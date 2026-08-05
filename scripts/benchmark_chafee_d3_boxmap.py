#!/usr/bin/env python3
"""Bounded D3 CMGDB benchmark: on-demand neural BoxMap versus precomputation.

The command is plan-only unless ``--execute`` is supplied.  Even in execute
mode it refuses subdivisions above 18, runs every graph in a supervised child
process, and never writes inside the canonical D3 study directory.  The
production level-24 graph is estimated from levels 12, 15, and 18; it is never
launched by this harness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


CODE_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_D3_ROOT = (
    CODE_ROOT / "output" / "chafee_latent_dimension_study" / "latent_3d" / "seed_0"
)
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "output" / "chafee_d3_boxmap_benchmark_v1"
DEFAULT_CHECKPOINT_ID = "canonical_d3_seed0"
DEFAULT_SUBDIVISIONS = (12, 15, 18)
BACKENDS = ("ondemand", "precomputed")
TARGET_SUBDIVISION = 24
DIMENSION = 3

# These hard ceilings cannot be raised through the CLI.  A different experiment
# requires a conscious source change and review.
HARD_MAX_SUBDIVISION = 18
HARD_MAX_REPEATS = 5
HARD_MAX_WARMUPS = 2
HARD_MAX_TIMEOUT_SECONDS = 120.0
HARD_MAX_RSS_MIB = 4_096
HARD_MAX_EDGES = 40_000_000
HARD_MAX_NEURAL_POINTS = 10_000_000
HARD_MAX_CALLBACK_RECTANGLES = 1_000_000

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RSS_MIB = 3_072
DEFAULT_MAX_EDGES = 30_000_000
DEFAULT_MAX_NEURAL_POINTS = 8_000_000
DEFAULT_MAX_CALLBACK_RECTANGLES = 750_000
DEFAULT_MAX_FORWARD_POINTS = 262_144
DEFAULT_REPEATS = 3
DEFAULT_WARMUPS = 1
SUPERVISOR_POLL_SECONDS = 0.05

EXISTING_ONDEMAND_CONTEXT = CANONICAL_D3_ROOT / "MG_uniform_s18_ondemand" / "summary.json"
EXISTING_PRECOMPUTED_CONTEXT = CANONICAL_D3_ROOT / "MG_uniform_s24" / "summary.json"


class BenchmarkBudgetExceeded(RuntimeError):
    """Raised by a callback before it exceeds an in-process work budget."""


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    run_root: Path
    models_dir: Path
    bounds_path: Path
    checkpoint_path: Path
    sidecar_path: Path
    checkpoint_sha256: str
    sidecar_sha256: str
    bounds_sha256: str


@dataclass(frozen=True)
class BenchmarkLimits:
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_rss_mib: int = DEFAULT_MAX_RSS_MIB
    max_edges: int = DEFAULT_MAX_EDGES
    max_neural_points: int = DEFAULT_MAX_NEURAL_POINTS
    max_callback_rectangles: int = DEFAULT_MAX_CALLBACK_RECTANGLES
    max_forward_points: int = DEFAULT_MAX_FORWARD_POINTS


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    checkpoint_id: str
    run_root: str
    backend: str
    subdivision: int
    repeat: int
    warmup: bool
    device: str
    limits: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _file_reference(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _checkpoint_from_root(checkpoint_id: str, run_root: Path) -> CheckpointSpec:
    run_root = run_root.resolve()
    models = run_root / "models"
    checkpoint = models / "autoencoder.pt"
    sidecar = models / "autoencoder.json"
    bounds = run_root / "bounds.json"
    for path in (checkpoint, sidecar, bounds):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"checkpoint input is missing or empty: {path}")

    sidecar_payload = _read_json(sidecar)
    low_dims = sidecar_payload.get("arch", {}).get("low_dims")
    if int(low_dims) != DIMENSION:
        raise ValueError(
            f"{sidecar} has latent dimension {low_dims!r}; this harness accepts only D3"
        )
    bounds_payload = _read_json(bounds)
    lower = np.asarray(bounds_payload.get("lower"), dtype=np.float64)
    upper = np.asarray(bounds_payload.get("upper"), dtype=np.float64)
    if lower.shape != (DIMENSION,) or upper.shape != (DIMENSION,):
        raise ValueError(f"{bounds} must contain three-dimensional lower/upper bounds")
    if not np.all(lower < upper):
        raise ValueError(f"{bounds} does not satisfy lower < upper")

    return CheckpointSpec(
        checkpoint_id=checkpoint_id,
        run_root=run_root,
        models_dir=models,
        bounds_path=bounds,
        checkpoint_path=checkpoint,
        sidecar_path=sidecar,
        checkpoint_sha256=sha256_file(checkpoint),
        sidecar_sha256=sha256_file(sidecar),
        bounds_sha256=sha256_file(bounds),
    )


def parse_checkpoint_argument(raw: str) -> tuple[str, Path]:
    name, separator, path_text = raw.partition("=")
    if not separator or not name.strip() or not path_text.strip():
        raise ValueError("--checkpoint must have the form NAME=/path/to/d3/run_root")
    normalized = name.strip()
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in normalized):
        raise ValueError(f"checkpoint name contains unsupported characters: {normalized!r}")
    return normalized, Path(path_text.strip())


def validate_protocol(
    *,
    subdivisions: Sequence[int],
    repeats: int,
    warmups: int,
    limits: BenchmarkLimits,
) -> tuple[int, ...]:
    levels = tuple(sorted(set(int(value) for value in subdivisions)))
    if not levels:
        raise ValueError("at least one pilot subdivision is required")
    for level in levels:
        if level < DIMENSION or level % DIMENSION:
            raise ValueError(
                f"D3 uniform pilot subdivision {level} must be positive and divisible by 3"
            )
        if level > HARD_MAX_SUBDIVISION:
            raise ValueError(
                f"subdivision {level} exceeds the hard pilot cap "
                f"{HARD_MAX_SUBDIVISION}; level {TARGET_SUBDIVISION} is extrapolation-only"
            )
    if not 1 <= repeats <= HARD_MAX_REPEATS:
        raise ValueError(f"repeats must be in [1, {HARD_MAX_REPEATS}]")
    if not 0 <= warmups <= HARD_MAX_WARMUPS:
        raise ValueError(f"warmups must be in [0, {HARD_MAX_WARMUPS}]")
    if not 1.0 <= limits.timeout_seconds <= HARD_MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout_seconds must be in [1, {HARD_MAX_TIMEOUT_SECONDS}]"
        )
    if not 128 <= limits.max_rss_mib <= HARD_MAX_RSS_MIB:
        raise ValueError(f"max_rss_mib must be in [128, {HARD_MAX_RSS_MIB}]")
    if not 1 <= limits.max_edges <= HARD_MAX_EDGES:
        raise ValueError(f"max_edges must be in [1, {HARD_MAX_EDGES}]")
    if not 1 <= limits.max_neural_points <= HARD_MAX_NEURAL_POINTS:
        raise ValueError(
            f"max_neural_points must be in [1, {HARD_MAX_NEURAL_POINTS}]"
        )
    if not 1 <= limits.max_callback_rectangles <= HARD_MAX_CALLBACK_RECTANGLES:
        raise ValueError(
            "max_callback_rectangles must be in "
            f"[1, {HARD_MAX_CALLBACK_RECTANGLES}]"
        )
    if not 1 <= limits.max_forward_points <= limits.max_neural_points:
        raise ValueError("max_forward_points must be positive and no larger than max_neural_points")
    return levels


def _size_projection(subdivision: int) -> dict[str, int]:
    axis_cells = 2 ** (subdivision // DIMENSION)
    cells = 2**subdivision
    unique_corners = (axis_cells + 1) ** DIMENSION
    return {
        "subdivision": subdivision,
        "axis_cells": axis_cells,
        "uniform_cells": cells,
        "precomputed_unique_corner_points": unique_corners,
        "precomputed_float64_table_bytes": unique_corners * DIMENSION * 8,
        "one_corner_pass_per_cell": cells * (2**DIMENSION),
    }


def _existing_context() -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key, path in (
        ("prior_ondemand_level18", EXISTING_ONDEMAND_CONTEXT),
        ("existing_precomputed_level24", EXISTING_PRECOMPUTED_CONTEXT),
    ):
        if not path.is_file():
            continue
        payload = _read_json(path)
        context[key] = {
            "reference": _file_reference(path),
            "backend": payload.get("backend"),
            "subdivision": payload.get("subdiv_max"),
            "duration_seconds": payload.get("duration_seconds"),
            "map_cells": payload.get("map_cells"),
            "cached_edges": payload.get("cached_edges"),
            "n_morse_nodes": payload.get("n_morse_nodes"),
            "callback": payload.get("callback"),
            "interpretation": (
                "Read-only historical context, not a repeat from this benchmark."
            ),
        }
    return context


def _trial_specs(
    checkpoints: Sequence[CheckpointSpec],
    *,
    subdivisions: Sequence[int],
    repeats: int,
    warmups: int,
    device: str,
    limits: BenchmarkLimits,
) -> list[TrialSpec]:
    trials: list[TrialSpec] = []
    limit_payload = asdict(limits)
    first_level = min(subdivisions)
    for checkpoint in checkpoints:
        for warmup_index in range(warmups):
            for backend in BACKENDS:
                trials.append(
                    TrialSpec(
                        trial_id=(
                            f"{checkpoint.checkpoint_id}_warmup{warmup_index + 1}_"
                            f"s{first_level}_{backend}"
                        ),
                        checkpoint_id=checkpoint.checkpoint_id,
                        run_root=str(checkpoint.run_root),
                        backend=backend,
                        subdivision=first_level,
                        repeat=warmup_index,
                        warmup=True,
                        device=device,
                        limits=limit_payload,
                    )
                )
        for subdivision in subdivisions:
            for repeat in range(repeats):
                order = BACKENDS if repeat % 2 == 0 else tuple(reversed(BACKENDS))
                for backend in order:
                    trials.append(
                        TrialSpec(
                            trial_id=(
                                f"{checkpoint.checkpoint_id}_s{subdivision}_"
                                f"r{repeat + 1}_{backend}"
                            ),
                            checkpoint_id=checkpoint.checkpoint_id,
                            run_root=str(checkpoint.run_root),
                            backend=backend,
                            subdivision=subdivision,
                            repeat=repeat,
                            warmup=False,
                            device=device,
                            limits=limit_payload,
                        )
                    )
    return trials


def build_plan(
    checkpoints: Sequence[CheckpointSpec],
    *,
    subdivisions: Sequence[int] = DEFAULT_SUBDIVISIONS,
    repeats: int = DEFAULT_REPEATS,
    warmups: int = DEFAULT_WARMUPS,
    device: str = "cpu",
    limits: BenchmarkLimits = BenchmarkLimits(),
) -> dict[str, Any]:
    if not checkpoints:
        raise ValueError("at least one D3 checkpoint is required")
    if len({checkpoint.checkpoint_id for checkpoint in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint IDs must be unique")
    levels = validate_protocol(
        subdivisions=subdivisions,
        repeats=repeats,
        warmups=warmups,
        limits=limits,
    )
    trials = _trial_specs(
        checkpoints,
        subdivisions=levels,
        repeats=repeats,
        warmups=warmups,
        device=device,
        limits=limits,
    )
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "mode": "bounded_pilot_with_extrapolation_only_for_production_level",
        "dimension": DIMENSION,
        "backends": {
            "ondemand": (
                "Batched neural evaluation of all eight rectangle corners inside "
                "each CMGDB callback; no persistent corner table."
            ),
            "precomputed": (
                "One bounded neural pass over unique grid corners followed by "
                "lookup-only batched CMGDB callbacks."
            ),
        },
        "primary_timing_contrasts": [
            "CMGDB phase: ondemand neural callbacks versus precomputed lookup callbacks",
            "single-build end to end: ondemand CMGDB versus precompute plus lookup CMGDB",
            "precompute amortization across repeated graph consumers",
        ],
        "checkpoints": [
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "run_root": str(checkpoint.run_root),
                "checkpoint": _file_reference(checkpoint.checkpoint_path),
                "sidecar": _file_reference(checkpoint.sidecar_path),
                "bounds": _file_reference(checkpoint.bounds_path),
                "selection_reason": (
                    "Canonical completed D3 seed-0 model; no second D3 checkpoint "
                    "currently exists in the study archive."
                ),
            }
            for checkpoint in checkpoints
        ],
        "pilot_subdivisions": list(levels),
        "pilot_sizes": [_size_projection(level) for level in levels],
        "target_extrapolation": {
            **_size_projection(TARGET_SUBDIVISION),
            "launch_permitted": False,
            "reason": (
                "The level-24 graph has 64 times as many cells as level 18 and "
                "the existing graph contains over one billion cached edges."
            ),
        },
        "warmups": {
            "full_graph_warmups_per_backend": warmups,
            "subdivision": min(levels),
            "included_in_statistics": False,
            "per_worker_neural_kernel_warmup": True,
        },
        "measured_repeats_per_backend_and_level": repeats,
        "ordering": (
            "Backends alternate first/second position by repeat to reduce order "
            "and thermal bias."
        ),
        "limits": asdict(limits),
        "hard_limits": {
            "max_subdivision": HARD_MAX_SUBDIVISION,
            "max_timeout_seconds": HARD_MAX_TIMEOUT_SECONDS,
            "max_rss_mib": HARD_MAX_RSS_MIB,
            "max_edges": HARD_MAX_EDGES,
            "max_neural_points": HARD_MAX_NEURAL_POINTS,
            "max_callback_rectangles": HARD_MAX_CALLBACK_RECTANGLES,
        },
        "instrumentation": {
            "time": [
                "model load",
                "neural warmup",
                "precompute",
                "CMGDB",
                "neural forward time",
                "callback time",
                "worker total",
            ],
            "memory": (
                "Supervisor samples whole-process RSS; worker also records "
                "resource.ru_maxrss and phase RSS snapshots."
            ),
            "neural_work": [
                "callback scalar and batch calls",
                "callback rectangles",
                "neural forward calls",
                "neural corner points",
            ],
            "graph": [
                "cells",
                "cached edges",
                "Morse nodes and edges",
                "minimal nodes",
                "Morse-set sizes",
                "graph fingerprint",
            ],
        },
        "stop_rules": [
            "Supervisor terminates a worker at the wall-time or RSS limit.",
            "CMGDB aborts before caching more than the configured edge limit.",
            "Callbacks abort before rectangle or neural-point budgets are exceeded.",
            "Any warmup or measured failure stops higher subdivisions for that checkpoint.",
            "Any on-demand/precomputed graph-fingerprint disagreement stops escalation.",
            "Subdivision 24 is never dispatched by this harness.",
        ],
        "extrapolation": {
            "methods": [
                "constant per-cell projection from the largest completed pilot",
                "power-law fit to median pilot measurements",
            ],
            "metrics": [
                "CMGDB seconds",
                "end-to-end seconds excluding model load",
                "cached edges",
                "neural corner points",
                "peak RSS",
            ],
            "claim_boundary": (
                "Heuristic capacity estimate only, not a production runtime "
                "measurement or statistical confidence interval."
            ),
        },
        "planned_trials": [asdict(trial) for trial in trials],
        "existing_read_only_context": _existing_context(),
    }


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def assert_safe_output_root(
    output_root: Path,
    checkpoints: Sequence[CheckpointSpec],
) -> Path:
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark output: {output}")
    protected = [CANONICAL_D3_ROOT.resolve()]
    protected.extend(checkpoint.run_root.resolve() for checkpoint in checkpoints)
    for path in protected:
        if _paths_overlap(output, path):
            raise ValueError(
                f"benchmark output {output} overlaps protected D3 input {path}"
            )
    if output == CODE_ROOT.resolve() or output == (CODE_ROOT / "output").resolve():
        raise ValueError("benchmark output must be a fresh dedicated directory")
    return output


class CountingEvaluator:
    """Count bounded neural-style point evaluations around any batch callable."""

    def __init__(
        self,
        evaluator: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        *,
        max_points: int,
        deadline: float | None = None,
    ):
        self.evaluator = evaluator
        self.max_points = int(max_points)
        self.deadline = deadline
        self.forward_calls = 0
        self.points = 0
        self.seconds = 0.0

    def reset(self) -> None:
        self.forward_calls = 0
        self.points = 0
        self.seconds = 0.0

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        n_points = int(values.shape[0])
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise BenchmarkBudgetExceeded("callback deadline reached")
        if self.points + n_points > self.max_points:
            raise BenchmarkBudgetExceeded(
                f"neural point budget would be exceeded: "
                f"{self.points + n_points}>{self.max_points}"
            )
        started = time.perf_counter()
        result = np.asarray(self.evaluator(values), dtype=np.float64)
        self.seconds += time.perf_counter() - started
        self.forward_calls += 1
        self.points += n_points
        if result.ndim != 2 or result.shape[0] != n_points:
            raise ValueError(
                "evaluator must return shape (n_points, output_dim); "
                f"got {result.shape}"
            )
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "neural_forward_calls": self.forward_calls,
            "neural_corner_points": self.points,
            "neural_forward_seconds": self.seconds,
        }


class DirectNeuralBoxMap:
    """Vectorized direct corner evaluation with callback and work counters."""

    def __init__(
        self,
        evaluator: CountingEvaluator,
        *,
        dimension: int = DIMENSION,
        padding: bool = True,
        max_rectangles: int = DEFAULT_MAX_CALLBACK_RECTANGLES,
        max_forward_points: int = DEFAULT_MAX_FORWARD_POINTS,
    ):
        self.evaluator = evaluator
        self.dimension = int(dimension)
        self.padding = bool(padding)
        self.max_rectangles = int(max_rectangles)
        self.max_forward_points = int(max_forward_points)
        self._corner_bits = np.asarray(
            list(product((0, 1), repeat=self.dimension)),
            dtype=bool,
        )
        self.scalar_calls = 0
        self.batch_calls = 0
        self.rectangles = 0
        self.callback_seconds = 0.0

    def __call__(self, rect: Any) -> list[float]:
        self.scalar_calls += 1
        return self._map(np.asarray(rect, dtype=np.float64).reshape(1, -1))[0]

    def batch(self, rects: Any) -> list[list[float]]:
        self.batch_calls += 1
        return self._map(np.asarray(rects, dtype=np.float64))

    def _map(self, rects: NDArray[np.float64]) -> list[list[float]]:
        started = time.perf_counter()
        if rects.ndim != 2 or rects.shape[1] != 2 * self.dimension:
            raise ValueError(
                f"rectangles must have shape (n, {2 * self.dimension}); "
                f"got {rects.shape}"
            )
        n_rectangles = int(rects.shape[0])
        if self.rectangles + n_rectangles > self.max_rectangles:
            raise BenchmarkBudgetExceeded(
                "callback rectangle budget would be exceeded: "
                f"{self.rectangles + n_rectangles}>{self.max_rectangles}"
            )
        lower = rects[:, : self.dimension]
        upper = rects[:, self.dimension :]
        widths = upper - lower
        if np.any(widths <= 0):
            raise ValueError("all rectangle widths must be positive")
        points_per_rect = 2**self.dimension
        rectangles_per_forward = max(
            1,
            self.max_forward_points // points_per_rect,
        )
        output = np.empty((n_rectangles, 2 * self.dimension), dtype=np.float64)
        for start in range(0, n_rectangles, rectangles_per_forward):
            end = min(start + rectangles_per_forward, n_rectangles)
            chunk_lower = lower[start:end, None, :]
            chunk_upper = upper[start:end, None, :]
            corners = np.where(
                self._corner_bits[None, :, :],
                chunk_upper,
                chunk_lower,
            )
            values = self.evaluator(
                corners.reshape((end - start) * points_per_rect, self.dimension)
            ).reshape(end - start, points_per_rect, -1)
            if values.shape[2] != self.dimension:
                raise ValueError(
                    f"latent map returned dimension {values.shape[2]}; "
                    f"expected {self.dimension}"
                )
            image_lower = values.min(axis=1)
            image_upper = values.max(axis=1)
            if self.padding:
                image_lower -= widths[start:end]
                image_upper += widths[start:end]
            output[start:end] = np.concatenate((image_lower, image_upper), axis=1)
        self.rectangles += n_rectangles
        self.callback_seconds += time.perf_counter() - started
        return output.tolist()

    def stats(self) -> dict[str, Any]:
        return {
            "scalar_calls": self.scalar_calls,
            "batch_calls": self.batch_calls,
            "rectangles": self.rectangles,
            "callback_seconds": self.callback_seconds,
            **self.evaluator.stats(),
        }


class InstrumentedLookupBoxMap:
    """Count lookup callbacks while preserving a precomputed map's batch API."""

    def __init__(self, box_map: Any):
        self.box_map = box_map
        self.scalar_calls = 0
        self.batch_calls = 0
        self.rectangles = 0
        self.callback_seconds = 0.0

    def __call__(self, rect: Any) -> list[float]:
        self.scalar_calls += 1
        self.rectangles += 1
        started = time.perf_counter()
        result = self.box_map(rect)
        self.callback_seconds += time.perf_counter() - started
        return result

    def batch(self, rects: Any) -> list[list[float]]:
        values = list(rects)
        self.batch_calls += 1
        self.rectangles += len(values)
        started = time.perf_counter()
        result = self.box_map.batch(values)
        self.callback_seconds += time.perf_counter() - started
        return result

    def stats(self, evaluator: CountingEvaluator) -> dict[str, Any]:
        return {
            "scalar_calls": self.scalar_calls,
            "batch_calls": self.batch_calls,
            "rectangles": self.rectangles,
            "callback_seconds": self.callback_seconds,
            **evaluator.stats(),
        }


def _ru_maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _current_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _torch_batch_evaluator(model: Any, device: Any) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    import torch

    model = model.to(device)
    model.eval()

    def evaluate(points: NDArray[np.float64]) -> NDArray[np.float64]:
        with torch.no_grad():
            tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
            values = model(tensor).detach().cpu().numpy()
        return np.asarray(values, dtype=np.float64)

    return evaluate


def _graph_summary(morse_graph: Any, map_graph: Any) -> dict[str, Any]:
    vertices = sorted(int(value) for value in morse_graph.vertices())
    edges = sorted(
        (int(source), int(target))
        for source, target in morse_graph.edges()
    )
    minimal_nodes = [
        node for node in vertices if len(morse_graph.adjacencies(node)) == 0
    ]
    morse_set_sizes = {
        str(node): len(morse_graph.morse_set(node)) for node in vertices
    }
    core = {
        "map_cells": int(map_graph.num_vertices()),
        "cached_edges": int(map_graph.num_cached_edges()),
        "morse_nodes": len(vertices),
        "morse_edges": edges,
        "minimal_nodes": minimal_nodes,
        "morse_set_sizes": morse_set_sizes,
    }
    core["graph_fingerprint"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return core


def _worker_result(spec: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import CMGDB
    from CMGDB.PrecomputedBoxMap import precompute_corner_grid

    from latentdynamics.analysis.hierarchical_precomputed import (
        HierarchicalPrecomputedBoxMap,
    )
    from latentdynamics.training import load_checkpoint

    started_total = time.perf_counter()
    limits = BenchmarkLimits(**spec["limits"])
    subdivision = int(spec["subdivision"])
    if subdivision > HARD_MAX_SUBDIVISION:
        raise ValueError("worker refused a subdivision above the hard pilot cap")

    os.environ.pop("CMGDB_MAPGRAPH_RESERVE_EDGES", None)
    os.environ.pop("CMGDB_MAPGRAPH_RESERVE_MIN_VERTICES", None)

    run_root = Path(spec["run_root"]).resolve()
    bounds_payload = _read_json(run_root / "bounds.json")
    lower = np.asarray(bounds_payload["lower"], dtype=np.float64)
    upper = np.asarray(bounds_payload["upper"], dtype=np.float64)
    if lower.shape != (DIMENSION,) or upper.shape != (DIMENSION,):
        raise ValueError("worker received non-D3 bounds")

    device = torch.device(str(spec["device"]))
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    model_started = time.perf_counter()
    model, arch = load_checkpoint(run_root / "models", map_location=device)
    if int(arch.low_dims) != DIMENSION:
        raise ValueError(f"worker checkpoint dimension is {arch.low_dims}, not 3")
    latent_map = model.latent_map.to(device)
    latent_map.eval()
    model_load_seconds = time.perf_counter() - model_started

    raw_evaluator = _torch_batch_evaluator(latent_map, device)
    warmup_started = time.perf_counter()
    warmup_points = np.stack(
        [
            lower,
            upper,
            0.5 * (lower + upper),
            lower + 0.25 * (upper - lower),
        ],
        axis=0,
    )
    raw_evaluator(warmup_points)
    neural_warmup_seconds = time.perf_counter() - warmup_started

    deadline = time.monotonic() + max(1.0, 0.9 * limits.timeout_seconds)
    evaluator = CountingEvaluator(
        raw_evaluator,
        max_points=limits.max_neural_points,
        deadline=deadline,
    )
    baseline_rss = _current_rss_bytes()
    precompute_seconds = 0.0
    precomputed_table: dict[str, Any] | None = None
    backend = str(spec["backend"])
    if backend == "ondemand":
        box_map: Any = DirectNeuralBoxMap(
            evaluator,
            dimension=DIMENSION,
            padding=True,
            max_rectangles=limits.max_callback_rectangles,
            max_forward_points=limits.max_forward_points,
        )
    elif backend == "precomputed":
        precompute_started = time.perf_counter()
        corners_per_axis = 2 ** (subdivision // DIMENSION) + 1
        values, output_dimension = precompute_corner_grid(
            evaluator,
            lower_bounds=lower,
            upper_bounds=upper,
            corners_per_axis=corners_per_axis,
            batch_points=limits.max_forward_points,
            device="cpu",
        )
        if int(output_dimension) != DIMENSION:
            raise ValueError(
                f"precompute output dimension {output_dimension}, expected 3"
            )
        lookup = HierarchicalPrecomputedBoxMap(
            lower=lower,
            upper=upper,
            coarse_subdiv=subdivision,
            fine_subdiv=subdivision,
            coarse_values=values,
            padding=True,
        )
        box_map = InstrumentedLookupBoxMap(lookup)
        precompute_seconds = time.perf_counter() - precompute_started
        precomputed_table = {
            "corners_per_axis": corners_per_axis,
            "unique_corner_points": corners_per_axis**DIMENSION,
            "array_shape": list(values.shape),
            "array_nbytes": int(values.nbytes),
        }
    else:
        raise ValueError(f"unsupported backend: {backend}")

    after_backend_rss = _current_rss_bytes()
    cmgdb_started = time.perf_counter()
    cmgdb_model = CMGDB.Model(
        subdivision,
        subdivision,
        subdivision,
        10_000,
        lower.tolist(),
        upper.tolist(),
        box_map,
    )
    if not hasattr(cmgdb_model, "set_batch_map"):
        raise RuntimeError("CMGDB.Model.set_batch_map is required")
    cmgdb_model.set_batch_map(box_map.batch)
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(cmgdb_model)
    cmgdb_seconds = time.perf_counter() - cmgdb_started
    if not bool(map_graph.has_cache()):
        raise RuntimeError("CMGDB did not retain the optimized batch cache")
    graph = _graph_summary(morse_graph, map_graph)
    if graph["map_cells"] != 2**subdivision:
        raise RuntimeError(
            f"CMGDB returned {graph['map_cells']} cells; expected {2**subdivision}"
        )

    callback = (
        box_map.stats()
        if isinstance(box_map, DirectNeuralBoxMap)
        else box_map.stats(evaluator)
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "trial": dict(spec),
        "timings": {
            "model_load_seconds": model_load_seconds,
            "neural_warmup_seconds": neural_warmup_seconds,
            "precompute_seconds": precompute_seconds,
            "cmgdb_seconds": cmgdb_seconds,
            "end_to_end_excluding_model_load_seconds": (
                precompute_seconds + cmgdb_seconds
            ),
            "worker_total_seconds": time.perf_counter() - started_total,
        },
        "memory": {
            "baseline_rss_bytes": baseline_rss,
            "after_backend_build_rss_bytes": after_backend_rss,
            "worker_ru_maxrss_bytes": _ru_maxrss_bytes(),
        },
        "callback": callback,
        "precomputed_table": precomputed_table,
        "graph": graph,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cmgdb_module": str(Path(CMGDB.__file__).resolve()),
        },
    }


def _worker_main(spec_path: Path, result_path: Path) -> int:
    spec = _read_json(spec_path)
    try:
        result = _worker_result(spec)
        exit_code = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "status": "failed",
            "trial": spec,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "memory": {"worker_ru_maxrss_bytes": _ru_maxrss_bytes()},
        }
        exit_code = 1
    write_json(result_path, result)
    return exit_code


def _terminate_worker(process: subprocess.Popen[Any]) -> None:
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def _read_tail(path: Path, max_characters: int = 8_000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_characters:]


def run_supervised_trial(
    trial: TrialSpec,
    *,
    worker_root: Path,
) -> dict[str, Any]:
    trial_root = worker_root / trial.trial_id
    trial_root.mkdir(parents=True)
    spec_path = trial_root / "spec.json"
    result_path = trial_root / "result.json"
    stdout_path = trial_root / "stdout.log"
    stderr_path = trial_root / "stderr.log"
    write_json(spec_path, asdict(trial))

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-spec",
        str(spec_path),
        "--worker-result",
        str(result_path),
    ]
    started = time.monotonic()
    peak_rss = 0
    stop_reason: str | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout_stream, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_stream:
        process = subprocess.Popen(
            command,
            cwd=CODE_ROOT,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
        )
        try:
            import psutil

            monitored_process = psutil.Process(process.pid)
        except (ImportError, OSError):
            monitored_process = None

        while process.poll() is None:
            elapsed = time.monotonic() - started
            if monitored_process is not None:
                try:
                    rss = int(monitored_process.memory_info().rss)
                    peak_rss = max(peak_rss, rss)
                except OSError:
                    pass
            if elapsed > trial.limits["timeout_seconds"]:
                stop_reason = "wall_timeout"
                _terminate_worker(process)
                break
            if peak_rss > int(trial.limits["max_rss_mib"]) * 1024**2:
                stop_reason = "rss_limit"
                _terminate_worker(process)
                break
            time.sleep(SUPERVISOR_POLL_SECONDS)
        exit_code = process.wait()

    elapsed = time.monotonic() - started
    if result_path.is_file():
        result = _read_json(result_path)
    else:
        result = {
            "schema_version": 1,
            "status": "supervisor_stopped" if stop_reason else "worker_no_result",
            "trial": asdict(trial),
        }
    result["supervisor"] = {
        "elapsed_seconds": elapsed,
        "sampled_peak_rss_bytes": peak_rss or None,
        "stop_reason": stop_reason,
        "worker_exit_code": exit_code,
        "command": command,
        "stdout_tail": _read_tail(stdout_path),
        "stderr_tail": _read_tail(stderr_path),
    }
    write_json(result_path, result)
    return result


def _completed(result: Mapping[str, Any]) -> bool:
    return result.get("status") == "complete"


def graph_parity(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_backend: dict[str, set[str]] = {backend: set() for backend in BACKENDS}
    for result in results:
        if not _completed(result):
            continue
        backend = str(result["trial"]["backend"])
        by_backend[backend].add(str(result["graph"]["graph_fingerprint"]))
    status = "verified"
    reason = ""
    if any(not values for values in by_backend.values()):
        status = "unavailable"
        reason = "one or both backends have no completed result"
    elif any(len(values) != 1 for values in by_backend.values()):
        status = "mismatch"
        reason = "a backend was not internally deterministic across repeats"
    elif by_backend["ondemand"] != by_backend["precomputed"]:
        status = "mismatch"
        reason = "on-demand and precomputed graph fingerprints differ"
    return {
        "status": status,
        "reason": reason,
        "fingerprints": {
            backend: sorted(values) for backend, values in by_backend.items()
        },
    }


def _describe(values: Sequence[float]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "sample_standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "sample_standard_deviation": (
            statistics.stdev(numbers) if len(numbers) > 1 else 0.0
        ),
        "minimum": min(numbers),
        "maximum": max(numbers),
    }


def _nested_number(result: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value: Any = result
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


SUMMARY_METRICS: dict[str, tuple[str, ...]] = {
    "cmgdb_seconds": ("timings", "cmgdb_seconds"),
    "precompute_seconds": ("timings", "precompute_seconds"),
    "end_to_end_excluding_model_load_seconds": (
        "timings",
        "end_to_end_excluding_model_load_seconds",
    ),
    "supervisor_peak_rss_bytes": ("supervisor", "sampled_peak_rss_bytes"),
    "cached_edges": ("graph", "cached_edges"),
    "neural_corner_points": ("callback", "neural_corner_points"),
    "neural_forward_calls": ("callback", "neural_forward_calls"),
    "callback_rectangles": ("callback", "rectangles"),
}


def summarize_trials(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    measured = [
        result
        for result in results
        if not bool(result.get("trial", {}).get("warmup", False))
    ]
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for result in measured:
        trial = result.get("trial", {})
        grouped[
            (
                str(trial.get("checkpoint_id")),
                int(trial.get("subdivision")),
                str(trial.get("backend")),
            )
        ].append(result)

    groups = []
    for (checkpoint_id, subdivision, backend), group in sorted(grouped.items()):
        completed = [result for result in group if _completed(result)]
        metrics = {
            name: _describe(
                [
                    value
                    for result in completed
                    if (value := _nested_number(result, path)) is not None
                ]
            )
            for name, path in SUMMARY_METRICS.items()
        }
        groups.append(
            {
                "checkpoint_id": checkpoint_id,
                "subdivision": subdivision,
                "backend": backend,
                "requested_repeats": len(group),
                "completed_repeats": len(completed),
                "failure_status_counts": dict(
                    sorted(
                        {
                            status: sum(
                                int(str(result.get("status")) == status)
                                for result in group
                            )
                            for status in {
                                str(result.get("status")) for result in group
                            }
                            if status != "complete"
                        }.items()
                    )
                ),
                "metrics": metrics,
                "graph_fingerprints": sorted(
                    {
                        str(result["graph"]["graph_fingerprint"])
                        for result in completed
                    }
                ),
            }
        )

    parity = []
    for checkpoint_id in sorted(
        {str(result.get("trial", {}).get("checkpoint_id")) for result in measured}
    ):
        for subdivision in sorted(
            {
                int(result.get("trial", {}).get("subdivision"))
                for result in measured
                if str(result.get("trial", {}).get("checkpoint_id")) == checkpoint_id
            }
        ):
            selected = [
                result
                for result in measured
                if str(result.get("trial", {}).get("checkpoint_id")) == checkpoint_id
                and int(result.get("trial", {}).get("subdivision")) == subdivision
            ]
            parity.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "subdivision": subdivision,
                    **graph_parity(selected),
                }
            )
    return {"groups": groups, "graph_parity": parity}


def _power_law_projection(
    points: Sequence[tuple[float, float]],
    target_x: float,
) -> dict[str, float | int | None]:
    usable = [(float(x), float(y)) for x, y in points if x > 0 and y > 0]
    if len(usable) < 2:
        return {
            "n_pilot_levels": len(usable),
            "exponent": None,
            "target_estimate": None,
        }
    logs_x = [math.log(x) for x, _ in usable]
    logs_y = [math.log(y) for _, y in usable]
    mean_x = statistics.fmean(logs_x)
    mean_y = statistics.fmean(logs_y)
    denominator = sum((value - mean_x) ** 2 for value in logs_x)
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(logs_x, logs_y, strict=True)
    ) / denominator
    intercept = mean_y - slope * mean_x
    return {
        "n_pilot_levels": len(usable),
        "exponent": slope,
        "target_estimate": math.exp(intercept + slope * math.log(target_x)),
    }


def extrapolate_target(
    summary: Mapping[str, Any],
    *,
    target_subdivision: int = TARGET_SUBDIVISION,
    limits: BenchmarkLimits = BenchmarkLimits(),
) -> dict[str, Any]:
    target_cells = float(2**target_subdivision)
    by_checkpoint_backend: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for group in summary["groups"]:
        if int(group["completed_repeats"]) > 0:
            by_checkpoint_backend[
                (str(group["checkpoint_id"]), str(group["backend"]))
            ].append(group)

    projections = []
    for (checkpoint_id, backend), groups in sorted(by_checkpoint_backend.items()):
        ordered = sorted(groups, key=lambda group: int(group["subdivision"]))
        metrics: dict[str, Any] = {}
        for metric in (
            "cmgdb_seconds",
            "end_to_end_excluding_model_load_seconds",
            "supervisor_peak_rss_bytes",
            "cached_edges",
            "neural_corner_points",
        ):
            points = [
                (
                    float(2 ** int(group["subdivision"])),
                    float(group["metrics"][metric]["median"]),
                )
                for group in ordered
                if group["metrics"][metric]["median"] is not None
            ]
            power = _power_law_projection(points, target_cells)
            if points:
                highest_cells, highest_value = points[-1]
                constant_per_cell = highest_value * target_cells / highest_cells
            else:
                constant_per_cell = None
            metrics[metric] = {
                "constant_per_cell_from_highest_pilot": constant_per_cell,
                "power_law": power,
            }

        edge_estimate = metrics["cached_edges"][
            "constant_per_cell_from_highest_pilot"
        ]
        csr_lower_bound = (
            None
            if edge_estimate is None
            else int((target_cells + 1.0) * 8 + float(edge_estimate) * 8)
        )
        estimated_peak = metrics["supervisor_peak_rss_bytes"][
            "constant_per_cell_from_highest_pilot"
        ]
        reasons = []
        if edge_estimate is not None and edge_estimate > limits.max_edges:
            # A benchmark-harness go/no-go on its own projections, not a CMGDB
            # limit: nothing here constrains what a real run may attempt.
            reasons.append("estimated cached edges exceed the pilot edge ceiling")
        if csr_lower_bound is not None and csr_lower_bound > limits.max_rss_mib * 1024**2:
            reasons.append("CSR lower bound alone exceeds the pilot RSS ceiling")
        if estimated_peak is not None and estimated_peak > limits.max_rss_mib * 1024**2:
            reasons.append("linear peak-RSS projection exceeds the pilot RSS ceiling")
        projections.append(
            {
                "checkpoint_id": checkpoint_id,
                "backend": backend,
                "target_subdivision": target_subdivision,
                "target_cells": int(target_cells),
                "metrics": metrics,
                "estimated_csr_storage_lower_bound_bytes": csr_lower_bound,
                "dispatch_recommendation": "do_not_run" if reasons else "manual_review",
                "dispatch_reasons": reasons,
            }
        )

    unique_corners = (2 ** (target_subdivision // DIMENSION) + 1) ** DIMENSION
    return {
        "schema_version": 1,
        "target_subdivision": target_subdivision,
        "target_dispatch_permitted_by_harness": False,
        "target_uniform_cells": int(target_cells),
        "precomputed_unique_corner_points": unique_corners,
        "precomputed_float64_table_bytes": unique_corners * DIMENSION * 8,
        "claim_boundary": (
            "These are heuristic capacity projections from bounded pilots, not "
            "measurements or confidence intervals."
        ),
        "projections": projections,
    }


def run_benchmark(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    worker_root = output_root / "workers"
    worker_root.mkdir()
    all_results: list[dict[str, Any]] = []
    planned = [TrialSpec(**payload) for payload in plan["planned_trials"]]
    by_checkpoint: dict[str, list[TrialSpec]] = defaultdict(list)
    for trial in planned:
        by_checkpoint[trial.checkpoint_id].append(trial)

    stop_events = []
    for checkpoint_id, checkpoint_trials in by_checkpoint.items():
        warmups = [trial for trial in checkpoint_trials if trial.warmup]
        checkpoint_stopped = False
        for trial in warmups:
            result = run_supervised_trial(trial, worker_root=worker_root)
            all_results.append(result)
            if not _completed(result):
                checkpoint_stopped = True
                stop_events.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "after_trial": trial.trial_id,
                        "reason": "warmup_failed",
                    }
                )
                break
        if checkpoint_stopped:
            continue

        measured = [trial for trial in checkpoint_trials if not trial.warmup]
        levels = sorted({trial.subdivision for trial in measured})
        for level in levels:
            level_trials = [trial for trial in measured if trial.subdivision == level]
            level_results = []
            for trial in level_trials:
                result = run_supervised_trial(trial, worker_root=worker_root)
                all_results.append(result)
                level_results.append(result)
            failure = next(
                (result for result in level_results if not _completed(result)),
                None,
            )
            parity = graph_parity(level_results)
            if failure is not None or parity["status"] != "verified":
                stop_events.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "after_subdivision": level,
                        "reason": (
                            f"trial_status:{failure.get('status')}"
                            if failure is not None
                            else f"graph_parity:{parity['status']}"
                        ),
                        "parity": parity,
                    }
                )
                break

    summary = summarize_trials(all_results)
    extrapolation = extrapolate_target(
        summary,
        limits=BenchmarkLimits(**plan["limits"]),
    )
    result = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "status": "complete" if not stop_events else "complete_with_stops",
        "trial_counts": {
            "planned": len(planned),
            "dispatched": len(all_results),
            "complete": sum(int(_completed(value)) for value in all_results),
            "failed_or_stopped": sum(int(not _completed(value)) for value in all_results),
        },
        "stop_events": stop_events,
        "summary": summary,
        "extrapolation": extrapolation,
        "results": all_results,
    }
    return result


def _csv_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for trial_result in result["results"]:
        trial = trial_result.get("trial", {})
        row = {
            "trial_id": trial.get("trial_id"),
            "checkpoint_id": trial.get("checkpoint_id"),
            "backend": trial.get("backend"),
            "subdivision": trial.get("subdivision"),
            "repeat": trial.get("repeat"),
            "warmup": trial.get("warmup"),
            "status": trial_result.get("status"),
            "supervisor_stop_reason": trial_result.get("supervisor", {}).get(
                "stop_reason"
            ),
            "cmgdb_seconds": _nested_number(
                trial_result, ("timings", "cmgdb_seconds")
            ),
            "precompute_seconds": _nested_number(
                trial_result, ("timings", "precompute_seconds")
            ),
            "end_to_end_excluding_model_load_seconds": _nested_number(
                trial_result,
                ("timings", "end_to_end_excluding_model_load_seconds"),
            ),
            "sampled_peak_rss_bytes": _nested_number(
                trial_result, ("supervisor", "sampled_peak_rss_bytes")
            ),
            "map_cells": _nested_number(trial_result, ("graph", "map_cells")),
            "cached_edges": _nested_number(
                trial_result, ("graph", "cached_edges")
            ),
            "morse_nodes": _nested_number(
                trial_result, ("graph", "morse_nodes")
            ),
            "graph_fingerprint": trial_result.get("graph", {}).get(
                "graph_fingerprint"
            ),
            "callback_batch_calls": _nested_number(
                trial_result, ("callback", "batch_calls")
            ),
            "callback_rectangles": _nested_number(
                trial_result, ("callback", "rectangles")
            ),
            "neural_forward_calls": _nested_number(
                trial_result, ("callback", "neural_forward_calls")
            ),
            "neural_corner_points": _nested_number(
                trial_result, ("callback", "neural_corner_points")
            ),
            "error_type": trial_result.get("error_type"),
            "error_message": trial_result.get("error_message"),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty benchmark CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def render_readme(plan: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    lines = [
        "# Bounded Chafee-Infante D3 BoxMap benchmark",
        "",
        "This package compares batched on-demand neural corner evaluation with "
        "precomputed unique-corner lookup. New CMGDB work is hard-capped at "
        "subdivision 18; subdivision 24 is extrapolation-only.",
        "",
        "## Safety outcome",
        "",
        f"- Status: `{result['status']}`",
        f"- Dispatched: {result['trial_counts']['dispatched']} of "
        f"{result['trial_counts']['planned']} planned trials",
        f"- Complete: {result['trial_counts']['complete']}; failed/stopped: "
        f"{result['trial_counts']['failed_or_stopped']}",
        f"- Wall limit per worker: {plan['limits']['timeout_seconds']} seconds",
        f"- RSS limit per worker: {plan['limits']['max_rss_mib']} MiB",
        f"- Cached-edge projection ceiling: {plan['limits']['max_edges']:,}",
        "",
        "## Measured medians",
        "",
        "| Checkpoint | Subdivision | Backend | n | CMGDB (s) | Precompute (s) | "
        "End to end (s) | Peak RSS (MiB) | Cached edges | Neural points |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in result["summary"]["groups"]:
        metrics = group["metrics"]
        peak = metrics["supervisor_peak_rss_bytes"]["median"]
        lines.append(
            f"| {group['checkpoint_id']} | {group['subdivision']} | "
            f"{group['backend']} | {group['completed_repeats']} | "
            f"{_format_number(metrics['cmgdb_seconds']['median'])} | "
            f"{_format_number(metrics['precompute_seconds']['median'])} | "
            f"{_format_number(metrics['end_to_end_excluding_model_load_seconds']['median'])} | "
            f"{_format_number(None if peak is None else float(peak) / 1024**2, 1)} | "
            f"{_format_number(metrics['cached_edges']['median'], 0)} | "
            f"{_format_number(metrics['neural_corner_points']['median'], 0)} |"
        )

    lines.extend(
        [
            "",
            "## Graph parity",
            "",
            "| Checkpoint | Subdivision | Status |",
            "|---|---:|---|",
        ]
    )
    for parity in result["summary"]["graph_parity"]:
        lines.append(
            f"| {parity['checkpoint_id']} | {parity['subdivision']} | "
            f"{parity['status']} |"
        )

    lines.extend(
        [
            "",
            "## Production-level extrapolation",
            "",
            "The estimates below are capacity heuristics, not measurements or "
            "confidence intervals. This harness cannot dispatch subdivision 24.",
            "",
            "| Checkpoint | Backend | Linear CMGDB (s) | Power-law CMGDB (s) | "
            "Linear edges | Recommendation |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for projection in result["extrapolation"]["projections"]:
        cmgdb = projection["metrics"]["cmgdb_seconds"]
        edges = projection["metrics"]["cached_edges"]
        lines.append(
            f"| {projection['checkpoint_id']} | {projection['backend']} | "
            f"{_format_number(cmgdb['constant_per_cell_from_highest_pilot'])} | "
            f"{_format_number(cmgdb['power_law']['target_estimate'])} | "
            f"{_format_number(edges['constant_per_cell_from_highest_pilot'], 0)} | "
            f"{projection['dispatch_recommendation']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- The canonical seed-0 model is the only completed D3 checkpoint "
            "currently available; additional D3 checkpoints may be supplied explicitly.",
            "- Model loading is excluded from the primary timing contrast.",
            "- Precompute time and lookup-only CMGDB time are both reported, so "
            "single-use and amortized comparisons remain distinct.",
            "- RSS sampling includes the Python and native CMGDB process. Accelerator "
            "allocator accounting may be incomplete on non-CPU devices.",
            "- A matching Morse fingerprint and cached-edge count are a strong parity "
            "check, not a bytewise audit of every native adjacency.",
            "- Existing subdivision-18 on-demand and subdivision-24 precomputed "
            "artifacts are read-only context, not benchmark repeats.",
            "",
        ]
    )
    return "\n".join(lines)


def write_benchmark_package(
    output_root: Path,
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    write_json(output_root / "plan.json", plan)
    write_json(output_root / "results.json", result)
    write_json(output_root / "summary.json", result["summary"])
    write_json(output_root / "extrapolation.json", result["extrapolation"])
    write_csv(output_root / "results.csv", _csv_rows(result))
    (output_root / "README.md").write_text(
        render_readme(plan, result),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run bounded child-process pilots; omitted means print the plan only",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=RUN_ROOT",
        help="repeatable D3 run root; defaults to the canonical seed-0 run",
    )
    parser.add_argument(
        "--subdivisions",
        type=int,
        nargs="+",
        default=list(DEFAULT_SUBDIVISIONS),
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-rss-mib", type=int, default=DEFAULT_MAX_RSS_MIB)
    parser.add_argument("--max-edges", type=int, default=DEFAULT_MAX_EDGES)
    parser.add_argument(
        "--max-neural-points",
        type=int,
        default=DEFAULT_MAX_NEURAL_POINTS,
    )
    parser.add_argument(
        "--max-callback-rectangles",
        type=int,
        default=DEFAULT_MAX_CALLBACK_RECTANGLES,
    )
    parser.add_argument(
        "--max-forward-points",
        type=int,
        default=DEFAULT_MAX_FORWARD_POINTS,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec or args.worker_result:
        if args.worker_spec is None or args.worker_result is None:
            raise ValueError("internal worker mode requires both worker paths")
        return _worker_main(args.worker_spec, args.worker_result)

    checkpoint_arguments = (
        [parse_checkpoint_argument(raw) for raw in args.checkpoint]
        if args.checkpoint
        else [(DEFAULT_CHECKPOINT_ID, CANONICAL_D3_ROOT)]
    )
    checkpoints = [
        _checkpoint_from_root(checkpoint_id, root)
        for checkpoint_id, root in checkpoint_arguments
    ]
    limits = BenchmarkLimits(
        timeout_seconds=args.timeout_seconds,
        max_rss_mib=args.max_rss_mib,
        max_edges=args.max_edges,
        max_neural_points=args.max_neural_points,
        max_callback_rectangles=args.max_callback_rectangles,
        max_forward_points=args.max_forward_points,
    )
    plan = build_plan(
        checkpoints,
        subdivisions=args.subdivisions,
        repeats=args.repeats,
        warmups=args.warmups,
        device=args.device,
        limits=limits,
    )
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0

    output_root = assert_safe_output_root(args.output_root, checkpoints)
    output_root.mkdir(parents=True)
    write_json(output_root / "plan.json", plan)
    result = run_benchmark(plan, output_root=output_root)
    write_benchmark_package(output_root, plan, result)
    print(
        f"Wrote bounded D3 BoxMap benchmark: {output_root} "
        f"({result['trial_counts']['complete']}/"
        f"{result['trial_counts']['dispatched']} dispatched trials complete)",
        flush=True,
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
