#!/usr/bin/env python3
"""Run the fixed Patrick Leslie3D latent map over six CMGDB ladders.

The expensive neural corner table is built exactly once at subdivision 30 and
is reused by every CMGDB model.  Each model still performs its own adaptive
Morse decomposition because ``subdiv_init`` and ``subdiv_min`` change the SCC
hierarchy and cannot be recovered by coarsening another run's graph.

The frozen run matrix is::

    (init, min, max, limit)
    (16, 18, 30, 1_000_000)
    (18, 20, 30, 1_000_000)
    (20, 22, 30, 1_000_000)
    (22, 24, 30, 1_000_000)
    (24, 26, 30, 1_000_000)
    (26, 28, 30, 1_000_000)

Only Patrick's archived latent dynamics checkpoint and recorded latent bounds
enter the CMGDB computation.  The encoder and decoder are copied into each
run's provenance bundle but are not evaluated here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import CMGDB
import numpy as np
import psutil
import torch

from latentdynamics.analysis.morse_graph_parser import MorseGraph
from latentdynamics.viz.morse_plots import save_morse_graph_artifacts


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"
SOURCE_ROOT = CODE_ROOT / "replay_sources" / "leslie3d_example2"
MODEL_ROOT = SOURCE_ROOT / "models"
ARCHIVED_MG_ROOT = SOURCE_ROOT / "MG"

EXPERIMENT = "leslie3d_example2_patrick_cmgdb_initmin_sweep_max30_limit1000000_v1"
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "output" / EXPERIMENT

LOWER = np.array([-0.37490714, -0.4695556], dtype=np.float64)
UPPER = np.array([0.3535685, 0.455769], dtype=np.float64)
SUBDIV_MAX = 30
SUBDIV_LIMIT = 1_000_000
RUN_MATRIX: tuple[tuple[int, int], ...] = (
    (16, 18),
    (18, 20),
    (20, 22),
    (22, 24),
    (24, 26),
    (26, 28),
)
PADDING = True
EVAL_MODE = "corners"
PRECOMPUTE_BATCH_POINTS = 262_144
PLAN_SCHEMA_VERSION = 1
COMPLETION_ARTIFACT_KEYS = frozenset(
    {"morse_graph", "morse_sets", "parameter_log", "cell_summary", "run_manifest"}
)
CHECKPOINT_NAMES = frozenset({"encoder.pt", "dynamics.pt", "decoder.pt"})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _artifact_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty artifact: {path}")
    return {
        "path": str(path.relative_to(relative_to)),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _git_record(path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {
            "path": str(path),
            "commit": commit,
            "dirty": bool(status.strip()),
            "status_line_count": len(status.splitlines()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(path), "commit": None, "dirty": None}


def _run_id(index: int, init: int, minimum: int) -> str:
    return f"run_{index:02d}_init{init}_min{minimum}_max{SUBDIV_MAX}"


def _source_records() -> dict[str, Any]:
    paths = {
        "encoder": MODEL_ROOT / "encoder.pt",
        "dynamics": MODEL_ROOT / "dynamics.pt",
        "decoder": MODEL_ROOT / "decoder.pt",
        "archived_morse_graph": ARCHIVED_MG_ROOT / "morse_graph",
        "archived_morse_sets": ARCHIVED_MG_ROOT / "morse_sets",
        "archived_parameter_log": SOURCE_ROOT / "mg_params_log.txt",
    }
    return {
        name: {
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }


def _build_plan(*, output_root: Path, batch_points: int) -> dict[str, Any]:
    axis_depths = [
        (SUBDIV_MAX - axis + 2 - 1) // 2
        for axis in range(2)
    ]
    corner_shape = [2**depth + 1 for depth in axis_depths]
    n_corners = int(np.prod(np.asarray(corner_shape, dtype=object)))
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "output_root": str(output_root.resolve()),
        "source": _source_records(),
        "bounds": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
        "box_map": {
            "factory": "CMGDB.make_precomputed_box_map",
            "mode": "adaptive",
            "eval_mode": EVAL_MODE,
            "padding": PADDING,
            "device": "cpu",
            "input_dtype": "torch.float32",
            "stored_table_dtype": "numpy.float64",
            "subdiv_max": SUBDIV_MAX,
            "axis_depths": axis_depths,
            "corner_lattice_shape": corner_shape,
            "corner_point_count": n_corners,
            "output_dimension": 2,
            "estimated_table_bytes": n_corners * 2 * 8,
            "batch_points": batch_points,
            "shared_across_all_runs": True,
        },
        "cmgdb": {
            "subdiv_max": SUBDIV_MAX,
            "subdiv_limit": SUBDIV_LIMIT,
            "compute_entrypoint": "ComputeConleyMorseGraphOnly",
            "batch_callback": True,
        },
        "runs": [
            {
                "index": index,
                "run_id": _run_id(index, init, minimum),
                "subdiv_init": init,
                "subdiv_min": minimum,
                "subdiv_max": SUBDIV_MAX,
                "subdiv_limit": SUBDIV_LIMIT,
            }
            for index, (init, minimum) in enumerate(RUN_MATRIX)
        ],
    }


def _ensure_plan(*, output_root: Path, batch_points: int) -> tuple[dict[str, Any], str]:
    plan = _build_plan(output_root=output_root, batch_points=batch_points)
    plan_hash = _canonical_hash(plan)
    envelope = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_sha256": plan_hash,
        "plan": plan,
    }
    path = output_root / "experiment_plan.json"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != envelope:
            raise ValueError(f"existing plan differs from requested plan: {path}")
        return plan, plan_hash
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing nonempty output root without matching plan: {output_root}"
        )
    _write_json_atomic(path, envelope)
    return plan, plan_hash


class PeakRSSMonitor(AbstractContextManager["PeakRSSMonitor"]):
    """Sample process RSS, including recursive children, in a background thread."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            total = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_bytes = max(self.peak_bytes, int(total))
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "PeakRSSMonitor":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _setup_logging(output_root: Path) -> logging.Logger:
    logger = logging.getLogger(EXPERIMENT)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    log_path = output_root / "logs" / "launcher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _load_dynamics() -> torch.nn.Module:
    sys.path.insert(0, str(WORKSPACE_ROOT / "archive" / "brittany"))
    dynamics = torch.load(
        MODEL_ROOT / "dynamics.pt",
        map_location="cpu",
        weights_only=False,
    )
    dynamics = dynamics.float().cpu().eval()
    for parameter in dynamics.parameters():
        parameter.requires_grad_(False)
    return dynamics


def _graph_summary(dot_path: Path) -> dict[str, Any]:
    graph = MorseGraph.from_dot(dot_path)
    edges = sorted(
        (int(source), int(target))
        for source, targets in graph.edges.items()
        for target in targets
    )
    indegree = dict.fromkeys(graph.nodes, 0)
    for _source, target in edges:
        indegree[target] = indegree.get(target, 0) + 1
    return {
        "node_count": len(graph.nodes),
        "edge_count": len(edges),
        "nodes": graph.nodes,
        "edges": [list(edge) for edge in edges],
        "minimal_nodes": sorted(graph.minimal),
        "source_nodes": sorted(node for node in graph.nodes if indegree.get(node, 0) == 0),
        "conley_indices": {str(node): graph.labels.get(node, "") for node in graph.nodes},
        "colors": {str(node): graph.colors.get(node, "") for node in graph.nodes},
    }


def _morse_set_summary(path: Path) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    area = defaultdict(float)
    lower = defaultdict(lambda: np.array([np.inf, np.inf], dtype=np.float64))
    upper = defaultdict(lambda: np.array([-np.inf, -np.inf], dtype=np.float64))
    width_counts: dict[int, Counter[tuple[float, float]]] = defaultdict(Counter)
    row_count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            values = np.fromstring(raw, sep=",")
            if values.shape != (5,) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid Morse-set row {line_number} in {path}")
            lo = values[:2]
            hi = values[2:4]
            label_value = values[4]
            label = int(round(float(label_value)))
            if label_value != label or np.any(lo >= hi):
                raise ValueError(f"invalid box/label at row {line_number} in {path}")
            if np.any(lo < LOWER - 1e-12) or np.any(hi > UPPER + 1e-12):
                raise ValueError(f"box outside archived bounds at row {line_number}")
            width = hi - lo
            counts[label] += 1
            area[label] += float(np.prod(width))
            lower[label] = np.minimum(lower[label], lo)
            upper[label] = np.maximum(upper[label], hi)
            width_counts[label][tuple(float(x) for x in width)] += 1
            row_count += 1
    if row_count == 0:
        raise ValueError(f"empty Morse-set file: {path}")
    labels = sorted(counts)
    per_node: dict[str, Any] = {}
    for label in labels:
        per_node[str(label)] = {
            "box_count": counts[label],
            "summed_cover_area": area[label],
            "lower": lower[label].tolist(),
            "upper": upper[label].tolist(),
            "extent": (upper[label] - lower[label]).tolist(),
            "width_histogram": [
                {"width": list(width), "count": count}
                for width, count in sorted(width_counts[label].items())
            ],
        }
    return {
        "row_count": row_count,
        "labels": labels,
        "summed_cover_area": float(sum(area.values())),
        "per_node": per_node,
    }


def _copy_models(run_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    destination = run_root / "models"
    destination.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for name in ("encoder.pt", "dynamics.pt", "decoder.pt"):
        source = MODEL_ROOT / name
        target = destination / name
        if overwrite or not target.exists():
            shutil.copy2(source, target)
        source_hash = _sha256(source)
        target_hash = _sha256(target)
        if source_hash != target_hash:
            raise ValueError(f"checkpoint copy mismatch: {target}")
        records[name] = {
            "source_sha256": source_hash,
            "copy": _artifact_record(target, relative_to=run_root),
            "byte_identical": True,
        }
    return records


def _write_parameter_log(
    path: Path,
    *,
    init: int,
    minimum: int,
    graph_seconds: float,
    shared_precompute_seconds: float,
) -> None:
    path.write_text(
        "\n".join(
            [
                "--- CMGDB Computation Parameters ---",
                f"Lower bounds: {LOWER.tolist()}",
                f"Upper bounds: {UPPER.tolist()}",
                "Box map: shared level-30 adaptive precomputed corner table",
                f"Padding: {PADDING}",
                "Device: cpu",
                f"Subdivision init: {init}",
                f"Subdivision min: {minimum}",
                f"Subdivision max: {SUBDIV_MAX}",
                f"Subdivision limit: {SUBDIV_LIMIT}",
                f"Shared precompute duration seconds: {shared_precompute_seconds:.6f}",
                f"Graph computation duration seconds: {graph_seconds:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _completion_is_valid(run_root: Path, *, plan_hash: str) -> bool:
    marker_path = run_root / "stage_morse_complete.json"
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if marker.get("plan_sha256") != plan_hash or marker.get("status") != "complete":
        return False
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != COMPLETION_ARTIFACT_KEYS:
        return False
    resolved_root = run_root.resolve()
    for record in artifacts.values():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            return False
        path = (run_root / record["path"]).resolve()
        if not path.is_relative_to(resolved_root):
            return False
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            return False
    try:
        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("plan_sha256") != plan_hash or manifest.get("status") != "complete":
        return False
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != CHECKPOINT_NAMES:
        return False
    for name, record in checkpoints.items():
        if not isinstance(record, dict) or not isinstance(record.get("copy"), dict):
            return False
        source = MODEL_ROOT / name
        copy_record = record["copy"]
        if not isinstance(copy_record.get("path"), str):
            return False
        copied = (run_root / copy_record["path"]).resolve()
        if not copied.is_relative_to(resolved_root) or not copied.is_file():
            return False
        source_hash = _sha256(source)
        copied_hash = _sha256(copied)
        if (
            source_hash != copied_hash
            or record.get("source_sha256") != source_hash
            or copy_record.get("sha256") != copied_hash
            or record.get("byte_identical") is not True
        ):
            return False
    return True


def _environment_record() -> dict[str, Any]:
    return {
        "created_utc": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cmgdb_module": str(Path(CMGDB.__file__).resolve()),
        "cpu_count": os.cpu_count(),
        "system_memory_bytes": int(psutil.virtual_memory().total),
        "repositories": {
            "code": _git_record(CODE_ROOT),
            "cmgdb": _git_record(WORKSPACE_ROOT / "archive" / "CMGDB"),
        },
        "allocation_environment": {
            key: os.environ.get(key)
            for key in (
                "CMGDB_MAPGRAPH_MAX_VERTICES",
                "CMGDB_MAPGRAPH_MAX_EDGES",
                "CMGDB_MAPGRAPH_CACHE",
                "CMGDB_MAPGRAPH_RESERVE_EDGES",
            )
        },
    }


def _collect_completed_runs(output_root: Path, *, plan_hash: str) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for index, (init, minimum) in enumerate(RUN_MATRIX):
        run_id = _run_id(index, init, minimum)
        run_root = output_root / "runs" / run_id
        if not _completion_is_valid(run_root, plan_hash=plan_hash):
            continue
        summary = json.loads((run_root / "cell_summary.json").read_text(encoding="utf-8"))
        if summary.get("run_id") != run_id or summary.get("plan_sha256") != plan_hash:
            raise ValueError(f"completed run summary does not match the plan: {run_root}")
        completed.append(summary)
    return completed


def _write_sweep_summary(
    output_root: Path,
    *,
    plan_hash: str,
    precompute_record: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    completed = _collect_completed_runs(output_root, plan_hash=plan_hash)
    sweep_summary = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "complete" if len(completed) == len(RUN_MATRIX) and not failures else "partial",
        "plan_sha256": plan_hash,
        "shared_precompute": precompute_record,
        "completed_run_count": len(completed),
        "expected_run_count": len(RUN_MATRIX),
        "failed_run_ids": failures,
        "runs": completed,
        "updated_utc": _utc_now(),
    }
    _write_json_atomic(output_root / "sweep_summary.json", sweep_summary)
    return sweep_summary


def run_sweep(
    *,
    output_root: Path,
    batch_points: int,
    force: bool,
) -> int:
    plan, plan_hash = _ensure_plan(
        output_root=output_root,
        batch_points=batch_points,
    )
    logger = _setup_logging(output_root)
    provenance_path = output_root / "source_provenance.json"
    if not provenance_path.exists():
        _write_json_atomic(provenance_path, _environment_record())
    logger.info("frozen plan sha256=%s", plan_hash)
    logger.info("output root: %s", output_root)

    incomplete: list[tuple[int, int, int, Path]] = []
    for index, (init, minimum) in enumerate(RUN_MATRIX):
        run_root = output_root / "runs" / _run_id(index, init, minimum)
        valid = _completion_is_valid(run_root, plan_hash=plan_hash)
        if valid and not force:
            logger.info("verified completed run; skipping %s", run_root.name)
            continue
        marker_path = run_root / "stage_morse_complete.json"
        if marker_path.exists() and not valid and not force:
            raise RuntimeError(
                f"existing completion marker failed validation: {marker_path}; "
                "use --force to recompute or choose a new output root"
            )
        incomplete.append((index, init, minimum, run_root))
    if not incomplete:
        precompute_path = output_root / "shared_precompute.json"
        try:
            precompute_record = json.loads(precompute_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"missing or invalid shared precompute record: {precompute_path}"
            ) from exc
        sweep_summary = _write_sweep_summary(
            output_root,
            plan_hash=plan_hash,
            precompute_record=precompute_record,
            failures=[],
        )
        if sweep_summary["status"] != "complete":
            raise RuntimeError("completion markers did not yield a complete sweep summary")
        logger.info("all six runs are already complete")
        return 0

    execution_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    execution_provenance = {
        **_environment_record(),
        "execution_id": execution_id,
        "plan_sha256": plan_hash,
        "force": force,
        "scheduled_run_ids": [
            _run_id(index, init, minimum)
            for index, init, minimum, _run_root in incomplete
        ],
    }
    _write_json_atomic(
        output_root / "provenance" / f"execution_{execution_id}.json",
        execution_provenance,
    )

    dynamics = _load_dynamics()
    logger.info(
        "building one shared level-%d corner table (%s points, %.2f GiB table)",
        SUBDIV_MAX,
        f"{plan['box_map']['corner_point_count']:,}",
        plan["box_map"]["estimated_table_bytes"] / (1024**3),
    )
    precompute_started = _utc_now()
    precompute_t0 = time.perf_counter()
    with PeakRSSMonitor() as monitor:
        box_map = CMGDB.make_precomputed_box_map(
            dynamics,
            LOWER,
            UPPER,
            subdiv_max=SUBDIV_MAX,
            mode="adaptive",
            eval_mode=EVAL_MODE,
            padding=PADDING,
            batch_points=batch_points,
            device="cpu",
        )
    precompute_seconds = time.perf_counter() - precompute_t0
    precompute_record = {
        "status": "complete",
        "started_utc": precompute_started,
        "completed_utc": _utc_now(),
        "duration_seconds": precompute_seconds,
        "peak_rss_bytes": monitor.peak_bytes,
        **plan["box_map"],
    }
    _write_json_atomic(output_root / "shared_precompute.json", precompute_record)
    logger.info(
        "shared precompute complete in %.2f s; sampled peak RSS %.2f GiB",
        precompute_seconds,
        monitor.peak_bytes / (1024**3),
    )

    failures: list[str] = []
    cell_summaries: list[dict[str, Any]] = []
    for index, init, minimum, run_root in incomplete:
        run_id = _run_id(index, init, minimum)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "logs").mkdir(exist_ok=True)
        checkpoint_records = _copy_models(run_root, overwrite=force)
        logger.info(
            "starting %s: init=%d min=%d max=%d limit=%d",
            run_id,
            init,
            minimum,
            SUBDIV_MAX,
            SUBDIV_LIMIT,
        )
        started_utc = _utc_now()
        t0 = time.perf_counter()
        try:
            model = CMGDB.Model(
                minimum,
                SUBDIV_MAX,
                init,
                SUBDIV_LIMIT,
                LOWER.tolist(),
                UPPER.tolist(),
                box_map,
            )
            if not hasattr(model, "set_batch_map") or not hasattr(box_map, "batch"):
                raise RuntimeError("CMGDB batch-map interface is unavailable")
            model.set_batch_map(box_map.batch)
            compute_only = getattr(CMGDB, "ComputeConleyMorseGraphOnly", None)
            if not callable(compute_only):
                raise RuntimeError("CMGDB.ComputeConleyMorseGraphOnly is unavailable")
            with PeakRSSMonitor() as run_monitor:
                morse_graph = compute_only(model)
            graph_seconds = time.perf_counter() - t0
            mg_root = run_root / "MG"
            dot_path, sets_path = save_morse_graph_artifacts(morse_graph, mg_root)
            graph_summary = _graph_summary(dot_path)
            set_summary = _morse_set_summary(sets_path)
            if set(graph_summary["nodes"]) != set(set_summary["labels"]):
                raise ValueError(
                    f"graph/set label mismatch for {run_id}: "
                    f"{graph_summary['nodes']} vs {set_summary['labels']}"
                )
            _write_parameter_log(
                run_root / "mg_params_log.txt",
                init=init,
                minimum=minimum,
                graph_seconds=graph_seconds,
                shared_precompute_seconds=precompute_seconds,
            )
            artifacts = {
                "morse_graph": _artifact_record(dot_path, relative_to=run_root),
                "morse_sets": _artifact_record(sets_path, relative_to=run_root),
                "parameter_log": _artifact_record(
                    run_root / "mg_params_log.txt", relative_to=run_root
                ),
            }
            summary = {
                "schema_version": 1,
                "status": "complete",
                "plan_sha256": plan_hash,
                "index": index,
                "run_id": run_id,
                "parameters": {
                    "subdiv_init": init,
                    "subdiv_min": minimum,
                    "subdiv_max": SUBDIV_MAX,
                    "subdiv_limit": SUBDIV_LIMIT,
                    "bounds_lower": LOWER.tolist(),
                    "bounds_upper": UPPER.tolist(),
                    "padding": PADDING,
                    "eval_mode": EVAL_MODE,
                },
                "timing": {
                    "shared_precompute_seconds": precompute_seconds,
                    "graph_seconds": graph_seconds,
                },
                "memory": {"sampled_peak_rss_bytes": run_monitor.peak_bytes},
                "started_utc": started_utc,
                "completed_utc": _utc_now(),
                "graph": graph_summary,
                "morse_sets": set_summary,
                "checkpoints": checkpoint_records,
                "artifacts": artifacts,
            }
            _write_json_atomic(run_root / "cell_summary.json", summary)
            summary["artifacts"]["cell_summary"] = _artifact_record(
                run_root / "cell_summary.json", relative_to=run_root
            )
            _write_json_atomic(run_root / "run_manifest.json", summary)
            completion_artifacts = {
                **artifacts,
                "cell_summary": _artifact_record(
                    run_root / "cell_summary.json", relative_to=run_root
                ),
                "run_manifest": _artifact_record(
                    run_root / "run_manifest.json", relative_to=run_root
                ),
            }
            _write_json_atomic(
                run_root / "stage_morse_complete.json",
                {
                    "schema_version": 1,
                    "status": "complete",
                    "plan_sha256": plan_hash,
                    "run_id": run_id,
                    "completed_utc": _utc_now(),
                    "artifacts": completion_artifacts,
                },
            )
            cell_summaries.append(summary)
            logger.info(
                "completed %s in %.2f s: %d nodes, %d edges, %d boxes; peak RSS %.2f GiB",
                run_id,
                graph_seconds,
                graph_summary["node_count"],
                graph_summary["edge_count"],
                set_summary["row_count"],
                run_monitor.peak_bytes / (1024**3),
            )
            del morse_graph, model
            gc.collect()
        except Exception as exc:
            failures.append(run_id)
            failure = {
                "schema_version": 1,
                "status": "failed",
                "plan_sha256": plan_hash,
                "run_id": run_id,
                "parameters": {
                    "subdiv_init": init,
                    "subdiv_min": minimum,
                    "subdiv_max": SUBDIV_MAX,
                    "subdiv_limit": SUBDIV_LIMIT,
                },
                "started_utc": started_utc,
                "failed_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_json_atomic(run_root / "run_failure.json", failure)
            logger.exception("failed %s", run_id)
            gc.collect()

    sweep_summary = _write_sweep_summary(
        output_root,
        plan_hash=plan_hash,
        precompute_record=precompute_record,
        failures=failures,
    )
    completed_count = int(sweep_summary["completed_run_count"])
    if failures or completed_count != len(RUN_MATRIX):
        logger.error(
            "sweep incomplete: %d/%d complete; failures=%s",
            completed_count,
            len(RUN_MATRIX),
            failures,
        )
        return 1
    logger.info("all six CMGDB runs completed successfully")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-points", type=int, default=PRECOMPUTE_BATCH_POINTS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly recompute and overwrite all six completed cells",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the frozen plan without creating output or running CMGDB",
    )
    args = parser.parse_args()
    if args.batch_points < 1:
        parser.error("--batch-points must be positive")
    output_root = args.output_root.resolve()
    if args.plan_only:
        plan = _build_plan(output_root=output_root, batch_points=args.batch_points)
        print(json.dumps(plan, indent=2))
        return 0
    return run_sweep(
        output_root=output_root,
        batch_points=args.batch_points,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
