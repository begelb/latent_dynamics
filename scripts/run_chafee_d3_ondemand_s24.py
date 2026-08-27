"""Run one exact uniform-s24 D3 CMGDB graph with a neural BoxMap.

This executable is intentionally narrower than the D3 BoxMap benchmark:

* the only accepted graph resolution is uniform 24/24/24;
* the target is the completed ``dataset_01_seed_00`` training attempt;
* all fifteen matched D3 training completion markers must exist first;
* the latent map is evaluated on demand through ``Model.set_batch_map``;
* no corner table or other precomputed BoxMap is constructed;
* missing native cache support or any scalar callback is a hard failure; and
* output must be a fresh directory disjoint from every training input.

Without ``--execute`` the script performs read-only preflight validation and
prints the frozen plan.  It never launches CMGDB implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

TRAINING_ROOT = (
    CODE_ROOT / "output" / "chafee_d3_matched_d2_archive_5x3_training_v1"
)
TARGET_RUN_ID = "dataset_01_seed_00_lr3e3_e4000"
DEFAULT_RUN_ROOT = TRAINING_ROOT / "runs" / TARGET_RUN_ID
DEFAULT_OUTPUT_ROOT = (
    CODE_ROOT / "output" / "chafee_d3_ondemand_s24_dataset_01_seed_00_v1"
)
LEGACY_D3_ROOT = (
    CODE_ROOT
    / "output"
    / "chafee_latent_dimension_study"
    / "latent_3d"
    / "seed_0"
)

DIMENSION = 3
HIGH_DIMENSION = 64
TRAINING_ROWS = 30_000
SUBDIVISION = 24
SUBDIV_LIMIT = 10_000
EXPECTED_CELLS = 2**SUBDIVISION
CELLS_PER_AXIS = 2 ** (SUBDIVISION // DIMENSION)
NATIVE_BATCH_CHUNK = 100_000
EXPECTED_CALLBACK_RECTANGLES = 2 * EXPECTED_CELLS
EXPECTED_NEURAL_CORNER_POINTS = (
    EXPECTED_CALLBACK_RECTANGLES * (2**DIMENSION)
)
EXPECTED_BATCH_CALLS = 2 * (
    (EXPECTED_CELLS + NATIVE_BATCH_CHUNK - 1) // NATIVE_BATCH_CHUNK
)
BOUNDS_EPSILON_FRAC = 0.1
PADDING = True
EXPECTED_TARGET_LOWER = np.asarray(
    (-3.9116177558898926, -2.707045078277588, -3.076873302459717),
    dtype=np.float64,
)
EXPECTED_TARGET_UPPER = np.asarray(
    (3.6786651611328125, 2.518048048019409, 3.3541271686553955),
    dtype=np.float64,
)
DEFAULT_RESERVE_EDGES = 1_200_000_000
DEFAULT_MAX_FORWARD_POINTS = 800_000
DEFAULT_RSS_SAMPLE_SECONDS = 0.1
EXPECTED_TRAINING_RUNS = 15
SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"cannot parse JSON from {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return path


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _file_record(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = _sha256(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for {resolved}: expected {expected_sha256}, "
            f"observed {digest}"
        )
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": int(resolved.stat().st_size),
    }


def _safe_relative_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise ValueError(f"artifact path escapes {resolved_root}: {relative!r}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _paths_overlap(left: Path, right: Path) -> bool:
    a = left.resolve()
    b = right.resolve()
    return a == b or a in b.parents or b in a.parents


@dataclass(frozen=True)
class TargetInputs:
    training_root: Path
    run_root: Path
    attempt_root: Path
    checkpoint: Path
    checkpoint_sidecar: Path
    train_data: Path
    plan_sha256: str
    source_records: dict[str, dict[str, Any]]
    matrix_status: dict[str, Any]


def _matrix_completion_status(
    training_root: Path,
    experiment_plan: Mapping[str, Any],
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    plan = experiment_plan.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("training experiment plan has no object-valued plan")
    trials = plan.get("trials")
    if not isinstance(trials, list) or len(trials) != EXPECTED_TRAINING_RUNS:
        raise ValueError(
            "training experiment plan must contain exactly "
            f"{EXPECTED_TRAINING_RUNS} trials"
        )
    completed: list[str] = []
    incomplete: list[str] = []
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("malformed trial in training experiment plan")
        run_id = str(trial.get("run_id"))
        marker_path = training_root / "runs" / run_id / "completed.json"
        if not marker_path.is_file():
            incomplete.append(run_id)
            continue
        marker = _read_json(marker_path)
        if (
            marker.get("status") != "completed"
            or marker.get("plan_sha256") != plan_sha256
        ):
            incomplete.append(run_id)
            continue
        completed.append(run_id)
    return {
        "required_runs": EXPECTED_TRAINING_RUNS,
        "completed_runs": len(completed),
        "complete": len(completed) == EXPECTED_TRAINING_RUNS,
        "completed_run_ids": completed,
        "incomplete_run_ids": incomplete,
    }


def resolve_target_inputs(run_root: Path = DEFAULT_RUN_ROOT) -> TargetInputs:
    """Resolve and hash the immutable target attempt from its completion marker."""

    resolved_run = run_root.resolve()
    if resolved_run.name != TARGET_RUN_ID:
        raise ValueError(
            f"this runner accepts only {TARGET_RUN_ID}; got {resolved_run.name}"
        )
    training_root = resolved_run.parents[1]
    experiment_plan_path = training_root / "experiment_plan.json"
    experiment_plan = _read_json(experiment_plan_path)
    plan = experiment_plan.get("plan")
    plan_sha256 = str(experiment_plan.get("plan_sha256", ""))
    if not isinstance(plan, dict) or not plan_sha256:
        raise ValueError("malformed training experiment plan envelope")
    if _payload_sha256(plan) != plan_sha256:
        raise ValueError("training experiment plan body hash mismatch")

    matrix_status = _matrix_completion_status(
        training_root,
        experiment_plan,
        plan_sha256=plan_sha256,
    )
    trial_records = [
        trial
        for trial in plan.get("trials", [])
        if isinstance(trial, dict) and trial.get("run_id") == TARGET_RUN_ID
    ]
    if len(trial_records) != 1:
        raise ValueError(f"training plan does not identify {TARGET_RUN_ID} once")
    trial = trial_records[0]
    if int(trial.get("dataset", -1)) != 1 or int(
        trial.get("training_seed", -1)
    ) != 0:
        raise ValueError("target trial is not dataset 1, training seed 0")

    run_spec_path = resolved_run / "run_spec.json"
    run_spec = _read_json(run_spec_path)
    if (
        run_spec.get("plan_sha256") != plan_sha256
        or run_spec.get("run", {}).get("run_id") != TARGET_RUN_ID
    ):
        raise ValueError("target run specification differs from the frozen plan")

    completed_path = resolved_run / "completed.json"
    completed = _read_json(completed_path)
    if (
        completed.get("status") != "completed"
        or completed.get("plan_sha256") != plan_sha256
        or completed.get("run", {}).get("run_id") != TARGET_RUN_ID
    ):
        raise ValueError("target completion marker is not valid")
    attempt_number = int(completed.get("attempt", 0))
    attempt_root = (
        resolved_run / "attempts" / f"attempt_{attempt_number:03d}"
    ).resolve()
    if resolved_run not in attempt_root.parents:
        raise ValueError("resolved attempt root escapes the target run")

    checkpoint_entry = completed.get("checkpoint")
    if not isinstance(checkpoint_entry, dict):
        raise ValueError("target completion marker has no checkpoint record")
    checkpoint = _safe_relative_file(
        resolved_run,
        str(checkpoint_entry.get("path")),
    )
    expected_checkpoint = attempt_root / "models" / "autoencoder.pt"
    if checkpoint != expected_checkpoint:
        raise ValueError(
            f"completion marker selected {checkpoint}, expected {expected_checkpoint}"
        )
    checkpoint_record = _file_record(
        checkpoint,
        expected_sha256=str(checkpoint_entry.get("sha256")),
    )
    if checkpoint_record["size_bytes"] != int(
        checkpoint_entry.get("size_bytes", -1)
    ):
        raise ValueError("checkpoint size differs from completion marker")

    checkpoint_sidecar = checkpoint.with_suffix(".json")
    sidecar = _read_json(checkpoint_sidecar)
    sidecar_arch = sidecar.get("arch")
    if (
        sidecar.get("version") != 1
        or not isinstance(sidecar_arch, dict)
        or int(sidecar_arch.get("low_dims", -1)) != DIMENSION
        or int(sidecar_arch.get("high_dims", -1)) != HIGH_DIMENSION
        or sidecar_arch != plan.get("architecture")
    ):
        raise ValueError("target checkpoint sidecar architecture is not frozen D3")

    training_summary_path = attempt_root / "training_summary.json"
    training_summary = _read_json(training_summary_path)
    if (
        training_summary.get("arch") != sidecar_arch
        or int(training_summary.get("seed", -1)) != 0
        or int(training_summary.get("epochs_completed", -1)) != 4_000
    ):
        raise ValueError("target training summary differs from its checkpoint")

    artifact_entry = completed.get("artifact_manifest")
    if not isinstance(artifact_entry, dict):
        raise ValueError("target completion marker has no artifact manifest")
    artifact_manifest_path = _safe_relative_file(
        resolved_run,
        str(artifact_entry.get("path")),
    )
    if attempt_root not in artifact_manifest_path.parents:
        raise ValueError("artifact manifest does not belong to the target attempt")

    dataset_record = plan.get("sources", {}).get("train_data_dataset_1")
    if not isinstance(dataset_record, dict):
        raise ValueError("training plan has no dataset-1 source record")
    train_data = Path(str(dataset_record.get("path"))).resolve()

    sources = {
        "training_experiment_plan": _file_record(experiment_plan_path),
        "target_run_spec": _file_record(run_spec_path),
        "target_completion_marker": _file_record(completed_path),
        "target_artifact_manifest": _file_record(
            artifact_manifest_path,
            expected_sha256=str(artifact_entry.get("sha256")),
        ),
        "target_training_summary": _file_record(training_summary_path),
        "checkpoint": checkpoint_record,
        "checkpoint_sidecar": _file_record(checkpoint_sidecar),
        "train_data": _file_record(
            train_data,
            expected_sha256=str(dataset_record.get("sha256")),
        ),
        "runner": _file_record(SCRIPT_PATH),
    }
    return TargetInputs(
        training_root=training_root.resolve(),
        run_root=resolved_run,
        attempt_root=attempt_root,
        checkpoint=checkpoint,
        checkpoint_sidecar=checkpoint_sidecar,
        train_data=train_data,
        plan_sha256=plan_sha256,
        source_records=sources,
        matrix_status=matrix_status,
    )


def assert_safe_fresh_output(output_root: Path, inputs: TargetInputs) -> Path:
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    protected = (
        inputs.training_root,
        inputs.run_root,
        inputs.attempt_root,
        LEGACY_D3_ROOT.resolve(),
    )
    for path in protected:
        if _paths_overlap(output, path):
            raise ValueError(
                f"output {output} overlaps protected input or broad root {path}"
            )
    if output in (CODE_ROOT.resolve(), (CODE_ROOT / "output").resolve()):
        raise ValueError("output must be a fresh dedicated subdirectory")
    if output_root.is_symlink():
        raise ValueError("output root must not be a symlink")
    return output


def _resolve_device(name: str) -> Any:
    import torch

    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_training_pairs(
    train_data: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pairs = np.loadtxt(train_data, delimiter=",", dtype=np.float64)
    expected = (TRAINING_ROWS, 2 * HIGH_DIMENSION)
    if pairs.shape != expected:
        raise ValueError(f"{train_data} has shape {pairs.shape}; expected {expected}")
    if not np.all(np.isfinite(pairs)):
        raise ValueError(f"{train_data} contains non-finite values")
    return (
        np.ascontiguousarray(pairs[:, :HIGH_DIMENSION]),
        np.ascontiguousarray(pairs[:, HIGH_DIMENSION:]),
    )


def compute_cpu_bounds(
    encoder: Any,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exactly reproduce the established one-call CPU bounds computation."""

    import torch

    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape != y.shape
        or x.shape[1] != HIGH_DIMENSION
    ):
        raise ValueError(
            "bounds inputs must be matching (n, 64) current/next arrays"
        )
    encoder = encoder.to(torch.device("cpu"))
    encoder.eval()
    # This deliberately uses one 60,000-row encoder call. Splitting the GEMM
    # into smaller batches changes float32 rounding by roughly 1e-7 and would
    # make this graph's domain differ from the established D3 protocol.
    all_states = np.concatenate((x, y), axis=0)
    with torch.no_grad():
        tensor = torch.as_tensor(
            all_states,
            dtype=torch.float32,
            device="cpu",
        )
        encoded = encoder(tensor).detach().cpu().numpy()
    lower = encoded.min(axis=0)
    upper = encoded.max(axis=0)
    if not np.all(np.isfinite(lower)) or not np.all(lower < upper):
        raise ValueError("encoder produced invalid latent bounds")
    buffer = np.float32(BOUNDS_EPSILON_FRAC) * (upper - lower)
    return (
        (lower - buffer).astype(np.float64),
        (upper + buffer).astype(np.float64),
    )


class NeuralEvaluator:
    """Batched Torch evaluator with exact point/time/accelerator counters."""

    def __init__(self, latent_map: Any, device: Any):
        import torch

        self.torch = torch
        self.device = device
        self.latent_map = latent_map.to(device)
        self.latent_map.eval()
        self.forward_calls = 0
        self.points = 0
        self.forward_seconds = 0.0
        self.max_forward_points_observed = 0
        self.accelerator_peak_allocated_bytes: int | None = None
        self.accelerator_peak_driver_allocated_bytes: int | None = None

    def _sample_accelerator_memory(self) -> None:
        allocated: int | None = None
        driver: int | None = None
        if self.device.type == "mps":
            current = getattr(self.torch.mps, "current_allocated_memory", None)
            driver_current = getattr(
                self.torch.mps,
                "driver_allocated_memory",
                None,
            )
            if callable(current):
                allocated = int(current())
            if callable(driver_current):
                driver = int(driver_current())
        elif self.device.type == "cuda":
            allocated = int(self.torch.cuda.max_memory_allocated(self.device))
            reserved = getattr(self.torch.cuda, "max_memory_reserved", None)
            if callable(reserved):
                driver = int(reserved(self.device))
        if allocated is not None:
            self.accelerator_peak_allocated_bytes = max(
                allocated,
                self.accelerator_peak_allocated_bytes or 0,
            )
        if driver is not None:
            self.accelerator_peak_driver_allocated_bytes = max(
                driver,
                self.accelerator_peak_driver_allocated_bytes or 0,
            )

    def __call__(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        values = np.ascontiguousarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != DIMENSION:
            raise ValueError(
                f"latent points must have shape (n, {DIMENSION}); got {values.shape}"
            )
        started = time.perf_counter()
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(
                values,
                dtype=self.torch.float32,
                device=self.device,
            )
            mapped = self.latent_map(tensor).detach().cpu().numpy()
        elapsed = time.perf_counter() - started
        result = np.asarray(mapped, dtype=np.float64)
        if result.shape != values.shape or not np.all(np.isfinite(result)):
            raise ValueError(
                f"latent map returned invalid shape/data: {result.shape}"
            )
        n_points = int(values.shape[0])
        self.forward_calls += 1
        self.points += n_points
        self.forward_seconds += elapsed
        self.max_forward_points_observed = max(
            self.max_forward_points_observed,
            n_points,
        )
        self._sample_accelerator_memory()
        return result

    def reset_counters(self) -> None:
        self.forward_calls = 0
        self.points = 0
        self.forward_seconds = 0.0
        self.max_forward_points_observed = 0
        self.accelerator_peak_allocated_bytes = None
        self.accelerator_peak_driver_allocated_bytes = None

    def stats(self) -> dict[str, Any]:
        return {
            "neural_forward_calls": self.forward_calls,
            "neural_corner_points": self.points,
            "neural_forward_seconds": self.forward_seconds,
            "max_forward_points_observed": self.max_forward_points_observed,
            "accelerator_peak_allocated_bytes": (
                self.accelerator_peak_allocated_bytes
            ),
            "accelerator_peak_driver_allocated_bytes": (
                self.accelerator_peak_driver_allocated_bytes
            ),
        }


class OnDemandNeuralBoxMap:
    """Evaluate all eight D3 box corners in vectorized neural batches."""

    def __init__(
        self,
        evaluator: NeuralEvaluator | Callable[
            [NDArray[np.float64]],
            NDArray[np.float64],
        ],
        *,
        max_forward_points: int = DEFAULT_MAX_FORWARD_POINTS,
        padding: bool = PADDING,
    ):
        if max_forward_points < 2**DIMENSION:
            raise ValueError("max_forward_points must fit at least one D3 box")
        self.evaluator = evaluator
        self.max_forward_points = int(max_forward_points)
        self.padding = bool(padding)
        self.corner_bits = np.asarray(
            list(product((0, 1), repeat=DIMENSION)),
            dtype=bool,
        )
        self.scalar_calls = 0
        self.batch_calls = 0
        self.rectangles = 0
        self.callback_seconds = 0.0
        self.max_batch_rectangles_observed = 0

    def __call__(self, rect: Any) -> list[float]:
        self.scalar_calls += 1
        return self._map(np.asarray(rect, dtype=np.float64).reshape(1, -1))[0]

    def batch(self, rects: Any) -> list[list[float]]:
        self.batch_calls += 1
        return self._map(np.asarray(rects, dtype=np.float64))

    def _map(self, rects: NDArray[np.float64]) -> list[list[float]]:
        started = time.perf_counter()
        if rects.ndim != 2 or rects.shape[1] != 2 * DIMENSION:
            raise ValueError(
                f"rectangles must have shape (n, {2 * DIMENSION}); "
                f"got {rects.shape}"
            )
        if not np.all(np.isfinite(rects)):
            raise ValueError("rectangles contain non-finite coordinates")
        n_rectangles = int(rects.shape[0])
        lower = rects[:, :DIMENSION]
        upper = rects[:, DIMENSION:]
        widths = upper - lower
        if np.any(widths <= 0.0):
            raise ValueError("all rectangle widths must be positive")

        points_per_rectangle = 2**DIMENSION
        rectangles_per_forward = max(
            1,
            self.max_forward_points // points_per_rectangle,
        )
        output = np.empty(
            (n_rectangles, 2 * DIMENSION),
            dtype=np.float64,
        )
        for start in range(0, n_rectangles, rectangles_per_forward):
            stop = min(start + rectangles_per_forward, n_rectangles)
            chunk_lower = lower[start:stop, None, :]
            chunk_upper = upper[start:stop, None, :]
            corners = np.where(
                self.corner_bits[None, :, :],
                chunk_upper,
                chunk_lower,
            )
            flat_corners = corners.reshape(
                (stop - start) * points_per_rectangle,
                DIMENSION,
            )
            mapped = np.asarray(
                self.evaluator(flat_corners),
                dtype=np.float64,
            ).reshape(stop - start, points_per_rectangle, DIMENSION)
            image_lower = mapped.min(axis=1)
            image_upper = mapped.max(axis=1)
            if self.padding:
                image_lower -= widths[start:stop]
                image_upper += widths[start:stop]
            output[start:stop] = np.concatenate(
                (image_lower, image_upper),
                axis=1,
            )

        self.rectangles += n_rectangles
        self.max_batch_rectangles_observed = max(
            self.max_batch_rectangles_observed,
            n_rectangles,
        )
        self.callback_seconds += time.perf_counter() - started
        return output.tolist()

    def stats(self) -> dict[str, Any]:
        evaluator_stats = (
            self.evaluator.stats()
            if isinstance(self.evaluator, NeuralEvaluator)
            else {}
        )
        return {
            "scalar_calls": self.scalar_calls,
            "batch_calls": self.batch_calls,
            "rectangles": self.rectangles,
            "callback_seconds": self.callback_seconds,
            "max_batch_rectangles_observed": (
                self.max_batch_rectangles_observed
            ),
            "configured_max_forward_points": self.max_forward_points,
            **evaluator_stats,
        }


class PeakRSSSampler:
    """Sample process RSS in a background thread; ru_maxrss remains fallback."""

    def __init__(self, interval_seconds: float):
        if interval_seconds <= 0.0:
            raise ValueError("RSS sampling interval must be positive")
        self.interval_seconds = float(interval_seconds)
        self.available = False
        self.error: str | None = None
        self.baseline_bytes: int | None = None
        self.peak_bytes: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil

            self._process = psutil.Process()
            self.available = True
        except (ImportError, OSError) as error:
            self._process = None
            self.error = f"{type(error).__name__}: {error}"

    def _sample(self) -> None:
        if self._process is None:
            return
        try:
            rss = int(self._process.memory_info().rss)
        except OSError as error:
            self.error = f"{type(error).__name__}: {error}"
            return
        if self.baseline_bytes is None:
            self.baseline_bytes = rss
        self.peak_bytes = max(rss, self.peak_bytes or 0)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        if self.available:
            self._thread = threading.Thread(
                target=self._run,
                name="peak-rss-sampler",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.interval_seconds))

    def stats(self) -> dict[str, Any]:
        return {
            "rss_sampler_available": self.available,
            "rss_sampler_error": self.error,
            "rss_sample_interval_seconds": self.interval_seconds,
            "rss_samples": self.samples,
            "baseline_rss_bytes": self.baseline_bytes,
            "sampled_peak_rss_bytes": self.peak_bytes,
            "ru_maxrss_bytes": _ru_maxrss_bytes(),
        }


def _ru_maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _usage_snapshot() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "process_seconds": time.process_time(),
        "user_seconds": float(usage.ru_utime),
        "system_seconds": float(usage.ru_stime),
    }


def _usage_delta(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: float(after[key] - before[key])
        for key in ("process_seconds", "user_seconds", "system_seconds")
    }


def _graph_summary(morse_graph: Any, map_graph: Any) -> dict[str, Any]:
    vertices = sorted(int(value) for value in morse_graph.vertices())
    unreduced_method = getattr(morse_graph, "edges_unreduced", None)
    if not callable(unreduced_method):
        raise RuntimeError(
            "MorseGraph.edges_unreduced is required to avoid recomputing "
            "the native transitive reduction"
        )
    unreduced_edges = sorted(
        (int(source), int(target))
        for source, target in unreduced_method()
    )
    nonminimal = {
        source for source, target in unreduced_edges if source != target
    }
    minimal_nodes = [node for node in vertices if node not in nonminimal]
    morse_set_sizes = {
        str(node): len(morse_graph.morse_set(node)) for node in vertices
    }
    core = {
        "map_cells": int(map_graph.num_vertices()),
        "cached_edges": int(map_graph.num_cached_edges()),
        "morse_nodes": len(vertices),
        "morse_unreduced_edge_count": len(unreduced_edges),
        "morse_unreduced_edges_sha256": _payload_sha256(unreduced_edges),
        "minimal_nodes": minimal_nodes,
        "morse_set_sizes": morse_set_sizes,
    }
    core["graph_fingerprint"] = _payload_sha256(core)
    return core


def _runtime_provenance(device: Any) -> dict[str, Any]:
    import CMGDB
    import torch

    native_path = Path(CMGDB._cmgdb.__file__).resolve()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cmgdb_python": _file_record(Path(CMGDB.__file__)),
        "cmgdb_native": _file_record(native_path),
    }


def build_plan(
    inputs: TargetInputs,
    *,
    output_root: Path,
    device: str,
    reserve_edges: int,
    max_forward_points: int,
    rss_sample_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "one exact uniform-s24 D3 CMGDB computation using a batched "
            "on-demand neural BoxMap"
        ),
        "created_at_utc": _utc_now(),
        "target": {
            "run_id": TARGET_RUN_ID,
            "training_plan_sha256": inputs.plan_sha256,
            "run_root": str(inputs.run_root),
            "attempt_root": str(inputs.attempt_root),
        },
        "training_matrix": inputs.matrix_status,
        "computation": {
            "backend": "batched_on_demand_neural",
            "precomputed": False,
            "dimension": DIMENSION,
            "subdiv_init": SUBDIVISION,
            "subdiv_min": SUBDIVISION,
            "subdiv_max": SUBDIVISION,
            "subdiv_limit": SUBDIV_LIMIT,
            "cells_per_axis": CELLS_PER_AXIS,
            "expected_cells": EXPECTED_CELLS,
            "padding": PADDING,
            "bounds_epsilon_frac": BOUNDS_EPSILON_FRAC,
            "bounds_device": "cpu",
            "neural_device_requested": device,
            "max_forward_points": max_forward_points,
            "reserve_edges": reserve_edges,
            "cache_map_graph": True,
            "conley_indices": False,
        },
        "instrumentation": {
            "wall_time": True,
            "process_cpu_time": True,
            "resource_user_system_cpu_time": True,
            "sampled_peak_rss": True,
            "rss_sample_seconds": rss_sample_seconds,
            "resource_ru_maxrss_fallback": True,
            "callback_counts": [
                "scalar_calls",
                "batch_calls",
                "rectangles",
                "neural_forward_calls",
                "neural_corner_points",
            ],
        },
        "hard_postconditions": {
            "all_15_training_runs_complete_before_launch": True,
            "map_cells_equal_2_pow_24": True,
            "native_map_graph_cache_present": True,
            "scalar_callback_calls_equal_zero": True,
            "batch_callback_calls_positive": True,
            "cached_edges_positive": True,
            "callback_rectangles_equal_two_full_cell_passes": (
                EXPECTED_CALLBACK_RECTANGLES
            ),
            "neural_corner_points_equal_eight_per_rectangle": (
                EXPECTED_NEURAL_CORNER_POINTS
            ),
        },
        "output_root": str(output_root.resolve()),
        "sources": inputs.source_records,
    }


def run_exact_s24(
    *,
    inputs: TargetInputs,
    output_root: Path,
    device_name: str,
    reserve_edges: int,
    max_forward_points: int,
    rss_sample_seconds: float,
) -> dict[str, Any]:
    """Execute the single authorized production graph and persist its artifacts."""

    import CMGDB

    from latentdynamics.training import load_checkpoint
    from latentdynamics.viz import save_morse_graph_artifacts

    if not inputs.matrix_status["complete"]:
        raise RuntimeError(
            "all 15 matched D3 trainings must complete before s24; still "
            f"missing {inputs.matrix_status['incomplete_run_ids']}"
        )
    if reserve_edges < 1:
        raise ValueError("reserve_edges must be positive")
    if max_forward_points < 2**DIMENSION:
        raise ValueError("max_forward_points is too small")

    output = assert_safe_fresh_output(output_root, inputs)
    output.mkdir(parents=True, exist_ok=False)
    plan = build_plan(
        inputs,
        output_root=output,
        device=device_name,
        reserve_edges=reserve_edges,
        max_forward_points=max_forward_points,
        rss_sample_seconds=rss_sample_seconds,
    )
    plan_path = _write_json_exclusive(output / "launch_manifest.json", plan)
    status_path = output / "status.json"
    _write_json_exclusive(
        status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "phase": "load_checkpoint_and_compute_bounds",
            "started_at_utc": _utc_now(),
            "launch_manifest": _file_record(plan_path),
        },
    )

    started_wall = time.perf_counter()
    started_cpu = _usage_snapshot()
    sampler = PeakRSSSampler(rss_sample_seconds)
    sampler.start()
    try:
        # Bounds intentionally use the CPU checkpoint and all current/next
        # training states, matching the established D3 study's 10% rule.
        model, arch = load_checkpoint(
            inputs.checkpoint.parent,
            map_location="cpu",
        )
        if int(arch.low_dims) != DIMENSION or int(arch.high_dims) != HIGH_DIMENSION:
            raise ValueError("loaded checkpoint is not the expected 64-to-3 model")
        model.eval()
        x, y = _load_training_pairs(inputs.train_data)
        bounds_started = time.perf_counter()
        lower, upper = compute_cpu_bounds(model.encoder, x, y)
        bounds_seconds = time.perf_counter() - bounds_started
        if not np.array_equal(
            lower,
            EXPECTED_TARGET_LOWER,
        ) or not np.array_equal(upper, EXPECTED_TARGET_UPPER):
            raise ValueError(
                "computed target bounds differ from the independently "
                "verified dataset-1/seed-0 bounds"
            )
        bounds_payload = {
            "schema_version": SCHEMA_VERSION,
            "dimension": DIMENSION,
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "epsilon_frac": BOUNDS_EPSILON_FRAC,
            "source": "encoder(train_data current states + next states)",
            "n_encoded_states": 2 * TRAINING_ROWS,
            "encoder_device": "cpu",
            "checkpoint_sha256": inputs.source_records["checkpoint"]["sha256"],
            "train_data_sha256": inputs.source_records["train_data"]["sha256"],
            "independent_reference": {
                "lower": EXPECTED_TARGET_LOWER.tolist(),
                "upper": EXPECTED_TARGET_UPPER.tolist(),
                "comparison": "exact float64 values promoted from float32",
                "matched": True,
            },
        }
        bounds_path = _write_json_exclusive(
            output / "bounds.json",
            bounds_payload,
        )
        bounds_record = _file_record(bounds_path)
        del x, y

        device = _resolve_device(device_name)
        latent_map = model.latent_map.to(device)
        latent_map.eval()
        del model
        evaluator = NeuralEvaluator(latent_map, device)
        warmup_points = np.stack(
            (
                lower,
                upper,
                0.5 * (lower + upper),
                lower + 0.25 * (upper - lower),
            ),
            axis=0,
        )
        warmup_started = time.perf_counter()
        evaluator(warmup_points)
        warmup_seconds = time.perf_counter() - warmup_started
        evaluator.reset_counters()
        box_map = OnDemandNeuralBoxMap(
            evaluator,
            max_forward_points=max_forward_points,
            padding=PADDING,
        )

        # The edge-buffer reservation is passed straight to CMGDB's
        # reserve_edges option (a sizing hint only -- exceeding it grows the
        # buffer, it does not fail), keeping timing and peak-memory
        # comparisons apples-to-apples across runs.
        runtime = _runtime_provenance(device)
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "recorded_at_utc": _utc_now(),
            "launch_manifest": _file_record(plan_path),
            "bounds": bounds_record,
            "sources": inputs.source_records,
            "runtime": runtime,
            "cmgdb_options": {
                "reserve_edges": reserve_edges,
                "cache_map_graph": True,
            },
        }
        provenance_path = _write_json_exclusive(
            output / "provenance.json",
            provenance,
        )

        _write_json_atomic(
            status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "phase": "CMGDB.ComputeMorseGraph",
                "started_at_utc": _read_json(status_path)["started_at_utc"],
                "bounds": bounds_record,
                "device": str(device),
            },
        )
        print(
            "Launching exact D3 uniform 24/24/24 on-demand CMGDB "
            f"({EXPECTED_CELLS:,} cells) on {device}.",
            flush=True,
        )
        cmgdb_model = CMGDB.Model(
            SUBDIVISION,
            SUBDIVISION,
            SUBDIVISION,
            SUBDIV_LIMIT,
            lower.tolist(),
            upper.tolist(),
            box_map,
        )
        if not hasattr(cmgdb_model, "set_batch_map"):
            raise RuntimeError("CMGDB.Model.set_batch_map is required")
        cmgdb_model.set_batch_map(box_map.batch)
        graph_cpu_before = _usage_snapshot()
        graph_started = time.perf_counter()
        morse_graph, map_graph = CMGDB.ComputeMorseGraph(
            cmgdb_model, cache_map_graph=True, reserve_edges=reserve_edges
        )
        graph_seconds = time.perf_counter() - graph_started
        graph_cpu = _usage_delta(graph_cpu_before, _usage_snapshot())

        has_cache_method = getattr(map_graph, "has_cache", None)
        edge_count_method = getattr(map_graph, "num_cached_edges", None)
        if not callable(has_cache_method) or not bool(has_cache_method()):
            raise RuntimeError("CMGDB native MapGraph batch cache is missing")
        if not callable(edge_count_method):
            raise RuntimeError("CMGDB MapGraph does not expose cached edge count")
        if int(map_graph.num_vertices()) != EXPECTED_CELLS:
            raise RuntimeError(
                f"CMGDB returned {int(map_graph.num_vertices())} cells; "
                f"expected {EXPECTED_CELLS}"
            )
        cached_edges = int(edge_count_method())
        if cached_edges <= 0:
            raise RuntimeError("CMGDB native MapGraph cache has no edges")
        callback = box_map.stats()
        if int(callback["scalar_calls"]) != 0:
            raise RuntimeError(
                "CMGDB used the forbidden scalar callback "
                f"{callback['scalar_calls']} times"
            )
        if int(callback["batch_calls"]) <= 0:
            raise RuntimeError("CMGDB never used the required batch callback")
        if int(callback["rectangles"]) != EXPECTED_CALLBACK_RECTANGLES:
            raise RuntimeError(
                "CMGDB callback coverage was incomplete: observed "
                f"{callback['rectangles']} rectangles, expected "
                f"{EXPECTED_CALLBACK_RECTANGLES}"
            )
        if (
            int(callback["neural_corner_points"])
            != EXPECTED_NEURAL_CORNER_POINTS
        ):
            raise RuntimeError(
                "neural corner coverage was incomplete: observed "
                f"{callback['neural_corner_points']} points, expected "
                f"{EXPECTED_NEURAL_CORNER_POINTS}"
            )

        _write_json_atomic(
            status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "phase": "save_morse_artifacts",
                "started_at_utc": _read_json(status_path)["started_at_utc"],
                "cmgdb_seconds": graph_seconds,
                "cached_edges": cached_edges,
                "callback": callback,
            },
        )
        graph_dir = output / "MG_uniform_s24"
        save_started = time.perf_counter()
        dot_path, morse_sets_path = save_morse_graph_artifacts(
            morse_graph,
            graph_dir,
        )
        artifact_seconds = time.perf_counter() - save_started
        graph = _graph_summary(morse_graph, map_graph)
        if graph["cached_edges"] != cached_edges:
            raise RuntimeError("cached edge count changed during serialization")

        sampler.stop()
        completed_cpu = _usage_snapshot()
        total_seconds = time.perf_counter() - started_wall
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "target_run_id": TARGET_RUN_ID,
            "backend": "batched_on_demand_neural",
            "precomputed": False,
            "dimension": DIMENSION,
            "subdiv_init": SUBDIVISION,
            "subdiv_min": SUBDIVISION,
            "subdiv_max": SUBDIVISION,
            "subdiv_limit": SUBDIV_LIMIT,
            "cells_per_axis": CELLS_PER_AXIS,
            "expected_cells": EXPECTED_CELLS,
            "padding": PADDING,
            "bounds": bounds_payload,
            "timings": {
                "bounds_seconds": bounds_seconds,
                "neural_warmup_seconds": warmup_seconds,
                "cmgdb_seconds": graph_seconds,
                "morse_artifact_seconds": artifact_seconds,
                "total_seconds": total_seconds,
                "cmgdb_cpu": graph_cpu,
                "total_cpu": _usage_delta(started_cpu, completed_cpu),
            },
            "memory": sampler.stats(),
            "callback": {
                **callback,
                "native_batch_chunk": NATIVE_BATCH_CHUNK,
                "expected_batch_calls_for_two_full_cell_passes": (
                    EXPECTED_BATCH_CALLS
                ),
                "batch_calls_match_native_chunk_expectation": (
                    int(callback["batch_calls"]) == EXPECTED_BATCH_CALLS
                ),
                "expected_rectangles_for_two_full_cell_passes": (
                    EXPECTED_CALLBACK_RECTANGLES
                ),
                "expected_corner_points_if_two_full_cell_passes": (
                    EXPECTED_NEURAL_CORNER_POINTS
                ),
                "rectangles_per_map_cell": (
                    float(callback["rectangles"]) / EXPECTED_CELLS
                ),
            },
            "graph": graph,
            "artifacts": {
                "launch_manifest": _file_record(plan_path),
                "bounds": bounds_record,
                "provenance": _file_record(provenance_path),
                "morse_graph_dot": _file_record(dot_path),
                "morse_sets": _file_record(morse_sets_path),
                "map_graph_serialized": False,
                "map_graph_serialization_note": (
                    "the billion-edge native cache is retained only in memory; "
                    "cell/edge counts and Morse artifacts are persisted"
                ),
            },
            "runtime": runtime,
            "postconditions": {
                "all_15_training_runs_complete": True,
                "map_cells_equal_2_pow_24": True,
                "native_map_graph_cache_present": True,
                "scalar_callback_calls_equal_zero": True,
                "batch_callback_calls_positive": True,
                "cached_edges_positive": True,
                "callback_rectangles_equal_33554432": True,
                "neural_corner_points_equal_268435456": True,
            },
        }
        summary_path = _write_json_exclusive(
            graph_dir / "summary.json",
            summary,
        )
        _write_json_atomic(
            status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "summary": _file_record(summary_path),
            },
        )
        print(
            f"Completed exact s24 graph in {graph_seconds:.3f}s: "
            f"{cached_edges:,} cached edges, "
            f"{graph['morse_nodes']} Morse nodes.",
            flush=True,
        )
        return summary
    except BaseException as error:
        sampler.stop()
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started_wall,
            "cpu": _usage_delta(started_cpu, _usage_snapshot()),
            "memory": sampler.stats(),
        }
        _write_json_atomic(output / "failure.json", failure)
        _write_json_atomic(
            status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "failed_at_utc": failure["failed_at_utc"],
                "failure": _file_record(output / "failure.json"),
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="target dataset_01_seed_00 run root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="fresh, dedicated production output directory",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="device for on-demand latent-map evaluation; bounds stay on CPU",
    )
    parser.add_argument(
        "--reserve-edges",
        type=int,
        default=DEFAULT_RESERVE_EDGES,
        help=(
            "edge-buffer capacity to allocate up front; a sizing hint, not a "
            "ceiling -- the cache grows past it if the graph is larger"
        ),
    )
    parser.add_argument(
        "--max-forward-points",
        type=int,
        default=DEFAULT_MAX_FORWARD_POINTS,
        help="maximum neural corner points per Torch forward call",
    )
    parser.add_argument(
        "--rss-sample-seconds",
        type=float,
        default=DEFAULT_RSS_SAMPLE_SECONDS,
        help="process RSS sampling interval",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch the exact s24 graph after preflight; otherwise plan only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = resolve_target_inputs(args.run_root)
        plan = build_plan(
            inputs,
            output_root=args.output_root,
            device=args.device,
            reserve_edges=args.reserve_edges,
            max_forward_points=args.max_forward_points,
            rss_sample_seconds=args.rss_sample_seconds,
        )
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            if not inputs.matrix_status["complete"]:
                print(
                    "\nPreflight only: production remains gated on "
                    f"{len(inputs.matrix_status['incomplete_run_ids'])} "
                    "unfinished training runs.",
                    file=sys.stderr,
                )
            return 0
        run_exact_s24(
            inputs=inputs,
            output_root=args.output_root,
            device_name=args.device,
            reserve_edges=args.reserve_edges,
            max_forward_points=args.max_forward_points,
            rss_sample_seconds=args.rss_sample_seconds,
        )
        return 0
    except Exception as error:
        traceback.print_exc()
        print(
            f"Exact D3 s24 run failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
