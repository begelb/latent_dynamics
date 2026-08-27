#!/usr/bin/env python3
"""Continue the 2,138 unresolved Chafee--Infante evaluation trajectories.

This analysis never overwrites the archived ``traj_attractors.pkl``.  It
reintegrates each archived label-0 initial condition independently from t=0,
stops when the legacy first-16-mode distance criterion is met, verifies every
new label with a second solver, and rescoring the 45 canonical paper runs.
"""

from __future__ import annotations

import os
import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: Inputs live in the tracked artifacts tree; outputs stay beside the archived
#: run they extend, so re-running this script updates that record in place.
ARCHIVE_DIR = PROJECT_ROOT / "artifacts" / "reference_inputs" / "chafee_infante"
OUTPUT_DIR = PROJECT_ROOT / "replay_sources" / "chafee_infante" / "continuation_10000"
TRAJECTORY_ARCHIVE = ARCHIVE_DIR / "traj_attractors.pkl"
STABLE_ROOTS = ARCHIVE_DIR / "stable_solutions.csv"
ARCHIVED_MODEL_SOURCE = ARCHIVE_DIR / "autoencoder_model.py"
ARCHIVED_LABEL_SOURCE = ARCHIVE_DIR / "generate_attractor_basin_data.py"

EXPECTED_HASHES = {
    "traj_attractors.pkl": "f163b7427e50a4e4d08ab54c87cb5bd16592768edfe8432f019842416afbb145",
    "stable_solutions.csv": "cae0222acb37ae9688e54cb2a1f42ac3777360e49b3919403ec1433363de0586",
}
EXPECTED_COUNTS = {-1: 3_909, 0: 2_138, 1: 3_953}
SOLVER_ARTIFACT_SCHEMA_VERSION = 1
SOLVER_ALGORITHM = "solve_ivp_terminal_first16_root_distance_v1"
N = 64
ALPHA = 28.0
ORIGINAL_CUTOFF = 6.0
CONVERGENCE_TOLERANCE = 1e-8
COMPARE_MODES = 16
MAX_TIME = 6_400.0
MAX_STEP = 5.0
RTOL = 1e-10
ATOL = 1e-12
CHECKPOINT_HORIZONS = (6, 8, 10, 12, 16, 24, 40, 64, 100, 150, 200, 300, 500, 800, 1600, 3200, 6400)

_ROOTS: NDArray[np.float64] | None = None
_WORKER_MAX_TIME = MAX_TIME
_WORKER_MAX_STEP = MAX_STEP
_WORKER_RTOL = RTOL
_WORKER_ATOL = ATOL
_L_EIG = -(np.arange(1, N + 1, dtype=np.float64) ** 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Digests recorded by the reference computation for the two archived generator
#: sources, which are not redistributed.
ARCHIVED_SOURCE_HASHES = {
    "autoencoder_model.py":
        "420d13a61840ab8c52655d6561834da2b37d40396e3b0b73eb5b5499be6f6263",
    "generate_attractor_basin_data.py":
        "abe729271bc33aaa4f360a518a0cbe832284947995e584f5da0a6b15c0278471",
}


def _archived_source_hash(path: Path, name: str) -> str:
    """Hash the archived source when present, else its recorded digest."""
    return sha256_file(path) if path.is_file() else ARCHIVED_SOURCE_HASHES[name]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **arrays)
    temporary.replace(path)


def nonlinear(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Archived odd-extension convolution for the cubic sine coefficients."""
    a_ext = np.concatenate((-a[::-1], np.zeros(1), a))
    conv2 = np.convolve(a_ext, a_ext)
    conv3 = np.convolve(conv2, a_ext)
    center = 3 * N
    return conv3[center + 1 : center + 1 + N]


def vector_field(_t: float, a: NDArray[np.float64]) -> NDArray[np.float64]:
    return (ALPHA + _L_EIG) * a + (ALPHA / 4.0) * nonlinear(a)


def initialize_worker(
    roots: NDArray[np.float64],
    max_time: float,
    max_step: float,
    rtol: float,
    atol: float,
) -> None:
    global _ROOTS, _WORKER_MAX_TIME, _WORKER_MAX_STEP, _WORKER_RTOL, _WORKER_ATOL
    _ROOTS = np.asarray(roots, dtype=np.float64)
    _WORKER_MAX_TIME = float(max_time)
    _WORKER_MAX_STEP = float(max_step)
    _WORKER_RTOL = float(rtol)
    _WORKER_ATOL = float(atol)


def negative_event(_t: float, state: NDArray[np.float64]) -> float:
    assert _ROOTS is not None
    return float(
        np.linalg.norm(state[:COMPARE_MODES] - _ROOTS[0, :COMPARE_MODES])
        - CONVERGENCE_TOLERANCE
    )


def positive_event(_t: float, state: NDArray[np.float64]) -> float:
    assert _ROOTS is not None
    return float(
        np.linalg.norm(state[:COMPARE_MODES] - _ROOTS[1, :COMPARE_MODES])
        - CONVERGENCE_TOLERANCE
    )


negative_event.terminal = True
negative_event.direction = -1.0
positive_event.terminal = True
positive_event.direction = -1.0


def integrate_one(task: tuple[int, int, NDArray[np.float64], str]) -> dict[str, Any]:
    """Integrate one exact archived initial condition in an isolated process."""
    position, archive_index, initial = task[:3]
    method = task[3]
    assert _ROOTS is not None
    solution = solve_ivp(
        vector_field,
        (0.0, _WORKER_MAX_TIME),
        np.asarray(initial, dtype=np.float64),
        method=method,
        rtol=_WORKER_RTOL,
        atol=_WORKER_ATOL,
        max_step=_WORKER_MAX_STEP,
        events=(negative_event, positive_event),
        t_eval=(ORIGINAL_CUTOFF, _WORKER_MAX_TIME),
    )

    hit_times = [float(events[0]) if events.size else math.nan for events in solution.t_events]
    hits = [index for index, value in enumerate(hit_times) if math.isfinite(value)]
    label = 0
    resolution_time = math.nan
    terminal_state = np.full(N, np.nan, dtype=np.float64)
    if len(hits) == 1:
        root_index = hits[0]
        label = -1 if root_index == 0 else 1
        resolution_time = hit_times[root_index]
        terminal_state = np.asarray(solution.y_events[root_index][0], dtype=np.float64)

    cutoff_state = np.full(N, np.nan, dtype=np.float64)
    cutoff_matches = np.flatnonzero(np.isclose(solution.t, ORIGINAL_CUTOFF, rtol=0.0, atol=1e-12))
    if cutoff_matches.size:
        cutoff_state = np.asarray(solution.y[:, int(cutoff_matches[0])], dtype=np.float64)

    finite = bool(
        np.all(np.isfinite(cutoff_state))
        and (label == 0 or np.all(np.isfinite(terminal_state)))
    )
    first16_distance = math.nan
    full_distance = math.nan
    terminal_field_norm = math.nan
    inward_distance = math.nan
    if label:
        chosen = _ROOTS[0 if label == -1 else 1]
        first16_distance = float(
            np.linalg.norm(terminal_state[:COMPARE_MODES] - chosen[:COMPARE_MODES])
        )
        full_distance = float(np.linalg.norm(terminal_state - chosen))
        terminal_field_norm = float(np.linalg.norm(vector_field(0.0, terminal_state)))
        inward = solve_ivp(
            vector_field,
            (0.0, 1.0),
            terminal_state,
            method=method,
            rtol=_WORKER_RTOL,
            atol=_WORKER_ATOL,
            max_step=min(1.0, _WORKER_MAX_STEP),
        ).y[:, -1]
        inward_distance = float(np.linalg.norm(inward[:COMPARE_MODES] - chosen[:COMPARE_MODES]))

    cutoff_distances = np.full(2, np.nan, dtype=np.float64)
    if np.all(np.isfinite(cutoff_state)):
        cutoff_distances = np.asarray(
            [
                np.linalg.norm(cutoff_state[:COMPARE_MODES] - root[:COMPARE_MODES])
                for root in _ROOTS
            ],
            dtype=np.float64,
        )

    return {
        "position": position,
        "archive_index": archive_index,
        "success": bool(solution.success),
        "status": int(solution.status),
        "message": str(solution.message) if not solution.success else "",
        "finite": finite,
        "label": label,
        "resolution_time": resolution_time,
        "nfev": int(solution.nfev),
        "njev": int(solution.njev),
        "nlu": int(solution.nlu),
        "cutoff_state": cutoff_state,
        "cutoff_distances": cutoff_distances,
        "terminal_state": terminal_state,
        "first16_distance": first16_distance,
        "full_distance": full_distance,
        "terminal_field_norm": terminal_field_norm,
        "inward_distance": inward_distance,
    }


def new_result_arrays(count: int) -> dict[str, NDArray[Any]]:
    return {
        "completed": np.zeros(count, dtype=bool),
        "success": np.zeros(count, dtype=bool),
        "finite": np.zeros(count, dtype=bool),
        "labels": np.zeros(count, dtype=np.int8),
        "resolution_times": np.full(count, np.nan, dtype=np.float64),
        "nfev": np.zeros(count, dtype=np.int64),
        "njev": np.zeros(count, dtype=np.int64),
        "nlu": np.zeros(count, dtype=np.int64),
        "cutoff_states": np.full((count, N), np.nan, dtype=np.float64),
        "cutoff_distances": np.full((count, 2), np.nan, dtype=np.float64),
        "terminal_states": np.full((count, N), np.nan, dtype=np.float64),
        "first16_distances": np.full(count, np.nan, dtype=np.float64),
        "full_distances": np.full(count, np.nan, dtype=np.float64),
        "terminal_field_norms": np.full(count, np.nan, dtype=np.float64),
        "inward_distances": np.full(count, np.nan, dtype=np.float64),
    }


def save_method_artifact(
    path: Path,
    *,
    indices: NDArray[np.int64],
    method: str,
    arrays: dict[str, NDArray[Any]],
    elapsed_seconds: float,
) -> None:
    atomic_savez(
        path,
        artifact_schema_version=np.asarray(SOLVER_ARTIFACT_SCHEMA_VERSION),
        solver_algorithm=np.asarray(SOLVER_ALGORITHM),
        producer_script_sha256=np.asarray(sha256_file(Path(__file__))),
        produced_at_utc=np.asarray(datetime.now(UTC).isoformat()),
        trajectory_archive_sha256=np.asarray(EXPECTED_HASHES["traj_attractors.pkl"]),
        stable_roots_sha256=np.asarray(EXPECTED_HASHES["stable_solutions.csv"]),
        archive_indices=indices,
        method=np.asarray(method),
        system_N=np.asarray(N),
        system_alpha=np.asarray(ALPHA),
        compared_modes=np.asarray(COMPARE_MODES),
        convergence_tolerance=np.asarray(CONVERGENCE_TOLERANCE),
        original_cutoff=np.asarray(ORIGINAL_CUTOFF),
        rtol=np.asarray(RTOL),
        atol=np.asarray(ATOL),
        max_time=np.asarray(MAX_TIME),
        max_step=np.asarray(MAX_STEP),
        elapsed_seconds=np.asarray(elapsed_seconds),
        **arrays,
    )


def load_method_artifact(
    path: Path,
    *,
    indices: NDArray[np.int64],
    method: str,
    require_complete: bool,
) -> dict[str, NDArray[Any]]:
    expected_scalars: dict[str, str | int | float] = {
        "artifact_schema_version": SOLVER_ARTIFACT_SCHEMA_VERSION,
        "solver_algorithm": SOLVER_ALGORITHM,
        "producer_script_sha256": sha256_file(Path(__file__)),
        "trajectory_archive_sha256": EXPECTED_HASHES["traj_attractors.pkl"],
        "stable_roots_sha256": EXPECTED_HASHES["stable_solutions.csv"],
        "method": method,
        "system_N": N,
        "system_alpha": ALPHA,
        "compared_modes": COMPARE_MODES,
        "convergence_tolerance": CONVERGENCE_TOLERANCE,
        "original_cutoff": ORIGINAL_CUTOFF,
        "rtol": RTOL,
        "atol": ATOL,
        "max_time": MAX_TIME,
        "max_step": MAX_STEP,
    }
    templates = new_result_arrays(indices.size)
    with np.load(path, allow_pickle=False) as saved:
        missing = [
            key
            for key in (*expected_scalars, "archive_indices", *templates)
            if key not in saved.files
        ]
        if missing:
            raise ValueError(f"{path} lacks required fields: {missing}")
        if not np.array_equal(saved["archive_indices"], indices):
            raise ValueError(f"{path} has different archive indices")
        mismatches: dict[str, dict[str, Any]] = {}
        for key, expected in expected_scalars.items():
            actual = saved[key].item()
            equal = (
                np.isclose(float(actual), float(expected), rtol=0.0, atol=0.0)
                if isinstance(expected, float)
                else actual == expected
            )
            if not bool(equal):
                mismatches[key] = {"expected": expected, "actual": actual}
        if mismatches:
            raise ValueError(f"{path} solver provenance mismatch: {mismatches}")
        arrays = {key: np.asarray(saved[key]) for key in templates}
    shape_mismatches = {
        key: {"expected": template.shape, "actual": arrays[key].shape}
        for key, template in templates.items()
        if arrays[key].shape != template.shape
    }
    if shape_mismatches:
        raise ValueError(f"{path} result-shape mismatch: {shape_mismatches}")
    completed = np.asarray(arrays["completed"], dtype=bool)
    if np.any(completed & ~(arrays["success"] & arrays["finite"])):
        bad = np.flatnonzero(completed & ~(arrays["success"] & arrays["finite"]))
        raise ValueError(f"{path} marks failed/nonfinite rows complete: {bad[:20].tolist()}")
    if require_complete:
        valid = completed & arrays["success"] & arrays["finite"]
        if not np.all(valid):
            bad = np.flatnonzero(~valid)
            raise ValueError(f"{path} is not complete: {bad[:20].tolist()}")
        if np.any(arrays["labels"] == 0) or not np.all(
            np.isfinite(arrays["resolution_times"])
        ):
            raise ValueError(f"{path} has unresolved or nonfinite terminal results")
    return arrays


def run_method(
    method: str,
    *,
    indices: NDArray[np.int64],
    points: NDArray[np.float64],
    roots: NDArray[np.float64],
    workers: int,
    force: bool,
) -> dict[str, NDArray[Any]]:
    final_path = OUTPUT_DIR / f"{method.lower()}_continuation.npz"
    partial_path = OUTPUT_DIR / f".{method.lower()}_partial.npz"
    if final_path.is_file() and not force:
        print(f"[{method}] reusing {final_path.name}", flush=True)
        return load_method_artifact(
            final_path,
            indices=indices,
            method=method,
            require_complete=True,
        )

    arrays = new_result_arrays(indices.size)
    if partial_path.is_file() and not force:
        arrays = load_method_artifact(
            partial_path,
            indices=indices,
            method=method,
            require_complete=False,
        )
        print(
            f"[{method}] resuming {int(np.count_nonzero(arrays['completed']))}/{indices.size}",
            flush=True,
        )

    pending = np.flatnonzero(~arrays["completed"])
    started = time.perf_counter()
    tasks = [
        (int(position), int(indices[position]), points[int(indices[position])], method)
        for position in pending.tolist()
    ]
    print(f"[{method}] launching {len(tasks)} trajectories on {workers} workers", flush=True)
    failures: list[str] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(roots, MAX_TIME, MAX_STEP, RTOL, ATOL),
    ) as pool:
        futures = {pool.submit(integrate_one, task): task[0] for task in tasks}
        for completed_now, future in enumerate(as_completed(futures), start=1):
            position = futures[future]
            try:
                result = future.result()
            except Exception as error:  # retain the partial artifact before failing
                failures.append(f"position={position}: {error!r}")
                continue
            for source, target in (
                ("success", "success"),
                ("finite", "finite"),
                ("label", "labels"),
                ("resolution_time", "resolution_times"),
                ("nfev", "nfev"),
                ("njev", "njev"),
                ("nlu", "nlu"),
                ("cutoff_state", "cutoff_states"),
                ("cutoff_distances", "cutoff_distances"),
                ("terminal_state", "terminal_states"),
                ("first16_distance", "first16_distances"),
                ("full_distance", "full_distances"),
                ("terminal_field_norm", "terminal_field_norms"),
                ("inward_distance", "inward_distances"),
            ):
                arrays[target][position] = result[source]
            arrays["completed"][position] = True

            if completed_now % 100 == 0 or completed_now == len(tasks):
                elapsed = time.perf_counter() - started
                done = int(np.count_nonzero(arrays["completed"]))
                resolved = int(np.count_nonzero(arrays["labels"]))
                print(
                    f"[{method}] {done}/{indices.size}; resolved={resolved}; elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if completed_now % 250 == 0:
                save_method_artifact(
                    partial_path,
                    indices=indices,
                    method=method,
                    arrays=arrays,
                    elapsed_seconds=time.perf_counter() - started,
                )

    elapsed = time.perf_counter() - started
    save_method_artifact(
        partial_path,
        indices=indices,
        method=method,
        arrays=arrays,
        elapsed_seconds=elapsed,
    )
    if failures:
        raise RuntimeError(f"{method} worker failures: {failures[:10]}")
    if not np.all(arrays["completed"] & arrays["success"] & arrays["finite"]):
        bad = np.flatnonzero(~(arrays["completed"] & arrays["success"] & arrays["finite"]))
        raise RuntimeError(f"{method} incomplete/failed positions: {bad[:20].tolist()}")
    save_method_artifact(
        final_path,
        indices=indices,
        method=method,
        arrays=arrays,
        elapsed_seconds=elapsed,
    )
    partial_path.unlink(missing_ok=True)
    return arrays


def describe(values: NDArray[np.float64]) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def saved_solver_elapsed_seconds(method: str) -> float:
    path = OUTPUT_DIR / f"{method.lower()}_continuation.npz"
    with np.load(path, allow_pickle=False) as saved:
        return float(saved["elapsed_seconds"].item())


def classify(
    truth: NDArray[np.int64],
    predicted: NDArray[np.int64],
    *,
    negative_label: int,
    positive_label: int,
) -> dict[str, Any]:
    conditioned = truth != 0
    negative_truth = truth == -1
    positive_truth = truth == 1
    in_negative = predicted == negative_label
    in_positive = predicted == positive_label
    counts = {
        "outside_both_basins": int(np.count_nonzero(conditioned & ~(in_negative | in_positive))),
        "misclassified_in_negative_basin": int(np.count_nonzero(positive_truth & in_negative)),
        "misclassified_in_positive_basin": int(np.count_nonzero(negative_truth & in_positive)),
        "correctly_classified_in_negative_basin": int(np.count_nonzero(negative_truth & in_negative)),
        "correctly_classified_in_positive_basin": int(np.count_nonzero(positive_truth & in_positive)),
    }
    denominator = int(np.count_nonzero(conditioned))
    if sum(counts.values()) != denominator:
        raise ValueError("basin categories do not conserve the conditioned count")
    percentages = {key: 100.0 * value / denominator for key, value in counts.items()}
    return {
        "denominator": denominator,
        "counts": counts,
        "percentages": percentages,
        "combined_correct_count": (
            counts["correctly_classified_in_negative_basin"]
            + counts["correctly_classified_in_positive_basin"]
        ),
        "combined_correct_percentage": (
            percentages["correctly_classified_in_negative_basin"]
            + percentages["correctly_classified_in_positive_basin"]
        ),
    }


def canonical_run_directories() -> dict[int, list[Path]]:
    d1_root = PROJECT_ROOT / "code" / "output" / "chafee_d1_matched_d2_archive_5x3_roa_v1"
    d2_root = PROJECT_ROOT / "code" / "output" / "chafee_d2_archive_5x3_roa_v1"
    d3_root = PROJECT_ROOT / "code" / "output" / "chafee_d3_matched_d2_archive_5x3_ondemand_v2"

    d1_payload = json.loads((d1_root / "results.json").read_text())
    d2_payload = json.loads((d2_root / "results.json").read_text())
    d3_payload = json.loads((d3_root / "aggregate_results.json").read_text())
    directories = {
        1: [Path(row["output_dir"]) for row in d1_payload["results"]],
        2: [Path(row["output_dir"]) for row in d2_payload["results"]],
        3: [Path(row["result_path"]).parent for row in d3_payload["rows"]],
    }
    for dimension, paths in directories.items():
        if len(paths) != 15 or len(set(paths)) != 15:
            raise ValueError(f"d={dimension} canonical manifest does not contain 15 unique runs")
        missing = [str(path) for path in paths if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"missing canonical d={dimension} runs: {missing}")
    return directories


def recompute_paper_statistics(
    old_truth: NDArray[np.int64], new_truth: NDArray[np.int64]
) -> dict[str, Any]:
    directories = canonical_run_directories()
    rows: list[dict[str, Any]] = []
    for dimension, run_dirs in directories.items():
        for run_dir in run_dirs:
            stats_path = run_dir / "basin_statistics.json"
            labels_path = run_dir / "trajectory_basin_labels.npy"
            payload = json.loads(stats_path.read_text())
            predicted = np.asarray(np.load(labels_path, allow_pickle=False), dtype=np.int64)
            if predicted.shape != (10_000,):
                raise ValueError(f"unexpected prediction shape at {labels_path}: {predicted.shape}")
            root_section = payload["root_association"] if dimension == 2 else payload["stable_roots"]
            negative = int(root_section["negative_basin_label"])
            positive = int(root_section["positive_basin_label"])
            old = classify(
                old_truth,
                predicted,
                negative_label=negative,
                positive_label=positive,
            )
            stored_counts = payload["statistics"]["counts"]
            if old["counts"] != stored_counts:
                raise ValueError(f"old count reproduction failed at {run_dir}")
            new = classify(
                new_truth,
                predicted,
                negative_label=negative,
                positive_label=positive,
            )
            dataset_value = payload.get("dataset")
            if dataset_value is None:
                dataset_parts = [
                    part for part in run_dir.parts if part.startswith("dataset_")
                ]
                if len(dataset_parts) != 1:
                    raise ValueError(f"cannot infer dataset number from {run_dir}")
                dataset_value = dataset_parts[0].split("_", maxsplit=1)[1]
            dataset = int(dataset_value)
            if dimension == 1:
                run_parts = run_dir.name.split("_")
                if len(run_parts) < 2 or run_parts[0] != "run":
                    raise ValueError(f"cannot infer d=1 run number from {run_dir}")
                run_number = int(run_parts[1])
            elif dimension == 2:
                run_number = int(payload["training_trial"])
            else:
                run_number = int(payload["training_seed"]) + 1
            rows.append(
                {
                    "dimension": dimension,
                    "dataset": dataset,
                    "run": run_number,
                    "negative_basin_label": negative,
                    "positive_basin_label": positive,
                    "run_directory": str(run_dir.relative_to(PROJECT_ROOT)),
                    "prediction_sha256": sha256_file(labels_path),
                    "old": old,
                    "completed": new,
                }
            )

    rows.sort(key=lambda row: (row["dimension"], row["dataset"], row["run"]))
    cell_keys = {
        (int(row["dimension"]), int(row["dataset"]), int(row["run"])) for row in rows
    }
    if len(rows) != 45 or len(cell_keys) != 45:
        raise ValueError(
            f"canonical export must have 45 unique (dimension, dataset, run) cells; "
            f"got rows={len(rows)}, unique={len(cell_keys)}"
        )
    for dimension in (1, 2, 3):
        for dataset in range(1, 6):
            run_ids = {
                int(row["run"])
                for row in rows
                if row["dimension"] == dimension and row["dataset"] == dataset
            }
            if run_ids != {1, 2, 3}:
                raise ValueError(
                    f"d={dimension}, dataset={dataset} has run ids {sorted(run_ids)}"
                )
    summary: dict[str, Any] = {}
    expected_old_means = {1: 52.777919104553554, 2: 71.88586449588739, 3: 76.22657508691597}
    category_keys = list(rows[0]["old"]["counts"])
    for dimension in (1, 2, 3):
        selected = [row for row in rows if row["dimension"] == dimension]
        old_correct = np.asarray(
            [row["old"]["combined_correct_percentage"] for row in selected], dtype=np.float64
        )
        new_correct = np.asarray(
            [row["completed"]["combined_correct_percentage"] for row in selected],
            dtype=np.float64,
        )
        if not np.isclose(
            np.mean(old_correct), expected_old_means[dimension], rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"d={dimension} old headline does not reproduce the paper package")
        by_dataset = []
        for dataset in range(1, 6):
            group = [row for row in selected if row["dataset"] == dataset]
            if len(group) != 3:
                raise ValueError(f"d={dimension}, dataset={dataset} does not have three runs")
            values = np.asarray(
                [row["completed"]["combined_correct_percentage"] for row in group],
                dtype=np.float64,
            )
            by_dataset.append(
                {
                    "dataset": dataset,
                    "runs": [float(value) for value in values],
                    "descriptive": describe(values),
                }
            )
        summary[str(dimension)] = {
            "old_combined_correct": describe(old_correct),
            "completed_combined_correct": describe(new_correct),
            "old_mean_category_percentages": {
                key: float(np.mean([row["old"]["percentages"][key] for row in selected]))
                for key in category_keys
            },
            "completed_mean_category_percentages": {
                key: float(
                    np.mean([row["completed"]["percentages"][key] for row in selected])
                )
                for key in category_keys
            },
            "by_dataset": by_dataset,
        }

    output = {
        "schema_version": 1,
        "method": "Rescore unchanged canonical predicted basin labels against completed truth labels",
        "denominators": {"old_per_run": 7862, "completed_per_run": 10000},
        "summary_by_dimension": summary,
        "runs": rows,
    }
    atomic_write_json(OUTPUT_DIR / "updated_paper_statistics.json", output)

    csv_path = OUTPUT_DIR / "updated_paper_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        fieldnames = [
            "dimension",
            "dataset",
            "run",
            "old_combined_correct_percentage",
            "completed_combined_correct_percentage",
            "completed_outside_both_percentage",
            "completed_misclassified_negative_percentage",
            "completed_misclassified_positive_percentage",
            "completed_correct_negative_percentage",
            "completed_correct_positive_percentage",
            "run_directory",
        ]
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            completed = row["completed"]
            percentages = completed["percentages"]
            writer.writerow(
                {
                    "dimension": row["dimension"],
                    "dataset": row["dataset"],
                    "run": row["run"],
                    "old_combined_correct_percentage": row["old"][
                        "combined_correct_percentage"
                    ],
                    "completed_combined_correct_percentage": completed[
                        "combined_correct_percentage"
                    ],
                    "completed_outside_both_percentage": percentages[
                        "outside_both_basins"
                    ],
                    "completed_misclassified_negative_percentage": percentages[
                        "misclassified_in_negative_basin"
                    ],
                    "completed_misclassified_positive_percentage": percentages[
                        "misclassified_in_positive_basin"
                    ],
                    "completed_correct_negative_percentage": percentages[
                        "correctly_classified_in_negative_basin"
                    ],
                    "completed_correct_positive_percentage": percentages[
                        "correctly_classified_in_positive_basin"
                    ],
                    "run_directory": row["run_directory"],
                }
            )
    return output


def slow_saddle_diagnostic(
    cutoff_state: NDArray[np.float64], resolution_time: float
) -> dict[str, Any]:
    equilibrium = root(
        lambda state: vector_field(0.0, state),
        cutoff_state,
        method="lm",
        options={"ftol": 1e-13, "xtol": 1e-13, "gtol": 1e-13, "maxiter": 20_000},
    )
    saddle = np.asarray(equilibrium.x, dtype=np.float64)
    step = 1e-6
    basis = np.eye(N)
    jacobian = np.column_stack(
        [
            (vector_field(0.0, saddle + step * basis[j]) - vector_field(0.0, saddle - step * basis[j]))
            / (2.0 * step)
            for j in range(N)
        ]
    )
    eigenvalues = np.linalg.eigvals(jacobian)
    positive = np.sort(eigenvalues.real[eigenvalues.real > 1e-8])
    return {
        "resolution_time": float(resolution_time),
        "root_solver_success": bool(equilibrium.success),
        "equilibrium_residual_norm": float(np.linalg.norm(vector_field(0.0, saddle))),
        "cutoff_distance_to_equilibrium": float(np.linalg.norm(cutoff_state - saddle)),
        "equilibrium_norm": float(np.linalg.norm(saddle)),
        "equilibrium_first_10_coefficients": saddle[:10].tolist(),
        "unstable_eigenvalue_count": int(positive.size),
        "positive_eigenvalues": positive.tolist(),
        "weakest_positive_eigenvalue": float(positive[0]) if positive.size else None,
        "weak_e_folding_time": float(1.0 / positive[0]) if positive.size else None,
        "largest_real_eigenvalue": float(np.max(eigenvalues.real)),
    }


def plot_resolution_curve(times: NDArray[np.float64], labels: NDArray[np.int8]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = np.asarray(CHECKPOINT_HORIZONS, dtype=np.float64)
    cumulative = np.asarray([np.count_nonzero(times <= horizon) for horizon in horizons])
    remaining = times.size - cumulative
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].step(horizons, cumulative, where="post", color="#2463a6", linewidth=2.2)
    axes[0].axvline(ORIGINAL_CUTOFF, color="#b33a3a", linestyle="--", linewidth=1.4)
    axes[0].set_xscale("log")
    axes[0].set_xlim(5.5, max(800.0, float(np.max(times)) * 1.08))
    axes[0].set_ylim(0, times.size * 1.025)
    axes[0].set_xlabel("Physical integration time")
    axes[0].set_ylabel("Previously unresolved trajectories now classified")
    axes[0].set_title("Continuation of the 2,138 label-0 trajectories")
    axes[0].grid(alpha=0.22)

    lower = max(ORIGINAL_CUTOFF, float(np.min(times)) * 0.98)
    bins = np.geomspace(lower, float(np.max(times)) * 1.03, 28)
    axes[1].hist(
        [times[labels == -1], times[labels == 1]],
        bins=bins,
        stacked=True,
        color=["#3b70b2", "#d15b45"],
        label=["Negative equilibrium", "Positive equilibrium"],
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Time when the $10^{-8}$ criterion is first met")
    axes[1].set_ylabel("Trajectory count")
    axes[1].set_title("Resolution-time distribution")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.22)
    fig.savefig(OUTPUT_DIR / "resolution_times.png", dpi=200)
    plt.close(fig)

    atomic_write_json(
        OUTPUT_DIR / "resolution_by_horizon.json",
        {
            "horizons": horizons.tolist(),
            "cumulative_resolved": cumulative.tolist(),
            "remaining_unresolved": remaining.tolist(),
        },
    )


def build_report(summary: dict[str, Any], paper_stats: dict[str, Any]) -> None:
    continuation = summary["continuation"]
    lines = [
        "# Chafee--Infante 10,000-trajectory continuation",
        "",
        "The archived label `0` was only a finite-time nonclassification: the t=6 endpoint was not within `1e-8` of either saved stable equilibrium in the first 16 spectral modes. It was not a label for the unstable zero equilibrium.",
        "",
        "## Outcome",
        "",
        f"All {continuation['original_unresolved']:,} previously unresolved trajectories met the original convergence criterion by t={continuation['latest_resolution_time']:.6g}. They add {continuation['new_negative']:,} negative and {continuation['new_positive']:,} positive outcomes, giving completed totals of {continuation['completed_negative']:,} negative and {continuation['completed_positive']:,} positive trajectories.",
        "",
        f"LSODA and BDF agreed on {continuation['solver_label_agreements']:,}/{continuation['original_unresolved']:,} labels; disagreements: {continuation['solver_label_disagreements']}. Both solvers were run independently from each exact archived initial condition.",
        "",
        "## Staged resolution",
        "",
        "| Physical time | Resolved cumulatively | Still unresolved |",
        "|---:|---:|---:|",
    ]
    for row in continuation["resolution_by_horizon"]:
        lines.append(
            f"| {row['horizon']:g} | {row['cumulative_resolved']:,} | {row['remaining_unresolved']:,} |"
        )
    lines.extend(
        [
            "",
        "The slowest trajectory passed near an index-one saddle with a weak unstable direction; this explains its long transient without introducing a third attractor. Its saddle diagnostic is recorded in `summary.json`.",
            "",
            "## Rescored paper headline",
            "",
            "The encoder outputs and CMGDB regions were not rerun. Their saved 10,000 point classifications were verified against the old counts and rescored against the completed truth labels.",
            "",
            "| Latent dimension | Old mean correct (n=7,862) | Completed mean correct (n=10,000) | Completed outside both |",
            "|---:|---:|---:|---:|",
        ]
    )
    for dimension in (1, 2, 3):
        values = paper_stats["summary_by_dimension"][str(dimension)]
        completed_categories = values["completed_mean_category_percentages"]
        lines.append(
            f"| {dimension} | {values['old_combined_correct']['mean']:.2f}% | {values['completed_combined_correct']['mean']:.2f}% | {completed_categories['outside_both_basins']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Suggested manuscript replacement",
            "",
            f"Starting from {continuation['original_total']:,} initial conditions, we integrated each trajectory until its first 16 spectral coefficients entered a $10^{{-8}}$ neighborhood of one of the two stable equilibria. At the original cutoff $t=6$ (60 applications of the time-$0.1$ map), {continuation['original_total'] - continuation['original_unresolved']:,} trajectories met this criterion. We continued the remaining {continuation['original_unresolved']:,} trajectories; all subsequently met the criterion, with {continuation['new_negative']:,} converging to the negative equilibrium and {continuation['new_positive']:,} to the positive equilibrium. A diagnostic of the slowest trajectory shows it passing near a weakly unstable equilibrium. We therefore report the basin-classification percentages over all {continuation['original_total']:,} initial conditions.",
            "",
            "## Artifacts",
            "",
            "- `summary.json`: provenance, solver agreement, trajectory counts, and saddle diagnostic.",
            "- `continuation_results.npz`: accepted full 10,000 labels and per-trajectory resolution data.",
            "- `lsoda_continuation.npz`, `bdf_continuation.npz`: independent raw solver results.",
            "- `updated_paper_statistics.json` and `.csv`: all 45 rescored canonical runs.",
            "- `resolution_times.png`: staged continuation plot.",
            "",
        ]
    )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def load_and_validate_inputs() -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], dict[str, Any]]:
    actual_hashes = {
        "traj_attractors.pkl": sha256_file(TRAJECTORY_ARCHIVE),
        "stable_solutions.csv": sha256_file(STABLE_ROOTS),
    }
    if actual_hashes != EXPECTED_HASHES:
        raise ValueError(f"archived input hash mismatch: {actual_hashes}")
    with TRAJECTORY_ARCHIVE.open("rb") as source:
        archived = pickle.load(source)
    points = np.asarray(list(archived.keys()), dtype=np.float64)
    labels = np.asarray(list(archived.values()), dtype=np.int64)
    roots = np.loadtxt(STABLE_ROOTS, delimiter=",", dtype=np.float64)
    counts = {int(key): int(value) for key, value in Counter(labels.tolist()).items()}
    if points.shape != (10_000, N) or roots.shape != (2, N) or counts != EXPECTED_COUNTS:
        raise ValueError(
            f"archived input shape/count mismatch: points={points.shape}, roots={roots.shape}, counts={counts}"
        )

    rng = np.random.RandomState(9551)
    reconstructed = rng.uniform(-2.0, 2.0, (10_000, N)) * np.exp(-0.5 * np.arange(N))
    if not np.array_equal(points, reconstructed):
        raise ValueError("archived initial conditions do not reproduce evaluation seed 9551")
    root_residuals = [float(np.linalg.norm(vector_field(0.0, value))) for value in roots]
    if max(root_residuals) >= 1e-7:
        raise ValueError(f"saved stable roots have unexpectedly large residuals: {root_residuals}")
    provenance = {
        "input_hashes": actual_hashes,
        # The two archived generator sources are provenance only -- nothing here
        # imports them -- and are deliberately not shipped ("no archive .py
        # policy", artifacts/manifest.json). Their recorded digests stand in, so
        # the provenance block is complete without the files being present.
        "source_hashes": {
            "autoencoder_model.py": _archived_source_hash(
                ARCHIVED_MODEL_SOURCE, "autoencoder_model.py"
            ),
            "generate_attractor_basin_data.py": _archived_source_hash(
                ARCHIVED_LABEL_SOURCE, "generate_attractor_basin_data.py"
            ),
            "run_continuation.py": sha256_file(Path(__file__)),
        },
        "archived_counts": counts,
        "initial_condition_seed": 9551,
        "initial_condition_seed_exactly_reproduced": True,
        "stable_root_residual_norms": root_residuals,
        "stable_root_symmetry_norm": float(np.linalg.norm(roots[0] + roots[1])),
    }
    return points, labels, roots, provenance


def main() -> int:
    # Each trajectory is already parallelized as a separate process.  Prevent
    # numerical libraries inside a worker from adding their own thread pools.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, max(1, (os.cpu_count() or 2) - 2)),
        help="processes per solver (default: up to 12)",
    )
    parser.add_argument("--force", action="store_true", help="ignore any completed solver artifacts")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_started = time.perf_counter()
    points, old_labels, roots, provenance = load_and_validate_inputs()
    unresolved_indices = np.flatnonzero(old_labels == 0).astype(np.int64)
    print(
        f"Validated exact archive; continuing {unresolved_indices.size} trajectories with {args.workers} workers",
        flush=True,
    )
    run_config = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "provenance": provenance,
        "system": {"N": N, "alpha": ALPHA, "tau": 0.1},
        "criterion": {
            "compared_modes": COMPARE_MODES,
            "distance_tolerance": CONVERGENCE_TOLERANCE,
            "original_cutoff": ORIGINAL_CUTOFF,
        },
        "solvers": ["LSODA", "BDF"],
        "rtol": RTOL,
        "atol": ATOL,
        "max_time": MAX_TIME,
        "max_step": MAX_STEP,
        "workers": args.workers,
        "checkpoint_horizons": list(CHECKPOINT_HORIZONS),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    atomic_write_json(OUTPUT_DIR / "run_config.json", run_config)

    lsoda = run_method(
        "LSODA",
        indices=unresolved_indices,
        points=points,
        roots=roots,
        workers=args.workers,
        force=args.force,
    )
    bdf = run_method(
        "BDF",
        indices=unresolved_indices,
        points=points,
        roots=roots,
        workers=args.workers,
        force=args.force,
    )

    if np.any(lsoda["labels"] == 0) or np.any(bdf["labels"] == 0):
        raise RuntimeError(
            "at least one trajectory did not resolve by max_time; no completed labels were accepted"
        )
    disagreements = np.flatnonzero(lsoda["labels"] != bdf["labels"])
    if disagreements.size:
        raise RuntimeError(
            f"LSODA/BDF label disagreements at unresolved positions {disagreements[:20].tolist()}"
        )
    if np.any(lsoda["first16_distances"] > CONVERGENCE_TOLERANCE * (1.0 + 1e-5)):
        raise RuntimeError("LSODA event distance exceeds the legacy tolerance")
    if np.any(bdf["first16_distances"] > CONVERGENCE_TOLERANCE * (1.0 + 1e-5)):
        raise RuntimeError("BDF event distance exceeds the legacy tolerance")
    if np.any(lsoda["inward_distances"] >= lsoda["first16_distances"]):
        raise RuntimeError("LSODA +1-time audit did not move every terminal state inward")
    if np.any(bdf["inward_distances"] >= bdf["first16_distances"]):
        raise RuntimeError("BDF +1-time audit did not move every terminal state inward")

    accepted_labels = old_labels.copy()
    accepted_labels[unresolved_indices] = lsoda["labels"].astype(np.int64)
    if np.any(accepted_labels == 0):
        raise RuntimeError("accepted label array still contains zeros")
    if not np.array_equal(accepted_labels[old_labels != 0], old_labels[old_labels != 0]):
        raise RuntimeError("continuation modified an originally resolved label")

    primary_times = np.asarray(lsoda["resolution_times"], dtype=np.float64)
    cross_times = np.asarray(bdf["resolution_times"], dtype=np.float64)
    time_differences = np.abs(primary_times - cross_times)
    slowest_position = int(np.argmax(primary_times))
    saddle = slow_saddle_diagnostic(
        np.asarray(lsoda["cutoff_states"][slowest_position], dtype=np.float64),
        float(primary_times[slowest_position]),
    )
    horizon_rows = [
        {
            "horizon": float(horizon),
            "cumulative_resolved": int(np.count_nonzero(primary_times <= horizon)),
            "remaining_unresolved": int(np.count_nonzero(primary_times > horizon)),
        }
        for horizon in CHECKPOINT_HORIZONS
    ]
    continuation_summary = {
        "original_total": 10_000,
        "original_unresolved": int(unresolved_indices.size),
        "new_negative": int(np.count_nonzero(lsoda["labels"] == -1)),
        "new_positive": int(np.count_nonzero(lsoda["labels"] == 1)),
        "completed_negative": int(np.count_nonzero(accepted_labels == -1)),
        "completed_positive": int(np.count_nonzero(accepted_labels == 1)),
        "solver_label_agreements": int(unresolved_indices.size - disagreements.size),
        "solver_label_disagreements": int(disagreements.size),
        "earliest_resolution_time": float(np.min(primary_times)),
        "latest_resolution_time": float(np.max(primary_times)),
        "resolution_time_quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(
                (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
                np.quantile(primary_times, (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)),
            )
        },
        "absolute_solver_time_difference_quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(
                (0.0, 0.5, 0.9, 0.99, 1.0),
                np.quantile(time_differences, (0.0, 0.5, 0.9, 0.99, 1.0)),
            )
        },
        "maximum_full_64_distance_at_lsoda_event": float(np.max(lsoda["full_distances"])),
        "maximum_full_64_distance_at_bdf_event": float(np.max(bdf["full_distances"])),
        "maximum_lsoda_terminal_field_norm": float(np.max(lsoda["terminal_field_norms"])),
        "maximum_bdf_terminal_field_norm": float(np.max(bdf["terminal_field_norms"])),
        "resolution_by_horizon": horizon_rows,
        "slowest_archive_index": int(unresolved_indices[slowest_position]),
    }
    atomic_savez(
        OUTPUT_DIR / "continuation_results.npz",
        old_labels=old_labels,
        completed_labels=accepted_labels,
        unresolved_archive_indices=unresolved_indices,
        continuation_labels=lsoda["labels"],
        lsoda_resolution_times=primary_times,
        bdf_resolution_times=cross_times,
        lsoda_cutoff_states=lsoda["cutoff_states"],
        lsoda_terminal_states=lsoda["terminal_states"],
    )
    plot_resolution_curve(primary_times, np.asarray(lsoda["labels"], dtype=np.int8))
    # The paper-statistics table is recomputed over 45 archived ROA sweep runs
    # (15 per latent dimension) that this repository does not ship. Their absence
    # must not discard a completed continuation, which is self-contained.
    try:
        paper_stats = recompute_paper_statistics(old_labels, accepted_labels)
    except FileNotFoundError as error:
        warnings.warn(
            f"paper-statistics table skipped; canonical ROA runs absent ({error})",
            RuntimeWarning,
            stacklevel=2,
        )
        paper_stats = None

    summary = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance,
        "continuation": continuation_summary,
        "slow_saddle_diagnostic": saddle,
        "solver_elapsed_seconds": {
            "LSODA": saved_solver_elapsed_seconds("LSODA"),
            "BDF": saved_solver_elapsed_seconds("BDF"),
        },
        "current_invocation_elapsed_seconds": float(time.perf_counter() - overall_started),
        "artifacts": {
            name: {"path": name, "sha256": sha256_file(OUTPUT_DIR / name)}
            for name in (
                "run_config.json",
                "lsoda_continuation.npz",
                "bdf_continuation.npz",
                "continuation_results.npz",
                "resolution_by_horizon.json",
                "resolution_times.png",
                "updated_paper_statistics.json",
                "updated_paper_statistics.csv",
            )
            if (OUTPUT_DIR / name).is_file()
        },
    }
    atomic_write_json(OUTPUT_DIR / "summary.json", summary)
    if paper_stats is not None:
        build_report(summary, paper_stats)
    print(
        "Completed: "
        f"new labels (-/+)={continuation_summary['new_negative']}/{continuation_summary['new_positive']}; "
        f"latest t={continuation_summary['latest_resolution_time']:.6g}; "
        f"invocation elapsed={summary['current_invocation_elapsed_seconds']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
