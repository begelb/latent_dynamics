"""Run the matched five-dataset by three-seed Chafee d=1 experiment.

This exploratory driver pairs each of the five archived reference training datasets
with three explicit, independent model-initialization seeds (0, 1, and 2).
Training uses the validated direct full-batch implementation and changes only
the latent dimension from the archived d=2 computation to d=1.

The primary analysis is deliberately the archived-comparison protocol:

* encoder bounds from the exact dataset used to train that model;
* a persisted 257-corner lookup for the level-8 one-dimensional box map;
* a uniform CMGDB graph at subdivisions 8/8/8 with padding enabled; and
* the archived strict ``MorseSingletonReachability`` basin classification.

Headline validity requires exactly two uniform minimal attractors and distinct
unique singleton associations for the encoded negative and positive roots.
Adaptive/Conley topology and full-grid strict plus blocker/LCA RoA products are
optional post-processing and never gate the primary comparison.

The outer plan, each dataset training sweep, and every analysis attempt are
immutable and hash-bound.  Training resumes at model boundaries.  Interrupted
analysis attempts are preserved and retried in a new attempt directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import re
import statistics
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

import analyze_chafee_d1_full_batch_sweep as analyze
import chafee_latent_dimension_study as study
import repeat_chafee_d1_full_batch as repeat
import sweep_chafee_d1_full_batch as sweep

from latentdynamics._paths import get_repo_root

REPO_ROOT = get_repo_root()
DRIVER_IMPLEMENTATION = Path(__file__).resolve()
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "replay_sources" / "chafee_infante" / "reference_inputs"
# Rebound in main() when --reference-root is supplied.
REFERENCE_ROOT = DEFAULT_REFERENCE_ROOT
DEFAULT_OUTPUT = REPO_ROOT / "output" / "chafee_d1_matched_d2_archive_5x3_roa_v1"

TRAINING_SEEDS = (0, 1, 2)
EPOCHS = 4_000
LEARNING_RATE = 0.003
DEVICE = "cpu"
PLAN_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
EXPECTED_CONDITIONED = 7_862
EXPECTED_DATASET_HASHES = {
    1: "f50e3f44b4d6a4e5cf516ead5f3e44d2b4a3afdb0dff34c76613d0444caf8f7b",
    2: "b183613bf887b8eb2e85f780e440295dc16b1320bff178da6886c0d0dc93353d",
    3: "980f9320588ca49d87e319f7dc727a885692d12ccb523bac3c9ad92ffc33ad46",
    4: "6025463fc0b50334a6bdad35dd8aa039e7ce78ffde6c9d7b64b7879f77261077",
    5: "35971c3db99009acf2f9163d2fdceaf333cf156dc9d357e83f6f957732047adb",
}
DATASET_INITIAL_CONDITION_SEEDS = {
    1: 2_158,
    2: 4_792,
    3: 3_174,
    4: 688,
    5: 5_727,
}
ANALYSIS_ATTEMPT = re.compile(r"^analysis_attempt_(\d{3,})$")
COUNT_KEYS = (
    "correctly_classified_in_negative_basin",
    "correctly_classified_in_positive_basin",
    "misclassified_in_negative_basin",
    "misclassified_in_positive_basin",
    "outside_both_basins",
)
RESULT_FIELDS = (
    "dataset",
    "training_seed",
    "dataset_initial_condition_seed",
    "status",
    "root_association_status",
    "headline_valid",
    "training_data_sha256",
    "checkpoint_sha256",
    "conditioned_trajectories",
    "correctly_classified_in_negative_basin",
    "correctly_classified_in_negative_basin_percentage",
    "correctly_classified_in_positive_basin",
    "correctly_classified_in_positive_basin_percentage",
    "misclassified_in_negative_basin",
    "misclassified_in_negative_basin_percentage",
    "misclassified_in_positive_basin",
    "misclassified_in_positive_basin_percentage",
    "outside_both_basins",
    "outside_both_basins_percentage",
    "combined_correct_count",
    "combined_correct_percentage",
    "attractor_count",
    "attractor_nodes",
    "negative_basin_label",
    "positive_basin_label",
    "morse_nodes",
    "morse_edges",
    "analysis_status",
    "analysis_attempt",
    "full_strict_roa_status",
    "exact_blocker_lca_status",
    "elapsed_seconds",
    "output_dir",
    "failure_reason",
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset: int
    initial_condition_seed: int
    train_data: Path
    train_data_sha256: str

    @property
    def directory_name(self) -> str:
        return f"dataset_{self.dataset}"

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_initial_condition_seed": self.initial_condition_seed,
            "train_data": _file_record(
                self.train_data,
                expected_sha256=self.train_data_sha256,
            ),
        }


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
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


def _dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            dataset=dataset,
            initial_condition_seed=DATASET_INITIAL_CONDITION_SEEDS[dataset],
            train_data=(
                REFERENCE_ROOT
                / "computations"
                / f"run_dataset_{dataset}"
                / "train_data.csv"
            ),
            train_data_sha256=EXPECTED_DATASET_HASHES[dataset],
        )
        for dataset in sorted(EXPECTED_DATASET_HASHES)
    )


def _training_specs() -> tuple[sweep.FullBatchRunSpec, ...]:
    return tuple(
        sweep.FullBatchRunSpec(
            run_id=f"seed_{seed:02d}_lr3e3_e4000",
            seed=seed,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
        )
        for seed in TRAINING_SEEDS
    )


def _cmgdb_provenance() -> dict[str, Any]:
    native_module_name = getattr(study.CMGDB.ComputeMorseGraph, "__module__", "")
    native_module = importlib.import_module(native_module_name)
    native_binary = Path(native_module.__file__).resolve()
    return {
        "version": importlib.metadata.version("cmgdb"),
        "python_module": _file_record(Path(study.CMGDB.__file__)),
        "native_extension": _file_record(native_binary),
    }


def _runtime_provenance() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _build_plan() -> dict[str, Any]:
    canonical_inputs = study.verify_exact_inputs(REFERENCE_ROOT)
    datasets = _dataset_specs()
    implementations = {
        name: _file_record(path)
        for name, path in {
            "matched_driver": DRIVER_IMPLEMENTATION,
            "training_sweep": Path(sweep.__file__),
            "batch_analyzer": Path(analyze.__file__),
            "dimension_study": Path(study.__file__),
            "optional_roa_augmentation": Path(repeat.__file__),
        }.items()
    }
    return {
        "purpose": (
            "matched d=1 robustness experiment over the five archived d=2 "
            "training datasets and three explicit model initializations"
        ),
        "created_at_utc": _utc_now(),
        "design": {
            "datasets": [dataset.provenance() for dataset in datasets],
            "training_seeds": list(TRAINING_SEEDS),
            "runs_per_dataset": len(TRAINING_SEEDS),
            "total_runs": len(datasets) * len(TRAINING_SEEDS),
            "same_seed_across_datasets_is_a_paired_initialization": True,
            "d2_archive_trials_have_unknown_initialization_seeds": True,
            "cellwise_d1_d2_trial_pairing_permitted": False,
            "dataset_level_dimension_comparison_is_primary": True,
        },
        "training_protocol": {
            "entrypoint": "latentdynamics.training.train_reference_full_batch",
            "device": DEVICE,
            "architecture": study.reference_architecture(1).model_dump(mode="json"),
            "epochs": EPOCHS,
            "updates_per_epoch": 1,
            "full_batch_rows": sweep.TRAINING_ROWS,
            "dtype": "float32",
            "optimizer": {
                "name": "Adam",
                "learning_rate": LEARNING_RATE,
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "monitor": "full_batch_train_loss_total",
                "mode": "min",
                "factor": 0.5,
                "patience": 100,
                "threshold": 1e-4,
                "threshold_mode": "rel",
                "min_lr": 1e-6,
            },
            "objective": "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)",
            "validation": False,
            "early_stopping": False,
            "checkpoint_selection": "fixed final epoch",
        },
        "primary_analysis_protocol": {
            "dimension": 1,
            "bounds": "E(current and next training states) plus 10% margin",
            "precompute": "257 persisted level-8 corner values",
            "cmgdb_subdivisions": [8, 8, 8],
            "uniform_cells": 256,
            "padding": True,
            "basin_semantics": (
                "CMGDB.MorseSingletonReachability complete reachable set "
                "equals one singleton attractor"
            ),
            "headline_validity": (
                "exactly two uniform minimal attractors and distinct unique "
                "root singleton associations"
            ),
            "adaptive_required": False,
            "trajectory_truth_used_for_training_or_selection": False,
        },
        "optional_postprocessing": {
            "full_strict_singleton_roa": True,
            "exact_blocker_lca_roa": True,
            "adaptive_topology": True,
            "required_for_primary_statistics": False,
        },
        "fixed_evaluation_inputs": {
            "trajectory_labels": _file_record(
                canonical_inputs.trajectory_labels,
                expected_sha256=canonical_inputs.hashes[
                    "traj_attractors.pkl"
                ],
            ),
            "stable_roots": _file_record(
                canonical_inputs.stable_roots,
                expected_sha256=canonical_inputs.hashes[
                    "stable_solutions.csv"
                ],
            ),
        },
        "implementations": implementations,
        "cmgdb": _cmgdb_provenance(),
        "runtime_at_plan_creation": _runtime_provenance(),
    }


def _plan_envelope(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_sha256": _payload_sha256(plan),
        "plan": plan,
    }


def _iter_frozen_file_records(plan: dict[str, Any]) -> Sequence[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(
        dataset["train_data"] for dataset in plan["design"]["datasets"]
    )
    records.extend(plan["fixed_evaluation_inputs"].values())
    records.extend(plan["implementations"].values())
    records.extend(
        (
            plan["cmgdb"]["python_module"],
            plan["cmgdb"]["native_extension"],
        )
    )
    return records


def _validate_plan_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"schema_version", "plan_sha256", "plan"}:
        raise ValueError("malformed matched-run plan envelope")
    if payload["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported matched-run plan schema")
    plan = payload["plan"]
    if not isinstance(plan, dict) or payload["plan_sha256"] != _payload_sha256(plan):
        raise ValueError("matched-run plan hash mismatch")
    if plan.get("training_protocol", {}).get("device") != DEVICE:
        raise ValueError("matched-run plan is not frozen to the audited CPU backend")
    for record in _iter_frozen_file_records(plan):
        path = Path(record["path"])
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"frozen matched-run source changed: {path}")
    current_cmgdb = _cmgdb_provenance()
    for key in ("version", "python_module", "native_extension"):
        if current_cmgdb[key] != plan["cmgdb"][key]:
            raise ValueError(f"installed cmgdb changed at field {key}")
    return plan


def _create_or_load_plan(output_root: Path) -> tuple[dict[str, Any], str]:
    target = output_root.resolve()
    plan_path = target / "experiment_plan.json"
    if plan_path.exists():
        envelope = _read_json(plan_path)
        plan = _validate_plan_envelope(envelope)
        return plan, str(envelope["plan_sha256"])
    if target.exists():
        raise FileExistsError(
            f"{target} exists without experiment_plan.json; refusing reuse"
        )
    plan = _build_plan()
    envelope = _plan_envelope(plan)
    target.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(plan_path, envelope)
    return plan, str(envelope["plan_sha256"])


def _dataset_from_plan(plan: dict[str, Any], dataset: int) -> DatasetSpec:
    records = {
        int(record["dataset"]): record
        for record in plan["design"]["datasets"]
    }
    record = records[dataset]
    train = record["train_data"]
    return DatasetSpec(
        dataset=dataset,
        initial_condition_seed=int(record["dataset_initial_condition_seed"]),
        train_data=Path(train["path"]),
        train_data_sha256=str(train["sha256"]),
    )


def _run_training(
    *,
    output_root: Path,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset in sorted(EXPECTED_DATASET_HASHES):
        spec = _dataset_from_plan(plan, dataset)
        sweep_root = output_root / spec.directory_name / "training_sweep"
        print(
            f"dataset={dataset} stage=train seeds={TRAINING_SEEDS}",
            flush=True,
        )
        try:
            summary = sweep.run_sweep(
                output_dir=sweep_root,
                device_name=DEVICE,
                run_specs=_training_specs(),
                train_data=spec.train_data,
                verbose=True,
            )
            summaries.append(
                {
                    "dataset": dataset,
                    "status": (
                        "complete"
                        if summary["all_runs_completed"]
                        else "complete_with_failures"
                    ),
                    "counts": summary["counts"],
                    "plan_sha256": summary["plan_sha256"],
                }
            )
        except Exception as error:
            summaries.append(
                {
                    "dataset": dataset,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
    return summaries


def _analysis_attempts(dataset_root: Path) -> list[tuple[int, Path]]:
    attempts: list[tuple[int, Path]] = []
    if not dataset_root.is_dir():
        return attempts
    for child in dataset_root.iterdir():
        match = ANALYSIS_ATTEMPT.fullmatch(child.name)
        if match is not None and child.is_dir():
            attempts.append((int(match.group(1)), child))
    return sorted(attempts)


def _terminal_analysis(path: Path, expected_runs: int) -> bool:
    try:
        manifest = _read_json(path / "batch_manifest.json")
        results = _read_json(path / "results_by_run.json")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return False
    rows = results.get("results_in_frozen_plan_order")
    return bool(
        manifest.get("status") in {"complete", "complete_with_failures"}
        and manifest.get("rows_processed") == expected_runs
        and isinstance(rows, list)
        and len(rows) == expected_runs
    )


def _latest_terminal_analysis(
    dataset_root: Path,
    *,
    expected_runs: int,
) -> tuple[int, Path] | None:
    for index, path in reversed(_analysis_attempts(dataset_root)):
        if _terminal_analysis(path, expected_runs):
            return index, path
    return None


def _run_analysis(
    *,
    output_root: Path,
    plan: dict[str, Any],
    batch_points: int | str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    expected = len(TRAINING_SEEDS)
    for dataset in sorted(EXPECTED_DATASET_HASHES):
        spec = _dataset_from_plan(plan, dataset)
        dataset_root = output_root / spec.directory_name
        source = dataset_root / "training_sweep"
        try:
            frozen = analyze._verify_source_sweep(source)
        except Exception as error:
            summaries.append(
                {
                    "dataset": dataset,
                    "status": "waiting_for_valid_training_sweep",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            continue
        if len(frozen.candidates) != expected:
            summaries.append(
                {
                    "dataset": dataset,
                    "status": "waiting_for_completed_training",
                    "completed_valid": len(frozen.candidates),
                    "expected": expected,
                }
            )
            continue
        terminal = _latest_terminal_analysis(
            dataset_root,
            expected_runs=expected,
        )
        if terminal is not None:
            index, path = terminal
            summaries.append(
                {
                    "dataset": dataset,
                    "status": "already_complete",
                    "analysis_attempt": index,
                    "analysis_root": str(path),
                }
            )
            continue
        attempts = _analysis_attempts(dataset_root)
        index = max((item[0] for item in attempts), default=0) + 1
        target = dataset_root / f"analysis_attempt_{index:03d}"
        print(
            f"dataset={dataset} stage=uniform_analysis attempt={index:03d}",
            flush=True,
        )
        try:
            result = analyze.run_batch_analysis(
                source_sweep=source,
                analysis_root=target,
                device_name=DEVICE,
                batch_points=batch_points,
                train_data=spec.train_data,
                uniform_only=True,
            )
            summaries.append(
                {
                    "dataset": dataset,
                    "status": result["status"],
                    "analysis_attempt": index,
                    "analysis_root": str(target),
                    "analysis_complete": result["analysis_complete"],
                    "analysis_failed": result["analysis_failed"],
                }
            )
        except Exception as error:
            summaries.append(
                {
                    "dataset": dataset,
                    "status": "failed",
                    "analysis_attempt": index,
                    "analysis_root": str(target),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
    return summaries


def _optional_roa_augmentation(
    *,
    output_root: Path,
    batch_points: int | str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset in sorted(EXPECTED_DATASET_HASHES):
        dataset_root = output_root / f"dataset_{dataset}"
        terminal = _latest_terminal_analysis(
            dataset_root,
            expected_runs=len(TRAINING_SEEDS),
        )
        if terminal is None:
            summaries.append(
                {"dataset": dataset, "status": "waiting_for_analysis"}
            )
            continue
        attempt_index, analysis_root = terminal
        for plan_index, training_seed in enumerate(TRAINING_SEEDS, start=1):
            run_id = f"seed_{training_seed:02d}_lr3e3_e4000"
            run = analysis_root / "by_run" / f"run_{plan_index:03d}_{run_id}"
            manifest = run / "topology_roa_augmentation.json"
            if manifest.is_file():
                record = _read_json(manifest)
                summaries.append(
                    {
                        "dataset": dataset,
                        "training_seed": training_seed,
                        "status": "already_processed",
                        "augmentation_status": record.get("status"),
                    }
                )
                continue
            try:
                record = repeat._recover_topology_and_roa(
                    run=run,
                    device=torch.device(DEVICE),
                    batch_points=batch_points,
                )
                summaries.append(
                    {
                        "dataset": dataset,
                        "training_seed": training_seed,
                        "status": record["status"],
                        "analysis_attempt": attempt_index,
                    }
                )
            except Exception as error:
                summaries.append(
                    {
                        "dataset": dataset,
                        "training_seed": training_seed,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
    return summaries


def _analysis_failure_reason(run: Path, analysis_row: dict[str, Any]) -> str:
    parts = [
        str(analysis_row.get(field, "")).strip()
        for field in ("error_stage", "error_type", "error_message")
        if str(analysis_row.get(field, "")).strip()
    ]
    manifest = run / "analysis_manifest.json"
    if not parts and manifest.is_file():
        payload = _read_json(manifest)
        parts = [
            str(payload.get(field, "")).strip()
            for field in ("error_stage", "error_type", "error_message")
            if str(payload.get(field, "")).strip()
        ]
    return ": ".join(parts)


def _flat_result_row(
    *,
    dataset: DatasetSpec,
    training_seed: int,
    plan_index: int,
    attempt_index: int,
    analysis_root: Path,
    analysis_row: dict[str, Any],
) -> dict[str, Any]:
    run_id = f"seed_{training_seed:02d}_lr3e3_e4000"
    run = analysis_root / "by_run" / f"run_{plan_index:03d}_{run_id}"
    statistics_path = run / "basin_statistics.json"
    uniform_marker = run / "stage_status" / "uniform.json"
    row: dict[str, Any] = {
        "dataset": dataset.dataset,
        "training_seed": training_seed,
        "dataset_initial_condition_seed": dataset.initial_condition_seed,
        "status": "invalid",
        "root_association_status": "unavailable",
        "headline_valid": False,
        "training_data_sha256": dataset.train_data_sha256,
        "checkpoint_sha256": analysis_row.get("checkpoint_sha256"),
        "analysis_status": analysis_row.get("status"),
        "analysis_attempt": attempt_index,
        "output_dir": str(run.resolve()),
        "failure_reason": _analysis_failure_reason(run, analysis_row),
    }
    for key in COUNT_KEYS:
        row[key] = None
        row[f"{key}_percentage"] = None
    row.update(
        {
            "conditioned_trajectories": None,
            "combined_correct_count": None,
            "combined_correct_percentage": None,
            "attractor_count": None,
            "attractor_nodes": None,
            "negative_basin_label": None,
            "positive_basin_label": None,
            "morse_nodes": None,
            "morse_edges": None,
            "elapsed_seconds": None,
        }
    )
    if not statistics_path.is_file() or not uniform_marker.is_file():
        if not row["failure_reason"]:
            row["failure_reason"] = "uniform basin statistics unavailable"
        return row
    payload = _read_json(statistics_path)
    marker = _read_json(uniform_marker)
    statistics_payload = payload["statistics"]
    counts = statistics_payload["counts"]
    percentages = statistics_payload["percentages"]
    attractors = [int(value) for value in payload["cmgdb"]["attractor_nodes"]]
    negative = int(payload["stable_roots"]["negative_basin_label"])
    positive = int(payload["stable_roots"]["positive_basin_label"])
    root_valid = (
        len(attractors) == 2
        and negative != positive
        and {negative, positive}.issubset(set(attractors))
    )
    conditioned = int(statistics_payload["conditioned_trajectories"])
    count_sum = sum(int(counts[key]) for key in COUNT_KEYS)
    percentage_sum = sum(float(percentages[key]) for key in COUNT_KEYS)
    conservation_valid = (
        conditioned == EXPECTED_CONDITIONED
        and count_sum == conditioned
        and math.isclose(
            percentage_sum,
            100.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    headline_valid = root_valid and conservation_valid
    for key in COUNT_KEYS:
        row[key] = int(counts[key])
        row[f"{key}_percentage"] = float(percentages[key])
    combined_count = int(counts[COUNT_KEYS[0]]) + int(counts[COUNT_KEYS[1]])
    row.update(
        {
            "status": "complete" if headline_valid else "invalid",
            "root_association_status": "valid" if root_valid else "invalid",
            "headline_valid": headline_valid,
            "conditioned_trajectories": conditioned,
            "combined_correct_count": combined_count,
            "combined_correct_percentage": 100.0
            * combined_count
            / conditioned,
            "attractor_count": len(attractors),
            "attractor_nodes": attractors,
            "negative_basin_label": negative,
            "positive_basin_label": positive,
            "morse_nodes": int(marker["nodes"]),
            "morse_edges": int(marker["edges"]),
            "elapsed_seconds": (
                float(_read_json(run / "analysis_manifest.json")["elapsed_seconds"])
                if (run / "analysis_manifest.json").is_file()
                and "elapsed_seconds"
                in _read_json(run / "analysis_manifest.json")
                else None
            ),
            "failure_reason": (
                "" if headline_valid else "uniform validity/conservation failed"
            ),
        }
    )
    augmentation_path = run / "topology_roa_augmentation.json"
    augmentation = (
        _read_json(augmentation_path) if augmentation_path.is_file() else {}
    )
    row["full_strict_roa_status"] = (
        augmentation.get("full_uniform_roa", {}).get("status")
        if augmentation
        else "not_requested"
    )
    row["exact_blocker_lca_status"] = (
        "complete"
        if (run / "MG_uniform_s8" / "regions_of_attraction_exact.npz").is_file()
        else "not_requested"
    )
    return row


def _collect_results(
    *,
    output_root: Path,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_index in sorted(EXPECTED_DATASET_HASHES):
        dataset = _dataset_from_plan(plan, dataset_index)
        dataset_root = output_root / dataset.directory_name
        terminal = _latest_terminal_analysis(
            dataset_root,
            expected_runs=len(TRAINING_SEEDS),
        )
        if terminal is None:
            continue
        attempt_index, analysis_root = terminal
        payload = _read_json(analysis_root / "results_by_run.json")
        analysis_rows = {
            int(row["seed"]): row
            for row in payload["results_in_frozen_plan_order"]
        }
        for plan_index, training_seed in enumerate(TRAINING_SEEDS, start=1):
            row = analysis_rows.get(
                training_seed,
                {
                    "status": "source_not_completed",
                    "error_message": "analysis row unavailable",
                },
            )
            rows.append(
                _flat_result_row(
                    dataset=dataset,
                    training_seed=training_seed,
                    plan_index=plan_index,
                    attempt_index=attempt_index,
                    analysis_root=analysis_root,
                    analysis_row=row,
                )
            )
    return sorted(rows, key=lambda row: (row["dataset"], row["training_seed"]))


def _describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "sample_standard_deviation": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": (
            statistics.stdev(values) if len(values) > 1 else None
        ),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "complete"]
    by_dataset = []
    for dataset in sorted(EXPECTED_DATASET_HASHES):
        selected = [
            float(row["combined_correct_percentage"])
            for row in complete
            if row["dataset"] == dataset
        ]
        by_dataset.append(
            {
                "dataset": dataset,
                "combined_correct_percentage": _describe(selected),
            }
        )
    by_seed = []
    for seed in TRAINING_SEEDS:
        selected = [
            float(row["combined_correct_percentage"])
            for row in complete
            if row["training_seed"] == seed
        ]
        by_seed.append(
            {
                "training_seed": seed,
                "combined_correct_percentage": _describe(selected),
            }
        )
    pooled_counts = {
        key: sum(int(row[key]) for row in complete)
        for key in COUNT_KEYS
    }
    pooled_conditioned = sum(
        int(row["conditioned_trajectories"]) for row in complete
    )
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "status": (
            "complete"
            if len(rows) == 15 and len(complete) == 15
            else (
                "complete_with_invalid_runs"
                if len(rows) == 15
                else "incomplete"
            )
        ),
        "run_counts": {
            "expected": 15,
            "available": len(rows),
            "headline_valid": len(complete),
            "invalid": len(rows) - len(complete),
        },
        "combined_correct_percentage_descriptive": _describe(
            [float(row["combined_correct_percentage"]) for row in complete]
        ),
        "by_dataset": by_dataset,
        "by_training_seed": by_seed,
        "pooled_conditioned_statistics": {
            "conditioned_trajectories": pooled_conditioned,
            "counts": pooled_counts,
            "combined_correct_count": (
                pooled_counts["correctly_classified_in_negative_basin"]
                + pooled_counts["correctly_classified_in_positive_basin"]
            ),
            "combined_correct_percentage": (
                None
                if pooled_conditioned == 0
                else 100.0
                * (
                    pooled_counts["correctly_classified_in_negative_basin"]
                    + pooled_counts["correctly_classified_in_positive_basin"]
                )
                / pooled_conditioned
            ),
        },
        "results": list(rows),
    }


def _write_results_package(
    *,
    output_root: Path,
    plan_sha256: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = _aggregate(rows)
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "status": aggregate["status"],
        "experiment_plan_sha256": plan_sha256,
        "design": {
            "datasets": sorted(EXPECTED_DATASET_HASHES),
            "training_seeds": list(TRAINING_SEEDS),
            "primary_comparison_level": "dataset",
            "d2_trial_level_pairing": "prohibited_unknown_d2_initialization_seeds",
        },
        "results": list(rows),
    }
    _write_json_atomic(output_root / "results.json", payload)
    _write_json_atomic(output_root / "aggregate_statistics.json", aggregate)
    csv_path = output_root / "results.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(row.get(field), separators=(",", ":"))
                        if isinstance(row.get(field), (list, dict))
                        else row.get(field)
                    )
                    for field in RESULT_FIELDS
                }
            )
    temporary.replace(csv_path)
    return aggregate


def _package_manifest(output_root: Path, plan_sha256: str) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "package_manifest.json":
            continue
        files[str(path.relative_to(output_root))] = {
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
    payload = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "experiment_plan_sha256": plan_sha256,
        "file_count": len(files),
        "files": files,
    }
    _write_json_atomic(output_root / "package_manifest.json", payload)
    return payload


def _next_invocation_path(output_root: Path) -> tuple[int, Path]:
    summaries = output_root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in summaries.iterdir():
        match = re.fullmatch(r"invocation_(\d{4,})\.json", path.name)
        if match is not None:
            indices.append(int(match.group(1)))
    index = max(indices, default=0) + 1
    return index, summaries / f"invocation_{index:04d}.json"


def run_experiment(
    *,
    output_root: Path,
    stage: str,
    batch_points: int | str,
    augment_full_roa: bool,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    started = time.perf_counter()
    plan, plan_sha256 = _create_or_load_plan(output_root)
    invocation_index, invocation_path = _next_invocation_path(output_root)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "invocation": invocation_index,
        "started_at_utc": _utc_now(),
        "stage": stage,
        "output_root": str(output_root),
        "experiment_plan_sha256": plan_sha256,
        "training": [],
        "analysis": [],
        "optional_roa_augmentation": [],
    }
    try:
        if stage in {"train", "all"}:
            summary["training"] = _run_training(
                output_root=output_root,
                plan=plan,
            )
        if stage in {"analyze", "all"}:
            summary["analysis"] = _run_analysis(
                output_root=output_root,
                plan=plan,
                batch_points=batch_points,
            )
        if augment_full_roa:
            summary["optional_roa_augmentation"] = _optional_roa_augmentation(
                output_root=output_root,
                batch_points=batch_points,
            )
        rows = _collect_results(output_root=output_root, plan=plan)
        aggregate = _write_results_package(
            output_root=output_root,
            plan_sha256=plan_sha256,
            rows=rows,
        )
        summary.update(
            {
                "status": aggregate["status"],
                "run_counts": aggregate["run_counts"],
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    except Exception as error:
        summary.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _write_json_exclusive(invocation_path, summary)
        raise
    _write_json_exclusive(invocation_path, summary)
    _write_json_atomic(
        output_root / "latest_summary.json",
        {
            "schema_version": 1,
            "invocation": invocation_index,
            "summary": {
                "path": str(invocation_path.relative_to(output_root)),
                "sha256": _sha256(invocation_path),
            },
            "status": summary["status"],
            "run_counts": summary["run_counts"],
        },
    )
    _package_manifest(output_root, plan_sha256)
    return summary


def _batch_points(value: str) -> int | str:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("batch points must be positive or auto")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help=(
            "root of the archived reference inputs "
            "(computations/run_dataset_N/train_data.csv, traj_attractors.pkl, ...)"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("plan", "train", "analyze", "all"),
        default="all",
    )
    parser.add_argument(
        "--device",
        choices=(DEVICE,),
        default=DEVICE,
        help="audited matched backend; only cpu is accepted",
    )
    parser.add_argument(
        "--batch-points",
        type=_batch_points,
        default="auto",
    )
    parser.add_argument(
        "--augment-full-roa",
        action="store_true",
        help=(
            "after primary analysis, add full strict-singleton, exact "
            "blocker/LCA, and adaptive topology products"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    global REFERENCE_ROOT
    REFERENCE_ROOT = args.reference_root.resolve()
    summary = run_experiment(
        output_root=args.output_root,
        stage=args.stage,
        batch_points=args.batch_points,
        augment_full_roa=args.augment_full_roa,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "run_counts": summary["run_counts"],
                "output_root": summary["output_root"],
                "experiment_plan_sha256": summary[
                    "experiment_plan_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if args.stage == "plan" or summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
