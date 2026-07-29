"""Run the training-only matched Chafee d=3 five-dataset by three-seed matrix.

The fifteen fixed trials use Marcio's five archived training-pair CSVs and
explicit model-initialization seeds 0, 1, and 2.  Every trial performs exactly
4,000 direct full-batch Adam updates on CPU with the decoded reconstruction
plus decoded one-step prediction objective.  The architecture is the validated
Chafee architecture with latent width three.

This driver deliberately has no analysis dependency and never invokes CMGDB.
It writes to a fresh output root that is disjoint from the existing d=3 study.
The first invocation freezes the complete matrix, source hashes, architecture,
optimizer/scheduler semantics, runtime, and implementation hashes.  Subsequent
invocations resume at trial boundaries: verified completed trials are skipped,
while failed or interrupted trials receive a new immutable attempt directory.
No checkpoint, history, summary, or manifest is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

import sweep_chafee_d1_full_batch as reusable
from latentdynamics.config import ArchConfig
from latentdynamics.training import train_marcio_full_batch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
RUNNER_IMPLEMENTATION = Path(__file__).resolve()
DEFAULT_OUTPUT = (
    CODE_ROOT / "output" / "chafee_d3_matched_d2_archive_5x3_training_v1"
)
EXISTING_D3_RUN = (
    CODE_ROOT
    / "output"
    / "chafee_latent_dimension_study"
    / "latent_3d"
    / "seed_0"
)
EXISTING_D3_SIDECAR = EXISTING_D3_RUN / "models" / "autoencoder.json"
EXISTING_D3_SIDECAR_SHA256 = (
    "54ebb8b172b3a23de981ce2495de6bab6a605c3ccad285188ad433d5b73197d1"
)

TRAINING_SEEDS = (0, 1, 2)
DATASETS = (1, 2, 3, 4, 5)
DATASET_INITIAL_CONDITION_SEEDS = {
    1: 2_158,
    2: 4_792,
    3: 3_174,
    4: 688,
    5: 5_727,
}
DATASET_SHA256 = {
    1: "f50e3f44b4d6a4e5cf516ead5f3e44d2b4a3afdb0dff34c76613d0444caf8f7b",
    2: "b183613bf887b8eb2e85f780e440295dc16b1320bff178da6886c0d0dc93353d",
    3: "980f9320588ca49d87e319f7dc727a885692d12ccb523bac3c9ad92ffc33ad46",
    4: "6025463fc0b50334a6bdad35dd8aa039e7ce78ffde6c9d7b64b7879f77261077",
    5: "35971c3db99009acf2f9163d2fdceaf333cf156dc9d357e83f6f957732047adb",
}

DEVICE = "cpu"
EPOCHS = 4_000
LEARNING_RATE = 0.003
PLAN_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
EXPECTED_TRIALS = len(DATASETS) * len(TRAINING_SEEDS)
RESULT_FIELDS = (
    "plan_index",
    "dataset",
    "dataset_initial_condition_seed",
    "training_seed",
    "run_id",
    "status",
    "attempt",
    "training_data_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "history_path",
    "history_sha256",
    "training_summary_path",
    "training_summary_sha256",
    "loss_reconstruction",
    "loss_prediction",
    "loss_total",
    "final_learning_rate",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class D3Trial:
    """One dataset/initialization cell in the frozen 5 x 3 design."""

    plan_index: int
    dataset: int
    training_seed: int
    run_id: str
    training_spec: reusable.FullBatchRunSpec

    @property
    def source_key(self) -> str:
        return f"train_data_dataset_{self.dataset}"

    def plan_record(self) -> dict[str, Any]:
        return {
            "plan_index": self.plan_index,
            "dataset": self.dataset,
            "dataset_initial_condition_seed": (
                DATASET_INITIAL_CONDITION_SEEDS[self.dataset]
            ),
            "training_seed": self.training_seed,
            "run_id": self.run_id,
            "training_data_source": self.source_key,
            "training_spec": asdict(self.training_spec),
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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"cannot read valid JSON from {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
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


def _dataset_path(dataset: int) -> Path:
    return (
        PROJECT_ROOT
        / "archive"
        / "marcio"
        / "computations"
        / f"run_dataset_{dataset}"
        / "train_data.csv"
    )


def _d3_architecture() -> ArchConfig:
    expected = ArchConfig(
        high_dims=64,
        low_dims=3,
        encoder={
            "hidden_shapes": [64, 32],
            "activation": "tanh",
            "out_activation": "none",
        },
        latent_map={
            "hidden_shapes": [32, 32],
            "activation": "tanh",
            "out_activation": "none",
        },
        decoder={
            "hidden_shapes": [32, 64],
            "activation": "tanh",
            "out_activation": "none",
        },
    )
    sidecar_record = _file_record(
        EXISTING_D3_SIDECAR,
        expected_sha256=EXISTING_D3_SIDECAR_SHA256,
    )
    sidecar = _read_json(Path(sidecar_record["path"]))
    archived = ArchConfig.model_validate(sidecar.get("arch"))
    if archived != expected:
        raise ValueError(
            "the existing d=3 architecture sidecar differs from the explicit "
            "64-to-3 matched architecture"
        )
    return expected


def _trial_matrix() -> tuple[D3Trial, ...]:
    trials: list[D3Trial] = []
    for plan_index, (dataset, seed) in enumerate(
        (
            (dataset, seed)
            for dataset in DATASETS
            for seed in TRAINING_SEEDS
        ),
        start=1,
    ):
        run_id = (
            f"dataset_{dataset:02d}_seed_{seed:02d}_lr3e3_e4000"
        )
        trials.append(
            D3Trial(
                plan_index=plan_index,
                dataset=dataset,
                training_seed=seed,
                run_id=run_id,
                training_spec=reusable.FullBatchRunSpec(
                    run_id=run_id,
                    seed=seed,
                    epochs=EPOCHS,
                    learning_rate=LEARNING_RATE,
                ),
            )
        )
    return tuple(trials)


def _current_sources() -> dict[str, dict[str, Any]]:
    sources = {
        f"train_data_dataset_{dataset}": _file_record(
            _dataset_path(dataset),
            expected_sha256=DATASET_SHA256[dataset],
        )
        for dataset in DATASETS
    }
    implementation_paths = {
        "d3_runner_implementation": RUNNER_IMPLEMENTATION,
        "reusable_sweep_implementation": Path(reusable.__file__),
        "marcio_training_implementation": (
            CODE_ROOT / "src" / "latentdynamics" / "training" / "marcio.py"
        ),
        "checkpoint_implementation": (
            CODE_ROOT
            / "src"
            / "latentdynamics"
            / "training"
            / "checkpoints.py"
        ),
        "autoencoder_implementation": (
            CODE_ROOT
            / "src"
            / "latentdynamics"
            / "models"
            / "autoencoder.py"
        ),
        "architecture_schema_implementation": (
            CODE_ROOT
            / "src"
            / "latentdynamics"
            / "config"
            / "schema.py"
        ),
        "existing_d3_architecture_sidecar": EXISTING_D3_SIDECAR,
    }
    sources.update(
        {
            name: _file_record(
                path,
                expected_sha256=(
                    EXISTING_D3_SIDECAR_SHA256
                    if name == "existing_d3_architecture_sidecar"
                    else None
                ),
            )
            for name, path in implementation_paths.items()
        }
    )
    return sources


def _build_plan(
    *,
    sources: dict[str, dict[str, Any]],
    arch: ArchConfig,
    trials: Sequence[D3Trial],
) -> dict[str, Any]:
    frozen_trials = tuple(trials)
    if len(frozen_trials) != EXPECTED_TRIALS:
        raise ValueError(
            f"matched d=3 plan requires {EXPECTED_TRIALS} trials"
        )
    return {
        "purpose": (
            "training-only matched Chafee d=3 robustness matrix over five "
            "archived datasets and three explicit initialization seeds"
        ),
        "created_at_utc": _utc_now(),
        "training_entrypoint": (
            "latentdynamics.training.train_marcio_full_batch"
        ),
        "training_semantics": {
            "data_rows": reusable.TRAINING_ROWS,
            "high_dimension": reusable.HIGH_DIMENSION,
            "latent_dimension": 3,
            "dtype": "float32",
            "full_batch": True,
            "updates_per_epoch": 1,
            "epochs": EPOCHS,
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
                "monitor": "train.loss_total",
                "mode": "min",
                "factor": 0.5,
                "patience": 100,
                "threshold": 1e-4,
                "threshold_mode": "rel",
                "min_lr": 1e-6,
            },
            "objective": "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)",
            "validation_used": False,
            "early_stopping_used": False,
            "checkpoint_selection": "fixed final epoch",
        },
        "resolved_device": DEVICE,
        "architecture": arch.model_dump(mode="json"),
        "design": {
            "datasets": list(DATASETS),
            "dataset_initial_condition_seeds": {
                str(key): value
                for key, value in DATASET_INITIAL_CONDITION_SEEDS.items()
            },
            "training_seeds": list(TRAINING_SEEDS),
            "runs_per_dataset": len(TRAINING_SEEDS),
            "total_runs": len(frozen_trials),
            "same_seed_across_datasets_is_a_paired_initialization": True,
        },
        "trials": [trial.plan_record() for trial in frozen_trials],
        "sources": sources,
        "scope_guards": {
            "training_only": True,
            "cmgdb_imported_or_invoked": False,
            "morse_graph_or_roa_analysis_performed": False,
            "existing_d3_artifacts_mutated": False,
        },
        "runtime_at_plan_creation": reusable._runtime_provenance(),
    }


def _plan_envelope(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_sha256": _payload_sha256(plan),
        "plan": plan,
    }


def _validate_plan_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"schema_version", "plan_sha256", "plan"}:
        raise ValueError("malformed d=3 matched training plan")
    if payload["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported d=3 matched training plan schema")
    plan = payload["plan"]
    if not isinstance(plan, dict):
        raise ValueError("d=3 matched plan body must be an object")
    if payload["plan_sha256"] != _payload_sha256(plan):
        raise ValueError("d=3 matched training plan hash mismatch")
    if (
        plan.get("resolved_device") != DEVICE
        or plan.get("training_semantics", {}).get("latent_dimension") != 3
        or plan.get("scope_guards", {}).get("training_only") is not True
        or plan.get("scope_guards", {}).get(
            "cmgdb_imported_or_invoked"
        )
        is not False
    ):
        raise ValueError("d=3 matched plan violates frozen scope/protocol")
    reusable._validate_plan_sources(plan)
    arch = ArchConfig.model_validate(plan.get("architecture"))
    if arch != _d3_architecture():
        raise ValueError("frozen d=3 architecture changed")
    _trials_from_plan(plan)
    return plan


def _trials_from_plan(plan: dict[str, Any]) -> tuple[D3Trial, ...]:
    records = plan.get("trials")
    if not isinstance(records, list) or len(records) != EXPECTED_TRIALS:
        raise ValueError("frozen d=3 plan does not contain exactly 15 trials")
    trials: list[D3Trial] = []
    for expected_index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError("malformed d=3 trial record")
        spec = reusable.FullBatchRunSpec.from_payload(
            record["training_spec"]
        )
        trial = D3Trial(
            plan_index=int(record["plan_index"]),
            dataset=int(record["dataset"]),
            training_seed=int(record["training_seed"]),
            run_id=str(record["run_id"]),
            training_spec=spec,
        )
        if (
            trial.plan_index != expected_index
            or trial.dataset not in DATASETS
            or trial.training_seed not in TRAINING_SEEDS
            or trial.run_id != spec.run_id
            or record.get("training_data_source") != trial.source_key
        ):
            raise ValueError("d=3 trial differs from its frozen matrix identity")
        trials.append(trial)
    expected_pairs = {
        (dataset, seed)
        for dataset in DATASETS
        for seed in TRAINING_SEEDS
    }
    if {(trial.dataset, trial.training_seed) for trial in trials} != expected_pairs:
        raise ValueError("d=3 plan is not the complete 5 x 3 design")
    return tuple(trials)


def _assert_safe_output(output_root: Path) -> None:
    target = output_root.resolve()
    protected = EXISTING_D3_RUN.resolve()
    if (
        target == protected
        or target in protected.parents
        or protected in target.parents
    ):
        raise ValueError(
            f"training output {target} overlaps existing d=3 artifacts "
            f"at {protected}"
        )
    if output_root.is_symlink():
        raise ValueError("d=3 training output must not be a symlink")


def _create_or_load_plan(
    output_root: Path,
) -> tuple[dict[str, Any], str, tuple[D3Trial, ...], ArchConfig]:
    _assert_safe_output(output_root)
    plan_path = output_root / "experiment_plan.json"
    if plan_path.exists():
        envelope = _read_json(plan_path)
        plan = _validate_plan_envelope(envelope)
        return (
            plan,
            str(envelope["plan_sha256"]),
            _trials_from_plan(plan),
            ArchConfig.model_validate(plan["architecture"]),
        )
    if output_root.exists():
        raise FileExistsError(
            f"{output_root} exists without experiment_plan.json; refusing reuse"
        )
    sources = _current_sources()
    arch = _d3_architecture()
    trials = _trial_matrix()
    plan = _build_plan(sources=sources, arch=arch, trials=trials)
    envelope = _plan_envelope(plan)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(plan_path, envelope)
    return plan, str(envelope["plan_sha256"]), trials, arch


def _load_training_pairs(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return reusable._load_training_pairs(path)


def _validate_completed_trial(
    *,
    run_root: Path,
    trial: D3Trial,
    plan_sha256: str,
    arch: ArchConfig,
) -> dict[str, Any] | None:
    completion = reusable._validate_completed_run(
        run_root=run_root,
        spec=trial.training_spec,
        plan_sha256=plan_sha256,
    )
    if completion is None:
        return None
    attempt = run_root / "attempts" / f"attempt_{int(completion['attempt']):03d}"
    reusable._validate_training_artifacts(
        attempt=attempt,
        spec=trial.training_spec,
        arch=arch,
    )
    return completion


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


def _run_training(
    *,
    output_root: Path,
    plan: dict[str, Any],
    plan_sha256: str,
    trials: Sequence[D3Trial],
    arch: ArchConfig,
    verbose: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cached_dataset: int | None = None
    training_pairs: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ] | None = None
    for trial in trials:
        spec = trial.training_spec
        run_root = output_root / "runs" / trial.run_id
        row: dict[str, Any] = {
            "plan_index": trial.plan_index,
            "dataset": trial.dataset,
            "training_seed": trial.training_seed,
            "run_id": trial.run_id,
        }
        try:
            reusable._prepare_run_root(
                run_root,
                spec=spec,
                plan_sha256=plan_sha256,
            )
            completion = _validate_completed_trial(
                run_root=run_root,
                trial=trial,
                plan_sha256=plan_sha256,
                arch=arch,
            )
            if completion is not None:
                row.update(
                    {
                        "status": "already_completed",
                        "attempt": int(completion["attempt"]),
                        "checkpoint": completion["checkpoint"],
                    }
                )
                rows.append(row)
                if verbose:
                    print(
                        f"{trial.run_id}: verified complete; skipped",
                        flush=True,
                    )
                continue
        except Exception as error:
            row.update(
                {
                    "status": "invalid_existing_run",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            rows.append(row)
            if verbose:
                print(
                    f"{trial.run_id}: invalid existing run: {error}",
                    flush=True,
                )
            continue

        attempt_index, attempt = reusable._next_attempt_directory(run_root)
        _write_json_exclusive(
            attempt / "attempt_started.json",
            {
                "schema_version": 1,
                "status": "started",
                "started_at_utc": _utc_now(),
                "plan_sha256": plan_sha256,
                "trial": trial.plan_record(),
                "attempt": attempt_index,
            },
        )
        if verbose:
            print(
                f"{trial.run_id}: starting attempt {attempt_index:03d} "
                f"dataset={trial.dataset} seed={trial.training_seed} "
                f"epochs={spec.epochs} device={DEVICE}",
                flush=True,
            )
        try:
            if cached_dataset != trial.dataset or training_pairs is None:
                source = Path(plan["sources"][trial.source_key]["path"])
                training_pairs = _load_training_pairs(source)
                cached_dataset = trial.dataset
            x, y = training_pairs
            train_marcio_full_batch(
                arch=arch,
                x=x,
                y=y,
                epochs=spec.epochs,
                learning_rate=spec.learning_rate,
                seed=spec.seed,
                device=torch.device(DEVICE),
                output_dir=attempt,
                scheduler_factor=spec.scheduler_factor,
                scheduler_patience=spec.scheduler_patience,
                scheduler_threshold=spec.scheduler_threshold,
                scheduler_min_lr=spec.scheduler_min_lr,
                verbose=verbose,
            )
            manifest = reusable._build_artifact_manifest(
                attempt=attempt,
                run_root=run_root,
                spec=spec,
                arch=arch,
                plan=plan,
                plan_sha256=plan_sha256,
                attempt_index=attempt_index,
            )
            manifest.update(
                {
                    "matched_d3_trial": trial.plan_record(),
                    "training_data": plan["sources"][trial.source_key],
                    "scope_guards": plan["scope_guards"],
                }
            )
            manifest_path = _write_json_exclusive(
                attempt / "artifact_manifest.json",
                manifest,
            )
            completion = reusable._completion_payload(
                manifest_path=manifest_path,
                run_root=run_root,
                manifest=manifest,
            )
            _write_json_exclusive(run_root / "completed.json", completion)
            row.update(
                {
                    "status": "completed",
                    "attempt": attempt_index,
                    "checkpoint": completion["checkpoint"],
                }
            )
            if verbose:
                print(f"{trial.run_id}: completed", flush=True)
        except Exception as error:
            _write_json_exclusive(
                attempt / "attempt_failed.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "failed_at_utc": _utc_now(),
                    "plan_sha256": plan_sha256,
                    "trial": trial.plan_record(),
                    "attempt": attempt_index,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            row.update(
                {
                    "status": "failed",
                    "attempt": attempt_index,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            if verbose:
                print(
                    f"{trial.run_id}: failed attempt {attempt_index:03d}: "
                    f"{error}",
                    flush=True,
                )
        rows.append(row)
    return rows


def _completed_result(
    *,
    output_root: Path,
    trial: D3Trial,
    plan: dict[str, Any],
    plan_sha256: str,
    arch: ArchConfig,
) -> dict[str, Any]:
    run_root = output_root / "runs" / trial.run_id
    base: dict[str, Any] = {
        "plan_index": trial.plan_index,
        "dataset": trial.dataset,
        "dataset_initial_condition_seed": (
            DATASET_INITIAL_CONDITION_SEEDS[trial.dataset]
        ),
        "training_seed": trial.training_seed,
        "run_id": trial.run_id,
        "training_data_sha256": plan["sources"][trial.source_key]["sha256"],
        "status": "not_completed",
        "error_type": "",
        "error_message": "",
    }
    try:
        completion = _validate_completed_trial(
            run_root=run_root,
            trial=trial,
            plan_sha256=plan_sha256,
            arch=arch,
        )
    except Exception as error:
        return {
            **base,
            "status": "invalid_existing_run",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    if completion is None:
        return base
    attempt_index = int(completion["attempt"])
    attempt = run_root / "attempts" / f"attempt_{attempt_index:03d}"
    manifest = _read_json(attempt / "artifact_manifest.json")
    summary = _read_json(attempt / "training_summary.json")
    history = _read_json(attempt / "logs" / "history.json")["train"]
    artifacts = manifest["artifacts"]
    return {
        **base,
        "status": "complete",
        "attempt": attempt_index,
        "checkpoint_path": artifacts["checkpoint"]["path"],
        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "checkpoint_size_bytes": artifacts["checkpoint"]["size_bytes"],
        "history_path": artifacts["history"]["path"],
        "history_sha256": artifacts["history"]["sha256"],
        "training_summary_path": artifacts["training_summary"]["path"],
        "training_summary_sha256": artifacts["training_summary"]["sha256"],
        **{
            name: float(summary["final_epoch_train"][name])
            for name in (
                "loss_reconstruction",
                "loss_prediction",
                "loss_total",
            )
        },
        "final_learning_rate": float(history["learning_rate"][-1]),
    }


def _collect_results(
    *,
    output_root: Path,
    plan: dict[str, Any],
    plan_sha256: str,
    trials: Sequence[D3Trial],
    arch: ArchConfig,
) -> list[dict[str, Any]]:
    return [
        _completed_result(
            output_root=output_root,
            trial=trial,
            plan=plan,
            plan_sha256=plan_sha256,
            arch=arch,
        )
        for trial in trials
    ]


def _write_results(
    *,
    output_root: Path,
    plan_sha256: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    completed = sum(row["status"] == "complete" for row in rows)
    invalid = sum(row["status"] == "invalid_existing_run" for row in rows)
    status = (
        "complete"
        if completed == EXPECTED_TRIALS
        else ("invalid" if invalid else "incomplete")
    )
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "status": status,
        "experiment_plan_sha256": plan_sha256,
        "counts": {
            "expected": EXPECTED_TRIALS,
            "complete": completed,
            "not_completed": EXPECTED_TRIALS - completed - invalid,
            "invalid": invalid,
        },
        "training_only": True,
        "cmgdb_invoked": False,
        "results_in_frozen_plan_order": list(rows),
    }
    _write_json_atomic(output_root / "training_results.json", payload)
    csv_path = output_root / "training_results.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in RESULT_FIELDS})
    temporary.replace(csv_path)
    return payload


def _package_manifest(
    *,
    output_root: Path,
    plan_sha256: str,
) -> dict[str, Any]:
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


def run_experiment(
    *,
    output_root: Path,
    stage: str,
    verbose: bool,
) -> dict[str, Any]:
    target = output_root.resolve()
    started = time.perf_counter()
    plan, plan_sha256, trials, arch = _create_or_load_plan(target)
    invocation_index, invocation_path = _next_invocation_path(target)
    invocation_rows: list[dict[str, Any]] = []
    if stage == "train":
        invocation_rows = _run_training(
            output_root=target,
            plan=plan,
            plan_sha256=plan_sha256,
            trials=trials,
            arch=arch,
            verbose=verbose,
        )
    results_rows = _collect_results(
        output_root=target,
        plan=plan,
        plan_sha256=plan_sha256,
        trials=trials,
        arch=arch,
    )
    results = _write_results(
        output_root=target,
        plan_sha256=plan_sha256,
        rows=results_rows,
    )
    summary = {
        "schema_version": 1,
        "invocation": invocation_index,
        "stage": stage,
        "started_at_utc": _utc_now(),
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "output_root": str(target),
        "experiment_plan_sha256": plan_sha256,
        "status": results["status"],
        "counts": results["counts"],
        "invocation_rows": invocation_rows,
        "training_only": True,
        "cmgdb_invoked": False,
        "runtime": reusable._runtime_provenance(),
    }
    _write_json_exclusive(invocation_path, summary)
    _write_json_atomic(
        target / "latest_summary.json",
        {
            "schema_version": 1,
            "invocation": invocation_index,
            "summary": {
                "path": str(invocation_path.relative_to(target)),
                "sha256": _sha256(invocation_path),
            },
            "status": results["status"],
            "counts": results["counts"],
        },
    )
    _package_manifest(output_root=target, plan_sha256=plan_sha256)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=("plan", "train", "validate"),
        default="train",
        help=(
            "plan creates/validates provenance only; train creates or resumes "
            "the fixed 15 trials; validate hash-checks completed artifacts"
        ),
    )
    parser.add_argument(
        "--device",
        choices=(DEVICE,),
        default=DEVICE,
        help="the audited matched backend is fixed to CPU",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_experiment(
        output_root=args.output_root,
        stage=args.stage,
        verbose=not args.quiet,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "counts": summary["counts"],
                "output_root": summary["output_root"],
                "experiment_plan_sha256": summary[
                    "experiment_plan_sha256"
                ],
                "training_only": True,
                "cmgdb_invoked": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.stage in {"plan", "validate"}:
        return 0
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
