"""Run the isolated Leslie3D Example 2 Marcio-style 5 x 3 experiment.

The experiment has two frozen seed axes: five independently sampled training
datasets and three model initializations per dataset.  Data and scalers are
shared only within a dataset.  Each model is trained on the resulting 30,000
scaled transition pairs by ``train_marcio_full_batch`` and is evaluated after
the final update on the fixed 6,000-pair validation holdout.

The seven stages can be launched together or independently::

    python scripts/run_leslie3d_example2_marcio_5x3.py \
        --stages data,scale,train,diagnose,morse,render,metrics \
        --skip-completed

The isolated layout is::

    data/leslie3d_example2_marcio_5x3_v1/dataset_01/
    output/leslie3d_example2_marcio_5x3_v1/dataset_01/seed_0/

Resumption is at stage/cell boundaries.  ``--skip-completed`` verifies the
expected artifacts before skipping them.  In particular, an interrupted train
stage that already wrote a complete checkpoint resumes with holdout evaluation
instead of repeating 4,000 optimizer updates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from latentdynamics.cli import diagnose as diagnose_stage
from latentdynamics.cli import make_data as data_stage
from latentdynamics.cli import metrics as metrics_stage_module
from latentdynamics.cli import morse_graph as morse_stage
from latentdynamics.cli import provenance as provenance_stage
from latentdynamics.cli import render as render_stage_module
from latentdynamics.cli import scale_data as scale_stage
from latentdynamics.config import ExperimentConfig, load_config
from latentdynamics.sampling import load_scaler
from latentdynamics.training import (
    has_new_checkpoint,
    load_any_checkpoint,
    train_marcio_full_batch,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_STEM = "leslie3d_example2_marcio_5x3"
EXPERIMENT_LABEL = "leslie3d_example2_marcio_5x3_v1"
DEFAULT_DATA_ROOT = CODE_ROOT / "data" / EXPERIMENT_LABEL
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "output" / EXPERIMENT_LABEL

DATASET_INITIAL_CONDITION_SEEDS: dict[int, int] = {
    1: 2_158,
    2: 4_792,
    3: 3_174,
    4: 688,
    5: 5_727,
}
MODEL_SEEDS: tuple[int, ...] = (0, 1, 2)
ALL_STAGES: tuple[str, ...] = (
    "data",
    "scale",
    "train",
    "diagnose",
    "morse",
    "render",
    "metrics",
)
DATASET_STAGES = frozenset({"data", "scale"})
CELL_STAGES = frozenset(ALL_STAGES) - DATASET_STAGES

TRAIN_FILE = "train"
TRAIN_PAIRS = 30_000
HOLDOUT_PAIRS = 6_000
OBJECTIVE = "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)"
CMGDB_BOUNDS_DATA_ROLE = "train_pairs"
CMGDB_BOUNDS_SOURCE = "encoded_train_pairs"
PLAN_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Cell:
    """One model initialization trained on one generated dataset."""

    index: int
    dataset: int
    initial_condition_seed: int
    model_seed: int
    data_dir: str
    output_dir: str


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _artifact_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not _nonempty(path):
        raise FileNotFoundError(f"required artifact is missing or empty: {path}")
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _validate_protocol(cfg: ExperimentConfig) -> None:
    """Fail closed if the packaged experiment drifts from the requested run."""

    expected_components = {
        "encoder": ((64, 32), "tanh", "none"),
        "latent_map": ((32, 32), "tanh", "none"),
        "decoder": ((32, 64), "tanh", "none"),
    }
    observed_components = {
        name: (
            cfg.arch.component(name).hidden_shapes,
            cfg.arch.component(name).activation,
            cfg.arch.component(name).out_activation,
        )
        for name in expected_components
    }
    checks = {
        "system": cfg.system.name == "leslie3d",
        "system_params": cfg.system.params
        == {
            "th1": 28.9,
            "th2": 29.8,
            "th3": 22.0,
            "survival_p1": 0.7,
            "survival_p2": 0.7,
        },
        "dimensions": (cfg.arch.high_dims, cfg.arch.low_dims) == (3, 2),
        "architecture": observed_components == expected_components,
        "training": (
            cfg.training.learning_rate == 0.003
            and cfg.training.batch_size == TRAIN_PAIRS
            and cfg.training.epochs == 4_000
            and cfg.training.loss_weights == [1.0, 1.0, 0.0]
            and cfg.training.gradient_clip_norm is None
            and cfg.training.scheduler_factor == 0.5
            and cfg.training.lr_patience == 100
            and cfg.training.scheduler_threshold == 1e-4
            and cfg.training.scheduler_min_lr == 1e-6
        ),
        "data": (
            cfg.data.scaling == "minmax"
            and cfg.data.n_samples_train == 1_000
            and cfg.data.n_samples_val == 200
            and cfg.data.n_iterations == 30
            and cfg.data.skip == 0
        ),
        "cmgdb": (
            (
                cfg.cmgdb.subdiv_init,
                cfg.cmgdb.subdiv_min,
                cfg.cmgdb.subdiv_max,
                cfg.cmgdb.subdiv_limit,
            )
            == (25, 28, 29, 10_000)
            and cfg.cmgdb.padding is True
            and cfg.cmgdb.bounds_epsilon_frac == 0.01
            and cfg.cmgdb.lower_bounds is None
            and cfg.cmgdb.upper_bounds is None
            and cfg.cmgdb.box_map_backend == "adaptive_precomputed"
            and cfg.cmgdb.compute_roa is False
        ),
        "model_seeds": tuple(cfg.seeds) == MODEL_SEEDS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "packaged Leslie3D Marcio 5x3 protocol drifted at: "
            + ", ".join(failed)
        )


def _base_config() -> ExperimentConfig:
    cfg = load_config(CONFIG_STEM)
    _validate_protocol(cfg)
    return cfg


def _dataset_config(
    dataset: int,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    model_seeds: Sequence[int] = MODEL_SEEDS,
) -> ExperimentConfig:
    """Resolve one of the five frozen dataset configurations."""

    if dataset not in DATASET_INITIAL_CONDITION_SEEDS:
        raise ValueError(
            f"dataset must be one of {tuple(DATASET_INITIAL_CONDITION_SEEDS)}; "
            f"got {dataset}"
        )
    invalid_model_seeds = set(model_seeds) - set(MODEL_SEEDS)
    if invalid_model_seeds:
        raise ValueError(
            f"model seeds are frozen to {MODEL_SEEDS}; got invalid "
            f"{sorted(invalid_model_seeds)}"
        )

    cfg = _base_config().model_copy(deep=True)
    dataset_name = f"dataset_{dataset:02d}"
    dataset_output = Path(output_root).resolve() / dataset_name
    cfg.data.train_seed = DATASET_INITIAL_CONDITION_SEEDS[dataset]
    cfg.paths.data_dir = Path(data_root).resolve() / dataset_name
    cfg.paths.output_dir = dataset_output
    cfg.paths.scaler_dir_override = dataset_output / "scalers"
    cfg.seeds = list(model_seeds)
    cfg.experiment_name = f"{CONFIG_STEM}_{dataset_name}"
    return cfg


def _seed_config(cfg: ExperimentConfig, model_seed: int) -> ExperimentConfig:
    if model_seed not in MODEL_SEEDS:
        raise ValueError(f"model seed must be one of {MODEL_SEEDS}; got {model_seed}")
    seed_cfg = cfg.model_copy(deep=True)
    seed_cfg.paths.output_dir = cfg.paths.output_dir / f"seed_{model_seed}"
    seed_cfg.paths.scaler_dir_override = cfg.paths.scaler_dir
    seed_cfg.seeds = [model_seed]
    return seed_cfg


def plan_cells(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[Cell]:
    cells: list[Cell] = []
    for dataset, initial_condition_seed in DATASET_INITIAL_CONDITION_SEEDS.items():
        cfg = _dataset_config(
            dataset,
            data_root=data_root,
            output_root=output_root,
        )
        for model_seed in MODEL_SEEDS:
            seed_cfg = _seed_config(cfg, model_seed)
            cells.append(
                Cell(
                    index=len(cells),
                    dataset=dataset,
                    initial_condition_seed=initial_condition_seed,
                    model_seed=model_seed,
                    data_dir=str(cfg.paths.data_dir),
                    output_dir=str(seed_cfg.paths.output_dir),
                )
            )
    return cells


def _build_plan(*, data_root: Path, output_root: Path) -> dict[str, Any]:
    cfg = _base_config()
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "experiment": CONFIG_STEM,
        "layout_version": EXPERIMENT_LABEL,
        "data_root": str(data_root.resolve()),
        "output_root": str(output_root.resolve()),
        "design": {
            "dataset_initial_condition_seeds": {
                f"dataset_{dataset:02d}": seed
                for dataset, seed in DATASET_INITIAL_CONDITION_SEEDS.items()
            },
            "model_seeds": list(MODEL_SEEDS),
            "n_datasets": len(DATASET_INITIAL_CONDITION_SEEDS),
            "runs_per_dataset": len(MODEL_SEEDS),
            "total_runs": len(DATASET_INITIAL_CONDITION_SEEDS) * len(MODEL_SEEDS),
            "shared_validation_seed": cfg.data.val_seed,
        },
        "training": {
            "entrypoint": "latentdynamics.training.train_marcio_full_batch",
            "objective": OBJECTIVE,
            "full_batch_pairs": TRAIN_PAIRS,
            "epochs": cfg.training.epochs,
            "learning_rate": cfg.training.learning_rate,
            "loss_weights_metadata": cfg.training.loss_weights,
            "gradient_clipping": None,
            "validation_used_for_optimization": False,
            "holdout_pairs_evaluated_after_training": HOLDOUT_PAIRS,
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "factor": cfg.training.scheduler_factor,
                "patience": cfg.training.lr_patience,
                "threshold": cfg.training.scheduler_threshold,
                "min_lr": cfg.training.scheduler_min_lr,
            },
        },
        "cmgdb": cfg.cmgdb.model_dump(mode="json"),
        "cmgdb_bounds_inference": {
            "data_role": CMGDB_BOUNDS_DATA_ROLE,
            "source": CMGDB_BOUNDS_SOURCE,
            "included_arrays": ["train.x", "train.y"],
            "validation_pairs_included": False,
            "epsilon_frac": cfg.cmgdb.bounds_epsilon_frac,
        },
        "architecture": cfg.arch.model_dump(mode="json"),
        "data": cfg.data.model_dump(mode="json"),
        "stages": list(ALL_STAGES),
        "cells": [asdict(cell) for cell in plan_cells(data_root=data_root, output_root=output_root)],
    }


def _ensure_plan(*, data_root: Path, output_root: Path) -> tuple[dict[str, Any], str]:
    output_root = output_root.resolve()
    plan = _build_plan(data_root=data_root, output_root=output_root)
    plan_hash = _canonical_hash(plan)
    envelope = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_sha256": plan_hash,
        "plan": plan,
    }
    plan_path = output_root / "experiment_plan.json"
    if plan_path.exists():
        observed = _read_json(plan_path)
        if observed != envelope:
            raise ValueError(
                f"existing experiment plan differs from the requested frozen plan: {plan_path}"
            )
        return plan, plan_hash
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"{output_root} is non-empty but has no experiment_plan.json; refusing reuse"
        )
    _write_json_atomic(plan_path, envelope)
    return plan, plan_hash


def _resolve_device(name: str | torch.device | None) -> torch.device:
    if isinstance(name, torch.device):
        return name
    if name is not None and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_pairs(path: Path, *, high_dims: int, expected_rows: int) -> tuple[NDArray, NDArray]:
    if not _nonempty(path):
        raise FileNotFoundError(path)
    pairs = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    expected_shape = (expected_rows, 2 * high_dims)
    if pairs.shape != expected_shape:
        raise ValueError(f"{path} has shape {pairs.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(pairs)):
        raise ValueError(f"{path} contains non-finite values")
    return (
        np.ascontiguousarray(pairs[:, :high_dims]),
        np.ascontiguousarray(pairs[:, high_dims:]),
    )


def _load_scaled_pairs(
    cfg: ExperimentConfig,
    *,
    role: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], Path]:
    if role == "train":
        csv_path = cfg.paths.data_dir / f"{TRAIN_FILE}.csv"
        expected_rows = TRAIN_PAIRS
    elif role == "holdout":
        csv_path = cfg.paths.val_csv()
        expected_rows = HOLDOUT_PAIRS
    else:
        raise ValueError(f"unknown pair role {role!r}")
    x, y = _load_pairs(
        csv_path,
        high_dims=cfg.arch.high_dims,
        expected_rows=expected_rows,
    )
    scaler = load_scaler(cfg.paths.scaler_path(TRAIN_FILE))
    return (
        np.ascontiguousarray(scaler.transform(x), dtype=np.float64),
        np.ascontiguousarray(scaler.transform(y), dtype=np.float64),
        csv_path,
    )


@torch.no_grad()
def _evaluate_two_term_objective(
    model: torch.nn.Module,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    device: torch.device,
    batch_size: int = 8_192,
) -> dict[str, float]:
    """Sample-weighted post-update evaluation of Marcio's decoded objective."""

    if x.shape != y.shape or x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"invalid evaluation pair shapes: x={x.shape}, y={y.shape}")
    model = model.to(device)
    model.eval()
    reconstruction_sum = 0.0
    prediction_sum = 0.0
    n_elements = 0
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        x_batch = torch.as_tensor(x[start:stop], dtype=torch.float32, device=device)
        y_batch = torch.as_tensor(y[start:stop], dtype=torch.float32, device=device)
        encoded = model.encoder(x_batch)
        reconstructed = model.decoder(encoded)
        predicted = model.decoder(model.latent_map(encoded))
        reconstruction_sum += float((reconstructed - x_batch).square().sum().cpu())
        prediction_sum += float((predicted - y_batch).square().sum().cpu())
        n_elements += int(x_batch.numel())
    reconstruction = reconstruction_sum / n_elements
    prediction = prediction_sum / n_elements
    return {
        "loss_reconstruction": reconstruction,
        "loss_prediction": prediction,
        "loss_total": reconstruction + prediction,
    }


def _training_contract(
    dataset_cfg: ExperimentConfig,
    *,
    dataset: int,
    model_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": CONFIG_STEM,
        "dataset": dataset,
        "dataset_initial_condition_seed": DATASET_INITIAL_CONDITION_SEEDS[dataset],
        "model_seed": model_seed,
        "device": str(device),
        "training_entrypoint": "latentdynamics.training.train_marcio_full_batch",
        "objective": OBJECTIVE,
        "training": {
            "epochs": dataset_cfg.training.epochs,
            "learning_rate": dataset_cfg.training.learning_rate,
            "batch_size": dataset_cfg.training.batch_size,
            "loss_weights_metadata": dataset_cfg.training.loss_weights,
            "gradient_clip_norm": dataset_cfg.training.gradient_clip_norm,
            "early_stopping": False,
            "best_weight_restoration": False,
            "scheduler_factor": dataset_cfg.training.scheduler_factor,
            "scheduler_patience": dataset_cfg.training.lr_patience,
            "scheduler_threshold": dataset_cfg.training.scheduler_threshold,
            "scheduler_min_lr": dataset_cfg.training.scheduler_min_lr,
        },
        "architecture": dataset_cfg.arch.model_dump(mode="json"),
        "data": {
            "scaling": dataset_cfg.data.scaling,
            "train_pairs": TRAIN_PAIRS,
            "holdout_pairs": HOLDOUT_PAIRS,
            "holdout_used_for_training_or_selection": False,
        },
    }


def _training_core_paths(seed_cfg: ExperimentConfig) -> tuple[Path, ...]:
    root = seed_cfg.paths.output_dir
    return (
        root / "models" / "autoencoder.pt",
        root / "models" / "autoencoder.json",
        root / "logs" / "history.json",
        root / "training_summary.json",
    )


def _training_core_complete(seed_cfg: ExperimentConfig) -> bool:
    return has_new_checkpoint(seed_cfg.paths.model_dir) and all(
        _nonempty(path) for path in _training_core_paths(seed_cfg)
    )


def _training_core_partially_present(seed_cfg: ExperimentConfig) -> bool:
    return any(path.exists() for path in _training_core_paths(seed_cfg))


def _training_complete(seed_cfg: ExperimentConfig) -> bool:
    root = seed_cfg.paths.output_dir
    marker = root / "stage_train_complete.json"
    if not _nonempty(marker):
        return False
    try:
        payload = _read_json(marker)
        if payload.get("status") != "complete":
            return False
        for record in payload["artifacts"].values():
            path = root / record["path"]
            if (
                not _nonempty(path)
                or int(path.stat().st_size) != int(record["size_bytes"])
                or _sha256(path) != record["sha256"]
            ):
                return False
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return True


def _write_final_evaluation(
    *,
    dataset_cfg: ExperimentConfig,
    seed_cfg: ExperimentConfig,
    dataset: int,
    model_seed: int,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[Path, Path]:
    x_train, y_train, train_csv = _load_scaled_pairs(dataset_cfg, role="train")
    x_holdout, y_holdout, holdout_csv = _load_scaled_pairs(dataset_cfg, role="holdout")
    train_metrics = _evaluate_two_term_objective(
        model,
        x_train,
        y_train,
        device=device,
    )
    holdout_metrics = _evaluate_two_term_objective(
        model,
        x_holdout,
        y_holdout,
        device=device,
    )
    payload = {
        "schema_version": 1,
        "evaluated_at_utc": _utc_now(),
        "objective": OBJECTIVE,
        "checkpoint_selection": "fixed final epoch",
        "dataset": dataset,
        "dataset_initial_condition_seed": DATASET_INITIAL_CONDITION_SEEDS[dataset],
        "model_seed": model_seed,
        "device": str(device),
        "scaler": {
            "path": str(dataset_cfg.paths.scaler_path(TRAIN_FILE).resolve()),
            "sha256": _sha256(dataset_cfg.paths.scaler_path(TRAIN_FILE)),
        },
        "train": {
            "n_pairs": int(x_train.shape[0]),
            "csv": str(train_csv.resolve()),
            "csv_sha256": _sha256(train_csv),
            **train_metrics,
        },
        "holdout": {
            "n_pairs": int(x_holdout.shape[0]),
            "csv": str(holdout_csv.resolve()),
            "csv_sha256": _sha256(holdout_csv),
            "sampling_seed": dataset_cfg.data.val_seed,
            "used_for_optimization": False,
            "used_for_checkpoint_selection": False,
            **holdout_metrics,
        },
    }
    root = seed_cfg.paths.output_dir
    evaluation_path = _write_json_atomic(root / "holdout_evaluation.json", payload)
    final_losses_path = root / "final_losses.txt"
    final_losses_path.write_text(
        "\n".join(
            [
                "training_method: marcio_full_batch",
                f"checkpoint_epoch: {dataset_cfg.training.epochs}",
                "checkpoint_selection: fixed_final_epoch",
                f"train_loss_reconstruction: {train_metrics['loss_reconstruction']:.9e}",
                f"train_loss_prediction: {train_metrics['loss_prediction']:.9e}",
                f"train_loss_total: {train_metrics['loss_total']:.9e}",
                f"val_loss_reconstruction: {holdout_metrics['loss_reconstruction']:.9e}",
                f"val_loss_prediction: {holdout_metrics['loss_prediction']:.9e}",
                f"val_loss_total: {holdout_metrics['loss_total']:.9e}",
                "validation_used_for_optimization: false",
                "validation_used_for_checkpoint_selection: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return evaluation_path, final_losses_path


def _run_train(
    *,
    dataset_cfg: ExperimentConfig,
    seed_cfg: ExperimentConfig,
    dataset: int,
    model_seed: int,
    device: torch.device,
    verbose: bool,
    force_overwrite: bool,
) -> None:
    """Train one cell or finish post-training evaluation after interruption."""

    output_root = seed_cfg.paths.output_dir
    if _training_complete(seed_cfg) and not force_overwrite:
        raise RuntimeError(
            f"verified completed training already exists at {output_root}; "
            "use --skip-completed or --force-overwrite"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    contract = _training_contract(
        dataset_cfg,
        dataset=dataset,
        model_seed=model_seed,
        device=device,
    )
    contract_path = output_root / "training_contract.json"
    if contract_path.exists() and _read_json(contract_path) != contract and not force_overwrite:
        raise ValueError(f"existing training contract differs at {contract_path}")
    _write_json_atomic(contract_path, contract)

    if not force_overwrite and _training_core_complete(seed_cfg):
        if verbose:
            print(f"train: checkpoint complete; resuming holdout evaluation for {output_root}")
        model, _arch = load_any_checkpoint(seed_cfg.paths.model_dir, arch=seed_cfg.arch)
    else:
        if (
            not force_overwrite
            and _training_core_partially_present(seed_cfg)
            and not _training_core_complete(seed_cfg)
        ):
            raise RuntimeError(
                f"partial training core at {output_root}; refusing ambiguous overwrite. "
                "Inspect it or pass --force-overwrite."
            )
        x_train, y_train, _train_csv = _load_scaled_pairs(dataset_cfg, role="train")
        if x_train.shape[0] != dataset_cfg.training.batch_size:
            raise ValueError(
                f"configured full batch is {dataset_cfg.training.batch_size}, "
                f"but loaded {x_train.shape[0]} pairs"
            )
        result = train_marcio_full_batch(
            arch=dataset_cfg.arch,
            x=x_train,
            y=y_train,
            epochs=dataset_cfg.training.epochs,
            learning_rate=dataset_cfg.training.learning_rate,
            seed=model_seed,
            device=device,
            output_dir=output_root,
            scheduler_factor=dataset_cfg.training.scheduler_factor,
            scheduler_patience=dataset_cfg.training.lr_patience,
            scheduler_threshold=dataset_cfg.training.scheduler_threshold,
            scheduler_min_lr=dataset_cfg.training.scheduler_min_lr,
            verbose=verbose,
        )
        model = result.model
        if not _training_core_complete(seed_cfg):
            raise RuntimeError(
                "train_marcio_full_batch returned without its checkpoint, history, "
                "or training summary"
            )

    evaluation_path, final_losses_path = _write_final_evaluation(
        dataset_cfg=dataset_cfg,
        seed_cfg=seed_cfg,
        dataset=dataset,
        model_seed=model_seed,
        model=model,
        device=device,
    )
    artifact_paths = {
        "checkpoint": output_root / "models" / "autoencoder.pt",
        "checkpoint_metadata": output_root / "models" / "autoencoder.json",
        "history": output_root / "logs" / "history.json",
        "training_summary": output_root / "training_summary.json",
        "training_contract": contract_path,
        "holdout_evaluation": evaluation_path,
        "final_losses": final_losses_path,
    }
    marker = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "dataset": dataset,
        "model_seed": model_seed,
        "artifacts": {
            name: _artifact_record(path, relative_to=output_root)
            for name, path in artifact_paths.items()
        },
    }
    _write_json_atomic(output_root / "stage_train_complete.json", marker)


def _data_complete(cfg: ExperimentConfig) -> bool:
    paths = {
        "train": (
            cfg.paths.data_dir / "train.csv",
            cfg.paths.data_dir / "train_metadata.json",
            int(cfg.data.n_samples_train),
            int(cfg.data.train_seed),
        ),
        "val": (
            cfg.paths.data_dir / "val.csv",
            cfg.paths.data_dir / "val_metadata.json",
            int(cfg.data.n_samples_val),
            int(cfg.data.val_seed),
        ),
    }
    for role, (csv_path, metadata_path, n_samples, sampling_seed) in paths.items():
        if not _nonempty(csv_path) or not _nonempty(metadata_path):
            return False
        try:
            metadata = _read_json(metadata_path)
        except ValueError:
            return False
        expected = {
            "dataset_name": role,
            "role": role,
            "dimension": cfg.arch.high_dims,
            "n_samples": n_samples,
            "n_iterations": cfg.data.n_iterations,
            "skip_initial_steps": cfg.data.skip,
            "sampling_method": cfg.data.sampling_method,
            "sampling_seed": sampling_seed,
            "model_params": cfg.system.params,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
    return True


def _morse_complete(cfg: ExperimentConfig) -> bool:
    root = cfg.paths.output_dir
    log_path = root / "mg_params_log.txt"
    if not all(
        _nonempty(path)
        for path in (root / "MG" / "morse_graph", root / "MG" / "morse_sets", log_path)
    ):
        return False
    text = log_path.read_text(encoding="utf-8")
    required = (
        "subdiv_init: 25",
        "subdiv_min: 28",
        "subdiv_max: 29",
        "subdiv_limit: 10000",
        "bounds_epsilon_frac: 0.01",
        "padding: True",
        "box_map_backend: adaptive_precomputed",
        "compute_roa: False",
        f"bounds_source: {CMGDB_BOUNDS_SOURCE}",
    )
    return all(item in text for item in required)


def _render_complete(cfg: ExperimentConfig) -> bool:
    return all(
        _nonempty(cfg.paths.morse_dir / name)
        for name in (
            "morse_graph.pdf",
            "morse_graph.png",
            "morse_sets.pdf",
            "morse_sets.png",
        )
    )


def _stage_complete(
    stage: str,
    *,
    dataset_cfg: ExperimentConfig,
    seed_cfg: ExperimentConfig | None,
) -> bool:
    if stage == "data":
        return _data_complete(dataset_cfg)
    if stage == "scale":
        return scale_stage.scaler_is_current(dataset_cfg, TRAIN_FILE)
    if seed_cfg is None:
        raise ValueError(f"stage {stage!r} requires a model cell")
    if stage == "train":
        return _training_complete(seed_cfg)
    if stage == "diagnose":
        return _nonempty(seed_cfg.paths.output_dir / "diagnose.json")
    if stage == "morse":
        return _morse_complete(seed_cfg)
    if stage == "render":
        return _render_complete(seed_cfg)
    if stage == "metrics":
        return _nonempty(seed_cfg.paths.output_dir / "metrics.json")
    raise ValueError(f"unknown stage {stage!r}")


def _execute_stage(
    stage: str,
    *,
    dataset: int,
    model_seed: int | None,
    dataset_cfg: ExperimentConfig,
    seed_cfg: ExperimentConfig | None,
    device: torch.device,
    verbose: bool,
    force_overwrite: bool,
) -> None:
    if stage == "data":
        data_stage.run(dataset_cfg, verbose=verbose)
        return
    if stage == "scale":
        scale_stage.run(dataset_cfg, TRAIN_FILE, verbose=verbose)
        return
    if model_seed is None or seed_cfg is None:
        raise ValueError(f"stage {stage!r} requires a model seed")
    if stage == "train":
        _run_train(
            dataset_cfg=dataset_cfg,
            seed_cfg=seed_cfg,
            dataset=dataset,
            model_seed=model_seed,
            device=device,
            verbose=verbose,
            force_overwrite=force_overwrite,
        )
    elif stage == "diagnose":
        diagnose_stage.run(
            seed_cfg,
            train_file=TRAIN_FILE,
            device=device,
            verbose=verbose,
        )
    elif stage == "morse":
        morse_stage.run(
            seed_cfg,
            train_file=TRAIN_FILE,
            bounds_data_role=CMGDB_BOUNDS_DATA_ROLE,
            device=device,
            verbose=verbose,
            force_overwrite=force_overwrite,
        )
    elif stage == "render":
        # The requested products are the Morse graph and Morse sets.  Omit the
        # optional approximate RoA walk; exact RoA is disabled in the config.
        render_stage_module.render_stage(
            seed_cfg,
            train_file=TRAIN_FILE,
            device=device,
            verbose=verbose,
            figures={"morse", "overlay", "extras"},
        )
    elif stage == "metrics":
        metrics_stage_module.metrics_stage(
            seed_cfg,
            dataset_cfg,
            train_file=TRAIN_FILE,
            verbose=verbose,
        )
    else:
        raise ValueError(f"unknown stage {stage!r}")


def _morse_counts(dot_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_morse_nodes": None,
        "n_morse_edges": None,
        "n_sink_nodes": None,
    }
    if not _nonempty(dot_path):
        return result
    text = dot_path.read_text(encoding="utf-8")
    nodes = {
        int(match.group(1))
        for match in re.finditer(r"^\s*(\d+)\s+\[label", text, re.MULTILINE)
    }
    edges = [
        (int(source), int(target))
        for source, target in re.findall(
            r"^\s*(\d+)\s*->\s*(\d+)\s*;", text, re.MULTILINE
        )
    ]
    sources = {source for source, _target in edges}
    return {
        "n_morse_nodes": len(nodes),
        "n_morse_edges": len(edges),
        "n_sink_nodes": len(nodes - sources),
    }


def _cell_summary(
    *,
    dataset: int,
    model_seed: int,
    dataset_cfg: ExperimentConfig,
    seed_cfg: ExperimentConfig,
) -> dict[str, Any]:
    root = seed_cfg.paths.output_dir
    evaluation: dict[str, Any] | None = None
    evaluation_path = root / "holdout_evaluation.json"
    if _nonempty(evaluation_path):
        try:
            evaluation = _read_json(evaluation_path)
        except ValueError:
            evaluation = None
    metrics: dict[str, Any] | None = None
    metrics_path = root / "metrics.json"
    if _nonempty(metrics_path):
        try:
            metrics = _read_json(metrics_path)
        except ValueError:
            metrics = None
    return {
        "dataset": dataset,
        "dataset_name": f"dataset_{dataset:02d}",
        "dataset_initial_condition_seed": DATASET_INITIAL_CONDITION_SEEDS[dataset],
        "validation_seed": dataset_cfg.data.val_seed,
        "model_seed": model_seed,
        "data_dir": str(dataset_cfg.paths.data_dir),
        "output_dir": str(root),
        "stage_status": {
            stage: (
                "complete"
                if _stage_complete(
                    stage,
                    dataset_cfg=dataset_cfg,
                    seed_cfg=seed_cfg,
                )
                else "pending"
            )
            for stage in ALL_STAGES
        },
        "final_train": evaluation.get("train") if evaluation is not None else None,
        "fixed_holdout": evaluation.get("holdout") if evaluation is not None else None,
        "morse": _morse_counts(root / "MG" / "morse_graph"),
        "metrics": metrics,
    }


def _write_sweep_summary(*, data_root: Path, output_root: Path, plan_sha256: str) -> dict:
    cells: list[dict[str, Any]] = []
    dataset_configs = {
        dataset: _dataset_config(
            dataset,
            data_root=data_root,
            output_root=output_root,
        )
        for dataset in DATASET_INITIAL_CONDITION_SEEDS
    }
    for dataset, dataset_cfg in dataset_configs.items():
        for model_seed in MODEL_SEEDS:
            cells.append(
                _cell_summary(
                    dataset=dataset,
                    model_seed=model_seed,
                    dataset_cfg=dataset_cfg,
                    seed_cfg=_seed_config(dataset_cfg, model_seed),
                )
            )
    counts: dict[str, dict[str, int]] = {}
    for stage in ALL_STAGES:
        if stage in DATASET_STAGES:
            values = [
                _stage_complete(
                    stage,
                    dataset_cfg=cfg,
                    seed_cfg=None,
                )
                for cfg in dataset_configs.values()
            ]
        else:
            values = [cell["stage_status"][stage] == "complete" for cell in cells]
        counts[stage] = {
            "expected": len(values),
            "complete": sum(values),
            "pending": len(values) - sum(values),
        }
    payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "experiment": CONFIG_STEM,
        "experiment_plan_sha256": plan_sha256,
        "data_root": str(data_root.resolve()),
        "output_root": str(output_root.resolve()),
        "design": {
            "dataset_initial_condition_seeds": DATASET_INITIAL_CONDITION_SEEDS,
            "model_seeds": list(MODEL_SEEDS),
            "total_cells": len(cells),
        },
        "counts": counts,
        "cells": cells,
    }
    _write_json_atomic(output_root / "sweep_summary.json", payload)

    csv_path = output_root / "sweep_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    fields = [
        "dataset",
        "dataset_initial_condition_seed",
        "model_seed",
        *[f"stage_{stage}" for stage in ALL_STAGES],
        "train_loss_total",
        "holdout_loss_total",
        "n_morse_nodes",
        "n_morse_edges",
        "n_sink_nodes",
        "output_dir",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            train = cell["final_train"] or {}
            holdout = cell["fixed_holdout"] or {}
            writer.writerow(
                {
                    "dataset": cell["dataset"],
                    "dataset_initial_condition_seed": cell[
                        "dataset_initial_condition_seed"
                    ],
                    "model_seed": cell["model_seed"],
                    **{
                        f"stage_{stage}": cell["stage_status"][stage]
                        for stage in ALL_STAGES
                    },
                    "train_loss_total": train.get("loss_total"),
                    "holdout_loss_total": holdout.get("loss_total"),
                    **cell["morse"],
                    "output_dir": cell["output_dir"],
                }
            )
    temporary.replace(csv_path)
    return payload


def _write_standard_manifests(
    *,
    dataset: int,
    model_seed: int | None,
    dataset_cfg: ExperimentConfig,
    selected_model_seeds: Sequence[int],
    device: torch.device,
    requested_stages: Sequence[str],
    skipped_stage: str | None,
) -> list[Path]:
    """Refresh the standard pipeline provenance manifest for affected cells."""

    seeds = selected_model_seeds if model_seed is None else (model_seed,)
    written: list[Path] = []
    for seed in seeds:
        seed_cfg = _seed_config(dataset_cfg, seed)
        cell_index = (dataset - 1) * len(MODEL_SEEDS) + MODEL_SEEDS.index(seed)
        written.append(
            provenance_stage.write_run_manifest(
                seed_cfg,
                dataset_cfg,
                cell_summary={
                    "cell_index": cell_index,
                    "seed": seed,
                    "device": str(device),
                    "skipped_stages": [skipped_stage] if skipped_stage else [],
                },
                stages=list(requested_stages),
                train_file=TRAIN_FILE,
            )
        )
    return written


def _normalize_stages(stages: str | Iterable[str]) -> list[str]:
    if isinstance(stages, str):
        if stages.strip().lower() == "all":
            return list(ALL_STAGES)
        raw = stages.split(",")
    else:
        raw = list(stages)
    requested = {item.strip().lower() for item in raw if item.strip()}
    unknown = requested - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {ALL_STAGES}")
    return [stage for stage in ALL_STAGES if stage in requested]


def _parse_frozen_subset(raw: str, *, allowed: Sequence[int], label: str) -> list[int]:
    if raw.strip().lower() == "all":
        return list(allowed)
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{label} must be a comma-separated integer list or 'all'") from exc
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicates")
    invalid = set(values) - set(allowed)
    if invalid:
        raise ValueError(f"{label} contains values outside {tuple(allowed)}: {sorted(invalid)}")
    return [value for value in allowed if value in values]


def run_experiment(
    *,
    data_root: Path,
    output_root: Path,
    stages: Sequence[str],
    datasets: Sequence[int],
    model_seeds: Sequence[int],
    device: str | torch.device | None,
    skip_completed: bool,
    force_overwrite: bool,
    fail_fast: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Execute selected stages in global stage order and refresh summaries."""

    if skip_completed and force_overwrite:
        raise ValueError("--skip-completed and --force-overwrite are mutually exclusive")
    normalized = _normalize_stages(stages)
    invalid_datasets = set(datasets) - set(DATASET_INITIAL_CONDITION_SEEDS)
    invalid_seeds = set(model_seeds) - set(MODEL_SEEDS)
    if invalid_datasets or invalid_seeds:
        raise ValueError(
            f"selection is outside frozen design: datasets={sorted(invalid_datasets)}, "
            f"model_seeds={sorted(invalid_seeds)}"
        )
    plan, plan_sha256 = _ensure_plan(data_root=data_root, output_root=output_root)
    resolved_device = _resolve_device(device)
    dataset_configs = {
        dataset: _dataset_config(
            dataset,
            data_root=data_root,
            output_root=output_root,
            model_seeds=MODEL_SEEDS,
        )
        for dataset in datasets
    }
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for stage in normalized:
        if verbose:
            print(f"\n=== stage: {stage} ===", flush=True)
        if stage in DATASET_STAGES:
            targets = [(dataset, None) for dataset in datasets]
        else:
            targets = [
                (dataset, model_seed)
                for dataset in datasets
                for model_seed in model_seeds
            ]
        for dataset, model_seed in targets:
            dataset_cfg = dataset_configs[dataset]
            seed_cfg = (
                _seed_config(dataset_cfg, model_seed)
                if model_seed is not None
                else None
            )
            target = (
                f"dataset_{dataset:02d}"
                if model_seed is None
                else f"dataset_{dataset:02d}/seed_{model_seed}"
            )
            record: dict[str, Any] = {
                "stage": stage,
                "dataset": dataset,
                "dataset_initial_condition_seed": DATASET_INITIAL_CONDITION_SEEDS[dataset],
                "model_seed": model_seed,
                "target": target,
            }
            if skip_completed and _stage_complete(
                stage,
                dataset_cfg=dataset_cfg,
                seed_cfg=seed_cfg,
            ):
                record["status"] = "skipped_complete"
                records.append(record)
                if verbose:
                    print(f"{target}: skipped verified {stage}", flush=True)
                _write_standard_manifests(
                    dataset=dataset,
                    model_seed=model_seed,
                    dataset_cfg=dataset_cfg,
                    selected_model_seeds=model_seeds,
                    device=resolved_device,
                    requested_stages=normalized,
                    skipped_stage=stage,
                )
                continue
            if verbose:
                print(f"{target}: running {stage}", flush=True)
            try:
                _execute_stage(
                    stage,
                    dataset=dataset,
                    model_seed=model_seed,
                    dataset_cfg=dataset_cfg,
                    seed_cfg=seed_cfg,
                    device=resolved_device,
                    verbose=verbose,
                    force_overwrite=force_overwrite,
                )
                if not _stage_complete(
                    stage,
                    dataset_cfg=dataset_cfg,
                    seed_cfg=seed_cfg,
                ):
                    raise RuntimeError(
                        f"{stage} returned without its complete artifact contract for {target}"
                    )
                record["status"] = "completed"
            except Exception as exc:
                record.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                failures.append(record)
                if verbose:
                    print(f"{target}: {stage} failed: {exc}", flush=True)
            records.append(record)
            _write_standard_manifests(
                dataset=dataset,
                model_seed=model_seed,
                dataset_cfg=dataset_cfg,
                selected_model_seeds=model_seeds,
                device=resolved_device,
                requested_stages=normalized,
                skipped_stage=None,
            )
            if failures and fail_fast:
                _write_sweep_summary(
                    data_root=data_root,
                    output_root=output_root,
                    plan_sha256=plan_sha256,
                )
                raise RuntimeError(
                    f"fail-fast: {target} stage {stage} failed: "
                    f"{failures[-1]['error_message']}"
                )
        _write_sweep_summary(
            data_root=data_root,
            output_root=output_root,
            plan_sha256=plan_sha256,
        )

    sweep_summary = _write_sweep_summary(
        data_root=data_root,
        output_root=output_root,
        plan_sha256=plan_sha256,
    )
    invocation = {
        "schema_version": 1,
        "completed_at_utc": _utc_now(),
        "experiment_plan_sha256": plan_sha256,
        "device": str(resolved_device),
        "requested_stages": normalized,
        "selected_datasets": list(datasets),
        "selected_model_seeds": list(model_seeds),
        "skip_completed": skip_completed,
        "force_overwrite": force_overwrite,
        "status": "complete" if not failures else "complete_with_failures",
        "records": records,
        "failure_count": len(failures),
        "sweep_summary": "sweep_summary.json",
    }
    _write_json_atomic(output_root / "latest_invocation.json", invocation)
    return {
        "plan": plan,
        "plan_sha256": plan_sha256,
        "invocation": invocation,
        "sweep_summary": sweep_summary,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stages",
        default="all",
        help=f"comma-separated subset of {ALL_STAGES}, or 'all'",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="dataset indices from 1..5 as a comma-separated list, or 'all'",
    )
    parser.add_argument(
        "--model-seeds",
        default="all",
        help="model seeds from 0,1,2 as a comma-separated list, or 'all'",
    )
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="training/evaluation device: auto, cpu, mps, cuda, ...",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="verify and skip completed stage/cell artifacts",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="explicitly permit replacement of existing train/Morse artifacts",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        stages = _normalize_stages(args.stages)
        datasets = _parse_frozen_subset(
            args.datasets,
            allowed=tuple(DATASET_INITIAL_CONDITION_SEEDS),
            label="datasets",
        )
        model_seeds = _parse_frozen_subset(
            args.model_seeds,
            allowed=MODEL_SEEDS,
            label="model-seeds",
        )
        if args.max_datasets is not None:
            if args.max_datasets < 0:
                raise ValueError("--max-datasets must be non-negative")
            datasets = datasets[: args.max_datasets]
        if args.max_seeds is not None:
            if args.max_seeds < 0:
                raise ValueError("--max-seeds must be non-negative")
            model_seeds = model_seeds[: args.max_seeds]
        if args.skip_completed and args.force_overwrite:
            raise ValueError("--skip-completed and --force-overwrite are mutually exclusive")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        plan = _build_plan(data_root=args.data_root, output_root=args.output_root)
        selected = [
            cell
            for cell in plan["cells"]
            if cell["dataset"] in datasets and cell["model_seed"] in model_seeds
        ]
        print(
            json.dumps(
                {
                    "experiment": CONFIG_STEM,
                    "stages": stages,
                    "datasets": datasets,
                    "dataset_initial_condition_seeds": {
                        f"dataset_{dataset:02d}": DATASET_INITIAL_CONDITION_SEEDS[dataset]
                        for dataset in datasets
                    },
                    "model_seeds": model_seeds,
                    "n_cells": len(selected),
                    "device": args.device,
                    "data_root": str(args.data_root.resolve()),
                    "output_root": str(args.output_root.resolve()),
                    "cells": selected,
                    "frozen_plan_sha256": _canonical_hash(plan),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    result = run_experiment(
        data_root=args.data_root,
        output_root=args.output_root,
        stages=stages,
        datasets=datasets,
        model_seeds=model_seeds,
        device=args.device,
        skip_completed=args.skip_completed,
        force_overwrite=args.force_overwrite,
        fail_fast=args.fail_fast,
        verbose=not args.quiet,
    )
    invocation = result["invocation"]
    print(
        json.dumps(
            {
                "status": invocation["status"],
                "failure_count": invocation["failure_count"],
                "experiment_plan_sha256": result["plan_sha256"],
                "output_root": str(args.output_root.resolve()),
                "sweep_summary": str(
                    args.output_root.resolve() / "sweep_summary.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if invocation["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
