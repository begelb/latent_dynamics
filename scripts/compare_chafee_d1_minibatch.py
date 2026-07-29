"""Train one controlled mini-batch Chafee--Infante d=1 comparison run.

The canonical 4,000- and 10,000-update full-batch artifacts are read-only.
This driver keeps their exact unscaled 30,000 training pairs, seed-0
initialization, architecture, and two-term decoded objective, while changing
only the optimization regime to shuffled mini-batch Adam.

Checkpoint selection does not inspect the archived basin trajectories.  The
primary checkpoint is the epoch with the smallest normalized held-out latent
residual among epochs whose held-out decoded objective is within five percent
of its minimum.  The equal-update epoch, best decoded-validation epoch, and
final epoch are also retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from latentdynamics.config import ArchConfig
from latentdynamics.models import LatentDynamicsAutoencoder, build_autoencoder
from latentdynamics.training import load_checkpoint, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA = PROJECT_ROOT / "archive" / "marcio" / "scripts" / "train_data.csv"
VALIDATION_DATA = CODE_ROOT / "data" / "chafee_infante" / "val.csv"
CANONICAL_RUN = (
    CODE_ROOT
    / "output"
    / "chafee_latent_dimension_study"
    / "latent_1d"
    / "seed_0"
)
DEFAULT_OUTPUT = CANONICAL_RUN.parent / "seed_0_minibatch_b1024_lr1e3"

TRAIN_DATA_SHA256 = "890fea29a2d26b31b44bbc4fbd773af0ebf742b08893c27ddaf1bdbf87e30329"
VALIDATION_DATA_SHA256 = "957b7fe13d03550d88f7fd4845c0870af890b248ef91fab26950a61c5b9b10a3"
CANONICAL_CHECKPOINT_SHA256 = (
    "f2d1ad7dcc094e4565f25446e613d4b528261012810bb493ef70d1a3977c0f91"
)

HIGH_DIMENSION = 64
TRAINING_ROWS = 30_000
VALIDATION_ROWS = 6_000
OBJECTIVE = "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)"


@dataclass(frozen=True)
class MiniBatchSettings:
    """Fixed settings for the requested controlled Adam experiment."""

    seed: int = 0
    batch_size: int = 1_024
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_epochs: int = 1_000
    minimum_epochs: int = 334
    equal_update_epoch: int = 334
    scheduler_factor: float = 0.5
    scheduler_patience: int = 25
    scheduler_threshold: float = 1e-4
    scheduler_min_lr: float = 1e-6
    early_stopping_patience: int = 100
    validation_tolerance_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if not 1 <= self.minimum_epochs <= self.max_epochs:
            raise ValueError("minimum_epochs must lie in [1, max_epochs]")
        if not 1 <= self.equal_update_epoch <= self.max_epochs:
            raise ValueError("equal_update_epoch must lie in [1, max_epochs]")
        if self.early_stopping_patience <= self.scheduler_patience:
            raise ValueError(
                "early_stopping_patience must exceed scheduler_patience"
            )
        if not 0.0 <= self.validation_tolerance_fraction < 1.0:
            raise ValueError("validation_tolerance_fraction must lie in [0, 1)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_sha256(path: Path, expected: str, *, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{description} SHA256 mismatch for {path}: "
            f"expected {expected}, observed {actual}"
        )
    return actual


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _load_pairs(
    path: Path,
    *,
    rows: int,
    high_dimension: int,
    skiprows: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pairs = np.loadtxt(path, delimiter=",", skiprows=skiprows, dtype=np.float64)
    expected = (rows, 2 * high_dimension)
    if pairs.shape != expected:
        raise ValueError(f"{path} has shape {pairs.shape}; expected {expected}")
    if not np.all(np.isfinite(pairs)):
        raise ValueError(f"{path} contains non-finite values")
    return (
        np.ascontiguousarray(pairs[:, :high_dimension]),
        np.ascontiguousarray(pairs[:, high_dimension:]),
    )


def _validate_pair_arrays(
    x: np.ndarray | Tensor,
    y: np.ndarray | Tensor,
    *,
    high_dimension: int,
    name: str,
) -> None:
    x_shape = tuple(x.shape)
    y_shape = tuple(y.shape)
    if len(x_shape) != 2 or len(y_shape) != 2:
        raise ValueError(f"{name} arrays must both be rank-2")
    if x_shape != y_shape:
        raise ValueError(f"{name} x/y shapes differ: {x_shape} and {y_shape}")
    if x_shape[0] < 1 or x_shape[1] != high_dimension:
        raise ValueError(
            f"{name} arrays must have shape (n, {high_dimension}); got {x_shape}"
        )


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shuffled_batch_indices(
    n_rows: int,
    batch_size: int,
    *,
    generator: torch.Generator,
) -> list[Tensor]:
    """Return one seeded, complete, drop-last-false epoch partition."""

    if n_rows < 1 or batch_size < 1:
        raise ValueError("n_rows and batch_size must be positive")
    permutation = torch.randperm(n_rows, generator=generator)
    return [
        permutation[start : start + batch_size]
        for start in range(0, n_rows, batch_size)
    ]


def _two_term_losses(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    encoded = model.encoder(x)
    reconstruction = model.decoder(encoded)
    prediction = model.decoder(model.latent_map(encoded))
    loss_reconstruction = nn.functional.mse_loss(reconstruction, x)
    loss_prediction = nn.functional.mse_loss(prediction, y)
    return (
        loss_reconstruction,
        loss_prediction,
        loss_reconstruction + loss_prediction,
    )


def _train_epoch(
    model: nn.Module,
    x_cpu: Tensor,
    y_cpu: Tensor,
    *,
    optimizer: Adam,
    batch_size: int,
    shuffle_generator: torch.Generator,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    model.train()
    totals = {
        "loss_reconstruction": 0.0,
        "loss_prediction": 0.0,
        "loss_total": 0.0,
    }
    n_samples = 0
    batches = _shuffled_batch_indices(
        int(x_cpu.shape[0]),
        batch_size,
        generator=shuffle_generator,
    )
    for indices in batches:
        x = x_cpu[indices].to(device)
        y = y_cpu[indices].to(device)
        count = int(x.shape[0])
        optimizer.zero_grad(set_to_none=True)
        reconstruction, prediction, total = _two_term_losses(model, x, y)
        if not torch.isfinite(total).item():
            raise FloatingPointError(
                f"non-finite mini-batch objective: {float(total.detach())}"
            )
        total.backward()
        optimizer.step()
        totals["loss_reconstruction"] += float(reconstruction.detach()) * count
        totals["loss_prediction"] += float(prediction.detach()) * count
        totals["loss_total"] += float(total.detach()) * count
        n_samples += count
    if n_samples != int(x_cpu.shape[0]):
        raise RuntimeError(f"epoch consumed {n_samples} of {x_cpu.shape[0]} samples")
    return ({key: value / n_samples for key, value in totals.items()}, len(batches))


def _evaluate_model(
    model: nn.Module,
    x_cpu: Tensor,
    y_cpu: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    reconstruction_sse = 0.0
    prediction_sse = 0.0
    latent_sse = 0.0
    latent_values: list[Tensor] = []
    residual_norms: list[Tensor] = []
    n_samples = 0
    reconstruction_elements = 0
    prediction_elements = 0
    latent_elements = 0
    with torch.inference_mode():
        for start in range(0, int(x_cpu.shape[0]), batch_size):
            stop = min(start + batch_size, int(x_cpu.shape[0]))
            x = x_cpu[start:stop].to(device)
            y = y_cpu[start:stop].to(device)
            count = int(x.shape[0])
            encoded_x = model.encoder(x)
            encoded_y = model.encoder(y)
            predicted_latent = model.latent_map(encoded_x)
            reconstruction_error = model.decoder(encoded_x) - x
            prediction_error = model.decoder(predicted_latent) - y
            residual = predicted_latent - encoded_y
            norms = torch.linalg.vector_norm(residual, dim=1)
            reconstruction_sse += float(
                torch.sum(reconstruction_error.to(torch.float64).square())
            )
            prediction_sse += float(
                torch.sum(prediction_error.to(torch.float64).square())
            )
            latent_sse += float(torch.sum(residual.to(torch.float64).square()))
            reconstruction_elements += reconstruction_error.numel()
            prediction_elements += prediction_error.numel()
            latent_elements += residual.numel()
            latent_values.extend((encoded_x.detach().cpu(), encoded_y.detach().cpu()))
            residual_norms.append(norms.detach().cpu())
            n_samples += count

    latent = torch.cat(latent_values).numpy().astype(np.float64, copy=False)
    residual = torch.cat(residual_norms).numpy().astype(np.float64, copy=False)
    lower, upper = np.quantile(latent, [0.01, 0.99])
    span = float(upper - lower)
    latent_rmse = math.sqrt(latent_sse / n_samples)
    if not np.isfinite(span) or span <= 0:
        raise ValueError(f"held-out encoded q99-q01 span is invalid: {span}")
    reconstruction_mse = reconstruction_sse / reconstruction_elements
    prediction_mse = prediction_sse / prediction_elements
    metrics = {
        "loss_reconstruction": reconstruction_mse,
        "loss_prediction": prediction_mse,
        "loss_total": reconstruction_mse + prediction_mse,
    }
    metrics.update(
        {
            "latent_semiconjugacy_mse": latent_sse / latent_elements,
            "latent_residual_rmse": latent_rmse,
            "latent_span_q99_q01": span,
            "normalized_latent_residual_rmse": latent_rmse / span,
            "p99_euclidean_latent_residual": float(np.quantile(residual, 0.99)),
            "max_euclidean_latent_residual": float(np.max(residual)),
        }
    )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"non-finite validation metrics: {metrics}")
    return metrics


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _select_checkpoint_epochs(
    val_loss_total: list[float],
    normalized_latent_residual: list[float],
    *,
    tolerance_fraction: float,
) -> dict[str, int | float]:
    if not val_loss_total or len(val_loss_total) != len(normalized_latent_residual):
        raise ValueError("validation histories must be nonempty and equally sized")
    decoded = np.asarray(val_loss_total, dtype=np.float64)
    latent = np.asarray(normalized_latent_residual, dtype=np.float64)
    if not np.all(np.isfinite(decoded)) or not np.all(np.isfinite(latent)):
        raise ValueError("validation histories must be finite")
    best_validation_index = int(np.argmin(decoded))
    cutoff = float(decoded[best_validation_index] * (1.0 + tolerance_fraction))
    eligible = np.flatnonzero(decoded <= cutoff)
    selected_index = min(
        (int(index) for index in eligible),
        key=lambda index: (latent[index], decoded[index], index),
    )
    return {
        "best_validation_epoch": best_validation_index + 1,
        "selected_epoch": selected_index + 1,
        "validation_cutoff": cutoff,
        "eligible_epoch_count": int(eligible.size),
    }


def _model_from_state(
    arch: ArchConfig,
    state: dict[str, Tensor],
) -> LatentDynamicsAutoencoder:
    model = build_autoencoder(arch)
    model.load_state_dict(state)
    return model


def _save_state(
    state: dict[str, Tensor],
    arch: ArchConfig,
    model_dir: Path,
    *,
    basename: str = "autoencoder",
) -> tuple[Path, Path]:
    model = _model_from_state(arch, state)
    return save_checkpoint(model, arch, model_dir, basename=basename)


def _write_history_csv(
    path: Path,
    history: dict[str, dict[str, list[float]]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "epoch",
        "train_loss_reconstruction",
        "train_loss_prediction",
        "train_loss_total",
        "val_loss_reconstruction",
        "val_loss_prediction",
        "val_loss_total",
        "val_latent_semiconjugacy_mse",
        "val_latent_residual_rmse",
        "val_latent_span_q99_q01",
        "val_normalized_latent_residual_rmse",
        "val_p99_euclidean_latent_residual",
        "val_max_euclidean_latent_residual",
        "learning_rate",
    ]
    n_epochs = len(history["optimizer"]["learning_rate"])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index in range(n_epochs):
            row: dict[str, int | float] = {"epoch": index + 1}
            for key, values in history["train"].items():
                row[f"train_{key}"] = values[index]
            for key, values in history["val"].items():
                row[f"val_{key}"] = values[index]
            row["learning_rate"] = history["optimizer"]["learning_rate"][index]
            writer.writerow(row)
    return path


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "device": str(device),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "deterministic_algorithms_enforced": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def train_minibatch(
    *,
    arch: ArchConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    settings: MiniBatchSettings,
    device: torch.device,
    output_dir: Path,
    run_plan_sha256: str,
    verbose: bool,
) -> dict[str, Any]:
    """Train, select, and persist one controlled mini-batch experiment."""

    _validate_pair_arrays(
        x_train,
        y_train,
        high_dimension=arch.high_dims,
        name="training",
    )
    _validate_pair_arrays(
        x_val,
        y_val,
        high_dimension=arch.high_dims,
        name="validation",
    )
    existing = (
        {path.name for path in output_dir.iterdir()}
        if output_dir.exists()
        else set()
    )
    if existing not in (set(), {"run_plan.json"}):
        raise FileExistsError(
            f"{output_dir} contains {sorted(existing)}; refusing to overwrite "
            "an existing run"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if _sha256(output_dir / "run_plan.json") != run_plan_sha256:
        raise ValueError("run_plan.json changed before training began")

    _seed_everything(settings.seed)
    x_train_cpu = torch.as_tensor(x_train, dtype=torch.float32).contiguous()
    y_train_cpu = torch.as_tensor(y_train, dtype=torch.float32).contiguous()
    x_val_cpu = torch.as_tensor(x_val, dtype=torch.float32).contiguous()
    y_val_cpu = torch.as_tensor(y_val, dtype=torch.float32).contiguous()
    model = build_autoencoder(arch).to(device)
    validation_model = build_autoencoder(arch).cpu()
    optimizer = Adam(
        model.parameters(),
        lr=settings.learning_rate,
        betas=(settings.beta1, settings.beta2),
        eps=settings.epsilon,
        weight_decay=settings.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=settings.scheduler_factor,
        patience=settings.scheduler_patience,
        threshold=settings.scheduler_threshold,
        threshold_mode="rel",
        min_lr=settings.scheduler_min_lr,
    )
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(settings.seed)

    history: dict[str, dict[str, list[float]]] = {
        "train": {
            "loss_reconstruction": [],
            "loss_prediction": [],
            "loss_total": [],
        },
        "val": {
            "loss_reconstruction": [],
            "loss_prediction": [],
            "loss_total": [],
            "latent_semiconjugacy_mse": [],
            "latent_residual_rmse": [],
            "latent_span_q99_q01": [],
            "normalized_latent_residual_rmse": [],
            "p99_euclidean_latent_residual": [],
            "max_euclidean_latent_residual": [],
        },
        "optimizer": {"learning_rate": []},
    }
    states: list[dict[str, Tensor]] = []
    best_validation = float("inf")
    no_improvement = 0
    optimizer_updates = 0
    started = time.perf_counter()

    for epoch_index in range(settings.max_epochs):
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        train_metrics, updates = _train_epoch(
            model,
            x_train_cpu,
            y_train_cpu,
            optimizer=optimizer,
            batch_size=settings.batch_size,
            shuffle_generator=shuffle_generator,
            device=device,
        )
        state = _cpu_state_dict(model)
        validation_model.load_state_dict(state)
        val_metrics = _evaluate_model(
            validation_model,
            x_val_cpu,
            y_val_cpu,
            batch_size=settings.batch_size,
            device=torch.device("cpu"),
        )
        for key, value in train_metrics.items():
            history["train"][key].append(float(value))
        for key, value in val_metrics.items():
            history["val"][key].append(float(value))
        history["optimizer"]["learning_rate"].append(learning_rate_used)
        states.append(state)
        optimizer_updates += updates

        val_total = val_metrics["loss_total"]
        if val_total < best_validation:
            best_validation = val_total
            no_improvement = 0
        else:
            no_improvement += 1
        scheduler.step(val_total)

        epoch = epoch_index + 1
        if verbose and (
            epoch == 1
            or epoch % 10 == 0
            or epoch == settings.equal_update_epoch
            or epoch == settings.max_epochs
        ):
            print(
                f"epoch={epoch:04d} "
                f"train={train_metrics['loss_total']:.6e} "
                f"val={val_total:.6e} "
                f"norm_L3={val_metrics['normalized_latent_residual_rmse']:.6e} "
                f"lr={learning_rate_used:.3e} "
                f"no_improve={no_improvement}",
                flush=True,
            )
        if (
            epoch >= settings.minimum_epochs
            and no_improvement >= settings.early_stopping_patience
        ):
            if verbose:
                print(f"early_stopping_epoch={epoch}", flush=True)
            break

    duration = time.perf_counter() - started
    epochs_completed = len(states)
    if epochs_completed < settings.equal_update_epoch:
        raise RuntimeError(
            f"run stopped at epoch {epochs_completed}, before equal-update "
            f"epoch {settings.equal_update_epoch}"
        )
    selection = _select_checkpoint_epochs(
        history["val"]["loss_total"],
        history["val"]["normalized_latent_residual_rmse"],
        tolerance_fraction=settings.validation_tolerance_fraction,
    )
    selected_epoch = int(selection["selected_epoch"])
    best_validation_epoch = int(selection["best_validation_epoch"])
    candidate_epochs = {
        "equal_update": settings.equal_update_epoch,
        "best_validation": best_validation_epoch,
        "best_normalized_latent": selected_epoch,
        "final": epochs_completed,
    }

    candidate_manifest: dict[str, Any] = {}
    for name, epoch in candidate_epochs.items():
        model_dir = output_dir / "candidates" / name / "models"
        checkpoint, sidecar = _save_state(states[epoch - 1], arch, model_dir)
        candidate_manifest[name] = {
            "epoch": epoch,
            "validation": {
                key: values[epoch - 1]
                for key, values in history["val"].items()
            },
            "checkpoint": {
                "path": str(checkpoint.relative_to(output_dir)),
                "sha256": _sha256(checkpoint),
            },
            "sidecar": str(sidecar.relative_to(output_dir)),
        }

    primary_checkpoint, primary_sidecar = _save_state(
        states[selected_epoch - 1],
        arch,
        output_dir / "models",
        basename="selected",
    )
    history_payload = {
        "schema_version": 1,
        "training_method": "marcio_seeded_minibatch",
        "epoch_indexing": "list index 0 is epoch 1",
        **history,
    }
    history_path = _write_json(output_dir / "logs" / "history.json", history_payload)
    history_csv = _write_history_csv(output_dir / "logs" / "history.csv", history)
    manifest_path = _write_json(
        output_dir / "selection_record.json",
        {
            "schema_version": 1,
            "primary_candidate": "best_normalized_latent",
            "selected_basename": "selected",
            "run_plan_sha256": run_plan_sha256,
            "selection_inputs": ["independent validation pairs only"],
            "basin_artifacts_accessed_before_selection_freeze": False,
            "selection": selection,
            "selected_checkpoint": {
                "epoch": selected_epoch,
                "path": str(primary_checkpoint.relative_to(output_dir)),
                "sha256": _sha256(primary_checkpoint),
                "sidecar_path": str(primary_sidecar.relative_to(output_dir)),
                "sidecar_sha256": _sha256(primary_sidecar),
            },
            "candidates": candidate_manifest,
        },
    )

    batches_per_epoch = math.ceil(int(x_train_cpu.shape[0]) / settings.batch_size)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "training_method": "marcio_seeded_minibatch",
        "objective": OBJECTIVE,
        "settings": asdict(settings),
        "run_plan_sha256": run_plan_sha256,
        "optimizer": {
            "name": "Adam",
            "learning_rate": settings.learning_rate,
            "betas": [settings.beta1, settings.beta2],
            "epsilon": settings.epsilon,
            "weight_decay": settings.weight_decay,
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "monitor": "val.loss_total",
            "factor": settings.scheduler_factor,
            "patience": settings.scheduler_patience,
            "threshold": settings.scheduler_threshold,
            "threshold_mode": "rel",
            "min_lr": settings.scheduler_min_lr,
        },
        "early_stopping": {
            "monitor": "val.loss_total",
            "patience": settings.early_stopping_patience,
            "minimum_epochs": settings.minimum_epochs,
            "used": epochs_completed < settings.max_epochs,
        },
        "data": {
            "training_pairs": int(x_train_cpu.shape[0]),
            "validation_pairs": int(x_val_cpu.shape[0]),
            "high_dims": int(x_train_cpu.shape[1]),
            "dtype": "float32",
            "full_batch": False,
            "batch_size": settings.batch_size,
            "shuffle": True,
            "drop_last": False,
            "batches_per_epoch": batches_per_epoch,
            "last_batch_size": int(x_train_cpu.shape[0]) % settings.batch_size,
            "scaling": "none",
        },
        "arch": arch.model_dump(mode="json"),
        "epochs_requested": settings.max_epochs,
        "epochs_completed": epochs_completed,
        "optimizer_updates": optimizer_updates,
        "example_presentations": epochs_completed * int(x_train_cpu.shape[0]),
        "duration_seconds": duration,
        "duration_minutes": duration / 60.0,
        "checkpoint_epoch": selected_epoch,
        "checkpoint_selection": {
            "name": "best_normalized_latent_within_validation_tolerance",
            "rule": (
                "minimum val.normalized_latent_residual_rmse among epochs with "
                "val.loss_total <= 1.05 * minimum val.loss_total"
            ),
            **selection,
            "basin_trajectories_consulted": False,
            "validation_evaluation_device": "cpu",
            "validation_accumulation": "float64 sum of squared errors",
        },
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "runtime": _runtime_metadata(device),
        "artifacts": {
            "primary_checkpoint": str(primary_checkpoint.relative_to(output_dir)),
            "primary_checkpoint_sha256": _sha256(primary_checkpoint),
            "primary_sidecar": str(primary_sidecar.relative_to(output_dir)),
            "history": str(history_path.relative_to(output_dir)),
            "history_csv": str(history_csv.relative_to(output_dir)),
            "selection_record": str(manifest_path.relative_to(output_dir)),
        },
    }
    _write_json(output_dir / "training_summary.json", summary)
    return summary


def run_experiment(
    *,
    output_dir: Path,
    device_name: str,
    verbose: bool,
) -> dict[str, Any]:
    train_hash = _checked_sha256(
        TRAIN_DATA,
        TRAIN_DATA_SHA256,
        description="exact Marcio training data",
    )
    validation_hash = _checked_sha256(
        VALIDATION_DATA,
        VALIDATION_DATA_SHA256,
        description="independent Chafee validation data",
    )
    canonical_checkpoint = CANONICAL_RUN / "models" / "autoencoder.pt"
    canonical_hash = _checked_sha256(
        canonical_checkpoint,
        CANONICAL_CHECKPOINT_SHA256,
        description="canonical d=1 checkpoint",
    )
    _, arch = load_checkpoint(CANONICAL_RUN / "models", map_location="cpu")
    if arch.high_dims != HIGH_DIMENSION or arch.low_dims != 1:
        raise ValueError(
            f"canonical architecture is {arch.high_dims}->{arch.low_dims}, expected 64->1"
        )
    x_train, y_train = _load_pairs(
        TRAIN_DATA,
        rows=TRAINING_ROWS,
        high_dimension=HIGH_DIMENSION,
        skiprows=0,
    )
    x_val, y_val = _load_pairs(
        VALIDATION_DATA,
        rows=VALIDATION_ROWS,
        high_dimension=HIGH_DIMENSION,
        skiprows=1,
    )
    device = _resolve_device(device_name)
    settings = MiniBatchSettings()
    resolved_output = output_dir.resolve()
    if resolved_output.exists() and any(resolved_output.iterdir()):
        raise FileExistsError(
            f"{resolved_output} is not empty; refusing to overwrite an existing run"
        )
    resolved_output.mkdir(parents=True, exist_ok=True)
    run_plan = {
        "schema_version": 1,
        "status": "frozen_before_training",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": resolved_output.name,
        "training_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "architecture": arch.model_dump(mode="json"),
        "objective": {
            "formula": OBJECTIVE,
            "weights": [1.0, 1.0, 0.0],
        },
        "settings": asdict(settings),
        "data": {
            "training": {
                "path": str(TRAIN_DATA.resolve()),
                "sha256": train_hash,
                "shape": [TRAINING_ROWS, 2 * HIGH_DIMENSION],
                "role": "gradient updates only",
                "scaling": "none",
            },
            "validation": {
                "path": str(VALIDATION_DATA.resolve()),
                "sha256": validation_hash,
                "shape": [VALIDATION_ROWS, 2 * HIGH_DIMENSION],
                "role": "scheduler, early stopping, and checkpoint selection only",
                "sampling_seed": 9999,
                "scaling": "none",
            },
        },
        "checkpoint_selection": {
            "rule_id": "val_decoded_5pct_then_normalized_semiconjugacy_v1",
            "decoded_metric": "val.loss_reconstruction + val.loss_prediction",
            "eligibility": "decoded metric <= 1.05 * run-wide minimum",
            "ranking": [
                "normalized latent residual RMSE",
                "decoded metric",
                "earliest epoch",
            ],
            "basin_or_cmgdb_inputs_allowed": False,
        },
    }
    plan_path = _write_json(resolved_output / "run_plan.json", run_plan)
    plan_hash = _sha256(plan_path)
    summary = train_minibatch(
        arch=arch,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        settings=settings,
        device=device,
        output_dir=resolved_output,
        run_plan_sha256=plan_hash,
        verbose=verbose,
    )
    summary["sources"] = {
        "training_data": {
            "path": str(TRAIN_DATA.resolve()),
            "sha256": train_hash,
        },
        "validation_data": {
            "path": str(VALIDATION_DATA.resolve()),
            "sha256": validation_hash,
        },
        "architecture_source_checkpoint": {
            "path": str(canonical_checkpoint.resolve()),
            "sha256": canonical_hash,
            "weights_reused": False,
        },
    }
    _write_json(resolved_output / "training_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_experiment(
        output_dir=args.output_dir,
        device_name=args.device,
        verbose=not args.quiet,
    )
    print(
        f"completed_epochs={summary['epochs_completed']} "
        f"selected_epoch={summary['checkpoint_epoch']} "
        f"output={args.output_dir.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
