r"""Marcio-faithful full-batch training for the Chafee--Infante studies.

The archived paper computation in ``archive/marcio/scripts/train_model.py``
uses one full batch and the two-term objective

.. math::

    \operatorname{MSE}(D(E(x)), x)
    + \operatorname{MSE}(D(G(E(x))), y).

This module preserves those semantics in a reusable, checkpoint-compatible
function.  It intentionally does not use :class:`~latentdynamics.training.Trainer`:
that general-purpose trainer adds validation, early stopping, and best-weight
restoration, none of which occur in Marcio's Chafee--Infante loop.

The helper seeds Python, NumPy, and PyTorch RNGs but does not force PyTorch's
deterministic-algorithm mode.  Its metadata therefore records the backend and
runtime and explicitly does not claim bitwise reproducibility across devices,
hardware, or library versions.
"""

from __future__ import annotations

import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..config.schema import ArchConfig
from ..models.autoencoder import LatentDynamicsAutoencoder, build_autoencoder
from .checkpoints import DEFAULT_BASENAME, save_checkpoint

MARCIO_OBJECTIVE = "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)"


@dataclass(frozen=True)
class MarcioFullBatchResult:
    """In-memory result and artifact paths from a faithful full-batch run."""

    model: LatentDynamicsAutoencoder
    history: dict[str, list[float]]
    summary: dict[str, Any]
    checkpoint_path: Path
    checkpoint_metadata_path: Path
    history_path: Path
    summary_path: Path


def _seed_marcio_run(seed: int) -> None:
    """Seed every RNG that can affect model construction or PyTorch training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _full_batch_tensor(
    values: np.ndarray | Tensor,
    *,
    name: str,
    device: torch.device,
) -> Tensor:
    """Match Marcio's explicit float32 conversion and full-device transfer."""

    try:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} cannot be converted to a float32 tensor") from exc
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 array, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one training pair")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains non-finite values")
    return tensor.contiguous()


def train_marcio_full_batch(
    *,
    arch: ArchConfig,
    x: np.ndarray | Tensor,
    y: np.ndarray | Tensor,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str | torch.device,
    output_dir: str | Path,
    model: LatentDynamicsAutoencoder | None = None,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 100,
    scheduler_threshold: float = 1e-4,
    scheduler_min_lr: float = 1e-6,
    basename: str = DEFAULT_BASENAME,
    verbose: bool = False,
) -> MarcioFullBatchResult:
    """Train with the exact optimization structure used by Marcio's paper run.

    The complete ``x``/``y`` arrays are moved to ``device`` as float32 tensors
    and used in one optimizer update per epoch.  The function always executes
    exactly ``epochs`` updates unless a non-finite loss makes the run invalid.
    There is no validation pass, early stopping, best-epoch selection, gradient
    clipping, or best-weight restoration.  The saved checkpoint is therefore
    the model state immediately after the final requested optimizer update.

    When ``model`` is omitted it is constructed *after* all random-number
    generators are seeded.  Supplying a model is useful for controlled tests or
    warm starts; in that case the seed still controls all subsequent stochastic
    PyTorch operations, but the caller owns the supplied initialization.

    Args:
        arch: Architecture used to build the model and checkpoint sidecar.
        x: Current states, shaped ``(n_pairs, arch.high_dims)``.
        y: Corresponding one-step states, with the same shape as ``x``.
        epochs: Fixed number of full-batch Adam updates.
        learning_rate: Initial Adam learning rate.
        seed: Explicit Python, NumPy, and PyTorch seed.
        device: PyTorch device on which the full batch and model should live.
        output_dir: Run root. Artifacts use the standard ``models/`` and
            ``logs/`` subdirectories.
        model: Optional preconstructed compatible autoencoder.
        scheduler_factor: ``ReduceLROnPlateau`` reduction factor.
        scheduler_patience: Plateau epochs tolerated before reducing the rate.
        scheduler_threshold: Relative improvement threshold.
        scheduler_min_lr: Lower bound on the learning rate.
        basename: Checkpoint basename in ``output_dir/models``.
        verbose: Print Marcio-style progress every 100 epochs and at completion.

    Returns:
        A :class:`MarcioFullBatchResult` containing the trained model, compact
        loss history, summary metadata, and artifact paths.
    """

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0.0 < scheduler_factor < 1.0:
        raise ValueError("scheduler_factor must lie strictly between 0 and 1")
    if (
        isinstance(scheduler_patience, bool)
        or not isinstance(scheduler_patience, int)
        or scheduler_patience < 0
    ):
        raise ValueError("scheduler_patience must be a non-negative integer")
    if scheduler_threshold < 0:
        raise ValueError("scheduler_threshold must be non-negative")
    if scheduler_min_lr < 0:
        raise ValueError("scheduler_min_lr must be non-negative")
    if scheduler_min_lr > learning_rate:
        raise ValueError("scheduler_min_lr cannot exceed learning_rate")
    if not basename:
        raise ValueError("basename must not be empty")

    resolved_device = torch.device(device)
    _seed_marcio_run(seed)

    x_batch = _full_batch_tensor(x, name="x", device=resolved_device)
    y_batch = _full_batch_tensor(y, name="y", device=resolved_device)
    if x_batch.shape != y_batch.shape:
        raise ValueError(
            f"x and y must have the same shape, got {tuple(x_batch.shape)} "
            f"and {tuple(y_batch.shape)}"
        )
    if x_batch.shape[1] != arch.high_dims:
        raise ValueError(
            f"x/y feature dimension {x_batch.shape[1]} does not match "
            f"arch.high_dims={arch.high_dims}"
        )

    initialized_by_helper = model is None
    if model is None:
        model = build_autoencoder(arch)
    model = model.to(resolved_device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        threshold_mode="rel",
        min_lr=scheduler_min_lr,
    )

    history: dict[str, list[float]] = {
        "loss_reconstruction": [],
        "loss_prediction": [],
        "loss_total": [],
        "learning_rate": [],
    }

    for epoch in range(epochs):
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad()

        z_current = model.encoder(x_batch)
        z_forward = model.latent_map(z_current)
        x_current_decoded = model.decoder(z_current)
        loss_reconstruction = criterion(x_current_decoded, x_batch)
        x_forward_decoded = model.decoder(z_forward)
        loss_prediction = criterion(x_forward_decoded, y_batch)
        loss_total = loss_reconstruction + loss_prediction

        if not torch.isfinite(loss_total).item():
            raise FloatingPointError(
                f"non-finite full-batch loss at epoch {epoch + 1}: "
                f"{float(loss_total.detach())}"
            )

        history["loss_reconstruction"].append(float(loss_reconstruction.detach()))
        history["loss_prediction"].append(float(loss_prediction.detach()))
        history["loss_total"].append(float(loss_total.detach()))
        history["learning_rate"].append(learning_rate_used)

        loss_total.backward()
        optimizer.step()

        # Marcio's loop monitors the same full-batch training objective, after
        # the optimizer update and without a validation metric.
        scheduler.step(loss_total.detach())

        if verbose and (epoch % 100 == 0 or epoch + 1 == epochs):
            print(
                f"Epoch {epoch}/{epochs} | Total Loss: "
                f"{history['loss_total'][-1]:0.6f}"
            )

    output_root = Path(output_dir)
    checkpoint_path, checkpoint_metadata_path = save_checkpoint(
        model,
        arch,
        output_root / "models",
        basename=basename,
    )

    history_path = output_root / "logs" / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_payload = {
        "schema_version": 1,
        "training_method": "marcio_full_batch",
        "epoch_indexing": "list index 0 is epoch 1",
        "train": history,
    }
    history_path.write_text(
        json.dumps(history_payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    final_losses = {
        "loss_reconstruction": history["loss_reconstruction"][-1],
        "loss_prediction": history["loss_prediction"][-1],
        "loss_total": history["loss_total"][-1],
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "training_method": "marcio_full_batch",
        "objective": MARCIO_OBJECTIVE,
        "optimizer": {
            "name": "Adam",
            "learning_rate": float(learning_rate),
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "monitor": "train.loss_total",
            "mode": "min",
            "factor": float(scheduler_factor),
            "patience": int(scheduler_patience),
            "threshold": float(scheduler_threshold),
            "threshold_mode": "rel",
            "min_lr": float(scheduler_min_lr),
        },
        "seed": int(seed),
        "device": str(resolved_device),
        "reproducibility": {
            "seeded_rngs": [
                "python",
                "numpy",
                "torch_cpu",
                "torch_cuda_if_available",
            ],
            "resolved_backend": resolved_device.type,
            "deterministic_algorithms_enforced": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "bitwise_reproducible_across_backends_or_runtime_versions": False,
            "limitation": (
                "The seed fixes RNG streams, but deterministic PyTorch algorithms "
                "are not forced. Backend kernels, hardware, and runtime versions "
                "may produce numerically different trained checkpoints."
            ),
            "runtime": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": str(np.__version__),
                "torch": str(torch.__version__),
                "mps_available": bool(torch.backends.mps.is_available()),
                "cuda_available": bool(torch.cuda.is_available()),
            },
        },
        "data": {
            "n_pairs": int(x_batch.shape[0]),
            "high_dims": int(x_batch.shape[1]),
            "dtype": "float32",
            "full_batch": True,
        },
        "arch": arch.model_dump(mode="json"),
        "model_initialized_by_helper": initialized_by_helper,
        "epochs_requested": int(epochs),
        "epochs_completed": int(epochs),
        "checkpoint_epoch": int(epochs),
        "checkpoint_selection": "final_epoch",
        "validation_used": False,
        "early_stopping_used": False,
        "best_weight_restoration_used": False,
        "final_epoch_train": final_losses,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "artifacts": {
            "checkpoint": str(checkpoint_path.relative_to(output_root)),
            "checkpoint_metadata": str(checkpoint_metadata_path.relative_to(output_root)),
            "history": str(history_path.relative_to(output_root)),
        },
    }
    summary_path = output_root / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return MarcioFullBatchResult(
        model=model,
        history=history,
        summary=summary,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        history_path=history_path,
        summary_path=summary_path,
    )
