"""Fixed-epoch full-batch curriculum training without stopping or scheduling.

This loop extends the reference full-batch recipe for experiments that
progressively activate the reconstruction, decoded-prediction, and
semiconjugacy losses.  It is intentionally separate from both
:mod:`latentdynamics.training.reference_recipe` (whose archived two-term
scheduler semantics are preserved) and the generic ``Trainer`` (which performs
validation selection, early stopping, and learning rate scheduling).

No released configuration uses curriculum training; the loop is kept because
the configuration schema and training CLI support it.
"""

from __future__ import annotations

import copy
import json
import platform
import random
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import LBFGS, Adam, AdamW

from ..config.schema import (
    ArchConfig,
    CurriculumLBFGSPolishConfig,
    CurriculumOptimizerConfig,
    CurriculumStageConfig,
)
from ..models.autoencoder import LatentDynamicsAutoencoder, build_autoencoder
from .checkpoints import DEFAULT_BASENAME, save_checkpoint

CURRICULUM_OBJECTIVE = "w1*MSE(D(E(x)),x) + w2*MSE(D(G(E(x))),y) + w3*MSE(G(E(x)),E(y))"


@dataclass(frozen=True)
class CurriculumFullBatchResult:
    """Trained model and canonical artifacts from one curriculum run."""

    model: LatentDynamicsAutoencoder
    history: dict[str, Any]
    summary: dict[str, Any]
    checkpoint_path: Path
    checkpoint_metadata_path: Path
    history_path: Path
    summary_path: Path
    final_losses_path: Path


def _seed_run(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _full_batch_tensor(
    values: np.ndarray | Tensor,
    *,
    name: str,
    device: torch.device,
) -> Tensor:
    try:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} cannot be converted to a float32 tensor") from exc
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one pair")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains non-finite values")
    return tensor.contiguous()


def _loss_terms(
    model: LatentDynamicsAutoencoder,
    x: Tensor,
    y: Tensor,
    weights: Sequence[float],
) -> dict[str, Tensor]:
    fp = model(x, y)
    reconstruction = nn.functional.mse_loss(fp.x_t_hat, fp.x_t)
    prediction = nn.functional.mse_loss(fp.x_tau_hat, fp.x_tau)
    semiconjugacy = nn.functional.mse_loss(fp.z_tau_pred, fp.z_tau)
    total = (
        float(weights[0]) * reconstruction
        + float(weights[1]) * prediction
        + float(weights[2]) * semiconjugacy
    )
    losses = {
        "loss_reconstruction": reconstruction,
        "loss_prediction": prediction,
        "loss_semiconjugacy": semiconjugacy,
    }
    if len(weights) == 4 and float(weights[3]) != 0.0:
        cycle = nn.functional.mse_loss(fp.z_tau_pred_cycle, fp.z_tau_pred)
        total = total + float(weights[3]) * cycle
        losses["loss_cycle"] = cycle
    losses["loss_total"] = total
    return losses


def _detached(losses: dict[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach()) for name, value in losses.items()}


@torch.no_grad()
def _evaluate(
    model: LatentDynamicsAutoencoder,
    x: Tensor,
    y: Tensor,
    weights: Sequence[float],
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    result = _detached(_loss_terms(model, x, y, weights))
    model.train(was_training)
    return result


def _set_trainable_components(
    model: LatentDynamicsAutoencoder,
    names: Sequence[str],
) -> None:
    selected = set(names)
    modules = {
        "encoder": model.encoder,
        "latent_map": model.latent_map,
        "decoder": model.decoder,
    }
    unknown = selected - set(modules)
    if unknown:
        raise ValueError(f"unknown trainable component(s): {sorted(unknown)}")
    for name, module in modules.items():
        requires_grad = name in selected
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return normalized or "stage"


def _history_stats(series: Sequence[float], *, selected_index: int) -> dict[str, float]:
    values = [float(value) for value in series]
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "final": float(values[-1]),
        # Compatibility field for sweep readers. Selection is always the final
        # epoch in this trainer; it is not the minimum validation epoch.
        "best_epoch_value": float(values[selected_index]),
    }


def _first_order_optimizer(
    model: LatentDynamicsAutoencoder,
    config: CurriculumOptimizerConfig,
    *,
    learning_rate: float,
) -> torch.optim.Optimizer:
    kwargs = {
        "lr": float(learning_rate),
        "betas": (float(config.betas[0]), float(config.betas[1])),
        "eps": float(config.eps),
        "weight_decay": float(config.weight_decay),
        "amsgrad": bool(config.amsgrad),
        "foreach": bool(config.foreach),
        "fused": bool(config.fused),
    }
    optimizer_type = AdamW if config.name == "adamw" else Adam
    return optimizer_type(model.parameters(), **kwargs)


def _float32_model_copy(model: LatentDynamicsAutoencoder) -> LatentDynamicsAutoencoder:
    """Clone an optimizer-owned float64 model without changing its state."""

    candidate = copy.deepcopy(model).cpu().float()
    candidate.train(model.training)
    return candidate


def _loss_delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {name: float(after[name] - before[name]) for name in before}


def train_curriculum_full_batch(
    *,
    arch: ArchConfig,
    stages: Sequence[CurriculumStageConfig],
    x: np.ndarray | Tensor,
    y: np.ndarray | Tensor,
    x_validation: np.ndarray | Tensor,
    y_validation: np.ndarray | Tensor,
    seed: int,
    device: str | torch.device | None,
    output_dir: str | Path,
    first_order_optimizer: CurriculumOptimizerConfig | None = None,
    polish: CurriculumLBFGSPolishConfig | None = None,
    model: LatentDynamicsAutoencoder | None = None,
    basename: str = DEFAULT_BASENAME,
    verbose: bool = False,
) -> CurriculumFullBatchResult:
    """Run every curriculum phase for exactly its requested full-batch epochs.

    One configured Adam/AdamW optimizer is created before the first phase and
    its state is retained across phase boundaries. Each stage can change the
    fixed learning rate, active loss weights, and trainable modules. An
    optional final L-BFGS pass starts from fresh optimizer state on CPU
    float64, then the exact final endpoint is cast to float32 for checkpointing.
    There is no scheduler, patience threshold, early stopping, gradient
    clipping, validation-based selection, or best-weight restoration.
    Validation losses are always reporting-only.
    """

    stages = list(stages)
    if not stages:
        raise ValueError("stages must not be empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not basename:
        raise ValueError("basename must not be empty")

    optimizer_config = first_order_optimizer or CurriculumOptimizerConfig()

    resolved_device = _resolve_device(device)
    _seed_run(seed)
    x_train = _full_batch_tensor(x, name="x", device=resolved_device)
    y_train = _full_batch_tensor(y, name="y", device=resolved_device)
    x_val = _full_batch_tensor(
        x_validation,
        name="x_validation",
        device=resolved_device,
    )
    y_val = _full_batch_tensor(
        y_validation,
        name="y_validation",
        device=resolved_device,
    )
    if x_train.shape != y_train.shape:
        raise ValueError(
            f"x and y must have the same shape, got {tuple(x_train.shape)} "
            f"and {tuple(y_train.shape)}"
        )
    if x_val.shape != y_val.shape:
        raise ValueError(
            "x_validation and y_validation must have the same shape, got "
            f"{tuple(x_val.shape)} and {tuple(y_val.shape)}"
        )
    if x_train.shape[1] != arch.high_dims or x_val.shape[1] != arch.high_dims:
        raise ValueError(
            f"training and validation feature dimensions must equal arch.high_dims={arch.high_dims}"
        )

    initialized_by_helper = model is None
    if model is None:
        model = build_autoencoder(arch)
    model = model.to(resolved_device)
    model.train()

    optimizer = _first_order_optimizer(
        model,
        optimizer_config,
        learning_rate=float(stages[0].learning_rate),
    )
    train_history: dict[str, list[float]] = {}
    val_history: dict[str, list[float]] = {}
    stage_index_history: list[int] = []
    stage_name_history: list[str] = []
    learning_rate_history: list[float] = []
    loss_weights_history: list[list[float]] = []
    stage_records: list[dict[str, Any]] = []
    global_epoch = 0
    output_root = Path(output_dir)
    started = time.perf_counter()

    for stage_index, stage in enumerate(stages, start=1):
        _set_trainable_components(model, stage.trainable_components)
        for group in optimizer.param_groups:
            group["lr"] = float(stage.learning_rate)
        start_epoch = global_epoch + 1

        for local_epoch in range(1, stage.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = _loss_terms(model, x_train, y_train, stage.loss_weights)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError(
                    f"non-finite loss in curriculum stage {stage.name!r}, local epoch {local_epoch}"
                )
            train_values = _detached(losses)
            losses["loss_total"].backward()
            optimizer.step()
            val_values = _evaluate(model, x_val, y_val, stage.loss_weights)

            for name, value in train_values.items():
                train_history.setdefault(name, []).append(value)
            for name, value in val_values.items():
                val_history.setdefault(name, []).append(value)
            stage_index_history.append(stage_index)
            stage_name_history.append(stage.name)
            learning_rate_history.append(float(stage.learning_rate))
            loss_weights_history.append([float(value) for value in stage.loss_weights])
            global_epoch += 1

            if verbose and (
                local_epoch == 1 or local_epoch % 100 == 0 or local_epoch == stage.epochs
            ):
                print(
                    f"stage {stage_index}/{len(stages)} {stage.name} "
                    f"epoch {local_epoch}/{stage.epochs} global={global_epoch} "
                    f"train={train_values['loss_total']:.6e} "
                    f"holdout={val_values['loss_total']:.6e}"
                )

        train_endpoint = _evaluate(model, x_train, y_train, stage.loss_weights)
        val_endpoint = _evaluate(model, x_val, y_val, stage.loss_weights)
        stage_root = output_root / "stage_checkpoints" / f"{stage_index:02d}_{_slug(stage.name)}"
        stage_checkpoint, stage_metadata = save_checkpoint(
            model,
            arch,
            stage_root / "models",
            basename=basename,
        )
        stage_records.append(
            {
                "index": stage_index,
                "name": stage.name,
                "start_epoch_one_based": start_epoch,
                "end_epoch_one_based": global_epoch,
                "epochs": int(stage.epochs),
                "learning_rate": float(stage.learning_rate),
                "loss_weights": [float(value) for value in stage.loss_weights],
                "trainable_components": list(stage.trainable_components),
                "optimizer_state_continued_from_previous_stage": stage_index > 1,
                "train_endpoint_post_update": train_endpoint,
                "holdout_endpoint_post_update": val_endpoint,
                "checkpoint": str(stage_checkpoint.relative_to(output_root)),
                "checkpoint_metadata": str(stage_metadata.relative_to(output_root)),
            }
        )

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    first_order_elapsed = time.perf_counter() - started
    final_weights = stages[-1].loss_weights
    adamw_train = _evaluate(model, x_train, y_train, final_weights)
    adamw_val = _evaluate(model, x_val, y_val, final_weights)
    adamw_checkpoint_path, adamw_checkpoint_metadata_path = save_checkpoint(
        model,
        arch,
        output_root / "adamw_endpoint" / "models",
        basename=basename,
    )

    polish_records: list[dict[str, Any]] = []
    polish_summary: dict[str, Any] | None = None
    polish_elapsed = 0.0
    if polish is not None:
        if list(polish.loss_weights) != list(final_weights):
            raise ValueError("polish.loss_weights must match the final curriculum stage")
        polish_started = time.perf_counter()
        _set_trainable_components(model, polish.trainable_components)
        model = model.cpu().double()
        model.train()
        x_train_64 = x_train.detach().cpu().double().contiguous()
        y_train_64 = y_train.detach().cpu().double().contiguous()
        x_val_64 = x_val.detach().cpu().double().contiguous()
        y_val_64 = y_val.detach().cpu().double().contiguous()
        x_train_32 = x_train.detach().cpu().float().contiguous()
        y_train_32 = y_train.detach().cpu().float().contiguous()
        x_val_32 = x_val.detach().cpu().float().contiguous()
        y_val_32 = y_val.detach().cpu().float().contiguous()
        optimized_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not optimized_parameters:
            raise ValueError("L-BFGS polish has no trainable parameters")
        lbfgs = LBFGS(
            optimized_parameters,
            lr=float(polish.learning_rate),
            max_iter=int(polish.max_iter),
            max_eval=int(polish.max_eval),
            tolerance_grad=float(polish.tolerance_grad),
            tolerance_change=float(polish.tolerance_change),
            history_size=int(polish.history_size),
            line_search_fn=polish.line_search_fn,
        )
        total_closure_calls = 0
        previous_internal_iterations = 0
        previous_function_evaluations = 0

        for outer_step in range(1, polish.outer_steps + 1):
            closure_calls = 0

            def closure(_outer_step: int = outer_step) -> Tensor:
                nonlocal closure_calls
                closure_calls += 1
                lbfgs.zero_grad(set_to_none=True)
                losses = _loss_terms(model, x_train_64, y_train_64, polish.loss_weights)
                if not all(torch.isfinite(value).item() for value in losses.values()):
                    raise FloatingPointError(f"non-finite L-BFGS loss at outer step {_outer_step}")
                losses["loss_total"].backward()
                return losses["loss_total"]

            returned = lbfgs.step(closure)
            total_closure_calls += closure_calls
            state_blocks = [state for state in lbfgs.state.values() if isinstance(state, dict)]
            cumulative_internal_iterations = max(
                (int(state.get("n_iter", 0)) for state in state_blocks),
                default=0,
            )
            cumulative_function_evaluations = max(
                (int(state.get("func_evals", 0)) for state in state_blocks),
                default=total_closure_calls,
            )
            endpoint_64_train = _evaluate(model, x_train_64, y_train_64, polish.loss_weights)
            endpoint_64_holdout = _evaluate(model, x_val_64, y_val_64, polish.loss_weights)
            endpoint_32_model = _float32_model_copy(model)
            endpoint_32_train = _evaluate(
                endpoint_32_model, x_train_32, y_train_32, polish.loss_weights
            )
            endpoint_32_holdout = _evaluate(
                endpoint_32_model, x_val_32, y_val_32, polish.loss_weights
            )
            record = {
                "outer_step": outer_step,
                "closure_evaluations": int(closure_calls),
                "cumulative_closure_evaluations": int(total_closure_calls),
                "internal_iterations": int(
                    cumulative_internal_iterations - previous_internal_iterations
                ),
                "cumulative_internal_iterations": int(cumulative_internal_iterations),
                "optimizer_function_evaluations": int(
                    cumulative_function_evaluations - previous_function_evaluations
                ),
                "cumulative_optimizer_function_evaluations": int(cumulative_function_evaluations),
                # PyTorch returns the objective from the first closure call,
                # so this is provenance only and never treated as the endpoint.
                "optimizer_returned_initial_objective": float(returned.detach()),
                "endpoint_float64_train": endpoint_64_train,
                "endpoint_float64_holdout": endpoint_64_holdout,
                "endpoint_float32_train": endpoint_32_train,
                "endpoint_float32_holdout": endpoint_32_holdout,
            }
            polish_records.append(record)
            previous_internal_iterations = cumulative_internal_iterations
            previous_function_evaluations = cumulative_function_evaluations
            if verbose:
                print(
                    f"L-BFGS outer {outer_step}/{polish.outer_steps} "
                    f"closures={closure_calls} internal={record['internal_iterations']} "
                    f"train64={endpoint_64_train['loss_total']:.6e} "
                    f"train32={endpoint_32_train['loss_total']:.6e}"
                )

        polish_elapsed = time.perf_counter() - polish_started
        model = model.cpu().float()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        final_train = _evaluate(model, x_train_32, y_train_32, polish.loss_weights)
        final_val = _evaluate(model, x_val_32, y_val_32, polish.loss_weights)
        allowed_increase = max(1e-9, 1e-6 * abs(adamw_train["loss_total"]))
        if final_train["loss_total"] > adamw_train["loss_total"] + allowed_increase:
            raise FloatingPointError(
                "the float32 L-BFGS endpoint increased the full-batch training objective "
                f"from {adamw_train['loss_total']:.12e} to {final_train['loss_total']:.12e}"
            )
        final_optimizer_state = [state for state in lbfgs.state.values() if isinstance(state, dict)]
        polish_summary = {
            "name": "LBFGS",
            "starts_with_fresh_optimizer_state": True,
            "device": polish.device,
            "dtype": polish.dtype,
            "outer_steps_requested": int(polish.outer_steps),
            "outer_steps_completed": len(polish_records),
            "learning_rate": float(polish.learning_rate),
            "max_iter": int(polish.max_iter),
            "max_eval": int(polish.max_eval),
            "history_size": int(polish.history_size),
            "tolerance_grad": float(polish.tolerance_grad),
            "tolerance_change": float(polish.tolerance_change),
            "line_search_fn": polish.line_search_fn,
            "loss_weights": [float(value) for value in polish.loss_weights],
            "trainable_components": list(polish.trainable_components),
            "closure_evaluations": int(total_closure_calls),
            "optimizer_function_evaluations": max(
                (int(state.get("func_evals", 0)) for state in final_optimizer_state),
                default=total_closure_calls,
            ),
            "internal_iterations": max(
                (int(state.get("n_iter", 0)) for state in final_optimizer_state),
                default=0,
            ),
            "duration_seconds": round(polish_elapsed, 2),
        }
        checkpoint_selection = "final_lbfgs_float32_endpoint"
        checkpoint_source = "lbfgs_float32_endpoint"
    else:
        final_train = dict(adamw_train)
        final_val = dict(adamw_val)
        checkpoint_selection = "final_epoch"
        checkpoint_source = "final_first_order_epoch"

    checkpoint_path, checkpoint_metadata_path = save_checkpoint(
        model,
        arch,
        output_root / "models",
        basename=basename,
    )
    elapsed = time.perf_counter() - started

    history: dict[str, Any] = {
        "schema_version": 2,
        "training_method": "curriculum_full_batch",
        "epoch_indexing": "list index 0 is global epoch 1",
        "loss_timing": {
            "train": "pre-update full-batch loss used for backpropagation",
            "validation": "post-update reporting-only full-batch loss",
            "stage_endpoints": "post-update full-batch evaluation",
            "polish": "post-L-BFGS-outer-step evaluations; validation remains reporting-only",
        },
        "stage_index": stage_index_history,
        "stage_name": stage_name_history,
        "learning_rate": learning_rate_history,
        "loss_weights": loss_weights_history,
        "train": train_history,
        "val": val_history,
        "stages": stage_records,
        "polish": {
            "config": polish.model_dump(mode="json"),
            "records": polish_records,
        }
        if polish is not None
        else None,
    }
    history_path = output_root / "logs" / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, separators=(",", ":")) + "\n")

    selected_index = global_epoch - 1
    summary: dict[str, Any] = {
        "schema_version": 2,
        "training_method": "curriculum_full_batch",
        "objective": CURRICULUM_OBJECTIVE,
        "optimizer": {
            "sequence": [
                "AdamW" if optimizer_config.name == "adamw" else "Adam",
                *(["LBFGS"] if polish is not None else []),
            ],
            "first_order": {
                "name": "AdamW" if optimizer_config.name == "adamw" else "Adam",
                "betas": [float(value) for value in optimizer_config.betas],
                "eps": float(optimizer_config.eps),
                "weight_decay": float(optimizer_config.weight_decay),
                "amsgrad": bool(optimizer_config.amsgrad),
                "foreach": bool(optimizer_config.foreach),
                "fused": bool(optimizer_config.fused),
                "state_continues_across_stages": True,
                "stage_learning_rates": [float(stage.learning_rate) for stage in stages],
                "updates_completed": int(global_epoch),
                "device": str(resolved_device),
                "dtype": "float32",
            },
            "polish": polish_summary,
        },
        "scheduler": None,
        "seed": int(seed),
        "device": str(resolved_device),
        "reproducibility": {
            "seeded_rngs": ["python", "numpy", "torch_cpu", "torch_cuda_if_available"],
            "resolved_backend": resolved_device.type,
            "deterministic_algorithms_enforced": bool(torch.are_deterministic_algorithms_enabled()),
            "bitwise_reproducible_across_backends_or_runtime_versions": False,
            "runtime": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": str(np.__version__),
                "torch": str(torch.__version__),
            },
        },
        "data": {
            "n_training_pairs": int(x_train.shape[0]),
            "n_validation_pairs": int(x_val.shape[0]),
            "high_dims": int(x_train.shape[1]),
            "dtype": "float32",
            "full_batch": True,
        },
        "arch": arch.model_dump(mode="json"),
        "model_initialized_by_helper": initialized_by_helper,
        "curriculum": stage_records,
        "loss_weights": [float(value) for value in final_weights],
        "n_epochs_run": int(global_epoch),
        "epochs_requested": int(global_epoch),
        "epochs_completed": int(global_epoch),
        "first_order_epochs_completed": int(global_epoch),
        "checkpoint_epoch": None if polish is not None else int(global_epoch),
        "best_epoch": None,
        "best_source": (
            "not_applicable_final_lbfgs_endpoint_selected"
            if polish is not None
            else "not_applicable_final_epoch_selected"
        ),
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_source": checkpoint_source,
        "validation_evaluated": True,
        "validation_used_for_optimization": False,
        "validation_used_for_checkpoint_selection": False,
        "early_stopping_used": False,
        "patience_used": False,
        "scheduler_used": False,
        "gradient_clipping_used": False,
        "best_weight_restoration_used": False,
        "train_duration_seconds": round(elapsed, 2),
        "train_duration_minutes": round(elapsed / 60.0, 4),
        "first_order_duration_seconds": round(first_order_elapsed, 2),
        "polish_duration_seconds": round(polish_elapsed, 2),
        "adamw_endpoint_train": adamw_train,
        "adamw_endpoint_holdout": adamw_val,
        "final_checkpoint_train": final_train,
        "selected_val": final_val,
        "final_holdout": final_val,
        "polish_delta_train": _loss_delta(final_train, adamw_train),
        "polish_delta_holdout": _loss_delta(final_val, adamw_val),
        "final_learning_rate": float(stages[-1].learning_rate),
        "train": {
            name: _history_stats(values, selected_index=selected_index)
            for name, values in train_history.items()
        },
        "val": {
            name: _history_stats(values, selected_index=selected_index)
            for name, values in val_history.items()
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path.relative_to(output_root)),
            "checkpoint_metadata": str(checkpoint_metadata_path.relative_to(output_root)),
            "adamw_checkpoint": str(adamw_checkpoint_path.relative_to(output_root)),
            "adamw_checkpoint_metadata": str(
                adamw_checkpoint_metadata_path.relative_to(output_root)
            ),
            "history": str(history_path.relative_to(output_root)),
        },
    }
    if polish is None:
        # Backward-compatible alias for no-polish curriculum consumers only.
        summary["final_epoch_train"] = final_train
    summary_path = output_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    final_losses_path = output_root / "final_losses.txt"
    final_losses_path.write_text(
        "\n".join(
            [
                "training_method: curriculum_full_batch",
                f"first_order_epochs_completed: {global_epoch}",
                f"checkpoint_selection: {checkpoint_selection}",
                f"checkpoint_source: {checkpoint_source}",
                *(f"adamw_train_{name}: {value:.9e}" for name, value in adamw_train.items()),
                *(f"adamw_holdout_{name}: {value:.9e}" for name, value in adamw_val.items()),
                *(f"train_{name}: {value:.9e}" for name, value in final_train.items()),
                *(f"val_{name}: {value:.9e}" for name, value in final_val.items()),
                f"lbfgs_closure_evaluations: {polish_summary['closure_evaluations'] if polish_summary else 0}",
                f"lbfgs_internal_iterations: {polish_summary['internal_iterations'] if polish_summary else 0}",
                "validation_used_for_optimization: false",
                "validation_used_for_checkpoint_selection: false",
                "scheduler_used: false",
                "early_stopping_used: false",
            ]
        )
        + "\n"
    )

    return CurriculumFullBatchResult(
        model=model,
        history=history,
        summary=summary,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        history_path=history_path,
        summary_path=summary_path,
        final_losses_path=final_losses_path,
    )
