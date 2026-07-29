"""Run a long, fixed-horizon Chafee--Infante gradient-accumulation experiment.

The exact 30,000 unscaled Marcio pairs are processed in 1,024-row chunks, but
Adam is stepped exactly once per complete data pass.  Each chunk's mean loss is
weighted by its row count, so the accumulated gradient is the full-data mean
gradient even for the final 304-row chunk.

The run is fixed at 20,000 optimizer updates with no validation stopping.
Inference checkpoints are saved every 1,000 updates, and complete resumable
training state is committed every 250 updates at safe accumulation boundaries.
No basin artifact is read by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import compare_chafee_d1_minibatch as common
from latentdynamics.config import ArchConfig
from latentdynamics.models import LatentDynamicsAutoencoder, build_autoencoder
from latentdynamics.training import load_checkpoint, save_checkpoint

DEFAULT_OUTPUT = (
    common.CANONICAL_RUN.parent / "seed_0_gradaccum_b1024_epoch_20000"
)
OBJECTIVE = common.OBJECTIVE


@dataclass(frozen=True)
class GradientAccumulationSettings:
    seed: int = 0
    microbatch_size: int = 1_024
    effective_batch_size: int = 30_000
    learning_rate: float = 0.003
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    epochs: int = 20_000
    scheduler_factor: float = 0.5
    scheduler_patience: int = 100
    scheduler_threshold: float = 1e-4
    scheduler_min_lr: float = 1e-6
    resume_interval: int = 250
    milestone_interval: int = 1_000

    def __post_init__(self) -> None:
        if self.microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")
        if self.effective_batch_size < 1:
            raise ValueError("effective_batch_size must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.resume_interval < 1 or self.milestone_interval < 1:
            raise ValueError("checkpoint intervals must be positive")
        if self.epochs % self.resume_interval:
            raise ValueError("epochs must be divisible by resume_interval")
        if self.epochs % self.milestone_interval:
            raise ValueError("epochs must be divisible by milestone_interval")


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _to_cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


def _optimizer_to_device(optimizer: Adam, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _accumulated_epoch(
    model: nn.Module,
    x_cpu: Tensor,
    y_cpu: Tensor,
    *,
    optimizer: Adam,
    microbatch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    """Apply one exact full-mean Adam update via row-weighted chunks."""

    if x_cpu.shape != y_cpu.shape or x_cpu.ndim != 2:
        raise ValueError("x/y must be matching rank-2 tensors")
    n_rows = int(x_cpu.shape[0])
    if n_rows < 1:
        raise ValueError("x/y must not be empty")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums = {
        "loss_reconstruction": 0.0,
        "loss_prediction": 0.0,
        "loss_total": 0.0,
    }
    microbatches = 0
    consumed = 0
    for start in range(0, n_rows, microbatch_size):
        stop = min(start + microbatch_size, n_rows)
        x = x_cpu[start:stop].to(device)
        y = y_cpu[start:stop].to(device)
        count = int(x.shape[0])
        reconstruction, prediction, total = common._two_term_losses(model, x, y)
        if not torch.isfinite(total).item():
            raise FloatingPointError(
                f"non-finite accumulated objective: {float(total.detach())}"
            )
        (total * (count / n_rows)).backward()
        sums["loss_reconstruction"] += float(reconstruction.detach()) * count
        sums["loss_prediction"] += float(prediction.detach()) * count
        sums["loss_total"] += float(total.detach()) * count
        consumed += count
        microbatches += 1

    if consumed != n_rows:
        raise RuntimeError(f"accumulation consumed {consumed} of {n_rows} rows")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return ({key: value / n_rows for key, value in sums.items()}, microbatches)


def _cpu_model(
    arch: ArchConfig,
    state: dict[str, Tensor],
) -> LatentDynamicsAutoencoder:
    model = build_autoencoder(arch)
    model.load_state_dict(state)
    return model


def _atomic_torch_save(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _rng_state() -> dict[str, Any]:
    mps_state = None
    get_mps_state = getattr(torch.mps, "get_rng_state", None)
    if torch.backends.mps.is_available() and callable(get_mps_state):
        mps_state = get_mps_state()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_mps": mps_state,
    }


def _restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    mps_state = payload.get("torch_mps")
    set_mps_state = getattr(torch.mps, "set_rng_state", None)
    if mps_state is not None and callable(set_mps_state):
        set_mps_state(mps_state)


def _save_milestone(
    *,
    model_state: dict[str, Tensor],
    arch: ArchConfig,
    output_dir: Path,
    epoch: int,
    metrics: dict[str, float],
    learning_rate: float,
    run_plan_sha256: str,
) -> dict[str, Any]:
    milestone = output_dir / "milestones" / f"epoch_{epoch:05d}"
    model = _cpu_model(arch, model_state)
    checkpoint, sidecar = save_checkpoint(model, arch, milestone / "models")
    payload = {
        "schema_version": 1,
        "epoch": epoch,
        "optimizer_updates": epoch,
        "run_plan_sha256": run_plan_sha256,
        "train": metrics,
        "learning_rate": learning_rate,
        "checkpoint": {
            "path": str(checkpoint.relative_to(output_dir)),
            "sha256": common._sha256(checkpoint),
            "sidecar_path": str(sidecar.relative_to(output_dir)),
            "sidecar_sha256": common._sha256(sidecar),
        },
        "basin_artifacts_accessed": False,
    }
    common._write_json(milestone / "manifest.json", payload)
    return payload


def _save_resume_state(
    *,
    model: nn.Module,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    output_dir: Path,
    epoch: int,
    history: dict[str, list[float]],
    elapsed_seconds: float,
    run_plan_sha256: str,
    microbatches_completed: int,
) -> Path:
    if model.training and any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("resume state must be saved after gradients are cleared")
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    generation = output_dir / "resume" / f"epoch_{epoch:05d}"
    if generation.exists():
        raise FileExistsError(f"resume generation already exists: {generation}")
    generation.mkdir(parents=True)
    state_path = _atomic_torch_save(
        {
            "schema_version": 1,
            "epoch": epoch,
            "model_state": _cpu_state_dict(model),
            "optimizer_state": _to_cpu_tree(optimizer.state_dict()),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "elapsed_seconds": elapsed_seconds,
            "microbatches_completed": microbatches_completed,
            "rng_state": _rng_state(),
            "run_plan_sha256": run_plan_sha256,
        },
        generation / "state.pt",
    )
    manifest = {
        "schema_version": 1,
        "epoch": epoch,
        "run_plan_sha256": run_plan_sha256,
        "state": {
            "path": str(state_path.relative_to(output_dir)),
            "size_bytes": state_path.stat().st_size,
            "sha256": common._sha256(state_path),
        },
        "safe_boundary": (
            "after optimizer.step, scheduler.step, and optimizer.zero_grad"
        ),
    }
    common._write_json(generation / "manifest.json", manifest)
    common._write_json(
        output_dir / "resume" / "latest.json",
        {
            "schema_version": 1,
            "epoch": epoch,
            "generation_manifest": str(
                (generation / "manifest.json").relative_to(output_dir)
            ),
            "generation_manifest_sha256": common._sha256(
                generation / "manifest.json"
            ),
        },
    )
    return state_path


def _load_resume_state(
    output_dir: Path,
    *,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    latest = json.loads(
        (output_dir / "resume" / "latest.json").read_text(encoding="utf-8")
    )
    manifest_path = output_dir / latest["generation_manifest"]
    if common._sha256(manifest_path) != latest["generation_manifest_sha256"]:
        raise ValueError("latest resume manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_plan_sha256") != expected_plan_sha256:
        raise ValueError("resume generation run-plan hash mismatch")
    state_path = output_dir / manifest["state"]["path"]
    if common._sha256(state_path) != manifest["state"]["sha256"]:
        raise ValueError("resume state hash mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("run_plan_sha256") != expected_plan_sha256:
        raise ValueError("resume state run-plan hash mismatch")
    return state


def _new_run_plan(
    *,
    output_dir: Path,
    arch: ArchConfig,
    settings: GradientAccumulationSettings,
    train_hash: str,
    canonical_hash: str,
) -> dict[str, Any]:
    script = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "status": "frozen_before_training",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": output_dir.name,
        "purpose": "long fixed-horizon 1-D gradient-accumulation experiment",
        "training_script": {
            "path": str(script),
            "sha256": common._sha256(script),
        },
        "architecture": arch.model_dump(mode="json"),
        "architecture_source": {
            "path": str(
                (common.CANONICAL_RUN / "models" / "autoencoder.pt").resolve()
            ),
            "sha256": canonical_hash,
            "weights_reused": False,
        },
        "data": {
            "path": str(common.TRAIN_DATA.resolve()),
            "sha256": train_hash,
            "shape": [common.TRAINING_ROWS, 2 * common.HIGH_DIMENSION],
            "scaling": "none",
            "shuffle": False,
            "drop_last": False,
        },
        "objective": {
            "formula": OBJECTIVE,
            "weights": [1.0, 1.0, 0.0],
        },
        "settings": asdict(settings),
        "accumulation": {
            "microbatches_per_update": math.ceil(
                settings.effective_batch_size / settings.microbatch_size
            ),
            "last_microbatch_rows": (
                settings.effective_batch_size % settings.microbatch_size
            ),
            "loss_weighting": (
                "microbatch mean objective multiplied by "
                "microbatch_rows / effective_batch_size"
            ),
            "optimizer_steps_per_epoch": 1,
        },
        "milestone_epochs": list(
            range(
                settings.milestone_interval,
                settings.epochs + 1,
                settings.milestone_interval,
            )
        ),
        "early_stopping": False,
        "validation_used": False,
        "basin_or_cmgdb_inputs_allowed": False,
    }


def _prepare(
    *,
    output_dir: Path,
    resume: bool,
) -> tuple[
    ArchConfig,
    Tensor,
    Tensor,
    GradientAccumulationSettings,
    torch.device,
    str,
]:
    train_hash = common._checked_sha256(
        common.TRAIN_DATA,
        common.TRAIN_DATA_SHA256,
        description="exact Marcio training data",
    )
    canonical_checkpoint = common.CANONICAL_RUN / "models" / "autoencoder.pt"
    canonical_hash = common._checked_sha256(
        canonical_checkpoint,
        common.CANONICAL_CHECKPOINT_SHA256,
        description="canonical d=1 checkpoint",
    )
    _, arch = load_checkpoint(common.CANONICAL_RUN / "models", map_location="cpu")
    x, y = common._load_pairs(
        common.TRAIN_DATA,
        rows=common.TRAINING_ROWS,
        high_dimension=common.HIGH_DIMENSION,
        skiprows=0,
    )
    settings = GradientAccumulationSettings()
    device = common._resolve_device("mps")
    if resume:
        plan_path = output_dir / "run_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("settings") != asdict(settings):
            raise ValueError("frozen run settings differ from current settings")
        if plan.get("training_script", {}).get("sha256") != common._sha256(
            Path(__file__).resolve()
        ):
            raise ValueError("training script changed since the run plan was frozen")
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        plan = _new_run_plan(
            output_dir=output_dir,
            arch=arch,
            settings=settings,
            train_hash=train_hash,
            canonical_hash=canonical_hash,
        )
        common._write_json(output_dir / "run_plan.json", plan)
    plan_hash = common._sha256(output_dir / "run_plan.json")
    return (
        arch,
        torch.as_tensor(x, dtype=torch.float32).contiguous(),
        torch.as_tensor(y, dtype=torch.float32).contiguous(),
        settings,
        device,
        plan_hash,
    )


def run_training(
    *,
    output_dir: Path,
    resume: bool,
    verbose: bool,
) -> dict[str, Any]:
    output = output_dir.resolve()
    arch, x_cpu, y_cpu, settings, device, plan_hash = _prepare(
        output_dir=output,
        resume=resume,
    )
    common._seed_everything(settings.seed)
    model = build_autoencoder(arch).to(device)
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
    history = {
        "loss_reconstruction": [],
        "loss_prediction": [],
        "loss_total": [],
        "learning_rate": [],
    }
    start_epoch = 1
    elapsed_prior = 0.0
    microbatches_completed = 0
    if resume:
        state = _load_resume_state(output, expected_plan_sha256=plan_hash)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        _optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        elapsed_prior = float(state["elapsed_seconds"])
        microbatches_completed = int(state["microbatches_completed"])
        start_epoch = int(state["epoch"]) + 1
        _restore_rng_state(state["rng_state"])
        if start_epoch > settings.epochs:
            raise ValueError("run is already complete")

    started = time.perf_counter()
    for epoch in range(start_epoch, settings.epochs + 1):
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        metrics, microbatches = _accumulated_epoch(
            model,
            x_cpu,
            y_cpu,
            optimizer=optimizer,
            microbatch_size=settings.microbatch_size,
            device=device,
        )
        scheduler.step(metrics["loss_total"])
        history["loss_reconstruction"].append(metrics["loss_reconstruction"])
        history["loss_prediction"].append(metrics["loss_prediction"])
        history["loss_total"].append(metrics["loss_total"])
        history["learning_rate"].append(learning_rate_used)
        microbatches_completed += microbatches
        elapsed = elapsed_prior + time.perf_counter() - started

        if verbose and (epoch == start_epoch or epoch % 100 == 0):
            print(
                f"epoch={epoch:05d}/{settings.epochs} "
                f"loss={metrics['loss_total']:.6e} "
                f"lr={learning_rate_used:.3e} "
                f"microbatches={microbatches_completed} "
                f"elapsed_min={elapsed / 60.0:.2f}",
                flush=True,
            )
        if epoch % 100 == 0:
            common._write_json(
                output / "progress.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "epoch": epoch,
                    "target_epoch": settings.epochs,
                    "train": metrics,
                    "learning_rate": learning_rate_used,
                    "elapsed_seconds": elapsed,
                    "run_plan_sha256": plan_hash,
                },
            )

        model_state: dict[str, Tensor] | None = None
        if epoch % settings.milestone_interval == 0:
            model_state = _cpu_state_dict(model)
            _save_milestone(
                model_state=model_state,
                arch=arch,
                output_dir=output,
                epoch=epoch,
                metrics=metrics,
                learning_rate=learning_rate_used,
                run_plan_sha256=plan_hash,
            )
        if epoch % settings.resume_interval == 0:
            _save_resume_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                output_dir=output,
                epoch=epoch,
                history=history,
                elapsed_seconds=elapsed,
                run_plan_sha256=plan_hash,
                microbatches_completed=microbatches_completed,
            )
            common._write_json(
                output / "logs" / "history.json",
                {
                    "schema_version": 1,
                    "training_method": "marcio_gradient_accumulation",
                    "epoch_indexing": "list index 0 is epoch 1",
                    "train": history,
                },
            )

    elapsed = elapsed_prior + time.perf_counter() - started
    final_state = _cpu_state_dict(model)
    final_model = _cpu_model(arch, final_state)
    final_checkpoint, final_sidecar = save_checkpoint(
        final_model,
        arch,
        output / "models",
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "training_method": "marcio_gradient_accumulation",
        "objective": OBJECTIVE,
        "settings": asdict(settings),
        "run_plan_sha256": plan_hash,
        "epochs_requested": settings.epochs,
        "epochs_completed": settings.epochs,
        "optimizer_updates": settings.epochs,
        "microbatches_completed": microbatches_completed,
        "example_presentations": settings.epochs * int(x_cpu.shape[0]),
        "early_stopping_used": False,
        "validation_used": False,
        "duration_seconds": elapsed,
        "duration_minutes": elapsed / 60.0,
        "final_epoch_train": {
            key: history[key][-1]
            for key in (
                "loss_reconstruction",
                "loss_prediction",
                "loss_total",
            )
        },
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "runtime": common._runtime_metadata(device),
        "artifacts": {
            "checkpoint": str(final_checkpoint.relative_to(output)),
            "checkpoint_sha256": common._sha256(final_checkpoint),
            "sidecar": str(final_sidecar.relative_to(output)),
            "sidecar_sha256": common._sha256(final_sidecar),
            "history": "logs/history.json",
            "milestones": "milestones/",
            "latest_resume": "resume/latest.json",
        },
        "basin_artifacts_accessed": False,
    }
    common._write_json(output / "training_summary.json", summary)
    completion = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "run_plan_sha256": plan_hash,
        "training_summary_sha256": common._sha256(
            output / "training_summary.json"
        ),
        "final_checkpoint_sha256": common._sha256(final_checkpoint),
        "epochs_completed": settings.epochs,
    }
    common._write_json(output / "completion.json", completion)
    common._write_json(
        output / "progress.json",
        {
            "schema_version": 1,
            "status": "complete",
            "epoch": settings.epochs,
            "target_epoch": settings.epochs,
            "elapsed_seconds": elapsed,
            "run_plan_sha256": plan_hash,
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_training(
        output_dir=args.output_dir,
        resume=args.resume,
        verbose=not args.quiet,
    )
    print(
        f"training_complete epochs={summary['epochs_completed']} "
        f"output={args.output_dir.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
