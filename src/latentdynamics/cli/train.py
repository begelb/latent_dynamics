"""Train a LatentDynamicsAutoencoder from a config + scaled CSVs."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import ExperimentConfig
from ..models import build_autoencoder
from ..sampling import load_scaler
from ..training import Trainer, has_legacy_checkpoint


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _load_pair(csv: Path, high_dims: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    return data[:, :high_dims], data[:, high_dims:]


def _build_loaders(
    cfg: ExperimentConfig,
    train_file: str,
    seed: int | None,
) -> tuple[DataLoader, DataLoader]:
    high = cfg.arch.high_dims
    x_tr, y_tr = _load_pair(cfg.paths.data_dir / f"{train_file}.csv", high)
    x_va, y_va = _load_pair(cfg.paths.val_csv(), high)

    scaler = load_scaler(cfg.paths.scaler_path(train_file))
    x_tr, y_tr = scaler.transform(x_tr), scaler.transform(y_tr)
    x_va, y_va = scaler.transform(x_va), scaler.transform(y_va)

    train_ds = TensorDataset(
        torch.tensor(x_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_va, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return (
        DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, generator=generator),
        DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False),
    )


def run(
    cfg: ExperimentConfig,
    *,
    train_file: str = "train",
    seed: int | None = None,
    output_subdir: str | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    force_overwrite: bool = False,
) -> None:
    """Train the autoencoder for one (config, seed) combination."""
    output_root = cfg.paths.output_dir
    if output_subdir is not None:
        output_root = output_root / output_subdir

    model_dir = output_root / "models"
    if has_legacy_checkpoint(model_dir) and not force_overwrite:
        raise RuntimeError(
            f"legacy 3-file checkpoint detected at {model_dir} "
            f"(encoder.pt + dynamics.pt + decoder.pt). Refusing to write a "
            f"new-format checkpoint next to it (would orphan the legacy run). "
            f"Pass --force-overwrite to proceed."
        )

    if seed is not None:
        _seed_everything(seed)

    train_loader, val_loader = _build_loaders(cfg, train_file, seed)
    model = build_autoencoder(cfg.arch)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=cfg.training,
        arch_cfg=cfg.arch,
        device=device,
        verbose=verbose,
    )
    import time as _time

    _t0 = _time.perf_counter()
    history = trainer.fit()
    train_seconds = _time.perf_counter() - _t0

    trainer.save(output_root)

    best_epoch = trainer.best_epoch
    best_val = (
        {k: history.val[k][best_epoch] for k in history.val}
        if best_epoch >= 0 and history.val["loss_total"]
        else {k: float("nan") for k in history.val}
    )
    losses_path = output_root / "final_losses.txt"
    lines = [f"best_epoch: {best_epoch}"] + [f"val_{k}: {v:.6e}" for k, v in best_val.items()]
    losses_path.write_text("\n".join(lines) + "\n")

    import json as _json

    summary: dict[str, object] = {
        "best_epoch": int(best_epoch),
        "loss_weights": list(cfg.training.loss_weights),
        "n_epochs_run": len(history.train["loss_total"]),
        "train_duration_seconds": round(train_seconds, 2),
        "train_duration_minutes": round(train_seconds / 60.0, 4),
    }
    for split_name, hist in (("train", history.train), ("val", history.val)):
        per_loss: dict[str, dict[str, float]] = {}
        for key, series in hist.items():
            if not series:
                continue
            arr = [float(x) for x in series]
            per_loss[key] = {
                "mean": sum(arr) / len(arr),
                "min": min(arr),
                "max": max(arr),
                "final": arr[-1],
                "best_epoch_value": (
                    float(series[best_epoch]) if 0 <= best_epoch < len(series) else float("nan")
                ),
            }
        summary[split_name] = per_loss
    (output_root / "training_summary.json").write_text(_json.dumps(summary, indent=2))

    if verbose:
        print(f"trained {summary['n_epochs_run']} epochs in {train_seconds / 60:.2f} min")
        print(f"checkpoint and logs written to {output_root}")
