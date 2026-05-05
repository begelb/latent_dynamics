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
    x_te, y_te = _load_pair(cfg.paths.data_dir / "test.csv", high)

    scaler = load_scaler(cfg.paths.scaler_path(train_file))
    x_tr, y_tr = scaler.transform(x_tr), scaler.transform(y_tr)
    x_te, y_te = scaler.transform(x_te), scaler.transform(y_te)

    train_ds = TensorDataset(
        torch.tensor(x_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(x_te, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.float32),
    )

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return (
        DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, generator=generator),
        DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False),
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

    train_loader, test_loader = _build_loaders(cfg, train_file, seed)
    model = build_autoencoder(cfg.arch)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        training_cfg=cfg.training,
        arch_cfg=cfg.arch,
        device=device,
        verbose=verbose,
    )
    history = trainer.fit()

    trainer.save(output_root)

    final = {k: history.train[k][-1] if history.train[k] else float("nan") for k in history.train}
    losses_path = output_root / "final_losses.txt"
    losses_path.write_text("\n".join(f"{k}: {v:.6e}" for k, v in final.items()) + "\n")
    if verbose:
        print(f"checkpoint and logs written to {output_root}")
