"""Train a LatentDynamicsAutoencoder from a config + scaled CSVs."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import ExperimentConfig
from ..models import build_autoencoder
from ..sampling import load_scaler
from ..training import Trainer, has_legacy_checkpoint, load_any_checkpoint


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


def _checkpoint_source_hashes(checkpoint_dir: Path) -> dict[str, str]:
    """Hash every checkpoint payload used for a warm start."""
    names = ("autoencoder.pt", "autoencoder.json", "encoder.pt", "dynamics.pt", "decoder.pt")
    hashes: dict[str, str] = {}
    for name in names:
        path = checkpoint_dir / name
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashes


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

    warm_start_dir = cfg.training.warm_start_checkpoint_dir
    initialization: dict[str, object]
    if warm_start_dir is None:
        model = build_autoencoder(cfg.arch)
        initialization = {"type": "fresh_random"}
    else:
        source_dir = Path(warm_start_dir)
        if source_dir.resolve() == model_dir.resolve():
            raise RuntimeError(
                "warm-start checkpoint directory resolves to the training output model "
                f"directory ({model_dir}); use a distinct output path"
            )
        model, source_arch = load_any_checkpoint(source_dir, arch=cfg.arch)
        if source_arch.model_dump() != cfg.arch.model_dump():
            raise ValueError(
                f"warm-start architecture from {source_dir} does not match the experiment config"
            )
        source_hashes = _checkpoint_source_hashes(source_dir)
        if not source_hashes:
            raise FileNotFoundError(f"no checkpoint payloads found in {source_dir}")
        initialization = {
            "type": "warm_start_weights",
            "checkpoint_dir": str(source_dir),
            "checkpoint_sha256": source_hashes,
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
        }
        if verbose:
            print(f"warm-started model weights from {source_dir}")
            print("optimizer and scheduler start fresh")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=cfg.training,
        arch_cfg=cfg.arch,
        device=device,
        verbose=verbose,
    )
    initial_val = trainer.evaluate_validation() if warm_start_dir is not None else None
    if initial_val is not None:
        trainer.register_baseline(initial_val)
    import time as _time

    _t0 = _time.perf_counter()
    history = trainer.fit()
    train_seconds = _time.perf_counter() - _t0

    trainer.save(output_root)

    best_epoch = trainer.best_epoch
    if best_epoch >= 0 and history.val["loss_total"]:
        selected_val = {key: history.val[key][best_epoch] for key in history.val}
        best_source = "training_epoch"
    elif best_epoch == -1 and initial_val is not None:
        selected_val = dict(initial_val)
        best_source = "warm_start_initial"
    else:
        selected_val = {key: float("nan") for key in history.val}
        best_source = "unavailable"
    losses_path = output_root / "final_losses.txt"
    lines = [f"best_epoch: {best_epoch}", f"best_source: {best_source}"] + [
        f"val_{key}: {value:.6e}" for key, value in selected_val.items()
    ]
    losses_path.write_text("\n".join(lines) + "\n")

    import json as _json

    summary: dict[str, object] = {
        "best_epoch": int(best_epoch),
        "best_source": best_source,
        "selected_val": {key: float(value) for key, value in selected_val.items()},
        "initialization": initialization,
        "loss_weights": list(cfg.training.loss_weights),
        "n_epochs_run": len(history.train["loss_total"]),
        "train_duration_seconds": round(train_seconds, 2),
        "train_duration_minutes": round(train_seconds / 60.0, 4),
    }
    if initial_val is not None:
        summary["initial_val"] = {key: float(value) for key, value in initial_val.items()}
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
