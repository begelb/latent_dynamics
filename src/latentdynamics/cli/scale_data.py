"""Construct and persist an ambient-coordinate scaler for a training CSV."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..sampling import (
    fit_fixed_bounds_scaler,
    fit_identity_scaler,
    fit_minmax_scaler,
    save_scaler,
)
from ..systems import build_system


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scaler_metadata_path(scaler_path: Path) -> Path:
    return scaler_path.with_name("scaler_metadata.json")


def _scaling_parameters(cfg: ExperimentConfig) -> dict:
    """Configuration values that determine the scaler's numeric map."""
    if cfg.data.scaling != "fixed_bounds":
        return {}
    system = build_system(cfg.system.name, cfg.system.params)
    return {
        "lower_bounds": system.lower_bounds.tolist(),
        "upper_bounds": system.upper_bounds.tolist(),
        "scaling_epsilon": cfg.data.scaling_epsilon,
    }


def scaler_metadata(
    cfg: ExperimentConfig,
    train_file: str,
    csv_path: Path,
) -> dict:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "train_file": train_file,
        "train_csv": str(csv_path),
        "train_csv_sha256": file_sha256(csv_path),
        "scaling": cfg.data.scaling,
        "high_dims": cfg.arch.high_dims,
        **_scaling_parameters(cfg),
    }


def scaler_is_current(cfg: ExperimentConfig, train_file: str) -> bool:
    scaler_path = cfg.paths.scaler_path(train_file)
    meta_path = scaler_metadata_path(scaler_path)
    csv_path = cfg.paths.data_dir / f"{train_file}.csv"
    if not scaler_path.is_file() or scaler_path.stat().st_size == 0:
        return False
    if not meta_path.is_file() or meta_path.stat().st_size == 0:
        return False
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    expected_parameters = _scaling_parameters(cfg)
    return (
        meta.get("train_file") == train_file
        and meta.get("train_csv_sha256") == file_sha256(csv_path)
        and meta.get("scaling") == cfg.data.scaling
        and meta.get("high_dims") == cfg.arch.high_dims
        and all(meta.get(key) == value for key, value in expected_parameters.items())
    )


def run(cfg: ExperimentConfig, train_file: str = "train", *, verbose: bool = True) -> None:
    if cfg.paths.scaler_read_only:
        raise RuntimeError(
            "config sets paths.scaler_read_only=true; refusing to fit or overwrite "
            f"the protected scaler at {cfg.paths.scaler_path(train_file)}. "
            "Omit the scale stage and use the existing scaler."
        )
    high_dims = cfg.arch.high_dims
    csv_path = cfg.paths.data_dir / f"{train_file}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"training CSV not found: {csv_path}")

    train_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    x = train_data[:, :high_dims]
    y = train_data[:, high_dims:]
    if cfg.data.scaling == "none":
        scaler = fit_identity_scaler(high_dims)
    elif cfg.data.scaling == "fixed_bounds":
        system = build_system(cfg.system.name, cfg.system.params)
        if system.dim != high_dims:
            raise ValueError(
                f"fixed-bounds scaler has {system.dim} system dimensions, "
                f"but arch.high_dims is {high_dims}"
            )
        scaler = fit_fixed_bounds_scaler(
            system.lower_bounds,
            system.upper_bounds,
            epsilon=cfg.data.scaling_epsilon,
        )
    else:
        scaler = fit_minmax_scaler(x, y)
    if verbose:
        print(
            f"fitted {cfg.data.scaling} scaler on "
            f"{x.shape[0] + y.shape[0]} samples ({high_dims} dims)"
        )

    scaler_path = cfg.paths.scaler_path(train_file)
    save_scaler(scaler, scaler_path)
    scaler_metadata_path(scaler_path).write_text(
        json.dumps(scaler_metadata(cfg, train_file, csv_path), indent=2) + "\n"
    )
    if verbose:
        print(f"wrote {scaler_path}")
