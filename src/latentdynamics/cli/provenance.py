"""Run provenance manifests for pipeline cells."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from latentdynamics import __version__

from ..config import ExperimentConfig


def _canonical_config(cfg: ExperimentConfig) -> dict[str, Any]:
    return cfg.model_dump(mode="json")


def config_hash(cfg: ExperimentConfig) -> str:
    payload = json.dumps(_canonical_config(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_manifest(
    seed_cfg: ExperimentConfig,
    root_cfg: ExperimentConfig,
    *,
    cell_summary: dict[str, Any],
    stages: list[str],
    train_file: str,
) -> Path:
    """Write ``run_manifest.json`` beside a cell's outputs."""
    output_dir = seed_cfg.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = root_cfg.paths.data_dir / f"{train_file}.csv"
    scaler_path = root_cfg.paths.scaler_path(train_file)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "latentdynamics_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "requested_stages": stages,
        "cell": {
            "cell_index": cell_summary.get("cell_index"),
            "train_file": train_file,
            "seed": cell_summary.get("seed"),
            "output_dir": str(output_dir),
            "device": cell_summary.get("device"),
            "skipped_stages": cell_summary.get("skipped_stages", []),
        },
        "config_hash": config_hash(root_cfg),
        "config": _canonical_config(root_cfg),
        "artefacts": {
            "train_csv": str(train_csv),
            "train_csv_sha256": _file_sha256(train_csv),
            "scaler": str(scaler_path),
            "scaler_sha256": _file_sha256(scaler_path),
            "model_dir": str(seed_cfg.paths.model_dir),
            "morse_dir": str(seed_cfg.paths.morse_dir),
            "metrics": str(output_dir / "metrics.json"),
        },
    }

    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
