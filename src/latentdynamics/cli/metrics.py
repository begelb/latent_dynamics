"""Compute paper-specific metrics from saved artefacts.

Per-system dispatch on ``cfg.system.name``:

- ``coral``       -> :func:`unique_membership_metric` for a0, a1, r
- ``leslie3d``    -> tau-bar tolerance vs max semiconjugacy error
- otherwise       -> no metric (returns ``{}``)

All inputs are read from disk (state_dict + JSON sidecar OR legacy 3-file
format, plus the saved ``MG/morse_graph`` DOT and ``MG/morse_sets`` CSV);
nothing is recomputed by CMGDB.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from ..analysis import (
    MorseSet,
    check_unique_membership,
    compute_max_semiconjugacy_error,
    compute_min_boundary_separation,
)
from ..config import ExperimentConfig
from ..systems import build_system
from ..systems.base import DiscreteMap
from ..systems.coral import RedCoralModel
from ..training import has_legacy_checkpoint, has_new_checkpoint, load_any_checkpoint


def metrics_stage(
    seed_cfg: ExperimentConfig,
    cfg: ExperimentConfig,
    *,
    train_file: str = "train",
    verbose: bool = True,
) -> dict:
    """Dispatch to a system-specific metric; persist the result."""
    name = seed_cfg.system.name
    if name == "coral":
        result = _coral_metrics(seed_cfg, cfg, train_file=train_file)
    elif name == "leslie3d":
        result = _leslie3d_metrics(seed_cfg, cfg, train_file=train_file)
    else:
        result = {}

    out_path = seed_cfg.paths.output_dir / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    if verbose:
        print(f"metrics -> {out_path}: {result}")
    return result


def _model_dir(seed_cfg: ExperimentConfig) -> Path:
    return seed_cfg.paths.output_dir / "models"


def _coral_metrics(seed_cfg: ExperimentConfig, cfg: ExperimentConfig, *, train_file: str) -> dict:
    morse_sets_path = seed_cfg.paths.morse_dir / "morse_sets"
    morse_graph_path = seed_cfg.paths.morse_dir / "morse_graph"
    if not morse_sets_path.exists() or not morse_graph_path.exists():
        return {"error": "missing morse_sets or morse_graph file"}
    if morse_sets_path.stat().st_size == 0 or morse_graph_path.stat().st_size == 0:
        return {"error": "empty morse_sets or morse_graph file"}

    model_dir = _model_dir(seed_cfg)
    if not (has_legacy_checkpoint(model_dir) or has_new_checkpoint(model_dir)):
        return {"error": f"missing checkpoint at {model_dir}"}

    model, _arch = load_any_checkpoint(model_dir, arch=seed_cfg.arch)
    scaler = joblib.load(cfg.paths.scaler_path(train_file))

    labels, metrics = check_unique_membership(
        encoder=model.encoder,
        scaler=scaler,
        morse_sets_path=morse_sets_path,
        morse_graph_path=morse_graph_path,
        fixed_points=RedCoralModel.FIXED_POINTS,
        device=torch.device("cpu"),
    )
    return {"labels": labels, "metrics": metrics}


def _leslie3d_metrics(seed_cfg: ExperimentConfig, cfg: ExperimentConfig, *, train_file: str) -> dict:
    morse_sets_path = seed_cfg.paths.morse_dir / "morse_sets"
    if not morse_sets_path.exists():
        return {"error": "missing morse_sets file"}
    if morse_sets_path.stat().st_size == 0:
        return {"error": "empty morse_sets file"}

    model_dir = _model_dir(seed_cfg)
    if not (has_legacy_checkpoint(model_dir) or has_new_checkpoint(model_dir)):
        return {"error": f"missing checkpoint at {model_dir}"}

    model, _arch = load_any_checkpoint(model_dir, arch=seed_cfg.arch)
    device = torch.device("cpu")
    model.to(device)
    scaler = joblib.load(cfg.paths.scaler_path(train_file))

    target_label = 0  # paper Sec. 1.117 names the suspect Morse node 0
    morse_set = MorseSet(morse_sets_path, label=target_label)

    @torch.no_grad()
    def _g(verts):
        x = torch.as_tensor(verts, dtype=torch.float32, device=device)
        return model.latent_map(x).cpu().numpy()

    tau_bar = compute_min_boundary_separation(morse_set, _g)

    system = build_system(cfg.system.name, cfg.system.params)
    if not isinstance(system, DiscreteMap):
        return {"tau_bar": float(tau_bar), "warning": "system is not a DiscreteMap; skipped semiconjugacy error"}

    rng = np.random.default_rng(0)
    pts = rng.uniform(system.lower_bounds, system.upper_bounds, size=(256, system.dim))
    iterated = pts.copy()
    for _ in range(min(cfg.data.n_iterations, 20)):
        iterated = system.step(iterated)

    pts_scaled = scaler.transform(iterated)
    next_scaled = scaler.transform(system.step(iterated))
    max_err = compute_max_semiconjugacy_error(
        encoder=model.encoder,
        latent_map=model.latent_map,
        points_in_block=pts_scaled,
        next_points_true=next_scaled,
        device=device,
    )

    return {
        "target_label": int(target_label),
        "tau_bar": float(tau_bar),
        "max_semiconjugacy_error": float(max_err),
        "is_spurious_attractor": bool(max_err > tau_bar),
    }
