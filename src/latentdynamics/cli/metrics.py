"""Compute paper-specific metrics from saved artifacts.

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


def _morse_set_contains(points: np.ndarray, morse_set: MorseSet) -> np.ndarray:
    """Boolean mask for latent ``points`` lying in any box of ``morse_set``."""
    mask = np.zeros(points.shape[0], dtype=bool)
    for box in morse_set:
        in_box = (
            (box.lower_x <= points[:, 0])
            & (points[:, 0] <= box.upper_x)
            & (box.lower_y <= points[:, 1])
            & (points[:, 1] <= box.upper_y)
        )
        mask |= in_box
    return mask


@torch.no_grad()
def _filter_samples_in_target_morse_set(
    *,
    encoder: torch.nn.Module,
    morse_set: MorseSet,
    points_scaled: np.ndarray,
    next_scaled: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep high-dimensional samples whose encoded current point lies in ``morse_set``."""
    encoder.eval()
    z = encoder(torch.as_tensor(points_scaled, dtype=torch.float32, device=device))
    encoded = z.cpu().numpy()
    if encoded.shape[1] != 2:
        raise ValueError(
            f"Leslie tolerance metric expects a 2D latent space; got {encoded.shape[1]}D"
        )
    mask = _morse_set_contains(encoded, morse_set)
    return points_scaled[mask], next_scaled[mask]


def metrics_stage(
    seed_cfg: ExperimentConfig,
    cfg: ExperimentConfig,
    *,
    train_file: str = "train",
    verbose: bool = True,
    out_dir: Path | None = None,
) -> dict:
    """Dispatch to a system-specific metric; persist the result.

    Reads model checkpoint, scaler, and Morse artifacts from the source paths
    in ``seed_cfg``/``cfg``. ``metrics.json`` is written to ``out_dir`` when
    provided (replay-routing) or to ``seed_cfg.paths.output_dir`` otherwise.
    """
    name = seed_cfg.system.name
    if name == "coral":
        result = _coral_metrics(seed_cfg, cfg, train_file=train_file)
    else:
        # Generic 2D-latent path: tau_bar + max-semiconjugacy-error per minimal
        # Morse set. Subsumes the old hardcoded leslie3d target_label=0 logic.
        if seed_cfg.arch.low_dims == 2:
            result = _per_minimal_tolerance_metrics(seed_cfg, cfg, train_file=train_file)
        else:
            result = {}

    # System-agnostic faithfulness flag: every attractor-type Morse set (stable
    # Conley index) must be a minimal node. Non-minimal ones signal a too-coarse
    # subdivision (spurious outgoing edges) -- an unfaithful Morse graph.
    if isinstance(result, dict):
        result["morse_graph_consistency"] = _morse_graph_consistency(
            seed_cfg.paths.morse_dir / "morse_graph"
        )

    write_root = Path(out_dir) if out_dir is not None else seed_cfg.paths.output_dir
    out_path = write_root / "metrics.json"
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


def _minimal_morse_labels(morse_graph_dot: Path) -> list[int]:
    """Parse a CMGDB DOT file and return Morse-graph nodes with no outgoing edges."""
    import re

    text = morse_graph_dot.read_text()
    node_ids = {int(m.group(1)) for m in re.finditer(r"^(\d+)\s+\[label", text, re.M)}
    edge_pairs = re.findall(r"^(\d+)\s*->\s*(\d+);", text, re.M)
    has_out = {int(s) for s, _ in edge_pairs}
    return sorted(n for n in node_ids if n not in has_out)


def _morse_graph_consistency(morse_graph_dot: Path) -> dict:
    """Flag attractor-type Morse sets that are not minimal (subdivision too coarse).

    The Conley index carries dynamical type: a nontrivial degree-0 component
    (``x-1`` stable fixed point, ``x^n-1`` stable periodic orbit, ``x-1, x-1``
    stable set) marks an *attracting* Morse set, which must therefore be a
    minimal node (a sink). An attractor-type set with outgoing edges indicates
    a spurious edge from too-coarse subdivision -> raise ``subdiv_init``. Also
    reports trivial-index ``(0,0,0)`` sets (raise ``subdiv_max`` to dissolve).
    """
    import re

    if not morse_graph_dot.exists() or morse_graph_dot.stat().st_size == 0:
        return {}
    text = morse_graph_dot.read_text()
    idx = {
        int(m.group(1)): m.group(2).strip()
        for m in re.finditer(r'^(\d+)\s+\[label="\d+\s*:\s*\(([^)]*)\)"', text, re.M)
    }
    has_out = {int(s) for s, _ in re.findall(r"^(\d+)\s*->\s*(\d+);", text, re.M)}
    attractor_type = [n for n in idx if idx[n].split(",")[0].strip() not in ("0", "")]
    nonminimal = sorted(n for n in attractor_type if n in has_out)
    trivial = sorted(n for n, v in idx.items() if v.replace(" ", "") == "0,0,0")
    return {
        "n_morse_sets": len(idx),
        "n_minimal_attractors": len([n for n in attractor_type if n not in has_out]),
        "attractor_type_nonminimal": nonminimal,
        "n_trivial_index": len(trivial),
        "consistent": len(nonminimal) == 0,
    }


def _per_minimal_tolerance_metrics(
    seed_cfg: ExperimentConfig,
    cfg: ExperimentConfig,
    *,
    train_file: str,
) -> dict:
    """Compute tau_bar and max-semiconjugacy-error for every minimal Morse set.

    Writes one entry per minimal Morse node, keyed by its node id. Both
    quantities are bounds/estimates: ``tau_bar`` is the minimum distance from
    G(corner) to the Morse-set boundary, taken over the box corner vertices,
    and ``max_semiconjugacy_error`` is the sup over sampled high-dim points
    that encode into the block of ``||E(f(x)) - G(E(x))||``.
    """
    morse_sets_path = seed_cfg.paths.morse_dir / "morse_sets"
    morse_graph_path = seed_cfg.paths.morse_dir / "morse_graph"
    if not morse_sets_path.exists():
        return {"error": "missing morse_sets file"}
    if morse_sets_path.stat().st_size == 0:
        return {"error": "empty morse_sets file"}
    if not morse_graph_path.exists() or morse_graph_path.stat().st_size == 0:
        return {"error": "missing or empty morse_graph file"}

    model_dir = _model_dir(seed_cfg)
    if not (has_legacy_checkpoint(model_dir) or has_new_checkpoint(model_dir)):
        return {"error": f"missing checkpoint at {model_dir}"}

    model, _arch = load_any_checkpoint(model_dir, arch=seed_cfg.arch)
    device = torch.device("cpu")
    model.to(device)
    scaler = joblib.load(cfg.paths.scaler_path(train_file))

    @torch.no_grad()
    def _g(verts):
        x = torch.as_tensor(verts, dtype=torch.float32, device=device)
        return model.latent_map(x).cpu().numpy()

    minimal_labels = _minimal_morse_labels(morse_graph_path)

    # One pass of high-dim samples shared across minimal Morse sets.
    system = build_system(cfg.system.name, cfg.system.params)
    pts_scaled = next_scaled = None
    semiconj_supported = isinstance(system, DiscreteMap)
    if semiconj_supported:
        rng = np.random.default_rng(0)
        # 4096 samples is enough to populate small Morse sets without ballooning runtime.
        pts = rng.uniform(system.lower_bounds, system.upper_bounds, size=(4096, system.dim))
        iterated = pts.copy()
        for _ in range(min(cfg.data.n_iterations, 20)):
            iterated = system.step(iterated)
        pts_scaled = scaler.transform(iterated)
        next_scaled = scaler.transform(system.step(iterated))

    per_minimal: dict[str, dict] = {}
    for lbl in minimal_labels:
        ms = MorseSet(morse_sets_path, label=lbl)
        entry: dict[str, object] = {"n_boxes": len(ms)}
        if len(ms) == 0:
            entry["error"] = "label not present in morse_sets"
            per_minimal[str(lbl)] = entry
            continue
        tau_bar = float(compute_min_boundary_separation(ms, _g))
        entry["tau_bar"] = tau_bar
        if not semiconj_supported:
            entry["warning"] = "system is not a DiscreteMap; skipped semiconjugacy error"
            per_minimal[str(lbl)] = entry
            continue
        pts_in_block, next_in_block = _filter_samples_in_target_morse_set(
            encoder=model.encoder,
            morse_set=ms,
            points_scaled=pts_scaled,
            next_scaled=next_scaled,
            device=device,
        )
        entry["n_semiconjugacy_samples"] = int(pts_in_block.shape[0])
        if pts_in_block.shape[0] == 0:
            entry["max_semiconjugacy_error"] = None
            entry["is_spurious_attractor"] = None
            per_minimal[str(lbl)] = entry
            continue
        max_err = float(
            compute_max_semiconjugacy_error(
                encoder=model.encoder,
                latent_map=model.latent_map,
                points_in_block=pts_in_block,
                next_points_true=next_in_block,
                device=device,
            )
        )
        entry["max_semiconjugacy_error"] = max_err
        entry["is_spurious_attractor"] = bool(max_err > tau_bar)
        per_minimal[str(lbl)] = entry

    return {
        "minimal_morse_labels": minimal_labels,
        "minimal_morse_sets": per_minimal,
    }
