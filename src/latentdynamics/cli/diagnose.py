"""Quick pre-CMGDB collapse-detection diagnostic.

Goal: detect a broken trained model in seconds, before paying for a 30+ min
CMGDB run. The stage answers "is training producing a non-degenerate model"
not "what are its attractors" — the attractor question is CMGDB's job.

Three artifacts written under the run's write-root (``cfg.paths.output_dir``
by default, or the replay tree when invoked with ``out_dir=...``):

- ``figures/latent_pointcloud.png`` : encoded train+val data scattered in the
  latent box. Visual sanity for encoder collapse.
- ``figures/latent_map_one_step.png`` : a single application of the latent map
  on a coarse grid (1D function plot or 2D scatter overlay).
- ``diagnose.json`` : structured summary with two hard flags (encoder
  collapse, latent map over-contraction) and three soft notes
  (G/E/D near-identity). See
  docs/superpowers/specs/2026-05-16-diagnose-redesign-design.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from ..analysis.morse import LatentBounds
from ..config import ExperimentConfig
from ..training import load_any_checkpoint


_BOUNDED_ACTIVATION_SPANS: dict[str, float] = {
    "tanh": 2.0,
    "sigmoid": 1.0,
}


def _encoder_extent_report(
    encoded: NDArray[np.float64],
    *,
    out_activation: str,
    collapse_thresh: float,
) -> tuple[dict, bool]:
    """Encoder block of diagnose.json plus the encoder_collapsed flag.

    For tanh/sigmoid out the flag fires when ``max_extent / reference_span <
    collapse_thresh``. The linear-out case is handled separately (Task 3).
    """
    extent_per_axis = (encoded.max(axis=0) - encoded.min(axis=0)).astype(float)
    max_extent = float(extent_per_axis.max())
    reference_span = _BOUNDED_ACTIVATION_SPANS.get(out_activation)
    if reference_span is None:
        # Linear out: caller (Task 3) handles the absolute fallback.
        max_extent_relative = None
        collapsed = max_extent < collapse_thresh
    else:
        max_extent_relative = max_extent / reference_span
        collapsed = max_extent_relative < collapse_thresh
    block = {
        "extent_per_axis": extent_per_axis.tolist(),
        "max_extent": max_extent,
        "out_activation": out_activation,
        "reference_span": reference_span,
        "max_extent_relative": max_extent_relative,
    }
    return block, collapsed


def _resolve_bounds(
    cfg: ExperimentConfig,
    encoder: torch.nn.Module,
    encoded_pts: NDArray[np.float64],
) -> tuple[LatentBounds, str]:
    if cfg.cmgdb.lower_bounds is not None and cfg.cmgdb.upper_bounds is not None:
        return (
            LatentBounds(
                lower=np.asarray(cfg.cmgdb.lower_bounds, dtype=np.float64),
                upper=np.asarray(cfg.cmgdb.upper_bounds, dtype=np.float64),
            ),
            "config",
        )
    lower = encoded_pts.min(axis=0)
    upper = encoded_pts.max(axis=0)
    buffer = cfg.cmgdb.bounds_epsilon_frac * (upper - lower)
    return LatentBounds(lower=lower - buffer, upper=upper + buffer), "encoded_data"


def _grid_in_bounds(bounds: LatentBounds, *, points_per_axis: int) -> NDArray[np.float64]:
    """Uniform grid covering the latent box, with sensible 1D vs 2D layouts."""
    axes = [
        np.linspace(lo, hi, points_per_axis)
        for lo, hi in zip(bounds.lower, bounds.upper, strict=True)
    ]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=-1)


@torch.no_grad()
def _latent_map_one_step_report(
    latent_map: torch.nn.Module,
    grid: NDArray[np.float64],
    *,
    device: torch.device,
    bounds: LatentBounds,
    contraction_thresh: float,
    near_identity_thresh: float,
) -> tuple[dict, NDArray[np.float64], bool]:
    """One forward pass of G on the grid. Returns (block, image, overcontracted).

    The block dict contains:
    - n_grid_points: number of grid points
    - grid_diameter: Euclidean diameter of the grid (max-min distance)
    - image_diameter: Euclidean diameter of the image
    - contraction_ratio: image_diameter / grid_diameter
    - mean_step_relative: mean ||G(z) - z|| / box_diameter
    - near_identity: bool, True when mean_step_relative < near_identity_thresh

    The overcontracted flag is True when contraction_ratio < contraction_thresh.
    """
    z = torch.as_tensor(grid, dtype=torch.float32, device=device)
    image_t = latent_map(z)
    image = image_t.cpu().numpy().astype(np.float64)

    def _diam(points: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

    grid_diameter = _diam(grid)
    image_diameter = _diam(image)
    contraction_ratio = image_diameter / grid_diameter if grid_diameter > 0 else 0.0

    box_diameter = float(np.linalg.norm(bounds.upper - bounds.lower))
    step_norms = np.linalg.norm(image - grid, axis=-1)
    mean_step_relative = float(step_norms.mean() / box_diameter) if box_diameter > 0 else 0.0

    block = {
        "n_grid_points": int(grid.shape[0]),
        "grid_diameter": grid_diameter,
        "image_diameter": image_diameter,
        "contraction_ratio": contraction_ratio,
        "mean_step_relative": mean_step_relative,
        "near_identity": mean_step_relative < near_identity_thresh,
    }
    overcontracted = contraction_ratio < contraction_thresh
    return block, image, overcontracted


@torch.no_grad()
def _matched_dim_identity_report(
    model: object,
    *,
    high_dims: int,
    low_dims: int,
    encoder_out_activation: str,
    decoder_out_activation: str,
    data_sample_scaled: NDArray[np.float64],
    grid: NDArray[np.float64],
    bounds: LatentBounds,
    device: torch.device,
    near_identity_thresh: float,
) -> dict:
    """Soft note: are E and D close to the identity?

    Each side is gated independently: when high_dims == low_dims, a side is
    computed iff its own (resolved) output activation is 'none'. When dims
    don't match, all four optional fields are null regardless of activations.

    Pass the *resolved* per-component out_activation values (i.e.
    ``cfg.arch.component('encoder').out_activation``), not the top-level
    ``arch.encoder_out_activation`` default.
    """
    matched_dims = high_dims == low_dims
    base = {
        "matched_dims": bool(matched_dims),
        "encoder_near_identity": None,
        "decoder_near_identity": None,
        "mean_step_E_relative": None,
        "mean_step_D_relative": None,
    }
    if not matched_dims:
        return base

    box_diameter = float(np.linalg.norm(bounds.upper - bounds.lower))
    if box_diameter == 0:
        return base

    enc_eligible = encoder_out_activation == "none"
    dec_eligible = decoder_out_activation == "none"

    if enc_eligible:
        x = torch.as_tensor(data_sample_scaled, dtype=torch.float32, device=device)
        e_x = model.encoder(x).cpu().numpy().astype(np.float64)
        mean_step_e = float(np.linalg.norm(e_x - data_sample_scaled, axis=-1).mean()
                            / box_diameter)
        base["mean_step_E_relative"] = mean_step_e
        base["encoder_near_identity"] = mean_step_e < near_identity_thresh

    if dec_eligible:
        z = torch.as_tensor(grid, dtype=torch.float32, device=device)
        d_z = model.decoder(z).cpu().numpy().astype(np.float64)
        mean_step_d = float(np.linalg.norm(d_z - grid, axis=-1).mean() / box_diameter)
        base["mean_step_D_relative"] = mean_step_d
        base["decoder_near_identity"] = mean_step_d < near_identity_thresh

    return base




def _save_pointcloud_plot(
    encoded_pts: NDArray[np.float64],
    bounds: LatentBounds,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if encoded_pts.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.scatter(encoded_pts[:, 0], np.zeros(encoded_pts.shape[0]), s=2, alpha=0.3)
        ax.set_xlim(bounds.lower[0], bounds.upper[0])
        ax.set_yticks([])
        ax.set_xlabel("$z_1$")
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(encoded_pts[:, 0], encoded_pts[:, 1], s=1, alpha=0.3)
        ax.set_xlim(bounds.lower[0], bounds.upper[0])
        ax.set_ylim(bounds.lower[1], bounds.upper[1])
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        ax.set_aspect("equal", adjustable="box")
    ax.set_title("encoded train+val data")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_one_step_plot(
    grid: NDArray[np.float64],
    image: NDArray[np.float64],
    bounds: LatentBounds,
    out_path: Path,
) -> None:
    """Show a single application of G overlaid on the grid in latent space."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if grid.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        order = np.argsort(grid[:, 0])
        z = grid[order, 0]
        gz = image[order, 0]
        ax.plot(z, gz, label="$G(z)$", linewidth=2)
        ax.plot(z, z, "--", color="grey", label="$z$ (identity)")
        ax.set_xlim(bounds.lower[0], bounds.upper[0])
        ax.set_xlabel("$z$")
        ax.set_ylabel("$G(z)$")
        ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(grid[:, 0], grid[:, 1], s=6, alpha=0.4, label="grid $S$",
                   color="tab:blue")
        ax.scatter(image[:, 0], image[:, 1], s=6, alpha=0.6, label="$G(S)$",
                   color="tab:red")
        ax.set_xlim(bounds.lower[0], bounds.upper[0])
        ax.set_ylim(bounds.lower[1], bounds.upper[1])
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", framealpha=0.8)
    ax.set_title("latent_map: one-step image of grid")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_train_data_scaled(cfg: ExperimentConfig, train_file: str) -> NDArray[np.float64]:
    high = cfg.arch.high_dims
    train = np.loadtxt(cfg.paths.data_dir / f"{train_file}.csv", delimiter=",", skiprows=1)
    val = np.loadtxt(cfg.paths.val_csv(), delimiter=",", skiprows=1)
    scaler = joblib.load(cfg.paths.scaler_path(train_file))
    pieces = [
        scaler.transform(train[:, :high]),
        scaler.transform(val[:, :high]),
        scaler.transform(train[:, high:]),
        scaler.transform(val[:, high:]),
    ]
    return np.vstack(pieces).astype(np.float64)


def run(
    cfg: ExperimentConfig,
    *,
    train_file: str = "train",
    device: torch.device | str | None = None,
    points_per_axis_2d: int = 50,
    points_per_axis_1d: int = 200,
    encoder_collapse_thresh: float = 0.02,
    latent_contraction_thresh: float = 0.05,
    near_identity_thresh: float = 0.01,
    verbose: bool = True,
    out_dir: Path | None = None,
) -> dict:
    """Quick pre-CMGDB collapse-detection diagnostic.

    Two hard flags + three soft notes. No iteration, no clustering. See
    docs/superpowers/specs/2026-05-16-diagnose-redesign-design.md.
    """
    source_root = cfg.paths.output_dir
    write_root = Path(out_dir) if out_dir is not None else source_root
    model, _arch = load_any_checkpoint(source_root / "models", arch=cfg.arch)

    if device is None:
        device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    model.to(device)

    # 1. Encoded data scatter -> encoder collapse check + pointcloud figure.
    high_data_scaled = _load_train_data_scaled(cfg, train_file)
    with torch.no_grad():
        encoded = (
            model.encoder(torch.as_tensor(high_data_scaled, dtype=torch.float32, device=device))
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    # Resolved per-component out_activations (per-component override takes
    # precedence over arch-level default; see ArchConfig.component()).
    encoder_out_activation = cfg.arch.component("encoder").out_activation
    decoder_out_activation = cfg.arch.component("decoder").out_activation

    encoder_block, encoder_collapsed = _encoder_extent_report(
        encoded,
        out_activation=encoder_out_activation,
        collapse_thresh=encoder_collapse_thresh,
    )

    bounds, bounds_source = _resolve_bounds(cfg, model.encoder, encoded)

    # 2. Latent-map one-step report.
    pts_per_axis = points_per_axis_1d if cfg.arch.low_dims == 1 else points_per_axis_2d
    grid = _grid_in_bounds(bounds, points_per_axis=pts_per_axis)
    lm_block, image, latent_overcontracted = _latent_map_one_step_report(
        model.latent_map,
        grid,
        device=device,
        bounds=bounds,
        contraction_thresh=latent_contraction_thresh,
        near_identity_thresh=near_identity_thresh,
    )

    # 3. Matched-dim E/D identity soft notes.
    identity_block = _matched_dim_identity_report(
        model,
        high_dims=cfg.arch.high_dims,
        low_dims=cfg.arch.low_dims,
        encoder_out_activation=encoder_out_activation,
        decoder_out_activation=decoder_out_activation,
        data_sample_scaled=high_data_scaled,
        grid=grid,
        bounds=bounds,
        device=device,
        near_identity_thresh=near_identity_thresh,
    )

    # 4. Plots.
    fig_dir = write_root / "figures"
    _save_pointcloud_plot(encoded, bounds, fig_dir / "latent_pointcloud.png")
    _save_one_step_plot(grid, image, bounds, fig_dir / "latent_map_one_step.png")

    # 5. Diagnostic label.
    if encoder_collapsed and latent_overcontracted:
        diagnostic = "encoder_collapsed_and_latent_overcontracted"
    elif encoder_collapsed:
        diagnostic = "encoder_collapsed"
    elif latent_overcontracted:
        diagnostic = "latent_map_overcontracted"
    else:
        diagnostic = "ok"

    payload = {
        "diagnostic": diagnostic,
        "hard_flags": {
            "encoder_collapsed": bool(encoder_collapsed),
            "latent_map_overcontracted": bool(latent_overcontracted),
        },
        "encoder": encoder_block,
        "latent_map": lm_block,
        **identity_block,
        "bounds": {
            "lower": bounds.lower.tolist(),
            "upper": bounds.upper.tolist(),
            "source": bounds_source,
        },
    }

    write_root.mkdir(parents=True, exist_ok=True)
    json_path = write_root / "diagnose.json"
    json_path.write_text(json.dumps(payload, indent=2))

    if verbose:
        flag_summary = []
        if encoder_collapsed:
            flag_summary.append(f"encoder_collapsed (max_extent={encoder_block['max_extent']:.3g})")
        if latent_overcontracted:
            flag_summary.append(
                f"latent_map_overcontracted (ratio={lm_block['contraction_ratio']:.3g})"
            )
        notes = []
        if lm_block["near_identity"]:
            notes.append(f"G~id (mean step rel={lm_block['mean_step_relative']:.3g})")
        if identity_block.get("encoder_near_identity"):
            notes.append("E~id")
        if identity_block.get("decoder_near_identity"):
            notes.append("D~id")
        if flag_summary:
            print(f"diagnose [{diagnostic}]: " + "; ".join(flag_summary))
        else:
            print("diagnose: ok")
        if notes:
            print("  notes: " + ", ".join(notes))
        print(f"  -> {json_path}")
        print(f"  -> {fig_dir / 'latent_pointcloud.png'}")
        print(f"  -> {fig_dir / 'latent_map_one_step.png'}")

    return payload
