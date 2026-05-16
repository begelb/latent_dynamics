"""CMGDB-free diagnostic of the trained latent dynamics.

Goal: detect a collapsed-latent regime in seconds, before paying for a 30+ min
CMGDB run on a model that has only one global attractor in the latent.

Three artifacts written under the run's write-root (``cfg.paths.output_dir``
by default, or the replay tree when invoked with ``out_dir=...``):

- ``figures/latent_pointcloud.png`` : encoded train+test data scattered in the
  latent box. A healthy run shows visible structure spanning the box; a
  collapsed run shows a tight blob.
- ``figures/latent_orbits.png`` : terminal points after iterating ``latent_map``
  from a uniform grid of starting points. Reveals attractor count and basin
  structure cheaply.
- ``diagnose.json`` : structured summary with ``n_distinct_limit_points``,
  ``latent_extent``, ``mean_iter_to_convergence``. The metrics stage cross-
  checks this against the saved Morse graph.
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
    arch: object,
    data_sample_scaled: NDArray[np.float64],
    grid: NDArray[np.float64],
    bounds: LatentBounds,
    device: torch.device,
    near_identity_thresh: float,
) -> dict:
    """Soft note: are E and D close to the identity?

    Each side is gated independently: when high_dims == low_dims, a side is
    computed iff its own output activation is 'none'. When dims don't match,
    all four optional fields are null regardless of activations.
    """
    matched_dims = arch.high_dims == arch.low_dims
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

    enc_eligible = getattr(arch, "encoder_out_activation", None) == "none"
    dec_eligible = getattr(arch, "decoder_out_activation", None) == "none"

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


@torch.no_grad()
def _iterate_latent(
    latent_map: torch.nn.Module,
    grid: NDArray[np.float64],
    *,
    n_iter: int,
    device: torch.device,
    convergence_eps: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Iterate the latent map ``n_iter`` times; record terminal points and steps to convergence."""
    z = torch.as_tensor(grid, dtype=torch.float32, device=device)
    iter_counts = np.full(grid.shape[0], n_iter, dtype=np.int64)
    z_prev = z.clone()
    for k in range(n_iter):
        z = latent_map(z)
        diff = (z - z_prev).cpu().numpy()
        moving_mask = np.linalg.norm(diff, axis=-1) > convergence_eps
        # iter_counts records the first iteration after which a point stopped moving
        first_converged = (iter_counts == n_iter) & ~moving_mask
        iter_counts[first_converged] = k + 1
        z_prev = z.clone()
    return z.cpu().numpy(), iter_counts


def _cluster_terminal_points(
    terminal: NDArray[np.float64], *, eps: float
) -> tuple[int, NDArray[np.int64]]:
    """Greedy single-link clustering with cutoff ``eps``; returns (n_clusters, labels)."""
    n = terminal.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    next_label = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = next_label
        # link any later point within eps to this seed
        deltas = np.linalg.norm(terminal[i + 1 :] - terminal[i], axis=-1)
        same = np.where(deltas < eps)[0] + (i + 1)
        labels[same] = next_label
        next_label += 1
    return next_label, labels


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
    ax.set_title("encoded train+test data")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_orbits_plot(
    grid: NDArray[np.float64],
    terminal: NDArray[np.float64],
    labels: NDArray[np.int64],
    bounds: LatentBounds,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if grid.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.scatter(
            terminal[:, 0], np.zeros(terminal.shape[0]), c=labels, cmap="tab10", s=10, alpha=0.6
        )
        ax.set_xlim(bounds.lower[0], bounds.upper[0])
        ax.set_yticks([])
        ax.set_xlabel("$z_1$")
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(terminal[:, 0], terminal[:, 1], c=labels, cmap="tab10", s=10, alpha=0.6)
        ax.set_xlim(bounds.lower[0], bounds.upper[0])
        ax.set_ylim(bounds.lower[1], bounds.upper[1])
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"latent_map orbits, terminal points (n_clusters={int(labels.max() + 1)})")
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
    n_iter: int = 200,
    points_per_axis_2d: int = 50,
    points_per_axis_1d: int = 200,
    cluster_eps_frac: float = 0.01,
    convergence_eps_frac: float = 1e-4,
    verbose: bool = True,
    out_dir: Path | None = None,
) -> dict:
    """Diagnose the trained latent map without invoking CMGDB.

    Reads source artifacts from ``cfg.paths``: the checkpoint at
    ``cfg.paths.output_dir/models/``, the scaler at
    ``cfg.paths.scaler_path(train_file)``, and the train/test CSVs at
    ``cfg.paths.data_dir``. Writes to ``out_dir`` when provided
    (replay-routing) or to ``cfg.paths.output_dir`` otherwise:

    - ``<write_root>/figures/latent_pointcloud.png``
    - ``<write_root>/figures/latent_orbits.png``
    - ``<write_root>/diagnose.json``

    Returns the JSON payload as a dict.
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

    # 1. Encoded data scatter.
    high_data_scaled = _load_train_data_scaled(cfg, train_file)
    with torch.no_grad():
        encoded = (
            model.encoder(torch.as_tensor(high_data_scaled, dtype=torch.float32, device=device))
            .cpu()
            .numpy()
        )

    bounds, bounds_source = _resolve_bounds(cfg, model.encoder, encoded)

    # 2. Grid + iteration.
    pts_per_axis = points_per_axis_1d if cfg.arch.low_dims == 1 else points_per_axis_2d
    grid = _grid_in_bounds(bounds, points_per_axis=pts_per_axis)
    bounds_diam = float(np.linalg.norm(bounds.upper - bounds.lower))
    convergence_eps = convergence_eps_frac * bounds_diam
    cluster_eps = cluster_eps_frac * bounds_diam

    terminal, iter_counts = _iterate_latent(
        model.latent_map,
        grid,
        n_iter=n_iter,
        device=device,
        convergence_eps=convergence_eps,
    )

    n_clusters, labels = _cluster_terminal_points(terminal, eps=cluster_eps)

    # Cluster only the points that converged. When iteration didn't reach
    # limit points, the all-points cluster count overcounts trajectories.
    converged_mask = iter_counts < n_iter
    if converged_mask.any():
        n_converged_clusters, _ = _cluster_terminal_points(
            terminal[converged_mask], eps=cluster_eps
        )
    else:
        n_converged_clusters = 0

    # 3. Plots.
    fig_dir = write_root / "figures"
    _save_pointcloud_plot(encoded, bounds, fig_dir / "latent_pointcloud.png")
    _save_orbits_plot(grid, terminal, labels, bounds, fig_dir / "latent_orbits.png")

    # 4. JSON summary.
    encoded_extent = (encoded.max(axis=0) - encoded.min(axis=0)).tolist()
    frac_unconverged = float(np.mean(iter_counts == n_iter))
    # The "trustworthy" attractor count is the converged-only cluster count.
    # Fall back to the all-points count when nothing converged, but flag it.
    if frac_unconverged >= 0.5:
        diagnostic = "iteration_did_not_converge"
        trusted_n_limit_points = -1
    else:
        diagnostic = "converged"
        trusted_n_limit_points = int(n_converged_clusters)

    payload = {
        # Best-estimate attractor count, or -1 when the iteration never
        # reached limit points (latent_map likely near-identity, or n_iter
        # too small).
        "n_distinct_limit_points": trusted_n_limit_points,
        "diagnostic": diagnostic,
        # Raw cluster counts kept for inspection.
        "n_terminal_clusters_all": int(n_clusters),
        "n_terminal_clusters_converged": int(n_converged_clusters),
        "encoded_extent": encoded_extent,
        "encoded_image_diameter": float(np.linalg.norm(np.asarray(encoded_extent))),
        "bounds_lower": bounds.lower.tolist(),
        "bounds_upper": bounds.upper.tolist(),
        "bounds_source": bounds_source,
        "n_iter": int(n_iter),
        "n_grid_points": int(grid.shape[0]),
        "convergence_eps": float(convergence_eps),
        "cluster_eps": float(cluster_eps),
        "mean_iter_to_convergence": float(np.mean(iter_counts)),
        "max_iter_to_convergence": int(np.max(iter_counts)),
        "frac_unconverged": frac_unconverged,
    }

    write_root.mkdir(parents=True, exist_ok=True)
    json_path = write_root / "diagnose.json"
    json_path.write_text(json.dumps(payload, indent=2))

    if verbose:
        if diagnostic == "converged":
            print(
                f"diagnose: {trusted_n_limit_points} distinct limit point(s), "
                f"encoded extent {[round(x, 3) for x in encoded_extent]}, "
                f"{int(frac_unconverged * 100)}% unconverged after {n_iter} iters"
            )
        else:
            print(
                f"diagnose: latent_map iteration did not converge "
                f"({int(frac_unconverged * 100)}% of {grid.shape[0]} grid points "
                f"still moving > {convergence_eps:.2e} after {n_iter} iters); "
                f"latent_map is likely near-identity or has slow/non-fixed dynamics. "
                f"all-terminal clusters={n_clusters}; encoded extent "
                f"{[round(x, 3) for x in encoded_extent]}."
            )
        print(f"  -> {json_path}")
        print(f"  -> {fig_dir / 'latent_pointcloud.png'}")
        print(f"  -> {fig_dir / 'latent_orbits.png'}")

    return payload
