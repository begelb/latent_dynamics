"""Phase-space plots of training-data trajectories.

The data stage writes pairs ``(x_t, x_{t+1})`` of dim ``2 * high_dims`` per row,
grouped by trajectory in blocks of ``n_iterations`` rows. This module loads
that CSV, regroups it into ``(n_trajectories, n_iterations + 1, high_dims)``,
and renders a 2D scatter (when ``dim == 2``) or a grid of 2D pair projections
(when ``dim >= 3``). Single 3D scatter rendering is intentionally avoided —
matplotlib's 3D depth cues collapse under high overplot and obscure the
flow direction the time-coded color is meant to show.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def _load_trajectories(csv_path: Path, metadata_path: Path) -> tuple[np.ndarray, dict]:
    """Return ``(trajectories, metadata)`` with shape
    ``(n_trajectories, n_iterations + 1, dim)``."""
    metadata = json.loads(metadata_path.read_text())
    n_iter = int(metadata["n_iterations"])
    dim = int(metadata["dimension"])

    df = pd.read_csv(csv_path)
    arr = df.to_numpy()
    if arr.shape[1] != 2 * dim:
        raise ValueError(
            f"{csv_path} has {arr.shape[1]} columns but metadata claims dim={dim} "
            f"(expected {2 * dim})"
        )

    n_traj = arr.shape[0] // n_iter
    if n_traj == 0:
        raise ValueError(f"{csv_path} has fewer than n_iterations={n_iter} rows")
    arr = arr[: n_traj * n_iter]
    x = arr[:, :dim].reshape(n_traj, n_iter, dim)
    last = arr[:, dim:].reshape(n_traj, n_iter, dim)[:, -1:, :]
    trajectories = np.concatenate([x, last], axis=1)
    return trajectories, metadata


def _draw_pair_panel(
    ax: plt.Axes,
    trajectories: np.ndarray,
    proj: tuple[int, int],
    *,
    times: np.ndarray,
    cmap: str,
    point_size: float,
    point_alpha: float,
    overlay_idx: np.ndarray,
    ic_size: float,
    final_size: float,
) -> object:
    i, j = proj
    n_steps = trajectories.shape[1]
    xs = trajectories[..., i].reshape(-1)
    ys = trajectories[..., j].reshape(-1)
    flat_times = times.reshape(-1)
    order = np.argsort(flat_times)
    sc = ax.scatter(
        xs[order], ys[order],
        c=flat_times[order], cmap=cmap, vmin=0, vmax=n_steps - 1,
        s=point_size, alpha=point_alpha,
        edgecolors="none", rasterized=True,
    )
    for k in overlay_idx:
        ax.plot(
            trajectories[k, :, i], trajectories[k, :, j],
            color="0.2", linewidth=0.7, alpha=0.6, zorder=3,
        )
    ax.scatter(
        trajectories[:, 0, i], trajectories[:, 0, j],
        s=ic_size, facecolors="none", edgecolors="black",
        linewidths=0.5, alpha=0.7, zorder=4,
    )
    ax.scatter(
        trajectories[:, -1, i], trajectories[:, -1, j],
        s=final_size, color="black", alpha=0.7, zorder=5,
    )
    ax.set_xlabel(f"x[{i}]")
    ax.set_ylabel(f"x[{j}]")
    return sc


def plot_data_trajectories(
    csv_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    proj: Sequence[int] | None = None,
    n_overlay_lines: int = 8,
    cmap: str = "viridis",
    point_size: float = 4.0,
    point_alpha: float = 0.35,
    ic_size: float = 24.0,
    final_size: float = 20.0,
) -> Figure:
    """Phase-space scatter of training-data trajectories.

    For ``dim == 2`` returns a single-panel scatter. For ``dim >= 3`` returns
    a 3-panel grid of pair projections (``(p0, p1)``, ``(p0, p2)``,
    ``(p1, p2)``) using the first three components of ``proj`` (default:
    indices 0, 1, 2). All scatter points are colored by iteration index
    (early → late under ``viridis``); initial conditions are open circles
    and final states are filled circles.
    """
    csv_path = Path(csv_path)
    if metadata_path is None:
        metadata_path = csv_path.with_name(csv_path.stem + "_metadata.json")
    metadata_path = Path(metadata_path)

    trajectories, metadata = _load_trajectories(csv_path, metadata_path)
    n_traj, n_steps, dim = trajectories.shape

    if dim == 1:
        raise ValueError("1D system: plot as time series, not phase space")

    if proj is None:
        proj = tuple(range(min(3, dim)))
    proj = tuple(proj)
    if dim == 2 and len(proj) != 2:
        raise ValueError(f"2D system requires len(proj)==2, got {proj}")
    if dim >= 3 and len(proj) < 3:
        # Allow user to ask for just a pair on a >=3 dim system; falls through to single-panel
        pass
    if not all(0 <= p < dim for p in proj):
        raise ValueError(f"proj indices out of range for dim={dim}: {proj}")

    sys_name = metadata.get("system", csv_path.parent.name)
    n_lines = min(n_overlay_lines, n_traj)
    overlay_idx = np.linspace(0, n_traj - 1, n_lines, dtype=int) if n_lines else np.array([], int)
    times = np.broadcast_to(np.arange(n_steps), trajectories.shape[:2])

    if len(proj) == 2:
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        sc = _draw_pair_panel(
            ax, trajectories, (proj[0], proj[1]),
            times=times, cmap=cmap,
            point_size=point_size, point_alpha=point_alpha,
            overlay_idx=overlay_idx, ic_size=ic_size, final_size=final_size,
        )
        ax.set_title(f"{sys_name}  dim={dim}, {n_traj} trajectories × {n_steps} steps")
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85, label="iteration t")
        cbar.set_alpha(1.0)
        _add_marker_legend(ax, n_steps)
        fig.tight_layout()
        return fig

    pairs = list(itertools.combinations(proj[:3], 2))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    last_sc = None
    for ax, pair in zip(axes, pairs, strict=True):
        last_sc = _draw_pair_panel(
            ax, trajectories, pair,
            times=times, cmap=cmap,
            point_size=point_size, point_alpha=point_alpha,
            overlay_idx=overlay_idx, ic_size=ic_size, final_size=final_size,
        )
    _add_marker_legend(axes[0], n_steps)
    projection_note = ""
    if dim > 3:
        projection_note = f"  (showing first 3 of {dim} dims)"
    fig.suptitle(
        f"{sys_name}  dim={dim}, {n_traj} trajectories × {n_steps} steps{projection_note}",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 0.94, 0.97))
    cbar_ax = fig.add_axes([0.95, 0.15, 0.012, 0.7])
    fig.colorbar(last_sc, cax=cbar_ax, label="iteration t")
    return fig


def _add_marker_legend(ax: plt.Axes, n_steps: int) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor="none", markeredgecolor="black",
                   markersize=6, label="t=0"),
        plt.Line2D([0], [0], marker="o", linestyle="none",
                   color="black", markersize=6, label=f"t={n_steps - 1}"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.85)


def save_default_trajectory_plot(
    data_dir: str | Path,
    out_path: str | Path | None = None,
    *,
    train_label: str = "train",
    max_dim: int = 10,
) -> Path | None:
    """Render the canonical trajectory plot for one dataset.

    Picks ``train.csv`` (or the first sweep ``train_<N>.csv``). Returns
    ``None`` and skips when ``dim > max_dim``.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{train_label}.csv"
    if not csv_path.exists():
        sweep = sorted(
            p for p in data_dir.glob("train_*.csv") if not p.name.endswith("_metadata.json")
        )
        if not sweep:
            raise FileNotFoundError(f"no {train_label}.csv (or train_*.csv) under {data_dir}")
        csv_path = sweep[0]

    meta_path = csv_path.with_name(csv_path.stem + "_metadata.json")
    dim = int(json.loads(meta_path.read_text())["dimension"])
    if dim > max_dim:
        return None

    fig = plot_data_trajectories(csv_path, meta_path)
    if out_path is None:
        out_path = data_dir / "trajectories.png"
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
