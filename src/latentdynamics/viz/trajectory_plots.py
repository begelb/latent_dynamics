"""Latent-trajectory overlay plot for the Leslie 3D failure case (paper Fig. 1.214)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .style import (
    LATENT_ARROW_MUTATION_SCALE,
    LATENT_FIG_WIDTH_IN,
    PALETTE,
    save_latent_figure,
    style_latent_axes,
)


def _scatter_morse_sets(
    ax: plt.Axes,
    morse_set_data: NDArray[np.float64],
    palette: list[str],
) -> None:
    lx, ly, ux, uy = (
        morse_set_data[:, 0],
        morse_set_data[:, 1],
        morse_set_data[:, 2],
        morse_set_data[:, 3],
    )
    labels = morse_set_data[:, 4].astype(int)
    cx = 0.5 * (lx + ux)
    cy = 0.5 * (ly + uy)
    for lbl in np.unique(labels):
        mask = labels == lbl
        ax.scatter(
            cx[mask],
            cy[mask],
            color=palette[int(lbl) % len(palette)],
            marker="s",
            s=12,
            alpha=1.0,
            edgecolors="none",
            zorder=1,
        )


def plot_latent_trajectory(
    morse_set_data: NDArray[np.float64],
    periodic_pts: dict[int, list[list[float]]],
    encode: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    advance_latent: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    save_path: str | Path,
    *,
    trajectory_steps: int = 4,
    palette: list[str] = PALETTE,
) -> Path:
    """Overlay periodic-orbit latent trajectories on the Morse-set partition."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    markers = ["s", "*", "D", "*", "^", "p"]
    gray_to_black = mcolors.LinearSegmentedColormap.from_list("gb", ["#cccccc", "#000000"])
    is_short = trajectory_steps < 5

    fig, ax = plt.subplots(
        figsize=(LATENT_FIG_WIDTH_IN, round(LATENT_FIG_WIDTH_IN * 0.8, 3)),
        layout="constrained",
    )
    _scatter_morse_sets(ax, morse_set_data, palette)

    for label, points in periodic_pts.items():
        m_shape = markers[int(label) % len(markers)]
        z = encode(np.asarray([points[0]], dtype=np.float64))[0]
        trajectory = [z]
        for _ in range(trajectory_steps):
            z = advance_latent(z[None, :])[0]
            trajectory.append(z)

        traj = np.asarray(trajectory)
        # Grey, trimmed connecting segments: pull each end 8% toward the
        # segment center so the path clears the Morse-set boxes (matches the
        # shortened arrows) rather than running edge-to-edge.
        line_alpha = 0.35 if is_short else 0.12
        for start, end in zip(traj[:-1], traj[1:]):
            delta = end - start
            ax.plot(
                [start[0] + 0.08 * delta[0], end[0] - 0.08 * delta[0]],
                [start[1] + 0.08 * delta[1], end[1] - 0.08 * delta[1]],
                color="#6e6e6e",
                alpha=line_alpha,
                linestyle="-",
                linewidth=0.8,
                zorder=5,
            )
        for i in range(len(traj)):
            prog = i / (len(traj) - 1) if len(traj) > 1 else 1.0
            if is_short:
                size, current_color, lw, arrow_alpha = 20, "black", 1.0, 1.0
            else:
                size = 25 + (prog * 45)
                current_color = gray_to_black(prog)
                lw = 0.5 + (prog * 0.7)
                arrow_alpha = 0.2 + (prog * 0.4)
            ax.scatter(
                traj[i, 0],
                traj[i, 1],
                facecolor=current_color,
                marker=m_shape,
                s=size,
                edgecolors="black",
                linewidths=lw,
                zorder=10 + i,
            )
            if i < len(traj) - 1:
                ax.annotate(
                    "",
                    xy=(traj[i + 1, 0], traj[i + 1, 1]),
                    xytext=(traj[i, 0], traj[i, 1]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#6e6e6e",
                        "lw": 0.8,
                        "alpha": arrow_alpha,
                        "mutation_scale": LATENT_ARROW_MUTATION_SCALE,
                        # Shrink both ends so arrows sit between, not over, the
                        # Morse-set boxes they connect.
                        "shrinkA": 7.0,
                        "shrinkB": 7.0,
                    },
                    zorder=100,
                )

    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    style_latent_axes(ax, two_d=True)
    save_latent_figure(fig, save_path, dpi=300, close=True)
    return save_path
