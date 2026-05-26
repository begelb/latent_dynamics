"""Plot regions of attraction (RoA) computed from the cell graph.

Two layers:
- The uniform grid colored by ``box_roa`` at low alpha (the basin of
  attraction of each minimal Morse set, plus boundary/escape regions).
- The original CMGDB Morse-set boxes overlaid at full alpha, colored by
  their Morse node — so minimal Morse sets and saddle/source Morse sets
  render as in CMGDB's ``morse_sets.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ..analysis.cmgdb_roa import BOUNDARY, ESCAPE, CellROA, load_exact_roa
from ..analysis.cell_graph import CellGraphROA, compute_cell_graph_roa
from ..analysis.regions_of_attraction import BoxROATable, load_box_roa
from .style import save_figure


_BOUNDARY_COLOR = "#bbbbbb"
_ESCAPE_COLOR = "#ffffff"


def _rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    r, g, b, _ = to_rgba(hex_color)
    return (r, g, b, alpha)


def plot_roa_overlay_cell_graph(
    cg: CellGraphROA,
    morse_table: BoxROATable,
    *,
    roa_alpha: float = 0.35,
    morse_alpha: float = 0.95,
) -> Figure:
    """Render basin (cell-graph RoA, low alpha) + Morse sets (full alpha)."""
    grid = cg.grid
    mg = cg.morse_graph
    r = grid.resolution

    # Build a (resolution, resolution, 4) RGBA image from box_roa.
    img = np.zeros((r, r, 4), dtype=np.float64)
    roa = cg.box_roa.reshape(r, r)
    for label in np.unique(roa):
        mask = roa == label
        if label == CellGraphROA.ESCAPE:
            color = _rgba(_ESCAPE_COLOR, 0.0)
        elif label == CellGraphROA.BOUNDARY:
            color = _rgba(_BOUNDARY_COLOR, roa_alpha)
        else:
            hex_color = mg.colors.get(int(label), "#888888")
            color = _rgba(hex_color, roa_alpha)
        img[mask] = color

    # Note: imshow expects (rows=y, cols=x); our grid is laid out as
    # (i over x, j over y), so transpose for display and use origin='lower'.
    display = np.transpose(img, (1, 0, 2))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(
        display,
        origin="lower",
        extent=(grid.bounds_lo[0], grid.bounds_hi[0], grid.bounds_lo[1], grid.bounds_hi[1]),
        interpolation="nearest",
        aspect="equal",
    )

    # Overlay recurrent Morse sets at full alpha, colored by their own Morse
    # node. Non-minimal recurrent sets are not part of a lower set's transient
    # RoA even when the Morse graph has a path to that lower set.
    patches = []
    facecolors = []
    for _, row in morse_table.boxes.iterrows():
        lo_x, lo_y = row["lower_0"], row["lower_1"]
        hi_x, hi_y = row["upper_0"], row["upper_1"]
        patches.append(Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y))
        morse_node = int(row["morse_node"])
        facecolors.append(mg.colors.get(morse_node, "#888888"))
    pc = PatchCollection(patches, match_original=False)
    pc.set_facecolor(facecolors)
    pc.set_edgecolor("none")
    pc.set_alpha(morse_alpha)
    ax.add_collection(pc)

    ax.set_xlim(grid.bounds_lo[0], grid.bounds_hi[0])
    ax.set_ylim(grid.bounds_lo[1], grid.bounds_hi[1])
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")

    fig.tight_layout()
    return fig


def render_cell_graph_roa(
    morse_graph_dot: str | Path,
    morse_sets_csv: str | Path,
    latent_map: object,
    out_path: str | Path,
    *,
    resolution: int = 128,
    bounds_padding: float = 0.05,
    device: str = "cpu",
) -> Path:
    """One-shot: compute cell-graph RoA and save the plot."""
    cg = compute_cell_graph_roa(
        latent_map,
        morse_graph_dot,
        morse_sets_csv,
        resolution=resolution,
        bounds_padding=bounds_padding,
        device=device,
    )
    table = load_box_roa(morse_graph_dot, morse_sets_csv)
    fig = plot_roa_overlay_cell_graph(cg, table)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=140, bbox_inches="tight", close=True)
    return out_path


def plot_exact_roa(
    roa: CellROA,
    morse_table: BoxROATable,
    *,
    roa_alpha: float = 0.55,
) -> Figure:
    """Render exact CMGDB-cell RoA labels from a saved artifact."""
    mg = morse_table.morse_graph
    labels = np.asarray(roa.box_roa, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    if roa.grid_shape is not None and len(roa.grid_shape) == 2:
        shape = tuple(int(v) for v in roa.grid_shape.tolist())
        if int(np.prod(shape)) != labels.size:
            raise ValueError(
                f"exact RoA grid_shape {shape} does not match {labels.size} labels"
            )
        img = np.zeros(shape + (4,), dtype=np.float64)
        label_grid = labels.reshape(shape)
        for label in np.unique(label_grid):
            mask = label_grid == label
            if label == ESCAPE:
                color = _rgba(_ESCAPE_COLOR, 0.0)
            elif label == BOUNDARY:
                color = _rgba(_BOUNDARY_COLOR, roa_alpha)
            else:
                color = _rgba(mg.colors.get(int(label), "#888888"), roa_alpha)
            img[mask] = color
        display = np.transpose(img, (1, 0, 2))
        if roa.bounds_lower is None or roa.bounds_upper is None:
            extent = (0.0, float(shape[0]), 0.0, float(shape[1]))
        else:
            extent = (
                float(roa.bounds_lower[0]),
                float(roa.bounds_upper[0]),
                float(roa.bounds_lower[1]),
                float(roa.bounds_upper[1]),
            )
        ax.imshow(
            display,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    elif roa.boxes is not None:
        patches = []
        facecolors = []
        for box, label in zip(roa.boxes, labels, strict=True):
            lo_x, lo_y, hi_x, hi_y = box[:4]
            patches.append(Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y))
            if label == ESCAPE:
                facecolors.append(_rgba(_ESCAPE_COLOR, 0.0))
            elif label == BOUNDARY:
                facecolors.append(_rgba(_BOUNDARY_COLOR, roa_alpha))
            else:
                facecolors.append(_rgba(mg.colors.get(int(label), "#888888"), roa_alpha))
        pc = PatchCollection(patches, match_original=False)
        pc.set_facecolor(facecolors)
        pc.set_edgecolor("none")
        ax.add_collection(pc)
        if roa.bounds_lower is not None and roa.bounds_upper is not None:
            ax.set_xlim(roa.bounds_lower[0], roa.bounds_upper[0])
            ax.set_ylim(roa.bounds_lower[1], roa.bounds_upper[1])
        else:
            ax.autoscale()
    else:
        raise ValueError("exact RoA artifact has neither grid_shape nor box geometry")

    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    fig.tight_layout()
    return fig


def render_exact_roa_artifact(
    artifact_path: str | Path,
    morse_graph_dot: str | Path,
    out_path: str | Path,
) -> Path:
    roa = load_exact_roa(artifact_path)
    morse_sets_csv = Path(morse_graph_dot).with_name("morse_sets")
    table = load_box_roa(morse_graph_dot, morse_sets_csv)
    fig = plot_exact_roa(roa, table)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=140, bbox_inches="tight", close=True)
    return out_path
