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
from matplotlib.patches import Patch, Rectangle

from ..analysis.cell_graph import CellGraphROA, compute_cell_graph_roa
from ..analysis.regions_of_attraction import BoxROATable, load_box_roa


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
    title: str | None = None,
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

    # Overlay Morse sets at full alpha, colored by the LCA label of the
    # underlying box so the overlay hue matches the basin underneath.
    patches = []
    facecolors = []
    for _, row in morse_table.boxes.iterrows():
        lo_x, lo_y = row["lower_0"], row["lower_1"]
        hi_x, hi_y = row["upper_0"], row["upper_1"]
        patches.append(Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y))
        lca = mg.roa_label.get(int(row["morse_node"]), int(row["morse_node"]))
        facecolors.append(mg.colors.get(lca, "#888888"))
    pc = PatchCollection(patches, match_original=False)
    pc.set_facecolor(facecolors)
    pc.set_edgecolor("none")
    pc.set_alpha(morse_alpha)
    ax.add_collection(pc)

    ax.set_xlim(grid.bounds_lo[0], grid.bounds_hi[0])
    ax.set_ylim(grid.bounds_lo[1], grid.bounds_hi[1])
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    if title is None:
        n_min = len(mg.minimal)
        title = (
            f"Regions of attraction  ({n_min} minimal Morse set"
            f"{'s' if n_min != 1 else ''}, grid {r}×{r})"
        )
    ax.set_title(title)

    used_labels = sorted(
        int(v) for v in np.unique(roa)
        if v not in (CellGraphROA.ESCAPE, CellGraphROA.BOUNDARY)
    )
    handles = [
        Patch(facecolor=mg.colors.get(n, "#888888"), edgecolor="black", label=f"Morse set {n}")
        for n in used_labels
    ]
    if (cg.box_roa == CellGraphROA.BOUNDARY).any():
        handles.append(Patch(facecolor=_BOUNDARY_COLOR, edgecolor="black", label="multi-basin"))
    if (cg.box_roa == CellGraphROA.ESCAPE).any():
        handles.append(Patch(facecolor="white", edgecolor="black", label="escape"))
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, framealpha=0.85, title="Legend",
    )

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
    title: str | None = None,
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
    fig = plot_roa_overlay_cell_graph(cg, table, title=title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
