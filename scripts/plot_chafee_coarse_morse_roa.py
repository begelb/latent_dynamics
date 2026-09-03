"""Render the archived attraction basins, with an optional coarse-Morse overlay.

The basin layer reproduces the archived uniform 256-by-256 basin computation
(``compute_att_basins_statistics.py`` in the reference archive). The foreground
uses the connection-complete coarse Morse sets produced by
``scripts/coarsen_chafee_infante.py`` from the archived adaptive cell graph.
Both computations use the archived data-derived latent bounds.

Everything CMGDB can answer is asked of CMGDB: it computes the uniform Morse
graph and cell graph, reports which cells lie in an attractor's basin
(``MorseSingletonReachability``), gives the box of every cell, reads the saved
coarse Morse sets back, and draws every layer. The version that walked the cell
graph in networkx and painted one rectangle per cell is kept beside this one as
``plot_chafee_coarse_morse_roa_overlay_old.py``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import CMGDB
import numpy as np
import torch
from matplotlib.colors import to_hex, to_rgba

import coarsen_chafee_infante as coarsen
from coarsen_chafee_infante import (
    DEFAULT_REFERENCE_ROOT,
    REPO_ROOT,
    _load_reference_model,
    _reference_bounds,
)
from latentdynamics.viz.style import (
    CHAFEE_CONNECTING_COLOR,
    CHAFEE_NEGATIVE_COLOR,
    CHAFEE_POSITIVE_COLOR,
    PAPER_RCPARAMS,
    apply_paper_style,
    save_figure,
)

def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


DEFAULT_COARSE_SETS = _first_existing(
    REPO_ROOT
    / "replay_sources"
    / "chafee_infante"
    / "coarsened"
    / "MG"
    / "morse_sets",
    REPO_ROOT / "output" / "chafee_coarsened" / "MG" / "morse_sets",
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "chafee_coarsened" / "morse_roa_overlay"
DEFAULT_BASIN_OUTPUT = DEFAULT_OUTPUT.with_name("attractor_basins")

BASIN_ALPHA = 0.35
ATTRACTOR_ALPHA = 1.0

# The figure is one CMGDB.PlotMorseSets call, so every layer of it is a labelled
# box set and the order below is the order they are painted in: the merged
# coarse set first, the basins over it, the attracting sets last. The labels
# belong to the drawing, not to any Morse graph. Each stays on a single grid --
# the uniform one for the basins, the coarsened one for the overlay -- because a
# label whose boxes come from two grids cannot be merged into one outline.
LAYER_MERGED = 0
LAYER_BASIN_NEGATIVE = 1
LAYER_BASIN_POSITIVE = 2
LAYER_UNIFORM_NEGATIVE = 3
LAYER_UNIFORM_POSITIVE = 4
LAYER_COARSE_NEGATIVE = 5
LAYER_COARSE_POSITIVE = 6
PAINT_ORDER = (
    LAYER_MERGED,
    LAYER_BASIN_NEGATIVE,
    LAYER_BASIN_POSITIVE,
    LAYER_UNIFORM_NEGATIVE,
    LAYER_UNIFORM_POSITIVE,
    LAYER_COARSE_NEGATIVE,
    LAYER_COARSE_POSITIVE,
)
#: Coarse Morse node -> layer. Node 2 is the merged unstable/connecting class.
COARSE_LAYER = {
    0: LAYER_COARSE_POSITIVE,
    1: LAYER_COARSE_NEGATIVE,
    2: LAYER_MERGED,
}


def _over_white(color: str, alpha: float) -> str:
    """The color a translucent fill leaves on the white page.

    The layers are painted opaque and the basin alpha is composited here
    instead: drawn translucent, a set paints the edge shared by two of its
    boxes twice, and every seam reads as a darker line.
    """
    red, green, blue = to_rgba(color)[:3]
    return to_hex((
        alpha * red + (1.0 - alpha),
        alpha * green + (1.0 - alpha),
        alpha * blue + (1.0 - alpha),
    ))


LAYER_COLORS = (
    CHAFEE_CONNECTING_COLOR,
    _over_white(CHAFEE_NEGATIVE_COLOR, BASIN_ALPHA),
    _over_white(CHAFEE_POSITIVE_COLOR, BASIN_ALPHA),
    CHAFEE_NEGATIVE_COLOR,
    CHAFEE_POSITIVE_COLOR,
    CHAFEE_NEGATIVE_COLOR,
    CHAFEE_POSITIVE_COLOR,
)


def attractor_basins(map_graph, morse_graph, attractors=None):
    """Boxes whose complete reachable Morse-node set is exactly one attractor.

    This is the archived basin semantics -- a box belongs to the basin of
    ``att`` only when every Morse node reachable from it is ``att`` itself --
    and it is exactly what CMGDB.MorseSingletonReachability reports per cell:
    the id of the one Morse node reachable from it, -1 when none is and -2 when
    several are. The query needs the cached cell graph the computation asked
    for; it refuses a lazy one rather than re-evaluating the map per edge.
    """
    if attractors is None:
        attractors = [
            node
            for node in range(int(morse_graph.num_vertices()))
            if len(list(morse_graph.adjacencies(node))) == 0
        ]
    cells = np.arange(int(map_graph.num_vertices()), dtype=np.uint64)
    single = CMGDB.MorseSingletonReachability(map_graph, morse_graph, cells)
    return {
        int(attractor): cells[single == int(attractor)].astype(np.int64).tolist()
        for attractor in attractors
    }


def _compute_uniform_basins(device: str):
    """Reproduce the archived 16/16/16 padded uniform-grid basin graph."""
    model = _load_reference_model(device)
    bounds = _reference_bounds(model, device)
    resolution = 2 ** (16 // 2)
    xs = np.linspace(bounds.lower[0], bounds.upper[0], resolution + 1)
    ys = np.linspace(bounds.lower[1], bounds.upper[1], resolution + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack((xx.ravel(), yy.ravel()), axis=-1).astype(np.float32)

    images = []
    with torch.no_grad():
        for start in range(0, len(points), 8192):
            chunk = torch.from_numpy(points[start : start + 8192]).to(device)
            images.append(model.latent_map(chunk).cpu().numpy())
    image_table = np.concatenate(images, axis=0).reshape(
        resolution + 1,
        resolution + 1,
        2,
    )
    cell_size = (bounds.upper - bounds.lower) / resolution

    def latent_map(point):
        point_array = np.asarray(point, dtype=np.float64)
        index = np.rint((point_array - bounds.lower) / cell_size).astype(int)
        index = np.clip(index, 0, resolution)
        return image_table[index[0], index[1]]

    def box_map(rect):
        return CMGDB.BoxMap(latent_map, rect, padding=True)

    cmgdb_model = CMGDB.Model(
        16,
        16,
        16,
        10000,
        bounds.lower.tolist(),
        bounds.upper.tolist(),
        box_map,
    )
    # The cached cell graph is what the basin query runs on: one extra batched
    # pass over the 65,536-cell grid, after which reachability is a C++ sweep
    # over a CSR array instead of a Python callback per edge.
    morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(
        cmgdb_model, cache_map_graph=True)
    attractors = [
        node
        for node in range(int(morse_graph.num_vertices()))
        if not list(morse_graph.adjacencies(node))
    ]
    if len(attractors) != 2:
        raise ValueError(
            f"archived uniform basin graph has {len(attractors)} attractors, expected 2"
        )
    basins = attractor_basins(map_graph, morse_graph, attractors)
    stable_roots = np.loadtxt(
        coarsen.REFERENCE_ROOT / "stable_solutions.csv",
        delimiter=",",
        ndmin=2,
        dtype=np.float32,
    )
    if stable_roots.shape[0] != 2:
        raise ValueError(
            f"expected two saved stable roots, got shape {stable_roots.shape}"
        )
    with torch.no_grad():
        encoded_roots = (
            model.encoder(torch.from_numpy(stable_roots).to(device)).cpu().numpy()
        )
    physical_attractors = {
        name: _attractor_containing_point(
            encoded_root,
            morse_graph,
            basins,
            attractors,
        )
        for name, encoded_root in zip(
            ("negative", "positive"),
            encoded_roots,
            strict=True,
        )
    }
    if len(set(physical_attractors.values())) != 2:
        raise ValueError(
            "the two saved physical equilibria did not map to distinct attractors"
        )
    return (
        morse_graph,
        map_graph,
        bounds,
        basins,
        attractors,
        physical_attractors,
        resolution,
    )


def _attractor_containing_point(
    point: np.ndarray,
    morse_graph,
    basins,
    attractors,
) -> int:
    matches = []
    for attractor in attractors:
        for cell in basins[attractor]:
            lo_x, lo_y, hi_x, hi_y = morse_graph.phase_space_box(cell)
            if lo_x <= point[0] <= hi_x and lo_y <= point[1] <= hi_y:
                matches.append(attractor)
                break
    if len(matches) != 1:
        raise ValueError(
            f"encoded stable root belongs to {len(matches)} attractor basins"
        )
    return matches[0]


def _attractor_center(morse_graph, node: int) -> np.ndarray:
    centers = []
    for cell in morse_graph.morse_set(node):
        lo_x, lo_y, hi_x, hi_y = morse_graph.phase_space_box(cell)
        centers.append(((lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0))
    if not centers:
        raise ValueError(f"uniform attractor node {node} has no cells")
    return np.mean(np.asarray(centers), axis=0)


def _cell_rows(morse_graph, cells, layer: int) -> list[list[float]]:
    """CMGDB's box for each cell, labelled for one layer of the drawing."""
    return [
        [*morse_graph.phase_space_box(int(cell)), float(layer)] for cell in cells
    ]


def _basin_rows(morse_graph, basins, physical_attractors) -> list[list[float]]:
    """The basin cells of each attractor, with the attractor's own cells over them.

    ``physical_attractors`` names the two attractors by the saved steady
    solutions, negative and positive, rather than by latent position or node id.
    """
    if len(physical_attractors) != 2:
        raise ValueError(
            f"expected negative and positive attractors, got {len(physical_attractors)}"
        )
    rows: list[list[float]] = []
    for state, basin_layer, set_layer in (
        ("negative", LAYER_BASIN_NEGATIVE, LAYER_UNIFORM_NEGATIVE),
        ("positive", LAYER_BASIN_POSITIVE, LAYER_UNIFORM_POSITIVE),
    ):
        attractor = physical_attractors[state]
        rows += _cell_rows(morse_graph, basins[attractor], basin_layer)
        rows += _cell_rows(morse_graph, morse_graph.morse_set(attractor), set_layer)
    return rows


def _coarse_rows(path: Path) -> list[list[float]]:
    """The saved coarse Morse sets, read back by CMGDB and relabelled by layer."""
    rows = []
    for box in CMGDB.LoadMorseSetFile(str(path)):
        *corners, node = box
        rows.append([float(value) for value in corners]
                    + [float(COARSE_LAYER[int(node)])])
    return rows


def _plot_layers(
    rows: list[list[float]],
    bounds,
    *,
    show_ticks: bool = True,
    show_axis_labels: bool = True,
):
    """Draw every layer of the figure in one CMGDB.PlotMorseSets call.

    CMGDB merges each label into the outline of its union, which is what the
    per-cell loop this replaces was approximating: it drew every cell as its own
    rectangle with an edge in the face color, so that the antialiasing seams
    between neighbours closed. One outline has no interior seams to close, and
    costs a few polygons where the cells cost tens of thousands.

    PAINT_ORDER carries the layering the separate collections used to get from
    their z-orders: the merged coarse set under the basins, so it reads as the
    structure the basins are drawn over rather than a patch covering the
    boundary between them, and the attracting sets over both.
    """
    fig, ax = CMGDB.PlotMorseSets(
        rows,
        morse_nodes=PAINT_ORDER,
        clist=LAYER_COLORS,
        fig_w=6.6,
        fig_h=6.4,
        xlim=(float(bounds.lower[0]), float(bounds.upper[0])),
        ylim=(float(bounds.lower[1]), float(bounds.upper[1])),
        axis_labels=show_axis_labels,
        xlabel="$z_1$",
        ylabel="$z_2$",
        fontsize=float(PAPER_RCPARAMS["axes.labelsize"]),
        show=False,
    )
    # CMGDB sizes labels and tick numbers together; the paper style sets them
    # apart, so the tick numbers go back to their own size.
    ax.tick_params(labelsize=float(PAPER_RCPARAMS["xtick.labelsize"]))
    ax.set_aspect("equal")
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(False)
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-sets", type=Path, default=DEFAULT_COARSE_SETS)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output stem for the attraction-basin/coarse-Morse overlay",
    )
    parser.add_argument(
        "--basin-output",
        type=Path,
        default=DEFAULT_BASIN_OUTPUT,
        help="output stem for the attraction basins without the coarse overlay",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help=(
            "directory holding the archived reference inputs "
            "(ci_model_weights.pth, train_data.csv, stable_solutions.csv)"
        ),
    )
    parser.add_argument(
        "--hide-ticks",
        action="store_true",
        help="drop the coordinate tick marks and values",
    )
    parser.add_argument(
        "--hide-axis-labels",
        action="store_true",
        help="drop the z_1 and z_2 labels",
    )
    args = parser.parse_args(argv)

    reference_root = args.reference_root.resolve()
    coarsen.REFERENCE_ROOT = reference_root
    coarsen.REFERENCE_WEIGHTS = reference_root / "ci_model_weights.pth"
    coarsen.REFERENCE_DATA = reference_root / "train_data.csv"

    (
        morse_graph,
        map_graph,
        bounds,
        basins,
        attractors,
        physical_attractors,
        resolution,
    ) = _compute_uniform_basins(args.device)
    ordered_attractors = sorted(
        attractors,
        key=lambda node: float(_attractor_center(morse_graph, node)[0]),
    )
    basin_rows = _basin_rows(morse_graph, basins, physical_attractors)
    coarse_rows = _coarse_rows(args.coarse_sets)

    apply_paper_style()
    basin_fig = _plot_layers(
        basin_rows,
        bounds,
        show_ticks=not args.hide_ticks,
        show_axis_labels=not args.hide_axis_labels,
    )
    basin_written = save_figure(
        basin_fig,
        args.basin_output,
        dpi=180,
        bbox_inches="tight",
        close=True,
    )
    overlay_fig = _plot_layers(
        basin_rows + coarse_rows,
        bounds,
        show_ticks=not args.hide_ticks,
        show_axis_labels=not args.hide_axis_labels,
    )
    overlay_written = save_figure(
        overlay_fig,
        args.output,
        dpi=180,
        bbox_inches="tight",
        close=True,
    )

    metadata = {
        "source": str(coarsen.REFERENCE_ROOT),
        "uniform_grid_resolution": [resolution, resolution],
        "uniform_map_vertices": int(map_graph.num_vertices()),
        "uniform_morse_nodes": int(morse_graph.num_vertices()),
        "uniform_attractors_left_to_right": ordered_attractors,
        "uniform_attractors_by_physical_state": physical_attractors,
        "basin_cell_counts_left_to_right": [
            len(basins[node]) for node in ordered_attractors
        ],
        "basin_cell_counts_by_physical_state": {
            state: len(basins[node])
            for state, node in physical_attractors.items()
        },
        "physical_colors": {
            "negative": CHAFEE_NEGATIVE_COLOR,
            "positive": CHAFEE_POSITIVE_COLOR,
            "coarse_unstable_connecting": CHAFEE_CONNECTING_COLOR,
        },
        "basin_method": "CMGDB.MorseSingletonReachability on the cached cell graph",
        "rendering": {
            "basin_layer": "CMGDB.PlotMorseSets, one merged outline per layer",
            "per_cell_scatter": False,
            "basin_alpha": BASIN_ALPHA,
            "attractor_set_alpha": ATTRACTOR_ALPHA,
        },
        "axis_visibility": {
            "ticks": bool(not args.hide_ticks),
            "latent_coordinate_labels": bool(not args.hide_axis_labels),
            "grid": False,
        },
        "coarse_morse_sets": str(args.coarse_sets),
        "basin_only_outputs": [str(path) for path in basin_written],
        "overlay_outputs": [str(path) for path in overlay_written],
        "outputs": [str(path) for path in (*basin_written, *overlay_written)],
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
