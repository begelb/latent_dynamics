"""Render Marcio's attraction basins, with an optional coarse-Morse overlay.

The basin layer reproduces the uniform 256-by-256 computation in
``archive/marcio/scripts/compute_att_basins_statistics.py``. The foreground
uses the connection-complete coarse Morse sets produced by
``scripts/coarsen_chafee_infante.py`` from Marcio's adaptive cell graph.
Both computations use his data-derived latent bounds.

Unlike the archived plotting routine, both outputs draw the uniform basin
partition as one RGBA image. This preserves the exact cell assignment without
the vector seams caused by plotting every cell as a separate square marker.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import CMGDB
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from coarsen_chafee_infante import (
    CODE_ROOT,
    MARCIO_ROOT,
    _load_marcio_model,
    _marcio_bounds,
)
from latentdynamics.viz.style import (
    CHAFEE_CONNECTING_COLOR,
    CHAFEE_NEGATIVE_COLOR,
    CHAFEE_POSITIVE_COLOR,
    apply_paper_style,
    save_figure,
)

sys.path.insert(0, str(MARCIO_ROOT))
from basins_attraction import attractor_basins

DEFAULT_COARSE_SETS = (
    CODE_ROOT / "paper_figures" / "coarsened" / "chafee_infante" / "MG" / "morse_sets"
)
DEFAULT_OUTPUT = (
    CODE_ROOT
    / "paper_figures"
    / "coarsened"
    / "chafee_infante"
    / "morse_roa_overlay"
)
DEFAULT_BASIN_OUTPUT = DEFAULT_OUTPUT.with_name("attractor_basins")

BASIN_ALPHA = 0.35
ATTRACTOR_ALPHA = 1.0


def _compute_uniform_basins(device: str):
    """Reproduce Marcio's 16/16/16 padded uniform-grid basin graph."""
    model = _load_marcio_model(device)
    bounds = _marcio_bounds(model, device)
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
    morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(cmgdb_model)
    attractors = [
        node
        for node in range(int(morse_graph.num_vertices()))
        if not list(morse_graph.adjacencies(node))
    ]
    if len(attractors) != 2:
        raise ValueError(
            f"Marcio uniform basin graph has {len(attractors)} attractors, expected 2"
        )
    basins = attractor_basins(map_graph, morse_graph, attractors)
    stable_roots = np.loadtxt(
        MARCIO_ROOT / "stable_solutions.csv",
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


def _cell_index(morse_graph, cell: int, bounds, cell_size: np.ndarray) -> tuple[int, int]:
    lo_x, lo_y, hi_x, hi_y = morse_graph.phase_space_box(cell)
    center = np.asarray([(lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0])
    return tuple(np.floor((center - bounds.lower) / cell_size).astype(int))


def _paint_cells(
    image: np.ndarray,
    morse_graph,
    cells,
    *,
    bounds,
    cell_size: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    rgb = tuple(int(color[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    resolution_y, resolution_x = image.shape[:2]
    for cell in cells:
        i, j = _cell_index(morse_graph, cell, bounds, cell_size)
        if 0 <= i < resolution_x and 0 <= j < resolution_y:
            image[j, i] = (*rgb, alpha)


def _basin_image(morse_graph, basins, physical_attractors, bounds, resolution):
    """Return one exact RGBA layer with opaque attracting Morse-set cells.

    ``physical_attractors`` is ordered as negative then positive using the two
    saved steady solutions, not inferred from latent position or node id.
    """
    if len(physical_attractors) != 2:
        raise ValueError(
            f"expected negative and positive attractors, got {len(physical_attractors)}"
        )
    image = np.zeros((resolution, resolution, 4), dtype=np.float64)
    cell_size = (bounds.upper - bounds.lower) / resolution
    physical_colors = (CHAFEE_NEGATIVE_COLOR, CHAFEE_POSITIVE_COLOR)
    for color, attractor in zip(physical_colors, physical_attractors, strict=True):
        _paint_cells(
            image,
            morse_graph,
            basins[attractor],
            bounds=bounds,
            cell_size=cell_size,
            color=color,
            alpha=BASIN_ALPHA,
        )
        _paint_cells(
            image,
            morse_graph,
            morse_graph.morse_set(attractor),
            bounds=bounds,
            cell_size=cell_size,
            color=color,
            alpha=ATTRACTOR_ALPHA,
        )
    return image


def _add_coarse_sets(ax, coarse_sets: np.ndarray) -> None:
    styles = {
        0: (CHAFEE_POSITIVE_COLOR, 1.0),
        1: (CHAFEE_NEGATIVE_COLOR, 1.0),
        2: (CHAFEE_CONNECTING_COLOR, 1.0),
    }
    for label in (2, 0, 1):
        rows = coarse_sets[coarse_sets[:, -1].astype(int) == label]
        patches = [
            Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y)
            for lo_x, lo_y, hi_x, hi_y, _ in rows
        ]
        if not patches:
            continue
        color, alpha = styles[label]
        collection = PatchCollection(patches, match_original=False)
        collection.set_facecolor(color)
        collection.set_edgecolor("none")
        collection.set_linewidth(0)
        collection.set_antialiased(False)
        collection.set_rasterized(True)
        collection.set_alpha(alpha)
        collection.set_zorder(3 if label in (0, 1) else 2)
        ax.add_collection(collection)


def _style_axes(
    ax,
    *,
    show_ticks: bool = False,
    show_axis_labels: bool = False,
) -> None:
    ax.set_xlabel("$z_1$" if show_axis_labels else "")
    ax.set_ylabel("$z_2$" if show_axis_labels else "")
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(False)


def _plot_basin_image(
    basin_image: np.ndarray,
    bounds,
    *,
    coarse_sets: np.ndarray | None = None,
    show_ticks: bool = False,
    show_axis_labels: bool = False,
):
    """Plot one RGBA basin image, optionally with the adaptive coarse sets."""
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    ax.imshow(
        basin_image,
        origin="lower",
        extent=(
            bounds.lower[0],
            bounds.upper[0],
            bounds.lower[1],
            bounds.upper[1],
        ),
        interpolation="nearest",
        aspect="equal",
    )
    if coarse_sets is not None:
        _add_coarse_sets(ax, coarse_sets)
    ax.set_xlim(bounds.lower[0], bounds.upper[0])
    ax.set_ylim(bounds.lower[1], bounds.upper[1])
    _style_axes(
        ax,
        show_ticks=show_ticks,
        show_axis_labels=show_axis_labels,
    )
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
        "--show-ticks",
        action="store_true",
        help="show coordinate tick marks and values (hidden in the paper-ready default)",
    )
    parser.add_argument(
        "--show-axis-labels",
        action="store_true",
        help="show z_1 and z_2 labels (hidden in the paper-ready default)",
    )
    args = parser.parse_args(argv)

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
    basin_image = _basin_image(
        morse_graph,
        basins,
        (
            physical_attractors["negative"],
            physical_attractors["positive"],
        ),
        bounds,
        resolution,
    )
    coarse_sets = np.loadtxt(args.coarse_sets, delimiter=",", ndmin=2)

    apply_paper_style()
    basin_fig = _plot_basin_image(
        basin_image,
        bounds,
        show_ticks=args.show_ticks,
        show_axis_labels=args.show_axis_labels,
    )
    basin_written = save_figure(
        basin_fig,
        args.basin_output,
        dpi=180,
        bbox_inches="tight",
        close=True,
    )
    overlay_fig = _plot_basin_image(
        basin_image,
        bounds,
        coarse_sets=coarse_sets,
        show_ticks=args.show_ticks,
        show_axis_labels=args.show_axis_labels,
    )
    overlay_written = save_figure(
        overlay_fig,
        args.output,
        dpi=180,
        bbox_inches="tight",
        close=True,
    )

    metadata = {
        "source": "archive/marcio/scripts",
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
        "rendering": {
            "basin_layer": "single RGBA image via imshow",
            "per_cell_scatter": False,
            "basin_alpha": BASIN_ALPHA,
            "attractor_set_alpha": ATTRACTOR_ALPHA,
        },
        "axis_visibility": {
            "ticks": bool(args.show_ticks),
            "latent_coordinate_labels": bool(args.show_axis_labels),
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
