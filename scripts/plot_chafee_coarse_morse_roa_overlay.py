"""Overlay Marcio's coarse Morse representation on his attraction basins.

The basin layer reproduces the uniform 256-by-256 computation in
``archive/marcio/scripts/compute_att_basins_statistics.py``. The foreground
uses the connection-complete coarse Morse sets produced by
``scripts/coarsen_chafee_infante.py`` from Marcio's adaptive cell graph.
Both computations use his data-derived latent bounds.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from latentdynamics.viz import PALETTE
from latentdynamics.viz.style import apply_paper_style, save_figure

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
    return morse_graph, map_graph, bounds, basins, attractors, resolution


def _attractor_center(morse_graph, node: int) -> np.ndarray:
    centers = []
    for cell in morse_graph.morse_set(node):
        lo_x, lo_y, hi_x, hi_y = morse_graph.phase_space_box(cell)
        centers.append(((lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0))
    if not centers:
        raise ValueError(f"uniform attractor node {node} has no cells")
    return np.mean(np.asarray(centers), axis=0)


def _basin_image(morse_graph, basins, ordered_attractors, bounds, resolution):
    image = np.zeros((resolution, resolution, 4), dtype=np.float64)
    cell_size = (bounds.upper - bounds.lower) / resolution
    basin_colors = (PALETTE[0], PALETTE[1])
    for color, attractor in zip(basin_colors, ordered_attractors, strict=True):
        rgb = tuple(int(color[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
        for cell in basins[attractor]:
            lo_x, lo_y, hi_x, hi_y = morse_graph.phase_space_box(cell)
            center = np.asarray([(lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0])
            i, j = np.floor((center - bounds.lower) / cell_size).astype(int)
            if 0 <= i < resolution and 0 <= j < resolution:
                image[j, i] = (*rgb, 0.28)
    return image


def _add_coarse_sets(ax, coarse_sets: np.ndarray) -> None:
    styles = {
        0: (PALETTE[0], 1.0),
        1: (PALETTE[1], 1.0),
        2: (PALETTE[2], 1.0),
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
        collection.set_alpha(alpha)
        collection.set_zorder(3 if label in (0, 1) else 2)
        ax.add_collection(collection)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-sets", type=Path, default=DEFAULT_COARSE_SETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    morse_graph, map_graph, bounds, basins, attractors, resolution = (
        _compute_uniform_basins(args.device)
    )
    ordered_attractors = sorted(
        attractors,
        key=lambda node: float(_attractor_center(morse_graph, node)[0]),
    )
    basin_image = _basin_image(
        morse_graph,
        basins,
        ordered_attractors,
        bounds,
        resolution,
    )
    coarse_sets = np.loadtxt(args.coarse_sets, delimiter=",", ndmin=2)

    apply_paper_style()
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
    _add_coarse_sets(ax, coarse_sets)
    ax.set_xlim(bounds.lower[0], bounds.upper[0])
    ax.set_ylim(bounds.lower[1], bounds.upper[1])
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    written = save_figure(
        fig,
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
        "basin_cell_counts_left_to_right": [
            len(basins[node]) for node in ordered_attractors
        ],
        "coarse_morse_sets": str(args.coarse_sets),
        "outputs": [str(path) for path in written],
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
