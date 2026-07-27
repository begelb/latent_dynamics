"""Compute uniform-grid Chafee--Infante Morse and quotient backups.

This reproduces Marcio's padded ``16/16/16`` computation on the uniform
``256 x 256`` grid.  The uniform graph has many more recurrent components than
the adaptive graph, so the faithful coarse representation retains its two
minimal nodes and collapses every other recurrent component into ``M(1)``.

The generated figures are backups only.  They do not replace or modify any
manuscript references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import CMGDB
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from coarsen_chafee_infante import (
    CODE_ROOT,
    MARCIO_COARSE_PALETTE,
    MARCIO_DATA,
    MARCIO_PALETTE,
    MARCIO_WEIGHTS,
    PROJECT_ROOT,
    _compute_marcio_graph,
    _parsed_graph_from_live,
)
from latentdynamics.analysis.morse_coarsening import (
    _backward_reachable,
    _forward_reachable,
    _reverse_csr,
    _transitive_reduction,
    compute_connection_complete_morse_sets,
    write_connection_complete_morse_sets,
    write_morse_graph_dot,
)
from latentdynamics.analysis.morse_graph_parser import MorseGraph
from latentdynamics.viz.morse_plots import render_morse_graph_from_dot
from latentdynamics.viz.style import (
    LATENT_FIG_WIDTH_IN,
    apply_paper_style,
    save_latent_figure,
)
from plot_chafee_coarse_morse_roa_overlay import _compute_uniform_basins

DEFAULT_OUTPUT = (
    CODE_ROOT
    / "paper_figures"
    / "coarsened"
    / "chafee_infante_uniform_s16"
)
DEFAULT_PAPER_OUTPUT = (
    PROJECT_ROOT
    / "paper"
    / "figures"
    / "chafee_infante"
    / "backup_uniform_s16"
)
ZERO_INDEX_COLOR = "#bdbdbd"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_signature(morse_graph, node: int) -> frozenset[tuple[float, ...]]:
    return frozenset(
        tuple(float(value) for value in morse_graph.phase_space_box(int(cell)))
        for cell in morse_graph.morse_set(node)
    )


def _match_nonzero_uniform_nodes(uniform_graph) -> dict[int, int]:
    """Match the seven nonzero-index uniform nodes to adaptive node ids."""
    adaptive_graph, _, _ = _compute_marcio_graph("cpu")
    adaptive_signatures = {
        _box_signature(adaptive_graph, node): node
        for node in range(int(adaptive_graph.num_vertices()))
    }

    matched: dict[int, int] = {}
    for node in range(int(uniform_graph.num_vertices())):
        annotations = tuple(uniform_graph.annotations(node))
        if annotations == ("0", "0", "0"):
            continue
        signature = _box_signature(uniform_graph, node)
        if signature not in adaptive_signatures:
            raise ValueError(
                f"uniform nonzero-index node {node} does not match an adaptive Morse set"
            )
        matched[node] = int(adaptive_signatures[signature])

    expected = set(range(int(adaptive_graph.num_vertices())))
    if set(matched.values()) != expected:
        raise ValueError(
            "uniform/adaptive nonzero-index matching is not bijective: "
            f"matched adaptive nodes {sorted(matched.values())}, expected {sorted(expected)}"
        )
    return matched


def _rgba_image(
    morse_graph,
    cells_by_label: dict[int, np.ndarray],
    colors: dict[int, str],
    bounds,
    resolution: int,
) -> np.ndarray:
    image = np.zeros((resolution, resolution, 4), dtype=np.float64)
    cell_size = (bounds.upper - bounds.lower) / resolution
    for label, cells in cells_by_label.items():
        rgba = to_rgba(colors[label])
        for raw_cell in cells:
            lo_x, lo_y, hi_x, hi_y = morse_graph.phase_space_box(int(raw_cell))
            center = np.asarray(
                [(lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0],
                dtype=np.float64,
            )
            i, j = np.floor((center - bounds.lower) / cell_size).astype(int)
            if not (0 <= i < resolution and 0 <= j < resolution):
                raise ValueError(
                    f"cell {int(raw_cell)} center {center.tolist()} lies outside the uniform image"
                )
            if image[j, i, 3] != 0.0:
                raise ValueError(f"uniform image cell ({i}, {j}) received multiple labels")
            image[j, i] = rgba
    return image


def _save_uniform_image(
    image: np.ndarray,
    bounds,
    destination: Path,
    *,
    background: np.ndarray | None = None,
) -> list[Path]:
    apply_paper_style()
    fig, ax = plt.subplots(
        figsize=(LATENT_FIG_WIDTH_IN, LATENT_FIG_WIDTH_IN),
        layout="constrained",
    )
    extent = (
        float(bounds.lower[0]),
        float(bounds.upper[0]),
        float(bounds.lower[1]),
        float(bounds.upper[1]),
    )
    if background is not None:
        ax.imshow(
            background,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
        )
    ax.imshow(
        image,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xlim(bounds.lower[0], bounds.upper[0])
    ax.set_ylim(bounds.lower[1], bounds.upper[1])
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_xticks([])
    ax.set_yticks([])
    return save_latent_figure(fig, destination, close=True)


def _filtered_nonzero_graph(
    fine_graph: MorseGraph,
    uniform_to_adaptive: dict[int, int],
) -> MorseGraph:
    retained = sorted(uniform_to_adaptive)
    order = {
        source: {
            target
            for target in retained
            if target != source and target in fine_graph.descendants[source]
        }
        for source in retained
    }
    reduced = _transitive_reduction(retained, order)
    canonical_edges: dict[int, list[int]] = {}
    for source, targets in reduced.items():
        canonical_source = uniform_to_adaptive[source]
        canonical_edges[canonical_source] = sorted(
            uniform_to_adaptive[target] for target in targets
        )

    labels = {}
    colors = {}
    for uniform_node, adaptive_node in uniform_to_adaptive.items():
        labels[adaptive_node] = fine_graph.labels[uniform_node]
        colors[adaptive_node] = f"{MARCIO_PALETTE[adaptive_node]}ff"
    return MorseGraph(
        nodes=sorted(uniform_to_adaptive.values()),
        edges=canonical_edges,
        colors=colors,
        labels=labels,
    )


def _basin_background(
    morse_graph,
    basins: dict[int, list[int]],
    attractor_to_coarse: dict[int, int],
    bounds,
    resolution: int,
) -> np.ndarray:
    image = np.zeros((resolution, resolution, 4), dtype=np.float64)
    cell_size = (bounds.upper - bounds.lower) / resolution
    for attractor, basin_cells in basins.items():
        coarse = attractor_to_coarse[attractor]
        rgba = np.asarray(to_rgba(MARCIO_COARSE_PALETTE[coarse]), dtype=np.float64)
        rgba[3] = 0.35
        for raw_cell in basin_cells:
            lo_x, lo_y, hi_x, hi_y = morse_graph.phase_space_box(int(raw_cell))
            center = np.asarray(
                [(lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0],
                dtype=np.float64,
            )
            i, j = np.floor((center - bounds.lower) / cell_size).astype(int)
            if image[j, i, 3] != 0.0:
                raise ValueError(f"uniform basin cell ({i}, {j}) has multiple basin labels")
            image[j, i] = rgba
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-paper-copy",
        action="store_true",
        help="generate code artifacts without copying backup PDFs into paper/",
    )
    args = parser.parse_args()

    (
        uniform_graph,
        map_graph,
        bounds,
        basins,
        attractors,
        resolution,
    ) = _compute_uniform_basins(args.device)
    nodes = list(range(int(uniform_graph.num_vertices())))
    if resolution != 256 or int(map_graph.num_vertices()) != resolution**2:
        raise ValueError(
            f"unexpected uniform grid: resolution={resolution}, cells={int(map_graph.num_vertices())}"
        )
    if len(nodes) != 137 or len(attractors) != 2:
        raise ValueError(
            f"unexpected uniform Morse graph: nodes={len(nodes)}, attractors={attractors}"
        )

    uniform_to_adaptive = _match_nonzero_uniform_nodes(uniform_graph)
    adaptive_to_uniform = {
        adaptive: uniform for uniform, adaptive in uniform_to_adaptive.items()
    }
    positive_attractor = adaptive_to_uniform[0]
    negative_attractor = adaptive_to_uniform[1]
    if set(attractors) != {positive_attractor, negative_attractor}:
        raise ValueError(
            "the adaptive-matched stable nodes do not equal the uniform minimal nodes"
        )

    projection = {
        node: (
            0
            if node == positive_attractor
            else 1
            if node == negative_attractor
            else 2
        )
        for node in nodes
    }
    completed = compute_connection_complete_morse_sets(
        map_graph,
        uniform_graph,
        projection,
    )
    if completed.overlaps:
        raise ValueError(f"uniform coarse Morse sets overlap: {completed.overlaps}")

    fine_graph = _parsed_graph_from_live(uniform_graph)
    fine_colors = {
        node: (
            MARCIO_PALETTE[uniform_to_adaptive[node]]
            if node in uniform_to_adaptive
            else ZERO_INDEX_COLOR
        )
        for node in nodes
    }
    fine_graph.colors = {node: f"{color}ff" for node, color in fine_colors.items()}

    fine_dir = args.output / "fine"
    fine_mg_dir = fine_dir / "MG"
    fine_mg_dir.mkdir(parents=True, exist_ok=True)
    fine_dot = write_morse_graph_dot(fine_graph, fine_mg_dir / "morse_graph")
    CMGDB.SaveMorseSets(uniform_graph, str(fine_mg_dir / "morse_sets"))
    fine_graph_paths = render_morse_graph_from_dot(
        fine_dot,
        fine_dir,
        basename="morse_graph",
        formats=("pdf", "png"),
    )
    fine_cells = {
        node: np.fromiter(
            (int(cell) for cell in uniform_graph.morse_set(node)),
            dtype=np.int64,
        )
        for node in nodes
    }
    fine_image = _rgba_image(
        uniform_graph,
        fine_cells,
        fine_colors,
        bounds,
        resolution,
    )
    fine_set_paths = _save_uniform_image(
        fine_image,
        bounds,
        fine_dir / "morse_sets",
    )

    filtered_graph = _filtered_nonzero_graph(fine_graph, uniform_to_adaptive)
    filtered_dir = args.output / "fine_nonzero_filtered"
    filtered_dot = write_morse_graph_dot(
        filtered_graph,
        filtered_dir / "MG" / "morse_graph",
    )
    filtered_graph_paths = render_morse_graph_from_dot(
        filtered_dot,
        filtered_dir,
        basename="morse_graph",
        formats=("pdf", "png"),
    )
    filtered_singleton_projection = dict(uniform_to_adaptive)
    filtered_sets = compute_connection_complete_morse_sets(
        map_graph,
        uniform_graph,
        filtered_singleton_projection,
    )
    write_connection_complete_morse_sets(
        uniform_graph,
        filtered_sets,
        filtered_dir / "MG" / "morse_sets",
    )
    filtered_colors = {
        adaptive: MARCIO_PALETTE[adaptive]
        for adaptive in sorted(uniform_to_adaptive.values())
    }
    filtered_image = _rgba_image(
        uniform_graph,
        filtered_sets.cells,
        filtered_colors,
        bounds,
        resolution,
    )
    filtered_set_paths = _save_uniform_image(
        filtered_image,
        bounds,
        filtered_dir / "morse_sets",
    )

    coarse_graph = MorseGraph(
        nodes=[0, 1, 2],
        edges={2: [0, 1]},
        colors={
            0: f"{MARCIO_COARSE_PALETTE[0]}ff",
            1: f"{MARCIO_COARSE_PALETTE[1]}ff",
            2: f"{MARCIO_COARSE_PALETTE[2]}ff",
        },
        labels={0: "M(0+)", 1: "M(0-)", 2: "M(1)"},
    )
    coarse_dir = args.output / "coarse"
    coarse_mg_dir = coarse_dir / "MG"
    coarse_dot = write_morse_graph_dot(
        coarse_graph,
        coarse_mg_dir / "morse_graph",
    )
    write_connection_complete_morse_sets(
        uniform_graph,
        completed,
        coarse_mg_dir / "morse_sets",
    )
    coarse_graph_paths = render_morse_graph_from_dot(
        coarse_dot,
        coarse_dir,
        basename="morse_graph",
        formats=("pdf", "png"),
    )
    coarse_colors = dict(enumerate(MARCIO_COARSE_PALETTE))
    coarse_image = _rgba_image(
        uniform_graph,
        completed.cells,
        coarse_colors,
        bounds,
        resolution,
    )
    coarse_set_paths = _save_uniform_image(
        coarse_image,
        bounds,
        coarse_dir / "morse_sets",
    )

    filtered_coarse_projection = {
        uniform: (
            0
            if adaptive == 0
            else 1
            if adaptive == 1
            else 2
        )
        for uniform, adaptive in uniform_to_adaptive.items()
    }
    filtered_completed = compute_connection_complete_morse_sets(
        map_graph,
        uniform_graph,
        filtered_coarse_projection,
    )
    if filtered_completed.overlaps:
        raise ValueError(
            "filtered nonzero-index coarse Morse sets overlap: "
            f"{filtered_completed.overlaps}"
        )
    filtered_coarse_dir = args.output / "coarse_nonzero_filtered"
    filtered_coarse_dot = write_morse_graph_dot(
        coarse_graph,
        filtered_coarse_dir / "MG" / "morse_graph",
    )
    write_connection_complete_morse_sets(
        uniform_graph,
        filtered_completed,
        filtered_coarse_dir / "MG" / "morse_sets",
    )
    filtered_coarse_graph_paths = render_morse_graph_from_dot(
        filtered_coarse_dot,
        filtered_coarse_dir,
        basename="morse_graph",
        formats=("pdf", "png"),
    )
    filtered_coarse_image = _rgba_image(
        uniform_graph,
        filtered_completed.cells,
        coarse_colors,
        bounds,
        resolution,
    )
    filtered_coarse_set_paths = _save_uniform_image(
        filtered_coarse_image,
        bounds,
        filtered_coarse_dir / "morse_sets",
    )

    attractor_to_coarse = {
        positive_attractor: 0,
        negative_attractor: 1,
    }
    basin_background = _basin_background(
        uniform_graph,
        basins,
        attractor_to_coarse,
        bounds,
        resolution,
    )
    overlay_paths = _save_uniform_image(
        coarse_image,
        bounds,
        args.output / "coarse_morse_roa_overlay",
        background=basin_background,
    )
    filtered_overlay_paths = _save_uniform_image(
        filtered_coarse_image,
        bounds,
        args.output / "coarse_nonzero_filtered_morse_roa_overlay",
        background=basin_background,
    )

    fiber = {node for node in nodes if projection[node] == 2}
    maxima = sorted(
        node
        for node in fiber
        if not any(
            node in fine_graph.descendants[other]
            for other in fiber
            if other != node
        )
    )
    minima = sorted(
        node
        for node in fiber
        if not any(
            other in fine_graph.descendants[node]
            for other in fiber
            if other != node
        )
    )
    n_vertices = int(map_graph.num_vertices())
    max_cells = np.unique(
        np.concatenate(
            [
                np.fromiter(
                    (int(cell) for cell in uniform_graph.morse_set(node)),
                    dtype=np.int64,
                )
                for node in maxima
            ]
        )
    )
    min_cells = np.unique(
        np.concatenate(
            [
                np.fromiter(
                    (int(cell) for cell in uniform_graph.morse_set(node)),
                    dtype=np.int64,
                )
                for node in minima
            ]
        )
    )
    reverse_pointers, reverse_neighbors = _reverse_csr(map_graph, n_vertices)
    endpoint_completion = np.flatnonzero(
        _forward_reachable(map_graph, max_cells, n_vertices)
        & _backward_reachable(
            reverse_pointers,
            reverse_neighbors,
            min_cells,
            n_vertices,
        )
    )
    if not np.array_equal(endpoint_completion, completed.cells[2]):
        raise ValueError(
            "all-fiber and top-down/bottom-up uniform completions do not agree"
        )

    basin_overlaps = {
        str(coarse): {
            str(attractor): int(
                np.intersect1d(
                    cells,
                    np.asarray(basins[attractor], dtype=np.int64),
                ).size
            )
            for attractor in attractors
        }
        for coarse, cells in completed.cells.items()
    }
    annotation_histogram = Counter(
        tuple(uniform_graph.annotations(node)) for node in nodes
    )
    recurrent_counts = {
        str(coarse): int(
            completed.cells[coarse].size
            - completed.connection_cells[coarse].size
        )
        for coarse in completed.cells
    }
    connection_counts = {
        str(coarse): int(completed.connection_cells[coarse].size)
        for coarse in completed.cells
    }
    filtered_recurrent_counts = {
        str(coarse): int(
            filtered_completed.cells[coarse].size
            - filtered_completed.connection_cells[coarse].size
        )
        for coarse in filtered_completed.cells
    }
    filtered_connection_counts = {
        str(coarse): int(filtered_completed.connection_cells[coarse].size)
        for coarse in filtered_completed.cells
    }

    manifest = {
        "source": "archive/marcio/scripts/compute_att_basins_statistics.py",
        "inputs": {
            "weights": str(MARCIO_WEIGHTS.relative_to(PROJECT_ROOT)),
            "weights_sha256": _sha256(MARCIO_WEIGHTS),
            "data": str(MARCIO_DATA.relative_to(PROJECT_ROOT)),
            "data_sha256": _sha256(MARCIO_DATA),
        },
        "bounds": {
            "lower": bounds.lower.tolist(),
            "upper": bounds.upper.tolist(),
            "rule": "encoded transition extrema plus 10 percent per-axis padding",
        },
        "cmgdb": {
            "subdiv_init": 16,
            "subdiv_min": 16,
            "subdiv_max": 16,
            "subdiv_limit": 10000,
            "padding": True,
            "resolution": [resolution, resolution],
            "cells": n_vertices,
        },
        "fine_graph": {
            "nodes": len(nodes),
            "reduced_edges": len(list(uniform_graph.edges())),
            "attractors": attractors,
            "annotation_histogram": {
                ",".join(annotation): count
                for annotation, count in sorted(annotation_histogram.items())
            },
            "uniform_to_adaptive_nonzero_nodes": {
                str(uniform): adaptive
                for uniform, adaptive in sorted(uniform_to_adaptive.items())
            },
            "zero_index_nodes": len(nodes) - len(uniform_to_adaptive),
            "zero_index_color": ZERO_INDEX_COLOR,
            "filtered_graph_note": (
                "The seven-node graph is filtered to the nonzero-index components "
                "and is not the full uniform Morse graph."
            ),
        },
        "quotient": {
            "projection": {
                str(node): coarse for node, coarse in sorted(projection.items())
            },
            "edges": {"2": [0, 1]},
            "labels": {"0": "M(0+)", "1": "M(0-)", "2": "M(1)"},
            "fiber_2_maxima": maxima,
            "fiber_2_minima": minima,
            "top_down_bottom_up_verified": True,
            "recurrent_cell_counts": recurrent_counts,
            "connection_cell_counts": connection_counts,
            "completed_cell_counts": {
                str(coarse): int(cells.size)
                for coarse, cells in completed.cells.items()
            },
            "overlaps": len(completed.overlaps),
            "uniform_basin_overlaps": basin_overlaps,
        },
        "filtered_nonzero_view": {
            "fine_nodes": len(uniform_to_adaptive),
            "omitted_zero_index_nodes": len(nodes) - len(uniform_to_adaptive),
            "note": (
                "This is a filtered view, not a quotient of the complete "
                "137-node uniform Morse graph."
            ),
            "coarse_projection": {
                str(node): coarse
                for node, coarse in sorted(filtered_coarse_projection.items())
            },
            "coarse_recurrent_cell_counts": filtered_recurrent_counts,
            "coarse_connection_cell_counts": filtered_connection_counts,
            "coarse_completed_cell_counts": {
                str(coarse): int(cells.size)
                for coarse, cells in filtered_completed.cells.items()
            },
            "coarse_overlaps": len(filtered_completed.overlaps),
        },
        "figures": {
            "fine_graph": [str(path) for path in fine_graph_paths],
            "fine_sets": [str(path) for path in fine_set_paths],
            "filtered_nonzero_graph": [
                str(path) for path in filtered_graph_paths
            ],
            "filtered_nonzero_sets": [
                str(path) for path in filtered_set_paths
            ],
            "coarse_graph": [str(path) for path in coarse_graph_paths],
            "coarse_sets": [str(path) for path in coarse_set_paths],
            "filtered_nonzero_coarse_graph": [
                str(path) for path in filtered_coarse_graph_paths
            ],
            "filtered_nonzero_coarse_sets": [
                str(path) for path in filtered_coarse_set_paths
            ],
            "same_grid_overlay": [str(path) for path in overlay_paths],
            "filtered_nonzero_same_grid_overlay": [
                str(path) for path in filtered_overlay_paths
            ],
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    if not args.no_paper_copy:
        args.paper_output.mkdir(parents=True, exist_ok=True)
        paper_figures = {
            fine_dir / "morse_graph.pdf": "ci_morse_graph_uniform_s16_all.pdf",
            fine_dir / "morse_sets.pdf": "ci_morse_sets_uniform_s16_all.pdf",
            filtered_dir / "morse_graph.pdf": (
                "ci_morse_graph_uniform_s16_nonzero_filtered.pdf"
            ),
            filtered_dir / "morse_sets.pdf": (
                "ci_morse_sets_uniform_s16_nonzero_filtered.pdf"
            ),
            coarse_dir / "morse_graph.pdf": (
                "ci_coarse_morse_graph_uniform_s16_all.pdf"
            ),
            coarse_dir / "morse_sets.pdf": (
                "ci_coarse_morse_sets_uniform_s16_all.pdf"
            ),
            filtered_coarse_dir / "morse_graph.pdf": (
                "ci_coarse_morse_graph_uniform_s16_nonzero_filtered.pdf"
            ),
            filtered_coarse_dir / "morse_sets.pdf": (
                "ci_coarse_morse_sets_uniform_s16_nonzero_filtered.pdf"
            ),
            args.output / "coarse_morse_roa_overlay.pdf": (
                "ci_coarse_morse_roa_uniform_s16_all.pdf"
            ),
            args.output / "coarse_nonzero_filtered_morse_roa_overlay.pdf": (
                "ci_coarse_morse_roa_uniform_s16_nonzero_filtered.pdf"
            ),
        }
        for source, filename in paper_figures.items():
            shutil.copy2(source, args.paper_output / filename)

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
