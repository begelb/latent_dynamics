"""Coarsen the saved three-dimensional Chafee--Infante Morse computation.

The two minimal Morse nodes stay distinct and fine nodes 2--10 form the
coarse unstable node M(1).  As in the two-dimensional construction, the
coarse cell set is connection-complete: it contains the recurrent cells in
the quotient fiber and every cached-map-graph cell on a directed path between
two of those recurrent cells.

The adaptive CMGDB computation is replayed from its persisted lookup table.
No neural network is evaluated by the CMGDB callback or by the path
completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import CMGDB
import numpy as np

from latentdynamics.analysis.hierarchical_precomputed import (
    HierarchicalPrecomputedBoxMap,
)
from latentdynamics.analysis.morse import LatentBounds
from latentdynamics.analysis.morse_coarsening import (
    coarsen_morse_graph,
    compute_connection_complete_morse_sets,
    write_connection_complete_morse_sets,
    write_morse_graph_dot,
)
from latentdynamics.analysis.morse_graph_parser import MorseGraph

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_ROOT = CODE_ROOT / "output" / "chafee_latent_dimension_study"
DEFAULT_RUN = DEFAULT_STUDY_ROOT / "latent_3d" / "seed_0"
DEFAULT_OUTPUT = DEFAULT_RUN / "MG_adaptive_coarse_marcio"

FINE_NODES = tuple(range(11))
MERGED_FINE_NODES = frozenset(range(2, 11))
EXPECTED_FINE_EDGES = {
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (4, 2),
    (4, 3),
    (5, 4),
    (6, 5),
    (7, 2),
    (7, 3),
    (8, 7),
    (9, 8),
    (10, 6),
    (10, 9),
}
IDENTITY_PROJECTION = {
    **{0: 0, 1: 1},
    **dict.fromkeys(MERGED_FINE_NODES, 2),
}
IDENTITY_LABELS = {
    frozenset({0}): "M(0⁻)",
    frozenset({1}): "M(0⁺)",
    MERGED_FINE_NODES: "M(1)",
}
IDENTITY_COLORS = {
    0: "#ffb000ff",
    1: "#dc267fff",
    2: "#7f7f7fff",
}
SEMANTIC_DISPLAY_RELABEL = {
    "fine_to_display": {
        "0": 1,
        "1": 0,
        **{str(node): 2 for node in MERGED_FINE_NODES},
    },
    "display_labels": {
        "0": "M(0⁺)",
        "1": "M(0⁻)",
        "2": "M(1)",
    },
    "display_colors": {
        "0": "#1f77b4ff",
        "1": "#e6550dff",
        "2": "#7f7f7fff",
    },
    "note": (
        "Optional display relabel matching the archived 2-D semantic palette. "
        "The canonical coarse data retain the 3-D fine-node identities."
    ),
}

SUBDIV_INIT = 21
SUBDIV_MIN = 24
SUBDIV_MAX = 33
SUBDIV_LIMIT = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bounds(path: Path) -> LatentBounds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lower = np.asarray(payload["lower"], dtype=np.float64)
    upper = np.asarray(payload["upper"], dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(lower < upper):
        raise ValueError(f"invalid three-dimensional bounds in {path}")
    return LatentBounds(lower=lower, upper=upper)


def _live_edges(morse_graph: Any) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, target in morse_graph.edges()
    }


def _root_owner(
    morse_graph: Any,
    point: np.ndarray,
) -> int:
    owners: list[int] = []
    for node in FINE_NODES:
        for raw_cell in morse_graph.morse_set(node):
            box = np.asarray(
                morse_graph.phase_space_box(int(raw_cell)),
                dtype=np.float64,
            )
            if box.shape != (6,):
                raise ValueError(
                    f"phase_space_box returned shape {box.shape}; expected (6,)"
                )
            if np.all(point >= box[:3]) and np.all(point <= box[3:]):
                owners.append(node)
                break
    if len(owners) != 1:
        raise ValueError(
            f"encoded stable root {point.tolist()} belongs to fine nodes {owners}; "
            "expected exactly one"
        )
    return owners[0]


def _width_distribution(morse_sets_path: Path) -> dict[str, Any]:
    data = np.loadtxt(
        morse_sets_path,
        delimiter=",",
        ndmin=2,
        dtype=np.float64,
    )
    if data.shape[1] != 7:
        raise ValueError(
            f"{morse_sets_path} has {data.shape[1]} columns; expected 7"
        )
    widths = data[:, 3:6] - data[:, :3]
    if np.any(widths <= 0.0):
        raise ValueError("coarse Morse boxes must have strictly positive widths")
    terminal_widths = widths.min(axis=0)
    scales = np.rint(widths / terminal_widths).astype(np.int64)
    reconstructed = scales * terminal_widths
    if not np.allclose(widths, reconstructed, rtol=1e-11, atol=1e-13):
        raise ValueError(
            "coarse Morse boxes are not aligned to integer multiples of the "
            "terminal cell widths"
        )
    unique, counts = np.unique(scales, axis=0, return_counts=True)
    return {
        "terminal_widths": terminal_widths.tolist(),
        "uniform_terminal_grid": bool(unique.shape[0] == 1),
        "scale_distribution": [
            {
                "scale": row.tolist(),
                "widths": (row * terminal_widths).tolist(),
                "count": int(count),
            }
            for row, count in zip(unique, counts, strict=True)
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cmgdb-max-vertices",
        type=int,
        default=2**24,
    )
    parser.add_argument(
        "--cmgdb-max-edges",
        type=int,
        default=1_200_000_000,
    )
    parser.add_argument(
        "--cmgdb-reserve-edges",
        type=int,
        default=300_000_000,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing coarse output: {output}"
        )

    fine_dir = run / "MG_adaptive"
    fine_dot = fine_dir / "morse_graph"
    fine_sets = fine_dir / "morse_sets"
    bounds_path = run / "bounds.json"
    table_path = run / "precomputed_level24_to33"
    roots_path = run / "basin_statistics.json"
    required = (fine_dot, fine_sets, bounds_path, table_path, roots_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing 3-D study artifacts: {missing}")

    native = getattr(CMGDB, "MorseDirectedPathCells", None)
    if not callable(native):
        raise RuntimeError(
            "CMGDB.MorseDirectedPathCells is unavailable; rebuild the bundled "
            "CMGDB extension. The large 3-D computation will not fall back to "
            "Python reverse-graph construction."
        )

    os.environ["CMGDB_MAPGRAPH_MAX_VERTICES"] = str(args.cmgdb_max_vertices)
    os.environ["CMGDB_MAPGRAPH_MAX_EDGES"] = str(args.cmgdb_max_edges)
    os.environ["CMGDB_MAPGRAPH_RESERVE_EDGES"] = str(args.cmgdb_reserve_edges)

    bounds = _load_bounds(bounds_path)
    lookup = HierarchicalPrecomputedBoxMap.load(table_path, mmap_mode="r")
    model = CMGDB.Model(
        SUBDIV_MIN,
        SUBDIV_MAX,
        SUBDIV_INIT,
        SUBDIV_LIMIT,
        bounds.lower.tolist(),
        bounds.upper.tolist(),
        lookup,
    )
    if not hasattr(model, "set_batch_map"):
        raise RuntimeError("CMGDB.Model.set_batch_map is required")
    model.set_batch_map(lookup.batch)

    cmgdb_started = time.perf_counter()
    live_morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    cmgdb_seconds = time.perf_counter() - cmgdb_started
    has_cache = getattr(map_graph, "has_cache", None)
    if not callable(has_cache) or not bool(has_cache()):
        raise RuntimeError(
            "CMGDB did not retain the batched map-graph cache; refusing "
            "on-demand callbacks"
        )
    if int(live_morse_graph.num_vertices()) != len(FINE_NODES):
        raise ValueError(
            "live adaptive graph changed: "
            f"expected 11 nodes, got {int(live_morse_graph.num_vertices())}"
        )
    live_edges = _live_edges(live_morse_graph)
    if live_edges != EXPECTED_FINE_EDGES:
        raise ValueError(
            "live adaptive graph changed: "
            f"expected edges {sorted(EXPECTED_FINE_EDGES)}, got {sorted(live_edges)}"
        )

    saved_graph = MorseGraph.from_dot(fine_dot)
    saved_edges = {
        (source, target)
        for source, targets in saved_graph.edges.items()
        for target in targets
    }
    if saved_graph.nodes != list(FINE_NODES) or saved_edges != live_edges:
        raise ValueError(
            "saved adaptive DOT does not match the live lookup-only recomputation"
        )

    quotient = coarsen_morse_graph(
        saved_graph,
        [MERGED_FINE_NODES],
        labels=IDENTITY_LABELS,
    )
    if quotient.projection != IDENTITY_PROJECTION:
        raise ValueError(
            f"unexpected quotient projection: {quotient.projection}"
        )
    if quotient.graph.edges != {2: [0, 1]}:
        raise ValueError(
            f"unexpected quotient edges: {quotient.graph.edges}"
        )
    quotient.graph.colors.update(IDENTITY_COLORS)

    completion_started = time.perf_counter()
    completed = compute_connection_complete_morse_sets(
        map_graph,
        live_morse_graph,
        quotient.projection,
    )
    completion_seconds = time.perf_counter() - completion_started
    if completed.overlaps:
        examples = list(completed.overlaps.items())[:10]
        raise ValueError(
            "connection-complete coarse Morse sets overlap; "
            f"examples={examples}"
        )

    output.mkdir(parents=True)
    cells_dir = output / "cells"
    cells_dir.mkdir()
    write_morse_graph_dot(quotient.graph, output / "morse_graph")
    write_connection_complete_morse_sets(
        live_morse_graph,
        completed,
        output / "morse_sets",
    )
    for coarse in sorted(completed.cells):
        np.save(cells_dir / f"coarse_{coarse}_cells.npy", completed.cells[coarse])
        np.save(
            cells_dir / f"coarse_{coarse}_connection_cells.npy",
            completed.connection_cells[coarse],
        )

    recurrent_counts = {
        str(coarse): int(
            completed.cells[coarse].size
            - completed.connection_cells[coarse].size
        )
        for coarse in sorted(completed.cells)
    }
    connection_counts = {
        str(coarse): int(completed.connection_cells[coarse].size)
        for coarse in sorted(completed.cells)
    }
    total_counts = {
        str(coarse): int(completed.cells[coarse].size)
        for coarse in sorted(completed.cells)
    }
    fine_recurrent_counts = {
        str(node): len(live_morse_graph.morse_set(node))
        for node in FINE_NODES
    }

    roots_payload = json.loads(roots_path.read_text(encoding="utf-8"))
    roots = np.asarray(
        roots_payload["stable_roots"]["encoded"],
        dtype=np.float64,
    )
    if roots.shape != (2, 3):
        raise ValueError(
            f"{roots_path} has shape {roots.shape}; expected (2, 3)"
        )
    root_owners = {
        "negative": _root_owner(live_morse_graph, roots[0]),
        "positive": _root_owner(live_morse_graph, roots[1]),
    }
    if root_owners != {"negative": 0, "positive": 1}:
        raise ValueError(
            f"unexpected 3-D attractor sign mapping: {root_owners}"
        )

    map_edge_count_method = getattr(map_graph, "num_cached_edges", None)
    map_edge_count = (
        int(map_edge_count_method())
        if callable(map_edge_count_method)
        else None
    )
    manifest = {
        "schema_version": 1,
        "source": {
            "run": str(run),
            "fine_morse_graph": {
                "path": str(fine_dot),
                "sha256": _sha256(fine_dot),
            },
            "fine_morse_sets": {
                "path": str(fine_sets),
                "sha256": _sha256(fine_sets),
            },
            "bounds": {
                "path": str(bounds_path),
                "sha256": _sha256(bounds_path),
                "lower": bounds.lower.tolist(),
                "upper": bounds.upper.tolist(),
            },
            "lookup_table": str(table_path),
        },
        "computation": {
            "callback": "persisted HierarchicalPrecomputedBoxMap lookup",
            "callback_neural_evaluations": 0,
            "cmgdb_routine": "CMGDB.ComputeMorseGraph",
            "cmgdb_seconds": cmgdb_seconds,
            "path_completion": "CMGDB.MorseDirectedPathCells cached-CSR traversal",
            "path_completion_seconds": completion_seconds,
            "subdivisions": {
                "init": SUBDIV_INIT,
                "min": SUBDIV_MIN,
                "max": SUBDIV_MAX,
                "limit": SUBDIV_LIMIT,
            },
            "padding": True,
            "map_cells": int(map_graph.num_vertices()),
            "cached_edges": map_edge_count,
            "environment": {
                "CMGDB_MAPGRAPH_MAX_VERTICES": os.environ[
                    "CMGDB_MAPGRAPH_MAX_VERTICES"
                ],
                "CMGDB_MAPGRAPH_MAX_EDGES": os.environ[
                    "CMGDB_MAPGRAPH_MAX_EDGES"
                ],
                "CMGDB_MAPGRAPH_RESERVE_EDGES": os.environ[
                    "CMGDB_MAPGRAPH_RESERVE_EDGES"
                ],
            },
        },
        "fine_graph": {
            "nodes": list(FINE_NODES),
            "edges": [list(edge) for edge in sorted(live_edges)],
            "recurrent_cell_counts": fine_recurrent_counts,
        },
        "quotient": {
            "projection": {
                str(fine): coarse
                for fine, coarse in sorted(quotient.projection.items())
            },
            "fibers": {
                str(coarse): sorted(fiber)
                for coarse, fiber in quotient.fibers.items()
            },
            "edges": {"2": [0, 1]},
            "labels": {
                "0": "M(0⁻)",
                "1": "M(0⁺)",
                "2": "M(1)",
            },
            "colors": {
                str(node): color
                for node, color in IDENTITY_COLORS.items()
            },
            "root_owners": root_owners,
            "recurrent_cell_counts": recurrent_counts,
            "connection_cell_counts": connection_counts,
            "total_cell_counts": total_counts,
            "overlap_count": 0,
            "merged_node_conley_index": None,
            "merged_node_conley_note": (
                "No Conley index is inferred for the merged fiber."
            ),
        },
        "alternative_2d_semantic_display": SEMANTIC_DISPLAY_RELABEL,
        "rendering_diagnostics": {
            "box_width_distribution": _width_distribution(
                output / "morse_sets"
            ),
        },
        "outputs": {
            "morse_graph": {
                "path": str(output / "morse_graph"),
                "sha256": _sha256(output / "morse_graph"),
            },
            "morse_sets": {
                "path": str(output / "morse_sets"),
                "sha256": _sha256(output / "morse_sets"),
            },
            "cells": {
                path.name: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for path in sorted(cells_dir.glob("*.npy"))
            },
        },
    }
    manifest_path = output / "quotient.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    diagnostics = {
        "cmgdb_seconds": cmgdb_seconds,
        "path_completion_seconds": completion_seconds,
        "map_cells": int(map_graph.num_vertices()),
        "cached_edges": map_edge_count,
        "fine_recurrent_cell_counts": fine_recurrent_counts,
        "coarse_recurrent_cell_counts": recurrent_counts,
        "coarse_connection_cell_counts": connection_counts,
        "coarse_total_cell_counts": total_counts,
        "root_owners": root_owners,
        "morse_sets_rows": sum(
            int(entry["count"])
            for entry in manifest["rendering_diagnostics"][
                "box_width_distribution"
            ]["scale_distribution"]
        ),
    }
    (output / "run_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(f"artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
