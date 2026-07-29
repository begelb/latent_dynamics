"""Build Marcio-style three-node Morse quotient for the learned 3-D model.

The adaptive 3-D Morse graph is recomputed from the persisted hierarchical
corner table, so every CMGDB callback is an array lookup.  Fine nodes 0 and 1
remain distinct and fine nodes 2--10 are collapsed to ``M(1)``.  The cell set
for a quotient fiber is the same connection-complete enclosure used for the
2-D computation:

    forward_reachable(fiber cells) ∩ backward_reachable(fiber cells).

For the 276-million-edge 3-D graph this traversal runs directly against
CMGDB's cached CSR.  No neural network is loaded or evaluated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import CMGDB
import numpy as np

from chafee_latent_dimension_study import (
    DEFAULT_OUTPUT_ROOT,
    RESOLUTIONS,
    DimensionPaths,
    _load_bounds,
    _map_graph_cache_metadata,
    _run_lookup_cmgdb,
)
from latentdynamics.analysis.hierarchical_precomputed import (
    HierarchicalPrecomputedBoxMap,
)
from latentdynamics.analysis.morse_coarsening import (
    coarsen_morse_graph,
    compute_connection_complete_morse_sets,
    write_connection_complete_morse_sets,
    write_morse_graph_dot,
)
from latentdynamics.analysis.morse_graph_parser import MorseGraph

DIMENSION = 3
SCHEMA_VERSION = 1
EXPECTED_FINE_NODES = list(range(11))
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
EXPECTED_MINIMAL_NODES = {0, 1}
MERGED_FINE_NODES = frozenset(range(2, 11))
EXPECTED_QUOTIENT_EDGES = {2: [0, 1]}
BLUE = "#1f77b4"
ORANGE = "#e6550d"
GRAY = "#7f7f7f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    label = str(resolved)
    if relative_to is not None:
        try:
            label = str(resolved.relative_to(relative_to.resolve()))
        except ValueError:
            pass
    return {
        "path": label,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _edge_set(graph: MorseGraph) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, targets in graph.edges.items()
        for target in targets
    }


def _live_edge_set(morse_graph: Any) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, target in morse_graph.edges()
    }


def _cells(morse_graph: Any, nodes: set[int] | frozenset[int] | list[int]) -> np.ndarray:
    parts = [
        np.fromiter(
            (int(cell) for cell in morse_graph.morse_set(int(node))),
            dtype=np.int64,
        )
        for node in sorted(nodes)
    ]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts))


def _fiber_extrema(
    graph: MorseGraph,
    fiber: frozenset[int],
) -> tuple[list[int], list[int]]:
    maxima = sorted(
        node
        for node in fiber
        if not any(
            node in graph.descendants[other]
            for other in fiber
            if other != node
        )
    )
    minima = sorted(
        node
        for node in fiber
        if not any(
            other in graph.descendants[node]
            for other in fiber
            if other != node
        )
    )
    return maxima, minima


def _validate_epimorphism(
    fine_graph: MorseGraph,
    projection: dict[int, int],
    quotient_graph: MorseGraph,
) -> dict[str, Any]:
    if set(projection) != set(fine_graph.nodes):
        raise ValueError("projection is not defined on every fine Morse node")
    image = set(projection.values())
    if image != set(quotient_graph.nodes):
        raise ValueError(
            f"projection image {sorted(image)} is not the quotient node set "
            f"{quotient_graph.nodes}"
        )

    violations: list[tuple[int, int, int, int]] = []
    relation_count = 0
    for source in fine_graph.nodes:
        for target in fine_graph.descendants[source]:
            if source == target:
                continue
            relation_count += 1
            coarse_source = projection[source]
            coarse_target = projection[target]
            if (
                coarse_source != coarse_target
                and coarse_target
                not in quotient_graph.descendants[coarse_source]
            ):
                violations.append(
                    (source, target, coarse_source, coarse_target)
                )
    if violations:
        raise ValueError(
            f"projection is not order preserving; examples: {violations[:10]}"
        )

    fibers = {
        coarse: frozenset(
            fine for fine, image_node in projection.items() if image_node == coarse
        )
        for coarse in quotient_graph.nodes
    }
    convexity_violations: list[tuple[int, int, int, int]] = []
    intervals_checked = 0
    for coarse, fiber in fibers.items():
        for upper in fiber:
            for lower in fiber:
                if lower not in fine_graph.descendants[upper]:
                    continue
                intervals_checked += 1
                for middle in fine_graph.nodes:
                    if (
                        middle in fine_graph.descendants[upper]
                        and lower in fine_graph.descendants[middle]
                        and middle not in fiber
                    ):
                        convexity_violations.append(
                            (coarse, upper, middle, lower)
                        )
    if convexity_violations:
        raise ValueError(
            "projection fibers are not order-convex; examples: "
            f"{convexity_violations[:10]}"
        )

    edge_witnesses: dict[str, list[list[int]]] = {}
    for coarse_source, coarse_targets in quotient_graph.edges.items():
        for coarse_target in coarse_targets:
            witnesses = [
                [source, target]
                for source in fine_graph.nodes
                for target in fine_graph.nodes
                if projection[source] == coarse_source
                and projection[target] == coarse_target
                and target in fine_graph.descendants[source]
            ]
            if not witnesses:
                raise ValueError(
                    f"quotient edge {coarse_source}->{coarse_target} has no "
                    "fine-order witness"
                )
            edge_witnesses[f"{coarse_source}->{coarse_target}"] = witnesses

    return {
        "surjective": True,
        "order_preserving": True,
        "order_convex_fibers": True,
        "fine_strict_relations_checked": relation_count,
        "fiber_intervals_checked": intervals_checked,
        "quotient_edge_fine_order_witnesses": edge_witnesses,
    }


def _adaptive_root_signs(
    fine_morse_sets: Path,
    statistics_path: Path,
) -> dict[str, Any]:
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    roots = np.asarray(
        statistics["stable_roots"]["encoded"],
        dtype=np.float64,
    )
    if roots.shape != (2, DIMENSION):
        raise ValueError(
            f"encoded stable roots have shape {roots.shape}; expected {(2, DIMENSION)}"
        )
    rows = np.loadtxt(fine_morse_sets, delimiter=",", ndmin=2, dtype=np.float64)
    if rows.shape[1] != 2 * DIMENSION + 1:
        raise ValueError(
            f"{fine_morse_sets} has {rows.shape[1]} columns; expected 7"
        )

    labels: list[int] = []
    box_counts: list[int] = []
    for root in roots:
        contains = np.all(
            (rows[:, :DIMENSION] <= root)
            & (root <= rows[:, DIMENSION : 2 * DIMENSION]),
            axis=1,
        )
        matched = sorted(set(rows[contains, -1].astype(int).tolist()))
        if len(matched) != 1:
            raise ValueError(
                f"encoded stable root {root.tolist()} lies in adaptive labels "
                f"{matched}; expected exactly one"
            )
        labels.append(matched[0])
        box_counts.append(int(contains.sum()))
    if labels != [0, 1]:
        raise ValueError(
            "adaptive root-sign convention changed: expected negative root in "
            f"fine node 0 and positive root in fine node 1, got {labels}"
        )
    return {
        "root_order": ["negative", "positive"],
        "encoded_roots": roots.tolist(),
        "adaptive_fine_nodes": labels,
        "containing_box_counts": box_counts,
        "verified_by_closed_box_containment": True,
    }


def _verify_marker_files(run: Path, marker_path: Path) -> dict[str, Any]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    outputs = marker["provenance"]["outputs"]
    verified: dict[str, Any] = {}
    for name, recorded in sorted(outputs["files"].items()):
        path = run / recorded["path"]
        actual = _signature(path, relative_to=run)
        if (
            actual["size_bytes"] != int(recorded["size_bytes"])
            or actual["sha256"] != str(recorded["sha256"])
        ):
            raise ValueError(
                f"{marker_path}: current {name} does not match recorded provenance"
            )
        verified[name] = actual
    return {
        "marker": _signature(marker_path, relative_to=run),
        "output_fingerprint": str(outputs["fingerprint"]),
        "verified_files": verified,
    }


def _save_cell_arrays(
    output: Path,
    completed: Any,
    recurrent: dict[int, np.ndarray],
) -> dict[str, Any]:
    cells_dir = output / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for coarse in sorted(completed.cells):
        arrays = {
            "recurrent": np.asarray(recurrent[coarse], dtype=np.int64),
            "connection": np.asarray(
                completed.connection_cells[coarse],
                dtype=np.int64,
            ),
            "complete": np.asarray(completed.cells[coarse], dtype=np.int64),
        }
        for kind, values in arrays.items():
            path = cells_dir / f"{kind}_coarse_{coarse}.npy"
            np.save(path, values, allow_pickle=False)
            artifacts[f"{kind}_coarse_{coarse}"] = {
                **_signature(path, relative_to=output),
                "dtype": str(values.dtype),
                "shape": list(values.shape),
            }
    return artifacts


def _artifact_hashes_match(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> bool | None:
    if previous is None:
        return None
    try:
        previous_artifacts = previous["artifacts"]
    except (KeyError, TypeError):
        return False
    if set(previous_artifacts) != set(current):
        return False
    return all(
        previous_artifacts[name]["sha256"] == current[name]["sha256"]
        and int(previous_artifacts[name]["size_bytes"])
        == int(current[name]["size_bytes"])
        for name in current
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: <3-D run>/MG_adaptive_coarse_marcio",
    )
    parser.add_argument("--cmgdb-max-edges", type=int, default=1_200_000_000)
    parser.add_argument("--cmgdb-reserve-edges", type=int, default=1_200_000_000)
    args = parser.parse_args()

    if args.cmgdb_max_edges <= 0 or args.cmgdb_reserve_edges <= 0:
        raise ValueError("CMGDB edge limits must be positive")
    if args.cmgdb_reserve_edges > args.cmgdb_max_edges:
        raise ValueError("--cmgdb-reserve-edges cannot exceed --cmgdb-max-edges")
    os.environ["CMGDB_MAPGRAPH_MAX_EDGES"] = str(args.cmgdb_max_edges)
    os.environ["CMGDB_MAPGRAPH_RESERVE_EDGES"] = str(args.cmgdb_reserve_edges)

    paths = DimensionPaths(
        output_root=args.output_root.resolve(),
        dimension=DIMENSION,
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else paths.run / "MG_adaptive_coarse_marcio"
    )
    source_dot = paths.adaptive / "morse_graph"
    source_sets = paths.adaptive / "morse_sets"
    source_adaptive_marker = paths.stage_marker("adaptive")
    source_precompute_marker = paths.stage_marker("precompute-fine")
    statistics_path = paths.stats
    required = (
        paths.bounds,
        paths.hierarchical_table / "metadata.json",
        source_dot,
        source_sets,
        source_adaptive_marker,
        source_precompute_marker,
        statistics_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required 3-D study artifacts: {missing}")

    native = getattr(CMGDB, "MorseDirectedPathCells", None)
    if not callable(native):
        raise RuntimeError(
            "CMGDB.MorseDirectedPathCells is required for the large cached "
            "3-D connection completion; rebuild archive/CMGDB"
        )

    previous_manifest_path = output / "quotient.json"
    previous_manifest = (
        json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        if previous_manifest_path.is_file()
        else None
    )
    output.mkdir(parents=True, exist_ok=True)

    fine_graph = MorseGraph.from_dot(source_dot)
    if fine_graph.nodes != EXPECTED_FINE_NODES:
        raise ValueError(
            f"unexpected 3-D fine nodes {fine_graph.nodes}; "
            f"expected {EXPECTED_FINE_NODES}"
        )
    if _edge_set(fine_graph) != EXPECTED_FINE_EDGES:
        raise ValueError(
            f"unexpected 3-D fine edges {sorted(_edge_set(fine_graph))}"
        )
    if fine_graph.minimal != EXPECTED_MINIMAL_NODES:
        raise ValueError(
            f"unexpected 3-D minimal nodes {sorted(fine_graph.minimal)}"
        )

    quotient = coarsen_morse_graph(
        fine_graph,
        [MERGED_FINE_NODES],
        labels={
            frozenset({0}): "M(0-)",
            frozenset({1}): "M(0+)",
            MERGED_FINE_NODES: "M(1)",
        },
    )
    if quotient.projection != {
        **{0: 0, 1: 1},
        **{node: 2 for node in range(2, 11)},
    }:
        raise ValueError(f"unexpected quotient projection {quotient.projection}")
    if quotient.graph.edges != EXPECTED_QUOTIENT_EDGES:
        raise ValueError(f"unexpected quotient edges {quotient.graph.edges}")
    if quotient.graph.minimal != EXPECTED_MINIMAL_NODES:
        raise ValueError(
            f"unexpected quotient minima {sorted(quotient.graph.minimal)}"
        )
    quotient.graph.colors = {
        0: f"{ORANGE}ff",
        1: f"{BLUE}ff",
        2: f"{GRAY}ff",
    }
    epimorphism_validation = _validate_epimorphism(
        fine_graph,
        quotient.projection,
        quotient.graph,
    )
    root_signs = _adaptive_root_signs(source_sets, statistics_path)

    precompute_provenance = _verify_marker_files(
        paths.run,
        source_precompute_marker,
    )
    adaptive_provenance = _verify_marker_files(
        paths.run,
        source_adaptive_marker,
    )
    table_metadata = json.loads(
        (paths.hierarchical_table / "metadata.json").read_text(encoding="utf-8")
    )
    if (
        int(table_metadata["dimension"]) != DIMENSION
        or int(table_metadata["coarse_subdiv"]) != 24
        or int(table_metadata["fine_subdiv"]) != 33
        or table_metadata.get("callback_neural_evaluations") != 0
    ):
        raise ValueError(
            "hierarchical lookup-table metadata no longer describes the "
            "zero-neural-evaluation 24-to-33 3-D table"
        )

    resolution = RESOLUTIONS[DIMENSION]
    bounds = _load_bounds(paths.bounds, DIMENSION)
    box_map = HierarchicalPrecomputedBoxMap.load(
        paths.hierarchical_table,
        mmap_mode="r",
    )
    topology_started = time.perf_counter()
    morse_graph, map_graph, cmgdb_duration, conley_status = _run_lookup_cmgdb(
        box_map,
        bounds,
        subdiv_init=resolution.adaptive_init,
        subdiv_min=resolution.adaptive_min,
        subdiv_max=resolution.adaptive_max,
        compute_conley=False,
    )
    topology_wall = time.perf_counter() - topology_started

    if int(morse_graph.num_vertices()) != len(EXPECTED_FINE_NODES):
        raise ValueError(
            f"live 3-D graph has {morse_graph.num_vertices()} nodes; expected 11"
        )
    if _live_edge_set(morse_graph) != EXPECTED_FINE_EDGES:
        raise ValueError(
            f"live 3-D graph edges differ: {sorted(_live_edge_set(morse_graph))}"
        )
    with tempfile.TemporaryDirectory(prefix="ci_3d_live_morse_") as tmp:
        live_sets = Path(tmp) / "morse_sets"
        CMGDB.SaveMorseSets(morse_graph, str(live_sets))
        live_sets_signature = _signature(live_sets)
    source_sets_signature = _signature(source_sets, relative_to=paths.run)
    if live_sets_signature["sha256"] != source_sets_signature["sha256"]:
        raise ValueError(
            "live lookup-only recomputation does not reproduce the persisted "
            "fine Morse sets byte-for-byte"
        )

    completion_started = time.perf_counter()
    completed = compute_connection_complete_morse_sets(
        map_graph,
        morse_graph,
        quotient.projection,
    )
    completion_duration = time.perf_counter() - completion_started
    if completed.overlaps:
        examples = list(completed.overlaps.items())[:10]
        raise ValueError(
            f"connection-complete quotient sets overlap; examples: {examples}"
        )

    recurrent = {
        coarse: _cells(morse_graph, fiber)
        for coarse, fiber in completed.fibers.items()
    }
    for coarse in sorted(completed.cells):
        expected_connections = np.setdiff1d(
            completed.cells[coarse],
            recurrent[coarse],
            assume_unique=True,
        )
        if not np.array_equal(
            expected_connections,
            completed.connection_cells[coarse],
        ):
            raise ValueError(
                f"coarse node {coarse} recurrent/connection partition is inconsistent"
            )
        if np.intersect1d(
            recurrent[coarse],
            completed.connection_cells[coarse],
            assume_unique=True,
        ).size:
            raise ValueError(
                f"coarse node {coarse} recurrent and connection cells overlap"
            )

    maxima, minima = _fiber_extrema(fine_graph, MERGED_FINE_NODES)
    endpoint_started = time.perf_counter()
    endpoint_complete = np.asarray(
        native(map_graph, morse_graph, maxima, minima),
        dtype=np.uint64,
    ).astype(np.int64, copy=False)
    endpoint_duration = time.perf_counter() - endpoint_started
    if not np.array_equal(endpoint_complete, completed.cells[2]):
        raise ValueError(
            "all-fiber completion and maximal-to-minimal endpoint completion "
            "do not agree"
        )

    foreign_recurrent = np.concatenate((recurrent[0], recurrent[1]))
    if np.intersect1d(
        completed.cells[2],
        foreign_recurrent,
        assume_unique=True,
    ).size:
        raise ValueError("M(1) completion contains a retained attractor cell")

    write_started = time.perf_counter()
    graph_path = write_morse_graph_dot(
        quotient.graph,
        output / "morse_graph",
    )
    sets_path = write_connection_complete_morse_sets(
        morse_graph,
        completed,
        output / "morse_sets",
    )
    cell_artifacts = _save_cell_arrays(output, completed, recurrent)
    write_duration = time.perf_counter() - write_started

    artifacts = {
        "morse_graph": _signature(graph_path, relative_to=output),
        "morse_sets": _signature(sets_path, relative_to=output),
        **cell_artifacts,
    }
    deterministic_match = _artifact_hashes_match(previous_manifest, artifacts)

    recurrent_counts = {
        str(coarse): int(recurrent[coarse].size)
        for coarse in sorted(recurrent)
    }
    connection_counts = {
        str(coarse): int(completed.connection_cells[coarse].size)
        for coarse in sorted(completed.connection_cells)
    }
    completed_counts = {
        str(coarse): int(completed.cells[coarse].size)
        for coarse in sorted(completed.cells)
    }
    fine_counts = {
        str(node): int(len(morse_graph.morse_set(node)))
        for node in EXPECTED_FINE_NODES
    }

    project_root = Path(__file__).resolve().parents[2]
    cmgdb_binary = next(
        Path(CMGDB.__file__).resolve().parent.glob("_cmgdb*.so")
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dimension": DIMENSION,
        "construction": {
            "name": "Marcio-style three-node Morse-poset epimorphism",
            "definition": (
                "For fiber S with recurrent-cell union U(S), the completed "
                "cell enclosure is forward_reachable(U(S)) intersect "
                "backward_reachable(U(S))."
            ),
            "connection_semantics": (
                "Every fine recurrent cell in a quotient fiber plus every "
                "cell on a directed cell-graph path between recurrent cells "
                "in that same fiber."
            ),
            "conley_index_note": (
                "No Conley index is assigned to merged node M(1); it cannot "
                "be inferred by combining fine-node annotations."
            ),
            "native_traversal": "CMGDB.MorseDirectedPathCells",
            "native_traversal_source": _signature(
                project_root
                / "archive"
                / "CMGDB"
                / "src"
                / "CMGDB"
                / "_cmgdb"
                / "CMGDB.cpp",
                relative_to=project_root,
            ),
            "python_construction_source": _signature(
                project_root
                / "code"
                / "src"
                / "latentdynamics"
                / "analysis"
                / "morse_coarsening.py",
                relative_to=project_root,
            ),
            "driver_source": _signature(
                Path(__file__),
                relative_to=project_root,
            ),
        },
        "source_adaptive_computation": {
            "run": str(paths.run.resolve()),
            "bounds": {
                "lower": bounds.lower.tolist(),
                "upper": bounds.upper.tolist(),
            },
            "subdivisions": {
                "init": resolution.adaptive_init,
                "min": resolution.adaptive_min,
                "max": resolution.adaptive_max,
                "per_axis": [
                    resolution.adaptive_init // DIMENSION,
                    resolution.adaptive_min // DIMENSION,
                    resolution.adaptive_max // DIMENSION,
                ],
            },
            "padding": True,
            "fine_graph": {
                "nodes": EXPECTED_FINE_NODES,
                "hasse_edges": [list(edge) for edge in sorted(EXPECTED_FINE_EDGES)],
                "minimal_nodes": sorted(EXPECTED_MINIMAL_NODES),
                "recurrent_cell_counts": fine_counts,
                "morse_graph": _signature(source_dot, relative_to=paths.run),
                "morse_sets": source_sets_signature,
            },
            "live_recomputation": {
                "morse_sets_sha256": live_sets_signature["sha256"],
                "matches_persisted_morse_sets_byte_for_byte": True,
                "map_graph": _map_graph_cache_metadata(map_graph),
                "conley": conley_status,
                "callback": (
                    "persisted HierarchicalPrecomputedBoxMap "
                    "dense-coarse/sparse-fine lookup"
                ),
                "callback_neural_evaluations": 0,
            },
            "root_sign_validation": root_signs,
        },
        "epimorphism": {
            "canonical_raw_id_convention": {
                "projection": {
                    str(fine): coarse
                    for fine, coarse in sorted(quotient.projection.items())
                },
                "fibers": {
                    str(coarse): sorted(fiber)
                    for coarse, fiber in sorted(quotient.fibers.items())
                },
                "quotient_hasse_edges": {"2": [0, 1]},
                "coarse_nodes": {
                    "0": {
                        "label": "M(0-)",
                        "sign": "negative",
                        "source_fine_nodes": [0],
                        "color": ORANGE,
                    },
                    "1": {
                        "label": "M(0+)",
                        "sign": "positive",
                        "source_fine_nodes": [1],
                        "color": BLUE,
                    },
                    "2": {
                        "label": "M(1)",
                        "sign": "nonminimal merged fiber",
                        "source_fine_nodes": sorted(MERGED_FINE_NODES),
                        "color": GRAY,
                    },
                },
                "artifact_label_convention": (
                    "morse_graph, morse_sets, and cells/*.npy use these "
                    "canonical raw quotient ids"
                ),
            },
            "optional_2d_semantic_display_convention": {
                "purpose": (
                    "Match Marcio's 2-D display ids/palette: display node 0 "
                    "is M(0+), display node 1 is M(0-), display node 2 is M(1)."
                ),
                "canonical_to_display": {"0": 1, "1": 0, "2": 2},
                "fine_to_display": {
                    "0": 1,
                    "1": 0,
                    **{str(node): 2 for node in range(2, 11)},
                },
                "display_coarse_nodes": {
                    "0": {"label": "M(0+)", "sign": "positive", "color": BLUE},
                    "1": {"label": "M(0-)", "sign": "negative", "color": ORANGE},
                    "2": {
                        "label": "M(1)",
                        "sign": "nonminimal merged fiber",
                        "color": GRAY,
                    },
                },
                "quotient_hasse_edges": {"2": [0, 1]},
            },
            "validation": epimorphism_validation,
        },
        "connection_complete_morse_sets": {
            "fiber_2_maxima": maxima,
            "fiber_2_minima": minima,
            "all_fiber_equals_top_down_bottom_up_endpoints": True,
            "endpoint_complete_sha256": hashlib.sha256(
                endpoint_complete.tobytes(order="C")
            ).hexdigest(),
            "recurrent_cell_counts": recurrent_counts,
            "connection_cell_counts": connection_counts,
            "completed_cell_counts": completed_counts,
            "overlap_cell_count": len(completed.overlaps),
            "retained_attractor_cells_inside_M1": 0,
            "partition_verified": True,
        },
        "lookup_only_guarantee": {
            "neural_model_loaded": False,
            "neural_evaluations": 0,
            "table_metadata": table_metadata,
            "precompute_fine_provenance": precompute_provenance,
            "adaptive_source_provenance": adaptive_provenance,
            "cmgdb_binary_used_for_coarsening": _signature(
                cmgdb_binary,
                relative_to=project_root,
            ),
        },
        "determinism": {
            "sorted_cell_ids": True,
            "stable_csv_order": "coarse node id, then ascending map-cell id",
            "live_source_reproduction": "byte-identical Morse-set CSV",
            "independent_completion_identity": (
                "Path(all fiber nodes, all fiber nodes) equals "
                "Path(fiber maxima, fiber minima)"
            ),
            "artifact_sha256_recorded": True,
        },
        "artifacts": artifacts,
    }
    _write_json(previous_manifest_path, manifest)

    diagnostics = {
        "schema_version": 1,
        "durations_seconds": {
            "cmgdb_reported_topology": cmgdb_duration,
            "topology_wall": topology_wall,
            "connection_completion": completion_duration,
            "endpoint_crosscheck": endpoint_duration,
            "artifact_write": write_duration,
        },
        "previous_manifest_found": previous_manifest is not None,
        "deterministic_artifact_hashes_match_previous": deterministic_match,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cmgdb_module": str(Path(CMGDB.__file__).resolve()),
        },
    }
    _write_json(output / "run_diagnostics.json", diagnostics)

    print(f"output: {output}")
    print(f"projection: {manifest['epimorphism']['canonical_raw_id_convention']['projection']}")
    print(f"quotient edges: {EXPECTED_QUOTIENT_EDGES}")
    print(f"recurrent cells: {recurrent_counts}")
    print(f"connection cells: {connection_counts}")
    print(f"completed cells: {completed_counts}")
    print(f"overlaps: {len(completed.overlaps)}")
    print(f"previous-run artifact hashes match: {deterministic_match}")

    del endpoint_complete, completed, recurrent, box_map, morse_graph, map_graph
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
