#!/usr/bin/env python3
"""Fixed-depth CMGDB recomputation of the leslie3d_example1 latent Morse graph.

The script computes the Morse graph of the author-provided
``spurious_attractor_ex`` checkpoint on a uniform grid
(``subdiv_init = subdiv_min = subdiv_max = depth``), identifies the recurrent
component containing the learned stable fixed point and the component
containing the learned unstable period-two orbit, collapses their full order
interval, adds every cell on connections internal to that interval, and
recomputes the Conley index of the resulting coarse Morse set.

``--depth 22`` is the paper's coarse example, computed as (22, 22, 24): the
base grid is depth 22 and refinement may reach 24. The Morse sets come back on
the depth-22 grid either way, so the fixed-depth analysis downstream is
unaffected -- what refinement buys is the removal of 20 one-cell components of
trivial index that a run pinned at (22, 22, 22) reports as Morse sets.
The distinguished-point coordinates are the learned invariant objects of the
shipped checkpoint
(:data:`latentdynamics.analysis.conley_index.LESLIE3D_EXAMPLE1_FIXED_POINT`).

Inputs resolve under ``replay_sources/leslie3d_example1/``.  Outputs go to
``<--output>/fixed<depth>`` (default ``output/leslie3d_example1_study``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import CMGDB
from latentdynamics._paths import get_repo_root
from latentdynamics.analysis.conley_index import (
    LESLIE3D_EXAMPLE1_FIXED_POINT as FIXED_POINT,
    LESLIE3D_EXAMPLE1_PERIOD_TWO as PERIOD_TWO,
    LocalIndexComputer,
    UniformCoordinates,
    component_index_labels,
    morse_graph_cells,
    parsed_live_graph,
)
from latentdynamics.analysis.morse import LatentBounds, _build_box_map
from latentdynamics.analysis.morse_coarsening import (
    coarsen_morse_graph,
    compute_connection_complete_morse_sets,
    write_connection_complete_morse_sets,
    write_morse_graph_dot,
)
from latentdynamics.analysis.morse_graph_parser import MorseGraph
from latentdynamics.config.schema import CMGDBConfig
from latentdynamics.replay import load_experiment
from latentdynamics.viz.style import PALETTE


REPO_ROOT = get_repo_root()
RUN = REPO_ROOT / "replay_sources" / "leslie3d_example1" / "spurious_attractor_ex"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "leslie3d_example1_study"

#: Refinement ceiling for the coarse example. The base grid depth is chosen
#: by --depth; refinement above it is what removes the trivial one-cell
#: components a fully pinned grid reports.
SUBDIV_MAX = 24
_EXPERIMENT = None


def cached_experiment():
    global _EXPERIMENT
    if _EXPERIMENT is None:
        _EXPERIMENT = load_experiment("leslie3d_example1_replay", device="cpu")
    return _EXPERIMENT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hash(models_dir: Path) -> tuple[str, str]:
    """Hash the shipped checkpoint: migrated single file, legacy fallback."""
    for name in ("autoencoder.pt", "dynamics.pt"):
        candidate = models_dir / name
        if candidate.is_file():
            return name, sha256(candidate)
    raise FileNotFoundError(f"no checkpoint file to hash under {models_dir}")


def recorded_path(path: Path) -> str:
    """Repo-relative path when possible, otherwise the path as given."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def render_graph(dot_path: Path, output_stem: Path) -> Path | None:
    try:
        from graphviz import Source

        rendered = Source(dot_path.read_text(encoding="utf-8")).render(
            filename=str(output_stem), format="png", cleanup=True
        )
        return Path(rendered)
    except Exception as exc:
        print(f"graph render skipped: {exc}", flush=True)
        return None


def fixed_config(experiment: Any, depth: int) -> CMGDBConfig:
    """CMGDB settings on a depth-``depth`` base grid, refining up to SUBDIV_MAX.

    Every recurrent cell is still carried down to ``depth``, so the Morse sets
    come back on the depth-``depth`` grid and the fixed-depth analysis
    downstream is unaffected.
    """
    raw = experiment.seed_cfg.cmgdb.model_dump()
    raw.update(
        subdiv_init=depth,
        subdiv_min=depth,
        subdiv_max=SUBDIV_MAX,
        compute_roa=False,
    )
    return CMGDBConfig.model_validate(raw)


def point_key(point: np.ndarray, coordinates: UniformCoordinates) -> tuple[int, ...]:
    raw = (np.asarray(point, dtype=np.float64) - coordinates.lower) / coordinates.widths
    key = np.floor(raw).astype(np.int64)
    key = np.maximum(key, 0)
    key = np.minimum(key, coordinates.sizes - 1)
    return tuple(int(value) for value in key)


def point_owners(
    point: np.ndarray,
    cells_by_node: dict[int, set[tuple[int, ...]]],
    coordinates: UniformCoordinates,
) -> list[int]:
    key = point_key(point, coordinates)
    return sorted(node for node, cells in cells_by_node.items() if key in cells)


def adaptive_parent_cells(
    coordinates: UniformCoordinates,
) -> dict[int, set[tuple[int, ...]]]:
    """Project the saved adaptive boxes to cells of the requested fixed grid."""
    result: dict[int, set[tuple[int, ...]]] = {}
    source = RUN / "MG" / "morse_sets"
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            values = np.asarray([float(value) for value in row[:-1]])
            dimension = len(coordinates.lower)
            center = 0.5 * (values[:dimension] + values[dimension:])
            label = int(float(row[-1]))
            result.setdefault(label, set()).add(point_key(center, coordinates))
    return result


def overlap_diagnostics(
    live: dict[int, set[tuple[int, ...]]],
    adaptive: dict[int, set[tuple[int, ...]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in sorted(live):
        overlaps = []
        for label in sorted(adaptive):
            intersection = len(live[node] & adaptive[label])
            if not intersection:
                continue
            overlaps.append(
                {
                    "adaptive_label": label,
                    "intersection": intersection,
                    "live_fraction": intersection / len(live[node]),
                    "adaptive_fraction": intersection / len(adaptive[label]),
                }
            )
        result[str(node)] = {
            "live_cells": len(live[node]),
            "adaptive_overlaps": overlaps,
        }
    return result


def graph_edges(graph: MorseGraph) -> list[list[int]]:
    return [
        [source, target]
        for source in graph.nodes
        for target in graph.edges.get(source, [])
    ]


def write_raw_morse_sets(morse_graph: Any, destination: Path) -> Path:
    """Write the uncoarsened recurrent boxes with their raw Morse labels."""
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for node in sorted(int(value) for value in morse_graph.vertices()):
            boxes = np.asarray(morse_graph.morse_set_boxes(node), dtype=np.float64)
            for row in boxes:
                writer.writerow([*row.tolist(), node])
    return destination


def nontrivial_skeleton(
    quotient: Any,
    indices: dict[int, list[str]],
    merged_coarse: int,
    merged_index: list[str],
) -> tuple[MorseGraph, dict[int, list[str]]]:
    coarse_indices: dict[int, list[str]] = {}
    for coarse, fiber in quotient.fibers.items():
        if coarse == merged_coarse:
            coarse_indices[coarse] = list(merged_index)
        elif len(fiber) == 1:
            coarse_indices[coarse] = list(indices[next(iter(fiber))])
        else:
            raise RuntimeError(f"unexpected additional merged fiber: {sorted(fiber)}")

    retained = sorted(
        coarse
        for coarse, index in coarse_indices.items()
        if any(value != "0" for value in index)
    )
    retained_set = set(retained)
    edges: dict[int, list[int]] = {}
    for source in retained:
        targets = (
            set(quotient.graph.descendants[source]) & retained_set
        ) - {source}
        covers = sorted(
            target
            for target in targets
            if not any(
                target in quotient.graph.descendants[other]
                for other in targets
                if other != target
            )
        )
        if covers:
            edges[source] = covers
    labels = {
        node: quotient.graph.labels.get(
            node,
            f"{node} : ({', '.join(coarse_indices[node])})",
        )
        for node in retained
    }
    colors = {
        node: quotient.graph.colors[node]
        for node in retained
        if node in quotient.graph.colors
    }
    return (
        MorseGraph(nodes=retained, edges=edges, colors=colors, labels=labels),
        {node: coarse_indices[node] for node in retained},
    )


def main(depth: int, output_root: Path) -> None:
    if depth < 1:
        raise ValueError(f"depth must be positive; got {depth}")
    out = output_root / f"fixed{depth}"
    stem = f"fixed{depth}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    experiment = cached_experiment()
    lower_raw, upper_raw = experiment.morse_bounds()
    lower = np.asarray(lower_raw, dtype=np.float64)
    upper = np.asarray(upper_raw, dtype=np.float64)
    bounds = LatentBounds(lower=lower, upper=upper)
    coordinates = UniformCoordinates(lower, upper, depth)
    config = fixed_config(experiment, depth)

    print(
        f"building fixed {depth}/{depth}/{config.subdiv_max} graph on "
        f"{coordinates.sizes.tolist()} grid",
        flush=True,
    )
    graph_started = time.perf_counter()
    box_map = _build_box_map(
        experiment.model.latent_map,
        bounds,
        config,
        device=torch.device("cpu"),
    )
    model = CMGDB.Model(
        config.subdiv_min,
        config.subdiv_max,
        config.subdiv_init,
        config.subdiv_limit,
        lower.tolist(),
        upper.tolist(),
        box_map,
    )
    if hasattr(model, "set_batch_map"):
        model.set_batch_map(box_map.batch)
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    graph_seconds = time.perf_counter() - graph_started
    print(
        f"fixed graph: {morse_graph.num_vertices()} recurrent sets, "
        f"{map_graph.num_vertices():,} cells, {graph_seconds:.1f}s",
        flush=True,
    )

    indices = component_index_labels(
        model, morse_graph, map_graph=map_graph, coordinates=coordinates
    )
    live_graph = parsed_live_graph(morse_graph, indices, PALETTE)
    live_cells = morse_graph_cells(morse_graph, coordinates)
    adaptive_cells = adaptive_parent_cells(coordinates)
    overlaps = overlap_diagnostics(live_cells, adaptive_cells)

    fixed_owners = point_owners(FIXED_POINT, live_cells, coordinates)
    period_owners = [
        point_owners(phase, live_cells, coordinates)
        for phase in PERIOD_TWO
    ]
    if len(fixed_owners) != 1:
        raise RuntimeError(f"fixed point has unexpected owners: {fixed_owners}")
    if any(len(owners) != 1 for owners in period_owners):
        raise RuntimeError(f"period-two phases have unexpected owners: {period_owners}")
    fixed_node = fixed_owners[0]
    period_nodes = sorted({owners[0] for owners in period_owners})
    if len(period_nodes) != 1:
        raise RuntimeError(
            f"period-two phases lie in different recurrent nodes: {period_nodes}"
        )
    period_node = period_nodes[0]
    if fixed_node not in live_graph.descendants[period_node]:
        raise RuntimeError(
            f"period-two node {period_node} does not reach fixed node {fixed_node}"
        )

    merge_group = frozenset(
        node
        for node in live_graph.nodes
        if node in live_graph.descendants[period_node]
        and fixed_node in live_graph.descendants[node]
    )
    print(
        f"fixed point node={fixed_node} index={indices[fixed_node]}; "
        f"period-two node={period_node} index={indices[period_node]}; "
        f"order interval={sorted(merge_group)}",
        flush=True,
    )

    quotient = coarsen_morse_graph(live_graph, [merge_group])
    merged_coarse = quotient.projection[fixed_node]
    completed = compute_connection_complete_morse_sets(
        map_graph,
        morse_graph,
        quotient.projection,
    )
    base_parts = [
        np.asarray(morse_graph.morse_set(node), dtype=np.int64)
        for node in sorted(merge_group)
    ]
    base_cells = np.unique(np.concatenate(base_parts))
    merged_cells = completed.cells[merged_coarse]
    connection_cells = completed.connection_cells[merged_coarse]

    index_computer = LocalIndexComputer(
        model,
        map_graph,
        morse_graph,
        coordinates,
    )
    literal = index_computer.compute("literal_recurrent_union", base_cells)
    merged = index_computer.compute("connection_complete_order_interval", merged_cells)
    print(
        f"merged cells: recurrent={len(base_cells)} + "
        f"connections={len(connection_cells)} = {len(merged_cells)}; "
        f"index={merged['conley_index']}",
        flush=True,
    )

    same_component = fixed_node == period_node
    merged_label = (
        (
            f"Cflip {fixed_node} : "
            if same_component
            else f"Mflip {sorted(merge_group)} : "
        )
        + f"({', '.join(merged['conley_index'] or ['undefined'])})"
    )
    quotient = coarsen_morse_graph(
        live_graph,
        [merge_group],
        labels={merge_group: merged_label},
    )
    merged_coarse = quotient.projection[fixed_node]
    quotient.graph.colors[merged_coarse] = "#008080ff"
    skeleton, skeleton_indices = nontrivial_skeleton(
        quotient,
        indices,
        merged_coarse,
        list(merged["conley_index"]),
    )

    raw_dot = write_morse_graph_dot(live_graph, out / f"morse_graph_{stem}.dot")
    raw_png = render_graph(raw_dot, out / f"morse_graph_{stem}")
    coarse_dot = write_morse_graph_dot(
        quotient.graph,
        out / f"morse_graph_{stem}_coarsened.dot",
    )
    coarse_png = render_graph(coarse_dot, out / f"morse_graph_{stem}_coarsened")
    skeleton_dot = write_morse_graph_dot(
        skeleton,
        out / f"morse_graph_{stem}_nontrivial_skeleton.dot",
    )
    skeleton_png = render_graph(
        skeleton_dot,
        out / f"morse_graph_{stem}_nontrivial_skeleton",
    )
    raw_sets_csv = write_raw_morse_sets(
        morse_graph,
        out / f"morse_sets_{stem}_raw.csv",
    )
    sets_csv = write_connection_complete_morse_sets(
        morse_graph,
        completed,
        out / f"morse_sets_{stem}_connection_complete.csv",
        allow_overlaps=bool(completed.overlaps),
    )

    component_summary: dict[str, Any] = {}
    for node in live_graph.nodes:
        boxes = np.asarray(morse_graph.morse_set_boxes(node), dtype=np.float64)
        component_summary[str(node)] = {
            "cells": len(live_cells[node]),
            "conley_index": indices[node],
            "minimal": node in live_graph.minimal,
            "extent": {
                "lower": boxes[:, :2].min(axis=0).tolist(),
                "upper": boxes[:, 2:].max(axis=0).tolist(),
            },
            "adaptive_overlaps": overlaps[str(node)]["adaptive_overlaps"],
        }

    checkpoint_file, checkpoint_sha256 = checkpoint_hash(RUN / "models")
    summary: dict[str, Any] = {
        "status": "complete",
        "purpose": (
            "leslie3d_example1 fixed-depth "
            f"subdiv_init=min={depth} max={config.subdiv_max} recomputation"
        ),
        "source_run": recorded_path(RUN),
        "source_hashes": {
            "checkpoint_file": f"models/{checkpoint_file}",
            "checkpoint_sha256": checkpoint_sha256,
            "adaptive_morse_graph": sha256(RUN / "MG" / "morse_graph"),
            "adaptive_morse_sets": sha256(RUN / "MG" / "morse_sets"),
        },
        "bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
        "subdivision": {
            "init": depth,
            "min": depth,
            "max": config.subdiv_max,
            "limit": config.subdiv_limit,
            "sizes": coordinates.sizes.tolist(),
            "cell_count": int(map_graph.num_vertices()),
            "padding": config.padding,
        },
        "fixed_graph": {
            "node_count": len(live_graph.nodes),
            "edges": graph_edges(live_graph),
            "minimal": sorted(live_graph.minimal),
            "components": component_summary,
        },
        "distinguished_objects": {
            "fixed_point": {
                "point": FIXED_POINT.tolist(),
                "node": fixed_node,
                "conley_index": indices[fixed_node],
            },
            "period_two": {
                "phases": PERIOD_TWO.tolist(),
                "node": period_node,
                "conley_index": indices[period_node],
            },
        },
        "coarsening": {
            "operation": (
                "identity_existing_component"
                if same_component
                else "order_interval_quotient"
            ),
            "same_component": same_component,
            "fine_order_interval": sorted(merge_group),
            "coarse_node": merged_coarse,
            "projection": {
                str(node): coarse for node, coarse in quotient.projection.items()
            },
            "edges": graph_edges(quotient.graph),
            "minimal": sorted(quotient.graph.minimal),
            "recurrent_cells": len(base_cells),
            "added_connection_cells": len(connection_cells),
            "merged_cells": len(merged_cells),
            "overlap_count": len(completed.overlaps),
            "literal_union": literal,
            "connection_complete": merged,
        },
        "nontrivial_skeleton": {
            "nodes": skeleton.nodes,
            "indices": {
                str(node): index for node, index in skeleton_indices.items()
            },
            "edges": graph_edges(skeleton),
            "minimal": sorted(skeleton.minimal),
        },
        "artifacts": {
            "fixed_graph_dot": recorded_path(raw_dot),
            "fixed_graph_png": recorded_path(raw_png) if raw_png else None,
            "coarse_graph_dot": recorded_path(coarse_dot),
            "coarse_graph_png": recorded_path(coarse_png) if coarse_png else None,
            "nontrivial_skeleton_dot": recorded_path(skeleton_dot),
            "nontrivial_skeleton_png": (
                recorded_path(skeleton_png) if skeleton_png else None
            ),
            "raw_morse_sets_csv": recorded_path(raw_sets_csv),
            "connection_complete_sets_csv": recorded_path(sets_csv),
        },
        "timings_seconds": {
            "fixed_graph_and_map": graph_seconds,
            "total": time.perf_counter() - started,
        },
    }
    result_path = out / "result.json"
    result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    operation_lines = (
        [
            (
                "The fixed point and both period-two phases are already "
                f"enclosed by combinatorial Morse component {fixed_node}."
            ),
            (
                "No quotient or connecting cells are required; the existing "
                f"{len(merged_cells)}-cell component has Conley index "
                f"`{merged['conley_index']}`."
            ),
            (
                "This is a coarse combinatorial enclosure: the exact learned "
                "fixed point and two-cycle remain distinct invariant objects."
            ),
        ]
        if same_component
        else [
            f"Order interval collapsed: `{sorted(merge_group)}`.",
            (
                f"Connection-complete cells: {len(base_cells)} recurrent + "
                f"{len(connection_cells)} connecting = {len(merged_cells)}."
            ),
            f"Merged Conley index: `{merged['conley_index']}`.",
        ]
    )
    report = "\n".join(
        [
            f"# leslie3d_example1 fixed-depth {depth}/{depth}/{config.subdiv_max} recomputation",
            "",
            f"Raw fixed-depth graph: {len(live_graph.nodes)} nodes, minima {sorted(live_graph.minimal)}.",
            f"Stable fixed point: node {fixed_node}, index `{indices[fixed_node]}`.",
            f"Unstable period-two orbit: node {period_node}, index `{indices[period_node]}`.",
            *operation_lines,
            f"Coarse Hasse edges: `{quotient.graph.edges}`; minima `{sorted(quotient.graph.minimal)}`.",
            f"Nontrivial skeleton edges: `{skeleton.edges}`; minima `{sorted(skeleton.minimal)}`.",
            f"Index-pair validity: `{literal['pair_valid']}`.",
            "",
        ]
    )
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--depth",
        type=int,
        default=23,
        help=f"base grid depth; 22 is the paper's coarse run, computed as "
             f"(22, 22, {SUBDIV_MAX})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "study root; results are written to <output>/fixed<depth> "
            "(default: output/leslie3d_example1_study)"
        ),
    )
    arguments = parser.parse_args()
    main(arguments.depth, arguments.output)
