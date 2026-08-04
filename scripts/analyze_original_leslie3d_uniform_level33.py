#!/usr/bin/env python3
"""Recompute Leslie reachability among saved recurrent sets at level 33.

The paper's adaptive CMGDB run localizes six strongly connected recurrent seed
sets within level-33 descendant subgrids, but its displayed Morse order also
inherits reachability computed on the coarser levels 29--32.  A literal
level-33 whole-domain graph has 2**33 vertices and is not practical to
materialize.

This postprocessor instead takes every box in each saved level-33 recurrent
set as a seed and exhausts its forward closure in the *uniform* level-33
box-map graph.  This is sufficient to decide reachability among the saved
strongly connected seed sets: any path beginning in a seed set lies entirely
in that forward closure.  The resulting relation is transitively reduced and
rendered as a seeded reachability graph.  The saved recurrent boxes are also
rendered in all three coordinate projections.

The calculation is exact for the floating-point, eight-corner box graph used
by ``CMGDB.BoxMap(..., padding=False)``.  It is not a validated enclosure of
the continuous map, and it cannot exclude a wholly new level-33 recurrent SCC
outside every seed region localized by the adaptive run.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
LOCAL_CMGDB_ROOT = PROJECT_ROOT / "archive" / "CMGDB"
LOCAL_CMGDB_SRC = LOCAL_CMGDB_ROOT / "src"

DEFAULT_SOURCE = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "ground_truth"
    / "absorbing_B_i29_m33_M36_L10000"
    / "screen"
)
DEFAULT_OUTPUT = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "ground_truth"
    / "absorbing_B_uniform_level33_recurrent_closure"
)
DEFAULT_CONLEY_SUMMARY = (
    DEFAULT_SOURCE.parent / "saved_set_conley" / "summary.json"
)

DOMAIN_LOWER = (0.0, 0.0, 0.0)
DOMAIN_UPPER = (110.0, 77.0, 54.0)
LEVEL = 33
THETA = (28.9, 29.8, 22.0)
SURVIVAL = (0.7, 0.7)

CORNER_SELECTORS = tuple(itertools.product((0, 1), repeat=3))
INTEGER_PHASE_WIDTH = 1 << 60
TRUNCATION_ERROR = 1 << 10

NODE_NAMES = {
    0: "P0",
    1: "P1",
    2: "S2",
    3: "S4",
    4: "p*",
    5: "origin",
}
NODE_COLORS = {
    0: "#FFB000",
    1: "#DC267F",
    2: "#FE6100",
    3: "#648FFF",
    4: "#785EF0",
    5: "#008080",
}

REQUIRED_INDEX_PAIR_CHECKS = (
    "S_subset_F_S",
    "A_subset_X",
    "X_minus_A_equals_S",
    "F_X_minus_A_subset_X",
    "F_A_intersect_X_subset_A",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repository: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"revision": revision, "dirty": bool(status.strip())}


def leslie(point: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = point
    head = (THETA[0] * x + THETA[1] * y + THETA[2] * z) * math.exp(-0.1 * (x + y + z))
    return head, SURVIVAL[0] * x, SURVIVAL[1] * y


@dataclass(frozen=True)
class UniformGrid:
    """CMGDB's cyclic dyadic tree at one uniform subdivision level."""

    level: int
    lower: tuple[float, float, float] = DOMAIN_LOWER
    upper: tuple[float, float, float] = DOMAIN_UPPER

    @cached_property
    def splits(self) -> tuple[int, int, int]:
        return tuple((self.level + 2 - axis) // 3 for axis in range(3))

    @cached_property
    def counts(self) -> tuple[int, int, int]:
        return tuple(1 << split for split in self.splits)

    @cached_property
    def widths(self) -> tuple[float, float, float]:
        return tuple((self.upper[axis] - self.lower[axis]) / self.counts[axis] for axis in range(3))

    @property
    def size(self) -> int:
        return math.prod(self.counts)

    def encode(self, indices: Sequence[int]) -> int:
        ix, iy, iz = indices
        _, ny, nz = self.counts
        return (int(ix) * ny + int(iy)) * nz + int(iz)

    def decode(self, code: int) -> tuple[int, int, int]:
        _, ny, nz = self.counts
        ix, remainder = divmod(code, ny * nz)
        iy, iz = divmod(remainder, nz)
        return ix, iy, iz

    def box_bounds(
        self, code: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        indices = self.decode(code)
        lower = tuple(self.lower[axis] + indices[axis] * self.widths[axis] for axis in range(3))
        upper = tuple(lower[axis] + self.widths[axis] for axis in range(3))
        return lower, upper

    def cover_interval(self, lower: float, upper: float, axis: int) -> range:
        """Match the local TreeGrid::coverAccept endpoint convention."""

        domain_lower = self.lower[axis]
        domain_upper = self.upper[axis]
        domain_width = domain_upper - domain_lower
        if upper < domain_lower or lower > domain_upper:
            return range(0)

        normalized_lower = max(0.0, min(1.0, (lower - domain_lower) / domain_width))
        normalized_upper = max(0.0, min(1.0, (upper - domain_lower) / domain_width))
        integer_lower = max(
            0,
            int(INTEGER_PHASE_WIDTH * normalized_lower) - TRUNCATION_ERROR,
        )
        integer_upper = min(
            INTEGER_PHASE_WIDTH,
            int(INTEGER_PHASE_WIDTH * normalized_upper) + TRUNCATION_ERROR,
        )

        step = 1 << (60 - self.splits[axis])
        first = max(0, -(-integer_lower // step) - 1)
        last = min(self.counts[axis] - 1, integer_upper // step)
        return range(first, last + 1)

    def image_hull(
        self, code: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        lower, upper = self.box_bounds(code)
        heads = []
        for selector in CORNER_SELECTORS:
            point = tuple(upper[axis] if selector[axis] else lower[axis] for axis in range(3))
            heads.append(leslie(point)[0])
        return (
            (min(heads), SURVIVAL[0] * lower[0], SURVIVAL[1] * lower[1]),
            (max(heads), SURVIVAL[0] * upper[0], SURVIVAL[1] * upper[1]),
        )

    def adjacencies(self, code: int) -> Iterator[int]:
        image_lower, image_upper = self.image_hull(code)
        ranges = tuple(
            self.cover_interval(image_lower[axis], image_upper[axis], axis) for axis in range(3)
        )
        return (
            self.encode((ix, iy, iz)) for ix in ranges[0] for iy in ranges[1] for iz in ranges[2]
        )


def require_local_cmgdb() -> tuple[object, Path]:
    """Import CMGDB, preferring the checkout, and confirm it is the fork."""
    if LOCAL_CMGDB_SRC.is_dir() and str(LOCAL_CMGDB_SRC) not in sys.path:
        sys.path.insert(0, str(LOCAL_CMGDB_SRC))
    import CMGDB

    from latentdynamics.analysis.cmgdb_fork import require_fork_cmgdb

    return CMGDB, require_fork_cmgdb()


def validate_adjacencies(grid: UniformGrid, level: int) -> dict[str, object]:
    """Exhaustively compare the replicated adjacency at a small level."""

    CMGDB, module_path = require_local_cmgdb()
    small = UniformGrid(level, grid.lower, grid.upper)

    def box_map(rect: Sequence[float]) -> list[float]:
        return CMGDB.BoxMap(leslie, rect, padding=False)

    model = CMGDB.Model(
        level,
        level,
        level,
        10_000,
        list(small.lower),
        list(small.upper),
        box_map,
    )
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    grid_id_to_code: dict[int, int] = {}
    for grid_id in range(map_graph.num_vertices()):
        bounds = morse_graph.phase_space_box(grid_id)
        indices = tuple(
            round((bounds[axis] - small.lower[axis]) / small.widths[axis]) for axis in range(3)
        )
        grid_id_to_code[grid_id] = small.encode(indices)

    mismatches = 0
    for grid_id, code in grid_id_to_code.items():
        observed = {grid_id_to_code[target] for target in map_graph.adjacencies(grid_id)}
        expected = set(small.adjacencies(code))
        mismatches += int(observed != expected)
    return {
        "level": level,
        "source_boxes_checked": int(map_graph.num_vertices()),
        "mismatches": mismatches,
        "cmgdb_version": version("CMGDB"),
        "cmgdb_module": str(module_path),
        "cmgdb_repository": str(LOCAL_CMGDB_ROOT),
        **git_state(LOCAL_CMGDB_ROOT),
    }


def load_saved_morse_sets(
    path: Path, grid: UniformGrid
) -> tuple[dict[int, NDArray[np.uint64]], NDArray[np.float64]]:
    """Load and verify the saved level-33 boxes."""

    raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError(f"Expected seven CSV columns in {path}; got {raw.shape}")

    lower = raw[:, :3]
    upper = raw[:, 3:6]
    nodes_float = raw[:, 6]
    nodes = nodes_float.astype(np.int16)
    if not np.array_equal(nodes_float, nodes):
        raise ValueError("Morse-set labels are not integral")

    expected_widths = np.asarray(grid.widths)
    if not np.allclose(upper - lower, expected_widths, rtol=0.0, atol=1e-12):
        raise ValueError("Not every saved Morse box is a uniform level-33 box")
    indices_float = (lower - np.asarray(grid.lower)) / expected_widths
    indices = np.rint(indices_float).astype(np.int64)
    if not np.allclose(indices_float, indices, rtol=0.0, atol=1e-9):
        raise ValueError("Saved Morse boxes do not align with the uniform grid")
    if np.any(indices < 0) or np.any(indices >= np.asarray(grid.counts)):
        raise ValueError("Saved Morse box lies outside the requested grid")

    _, ny, nz = grid.counts
    codes = ((indices[:, 0] * ny + indices[:, 1]) * nz + indices[:, 2]).astype(np.uint64)
    if np.unique(codes).size != codes.size:
        raise ValueError("A level-33 cell occurs in more than one saved Morse set")

    by_node = {int(node): np.sort(codes[nodes == node]) for node in sorted(np.unique(nodes))}
    return by_node, raw


def load_verified_conley_indices(
    path: Path,
    source_sets: Path,
    saved_sets: dict[int, NDArray[np.uint64]],
) -> tuple[dict[int, tuple[str, ...]], dict[str, object]]:
    """Load indices whose saved sets and local index pairs were verified."""

    summary = json.loads(path.read_text(encoding="utf-8"))
    nodes = sorted(saved_sets)
    if summary.get("status") != "complete" or summary.get("failed_nodes") != []:
        raise ValueError(f"Conley computation is not complete: {path}")
    if sorted(summary.get("completed_nodes", [])) != nodes:
        raise ValueError(f"Conley computation does not cover nodes {nodes}: {path}")
    if summary.get("algorithm") != (
        "saved uniform Morse sets -> verified local index pair -> ComputeConleyIndex"
    ):
        raise ValueError(f"Unexpected Conley-index algorithm in {path}")

    observed_source_hash = sha256(source_sets)
    recorded_source_hash = (
        summary.get("source_artifacts", {}).get("morse_sets", {}).get("sha256")
    )
    if recorded_source_hash != observed_source_hash:
        raise ValueError(
            "Conley indices were not computed from the current saved sets: "
            f"{recorded_source_hash} != {observed_source_hash}"
        )

    recorded_counts = summary.get("observed_morse_boxes_per_node", {})
    expected_counts = {str(node): len(saved_sets[node]) for node in nodes}
    if recorded_counts != expected_counts:
        raise ValueError(
            f"Conley saved-set counts differ: {recorded_counts} != {expected_counts}"
        )

    raw_indices = summary.get("conley_indices")
    if not isinstance(raw_indices, dict):
        raise ValueError(f"Missing Conley indices in {path}")
    results = summary.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Missing per-node Conley results in {path}")
    results_by_node = {int(result["node"]): result for result in results}
    if set(results_by_node) != set(nodes):
        raise ValueError(f"Unexpected per-node Conley results in {path}")

    indices: dict[int, tuple[str, ...]] = {}
    for node in nodes:
        result = results_by_node[node]
        if result.get("status") != "complete" or result.get("depth") != LEVEL:
            raise ValueError(f"Node {node} lacks a complete level-{LEVEL} index")
        checks = result.get("index_pair_checks", {})
        if any(checks.get(name) is not True for name in REQUIRED_INDEX_PAIR_CHECKS):
            raise ValueError(f"Node {node} failed an index-pair check")
        if result.get("acyclic_fiber_check") is not True:
            raise ValueError(f"Node {node} failed its acyclic-fiber check")
        raw_index = result.get("conley_index")
        if (
            not isinstance(raw_index, list)
            or len(raw_index) != 4
            or not all(isinstance(entry, str) for entry in raw_index)
        ):
            raise ValueError(f"Invalid Conley index for node {node}: {raw_index}")
        if raw_indices.get(str(node)) != raw_index:
            raise ValueError(f"Node {node} summary and detailed indices disagree")
        indices[node] = tuple(raw_index)

    return indices, summary


def exhaustive_forward_closure(
    grid: UniformGrid,
    starts: NDArray[np.uint64],
    owner: dict[int, int],
    saved_sets: dict[int, NDArray[np.uint64]],
    *,
    progress_every: int,
) -> dict[str, object]:
    """Exhaust the full uniform-graph closure of one saved seed set."""

    started = time.perf_counter()
    seen = {int(code) for code in starts}
    frontier = set(seen)
    reached_nodes = {owner[code] for code in seen if code in owner}
    distance = 0
    adjacency_count = 0
    max_frontier = len(frontier)

    while frontier:
        next_frontier: set[int] = set()
        for source in frontier:
            for target in grid.adjacencies(source):
                adjacency_count += 1
                target_node = owner.get(target)
                if target_node is not None:
                    reached_nodes.add(target_node)
                if target not in seen:
                    next_frontier.add(target)
        seen.update(next_frontier)
        frontier = next_frontier
        distance += 1
        max_frontier = max(max_frontier, len(frontier))
        if progress_every and distance % progress_every == 0:
            print(
                f"    distance={distance} visited={len(seen):,} frontier={len(frontier):,}",
                flush=True,
            )

    intersection_counts = {
        str(node): sum(int(code) in seen for code in codes) for node, codes in saved_sets.items()
    }
    return {
        "frontier_exhausted": True,
        "visited_boxes": len(seen),
        "graph_distances_completed": distance,
        "max_frontier_boxes": max_frontier,
        "adjacencies_examined": adjacency_count,
        "reached_saved_nodes": sorted(reached_nodes),
        "saved_box_intersection_counts": intersection_counts,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def transitive_reduction(
    nodes: Sequence[int], reachability: dict[int, set[int]]
) -> list[tuple[int, int]]:
    """Compute the cover relation of a finite strict partial order."""

    edges: list[tuple[int, int]] = []
    for source in nodes:
        for target in sorted(reachability[source]):
            if any(
                middle != target
                and middle in reachability[source]
                and target in reachability[middle]
                for middle in nodes
            ):
                continue
            edges.append((source, target))
    return edges


@dataclass(frozen=True)
class SavedSetReachabilityGraph:
    """Read-only adapter for :func:`CMGDB.PlotMorseGraph`.

    The adapter intentionally supplies no annotations: the known-object names
    are not Conley-index annotations and belong in surrounding explanatory
    text.  Its edges are the transitive reduction of reachability among the
    saved recurrent seed sets, not a complete uniform Morse decomposition.
    """

    node_ids: tuple[int, ...]
    reduced_edges: tuple[tuple[int, int], ...]
    conley_indices: dict[int, tuple[str, ...]]

    def vertices(self) -> list[int]:
        return list(self.node_ids)

    def edges(self) -> list[tuple[int, int]]:
        return list(self.reduced_edges)

    def adjacencies(self, node: int) -> list[int]:
        return [target for source, target in self.reduced_edges if source == node]

    def annotations(self, node: int) -> list[str]:
        return list(self.conley_indices.get(node, ()))


def node_palette(nodes: Sequence[int]) -> list[str]:
    return [NODE_COLORS.get(node, "#7f7f7f") for node in range(max(nodes) + 1)]


def render_graph(
    output: Path,
    nodes: Sequence[int],
    edges: Sequence[tuple[int, int]],
    CMGDB: object,
    *,
    basename: str,
    conley_indices: dict[int, tuple[str, ...]] | None = None,
) -> None:
    """Render saved-set reachability with CMGDB's native graph plotter."""

    adapter = SavedSetReachabilityGraph(
        tuple(nodes),
        tuple(edges),
        {} if conley_indices is None else conley_indices,
    )
    plot = CMGDB.PlotMorseGraph(adapter, clist=node_palette(nodes))
    (output / f"{basename}.dot").write_text(plot.source, encoding="utf-8")
    for fmt in ("pdf", "png"):
        plot.render(
            filename=basename,
            directory=str(output),
            format=fmt,
            view=False,
            cleanup=True,
        )


def render_morse_sets(
    output: Path,
    raw: NDArray[np.float64],
    nodes: Sequence[int],
    CMGDB: object,
) -> None:
    """Render all coordinate projections with CMGDB's native set plotter."""

    projections = (
        (0, 1, "$x_1$", "$x_2$", "x1_x2"),
        (0, 2, "$x_1$", "$x_3$", "x1_x3"),
        (1, 2, "$x_2$", "$x_3$", "x2_x3"),
    )
    palette = node_palette(nodes)
    for first, second, xlabel, ylabel, suffix in projections:
        png = output / f"morse_sets_{suffix}.png"
        pdf = output / f"morse_sets_{suffix}.pdf"
        CMGDB.PlotMorseSets(
            raw,
            morse_nodes=list(nodes),
            proj_dims=[first, second],
            clist=palette,
            fig_w=8,
            fig_h=6,
            xlim=[DOMAIN_LOWER[first], DOMAIN_UPPER[first]],
            ylim=[DOMAIN_LOWER[second], DOMAIN_UPPER[second]],
            xlabel=xlabel,
            ylabel=ylabel,
            fontsize=15,
            fig_fname=str(png),
            dpi=300,
        )
        fig = plt.gcf()
        for collection in fig.axes[0].collections:
            collection.set_rasterized(True)
        fig.savefig(pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--conley-summary",
        type=Path,
        default=DEFAULT_CONLEY_SUMMARY,
        help="Verified local Conley-index summary for the exact saved sets.",
    )
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument(
        "--validate-levels",
        type=int,
        nargs="*",
        default=[9, 12, 15, 18],
        help="Small uniform levels exhaustively compared with local CMGDB.",
    )
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help=(
            "Reuse exhausted closures from an existing output manifest after "
            "verifying source hashes. Intended for render-only updates."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print one progress line every N graph distances; zero disables it.",
    )
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    output = args.output_dir.resolve()
    conley_summary_path = args.conley_summary.resolve()
    existing_manifest_path = output / "manifest.json"
    existing_manifest_bytes = (
        existing_manifest_path.read_bytes()
        if args.reuse_output and existing_manifest_path.is_file()
        else None
    )
    existing_manifest = (
        json.loads(existing_manifest_bytes) if existing_manifest_bytes is not None else None
    )
    if output.exists() and existing_manifest is None:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.level != 33:
        raise ValueError("The saved recurrent boxes used by this script are level 33")
    if any(level < 1 or level > 20 for level in args.validate_levels):
        raise ValueError("validate-levels must lie between 1 and 20")
    if len(set(args.validate_levels)) != len(args.validate_levels):
        raise ValueError("validate-levels cannot contain duplicates")
    if args.progress_every < 0:
        raise ValueError("progress-every cannot be negative")

    source_sets = source_run / "MG" / "morse_sets"
    source_manifest = source_run / "manifest.json"
    source_graph = source_run / "morse_graph"
    for path in (source_sets, source_manifest, source_graph):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not conley_summary_path.is_file():
        raise FileNotFoundError(conley_summary_path)

    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    grid = UniformGrid(args.level)
    print(
        f"uniform level={grid.level} splits={grid.splits} cells={grid.size:,} widths={grid.widths}",
        flush=True,
    )
    validation = [validate_adjacencies(grid, level) for level in args.validate_levels]
    for item in validation:
        print(f"adjacency validation: {item}", flush=True)
        if item["mismatches"]:
            raise RuntimeError("Replicated adjacency does not match local CMGDB")

    by_node, raw = load_saved_morse_sets(source_sets, grid)
    nodes = sorted(by_node)
    conley_indices, conley_summary = load_verified_conley_indices(
        conley_summary_path,
        source_sets,
        by_node,
    )
    owner = {int(code): node for node, codes in by_node.items() for code in codes}
    print(
        "loaded saved level-33 strongly connected seed sets: "
        + ", ".join(f"{node}={len(by_node[node]):,}" for node in nodes),
        flush=True,
    )

    current_hashes = {
        "morse_sets_sha256": sha256(source_sets),
        "manifest_sha256": sha256(source_manifest),
        "morse_graph_sha256": sha256(source_graph),
    }
    reused_manifest_sha256 = None
    if existing_manifest is not None:
        for key, observed in current_hashes.items():
            expected = existing_manifest.get("source", {}).get(key)
            if observed != expected:
                raise RuntimeError(
                    f"Cannot reuse closures: source {key} changed from {expected} to {observed}"
                )
        if existing_manifest.get("uniform_grid", {}).get("level") != grid.level:
            raise RuntimeError("Cannot reuse closures from a different grid level")
        reused_manifest_sha256 = hashlib.sha256(existing_manifest_bytes).hexdigest()

    closures: dict[str, dict[str, object]] = {}
    reachability: dict[int, set[int]] = {}
    for node in nodes:
        if existing_manifest is not None:
            print(
                f"reusing exhausted closure for node {node} ({NODE_NAMES.get(node)})",
                flush=True,
            )
            result = existing_manifest["closures"][str(node)]
            if result.get("frontier_exhausted") is not True:
                raise RuntimeError(f"Saved closure for node {node} was not exhaustive")
        else:
            print(
                f"exhausting closure from node {node} ({NODE_NAMES.get(node)})",
                flush=True,
            )
            result = exhaustive_forward_closure(
                grid,
                by_node[node],
                owner,
                by_node,
                progress_every=args.progress_every,
            )
        closures[str(node)] = result
        reached = {int(item) for item in result["reached_saved_nodes"]}
        reached.discard(node)
        reachability[node] = reached
        print(
            f"  visited={result['visited_boxes']:,} "
            f"reached={sorted(reached)} elapsed={result['elapsed_seconds']:.2f}s",
            flush=True,
        )

    mutual_relations = sorted(
        (source, target)
        for source in nodes
        for target in reachability[source]
        if source < target and source in reachability[target]
    )
    if mutual_relations:
        raise RuntimeError(
            "Saved seed sets are mutually reachable on the full level-33 graph; "
            f"coalescing is required before rendering: {mutual_relations}"
        )
    edges = transitive_reduction(nodes, reachability)
    CMGDB, _ = require_local_cmgdb()
    render_graph(
        output,
        nodes,
        edges,
        CMGDB,
        basename="saved_set_reachability_graph",
    )
    render_graph(
        output,
        nodes,
        edges,
        CMGDB,
        basename="conley_annotated_saved_set_reachability_graph",
        conley_indices=conley_indices,
    )
    render_morse_sets(output, raw, nodes, CMGDB)

    manifest = {
        "system": "original 3D Leslie map",
        "parameters": {
            "theta": list(THETA),
            "survival": list(SURVIVAL),
        },
        "domain": {"lower": list(DOMAIN_LOWER), "upper": list(DOMAIN_UPPER)},
        "uniform_grid": {
            "level": grid.level,
            "axis_splits": list(grid.splits),
            "axis_counts": list(grid.counts),
            "box_widths": list(grid.widths),
            "whole_domain_box_count": grid.size,
            "whole_domain_materialized": False,
        },
        "box_map": "eight corner samples, padding=False",
        "source": {
            "run": str(source_run),
            "morse_sets": str(source_sets),
            "morse_sets_sha256": sha256(source_sets),
            "manifest": str(source_manifest),
            "manifest_sha256": sha256(source_manifest),
            "morse_graph": str(source_graph),
            "morse_graph_sha256": sha256(source_graph),
            "adaptive_graph_edge_3_to_2_present": "3 -> 2;"
            in source_graph.read_text(encoding="utf-8"),
        },
        "method": {
            "name": (
                "saved level-33 strongly connected seed sets plus exhaustive "
                "uniform-graph forward closures"
            ),
            "reason_whole_grid_not_materialized": (
                "The uniform level-33 domain contains 2^33 vertices. For any "
                "saved seed set R, every path beginning in R lies in its exhaustive "
                "forward closure, so that closure decides all reachability "
                "from R without materializing unrelated cells."
            ),
            "saved_set_role": (
                "The adaptive CMGDB run computed these as strongly connected "
                "sets within level-33 descendant subgrids. The present "
                "calculation tests direct level-33 reachability among the "
                "saved sets and discards reachability inherited from levels "
                "29--32."
            ),
            "closures_reused_for_render_update": existing_manifest is not None,
            "reused_manifest_sha256": reused_manifest_sha256,
        },
        "validation": validation,
        "morse_boxes_per_node": {str(node): len(codes) for node, codes in by_node.items()},
        "conley_index_annotations": {
            "description": (
                "Verified local Conley indices of the six saved recurrent sets; "
                "these are not indices of their forward closures."
            ),
            "summary": str(conley_summary_path),
            "summary_sha256": sha256(conley_summary_path),
            "algorithm": conley_summary["algorithm"],
            "source_morse_sets_sha256": conley_summary["source_artifacts"]["morse_sets"][
                "sha256"
            ],
            "all_index_pair_checks_passed": True,
            "all_acyclic_fiber_checks_passed": True,
            "indices": {
                str(node): list(conley_indices[node]) for node in nodes
            },
            "cmgdb": conley_summary["cmgdb"],
            "figure": "conley_annotated_saved_set_reachability_graph",
        },
        "closures": closures,
        "strict_reachability": {str(node): sorted(reachability[node]) for node in nodes},
        "mutual_relations_between_saved_nodes": [list(edge) for edge in mutual_relations],
        "saved_set_reachability_graph": {
            "nodes": nodes,
            "node_names": {str(node): NODE_NAMES.get(node) for node in nodes},
            "transitive_reduction_edges": [list(edge) for edge in edges],
            "sink_nodes_among_saved_sets": [node for node in nodes if not reachability[node]],
            "uniform_reachability_3_to_2_present": 2 in reachability.get(3, set()),
            "uniform_reduced_edge_3_to_2_present": (3, 2) in edges,
        },
        "interpretation": {
            "discrete_graph": (
                "Exact for reachability among every saved strongly connected "
                "seed set in the replicated floating-point uniform level-33 "
                "graph; every closure was exhausted."
            ),
            "inventory_completeness": (
                "Not established. This postprocessor neither decomposes the "
                "closure union into maximal SCCs nor scans all 2^33 cells for "
                "a wholly new recurrent SCC outside the saved adaptive regions."
            ),
            "nonnesting_counterexample": (
                "Corner sampling is nonnested: level-29 source cell (15,2,101) "
                "has 18 coarse targets, while its 16 level-33 descendants "
                "project to 21 coarse targets, adding (768,14,1), (768,15,1), "
                "and (768,16,1). Therefore the adaptive inventory cannot by "
                "itself certify completeness of the full uniform SCC inventory."
            ),
            "continuous_map": (
                "Not a proof about the continuous map. Eight-corner sampling "
                "is not an interval enclosure of nonlinear interior extrema."
            ),
            "figures": (
                "The unannotated and Conley-index-annotated graphs are rendered "
                "by CMGDB.PlotMorseGraph through a read-only adapter exposing "
                "the saved-set reachability order. "
                "Each coordinate projection is rendered by CMGDB.PlotMorseSets "
                "from every saved level-33 box. Its native scatter collections "
                "are rasterized only when the corresponding PDF is saved, to "
                "keep the roughly two-million-box files tractable."
            ),
            "conley_graph_scope": (
                "The Conley annotations belong to the same six saved recurrent "
                "sets. They do not establish that this inventory is the complete "
                "uniform level-33 Morse decomposition."
            ),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["saved_set_reachability_graph"], indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
