"""Uniform-grid cell bookkeeping and combinatorial Conley index pairs.

Utilities for CMGDB cell graphs whose recurrent boxes all live on one uniform
dyadic grid:

* :class:`UniformCoordinates` maps phase-space boxes to exact integer grid
  keys (and validates that a box really is a grid cell);
* :func:`match_nodes` matches the recurrent components of a live recomputation
  to saved Morse-set labels by maximum-Jaccard assignment;
* :class:`LocalIndexComputer` builds the one-step image index pair of a cell
  set, checks its validity, and evaluates the Conley index with the CMGDB
  fork's ``ComputeConleyIndexForCells``;
* :func:`component_index_labels` and :func:`parsed_live_graph` annotate every
  recurrent component of a live Morse graph with its Conley index.

The module also records the learned invariant-object coordinates of the
author-provided ``leslie3d_example1`` checkpoint (``spurious_attractor_ex``),
which the fixed-depth and coarsening studies use as ownership checks.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Iterable
from typing import Any

import numpy as np

import CMGDB

from .morse_graph_parser import MorseGraph

# Learned invariant objects of the shipped leslie3d_example1 checkpoint
# (``replay_sources/leslie3d_example1/spurious_attractor_ex``), in the stored
# latent coordinates.  The fixed point is the attracting fixed point of the
# learned latent map g obtained by iterating g to convergence from the center
# of Morse node 4 (|g(z*) - z*| ~ 2.7e-7); the two phases are the unstable
# period-two orbit associated with Morse node 5.  They are properties of the
# trained network weights, not of the underlying Leslie system, and serve as
# ownership checks: each point must lie in exactly one recurrent component of
# a recomputed cell graph.
LESLIE3D_EXAMPLE1_FIXED_POINT = np.asarray(
    [-0.2157317188464123, -0.342826142886686]
)
LESLIE3D_EXAMPLE1_PERIOD_TWO = np.asarray(
    [
        [-0.21588962, -0.35433480],
        [-0.21641609, -0.33100010],
    ]
)


def split_counts(depth: int, dimension: int) -> list[int]:
    """Per-axis dyadic split counts for ``depth`` total subdivisions."""
    quotient, remainder = divmod(depth, dimension)
    return [quotient + int(axis < remainder) for axis in range(dimension)]


def normalize_index(index: Iterable[str]) -> list[str]:
    """Whitespace-insensitive form of a Conley index polynomial tuple."""
    return [str(value).replace(" ", "") for value in index]


class UniformCoordinates:
    """Exact integer coordinates of uniform level-``depth`` grid cells.

    ``key`` converts a phase-space box to its integer grid coordinates and
    raises if the box is not aligned with the grid or does not have the grid
    cell size; ``flat`` linearizes those coordinates row-major.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray, depth: int) -> None:
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.depth = int(depth)
        self.sizes = np.asarray(
            [2**count for count in split_counts(depth, len(lower))],
            dtype=np.int64,
        )
        self.widths = (self.upper - self.lower) / self.sizes

    def key(self, box: Iterable[float]) -> tuple[int, ...]:
        values = np.asarray(list(box), dtype=np.float64)
        dimension = len(self.lower)
        raw = (values[:dimension] - self.lower) / self.widths
        rounded = np.rint(raw).astype(np.int64)
        if not np.allclose(raw, rounded, rtol=0.0, atol=2e-7):
            raise ValueError(
                f"box is not aligned with the level-{self.depth} grid: {values}"
            )
        if np.any(rounded < 0) or np.any(rounded >= self.sizes):
            raise ValueError(f"box coordinate outside grid: {rounded.tolist()}")
        expected_upper = self.lower + (rounded + 1) * self.widths
        if not np.allclose(
            values[dimension:], expected_upper, rtol=0.0, atol=2e-10
        ):
            raise ValueError(
                f"box does not have level-{self.depth} side lengths: {values}"
            )
        return tuple(int(value) for value in rounded)

    def flat(self, box: Iterable[float]) -> int:
        coordinates = self.key(box)
        result = 0
        stride = 1
        for coordinate, size in zip(coordinates, self.sizes, strict=True):
            result += coordinate * stride
            stride *= int(size)
        return int(result)


def morse_graph_cells(
    morse_graph: Any, coordinates: UniformCoordinates
) -> dict[int, set[tuple[int, ...]]]:
    """Grid keys of every recurrent component of a live CMGDB Morse graph."""
    return {
        int(node): {
            coordinates.key(box) for box in morse_graph.morse_set_boxes(int(node))
        }
        for node in morse_graph.vertices()
    }


def match_nodes(
    live: dict[int, set[tuple[int, ...]]],
    saved: dict[int, set[tuple[int, ...]]],
) -> tuple[dict[int, int], dict[str, Any]]:
    """Assign live recurrent components to saved labels by maximum Jaccard.

    Exhaustively searches label permutations, so it is intended for graphs
    with few recurrent components. Raises when the component counts differ or
    when the best assignment leaves any pair below Jaccard 0.95.
    """
    live_nodes = sorted(live)
    saved_labels = sorted(saved)
    if len(live_nodes) != len(saved_labels):
        raise RuntimeError(
            f"uniform graph has {len(live_nodes)} recurrent sets; saved graph has "
            f"{len(saved_labels)}"
        )

    def score(node: int, label: int) -> float:
        union = live[node] | saved[label]
        return len(live[node] & saved[label]) / len(union) if union else 1.0

    best_total = -math.inf
    best_assignment: tuple[int, ...] | None = None
    for labels in itertools.permutations(saved_labels):
        total = sum(score(node, label) for node, label in zip(live_nodes, labels, strict=True))
        if total > best_total:
            best_total = total
            best_assignment = labels
    assert best_assignment is not None
    mapping = dict(zip(live_nodes, best_assignment, strict=True))
    diagnostics: dict[str, Any] = {
        "total_jaccard": best_total,
        "nodes": {},
    }
    for node, label in mapping.items():
        intersection = len(live[node] & saved[label])
        union = len(live[node] | saved[label])
        diagnostics["nodes"][str(node)] = {
            "saved_label": label,
            "live_cells": len(live[node]),
            "saved_cells": len(saved[label]),
            "intersection": intersection,
            "jaccard": intersection / union if union else 1.0,
            "exact": live[node] == saved[label],
        }
    if min(value["jaccard"] for value in diagnostics["nodes"].values()) < 0.95:
        raise RuntimeError(f"live-to-saved component match is too weak: {diagnostics}")
    return mapping, diagnostics


class LocalIndexComputer:
    """Index pairs and Conley indices for cell sets of one live cell graph.

    For a cell set ``S`` the candidate pair is ``(F(S), F(S) \\ S)``.  The
    pair is valid when ``S`` is contained in its one-step image and no exit
    cell maps back into ``S`` inside the pair.  Valid pairs are evaluated with
    the CMGDB fork's ``ComputeConleyIndexForCells``.
    """

    def __init__(
        self,
        model: Any,
        map_graph: Any,
        morse_graph: Any,
        coordinates: UniformCoordinates,
    ) -> None:
        self.model = model
        self.map_graph = map_graph
        self.morse_graph = morse_graph
        self.coordinates = coordinates
        self._adjacency: dict[int, tuple[int, ...]] = {}
        self._flat: dict[int, int] = {}

    def adjacency(self, cell: int) -> tuple[int, ...]:
        cell = int(cell)
        if cell not in self._adjacency:
            self._adjacency[cell] = tuple(
                int(target) for target in self.map_graph.adjacencies(cell)
            )
        return self._adjacency[cell]

    def flat(self, cell: int) -> int:
        cell = int(cell)
        if cell not in self._flat:
            self._flat[cell] = self.coordinates.flat(
                self.morse_graph.phase_space_box(cell)
            )
        return self._flat[cell]

    def compute(self, name: str, cells: Iterable[int]) -> dict[str, Any]:
        started = time.perf_counter()
        recurrent = {int(cell) for cell in cells}
        image = {
            target
            for source in recurrent
            for target in self.adjacency(source)
        }
        missing_from_image = recurrent - image
        pair = image
        exit_set = pair - recurrent

        exit_violations: list[tuple[int, int]] = []
        local_native: dict[int, list[int]] = {}
        for source in pair:
            restricted = [
                target for target in self.adjacency(source) if target in pair
            ]
            local_native[source] = restricted
            if source in exit_set:
                for target in restricted:
                    if target in recurrent:
                        exit_violations.append((source, target))
                        if len(exit_violations) >= 20:
                            break

        pair_valid = not missing_from_image and not exit_violations
        result: dict[str, Any] = {
            "name": name,
            "morse_set_cells": len(recurrent),
            "image_cells": len(image),
            "pair_cells": len(pair),
            "exit_cells": len(exit_set),
            "map_edges": sum(map(len, local_native.values())),
            "checks": {
                "S_subset_F_S": not missing_from_image,
                "S_subset_F_S_missing_count": len(missing_from_image),
                "F_A_intersect_X_subset_A": not exit_violations,
                "F_A_intersect_X_subset_A_violation_examples": [
                    [int(source), int(target)]
                    for source, target in exit_violations
                ],
            },
            "pair_valid": pair_valid,
            "method": "CMGDB.ComputeConleyIndexForCells",
            "conley_index": None,
        }
        if pair_valid:
            result["conley_index"] = list(
                CMGDB.ComputeConleyIndexForCells(
                    self.model,
                    self.morse_graph,
                    sorted(recurrent),
                )
            )
        result["seconds"] = time.perf_counter() - started
        return result


def component_index_labels(model: Any, morse_graph: Any) -> dict[int, list[str]]:
    """Conley index of every recurrent component of a live CMGDB Morse graph."""
    result: dict[int, list[str]] = {}
    for node in morse_graph.vertices():
        node = int(node)
        result[node] = list(
            CMGDB.ComputeConleyIndexForCells(
                model,
                morse_graph,
                morse_graph.morse_set(node),
            )
        )
    return result


def parsed_live_graph(
    morse_graph: Any,
    indices: dict[int, list[str]],
    palette: list[str],
) -> MorseGraph:
    """Index-labelled :class:`MorseGraph` view of a live CMGDB Morse graph."""
    nodes = sorted(int(node) for node in morse_graph.vertices())
    edges: dict[int, list[int]] = {}
    for raw_source, raw_target in morse_graph.edges():
        source = int(raw_source)
        target = int(raw_target)
        edges.setdefault(source, []).append(target)
    labels = {
        node: f"{node} : ({', '.join(indices[node])})"
        for node in nodes
    }
    colors = {
        node: palette[node % len(palette)]
        for node in nodes
    }
    return MorseGraph(nodes=nodes, edges=edges, colors=colors, labels=labels)
