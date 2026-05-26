from __future__ import annotations

from pathlib import Path

import numpy as np

from latentdynamics.analysis.cmgdb_roa import (
    EXACT_ROA_FILENAME,
    MULTI,
    CellROA,
    collapse_roa_to_lca,
    compute_exact_roa,
    load_exact_roa,
    save_exact_roa,
)
from latentdynamics.analysis.regions_of_attraction import MorseGraph


class _MapGraph:
    def __init__(self, adjacency: list[list[int]]) -> None:
        self._adjacency = adjacency

    def num_vertices(self) -> int:
        return len(self._adjacency)

    def adjacencies(self, v: int):
        return self._adjacency[v]


class _MorseGraphCells:
    def __init__(self, node_to_cells: dict[int, list[int]]) -> None:
        self._node_to_cells = node_to_cells

    def morse_set(self, node: int):
        return self._node_to_cells[node]

    def phase_space_box(self, cell_id: int):
        lo = float(cell_id)
        return [lo, 0.0, lo + 1.0, 1.0]


def test_exact_roa_blocks_other_recurrent_morse_sets():
    # Cell-level version of:
    # a -> a -> b -> c -> c -> d -> e, with Morse sets a, c, e.
    # The multivalued edges a -> b and c -> d must not leak a,b,c into
    # the transient RoA of e because a and c are recurrent blockers.
    a, b, c, d, e = range(5)
    map_graph = _MapGraph(
        [
            [a, b],
            [c],
            [c, d],
            [e],
            [e],
        ]
    )
    cmgdb_morse = _MorseGraphCells({0: [a], 1: [c], 2: [e]})
    morse_dag = MorseGraph(nodes=[0, 1, 2], edges={}, colors={}, labels={})

    roa = compute_exact_roa(map_graph, cmgdb_morse, morse_dag)

    np.testing.assert_array_equal(roa.box_roa, np.array([0, 1, 1, 2, 2], dtype=np.int32))


def test_exact_roa_artifact_round_trips_uniform_shape(tmp_path: Path):
    roa = CellROA(
        box_roa=np.array([0, 1, -1, -2], dtype=np.int32),
        bounds_lower=np.array([-1.0, -1.0], dtype=np.float64),
        bounds_upper=np.array([1.0, 1.0], dtype=np.float64),
        grid_shape=np.array([2, 2], dtype=np.int64),
    )

    path = save_exact_roa(roa, tmp_path)
    loaded = load_exact_roa(path)

    assert path == tmp_path / EXACT_ROA_FILENAME
    np.testing.assert_array_equal(loaded.box_roa, roa.box_roa)
    np.testing.assert_allclose(loaded.bounds_lower, roa.bounds_lower)
    np.testing.assert_allclose(loaded.bounds_upper, roa.bounds_upper)
    np.testing.assert_array_equal(loaded.grid_shape, roa.grid_shape)


def _two_basin_fixture():
    """cell 2 flows to both minima (cell 0 and cell 1) without crossing a
    recurrent set; node 2 is the saddle LCA of the two minima."""
    map_graph = _MapGraph([[0], [1], [0, 1]])
    cmgdb_morse = _MorseGraphCells({0: [0], 1: [1], 2: []})
    morse_dag = MorseGraph(nodes=[0, 1, 2], edges={2: [0, 1]}, colors={}, labels={})
    return map_graph, cmgdb_morse, morse_dag


def test_exact_roa_uncollapsed_marks_multi_basin_cells():
    map_graph, cmgdb_morse, morse_dag = _two_basin_fixture()

    roa = compute_exact_roa(map_graph, cmgdb_morse, morse_dag, collapse_to_lca=False)

    # The multi-basin cell stays MULTI instead of collapsing to the LCA node 2.
    np.testing.assert_array_equal(roa.box_roa, np.array([0, 1, MULTI], dtype=np.int32))
    # The full reachable-minimal set is exposed via the bitmask + ordering.
    np.testing.assert_array_equal(roa.minimal_order, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        roa.reach_mask, np.array([0b01, 0b10, 0b11], dtype=np.uint64)
    )


def test_collapse_roa_to_lca_matches_inline_collapse():
    map_graph, cmgdb_morse, morse_dag = _two_basin_fixture()

    collapsed = compute_exact_roa(map_graph, cmgdb_morse, morse_dag, collapse_to_lca=True)
    uncollapsed = compute_exact_roa(
        map_graph, cmgdb_morse, morse_dag, collapse_to_lca=False
    )

    relabeled = collapse_roa_to_lca(uncollapsed, morse_dag)

    np.testing.assert_array_equal(collapsed.box_roa, np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(relabeled, collapsed.box_roa)
    # The bitmask is retained even when collapse runs inline (default path).
    assert collapsed.reach_mask is not None


def test_exact_roa_default_collapses_to_lca():
    map_graph, cmgdb_morse, morse_dag = _two_basin_fixture()

    roa = compute_exact_roa(map_graph, cmgdb_morse, morse_dag)

    np.testing.assert_array_equal(roa.box_roa, np.array([0, 1, 2], dtype=np.int32))


def test_exact_roa_artifact_round_trips_reach_mask_and_minimal_order(tmp_path: Path):
    roa = CellROA(
        box_roa=np.array([0, 1, MULTI], dtype=np.int32),
        reach_mask=np.array([1, 2, 3], dtype=np.uint64),
        minimal_order=np.array([0, 1], dtype=np.int32),
    )

    loaded = load_exact_roa(save_exact_roa(roa, tmp_path))

    np.testing.assert_array_equal(loaded.reach_mask, roa.reach_mask)
    assert loaded.reach_mask.dtype == np.uint64
    np.testing.assert_array_equal(loaded.minimal_order, roa.minimal_order)


def test_exact_roa_artifact_without_reach_mask_loads_none(tmp_path: Path):
    roa = CellROA(box_roa=np.array([0, 1], dtype=np.int32))

    loaded = load_exact_roa(save_exact_roa(roa, tmp_path))

    assert loaded.reach_mask is None
    assert loaded.minimal_order is None
