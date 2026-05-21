from __future__ import annotations

from pathlib import Path

import numpy as np

from latentdynamics.analysis.cmgdb_roa import (
    EXACT_ROA_FILENAME,
    CellROA,
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
