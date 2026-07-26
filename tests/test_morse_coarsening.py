from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latentdynamics.analysis.morse_coarsening import (
    coarsen_morse_graph,
    compute_connection_complete_morse_sets,
    write_connection_complete_morse_sets,
    write_quotient_morse_sets,
)
from latentdynamics.analysis.morse_graph_parser import MorseGraph


def _graph(tmp_path: Path, dot: str) -> MorseGraph:
    path = tmp_path / "morse_graph"
    path.write_text(dot)
    return MorseGraph.from_dot(path)


class _MapGraph:
    def __init__(self, adjacency: list[list[int]]) -> None:
        self._adjacency = adjacency

    def num_vertices(self) -> int:
        return len(self._adjacency)

    def adjacencies(self, node: int):
        return self._adjacency[node]


class _MorseCells:
    def __init__(self, cells: dict[int, list[int]]) -> None:
        self._cells = cells

    def morse_set(self, node: int):
        return self._cells[node]

    def phase_space_box(self, cell: int):
        lower = float(cell)
        return [lower, 0.0, lower + 1.0, 1.0]


def test_chafee_nonattracting_nodes_collapse_to_m1_epimorphism(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        """digraph {
0 [label="0 : (x-1, 0, 0)", fillcolor="#ffb000ff"];
1 [label="1 : (x-1, 0, 0)", fillcolor="#dc267fff"];
2 [label="2 : (0, x-1, 0)"];
3 [label="3 : (0, x-1, 0)"];
4 [label="4 : (0, 0, x-1)"];
5 [label="5 : (0, x-1, 0)"];
6 [label="6 : (0, 0, x-1)"];
2 -> 1;
3 -> 1;
3 -> 0;
4 -> 2;
4 -> 3;
5 -> 0;
6 -> 3;
6 -> 5;
}
""",
    )
    quotient = coarsen_morse_graph(
        graph,
        [{2, 3, 4, 5, 6}],
        labels={
            frozenset({0}): "M(0^-)",
            frozenset({1}): "M(0^+)",
            frozenset({2, 3, 4, 5, 6}): "M(1)",
        },
    )

    assert quotient.projection == {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}
    assert quotient.fibers[2] == frozenset({2, 3, 4, 5, 6})
    assert quotient.graph.edges == {2: [0, 1]}
    assert quotient.graph.minimal == {0, 1}
    assert quotient.graph.labels[0] == "M(0^-)"
    assert quotient.graph.labels[1] == "M(0^+)"
    assert quotient.graph.labels[2] == "M(1)"
    assert quotient.graph.colors[0] == "#ffb000ff"
    assert quotient.graph.colors[1] == "#dc267fff"

    # Every fine order relation is either collapsed inside a fiber or remains
    # an order relation after applying the surjective projection.
    for source in graph.nodes:
        for target in graph.descendants[source]:
            q_source = quotient.projection[source]
            q_target = quotient.projection[target]
            assert q_source == q_target or q_target in quotient.graph.descendants[q_source]


def test_nonconvex_merge_that_creates_cycle_is_rejected(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        """digraph {\n0 [label="0"];\n1 [label="1"];\n2 [label="2"];\n0 -> 1;\n1 -> 2;\n}\n""",
    )
    with pytest.raises(ValueError, match="creates a cycle"):
        coarsen_morse_graph(graph, [{0, 2}])


def test_morse_set_csv_is_relabelled_as_geometric_union(tmp_path: Path) -> None:
    source = tmp_path / "fine_sets"
    source.write_text("0,0,1,1,2\n1,0,2,1,3\n2,0,3,1,0\n")
    destination = tmp_path / "coarse_sets"

    write_quotient_morse_sets(source, destination, {0: 0, 2: 1, 3: 1})
    data = np.loadtxt(destination, delimiter=",", ndmin=2)

    assert data[:, -1].astype(int).tolist() == [1, 1, 0]
    np.testing.assert_allclose(data[:, :-1], np.loadtxt(source, delimiter=",", ndmin=2)[:, :-1])


def test_connection_completion_is_fiber_downset_intersect_upset() -> None:
    # Fine Morse cells 0 and 2 are collapsed. Cell 1 is on a connection
    # 0 -> 1 -> 2. Cell 3 can reach the fiber but is not reachable from it;
    # cell 4 is reachable from the fiber but cannot return to it.
    map_graph = _MapGraph(
        [
            [0, 1, 4],
            [2],
            [2],
            [2],
            [],
        ]
    )
    fine_morse_graph = _MorseCells({10: [0], 11: [2]})

    completed = compute_connection_complete_morse_sets(
        map_graph,
        fine_morse_graph,
        {10: 7, 11: 7},
    )

    assert completed.fibers == {7: frozenset({10, 11})}
    np.testing.assert_array_equal(completed.cells[7], np.array([0, 1, 2]))
    np.testing.assert_array_equal(completed.connection_cells[7], np.array([1]))
    assert completed.overlaps == {}


def test_connection_completion_leaves_singleton_fiber_verbatim() -> None:
    map_graph = _MapGraph([[0, 1], [2], [2]])
    fine_morse_graph = _MorseCells({10: [0]})

    completed = compute_connection_complete_morse_sets(
        map_graph,
        fine_morse_graph,
        {10: 7},
    )

    np.testing.assert_array_equal(completed.cells[7], np.array([0]))
    assert completed.connection_cells[7].size == 0


def test_connection_completion_rejects_foreign_recurrent_node_in_interval() -> None:
    map_graph = _MapGraph([[0, 1], [1, 2], [2]])
    fine_morse_graph = _MorseCells({10: [0], 11: [1], 12: [2]})

    with pytest.raises(ValueError, match="not order-convex"):
        compute_connection_complete_morse_sets(
            map_graph,
            fine_morse_graph,
            {10: 7, 11: 8, 12: 7},
        )


def test_connection_complete_csv_contains_added_path_cells(tmp_path: Path) -> None:
    map_graph = _MapGraph([[0, 1], [2], [2]])
    fine_morse_graph = _MorseCells({10: [0], 11: [2]})
    completed = compute_connection_complete_morse_sets(
        map_graph,
        fine_morse_graph,
        {10: 7, 11: 7},
    )

    destination = write_connection_complete_morse_sets(
        fine_morse_graph,
        completed,
        tmp_path / "coarse_sets",
    )
    data = np.loadtxt(destination, delimiter=",", ndmin=2)

    np.testing.assert_allclose(
        data,
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0, 7.0],
                [1.0, 0.0, 2.0, 1.0, 7.0],
                [2.0, 0.0, 3.0, 1.0, 7.0],
            ]
        ),
    )


def test_connection_completion_reports_outer_enclosure_overlap(tmp_path: Path) -> None:
    # Both quotient fibers have a connection through cell 4. This is not a
    # valid poset epimorphism, but detecting the overlap keeps the CSV writer
    # from silently assigning the shared enclosure cell to one set.
    map_graph = _MapGraph(
        [
            [0, 4],
            [1],
            [2, 4],
            [3],
            [1, 3],
        ]
    )
    fine_morse_graph = _MorseCells({10: [0], 11: [1], 12: [2], 13: [3]})
    completed = compute_connection_complete_morse_sets(
        map_graph,
        fine_morse_graph,
        {10: 7, 11: 7, 12: 8, 13: 8},
    )

    assert completed.overlaps == {4: frozenset({7, 8})}
    with pytest.raises(ValueError, match="enclosures overlap"):
        write_connection_complete_morse_sets(
            fine_morse_graph,
            completed,
            tmp_path / "coarse_sets",
        )
