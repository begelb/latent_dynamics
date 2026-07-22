from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latentdynamics.analysis.morse_coarsening import (
    coarsen_morse_graph,
    write_quotient_morse_sets,
)
from latentdynamics.analysis.morse_graph_parser import MorseGraph


def _graph(tmp_path: Path, dot: str) -> MorseGraph:
    path = tmp_path / "morse_graph"
    path.write_text(dot)
    return MorseGraph.from_dot(path)


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
