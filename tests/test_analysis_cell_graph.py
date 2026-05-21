from __future__ import annotations

import numpy as np
import torch

from latentdynamics.analysis.cell_graph import (
    build_reverse_csr,
    compute_cell_graph_roa,
    grid_boxes_overlapping_morse_set,
    reverse_reachable,
)
from latentdynamics.analysis.regions_of_attraction import load_box_roa


class _ConstantLatentMap(torch.nn.Module):
    def __init__(self, value: tuple[float, float]) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor(value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value.expand_as(x)


def _write_two_node_morse_artifacts(tmp_path):
    graph = tmp_path / "morse_graph"
    graph.write_text(
        "\n".join(
            [
                "digraph {",
                '0 [label="0", fillcolor="#111111"];',
                '1 [label="1", fillcolor="#eeeeee"];',
                "1 -> 0;",
                "}",
            ]
        )
    )
    sets = tmp_path / "morse_sets"
    np.savetxt(
        sets,
        np.array(
            [
                [0.0, 0.0, 0.1, 0.1, 0],
                [0.8, 0.8, 0.9, 0.9, 1],
            ],
            dtype=np.float64,
        ),
        delimiter=",",
    )
    return graph, sets


def test_cell_graph_roa_preserves_recurrent_morse_boxes_outside_lower_basin(tmp_path):
    graph, sets = _write_two_node_morse_artifacts(tmp_path)
    latent_map = _ConstantLatentMap((0.05, 0.05))

    table = load_box_roa(graph, sets)
    assert table.boxes.loc[table.boxes["morse_node"] == 1, "roa_label"].item() == 1

    cg = compute_cell_graph_roa(
        latent_map,
        graph,
        sets,
        resolution=8,
        bounds_padding=0.0,
        device="cpu",
    )

    node1_boxes = grid_boxes_overlapping_morse_set(
        cg.grid,
        np.array([[0.8, 0.8]], dtype=np.float64),
        np.array([[0.9, 0.9]], dtype=np.float64),
    )
    assert node1_boxes.size > 0
    assert np.all(cg.box_roa[node1_boxes] == 1)

    all_morse_boxes = grid_boxes_overlapping_morse_set(
        cg.grid,
        np.array([[0.0, 0.0], [0.8, 0.8]], dtype=np.float64),
        np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float64),
    )
    transient = np.setdiff1d(np.arange(cg.grid.n_boxes), all_morse_boxes)
    assert transient.size > 0
    assert np.any(cg.box_roa[transient] == 0)


def test_reverse_reachable_stops_at_other_recurrent_morse_sets():
    # Toy graph matching the intended semantics:
    # a is recurrent, b flows into recurrent c, d flows into recurrent e.
    # Spurious multivalued edges a -> b and c -> d should not make a,b,c part
    # of the transient RoA of e because a and c are blocking Morse sets.
    a, b, c, d, e = range(5)
    adjacency = [
        np.array([a, b], dtype=np.int64),
        np.array([c], dtype=np.int64),
        np.array([c, d], dtype=np.int64),
        np.array([e], dtype=np.int64),
        np.array([e], dtype=np.int64),
    ]
    rev_ptr, rev_neighbors = build_reverse_csr(adjacency, n_boxes=5)
    blocked = np.zeros(5, dtype=bool)
    blocked[[a, c]] = True

    reachable = reverse_reachable(
        rev_ptr,
        rev_neighbors,
        targets=np.array([e], dtype=np.int64),
        n_boxes=5,
        blocked=blocked,
    )

    np.testing.assert_array_equal(reachable, np.array([False, False, False, True, True]))
