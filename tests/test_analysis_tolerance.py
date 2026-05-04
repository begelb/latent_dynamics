"""Tests for the geometric Box/Edge/MorseSet primitives and tau-bar tolerance."""

from __future__ import annotations

import math

import numpy as np

from latentdynamics.analysis import (
    Box,
    Edge,
    MorseSet,
    compute_min_boundary_separation,
    distance_point_to_boundary,
    is_in_range,
    orthogonal_distance,
)


def _write_morse_sets_csv(path, rows):
    arr = np.asarray(rows, dtype=np.float64)
    np.savetxt(path, arr, delimiter=",")


class TestEdge:
    def test_make_horizontal(self):
        e = Edge.make((0.0, 0.0), (1.0, 0.0))
        assert e.orientation == "horizontal"
        assert e.u == (0.0, 0.0)
        assert e.v == (1.0, 0.0)

    def test_make_vertical(self):
        e = Edge.make((0.0, 1.0), (0.0, 0.0))
        assert e.orientation == "vertical"
        assert e.u == (0.0, 0.0)  # canonical ordering

    def test_edges_are_hashable_and_equal(self):
        a = Edge.make((0.0, 0.0), (1.0, 0.0))
        b = Edge.make((1.0, 0.0), (0.0, 0.0))
        assert a == b
        assert hash(a) == hash(b)


class TestBoxAndDistance:
    def test_is_in_range(self):
        edge = Edge.make((0.0, 0.0), (1.0, 0.0))
        assert is_in_range((0.5, 5.0), edge)
        assert not is_in_range((1.5, 0.0), edge)

    def test_orthogonal_distance_horizontal(self):
        edge = Edge.make((0.0, 0.0), (1.0, 0.0))
        assert orthogonal_distance((0.5, 3.0), edge) == 3.0

    def test_distance_to_boundary_picks_closest_in_range(self):
        edges = {
            Edge.make((0.0, 0.0), (1.0, 0.0)),
            Edge.make((0.0, 1.0), (1.0, 1.0)),
            Edge.make((0.0, 0.0), (0.0, 1.0)),
            Edge.make((1.0, 0.0), (1.0, 1.0)),
        }
        # interior point closest to bottom edge (distance 0.2)
        assert distance_point_to_boundary((0.5, 0.2), edges) == 0.2
        # closest to top edge (distance 0.1)
        assert math.isclose(distance_point_to_boundary((0.5, 0.9), edges), 0.1, abs_tol=1e-9)


class TestMorseSet:
    def test_loads_only_target_label(self, tmp_path):
        path = tmp_path / "morse_sets"
        _write_morse_sets_csv(
            path,
            [
                [0, 0, 1, 1, 0],
                [1, 0, 2, 1, 0],
                [0, 1, 1, 2, 1],  # different label, should be ignored
            ],
        )
        m = MorseSet(path, label=0)
        assert len(m) == 2
        assert all(b.M_label == 0 for b in m)

    def test_boundary_edges_of_two_adjacent_squares(self, tmp_path):
        path = tmp_path / "morse_sets"
        _write_morse_sets_csv(path, [[0, 0, 1, 1, 0], [1, 0, 2, 1, 0]])
        m = MorseSet(path, label=0)
        # Two unit squares glued along x=1: 6 outer edges remain.
        edges = m.boundary_edges()
        assert len(edges) == 6

    def test_vertices_unique(self, tmp_path):
        path = tmp_path / "morse_sets"
        _write_morse_sets_csv(path, [[0, 0, 1, 1, 0], [1, 0, 2, 1, 0]])
        m = MorseSet(path, label=0)
        vs = m.vertices()
        assert vs.shape == (6, 2)


class TestTauBar:
    def test_identity_dynamics_distance_is_zero_at_corners(self, tmp_path):
        path = tmp_path / "morse_sets"
        _write_morse_sets_csv(path, [[0, 0, 1, 1, 0]])
        m = MorseSet(path, label=0)
        # An identity map sends each corner to itself; the distance from the
        # corner to the boundary is zero.
        assert compute_min_boundary_separation(m, lambda x: x) == 0.0

    def test_inward_map_yields_positive_separation(self, tmp_path):
        path = tmp_path / "morse_sets"
        _write_morse_sets_csv(path, [[0, 0, 1, 1, 0]])
        m = MorseSet(path, label=0)
        # Pull every corner toward the center (0.5, 0.5) by a quarter.
        def inward(verts):
            return 0.75 * verts + 0.25 * np.array([0.5, 0.5])
        sep = compute_min_boundary_separation(m, inward)
        assert sep > 0.0
