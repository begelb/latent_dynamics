from __future__ import annotations

import numpy as np
import pytest

from latentdynamics.analysis.basin_statistics import (
    OUTSIDE,
    ChafeeBasinStatistics,
    classify_points_in_cell_roa,
    cmgdb_morton_cell_indices,
    compute_chafee_basin_statistics,
)
from latentdynamics.analysis.cmgdb_roa import BOUNDARY, ESCAPE, CellROA


def test_cmgdb_morton_indices_interleave_bits_msb_first():
    one_dimensional = np.arange(8, dtype=np.int64)[:, None]
    np.testing.assert_array_equal(
        cmgdb_morton_cell_indices(one_dimensional, [8]),
        np.arange(8, dtype=np.int64),
    )

    two_dimensional = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [3, 3],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(
        cmgdb_morton_cell_indices(two_dimensional, [4, 4]),
        np.array([0, 1, 2, 3, 8, 15], dtype=np.int64),
    )

    three_dimensional = np.array(
        list(np.ndindex(2, 2, 2)),
        dtype=np.int64,
    )
    expected = (
        4 * three_dimensional[:, 0]
        + 2 * three_dimensional[:, 1]
        + three_dimensional[:, 2]
    )
    np.testing.assert_array_equal(
        cmgdb_morton_cell_indices(three_dimensional, [2, 2, 2]),
        expected,
    )


def test_cmgdb_morton_indices_support_incomplete_final_axis_cycle():
    bins = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 1, 1],
        ],
        dtype=np.int64,
    )
    # Split path is x_msb, y, z, x_lsb for shape (4, 2, 2).
    np.testing.assert_array_equal(
        cmgdb_morton_cell_indices(bins, [4, 2, 2]),
        np.array([0, 1, 8, 15], dtype=np.int64),
    )


def test_uniform_grid_classification_uses_morton_ids_and_closed_boundaries():
    # Morton ids on a 2x2 grid are:
    #   (x-low, y-low)=0, (x-low, y-high)=1,
    #   (x-high,y-low)=2, (x-high,y-high)=3.
    roa = CellROA(
        box_roa=np.array([10, 10, 20, ESCAPE], dtype=np.int32),
        bounds_lower=np.array([0.0, 0.0]),
        bounds_upper=np.array([1.0, 1.0]),
        grid_shape=np.array([2, 2]),
        minimal_order=np.array([10, 20], dtype=np.int32),
    )
    points = np.array(
        [
            [0.25, 0.25],  # id 0: negative basin
            [0.25, 0.75],  # id 1: same basin, verifies Morton ordering
            [0.75, 0.25],  # id 2: positive basin
            [0.75, 0.75],  # id 3: escaping/unassigned cell
            [0.50, 0.25],  # shared by ids 0 and 2: conflicting boundary
            [0.25, 0.50],  # shared by ids 0 and 1: labels agree
            [0.75, 0.50],  # basin/escape shared face: boundary
            [-0.01, 0.25],  # outside the cell complex
        ]
    )

    result = classify_points_in_cell_roa(points, roa)

    np.testing.assert_array_equal(
        result.basin_labels,
        np.array([10, 10, 20, ESCAPE, BOUNDARY, 10, BOUNDARY, OUTSIDE]),
    )
    np.testing.assert_array_equal(
        result.matched_cell_counts,
        np.array([1, 1, 1, 1, 2, 2, 2, 0]),
    )
    assert result.geometry_mode == "uniform_cmgdb_morton"


def test_uniform_three_dimensional_lookup_never_needs_explicit_boxes():
    bins = np.array(list(np.ndindex(2, 2, 2)), dtype=np.int64)
    points = (bins + 0.5) / 2.0
    cell_labels = np.arange(8, dtype=np.int32)
    roa = CellROA(
        box_roa=cell_labels,
        bounds_lower=np.zeros(3),
        bounds_upper=np.ones(3),
        grid_shape=np.full(3, 2),
        boxes=None,
        minimal_order=cell_labels,
    )

    result = classify_points_in_cell_roa(points, roa)

    expected = cmgdb_morton_cell_indices(bins, [2, 2, 2]).astype(np.int32)
    np.testing.assert_array_equal(result.basin_labels, expected)
    np.testing.assert_array_equal(result.matched_cell_counts, np.ones(8, dtype=np.int32))


def test_uniform_lookup_accepts_an_explicit_cell_index_callback():
    roa = CellROA(
        # C-order labels: (0,0), (0,1), (1,0), (1,1).
        box_roa=np.array([10, 11, 12, 13], dtype=np.int32),
        bounds_lower=np.zeros(2),
        bounds_upper=np.ones(2),
        grid_shape=np.array([2, 2]),
        minimal_order=np.array([10, 11, 12, 13]),
    )

    def c_order(bins, shape):
        return np.ravel_multi_index(bins.T, tuple(shape.tolist()))

    result = classify_points_in_cell_roa(
        np.array([[0.25, 0.75], [0.75, 0.25]]),
        roa,
        uniform_grid_order="custom",
        cell_indexer=c_order,
    )

    np.testing.assert_array_equal(result.basin_labels, np.array([11, 12]))
    assert result.geometry_mode == "uniform_custom_indexer"


def test_explicit_box_classification_is_dimension_generic_and_order_independent():
    # Explicit CMGDB layout is lower[3], upper[3].
    boxes = np.array(
        [
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
            [0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    roa = CellROA(
        box_roa=np.array([7, 9], dtype=np.int32),
        boxes=boxes,
        minimal_order=np.array([7, 9], dtype=np.int32),
    )
    points = np.array(
        [
            [0.25, 0.5, 0.5],
            [0.75, 0.5, 0.5],
            [0.50, 0.5, 0.5],
            [1.25, 0.5, 0.5],
        ]
    )

    forward = classify_points_in_cell_roa(points, roa)
    reversed_roa = CellROA(
        box_roa=roa.box_roa[::-1],
        boxes=boxes[::-1],
        minimal_order=roa.minimal_order,
    )
    reversed_result = classify_points_in_cell_roa(points, reversed_roa)

    expected = np.array([7, 9, BOUNDARY, OUTSIDE], dtype=np.int32)
    np.testing.assert_array_equal(forward.basin_labels, expected)
    np.testing.assert_array_equal(reversed_result.basin_labels, expected)
    np.testing.assert_array_equal(forward.matched_cell_counts, np.array([1, 1, 2, 0]))
    assert forward.geometry_mode == "explicit_boxes"


def test_chafee_statistics_condition_on_nonzero_truth_and_conserve_counts():
    truth = np.array([-1, -1, -1, 1, 1, 1, 0])
    predicted = np.array([10, 20, OUTSIDE, 10, 20, BOUNDARY, 10])

    result = compute_chafee_basin_statistics(
        truth,
        predicted,
        negative_basin_label=10,
        positive_basin_label=20,
    )

    assert result.total_trajectories == 7
    assert result.excluded_zero_trajectories == 1
    assert result.conditioned_trajectories == 6
    assert result.counts() == {
        "outside_both_basins": 2,
        "misclassified_in_negative_basin": 1,
        "misclassified_in_positive_basin": 1,
        "correctly_classified_in_negative_basin": 1,
        "correctly_classified_in_positive_basin": 1,
    }
    assert sum(result.counts().values()) == result.conditioned_trajectories
    assert result.percentages()["outside_both_basins"] == pytest.approx(100.0 / 3.0)
    assert sum(result.percentages().values()) == pytest.approx(100.0)


def test_chafee_statistics_accept_point_classification_result():
    roa = CellROA(
        box_roa=np.array([3, 4], dtype=np.int32),
        boxes=np.array([[0.0, 0.5], [0.5, 1.0]]),
        minimal_order=np.array([3, 4], dtype=np.int32),
    )
    classification = classify_points_in_cell_roa(
        np.array([[0.25], [0.75], [1.25]]),
        roa,
    )

    result = compute_chafee_basin_statistics(
        np.array([-1, 1, 1]),
        classification,
        negative_basin_label=3,
        positive_basin_label=4,
    )

    assert result.correctly_classified_in_negative_basin == 1
    assert result.correctly_classified_in_positive_basin == 1
    assert result.outside_both_basins == 1


def test_chafee_percentages_reproduce_reference_table_denominator():
    result = ChafeeBasinStatistics(
        total_trajectories=10_000,
        excluded_zero_trajectories=2_138,
        conditioned_trajectories=7_862,
        outside_both_basins=1_694,
        misclassified_in_negative_basin=2,
        misclassified_in_positive_basin=3,
        correctly_classified_in_negative_basin=3_094,
        correctly_classified_in_positive_basin=3_069,
    )

    percentages = result.percentages()

    assert round(percentages["outside_both_basins"], 2) == 21.55
    assert round(percentages["misclassified_in_negative_basin"], 3) == 0.025
    assert round(percentages["misclassified_in_positive_basin"], 3) == 0.038
    assert round(percentages["correctly_classified_in_negative_basin"], 2) == 39.35
    assert round(percentages["correctly_classified_in_positive_basin"], 2) == 39.04


def test_chafee_statistics_reject_nonconserving_manual_counts():
    with pytest.raises(ValueError, match="conditioned trajectories"):
        ChafeeBasinStatistics(
            total_trajectories=3,
            excluded_zero_trajectories=0,
            conditioned_trajectories=3,
            outside_both_basins=1,
            misclassified_in_negative_basin=0,
            misclassified_in_positive_basin=0,
            correctly_classified_in_negative_basin=0,
            correctly_classified_in_positive_basin=0,
        )
