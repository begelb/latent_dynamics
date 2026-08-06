"""Tests for the direct original-3D Ives CMGDB reference computation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import CMGDB
import numpy as np

from latentdynamics.systems import IvesModel

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compute_ives_myvatn_3d_ground_truth.py"
)
SPEC = importlib.util.spec_from_file_location("compute_ives_ground_truth", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GROUND_TRUTH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GROUND_TRUTH)


def test_degenerate_intervals_contain_exact_point_map() -> None:
    system = IvesModel()
    box_map = GROUND_TRUTH.IvesLogIntervalBoxMap(system)
    points = np.asarray(
        [
            [-2.0, -5.0, -1.0],
            [0.0, 0.0, 0.0],
            [-1.293838452429635, -0.24529315028655657, -1.2247337524036408],
        ],
        dtype=np.float64,
    )
    rectangles = np.column_stack((points, points))

    intervals = box_map._evaluate(rectangles)
    images = system.step(points)

    assert np.all(images >= intervals[:, :3])
    assert np.all(images <= intervals[:, 3:])
    assert np.max(intervals[:, 3:] - intervals[:, :3]) < 1e-12


def test_degenerate_intervals_cover_algebraically_equivalent_double_paths() -> None:
    system = IvesModel()
    box_map = GROUND_TRUTH.IvesLogIntervalBoxMap(system)
    points = np.asarray(
        [
            [-0.597073415872267, -0.4689340634237006, -0.07377007950702641],
            [1.00661787437084, -0.5945418710728969, -0.2822470030662204],
        ],
        dtype=np.float64,
    )

    intervals = box_map._evaluate(np.column_stack((points, points)))
    images = system.step(points)

    assert np.all(images >= intervals[:, :3])
    assert np.all(images <= intervals[:, 3:])


def test_random_aligned_boxes_pass_enclosure_challenge() -> None:
    box_map = GROUND_TRUTH.IvesLogIntervalBoxMap(IvesModel())

    audit = GROUND_TRUTH.validate_interval_enclosure(
        box_map,
        domain_lower=GROUND_TRUTH.TRAPPING_LOWER,
        domain_upper=GROUND_TRUTH.TRAPPING_UPPER,
        seed=7,
        boxes_per_level=6,
        points_per_box=10,
    )

    assert audit["passed"] is True
    assert audit["total_points"] == 6 * 6 * 18
    assert audit["corners_per_box"] == 8
    assert all(value >= 0.0 for value in audit["minimum_lower_slack_by_coordinate"])
    assert all(value >= 0.0 for value in audit["minimum_upper_slack_by_coordinate"])


def test_trapping_box_is_interval_forward_invariant() -> None:
    box_map = GROUND_TRUTH.IvesLogIntervalBoxMap(IvesModel())
    rectangle = np.concatenate(
        (GROUND_TRUTH.TRAPPING_LOWER, GROUND_TRUTH.TRAPPING_UPPER)
    )

    image = box_map._evaluate(rectangle.reshape(1, 6))[0]

    assert np.all(image[:3] >= GROUND_TRUTH.TRAPPING_LOWER)
    assert np.all(image[3:] <= GROUND_TRUTH.TRAPPING_UPPER)


def test_archived_sampling_box_enters_trapping_box_by_fifth_iterate() -> None:
    box_map = GROUND_TRUTH.IvesLogIntervalBoxMap(IvesModel())

    audit = GROUND_TRUTH.sampling_absorption_audit(box_map)

    assert audit["passed"] is True
    assert audit["steps"] == 5
    assert all(value > 0.0 for value in audit["final_lower_margin"])
    assert all(value > 0.0 for value in audit["final_upper_margin"])


def test_scalar_and_batch_callbacks_agree_and_count_work() -> None:
    box_map = GROUND_TRUTH.IvesLogIntervalBoxMap(IvesModel())
    rectangles = [
        [-3.0, -6.0, -2.0, -2.5, -5.5, -1.5],
        [-1.0, -1.0, -1.0, -0.5, -0.5, -0.5],
    ]

    batch = np.asarray(box_map.batch(rectangles))
    scalar = np.asarray([box_map(rectangle) for rectangle in rectangles])

    np.testing.assert_array_equal(batch, scalar)
    assert box_map.stats() == {
        "scalar_calls": 2,
        "batch_calls": 1,
        "rectangles": 4,
        "internal_refinement": 0,
        "enclosure_leaf_rectangles": 4,
    }


def test_internal_refinement_hulls_eight_children_without_widening() -> None:
    system = IvesModel()
    direct = GROUND_TRUTH.IvesLogIntervalBoxMap(system)
    refined = GROUND_TRUTH.IvesLogIntervalBoxMap(system, internal_refinement=1)
    rectangles = np.asarray(
        [
            [-2.0, -5.0, -2.0, -1.5, -4.0, -1.0],
            [-0.5, -0.8, -0.4, 0.4, 0.2, 0.5],
        ],
        dtype=np.float64,
    )

    coarse_images = direct._evaluate(rectangles)
    refined_images = refined._evaluate(rectangles)

    assert np.all(refined_images[:, :3] >= coarse_images[:, :3])
    assert np.all(refined_images[:, 3:] <= coarse_images[:, 3:])
    assert np.any(
        (refined_images[:, :3] > coarse_images[:, :3])
        | (refined_images[:, 3:] < coarse_images[:, 3:])
    )

    fractions = np.asarray([0.0, 0.5, 1.0])
    lattice = np.asarray(
        np.meshgrid(fractions, fractions, fractions, indexing="ij")
    ).reshape(3, -1).T
    for rectangle, enclosure in zip(rectangles, refined_images, strict=True):
        lower = rectangle[:3]
        upper = rectangle[3:]
        points = lower + lattice * (upper - lower)
        images = system.step(points)
        assert np.all(images >= enclosure[:3])
        assert np.all(images <= enclosure[3:])

    batch = np.asarray(refined.batch(rectangles.tolist()))
    np.testing.assert_array_equal(batch, refined_images)
    assert refined.stats() == {
        "scalar_calls": 0,
        "batch_calls": 1,
        "rectangles": 2,
        "internal_refinement": 1,
        "enclosure_leaf_rectangles": 16,
    }


def test_tiny_cmgdb_run_uses_batch_callback_and_finds_recurrence() -> None:
    box_map = GROUND_TRUTH.IvesLogIntervalBoxMap(IvesModel())
    model = CMGDB.Model(
        3,
        3,
        3,
        1_000,
        GROUND_TRUTH.TRAPPING_LOWER.tolist(),
        GROUND_TRUTH.TRAPPING_UPPER.tolist(),
        box_map,
    )
    model.set_batch_map(box_map.batch)

    morse_graph = CMGDB.ComputeMorseGraphOnly(model)

    assert list(morse_graph.vertices())
    assert box_map.stats()["batch_calls"] > 0
    assert box_map.stats()["rectangles"] == 8


def test_subdivision_limit_assessment_distinguishes_known_states() -> None:
    assessment = GROUND_TRUTH._subdivision_limit_assessment(
        {"0": 10, "1": 60, "2": 600},
        minimum=6,
        maximum=9,
        limit=1_000,
    )

    states = {
        node: record["state"]
        for node, record in assessment["assessment_by_node"].items()
    }
    assert states == {
        "0": "guaranteed_no_size_limit_stop_through_maximum",
        "1": "guaranteed_no_size_limit_stop_through_maximum",
        "2": "guaranteed_stop_before_first_post_minimum_decomposition",
    }
