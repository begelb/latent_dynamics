"""Regression tests for the canonical Lake Myvatn Ives map."""

from __future__ import annotations

import numpy as np
import pytest

from latentdynamics.systems import IvesModel, build_system


def test_canonical_parameters_and_log_sampling_box() -> None:
    system = IvesModel()

    assert system.params == {
        "r1": 3.873,
        "r2": 11.746,
        "c": 10**-6.435,
        "d": 0.5517,
        "p": 0.06659,
        "q": 0.9026,
        "coordinate_mode": "log",
    }
    np.testing.assert_array_equal(system.lower_bounds, [-3.0, -7.5, -3.0])
    np.testing.assert_array_equal(system.upper_bounds, [1.5, 1.5, 1.5])


def test_log_map_matches_archived_reference_value() -> None:
    system = IvesModel()

    actual = system.step(np.array([-2.0, -5.0, -1.0]))

    np.testing.assert_allclose(
        actual,
        [-1.7710508562222909, -4.034220371989642, -1.4172053595323961],
        rtol=1e-14,
        atol=0.0,
    )


def test_scalar_and_batched_steps_match() -> None:
    system = IvesModel()
    points = np.array(
        [
            [-2.0, -5.0, -1.0],
            [0.0, 0.0, 0.0],
            [-3.0, -7.5, -3.0],
        ]
    )

    batch = system.step(points)
    scalar = np.stack([system.step(point) for point in points])

    assert batch.shape == points.shape
    assert system.step(points[0]).shape == (3,)
    np.testing.assert_allclose(batch, scalar, rtol=1e-14, atol=0.0)


def test_registry_builds_ives_model() -> None:
    system = build_system("ives", {"q": 0.8})

    assert isinstance(system, IvesModel)
    assert system.q == 0.8


def test_non_log_coordinate_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="only coordinate_mode='log'"):
        IvesModel(coordinate_mode="linear")
