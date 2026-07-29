from __future__ import annotations

import numpy as np
import pytest
from scripts import chafee_d1_physics_coordinate_ceiling as ceiling

from latentdynamics.analysis.morse import LatentBounds


def test_analytic_map_has_required_odd_three_fixed_point_topology():
    spec = ceiling.AnalyticMapSpec(stable_root_magnitude=1.25, mu=0.75)
    fixed = spec.fixed_points

    np.testing.assert_allclose(spec.evaluate(fixed), fixed, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        spec.derivative(fixed),
        [0.25, 1.75, 0.25],
        rtol=0,
        atol=1e-15,
    )
    assert spec.theoretical_global_derivative_lower_bound == pytest.approx(0.0625)

    z = np.linspace(-3.0, 3.0, 10_001)
    np.testing.assert_allclose(spec.evaluate(-z), -spec.evaluate(z), rtol=0, atol=1e-15)
    assert np.min(spec.derivative(z)) > 0

    drift = spec.drift(z)
    assert np.all(drift[z < -1.25] > 0)
    assert np.all(drift[(z > -1.25) & (z < 0)] < 0)
    assert np.all(drift[(z > 0) & (z < 1.25)] > 0)
    assert np.all(drift[z > 1.25] < 0)


def test_analytic_map_rejects_strength_without_global_monotonicity_guarantee():
    with pytest.raises(ValueError, match=r"must lie in \(0, 0.8\)"):
        ceiling.AnalyticMapSpec(stable_root_magnitude=1.0, mu=0.8)
    with pytest.raises(ValueError, match="stable_root_magnitude must be positive"):
        ceiling.AnalyticMapSpec(stable_root_magnitude=0.0, mu=0.75)


def test_physics_encoder_is_float32_first_coefficient_only():
    states = np.zeros((3, 64), dtype=np.float64)
    states[:, 0] = [1.0 / 3.0, -1.0 / 7.0, 2.0]
    states[:, 1] = [999.0, -999.0, 123.0]

    encoded = ceiling.physics_encode(states)

    assert encoded.shape == (3, 1)
    assert encoded.dtype == np.float64
    np.testing.assert_array_equal(
        encoded[:, 0],
        np.asarray(states[:, 0], dtype=np.float32).astype(np.float64),
    )


def test_physics_bounds_use_current_and_next_rows_with_ten_percent_padding():
    x = np.zeros((2, 64))
    y = np.zeros((2, 64))
    x[:, 0] = [-2.0, 1.0]
    y[:, 0] = [-1.0, 3.0]

    bounds, payload = ceiling.infer_physics_bounds(x, y)

    np.testing.assert_allclose(bounds.lower, [-2.5])
    np.testing.assert_allclose(bounds.upper, [3.5])
    assert payload["n_encoded_states"] == 4
    assert payload["epsilon_fraction"] == 0.1


def test_dense_validation_confirms_roots_derivatives_oddness_and_drift():
    spec = ceiling.AnalyticMapSpec(stable_root_magnitude=1.25, mu=0.75)
    result = ceiling.validate_dense_topology(
        spec,
        LatentBounds(lower=np.asarray([-2.5]), upper=np.asarray([2.5])),
        grid_points=20_001,
    )

    assert result["all_checks_passed"]
    assert all(result["checks"].values())
    np.testing.assert_allclose(
        result["fixed_points"]["dense_sign_change_estimates"],
        [-1.25, 0.0, 1.25],
        atol=result["fixed_points"]["matching_tolerance"],
        rtol=0,
    )
    assert result["derivatives"]["dense_minimum"] > 0


def test_one_step_mu_fit_is_diagnostic_and_does_not_change_selected_map():
    spec = ceiling.AnalyticMapSpec(stable_root_magnitude=1.0, mu=0.75)
    z = np.linspace(-1.5, 1.5, 101)[:, None]
    fitted_spec = ceiling.AnalyticMapSpec(stable_root_magnitude=1.0, mu=0.2)
    z_next = fitted_spec.evaluate(z)

    diagnostic = ceiling.fit_one_step_mu_diagnostic(spec, z, z_next)

    assert diagnostic["unconstrained_least_squares_mu"] == pytest.approx(0.2)
    assert diagnostic["selected_mu"] == 0.75
    assert diagnostic["least_squares_one_step_mse"] < 1e-28


def test_output_target_must_be_new_and_disjoint_from_canonical(tmp_path):
    safe = tmp_path / "new-ceiling"
    assert ceiling._assert_isolated_output(safe) == safe.resolve()

    safe.mkdir()
    with pytest.raises(FileExistsError, match="fail-if-present"):
        ceiling._assert_isolated_output(safe)

    with pytest.raises(ValueError, match="overlaps protected"):
        ceiling._assert_isolated_output(
            ceiling.reference.DEFAULT_OUTPUT_ROOT / "latent_1d" / "bad"
        )
