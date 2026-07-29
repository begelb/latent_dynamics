from __future__ import annotations

import numpy as np
import pytest
from scripts import chafee_d1_polynomial_coordinate_map as polynomial

from latentdynamics.analysis.morse import LatentBounds


def test_polynomial_map_keeps_three_local_fixed_points_but_folds_globally():
    spec = polynomial.PolynomialMapSpec(
        stable_root_magnitude=1.25,
        mu=0.75,
    )
    fixed = spec.fixed_points

    np.testing.assert_allclose(spec.evaluate(fixed), fixed, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        spec.derivative(fixed),
        [-0.5, 1.75, -0.5],
        rtol=0,
        atol=1e-15,
    )
    assert spec.normalized_fold_magnitude == pytest.approx(
        np.sqrt(1.75 / 2.25)
    )
    assert spec.normalized_sign_reversal_magnitude == pytest.approx(
        np.sqrt(1.75 / 0.75)
    )

    z = np.linspace(-3.0, 3.0, 10_001)
    np.testing.assert_allclose(
        spec.evaluate(-z),
        -spec.evaluate(z),
        rtol=0,
        atol=1e-15,
    )
    assert np.min(spec.derivative(z)) < 0.0
    assert spec.evaluate(2.5) < 0.0
    assert spec.evaluate(-2.5) > 0.0


def test_dense_diagnostics_record_global_failures_without_raising():
    spec = polynomial.PolynomialMapSpec(
        stable_root_magnitude=1.25,
        mu=0.75,
    )
    result = polynomial.diagnose_dense_topology(
        spec,
        LatentBounds(
            lower=np.asarray([-2.5]),
            upper=np.asarray([2.5]),
        ),
        grid_points=20_001,
    )

    assert not result["all_checks_passed"]
    assert result["cmgdb_was_not_precluded_by_failed_dense_checks"]
    assert result["checks"]["dense_grid_has_exactly_three_fixed_points"]
    assert result["checks"]["outer_fixed_points_strictly_stable"]
    assert result["checks"]["origin_strictly_unstable"]
    assert not result["checks"]["strictly_positive_derivative_on_dense_domain"]
    assert not result["checks"]["map_preserves_nonzero_sign_on_dense_domain"]
    assert "strictly_positive_derivative_on_dense_domain" in result["failed_checks"]
    assert "map_preserves_nonzero_sign_on_dense_domain" in result["failed_checks"]


def test_polynomial_one_step_fit_is_diagnostic_only():
    selected = polynomial.PolynomialMapSpec(
        stable_root_magnitude=1.0,
        mu=0.75,
    )
    fitted = polynomial.PolynomialMapSpec(
        stable_root_magnitude=1.0,
        mu=0.2,
    )
    z = np.linspace(-1.5, 1.5, 101)[:, None]
    diagnostic = polynomial.fit_one_step_mu_diagnostic(
        selected,
        z,
        fitted.evaluate(z),
    )

    assert diagnostic["unconstrained_least_squares_mu"] == pytest.approx(0.2)
    assert diagnostic["least_squares_one_step_mse"] < 1e-28
    assert diagnostic["selected_mu"] == 0.75


def test_polynomial_spec_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="stable_root_magnitude must be positive"):
        polynomial.PolynomialMapSpec(stable_root_magnitude=0.0, mu=0.75)
    with pytest.raises(ValueError, match="mu must be positive"):
        polynomial.PolynomialMapSpec(stable_root_magnitude=1.0, mu=0.0)
