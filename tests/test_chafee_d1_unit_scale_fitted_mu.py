from __future__ import annotations

import numpy as np
import pytest
from scripts import chafee_d1_unit_scale_fitted_mu as experiment

from latentdynamics.analysis.morse import LatentBounds


def test_unit_scale_map_has_fixed_roots_at_minus_one_zero_plus_one():
    spec = experiment.UnitScaleCubicSpec(mu=0.2)

    np.testing.assert_array_equal(spec.fixed_points, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        spec.evaluate(spec.fixed_points),
        spec.fixed_points,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        spec.derivative(spec.fixed_points),
        [0.6, 1.2, 0.6],
        rtol=0.0,
        atol=1e-15,
    )


def test_fit_mu_recovers_exact_no_intercept_training_coefficient():
    true_spec = experiment.UnitScaleCubicSpec(mu=0.137)
    z = np.linspace(-1.8, 1.8, 301)[:, None]
    z_next = true_spec.evaluate(z)

    fitted, diagnostic = experiment.fit_mu_least_squares(z, z_next)

    assert fitted.mu == pytest.approx(true_spec.mu, abs=1e-15)
    assert diagnostic["n_training_pairs"] == 301
    assert diagnostic["one_step_mse"] < 1e-30
    assert not diagnostic["test_labels_used_in_fit"]
    assert not diagnostic["stable_roots_used_in_fit"]
    assert "no intercept" in diagnostic["objective"]


def test_fit_mu_rejects_invalid_or_degenerate_pairs():
    with pytest.raises(ValueError, match="nonempty and equally shaped"):
        experiment.fit_mu_least_squares(np.zeros((2, 1)), np.zeros((3, 1)))
    with pytest.raises(ValueError, match="feature has zero norm"):
        experiment.fit_mu_least_squares(np.zeros((3, 1)), np.zeros((3, 1)))


def test_fixed_point_mismatch_keeps_raw_encoded_roots_unmodified():
    spec = experiment.UnitScaleCubicSpec(mu=0.05)
    roots = np.asarray([[-1.2366], [1.2366]])

    result = experiment.fixed_point_mismatch_diagnostic(spec, roots)

    np.testing.assert_allclose(
        result["encoded_root_minus_corresponding_model_root"],
        [-0.2366, 0.2366],
    )
    np.testing.assert_allclose(result["absolute_mismatch"], [0.2366, 0.2366])
    assert not result["encoded_pde_roots_are_map_fixed_points"]
    assert result["drift_at_encoded_pde_roots"][0] > 0.0
    assert result["drift_at_encoded_pde_roots"][1] < 0.0


def test_small_fitted_mu_is_well_behaved_on_archived_sized_domain():
    spec = experiment.UnitScaleCubicSpec(mu=0.05)
    result = experiment.diagnose_dense_topology(
        spec,
        LatentBounds(
            lower=np.asarray([-2.4]),
            upper=np.asarray([2.4]),
        ),
        grid_points=20_001,
    )

    assert result["all_archived_domain_checks_passed"]
    assert all(result["checks"].values())
    assert result["derivatives"]["folds_are_outside_archived_domain"]
    assert result["sign_reversal"]["sign_reversal_is_outside_archived_domain"]
    np.testing.assert_allclose(
        result["fixed_points"]["dense_sign_change_estimates"],
        [-1.0, 0.0, 1.0],
        rtol=0.0,
        atol=result["fixed_points"]["matching_tolerance"],
    )


def test_unit_scale_spec_rejects_nonpositive_mu():
    with pytest.raises(ValueError, match="mu must be positive"):
        experiment.UnitScaleCubicSpec(mu=0.0)
