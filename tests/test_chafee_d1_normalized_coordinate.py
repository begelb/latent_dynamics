from __future__ import annotations

import numpy as np
import pytest
from scripts import chafee_d1_normalized_coordinate as experiment


def test_normalized_encoder_divides_float32_first_coefficient_by_fixed_a():
    states = np.zeros((3, 64), dtype=np.float64)
    states[:, 0] = [1.0 / 3.0, -1.0 / 7.0, 2.0]
    states[:, 1] = [99.0, 100.0, 101.0]

    encoded = experiment.normalized_encode(states)

    expected = (
        np.asarray(states[:, 0], dtype=np.float32).astype(np.float64)
        / experiment.NORMALIZATION_A
    )
    np.testing.assert_array_equal(encoded[:, 0], expected)
    assert experiment.NORMALIZATION_A == 1.2365946


def test_normalized_fit_recovers_exact_synthetic_mu():
    true_mu = 0.173
    z = np.linspace(-1.6, 1.6, 501)[:, None]
    z_next = z + true_mu * z * (1.0 - z * z)

    fitted_mu, report = experiment.fit_mu_and_residual_report(z, z_next)

    assert fitted_mu == pytest.approx(true_mu, abs=1e-15)
    assert report["fitted"]["mse"] < 1e-30
    assert not report["test_labels_used_in_fit"]
    assert not report["stable_roots_used_in_fit"]
    assert report["identity_baseline"]["description"].startswith("G(z)=z")


def test_residual_metrics_include_requested_norms_and_quantiles():
    z = np.asarray([[-1.2], [-0.2], [0.4], [1.3]])
    z_next = z + 0.1 * z * (1.0 - z * z)

    metrics = experiment.residual_metrics(z, z_next, mu=0.2)

    assert metrics["mse"] > 0.0
    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))
    assert metrics["mae"] > 0.0
    assert metrics["max_absolute_error"] >= metrics["mae"]
    assert set(metrics["absolute_error_quantiles"]) == {
        "q50",
        "q90",
        "q95",
        "q99",
        "q999",
    }


def test_normalized_bounds_scale_both_current_and_next_coordinates():
    x = np.zeros((2, 64))
    y = np.zeros((2, 64))
    x[:, 0] = [-2.0, 1.0]
    y[:, 0] = [-1.0, 3.0]

    bounds, payload = experiment.infer_normalized_bounds(x, y, a=2.0)

    np.testing.assert_allclose(bounds.lower, [-1.25])
    np.testing.assert_allclose(bounds.upper, [1.75])
    assert payload["n_encoded_states"] == 4
    assert payload["epsilon_fraction"] == 0.1


def test_scan_inventory_labels_only_fitted_candidate_as_not_test_informed():
    candidates = experiment._scan_candidates(0.159)

    assert candidates[0]["candidate_id"] == "least_squares"
    assert not candidates[0]["test_informed"]
    assert all(candidate["test_informed"] for candidate in candidates[1:])
    assert any(candidate["mu"] == 0.75 for candidate in candidates)


def test_cell_boxes_merge_into_connected_closed_intervals():
    boxes = np.asarray(
        [
            [0.0, 0.25],
            [-1.0, -0.5],
            [-0.5, 0.0],
            [0.75, 1.0],
        ]
    )

    intervals = experiment.merge_cell_intervals(boxes)

    assert intervals == [(-1.0, 0.25), (0.75, 1.0)]


def test_exact_cubic_tau_uses_endpoints_and_derivative_critical_points():
    spec = experiment.cubic.UnitScaleCubicSpec(mu=0.2)

    result = experiment.exact_cubic_attracting_block_tau(
        spec,
        [(-1.2, -0.8), (0.8, 1.2)],
    )

    assert result["positive_attracting_margin"]
    assert result["tau"] == pytest.approx(0.0576, abs=1e-15)
    assert abs(result["witness"]["z"]) == pytest.approx(0.8)


def test_conditioned_residual_reports_witness_and_tau_logic():
    spec = experiment.cubic.UnitScaleCubicSpec(mu=0.2)
    z = np.asarray([[-1.0], [-0.9], [0.0], [0.9], [1.0]])
    z_next = np.asarray([[-1.0], [-0.85], [0.0], [0.85], [1.0]])

    result = experiment.conditional_block_residuals(
        spec,
        z,
        z_next,
        [(-1.1, -0.8), (0.8, 1.1)],
        tau=0.01,
    )

    assert result["accepted_pairs"] == 4
    assert result["max_absolute_residual"] > 0.01
    assert result["comparison_to_tau"] == "sample_counterexample_exceeds_tau"
    assert result["witness"]["training_pair_index"] in {1, 3}
