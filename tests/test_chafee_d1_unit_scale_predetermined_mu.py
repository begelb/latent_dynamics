from __future__ import annotations

import numpy as np
import pytest
from scripts import chafee_d1_unit_scale_predetermined_mu as experiment


def test_predetermined_mu_is_not_replaced_by_least_squares_value():
    true_mu = 0.1
    z = np.linspace(-1.8, 1.8, 301)[:, None]
    feature = z * (1.0 - z * z)
    z_next = z + true_mu * feature

    selected, diagnostic = experiment.predetermined_mu_diagnostic(z, z_next)

    assert selected.mu == experiment.PREDETERMINED_MU
    assert diagnostic["selected_mu"] == experiment.PREDETERMINED_MU
    assert diagnostic["least_squares_mu_diagnostic"] == pytest.approx(true_mu)
    assert diagnostic["selection_status"] == "post_hoc_test_informed"
    assert diagnostic["test_labels_informed_selected_mu"]
    assert not diagnostic["test_labels_used_in_residual_diagnostic"]


def test_post_hoc_comparability_is_explicit():
    comparability = experiment._post_hoc_comparability()

    assert not comparability["paper_eligible"]
    assert comparability["learned_parameter_count"] == 0
    assert comparability["test_labels_informed_mu_selection"]
    assert any(
        "least-squares residual minimizer" in item
        for item in comparability["invalid_interpretations"]
    )
