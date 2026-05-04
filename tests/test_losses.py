"""Tests for loss-mode dispatch and component decomposition."""

from __future__ import annotations

import pytest
import torch

from latentdynamics.models.autoencoder import ForwardPass
from latentdynamics.training.losses import (
    AdditiveReconstructionLoss,
    WeightedReconstructionLoss,
    build_loss,
)


def _fake_pass() -> ForwardPass:
    torch.manual_seed(0)
    return ForwardPass(
        x_t=torch.randn(4, 3),
        x_tau=torch.randn(4, 3),
        z_t=torch.randn(4, 2),
        z_tau=torch.randn(4, 2),
        z_tau_pred=torch.randn(4, 2),
        x_t_hat=torch.randn(4, 3),
        x_tau_hat=torch.randn(4, 3),
    )


class TestWeightedLoss:
    def test_total_equals_weighted_sum(self):
        fp = _fake_pass()
        loss = WeightedReconstructionLoss([2.0, 3.0, 5.0])
        out = loss(fp)
        manual = 2.0 * out.loss_ae1 + 3.0 * out.loss_ae2 + 5.0 * out.loss_dyn
        torch.testing.assert_close(out.total, manual)

    def test_zero_weight_does_not_blow_up(self):
        fp = _fake_pass()
        loss = WeightedReconstructionLoss([1.0, 1.0, 0.0])
        out = loss(fp)
        torch.testing.assert_close(out.total, out.loss_ae1 + out.loss_ae2)
        # And the dyn component is still reported, not divided by anything.
        assert out.loss_dyn.item() > 0.0

    def test_weights_length_validation(self):
        with pytest.raises(ValueError):
            WeightedReconstructionLoss([1.0, 1.0])


class TestAdditiveLoss:
    def test_total_equals_recon_sum(self):
        fp = _fake_pass()
        out = AdditiveReconstructionLoss()(fp)
        torch.testing.assert_close(out.total, out.loss_ae1 + out.loss_ae2)

    def test_dyn_component_is_reported_but_not_in_total(self):
        fp = _fake_pass()
        out = AdditiveReconstructionLoss()(fp)
        assert out.loss_dyn.item() > 0.0  # reported
        assert not torch.allclose(out.total, out.loss_ae1 + out.loss_ae2 + out.loss_dyn)


class TestBuildLoss:
    def test_weighted(self):
        loss = build_loss("weighted", [1.0, 1.0, 1.0])
        assert isinstance(loss, WeightedReconstructionLoss)

    def test_additive(self):
        loss = build_loss("additive", [1.0, 1.0, 0.0])
        assert isinstance(loss, AdditiveReconstructionLoss)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build_loss("hybrid", [1.0, 1.0, 1.0])


class TestLossBreakdownDict:
    def test_detach_dict_keys_and_floats(self):
        fp = _fake_pass()
        out = WeightedReconstructionLoss([1, 1, 1])(fp)
        d = out.detach_dict()
        assert set(d.keys()) == {"loss_ae1", "loss_ae2", "loss_dyn", "loss_total"}
        for v in d.values():
            assert isinstance(v, float)
