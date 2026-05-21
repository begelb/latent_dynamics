"""Tests for the weighted reconstruction loss."""

from __future__ import annotations

import pytest
import torch

from latentdynamics.models.autoencoder import ForwardPass
from latentdynamics.training.losses import ReconstructionLoss


def _fake_pass() -> ForwardPass:
    torch.manual_seed(0)
    return ForwardPass(
        x_t=torch.randn(4, 3),
        x_tau=torch.randn(4, 3),
        z_t=torch.randn(4, 2),
        z_tau=torch.randn(4, 2),
        z_tau_pred=torch.randn(4, 2),
        z_tau_pred_cycle=torch.randn(4, 2),
        x_t_hat=torch.randn(4, 3),
        x_tau_hat=torch.randn(4, 3),
    )


class TestReconstructionLoss:
    def test_total_equals_weighted_sum(self):
        fp = _fake_pass()
        loss = ReconstructionLoss([2.0, 3.0, 5.0, 7.0])
        out = loss(fp)
        manual = (
            2.0 * out.loss_ae1
            + 3.0 * out.loss_ae2
            + 5.0 * out.loss_dyn
            + 7.0 * out.loss_cycle_pred
        )
        torch.testing.assert_close(out.total, manual)

    def test_zero_weight_drops_dyn(self):
        fp = _fake_pass()
        loss = ReconstructionLoss([1.0, 1.0, 0.0])
        out = loss(fp)
        torch.testing.assert_close(out.total, out.loss_ae1 + out.loss_ae2)
        assert out.loss_dyn.item() > 0.0

    def test_three_weights_default_cycle_weight_to_zero(self):
        fp = _fake_pass()
        loss = ReconstructionLoss([1.0, 1.0, 0.0])
        out = loss(fp)
        assert loss.weights.tolist() == [1.0, 1.0, 0.0, 0.0]
        assert out.loss_cycle_pred.item() > 0.0
        torch.testing.assert_close(out.total, out.loss_ae1 + out.loss_ae2)

    def test_weights_length_validation(self):
        for weights in ([1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 2.0]):
            with pytest.raises(ValueError):
                ReconstructionLoss(weights)


class TestLossBreakdownDict:
    def test_detach_dict_keys_and_floats(self):
        fp = _fake_pass()
        out = ReconstructionLoss([1, 1, 1, 1])(fp)
        d = out.detach_dict()
        assert set(d.keys()) == {
            "loss_ae1",
            "loss_ae2",
            "loss_dyn",
            "loss_cycle_pred",
            "loss_total",
        }
        for v in d.values():
            assert isinstance(v, float)
