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
            2.0 * out.loss_reconstruction
            + 3.0 * out.loss_prediction
            + 5.0 * out.loss_semiconjugacy
            + 7.0 * out.loss_cycle
        )
        torch.testing.assert_close(out.total, manual)

    def test_zero_weight_drops_semiconjugacy(self):
        fp = _fake_pass()
        loss = ReconstructionLoss([1.0, 1.0, 0.0])
        out = loss(fp)
        torch.testing.assert_close(
            out.total, out.loss_reconstruction + out.loss_prediction
        )
        # The term is still computed (just zero-weighted in the total).
        assert out.loss_semiconjugacy.item() > 0.0

    def test_three_weights_omit_cycle(self):
        fp = _fake_pass()
        loss = ReconstructionLoss([1.0, 1.0, 1.0])
        out = loss(fp)
        assert loss.use_cycle is False
        # The cycle term is dropped entirely when no non-zero 4th weight is given.
        assert out.loss_cycle is None
        assert "loss_cycle" not in out.detach_dict()

    def test_zero_fourth_weight_omits_cycle(self):
        fp = _fake_pass()
        loss = ReconstructionLoss([1.0, 1.0, 1.0, 0.0])
        out = loss(fp)
        assert loss.use_cycle is False
        assert out.loss_cycle is None
        assert "loss_cycle" not in out.detach_dict()

    def test_nonzero_fourth_weight_uses_cycle(self):
        fp = _fake_pass()
        loss = ReconstructionLoss([1.0, 1.0, 1.0, 2.0])
        out = loss(fp)
        assert loss.use_cycle is True
        assert out.loss_cycle is not None and out.loss_cycle.item() > 0.0
        assert "loss_cycle" in out.detach_dict()

    def test_weights_length_validation(self):
        for weights in ([1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 2.0]):
            with pytest.raises(ValueError):
                ReconstructionLoss(weights)


class TestLossBreakdownDict:
    def test_detach_dict_keys_without_cycle(self):
        fp = _fake_pass()
        out = ReconstructionLoss([1, 1, 1])(fp)
        d = out.detach_dict()
        assert set(d.keys()) == {
            "loss_reconstruction",
            "loss_prediction",
            "loss_semiconjugacy",
            "loss_total",
        }
        for v in d.values():
            assert isinstance(v, float)

    def test_detach_dict_keys_with_cycle(self):
        fp = _fake_pass()
        out = ReconstructionLoss([1, 1, 1, 1])(fp)
        d = out.detach_dict()
        assert set(d.keys()) == {
            "loss_reconstruction",
            "loss_prediction",
            "loss_semiconjugacy",
            "loss_cycle",
            "loss_total",
        }
        for v in d.values():
            assert isinstance(v, float)
