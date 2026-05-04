"""Loss functions for joint autoencoder + latent-map training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from ..models.autoencoder import ForwardPass


@dataclass(frozen=True)
class LossBreakdown:
    """Per-component MSE values and the scalar driving backprop."""

    loss_ae1: Tensor
    loss_ae2: Tensor
    loss_dyn: Tensor
    total: Tensor

    def detach_dict(self) -> dict[str, float]:
        return {
            "loss_ae1": float(self.loss_ae1.detach()),
            "loss_ae2": float(self.loss_ae2.detach()),
            "loss_dyn": float(self.loss_dyn.detach()),
            "loss_total": float(self.total.detach()),
        }


class _ReconstructionLossBase(nn.Module):
    """Base class that computes the three component MSE values.

    ``loss_ae1`` is the autoencoder reconstruction error at time t,
    ``loss_ae2`` is the predicted reconstruction at time t+tau, and
    ``loss_dyn`` is the latent-space semiconjugacy error.
    """

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")

    def _components(self, fp: ForwardPass) -> tuple[Tensor, Tensor, Tensor]:
        loss_ae1 = self.criterion(fp.x_t, fp.x_t_hat)
        loss_ae2 = self.criterion(fp.x_tau, fp.x_tau_hat)
        loss_dyn = self.criterion(fp.z_tau_pred, fp.z_tau)
        return loss_ae1, loss_ae2, loss_dyn


class WeightedReconstructionLoss(_ReconstructionLossBase):
    """Linear combination ``w0 * L_ae1 + w1 * L_ae2 + w2 * L_dyn``."""

    def __init__(self, weights: Sequence[float]) -> None:
        super().__init__()
        if len(weights) != 3:
            raise ValueError("weights must have length 3 (recon_t, recon_tau, dyn)")
        self.register_buffer("weights", torch.tensor(list(weights), dtype=torch.float32))

    def forward(self, fp: ForwardPass) -> LossBreakdown:
        loss_ae1, loss_ae2, loss_dyn = self._components(fp)
        total = (
            self.weights[0] * loss_ae1
            + self.weights[1] * loss_ae2
            + self.weights[2] * loss_dyn
        )
        return LossBreakdown(loss_ae1=loss_ae1, loss_ae2=loss_ae2, loss_dyn=loss_dyn, total=total)


class AdditiveReconstructionLoss(_ReconstructionLossBase):
    """Marcio's loss: ``L_ae1 + L_ae2`` with no latent-space dynamics term."""

    def forward(self, fp: ForwardPass) -> LossBreakdown:
        loss_ae1, loss_ae2, loss_dyn = self._components(fp)
        total = loss_ae1 + loss_ae2
        return LossBreakdown(loss_ae1=loss_ae1, loss_ae2=loss_ae2, loss_dyn=loss_dyn, total=total)


def build_loss(loss_mode: str, weights: Sequence[float]) -> _ReconstructionLossBase:
    """Construct a loss module from the ``training.loss_mode`` config field."""
    mode = loss_mode.lower()
    if mode == "weighted":
        return WeightedReconstructionLoss(weights)
    if mode == "additive":
        return AdditiveReconstructionLoss()
    raise ValueError(f"unknown loss_mode: {loss_mode!r}")
