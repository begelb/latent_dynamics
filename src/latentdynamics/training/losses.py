"""Loss function for joint autoencoder + latent-map training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..models.autoencoder import ForwardPass


@dataclass(frozen=True)
class LossBreakdown:
    """Per-component MSE values and the scalar driving backprop."""

    loss_ae1: Tensor
    loss_ae2: Tensor
    loss_dyn: Tensor
    loss_cycle_pred: Tensor
    total: Tensor

    def detach_dict(self) -> dict[str, float]:
        return {
            "loss_ae1": float(self.loss_ae1.detach()),
            "loss_ae2": float(self.loss_ae2.detach()),
            "loss_dyn": float(self.loss_dyn.detach()),
            "loss_cycle_pred": float(self.loss_cycle_pred.detach()),
            "loss_total": float(self.total.detach()),
        }


class ReconstructionLoss(nn.Module):
    """Weighted sum of reconstruction, prediction, semiconjugacy, and cycle losses.

    ``loss_ae1`` is the autoencoder reconstruction error at time t,
    ``loss_ae2`` is the predicted reconstruction at time t+tau, and
    ``loss_dyn`` is the latent-space semiconjugacy error. Set a weight to
    zero to drop the corresponding term (e.g. ``[1, 1, 0]`` for a
    PDE setup with no semiconjugacy term). A fourth weight enables
    ``loss_cycle_pred = ||E(D(G(E(x_t)))) - G(E(x_t))||^2``.
    """

    def __init__(self, weights: Sequence[float]) -> None:
        super().__init__()
        if len(weights) not in (3, 4):
            raise ValueError("weights must have length 3 or 4 (recon_t, recon_tau, dyn, cycle_pred)")
        self.criterion = nn.MSELoss(reduction="mean")
        normalized = list(weights)
        if len(normalized) == 3:
            normalized.append(0.0)
        self.register_buffer("weights", torch.tensor(normalized, dtype=torch.float32))

    def forward(self, fp: ForwardPass) -> LossBreakdown:
        loss_ae1 = self.criterion(fp.x_t, fp.x_t_hat)
        loss_ae2 = self.criterion(fp.x_tau, fp.x_tau_hat)
        loss_dyn = self.criterion(fp.z_tau_pred, fp.z_tau)
        loss_cycle_pred = self.criterion(fp.z_tau_pred_cycle, fp.z_tau_pred)
        total = (
            self.weights[0] * loss_ae1
            + self.weights[1] * loss_ae2
            + self.weights[2] * loss_dyn
            + self.weights[3] * loss_cycle_pred
        )
        return LossBreakdown(
            loss_ae1=loss_ae1,
            loss_ae2=loss_ae2,
            loss_dyn=loss_dyn,
            loss_cycle_pred=loss_cycle_pred,
            total=total,
        )
