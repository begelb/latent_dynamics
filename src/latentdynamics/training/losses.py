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
    total: Tensor

    def detach_dict(self) -> dict[str, float]:
        return {
            "loss_ae1": float(self.loss_ae1.detach()),
            "loss_ae2": float(self.loss_ae2.detach()),
            "loss_dyn": float(self.loss_dyn.detach()),
            "loss_total": float(self.total.detach()),
        }


class ReconstructionLoss(nn.Module):
    """Weighted sum ``w0 * L_ae1 + w1 * L_ae2 + w2 * L_dyn``.

    ``loss_ae1`` is the autoencoder reconstruction error at time t,
    ``loss_ae2`` is the predicted reconstruction at time t+tau, and
    ``loss_dyn`` is the latent-space semiconjugacy error. Set a weight to
    zero to drop the corresponding term (e.g. ``[1, 1, 0]`` for Marcio's
    PDE setup with no semiconjugacy term).
    """

    def __init__(self, weights: Sequence[float]) -> None:
        super().__init__()
        if len(weights) != 3:
            raise ValueError("weights must have length 3 (recon_t, recon_tau, dyn)")
        self.criterion = nn.MSELoss(reduction="mean")
        self.register_buffer("weights", torch.tensor(list(weights), dtype=torch.float32))

    def forward(self, fp: ForwardPass) -> LossBreakdown:
        loss_ae1 = self.criterion(fp.x_t, fp.x_t_hat)
        loss_ae2 = self.criterion(fp.x_tau, fp.x_tau_hat)
        loss_dyn = self.criterion(fp.z_tau_pred, fp.z_tau)
        total = self.weights[0] * loss_ae1 + self.weights[1] * loss_ae2 + self.weights[2] * loss_dyn
        return LossBreakdown(loss_ae1=loss_ae1, loss_ae2=loss_ae2, loss_dyn=loss_dyn, total=total)
