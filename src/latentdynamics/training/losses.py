"""Loss function for joint autoencoder + latent-map training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..models.autoencoder import ForwardPass


@dataclass(frozen=True)
class LossBreakdown:
    """Per-component MSE values and the scalar driving backprop.

    The three core terms mirror the paper's $L_1, L_2, L_3$:
    ``loss_reconstruction`` ($L_1$), ``loss_prediction`` ($L_2$), and
    ``loss_semiconjugacy`` ($L_3$). ``loss_cycle`` is an optional fourth
    term that is present only when an explicit, non-zero fourth weight is
    supplied (otherwise it is ``None`` and is omitted from the reported
    breakdown).
    """

    loss_reconstruction: Tensor
    loss_prediction: Tensor
    loss_semiconjugacy: Tensor
    total: Tensor
    loss_cycle: Tensor | None = None

    def detach_dict(self) -> dict[str, float]:
        out = {
            "loss_reconstruction": float(self.loss_reconstruction.detach()),
            "loss_prediction": float(self.loss_prediction.detach()),
            "loss_semiconjugacy": float(self.loss_semiconjugacy.detach()),
        }
        if self.loss_cycle is not None:
            out["loss_cycle"] = float(self.loss_cycle.detach())
        out["loss_total"] = float(self.total.detach())
        return out


class ReconstructionLoss(nn.Module):
    r"""Weighted sum of reconstruction, prediction, and semiconjugacy losses.

    Given weights ``(w1, w2, w3)`` the objective is

    .. math::
        w_1 \lVert D(E(x)) - x\rVert^2
        + w_2 \lVert D(G(E(x))) - y\rVert^2
        + w_3 \lVert G(E(x)) - E(y)\rVert^2,

    where ``loss_reconstruction`` ($L_1$) is the autoencoder reconstruction
    error at time t, ``loss_prediction`` ($L_2$) is the one-step prediction
    error at time t+tau, and ``loss_semiconjugacy`` ($L_3$) is the latent-space
    semiconjugacy residual. Setting a weight to zero drops that term from the
    total (e.g. ``[1, 1, 0]`` for a PDE setup with no semiconjugacy term).

    An optional fourth weight enables a cycle-consistency term
    ``loss_cycle = ||E(D(G(E(x)))) - G(E(x))||^2``. It is computed and reported
    only when a non-zero fourth weight is given; with a length-3 weight vector
    (or a zero fourth weight) the cycle term is omitted entirely.
    """

    def __init__(self, weights: Sequence[float]) -> None:
        super().__init__()
        if len(weights) not in (3, 4):
            raise ValueError(
                "weights must have length 3 (reconstruction, prediction, "
                "semiconjugacy) or 4 (+ cycle)"
            )
        self.criterion = nn.MSELoss(reduction="mean")
        normalized = list(weights)
        # The cycle term is "explicitly used" only when a non-zero fourth
        # weight is supplied; otherwise it is dropped (not just zero-weighted).
        self.use_cycle = len(normalized) == 4 and normalized[3] != 0
        if len(normalized) == 3:
            normalized.append(0.0)
        self.register_buffer("weights", torch.tensor(normalized, dtype=torch.float32))

    def forward(self, fp: ForwardPass) -> LossBreakdown:
        loss_reconstruction = self.criterion(fp.x_t, fp.x_t_hat)
        loss_prediction = self.criterion(fp.x_tau, fp.x_tau_hat)
        loss_semiconjugacy = self.criterion(fp.z_tau_pred, fp.z_tau)
        total = (
            self.weights[0] * loss_reconstruction
            + self.weights[1] * loss_prediction
            + self.weights[2] * loss_semiconjugacy
        )
        loss_cycle: Tensor | None = None
        if self.use_cycle:
            loss_cycle = self.criterion(fp.z_tau_pred_cycle, fp.z_tau_pred)
            total = total + self.weights[3] * loss_cycle
        return LossBreakdown(
            loss_reconstruction=loss_reconstruction,
            loss_prediction=loss_prediction,
            loss_semiconjugacy=loss_semiconjugacy,
            total=total,
            loss_cycle=loss_cycle,
        )
