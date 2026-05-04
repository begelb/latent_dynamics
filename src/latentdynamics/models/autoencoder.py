"""Encoder, latent map, decoder, and the unified LatentDynamicsAutoencoder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config.schema import ArchConfig
from .activations import hidden_activation, terminal_activation


def _build_mlp(
    in_dim: int,
    hidden_shapes: Sequence[int],
    out_dim: int,
    activation: str,
    terminal: str,
) -> nn.Sequential:
    """Build an MLP with explicit hidden widths + one linear output."""
    layers: list[nn.Module] = []
    last = in_dim
    for hidden_dim in hidden_shapes:
        layers.append(nn.Linear(last, int(hidden_dim)))
        layers.append(hidden_activation(activation))
        last = int(hidden_dim)
    layers.append(nn.Linear(last, out_dim))
    term = terminal_activation(terminal)
    if term is not None:
        layers.append(term)
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, arch: ArchConfig) -> None:
        super().__init__()
        spec = arch.component("encoder")
        self.net = _build_mlp(
            in_dim=arch.high_dims,
            hidden_shapes=spec.hidden_shapes,
            out_dim=arch.low_dims,
            activation=spec.activation,
            terminal=spec.out_activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, arch: ArchConfig) -> None:
        super().__init__()
        spec = arch.component("decoder")
        self.net = _build_mlp(
            in_dim=arch.low_dims,
            hidden_shapes=spec.hidden_shapes,
            out_dim=arch.high_dims,
            activation=spec.activation,
            terminal=spec.out_activation,
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class LatentMap(nn.Module):
    """Discrete map on the latent space."""

    def __init__(self, arch: ArchConfig, terminal: str | None = None) -> None:
        super().__init__()
        spec = arch.component("latent_map")
        self.net = _build_mlp(
            in_dim=arch.low_dims,
            hidden_shapes=spec.hidden_shapes,
            out_dim=arch.low_dims,
            activation=spec.activation,
            terminal=terminal or spec.out_activation,
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


@dataclass(frozen=True)
class ForwardPass:
    """Snapshot of every intermediate tensor produced by one forward call."""

    x_t: Tensor
    x_tau: Tensor
    z_t: Tensor
    z_tau: Tensor
    z_tau_pred: Tensor
    x_t_hat: Tensor
    x_tau_hat: Tensor


class LatentDynamicsAutoencoder(nn.Module):
    """Encoder + latent map + decoder bundle; the unit of save/load."""

    def __init__(self, encoder: Encoder, latent_map: LatentMap, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_map = latent_map
        self.decoder = decoder

    def forward(self, x_t: Tensor, x_tau: Tensor) -> ForwardPass:
        z_t = self.encoder(x_t)
        x_t_hat = self.decoder(z_t)
        z_tau = self.encoder(x_tau)
        z_tau_pred = self.latent_map(z_t)
        x_tau_hat = self.decoder(z_tau_pred)
        return ForwardPass(
            x_t=x_t,
            x_tau=x_tau,
            z_t=z_t,
            z_tau=z_tau,
            z_tau_pred=z_tau_pred,
            x_t_hat=x_t_hat,
            x_tau_hat=x_tau_hat,
        )

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    @torch.no_grad()
    def step_latent(self, z: Tensor) -> Tensor:
        return self.latent_map(z)


def build_autoencoder(arch: ArchConfig) -> LatentDynamicsAutoencoder:
    """Construct a :class:`LatentDynamicsAutoencoder` from an :class:`ArchConfig`."""
    return LatentDynamicsAutoencoder(
        encoder=Encoder(arch),
        latent_map=LatentMap(arch),
        decoder=Decoder(arch),
    )
