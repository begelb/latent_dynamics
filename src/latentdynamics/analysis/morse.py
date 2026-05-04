"""CMGDB wrapper: bounds inference and Morse-graph computation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import CMGDB
import numpy as np
import torch
from numpy.typing import NDArray

from ..config.schema import CMGDBConfig
from ..models.autoencoder import LatentDynamicsAutoencoder


@dataclass(frozen=True)
class LatentBounds:
    """Min/max extents of a point cloud in latent space, with an optional buffer."""

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    @property
    def dim(self) -> int:
        return int(self.lower.shape[0])


def infer_latent_bounds(
    encoder: torch.nn.Module,
    all_data_scaled: NDArray[np.float64],
    *,
    epsilon_frac: float = 0.01,
    device: torch.device | None = None,
) -> LatentBounds:
    """Encode ``all_data_scaled`` and compute axis-aligned bounds, expanded by ``epsilon_frac``."""
    device = device or next(encoder.parameters()).device
    encoder.eval()
    with torch.no_grad():
        z = encoder(torch.as_tensor(all_data_scaled, dtype=torch.float32, device=device))
    z = z.cpu().numpy()
    lower = z.min(axis=0)
    upper = z.max(axis=0)
    buffer = epsilon_frac * (upper - lower)
    return LatentBounds(lower=lower - buffer, upper=upper + buffer)


def make_box_map(
    latent_map: torch.nn.Module,
    *,
    device: torch.device | None = None,
    padding: bool = True,
) -> Callable[[Any], Any]:
    """Build the CMGDB BoxMap callable from a torch latent-dynamics module."""
    device = device or next(latent_map.parameters()).device
    latent_map.eval()

    @torch.no_grad()
    def g(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        return latent_map(x_t).cpu().numpy()

    def box_map(rect):
        return CMGDB.BoxMap(g, rect, padding=padding)

    return box_map


def compute_morse_graph(
    autoencoder: LatentDynamicsAutoencoder,
    bounds: LatentBounds,
    cmgdb_cfg: CMGDBConfig,
    *,
    device: torch.device | None = None,
):
    """Run CMGDB on the given latent map and return ``(morse_graph, map_graph)``."""
    box_map = make_box_map(autoencoder.latent_map, device=device, padding=cmgdb_cfg.padding)
    model = CMGDB.Model(
        cmgdb_cfg.subdiv_min,
        cmgdb_cfg.subdiv_max,
        cmgdb_cfg.subdiv_init,
        cmgdb_cfg.subdiv_limit,
        bounds.lower.tolist(),
        bounds.upper.tolist(),
        box_map,
    )
    return CMGDB.ComputeConleyMorseGraph(model)
