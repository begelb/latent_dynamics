"""Tests for the CMGDB wrapper helpers (bounds inference and BoxMap)."""

from __future__ import annotations

import numpy as np
import torch

from latentdynamics.analysis.morse import infer_latent_bounds, make_box_map
from latentdynamics.config.schema import ArchConfig
from latentdynamics.models import build_autoencoder


def _arch() -> ArchConfig:
    return ArchConfig(num_layers=1, hidden_shape=4, high_dims=4, low_dims=2)


class TestInferLatentBounds:
    def test_bounds_contain_data_after_buffer(self):
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        data = np.random.RandomState(0).rand(50, 4).astype(np.float32)
        b = infer_latent_bounds(m.encoder, data, epsilon_frac=0.05, device=torch.device("cpu"))
        with torch.no_grad():
            z = m.encoder(torch.as_tensor(data)).numpy()
        assert (b.lower <= z.min(axis=0)).all()
        assert (b.upper >= z.max(axis=0)).all()

    def test_zero_epsilon_gives_tight_bounds(self):
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        data = np.random.RandomState(1).rand(20, 4).astype(np.float32)
        b = infer_latent_bounds(m.encoder, data, epsilon_frac=0.0, device=torch.device("cpu"))
        with torch.no_grad():
            z = m.encoder(torch.as_tensor(data)).numpy()
        np.testing.assert_allclose(b.lower, z.min(axis=0), atol=1e-6)
        np.testing.assert_allclose(b.upper, z.max(axis=0), atol=1e-6)

    def test_bounds_dim_property(self):
        m = build_autoencoder(_arch())
        data = np.random.rand(10, 4).astype(np.float32)
        b = infer_latent_bounds(m.encoder, data, device=torch.device("cpu"))
        assert b.dim == 2


class TestBoxMap:
    def test_returns_callable(self):
        m = build_autoencoder(_arch())
        G = make_box_map(m.latent_map, device=torch.device("cpu"))
        assert callable(G)
