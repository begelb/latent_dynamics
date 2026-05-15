"""Tests for the CMGDB wrapper helpers (bounds inference and BoxMap backends)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from latentdynamics.analysis.morse import (
    LatentBounds,
    _extract_numpy_mlp,
    _numpy_mlp_forward,
    infer_latent_bounds,
    make_box_map,
    make_box_map_numpy,
    make_box_map_uniform_precomputed,
)
from latentdynamics.config.schema import ArchConfig, CMGDBConfig
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


class TestBoxMapPytorch:
    def test_returns_callable(self):
        m = build_autoencoder(_arch())
        G = make_box_map(m.latent_map, device=torch.device("cpu"))
        assert callable(G)


class TestBoxMapNumpy:
    def test_returns_callable(self):
        m = build_autoencoder(_arch())
        G = make_box_map_numpy(m.latent_map)
        assert callable(G)

    def test_matches_pytorch_forward(self):
        """NumPy forward should match torch forward on the same inputs."""
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        latent_map = m.latent_map
        latent_map.eval()
        layers = _extract_numpy_mlp(latent_map)
        X = np.random.RandomState(0).randn(8, 2).astype(np.float64)
        Y_np = _numpy_mlp_forward(layers, X)
        with torch.no_grad():
            Y_t = latent_map(torch.as_tensor(X, dtype=torch.float32)).numpy().astype(np.float64)
        # float32 latent_map vs float64 numpy: tolerance to dtype rounding.
        np.testing.assert_allclose(Y_np, Y_t, atol=1e-5, rtol=1e-5)

    def test_output_rect_format(self):
        """box_map returns a flat [l_0..l_{d-1}, u_0..u_{d-1}] list with l <= u."""
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        G = make_box_map_numpy(m.latent_map, padding=False)
        rect = [0.0, 0.0, 0.5, 0.5]
        out = G(rect)
        assert len(out) == 4
        assert out[0] <= out[2] and out[1] <= out[3]

    def test_padding_widens_rect(self):
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        G_pad = make_box_map_numpy(m.latent_map, padding=True)
        G_nopad = make_box_map_numpy(m.latent_map, padding=False)
        rect = [0.0, 0.0, 0.5, 0.5]
        out_pad = np.array(G_pad(rect))
        out_nopad = np.array(G_nopad(rect))
        box_size = np.array([0.5, 0.5])
        # Lower bounds should differ by -box_size, upper by +box_size.
        np.testing.assert_allclose(out_pad[:2], out_nopad[:2] - box_size, atol=1e-10)
        np.testing.assert_allclose(out_pad[2:], out_nopad[2:] + box_size, atol=1e-10)


class TestBoxMapUniformPrecomputed:
    def test_returns_callable(self):
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        G = make_box_map_uniform_precomputed(m.latent_map, bounds, subdiv_k=2)
        assert callable(G)

    def test_rejects_non_divisible_k(self):
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="divisible"):
            make_box_map_uniform_precomputed(m.latent_map, bounds, subdiv_k=3)

    def test_matches_numpy_backend_on_grid_aligned_rect(self):
        """On a rect that exactly matches a level-k grid cell, the precomputed
        backend and the numpy backend should agree up to float32 rounding."""
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        # k=4 in d=2 -> 4 boxes per axis, side length 0.5.
        G_pre = make_box_map_uniform_precomputed(m.latent_map, bounds, subdiv_k=4, padding=False)
        G_np = make_box_map_numpy(m.latent_map, padding=False)
        # Cell (1, 2): x in [-0.5, 0.0], y in [0.0, 0.5].
        rect = [-0.5, 0.0, 0.0, 0.5]
        out_pre = np.array(G_pre(rect))
        out_np = np.array(G_np(rect))
        # Precomputed runs the network in float32 then casts; numpy backend runs
        # everything in float64. They agree to float32 precision.
        np.testing.assert_allclose(out_pre, out_np, atol=1e-5, rtol=1e-5)

    def test_padding_widens_rect(self):
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        G_pad = make_box_map_uniform_precomputed(m.latent_map, bounds, subdiv_k=4, padding=True)
        G_nopad = make_box_map_uniform_precomputed(
            m.latent_map, bounds, subdiv_k=4, padding=False
        )
        rect = [-0.5, 0.0, 0.0, 0.5]
        out_pad = np.array(G_pad(rect))
        out_nopad = np.array(G_nopad(rect))
        box_size = np.array([0.5, 0.5])
        np.testing.assert_allclose(out_pad[:2], out_nopad[:2] - box_size, atol=1e-10)
        np.testing.assert_allclose(out_pad[2:], out_nopad[2:] + box_size, atol=1e-10)


class TestCMGDBConfigBackendValidation:
    def test_uniform_precomputed_requires_uniform_mode(self):
        with pytest.raises(ValueError, match="uniform mode"):
            CMGDBConfig(
                subdiv_init=4,
                subdiv_min=6,
                subdiv_max=8,
                box_map_backend="uniform_precomputed",
            )

    def test_uniform_precomputed_accepted_in_uniform_mode(self):
        cfg = CMGDBConfig(
            subdiv_init=8,
            subdiv_min=8,
            subdiv_max=8,
            box_map_backend="uniform_precomputed",
        )
        assert cfg.box_map_backend == "uniform_precomputed"

    def test_pytorch_backend_default(self):
        assert CMGDBConfig().box_map_backend == "pytorch"

    def test_numpy_backend_accepts_adaptive(self):
        cfg = CMGDBConfig(
            subdiv_init=4, subdiv_min=6, subdiv_max=8, box_map_backend="numpy"
        )
        assert cfg.box_map_backend == "numpy"
