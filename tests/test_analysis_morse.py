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

    def test_memory_cap_raises(self):
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        # subdiv_k=10, d=2 -> n_per_axis=32, corners=33^2=1089.
        with pytest.raises(ValueError) as excinfo:
            make_box_map_uniform_precomputed(
                m.latent_map, bounds, subdiv_k=10, max_table_points=500
            )
        msg = str(excinfo.value)
        assert "1089" in msg
        assert "500" in msg
        assert "max_table_points" in msg


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

    def test_adaptive_precomputed_accepts_adaptive_subdivs(self):
        cfg = CMGDBConfig(
            subdiv_init=4,
            subdiv_min=6,
            subdiv_max=8,
            box_map_backend="adaptive_precomputed",
        )
        assert cfg.box_map_backend == "adaptive_precomputed"

    def test_adaptive_precomputed_accepts_uniform_subdivs(self):
        cfg = CMGDBConfig(
            subdiv_init=8,
            subdiv_min=8,
            subdiv_max=8,
            box_map_backend="adaptive_precomputed",
        )
        assert cfg.box_map_backend == "adaptive_precomputed"

    def test_max_table_points_default_and_override(self):
        cfg = CMGDBConfig()
        assert cfg.max_table_points == 10_000_000
        cfg2 = CMGDBConfig(max_table_points=1_000)
        assert cfg2.max_table_points == 1_000


class TestBoxMapAdaptivePrecomputed:
    def test_returns_callable(self):
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        from latentdynamics.analysis.morse import make_box_map_adaptive_precomputed

        G = make_box_map_adaptive_precomputed(m.latent_map, bounds, subdiv_max=4)
        assert callable(G)

    def test_bit_equivalent_to_uniform_precomputed(self):
        """In the uniform case (k % d == 0), adaptive and uniform precompute
        must agree exactly on every cell of the level-k partition.

        Same lattice, same forward pass, same gather -- the only difference is
        the lookup formula, which collapses to the same 2^d corners.
        """
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        k = 4  # 4 boxes per axis in d=2, 16 cells total
        n_per_axis = 2 ** (k // 2)
        L, U = bounds.lower, bounds.upper
        side = (U - L) / n_per_axis

        from latentdynamics.analysis.morse import make_box_map_adaptive_precomputed

        G_uni = make_box_map_uniform_precomputed(
            m.latent_map, bounds, subdiv_k=k, padding=False
        )
        G_adp = make_box_map_adaptive_precomputed(
            m.latent_map, bounds, subdiv_max=k, padding=False
        )
        for i in range(n_per_axis):
            for j in range(n_per_axis):
                rect = [
                    L[0] + i * side[0],
                    L[1] + j * side[1],
                    L[0] + (i + 1) * side[0],
                    L[1] + (j + 1) * side[1],
                ]
                out_uni = np.array(G_uni(rect))
                out_adp = np.array(G_adp(rect))
                np.testing.assert_array_equal(out_adp, out_uni)

    def test_agrees_with_numpy_backend_at_multiple_depths(self):
        """For sample rects at depths in [subdiv_init, subdiv_max], the adaptive
        precompute backend must agree with the numpy backend up to float32 cast.

        Rect construction follows CMGDB's bisection-cycle convention:
        at depth k, axis j has been bisected n_j(k) = (k + d - 1 - j) // d
        times, so cells are aligned to the lattice (2^n_j(k) + 1) along axis j.
        """
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        subdiv_max = 14
        d = bounds.dim

        from latentdynamics.analysis.morse import make_box_map_adaptive_precomputed

        G_adp = make_box_map_adaptive_precomputed(
            m.latent_map, bounds, subdiv_max=subdiv_max, padding=False
        )
        G_np = make_box_map_numpy(m.latent_map, padding=False)

        rng = np.random.default_rng(0)
        L, U = bounds.lower, bounds.upper
        for depth in (8, 10, 12, 14):
            n_per_axis = np.array(
                [2 ** ((depth + d - 1 - j) // d) for j in range(d)], dtype=np.int64
            )
            side = (U - L) / n_per_axis
            for _ in range(50):
                i = np.array([rng.integers(0, n_per_axis[j]) for j in range(d)])
                rect = list(L + i * side) + list(L + (i + 1) * side)
                out_adp = np.array(G_adp(rect))
                out_np = np.array(G_np(rect))
                np.testing.assert_allclose(
                    out_adp, out_np, atol=1e-5, rtol=1e-5,
                    err_msg=f"depth={depth}, i={i.tolist()}",
                )

    def test_handles_non_divisible_subdiv_max(self):
        """subdiv_max=5 with d=2 -> M = ceil(5/2) = 3, n_per_axis = 8.

        A cell at depth 5 has axis-0 bisected 3 times and axis-1 bisected 2
        times (per CMGDB's split cycle). Both align to the M=3 lattice; the
        backend should construct and answer without error.
        """
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))

        from latentdynamics.analysis.morse import make_box_map_adaptive_precomputed

        G = make_box_map_adaptive_precomputed(
            m.latent_map, bounds, subdiv_max=5, padding=False
        )
        # Cell at depth 5, axis-0 index 3 of 8, axis-1 index 1 of 4.
        side = np.array([0.25, 0.5])
        L = bounds.lower
        i0, i1 = 3, 1
        rect = [
            L[0] + i0 * side[0],
            L[1] + i1 * side[1],
            L[0] + (i0 + 1) * side[0],
            L[1] + (i1 + 1) * side[1],
        ]
        out = np.array(G(rect))
        assert out.shape == (4,)
        assert out[0] <= out[2] and out[1] <= out[3]

    def test_memory_cap_raises_with_budget_and_actual(self):
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))

        from latentdynamics.analysis.morse import make_box_map_adaptive_precomputed

        # subdiv_max=10, d=2 -> M=5, n_per_axis=32, corners=33^2=1089.
        # Cap below 1089 must raise; message must include both numbers.
        with pytest.raises(ValueError) as excinfo:
            make_box_map_adaptive_precomputed(
                m.latent_map, bounds, subdiv_max=10, max_table_points=500
            )
        msg = str(excinfo.value)
        assert "1089" in msg
        assert "500" in msg
        assert "max_table_points" in msg

    def test_dispatched_through_compute_morse_graph_config(self):
        """End-to-end: a CMGDBConfig with backend='adaptive_precomputed' must
        be accepted by _build_box_map and produce a working callable."""
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        cfg = CMGDBConfig(
            subdiv_init=4,
            subdiv_min=6,
            subdiv_max=8,
            box_map_backend="adaptive_precomputed",
            padding=False,
        )
        from latentdynamics.analysis.morse import _build_box_map

        box_map = _build_box_map(m.latent_map, bounds, cfg, device=torch.device("cpu"))
        out = box_map([-1.0, -1.0, -0.5, -0.5])
        assert len(out) == 4
        assert out[0] <= out[2] and out[1] <= out[3]
