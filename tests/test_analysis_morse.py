"""Tests for the CMGDB wrapper helpers (bounds inference and BoxMap backends)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from latentdynamics.analysis.morse import (
    LatentBounds,
    _extract_numpy_mlp,
    _numpy_mlp_forward,
    _resolve_box_map_backend,
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

    def test_auto_backend_default(self):
        assert CMGDBConfig().box_map_backend == "auto"

    def test_auto_backend_resolves_uniform_precomputed_for_uniform_divisible_grid(self):
        cfg = CMGDBConfig(
            subdiv_init=8,
            subdiv_min=8,
            subdiv_max=8,
            box_map_backend="auto",
        )
        assert _resolve_box_map_backend(cfg, dim=2) == "uniform_precomputed"

    def test_auto_backend_resolves_adaptive_precomputed_for_adaptive_grid(self):
        cfg = CMGDBConfig(
            subdiv_init=4,
            subdiv_min=6,
            subdiv_max=8,
            box_map_backend="auto",
        )
        assert _resolve_box_map_backend(cfg, dim=2) == "adaptive_precomputed"

    def test_auto_backend_resolves_adaptive_precomputed_when_uniform_grid_not_divisible(self):
        cfg = CMGDBConfig(
            subdiv_init=5,
            subdiv_min=5,
            subdiv_max=5,
            box_map_backend="auto",
        )
        assert _resolve_box_map_backend(cfg, dim=2) == "adaptive_precomputed"

    @pytest.mark.parametrize("backend", ["pytorch", "numpy"])
    def test_reference_backends_are_not_public_config_options(self, backend):
        """New CMGDB runs must go through precomputed backends.

        The direct Python/NumPy helpers remain available inside tests as
        reference implementations, but user-facing configs should not be able
        to select the slow per-box PyTorch path or the approximate NumPy path.
        """
        with pytest.raises(ValueError):
            CMGDBConfig(
                subdiv_init=4,
                subdiv_min=6,
                subdiv_max=8,
                box_map_backend=backend,
            )

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

    def test_precompute_batch_points_default_is_auto(self):
        cfg = CMGDBConfig()
        assert cfg.precompute_batch_points == "auto"

    def test_precompute_batch_points_accepts_positive_int(self):
        cfg = CMGDBConfig(precompute_batch_points=4096)
        assert cfg.precompute_batch_points == 4096

    def test_precompute_batch_points_rejects_zero(self):
        with pytest.raises(ValueError):
            CMGDBConfig(precompute_batch_points=0)

    def test_precompute_batch_points_rejects_negative(self):
        with pytest.raises(ValueError):
            CMGDBConfig(precompute_batch_points=-100)

    def test_precompute_batch_points_rejects_unknown_string(self):
        with pytest.raises(ValueError):
            CMGDBConfig(precompute_batch_points="huge")


class TestBoxMapAdaptivePrecomputed:
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

    @pytest.mark.parametrize("subdiv_max", [5, 7, 9, 11, 13])
    def test_agrees_with_numpy_backend_at_odd_subdiv_max(self, subdiv_max):
        """Adaptive precompute must match numpy backend at every depth in
        [2, subdiv_max] for odd subdiv_max, where CMGDB's bisection cycle
        produces rectangular cells (axis-0 bisected one more time than
        axis-1 at odd depths).

        Paper schedules (leslie3d_spurious: max=27, leslie_contraction:
        max=28 with subdiv_init=25, leslie3d_success: max=29) all hit odd
        depths in their (init, min, max) ladder. The existing
        ``test_agrees_with_numpy_backend_at_multiple_depths`` only exercises
        subdiv_max=14 at even depths {8, 10, 12, 14}, so all tested cells
        are square. This test fills the rectangular-cell gap up to a
        memory-friendly subdiv_max.
        """
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        d = bounds.dim

        from latentdynamics.analysis.morse import make_box_map_adaptive_precomputed

        G_adp = make_box_map_adaptive_precomputed(
            m.latent_map, bounds, subdiv_max=subdiv_max, padding=False
        )
        G_np = make_box_map_numpy(m.latent_map, padding=False)

        rng = np.random.default_rng(subdiv_max)
        L, U = bounds.lower, bounds.upper
        for depth in range(2, subdiv_max + 1):
            n_per_axis = np.array(
                [2 ** ((depth + d - 1 - j) // d) for j in range(d)], dtype=np.int64
            )
            side = (U - L) / n_per_axis
            for _ in range(20):
                i = np.array([rng.integers(0, n_per_axis[j]) for j in range(d)])
                rect = list(L + i * side) + list(L + (i + 1) * side)
                out_adp = np.array(G_adp(rect))
                out_np = np.array(G_np(rect))
                np.testing.assert_allclose(
                    out_adp, out_np, atol=1e-5, rtol=1e-5,
                    err_msg=f"subdiv_max={subdiv_max}, depth={depth}, i={i.tolist()}",
                )

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

    def test_morse_graph_matches_numpy_backend(self):
        """End-to-end equivalence: same model + bounds + subdivs, two backends,
        same Morse graph (num_vertices, edges, Conley index strings)."""
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))

        import CMGDB

        from latentdynamics.analysis.morse import compute_morse_graph

        cfg_adp = CMGDBConfig(
            subdiv_init=4,
            subdiv_min=6,
            subdiv_max=8,
            box_map_backend="adaptive_precomputed",
        )
        mg_adp, _ = compute_morse_graph(m, bounds, cfg_adp, device=torch.device("cpu"))
        model_np = CMGDB.Model(
            cfg_adp.subdiv_min,
            cfg_adp.subdiv_max,
            cfg_adp.subdiv_init,
            cfg_adp.subdiv_limit,
            bounds.lower.tolist(),
            bounds.upper.tolist(),
            make_box_map_numpy(m.latent_map, padding=cfg_adp.padding),
        )
        mg_np, _ = CMGDB.ComputeConleyMorseGraph(model_np)

        assert mg_adp.num_vertices() == mg_np.num_vertices()
        ann_adp = sorted(mg_adp.annotations(v) for v in range(mg_adp.num_vertices()))
        ann_np = sorted(mg_np.annotations(v) for v in range(mg_np.num_vertices()))
        assert ann_adp == ann_np
        assert set(mg_adp.edges()) == set(mg_np.edges())

    def test_padding_widens_rect(self):
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        from latentdynamics.analysis.morse import make_box_map_adaptive_precomputed

        G_pad = make_box_map_adaptive_precomputed(
            m.latent_map, bounds, subdiv_max=4, padding=True
        )
        G_nopad = make_box_map_adaptive_precomputed(
            m.latent_map, bounds, subdiv_max=4, padding=False
        )
        rect = [-0.5, 0.0, 0.0, 0.5]
        out_pad = np.array(G_pad(rect))
        out_nopad = np.array(G_nopad(rect))
        box_size = np.array([0.5, 0.5])
        np.testing.assert_allclose(out_pad[:2], out_nopad[:2] - box_size, atol=1e-10)
        np.testing.assert_allclose(out_pad[2:], out_nopad[2:] + box_size, atol=1e-10)

    def test_gelu_precomputed_matches_direct_pytorch_boxmap(self):
        """Precomputed backends should use the actual Torch network output.

        GELU is important here because the old NumPy helper uses the tanh
        approximation, while ``torch.nn.GELU()`` defaults to the exact form.
        """
        torch.manual_seed(0)
        arch = ArchConfig(
            num_layers=1,
            hidden_shape=4,
            high_dims=4,
            low_dims=2,
            activation="gelu",
        )
        m = build_autoencoder(arch)
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        G_pre = make_box_map_uniform_precomputed(
            m.latent_map,
            bounds,
            subdiv_k=4,
            padding=False,
            device=torch.device("cpu"),
        )
        G_torch = make_box_map(m.latent_map, device=torch.device("cpu"), padding=False)

        rect = [-0.5, 0.0, 0.0, 0.5]
        np.testing.assert_allclose(G_pre(rect), G_torch(rect), atol=1e-7, rtol=1e-7)


class TestPrecomputeForwardChunking:
    """The precompute backends must split the latent-map forward pass into
    bounded chunks so that high subdiv_max grids do not allocate one giant
    activation tensor."""

    def _record_forward_sizes(self, latent_map):
        """Register a pre-forward hook that records each forward batch size."""
        sizes: list[int] = []

        def hook(_module, args):
            x = args[0]
            sizes.append(int(x.shape[0]))

        handle = latent_map.register_forward_pre_hook(hook)
        return sizes, handle

    def test_adaptive_precompute_chunks_when_batch_smaller_than_total(self):
        from latentdynamics.analysis.morse import make_box_map_adaptive_precomputed

        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))

        sizes, handle = self._record_forward_sizes(m.latent_map)
        try:
            # subdiv_max=8, d=2 -> M=4, n_per_axis=16, corners=17^2=289.
            G = make_box_map_adaptive_precomputed(
                m.latent_map, bounds, subdiv_max=8, precompute_batch_points=64
            )
        finally:
            handle.remove()
        assert callable(G)
        assert len(sizes) >= 2, f"expected multiple forward calls, got {sizes}"
        assert max(sizes) <= 64, f"chunk size exceeded cap: {sizes}"
        assert sum(sizes) == 289

    def test_uniform_precompute_chunks_when_batch_smaller_than_total(self):
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))

        sizes, handle = self._record_forward_sizes(m.latent_map)
        try:
            # subdiv_k=8, d=2 -> n_per_axis=16, corners=17^2=289.
            G = make_box_map_uniform_precomputed(
                m.latent_map, bounds, subdiv_k=8, precompute_batch_points=64
            )
        finally:
            handle.remove()
        assert callable(G)
        assert len(sizes) >= 2, f"expected multiple forward calls, got {sizes}"
        assert max(sizes) <= 64
        assert sum(sizes) == 289

    def test_chunked_output_matches_one_shot(self):
        """Splitting the forward pass must not change the precomputed values."""
        from latentdynamics.analysis.morse import (
            _precompute_corner_grid,
        )

        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        L = np.array([-1.0, -1.0])
        U = np.array([1.0, 1.0])

        ys_chunked, _ = _precompute_corner_grid(
            m.latent_map, L, U, corners_per_axis=17, d=2,
            device=torch.device("cpu"), batch_points=64,
        )
        # Large explicit batch -> single forward pass.
        ys_oneshot, _ = _precompute_corner_grid(
            m.latent_map, L, U, corners_per_axis=17, d=2,
            device=torch.device("cpu"), batch_points=10_000,
        )
        np.testing.assert_array_equal(ys_chunked, ys_oneshot)

    def test_explicit_int_batch_used_directly(self):
        from latentdynamics.analysis.morse import make_box_map_adaptive_precomputed

        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))

        sizes, handle = self._record_forward_sizes(m.latent_map)
        try:
            G = make_box_map_adaptive_precomputed(
                m.latent_map, bounds, subdiv_max=8, precompute_batch_points=200
            )
        finally:
            handle.remove()
        assert callable(G)
        # 289 points, batch=200 -> chunks of size 200 and 89.
        assert sizes == [200, 89]

    def test_auto_resolver_returns_positive_int_within_total(self):
        from latentdynamics.analysis.morse import _resolve_precompute_batch_points

        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        n_total = 289
        chunk = _resolve_precompute_batch_points(
            "auto", latent_map=m.latent_map, n_total=n_total,
            device=torch.device("cpu"),
        )
        assert isinstance(chunk, int)
        assert 1 <= chunk <= n_total

    def test_auto_resolver_caps_at_total(self):
        from latentdynamics.analysis.morse import _resolve_precompute_batch_points

        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        chunk = _resolve_precompute_batch_points(
            "auto", latent_map=m.latent_map, n_total=10,
            device=torch.device("cpu"),
        )
        assert chunk == 10

    def test_explicit_int_clamped_to_total(self):
        from latentdynamics.analysis.morse import _resolve_precompute_batch_points

        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        chunk = _resolve_precompute_batch_points(
            1_000_000, latent_map=m.latent_map, n_total=500,
            device=torch.device("cpu"),
        )
        assert chunk == 500

    def test_morse_graph_matches_numpy_backend_with_finite_batch(self):
        """End-to-end parity: a config with a small precompute batch produces
        the same Morse graph as the numpy backend."""
        torch.manual_seed(0)
        m = build_autoencoder(_arch())
        bounds = LatentBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))

        import CMGDB

        from latentdynamics.analysis.morse import compute_morse_graph

        cfg_adp = CMGDBConfig(
            subdiv_init=4,
            subdiv_min=6,
            subdiv_max=8,
            box_map_backend="adaptive_precomputed",
            precompute_batch_points=32,
        )
        mg_adp, _ = compute_morse_graph(m, bounds, cfg_adp, device=torch.device("cpu"))
        model_np = CMGDB.Model(
            cfg_adp.subdiv_min,
            cfg_adp.subdiv_max,
            cfg_adp.subdiv_init,
            cfg_adp.subdiv_limit,
            bounds.lower.tolist(),
            bounds.upper.tolist(),
            make_box_map_numpy(m.latent_map, padding=cfg_adp.padding),
        )
        mg_np, _ = CMGDB.ComputeConleyMorseGraph(model_np)

        assert mg_adp.num_vertices() == mg_np.num_vertices()
        ann_adp = sorted(mg_adp.annotations(v) for v in range(mg_adp.num_vertices()))
        ann_np = sorted(mg_np.annotations(v) for v in range(mg_np.num_vertices()))
        assert ann_adp == ann_np
        assert set(mg_adp.edges()) == set(mg_np.edges())
