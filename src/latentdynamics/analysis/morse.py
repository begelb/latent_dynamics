"""CMGDB wrapper: bounds inference, BoxMap backends, and Morse-graph computation."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import CMGDB
import numpy as np
import torch
from numpy.typing import NDArray

from ..config.schema import CMGDBConfig
from ..models.autoencoder import LatentDynamicsAutoencoder

NumpyMLP = list[tuple[NDArray[np.float64], NDArray[np.float64], str | None]]


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


def _extract_numpy_mlp(latent_map: torch.nn.Module) -> NumpyMLP:
    """Pull (W, b, activation_name) for each Linear in the latent MLP, as float64 NumPy."""
    layers: NumpyMLP = []
    seq = latent_map.net  # nn.Sequential built by _build_mlp
    children = list(seq.children())
    i = 0
    while i < len(children):
        layer = children[i]
        if isinstance(layer, torch.nn.Linear):
            W = layer.weight.detach().cpu().numpy().astype(np.float64)
            b = layer.bias.detach().cpu().numpy().astype(np.float64)
            act: str | None = None
            if i + 1 < len(children):
                nxt = children[i + 1]
                if isinstance(nxt, torch.nn.ReLU):
                    act = "relu"
                elif isinstance(nxt, torch.nn.Tanh):
                    act = "tanh"
                elif isinstance(nxt, torch.nn.Sigmoid):
                    act = "sigmoid"
                elif isinstance(nxt, torch.nn.GELU):
                    act = "gelu"
                if act is not None:
                    i += 1
            layers.append((W, b, act))
        i += 1
    return layers


def _numpy_mlp_forward(layers: NumpyMLP, X: NDArray[np.float64]) -> NDArray[np.float64]:
    Y = X
    for W, b, act in layers:
        Y = Y @ W.T + b
        if act == "relu":
            Y = np.maximum(Y, 0.0)
        elif act == "tanh":
            Y = np.tanh(Y)
        elif act == "sigmoid":
            Y = 1.0 / (1.0 + np.exp(-Y))
        elif act == "gelu":
            Y = 0.5 * Y * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (Y + 0.044715 * Y**3)))
    return Y


def make_box_map(
    latent_map: torch.nn.Module,
    *,
    device: torch.device | None = None,
    padding: bool = True,
) -> Callable[[Any], Any]:
    """PyTorch scalar-call BoxMap (the default, conservative backend).

    Each box pays full PyTorch per-tensor overhead for every one of its ``2^d``
    corners, via CMGDB's ``mode='corners'`` list comprehension. For trained MLP
    latent maps in low dimension this overhead — not arithmetic — is the
    dominant cost; prefer the ``numpy`` or ``uniform_precomputed`` backends.
    """
    device = device or next(latent_map.parameters()).device
    latent_map.eval()

    @torch.no_grad()
    def g(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        return latent_map(x_t).cpu().numpy()

    def box_map(rect):
        return CMGDB.BoxMap(g, rect, padding=padding)

    return box_map


def make_box_map_numpy(
    latent_map: torch.nn.Module,
    *,
    padding: bool = True,
) -> Callable[[Any], Any]:
    """NumPy-matmul BoxMap: ``2^d`` corners stacked, one matmul chain per box.

    Extracts Linear weights and biases once; the per-box forward pass is a
    fixed-shape NumPy matmul over all corners. PyTorch overhead is paid zero
    times in the inner loop. Output is bit-equivalent in float64 to the
    PyTorch path up to dtype rounding, and the resulting Morse decomposition
    is identical for the cases in ``tests/test_analysis_morse.py``.
    """
    layers = _extract_numpy_mlp(latent_map)

    def box_map(rect):
        dim = len(rect) // 2
        list_intvals = [[rect[d], rect[d + dim]] for d in range(dim)]
        X = np.array(list(itertools.product(*list_intvals)), dtype=np.float64)
        Y = _numpy_mlp_forward(layers, X)
        Y_l = Y.min(axis=0)
        Y_u = Y.max(axis=0)
        if padding:
            box_size = np.array([rect[d + dim] - rect[d] for d in range(dim)])
            Y_l = Y_l - box_size
            Y_u = Y_u + box_size
        return Y_l.tolist() + Y_u.tolist()

    return box_map


def make_box_map_uniform_precomputed(
    latent_map: torch.nn.Module,
    bounds: LatentBounds,
    subdiv_k: int,
    *,
    padding: bool = True,
    device: torch.device | None = None,
) -> Callable[[Any], Any]:
    """Whole-grid pre-evaluation for uniform CMGDB grids.

    At uniform level ``k`` the box partition is a product grid with
    ``2^(k/d)`` boxes along each axis. Every corner is shared between up to
    ``2^d`` neighbouring boxes; the unique corner set has ``(2^(k/d) + 1)^d``
    points. This function evaluates ``latent_map`` on the entire set in a
    single batched forward pass, then returns a ``box_map(rect)`` that does
    only an O(1) index lookup and min/max over ``2^d`` precomputed outputs.

    Requires ``k % d == 0`` (each axis subdivided the same number of times).
    Caller must ensure CMGDB is configured uniform mode
    (``subdiv_init == subdiv_min == subdiv_max == k``); the schema validator
    enforces the latter.
    """
    d = bounds.dim
    if subdiv_k % d != 0:
        raise ValueError(
            f"uniform_precomputed backend requires subdiv_max ({subdiv_k}) divisible "
            f"by latent dim ({d}); got remainder {subdiv_k % d}"
        )

    device = device or next(latent_map.parameters()).device
    latent_map.eval()

    n_per_axis = 2 ** (subdiv_k // d)
    corners_per_axis = n_per_axis + 1
    L = np.asarray(bounds.lower, dtype=np.float64)
    U = np.asarray(bounds.upper, dtype=np.float64)
    box_side = (U - L) / n_per_axis

    axes = [np.linspace(L[i], U[i], corners_per_axis, dtype=np.float64) for i in range(d)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([m.ravel() for m in mesh], axis=-1)  # (N, d)

    with torch.no_grad():
        x_t = torch.as_tensor(points, dtype=torch.float32, device=device)
        y_t = latent_map(x_t)
        ys = y_t.cpu().numpy().astype(np.float64)

    out_dim = ys.shape[-1]
    ys_grid = ys.reshape((corners_per_axis,) * d + (out_dim,))

    def box_map(rect):
        rect_arr = np.asarray(rect, dtype=np.float64)
        center = (rect_arr[:d] + rect_arr[d:]) / 2.0
        idx = np.floor((center - L) / box_side).astype(np.int64)
        # Tolerate floating-point drift at the upper boundary.
        idx = np.clip(idx, 0, n_per_axis - 1)
        slicer = tuple(slice(int(idx[i]), int(idx[i]) + 2) for i in range(d))
        sub = ys_grid[slicer]  # shape (2,)*d + (out_dim,)
        flat = sub.reshape(2**d, out_dim)
        Y_l = flat.min(axis=0)
        Y_u = flat.max(axis=0)
        if padding:
            box_size = rect_arr[d:] - rect_arr[:d]
            Y_l = Y_l - box_size
            Y_u = Y_u + box_size
        return Y_l.tolist() + Y_u.tolist()

    return box_map


def _build_box_map(
    latent_map: torch.nn.Module,
    bounds: LatentBounds,
    cmgdb_cfg: CMGDBConfig,
    *,
    device: torch.device | None = None,
) -> Callable[[Any], Any]:
    backend = cmgdb_cfg.box_map_backend
    if backend == "pytorch":
        return make_box_map(latent_map, device=device, padding=cmgdb_cfg.padding)
    if backend == "numpy":
        return make_box_map_numpy(latent_map, padding=cmgdb_cfg.padding)
    if backend == "uniform_precomputed":
        return make_box_map_uniform_precomputed(
            latent_map,
            bounds,
            cmgdb_cfg.subdiv_max,
            padding=cmgdb_cfg.padding,
            device=device,
        )
    raise ValueError(f"unknown box_map_backend: {backend!r}")


def compute_morse_graph(
    autoencoder: LatentDynamicsAutoencoder,
    bounds: LatentBounds,
    cmgdb_cfg: CMGDBConfig,
    *,
    device: torch.device | None = None,
):
    """Run CMGDB on the given latent map and return ``(morse_graph, map_graph)``."""
    box_map = _build_box_map(autoencoder.latent_map, bounds, cmgdb_cfg, device=device)
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
