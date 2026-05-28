"""CMGDB wrapper: bounds inference, BoxMap backends, and Morse-graph computation."""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import CMGDB
import numpy as np
import torch
from numpy.typing import NDArray

from ..config.schema import CMGDBConfig
from ..models.autoencoder import LatentDynamicsAutoencoder

# Auto-mode chunk bounds. The lower bound keeps the inner loop from
# being dominated by per-call PyTorch overhead on tiny chunks; the upper
# bound caps single-buffer allocations on devices (notably MPS) where
# very large tensors can fail even when nominal "available" memory is
# larger.
_AUTO_BATCH_MIN = 4096
_AUTO_BATCH_MAX = 4 * 1024 * 1024
# Fraction of detected free memory the auto heuristic is allowed to spend
# on a single chunk's transient float32 activations. Conservative: leaves
# headroom for the persisted float64 lookup table, the latent map's own
# parameter buffers, and arbitrary other tensors live during precompute.
_AUTO_BATCH_MEMORY_FRACTION = 0.25
# Hard cap on the per-chunk allocation budget for MPS, because the MPS
# allocator imposes a single-buffer size limit roughly governed by
# `recommendedMaxWorkingSetSize` that is well below total unified RAM on
# Apple Silicon. 2 GiB stays under all known caps without being so small
# that the loop becomes overhead-bound on big lattices.
_MPS_PER_CHUNK_BUDGET_BYTES = 2 * 1024 * 1024 * 1024

NumpyMLP = list[tuple[NDArray[np.float64], NDArray[np.float64], str | None]]
ResolvedBoxMapBackend = Literal[
    "uniform_precomputed",
    "adaptive_precomputed",
]


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
    """PyTorch scalar-call BoxMap backend.

    Each box pays full PyTorch per-tensor overhead for every one of its ``2^d``
    corners, via CMGDB's ``mode='corners'`` list comprehension. For trained MLP
    latent maps in low dimension this overhead — not arithmetic — is the
    dominant cost; prefer ``auto`` unless you need this backend for debugging.
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


def _max_linear_width(latent_map: torch.nn.Module) -> int:
    """Maximum ``out_features`` across the ``nn.Linear`` layers in
    ``latent_map``. Drives the per-point memory estimate for auto-chunking."""
    widths = [
        int(layer.out_features)
        for layer in latent_map.modules()
        if isinstance(layer, torch.nn.Linear)
    ]
    if not widths:
        return 1
    return max(widths)


def _get_memory_budget_bytes() -> int | None:
    """Resolve the available memory budget in bytes for transient allocations.

    Checks the env var LATENTDYNAMICS_MEM_BUDGET_BYTES first (supporting
    K/M/G/T unit suffixes). Returns None if not set.
    """
    raw = os.environ.get("LATENTDYNAMICS_MEM_BUDGET_BYTES")
    if not raw:
        return None
    raw = raw.strip()
    unit_scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if raw[-1:] in unit_scale:
        try:
            value = int(raw[:-1])
        except ValueError:
            return None
        scale = unit_scale[raw[-1]]
    else:
        try:
            value = int(raw)
        except ValueError:
            return None
        scale = 1
    if value <= 0:
        return None
    return value * scale


def _available_memory_bytes(device: torch.device | None) -> int | None:
    """Best-effort estimate of free memory the caller is allowed to spend on
    a transient activation buffer on ``device``. Returns ``None`` when no
    signal is available (caller falls back to a conservative constant)."""
    if device is not None and device.type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info(device)
            return int(free)
        except Exception:
            return None

    budget = _get_memory_budget_bytes()
    if budget is not None:
        return budget

    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _resolve_precompute_batch_points(
    batch_points: int | Literal["auto"],
    *,
    latent_map: torch.nn.Module,
    n_total: int,
    device: torch.device | None,
) -> int:
    """Resolve ``CMGDBConfig.precompute_batch_points`` to a concrete chunk
    size. Explicit ints are clamped to ``[1, n_total]``; ``"auto"`` picks a
    device-aware budget using the latent map's max linear width."""
    if isinstance(batch_points, int) and not isinstance(batch_points, bool):
        if batch_points <= 0:
            raise ValueError(
                f"precompute_batch_points must be positive when an int; got {batch_points}"
            )
        return max(1, min(batch_points, n_total))
    if batch_points != "auto":
        raise ValueError(
            f"precompute_batch_points must be a positive int or 'auto'; got {batch_points!r}"
        )

    max_width = _max_linear_width(latent_map)
    # Per-point peak transient cost: input + max-width activation in float32
    # (4 bytes), plus a 2x safety factor for ReLU/Tanh temporaries, copies
    # to/from the device, and the float64 cpu->numpy buffer.
    bytes_per_point = max(64, 8 * max_width)

    available = _available_memory_bytes(device)
    if available is None:
        # Conservative fallback when memory cannot be detected: 1 GiB.
        budget = 1024 * 1024 * 1024
    else:
        budget = int(available * _AUTO_BATCH_MEMORY_FRACTION)

    if device is not None and device.type == "mps":
        budget = min(budget, _MPS_PER_CHUNK_BUDGET_BYTES)

    chunk = budget // bytes_per_point
    chunk = max(_AUTO_BATCH_MIN, min(chunk, _AUTO_BATCH_MAX))
    return int(min(chunk, n_total))


def _precompute_corner_grid(
    latent_map: torch.nn.Module,
    L: NDArray[np.float64],
    U: NDArray[np.float64],
    corners_per_axis: int,
    d: int,
    *,
    device: torch.device | None = None,
    batch_points: int | Literal["auto"] = "auto",
) -> tuple[NDArray[np.float64], int]:
    """Evaluate ``latent_map`` on the ``corners_per_axis^d`` product lattice
    over ``[L, U]`` in chunks and return ``(ys_grid, out_dim)`` with
    ``ys_grid.shape == (corners_per_axis,) * d + (out_dim,)``.

    The forward pass runs in float32 on ``device`` and the result is cast to
    float64. Lattice coordinates are generated chunk-by-chunk from flat
    indices via ``np.unravel_index``; the full coordinate array is never
    materialised. ``batch_points`` controls the forward chunk size --
    ``"auto"`` resolves via :func:`_resolve_precompute_batch_points`.
    """
    device = device or next(latent_map.parameters()).device
    latent_map.eval()

    n_total = int(corners_per_axis) ** int(d)
    # Per-axis step: linspace endpoints L and U give corners_per_axis nodes,
    # so the step between consecutive nodes is (U - L) / (corners_per_axis - 1).
    # When corners_per_axis == 1 the whole lattice collapses to L; treat the
    # step as zero in that edge case.
    if corners_per_axis > 1:
        step = (U - L) / float(corners_per_axis - 1)
    else:
        step = np.zeros_like(U - L)

    chunk_size = _resolve_precompute_batch_points(
        batch_points, latent_map=latent_map, n_total=n_total, device=device
    )

    shape = (corners_per_axis,) * d
    ys_flat: NDArray[np.float64] | None = None
    out_dim = -1

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        flat_idx = np.arange(start, end, dtype=np.int64)
        # multi_idx[:, j] = index of this point along axis j.
        multi_idx = np.stack(np.unravel_index(flat_idx, shape), axis=-1).astype(np.float64)
        points = L + multi_idx * step  # broadcasts; no full meshgrid alloc.
        with torch.no_grad():
            x_t = torch.as_tensor(points, dtype=torch.float32, device=device)
            y_chunk = latent_map(x_t).cpu().numpy().astype(np.float64)
        if ys_flat is None:
            out_dim = int(y_chunk.shape[-1])
            ys_flat = np.empty((n_total, out_dim), dtype=np.float64)
        ys_flat[start:end] = y_chunk

    assert ys_flat is not None, "n_total must be >= 1"
    ys_grid = ys_flat.reshape(shape + (out_dim,))
    return ys_grid, out_dim


def make_box_map_uniform_precomputed(
    latent_map: torch.nn.Module,
    bounds: LatentBounds,
    subdiv_k: int,
    *,
    padding: bool = True,
    device: torch.device | None = None,
    max_table_points: int = 10_000_000,
    precompute_batch_points: int | Literal["auto"] = "auto",
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

    n_per_axis = 2 ** (subdiv_k // d)
    corners_per_axis = n_per_axis + 1
    table_points = corners_per_axis**d
    if table_points > max_table_points:
        raise ValueError(
            f"uniform_precomputed table size ({table_points} corners) exceeds "
            f"max_table_points ({max_table_points}). For d={d}, subdiv_k="
            f"{subdiv_k} -> (2^{subdiv_k // d}+1)^{d} corners. "
            f"Lower subdiv_k or raise max_table_points."
        )

    L = np.asarray(bounds.lower, dtype=np.float64)
    U = np.asarray(bounds.upper, dtype=np.float64)
    box_side = (U - L) / n_per_axis
    ys_grid, out_dim = _precompute_corner_grid(
        latent_map, L, U, corners_per_axis, d,
        device=device, batch_points=precompute_batch_points,
    )

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


def make_box_map_adaptive_precomputed(
    latent_map: torch.nn.Module,
    bounds: LatentBounds,
    subdiv_max: int,
    *,
    padding: bool = True,
    device: torch.device | None = None,
    max_table_points: int = 10_000_000,
    precompute_batch_points: int | Literal["auto"] = "auto",
) -> Callable[[Any], Any]:
    """Whole-grid pre-evaluation for adaptive CMGDB grids.

    Generalises ``make_box_map_uniform_precomputed`` to the adaptive ladder
    ``subdiv_init <= subdiv_min <= subdiv_max``. At any depth ``k`` reached by
    CMGDB's binary subdivision tree, every cell corner sits on the
    ``(2^M + 1)^d`` lattice with ``M = ceil(subdiv_max / d)``. We evaluate the
    latent map on that entire lattice once, then answer ``box_map(rect)`` by
    snapping the rect's lower and upper bounds to integer indices and
    gathering the ``2^d`` corners they span.
    """
    d = bounds.dim
    M = (subdiv_max + d - 1) // d  # ceil(subdiv_max / d)
    n_per_axis = 2**M
    corners_per_axis = n_per_axis + 1

    table_points = corners_per_axis**d
    if table_points > max_table_points:
        raise ValueError(
            f"adaptive_precomputed table size ({table_points} corners) exceeds "
            f"max_table_points ({max_table_points}). For d={d}, subdiv_max="
            f"{subdiv_max}, M=ceil(subdiv_max/d)={M} -> (2^{M}+1)^{d} corners. "
            f"Lower subdiv_max or raise max_table_points."
        )

    L = np.asarray(bounds.lower, dtype=np.float64)
    U = np.asarray(bounds.upper, dtype=np.float64)
    finest_box_side = (U - L) / n_per_axis
    ys_grid, out_dim = _precompute_corner_grid(
        latent_map, L, U, corners_per_axis, d,
        device=device, batch_points=precompute_batch_points,
    )

    # Precompute the 2^d corner-combination matrix once.
    combos = np.array(
        list(itertools.product(range(2), repeat=d)), dtype=np.int64
    )  # (2^d, d), entries in {0, 1}
    axis_idx = np.arange(d, dtype=np.int64)

    def box_map(rect):
        rect_arr = np.asarray(rect, dtype=np.float64)
        i_lo = np.round((rect_arr[:d] - L) / finest_box_side).astype(np.int64)
        i_hi = np.round((rect_arr[d:] - L) / finest_box_side).astype(np.int64)
        np.clip(i_lo, 0, n_per_axis, out=i_lo)
        np.clip(i_hi, 0, n_per_axis, out=i_hi)
        idx_per_axis = np.stack([i_lo, i_hi], axis=0)        # (2, d)
        corner_indices = idx_per_axis[combos, axis_idx]      # (2^d, d)
        corners = ys_grid[tuple(corner_indices.T)]           # (2^d, out_dim)
        Y_l = corners.min(axis=0)
        Y_u = corners.max(axis=0)
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
    backend = _resolve_box_map_backend(cmgdb_cfg, bounds.dim)
    if backend == "uniform_precomputed":
        return make_box_map_uniform_precomputed(
            latent_map,
            bounds,
            cmgdb_cfg.subdiv_max,
            padding=cmgdb_cfg.padding,
            device=device,
            max_table_points=cmgdb_cfg.max_table_points,
            precompute_batch_points=cmgdb_cfg.precompute_batch_points,
        )
    if backend == "adaptive_precomputed":
        return make_box_map_adaptive_precomputed(
            latent_map,
            bounds,
            cmgdb_cfg.subdiv_max,
            padding=cmgdb_cfg.padding,
            device=device,
            max_table_points=cmgdb_cfg.max_table_points,
            precompute_batch_points=cmgdb_cfg.precompute_batch_points,
        )
    raise ValueError(f"unknown box_map_backend: {backend!r}")


def _resolve_box_map_backend(cmgdb_cfg: CMGDBConfig, dim: int) -> ResolvedBoxMapBackend:
    backend = cmgdb_cfg.box_map_backend
    if backend != "auto":
        return cast(ResolvedBoxMapBackend, backend)
    if (
        cmgdb_cfg.subdiv_init == cmgdb_cfg.subdiv_min == cmgdb_cfg.subdiv_max
        and cmgdb_cfg.subdiv_max % dim == 0
    ):
        return "uniform_precomputed"
    return "adaptive_precomputed"


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
