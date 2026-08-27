"""CMGDB wrapper: bounds inference, BoxMap backends, and Morse-graph computation."""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Sequence
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
    """Axis-aligned min/max extents of the latent phase space."""

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    @property
    def dim(self) -> int:
        return int(self.lower.shape[0])


class _PrecomputedBoxMap:
    """Callable lookup table with CMGDB's optional batched-map interface.

    ``batch_lookup``, when supplied, evaluates a whole chunk of rectangles with
    array operations instead of one NumPy call chain per rectangle. That matters
    a great deal: CMGDB routes every adjacency query through this interface, so
    the per-rectangle constant is multiplied by millions. Measured on a 2-D
    latent map, the per-rectangle loop costs 12.3 us/rect and the vectorized
    form 0.48 us/rect, for 85% of ``ComputeConleyMorseGraph`` wall clock.

    The scalar ``__call__`` keeps its own dedicated implementation rather than
    routing through the batch path, so single-rectangle latency is unchanged.
    """

    def __init__(
        self,
        lookup: Callable[[Any], list[float]],
        batch_lookup: Callable[[Any], list[list[float]]] | None = None,
    ) -> None:
        self._lookup = lookup
        self._batch_lookup = batch_lookup

    def __call__(self, rect: Any) -> list[float]:
        return self._lookup(rect)

    def batch(self, rects: Any) -> list[list[float]]:
        if self._batch_lookup is not None:
            return self._batch_lookup(rects)
        return [self._lookup(rect) for rect in rects]


def infer_latent_bounds(
    encoder: torch.nn.Module,
    all_data_scaled: NDArray[np.float64],
    *,
    epsilon_frac: float = 0.01,
    device: torch.device | None = None,
    latent_map: torch.nn.Module | None = None,
    clip_lower: Sequence[float] | None = None,
    clip_upper: Sequence[float] | None = None,
) -> LatentBounds:
    """Infer an expanded latent rectangle from encoded ambient points.

    ``latent_map`` optionally adds the one-step latent images to the bound
    cloud.  This reproduces archived examples whose CMGDB domain was inferred
    from ``E(X)`` and ``G(E(X))`` rather than ``E(X)`` alone.  Optional clip
    vectors make an activation range (for example tanh's ``[-1, 1]``) an
    explicit, provenance-tracked part of the recipe.
    """
    device = device or next(encoder.parameters()).device
    encoder.eval()
    if latent_map is not None:
        latent_map.eval()
    with torch.no_grad():
        z = encoder(torch.as_tensor(all_data_scaled, dtype=torch.float32, device=device))
        clouds = [z]
        if latent_map is not None:
            clouds.append(latent_map(z))
        latent = torch.cat(clouds, dim=0).cpu().numpy()
    lower = latent.min(axis=0)
    upper = latent.max(axis=0)
    buffer = epsilon_frac * (upper - lower)
    lower = lower - buffer
    upper = upper + buffer
    if (clip_lower is None) != (clip_upper is None):
        raise ValueError("clip_lower and clip_upper must be supplied together")
    if clip_lower is not None and clip_upper is not None:
        clip_lower_array = np.asarray(clip_lower, dtype=np.float64)
        clip_upper_array = np.asarray(clip_upper, dtype=np.float64)
        if clip_lower_array.shape != lower.shape or clip_upper_array.shape != upper.shape:
            raise ValueError("latent-bound clip vectors must match the latent dimension")
        lower = np.maximum(lower, clip_lower_array)
        upper = np.minimum(upper, clip_upper_array)
    if np.any(lower >= upper):
        raise ValueError(
            "inferred latent bounds are empty after expansion/clipping: "
            f"lower={lower.tolist()} upper={upper.tolist()}"
        )
    return LatentBounds(lower=lower, upper=upper)


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
    K/M/G/T unit suffixes, case-insensitive). Returns None if not set.
    """
    raw = os.environ.get("LATENTDYNAMICS_MEM_BUDGET_BYTES")
    if not raw:
        return None
    raw = raw.strip()
    unit_scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if raw[-1:].upper() in unit_scale:
        try:
            value = int(raw[:-1])
        except ValueError:
            return None
        scale = unit_scale[raw[-1].upper()]
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
    corners_per_axis: int | Sequence[int],
    d: int,
    *,
    device: torch.device | None = None,
    batch_points: int | Literal["auto"] = "auto",
) -> tuple[NDArray[np.float64], int]:
    """Evaluate ``latent_map`` on the corner lattice over ``[L, U]`` in chunks
    and return ``(ys_grid, out_dim)``.

    ``corners_per_axis`` is either a single count used for every axis, or one
    count per axis. The per-axis form matters because CMGDB bisects a different
    coordinate at each depth (``depth % d``), so at a subdivision level that is
    not a multiple of ``d`` the axes are refined unequally and a cubic lattice
    over-samples the coarse ones. ``ys_grid.shape == tuple(corners_per_axis) + (out_dim,)``.

    The forward pass runs in float32 on ``device`` and the result is cast to
    float64. Lattice coordinates are generated chunk-by-chunk from flat
    indices via ``np.unravel_index``; the full coordinate array is never
    materialised. ``batch_points`` controls the forward chunk size --
    ``"auto"`` resolves via :func:`_resolve_precompute_batch_points`.
    """
    device = device or next(latent_map.parameters()).device
    latent_map.eval()

    if isinstance(corners_per_axis, int):
        shape = (int(corners_per_axis),) * int(d)
    else:
        shape = tuple(int(c) for c in corners_per_axis)
        if len(shape) != int(d):
            raise ValueError(f"corners_per_axis has {len(shape)} entries but d={d}")

    n_total = 1
    for c in shape:
        n_total *= c

    # Per-axis step: linspace endpoints L and U give shape[j] nodes on axis j,
    # so the step between consecutive nodes is (U - L) / (shape[j] - 1). When an
    # axis has a single node the lattice collapses to L there; step is zero.
    counts = np.asarray(shape, dtype=np.int64)
    step = np.zeros_like(U - L, dtype=np.float64)
    multi = counts > 1
    step[multi] = (U - L)[multi] / (counts[multi] - 1).astype(np.float64)

    chunk_size = _resolve_precompute_batch_points(
        batch_points, latent_map=latent_map, n_total=n_total, device=device
    )
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
    ys_grid = ys_flat.reshape((*shape, out_dim))
    return ys_grid, out_dim


def make_box_map_uniform_precomputed(
    latent_map: torch.nn.Module,
    bounds: LatentBounds,
    subdiv_k: int,
    *,
    padding: bool = True,
    device: torch.device | None = None,
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
    L = np.asarray(bounds.lower, dtype=np.float64)
    U = np.asarray(bounds.upper, dtype=np.float64)
    box_side = (U - L) / n_per_axis
    ys_grid, out_dim = _precompute_corner_grid(
        latent_map,
        L,
        U,
        corners_per_axis,
        d,
        device=device,
        batch_points=precompute_batch_points,
    )

    # Corner offsets, hoisted out of the per-box work. Order is irrelevant
    # because only min/max over the 2^d corners is taken.
    combos = np.array(list(itertools.product(range(2), repeat=d)), dtype=np.int64)  # (2^d, d)

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

    def box_map_batch(rects):
        R = np.asarray(rects, dtype=np.float64)
        if R.size == 0:
            return []
        R = R.reshape(-1, 2 * d)
        center = (R[:, :d] + R[:, d:]) / 2.0
        idx = np.floor((center - L) / box_side).astype(np.int64)
        np.clip(idx, 0, n_per_axis - 1, out=idx)
        corner_indices = idx[:, None, :] + combos[None, :, :]  # (m, 2^d, d)
        corners = ys_grid[tuple(corner_indices[..., k] for k in range(d))]
        Y_l = corners.min(axis=1)
        Y_u = corners.max(axis=1)
        if padding:
            box_size = R[:, d:] - R[:, :d]
            Y_l = Y_l - box_size
            Y_u = Y_u + box_size
        return np.concatenate([Y_l, Y_u], axis=1).tolist()

    return _PrecomputedBoxMap(box_map, box_map_batch)


# float32 matmul results depend on the batch shape: the backend selects
# different blocking for small or ragged batches, and the outputs differ by up
# to one ulp. Table construction runs in large chunks, so on-demand evaluation
# must too, or the same point evaluated the two ways disagrees. Padding every
# batch up to a multiple of this block makes them agree exactly.
#
# 256 is the smallest block that is exact; 128 and below are not. It is also
# the smallest that is affordable: on-demand calls arrive tiny and frequent
# (~15 points per call on the production ladder), so padding to 4096 would
# evaluate 314M rows to serve 1.1M points -- worse than the table it replaces.
_GEMM_BLOCK = 256


def _eval_points(
    latent_map: torch.nn.Module,
    points: NDArray[np.float64],
    *,
    device: torch.device,
    chunk: int,
) -> NDArray[np.float64]:
    """Evaluate ``latent_map`` at arbitrary points, matching the table recipe.

    Uses the same float32 forward / float64 cast as
    :func:`_precompute_corner_grid`, and pads each batch to a multiple of
    ``_GEMM_BLOCK``, so a point evaluated here and the same point read from the
    table agree bit for bit.
    """
    latent_map.eval()
    n = int(points.shape[0])
    if n == 0:
        raise ValueError("points must be non-empty")
    pad = (-n) % _GEMM_BLOCK
    if pad:
        points = np.vstack([points, np.repeat(points[:1], pad, axis=0)])
    chunk = max(_GEMM_BLOCK, (chunk // _GEMM_BLOCK) * _GEMM_BLOCK)

    total = int(points.shape[0])
    out: NDArray[np.float64] | None = None
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        with torch.no_grad():
            x_t = torch.as_tensor(points[start:end], dtype=torch.float32, device=device)
            y = latent_map(x_t).cpu().numpy().astype(np.float64)
        if out is None:
            out = np.empty((total, y.shape[-1]), dtype=np.float64)
        out[start:end] = y
    assert out is not None, "points must be non-empty"
    return out[:n]


def make_box_map_adaptive_precomputed(
    latent_map: torch.nn.Module,
    bounds: LatentBounds,
    subdiv_max: int,
    *,
    padding: bool = True,
    device: torch.device | None = None,
    precompute_batch_points: int | Literal["auto"] = "auto",
    dense_subdiv: int | None = None,
) -> Callable[[Any], Any]:
    """Whole-grid pre-evaluation for adaptive CMGDB grids.

    With ``dense_subdiv`` set, the dense table is built at that depth instead of
    at ``subdiv_max``, and corners that do not land on the coarser lattice are
    evaluated on demand. This matters because CMGDB reaches ``subdiv_max`` only
    inside the recurrent set: on the production 24/27/28 ladder, 97.8% of the
    17.2M rectangle queries are at depth 24, and a table sized for depth 28
    costs 268M evaluations to serve them against 16.8M for a depth-24 table.

    Results are unchanged. Every box corner at any depth <= ``subdiv_max`` lies
    on the finest lattice, and the two ways of computing such a point agree
    exactly because the lattice spacings differ by a power of two, which scales
    floating-point values without rounding.

    Generalises ``make_box_map_uniform_precomputed`` to the adaptive ladder
    ``subdiv_init <= subdiv_min <= subdiv_max``. At any depth ``k`` reached by
    CMGDB's binary subdivision tree, every cell corner sits on the
    ``(2^M + 1)^d`` lattice with ``M = ceil(subdiv_max / d)``. We evaluate the
    latent map on that entire lattice once, then answer ``box_map(rect)`` by
    snapping the rect's lower and upper bounds to integer indices and
    gathering the ``2^d`` corners they span.
    """
    d = bounds.dim
    # CMGDB bisects coordinate ``depth % d`` at each depth, so after
    # ``subdiv_max`` subdivisions axis j has been split
    # ``(subdiv_max - j + d - 1) // d`` times -- not ``ceil(subdiv_max / d)``
    # times on every axis. Using the max on all axes over-samples every axis but
    # the first whenever ``subdiv_max % d != 0``, which in 2-D doubles the table
    # (at subdiv_max=29: 32769^2 instead of 32769 x 16385, 16 GiB instead of 8).
    # Verified empirically against CMGDB's finest box widths.
    axis_depths = np.array([(subdiv_max - j + d - 1) // d for j in range(d)], dtype=np.int64)
    n_per_axis = (2**axis_depths).astype(np.int64)

    # Depth at which the dense table is built. `dense_subdiv=None` keeps the
    # historical behaviour of tabulating the finest level over the whole domain.
    table_subdiv = subdiv_max if dense_subdiv is None else int(dense_subdiv)
    if table_subdiv > subdiv_max:
        raise ValueError(f"dense_subdiv ({table_subdiv}) must not exceed subdiv_max ({subdiv_max})")
    table_axis_depths = np.array(
        [(table_subdiv - j + d - 1) // d for j in range(d)], dtype=np.int64
    )
    table_n_per_axis = (2**table_axis_depths).astype(np.int64)
    corners_per_axis = table_n_per_axis + 1
    # Index stride from the finest lattice down to the table lattice. Always a
    # power of two per axis, which is what makes the two point computations
    # agree bitwise.
    stride = (2 ** (axis_depths - table_axis_depths)).astype(np.int64)

    L = np.asarray(bounds.lower, dtype=np.float64)
    U = np.asarray(bounds.upper, dtype=np.float64)
    finest_box_side = (U - L) / n_per_axis
    ys_grid, _out_dim = _precompute_corner_grid(
        latent_map,
        L,
        U,
        corners_per_axis.tolist(),
        d,
        device=device,
        batch_points=precompute_batch_points,
    )
    eval_device = device or next(latent_map.parameters()).device
    ondemand_chunk = _resolve_precompute_batch_points(
        precompute_batch_points,
        latent_map=latent_map,
        n_total=int(np.prod(n_per_axis.astype(object))),
        device=eval_device,
    )

    out_dim = int(ys_grid.shape[-1])
    # Strides are powers of two, so alignment is a bit test and the table index
    # is a shift. Both are much cheaper than % and // over ~70M corner indices.
    stride_mask = (stride - 1).astype(np.int64)
    stride_shift = (axis_depths - table_axis_depths).astype(np.int64)
    # Off-lattice corners are shared between neighbouring boxes and re-queried
    # across refinement passes, so memoize them. Keys pack the finest-lattice
    # index; on the production ladder the whole cache holds ~10^5 entries.
    pack_dims = (n_per_axis + 1).astype(np.int64)
    ondemand_cache: dict[int, NDArray[np.float64]] = {}

    def _corner_values(corner_idx: NDArray[np.int64]) -> NDArray[np.float64]:
        """Values at finest-lattice integer corner indices, shape (..., out_dim).

        Corners on the table lattice are read from it; the rest are evaluated
        once and cached. On the production ladder the second branch handles
        ~2% of rectangles.
        """
        flat = corner_idx.reshape(-1, d)
        resid = flat & stride_mask
        if not resid.any():
            # Fast path: everything is on the table. This is 97.8% of queries
            # on the production ladder, so it must not pay for the split.
            t_idx = flat >> stride_shift
            vals = ys_grid[tuple(t_idx[:, k] for k in range(d))]
            return vals.reshape(*corner_idx.shape[:-1], out_dim)

        on_table = ~resid.any(axis=1)
        out = np.empty((flat.shape[0], out_dim), dtype=np.float64)
        if on_table.any():
            t_idx = flat[on_table] >> stride_shift
            out[on_table] = ys_grid[tuple(t_idx[:, k] for k in range(d))]

        off_rows = np.nonzero(~on_table)[0]
        off_idx = flat[off_rows]
        keys = np.ravel_multi_index(tuple(off_idx[:, k] for k in range(d)), pack_dims)
        misses = []
        for i, key in enumerate(keys.tolist()):
            hit = ondemand_cache.get(key)
            if hit is None:
                misses.append(i)
            else:
                out[off_rows[i]] = hit
        if misses:
            m_idx = off_idx[misses]
            pts = L + m_idx.astype(np.float64) * finest_box_side
            vals = _eval_points(latent_map, pts, device=eval_device, chunk=ondemand_chunk)
            for j, i in enumerate(misses):
                ondemand_cache[int(keys[i])] = vals[j]
                out[off_rows[i]] = vals[j]
        return out.reshape(*corner_idx.shape[:-1], out_dim)

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
        idx_per_axis = np.stack([i_lo, i_hi], axis=0)  # (2, d)
        corner_indices = idx_per_axis[combos, axis_idx]  # (2^d, d)
        corners = _corner_values(corner_indices)  # (2^d, out_dim)
        Y_l = corners.min(axis=0)
        Y_u = corners.max(axis=0)
        if padding:
            box_size = rect_arr[d:] - rect_arr[:d]
            Y_l = Y_l - box_size
            Y_u = Y_u + box_size
        return Y_l.tolist() + Y_u.tolist()

    def box_map_batch(rects):
        R = np.asarray(rects, dtype=np.float64)
        if R.size == 0:
            return []
        R = R.reshape(-1, 2 * d)
        i_lo = np.round((R[:, :d] - L) / finest_box_side).astype(np.int64)
        i_hi = np.round((R[:, d:] - L) / finest_box_side).astype(np.int64)
        np.clip(i_lo, 0, n_per_axis, out=i_lo)
        np.clip(i_hi, 0, n_per_axis, out=i_hi)
        idx_per_axis = np.stack([i_lo, i_hi], axis=1)  # (m, 2, d)
        corner_indices = idx_per_axis[:, combos, axis_idx]  # (m, 2^d, d)
        corners = _corner_values(corner_indices)  # (m, 2^d, out_dim)
        Y_l = corners.min(axis=1)
        Y_u = corners.max(axis=1)
        if padding:
            box_size = R[:, d:] - R[:, :d]
            Y_l = Y_l - box_size
            Y_u = Y_u + box_size
        return np.concatenate([Y_l, Y_u], axis=1).tolist()

    return _PrecomputedBoxMap(box_map, box_map_batch)


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
            precompute_batch_points=cmgdb_cfg.precompute_batch_points,
        )
    if backend == "adaptive_precomputed":
        dense_subdiv = {
            "init": cmgdb_cfg.subdiv_init,
            "min": cmgdb_cfg.subdiv_min,
            "max": cmgdb_cfg.subdiv_max,
        }[cmgdb_cfg.adaptive_precompute_subdiv]
        return make_box_map_adaptive_precomputed(
            latent_map,
            bounds,
            cmgdb_cfg.subdiv_max,
            padding=cmgdb_cfg.padding,
            device=device,
            precompute_batch_points=cmgdb_cfg.precompute_batch_points,
            dense_subdiv=dense_subdiv,
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
    need_map_graph: bool = True,
):
    """Run CMGDB on the given latent map and return ``(morse_graph, map_graph)``.

    ``need_map_graph=False`` returns ``(morse_graph, None)``. With
    ``need_map_graph=True`` the returned ``MapGraph`` is eagerly cached
    (``cache_map_graph=True``): that costs a full extra pass of the box map
    over the whole phase space -- roughly half of all box-map evaluations --
    and is only needed for exact regions of attraction, whose Python-side
    adjacency walks would otherwise re-evaluate the map per box.
    """
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
    batch_map = getattr(box_map, "batch", None)
    if callable(batch_map) and hasattr(model, "set_batch_map"):
        model.set_batch_map(batch_map)
    if not need_map_graph:
        morse_graph, _ = CMGDB.ComputeConleyMorseGraph(model)
        return morse_graph, None
    return CMGDB.ComputeConleyMorseGraph(model, cache_map_graph=True)