"""Batched corner-lattice evaluation helpers, vendored from CMGDB.

CMGDB (>= 1.5.0) ships an equivalent lattice builder as
``CMGDB.PrecomputedBoxMap``, but importing helpers from CMGDB directly made
``latentdynamics.analysis`` fail to import outright against a build without
them, before any capability check could report something useful. They are
pure NumPy/Torch array plumbing with no CMGDB dependency, so vendoring costs
nothing and removes the hard coupling. CMGDB is MIT-licensed and the notice
is preserved in ``licenses/CMGDB-LICENSE``.

``latentdynamics.analysis.morse`` keeps its own equivalents for the ordinary
box-map backends. These exist for
:mod:`latentdynamics.analysis.hierarchical_precomputed`, whose dense-coarse /
sparse-fine lattice needs the chunked evaluator directly.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

__all__ = [
    "as_batched_evaluator",
    "precompute_corner_grid",
    "resolve_batch_points",
    "select_torch_device",
]


BatchPoints = Union[int, Literal["auto"]]

_AUTO_BATCH_MIN = 4096

_AUTO_BATCH_MAX = 4 * 1024 * 1024

_AUTO_BATCH_MEMORY_FRACTION = 0.25

_MPS_PER_CHUNK_BUDGET_BYTES = 2 * 1024 * 1024 * 1024

def _import_torch(*, required: bool):
    try:
        import torch
    except ImportError:
        if required:
            raise RuntimeError("Torch support requires torch to be installed") from None
        return None
    return torch

def select_torch_device(device: Any = "auto"):
    """Return a ``torch.device`` using CMGDB's default preference.

    ``device="auto"`` chooses ``mps`` when available, then ``cuda``, then
    ``cpu``. Explicit unavailable accelerators raise a clear error.
    """
    torch = _import_torch(required=True)
    if hasattr(device, "type"):
        return device
    if device is None or device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    torch_device = torch.device(device)
    if torch_device.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("requested torch device 'mps' is not available")
    elif torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested torch device 'cuda' is not available")
    return torch_device

def _max_linear_width(module: Any) -> int:
    torch = _import_torch(required=False)
    if torch is None:
        return 1
    widths = []
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            widths.extend([int(layer.in_features), int(layer.out_features)])
    return max(widths, default=1)

def _validate_batched_output(values: Any, n_points: int) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim == 1 and n_points == 1:
        out = out.reshape(1, -1)
    if out.ndim != 2:
        raise ValueError(
            "batched evaluator must return a 2D array with shape "
            f"(n_points, output_dim); got shape {out.shape}"
        )
    if out.shape[0] != n_points:
        raise ValueError(
            "batched evaluator returned the wrong number of rows: "
            f"expected {n_points}, got {out.shape[0]}"
        )
    return out

def as_batched_evaluator(f: Any, *, device: Any = "auto"):
    """Return a callable that maps ``(n, d)`` float64 NumPy arrays to arrays.

    Non-Torch callables are assumed to already be batched. If Torch is
    installed and ``f`` is a ``torch.nn.Module``, the returned evaluator runs
    the module in ``float32`` on ``device`` and returns ``float64`` NumPy data.
    """
    torch = _import_torch(required=False)
    if torch is not None and isinstance(f, torch.nn.Module):
        torch_device = select_torch_device(device)
        module = f.to(torch_device)
        module.eval()

        def torch_evaluator(points: np.ndarray) -> np.ndarray:
            points = np.asarray(points, dtype=np.float64)
            with torch.no_grad():
                x = torch.as_tensor(points, dtype=torch.float32, device=torch_device)
                values = module(x).detach().cpu().numpy()
            return _validate_batched_output(values, len(points))

        torch_evaluator._cmgdb_torch_device = torch_device
        torch_evaluator._cmgdb_width = _max_linear_width(module)
        return torch_evaluator

    def numpy_evaluator(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return _validate_batched_output(f(points), len(points))

    return numpy_evaluator

def _parse_slurm_mem_bytes() -> Optional[int]:
    for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        raw = os.environ.get(name)
        if not raw:
            continue
        try:
            return int(raw) * 1024 * 1024
        except ValueError:
            continue
    return None

def _available_memory_bytes(device: Any = None) -> Optional[int]:
    if device is not None and getattr(device, "type", None) == "cuda":
        torch = _import_torch(required=False)
        if torch is not None:
            try:
                free, _total = torch.cuda.mem_get_info(device)
                return int(free)
            except Exception:
                pass

    slurm_bytes = _parse_slurm_mem_bytes()
    if slurm_bytes is not None:
        return slurm_bytes

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        return None

def resolve_batch_points(
    batch_points: BatchPoints,
    *,
    n_total: int,
    input_dim: int = 1,
    evaluator_width: Optional[int] = None,
    device: Any = None,
) -> int:
    """Resolve ``batch_points`` to a concrete chunk size."""
    if isinstance(batch_points, int) and not isinstance(batch_points, bool):
        if batch_points <= 0:
            raise ValueError(
                f"batch_points must be positive when an int; got {batch_points}"
            )
        return max(1, min(int(batch_points), int(n_total)))
    if batch_points != "auto":
        raise ValueError(
            f"batch_points must be a positive int or 'auto'; got {batch_points!r}"
        )

    width = max(int(input_dim), int(evaluator_width or 1), 1)
    bytes_per_point = max(64, 8 * width)
    available = _available_memory_bytes(device)
    if available is None:
        budget = 1024 * 1024 * 1024
    else:
        budget = int(available * _AUTO_BATCH_MEMORY_FRACTION)

    if device is not None and getattr(device, "type", None) == "mps":
        budget = min(budget, _MPS_PER_CHUNK_BUDGET_BYTES)

    chunk = budget // bytes_per_point
    chunk = max(_AUTO_BATCH_MIN, min(int(chunk), _AUTO_BATCH_MAX))
    return int(max(1, min(chunk, int(n_total))))

def _validate_bounds(lower_bounds: Any, upper_bounds: Any) -> Tuple[np.ndarray, np.ndarray, int]:
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError("lower_bounds and upper_bounds must be one-dimensional")
    if lower.shape != upper.shape:
        raise ValueError("lower_bounds and upper_bounds must have the same shape")
    if lower.size == 0:
        raise ValueError("bounds must have at least one dimension")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be strictly less than its upper bound")
    return lower, upper, int(lower.size)

def precompute_corner_grid(
    f: Any,
    *,
    lower_bounds: Any,
    upper_bounds: Any,
    corners_per_axis: Union[int, Any],
    batch_points: BatchPoints = "auto",
    device: Any = "auto",
) -> Tuple[np.ndarray, int]:
    """Evaluate ``f`` on a product corner lattice in bounded chunks.

    ``corners_per_axis`` is either one count used for every axis, or a sequence
    giving one count per axis. The per-axis form exists because CMGDB bisects
    coordinate ``depth % dim`` at each depth, so at a subdivision level that is
    not a multiple of ``dim`` the axes are refined unequally.
    """
    lower, upper, dim = _validate_bounds(lower_bounds, upper_bounds)
    if np.isscalar(corners_per_axis):
        shape = (int(corners_per_axis),) * dim
    else:
        shape = tuple(int(c) for c in corners_per_axis)
        if len(shape) != dim:
            raise ValueError(
                f"corners_per_axis has {len(shape)} entries but dim={dim}"
            )
    if any(c < 1 for c in shape):
        raise ValueError(f"corners_per_axis must be positive; got {shape}")

    evaluator = as_batched_evaluator(f, device=device)
    evaluator_device = getattr(evaluator, "_cmgdb_torch_device", None)
    evaluator_width = getattr(evaluator, "_cmgdb_width", None)

    n_total = 1
    for c in shape:
        n_total *= c
    counts = np.asarray(shape, dtype=np.int64)
    step = np.zeros_like(upper - lower, dtype=np.float64)
    multi = counts > 1
    step[multi] = (upper - lower)[multi] / (counts[multi] - 1).astype(np.float64)

    chunk_size = resolve_batch_points(
        batch_points,
        n_total=n_total,
        input_dim=dim,
        evaluator_width=evaluator_width,
        device=evaluator_device,
    )
    ys_flat: Optional[np.ndarray] = None
    out_dim = -1

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        flat_idx = np.arange(start, end, dtype=np.int64)
        multi_idx = np.stack(np.unravel_index(flat_idx, shape), axis=-1).astype(np.float64)
        points = lower + multi_idx * step
        values = evaluator(points)
        if ys_flat is None:
            out_dim = int(values.shape[1])
            ys_flat = np.empty((n_total, out_dim), dtype=np.float64)
        if values.shape[1] != out_dim:
            raise ValueError(
                "batched evaluator output dimension changed between chunks: "
                f"expected {out_dim}, got {values.shape[1]}"
            )
        ys_flat[start:end] = values

    if ys_flat is None:
        raise RuntimeError("corner grid unexpectedly had no points")
    return ys_flat.reshape(shape + (out_dim,)), out_dim
