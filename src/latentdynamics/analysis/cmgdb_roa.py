"""Exact regions of attraction on CMGDB's returned cell graph.

This module works on the ``MapGraph`` returned by CMGDB during the ``morse``
stage. Unlike the render-time diagnostic cell graph, these labels live on the
same CMGDB cell ids used by ``morse_graph.morse_set(node)``.
"""

from __future__ import annotations

__all__ = [
    "BOUNDARY",
    "ESCAPE",
    "EXACT_ROA_FILENAME",
    "MULTI",
    "CellROA",
    "attractor_cells",
    "collapse_roa_to_lca",
    "compute_and_save_exact_roa",
    "compute_exact_roa",
    "load_exact_roa",
    "save_exact_roa",
]

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .morse_graph_parser import MorseGraph

EXACT_ROA_FILENAME = "regions_of_attraction_exact.npz"
BOUNDARY = -1
ESCAPE = -2
# Sentinel for a cell that reverse-reaches more than one minimal Morse set
# when the LCA collapse is disabled. The full set lives in ``reach_mask``.
MULTI = -3


@dataclass(frozen=True)
class CellROA:
    """Per-CMGDB-cell RoA labels plus enough geometry metadata to render them.

    ``reach_mask`` and ``minimal_order`` expose the full reachable-minimal set
    per cell: bit ``i`` of ``reach_mask[c]`` is set iff minimal Morse node
    ``minimal_order[i]`` reverse-reaches cell ``c``. They are populated by
    :func:`compute_exact_roa` regardless of whether the LCA collapse ran, so a
    caller can recover the set or re-collapse later via
    :func:`collapse_roa_to_lca`.
    """

    box_roa: np.ndarray
    bounds_lower: np.ndarray | None = None
    bounds_upper: np.ndarray | None = None
    grid_shape: np.ndarray | None = None
    boxes: np.ndarray | None = None
    reach_mask: np.ndarray | None = None
    minimal_order: np.ndarray | None = None


def _morse_cell_sets(cmgdb_morse_graph, morse_dag: MorseGraph) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for node in morse_dag.nodes:
        cells = np.asarray(list(cmgdb_morse_graph.morse_set(node)), dtype=np.int64)
        out[int(node)] = np.unique(cells)
    return out


def _recurrent_owners(
    n_vertices: int,
    morse_cells: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    owner = np.full(n_vertices, ESCAPE, dtype=np.int32)
    conflict = np.zeros(n_vertices, dtype=bool)
    for node in sorted(morse_cells):
        cells = morse_cells[node]
        if cells.size == 0:
            continue
        already_owned = owner[cells] != ESCAPE
        conflict[cells[already_owned]] = True
        owner[cells[~already_owned]] = int(node)
    return owner, conflict


def _build_reverse_csr(map_graph, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """Build reverse adjacency arrays for ``map_graph``.

    CMGDB's Python binding exposes outgoing adjacency only. Two passes avoid
    storing a Python list of arrays for every cell, which is too expensive at
    fine 2-D subdivision levels.
    """
    counts = np.zeros(n_vertices + 1, dtype=np.int64)
    for src in range(n_vertices):
        for dst in map_graph.adjacencies(src):
            d = int(dst)
            if 0 <= d < n_vertices:
                counts[d + 1] += 1
    ptr = np.cumsum(counts)
    neighbors = np.empty(int(ptr[-1]), dtype=np.int64)
    fill = ptr[:-1].copy()
    for src in range(n_vertices):
        for dst in map_graph.adjacencies(src):
            d = int(dst)
            if 0 <= d < n_vertices:
                neighbors[fill[d]] = src
                fill[d] += 1
    return ptr, neighbors


def _reverse_reachable(
    rev_ptr: np.ndarray,
    rev_neighbors: np.ndarray,
    targets: np.ndarray,
    n_vertices: int,
    *,
    blocked: np.ndarray,
) -> np.ndarray:
    visited = np.zeros(n_vertices, dtype=bool)
    stack = [int(t) for t in targets.tolist() if 0 <= int(t) < n_vertices]
    for t in stack:
        visited[t] = True
    while stack:
        cur = stack.pop()
        start, end = int(rev_ptr[cur]), int(rev_ptr[cur + 1])
        for k in range(start, end):
            pred = int(rev_neighbors[k])
            if blocked[pred] or visited[pred]:
                continue
            visited[pred] = True
            stack.append(pred)
    return visited


def _infer_uniform_grid_shape(n_vertices: int, dim: int) -> np.ndarray | None:
    if dim <= 0:
        return None
    if dim == 1:
        return np.asarray([n_vertices], dtype=np.int64)
    per_axis = round(n_vertices ** (1.0 / dim))
    if per_axis > 0 and per_axis**dim == n_vertices:
        return np.full(dim, per_axis, dtype=np.int64)
    return None


def _collect_boxes(cmgdb_morse_graph, n_vertices: int, *, max_vertices: int) -> np.ndarray | None:
    if n_vertices > max_vertices:
        return None
    rows = [cmgdb_morse_graph.phase_space_box(i) for i in range(n_vertices)]
    return np.asarray(rows, dtype=np.float64)


def _collapse_multi(
    box_roa: np.ndarray,
    reach_mask: np.ndarray,
    minimal_order,
    morse_dag: MorseGraph,
) -> np.ndarray:
    """Replace :data:`MULTI` cells in ``box_roa`` (in place) with the Morse-poset
    LCA of their reachable-minimal set, falling back to :data:`BOUNDARY` when the
    DAG admits no LCA. Bit ``i`` of ``reach_mask`` corresponds to
    ``minimal_order[i]``. Cells that are not :data:`MULTI` are left untouched.
    """
    multi_cells = np.flatnonzero(box_roa == MULTI)
    if multi_cells.size == 0:
        return box_roa
    order = [int(m) for m in minimal_order]
    masks = reach_mask[multi_cells]
    for mv in np.unique(masks):
        reached = [m for i, m in enumerate(order) if int(mv) & (1 << i)]
        lca = morse_dag.lca_of_minimals(frozenset(reached))
        cells = multi_cells[masks == mv]
        box_roa[cells] = int(lca) if lca is not None else BOUNDARY
    return box_roa


def compute_exact_roa(
    map_graph,
    cmgdb_morse_graph,
    morse_dag: MorseGraph,
    *,
    lower_bounds: NDArray[np.float64] | None = None,
    upper_bounds: NDArray[np.float64] | None = None,
    max_box_geometry_vertices: int = 2_000_000,
    collapse_to_lca: bool = True,
) -> CellROA:
    """Compute exact cell labels on CMGDB's ``MapGraph``.

    Other recurrent Morse sets are blockers for each minimal target, so a
    spurious multivalued edge leaving a recurrent set does not pull that set,
    or its upstream transient structure, into a lower attractor's RoA.

    When ``collapse_to_lca`` is true (default), a cell that reverse-reaches
    several minima is labelled with the Morse-poset LCA of that set (the
    saddle/source governing the basin boundary). When false, such cells are
    left as :data:`MULTI`; the full reachable-minimal set is always available
    in the returned ``reach_mask``/``minimal_order`` and can be collapsed later
    via :func:`collapse_roa_to_lca`.
    """
    n_vertices = int(map_graph.num_vertices())
    morse_cells = _morse_cell_sets(cmgdb_morse_graph, morse_dag)
    recurrent_owner, recurrent_conflict = _recurrent_owners(n_vertices, morse_cells)
    rev_ptr, rev_neighbors = _build_reverse_csr(map_graph, n_vertices)

    # One bit per minimal Morse set: the bitmask records the full set of
    # minima that can reverse-reach each cell. With this we can resolve
    # "multi-basin" cells via the Morse-poset LCA instead of dropping them
    # to a generic BOUNDARY label.
    minimal_sorted = sorted(int(n) for n in morse_dag.minimal)
    if len(minimal_sorted) > 64:
        raise ValueError(
            f"compute_exact_roa: {len(minimal_sorted)} minimal Morse nodes "
            f"exceeds the 64-bit reach-mask width"
        )
    # Shift by a uint64, not a Python int: under NumPy 2's NEP 50 the int is a
    # weak scalar that adopts uint64, but under NumPy 1.x uint64 and the int's
    # int64 promote to float64, for which left_shift has no loop. Explicit
    # uint64 on both sides behaves identically on either.
    bit_of = {m: np.uint64(1) << np.uint64(i) for i, m in enumerate(minimal_sorted)}
    reach_mask = np.zeros(n_vertices, dtype=np.uint64)
    for minimal in minimal_sorted:
        targets = morse_cells.get(minimal, np.empty(0, dtype=np.int64))
        if targets.size == 0:
            continue
        blocked = (recurrent_owner != ESCAPE) & (recurrent_owner != minimal)
        blocked[recurrent_conflict] = True
        blocked[targets] = False
        reachable = _reverse_reachable(
            rev_ptr,
            rev_neighbors,
            targets,
            n_vertices,
            blocked=blocked,
        )
        reach_mask[reachable] |= bit_of[minimal]

    box_roa = np.full(n_vertices, ESCAPE, dtype=np.int32)
    # Single-basin cells: assign that minimum directly.
    for minimal in minimal_sorted:
        single = reach_mask == bit_of[minimal]
        box_roa[single] = minimal
    # Multi-basin cells: mark them MULTI; the full reachable-minimal set stays
    # in ``reach_mask``. The optional LCA collapse below (or a later
    # collapse_roa_to_lca call) turns MULTI into a single label.
    multi_mask_values = np.unique(reach_mask[~np.isin(reach_mask, [*bit_of.values(), np.uint64(0)])])
    if multi_mask_values.size:
        box_roa[np.isin(reach_mask, multi_mask_values)] = MULTI

    owned = recurrent_owner != ESCAPE
    box_roa[owned] = recurrent_owner[owned]
    box_roa[recurrent_conflict] = BOUNDARY

    if collapse_to_lca:
        _collapse_multi(box_roa, reach_mask, minimal_sorted, morse_dag)

    dim = len(lower_bounds) if lower_bounds is not None else None
    grid_shape = _infer_uniform_grid_shape(n_vertices, dim) if dim is not None else None
    boxes = None
    if grid_shape is None:
        boxes = _collect_boxes(
            cmgdb_morse_graph,
            n_vertices,
            max_vertices=max_box_geometry_vertices,
        )

    return CellROA(
        box_roa=box_roa,
        bounds_lower=None if lower_bounds is None else np.asarray(lower_bounds, dtype=np.float64),
        bounds_upper=None if upper_bounds is None else np.asarray(upper_bounds, dtype=np.float64),
        grid_shape=grid_shape,
        boxes=boxes,
        reach_mask=reach_mask,
        minimal_order=np.asarray(minimal_sorted, dtype=np.int32),
    )


def collapse_roa_to_lca(roa: CellROA, morse_dag: MorseGraph) -> np.ndarray:
    """Collapse :data:`MULTI` cells to the Morse-poset LCA of their minimal set.

    This is the optional post-processing that turns an uncollapsed
    :class:`CellROA` (``collapse_to_lca=False``) into the single-label array
    that :func:`compute_exact_roa` produces by default. Cells that are not
    :data:`MULTI` are returned unchanged.
    """
    if roa.reach_mask is None or roa.minimal_order is None:
        raise ValueError(
            "collapse_roa_to_lca requires reach_mask and minimal_order; "
            "recompute with compute_exact_roa to populate them"
        )
    out = np.array(roa.box_roa, dtype=np.int32, copy=True)
    return _collapse_multi(out, roa.reach_mask, roa.minimal_order, morse_dag)


def save_exact_roa(roa: CellROA, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / EXACT_ROA_FILENAME
    payload: dict[str, np.ndarray] = {
        "box_roa": np.asarray(roa.box_roa, dtype=np.int32),
        "bounds_lower": (
            np.asarray([], dtype=np.float64)
            if roa.bounds_lower is None
            else np.asarray(roa.bounds_lower, dtype=np.float64)
        ),
        "bounds_upper": (
            np.asarray([], dtype=np.float64)
            if roa.bounds_upper is None
            else np.asarray(roa.bounds_upper, dtype=np.float64)
        ),
        "grid_shape": (
            np.asarray([], dtype=np.int64)
            if roa.grid_shape is None
            else np.asarray(roa.grid_shape, dtype=np.int64)
        ),
    }
    if roa.boxes is not None:
        payload["boxes"] = np.asarray(roa.boxes, dtype=np.float64)
    if roa.reach_mask is not None:
        payload["reach_mask"] = np.asarray(roa.reach_mask, dtype=np.uint64)
    if roa.minimal_order is not None:
        payload["minimal_order"] = np.asarray(roa.minimal_order, dtype=np.int32)
    np.savez_compressed(path, **payload)
    return path


def load_exact_roa(path: str | Path) -> CellROA:
    with np.load(Path(path)) as data:
        bounds_lower = data["bounds_lower"]
        bounds_upper = data["bounds_upper"]
        grid_shape = data["grid_shape"]
        boxes = data["boxes"] if "boxes" in data.files else None
        reach_mask = data["reach_mask"] if "reach_mask" in data.files else None
        minimal_order = data["minimal_order"] if "minimal_order" in data.files else None
        return CellROA(
            box_roa=np.asarray(data["box_roa"], dtype=np.int32),
            bounds_lower=None if bounds_lower.size == 0 else np.asarray(bounds_lower, dtype=np.float64),
            bounds_upper=None if bounds_upper.size == 0 else np.asarray(bounds_upper, dtype=np.float64),
            grid_shape=None if grid_shape.size == 0 else np.asarray(grid_shape, dtype=np.int64),
            boxes=None if boxes is None else np.asarray(boxes, dtype=np.float64),
            reach_mask=None if reach_mask is None else np.asarray(reach_mask, dtype=np.uint64),
            minimal_order=None if minimal_order is None else np.asarray(minimal_order, dtype=np.int32),
        )


def compute_and_save_exact_roa(
    *,
    map_graph,
    cmgdb_morse_graph,
    morse_graph_dot: str | Path,
    out_dir: str | Path,
    lower_bounds: NDArray[np.float64] | None = None,
    upper_bounds: NDArray[np.float64] | None = None,
    max_vertices: int,
    collapse_to_lca: bool = True,
) -> Path:
    n_vertices = int(map_graph.num_vertices())
    if n_vertices > max_vertices:
        raise ValueError(
            f"exact RoA map graph has {n_vertices} vertices, exceeding "
            f"the max_vertices limit ({max_vertices})"
        )
    morse_dag = MorseGraph.from_dot(morse_graph_dot)
    roa = compute_exact_roa(
        map_graph,
        cmgdb_morse_graph,
        morse_dag,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_box_geometry_vertices=max_vertices,
        collapse_to_lca=collapse_to_lca,
    )
    return save_exact_roa(roa, out_dir)


def attractor_cells(map_graph, cmgdb_morse_graph, attractor_nodes) -> set[int]:
    """Forward-invariant cell set of an attractor.

    Starts from the union of the Morse-set cells of ``attractor_nodes`` and
    closes that set under the ``map_graph``'s forward adjacency. The result is
    the attractor's own cells (forward-invariant set), not its basin of
    attraction (which is the reverse-reachable set, see :func:`compute_exact_roa`).
    """
    cells: set[int] = set()
    for node in attractor_nodes:
        cells.update(int(c) for c in cmgdb_morse_graph.morse_set(node))
    queue = list(cells)
    while queue:
        c = queue.pop()
        for d in map_graph.adjacencies(c):
            d = int(d)
            if d not in cells:
                cells.add(d)
                queue.append(d)
    return cells
