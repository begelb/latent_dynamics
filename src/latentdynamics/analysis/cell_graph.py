"""Cell graph computation: build the multivalued box map for a trained latent
map on a uniform grid, then compute per-box regions of attraction relative to
CMGDB's minimal Morse sets.

The pipeline:

1. ``UniformGrid`` — define a regular box decomposition of the latent space
   bounding box at resolution ``R`` per dim (``R**d`` boxes total).
2. ``evaluate_latent_corners`` — single batched forward pass of the trained
   ``latent_map`` on all ``(R+1)**d`` corner points.
3. ``box_image_bboxes`` — for each box, the image bbox is the min/max over
   its ``2**d`` corner images (an outer bound that's exact when the latent
   map is monotone per coordinate; otherwise a safe overestimate).
4. ``build_adjacency`` — vectorized: for each box, identify the grid boxes
   its image bbox overlaps. Boxes whose image escapes the grid have empty
   adjacency lists.
5. ``compute_box_roa`` — for each minimal Morse node, reverse-BFS in the
   adjacency graph from the grid boxes overlapping that Morse set. A transient
   box is in RoA(M) iff it can reach exactly one minimal Morse set's boxes.
   Grid boxes overlapping recurrent Morse sets keep their own Morse-node label;
   they are not assigned to the ROA of lower Morse sets.

Currently 2D-only. Higher-dim generalizes mechanically but is left for later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .regions_of_attraction import MorseGraph, load_box_roa


@dataclass(frozen=True)
class UniformGrid:
    bounds_lo: np.ndarray  # (dim,)
    bounds_hi: np.ndarray  # (dim,)
    resolution: int

    @property
    def dim(self) -> int:
        return int(self.bounds_lo.shape[0])

    @property
    def n_boxes(self) -> int:
        return int(self.resolution ** self.dim)

    @property
    def cell_size(self) -> np.ndarray:
        return (self.bounds_hi - self.bounds_lo) / self.resolution

    def box_index_2d(self, i: int, j: int) -> int:
        return i * self.resolution + j

    def box_lower_corners(self) -> np.ndarray:
        """``(n_boxes, dim)`` lower corners in row-major order."""
        d = self.dim
        cell = self.cell_size
        if d == 2:
            i = np.arange(self.resolution)
            xs = self.bounds_lo[0] + i * cell[0]
            ys = self.bounds_lo[1] + i * cell[1]
            xx, yy = np.meshgrid(xs, ys, indexing="ij")
            return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
        raise NotImplementedError("dim > 2 not yet supported")


def evaluate_latent_corners(
    latent_map: torch.nn.Module,
    grid: UniformGrid,
    *,
    device: str = "cpu",
    batch_size: int = 65536,
) -> np.ndarray:
    """Apply ``latent_map`` to every grid corner. Returns ``((R+1)**d, dim)``."""
    if grid.dim != 2:
        raise NotImplementedError("dim > 2 not yet supported")
    r = grid.resolution
    xs = np.linspace(grid.bounds_lo[0], grid.bounds_hi[0], r + 1)
    ys = np.linspace(grid.bounds_lo[1], grid.bounds_hi[1], r + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
    latent_map = latent_map.to(device).eval()
    out_chunks = []
    with torch.no_grad():
        for start in range(0, pts.shape[0], batch_size):
            chunk = torch.from_numpy(pts[start : start + batch_size]).to(device)
            out_chunks.append(latent_map(chunk).cpu().numpy())
    return np.concatenate(out_chunks, axis=0).astype(np.float64)


def box_image_bboxes_2d(corner_images: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Per box, return ``(image_lo, image_hi)`` arrays of shape ``(R**2, 2)``."""
    r = resolution
    imgs = corner_images.reshape(r + 1, r + 1, 2)
    c00 = imgs[:-1, :-1, :]
    c10 = imgs[1:, :-1, :]
    c01 = imgs[:-1, 1:, :]
    c11 = imgs[1:, 1:, :]
    stack = np.stack([c00, c10, c01, c11], axis=-2)
    lo = stack.min(axis=-2).reshape(-1, 2)
    hi = stack.max(axis=-2).reshape(-1, 2)
    return lo, hi


def build_adjacency_2d(
    image_lo: np.ndarray,
    image_hi: np.ndarray,
    grid: UniformGrid,
) -> list[np.ndarray]:
    """For each box, return the array of grid box indices its image overlaps.

    Vectorized: compute the (i0_min, i0_max, i1_min, i1_max) index window per
    box, then expand per box (the expansion loop is cheap because each box's
    window is small).
    """
    r = grid.resolution
    cell = grid.cell_size
    rel_lo = (image_lo - grid.bounds_lo) / cell
    rel_hi = (image_hi - grid.bounds_lo) / cell
    i0_min = np.clip(np.floor(rel_lo[:, 0]).astype(np.int64), 0, r - 1)
    i0_max = np.clip(np.ceil(rel_hi[:, 0]).astype(np.int64) - 1, 0, r - 1)
    i1_min = np.clip(np.floor(rel_lo[:, 1]).astype(np.int64), 0, r - 1)
    i1_max = np.clip(np.ceil(rel_hi[:, 1]).astype(np.int64) - 1, 0, r - 1)
    escapes = (
        (rel_hi[:, 0] < 0)
        | (rel_hi[:, 1] < 0)
        | (rel_lo[:, 0] > r)
        | (rel_lo[:, 1] > r)
    )
    adjacency: list[np.ndarray] = []
    for box_idx in range(image_lo.shape[0]):
        if escapes[box_idx]:
            adjacency.append(np.empty(0, dtype=np.int64))
            continue
        i_lo, i_hi = i0_min[box_idx], i0_max[box_idx]
        j_lo, j_hi = i1_min[box_idx], i1_max[box_idx]
        ii = np.arange(i_lo, i_hi + 1)
        jj = np.arange(j_lo, j_hi + 1)
        if ii.size == 0 or jj.size == 0:
            adjacency.append(np.empty(0, dtype=np.int64))
            continue
        I, J = np.meshgrid(ii, jj, indexing="ij")
        adjacency.append((I * r + J).reshape(-1))
    return adjacency


def build_reverse_csr(adjacency: list[np.ndarray], n_boxes: int) -> tuple[np.ndarray, np.ndarray]:
    """Compressed reverse adjacency: ``(ptr, neighbors)`` arrays."""
    counts = np.zeros(n_boxes + 1, dtype=np.int64)
    for adj in adjacency:
        for d in adj:
            counts[d + 1] += 1
    ptr = np.cumsum(counts)
    total = int(ptr[-1])
    neighbors = np.zeros(total, dtype=np.int64)
    fill = ptr[:-1].copy()
    for src, adj in enumerate(adjacency):
        for d in adj:
            neighbors[fill[d]] = src
            fill[d] += 1
    return ptr, neighbors


def reverse_reachable(
    rev_ptr: np.ndarray,
    rev_neighbors: np.ndarray,
    targets: np.ndarray,
    n_boxes: int,
    *,
    blocked: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean ``(n_boxes,)`` mask: True iff box can reach any of ``targets``."""
    visited = np.zeros(n_boxes, dtype=bool)
    stack = list(targets.tolist())
    for t in stack:
        visited[t] = True
    while stack:
        cur = stack.pop()
        start, end = rev_ptr[cur], rev_ptr[cur + 1]
        for k in range(start, end):
            pred = int(rev_neighbors[k])
            if blocked is not None and blocked[pred]:
                continue
            if not visited[pred]:
                visited[pred] = True
                stack.append(pred)
    return visited


def grid_boxes_overlapping_morse_set(
    grid: UniformGrid,
    morse_box_los: np.ndarray,
    morse_box_his: np.ndarray,
) -> np.ndarray:
    """Return grid box indices overlapping any of the given Morse-set boxes."""
    cell = grid.cell_size
    rel_lo = (morse_box_los - grid.bounds_lo) / cell
    rel_hi = (morse_box_his - grid.bounds_lo) / cell
    r = grid.resolution
    out: list[int] = []
    for k in range(morse_box_los.shape[0]):
        i_lo = max(0, int(np.floor(rel_lo[k, 0])))
        i_hi = min(r - 1, int(np.ceil(rel_hi[k, 0])) - 1)
        j_lo = max(0, int(np.floor(rel_lo[k, 1])))
        j_hi = min(r - 1, int(np.ceil(rel_hi[k, 1])) - 1)
        if i_hi < i_lo or j_hi < j_lo:
            continue
        for i in range(i_lo, i_hi + 1):
            for j in range(j_lo, j_hi + 1):
                out.append(i * r + j)
    return np.unique(np.asarray(out, dtype=np.int64)) if out else np.empty(0, dtype=np.int64)


def recurrent_grid_owners(
    grid: UniformGrid,
    morse_boxes,
    morse_nodes,
    nodes: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-grid-box recurrent owner and overlap-conflict mask.

    ``owner[i]`` is the Morse-node id whose saved recurrent boxes overlap grid
    box ``i``, or ``-1`` when no recurrent Morse set overlaps it. When multiple
    recurrent Morse sets overlap the same grid box, ``conflict[i]`` is true and
    the box is treated as a boundary for RoA traversal/plotting.
    """
    owner = np.full(grid.n_boxes, -1, dtype=np.int32)
    conflict = np.zeros(grid.n_boxes, dtype=bool)
    for node in sorted(nodes):
        rows = morse_boxes[morse_nodes == node]
        if rows.size == 0:
            continue
        node_lo = rows[:, :2]
        node_hi = rows[:, 2:4]
        node_idx = grid_boxes_overlapping_morse_set(grid, node_lo, node_hi)
        if node_idx.size == 0:
            continue
        already_owned = owner[node_idx] != -1
        conflict[node_idx[already_owned]] = True
        owner[node_idx[~already_owned]] = int(node)
    return owner, conflict


@dataclass
class CellGraphROA:
    grid: UniformGrid
    morse_graph: MorseGraph
    box_roa: np.ndarray
    """``(n_boxes,)`` int: ROA/recurrent label per grid box.
    Values are Morse-node ids for transient basin labels and recurrent boxes,
    or ``-1`` for boundary (reaches multiple minimal Morse sets), or ``-2`` for
    escape (no minimal Morse set reached)."""
    minimal_grid_boxes: dict[int, np.ndarray]
    """Per minimal Morse node, grid box indices overlapping its Morse set."""

    BOUNDARY = -1
    ESCAPE = -2


def compute_cell_graph_roa(
    latent_map: torch.nn.Module,
    morse_graph_dot: str | Path,
    morse_sets_csv: str | Path,
    *,
    resolution: int = 128,
    bounds_padding: float = 0.05,
    device: str = "cpu",
) -> CellGraphROA:
    """End-to-end: build cell graph and per-box RoA for the minimal Morse sets.

    The grid bounds default to the bounding box of all Morse-set boxes plus
    ``bounds_padding`` (fractional) on each side. Override by constructing
    ``UniformGrid`` directly and using the lower-level helpers.
    """
    table = load_box_roa(morse_graph_dot, morse_sets_csv)
    if table.dim != 2:
        raise NotImplementedError("cell-graph RoA is 2D-only for now")

    morse_lo = table.boxes[["lower_0", "lower_1"]].to_numpy()
    morse_hi = table.boxes[["upper_0", "upper_1"]].to_numpy()
    all_lo = morse_lo.min(axis=0)
    all_hi = morse_hi.max(axis=0)
    extent = all_hi - all_lo
    pad = bounds_padding * extent
    grid = UniformGrid(
        bounds_lo=all_lo - pad,
        bounds_hi=all_hi + pad,
        resolution=resolution,
    )

    corner_images = evaluate_latent_corners(latent_map, grid, device=device)
    image_lo, image_hi = box_image_bboxes_2d(corner_images, grid.resolution)
    adjacency = build_adjacency_2d(image_lo, image_hi, grid)
    rev_ptr, rev_neighbors = build_reverse_csr(adjacency, grid.n_boxes)

    mg = table.morse_graph
    morse_boxes = table.boxes[["lower_0", "lower_1", "upper_0", "upper_1"]].to_numpy()
    morse_nodes = table.boxes["morse_node"].to_numpy()
    recurrent_owner, recurrent_conflict = recurrent_grid_owners(
        grid,
        morse_boxes,
        morse_nodes,
        mg.nodes,
    )

    minimal_boxes: dict[int, np.ndarray] = {}
    reachable_masks: dict[int, np.ndarray] = {}
    for m in sorted(mg.minimal):
        rows = table.boxes[table.boxes["morse_node"] == m]
        m_lo = rows[["lower_0", "lower_1"]].to_numpy()
        m_hi = rows[["upper_0", "upper_1"]].to_numpy()
        m_idx = grid_boxes_overlapping_morse_set(grid, m_lo, m_hi)
        minimal_boxes[m] = m_idx
        blocked = (recurrent_owner != -1) & (recurrent_owner != m)
        blocked[recurrent_conflict] = True
        blocked[m_idx] = False
        reachable_masks[m] = reverse_reachable(
            rev_ptr,
            rev_neighbors,
            m_idx,
            grid.n_boxes,
            blocked=blocked,
        )

    # Pack the per-minimal-set masks into a bitmask per box so each box gets a
    # frozenset key cheaply.
    sorted_minimals = sorted(reachable_masks.keys())
    bitmask = np.zeros(grid.n_boxes, dtype=np.int64)
    for bit, m in enumerate(sorted_minimals):
        bitmask[reachable_masks[m]] |= np.int64(1) << bit

    # For each unique bitmask, decode the minimal set, resolve LCA via the
    # Morse graph, cache the result. A non-empty S with no LCA arises when
    # the Morse graph is a forest (disconnected components) and the cell
    # reaches minimals across components — flag as BOUNDARY, not ESCAPE.
    box_roa = np.full(grid.n_boxes, CellGraphROA.ESCAPE, dtype=np.int32)
    unique_keys = np.unique(bitmask)
    for key in unique_keys:
        if key == 0:
            continue  # genuine escape: no minimal Morse set reached
        S = frozenset(
            sorted_minimals[bit] for bit in range(len(sorted_minimals)) if (key >> bit) & 1
        )
        lca = mg.lca_of_minimals(S)
        if lca is None:
            box_roa[bitmask == key] = CellGraphROA.BOUNDARY
            continue
        box_roa[bitmask == key] = int(lca)

    # Recurrent Morse-set boxes are not transient RoA. Preserve their own
    # Morse-node label after the basin computation, including non-minimal
    # recurrent sets that can reach a lower attractor in the Morse graph.
    owned = recurrent_owner != -1
    box_roa[owned] = recurrent_owner[owned]
    box_roa[recurrent_conflict] = CellGraphROA.BOUNDARY

    return CellGraphROA(
        grid=grid,
        morse_graph=mg,
        box_roa=box_roa,
        minimal_grid_boxes=minimal_boxes,
    )
