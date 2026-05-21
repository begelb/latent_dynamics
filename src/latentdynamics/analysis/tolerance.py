"""Geometric primitives for Morse-set boundary analysis (paper Sec. 1.117).

Encapsulates the Box / Edge / MorseSet machinery and the tau-bar tolerance
computation that previously lived inside
``code/Leslie_analysis_scripts/Leslie3D_spurious_attractor_figure.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

EPS = 1e-9


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle in the latent plane belonging to one Morse set."""

    ID: int
    lower_x: float
    lower_y: float
    upper_x: float
    upper_y: float
    M_label: int


@dataclass(frozen=True)
class Edge:
    """Axis-aligned (or diagonal) segment between two corner vertices."""

    u: tuple[float, float]
    v: tuple[float, float]
    orientation: str

    @classmethod
    def make(cls, p1: tuple[float, float], p2: tuple[float, float]) -> Edge:
        u, v = (p1, p2) if p1 < p2 else (p2, p1)
        if abs(u[1] - v[1]) < EPS:
            orient = "horizontal"
        elif abs(u[0] - v[0]) < EPS:
            orient = "vertical"
        else:
            orient = "diagonal"
        return cls(u=u, v=v, orientation=orient)


class MorseSet:
    """Collection of boxes sharing one Morse label, loaded from a ``morse_sets`` CSV.

    The CSV columns are ``[lower_x, lower_y, upper_x, upper_y, label]`` for each box.
    """

    def __init__(self, file_path: str | Path, label: int) -> None:
        self.label = int(label)
        self.boxes: list[Box] = []
        self._load_from_file(Path(file_path))

    def _load_from_file(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        data = np.loadtxt(file_path, delimiter=",", ndmin=2)
        mask = np.isclose(data[:, 4], self.label)
        rows = data[mask]
        self.boxes = [
            Box(
                ID=i,
                lower_x=float(row[0]),
                lower_y=float(row[1]),
                upper_x=float(row[2]),
                upper_y=float(row[3]),
                M_label=int(row[4]),
            )
            for i, row in enumerate(rows)
        ]

    def __iter__(self):
        return iter(self.boxes)

    def __len__(self) -> int:
        return len(self.boxes)

    def boundary_edges(self) -> set[Edge]:
        """Edges that appear in exactly one box (i.e. the topological boundary)."""
        edges: set[Edge] = set()
        for box in self.boxes:
            corners = [
                (box.lower_x, box.lower_y),
                (box.upper_x, box.lower_y),
                (box.upper_x, box.upper_y),
                (box.lower_x, box.upper_y),
            ]
            for a, b in zip(corners, corners[1:] + corners[:1], strict=False):
                edge = Edge.make(a, b)
                if edge in edges:
                    edges.remove(edge)
                else:
                    edges.add(edge)
        return edges

    def vertices(self) -> NDArray[np.float64]:
        unique: set[tuple[float, float]] = set()
        for box in self.boxes:
            unique.update(
                {
                    (box.lower_x, box.lower_y),
                    (box.upper_x, box.lower_y),
                    (box.upper_x, box.upper_y),
                    (box.lower_x, box.upper_y),
                }
            )
        return np.asarray(sorted(unique), dtype=np.float64)


def is_in_range(point: tuple[float, float], edge: Edge) -> bool:
    px, py = point
    ux, uy = edge.u
    vx, vy = edge.v
    if edge.orientation == "horizontal":
        return min(ux, vx) - EPS <= px <= max(ux, vx) + EPS
    if edge.orientation == "vertical":
        return min(uy, vy) - EPS <= py <= max(uy, vy) + EPS
    return False


def orthogonal_distance(point: tuple[float, float], edge: Edge) -> float:
    px, py = point
    ux, uy = edge.u
    if edge.orientation == "horizontal":
        return abs(py - uy)
    if edge.orientation == "vertical":
        return abs(px - ux)
    return float("inf")


def _distance_point_to_segment(point: tuple[float, float], edge: Edge) -> float:
    """Euclidean distance from ``point`` to the finite segment ``edge``."""
    px, py = point
    ux, uy = edge.u
    vx, vy = edge.v
    dx = vx - ux
    dy = vy - uy
    length_sq = dx * dx + dy * dy
    if length_sq <= EPS:
        return float(np.hypot(px - ux, py - uy))
    t = ((px - ux) * dx + (py - uy) * dy) / length_sq
    t = min(1.0, max(0.0, t))
    closest_x = ux + t * dx
    closest_y = uy + t * dy
    return float(np.hypot(px - closest_x, py - closest_y))


def distance_point_to_boundary(point: tuple[float, float], boundary_edges: set[Edge]) -> float:
    """Minimum Euclidean distance from ``point`` to the Morse-set boundary."""
    best = float("inf")
    for edge in boundary_edges:
        d = _distance_point_to_segment(point, edge)
        if d < best:
            best = d
    return best


def compute_min_boundary_separation(
    morse_set: MorseSet,
    apply_dynamics: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    chunk_size: int = 4000,
) -> float:
    """Tau-bar bound: min distance from G(v) to the Morse-set boundary, over corners v.

    Axis-aligned boundary segments are clipped pointwise in NumPy, so the cost
    is :math:`O(VE)` floating-point ops but vectorized; chunking over ``V`` keeps
    the working set bounded.
    """
    edges = morse_set.boundary_edges()
    vertices = morse_set.vertices()
    if vertices.size == 0 or len(edges) == 0:
        return 0.0
    mapped = apply_dynamics(vertices)

    edge_count = len(edges)
    xmin = np.empty(edge_count)
    xmax = np.empty(edge_count)
    ymin = np.empty(edge_count)
    ymax = np.empty(edge_count)
    for i, e in enumerate(edges):
        ux, uy = e.u
        vx, vy = e.v
        xmin[i], xmax[i] = (ux, vx) if ux < vx else (vx, ux)
        ymin[i], ymax[i] = (uy, vy) if uy < vy else (vy, uy)

    px = np.asarray(mapped[:, 0], dtype=np.float64)
    py = np.asarray(mapped[:, 1], dtype=np.float64)
    best = float("inf")
    for i in range(0, px.shape[0], chunk_size):
        pxc = px[i : i + chunk_size, None]
        pyc = py[i : i + chunk_size, None]
        cx = np.clip(pxc, xmin[None, :], xmax[None, :])
        cy = np.clip(pyc, ymin[None, :], ymax[None, :])
        d = np.hypot(pxc - cx, pyc - cy).min(axis=1)
        if d.size:
            best = min(best, float(d.min()))
    return best


def compute_max_semiconjugacy_error(
    encoder: torch.nn.Module,
    latent_map: torch.nn.Module,
    points_in_block: NDArray[np.float64],
    next_points_true: NDArray[np.float64],
    *,
    device: torch.device | None = None,
) -> float:
    """Maximum error norm ``||E(f(x)) - G(E(x))||`` over a sample S.

    ``points_in_block`` are scaled samples in the high-dim space; ``next_points_true``
    are their scaled images under the ground-truth dynamics.
    """
    device = device or next(encoder.parameters()).device
    encoder.eval()
    latent_map.eval()
    with torch.no_grad():
        z_curr = encoder(torch.as_tensor(points_in_block, dtype=torch.float32, device=device))
        z_true = encoder(torch.as_tensor(next_points_true, dtype=torch.float32, device=device))
        z_pred = latent_map(z_curr)
        diff = z_true - z_pred
        norms = torch.linalg.vector_norm(diff, dim=1).cpu().numpy()
    if norms.size == 0:
        return 0.0
    return float(norms.max())
