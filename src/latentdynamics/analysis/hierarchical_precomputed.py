"""Hierarchical, lookup-only box maps for deep adaptive CMGDB runs.

The ordinary adaptive precomputed backend evaluates a map on the complete
corner lattice at ``subdiv_max``. That is appropriate in one and two
dimensions, but a three-dimensional level-33 lattice contains ``2049**3``
points. This module uses a dense corner lattice through a manageable coarse
level and dense fine corner blocks only inside coarse cells known to be
recurrent. After construction, CMGDB callbacks perform array lookup only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from .precomputed_box_map import (
    as_batched_evaluator,
    precompute_corner_grid,
    resolve_batch_points,
)
from numpy.typing import NDArray


@dataclass
class HierarchicalPrecomputedBoxMap:
    """Dense-coarse/sparse-fine lookup table for an isotropic CMGDB grid."""

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    coarse_subdiv: int
    fine_subdiv: int
    coarse_values: NDArray[np.float64]
    padding: bool = True
    active_coarse_indices: NDArray[np.int64] | None = None
    fine_block_values: NDArray[np.float64] | None = None
    _block_lookup: NDArray[np.int32] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)
        if self.lower.ndim != 1 or self.upper.shape != self.lower.shape:
            raise ValueError("lower and upper must be one-dimensional arrays with equal shape")
        if np.any(self.lower >= self.upper):
            raise ValueError("every lower bound must be strictly less than its upper bound")
        if self.coarse_subdiv < 1 or self.fine_subdiv < self.coarse_subdiv:
            raise ValueError("require 1 <= coarse_subdiv <= fine_subdiv")
        if self.coarse_subdiv % self.dim or self.fine_subdiv % self.dim:
            raise ValueError("coarse_subdiv and fine_subdiv must be divisible by dimension")

        expected = (self.coarse_cells_per_axis + 1,) * self.dim + (self.dim,)
        self.coarse_values = np.asarray(self.coarse_values, dtype=np.float64)
        if self.coarse_values.shape != expected:
            raise ValueError(
                f"coarse_values has shape {self.coarse_values.shape}; expected {expected}"
            )
        if (self.active_coarse_indices is None) != (self.fine_block_values is None):
            raise ValueError(
                "active_coarse_indices and fine_block_values must be supplied together"
            )
        if self.active_coarse_indices is not None:
            self._set_fine_tables(self.active_coarse_indices, self.fine_block_values)

    @property
    def dim(self) -> int:
        return int(self.lower.size)

    @property
    def coarse_axis_depth(self) -> int:
        return self.coarse_subdiv // self.dim

    @property
    def fine_axis_depth(self) -> int:
        return self.fine_subdiv // self.dim

    @property
    def coarse_cells_per_axis(self) -> int:
        return 2**self.coarse_axis_depth

    @property
    def fine_cells_per_axis(self) -> int:
        return 2**self.fine_axis_depth

    @property
    def fine_cells_per_coarse_axis(self) -> int:
        return 2 ** (self.fine_axis_depth - self.coarse_axis_depth)

    @property
    def coarse_side(self) -> NDArray[np.float64]:
        return (self.upper - self.lower) / self.coarse_cells_per_axis

    @property
    def fine_side(self) -> NDArray[np.float64]:
        return (self.upper - self.lower) / self.fine_cells_per_axis

    @classmethod
    def precompute_coarse(
        cls,
        latent_map: Any,
        *,
        lower: Any,
        upper: Any,
        coarse_subdiv: int,
        fine_subdiv: int,
        padding: bool = True,
        batch_points: int | str = "auto",
        device: Any = "auto",
    ) -> HierarchicalPrecomputedBoxMap:
        """Evaluate the complete coarse corner lattice in bounded batches."""

        lower_arr = np.asarray(lower, dtype=np.float64)
        upper_arr = np.asarray(upper, dtype=np.float64)
        if lower_arr.ndim != 1 or upper_arr.shape != lower_arr.shape:
            raise ValueError("lower and upper must be one-dimensional arrays with equal shape")
        dim = int(lower_arr.size)
        if coarse_subdiv % dim:
            raise ValueError("coarse_subdiv must be divisible by dimension")
        corners_per_axis = 2 ** (coarse_subdiv // dim) + 1
        values, out_dim = precompute_corner_grid(
            latent_map,
            lower_bounds=lower_arr,
            upper_bounds=upper_arr,
            corners_per_axis=corners_per_axis,
            batch_points=batch_points,
            device=device,
        )
        if out_dim != dim:
            raise ValueError(f"latent map returned dimension {out_dim}; expected {dim}")
        return cls(
            lower=lower_arr,
            upper=upper_arr,
            coarse_subdiv=coarse_subdiv,
            fine_subdiv=fine_subdiv,
            coarse_values=values,
            padding=padding,
        )

    def precompute_fine_blocks(
        self,
        latent_map: Any,
        coarse_boxes: Any,
        *,
        batch_points: int | str = "auto",
        device: Any = "auto",
    ) -> None:
        """Pre-evaluate fine corner blocks for selected uniform coarse cells."""

        boxes = np.asarray(coarse_boxes, dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 2 * self.dim:
            raise ValueError(
                f"coarse_boxes must have shape (n, {2 * self.dim}); got {boxes.shape}"
            )
        if boxes.shape[0] == 0:
            raise ValueError("at least one active coarse box is required")
        widths = boxes[:, self.dim :] - boxes[:, : self.dim]
        if not np.allclose(widths, self.coarse_side, rtol=1e-7, atol=1e-12):
            raise ValueError("every active box must be a uniform coarse-level cell")
        raw_indices = np.rint(
            (boxes[:, : self.dim] - self.lower) / self.coarse_side
        ).astype(np.int64)
        if np.any(raw_indices < 0) or np.any(raw_indices >= self.coarse_cells_per_axis):
            raise ValueError("an active coarse box lies outside the configured bounds")
        active = np.unique(raw_indices, axis=0)

        evaluator = as_batched_evaluator(latent_map, device=device)
        evaluator_device = getattr(evaluator, "_cmgdb_torch_device", None)
        evaluator_width = getattr(evaluator, "_cmgdb_width", None)
        refinement = self.fine_cells_per_coarse_axis
        local_corners = refinement + 1
        points_per_block = local_corners**self.dim
        n_total = int(active.shape[0]) * points_per_block
        chunk_size = resolve_batch_points(
            batch_points,
            n_total=n_total,
            input_dim=self.dim,
            evaluator_width=evaluator_width,
            device=evaluator_device,
        )

        values_flat: NDArray[np.float64] | None = None
        local_shape = (local_corners,) * self.dim
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            flat = np.arange(start, end, dtype=np.int64)
            block_ids = flat // points_per_block
            local_flat = flat % points_per_block
            local_indices = np.stack(
                np.unravel_index(local_flat, local_shape), axis=-1
            ).astype(np.int64)
            fine_indices = active[block_ids] * refinement + local_indices
            points = self.lower + fine_indices * self.fine_side
            values = evaluator(points)
            if values.shape[1] != self.dim:
                raise ValueError(
                    f"latent map returned dimension {values.shape[1]}; expected {self.dim}"
                )
            if values_flat is None:
                values_flat = np.empty((n_total, self.dim), dtype=np.float64)
            values_flat[start:end] = values

        if values_flat is None:
            raise RuntimeError("fine-block precomputation produced no values")
        fine_shape = (int(active.shape[0]), *local_shape, self.dim)
        self._set_fine_tables(active, values_flat.reshape(fine_shape))

    def _set_fine_tables(self, active: Any, values: Any) -> None:
        active_arr = np.asarray(active, dtype=np.int64)
        values_arr = np.asarray(values, dtype=np.float64)
        if active_arr.ndim != 2 or active_arr.shape[1] != self.dim:
            raise ValueError(
                f"active_coarse_indices must have shape (n, {self.dim}); "
                f"got {active_arr.shape}"
            )
        local_corners = self.fine_cells_per_coarse_axis + 1
        expected = (active_arr.shape[0],) + (local_corners,) * self.dim + (self.dim,)
        if values_arr.shape != expected:
            raise ValueError(f"fine_block_values has shape {values_arr.shape}; expected {expected}")
        if np.any(active_arr < 0) or np.any(active_arr >= self.coarse_cells_per_axis):
            raise ValueError("active coarse indices lie outside the coarse grid")
        if np.unique(active_arr, axis=0).shape[0] != active_arr.shape[0]:
            raise ValueError("active coarse indices must be unique")

        lookup_shape = (self.coarse_cells_per_axis,) * self.dim
        lookup = np.full(lookup_shape, -1, dtype=np.int32)
        lookup[tuple(active_arr[:, axis] for axis in range(self.dim))] = np.arange(
            active_arr.shape[0], dtype=np.int32
        )
        self.active_coarse_indices = active_arr
        self.fine_block_values = values_arr
        self._block_lookup = lookup

    def enable_lazy_fine_blocks(
        self,
        latent_map: Any,
        *,
        device: Any = "auto",
        max_points_per_eval: int = 1_000_000,
    ) -> None:
        """Fill fine blocks on demand instead of failing on unprepared cells.

        Opt-in: without this call the class keeps its strict lookup-only
        contract and raises on a fine query outside the prepared blocks. With
        it, the first query into an unprepared coarse cell evaluates that
        cell's complete fine corner block (batched, cached in the block
        store), so a single adaptive CMGDB run needs no preliminary pass to
        identify the recurrent cells. ``lazy_fill_stats`` records the work.
        """
        self._lazy_evaluator = as_batched_evaluator(latent_map, device=device)
        self._lazy_max_points = int(max_points_per_eval)
        self.lazy_fill_stats = {"blocks": 0, "points": 0, "seconds": 0.0}
        if self.active_coarse_indices is None:
            local_corners = self.fine_cells_per_coarse_axis + 1
            self._set_fine_tables(
                np.zeros((0, self.dim), dtype=np.int64),
                np.zeros((0,) + (local_corners,) * self.dim + (self.dim,), dtype=np.float64),
            )

    def _fill_fine_blocks(self, missing: NDArray[np.int64]) -> None:
        """Evaluate and append the fine blocks of the given coarse cells."""
        import time as _time

        started = _time.perf_counter()
        refinement = self.fine_cells_per_coarse_axis
        local_corners = refinement + 1
        local_shape = (local_corners,) * self.dim
        points_per_block = local_corners**self.dim
        local_indices = np.stack(
            np.unravel_index(np.arange(points_per_block), local_shape), axis=-1
        ).astype(np.int64)
        fine_indices = (
            missing[:, None, :] * refinement + local_indices[None, :, :]
        ).reshape(-1, self.dim)
        points = self.lower + fine_indices * self.fine_side
        values = np.empty((points.shape[0], self.dim), dtype=np.float64)
        step = max(points_per_block, self._lazy_max_points)
        for start in range(0, points.shape[0], step):
            values[start : start + step] = self._lazy_evaluator(points[start : start + step])
        blocks = values.reshape((missing.shape[0], *local_shape, self.dim))

        first_new = self.active_coarse_indices.shape[0]
        self.active_coarse_indices = np.concatenate([self.active_coarse_indices, missing])
        self.fine_block_values = np.concatenate([self.fine_block_values, blocks])
        self._block_lookup[tuple(missing[:, axis] for axis in range(self.dim))] = np.arange(
            first_new, first_new + missing.shape[0], dtype=np.int32
        )
        self.lazy_fill_stats["blocks"] += int(missing.shape[0])
        self.lazy_fill_stats["points"] += int(points.shape[0])
        self.lazy_fill_stats["seconds"] += _time.perf_counter() - started

    def __call__(self, rect: Any) -> list[float]:
        """Map one rectangle using precomputed corner values only."""

        return self.batch(np.asarray(rect, dtype=np.float64).reshape(1, -1))[0]

    def batch(self, rects: Any) -> list[list[float]]:
        """Map a rectangle batch with vectorized dense/sparse table lookup."""

        rect_arr = np.asarray(rects, dtype=np.float64)
        if rect_arr.ndim != 2 or rect_arr.shape[1] != 2 * self.dim:
            raise ValueError(
                f"rects must have shape (n, {2 * self.dim}); got {rect_arr.shape}"
            )
        if rect_arr.shape[0] == 0:
            return []
        widths = rect_arr[:, self.dim :] - rect_arr[:, : self.dim]
        if np.any(widths <= 0.0):
            raise ValueError("every rectangle must have positive width in every coordinate")
        if np.any(rect_arr[:, : self.dim] < self.lower - 1e-10) or np.any(
            rect_arr[:, self.dim :] > self.upper + 1e-10
        ):
            raise ValueError("rectangle lies outside the precomputed domain")

        result = np.empty((rect_arr.shape[0], 2 * self.dim), dtype=np.float64)
        coarse_mask = np.all(widths >= self.coarse_side * (1.0 - 1e-8), axis=1)
        if np.any(coarse_mask):
            result[coarse_mask] = self._map_coarse(rect_arr[coarse_mask])
        if np.any(~coarse_mask):
            result[~coarse_mask] = self._map_fine(rect_arr[~coarse_mask])
        return result.tolist()

    def _corner_bounds(
        self,
        values_for_combo,
        lower_indices: NDArray[np.int64],
        upper_indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        image_lower: NDArray[np.float64] | None = None
        image_upper: NDArray[np.float64] | None = None
        for combo in product((0, 1), repeat=self.dim):
            indices = np.where(np.asarray(combo, dtype=bool), upper_indices, lower_indices)
            values = values_for_combo(indices)
            image_lower = values if image_lower is None else np.minimum(image_lower, values)
            image_upper = values if image_upper is None else np.maximum(image_upper, values)
        assert image_lower is not None and image_upper is not None
        return image_lower, image_upper

    def _map_coarse(self, rects: NDArray[np.float64]) -> NDArray[np.float64]:
        lower_indices = np.rint(
            (rects[:, : self.dim] - self.lower) / self.coarse_side
        ).astype(np.int64)
        upper_indices = np.rint(
            (rects[:, self.dim :] - self.lower) / self.coarse_side
        ).astype(np.int64)

        def gather(indices: NDArray[np.int64]) -> NDArray[np.float64]:
            return self.coarse_values[
                tuple(indices[:, axis] for axis in range(self.dim))
            ]

        image_lower, image_upper = self._corner_bounds(gather, lower_indices, upper_indices)
        if self.padding:
            widths = rects[:, self.dim :] - rects[:, : self.dim]
            image_lower = image_lower - widths
            image_upper = image_upper + widths
        return np.concatenate((image_lower, image_upper), axis=1)

    def _fine_corner_values(
        self,
        global_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Look up fine-grid corners in any prepared adjacent coarse block.

        An internal coarse-grid boundary corner belongs to the cells on both
        sides of that boundary (and, in multiple dimensions, to every adjacent
        coarse-cell combination).  Prefer the block on the positive side, then
        deterministically try the remaining adjacent blocks until a prepared
        sparse block is found.
        """

        assert self.fine_block_values is not None and self._block_lookup is not None
        indices = np.asarray(global_indices, dtype=np.int64)
        if indices.ndim != 2 or indices.shape[1] != self.dim:
            raise ValueError(
                f"global fine indices must have shape (n, {self.dim}); "
                f"got {indices.shape}"
            )
        if np.any(indices < 0) or np.any(indices > self.fine_cells_per_axis):
            raise ValueError("fine corner lies outside the precomputed grid")

        refinement = self.fine_cells_per_coarse_axis
        primary = np.minimum(
            indices // refinement,
            self.coarse_cells_per_axis - 1,
        )
        internal_boundary = (
            (indices > 0)
            & (indices < self.fine_cells_per_axis)
            & (indices % refinement == 0)
        )
        block_ids = np.full(indices.shape[0], -1, dtype=np.int32)
        chosen_coarse = np.empty_like(primary)

        # product() starts with the all-positive-side candidate. A 1 bit means
        # use the negative-side cell along that boundary coordinate.
        for use_negative in product((0, 1), repeat=self.dim):
            unresolved = block_ids < 0
            if not np.any(unresolved):
                break
            candidate = primary.copy()
            eligible = unresolved.copy()
            for axis, choose_negative in enumerate(use_negative):
                if choose_negative:
                    eligible &= internal_boundary[:, axis]
                    candidate[:, axis] -= 1
            rows = np.flatnonzero(eligible)
            if rows.size == 0:
                continue
            candidate_ids = self._block_lookup[
                tuple(candidate[rows, axis] for axis in range(self.dim))
            ]
            prepared = candidate_ids >= 0
            selected = rows[prepared]
            if selected.size:
                block_ids[selected] = candidate_ids[prepared]
                chosen_coarse[selected] = candidate[selected]

        if np.any(block_ids < 0):
            if getattr(self, "_lazy_evaluator", None) is not None:
                unresolved = indices[block_ids < 0]
                missing_cells = np.unique(
                    np.minimum(
                        unresolved // refinement,
                        self.coarse_cells_per_axis - 1,
                    ),
                    axis=0,
                )
                self._fill_fine_blocks(missing_cells)
                return self._fine_corner_values(indices)
            missing = np.unique(indices[block_ids < 0], axis=0)
            raise KeyError(
                "unprepared coarse cell for fine corner (including every "
                f"adjacent block at a boundary): {missing[:10].tolist()}"
            )

        local_indices = indices - chosen_coarse * refinement
        if np.any(local_indices < 0) or np.any(local_indices > refinement):
            raise RuntimeError("resolved fine corner has an invalid block-local index")
        return self.fine_block_values[
            (block_ids, *(local_indices[:, axis] for axis in range(self.dim)))
        ]

    def _map_fine(self, rects: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.fine_block_values is None or self._block_lookup is None:
            raise RuntimeError("fine rectangle requested before fine blocks were precomputed")
        lower_global = np.rint(
            (rects[:, : self.dim] - self.lower) / self.fine_side
        ).astype(np.int64)
        upper_global = np.rint(
            (rects[:, self.dim :] - self.lower) / self.fine_side
        ).astype(np.int64)

        def gather(indices: NDArray[np.int64]) -> NDArray[np.float64]:
            return self._fine_corner_values(indices)

        image_lower, image_upper = self._corner_bounds(
            gather,
            lower_global,
            upper_global,
        )
        if self.padding:
            widths = rects[:, self.dim :] - rects[:, : self.dim]
            image_lower = image_lower - widths
            image_upper = image_upper + widths
        return np.concatenate((image_lower, image_upper), axis=1)

    def save(self, directory: str | Path) -> Path:
        """Persist tables without compression for fast reload."""

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "coarse_values.npy", self.coarse_values)
        if self.active_coarse_indices is not None and self.fine_block_values is not None:
            np.save(out / "active_coarse_indices.npy", self.active_coarse_indices)
            np.save(out / "fine_block_values.npy", self.fine_block_values)
        metadata = {
            "schema_version": 1,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "dimension": self.dim,
            "coarse_subdiv": self.coarse_subdiv,
            "fine_subdiv": self.fine_subdiv,
            "padding": bool(self.padding),
            "has_fine_blocks": self.fine_block_values is not None,
            "n_active_coarse_cells": (
                0
                if self.active_coarse_indices is None
                else int(self.active_coarse_indices.shape[0])
            ),
            "callback_neural_evaluations": 0,
        }
        path = out / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        mmap_mode: str | None = "r",
    ) -> HierarchicalPrecomputedBoxMap:
        """Reload persisted tables; the callback remains lookup-only."""

        source = Path(directory)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        active_path = source / "active_coarse_indices.npy"
        fine_path = source / "fine_block_values.npy"
        active = np.load(active_path, mmap_mode=mmap_mode) if active_path.exists() else None
        fine = np.load(fine_path, mmap_mode=mmap_mode) if fine_path.exists() else None
        return cls(
            lower=np.asarray(metadata["lower"], dtype=np.float64),
            upper=np.asarray(metadata["upper"], dtype=np.float64),
            coarse_subdiv=int(metadata["coarse_subdiv"]),
            fine_subdiv=int(metadata["fine_subdiv"]),
            coarse_values=np.load(source / "coarse_values.npy", mmap_mode=mmap_mode),
            padding=bool(metadata["padding"]),
            active_coarse_indices=active,
            fine_block_values=fine,
        )
