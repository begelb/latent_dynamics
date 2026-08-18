"""Pointwise basin classification and Chafee--Infante table statistics.

The exact CMGDB region-of-attraction artifact stores one label per cell.  This
module locates encoded points in those cells and keeps the geometric edge cases
explicit:

* a point in cells that all have the same unique-attractor label gets that
  label;
* a point shared by cells with disagreeing labels is :data:`BOUNDARY`;
* a point in an unassigned/escaping cell is :data:`ESCAPE`; and
* a point outside every stored cell is :data:`OUTSIDE`.

Uniform CMGDB ``TreeGrid`` artifacts are handled without materializing or
scanning their boxes.  Their cell ids are the depth-first binary path ids,
equivalently Morton/Z-order ids with coordinate bits interleaved most
significant bit first in axis order.  Explicit adaptive boxes use the CMGDB
layout ``[lower_0, ..., lower_d-1, upper_0, ..., upper_d-1]`` and a sweep-axis
point lookup rather than an ``O(points * cells)`` membership matrix.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cmgdb_roa import BOUNDARY, ESCAPE, MULTI, CellROA

__all__ = [
    "OUTSIDE",
    "CellIndexLookup",
    "ChafeeBasinStatistics",
    "PointBasinClassification",
    "classify_points_in_cell_roa",
    "cmgdb_morton_cell_indices",
    "compute_chafee_basin_statistics",
]

# Point did not intersect any cell geometry.  The other negative values are
# already part of the CellROA artifact format.
OUTSIDE = -4

CellIndexLookup: TypeAlias = Callable[
    [NDArray[np.int64], NDArray[np.int64]],
    NDArray[np.int64],
]


@dataclass(frozen=True)
class PointBasinClassification:
    """Unique-basin label and geometry diagnostics for each point.

    ``basin_labels`` contains a nonnegative CMGDB Morse-node id only when every
    cell containing the point agrees on that unique attractor.  Negative labels
    use :data:`BOUNDARY`, :data:`ESCAPE`, or :data:`OUTSIDE`.
    """

    basin_labels: NDArray[np.int32]
    matched_cell_counts: NDArray[np.int32]
    attractor_labels: NDArray[np.int32]
    geometry_mode: str

    @property
    def n_points(self) -> int:
        return int(self.basin_labels.size)

    @property
    def uniquely_classified_mask(self) -> NDArray[np.bool_]:
        return np.isin(self.basin_labels, self.attractor_labels)

    @property
    def boundary_mask(self) -> NDArray[np.bool_]:
        return self.basin_labels == BOUNDARY

    @property
    def escape_mask(self) -> NDArray[np.bool_]:
        return self.basin_labels == ESCAPE

    @property
    def outside_mask(self) -> NDArray[np.bool_]:
        return self.basin_labels == OUTSIDE


_CHAFEE_COUNT_FIELDS = (
    "outside_both_basins",
    "misclassified_in_negative_basin",
    "misclassified_in_positive_basin",
    "correctly_classified_in_negative_basin",
    "correctly_classified_in_positive_basin",
)


@dataclass(frozen=True)
class ChafeeBasinStatistics:
    """Counts underlying the Chafee--Infante basin-classification table.

    The five table counts are conditioned on ``trajectory_labels != 0``.
    ``excluded_zero_trajectories`` records the trajectories omitted by that
    conditioning, matching the archived reference computation.
    """

    total_trajectories: int
    excluded_zero_trajectories: int
    conditioned_trajectories: int
    outside_both_basins: int
    misclassified_in_negative_basin: int
    misclassified_in_positive_basin: int
    correctly_classified_in_negative_basin: int
    correctly_classified_in_positive_basin: int

    def __post_init__(self) -> None:
        self.validate_count_conservation()

    def validate_count_conservation(self) -> None:
        """Raise if either the full or conditioned counts fail to conserve."""
        values = (
            self.total_trajectories,
            self.excluded_zero_trajectories,
            self.conditioned_trajectories,
            *(getattr(self, field) for field in _CHAFEE_COUNT_FIELDS),
        )
        if any(value < 0 for value in values):
            raise ValueError("Chafee basin counts must be nonnegative")
        if (
            self.excluded_zero_trajectories + self.conditioned_trajectories
            != self.total_trajectories
        ):
            raise ValueError(
                "Chafee basin counts do not conserve total trajectories: "
                "excluded_zero_trajectories + conditioned_trajectories "
                "!= total_trajectories"
            )
        table_total = sum(getattr(self, field) for field in _CHAFEE_COUNT_FIELDS)
        if table_total != self.conditioned_trajectories:
            raise ValueError(
                "Chafee table counts do not conserve conditioned trajectories: "
                f"{table_total} != {self.conditioned_trajectories}"
            )

    def counts(self) -> dict[str, int]:
        """Return the five table rows in manuscript order."""
        return {field: int(getattr(self, field)) for field in _CHAFEE_COUNT_FIELDS}

    def percentages(self) -> dict[str, float]:
        """Return table percentages using nonzero trajectories as denominator."""
        if self.conditioned_trajectories == 0:
            raise ValueError("percentages are undefined with no nonzero trajectories")
        denominator = float(self.conditioned_trajectories)
        return {
            field: 100.0 * float(getattr(self, field)) / denominator
            for field in _CHAFEE_COUNT_FIELDS
        }


def _integer_array(values: ArrayLike, *, name: str, ndim: int) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {raw.shape}")
    if np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must contain integers, not booleans")
    if not np.issubdtype(raw.dtype, np.integer):
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain integers") from exc
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError(f"{name} must contain integers")
    return np.asarray(raw, dtype=np.int64)


def _cyclic_tree_bit_counts(grid_shape: NDArray[np.int64]) -> NDArray[np.int64]:
    bit_counts = np.empty(grid_shape.size, dtype=np.int64)
    for axis, size_value in enumerate(grid_shape.tolist()):
        size = int(size_value)
        if size <= 0 or size & (size - 1):
            raise ValueError(
                "CMGDB Morton grid_shape entries must be positive powers of two; "
                f"axis {axis} has {size}"
            )
        bit_counts[axis] = size.bit_length() - 1

    total_depth = int(bit_counts.sum())
    quotient, remainder = divmod(total_depth, int(grid_shape.size))
    expected = np.asarray(
        [quotient + (axis < remainder) for axis in range(grid_shape.size)],
        dtype=np.int64,
    )
    if not np.array_equal(bit_counts, expected):
        raise ValueError(
            "grid_shape is incompatible with CMGDB's cyclic TreeGrid splits; "
            f"got bits per axis {bit_counts.tolist()}, expected {expected.tolist()}"
        )
    if total_depth > 62:
        raise ValueError(
            "CMGDB Morton cell ids exceed the supported signed 64-bit range "
            f"at tree depth {total_depth}"
        )
    return bit_counts


def cmgdb_morton_cell_indices(
    bin_indices: ArrayLike,
    grid_shape: ArrayLike,
) -> NDArray[np.int64]:
    """Map coordinate-bin indices to uniform CMGDB ``TreeGrid`` cell ids.

    CMGDB recursively splits axis ``depth % dimension`` and numbers leaves in
    depth-first left-to-right order.  On a uniform tree, the leaf id is the
    binary root-to-leaf path: each coordinate's bin bits are interleaved
    most-significant first in axis order.
    """
    bins = _integer_array(bin_indices, name="bin_indices", ndim=2)
    shape = _integer_array(grid_shape, name="grid_shape", ndim=1)
    if shape.size == 0:
        raise ValueError("grid_shape must contain at least one axis")
    if bins.shape[1] != shape.size:
        raise ValueError(
            f"bin_indices dimension {bins.shape[1]} does not match "
            f"grid_shape dimension {shape.size}"
        )
    bit_counts = _cyclic_tree_bit_counts(shape)
    if np.any(bins < 0) or np.any(bins >= shape[None, :]):
        raise ValueError("bin_indices contain coordinates outside grid_shape")

    cell_ids = np.zeros(bins.shape[0], dtype=np.int64)
    total_depth = int(bit_counts.sum())
    dimension = int(shape.size)
    for depth in range(total_depth):
        axis = depth % dimension
        bit_ordinal = depth // dimension
        shift = int(bit_counts[axis]) - 1 - bit_ordinal
        cell_ids <<= 1
        cell_ids |= (bins[:, axis] >> shift) & 1
    return cell_ids


def _resolve_attractor_labels(
    roa: CellROA,
    attractor_labels: Iterable[int] | None,
) -> NDArray[np.int32]:
    if attractor_labels is None:
        if roa.minimal_order is None:
            raise ValueError(
                "attractor_labels are required when CellROA.minimal_order is absent"
            )
        raw = roa.minimal_order
    else:
        raw = list(attractor_labels)
    labels = _integer_array(raw, name="attractor_labels", ndim=1)
    if labels.size == 0:
        raise ValueError("attractor_labels must contain at least one label")
    if np.any(labels < 0):
        raise ValueError("attractor_labels must be nonnegative CMGDB Morse-node ids")
    labels = np.unique(labels)
    return np.asarray(labels, dtype=np.int32)


def _validate_cell_labels(roa: CellROA) -> NDArray[np.int64]:
    labels = _integer_array(roa.box_roa, name="CellROA.box_roa", ndim=1)
    invalid_negative = (labels < 0) & ~np.isin(labels, [BOUNDARY, ESCAPE, MULTI])
    if np.any(invalid_negative):
        invalid = np.unique(labels[invalid_negative]).tolist()
        raise ValueError(f"CellROA.box_roa has unknown negative labels: {invalid}")
    return labels


def _reduce_labels(
    cell_labels: NDArray[np.int64],
    attractor_set: set[int],
) -> int:
    unique = np.unique(cell_labels)
    basin_labels = [int(label) for label in unique if int(label) in attractor_set]
    boundary_like = any(
        int(label) not in attractor_set and int(label) != ESCAPE for label in unique
    )
    has_escape = bool(np.any(unique == ESCAPE))
    if len(basin_labels) == 1 and not boundary_like and not has_escape:
        return basin_labels[0]
    if not basin_labels and not boundary_like and has_escape:
        return ESCAPE
    return BOUNDARY


def _run_cell_indexer(
    bins: NDArray[np.int64],
    shape: NDArray[np.int64],
    *,
    cell_indexer: CellIndexLookup,
    n_cells: int,
) -> NDArray[np.int64]:
    cell_ids = _integer_array(
        cell_indexer(bins, shape),
        name="cell_indexer result",
        ndim=1,
    )
    if cell_ids.shape != (bins.shape[0],):
        raise ValueError(
            "cell_indexer must return one cell id per bin row; "
            f"got shape {cell_ids.shape} for {bins.shape[0]} rows"
        )
    if np.any(cell_ids < 0) or np.any(cell_ids >= n_cells):
        raise ValueError(f"cell_indexer returned an id outside [0, {n_cells})")
    return cell_ids


def _classify_uniform_grid(
    points: NDArray[np.float64],
    cell_labels: NDArray[np.int64],
    attractor_labels: NDArray[np.int32],
    *,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    shape: NDArray[np.int64],
    boundary_atol: float,
    cell_indexer: CellIndexLookup,
    geometry_mode: str,
) -> PointBasinClassification:
    n_points, dimension = points.shape
    n_cells = int(cell_labels.size)
    labels = np.full(n_points, OUTSIDE, dtype=np.int32)
    matched_counts = np.zeros(n_points, dtype=np.int32)
    if n_points == 0:
        return PointBasinClassification(
            labels,
            matched_counts,
            attractor_labels,
            geometry_mode,
        )

    outside = np.any(
        (points < lower[None, :] - boundary_atol)
        | (points > upper[None, :] + boundary_atol),
        axis=1,
    )
    inside_indices = np.flatnonzero(~outside)
    if inside_indices.size == 0:
        return PointBasinClassification(
            labels,
            matched_counts,
            attractor_labels,
            geometry_mode,
        )

    clipped = np.clip(points[inside_indices], lower, upper)
    widths = upper - lower
    scaled = (clipped - lower) / widths * shape
    base_bins = np.floor(scaled).astype(np.int64)
    base_bins = np.minimum(base_bins, shape - 1)

    nearest = np.rint(scaled)
    scaled_atol = boundary_atol / widths * shape
    internal_boundary = (
        (nearest > 0)
        & (nearest < shape)
        & (np.abs(scaled - nearest) <= scaled_atol)
    )
    ordinary_rows = ~np.any(internal_boundary, axis=1)
    ordinary_points = inside_indices[ordinary_rows]
    if ordinary_points.size:
        cell_ids = _run_cell_indexer(
            base_bins[ordinary_rows],
            shape,
            cell_indexer=cell_indexer,
            n_cells=n_cells,
        )
        raw = cell_labels[cell_ids]
        labels[ordinary_points] = BOUNDARY
        labels[ordinary_points[raw == ESCAPE]] = ESCAPE
        for attractor in attractor_labels.tolist():
            labels[ordinary_points[raw == attractor]] = int(attractor)
        matched_counts[ordinary_points] = 1

    attractor_set = {int(label) for label in attractor_labels.tolist()}
    boundary_rows = np.flatnonzero(~ordinary_rows)
    for row in boundary_rows.tolist():
        choices: list[tuple[int, ...]] = []
        for axis in range(dimension):
            if internal_boundary[row, axis]:
                boundary_bin = int(nearest[row, axis])
                choices.append((boundary_bin - 1, boundary_bin))
            else:
                choices.append((int(base_bins[row, axis]),))
        bins = np.asarray(list(product(*choices)), dtype=np.int64)
        cell_ids = np.unique(
            _run_cell_indexer(
                bins,
                shape,
                cell_indexer=cell_indexer,
                n_cells=n_cells,
            )
        )
        point_index = int(inside_indices[row])
        matched_counts[point_index] = int(cell_ids.size)
        labels[point_index] = _reduce_labels(cell_labels[cell_ids], attractor_set)

    return PointBasinClassification(
        labels,
        matched_counts,
        attractor_labels,
        geometry_mode,
    )


def _classify_explicit_boxes(
    points: NDArray[np.float64],
    cell_labels: NDArray[np.int64],
    attractor_labels: NDArray[np.int32],
    boxes: NDArray[np.float64],
    *,
    boundary_atol: float,
) -> PointBasinClassification:
    n_points, dimension = points.shape
    lower = boxes[:, :dimension]
    upper = boxes[:, dimension:]
    if not np.all(np.isfinite(boxes)):
        raise ValueError("CellROA.boxes must contain only finite bounds")
    if np.any(lower > upper):
        raise ValueError("CellROA.boxes contains a lower bound above an upper bound")

    labels = np.full(n_points, OUTSIDE, dtype=np.int32)
    matched_counts = np.zeros(n_points, dtype=np.int32)
    if n_points == 0 or boxes.shape[0] == 0:
        return PointBasinClassification(
            labels,
            matched_counts,
            attractor_labels,
            "explicit_boxes",
        )

    # Choose the coordinate whose interval sweep produces the fewest candidate
    # point-box pairs.  The remaining coordinates are checked only for those
    # candidates, so a regular 16.8M-cell grid is never expanded into an
    # O(points*cells) boolean matrix.
    sweep_data = []
    candidate_totals = []
    for axis in range(dimension):
        order = np.argsort(points[:, axis], kind="stable")
        values = points[order, axis]
        starts = np.searchsorted(
            values,
            lower[:, axis] - boundary_atol,
            side="left",
        )
        stops = np.searchsorted(
            values,
            upper[:, axis] + boundary_atol,
            side="right",
        )
        sweep_data.append((order, starts, stops))
        candidate_totals.append(int(np.sum(stops - starts, dtype=np.int64)))
    sweep_axis = int(np.argmin(candidate_totals))
    order, starts, stops = sweep_data[sweep_axis]

    candidate_attractor = np.full(n_points, OUTSIDE, dtype=np.int32)
    has_escape = np.zeros(n_points, dtype=bool)
    has_boundary = np.zeros(n_points, dtype=bool)
    has_conflict = np.zeros(n_points, dtype=bool)
    attractor_set = {int(label) for label in attractor_labels.tolist()}

    for cell_id, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        if start == stop:
            continue
        candidates = order[int(start) : int(stop)]
        candidate_points = points[candidates]
        inside = np.all(
            (candidate_points >= lower[cell_id] - boundary_atol)
            & (candidate_points <= upper[cell_id] + boundary_atol),
            axis=1,
        )
        matched = candidates[inside]
        if matched.size == 0:
            continue
        matched_counts[matched] += 1
        raw_label = int(cell_labels[cell_id])
        if raw_label in attractor_set:
            unseen = candidate_attractor[matched] == OUTSIDE
            candidate_attractor[matched[unseen]] = raw_label
            seen = matched[~unseen]
            has_conflict[seen] |= candidate_attractor[seen] != raw_label
        elif raw_label == ESCAPE:
            has_escape[matched] = True
        else:
            has_boundary[matched] = True

    matched = matched_counts > 0
    has_attractor = candidate_attractor != OUTSIDE
    unique_basin = (
        matched
        & has_attractor
        & ~has_escape
        & ~has_boundary
        & ~has_conflict
    )
    labels[unique_basin] = candidate_attractor[unique_basin]
    only_escape = matched & ~has_attractor & has_escape & ~has_boundary
    labels[only_escape] = ESCAPE
    boundary = matched & ~unique_basin & ~only_escape
    labels[boundary] = BOUNDARY

    return PointBasinClassification(
        labels,
        matched_counts,
        attractor_labels,
        "explicit_boxes",
    )


def classify_points_in_cell_roa(
    points: ArrayLike,
    roa: CellROA,
    *,
    attractor_labels: Iterable[int] | None = None,
    boundary_atol: float = 0.0,
    uniform_grid_order: str = "cmgdb_morton",
    cell_indexer: CellIndexLookup | None = None,
) -> PointBasinClassification:
    """Classify ``(n, d)`` points against a :class:`CellROA`.

    Uniform artifacts with ``bounds_lower``, ``bounds_upper``, and
    ``grid_shape`` use direct coordinate-to-cell lookup.  The default
    ``uniform_grid_order="cmgdb_morton"`` matches CMGDB's uniform ``TreeGrid``
    leaf numbering.  A caller working with a differently ordered artifact can
    pass ``cell_indexer(bin_indices, grid_shape)``; the callback must return one
    flat cell id per bin-index row.

    If uniform metadata is unavailable, ``roa.boxes`` is required in CMGDB's
    ``lower[d], upper[d]`` layout.  Boxes are closed.  Therefore a point on a
    shared face checks every touching cell: agreeing unique-attractor labels
    remain classified, while disagreement becomes :data:`BOUNDARY`.  Points
    matching no box are :data:`OUTSIDE`.
    """
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2:
        raise ValueError(f"points must have shape (n, d), got {point_array.shape}")
    if point_array.shape[1] == 0:
        raise ValueError("points must have at least one coordinate")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must contain only finite coordinates")
    if not np.isfinite(boundary_atol) or boundary_atol < 0:
        raise ValueError("boundary_atol must be a finite nonnegative number")

    cell_labels = _validate_cell_labels(roa)
    unique_attractors = _resolve_attractor_labels(roa, attractor_labels)
    dimension = int(point_array.shape[1])

    has_uniform_geometry = all(
        value is not None
        for value in (roa.bounds_lower, roa.bounds_upper, roa.grid_shape)
    )
    if has_uniform_geometry:
        lower = np.asarray(roa.bounds_lower, dtype=np.float64)
        upper = np.asarray(roa.bounds_upper, dtype=np.float64)
        shape = _integer_array(roa.grid_shape, name="CellROA.grid_shape", ndim=1)
        if lower.shape != (dimension,) or upper.shape != (dimension,):
            raise ValueError(
                "CellROA bounds dimension does not match points: "
                f"{lower.shape}, {upper.shape} versus d={dimension}"
            )
        if shape.shape != (dimension,):
            raise ValueError(
                f"CellROA.grid_shape has shape {shape.shape}, expected ({dimension},)"
            )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("CellROA bounds must contain only finite values")
        if np.any(lower >= upper):
            raise ValueError("CellROA bounds must satisfy lower < upper on every axis")
        if np.any(shape <= 0):
            raise ValueError("CellROA.grid_shape entries must be positive")
        expected_cells = prod(int(value) for value in shape.tolist())
        if expected_cells != cell_labels.size:
            raise ValueError(
                f"CellROA.grid_shape describes {expected_cells} cells but "
                f"box_roa has {cell_labels.size} labels"
            )

        if cell_indexer is None:
            if uniform_grid_order != "cmgdb_morton":
                raise ValueError(
                    "unsupported uniform_grid_order "
                    f"{uniform_grid_order!r}; use 'cmgdb_morton' or pass cell_indexer"
                )
            _cyclic_tree_bit_counts(shape)
            selected_indexer = cmgdb_morton_cell_indices
            geometry_mode = "uniform_cmgdb_morton"
        else:
            selected_indexer = cell_indexer
            geometry_mode = "uniform_custom_indexer"
        return _classify_uniform_grid(
            point_array,
            cell_labels,
            unique_attractors,
            lower=lower,
            upper=upper,
            shape=shape,
            boundary_atol=float(boundary_atol),
            cell_indexer=selected_indexer,
            geometry_mode=geometry_mode,
        )

    if roa.boxes is None:
        raise ValueError(
            "CellROA needs complete uniform metadata "
            "(bounds_lower, bounds_upper, grid_shape) or explicit boxes"
        )
    boxes = np.asarray(roa.boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 2 * dimension:
        raise ValueError(
            "CellROA.boxes must have shape (n_cells, 2*d) in lower[d], upper[d] "
            f"layout; got {boxes.shape} for d={dimension}"
        )
    if boxes.shape[0] != cell_labels.size:
        raise ValueError(
            f"CellROA.boxes has {boxes.shape[0]} rows but box_roa has "
            f"{cell_labels.size} labels"
        )
    return _classify_explicit_boxes(
        point_array,
        cell_labels,
        unique_attractors,
        boxes,
        boundary_atol=float(boundary_atol),
    )


def compute_chafee_basin_statistics(
    trajectory_labels: ArrayLike,
    point_basin_labels: ArrayLike | PointBasinClassification,
    *,
    negative_basin_label: int,
    positive_basin_label: int,
) -> ChafeeBasinStatistics:
    """Compute the reference Chafee basin counts.

    ``trajectory_labels`` must use ``-1`` for the negative steady state, ``1``
    for the positive steady state, and ``0`` for a trajectory excluded from the
    table.  All five table rows use the number of nonzero labels as their
    denominator.  Any point label other than the two supplied basin labels is
    counted as outside both basins.
    """
    truth = _integer_array(trajectory_labels, name="trajectory_labels", ndim=1)
    if isinstance(point_basin_labels, PointBasinClassification):
        predicted = np.asarray(point_basin_labels.basin_labels, dtype=np.int64)
    else:
        predicted = _integer_array(
            point_basin_labels,
            name="point_basin_labels",
            ndim=1,
        )
    if predicted.shape != truth.shape:
        raise ValueError(
            "trajectory_labels and point_basin_labels must have the same shape; "
            f"got {truth.shape} and {predicted.shape}"
        )
    invalid_truth = ~np.isin(truth, [-1, 0, 1])
    if np.any(invalid_truth):
        invalid = np.unique(truth[invalid_truth]).tolist()
        raise ValueError(
            "trajectory_labels must use only -1, 0, and 1; "
            f"found {invalid}"
        )

    negative_label = int(negative_basin_label)
    positive_label = int(positive_basin_label)
    if negative_label < 0 or positive_label < 0:
        raise ValueError("negative_basin_label and positive_basin_label must be nonnegative")
    if negative_label == positive_label:
        raise ValueError("negative and positive basin labels must be distinct")

    conditioned = truth != 0
    conditioned_total = int(np.count_nonzero(conditioned))
    negative_truth = truth == -1
    positive_truth = truth == 1
    in_negative = predicted == negative_label
    in_positive = predicted == positive_label
    in_neither = ~(in_negative | in_positive)

    result = ChafeeBasinStatistics(
        total_trajectories=int(truth.size),
        excluded_zero_trajectories=int(np.count_nonzero(~conditioned)),
        conditioned_trajectories=conditioned_total,
        outside_both_basins=int(np.count_nonzero(conditioned & in_neither)),
        misclassified_in_negative_basin=int(
            np.count_nonzero(positive_truth & in_negative)
        ),
        misclassified_in_positive_basin=int(
            np.count_nonzero(negative_truth & in_positive)
        ),
        correctly_classified_in_negative_basin=int(
            np.count_nonzero(negative_truth & in_negative)
        ),
        correctly_classified_in_positive_basin=int(
            np.count_nonzero(positive_truth & in_positive)
        ),
    )
    # __post_init__ already validates, but keep the conservation check visible
    # at the computation boundary where future category changes are most likely.
    result.validate_count_conservation()
    return result
