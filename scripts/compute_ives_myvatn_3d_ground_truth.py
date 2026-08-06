#!/usr/bin/env python3
"""Compute a direct 3D CMGDB reference for the analytic Ives map.

This computation never uses a learned encoder, decoder, or latent map.  Its
rectangle callback is a vectorized analytic monotone interval enclosure of the
original log10 midge-algae-detritus map, with a small multi-ULP outward guard.
An optional 2x2x2 internal partition reduces dependency overestimation before
the child images are hulled.  This is substantially stronger than
evaluating only the eight box corners, while remaining a numerical reference
rather than an arbitrary-precision formal proof.

Run from ``code/``.  A staged ladder is recommended before the final Conley
calculation::

    python scripts/compute_ives_myvatn_3d_ground_truth.py --subdiv 18 24 30
    python scripts/compute_ives_myvatn_3d_ground_truth.py --subdiv 18 27 33
    python scripts/compute_ives_myvatn_3d_ground_truth.py --subdiv 18 30 36 --conley
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import CMGDB
import numpy as np
from numpy.typing import NDArray

from latentdynamics.analysis.cmgdb_fork import cmgdb_provenance, require_fork_cmgdb
from latentdynamics.config import load_config
from latentdynamics.systems import IvesModel, build_system
from latentdynamics.viz import PALETTE, save_morse_graph_artifacts
from latentdynamics.viz.morse_plots import render_morse_graph_from_dot

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
LOCAL_CMGDB_ROOT = (PROJECT_ROOT / "archive" / "CMGDB").resolve()
DEFAULT_REFERENCE = (
    CODE_ROOT
    / "src"
    / "latentdynamics"
    / "reference_data"
    / "ives_myvatn_invariant_points.csv"
)
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "output" / "ives_myvatn_3d_ground_truth"
SAMPLING_LOWER = np.asarray([-3.0, -7.5, -3.0], dtype=np.float64)
SAMPLING_UPPER = np.asarray([1.5, 1.5, 1.5], dtype=np.float64)
# This interval box is forward invariant under the analytic enclosure and the
# fifth enclosure iterate of the complete archived sampling box lies strictly
# inside it.  The 0.065-log-unit margin below log10(c) keeps the cycle's clamped
# algae phase away from the computational boundary at useful grid levels.
TRAPPING_LOWER = np.asarray([-5.94, -6.50, -6.50], dtype=np.float64)
TRAPPING_UPPER = np.asarray([1.65, 1.04, 1.43], dtype=np.float64)
SAMPLING_ABSORPTION_STEPS = 5
SCHEMA_VERSION = 1
OUTWARD_ULPS = 8


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _outward(
    values: NDArray[np.float64] | float,
    direction: float,
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    for _ in range(OUTWARD_ULPS):
        result = np.nextafter(result, direction)
    return result


def _down(values: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return _outward(values, -np.inf)


def _up(values: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return _outward(values, np.inf)


class IvesLogIntervalBoxMap:
    """Batched outward-rounded monotone enclosure of :class:`IvesModel`."""

    def __init__(self, system: IvesModel, *, internal_refinement: int = 0) -> None:
        if internal_refinement not in (0, 1):
            raise ValueError("internal_refinement must be 0 or 1")
        self.system = system
        self.internal_refinement = internal_refinement
        self.scalar_calls = 0
        self.batch_calls = 0
        self.rectangles = 0
        self.enclosure_leaf_rectangles = 0

    def __call__(self, rect: list[float]) -> list[float]:
        self.scalar_calls += 1
        self.rectangles += 1
        self.enclosure_leaf_rectangles += 8**self.internal_refinement
        return self._evaluate(np.asarray(rect, dtype=np.float64).reshape(1, 6))[0].tolist()

    def batch(self, rects: list[list[float]]) -> list[list[float]]:
        array = np.asarray(rects, dtype=np.float64)
        if array.size == 0:
            return []
        array = array.reshape(-1, 6)
        self.batch_calls += 1
        self.rectangles += int(array.shape[0])
        self.enclosure_leaf_rectangles += int(
            array.shape[0] * 8**self.internal_refinement
        )
        return self._evaluate(array).tolist()

    def stats(self) -> dict[str, int]:
        return {
            "scalar_calls": self.scalar_calls,
            "batch_calls": self.batch_calls,
            "rectangles": self.rectangles,
            "internal_refinement": self.internal_refinement,
            "enclosure_leaf_rectangles": self.enclosure_leaf_rectangles,
        }

    def _evaluate(self, rects: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.internal_refinement == 0:
            return self._evaluate_leaf(rects)

        rectangles = np.asarray(rects, dtype=np.float64)
        if rectangles.ndim != 2 or rectangles.shape[1] != 6:
            raise ValueError(
                f"expected an N x 6 rectangle array, received {rectangles.shape}"
            )
        result = np.empty_like(rectangles)
        corner_bits = np.asarray(
            [
                [(corner >> axis) & 1 for axis in range(3)]
                for corner in range(8)
            ],
            dtype=bool,
        )
        # Bound temporary arrays even when CMGDB supplies a very large batch.
        parents_per_block = 8_192
        for start in range(0, rectangles.shape[0], parents_per_block):
            stop = min(start + parents_per_block, rectangles.shape[0])
            parent = rectangles[start:stop]
            lower = parent[:, :3]
            upper = parent[:, 3:]
            midpoint = lower + 0.5 * (upper - lower)
            child_lower = np.where(
                corner_bits[None, :, :], midpoint[:, None, :], lower[:, None, :]
            )
            child_upper = np.where(
                corner_bits[None, :, :], upper[:, None, :], midpoint[:, None, :]
            )
            children = np.concatenate((child_lower, child_upper), axis=2).reshape(-1, 6)
            images = self._evaluate_leaf(children).reshape(stop - start, 8, 6)
            result[start:stop, :3] = images[:, :, :3].min(axis=1)
            result[start:stop, 3:] = images[:, :, 3:].max(axis=1)
        return result

    def _evaluate_leaf(self, rects: NDArray[np.float64]) -> NDArray[np.float64]:
        if rects.ndim != 2 or rects.shape[1] != 6:
            raise ValueError(f"expected an N x 6 rectangle array, received {rects.shape}")
        if not np.isfinite(rects).all():
            raise ValueError("rectangle bounds must be finite")
        log_lower = rects[:, :3]
        log_upper = rects[:, 3:]
        if np.any(log_lower > log_upper):
            raise ValueError("each rectangle lower bound must not exceed its upper bound")

        system = self.system
        if not 0.0 <= system.q <= 1.0:
            raise ValueError("the monotone Ives enclosure requires q in [0, 1]")
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            linear_lower = _down(np.power(10.0, log_lower))
            linear_upper = _up(np.power(10.0, log_upper))
            midge_l, algae_l, detritus_l = linear_lower.T
            midge_u, algae_u, detritus_u = linear_upper.T

            # m' is increasing in both m and R=a+p*d because q lies in [0, 1].
            def midge_image(
                midge: NDArray[np.float64],
                algae: NDArray[np.float64],
                detritus: NDArray[np.float64],
            ) -> NDArray[np.float64]:
                resource = algae + system.p * detritus
                return (
                    system.r1
                    * midge
                    * np.power(resource / (resource + midge), system.q)
                )

            midge_next_l = _down(
                midge_image(midge_l, algae_l, detritus_l)
            )
            midge_next_u = _up(
                midge_image(midge_u, algae_u, detritus_u)
            )

            # h(t)=t(1+t)^(-q) is increasing for q in [0, 1].  The two
            # consumption terms have useful coordinate monotonicity that is
            # much tighter than multiplying independent a/R, p*d/R, and m'
            # intervals: C_a rises with (a,m) and falls with d, while C_d
            # rises with (d,m) and falls with a.
            def h(
                midge: NDArray[np.float64],
                algae: NDArray[np.float64],
                detritus: NDArray[np.float64],
            ) -> NDArray[np.float64]:
                ratio = midge / (algae + system.p * detritus)
                return ratio * np.power(1.0 + ratio, -system.q)

            algae_consumed_l = _down(
                system.r1 * algae_l * h(midge_l, algae_l, detritus_u)
            )
            algae_consumed_u = _up(
                system.r1 * algae_u * h(midge_u, algae_u, detritus_l)
            )
            algae_produced_l = _down(system.r2 * algae_l / (1.0 + algae_l))
            algae_produced_u = _up(system.r2 * algae_u / (1.0 + algae_u))
            algae_raw_l = _down(algae_produced_l - algae_consumed_u + system.c)
            algae_raw_u = _up(algae_produced_u - algae_consumed_l + system.c)
            algae_next_l = np.maximum(system.c, algae_raw_l)
            algae_next_u = np.maximum(system.c, algae_raw_u)

            detritus_consumed_l = _down(
                system.r1
                * system.p
                * detritus_l
                * h(midge_l, algae_u, detritus_l)
            )
            detritus_consumed_u = _up(
                system.r1
                * system.p
                * detritus_u
                * h(midge_u, algae_l, detritus_u)
            )
            detritus_produced_l = _down(_down(system.d * detritus_l) + algae_l)
            detritus_produced_u = _up(_up(system.d * detritus_u) + algae_u)
            detritus_raw_l = _down(detritus_produced_l - detritus_consumed_u + system.c)
            detritus_raw_u = _up(detritus_produced_u - detritus_consumed_l + system.c)
            detritus_next_l = np.maximum(system.c, detritus_raw_l)
            detritus_next_u = np.maximum(system.c, detritus_raw_u)

            image_lower = _down(
                np.log10(
                    np.column_stack((midge_next_l, algae_next_l, detritus_next_l))
                )
            )
            image_upper = _up(
                np.log10(
                    np.column_stack((midge_next_u, algae_next_u, detritus_next_u))
                )
            )

        result = np.column_stack((image_lower, image_upper))
        if not np.isfinite(result).all() or np.any(image_lower > image_upper):
            raise FloatingPointError("Ives interval evaluation produced invalid bounds")
        return result


def _axis_splits(level: int, dimension: int = 3) -> NDArray[np.int64]:
    return np.asarray(
        [(level - axis + dimension - 1) // dimension for axis in range(dimension)],
        dtype=np.int64,
    )


def validate_interval_enclosure(
    box_map: IvesLogIntervalBoxMap,
    *,
    domain_lower: NDArray[np.float64] | None = None,
    domain_upper: NDArray[np.float64] | None = None,
    seed: int = 20_260_806,
    boxes_per_level: int = 48,
    points_per_box: int = 32,
    challenge_points: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Numerically challenge the enclosure on random aligned boxes and points."""

    rng = np.random.default_rng(seed)
    system = box_map.system
    domain_lower = (
        system.lower_bounds
        if domain_lower is None
        else np.asarray(domain_lower, dtype=np.float64)
    )
    domain_upper = (
        system.upper_bounds
        if domain_upper is None
        else np.asarray(domain_upper, dtype=np.float64)
    )
    levels = (0, 6, 12, 18, 24, 30)
    records: list[dict[str, Any]] = []
    total_points = 0
    global_lower_slack = np.full(3, np.inf, dtype=np.float64)
    global_upper_slack = np.full(3, np.inf, dtype=np.float64)

    for level in levels:
        splits = _axis_splits(level)
        counts = np.left_shift(1, splits)
        widths = (domain_upper - domain_lower) / counts
        indices = np.column_stack(
            [rng.integers(0, int(count), size=boxes_per_level) for count in counts]
        )
        lower = domain_lower + indices * widths
        upper = lower + widths
        rectangles = np.column_stack((lower, upper))
        interval_images = box_map._evaluate(rectangles)

        unit = rng.random((boxes_per_level, points_per_box, 3))
        points = lower[:, None, :] + unit * (upper - lower)[:, None, :]
        corner_bits = np.asarray(
            [
                [(corner >> axis) & 1 for axis in range(3)]
                for corner in range(8)
            ],
            dtype=bool,
        )
        corners = np.where(
            corner_bits[None, :, :],
            upper[:, None, :],
            lower[:, None, :],
        )
        points = np.concatenate((points, corners), axis=1)
        images = system.step(points.reshape(-1, 3)).reshape(boxes_per_level, -1, 3)
        lower_slack = images - interval_images[:, None, :3]
        upper_slack = interval_images[:, None, 3:] - images
        if np.any(lower_slack < 0.0) or np.any(upper_slack < 0.0):
            bad = np.argwhere((lower_slack < 0.0) | (upper_slack < 0.0))[0]
            raise AssertionError(
                "interval enclosure failure at "
                f"level={level}, box={int(bad[0])}, point={int(bad[1])}, axis={int(bad[2])}"
            )
        global_lower_slack = np.minimum(global_lower_slack, lower_slack.min(axis=(0, 1)))
        global_upper_slack = np.minimum(global_upper_slack, upper_slack.min(axis=(0, 1)))
        tested = int(images.shape[0] * images.shape[1])
        total_points += tested
        records.append(
            {
                "level": level,
                "axis_splits": splits.tolist(),
                "axis_counts": counts.tolist(),
                "boxes": boxes_per_level,
                "points_including_endpoint_pairs": tested,
                "median_interval_width": np.median(
                    interval_images[:, 3:] - interval_images[:, :3], axis=0
                ).tolist(),
            }
        )

    challenge_record: dict[str, Any] | None = None
    if challenge_points is not None:
        challenge = np.asarray(challenge_points, dtype=np.float64)
        if challenge.ndim != 2 or challenge.shape[1] != 3:
            raise ValueError("challenge_points must have shape (n, 3)")
        if not np.isfinite(challenge).all():
            raise ValueError("challenge_points must be finite")
        intervals = box_map._evaluate(np.column_stack((challenge, challenge)))
        images = system.step(challenge)
        lower_slack = images - intervals[:, :3]
        upper_slack = intervals[:, 3:] - images
        if np.any(lower_slack < 0.0) or np.any(upper_slack < 0.0):
            raise AssertionError("interval enclosure failed on a challenge point")
        global_lower_slack = np.minimum(global_lower_slack, lower_slack.min(axis=0))
        global_upper_slack = np.minimum(global_upper_slack, upper_slack.min(axis=0))
        total_points += int(challenge.shape[0])
        challenge_record = {
            "count": int(challenge.shape[0]),
            "minimum_lower_slack_by_coordinate": lower_slack.min(axis=0).tolist(),
            "minimum_upper_slack_by_coordinate": upper_slack.min(axis=0).tolist(),
        }

    return {
        "passed": True,
        "seed": seed,
        "levels": records,
        "boxes_per_level": boxes_per_level,
        "random_points_per_box": points_per_box,
        "corners_per_box": 8,
        "challenge_points": challenge_record,
        "total_points": total_points,
        "minimum_lower_slack_by_coordinate": global_lower_slack.tolist(),
        "minimum_upper_slack_by_coordinate": global_upper_slack.tolist(),
        "scope": (
            "dense randomized containment check of the outward-rounded analytic "
            "interval extension; not an arbitrary-precision proof"
        ),
    }


def sampling_absorption_audit(
    box_map: IvesLogIntervalBoxMap,
    *,
    steps: int = SAMPLING_ABSORPTION_STEPS,
) -> dict[str, Any]:
    """Enclose successive images of the archived sampling box.

    Because every step is an outer interval enclosure, containment of the last
    interval in ``TRAPPING_*`` certifies that every sampled-domain trajectory
    has entered the trapping box by that time (subject to floating-point
    interval arithmetic, rather than arbitrary-precision directed rounding).
    """

    if steps < 1:
        raise ValueError("absorption steps must be positive")
    lower = SAMPLING_LOWER.copy()
    upper = SAMPLING_UPPER.copy()
    iterates: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        image = box_map._evaluate(np.concatenate((lower, upper)).reshape(1, 6))[0]
        lower = image[:3]
        upper = image[3:]
        iterates.append(
            {
                "step": step,
                "lower": lower.tolist(),
                "upper": upper.tolist(),
            }
        )
    contained = bool(
        np.all(lower >= TRAPPING_LOWER) and np.all(upper <= TRAPPING_UPPER)
    )
    return {
        "passed": contained,
        "steps": steps,
        "iterates": iterates,
        "trapping_lower": TRAPPING_LOWER.tolist(),
        "trapping_upper": TRAPPING_UPPER.tolist(),
        "final_lower_margin": (lower - TRAPPING_LOWER).tolist(),
        "final_upper_margin": (TRAPPING_UPPER - upper).tolist(),
        "scope": (
            "successive outward interval enclosures from the complete archived "
            "sampling box"
        ),
    }


def _load_reference_points(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            point = [
                float(raw["barycenter_x"]),
                float(raw["barycenter_y"]),
                float(raw["barycenter_z"]),
            ]
            if not np.isfinite(point).all():
                raise ValueError("reference invariant points must be finite")
            rows.append(
                {
                    "vertex": int(raw["vertex"]),
                    "component_id": int(raw["component_id"]),
                    "point": point,
                }
            )
    if len(rows) != 13:
        raise ValueError(f"expected 13 Ives invariant points, found {len(rows)}")
    if sum(row["vertex"] == 0 for row in rows) != 12:
        raise ValueError("reference CSV must contain 12 cycle phases")
    if sum(row["vertex"] == 1 for row in rows) != 1:
        raise ValueError("reference CSV must contain one fixed point")
    cycle_components = {
        row["component_id"] for row in rows if row["vertex"] == 0
    }
    if cycle_components != set(range(12)):
        raise ValueError("cycle component IDs must be exactly 0 through 11")
    fixed_components = [
        row["component_id"] for row in rows if row["vertex"] == 1
    ]
    if fixed_components != [0]:
        raise ValueError("fixed-point component ID must be 0")
    return rows


def _audit_saved_sets(
    path: Path,
    *,
    domain_lower: NDArray[np.float64],
    domain_upper: NDArray[np.float64],
    reference_rows: list[dict[str, Any]],
    source_level: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 7 or data.shape[0] == 0:
        raise ValueError(f"expected nonempty seven-column Morse sets, received {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("saved Morse sets contain non-finite values")

    lower = data[:, :3]
    upper = data[:, 3:6]
    raw_labels = data[:, 6]
    labels = np.rint(raw_labels).astype(np.int64)
    if not np.array_equal(raw_labels, labels):
        raise ValueError("saved Morse sets contain nonintegral labels")
    if np.any(labels < 0):
        raise ValueError("saved Morse-set labels must be nonnegative")
    if np.any(lower >= upper):
        raise ValueError("saved Morse set contains an invalid box")

    tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(domain_upper))
    if np.any(lower < domain_lower - tolerance) or np.any(
        upper > domain_upper + tolerance
    ):
        raise ValueError("saved Morse box leaves the computational domain")
    source_splits = _axis_splits(source_level)
    source_counts = np.left_shift(1, source_splits)
    expected_widths = (domain_upper - domain_lower) / source_counts
    widths = upper - lower
    if not np.allclose(widths, expected_widths, rtol=0.0, atol=1e-11):
        raise ValueError(f"not every saved Morse box is a level-{source_level} cell")
    indices_float = (lower - domain_lower) / expected_widths
    indices = np.rint(indices_float).astype(np.int64)
    if not np.allclose(indices_float, indices, rtol=0.0, atol=1e-9):
        raise ValueError("saved Morse boxes are not aligned with the source grid")
    if np.any(indices < 0) or np.any(indices >= source_counts):
        raise ValueError("saved Morse-box index lies outside the source grid")
    cell_ids = (
        (indices[:, 0] * source_counts[1] + indices[:, 1]) * source_counts[2]
        + indices[:, 2]
    )
    unique_cell_count = int(np.unique(cell_ids).size)
    if unique_cell_count != data.shape[0]:
        raise ValueError("saved Morse sets contain duplicate or cross-labeled cells")

    boundary_mask = np.any(
        np.isclose(lower, domain_lower, rtol=0.0, atol=tolerance), axis=1
    ) | np.any(np.isclose(upper, domain_upper, rtol=0.0, atol=tolerance), axis=1)
    boundary_nodes = sorted(np.unique(labels[boundary_mask]).astype(str).tolist(), key=int)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    counts = {
        str(label): int(count)
        for label, count in zip(unique_labels, label_counts, strict=True)
    }

    points = np.asarray([row["point"] for row in reference_rows], dtype=np.float64)
    rows_with_membership: list[dict[str, Any]] = []
    for source, point in zip(reference_rows, points, strict=True):
        inside = np.all(point >= lower - tolerance, axis=1) & np.all(
            point <= upper + tolerance, axis=1
        )
        memberships = sorted(np.unique(labels[inside]).astype(str).tolist(), key=int)
        rows_with_membership.append(
            {**source, "morse_node_memberships": memberships}
        )

    grid_volume = int(np.prod(source_counts, dtype=np.int64))

    return (
        {
            "box_count": int(data.shape[0]),
            "boxes_per_node": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
            "occupied_lower": lower.min(axis=0).tolist(),
            "occupied_upper": upper.max(axis=0).tolist(),
            "minimum_box_width": widths.min(axis=0).tolist(),
            "maximum_box_width": widths.max(axis=0).tolist(),
            "boundary_touching_nodes": boundary_nodes,
            "source_level": source_level,
            "source_axis_splits": source_splits.tolist(),
            "source_axis_counts": source_counts.tolist(),
            "expected_box_width": expected_widths.tolist(),
            "unique_grid_cells": unique_cell_count,
            "grid_volume": grid_volume,
        },
        rows_with_membership,
    )


def _classify_reference_memberships(
    rows: list[dict[str, Any]],
    *,
    sink_ids: list[str],
) -> dict[str, Any]:
    sinks = set(sink_ids)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        sink_memberships = sorted(
            sinks.intersection(row["morse_node_memberships"]), key=int
        )
        enriched.append({**row, "sink_memberships": sink_memberships})

    cycle = [row for row in enriched if row["vertex"] == 0]
    fixed = [row for row in enriched if row["vertex"] == 1]
    fixed_sink = fixed[0]["sink_memberships"][0] if len(fixed[0]["sink_memberships"]) == 1 else None
    unique_cycle = [
        row["sink_memberships"][0]
        for row in cycle
        if len(row["sink_memberships"]) == 1
    ]
    consensus = Counter(unique_cycle).most_common(1)
    cycle_sink = consensus[0][0] if consensus else None
    cycle_coverage = sum(
        row["sink_memberships"] == [cycle_sink] for row in cycle
    ) if cycle_sink is not None else 0
    cycle_conflicts = sum(
        bool(row["sink_memberships"]) and row["sink_memberships"] != [cycle_sink]
        for row in cycle
    ) if cycle_sink is not None else sum(bool(row["sink_memberships"]) for row in cycle)
    cycle_morse_nodes = sorted(
        {node for row in cycle for node in row["morse_node_memberships"]},
        key=int,
    )
    fixed_morse_nodes = fixed[0]["morse_node_memberships"]
    return {
        "rows": enriched,
        "fixed_morse_node_ids": fixed_morse_nodes,
        "fixed_sink_id": fixed_sink,
        "fixed_point_unique_sink": fixed_sink is not None,
        "cycle_morse_node_ids": cycle_morse_nodes,
        "cycle_sink_id": cycle_sink,
        "cycle_unique_target_count": cycle_coverage,
        "cycle_conflicting_phase_count": cycle_conflicts,
        "cycle_assignment_pass": cycle_coverage == 12 and cycle_conflicts == 0,
        "distinct_sink_pass": (
            fixed_sink is not None and cycle_sink is not None and fixed_sink != cycle_sink
        ),
    }


def _resolved_system() -> tuple[Any, IvesModel]:
    config = load_config("ives_myvatn")
    system = build_system(config.system.name, config.system.params)
    if not isinstance(system, IvesModel):
        raise TypeError("ives_myvatn must resolve to IvesModel")
    return config, system


def _default_output(
    subdiv: tuple[int, int, int],
    limit: int,
    *,
    conley: bool,
    domain: str,
    interval_refinement: int,
) -> Path:
    init, minimum, maximum = subdiv
    suffix = "conley" if conley else "morse"
    return DEFAULT_OUTPUT_ROOT / (
        f"{domain}_interval_r{interval_refinement}_i{init}_m{minimum}_M{maximum}_"
        f"L{limit}_{suffix}"
    )


def _subdivision_limit_assessment(
    boxes_per_node: dict[str, int],
    *,
    minimum: int,
    maximum: int,
    limit: int,
) -> dict[str, Any]:
    """Report what leaf counts can and cannot prove about the CMGDB limit."""

    gap = maximum - minimum
    nodes: dict[str, Any] = {}
    for node, raw_count in sorted(boxes_per_node.items(), key=lambda item: int(item[0])):
        count = int(raw_count)
        immediate_descendants = count * 2 if gap > 0 else count
        full_unpruned_descendants = count * (1 << gap)
        if gap == 0:
            state = "minimum_equals_maximum"
        elif immediate_descendants > limit:
            state = "guaranteed_stop_before_first_post_minimum_decomposition"
        elif full_unpruned_descendants <= limit:
            state = "guaranteed_no_size_limit_stop_through_maximum"
        else:
            state = "indeterminate_from_saved_minimum_level_count"
        nodes[node] = {
            "minimum_level_boxes": count,
            "first_post_minimum_boxes_before_pruning": immediate_descendants,
            "maximum_level_boxes_without_pruning": full_unpruned_descendants,
            "state": state,
        }
    return {
        "minimum": minimum,
        "maximum": maximum,
        "limit": limit,
        "assessment_by_node": nodes,
        "guaranteed_immediate_stop_nodes": [
            node
            for node, record in nodes.items()
            if record["state"]
            == "guaranteed_stop_before_first_post_minimum_decomposition"
        ],
        "guaranteed_through_maximum_nodes": [
            node
            for node, record in nodes.items()
            if record["state"] == "guaranteed_no_size_limit_stop_through_maximum"
        ],
        "indeterminate_nodes": [
            node
            for node, record in nodes.items()
            if record["state"] == "indeterminate_from_saved_minimum_level_count"
        ],
        "interpretation": (
            "CMGDB does not expose the deepest processed hierarchy level. These "
            "states are conservative deductions from the saved minimum-level "
            "Morse-set sizes, not a claim that an indeterminate node hit the limit."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subdiv",
        type=int,
        nargs=3,
        metavar=("INIT", "MIN", "MAX"),
        default=(15, 18, 24),
    )
    parser.add_argument("--subdiv-limit", type=int, default=100_000)
    parser.add_argument(
        "--interval-refinement",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Split each CMGDB rectangle once along all three axes and hull the "
            "eight analytic child enclosures (1), or evaluate it directly (0)."
        ),
    )
    parser.add_argument(
        "--domain",
        choices=("trapping", "sampling"),
        default="trapping",
        help=(
            "Use the forward-invariant recurrent-dynamics box or the archived "
            "training/sampling box."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--conley", action="store_true")
    parser.add_argument("--enclosure-boxes-per-level", type=int, default=48)
    parser.add_argument("--enclosure-points-per-box", type=int, default=32)
    args = parser.parse_args(argv)

    subdiv = tuple(args.subdiv)
    if not 1 <= subdiv[0] <= subdiv[1] <= subdiv[2]:
        raise ValueError(f"expected 1 <= init <= min <= max, received {subdiv}")
    if args.subdiv_limit < 1:
        raise ValueError("subdiv-limit must be positive")
    if args.enclosure_boxes_per_level < 1 or args.enclosure_points_per_box < 1:
        raise ValueError("enclosure audit sizes must be positive")

    output = (
        args.output
        or _default_output(
            subdiv,
            args.subdiv_limit,
            conley=args.conley,
            domain=args.domain,
            interval_refinement=args.interval_refinement,
        )
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    morse_dir = output / "MG"
    morse_dir.mkdir()

    reference_path = args.reference_csv.resolve()
    reference_rows = _load_reference_points(reference_path)
    config, system = _resolved_system()
    if args.domain == "trapping":
        domain_lower = TRAPPING_LOWER.copy()
        domain_upper = TRAPPING_UPPER.copy()
    else:
        domain_lower = SAMPLING_LOWER.copy()
        domain_upper = SAMPLING_UPPER.copy()
    module_path = require_fork_cmgdb()
    interval_map = IvesLogIntervalBoxMap(
        system, internal_refinement=args.interval_refinement
    )
    absorption_audit = sampling_absorption_audit(interval_map)
    if not absorption_audit["passed"]:
        raise AssertionError(
            "archived sampling box did not enter the configured trapping box "
            f"within {SAMPLING_ABSORPTION_STEPS} interval iterates"
        )
    full_domain_image = interval_map._evaluate(
        np.concatenate((domain_lower, domain_upper)).reshape(1, 6)
    )[0]
    forward_invariant = bool(
        np.all(full_domain_image[:3] >= domain_lower)
        and np.all(full_domain_image[3:] <= domain_upper)
    )
    if args.domain == "trapping" and not forward_invariant:
        raise AssertionError(
            "configured trapping box failed the interval forward-invariance check: "
            f"{full_domain_image.tolist()}"
        )

    run_config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "purpose": "direct original-3D Ives high-resolution CMGDB reference",
        "system": {
            "name": "IvesModel",
            "coordinate_system": "log10(midge, algae, detritus)",
            "parameters": system.params,
            "bounds": {
                "name": args.domain,
                "lower": domain_lower.tolist(),
                "upper": domain_upper.tolist(),
                "interval_image_lower": full_domain_image[:3].tolist(),
                "interval_image_upper": full_domain_image[3:].tolist(),
                "interval_forward_invariant": forward_invariant,
                "outside_images": (
                    "none under the interval enclosure"
                    if forward_invariant
                    else "treated as escape from the restricted domain"
                ),
            },
            "archived_sampling_bounds": {
                "lower": SAMPLING_LOWER.tolist(),
                "upper": SAMPLING_UPPER.tolist(),
            },
            "archived_sampling_absorption": absorption_audit,
            "config": str(
                (CODE_ROOT / "src" / "latentdynamics" / "configs" / "ives_myvatn.yaml").resolve()
            ),
            "config_experiment_name": config.experiment_name,
        },
        "rectangle_map": {
            "name": "outward-rounded analytic monotone interval enclosure",
            "coordinates": "log10",
            "batching": True,
            "padding": False,
            "internal_refinement": args.interval_refinement,
            "internal_subboxes_per_cmgdb_rectangle": 8**args.interval_refinement,
            "rounding": (
                f"{OUTWARD_ULPS} numpy.nextafter steps toward -inf/+inf after "
                "each interval bound"
            ),
            "scope": (
                "direct numerical interval reference; stronger than corner sampling, "
                "not an arbitrary-precision formal proof"
            ),
        },
        "cmgdb": {
            **cmgdb_provenance(LOCAL_CMGDB_ROOT),
            "module_path_verified": str(module_path),
            "algorithm": (
                "ComputeConleyMorseGraphOnly" if args.conley else "ComputeMorseGraphOnly"
            ),
            "subdivision": {
                "init": subdiv[0],
                "min": subdiv[1],
                "max": subdiv[2],
                "limit": args.subdiv_limit,
            },
        },
        "reference_invariant_points": _file_record(reference_path),
        "output": str(output),
    }
    _write_json(output / "run_config.json", run_config)
    _write_json(
        output / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "phase": "interval_enclosure_audit",
            "updated_at_utc": _utc_now(),
        },
    )
    print(json.dumps(run_config, indent=2, sort_keys=True), flush=True)

    audit = validate_interval_enclosure(
        interval_map,
        domain_lower=domain_lower,
        domain_upper=domain_upper,
        boxes_per_level=args.enclosure_boxes_per_level,
        points_per_box=args.enclosure_points_per_box,
        challenge_points=np.asarray(
            [row["point"] for row in reference_rows], dtype=np.float64
        ),
    )
    _write_json(output / "interval_enclosure_audit.json", audit)
    _write_json(
        output / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "phase": run_config["cmgdb"]["algorithm"],
            "updated_at_utc": _utc_now(),
            "interval_enclosure_audit": _file_record(
                output / "interval_enclosure_audit.json"
            ),
        },
    )

    model = CMGDB.Model(
        subdiv[1],
        subdiv[2],
        subdiv[0],
        args.subdiv_limit,
        domain_lower.tolist(),
        domain_upper.tolist(),
        interval_map,
    )
    if not hasattr(model, "set_batch_map"):
        raise RuntimeError("CMGDB.Model.set_batch_map is required")
    model.set_batch_map(interval_map.batch)

    print(
        f"launching {run_config['cmgdb']['algorithm']} at "
        f"{subdiv[0]}/{subdiv[1]}/{subdiv[2]}",
        flush=True,
    )
    started = time.perf_counter()
    compute = (
        CMGDB.ComputeConleyMorseGraphOnly
        if args.conley
        else CMGDB.ComputeMorseGraphOnly
    )
    morse_graph = compute(model)
    compute_seconds = time.perf_counter() - started

    dot_path, sets_path = save_morse_graph_artifacts(
        morse_graph,
        morse_dir,
        palette=PALETTE,
    )
    render_morse_graph_from_dot(dot_path, output, palette=PALETTE)

    vertices = [str(int(node)) for node in morse_graph.vertices()]
    edges = [
        [source, str(int(target))]
        for source in vertices
        for target in morse_graph.adjacencies(int(source))
    ]
    sink_ids = [
        node for node in vertices if not list(morse_graph.adjacencies(int(node)))
    ]
    set_audit, raw_memberships = _audit_saved_sets(
        sets_path,
        domain_lower=domain_lower,
        domain_upper=domain_upper,
        reference_rows=reference_rows,
        source_level=subdiv[1],
    )
    membership = _classify_reference_memberships(raw_memberships, sink_ids=sink_ids)
    membership["source"] = _file_record(reference_path)
    _write_json(output / "reference_membership.json", membership)

    limit_assessment = _subdivision_limit_assessment(
        set_audit["boxes_per_node"],
        minimum=subdiv[1],
        maximum=subdiv[2],
        limit=args.subdiv_limit,
    )
    boundary_nodes = set(set_audit["boundary_touching_nodes"])
    acceptance_checks = {
        "all_12_cycle_phases_in_one_sink": membership["cycle_assignment_pass"],
        "fixed_point_in_one_sink": membership["fixed_point_unique_sink"],
        "cycle_and_fixed_point_in_distinct_sinks": membership["distinct_sink_pass"],
        "exactly_two_sinks": len(sink_ids) == 2,
        "no_saved_morse_set_touches_domain_boundary": not boundary_nodes,
        "no_guaranteed_immediate_post_minimum_limit_stop": not limit_assessment[
            "guaranteed_immediate_stop_nodes"
        ],
        "every_reference_point_has_one_morse_membership": all(
            len(row["morse_node_memberships"]) == 1
            for row in membership["rows"]
        ),
    }
    scientific_acceptance = {
        "passed_intrinsic_checks": all(acceptance_checks.values()),
        "checks": acceptance_checks,
        "requires_cross_resolution_confirmation": True,
        "note": (
            "A completed computation is not automatically an accepted ground-truth "
            "resolution. Final acceptance also requires stable role-matched graph "
            "structure across consecutive minimum subdivision levels."
        ),
    }
    manifest = {
        **run_config,
        "completed_at_utc": _utc_now(),
        "compute_seconds": round(compute_seconds, 6),
        "box_map_callback": interval_map.stats(),
        "morse_graph": {
            "n_nodes": len(vertices),
            "n_edges": len(edges),
            "nodes": vertices,
            "edges": edges,
            "sink_ids": sink_ids,
            "n_sinks": len(sink_ids),
            "conley_indices": (
                {
                    node: list(morse_graph.annotations(int(node)))
                    for node in vertices
                }
                if args.conley
                else None
            ),
        },
        "morse_sets": {
            **set_audit,
            "subdivision_limit_assessment": limit_assessment,
        },
        "reference_membership": {
            key: value for key, value in membership.items() if key not in {"rows", "source"}
        },
        "scientific_acceptance": scientific_acceptance,
        "artifacts": {
            "run_config": _file_record(output / "run_config.json"),
            "interval_enclosure_audit": _file_record(
                output / "interval_enclosure_audit.json"
            ),
            "morse_graph": _file_record(dot_path),
            "morse_sets": _file_record(sets_path),
            "morse_graph_pdf": _file_record(output / "morse_graph.pdf"),
            "morse_graph_png": _file_record(output / "morse_graph.png"),
            "reference_membership": _file_record(output / "reference_membership.json"),
        },
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(
        output / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "phase": "complete",
            "updated_at_utc": _utc_now(),
            "manifest": _file_record(output / "manifest.json"),
            "scientific_acceptance": scientific_acceptance,
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
