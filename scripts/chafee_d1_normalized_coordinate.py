r"""Evaluate a normalized first-Fourier-coordinate Chafee--Infante model.

The experiment fixes

.. math::

    a=1.2365946,\qquad E(x)=x[:,0]/a,

and uses the one-parameter polynomial map

.. math::

    G_\mu(z)=z+\mu z(1-z^2).

Two primary variants are evaluated:

* ``mu`` fitted in closed form on all 30,000 archived training pairs; and
* the predetermined ``mu=0.75`` polynomial used in the earlier scale-aware
  limit test.

An additional frozen candidate scan is explicitly post-hoc and test-informed.
It is included only to expose the topology/statistics tradeoff.  All variants
use the same archived level-8 padded CMGDB and Marcio basin semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scripts import chafee_d1_physics_coordinate_ceiling as base
from scripts import chafee_d1_polynomial_coordinate_map as prior_polynomial
from scripts import chafee_d1_unit_scale_fitted_mu as cubic

from latentdynamics.analysis.cmgdb_roa import attractor_cells
from latentdynamics.analysis.morse import LatentBounds
from latentdynamics.viz import save_morse_graph_artifacts

reference = base.reference

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    CODE_ROOT / "output" / "exploratory_chafee_d1_normalized_a_1_2365946"
)
PRIOR_POLYNOMIAL_STATS = (
    CODE_ROOT
    / "output"
    / "exploratory_chafee_d1_polynomial_coordinate_map"
    / "basin_statistics.json"
)

NORMALIZATION_A = 1.2365946
PREDETERMINED_MU = 0.75
DENSE_GRID_POINTS = base.DENSE_GRID_POINTS
EXPERIMENT_LABEL = "normalized first-Fourier-coordinate cubic map"
SCAN_GRID = (
    0.05,
    0.075,
    0.1,
    0.125,
    0.15,
    0.175,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
)


@dataclass(frozen=True)
class EvaluationContext:
    """Input-independent cell lookup data shared by every ``mu``."""

    bounds: LatentBounds
    resolution: Any
    truth: NDArray[np.int64]
    encoded_roots: NDArray[np.float64]
    training_x: NDArray[np.float64]
    training_y: NDArray[np.float64]
    point_cells: Any
    root_cells: Any
    unique_cell_ids: NDArray[np.int64]
    inverse: NDArray[np.int64]
    point_candidate_count: int


def normalized_encode(
    values: NDArray[np.float64],
    *,
    a: float = NORMALIZATION_A,
) -> NDArray[np.float64]:
    """Return the float32-rounded first coefficient divided by exact ``a``."""

    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("a must be finite and positive")
    return base.physics_encode(values) / a


def infer_normalized_bounds(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    a: float = NORMALIZATION_A,
) -> tuple[LatentBounds, dict[str, Any]]:
    """Infer the canonical padded bounds after normalizing the coordinate."""

    encoded = np.concatenate(
        (normalized_encode(x, a=a), normalized_encode(y, a=a)),
        axis=0,
    )
    raw_lower = encoded.min(axis=0)
    raw_upper = encoded.max(axis=0)
    span = raw_upper - raw_lower
    if not np.all(span > 0.0):
        raise ValueError("normalized training range is degenerate")
    padding = reference.BOUNDS_EPSILON_FRAC * span
    bounds = LatentBounds(
        lower=raw_lower - padding,
        upper=raw_upper + padding,
    )
    return bounds, {
        "coordinate": "E(x)=float32(x[:,0])/a",
        "a": a,
        "a_decimal_is_fixed_exactly_as_requested": "1.2365946",
        "division_precision": "binary64 after float32 coefficient rounding",
        "encoded_rows": ["training current x", "training next y"],
        "n_encoded_states": int(encoded.shape[0]),
        "raw_lower": raw_lower.tolist(),
        "raw_upper": raw_upper.tolist(),
        "epsilon_fraction": reference.BOUNDS_EPSILON_FRAC,
        "lower": bounds.lower.tolist(),
        "upper": bounds.upper.tolist(),
    }


def residual_metrics(
    encoded_x: NDArray[np.float64],
    encoded_y: NDArray[np.float64],
    *,
    mu: float,
) -> dict[str, Any]:
    """Return full one-step residual diagnostics in normalized coordinates."""

    z = np.asarray(encoded_x, dtype=np.float64).reshape(-1)
    z_next = np.asarray(encoded_y, dtype=np.float64).reshape(-1)
    if z.shape != z_next.shape or z.size < 1:
        raise ValueError("encoded_x and encoded_y must be nonempty and equally shaped")
    spec = cubic.UnitScaleCubicSpec(mu=mu)
    prediction = np.asarray(spec.evaluate(z), dtype=np.float64)
    residual = prediction - z_next
    absolute = np.abs(residual)
    mse = float(np.mean(residual * residual))
    return {
        "mu": mu,
        "n_pairs": int(z.size),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(absolute)),
        "max_absolute_error": float(np.max(absolute)),
        "mean_signed_error": float(np.mean(residual)),
        "absolute_error_quantiles": {
            "q50": float(np.quantile(absolute, 0.50)),
            "q90": float(np.quantile(absolute, 0.90)),
            "q95": float(np.quantile(absolute, 0.95)),
            "q99": float(np.quantile(absolute, 0.99)),
            "q999": float(np.quantile(absolute, 0.999)),
        },
    }


def fit_mu_and_residual_report(
    encoded_x: NDArray[np.float64],
    encoded_y: NDArray[np.float64],
) -> tuple[float, dict[str, Any]]:
    """Fit the scalar and compare it with identity and predetermined maps."""

    z = np.asarray(encoded_x, dtype=np.float64).reshape(-1)
    z_next = np.asarray(encoded_y, dtype=np.float64).reshape(-1)
    if z.shape != z_next.shape or z.size < 1:
        raise ValueError("encoded_x and encoded_y must be nonempty and equally shaped")
    feature = z * (1.0 - z * z)
    delta = z_next - z
    denominator = float(np.dot(feature, feature))
    if denominator <= 0.0:
        raise ValueError("normalized cubic feature has zero norm")
    numerator = float(np.dot(feature, delta))
    fitted_mu = numerator / denominator
    if fitted_mu <= 0.0:
        raise ValueError("unconstrained fitted mu is not positive")

    identity_residual = z - z_next
    identity_absolute = np.abs(identity_residual)
    identity_mse = float(np.mean(identity_residual * identity_residual))
    fitted = residual_metrics(encoded_x, encoded_y, mu=fitted_mu)
    predetermined = residual_metrics(
        encoded_x,
        encoded_y,
        mu=PREDETERMINED_MU,
    )
    fitted_residual = (
        np.asarray(cubic.UnitScaleCubicSpec(fitted_mu).evaluate(z))
        - z_next
    )
    return fitted_mu, {
        "schema_version": 1,
        "coordinate": "E(x)=float32(x[:,0])/1.2365946",
        "map": "G_mu(z)=z+mu*z*(1-z^2)",
        "fit_objective": (
            "min_mu sum_i ((E(y_i)-E(x_i))-mu*E(x_i)*(1-E(x_i)^2))^2"
        ),
        "closed_form": "mu=(f dot delta_z)/(f dot f), f=z*(1-z^2)",
        "fit_intercept": False,
        "n_training_pairs": int(z.size),
        "test_labels_used_in_fit": False,
        "stable_roots_used_in_fit": False,
        "feature_dot_delta": numerator,
        "feature_dot_feature": denominator,
        "fitted_mu": fitted_mu,
        "normal_equation_residual_f_dot_error": float(
            np.dot(feature, fitted_residual)
        ),
        "identity_baseline": {
            "description": "G(z)=z, equivalently mu=0",
            "mu": 0.0,
            "mse": identity_mse,
            "rmse": math.sqrt(identity_mse),
            "mae": float(np.mean(identity_absolute)),
            "max_absolute_error": float(np.max(identity_absolute)),
        },
        "fitted": fitted,
        "predetermined_mu_0_75": predetermined,
        "fitted_improvement_over_identity": {
            "mse_reduction": identity_mse - fitted["mse"],
            "relative_mse_reduction": (
                (identity_mse - fitted["mse"]) / identity_mse
            ),
        },
        "predetermined_vs_fitted": {
            "mse_difference": predetermined["mse"] - fitted["mse"],
            "mse_ratio": predetermined["mse"] / fitted["mse"],
        },
    }


def _build_context(
    *,
    bounds: LatentBounds,
    encoded_points: NDArray[np.float64],
    truth: NDArray[np.int64],
    encoded_roots: NDArray[np.float64],
    training_x: NDArray[np.float64],
    training_y: NDArray[np.float64],
) -> EvaluationContext:
    resolution = reference.RESOLUTIONS[1]
    point_cells = reference._uniform_point_cells(
        encoded_points,
        bounds,
        resolution,
    )
    root_cells = reference._uniform_point_cells(
        encoded_roots,
        bounds,
        resolution,
    )
    candidates = np.concatenate(
        (point_cells.flat_cell_ids, root_cells.flat_cell_ids)
    )
    unique_cell_ids, inverse = np.unique(candidates, return_inverse=True)
    return EvaluationContext(
        bounds=bounds,
        resolution=resolution,
        truth=truth,
        encoded_roots=encoded_roots,
        training_x=training_x,
        training_y=training_y,
        point_cells=point_cells,
        root_cells=root_cells,
        unique_cell_ids=unique_cell_ids,
        inverse=inverse,
        point_candidate_count=point_cells.flat_cell_ids.size,
    )


def merge_cell_intervals(
    boxes: NDArray[np.float64],
) -> list[tuple[float, float]]:
    """Return connected closed intervals represented by D1 CMGDB cells."""

    rows = np.asarray(boxes, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 2 or rows.shape[0] < 1:
        raise ValueError("D1 cell boxes must have nonempty shape (n, 2)")
    if not np.all(np.isfinite(rows)) or np.any(rows[:, 1] <= rows[:, 0]):
        raise ValueError("D1 cell boxes must be finite nondegenerate intervals")
    order = np.argsort(rows[:, 0], kind="stable")
    sorted_rows = rows[order]
    scale = max(1.0, float(np.max(np.abs(sorted_rows))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    merged: list[tuple[float, float]] = []
    lower = float(sorted_rows[0, 0])
    upper = float(sorted_rows[0, 1])
    for row in sorted_rows[1:]:
        next_lower = float(row[0])
        next_upper = float(row[1])
        if next_lower <= upper + tolerance:
            upper = max(upper, next_upper)
            continue
        merged.append((lower, upper))
        lower, upper = next_lower, next_upper
    merged.append((lower, upper))
    return merged


def _interior_margin(
    value: float,
    intervals: list[tuple[float, float]],
) -> float:
    for lower, upper in intervals:
        if lower < value < upper:
            return min(value - lower, upper - value)
    return 0.0


def _monotone_breaks(
    spec: cubic.UnitScaleCubicSpec,
    lower: float,
    upper: float,
) -> list[float]:
    points = [lower, upper]
    critical = spec.fold_magnitude
    for value in (-critical, critical):
        if lower < value < upper:
            points.append(value)
    return sorted(points)


def _monotone_preimage(
    spec: cubic.UnitScaleCubicSpec,
    left: float,
    right: float,
    target: float,
) -> float:
    """Locate ``G(z)=target`` on a derivative-monotone closed segment."""

    lo = left
    hi = right
    value_lo = float(spec.evaluate(lo)) - target
    value_hi = float(spec.evaluate(hi)) - target
    if value_lo == 0.0:
        return lo
    if value_hi == 0.0:
        return hi
    if value_lo * value_hi > 0.0:
        raise ValueError("target is not bracketed on monotone cubic segment")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value_mid = float(spec.evaluate(mid)) - target
        if value_mid == 0.0:
            return mid
        if value_lo * value_mid <= 0.0:
            hi = mid
            value_hi = value_mid
        else:
            lo = mid
            value_lo = value_mid
    return 0.5 * (lo + hi)


def exact_cubic_attracting_block_tau(
    spec: cubic.UnitScaleCubicSpec,
    intervals: list[tuple[float, float]],
) -> dict[str, Any]:
    """Compute ``inf_N dist(G(z), complement(Int N))`` for D1 cubic blocks.

    The cubic is monotone between the two derivative-critical points.  On each
    such segment, the distance to the complement of the interval union attains
    its minimum at an endpoint unless the image crosses the complement, in
    which case the exact infimum is zero.
    """

    if not intervals:
        raise ValueError("attracting block must contain at least one interval")
    interval_results: list[dict[str, Any]] = []
    global_tau = float("inf")
    global_witness: dict[str, Any] | None = None
    for source_index, (lower, upper) in enumerate(intervals):
        breaks = _monotone_breaks(spec, lower, upper)
        source_tau = float("inf")
        source_witness: dict[str, Any] | None = None
        segment_rows: list[dict[str, Any]] = []
        for left, right in pairwise(breaks):
            mapped_left = float(spec.evaluate(left))
            mapped_right = float(spec.evaluate(right))
            margin_left = _interior_margin(mapped_left, intervals)
            margin_right = _interior_margin(mapped_right, intervals)
            segment_tau = min(margin_left, margin_right)
            witness_z = left if margin_left <= margin_right else right
            witness_mapped = (
                mapped_left if margin_left <= margin_right else mapped_right
            )

            # If the endpoints land in different connected interiors, the
            # continuous image crosses the complement and tau is exactly zero.
            endpoint_components = []
            for mapped in (mapped_left, mapped_right):
                component = next(
                    (
                        index
                        for index, (target_lower, target_upper) in enumerate(
                            intervals
                        )
                        if target_lower < mapped < target_upper
                    ),
                    None,
                )
                endpoint_components.append(component)
            if (
                endpoint_components[0] is not None
                and endpoint_components[1] is not None
                and endpoint_components[0] != endpoint_components[1]
            ):
                segment_tau = 0.0
                mapped_min = min(mapped_left, mapped_right)
                mapped_max = max(mapped_left, mapped_right)
                boundaries = sorted(
                    boundary
                    for interval in intervals
                    for boundary in interval
                    if mapped_min <= boundary <= mapped_max
                )
                if not boundaries:
                    raise AssertionError(
                        "different target components must have a boundary "
                        "between mapped endpoints"
                    )
                target_boundary = boundaries[len(boundaries) // 2]
                witness_z = _monotone_preimage(
                    spec,
                    left,
                    right,
                    target_boundary,
                )
                witness_mapped = float(spec.evaluate(witness_z))

            row = {
                "source_segment": [left, right],
                "mapped_endpoints": [mapped_left, mapped_right],
                "endpoint_interior_margins": [margin_left, margin_right],
                "tau": segment_tau,
                "witness": {
                    "z": witness_z,
                    "G_z": witness_mapped,
                    "distance_to_complement_of_interior": segment_tau,
                },
            }
            segment_rows.append(row)
            if segment_tau < source_tau:
                source_tau = segment_tau
                source_witness = {
                    **row["witness"],
                    "source_segment": row["source_segment"],
                }
        interval_results.append(
            {
                "source_interval_index": source_index,
                "source_interval": [lower, upper],
                "width": upper - lower,
                "derivative_partition": breaks,
                "tau": source_tau,
                "witness": source_witness,
                "segments": segment_rows,
            }
        )
        if source_tau < global_tau:
            global_tau = source_tau
            global_witness = {
                **(source_witness or {}),
                "source_interval_index": source_index,
            }
    return {
        "definition": "inf_{z in N} dist(G(z), R\\Int(N))",
        "method": (
            "analytic D1 cubic extrema: interval endpoints plus the exact "
            "derivative-critical points +/-sqrt((1+mu)/(3mu))"
        ),
        "tau": global_tau,
        "witness": global_witness,
        "intervals": interval_results,
        "positive_attracting_margin": global_tau > 0.0,
    }


def conditional_block_residuals(
    spec: cubic.UnitScaleCubicSpec,
    encoded_x: NDArray[np.float64],
    encoded_y: NDArray[np.float64],
    intervals: list[tuple[float, float]],
    *,
    tau: float,
) -> dict[str, Any]:
    """Summarize stored-pair residuals whose current encoding lies in ``N``."""

    z = np.asarray(encoded_x, dtype=np.float64).reshape(-1)
    z_next = np.asarray(encoded_y, dtype=np.float64).reshape(-1)
    if z.shape != z_next.shape:
        raise ValueError("encoded training arrays must have equal shape")
    membership = np.zeros(z.shape, dtype=bool)
    for lower, upper in intervals:
        membership |= (z >= lower) & (z <= upper)
    indices = np.flatnonzero(membership)
    if indices.size == 0:
        return {
            "evaluated_pairs": int(z.size),
            "accepted_pairs": 0,
            "status": "no_stored_training_pairs_in_block",
        }
    mapped = np.asarray(spec.evaluate(z[indices]), dtype=np.float64)
    residual = np.abs(mapped - z_next[indices])
    local_max = int(np.argmax(residual))
    global_index = int(indices[local_max])
    maximum = float(residual[local_max])
    mse = float(np.mean(residual * residual))
    if tau > 0.0:
        ratio: float | None = maximum / tau
        if maximum > tau:
            comparison = "sample_counterexample_exceeds_tau"
            implication = (
                "This stored pair proves the sampled residual exceeds tau and "
                "therefore contradicts the tolerance inequality."
            )
        else:
            comparison = "sample_max_below_or_equal_tau_inconclusive"
            implication = (
                "The finite sample maximum is only a lower bound on the true "
                "supremum; being below tau does not certify the hypothesis."
            )
    else:
        ratio = None
        comparison = "tau_nonpositive"
        implication = "No positive attracting-block margin is available."
    return {
        "evaluated_pairs": int(z.size),
        "accepted_pairs": int(indices.size),
        "membership": "closed interval-union membership E(x) in N",
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(residual)),
        "max_absolute_residual": maximum,
        "absolute_residual_quantiles": {
            "q50": float(np.quantile(residual, 0.50)),
            "q90": float(np.quantile(residual, 0.90)),
            "q95": float(np.quantile(residual, 0.95)),
            "q99": float(np.quantile(residual, 0.99)),
            "q999": float(np.quantile(residual, 0.999)),
        },
        "max_sample_over_tau": ratio,
        "comparison_to_tau": comparison,
        "logical_caveat": implication,
        "witness": {
            "training_pair_index": global_index,
            "E_x": float(z[global_index]),
            "E_y": float(z_next[global_index]),
            "G_E_x": float(spec.evaluate(z[global_index])),
            "absolute_residual": maximum,
        },
    }


def attracting_block_audit(
    spec: cubic.UnitScaleCubicSpec,
    map_graph: Any,
    morse_graph: Any,
    attractors: list[int],
    context: EvaluationContext,
    *,
    negative_attractor: int | None,
    positive_attractor: int | None,
) -> dict[str, Any]:
    """Compute theorem-aligned ``N_q``, tau, and conditioned residuals."""

    by_node: dict[str, Any] = {}
    for node in attractors:
        recurrent = {
            int(cell) for cell in morse_graph.morse_set(int(node))
        }
        closure = attractor_cells(map_graph, morse_graph, [int(node)])
        if not closure:
            raise ValueError(f"minimal node {node} has an empty forward closure")
        boxes = np.asarray(
            [
                morse_graph.phase_space_box(cell)
                for cell in sorted(closure)
            ],
            dtype=np.float64,
        )
        intervals = merge_cell_intervals(boxes)
        tau = exact_cubic_attracting_block_tau(spec, intervals)
        residuals = conditional_block_residuals(
            spec,
            context.training_x,
            context.training_y,
            intervals,
            tau=float(tau["tau"]),
        )
        forward_invariant = all(
            int(target) in closure
            for cell in closure
            for target in map_graph.adjacencies(cell)
        )
        root_membership = {
            "negative": {
                "value": float(context.encoded_roots[0, 0]),
                "closed_N": any(
                    lower <= float(context.encoded_roots[0, 0]) <= upper
                    for lower, upper in intervals
                ),
                "interior_N": any(
                    lower < float(context.encoded_roots[0, 0]) < upper
                    for lower, upper in intervals
                ),
            },
            "positive": {
                "value": float(context.encoded_roots[1, 0]),
                "closed_N": any(
                    lower <= float(context.encoded_roots[1, 0]) <= upper
                    for lower, upper in intervals
                ),
                "interior_N": any(
                    lower < float(context.encoded_roots[1, 0]) < upper
                    for lower, upper in intervals
                ),
            },
        }
        physical = (
            "negative"
            if node == negative_attractor
            else "positive"
            if node == positive_attractor
            else "unassociated"
        )
        by_node[str(node)] = {
            "morse_node": node,
            "physical_attractor": physical,
            "N_q_definition": (
                "cell-level forward closure from "
                "latentdynamics.analysis.cmgdb_roa.attractor_cells"
            ),
            "recurrent_cell_count": len(recurrent),
            "forward_closure_cell_count": len(closure),
            "forward_closure_equals_recurrent_cells": closure == recurrent,
            "forward_invariant_adjacency_verified": forward_invariant,
            "cell_ids": sorted(closure),
            "connected_interval_count": len(intervals),
            "intervals": [
                {
                    "lower": lower,
                    "upper": upper,
                    "width": upper - lower,
                }
                for lower, upper in intervals
            ],
            "total_interval_width": float(
                sum(upper - lower for lower, upper in intervals)
            ),
            "encoded_root_inclusion": root_membership,
            "tau": tau,
            "stored_pair_residuals_conditioned_on_E_x_in_N": residuals,
        }
    return {
        "schema_version": 1,
        "theorem_quantity": "tau(N_q,G)",
        "attracting_block_construction": (
            "N_q is the cell-level forward closure of the minimal Morse node"
        ),
        "sample_supremum_caveat": (
            "A stored-pair maximum is a lower bound on the true supremum. "
            "max_sample <= tau is inconclusive; max_sample > tau contradicts "
            "the tolerance hypothesis."
        ),
        "nodes": by_node,
    }


def evaluate_mu(
    mu: float,
    context: EvaluationContext,
    *,
    role: str,
    test_informed: bool,
    dense_grid_points: int,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one ``mu`` with exact uniform-graph basin semantics."""

    spec = cubic.UnitScaleCubicSpec(mu=mu)
    topology = cubic.diagnose_dense_topology(
        spec,
        context.bounds,
        grid_points=dense_grid_points,
    )
    callback = cubic.UnitScalePaddedBoxMap(spec)
    morse_graph, map_graph, compute_seconds, conley_status = (
        reference._run_lookup_cmgdb(
            callback,
            context.bounds,
            subdiv_init=context.resolution.uniform_init,
            subdiv_min=context.resolution.uniform_min,
            subdiv_max=context.resolution.uniform_max,
            compute_conley=False,
        )
    )
    uniform_cells = int(map_graph.num_vertices())
    if uniform_cells != context.resolution.uniform_cells:
        raise ValueError("normalized evaluation changed the level-8 cell count")
    attractors = reference._morse_attractors(morse_graph)

    query_started = time.perf_counter()
    singleton_unique = reference._native_singleton_reachability(
        map_graph,
        morse_graph,
        context.unique_cell_ids,
    )
    query_seconds = time.perf_counter() - query_started
    singleton_candidates = singleton_unique[context.inverse]
    point_singletons = np.asarray(
        singleton_candidates[: context.point_candidate_count],
        dtype=np.int32,
    )
    root_singletons = np.asarray(
        singleton_candidates[context.point_candidate_count :],
        dtype=np.int32,
    )

    negative_attractor: int | None = None
    positive_attractor: int | None = None
    root_resolution_error: str | None = None
    if len(attractors) == 2:
        try:
            negative_attractor = reference._root_attractor_label(
                root_singletons,
                context.root_cells,
                0,
                attractors,
            )
            positive_attractor = reference._root_attractor_label(
                root_singletons,
                context.root_cells,
                1,
                attractors,
            )
        except ValueError as error:
            root_resolution_error = str(error)

    graph_checks = {
        "uniform_cell_count_matches_level_8": (
            uniform_cells == context.resolution.uniform_cells
        ),
        "exactly_two_minimal_attractors": len(attractors) == 2,
        "encoded_roots_resolve_to_distinct_attractors": bool(
            negative_attractor is not None
            and positive_attractor is not None
            and negative_attractor != positive_attractor
        ),
    }
    valid = bool(all(graph_checks.values()))
    statistics: dict[str, Any] | None = None
    predicted: NDArray[np.int32] | None = None
    if valid:
        if negative_attractor is None or positive_attractor is None:
            raise AssertionError("validated attractors cannot be None")
        predicted = reference._point_basin_labels(
            point_singletons,
            context.point_cells,
            negative_attractor=negative_attractor,
            positive_attractor=positive_attractor,
        )
        _, statistics = base._statistics_payload(
            truth=context.truth,
            predicted=predicted,
            negative_attractor=negative_attractor,
            positive_attractor=positive_attractor,
        )

    result: dict[str, Any] = {
        "mu": mu,
        "role": role,
        "test_informed": test_informed,
        "map": "G_mu(z)=z+mu*z*(1-z^2)",
        "morse_nodes": int(morse_graph.num_vertices()),
        "minimal_attractors": attractors,
        "negative_attractor": negative_attractor,
        "positive_attractor": positive_attractor,
        "root_resolution_error": root_resolution_error,
        "graph_checks": graph_checks,
        "graph_valid_for_basin_statistics": valid,
        "statistics": statistics,
        "topology": {
            "all_archived_domain_checks_passed": topology[
                "all_archived_domain_checks_passed"
            ],
            "failed_checks": topology["failed_checks"],
            "fixed_points": topology["fixed_points"],
            "derivatives": topology["derivatives"],
            "sign_reversal": topology["sign_reversal"],
        },
        "cmgdb": {
            "subdivisions": [
                context.resolution.uniform_init,
                context.resolution.uniform_min,
                context.resolution.uniform_max,
            ],
            "uniform_cells": uniform_cells,
            "padding": True,
            "compute_seconds": compute_seconds,
            "reachability_query_seconds": query_seconds,
            "conley": conley_status,
            "callback": {
                "box_calls": callback.box_calls,
                "batch_calls": callback.batch_calls,
                "scalar_evaluations": callback.scalar_evaluations,
            },
        },
    }

    if artifact_dir is not None:
        block_audit = attracting_block_audit(
            spec,
            map_graph,
            morse_graph,
            attractors,
            context,
            negative_attractor=negative_attractor,
            positive_attractor=positive_attractor,
        )
        result["attracting_block_audit"] = block_audit
        artifact_dir.mkdir(parents=True)
        graph_dir = artifact_dir / "MG_uniform_s8"
        graph_dir.mkdir()
        dot_path, sets_path = save_morse_graph_artifacts(
            morse_graph,
            graph_dir,
        )
        intervals = prior_polynomial._morse_set_intervals(sets_path)
        result["morse_set_intervals"] = intervals
        result["graph_summary"] = reference._morse_summary(dot_path)
        result["morse_graph_path"] = str(dot_path.relative_to(artifact_dir))
        result["morse_sets_path"] = str(sets_path.relative_to(artifact_dir))
        base._write_json(artifact_dir / "topology_diagnostics.json", topology)
        base._write_json(
            artifact_dir / "attracting_block_audit.json",
            block_audit,
        )
        base._write_json(artifact_dir / "basin_statistics.json", result)
        if predicted is not None:
            np.save(artifact_dir / "trajectory_basin_labels.npy", predicted)
        np.savez_compressed(
            graph_dir / "marcio_singleton_reachability_queries.npz",
            queried_cell_ids=context.unique_cell_ids,
            singleton_node_by_queried_cell=singleton_unique,
            point_candidate_cell_ids=context.point_cells.flat_cell_ids,
            point_candidate_offsets=context.point_cells.offsets,
            point_singleton_nodes=point_singletons,
            root_candidate_cell_ids=context.root_cells.flat_cell_ids,
            root_candidate_offsets=context.root_cells.offsets,
            root_singleton_nodes=root_singletons,
            encoded_stable_roots=context.encoded_roots,
        )
    return result


def _scan_candidates(fitted_mu: float) -> list[dict[str, Any]]:
    candidates = [
        {
            "candidate_id": "least_squares",
            "mu": fitted_mu,
            "role": "training-only least-squares fit",
            "test_informed": False,
        }
    ]
    for mu in SCAN_GRID:
        if math.isclose(mu, fitted_mu, rel_tol=0.0, abs_tol=1e-14):
            continue
        candidates.append(
            {
                "candidate_id": f"grid_{mu:g}",
                "mu": mu,
                "role": "post-hoc exploratory scan",
                "test_informed": True,
            }
        )
    return candidates


def _compact_scan_row(result: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    statistics = result["statistics"]
    compact_statistics: dict[str, Any] | None = None
    if isinstance(statistics, dict):
        counts = statistics["counts"]
        compact_statistics = {
            "combined_correct": statistics["combined_correct"],
            "negative_correct_count": counts[
                "correctly_classified_in_negative_basin"
            ],
            "positive_correct_count": counts[
                "correctly_classified_in_positive_basin"
            ],
            "outside_count": counts["outside_both_basins"],
            "wrong_basin_count": (
                counts["misclassified_in_negative_basin"]
                + counts["misclassified_in_positive_basin"]
            ),
        }
    return {
        "candidate_id": candidate_id,
        "mu": result["mu"],
        "role": result["role"],
        "test_informed": result["test_informed"],
        "morse_nodes": result["morse_nodes"],
        "minimal_attractors": result["minimal_attractors"],
        "graph_valid_for_basin_statistics": result[
            "graph_valid_for_basin_statistics"
        ],
        "root_resolution_error": result["root_resolution_error"],
        "failed_dense_topology_checks": result["topology"]["failed_checks"],
        "statistics": compact_statistics,
    }


def _root_alignment_report(
    encoded_roots: NDArray[np.float64],
) -> dict[str, Any]:
    roots = np.asarray(encoded_roots, dtype=np.float64).reshape(-1)
    model_roots = np.asarray([-1.0, 1.0], dtype=np.float64)
    difference = roots - model_roots
    return {
        "a": NORMALIZATION_A,
        "model_outer_fixed_points": model_roots.tolist(),
        "encoded_pde_stable_roots": roots.tolist(),
        "encoded_minus_model": difference.tolist(),
        "absolute_mismatch": np.abs(difference).tolist(),
        "maximum_absolute_mismatch": float(np.max(np.abs(difference))),
    }


def run_experiment(
    *,
    output_dir: Path,
    archive_dir: Path,
    dense_grid_points: int = DENSE_GRID_POINTS,
) -> dict[str, Any]:
    """Run fitted, predetermined, and explicitly post-hoc scan variants."""

    output = base._assert_isolated_output(output_dir)
    started = time.perf_counter()
    inputs = reference.verify_exact_inputs(archive_dir)
    x, y = reference._load_training_pairs(inputs.train_data)
    encoded_x = normalized_encode(x)
    encoded_y = normalized_encode(y)
    fitted_mu, residual_report = fit_mu_and_residual_report(
        encoded_x,
        encoded_y,
    )
    bounds, bounds_payload = infer_normalized_bounds(x, y)

    physical_roots = reference._load_stable_roots(inputs.stable_roots)
    points, truth = reference._load_trajectory_labels(inputs.trajectory_labels)
    encoded_roots = normalized_encode(physical_roots)
    encoded_points = normalized_encode(points)
    context = _build_context(
        bounds=bounds,
        encoded_points=encoded_points,
        truth=truth,
        encoded_roots=encoded_roots,
        training_x=encoded_x,
        training_y=encoded_y,
    )

    output.mkdir(parents=True)
    base._write_json(output / "residual_report.json", residual_report)
    base._write_json(output / "bounds.json", bounds_payload)
    base._write_json(
        output / "fixed_root_alignment.json",
        _root_alignment_report(encoded_roots),
    )
    np.save(output / "encoded_stable_roots.npy", encoded_roots)

    fitted_result = evaluate_mu(
        fitted_mu,
        context,
        role="training-only least-squares fitted mu",
        test_informed=False,
        dense_grid_points=dense_grid_points,
        artifact_dir=output / "fitted_mu",
    )
    predetermined_result = evaluate_mu(
        PREDETERMINED_MU,
        context,
        role="predetermined mu=0.75 polynomial comparator",
        test_informed=False,
        dense_grid_points=dense_grid_points,
        artifact_dir=output / "predetermined_mu_0_75",
    )

    scan_rows: list[dict[str, Any]] = []
    for candidate in _scan_candidates(fitted_mu):
        result = evaluate_mu(
            float(candidate["mu"]),
            context,
            role=str(candidate["role"]),
            test_informed=bool(candidate["test_informed"]),
            dense_grid_points=dense_grid_points,
        )
        scan_rows.append(
            _compact_scan_row(result, str(candidate["candidate_id"]))
        )
    valid_rows = [
        row
        for row in scan_rows
        if row["graph_valid_for_basin_statistics"]
        and isinstance(row["statistics"], dict)
    ]
    post_hoc_best = max(
        valid_rows,
        key=lambda row: int(row["statistics"]["combined_correct"]["count"]),
    )
    scan_payload = {
        "schema_version": 1,
        "designation": "post-hoc test-informed topology/statistics scan",
        "paper_eligible": False,
        "same_labels_used_for_selection_and_reporting": True,
        "candidate_count": len(scan_rows),
        "rows": scan_rows,
        "post_hoc_selection": {
            "rule": (
                "maximum combined-correct count among graph-valid candidates "
                "in the frozen scan"
            ),
            "candidate_id": post_hoc_best["candidate_id"],
            "mu": post_hoc_best["mu"],
            "statistics": post_hoc_best["statistics"],
            "warning": "descriptive limit test, not an unbiased estimate",
        },
    }
    scan_dir = output / "posthoc_mu_scan"
    scan_dir.mkdir()
    base._write_json(scan_dir / "scan_results.json", scan_payload)

    prior = base._baseline_comparison(
        PRIOR_POLYNOMIAL_STATS,
        label="prior raw-coordinate a-scaled polynomial, mu=0.75",
    )
    comparison = {
        "fitted": {
            "mu": fitted_mu,
            "residual": residual_report["fitted"],
            "graph_valid": fitted_result["graph_valid_for_basin_statistics"],
            "statistics": fitted_result["statistics"],
        },
        "predetermined_mu_0_75": {
            "mu": PREDETERMINED_MU,
            "residual": residual_report["predetermined_mu_0_75"],
            "graph_valid": predetermined_result[
                "graph_valid_for_basin_statistics"
            ],
            "statistics": predetermined_result["statistics"],
            "prior_equivalent_baseline": prior,
        },
        "post_hoc_best": scan_payload["post_hoc_selection"],
    }
    base._write_json(output / "comparison.json", comparison)

    comparability = {
        "paper_eligible": False,
        "coordinate": "fixed physics coordinate x[:,0]/1.2365946",
        "learned_encoder": False,
        "learned_decoder": False,
        "fitted_variant_learned_parameter_count": 1,
        "predetermined_variant_learned_parameter_count": 0,
        "test_labels_used_in_least_squares_fit": False,
        "post_hoc_scan_is_test_informed": True,
        "valid_interpretation": (
            "Exploratory comparison of residual fit and finite-grid topology "
            "for a normalized one-parameter physical-coordinate map."
        ),
    }
    base._write_json(output / "comparability.json", comparability)
    (output / "COMPARABILITY.md").write_text(
        "\n".join(
            [
                "# Normalized first-Fourier-coordinate cubic map",
                "",
                "This is exploratory and not paper-eligible.",
                "",
                "The fitted mu uses only the 30,000 training pairs. The fixed "
                "mu=0.75 comparator is predetermined. The additional scan uses "
                "the archived basin labels for post-hoc selection and must not "
                "be interpreted as held-out performance.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "experiment_label": EXPERIMENT_LABEL,
        "output_dir": str(output),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": base._sha256(Path(__file__).resolve()),
        },
        "shared_drivers": [
            {
                "path": str(Path(base.__file__).resolve()),
                "sha256": base._sha256(Path(base.__file__).resolve()),
            },
            {
                "path": str(Path(cubic.__file__).resolve()),
                "sha256": base._sha256(Path(cubic.__file__).resolve()),
            },
        ],
        "runtime": base._runtime_metadata(),
        "inputs": inputs.provenance(),
        "parameters": {
            "a": NORMALIZATION_A,
            "coordinate": "E(x)=float32(x[:,0])/a",
            "map": "G_mu(z)=z+mu*z*(1-z^2)",
            "fitted_mu": fitted_mu,
            "predetermined_mu": PREDETERMINED_MU,
            "dense_grid_points": dense_grid_points,
            "cmgdb_subdivisions": [8, 8, 8],
            "cmgdb_padding": True,
        },
        "duration_seconds": time.perf_counter() - started,
        "fitted_graph_valid": fitted_result[
            "graph_valid_for_basin_statistics"
        ],
        "predetermined_graph_valid": predetermined_result[
            "graph_valid_for_basin_statistics"
        ],
        "post_hoc_selection": scan_payload["post_hoc_selection"],
    }
    base._write_json(output / "run_manifest.json", manifest)
    base._write_json(
        output / "artifact_manifest.json",
        base._artifact_manifest(output),
    )

    def summary(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "mu": result["mu"],
            "morse_nodes": result["morse_nodes"],
            "minimal_attractors": result["minimal_attractors"],
            "graph_valid": result["graph_valid_for_basin_statistics"],
            "statistics": result["statistics"],
        }

    return {
        "output_dir": str(output),
        "a": NORMALIZATION_A,
        "encoded_roots": encoded_roots[:, 0].tolist(),
        "fitted": summary(fitted_result),
        "predetermined_mu_0_75": summary(predetermined_result),
        "post_hoc_selection": scan_payload["post_hoc_selection"],
        "paper_eligible": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=reference.DEFAULT_ARCHIVE_DIR,
    )
    parser.add_argument(
        "--dense-grid-points",
        type=int,
        default=DENSE_GRID_POINTS,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_experiment(
        output_dir=args.output_dir,
        archive_dir=args.archive_dir,
        dense_grid_points=args.dense_grid_points,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
