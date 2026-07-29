r"""Fit and evaluate the raw-coordinate, unit-scale Chafee--Infante D1 map.

This isolated exploratory experiment implements exactly

.. math::

    E(x)=x[:,0],\qquad
    G(z)=z+\mu z(1-z^2).

The encoder is the raw first Fourier coefficient: it is *not* divided by the
archived stable-root magnitude.  Consequently, setting ``a=1`` forces the
model's nonzero fixed points to ``-1`` and ``+1``, while the encoded PDE stable
roots remain near ``-1.2366`` and ``+1.2366``.  That mismatch is measured
rather than corrected.

The only learned quantity is ``mu``.  It is obtained in closed form from all
30,000 archived training pairs by the no-intercept least-squares problem

.. math::

    \min_\mu \sum_i
      \left((E(y_i)-E(x_i))-\mu E(x_i)(1-E(x_i)^2)\right)^2.

No trajectory/test labels or stable-root values enter this fit.  Labels and
stable roots are loaded only after fitting, for the requested CMGDB basin
evaluation and fixed-point mismatch diagnostic.

The downstream calculation preserves the established archived D1 protocol:
float32-equivalent coordinate evaluation, bounds inferred from current and
next training states with 10 percent padding, level-8 uniform CMGDB,
``BoxMap(..., padding=True)``, native singleton reachability, and Marcio's
negative-first closed-cell basin semantics on the 7,862 conditioned points.

Outputs are fail-if-present and disjoint from all prior experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import CMGDB
import numpy as np
from numpy.typing import NDArray
from scripts import chafee_d1_physics_coordinate_ceiling as base
from scripts import chafee_d1_polynomial_coordinate_map as scaled_polynomial

from latentdynamics.analysis.basin_statistics import OUTSIDE
from latentdynamics.analysis.morse import LatentBounds
from latentdynamics.viz import save_morse_graph_artifacts

reference = base.reference

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    CODE_ROOT / "output" / "exploratory_chafee_d1_unit_scale_fitted_mu"
)
SCALED_POLYNOMIAL_STATS = (
    CODE_ROOT
    / "output"
    / "exploratory_chafee_d1_polynomial_coordinate_map"
    / "basin_statistics.json"
)
RATIONAL_CEILING_STATS = (
    CODE_ROOT
    / "output"
    / "exploratory_chafee_d1_physics_coordinate_ceiling"
    / "basin_statistics.json"
)

SCHEMA_VERSION = 1
DENSE_GRID_POINTS = base.DENSE_GRID_POINTS
EXPERIMENT_LABEL = "exploratory raw-coordinate unit-scale fitted-mu cubic map"
MODEL_FIXED_ROOT_MAGNITUDE = 1.0


@dataclass(frozen=True)
class UnitScaleCubicSpec:
    """The map ``G(z)=z+mu*z*(1-z^2)`` with fixed unit root scale."""

    mu: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu):
            raise ValueError("mu must be finite")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive")

    @property
    def fixed_points(self) -> NDArray[np.float64]:
        return np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)

    @property
    def fold_magnitude(self) -> float:
        """Return ``|z|`` where the derivative first vanishes."""

        return math.sqrt((1.0 + self.mu) / (3.0 * self.mu))

    @property
    def sign_reversal_magnitude(self) -> float:
        """Return nonzero ``|z|`` where the map itself vanishes."""

        return math.sqrt((1.0 + self.mu) / self.mu)

    def feature(
        self,
        values: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        z = np.asarray(values, dtype=np.float64)
        return z * (1.0 - z * z)

    def evaluate(
        self,
        values: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        z = np.asarray(values, dtype=np.float64)
        return z + self.mu * self.feature(z)

    def drift(
        self,
        values: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        z = np.asarray(values, dtype=np.float64)
        return self.mu * self.feature(z)

    def derivative(
        self,
        values: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        z = np.asarray(values, dtype=np.float64)
        return 1.0 + self.mu * (1.0 - 3.0 * z * z)


class UnitScalePaddedBoxMap:
    """CMGDB scalar and batch callbacks for the fitted cubic map."""

    def __init__(self, spec: UnitScaleCubicSpec) -> None:
        self.spec = spec
        self.scalar_evaluations = 0
        self.box_calls = 0
        self.batch_calls = 0

    def _point_map(self, point: Any) -> list[float]:
        values = np.asarray(point, dtype=np.float64).reshape(-1)
        if values.shape != (1,):
            raise ValueError(
                f"unit-scale D1 point must have shape (1,), got {values.shape}"
            )
        self.scalar_evaluations += 1
        return [float(self.spec.evaluate(values[0]))]

    def __call__(self, rectangle: Any) -> list[float]:
        self.box_calls += 1
        return CMGDB.BoxMap(self._point_map, rectangle, padding=True)

    def batch(self, rectangles: Any) -> list[list[float]]:
        self.batch_calls += 1
        return [self(rectangle) for rectangle in rectangles]


def fit_mu_least_squares(
    encoded_x: NDArray[np.float64],
    encoded_y: NDArray[np.float64],
) -> tuple[UnitScaleCubicSpec, dict[str, Any]]:
    """Fit the single no-intercept coefficient using all supplied pairs."""

    z = np.asarray(encoded_x, dtype=np.float64).reshape(-1)
    z_next = np.asarray(encoded_y, dtype=np.float64).reshape(-1)
    if z.shape != z_next.shape or z.size < 1:
        raise ValueError("encoded_x and encoded_y must be nonempty and equally shaped")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(z_next)):
        raise ValueError("encoded training pairs contain non-finite values")

    feature = z * (1.0 - z * z)
    delta = z_next - z
    denominator = float(np.dot(feature, feature))
    if denominator <= 0.0:
        raise ValueError("unit-scale cubic feature has zero norm")
    numerator = float(np.dot(feature, delta))
    mu = numerator / denominator
    spec = UnitScaleCubicSpec(mu=mu)

    fitted_delta = mu * feature
    fitted_prediction = z + fitted_delta
    residual = delta - fitted_delta
    persistence_residual = delta
    return spec, {
        "schema_version": 1,
        "fit_role": "the only learned parameter in the experiment",
        "coordinate": "raw first Fourier coefficient E(x)=x[:, 0]",
        "normalization": "none; a is fixed to 1 and E(x) is not divided by a",
        "map": "G(z)=z+mu*z*(1-z^2)",
        "residual_equation": (
            "delta_z=(E(y)-E(x)) ~= mu*f(z), f(z)=z*(1-z^2)"
        ),
        "objective": "min_mu sum_i (delta_z_i-mu*f(z_i))^2; no intercept",
        "closed_form": "mu=(f dot delta_z)/(f dot f)",
        "selection": "unconstrained least-squares minimizer on training pairs",
        "n_training_pairs": int(z.size),
        "test_labels_used_in_fit": False,
        "stable_roots_used_in_fit": False,
        "feature_dot_delta": numerator,
        "feature_dot_feature": denominator,
        "fitted_mu": mu,
        "one_step_mse": float(np.mean((fitted_prediction - z_next) ** 2)),
        "delta_residual_mse": float(np.mean(residual * residual)),
        "delta_residual_sse": float(np.dot(residual, residual)),
        "persistence_mu_zero_mse": float(
            np.mean(persistence_residual * persistence_residual)
        ),
        "mse_improvement_over_persistence": float(
            np.mean(persistence_residual * persistence_residual)
            - np.mean(residual * residual)
        ),
        "training_coordinate_range": {
            "z_minimum": float(np.min(z)),
            "z_maximum": float(np.max(z)),
            "z_next_minimum": float(np.min(z_next)),
            "z_next_maximum": float(np.max(z_next)),
        },
    }


def fixed_point_mismatch_diagnostic(
    spec: UnitScaleCubicSpec,
    encoded_stable_roots: NDArray[np.float64],
) -> dict[str, Any]:
    """Measure the intentional mismatch between fixed unit roots and PDE roots."""

    roots = np.asarray(encoded_stable_roots, dtype=np.float64)
    if roots.shape != (2, 1):
        raise ValueError(f"encoded roots must have shape (2, 1), got {roots.shape}")
    encoded = roots[:, 0]
    model_outer = np.asarray([-1.0, 1.0], dtype=np.float64)
    signed_difference = encoded - model_outer
    mapped = np.asarray(spec.evaluate(encoded), dtype=np.float64)
    drift = mapped - encoded
    derivatives = np.asarray(spec.derivative(encoded), dtype=np.float64)
    return {
        "schema_version": 1,
        "encoder": "raw E(x)=x[:, 0]",
        "model_fixed_points": spec.fixed_points.tolist(),
        "model_outer_fixed_points": model_outer.tolist(),
        "encoded_pde_stable_roots": encoded.tolist(),
        "encoded_root_minus_corresponding_model_root": signed_difference.tolist(),
        "absolute_mismatch": np.abs(signed_difference).tolist(),
        "mean_absolute_mismatch": float(np.mean(np.abs(signed_difference))),
        "relative_mismatch_to_encoded_root_magnitude": (
            np.abs(signed_difference) / np.abs(encoded)
        ).tolist(),
        "map_at_encoded_pde_roots": mapped.tolist(),
        "drift_at_encoded_pde_roots": drift.tolist(),
        "derivative_at_encoded_pde_roots": derivatives.tolist(),
        "encoded_pde_roots_are_map_fixed_points": bool(
            np.allclose(mapped, encoded, rtol=0.0, atol=1e-12)
        ),
        "interpretation": (
            "Because raw E is not normalized and a=1, the learned map attracts "
            "toward +/-1 rather than the encoded PDE equilibria near +/-1.2366."
        ),
    }


def diagnose_dense_topology(
    spec: UnitScaleCubicSpec,
    bounds: LatentBounds,
    *,
    grid_points: int = DENSE_GRID_POINTS,
) -> dict[str, Any]:
    """Diagnose fixed points and global behavior on the archived CMGDB domain."""

    if grid_points < 1_001 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer of at least 1001")
    extent = float(max(abs(bounds.lower[0]), abs(bounds.upper[0])))
    grid = np.linspace(-extent, extent, grid_points, dtype=np.float64)
    spacing = float(grid[1] - grid[0])
    mapped = np.asarray(spec.evaluate(grid), dtype=np.float64)
    drift = mapped - grid
    derivative = np.asarray(spec.derivative(grid), dtype=np.float64)
    expected_roots = spec.fixed_points
    roots = base._dense_sign_change_roots(grid, drift)
    root_tolerance = 3.0 * spacing

    distances = np.min(
        np.abs(grid[:, None] - expected_roots[None, :]),
        axis=1,
    )
    drift_mask = distances > 8.0 * spacing
    expected_drift_sign = np.sign(grid * (1.0 - grid * grid))
    signed_drift = expected_drift_sign[drift_mask] * drift[drift_mask]
    root_derivatives = np.asarray(
        spec.derivative(expected_roots),
        dtype=np.float64,
    )
    fixed_residuals = np.asarray(spec.drift(expected_roots), dtype=np.float64)
    odd_residual = np.asarray(spec.evaluate(grid) + spec.evaluate(-grid))
    positive = grid > 0.0
    negative = grid < 0.0

    checks = {
        "dense_grid_has_exactly_three_fixed_points": bool(roots.shape == (3,)),
        "dense_roots_match_minus_one_zero_plus_one": bool(
            roots.shape == (3,)
            and np.allclose(
                roots,
                expected_roots,
                rtol=0.0,
                atol=root_tolerance,
            )
        ),
        "fixed_point_residuals_near_machine_precision": bool(
            np.max(np.abs(fixed_residuals)) < 1e-12
        ),
        "oddness_near_machine_precision": bool(
            np.max(np.abs(odd_residual)) < 1e-12
        ),
        "correct_unit_scale_double_well_drift": bool(
            signed_drift.size > 0 and np.all(signed_drift > 0.0)
        ),
        "outer_fixed_points_strictly_stable": bool(
            abs(root_derivatives[0]) < 1.0
            and abs(root_derivatives[2]) < 1.0
        ),
        "origin_strictly_unstable": bool(root_derivatives[1] > 1.0),
        "strictly_positive_derivative_on_archived_domain": bool(
            np.min(derivative) > 0.0
        ),
        "map_preserves_nonzero_sign_on_archived_domain": bool(
            np.all(mapped[positive] > 0.0)
            and np.all(mapped[negative] < 0.0)
        ),
        "mapped_archived_domain_stays_inside_cmgdb_bounds": bool(
            np.min(mapped) >= float(bounds.lower[0])
            and np.max(mapped) <= float(bounds.upper[0])
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "formula": "G(z)=z+mu*z*(1-z^2); raw z=E(x)=x[:,0]; a=1",
        "parameters": asdict(spec),
        "dense_grid": {
            "points": grid_points,
            "lower": float(grid[0]),
            "upper": float(grid[-1]),
            "spacing": spacing,
        },
        "fixed_points": {
            "expected": expected_roots.tolist(),
            "dense_sign_change_estimates": roots.tolist(),
            "matching_tolerance": root_tolerance,
            "absolute_residuals": np.abs(fixed_residuals).tolist(),
        },
        "derivatives": {
            "at_minus_one_zero_plus_one": root_derivatives.tolist(),
            "expected_at_outer_roots": 1.0 - 2.0 * spec.mu,
            "expected_at_origin": 1.0 + spec.mu,
            "dense_minimum": float(np.min(derivative)),
            "dense_maximum": float(np.max(derivative)),
            "fold_magnitude": spec.fold_magnitude,
            "folds_are_outside_archived_domain": bool(spec.fold_magnitude > extent),
        },
        "sign_reversal": {
            "nonzero_G_zero_magnitude": spec.sign_reversal_magnitude,
            "sign_reversal_is_outside_archived_domain": bool(
                spec.sign_reversal_magnitude > extent
            ),
            "positive_grid_points_mapped_negative": int(
                np.count_nonzero(mapped[positive] < 0.0)
            ),
            "negative_grid_points_mapped_positive": int(
                np.count_nonzero(mapped[negative] > 0.0)
            ),
        },
        "range": {
            "mapped_minimum": float(np.min(mapped)),
            "mapped_maximum": float(np.max(mapped)),
            "cmgdb_lower": float(bounds.lower[0]),
            "cmgdb_upper": float(bounds.upper[0]),
        },
        "drift": {
            "minimum_correctly_signed_drift_away_from_roots": float(
                np.min(signed_drift)
            ),
            "maximum_absolute_drift": float(np.max(np.abs(drift))),
        },
        "oddness": {
            "maximum_absolute_Gz_plus_Gminusz": float(
                np.max(np.abs(odd_residual))
            )
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "all_archived_domain_checks_passed": not failed_checks,
        "global_caveat": (
            "The cubic is unbounded and eventually folds/sign-reverses outside "
            "the finite archived domain even when all finite-domain checks pass."
        ),
    }


def _comparison_entry(
    path: Path,
    *,
    label: str,
    result_count: int,
    result_percentage: float,
) -> dict[str, Any]:
    baseline = base._baseline_comparison(path, label=label)
    if baseline is None:
        return {"available": False, "path": str(path.resolve())}
    baseline_count = int(baseline["combined_correct_count"])
    baseline_percentage = float(baseline["combined_correct_percentage"])
    return {
        "available": True,
        "baseline": baseline,
        "unit_scale_minus_baseline_count": result_count - baseline_count,
        "unit_scale_minus_baseline_percentage_points": (
            result_percentage - baseline_percentage
        ),
    }


def _comparison_payload(statistics: dict[str, Any]) -> dict[str, Any]:
    result_count = int(statistics["combined_correct"]["count"])
    result_percentage = float(statistics["combined_correct"]["percentage"])
    return {
        "warning": "Exploratory descriptive comparison; only mu was fitted.",
        "unit_scale_combined_correct_count": result_count,
        "unit_scale_combined_correct_percentage": result_percentage,
        "baselines": {
            "scaled_polynomial_mu_0_75": _comparison_entry(
                SCALED_POLYNOMIAL_STATS,
                label="a=1.2366 scaled polynomial with predetermined mu=0.75",
                result_count=result_count,
                result_percentage=result_percentage,
            ),
            "rational_topology_ceiling": _comparison_entry(
                RATIONAL_CEILING_STATS,
                label="a=1.2366 saturated rational topology ceiling",
                result_count=result_count,
                result_percentage=result_percentage,
            ),
        },
    }


def _comparability_payload() -> dict[str, Any]:
    return {
        "paper_eligible": False,
        "designation": EXPERIMENT_LABEL,
        "learned_components": ["one scalar mu by closed-form least squares"],
        "not_learned": [
            "the fixed first-Fourier-coordinate encoder",
            "the unit scale a=1",
            "the cubic functional form",
        ],
        "no_test_label_leakage_into_fit": True,
        "fit_data": "all 30,000 archived training current/next pairs only",
        "evaluation_only_inputs": [
            "archived stable roots for attractor association and mismatch reporting",
            "archived trajectory points and labels for requested basin statistics",
        ],
        "preserved_evaluation_semantics": [
            "raw E(x)=x[:,0] with float32-equivalent rounding",
            "bounds from all current and next training pairs",
            "10 percent bounds padding",
            "level-8 uniform 256-cell CMGDB graph",
            "CMGDB BoxMap padding=True",
            "native CMGDB.MorseSingletonReachability",
            "negative-first closed-cell basin classification",
            "7,862 conditioned-trajectory denominator",
        ],
        "valid_interpretation": (
            "An exploratory one-parameter reduced map showing the effect of "
            "forcing a=1 while leaving the physical coordinate unnormalized."
        ),
        "invalid_interpretations": [
            "a trained autoencoder result",
            "a learned encoder result",
            "a reconstruction or full-state prediction model",
            "a paper-eligible model comparison",
        ],
    }


def _write_comparability_markdown(
    output_dir: Path,
    comparability: dict[str, Any],
) -> Path:
    lines = [
        "# Raw-coordinate unit-scale fitted-mu cubic",
        "",
        f"Designation: **{comparability['designation']}**.",
        "",
        "The only learned object is one scalar mu, fitted on 30,000 training pairs.",
        "No trajectory/test labels or stable-root values enter the fit.",
        "",
        "## Valid interpretation",
        "",
        comparability["valid_interpretation"],
        "",
        "## Not learned",
        "",
        *(f"- {item}" for item in comparability["not_learned"]),
        "",
        "## Evaluation protocol",
        "",
        *(f"- {item}" for item in comparability["preserved_evaluation_semantics"]),
        "",
        "## Invalid interpretations",
        "",
        *(f"- {item}" for item in comparability["invalid_interpretations"]),
        "",
    ]
    path = output_dir / "COMPARABILITY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_experiment(
    *,
    output_dir: Path,
    archive_dir: Path,
    dense_grid_points: int = DENSE_GRID_POINTS,
) -> dict[str, Any]:
    """Fit the one scalar, run CMGDB, and write a provenance-complete result."""

    output = base._assert_isolated_output(output_dir)
    started = time.perf_counter()
    inputs = reference.verify_exact_inputs(archive_dir)
    x, y = reference._load_training_pairs(inputs.train_data)

    encoded_x = base.physics_encode(x)
    encoded_y = base.physics_encode(y)
    spec, fit = fit_mu_least_squares(encoded_x, encoded_y)
    bounds, bounds_payload = base.infer_physics_bounds(x, y)
    topology = diagnose_dense_topology(
        spec,
        bounds,
        grid_points=dense_grid_points,
    )

    # Evaluation-only inputs are intentionally loaded after the fit is frozen.
    physical_roots = reference._load_stable_roots(inputs.stable_roots)
    points, truth = reference._load_trajectory_labels(inputs.trajectory_labels)
    encoded_roots = base.physics_encode(physical_roots)
    mismatch = fixed_point_mismatch_diagnostic(spec, encoded_roots)

    output.mkdir(parents=True)
    uniform_dir = output / "MG_uniform_s8"
    uniform_dir.mkdir()
    base._write_json(output / "fit.json", fit)
    base._write_json(output / "bounds.json", bounds_payload)
    base._write_json(output / "fixed_point_mismatch.json", mismatch)
    base._write_json(output / "topology_diagnostics.json", topology)

    box_map = UnitScalePaddedBoxMap(spec)
    resolution = reference.RESOLUTIONS[1]
    morse_graph, map_graph, cmgdb_seconds, conley_status = (
        reference._run_lookup_cmgdb(
            box_map,
            bounds,
            subdiv_init=resolution.uniform_init,
            subdiv_min=resolution.uniform_min,
            subdiv_max=resolution.uniform_max,
            compute_conley=False,
        )
    )
    uniform_cells = int(map_graph.num_vertices())
    if uniform_cells != resolution.uniform_cells:
        raise ValueError(
            f"uniform CMGDB returned {uniform_cells} cells; "
            f"expected {resolution.uniform_cells}"
        )
    attractors = reference._morse_attractors(morse_graph)
    dot_path, sets_path = save_morse_graph_artifacts(morse_graph, uniform_dir)
    graph_summary = reference._morse_summary(dot_path)
    morse_intervals = scaled_polynomial._morse_set_intervals(sets_path)

    encoded_points = base.physics_encode(points)
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
    all_candidate_ids = np.concatenate(
        (point_cells.flat_cell_ids, root_cells.flat_cell_ids)
    )
    unique_cell_ids, inverse = np.unique(all_candidate_ids, return_inverse=True)
    query_started = time.perf_counter()
    singleton_by_unique_cell = reference._native_singleton_reachability(
        map_graph,
        morse_graph,
        unique_cell_ids,
    )
    query_seconds = time.perf_counter() - query_started
    singleton_by_candidate = singleton_by_unique_cell[inverse]
    split = point_cells.flat_cell_ids.size
    point_singletons = np.asarray(
        singleton_by_candidate[:split],
        dtype=np.int32,
    )
    root_singletons = np.asarray(
        singleton_by_candidate[split:],
        dtype=np.int32,
    )

    graph_checks: dict[str, Any] = {
        "uniform_cell_count_matches_level_8": (
            uniform_cells == resolution.uniform_cells
        ),
        "exactly_two_minimal_attractors": len(attractors) == 2,
        "stable_roots_resolve_to_distinct_attractors": False,
    }
    root_resolution_error: str | None = None
    negative_attractor: int | None = None
    positive_attractor: int | None = None
    if len(attractors) == 2:
        try:
            negative_attractor = reference._root_attractor_label(
                root_singletons,
                root_cells,
                0,
                attractors,
            )
            positive_attractor = reference._root_attractor_label(
                root_singletons,
                root_cells,
                1,
                attractors,
            )
            graph_checks["stable_roots_resolve_to_distinct_attractors"] = (
                negative_attractor != positive_attractor
            )
        except ValueError as error:
            root_resolution_error = str(error)

    statistics_core: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    predicted: NDArray[np.int32] | None = None
    statistics_comparable = bool(all(graph_checks.values()))
    if statistics_comparable:
        if negative_attractor is None or positive_attractor is None:
            raise AssertionError("validated root attractors cannot be None")
        predicted = reference._point_basin_labels(
            point_singletons,
            point_cells,
            negative_attractor=negative_attractor,
            positive_attractor=positive_attractor,
        )
        _, statistics_core = base._statistics_payload(
            truth=truth,
            predicted=predicted,
            negative_attractor=negative_attractor,
            positive_attractor=positive_attractor,
        )
        comparison = _comparison_payload(statistics_core)
        np.save(output / "trajectory_basin_labels.npy", predicted)

    np.save(output / "encoded_stable_roots.npy", encoded_roots)
    np.savez_compressed(
        uniform_dir / "marcio_singleton_reachability_queries.npz",
        queried_cell_ids=unique_cell_ids,
        singleton_node_by_queried_cell=singleton_by_unique_cell,
        point_candidate_cell_ids=point_cells.flat_cell_ids,
        point_candidate_offsets=point_cells.offsets,
        point_singleton_nodes=point_singletons,
        root_candidate_cell_ids=root_cells.flat_cell_ids,
        root_candidate_offsets=root_cells.offsets,
        root_singleton_nodes=root_singletons,
        encoded_stable_roots=encoded_roots,
    )

    comparability = _comparability_payload()
    predicted_counts = (
        {
            str(label): int(count)
            for label, count in sorted(Counter(predicted.tolist()).items())
        }
        if predicted is not None
        else None
    )
    basin_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "computed" if statistics_comparable else "unavailable",
        "experiment": {
            "label": EXPERIMENT_LABEL,
            "paper_eligible": False,
            "learned_parameter_count": 1,
            "test_labels_used_in_fit": False,
        },
        "coordinate": {
            "formula": "E(x)=x_1=x[:,0]",
            "source": "raw first Fourier/spectral coefficient",
            "learned": False,
            "normalized": False,
            "encoded_stable_roots": encoded_roots.tolist(),
        },
        "unit_scale_cubic_map": {
            "formula": "G(z)=z+mu*z*(1-z^2)",
            "a": MODEL_FIXED_ROOT_MAGNITUDE,
            "parameters": asdict(spec),
            "mu_learned": True,
            "fit_path": "fit.json",
            "fixed_point_mismatch_path": "fixed_point_mismatch.json",
            "topology_diagnostics_path": "topology_diagnostics.json",
        },
        "inputs": inputs.provenance(),
        "bounds": bounds_payload,
        "cmgdb": {
            "subdivisions": [
                resolution.uniform_init,
                resolution.uniform_min,
                resolution.uniform_max,
            ],
            "uniform_cells": uniform_cells,
            "padding": True,
            "subdiv_limit": reference.SUBDIV_LIMIT,
            "morse_nodes": int(morse_graph.num_vertices()),
            "attractor_nodes": attractors,
            "nonminimal_recurrent_nodes": sorted(
                {int(node) for node in morse_intervals} - set(attractors)
            ),
            "negative_attractor": negative_attractor,
            "positive_attractor": positive_attractor,
            "morse_set_intervals": morse_intervals,
            "queried_uniform_cells": int(unique_cell_ids.size),
            "compute_seconds": cmgdb_seconds,
            "reachability_query_seconds": query_seconds,
            "conley": conley_status,
            "callback": {
                "box_calls": box_map.box_calls,
                "batch_calls": box_map.batch_calls,
                "scalar_evaluations": box_map.scalar_evaluations,
            },
            "graph_summary": graph_summary,
            "graph_checks": graph_checks,
            "root_resolution_error": root_resolution_error,
            "statistics_comparable": statistics_comparable,
            "morse_graph_path": str(dot_path.relative_to(output)),
            "morse_sets_path": str(sets_path.relative_to(output)),
        },
        "classification": {
            "method": (
                "Exact Marcio singleton-all-reachable-Morse-set basin semantics "
                "on the level-8 uniform CMGDB graph"
            ),
            "rule": (
                "complete reachable Morse-node set equals exactly the "
                "corresponding singleton attractor"
            ),
            "native_query": "CMGDB.MorseSingletonReachability",
            "closed_cell_boundary_policy": (
                "negative basin first, then positive basin, matching the archived loop"
            ),
            "counts_by_point_label": predicted_counts,
            "outside_label": OUTSIDE,
        },
        "statistics": statistics_core,
        "comparison": comparison,
        "comparability": comparability,
    }
    base._write_json(output / "basin_statistics.json", basin_payload)
    if comparison is not None:
        base._write_json(output / "comparison.json", comparison)
    base._write_json(output / "comparability.json", comparability)
    _write_comparability_markdown(output, comparability)

    run_manifest = {
        "schema_version": 1,
        "experiment_label": EXPERIMENT_LABEL,
        "output_dir": str(output),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": base._sha256(Path(__file__).resolve()),
        },
        "shared_reference_drivers": [
            {
                "path": str(Path(base.__file__).resolve()),
                "sha256": base._sha256(Path(base.__file__).resolve()),
            },
            {
                "path": str(Path(scaled_polynomial.__file__).resolve()),
                "sha256": base._sha256(Path(scaled_polynomial.__file__).resolve()),
            },
        ],
        "runtime": base._runtime_metadata(),
        "inputs": inputs.provenance(),
        "fit": {
            "training_pairs": fit["n_training_pairs"],
            "formula": fit["objective"],
            "mu": spec.mu,
            "one_step_mse": fit["one_step_mse"],
            "test_labels_used": False,
            "stable_roots_used": False,
        },
        "parameters": {
            "coordinate": "raw x[:,0] with float32-equivalent rounding",
            "a": MODEL_FIXED_ROOT_MAGNITUDE,
            "map": "G(z)=z+mu*z*(1-z^2)",
            "dense_grid_points": dense_grid_points,
            "cmgdb_subdivisions": [8, 8, 8],
            "cmgdb_padding": True,
            "cmgdb_subdiv_limit": reference.SUBDIV_LIMIT,
        },
        "duration_seconds": time.perf_counter() - started,
        "topology_checks": topology["checks"],
        "graph_checks": graph_checks,
        "statistics_comparable": statistics_comparable,
        "primary_result": statistics_core,
        "comparability_path": "COMPARABILITY.md",
    }
    base._write_json(output / "run_manifest.json", run_manifest)
    base._write_json(
        output / "artifact_manifest.json",
        base._artifact_manifest(output),
    )

    return {
        "output_dir": str(output),
        "fitted_mu": spec.mu,
        "one_step_mse": fit["one_step_mse"],
        "encoded_stable_roots": encoded_roots[:, 0].tolist(),
        "model_outer_fixed_points": [-1.0, 1.0],
        "mean_absolute_fixed_point_mismatch": (
            mismatch["mean_absolute_mismatch"]
        ),
        "all_archived_domain_topology_checks_passed": (
            topology["all_archived_domain_checks_passed"]
        ),
        "failed_topology_checks": topology["failed_checks"],
        "morse_nodes": graph_summary["nodes"],
        "minimal_attractors": graph_summary["minimal_nodes"],
        "graph_checks": graph_checks,
        "statistics_comparable": statistics_comparable,
        "combined_correct_count": (
            statistics_core["combined_correct"]["count"]
            if statistics_core is not None
            else None
        ),
        "combined_correct_percentage": (
            statistics_core["combined_correct"]["percentage"]
            if statistics_core is not None
            else None
        ),
        "counts": (
            statistics_core["counts"] if statistics_core is not None else None
        ),
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
