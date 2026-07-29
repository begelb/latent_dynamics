r"""Evaluate the unsaturated polynomial Chafee--Infante D1 map.

This is an isolated exploratory counterpart to
``chafee_d1_physics_coordinate_ceiling.py``.  It preserves the same fixed
physics coordinate, archived inputs, latent bounds, CMGDB resolution, padded
box-map construction, and basin-classification semantics, while replacing the
rational map by the user's simpler polynomial:

.. math::

    E(x)=x_1,\qquad
    G(z)=a\left(q+\mu q(1-q^2)\right),\quad q=z/a.

Here ``a`` is the mean magnitude of the first Fourier coefficient of the two
archived stable roots and ``mu=0.75``.  The polynomial has the desired three
fixed points locally, but it folds and reverses sign on the archived CMGDB
domain.  Those failed global checks are diagnostic results, not run-stopping
preconditions: the uniform graph is still computed so their practical effect
on the basin statistic can be measured.

This is not a trained model, an unbiased test result, or a paper-eligible
comparison.  Outputs are fail-if-present and never modify the rational-map
experiment.
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

from latentdynamics.analysis.basin_statistics import OUTSIDE
from latentdynamics.analysis.morse import LatentBounds
from latentdynamics.viz import save_morse_graph_artifacts

reference = base.reference

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    CODE_ROOT / "output" / "exploratory_chafee_d1_polynomial_coordinate_map"
)
RATIONAL_BASELINE_STATS = (
    CODE_ROOT
    / "output"
    / "exploratory_chafee_d1_physics_coordinate_ceiling"
    / "basin_statistics.json"
)

SCHEMA_VERSION = 1
SELECTED_MU = 0.75
DENSE_GRID_POINTS = base.DENSE_GRID_POINTS
EXPERIMENT_LABEL = "test-informed exploratory polynomial physics-coordinate map"


@dataclass(frozen=True)
class PolynomialMapSpec:
    """Parameters and formulas for the unsaturated cubic map."""

    stable_root_magnitude: float
    mu: float = SELECTED_MU

    def __post_init__(self) -> None:
        if not math.isfinite(self.stable_root_magnitude):
            raise ValueError("stable_root_magnitude must be finite")
        if self.stable_root_magnitude <= 0:
            raise ValueError("stable_root_magnitude must be positive")
        if not math.isfinite(self.mu):
            raise ValueError("mu must be finite")
        if self.mu <= 0:
            raise ValueError("mu must be positive")

    @property
    def fixed_points(self) -> NDArray[np.float64]:
        a = self.stable_root_magnitude
        return np.asarray([-a, 0.0, a], dtype=np.float64)

    @property
    def normalized_fold_magnitude(self) -> float:
        """Return |q| where G'(z)=0."""

        return math.sqrt((1.0 + self.mu) / (3.0 * self.mu))

    @property
    def normalized_sign_reversal_magnitude(self) -> float:
        """Return nonzero |q| where G(z)=0."""

        return math.sqrt((1.0 + self.mu) / self.mu)

    def evaluate(
        self,
        values: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        z = np.asarray(values, dtype=np.float64)
        q = z / self.stable_root_magnitude
        return self.stable_root_magnitude * (
            q + self.mu * q * (1.0 - q * q)
        )

    def drift(
        self,
        values: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        z = np.asarray(values, dtype=np.float64)
        return self.evaluate(z) - z

    def derivative(
        self,
        values: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        z = np.asarray(values, dtype=np.float64)
        q = z / self.stable_root_magnitude
        return 1.0 + self.mu * (1.0 - 3.0 * q * q)


class PolynomialPaddedBoxMap:
    """CMGDB scalar and batch callbacks for the polynomial map."""

    def __init__(self, spec: PolynomialMapSpec) -> None:
        self.spec = spec
        self.scalar_evaluations = 0
        self.box_calls = 0
        self.batch_calls = 0

    def _point_map(self, point: Any) -> list[float]:
        values = np.asarray(point, dtype=np.float64).reshape(-1)
        if values.shape != (1,):
            raise ValueError(
                f"polynomial D1 point must have shape (1,), got {values.shape}"
            )
        self.scalar_evaluations += 1
        return [float(self.spec.evaluate(values[0]))]

    def __call__(self, rectangle: Any) -> list[float]:
        self.box_calls += 1
        return CMGDB.BoxMap(self._point_map, rectangle, padding=True)

    def batch(self, rectangles: Any) -> list[list[float]]:
        self.batch_calls += 1
        return [self(rectangle) for rectangle in rectangles]


def fit_one_step_mu_diagnostic(
    spec: PolynomialMapSpec,
    encoded_x: NDArray[np.float64],
    encoded_y: NDArray[np.float64],
) -> dict[str, Any]:
    """Fit polynomial-map strength for diagnosis without changing the run."""

    z = np.asarray(encoded_x, dtype=np.float64).reshape(-1)
    z_next = np.asarray(encoded_y, dtype=np.float64).reshape(-1)
    if z.shape != z_next.shape or z.size < 1:
        raise ValueError("encoded_x and encoded_y must be nonempty and equally shaped")
    a = spec.stable_root_magnitude
    q = z / a
    feature = a * q * (1.0 - q * q)
    denominator = float(np.dot(feature, feature))
    if denominator <= 0:
        raise ValueError("one-step polynomial feature has zero norm")
    fitted_mu = float(np.dot(feature, z_next - z) / denominator)
    fitted_prediction = z + fitted_mu * feature
    selected_prediction = np.asarray(spec.evaluate(z), dtype=np.float64)
    return {
        "unconstrained_least_squares_mu": fitted_mu,
        "least_squares_one_step_mse": float(
            np.mean((fitted_prediction - z_next) ** 2)
        ),
        "selected_mu": spec.mu,
        "selected_mu_one_step_mse": float(
            np.mean((selected_prediction - z_next) ** 2)
        ),
        "selection_note": (
            "the fitted value is diagnostic only; mu=0.75 is held fixed to "
            "isolate the effect of removing the rational denominator"
        ),
    }


def diagnose_dense_topology(
    spec: PolynomialMapSpec,
    bounds: LatentBounds,
    *,
    grid_points: int = DENSE_GRID_POINTS,
) -> dict[str, Any]:
    """Record local successes and global failures without rejecting the graph."""

    if grid_points < 1_001 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer of at least 1001")
    extent = float(max(abs(bounds.lower[0]), abs(bounds.upper[0])))
    grid = np.linspace(-extent, extent, grid_points, dtype=np.float64)
    spacing = float(grid[1] - grid[0])
    mapped = np.asarray(spec.evaluate(grid), dtype=np.float64)
    drift = mapped - grid
    derivative = np.asarray(spec.derivative(grid), dtype=np.float64)
    roots = base._dense_sign_change_roots(grid, drift)
    expected_roots = spec.fixed_points
    root_tolerance = 3.0 * spacing

    root_exclusion = 8.0 * spacing
    distances = np.min(
        np.abs(grid[:, None] - expected_roots[None, :]),
        axis=1,
    )
    drift_mask = distances > root_exclusion
    q = grid / spec.stable_root_magnitude
    expected_drift_sign = np.sign(q * (1.0 - q * q))
    signed_drift = expected_drift_sign[drift_mask] * drift[drift_mask]

    root_derivatives = np.asarray(
        spec.derivative(expected_roots),
        dtype=np.float64,
    )
    fixed_residuals = np.asarray(
        spec.drift(expected_roots),
        dtype=np.float64,
    )
    odd_residual = np.asarray(spec.evaluate(grid) + spec.evaluate(-grid))
    positive = grid > 0.0
    negative = grid < 0.0
    checks = {
        "dense_grid_has_exactly_three_fixed_points": bool(roots.shape == (3,)),
        "dense_roots_match_minus_a_zero_plus_a": bool(
            roots.shape == (3,)
            and np.allclose(roots, expected_roots, rtol=0.0, atol=root_tolerance)
        ),
        "fixed_point_residuals_near_machine_precision": bool(
            np.max(np.abs(fixed_residuals)) < 1e-12
        ),
        "oddness_near_machine_precision": bool(
            np.max(np.abs(odd_residual)) < 1e-12
        ),
        "correct_local_double_well_drift": bool(
            signed_drift.size > 0 and np.all(signed_drift > 0.0)
        ),
        "outer_fixed_points_strictly_stable": bool(
            abs(root_derivatives[0]) < 1.0
            and abs(root_derivatives[2]) < 1.0
        ),
        "origin_strictly_unstable": bool(root_derivatives[1] > 1.0),
        "strictly_positive_derivative_on_dense_domain": bool(
            np.min(derivative) > 0.0
        ),
        "map_preserves_nonzero_sign_on_dense_domain": bool(
            np.all(mapped[positive] > 0.0)
            and np.all(mapped[negative] < 0.0)
        ),
        "mapped_dense_domain_stays_inside_cmgdb_bounds": bool(
            np.min(mapped) >= float(bounds.lower[0])
            and np.max(mapped) <= float(bounds.upper[0])
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    return {
        "schema_version": 1,
        "formula": "G(z)=a*(q + mu*q*(1-q^2)), q=z/a",
        "parameters": asdict(spec),
        "dense_grid": {
            "points": grid_points,
            "lower": float(grid[0]),
            "upper": float(grid[-1]),
            "spacing": spacing,
            "normalized_maximum_absolute_q": float(np.max(np.abs(q))),
        },
        "fixed_points": {
            "expected": expected_roots.tolist(),
            "dense_sign_change_estimates": roots.tolist(),
            "matching_tolerance": root_tolerance,
            "absolute_residuals": np.abs(fixed_residuals).tolist(),
        },
        "derivatives": {
            "at_minus_a_zero_plus_a": root_derivatives.tolist(),
            "expected_at_outer_roots": 1.0 - 2.0 * spec.mu,
            "expected_at_origin": 1.0 + spec.mu,
            "dense_minimum": float(np.min(derivative)),
            "dense_maximum": float(np.max(derivative)),
            "normalized_fold_magnitude": spec.normalized_fold_magnitude,
            "fold_locations_z": [
                -spec.stable_root_magnitude * spec.normalized_fold_magnitude,
                spec.stable_root_magnitude * spec.normalized_fold_magnitude,
            ],
        },
        "sign_reversal": {
            "normalized_nonzero_G_zero_magnitude": (
                spec.normalized_sign_reversal_magnitude
            ),
            "nonzero_G_zero_locations_z": [
                -spec.stable_root_magnitude
                * spec.normalized_sign_reversal_magnitude,
                spec.stable_root_magnitude
                * spec.normalized_sign_reversal_magnitude,
            ],
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
            "root_exclusion_radius": root_exclusion,
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
        "all_checks_passed": not failed_checks,
        "cmgdb_was_not_precluded_by_failed_dense_checks": True,
    }


def _morse_set_intervals(path: Path) -> dict[str, Any]:
    """Summarize each one-dimensional Morse-set enclosure."""

    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(f"unexpected D1 morse_sets format in {path}: {rows.shape}")
    result: dict[str, Any] = {}
    node_ids = rows[:, 2].astype(np.int64)
    for node in sorted(np.unique(node_ids).tolist()):
        selected = rows[node_ids == node]
        result[str(node)] = {
            "cell_count": int(selected.shape[0]),
            "enclosure_lower": float(np.min(selected[:, 0])),
            "enclosure_upper": float(np.max(selected[:, 1])),
        }
    return result


def _comparability_payload() -> dict[str, Any]:
    return {
        "paper_eligible": False,
        "generic_autoencoder_result": False,
        "unbiased_test_result": False,
        "designation": EXPERIMENT_LABEL,
        "preserved_from_rational_ceiling": [
            "exact SHA256-verified archived inputs",
            "E(x)=x[:, 0] with float32-equivalent rounding",
            "a from the same two archived stable roots",
            "mu=0.75",
            "bounds from all 30,000 current and next training pairs",
            "10 percent bounds padding",
            "level-8 uniform 256-cell CMGDB graph",
            "CMGDB BoxMap padding=True",
            "native CMGDB.MorseSingletonReachability",
            "negative-first closed-cell classification",
            "same 7,862 conditioned-trajectory denominator",
        ],
        "single_intentional_map_change": (
            "Removed the /(1+q^2) saturation factor: "
            "G(z)=a*(q+mu*q*(1-q^2))."
        ),
        "known_global_failure": (
            "The cubic folds, reverses sign for sufficiently large |q|, and is "
            "unbounded outside the finite CMGDB domain."
        ),
        "valid_interpretation": (
            "An isolated limit test of how the unsaturated cubic changes the "
            "same fixed-coordinate, fixed-grid basin calculation."
        ),
        "invalid_interpretations": [
            "a trained or learned model result",
            "an unbiased held-out generalization result",
            "a paper-eligible comparison",
            "proof that the polynomial is globally topology preserving",
        ],
    }


def _write_comparability_markdown(
    output_dir: Path,
    comparability: dict[str, Any],
) -> Path:
    lines = [
        "# Chafee--Infante D1 polynomial physics-coordinate map",
        "",
        f"Designation: **{comparability['designation']}**.",
        "",
        "This output is not trained, unbiased, or paper-eligible.",
        "",
        "## Valid interpretation",
        "",
        comparability["valid_interpretation"],
        "",
        "## Preserved from the rational-map ceiling",
        "",
        *(f"- {item}" for item in comparability["preserved_from_rational_ceiling"]),
        "",
        "## Map change",
        "",
        comparability["single_intentional_map_change"],
        "",
        "## Known global failure",
        "",
        comparability["known_global_failure"],
        "",
        "## Invalid interpretations",
        "",
        *(f"- {item}" for item in comparability["invalid_interpretations"]),
        "",
    ]
    path = output_dir / "COMPARABILITY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _comparison_payload(statistics_core: dict[str, Any]) -> dict[str, Any]:
    polynomial_count = int(statistics_core["combined_correct"]["count"])
    polynomial_percentage = float(
        statistics_core["combined_correct"]["percentage"]
    )
    rational = base._baseline_comparison(
        RATIONAL_BASELINE_STATS,
        label="rational saturated physics-coordinate ceiling",
    )
    rational_comparison: dict[str, Any]
    if rational is None:
        rational_comparison = {
            "available": False,
            "path": str(RATIONAL_BASELINE_STATS.resolve()),
        }
    else:
        rational_count = int(rational["combined_correct_count"])
        rational_percentage = float(rational["combined_correct_percentage"])
        rational_comparison = {
            "available": True,
            "baseline": rational,
            "polynomial_minus_rational_count": polynomial_count - rational_count,
            "polynomial_minus_rational_percentage_points": (
                polynomial_percentage - rational_percentage
            ),
        }
    canonical = base._comparison_payload(statistics_core)
    return {
        "warning": (
            "All comparisons are descriptive only; this map was designed after "
            "examining archived labels and is not learned."
        ),
        "polynomial_combined_correct_count": polynomial_count,
        "polynomial_combined_correct_percentage": polynomial_percentage,
        "rational_ceiling": rational_comparison,
        "canonical_learned_baselines": canonical["baselines"],
    }


def run_experiment(
    *,
    output_dir: Path,
    archive_dir: Path,
    dense_grid_points: int = DENSE_GRID_POINTS,
) -> dict[str, Any]:
    """Run the isolated polynomial experiment."""

    output = base._assert_isolated_output(output_dir)
    started = time.perf_counter()
    inputs = reference.verify_exact_inputs(archive_dir)
    x, y = reference._load_training_pairs(inputs.train_data)
    physical_roots = reference._load_stable_roots(inputs.stable_roots)
    points, truth = reference._load_trajectory_labels(inputs.trajectory_labels)

    bounds, bounds_payload = base.infer_physics_bounds(x, y)
    encoded_roots = base.physics_encode(physical_roots)
    root_magnitude = base.stable_root_magnitude(encoded_roots)
    spec = PolynomialMapSpec(
        stable_root_magnitude=root_magnitude,
        mu=SELECTED_MU,
    )
    one_step_diagnostic = fit_one_step_mu_diagnostic(
        spec,
        base.physics_encode(x),
        base.physics_encode(y),
    )
    topology = diagnose_dense_topology(
        spec,
        bounds,
        grid_points=dense_grid_points,
    )

    output.mkdir(parents=True)
    uniform_dir = output / "MG_uniform_s8"
    uniform_dir.mkdir()
    base._write_json(output / "bounds.json", bounds_payload)
    base._write_json(output / "topology_diagnostics.json", topology)
    base._write_json(
        output / "one_step_fit_diagnostic.json",
        one_step_diagnostic,
    )

    box_map = PolynomialPaddedBoxMap(spec)
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
    morse_intervals = _morse_set_intervals(sets_path)

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
            "generic_autoencoder_result": False,
            "test_labels_informed_design": True,
        },
        "coordinate": {
            "formula": "E(x)=x_1=x[:, 0]",
            "source": "first Fourier/spectral coefficient",
            "learned": False,
            "encoded_stable_roots": encoded_roots.tolist(),
        },
        "polynomial_map": {
            "formula": "G(z)=a*(q + mu*q*(1-q^2)), q=z/a",
            "parameters": asdict(spec),
            "learned": False,
            "topology_diagnostics_path": "topology_diagnostics.json",
            "one_step_fit_diagnostic_path": "one_step_fit_diagnostic.json",
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
                {int(node) for node in morse_intervals}
                - set(attractors)
            ),
            "negative_attractor": negative_attractor,
            "positive_attractor": positive_attractor,
            "morse_set_intervals": morse_intervals,
            "queried_uniform_cells": int(unique_cell_ids.size),
            "compute_seconds": cmgdb_seconds,
            "reachability_query_seconds": query_seconds,
            "conley": conley_status,
            "polynomial_callback": {
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
        "shared_reference_driver": {
            "path": str(Path(base.__file__).resolve()),
            "sha256": base._sha256(Path(base.__file__).resolve()),
        },
        "runtime": base._runtime_metadata(),
        "inputs": inputs.provenance(),
        "parameters": {
            "coordinate": "x[:, 0] with float32-equivalent rounding",
            "polynomial_map": asdict(spec),
            "dense_grid_points": dense_grid_points,
            "cmgdb_subdivisions": [8, 8, 8],
            "cmgdb_padding": True,
            "cmgdb_subdiv_limit": reference.SUBDIV_LIMIT,
        },
        "duration_seconds": time.perf_counter() - started,
        "dense_topology_all_checks_passed": topology["all_checks_passed"],
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
        "basin_statistics": str(output / "basin_statistics.json"),
        "dense_topology_all_checks_passed": topology["all_checks_passed"],
        "failed_dense_checks": topology["failed_checks"],
        "morse_nodes": graph_summary["nodes"],
        "minimal_attractors": graph_summary["minimal_nodes"],
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
