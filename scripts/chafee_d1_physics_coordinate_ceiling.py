r"""Compute a test-informed analytic ceiling for the Chafee--Infante D1 basin score.

This is deliberately *not* an autoencoder experiment and is not eligible for
the paper's learned-latent comparison.  It asks a narrower exploratory
question: how well can the exact archived basin benchmark score when the
one-dimensional coordinate is the first Fourier coefficient and the scalar
dynamics are given the known odd double-well topology?

The coordinate and map are

.. math::

    E(x) = x_1,\qquad
    G(z) = a\left(q + \mu\frac{q(1-q^2)}{1+q^2}\right),\quad q=z/a,

where ``a`` is the magnitude of the first Fourier coefficient of the two
archived stable roots.  The selected ``mu=0.75`` gives exactly three fixed
points, globally correct drift, stable outer roots, an unstable origin, and a
strictly positive derivative.  It is intentionally much stronger than the
least-squares one-step fit: this run is a topology/certification ceiling, not a
time-``tau`` prediction model.

The basin computation preserves the comparable part of the canonical D1
workflow:

* exact SHA256-verified training pairs, trajectory labels, and stable roots;
* float32-equivalent coordinate evaluation;
* bounds from current and next training states with 10 percent padding;
* the level-8 uniform 256-cell CMGDB graph with ``padding=True``;
* native ``MorseSingletonReachability`` queries; and
* Marcio's negative-first closed-cell classification and 7,862-point
  conditioned denominator.

Outputs are fail-if-present and must not overlap the canonical study tree.
There is no neural checkpoint, decoder, adaptive graph, or Conley-index stage.
The generated comparability record makes those limitations explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import CMGDB
import numpy as np
import torch
from numpy.typing import NDArray

from latentdynamics.analysis.basin_statistics import (
    OUTSIDE,
    compute_chafee_basin_statistics,
)
from latentdynamics.analysis.morse import LatentBounds
from latentdynamics.viz import save_morse_graph_artifacts

reference = importlib.import_module(
    "scripts.chafee_latent_dimension_study"
    if __package__
    else "chafee_latent_dimension_study"
)

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    CODE_ROOT / "output" / "exploratory_chafee_d1_physics_coordinate_ceiling"
)
CANONICAL_4K_STATS = (
    reference.DEFAULT_OUTPUT_ROOT
    / "latent_1d"
    / "seed_0"
    / "basin_statistics.json"
)
CANONICAL_10K_STATS = (
    reference.DEFAULT_OUTPUT_ROOT
    / "latent_1d"
    / "seed_0_epoch_10000"
    / "basin_statistics.json"
)

SCHEMA_VERSION = 1
SELECTED_MU = 0.75
GLOBAL_MONOTONICITY_MU_LIMIT = 0.8
DENSE_GRID_POINTS = 200_001
EXPECTED_CONDITIONED_TRAJECTORIES = 7_862
EXPERIMENT_LABEL = "test-informed exploratory physics-coordinate ceiling"


@dataclass(frozen=True)
class AnalyticMapSpec:
    """Parameters and exact formulas for the odd three-fixed-point map."""

    stable_root_magnitude: float
    mu: float = SELECTED_MU

    def __post_init__(self) -> None:
        if not math.isfinite(self.stable_root_magnitude):
            raise ValueError("stable_root_magnitude must be finite")
        if self.stable_root_magnitude <= 0:
            raise ValueError("stable_root_magnitude must be positive")
        if not math.isfinite(self.mu):
            raise ValueError("mu must be finite")
        if not 0.0 < self.mu < GLOBAL_MONOTONICITY_MU_LIMIT:
            raise ValueError(
                "mu must lie in (0, 0.8) so the analytic map is globally "
                "strictly increasing"
            )

    @property
    def fixed_points(self) -> NDArray[np.float64]:
        a = self.stable_root_magnitude
        return np.asarray([-a, 0.0, a], dtype=np.float64)

    @property
    def theoretical_global_derivative_lower_bound(self) -> float:
        # h'(q) for h(q)=q(1-q^2)/(1+q^2) has global minimum -5/4.
        return 1.0 - 1.25 * self.mu

    def evaluate(
        self,
        values: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        z = np.asarray(values, dtype=np.float64)
        q = z / self.stable_root_magnitude
        correction = self.mu * q * (1.0 - q * q) / (1.0 + q * q)
        return self.stable_root_magnitude * (q + correction)

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
        q2 = q * q
        correction_derivative = (1.0 - 4.0 * q2 - q2 * q2) / (1.0 + q2) ** 2
        return 1.0 + self.mu * correction_derivative


class AnalyticPaddedBoxMap:
    """CMGDB scalar and batch callbacks for the analytic map."""

    def __init__(self, spec: AnalyticMapSpec) -> None:
        self.spec = spec
        self.scalar_evaluations = 0
        self.box_calls = 0
        self.batch_calls = 0

    def _point_map(self, point: Any) -> list[float]:
        values = np.asarray(point, dtype=np.float64).reshape(-1)
        if values.shape != (1,):
            raise ValueError(f"analytic D1 point must have shape (1,), got {values.shape}")
        self.scalar_evaluations += 1
        return [float(self.spec.evaluate(values[0]))]

    def __call__(self, rectangle: Any) -> list[float]:
        self.box_calls += 1
        return CMGDB.BoxMap(self._point_map, rectangle, padding=True)

    def batch(self, rectangles: Any) -> list[list[float]]:
        self.batch_calls += 1
        return [self(rectangle) for rectangle in rectangles]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved.is_relative_to(right_resolved)
        or right_resolved.is_relative_to(left_resolved)
    )


def _assert_isolated_output(output_dir: Path) -> Path:
    """Reject existing, canonical-overlapping, or archive-overlapping targets."""

    target = output_dir.resolve()
    protected = (
        reference.DEFAULT_OUTPUT_ROOT.resolve(),
        reference.DEFAULT_ARCHIVE_DIR.resolve(),
    )
    for root in protected:
        if _paths_overlap(target, root):
            raise ValueError(
                f"output target {target} overlaps protected canonical/input root {root}"
            )
    if target.exists():
        raise FileExistsError(
            f"output target already exists; ceiling runs are fail-if-present: {target}"
        )
    return target


def physics_encode(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the first Fourier coefficient with float32 encoder semantics."""

    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != reference.HIGH_DIMENSION:
        raise ValueError(
            "physical states must have shape "
            f"(n, {reference.HIGH_DIMENSION}); got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("physical states contain non-finite values")
    return np.asarray(array[:, :1], dtype=np.float32).astype(np.float64)


def infer_physics_bounds(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[LatentBounds, dict[str, Any]]:
    """Match canonical bounds semantics using the fixed physical coordinate."""

    encoded = np.concatenate((physics_encode(x), physics_encode(y)), axis=0)
    raw_lower = encoded.min(axis=0)
    raw_upper = encoded.max(axis=0)
    span = raw_upper - raw_lower
    if not np.all(span > 0):
        raise ValueError("physics-coordinate training range is degenerate")
    buffer = reference.BOUNDS_EPSILON_FRAC * span
    bounds = LatentBounds(lower=raw_lower - buffer, upper=raw_upper + buffer)
    return bounds, {
        "dimension": 1,
        "coordinate": "first Fourier coefficient x[:, 0]",
        "float_evaluation": "input coefficient rounded to float32, then stored as float64",
        "encoded_rows": ["training current x", "training next y"],
        "n_encoded_states": int(encoded.shape[0]),
        "raw_lower": raw_lower.tolist(),
        "raw_upper": raw_upper.tolist(),
        "epsilon_fraction": reference.BOUNDS_EPSILON_FRAC,
        "lower": bounds.lower.tolist(),
        "upper": bounds.upper.tolist(),
    }


def stable_root_magnitude(encoded_roots: NDArray[np.float64]) -> float:
    """Validate odd root order and return their common first-mode magnitude."""

    roots = np.asarray(encoded_roots, dtype=np.float64)
    if roots.shape != (2, 1):
        raise ValueError(f"encoded roots must have shape (2, 1), got {roots.shape}")
    if not roots[0, 0] < 0.0 < roots[1, 0]:
        raise ValueError("encoded roots must retain negative-then-positive order")
    scale = max(1.0, float(np.max(np.abs(roots))))
    odd_residual = abs(float(roots[0, 0] + roots[1, 0]))
    if odd_residual > 32.0 * np.finfo(np.float32).eps * scale:
        raise ValueError(
            f"stable first-mode roots are not odd within float32 tolerance: {roots.ravel()}"
        )
    return float(np.mean(np.abs(roots[:, 0])))


def fit_one_step_mu_diagnostic(
    spec: AnalyticMapSpec,
    encoded_x: NDArray[np.float64],
    encoded_y: NDArray[np.float64],
) -> dict[str, float]:
    """Fit the map strength to one-step pairs for diagnosis, not selection."""

    z = np.asarray(encoded_x, dtype=np.float64).reshape(-1)
    z_next = np.asarray(encoded_y, dtype=np.float64).reshape(-1)
    if z.shape != z_next.shape or z.size < 1:
        raise ValueError("encoded_x and encoded_y must be nonempty and equally shaped")
    a = spec.stable_root_magnitude
    q = z / a
    feature = a * q * (1.0 - q * q) / (1.0 + q * q)
    denominator = float(np.dot(feature, feature))
    if denominator <= 0:
        raise ValueError("one-step map feature has zero norm")
    fitted_mu = float(np.dot(feature, z_next - z) / denominator)
    fitted_prediction = z + fitted_mu * feature
    selected_prediction = np.asarray(spec.evaluate(z), dtype=np.float64)
    return {
        "unconstrained_least_squares_mu": fitted_mu,
        "least_squares_one_step_mse": float(np.mean((fitted_prediction - z_next) ** 2)),
        "selected_mu": spec.mu,
        "selected_mu_one_step_mse": float(
            np.mean((selected_prediction - z_next) ** 2)
        ),
        "selection_note": (
            "the fitted value is diagnostic only; mu=0.75 was chosen for a "
            "strong globally monotone topology/certification ceiling"
        ),
    }


def _dense_sign_change_roots(
    grid: NDArray[np.float64],
    drift: NDArray[np.float64],
) -> NDArray[np.float64]:
    if grid.ndim != 1 or drift.shape != grid.shape or grid.size < 3:
        raise ValueError("grid and drift must be equally shaped 1-D arrays")
    roots: list[float] = []
    exact = np.flatnonzero(drift == 0.0)
    roots.extend(float(grid[index]) for index in exact.tolist())
    crossings = np.flatnonzero(drift[:-1] * drift[1:] < 0.0)
    for index in crossings.tolist():
        left = float(grid[index])
        right = float(grid[index + 1])
        left_value = float(drift[index])
        right_value = float(drift[index + 1])
        roots.append(
            left
            - left_value * (right - left) / (right_value - left_value)
        )
    roots.sort()
    spacing = float(grid[1] - grid[0])
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 2.0 * spacing:
            unique.append(root)
    return np.asarray(unique, dtype=np.float64)


def validate_dense_topology(
    spec: AnalyticMapSpec,
    bounds: LatentBounds,
    *,
    grid_points: int = DENSE_GRID_POINTS,
) -> dict[str, Any]:
    """Numerically and analytically validate roots, derivatives, oddness, and drift."""

    if grid_points < 1_001 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer of at least 1001")
    extent = float(max(abs(bounds.lower[0]), abs(bounds.upper[0])))
    grid = np.linspace(-extent, extent, grid_points, dtype=np.float64)
    spacing = float(grid[1] - grid[0])
    mapped = spec.evaluate(grid)
    drift = mapped - grid
    derivative = spec.derivative(grid)
    roots = _dense_sign_change_roots(grid, drift)
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
    checks = {
        "dense_grid_has_exactly_three_roots": bool(roots.shape == (3,)),
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
        "correct_drift_away_from_fixed_points": bool(
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
        "strictly_positive_theoretical_global_derivative_bound": bool(
            spec.theoretical_global_derivative_lower_bound > 0.0
        ),
        "map_preserves_nonzero_sign_on_dense_domain": bool(
            np.all(mapped[grid > 0.0] > 0.0)
            and np.all(mapped[grid < 0.0] < 0.0)
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"analytic topology validation failed: {failed}")

    return {
        "schema_version": 1,
        "formula": "G(z)=a*(q + mu*q*(1-q^2)/(1+q^2)), q=z/a",
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
            "at_minus_a_zero_plus_a": root_derivatives.tolist(),
            "expected_at_outer_roots": 1.0 - spec.mu,
            "expected_at_origin": 1.0 + spec.mu,
            "dense_minimum": float(np.min(derivative)),
            "dense_maximum": float(np.max(derivative)),
            "theoretical_global_lower_bound": (
                spec.theoretical_global_derivative_lower_bound
            ),
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
        "all_checks_passed": True,
    }


def _runtime_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cmgdb_module": str(Path(CMGDB.__file__).resolve()),
    }


def _statistics_payload(
    *,
    truth: NDArray[np.int64],
    predicted: NDArray[np.int32],
    negative_attractor: int,
    positive_attractor: int,
) -> tuple[Any, dict[str, Any]]:
    statistics = compute_chafee_basin_statistics(
        truth,
        predicted,
        negative_basin_label=negative_attractor,
        positive_basin_label=positive_attractor,
    )
    if statistics.conditioned_trajectories != EXPECTED_CONDITIONED_TRAJECTORIES:
        raise ValueError(
            "conditioned denominator changed: "
            f"{statistics.conditioned_trajectories} "
            f"!= {EXPECTED_CONDITIONED_TRAJECTORIES}"
        )
    combined_correct = (
        statistics.correctly_classified_in_negative_basin
        + statistics.correctly_classified_in_positive_basin
    )
    return statistics, {
        "total_trajectories": statistics.total_trajectories,
        "excluded_zero_trajectories": statistics.excluded_zero_trajectories,
        "conditioned_trajectories": statistics.conditioned_trajectories,
        "counts": statistics.counts(),
        "percentages": statistics.percentages(),
        "combined_correct": {
            "count": combined_correct,
            "percentage": (
                100.0 * combined_correct / statistics.conditioned_trajectories
            ),
        },
    }


def _baseline_comparison(path: Path, *, label: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    statistics = payload.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError(f"baseline lacks statistics object: {path}")
    denominator = int(statistics["conditioned_trajectories"])
    if denominator != EXPECTED_CONDITIONED_TRAJECTORIES:
        raise ValueError(f"baseline denominator changed in {path}: {denominator}")
    counts = statistics["counts"]
    combined = int(counts["correctly_classified_in_negative_basin"]) + int(
        counts["correctly_classified_in_positive_basin"]
    )
    return {
        "label": label,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "combined_correct_count": combined,
        "combined_correct_percentage": 100.0 * combined / denominator,
    }


def _comparison_payload(
    ceiling_statistics: dict[str, Any],
) -> dict[str, Any]:
    ceiling_count = int(ceiling_statistics["combined_correct"]["count"])
    ceiling_percentage = float(
        ceiling_statistics["combined_correct"]["percentage"]
    )
    comparisons: dict[str, Any] = {}
    for key, path, label in (
        ("canonical_4k", CANONICAL_4K_STATS, "canonical learned D1, 4,000 epochs"),
        ("canonical_10k", CANONICAL_10K_STATS, "learned D1, 10,000 epochs"),
    ):
        baseline = _baseline_comparison(path, label=label)
        if baseline is None:
            comparisons[key] = {"available": False, "path": str(path.resolve())}
            continue
        baseline_count = int(baseline["combined_correct_count"])
        baseline_percentage = float(
            baseline["combined_correct_percentage"]
        )
        comparisons[key] = {
            "available": True,
            "baseline": baseline,
            "ceiling_minus_baseline_count": ceiling_count - baseline_count,
            "ceiling_minus_baseline_percentage_points": (
                ceiling_percentage - baseline_percentage
            ),
            "ceiling_to_baseline_correct_count_ratio": (
                ceiling_count / baseline_count
            ),
        }
    return {
        "warning": (
            "These deltas are descriptive only. The ceiling was designed after "
            "examining the archived test labels and is not an unbiased learned-model "
            "comparison."
        ),
        "ceiling_combined_correct_count": ceiling_count,
        "ceiling_combined_correct_percentage": ceiling_percentage,
        "baselines": comparisons,
    }


def _comparability_payload() -> dict[str, Any]:
    return {
        "paper_eligible": False,
        "generic_autoencoder_result": False,
        "unbiased_test_result": False,
        "designation": EXPERIMENT_LABEL,
        "why_test_informed": (
            "The archived test labels were inspected while choosing the first "
            "Fourier coefficient as the ceiling coordinate."
        ),
        "preserved_from_canonical_d1_basin_table": [
            "exact SHA256-verified archived inputs",
            "all 10,000 archived initial conditions and labels",
            "conditioning on the exact 7,862 nonzero labels",
            "float32-equivalent coordinate evaluation",
            "bounds inferred from all 30,000 current and next training pairs",
            "10 percent bounds padding",
            "uniform CMGDB levels (8, 8, 8), giving 256 cells",
            "CMGDB BoxMap padding=True",
            "native CMGDB.MorseSingletonReachability",
            "complete reachable Morse-node set must equal a singleton attractor",
            "negative-basin-first closed-cell boundary rule",
            "same five count categories and percentage denominator",
        ],
        "intentional_differences": [
            "E(x)=x_1 is fixed physics, not a trained encoder",
            "G is an analytic odd double-well map, not a trained MLP",
            "mu=0.75 prioritizes topology and certification over time-tau fit",
            "there is no decoder, reconstruction loss, optimizer, or checkpoint",
            "the analytic callback is evaluated directly rather than through a "
            "persisted neural corner table",
            "only the uniform graph required for the basin table is computed",
            "there is no adaptive Morse graph or Conley-index annotation",
        ],
        "valid_interpretation": (
            "A problem-specific, test-informed ceiling showing what the fixed "
            "D1 CMGDB basin statistic can achieve with an almost ideal physical "
            "reaction coordinate and enforced three-fixed-point topology."
        ),
        "invalid_interpretations": [
            "evidence that a generic one-dimensional autoencoder learns this score",
            "an unbiased held-out generalization result",
            "a paper result directly comparable as a trained-model row",
            "an accurate learned surrogate of the time-0.1 PDE map",
        ],
    }


def _write_comparability_markdown(
    output_dir: Path,
    comparability: dict[str, Any],
) -> Path:
    lines = [
        "# Chafee--Infante D1 physics-coordinate ceiling",
        "",
        f"Designation: **{comparability['designation']}**.",
        "",
        "This output is not a trained autoencoder run and is not paper-eligible. "
        "It was designed after inspecting the archived test labels.",
        "",
        "## Valid interpretation",
        "",
        comparability["valid_interpretation"],
        "",
        "## Preserved basin-table semantics",
        "",
        *(f"- {item}" for item in comparability["preserved_from_canonical_d1_basin_table"]),
        "",
        "## Intentional differences",
        "",
        *(f"- {item}" for item in comparability["intentional_differences"]),
        "",
        "## Invalid interpretations",
        "",
        *(f"- {item}" for item in comparability["invalid_interpretations"]),
        "",
    ]
    path = output_dir / "COMPARABILITY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _artifact_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files[str(path.relative_to(output_dir))] = {
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
    return {
        "schema_version": 1,
        "root": str(output_dir.resolve()),
        "self_excluded": "artifact_manifest.json",
        "files": files,
    }


def run_experiment(
    *,
    output_dir: Path,
    archive_dir: Path,
    dense_grid_points: int = DENSE_GRID_POINTS,
) -> dict[str, Any]:
    """Run the isolated ceiling and return its compact result summary."""

    output = _assert_isolated_output(output_dir)
    started = time.perf_counter()
    inputs = reference.verify_exact_inputs(archive_dir)
    x, y = reference._load_training_pairs(inputs.train_data)
    physical_roots = reference._load_stable_roots(inputs.stable_roots)
    points, truth = reference._load_trajectory_labels(inputs.trajectory_labels)

    bounds, bounds_payload = infer_physics_bounds(x, y)
    encoded_roots = physics_encode(physical_roots)
    root_magnitude = stable_root_magnitude(encoded_roots)
    spec = AnalyticMapSpec(
        stable_root_magnitude=root_magnitude,
        mu=SELECTED_MU,
    )
    one_step_diagnostic = fit_one_step_mu_diagnostic(
        spec,
        physics_encode(x),
        physics_encode(y),
    )
    topology = validate_dense_topology(
        spec,
        bounds,
        grid_points=dense_grid_points,
    )

    output.mkdir(parents=True)
    uniform_dir = output / "MG_uniform_s8"
    uniform_dir.mkdir()
    _write_json(output / "bounds.json", bounds_payload)
    _write_json(output / "topology_validation.json", topology)
    _write_json(output / "one_step_fit_diagnostic.json", one_step_diagnostic)

    box_map = AnalyticPaddedBoxMap(spec)
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
    if int(map_graph.num_vertices()) != resolution.uniform_cells:
        raise ValueError(
            f"uniform CMGDB returned {int(map_graph.num_vertices())} cells; "
            f"expected {resolution.uniform_cells}"
        )
    attractors = reference._require_exactly_two_minimal_attractors(morse_graph)
    dot_path, sets_path = save_morse_graph_artifacts(morse_graph, uniform_dir)

    encoded_points = physics_encode(points)
    point_cells = reference._uniform_point_cells(encoded_points, bounds, resolution)
    root_cells = reference._uniform_point_cells(encoded_roots, bounds, resolution)
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
    if negative_attractor == positive_attractor:
        raise ValueError("physical stable roots resolved to the same attractor")
    predicted = reference._point_basin_labels(
        point_singletons,
        point_cells,
        negative_attractor=negative_attractor,
        positive_attractor=positive_attractor,
    )
    statistics, statistics_core = _statistics_payload(
        truth=truth,
        predicted=predicted,
        negative_attractor=negative_attractor,
        positive_attractor=positive_attractor,
    )

    np.save(output / "encoded_stable_roots.npy", encoded_roots)
    np.save(output / "trajectory_basin_labels.npy", predicted)
    np.savez_compressed(
        uniform_dir / "marcio_singleton_reachability_queries.npz",
        queried_cell_ids=unique_cell_ids,
        singleton_node_by_queried_cell=singleton_by_unique_cell,
        point_candidate_cell_ids=point_cells.flat_cell_ids,
        point_candidate_offsets=point_cells.offsets,
        point_singleton_nodes=point_singletons,
        point_basin_labels=predicted,
        root_candidate_cell_ids=root_cells.flat_cell_ids,
        root_candidate_offsets=root_cells.offsets,
        root_singleton_nodes=root_singletons,
        encoded_stable_roots=encoded_roots,
    )

    comparability = _comparability_payload()
    comparison = _comparison_payload(statistics_core)
    graph_summary = reference._morse_summary(dot_path)
    predicted_counts = {
        str(label): int(count)
        for label, count in sorted(Counter(predicted.tolist()).items())
    }
    basin_payload = {
        "schema_version": SCHEMA_VERSION,
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
        "analytic_map": {
            "formula": "G(z)=a*(q + mu*q*(1-q^2)/(1+q^2)), q=z/a",
            "parameters": asdict(spec),
            "learned": False,
            "topology_validation_path": "topology_validation.json",
            "one_step_fit_diagnostic_path": "one_step_fit_diagnostic.json",
        },
        "inputs": inputs.provenance(),
        "bounds": bounds_payload,
        "trajectory_data": {
            "total": reference.TRAJECTORY_ROWS,
            "label_counts": {
                str(label): count
                for label, count in reference.EXPECTED_TRAJECTORY_LABEL_COUNTS.items()
            },
        },
        "cmgdb": {
            "subdivisions": [
                resolution.uniform_init,
                resolution.uniform_min,
                resolution.uniform_max,
            ],
            "uniform_cells": resolution.uniform_cells,
            "padding": True,
            "subdiv_limit": reference.SUBDIV_LIMIT,
            "morse_nodes": int(morse_graph.num_vertices()),
            "attractor_nodes": attractors,
            "negative_attractor": negative_attractor,
            "positive_attractor": positive_attractor,
            "queried_uniform_cells": int(unique_cell_ids.size),
            "compute_seconds": cmgdb_seconds,
            "reachability_query_seconds": query_seconds,
            "conley": conley_status,
            "analytic_callback": {
                "box_calls": box_map.box_calls,
                "batch_calls": box_map.batch_calls,
                "scalar_evaluations": box_map.scalar_evaluations,
            },
            "graph_summary": graph_summary,
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
    _write_json(output / "basin_statistics.json", basin_payload)
    _write_json(output / "comparison.json", comparison)
    _write_json(output / "comparability.json", comparability)
    _write_comparability_markdown(output, comparability)

    run_manifest = {
        "schema_version": 1,
        "experiment_label": EXPERIMENT_LABEL,
        "output_dir": str(output),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "runtime": _runtime_metadata(),
        "inputs": inputs.provenance(),
        "parameters": {
            "coordinate": "x[:, 0] with float32-equivalent rounding",
            "analytic_map": asdict(spec),
            "dense_grid_points": dense_grid_points,
            "cmgdb_subdivisions": [8, 8, 8],
            "cmgdb_padding": True,
            "cmgdb_subdiv_limit": reference.SUBDIV_LIMIT,
        },
        "duration_seconds": time.perf_counter() - started,
        "primary_result": statistics_core,
        "comparability_path": "COMPARABILITY.md",
    }
    _write_json(output / "run_manifest.json", run_manifest)
    _write_json(output / "artifact_manifest.json", _artifact_manifest(output))

    del statistics, morse_graph, map_graph
    return {
        "output_dir": str(output),
        "basin_statistics": str(output / "basin_statistics.json"),
        "combined_correct_count": statistics_core["combined_correct"]["count"],
        "combined_correct_percentage": statistics_core["combined_correct"][
            "percentage"
        ],
        "counts": statistics_core["counts"],
        "percentages": statistics_core["percentages"],
        "morse_nodes": graph_summary["nodes"],
        "minimal_attractors": graph_summary["minimal_nodes"],
        "topology_checks_passed": topology["all_checks_passed"],
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
