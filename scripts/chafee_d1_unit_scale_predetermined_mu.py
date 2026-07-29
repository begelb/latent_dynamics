r"""Evaluate a post-hoc predetermined strength for the unit-scale cubic map.

This companion to ``chafee_d1_unit_scale_fitted_mu.py`` holds

.. math::

    E(x)=x[:,0],\qquad G(z)=z+0.35z(1-z^2)

fixed and evaluates it with the same level-8 archived CMGDB protocol.  The
value ``mu=0.35`` was selected *after* an exploratory basin/topology sweep and
is therefore test-informed, post-hoc, and not a learned residual minimizer.
The output is isolated from the least-squares experiment and records both the
selected value and the true training-only least-squares diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scripts import chafee_d1_physics_coordinate_ceiling as base
from scripts import chafee_d1_unit_scale_fitted_mu as fitted

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    CODE_ROOT
    / "output"
    / "exploratory_chafee_d1_unit_scale_predetermined_mu_0_35"
)
PREDETERMINED_MU = 0.35
EXPERIMENT_LABEL = "post-hoc unit-scale cubic topology-strength probe, mu=0.35"


def predetermined_mu_diagnostic(
    encoded_x: NDArray[np.float64],
    encoded_y: NDArray[np.float64],
) -> tuple[fitted.UnitScaleCubicSpec, dict[str, Any]]:
    """Return the fixed map plus its fit loss and the LS reference value."""

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
    least_squares_mu = numerator / denominator
    selected = fitted.UnitScaleCubicSpec(mu=PREDETERMINED_MU)
    selected_prediction = np.asarray(selected.evaluate(z), dtype=np.float64)
    least_squares_prediction = z + least_squares_mu * feature

    return selected, {
        "schema_version": 1,
        "fit_role": "diagnostic only; the selected mu was not learned",
        "coordinate": "raw first Fourier coefficient E(x)=x[:,0]",
        "normalization": "none; a=1",
        "map": "G(z)=z+mu*z*(1-z^2)",
        "residual_equation": (
            "delta_z=(E(y)-E(x)) ~= mu*f(z), f(z)=z*(1-z^2)"
        ),
        "objective": "diagnostic min_mu sum_i (delta_z_i-mu*f(z_i))^2",
        "closed_form": "mu_LS=(f dot delta_z)/(f dot f)",
        "selection": (
            "mu=0.35 predetermined post-hoc after an exploratory "
            "topology/basin-strength sweep"
        ),
        "selection_status": "post_hoc_test_informed",
        "n_training_pairs": int(z.size),
        "test_labels_used_in_residual_diagnostic": False,
        "test_labels_informed_selected_mu": True,
        "stable_roots_used_in_residual_diagnostic": False,
        "feature_dot_delta": numerator,
        "feature_dot_feature": denominator,
        "least_squares_mu_diagnostic": least_squares_mu,
        "fitted_mu": PREDETERMINED_MU,
        "selected_mu": PREDETERMINED_MU,
        "one_step_mse": float(np.mean((selected_prediction - z_next) ** 2)),
        "selected_mu_one_step_mse": float(
            np.mean((selected_prediction - z_next) ** 2)
        ),
        "least_squares_one_step_mse": float(
            np.mean((least_squares_prediction - z_next) ** 2)
        ),
        "persistence_mu_zero_mse": float(np.mean(delta * delta)),
        "training_coordinate_range": {
            "z_minimum": float(np.min(z)),
            "z_maximum": float(np.max(z)),
            "z_next_minimum": float(np.min(z_next)),
            "z_next_maximum": float(np.max(z_next)),
        },
    }


def _post_hoc_comparability() -> dict[str, Any]:
    return {
        "paper_eligible": False,
        "designation": EXPERIMENT_LABEL,
        "learned_parameter_count": 0,
        "predetermined_mu": PREDETERMINED_MU,
        "mu_selection_status": "post_hoc_test_informed",
        "no_test_labels_used_in_residual_diagnostic": True,
        "test_labels_informed_mu_selection": True,
        "valid_interpretation": (
            "A deliberately post-hoc limit test of how stronger unit-scale "
            "cubic drift changes the fixed level-8 CMGDB graph."
        ),
        "invalid_interpretations": [
            "the least-squares residual minimizer",
            "an unbiased hyperparameter choice",
            "a held-out generalization result",
            "a paper-eligible comparison",
        ],
    }


def _rewrite_post_hoc_metadata(output: Path) -> None:
    """Replace fitted-run labels and rebuild the self-verifying manifest."""

    comparability = _post_hoc_comparability()
    base._write_json(output / "comparability.json", comparability)
    (output / "COMPARABILITY.md").write_text(
        "\n".join(
            [
        "# Unit-scale cubic, predetermined mu=0.35",
                "",
                f"Designation: **{EXPERIMENT_LABEL}**.",
                "",
                "This is an explicitly post-hoc, test-informed topology-strength probe.",
                "It is not the least-squares minimizer and is not paper-eligible.",
                "",
                "## Valid interpretation",
                "",
                comparability["valid_interpretation"],
                "",
                "## Invalid interpretations",
                "",
                *(
                    f"- {item}"
                    for item in comparability["invalid_interpretations"]
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    basin_path = output / "basin_statistics.json"
    basin = json.loads(basin_path.read_text(encoding="utf-8"))
    basin["experiment"] = {
        "label": EXPERIMENT_LABEL,
        "paper_eligible": False,
        "learned_parameter_count": 0,
        "test_labels_used_in_residual_diagnostic": False,
        "test_labels_informed_mu_selection": True,
    }
    basin["unit_scale_cubic_map"]["mu_learned"] = False
    basin["unit_scale_cubic_map"]["mu_selection"] = (
        "post-hoc predetermined topology-strength probe"
    )
    basin["comparability"] = comparability
    if isinstance(basin.get("comparison"), dict):
        basin["comparison"]["warning"] = (
            "Descriptive only: mu=0.35 was selected post-hoc after inspecting "
            "exploratory topology/basin outcomes."
        )
    base._write_json(basin_path, basin)

    comparison_path = output / "comparison.json"
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        comparison["warning"] = (
            "Descriptive only: mu=0.35 was selected post-hoc after inspecting "
            "exploratory topology/basin outcomes."
        )
        base._write_json(comparison_path, comparison)

    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment_label"] = EXPERIMENT_LABEL
    manifest["script"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": base._sha256(Path(__file__).resolve()),
    }
    manifest["shared_evaluator"] = {
        "path": str(Path(fitted.__file__).resolve()),
        "sha256": base._sha256(Path(fitted.__file__).resolve()),
    }
    manifest["fit"] = {
        "role": "diagnostic only",
        "least_squares_mu": json.loads(
            (output / "fit.json").read_text(encoding="utf-8")
        )["least_squares_mu_diagnostic"],
        "selected_mu": PREDETERMINED_MU,
        "selection_status": "post_hoc_test_informed",
        "test_labels_used_in_residual_diagnostic": False,
        "test_labels_informed_selected_mu": True,
    }
    manifest["parameters"]["mu"] = PREDETERMINED_MU
    manifest["parameters"]["mu_learned"] = False
    manifest["parameters"]["mu_selection"] = "post_hoc_test_informed"
    base._write_json(manifest_path, manifest)
    base._write_json(
        output / "artifact_manifest.json",
        base._artifact_manifest(output),
    )


def run_experiment(
    *,
    output_dir: Path,
    archive_dir: Path,
    dense_grid_points: int = fitted.DENSE_GRID_POINTS,
) -> dict[str, Any]:
    """Run the shared evaluator with the explicitly fixed parameter."""

    original_fit = fitted.fit_mu_least_squares
    fitted.fit_mu_least_squares = predetermined_mu_diagnostic
    try:
        result = fitted.run_experiment(
            output_dir=output_dir,
            archive_dir=archive_dir,
            dense_grid_points=dense_grid_points,
        )
    finally:
        fitted.fit_mu_least_squares = original_fit

    output = Path(result["output_dir"])
    _rewrite_post_hoc_metadata(output)
    result["experiment_label"] = EXPERIMENT_LABEL
    result["selected_mu"] = PREDETERMINED_MU
    result["mu_selection_status"] = "post_hoc_test_informed"
    result["fitted_mu"] = None
    result["paper_eligible"] = False
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=fitted.reference.DEFAULT_ARCHIVE_DIR,
    )
    parser.add_argument(
        "--dense-grid-points",
        type=int,
        default=fitted.DENSE_GRID_POINTS,
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
