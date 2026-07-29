r"""Freeze the exploratory unit-scale cubic ``mu`` sensitivity scan.

Every candidate uses the raw first Fourier coordinate and the same map family

.. math:: G_\mu(z)=z+\mu z(1-z^2).

This scan is explicitly post-hoc and test-informed: basin labels are evaluated
for every candidate.  It is a limit/sensitivity diagnostic, not unbiased
model selection.  The training-only least-squares value remains the primary
fit and is included even though its level-8 graph is invalid for basin
statistics.  The reported ``mu=0.35`` companion is simply the best valid
candidate in this frozen exploratory inventory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scripts import chafee_d1_physics_coordinate_ceiling as base
from scripts import chafee_d1_unit_scale_fitted_mu as fitted

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = CODE_ROOT / "output" / "exploratory_chafee_d1_unit_scale_mu_scan"
EXPERIMENT_LABEL = "post-hoc unit-scale cubic mu sensitivity scan"

GRID_MUS = (
    0.06,
    0.075,
    0.1,
    0.125,
    0.15,
    0.175,
    0.2,
    0.3,
    0.325,
    0.35,
    0.375,
    0.4,
    0.425,
    0.45,
    0.475,
    0.5,
    0.525,
    0.55,
    0.6,
    0.75,
)

# Frozen expectations make accidental changes in CMGDB semantics visible.
EXPECTED_RESULTS: dict[str, dict[str, Any]] = {
    "least_squares": {"nodes": 12, "valid": False},
    "0.06": {"nodes": 53, "valid": False},
    "0.075": {"nodes": 67, "valid": False},
    "0.1": {"nodes": 65, "valid": False},
    "0.125": {"nodes": 39, "valid": False},
    "0.15": {"nodes": 31, "valid": False},
    "0.175": {"nodes": 24, "valid": False},
    "0.2": {"nodes": 20, "valid": False},
    "0.3": {"nodes": 12, "valid": False},
    "0.325": {"nodes": 12, "valid": False},
    "0.35": {
        "nodes": 11,
        "valid": True,
        "correct": 6_356,
        "outside": 1_506,
        "wrong": 0,
    },
    "root_alignment": {
        "nodes": 11,
        "valid": True,
        "correct": 3_841,
        "outside": 4_021,
        "wrong": 0,
    },
    "0.375": {"nodes": 10, "valid": False},
    "0.4": {"nodes": 10, "valid": False},
    "0.425": {"nodes": 10, "valid": False},
    "0.45": {"nodes": 9, "valid": False},
    "0.475": {
        "nodes": 8,
        "valid": True,
        "correct": 6_343,
        "outside": 603,
        "wrong": 916,
    },
    "0.5": {
        "nodes": 8,
        "valid": True,
        "correct": 6_201,
        "outside": 542,
        "wrong": 1_119,
    },
    "0.525": {
        "nodes": 8,
        "valid": True,
        "correct": 6_129,
        "outside": 426,
        "wrong": 1_307,
    },
    "0.55": {
        "nodes": 8,
        "valid": True,
        "correct": 6_000,
        "outside": 467,
        "wrong": 1_395,
    },
    "0.6": {
        "nodes": 8,
        "valid": True,
        "correct": 5_780,
        "outside": 476,
        "wrong": 1_606,
    },
    "0.75": {
        "nodes": 7,
        "valid": True,
        "correct": 5_427,
        "outside": 1_015,
        "wrong": 1_420,
    },
}


def _candidate_inventory(
    *,
    least_squares_mu: float,
    encoded_positive_root: float,
) -> list[dict[str, Any]]:
    root_alignment_mu = (1.0 - encoded_positive_root) / (
        encoded_positive_root * (1.0 - encoded_positive_root**2)
    )
    candidates = [
        {
            "candidate_id": "least_squares",
            "mu": least_squares_mu,
            "role": "training-only residual minimizer",
            "test_informed": False,
        }
    ]
    for mu in GRID_MUS:
        candidates.append(
            {
                "candidate_id": str(mu),
                "mu": mu,
                "role": "post-hoc exploratory grid",
                "test_informed": True,
            }
        )
        if mu == 0.35:
            candidates.append(
                {
                    "candidate_id": "root_alignment",
                    "mu": root_alignment_mu,
                    "role": (
                        "diagnostic satisfying G(encoded positive PDE root)=1"
                    ),
                    "test_informed": True,
                }
            )
    ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise AssertionError("candidate ids must be unique")
    return candidates


def _evaluate_candidate(
    candidate: dict[str, Any],
    *,
    bounds: Any,
    resolution: Any,
    unique_cell_ids: NDArray[np.int64],
    inverse: NDArray[np.int64],
    point_candidate_count: int,
    point_cells: Any,
    root_cells: Any,
    truth: NDArray[np.int64],
    dense_grid_points: int,
) -> dict[str, Any]:
    spec = fitted.UnitScaleCubicSpec(mu=float(candidate["mu"]))
    topology = fitted.diagnose_dense_topology(
        spec,
        bounds,
        grid_points=dense_grid_points,
    )
    callback = fitted.UnitScalePaddedBoxMap(spec)
    started = time.perf_counter()
    morse_graph, map_graph, cmgdb_seconds, _ = fitted.reference._run_lookup_cmgdb(
        callback,
        bounds,
        subdiv_init=resolution.uniform_init,
        subdiv_min=resolution.uniform_min,
        subdiv_max=resolution.uniform_max,
        compute_conley=False,
    )
    if int(map_graph.num_vertices()) != resolution.uniform_cells:
        raise ValueError("unit-scale scan changed the expected uniform cell count")
    attractors = fitted.reference._morse_attractors(morse_graph)
    singleton_unique = fitted.reference._native_singleton_reachability(
        map_graph,
        morse_graph,
        unique_cell_ids,
    )
    singleton_candidates = singleton_unique[inverse]
    point_singletons = np.asarray(
        singleton_candidates[:point_candidate_count],
        dtype=np.int32,
    )
    root_singletons = np.asarray(
        singleton_candidates[point_candidate_count:],
        dtype=np.int32,
    )

    negative_attractor: int | None = None
    positive_attractor: int | None = None
    root_resolution_error: str | None = None
    if len(attractors) == 2:
        try:
            negative_attractor = fitted.reference._root_attractor_label(
                root_singletons,
                root_cells,
                0,
                attractors,
            )
            positive_attractor = fitted.reference._root_attractor_label(
                root_singletons,
                root_cells,
                1,
                attractors,
            )
        except ValueError as error:
            root_resolution_error = str(error)

    valid = bool(
        len(attractors) == 2
        and negative_attractor is not None
        and positive_attractor is not None
        and negative_attractor != positive_attractor
    )
    statistics: dict[str, Any] | None = None
    if valid:
        predicted = fitted.reference._point_basin_labels(
            point_singletons,
            point_cells,
            negative_attractor=negative_attractor,
            positive_attractor=positive_attractor,
        )
        _, statistics = base._statistics_payload(
            truth=truth,
            predicted=predicted,
            negative_attractor=negative_attractor,
            positive_attractor=positive_attractor,
        )

    return {
        **candidate,
        "map": "G(z)=z+mu*z*(1-z^2)",
        "morse_nodes": int(morse_graph.num_vertices()),
        "minimal_attractors": attractors,
        "negative_attractor": negative_attractor,
        "positive_attractor": positive_attractor,
        "root_resolution_error": root_resolution_error,
        "graph_valid_for_basin_statistics": valid,
        "statistics": statistics,
        "topology": {
            "all_archived_domain_checks_passed": topology[
                "all_archived_domain_checks_passed"
            ],
            "failed_checks": topology["failed_checks"],
            "fold_magnitude": topology["derivatives"]["fold_magnitude"],
            "sign_reversal_magnitude": topology["sign_reversal"][
                "nonzero_G_zero_magnitude"
            ],
        },
        "cmgdb_seconds": cmgdb_seconds,
        "total_evaluation_seconds": time.perf_counter() - started,
        "callback": {
            "box_calls": callback.box_calls,
            "batch_calls": callback.batch_calls,
            "scalar_evaluations": callback.scalar_evaluations,
        },
    }


def _validate_frozen_results(rows: list[dict[str, Any]]) -> None:
    if {str(row["candidate_id"]) for row in rows} != set(EXPECTED_RESULTS):
        raise ValueError("frozen unit-scale scan candidate inventory changed")
    for row in rows:
        candidate_id = str(row["candidate_id"])
        expected = EXPECTED_RESULTS[candidate_id]
        if int(row["morse_nodes"]) != int(expected["nodes"]):
            raise ValueError(f"Morse-node count changed for mu candidate {candidate_id}")
        if bool(row["graph_valid_for_basin_statistics"]) != bool(expected["valid"]):
            raise ValueError(f"graph validity changed for mu candidate {candidate_id}")
        if not expected["valid"]:
            if row["statistics"] is not None:
                raise ValueError(f"invalid candidate {candidate_id} has statistics")
            continue
        statistics = row["statistics"]
        if not isinstance(statistics, dict):
            raise ValueError(f"valid candidate {candidate_id} lacks statistics")
        counts = statistics["counts"]
        correct = int(statistics["combined_correct"]["count"])
        outside = int(counts["outside_both_basins"])
        wrong = int(counts["misclassified_in_negative_basin"]) + int(
            counts["misclassified_in_positive_basin"]
        )
        if (
            correct != int(expected["correct"])
            or outside != int(expected["outside"])
            or wrong != int(expected["wrong"])
        ):
            raise ValueError(f"basin counts changed for mu candidate {candidate_id}")


def run_scan(
    *,
    output_dir: Path,
    archive_dir: Path,
    dense_grid_points: int = fitted.DENSE_GRID_POINTS,
) -> dict[str, Any]:
    """Run and freeze every candidate in the exploratory inventory."""

    output = base._assert_isolated_output(output_dir)
    started = time.perf_counter()
    inputs = fitted.reference.verify_exact_inputs(archive_dir)
    x, y = fitted.reference._load_training_pairs(inputs.train_data)
    encoded_x = base.physics_encode(x)
    encoded_y = base.physics_encode(y)
    least_squares_spec, fit = fitted.fit_mu_least_squares(encoded_x, encoded_y)
    bounds, bounds_payload = base.infer_physics_bounds(x, y)

    roots = fitted.reference._load_stable_roots(inputs.stable_roots)
    encoded_roots = base.physics_encode(roots)
    points, truth = fitted.reference._load_trajectory_labels(
        inputs.trajectory_labels
    )
    encoded_points = base.physics_encode(points)
    resolution = fitted.reference.RESOLUTIONS[1]
    point_cells = fitted.reference._uniform_point_cells(
        encoded_points,
        bounds,
        resolution,
    )
    root_cells = fitted.reference._uniform_point_cells(
        encoded_roots,
        bounds,
        resolution,
    )
    candidate_cells = np.concatenate(
        (point_cells.flat_cell_ids, root_cells.flat_cell_ids)
    )
    unique_cell_ids, inverse = np.unique(candidate_cells, return_inverse=True)

    inventory = _candidate_inventory(
        least_squares_mu=least_squares_spec.mu,
        encoded_positive_root=float(encoded_roots[1, 0]),
    )
    rows = [
        _evaluate_candidate(
            candidate,
            bounds=bounds,
            resolution=resolution,
            unique_cell_ids=unique_cell_ids,
            inverse=inverse,
            point_candidate_count=point_cells.flat_cell_ids.size,
            point_cells=point_cells,
            root_cells=root_cells,
            truth=truth,
            dense_grid_points=dense_grid_points,
        )
        for candidate in inventory
    ]
    _validate_frozen_results(rows)

    valid_rows = [
        row for row in rows if row["graph_valid_for_basin_statistics"]
    ]
    best = max(
        valid_rows,
        key=lambda row: int(row["statistics"]["combined_correct"]["count"]),
    )
    if best["candidate_id"] != "0.35":
        raise ValueError("post-hoc best valid candidate changed from mu=0.35")

    selection = {
        "selection_status": "post_hoc_test_informed",
        "selection_rule": (
            "maximum combined-correct count among graph-valid candidates in "
            "this frozen exploratory inventory"
        ),
        "selected_candidate_id": best["candidate_id"],
        "selected_mu": best["mu"],
        "combined_correct": best["statistics"]["combined_correct"],
        "warning": (
            "The same archived labels were used to select and report this "
            "candidate; it is not an unbiased estimate."
        ),
    }
    payload = {
        "schema_version": 1,
        "experiment_label": EXPERIMENT_LABEL,
        "paper_eligible": False,
        "map": "G(z)=z+mu*z*(1-z^2)",
        "coordinate": "raw E(x)=x[:,0]; no normalization; a=1",
        "inputs": inputs.provenance(),
        "bounds": bounds_payload,
        "fit": fit,
        "encoded_stable_roots": encoded_roots[:, 0].tolist(),
        "candidate_count": len(rows),
        "rows": rows,
        "selection": selection,
        "sensitivity_conclusion": (
            "Level-8 graph validity and basin counts change discontinuously "
            "with mu; residual-optimal mu is not graph-valid, and nearby "
            "post-hoc values can alternate between valid and invalid."
        ),
    }

    output.mkdir(parents=True)
    base._write_json(output / "scan_results.json", payload)
    base._write_json(output / "selection.json", selection)
    run_manifest = {
        "schema_version": 1,
        "experiment_label": EXPERIMENT_LABEL,
        "paper_eligible": False,
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": base._sha256(Path(__file__).resolve()),
        },
        "shared_evaluator": {
            "path": str(Path(fitted.__file__).resolve()),
            "sha256": base._sha256(Path(fitted.__file__).resolve()),
        },
        "runtime": base._runtime_metadata(),
        "inputs": inputs.provenance(),
        "candidate_ids": [row["candidate_id"] for row in rows],
        "frozen_expectations_validated": True,
        "duration_seconds": time.perf_counter() - started,
        "primary_fit_graph_valid": rows[0]["graph_valid_for_basin_statistics"],
        "post_hoc_selection": selection,
    }
    base._write_json(output / "run_manifest.json", run_manifest)
    base._write_json(
        output / "artifact_manifest.json",
        base._artifact_manifest(output),
    )
    return {
        "output_dir": str(output),
        "candidate_count": len(rows),
        "valid_candidate_count": len(valid_rows),
        "invalid_candidate_count": len(rows) - len(valid_rows),
        "least_squares_mu": least_squares_spec.mu,
        "least_squares_graph_valid": rows[0][
            "graph_valid_for_basin_statistics"
        ],
        "post_hoc_selected_mu": selection["selected_mu"],
        "post_hoc_selected_combined_correct": selection["combined_correct"],
        "paper_eligible": False,
    }


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
    result = run_scan(
        output_dir=args.output_dir,
        archive_dir=args.archive_dir,
        dense_grid_points=args.dense_grid_points,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
