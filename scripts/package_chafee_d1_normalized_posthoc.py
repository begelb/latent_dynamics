r"""Materialize and report the normalized scan's post-hoc ``mu=0.55`` result.

This companion never changes source models or prior experiment trees.  It adds
one detailed, explicitly test-informed evaluation beneath the normalized
experiment output, records theorem-aligned attracting-block audits, clarifies
the inherited exploratory provenance of ``mu=0.75``, and rebuilds the root
artifact manifest.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from scripts import chafee_d1_normalized_coordinate as experiment
from scripts import chafee_d1_physics_coordinate_ceiling as base

POSTHOC_MU = 0.55
POSTHOC_DIRNAME = "posthoc_best_mu_0_55"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _add_theorem_definition(
    audit: dict[str, Any],
    *,
    bounds: dict[str, Any],
) -> dict[str, Any]:
    audit["theorem_definition_on_latent_domain"] = (
        "tau(N_q,G)=inf_{z in N_q} dist(G(z), Z\\Int(N_q))"
    )
    audit["latent_domain_Z"] = {
        "lower": bounds["lower"],
        "upper": bounds["upper"],
    }
    audit["one_dimensional_equivalence_note"] = (
        "For these interior attracting intervals, the nearest points of "
        "Z\\Int(N_q) are the interval endpoints, so the analytic endpoint/"
        "critical-point computation equals the reported theorem quantity."
    )
    grid_width = (
        float(bounds["upper"][0]) - float(bounds["lower"][0])
    ) / 256.0
    audit["uniform_grid"] = {
        "cells": 256,
        "width_h": grid_width,
        "boxmap_padding": "one grid cell h on each side",
    }
    for node in audit["nodes"].values():
        tau = float(node["tau"]["tau"])
        node["numerical_grid_robustness"] = {
            "grid_width_h": grid_width,
            "boxmap_padding_cells_per_side": 1,
            "padded_invariance_margin_tau_minus_h": tau - grid_width,
            "tau_exceeds_one_padding_width": tau > grid_width,
            "interpretation": (
                "tau is the theorem quantity; tau-h is a separate diagnostic "
                "after accounting for one BoxMap padding cell per side"
            ),
        }
        clearances: dict[str, float | None] = {}
        for sign, membership in node["encoded_root_inclusion"].items():
            root = float(membership["value"])
            containing = [
                interval
                for interval in node["intervals"]
                if float(interval["lower"]) <= root <= float(interval["upper"])
            ]
            clearances[sign] = (
                min(
                    root - float(containing[0]["lower"]),
                    float(containing[0]["upper"]) - root,
                )
                if containing
                else None
            )
        node["encoded_root_to_nearest_block_boundary_clearance"] = clearances
    return audit


def _audit_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node, payload in sorted(audit["nodes"].items(), key=lambda item: int(item[0])):
        residual = payload["stored_pair_residuals_conditioned_on_E_x_in_N"]
        tau = payload["tau"]
        rows.append(
            {
                "node": int(node),
                "physical": payload["physical_attractor"],
                "cells": payload["forward_closure_cell_count"],
                "width": payload["total_interval_width"],
                "tau": tau["tau"],
                "accepted": residual["accepted_pairs"],
                "max_residual": residual["max_absolute_residual"],
                "ratio": residual["max_sample_over_tau"],
                "tau_minus_h": payload["numerical_grid_robustness"][
                    "padded_invariance_margin_tau_minus_h"
                ],
                "root_clearance": payload[
                    "encoded_root_to_nearest_block_boundary_clearance"
                ].get(payload["physical_attractor"]),
            }
        )
    return rows


def _report_markdown(
    *,
    residual: dict[str, Any],
    fitted: dict[str, Any],
    predetermined: dict[str, Any],
    posthoc: dict[str, Any],
    fitted_audit: dict[str, Any],
    predetermined_audit: dict[str, Any],
    posthoc_audit: dict[str, Any],
) -> str:
    fitted_stats = fitted["statistics"]
    predetermined_stats = predetermined["statistics"]
    posthoc_stats = posthoc["statistics"]

    lines = [
        "# Normalized Chafee--Infante D1 limit test",
        "",
        "## Setup",
        "",
        "The coordinate and map are",
        "",
        "\\[",
        "a=1.2365946,\\qquad E(x)=x_1/a,\\qquad "
        "G_\\mu(z)=z+\\mu z(1-z^2).",
        "\\]",
        "",
        "The encoded PDE roots are ±0.9999999658434106, within "
        "\\(3.42\\times10^{-8}\\) of the map roots ±1.",
        "",
        "## Residual versus basin topology",
        "",
        "| variant | μ | training MSE | Morse nodes | correct | outside | wrong |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| least-squares fit | {residual['fitted_mu']:.15g} | "
            f"{residual['fitted']['mse']:.12g} | {fitted['morse_nodes']} | "
            f"{fitted_stats['combined_correct']['count']}/7862 "
            f"({fitted_stats['combined_correct']['percentage']:.6f}%) | "
            f"{fitted_stats['counts']['outside_both_basins']} | 0 |"
        ),
        (
            f"| μ=0.75 inherited exploratory choice | 0.75 | "
            f"{residual['predetermined_mu_0_75']['mse']:.12g} | "
            f"{predetermined['morse_nodes']} | "
            f"{predetermined_stats['combined_correct']['count']}/7862 "
            f"({predetermined_stats['combined_correct']['percentage']:.6f}%) | "
            f"{predetermined_stats['counts']['outside_both_basins']} | "
            f"{predetermined_stats['counts']['misclassified_in_negative_basin'] + predetermined_stats['counts']['misclassified_in_positive_basin']} |"
        ),
        (
            f"| μ=0.55 post-hoc scan winner | 0.55 | "
            f"{posthoc['residual_metrics']['mse']:.12g} | "
            f"{posthoc['morse_nodes']} | "
            f"{posthoc_stats['combined_correct']['count']}/7862 "
            f"({posthoc_stats['combined_correct']['percentage']:.6f}%) | "
            f"{posthoc_stats['counts']['outside_both_basins']} | "
            f"{posthoc_stats['counts']['misclassified_in_negative_basin'] + posthoc_stats['counts']['misclassified_in_positive_basin']} |"
        ),
        "",
        "The least-squares value is genuinely the best one-step residual fit, "
        "yet it produces 26 Morse nodes and only 8.53% strict basin coverage. "
        "The post-hoc μ=0.55 map has roughly 4.11 times larger MSE but reaches "
        "99.35%. This is a direct residual-versus-topology disconnect.",
        "",
        "The basin rule is deliberately strict: a candidate cell is assigned "
        "only when its complete reachable Morse-node set is exactly one "
        "singleton attractor. With the fitted μ, the 24 nonminimal recurrent "
        "nodes create many intermediate reachable sets, so 7,191 conditioned "
        "points are classified outside even though there are two valid minima.",
        "",
        "## Theorem-aligned attracting-block audit",
        "",
        "For every minimal node, `N_q` is the cell-level forward closure from "
        "`attractor_cells`. In all rows below this closure equals the recurrent "
        "cells and forward invariance was verified. The margin is",
        "",
        "\\[",
        "\\tau(N_q,G)=\\inf_{z\\in N_q}\\operatorname{dist}"
        "(G(z),Z\\setminus\\operatorname{Int}N_q),",
        "\\]",
        "",
        "computed analytically from interval endpoints and every derivative-"
        "critical point of the cubic.",
        "",
        "The normalized uniform grid width is "
        "\\(h=(Z_{max}-Z_{min})/256=0.0151394755258\\). Since `BoxMap` "
        "pads one cell on each side, tau-h is also reported as a distinct "
        "numerical-robustness diagnostic; τ itself remains the theorem quantity.",
        "",
        "| variant | node/sign | cells | width | root clearance | tau | tau-h | stored pairs in N | sample max | max/tau |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, audit in (
        ("fitted", fitted_audit),
        ("μ=0.75", predetermined_audit),
        ("post-hoc μ=0.55", posthoc_audit),
    ):
        for row in _audit_rows(audit):
            lines.append(
                f"| {label} | {row['node']}/{row['physical']} | "
                f"{row['cells']} | {row['width']:.9g} | "
                f"{row['root_clearance']:.9g} | {row['tau']:.9g} | "
                f"{row['tau_minus_h']:.9g} | {row['accepted']} | "
                f"{row['max_residual']:.9g} | {row['ratio']:.6g} |"
            )
    lines.extend(
        [
            "",
            "Every sampled maximum exceeds its corresponding τ, so each stored "
            "witness directly contradicts the tolerance inequality for that "
            "block. More generally, a finite-sample maximum is only a lower "
            "bound on the true supremum: a sample maximum below τ would be "
            "inconclusive, while one above τ is a valid counterexample.",
            "",
            "## Provenance caveat",
            "",
            "The least-squares μ uses only the 30,000 training pairs. μ=0.75 "
            "was pre-specified for this normalized rerun but inherited from an "
            "earlier exploratory/test-informed design. μ=0.55 was selected "
            "post-hoc using these same archived basin labels. Neither the scan "
            "winner nor this report is an unbiased or paper-eligible result.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    root = experiment.DEFAULT_OUTPUT.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"normalized experiment output is absent: {root}")
    target = root / POSTHOC_DIRNAME
    if target.exists():
        raise FileExistsError(f"post-hoc materialization is fail-if-present: {target}")

    started = time.perf_counter()
    inputs = experiment.reference.verify_exact_inputs(
        experiment.reference.DEFAULT_ARCHIVE_DIR
    )
    x, y = experiment.reference._load_training_pairs(inputs.train_data)
    encoded_x = experiment.normalized_encode(x)
    encoded_y = experiment.normalized_encode(y)
    bounds, bounds_payload = experiment.infer_normalized_bounds(x, y)
    roots = experiment.reference._load_stable_roots(inputs.stable_roots)
    points, truth = experiment.reference._load_trajectory_labels(
        inputs.trajectory_labels
    )
    encoded_roots = experiment.normalized_encode(roots)
    context = experiment._build_context(
        bounds=bounds,
        encoded_points=experiment.normalized_encode(points),
        truth=truth,
        encoded_roots=encoded_roots,
        training_x=encoded_x,
        training_y=encoded_y,
    )
    result = experiment.evaluate_mu(
        POSTHOC_MU,
        context,
        role="post-hoc test-informed scan winner",
        test_informed=True,
        dense_grid_points=experiment.DENSE_GRID_POINTS,
        artifact_dir=target,
    )
    residual_metrics = experiment.residual_metrics(
        encoded_x,
        encoded_y,
        mu=POSTHOC_MU,
    )
    result["residual_metrics"] = residual_metrics

    # Add the theorem's ambient-domain notation to all three detailed audits.
    audits: dict[str, dict[str, Any]] = {}
    for directory in ("fitted_mu", "predetermined_mu_0_75", POSTHOC_DIRNAME):
        audit_path = root / directory / "attracting_block_audit.json"
        audit = _add_theorem_definition(
            _read_json(audit_path),
            bounds=bounds_payload,
        )
        base._write_json(audit_path, audit)
        audits[directory] = audit
        basin_path = root / directory / "basin_statistics.json"
        basin = _read_json(basin_path)
        basin["attracting_block_audit"] = audit
        if directory == POSTHOC_DIRNAME:
            basin["residual_metrics"] = residual_metrics
        if directory == "predetermined_mu_0_75":
            basin["role"] = (
                "pre-specified for this rerun; inherited exploratory/"
                "test-informed choice"
            )
            basin["test_informed"] = True
        base._write_json(basin_path, basin)

    base._write_json(target / "residual_metrics.json", residual_metrics)
    base._write_json(
        target / "run_manifest.json",
        {
            "schema_version": 1,
            "designation": "post-hoc test-informed scan winner",
            "paper_eligible": False,
            "mu": POSTHOC_MU,
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": base._sha256(Path(__file__).resolve()),
            },
            "shared_evaluator": {
                "path": str(Path(experiment.__file__).resolve()),
                "sha256": base._sha256(Path(experiment.__file__).resolve()),
            },
            "inputs": inputs.provenance(),
            "duration_seconds": time.perf_counter() - started,
            "primary_result": result["statistics"],
        },
    )

    comparison = _read_json(root / "comparison.json")
    comparison["predetermined_mu_0_75"]["selection_provenance"] = (
        "pre-specified for this normalized rerun; inherited from an earlier "
        "exploratory/test-informed choice"
    )
    comparison["post_hoc_best"]["artifact"] = POSTHOC_DIRNAME
    comparison["post_hoc_best"]["residual"] = residual_metrics
    comparison["post_hoc_best"]["full_statistics"] = result["statistics"]
    comparison["post_hoc_best"]["attracting_block_audit"] = (
        f"{POSTHOC_DIRNAME}/attracting_block_audit.json"
    )
    base._write_json(root / "comparison.json", comparison)

    comparability = _read_json(root / "comparability.json")
    comparability["predetermined_mu_0_75_provenance"] = (
        "pre-specified for this rerun; inherited exploratory/test-informed choice"
    )
    comparability["post_hoc_mu_0_55_is_test_informed"] = True
    base._write_json(root / "comparability.json", comparability)

    manifest = _read_json(root / "run_manifest.json")
    manifest["predetermined_mu_0_75_provenance"] = (
        "pre-specified for this rerun; inherited exploratory/test-informed choice"
    )
    manifest["post_hoc_materialization"] = {
        "mu": POSTHOC_MU,
        "path": POSTHOC_DIRNAME,
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": base._sha256(Path(__file__).resolve()),
        },
    }
    base._write_json(root / "run_manifest.json", manifest)

    residual_report = _read_json(root / "residual_report.json")
    fitted = _read_json(root / "fitted_mu" / "basin_statistics.json")
    predetermined = _read_json(
        root / "predetermined_mu_0_75" / "basin_statistics.json"
    )
    posthoc = _read_json(target / "basin_statistics.json")
    (root / "REPORT.md").write_text(
        _report_markdown(
            residual=residual_report,
            fitted=fitted,
            predetermined=predetermined,
            posthoc=posthoc,
            fitted_audit=audits["fitted_mu"],
            predetermined_audit=audits["predetermined_mu_0_75"],
            posthoc_audit=audits[POSTHOC_DIRNAME],
        ),
        encoding="utf-8",
    )

    base._write_json(
        root / "artifact_manifest.json",
        base._artifact_manifest(root),
    )
    return {
        "output_root": str(root),
        "post_hoc_dir": str(target),
        "mu": POSTHOC_MU,
        "morse_nodes": result["morse_nodes"],
        "statistics": result["statistics"],
        "residual_metrics": residual_metrics,
        "report": str(root / "REPORT.md"),
        "paper_eligible": False,
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
