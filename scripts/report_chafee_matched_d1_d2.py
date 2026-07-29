#!/usr/bin/env python3
"""Compare the matched 5x3 Chafee--Infante D1 and archived D2 packages.

This script is reporting-only.  It does not train models or recompute Morse
graphs.  The five training datasets are matched across dimensions, but the
three runs within a dataset are intentionally *not* paired: D1 has recorded
training seeds 0--2, whereas the archived D2 trial numbers are labels whose
random-number-generator seeds were not recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_D1_PACKAGE = CODE_ROOT / "output" / "chafee_d1_matched_d2_archive_5x3_roa_v1"
DEFAULT_D2_PACKAGE = CODE_ROOT / "output" / "chafee_d2_archive_5x3_roa_v1"
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "output" / "chafee_d1_d2_matched_comparison_v1"

EXPECTED_DATASETS = (1, 2, 3, 4, 5)
PAIRING_STATEMENT = (
    "The five training datasets are matched across dimensions, but D1 training "
    "seeds are not paired to D2 trial IDs. D2 trial IDs are archival labels, not "
    "recorded RNG seeds; all D1-D2 deltas are dataset- or dimension-level "
    "aggregate contrasts, never cellwise paired differences."
)

CORRECT_NEGATIVE = "correctly_classified_in_negative_basin"
CORRECT_POSITIVE = "correctly_classified_in_positive_basin"
MISCLASSIFIED_NEGATIVE = "misclassified_in_negative_basin"
MISCLASSIFIED_POSITIVE = "misclassified_in_positive_basin"
OUTSIDE = "outside_both_basins"
COMBINED_CORRECT = "combined_correct"
TOTAL_MISCLASSIFIED = "total_misclassified"

BASE_COUNT_FIELDS = (
    CORRECT_NEGATIVE,
    CORRECT_POSITIVE,
    MISCLASSIFIED_NEGATIVE,
    MISCLASSIFIED_POSITIVE,
    OUTSIDE,
)
REPORT_METRICS = (
    COMBINED_CORRECT,
    CORRECT_NEGATIVE,
    CORRECT_POSITIVE,
    TOTAL_MISCLASSIFIED,
    MISCLASSIFIED_NEGATIVE,
    MISCLASSIFIED_POSITIVE,
    OUTSIDE,
)
METRIC_LABELS = {
    COMBINED_CORRECT: "Correct",
    CORRECT_NEGATIVE: "Correct negative",
    CORRECT_POSITIVE: "Correct positive",
    TOTAL_MISCLASSIFIED: "Misclassified total",
    MISCLASSIFIED_NEGATIVE: "Misclassified negative",
    MISCLASSIFIED_POSITIVE: "Misclassified positive",
    OUTSIDE: "Outside both basins",
}


@dataclass(frozen=True)
class PackageSpec:
    key: str
    label: str
    dimension: int
    replicate_field: str
    replicate_kind: str
    replicate_prefix: str
    expected_replicates: tuple[int, int, int]


D1_SPEC = PackageSpec(
    key="d1",
    label="D1",
    dimension=1,
    replicate_field="training_seed",
    replicate_kind="recorded_training_seed",
    replicate_prefix="s",
    expected_replicates=(0, 1, 2),
)
D2_SPEC = PackageSpec(
    key="d2",
    label="D2",
    dimension=2,
    replicate_field="training_trial",
    replicate_kind="archived_unseeded_training_trial",
    replicate_prefix="t",
    expected_replicates=(1, 2, 3),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_results(package_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results_path = package_root / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"package results are missing: {results_path}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{results_path} must contain a JSON object")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise TypeError(f"{results_path} must contain a results list")
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"{results_path} contains a non-object result row")
    return payload, rows


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not a boolean")
    integer = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    return integer


def _finite_float(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _missing_row(dataset: int, replicate: int, spec: PackageSpec) -> dict[str, Any]:
    return {
        "dataset": dataset,
        spec.replicate_field: replicate,
        "status": "missing_result_row",
        "_synthetic_missing_row": True,
    }


def _normalize_row(
    source: Mapping[str, Any],
    *,
    spec: PackageSpec,
    dataset: int,
    replicate: int,
) -> dict[str, Any]:
    status = str(source.get("status", "missing")).strip().lower()
    has_analyzable_result = status in {"complete", "invalid"}
    statistics_reasons: list[str] = []
    topology_reasons: list[str] = []
    counts: dict[str, int | None] = dict.fromkeys(REPORT_METRICS)
    percentages: dict[str, float | None] = dict.fromkeys(REPORT_METRICS)
    conditioned: int | None = None

    if not has_analyzable_result:
        _append_reason(statistics_reasons, f"run_status:{status}")
        _append_reason(topology_reasons, f"run_status:{status}")
    else:
        try:
            conditioned = _integer(
                source["conditioned_trajectories"],
                field="conditioned_trajectories",
            )
            if conditioned <= 0:
                raise ValueError("conditioned_trajectories must be positive")
        except (KeyError, TypeError, ValueError) as error:
            _append_reason(statistics_reasons, f"invalid_denominator:{error}")

        for name in BASE_COUNT_FIELDS:
            try:
                count = _integer(source[name], field=name)
                if count < 0:
                    raise ValueError(f"{name} must be nonnegative")
                counts[name] = count
            except (KeyError, TypeError, ValueError) as error:
                _append_reason(statistics_reasons, f"invalid_count:{name}:{error}")

        if all(counts[name] is not None for name in BASE_COUNT_FIELDS):
            counts[COMBINED_CORRECT] = int(counts[CORRECT_NEGATIVE] or 0) + int(
                counts[CORRECT_POSITIVE] or 0
            )
            counts[TOTAL_MISCLASSIFIED] = int(counts[MISCLASSIFIED_NEGATIVE] or 0) + int(
                counts[MISCLASSIFIED_POSITIVE] or 0
            )
            if conditioned is not None:
                category_total = sum(int(counts[name] or 0) for name in BASE_COUNT_FIELDS)
                if category_total != conditioned:
                    _append_reason(
                        statistics_reasons,
                        f"count_conservation:{category_total}!={conditioned}",
                    )

        if counts[COMBINED_CORRECT] is not None:
            try:
                reported_correct = _integer(
                    source["combined_correct_count"],
                    field="combined_correct_count",
                )
                if reported_correct != counts[COMBINED_CORRECT]:
                    _append_reason(
                        statistics_reasons,
                        "combined_correct_count_disagrees_with_categories",
                    )
            except (KeyError, TypeError, ValueError) as error:
                _append_reason(statistics_reasons, f"invalid_combined_correct_count:{error}")

        if conditioned is not None and conditioned > 0:
            for name in REPORT_METRICS:
                count = counts[name]
                if count is not None:
                    percentages[name] = 100.0 * count / conditioned

        source_percentage_fields = {
            **{name: f"{name}_percentage" for name in BASE_COUNT_FIELDS},
            COMBINED_CORRECT: "combined_correct_percentage",
        }
        for name, source_field in source_percentage_fields.items():
            try:
                reported = _finite_float(source[source_field], field=source_field)
                derived = percentages[name]
                if derived is None or not math.isclose(
                    reported,
                    derived,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                ):
                    _append_reason(
                        statistics_reasons,
                        f"percentage_disagreement:{source_field}",
                    )
            except (KeyError, TypeError, ValueError) as error:
                _append_reason(
                    statistics_reasons,
                    f"invalid_percentage:{source_field}:{error}",
                )

        root_status = str(source.get("root_association_status", "missing")).lower()
        if root_status != "valid":
            _append_reason(
                topology_reasons,
                f"root_association_status:{root_status}",
            )
        try:
            attractor_count = _integer(source["attractor_count"], field="attractor_count")
            if attractor_count != 2:
                _append_reason(
                    topology_reasons,
                    f"attractor_count:{attractor_count}",
                )
        except (KeyError, TypeError, ValueError) as error:
            _append_reason(topology_reasons, f"invalid_attractor_count:{error}")

    dataset_seed: int | None
    try:
        raw_dataset_seed = source.get("dataset_initial_condition_seed")
        dataset_seed = (
            None
            if raw_dataset_seed is None
            else _integer(raw_dataset_seed, field="dataset_initial_condition_seed")
        )
    except (TypeError, ValueError):
        dataset_seed = None

    return {
        "dimension": spec.dimension,
        "dimension_label": spec.label,
        "dataset": dataset,
        "dataset_initial_condition_seed": dataset_seed,
        "replicate_kind": spec.replicate_kind,
        "replicate_id": replicate,
        "replicate_label": f"{spec.replicate_prefix}{replicate}",
        "input_status": status,
        "result_row_present": not bool(source.get("_synthetic_missing_row", False)),
        "statistics_valid": not statistics_reasons,
        "statistics_failure_reasons": statistics_reasons,
        "topology_valid": not topology_reasons,
        "topology_failure_reasons": topology_reasons,
        "reportable_valid": not statistics_reasons and not topology_reasons,
        "root_association_status": source.get("root_association_status"),
        "attractor_count": source.get("attractor_count"),
        "conditioned_trajectories": conditioned,
        "counts": counts,
        "percentages": percentages,
        "training_data_sha256": source.get("training_data_sha256"),
        "checkpoint_sha256": source.get("checkpoint_sha256"),
        "output_dir": source.get("output_dir"),
        "failure_reason": source.get("failure_reason"),
        "error_type": source.get("error_type"),
        "error_message": source.get("error_message"),
    }


def normalize_package(package_root: Path, spec: PackageSpec) -> dict[str, Any]:
    package_root = package_root.resolve()
    payload, source_rows = _read_results(package_root)
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for row in source_rows:
        if "dataset" not in row:
            raise KeyError(f"{spec.label} result row is missing dataset")
        if spec.replicate_field not in row:
            raise KeyError(
                f"{spec.label} result row is missing {spec.replicate_field}; "
                "D1 seeds and D2 trial labels must remain distinct"
            )
        dataset = _integer(row["dataset"], field="dataset")
        replicate = _integer(
            row[spec.replicate_field],
            field=spec.replicate_field,
        )
        if dataset not in EXPECTED_DATASETS:
            raise ValueError(f"{spec.label} has unexpected dataset {dataset}")
        if replicate not in spec.expected_replicates:
            raise ValueError(
                f"{spec.label} dataset {dataset} has unexpected {spec.replicate_field} {replicate}"
            )
        key = (dataset, replicate)
        if key in indexed:
            raise ValueError(f"{spec.label} contains duplicate result slot {key}")
        indexed[key] = row

    rows = []
    for dataset in EXPECTED_DATASETS:
        for replicate in spec.expected_replicates:
            source = indexed.get(
                (dataset, replicate),
                _missing_row(dataset, replicate, spec),
            )
            rows.append(
                _normalize_row(
                    source,
                    spec=spec,
                    dataset=dataset,
                    replicate=replicate,
                )
            )

    results_path = package_root / "results.json"
    return {
        "spec": spec,
        "package_root": package_root,
        "results_path": results_path,
        "results_sha256": sha256_file(results_path),
        "source_schema_version": payload.get("schema_version"),
        "source_status": payload.get("status"),
        "experiment_plan_sha256": payload.get("experiment_plan_sha256"),
        "source_row_count": len(source_rows),
        "rows": rows,
    }


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "n": 0,
            "mean": None,
            "sample_standard_deviation": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "sample_standard_deviation": (statistics.stdev(numeric) if len(numeric) > 1 else 0.0),
        "median": statistics.median(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def _reportable_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["reportable_valid"]]


def _means(rows: Sequence[dict[str, Any]], field: str) -> dict[str, float | None]:
    valid = _reportable_rows(rows)
    means: dict[str, float | None] = {}
    for metric in REPORT_METRICS:
        values = [row[field][metric] for row in valid]
        numeric = [float(value) for value in values if value is not None]
        means[metric] = statistics.fmean(numeric) if numeric else None
    return means


def _reason_counts(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    reasons: Counter[str] = Counter()
    for row in rows:
        reasons.update(str(value) for value in row[field])
    return dict(sorted(reasons.items()))


def _run_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "planned": len(rows),
        "result_rows_present": sum(int(row["result_row_present"]) for row in rows),
        "input_status_complete": sum(int(row["input_status"] == "complete") for row in rows),
        "statistics_valid": sum(int(row["statistics_valid"]) for row in rows),
        "statistics_failed": sum(int(not row["statistics_valid"]) for row in rows),
        "topology_valid": sum(int(row["topology_valid"]) for row in rows),
        "topology_failed": sum(int(not row["topology_valid"]) for row in rows),
        "reportable_valid": sum(
            int(row["statistics_valid"] and row["topology_valid"]) for row in rows
        ),
    }


def aggregate_dimension(package: dict[str, Any]) -> dict[str, Any]:
    rows = package["rows"]
    valid = _reportable_rows(rows)
    descriptive_percentages = {
        metric: describe(
            [
                float(row["percentages"][metric])
                for row in valid
                if row["percentages"][metric] is not None
            ]
        )
        for metric in REPORT_METRICS
    }
    descriptive_counts = {
        metric: describe(
            [float(row["counts"][metric]) for row in valid if row["counts"][metric] is not None]
        )
        for metric in REPORT_METRICS
    }
    pooled_conditioned = sum(int(row["conditioned_trajectories"]) for row in valid)
    pooled_counts = {
        metric: sum(int(row["counts"][metric]) for row in valid) for metric in REPORT_METRICS
    }
    pooled_percentages = {
        metric: (100.0 * pooled_counts[metric] / pooled_conditioned if pooled_conditioned else None)
        for metric in REPORT_METRICS
    }
    return {
        "dimension": package["spec"].dimension,
        "dimension_label": package["spec"].label,
        "replicate_kind": package["spec"].replicate_kind,
        "run_counts": _run_counts(rows),
        "input_status_counts": dict(
            sorted(Counter(str(row["input_status"]) for row in rows).items())
        ),
        "statistics_failure_reasons": _reason_counts(
            rows,
            "statistics_failure_reasons",
        ),
        "topology_failure_reasons": _reason_counts(
            rows,
            "topology_failure_reasons",
        ),
        "descriptive_across_reportable_runs": {
            "percentages": descriptive_percentages,
            "counts": descriptive_counts,
        },
        "pooled_across_reportable_runs": {
            "valid_runs": len(valid),
            "conditioned_rows": pooled_conditioned,
            "counts": pooled_counts,
            "percentages": pooled_percentages,
            "interpretation": (
                "Descriptive only. Evaluation trajectories are reused across model runs."
            ),
        },
    }


def _alignment_for_dataset(
    dataset: int,
    d1_rows: Sequence[dict[str, Any]],
    d2_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    d1_hashes = sorted(
        {str(row["training_data_sha256"]) for row in d1_rows if row["training_data_sha256"]}
    )
    d2_hashes = sorted(
        {str(row["training_data_sha256"]) for row in d2_rows if row["training_data_sha256"]}
    )
    d1_seeds = sorted(
        {
            int(row["dataset_initial_condition_seed"])
            for row in d1_rows
            if row["dataset_initial_condition_seed"] is not None
        }
    )
    d2_seeds = sorted(
        {
            int(row["dataset_initial_condition_seed"])
            for row in d2_rows
            if row["dataset_initial_condition_seed"] is not None
        }
    )

    if len(d1_hashes) > 1 or len(d2_hashes) > 1:
        hash_status = "internally_inconsistent"
    elif d1_hashes and d2_hashes:
        hash_status = "verified_match" if d1_hashes == d2_hashes else "mismatch"
    else:
        hash_status = "unverified_missing_hash"

    if len(d1_seeds) > 1 or len(d2_seeds) > 1:
        seed_status = "internally_inconsistent"
    elif d1_seeds and d2_seeds:
        seed_status = "verified_match" if d1_seeds == d2_seeds else "mismatch"
    else:
        seed_status = "unverified_missing_seed"

    contradictory = {"mismatch", "internally_inconsistent"}
    if hash_status in contradictory or seed_status in contradictory:
        overall = "mismatch"
    elif hash_status == "verified_match" and seed_status == "verified_match":
        overall = "verified_match"
    else:
        overall = "partially_verified"

    return {
        "dataset": dataset,
        "status": overall,
        "training_data_sha256_status": hash_status,
        "dataset_initial_condition_seed_status": seed_status,
        "d1_training_data_sha256_values": d1_hashes,
        "d2_training_data_sha256_values": d2_hashes,
        "d1_dataset_initial_condition_seed_values": d1_seeds,
        "d2_dataset_initial_condition_seed_values": d2_seeds,
    }


def _subtract(
    d1_values: Mapping[str, float | None],
    d2_values: Mapping[str, float | None],
) -> dict[str, float | None]:
    return {
        metric: (
            None
            if d1_values[metric] is None or d2_values[metric] is None
            else float(d1_values[metric]) - float(d2_values[metric])
        )
        for metric in REPORT_METRICS
    }


def _dataset_summary(
    dataset: int,
    d1_rows: Sequence[dict[str, Any]],
    d2_rows: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    d1_percentage_means = _means(d1_rows, "percentages")
    d2_percentage_means = _means(d2_rows, "percentages")
    d1_count_means = _means(d1_rows, "counts")
    d2_count_means = _means(d2_rows, "counts")
    seed_values = (
        alignment["d1_dataset_initial_condition_seed_values"]
        or alignment["d2_dataset_initial_condition_seed_values"]
    )
    return {
        "dataset": dataset,
        "dataset_initial_condition_seed": seed_values[0] if len(seed_values) == 1 else None,
        "dataset_alignment_status": alignment["status"],
        "d1": {
            "run_counts": _run_counts(d1_rows),
            "run_cells": [
                {
                    "replicate_label": row["replicate_label"],
                    "statistics_valid": row["statistics_valid"],
                    "topology_valid": row["topology_valid"],
                    "reportable_valid": row["reportable_valid"],
                    "combined_correct_percentage": row["percentages"][COMBINED_CORRECT],
                    "outside_both_basins_percentage": row["percentages"][OUTSIDE],
                    "total_misclassified_percentage": row["percentages"][TOTAL_MISCLASSIFIED],
                    "statistics_failure_reasons": row["statistics_failure_reasons"],
                    "topology_failure_reasons": row["topology_failure_reasons"],
                }
                for row in d1_rows
            ],
            "mean_percentages_across_reportable_runs": d1_percentage_means,
            "mean_counts_across_reportable_runs": d1_count_means,
        },
        "d2": {
            "run_counts": _run_counts(d2_rows),
            "run_cells": [
                {
                    "replicate_label": row["replicate_label"],
                    "statistics_valid": row["statistics_valid"],
                    "topology_valid": row["topology_valid"],
                    "reportable_valid": row["reportable_valid"],
                    "combined_correct_percentage": row["percentages"][COMBINED_CORRECT],
                    "outside_both_basins_percentage": row["percentages"][OUTSIDE],
                    "total_misclassified_percentage": row["percentages"][TOTAL_MISCLASSIFIED],
                    "statistics_failure_reasons": row["statistics_failure_reasons"],
                    "topology_failure_reasons": row["topology_failure_reasons"],
                }
                for row in d2_rows
            ],
            "mean_percentages_across_reportable_runs": d2_percentage_means,
            "mean_counts_across_reportable_runs": d2_count_means,
        },
        "delta_d1_minus_d2": {
            "mean_percentage_points": _subtract(
                d1_percentage_means,
                d2_percentage_means,
            ),
            "mean_counts_per_run": _subtract(d1_count_means, d2_count_means),
            "difference_in_reportable_valid_runs": (
                _run_counts(d1_rows)["reportable_valid"] - _run_counts(d2_rows)["reportable_valid"]
            ),
            "difference_in_topology_failures": (
                _run_counts(d1_rows)["topology_failed"] - _run_counts(d2_rows)["topology_failed"]
            ),
        },
    }


def build_comparison(
    d1_package: dict[str, Any],
    d2_package: dict[str, Any],
) -> dict[str, Any]:
    d1_rows = d1_package["rows"]
    d2_rows = d2_package["rows"]
    alignments = []
    by_dataset = []
    for dataset in EXPECTED_DATASETS:
        current_d1 = [row for row in d1_rows if row["dataset"] == dataset]
        current_d2 = [row for row in d2_rows if row["dataset"] == dataset]
        alignment = _alignment_for_dataset(dataset, current_d1, current_d2)
        alignments.append(alignment)
        by_dataset.append(
            _dataset_summary(
                dataset,
                current_d1,
                current_d2,
                alignment,
            )
        )

    mismatches = [alignment for alignment in alignments if alignment["status"] == "mismatch"]
    if mismatches:
        datasets = ", ".join(str(item["dataset"]) for item in mismatches)
        raise ValueError(
            "refusing to compare mismatched training datasets; "
            f"alignment failed for dataset(s) {datasets}"
        )

    d1_aggregate = aggregate_dimension(d1_package)
    d2_aggregate = aggregate_dimension(d2_package)
    d1_descriptive = d1_aggregate["descriptive_across_reportable_runs"]
    d2_descriptive = d2_aggregate["descriptive_across_reportable_runs"]
    overall_deltas = {
        "mean_percentage_points": {
            metric: (
                None
                if d1_descriptive["percentages"][metric]["mean"] is None
                or d2_descriptive["percentages"][metric]["mean"] is None
                else float(d1_descriptive["percentages"][metric]["mean"])
                - float(d2_descriptive["percentages"][metric]["mean"])
            )
            for metric in REPORT_METRICS
        },
        "mean_counts_per_run": {
            metric: (
                None
                if d1_descriptive["counts"][metric]["mean"] is None
                or d2_descriptive["counts"][metric]["mean"] is None
                else float(d1_descriptive["counts"][metric]["mean"])
                - float(d2_descriptive["counts"][metric]["mean"])
            )
            for metric in REPORT_METRICS
        },
    }

    if all(alignment["status"] == "verified_match" for alignment in alignments):
        alignment_status = "verified_match"
    else:
        alignment_status = "partially_verified"

    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "comparison_scope": "matched_training_datasets_unpaired_training_runs",
        "pairing_statement": PAIRING_STATEMENT,
        "delta_convention": (
            "All deltas are D1 minus D2. Percentage deltas are percentage points."
        ),
        "validity_convention": (
            "Metric summaries use only rows whose five category counts conserve "
            "the conditioned denominator and whose reported percentages agree. "
            "Topology validity is checked separately and requires exactly two "
            "attractors plus valid distinct stable-root association. An input row "
            "marked invalid can therefore be statistics-valid but topology-invalid, "
            "or vice versa; only rows passing both checks enter metric summaries."
        ),
        "design": {
            "datasets": list(EXPECTED_DATASETS),
            "planned_runs_per_dimension": 15,
            "d1_replicates": {
                "field": D1_SPEC.replicate_field,
                "values": list(D1_SPEC.expected_replicates),
                "recorded_rng_seeds": True,
            },
            "d2_replicates": {
                "field": D2_SPEC.replicate_field,
                "values": list(D2_SPEC.expected_replicates),
                "recorded_rng_seeds": False,
            },
        },
        "inputs": {
            "d1": {
                "package_root": str(d1_package["package_root"]),
                "results_path": str(d1_package["results_path"]),
                "results_sha256": d1_package["results_sha256"],
                "source_schema_version": d1_package["source_schema_version"],
                "source_status": d1_package["source_status"],
                "experiment_plan_sha256": d1_package["experiment_plan_sha256"],
                "source_row_count": d1_package["source_row_count"],
            },
            "d2": {
                "package_root": str(d2_package["package_root"]),
                "results_path": str(d2_package["results_path"]),
                "results_sha256": d2_package["results_sha256"],
                "source_schema_version": d2_package["source_schema_version"],
                "source_status": d2_package["source_status"],
                "experiment_plan_sha256": d2_package["experiment_plan_sha256"],
                "source_row_count": d2_package["source_row_count"],
                "training_trial_ids_are_not_recorded_rng_seeds": True,
            },
        },
        "dataset_alignment": {
            "status": alignment_status,
            "datasets": alignments,
        },
        "dimension_aggregates": {
            "d1": d1_aggregate,
            "d2": d2_aggregate,
        },
        "overall_delta_d1_minus_d2": overall_deltas,
        "by_dataset": by_dataset,
        "per_run": d1_rows + d2_rows,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _run_cell_text(cells: Sequence[dict[str, Any]]) -> str:
    parts = []
    for cell in cells:
        label = cell["replicate_label"]
        if not cell["statistics_valid"]:
            parts.append(f"{label}: STAT FAIL")
            continue
        topology_marker = "" if cell["topology_valid"] else "†"
        parts.append(
            f"{label}: {_fmt(cell['combined_correct_percentage'])} / "
            f"{_fmt(cell['outside_both_basins_percentage'])} / "
            f"{_fmt(cell['total_misclassified_percentage'])}{topology_marker}"
        )
    return "; ".join(parts)


def render_markdown(comparison: dict[str, Any]) -> str:
    d1 = comparison["dimension_aggregates"]["d1"]
    d2 = comparison["dimension_aggregates"]["d2"]
    lines = [
        "# Matched-dataset Chafee-Infante D1 versus D2 comparison",
        "",
        "## Interpretation boundary",
        "",
        comparison["pairing_statement"],
        "",
        "Each run cell below is `correct / outside / total misclassified` in percent. "
        "A dagger (`†`) marks a statistics-valid row whose topology check failed. "
        "Only rows passing both statistics and topology checks enter reported means.",
        "",
        "## Run validity",
        "",
        "| Dimension | Planned | Present | Statistics valid | Statistics failed | "
        "Topology valid | Topology failed | Reportable valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for aggregate in (d1, d2):
        counts = aggregate["run_counts"]
        lines.append(
            f"| {aggregate['dimension_label']} | {counts['planned']} | "
            f"{counts['result_rows_present']} | {counts['statistics_valid']} | "
            f"{counts['statistics_failed']} | {counts['topology_valid']} | "
            f"{counts['topology_failed']} | "
            f"{counts['reportable_valid']} |"
        )

    lines.extend(
        [
            "",
            "## Per-dataset three-run cells and means",
            "",
            "| Dataset | IC seed | D1 seeds 0/1/2: correct / outside / misclass (%) | "
            "D2 trials 1/2/3: correct / outside / misclass (%) | D1 mean correct | "
            "D2 mean correct | Delta D1-D2 |",
            "|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for entry in comparison["by_dataset"]:
        d1_means = entry["d1"]["mean_percentages_across_reportable_runs"]
        d2_means = entry["d2"]["mean_percentages_across_reportable_runs"]
        delta = entry["delta_d1_minus_d2"]["mean_percentage_points"]
        lines.append(
            f"| {entry['dataset']} | {entry['dataset_initial_condition_seed'] or '—'} | "
            f"{_run_cell_text(entry['d1']['run_cells'])} | "
            f"{_run_cell_text(entry['d2']['run_cells'])} | "
            f"{_fmt(d1_means[COMBINED_CORRECT])} | "
            f"{_fmt(d2_means[COMBINED_CORRECT])} | "
            f"{_fmt(delta[COMBINED_CORRECT])} |"
        )

    lines.extend(
        [
            "",
            "## Dimension-level statistics across valid runs",
            "",
            "| Metric (%) | D1 mean | D1 sample SD | D1 median | D2 mean | "
            "D2 sample SD | D2 median | Delta mean D1-D2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    overall_delta = comparison["overall_delta_d1_minus_d2"]["mean_percentage_points"]
    for metric in REPORT_METRICS:
        d1_stats = d1["descriptive_across_reportable_runs"]["percentages"][metric]
        d2_stats = d2["descriptive_across_reportable_runs"]["percentages"][metric]
        lines.append(
            f"| {METRIC_LABELS[metric]} | {_fmt(d1_stats['mean'])} | "
            f"{_fmt(d1_stats['sample_standard_deviation'])} | "
            f"{_fmt(d1_stats['median'])} | {_fmt(d2_stats['mean'])} | "
            f"{_fmt(d2_stats['sample_standard_deviation'])} | "
            f"{_fmt(d2_stats['median'])} | {_fmt(overall_delta[metric])} |"
        )

    lines.extend(
        [
            "",
            "## Per-dataset mean decomposition",
            "",
            "| Dataset | Dimension | Valid n | Correct negative (%) | "
            "Correct positive (%) | Misclass negative (%) | "
            "Misclass positive (%) | Outside (%) |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in comparison["by_dataset"]:
        for key, label in (("d1", "D1"), ("d2", "D2")):
            means = entry[key]["mean_percentages_across_reportable_runs"]
            valid_n = entry[key]["run_counts"]["reportable_valid"]
            lines.append(
                f"| {entry['dataset']} | {label} | {valid_n} | "
                f"{_fmt(means[CORRECT_NEGATIVE])} | "
                f"{_fmt(means[CORRECT_POSITIVE])} | "
                f"{_fmt(means[MISCLASSIFIED_NEGATIVE])} | "
                f"{_fmt(means[MISCLASSIFIED_POSITIVE])} | "
                f"{_fmt(means[OUTSIDE])} |"
            )

    lines.extend(
        [
            "",
            "## Pooled descriptive counts",
            "",
            "These counts reuse the same evaluation archive across model runs and are "
            "descriptive, not independent observations.",
            "",
            "| Dimension | Valid runs | Conditioned rows | Correct negative | "
            "Correct positive | Misclass negative | Misclass positive | Outside |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for aggregate in (d1, d2):
        pooled = aggregate["pooled_across_reportable_runs"]
        counts = pooled["counts"]
        lines.append(
            f"| {aggregate['dimension_label']} | {pooled['valid_runs']} | "
            f"{pooled['conditioned_rows']} | {counts[CORRECT_NEGATIVE]} | "
            f"{counts[CORRECT_POSITIVE]} | {counts[MISCLASSIFIED_NEGATIVE]} | "
            f"{counts[MISCLASSIFIED_POSITIVE]} | {counts[OUTSIDE]} |"
        )

    lines.extend(["", "## Failure accounting", ""])
    has_failures = False
    for aggregate in (d1, d2):
        for failure_kind in ("statistics_failure_reasons", "topology_failure_reasons"):
            for reason, count in aggregate[failure_kind].items():
                has_failures = True
                readable_kind = failure_kind.replace("_", " ")
                lines.append(
                    f"- {aggregate['dimension_label']} {readable_kind}: `{reason}` — {count}"
                )
    if not has_failures:
        lines.append("- No statistics or topology failures.")

    lines.extend(
        [
            "",
            "## Dataset alignment",
            "",
            f"Overall alignment status: `{comparison['dataset_alignment']['status']}`.",
            "",
            "| Dataset | Overall | Training-data hash | IC seed |",
            "|---:|---|---|---|",
        ]
    )
    for alignment in comparison["dataset_alignment"]["datasets"]:
        lines.append(
            f"| {alignment['dataset']} | {alignment['status']} | "
            f"{alignment['training_data_sha256_status']} | "
            f"{alignment['dataset_initial_condition_seed_status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _comparison_csv_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in comparison["by_dataset"]:
        row: dict[str, Any] = {
            "scope": "dataset",
            "dataset": entry["dataset"],
            "dataset_initial_condition_seed": entry["dataset_initial_condition_seed"],
            "dataset_alignment_status": entry["dataset_alignment_status"],
            "d1_run_cells": _run_cell_text(entry["d1"]["run_cells"]),
            "d2_run_cells": _run_cell_text(entry["d2"]["run_cells"]),
            "d1_statistics_valid": entry["d1"]["run_counts"]["statistics_valid"],
            "d2_statistics_valid": entry["d2"]["run_counts"]["statistics_valid"],
            "d1_reportable_valid": entry["d1"]["run_counts"]["reportable_valid"],
            "d2_reportable_valid": entry["d2"]["run_counts"]["reportable_valid"],
            "d1_topology_failed": entry["d1"]["run_counts"]["topology_failed"],
            "d2_topology_failed": entry["d2"]["run_counts"]["topology_failed"],
        }
        for metric in REPORT_METRICS:
            row[f"d1_mean_{metric}_percentage"] = entry["d1"][
                "mean_percentages_across_reportable_runs"
            ][metric]
            row[f"d2_mean_{metric}_percentage"] = entry["d2"][
                "mean_percentages_across_reportable_runs"
            ][metric]
            row[f"delta_d1_minus_d2_{metric}_percentage_points"] = entry["delta_d1_minus_d2"][
                "mean_percentage_points"
            ][metric]
            row[f"d1_mean_{metric}_count"] = entry["d1"]["mean_counts_across_reportable_runs"][
                metric
            ]
            row[f"d2_mean_{metric}_count"] = entry["d2"]["mean_counts_across_reportable_runs"][
                metric
            ]
            row[f"delta_d1_minus_d2_{metric}_mean_count"] = entry["delta_d1_minus_d2"][
                "mean_counts_per_run"
            ][metric]
        rows.append(row)

    d1 = comparison["dimension_aggregates"]["d1"]
    d2 = comparison["dimension_aggregates"]["d2"]
    aggregate_row: dict[str, Any] = {
        "scope": "dimension_aggregate",
        "dataset": "ALL",
        "dataset_initial_condition_seed": "",
        "dataset_alignment_status": comparison["dataset_alignment"]["status"],
        "d1_run_cells": "",
        "d2_run_cells": "",
        "d1_statistics_valid": d1["run_counts"]["statistics_valid"],
        "d2_statistics_valid": d2["run_counts"]["statistics_valid"],
        "d1_reportable_valid": d1["run_counts"]["reportable_valid"],
        "d2_reportable_valid": d2["run_counts"]["reportable_valid"],
        "d1_topology_failed": d1["run_counts"]["topology_failed"],
        "d2_topology_failed": d2["run_counts"]["topology_failed"],
    }
    for metric in REPORT_METRICS:
        aggregate_row[f"d1_mean_{metric}_percentage"] = d1["descriptive_across_reportable_runs"][
            "percentages"
        ][metric]["mean"]
        aggregate_row[f"d2_mean_{metric}_percentage"] = d2["descriptive_across_reportable_runs"][
            "percentages"
        ][metric]["mean"]
        aggregate_row[f"delta_d1_minus_d2_{metric}_percentage_points"] = comparison[
            "overall_delta_d1_minus_d2"
        ]["mean_percentage_points"][metric]
        aggregate_row[f"d1_mean_{metric}_count"] = d1["descriptive_across_reportable_runs"][
            "counts"
        ][metric]["mean"]
        aggregate_row[f"d2_mean_{metric}_count"] = d2["descriptive_across_reportable_runs"][
            "counts"
        ][metric]["mean"]
        aggregate_row[f"delta_d1_minus_d2_{metric}_mean_count"] = comparison[
            "overall_delta_d1_minus_d2"
        ]["mean_counts_per_run"][metric]
    rows.append(aggregate_row)
    return rows


def _per_run_csv_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for normalized in comparison["per_run"]:
        row = {
            key: normalized[key]
            for key in (
                "dimension",
                "dimension_label",
                "dataset",
                "dataset_initial_condition_seed",
                "replicate_kind",
                "replicate_id",
                "replicate_label",
                "input_status",
                "result_row_present",
                "statistics_valid",
                "topology_valid",
                "reportable_valid",
                "root_association_status",
                "attractor_count",
                "conditioned_trajectories",
                "training_data_sha256",
                "checkpoint_sha256",
                "output_dir",
                "failure_reason",
                "error_type",
                "error_message",
            )
        }
        row["statistics_failure_reasons"] = json.dumps(
            normalized["statistics_failure_reasons"],
            separators=(",", ":"),
        )
        row["topology_failure_reasons"] = json.dumps(
            normalized["topology_failure_reasons"],
            separators=(",", ":"),
        )
        for metric in REPORT_METRICS:
            row[f"{metric}_count"] = normalized["counts"][metric]
            row[f"{metric}_percentage"] = normalized["percentages"][metric]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(
    *,
    d1_package_root: Path,
    d2_package_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")

    d1_package = normalize_package(d1_package_root, D1_SPEC)
    d2_package = normalize_package(d2_package_root, D2_SPEC)
    comparison = build_comparison(d1_package, d2_package)

    output_root.mkdir(parents=True)
    write_json(output_root / "comparison.json", comparison)
    write_csv(output_root / "comparison.csv", _comparison_csv_rows(comparison))
    write_csv(output_root / "per_run.csv", _per_run_csv_rows(comparison))
    (output_root / "comparison.md").write_text(
        render_markdown(comparison),
        encoding="utf-8",
    )
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-package", type=Path, default=DEFAULT_D1_PACKAGE)
    parser.add_argument("--d2-package", type=Path, default=DEFAULT_D2_PACKAGE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    comparison = generate_report(
        d1_package_root=args.d1_package,
        d2_package_root=args.d2_package,
        output_root=args.output_root,
    )
    d1_counts = comparison["dimension_aggregates"]["d1"]["run_counts"]
    d2_counts = comparison["dimension_aggregates"]["d2"]["run_counts"]
    print(
        "Wrote matched-dataset, run-unpaired comparison: "
        f"{args.output_root.resolve()} "
        f"(D1 statistics valid {d1_counts['statistics_valid']}/15; "
        f"D2 statistics valid {d2_counts['statistics_valid']}/15)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
