"""Tests for the matched-dataset D1-versus-D2 report generator."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "report_chafee_matched_d1_d2.py"
    spec = importlib.util.spec_from_file_location(
        "report_chafee_matched_d1_d2",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = _load_module()


def _result_row(
    *,
    dataset: int,
    replicate_field: str,
    replicate: int,
    correct_shift: int,
    training_hash: str | None = None,
) -> dict[str, object]:
    conditioned = 100
    correct_negative = 25 + dataset + correct_shift
    correct_positive = 35 + replicate + correct_shift
    misclassified_negative = 2
    misclassified_positive = 3
    outside = conditioned - (
        correct_negative + correct_positive + misclassified_negative + misclassified_positive
    )
    counts = {
        "correctly_classified_in_negative_basin": correct_negative,
        "correctly_classified_in_positive_basin": correct_positive,
        "misclassified_in_negative_basin": misclassified_negative,
        "misclassified_in_positive_basin": misclassified_positive,
        "outside_both_basins": outside,
    }
    combined = correct_negative + correct_positive
    return {
        "status": "complete",
        "dataset": dataset,
        "dataset_initial_condition_seed": 1_000 + dataset,
        replicate_field: replicate,
        "training_data_sha256": training_hash or f"dataset-{dataset}-hash",
        "checkpoint_sha256": f"checkpoint-{replicate_field}-{dataset}-{replicate}",
        "root_association_status": "valid",
        "attractor_count": 2,
        "conditioned_trajectories": conditioned,
        **counts,
        **{f"{name}_percentage": float(value) for name, value in counts.items()},
        "combined_correct_count": combined,
        "combined_correct_percentage": float(combined),
    }


def _write_package(
    root: Path,
    *,
    replicate_field: str,
    replicates: tuple[int, int, int],
    correct_shift: int,
) -> list[dict[str, object]]:
    rows = [
        _result_row(
            dataset=dataset,
            replicate_field=replicate_field,
            replicate=replicate,
            correct_shift=correct_shift,
        )
        for dataset in range(1, 6)
        for replicate in replicates
    ]
    root.mkdir(parents=True)
    (root / "results.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "results": rows,
            }
        ),
        encoding="utf-8",
    )
    return rows


def _packages(tmp_path: Path) -> tuple[Path, Path]:
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    _write_package(
        d1,
        replicate_field="training_seed",
        replicates=(0, 1, 2),
        correct_shift=2,
    )
    _write_package(
        d2,
        replicate_field="training_trial",
        replicates=(1, 2, 3),
        correct_shift=0,
    )
    return d1, d2


def test_generate_report_writes_unpaired_multiformat_comparison(
    tmp_path: Path,
) -> None:
    d1, d2 = _packages(tmp_path)
    output = tmp_path / "report"

    comparison = REPORT.generate_report(
        d1_package_root=d1,
        d2_package_root=d2,
        output_root=output,
    )

    assert comparison["comparison_scope"] == ("matched_training_datasets_unpaired_training_runs")
    assert "not paired" in comparison["pairing_statement"]
    assert comparison["design"]["d2_replicates"]["recorded_rng_seeds"] is False
    assert comparison["dataset_alignment"]["status"] == "verified_match"
    assert len(comparison["per_run"]) == 30
    assert len(comparison["by_dataset"]) == 5
    assert comparison["dimension_aggregates"]["d1"]["run_counts"]["statistics_valid"] == 15
    assert comparison["dimension_aggregates"]["d2"]["run_counts"]["topology_failed"] == 0
    assert comparison["overall_delta_d1_minus_d2"]["mean_percentage_points"][
        "combined_correct"
    ] == pytest.approx(3.0)
    count_deltas = comparison["by_dataset"][0]["delta_d1_minus_d2"]
    assert count_deltas["difference_in_reportable_valid_runs"] == 0
    assert count_deltas["difference_in_topology_failures"] == 0
    assert "reportable_valid_runs" not in count_deltas

    assert {path.name for path in output.iterdir()} == {
        "comparison.json",
        "comparison.csv",
        "per_run.csv",
        "comparison.md",
    }
    markdown = (output / "comparison.md").read_text(encoding="utf-8")
    assert "D1 training seeds are not paired to D2 trial IDs" in markdown
    assert "D1 seeds 0/1/2" in markdown
    assert "s0:" in markdown
    assert "t3:" in markdown

    with (output / "comparison.csv").open(newline="", encoding="utf-8") as stream:
        comparison_rows = list(csv.DictReader(stream))
    assert len(comparison_rows) == 6
    assert comparison_rows[-1]["scope"] == "dimension_aggregate"
    assert comparison_rows[-1]["dataset"] == "ALL"

    with (output / "per_run.csv").open(newline="", encoding="utf-8") as stream:
        run_rows = list(csv.DictReader(stream))
    assert len(run_rows) == 30
    assert {row["replicate_kind"] for row in run_rows} == {
        "recorded_training_seed",
        "archived_unseeded_training_trial",
    }


def test_failures_are_retained_and_excluded_from_metric_means(
    tmp_path: Path,
) -> None:
    d1, d2 = _packages(tmp_path)
    payload = json.loads((d1 / "results.json").read_text(encoding="utf-8"))
    payload["results"][0] = {
        "status": "failed",
        "dataset": 1,
        "training_seed": 0,
        "dataset_initial_condition_seed": 1001,
        "training_data_sha256": "dataset-1-hash",
        "error_type": "RuntimeError",
        "error_message": "synthetic failure",
    }
    payload["results"][1]["root_association_status"] = "ambiguous"
    (d1 / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    comparison = REPORT.generate_report(
        d1_package_root=d1,
        d2_package_root=d2,
        output_root=tmp_path / "report",
    )

    counts = comparison["dimension_aggregates"]["d1"]["run_counts"]
    assert counts["statistics_valid"] == 14
    assert counts["statistics_failed"] == 1
    assert counts["topology_failed"] == 2
    dataset = comparison["by_dataset"][0]
    assert dataset["d1"]["run_counts"]["statistics_valid"] == 2
    assert dataset["d1"]["run_counts"]["topology_failed"] == 2
    assert dataset["d1"]["run_cells"][0]["statistics_valid"] is False
    assert dataset["d1"]["run_cells"][1]["statistics_valid"] is True
    assert dataset["d1"]["run_cells"][1]["topology_valid"] is False


def test_missing_result_slot_is_counted_as_a_failure(tmp_path: Path) -> None:
    d1, d2 = _packages(tmp_path)
    payload = json.loads((d1 / "results.json").read_text(encoding="utf-8"))
    payload["results"] = payload["results"][1:]
    (d1 / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    comparison = REPORT.generate_report(
        d1_package_root=d1,
        d2_package_root=d2,
        output_root=tmp_path / "report",
    )

    counts = comparison["dimension_aggregates"]["d1"]["run_counts"]
    assert counts["planned"] == 15
    assert counts["result_rows_present"] == 14
    assert counts["statistics_failed"] == 1
    first = comparison["per_run"][0]
    assert first["input_status"] == "missing_result_row"
    assert first["statistics_failure_reasons"] == ["run_status:missing_result_row"]


def test_statistics_and_topology_validity_are_independent(tmp_path: Path) -> None:
    d1, d2 = _packages(tmp_path)
    payload = json.loads((d1 / "results.json").read_text(encoding="utf-8"))
    topology_invalid = payload["results"][0]
    topology_invalid["status"] = "invalid"
    topology_invalid["root_association_status"] = "invalid"
    conservation_invalid = payload["results"][1]
    conservation_invalid["status"] = "invalid"
    conservation_invalid["outside_both_basins"] += 1
    conservation_invalid["outside_both_basins_percentage"] += 1.0
    (d1 / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    comparison = REPORT.generate_report(
        d1_package_root=d1,
        d2_package_root=d2,
        output_root=tmp_path / "report",
    )

    first, second = comparison["per_run"][:2]
    assert first["statistics_valid"] is True
    assert first["topology_valid"] is False
    assert first["reportable_valid"] is False
    assert second["statistics_valid"] is False
    assert second["topology_valid"] is True
    assert second["reportable_valid"] is False
    assert comparison["by_dataset"][0]["d1"]["run_counts"]["reportable_valid"] == 1


def test_training_dataset_hash_mismatch_is_rejected_before_writing(
    tmp_path: Path,
) -> None:
    d1, d2 = _packages(tmp_path)
    payload = json.loads((d1 / "results.json").read_text(encoding="utf-8"))
    for row in payload["results"]:
        if row["dataset"] == 3:
            row["training_data_sha256"] = "wrong-dataset-3-hash"
    (d1 / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "report"

    with pytest.raises(ValueError, match="alignment failed for dataset\\(s\\) 3"):
        REPORT.generate_report(
            d1_package_root=d1,
            d2_package_root=d2,
            output_root=output,
        )

    assert not output.exists()


def test_dimension_specific_replicate_field_is_required(tmp_path: Path) -> None:
    d1, _ = _packages(tmp_path)
    payload = json.loads((d1 / "results.json").read_text(encoding="utf-8"))
    payload["results"][0]["training_trial"] = payload["results"][0].pop("training_seed")
    (d1 / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KeyError, match="training_seed"):
        REPORT.normalize_package(d1, REPORT.D1_SPEC)
