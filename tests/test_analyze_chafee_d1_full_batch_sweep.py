"""Tests for the isolated full-batch sweep RoA analyzer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    script = scripts / "analyze_chafee_d1_full_batch_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "analyze_chafee_d1_full_batch_sweep",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZE = _load_module()


def _tiny_pairs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(444)
    return (
        rng.normal(size=(6, 64)).astype(np.float64),
        rng.normal(size=(6, 64)).astype(np.float64),
    )


def _patch_sweep_inputs(monkeypatch, tmp_path: Path) -> None:
    sources: dict[str, dict[str, object]] = {}
    for name in (
        "train_data",
        "canonical_checkpoint",
        "canonical_architecture_sidecar",
        "marcio_training_implementation",
        "sweep_runner_implementation",
    ):
        path = tmp_path / "frozen_sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        sources[name] = {
            "path": str(path),
            "sha256": ANALYZE.sweep._sha256(path),
            "size_bytes": path.stat().st_size,
        }
    arch = ANALYZE.study.marcio_architecture(1)
    monkeypatch.setattr(
        ANALYZE.sweep,
        "_current_source_provenance",
        lambda _train_data=ANALYZE.sweep.TRAIN_DATA: sources,
    )
    monkeypatch.setattr(ANALYZE.sweep, "_load_canonical_arch", lambda: arch)
    monkeypatch.setattr(
        ANALYZE.sweep,
        "_load_training_pairs",
        lambda _train_data=ANALYZE.sweep.TRAIN_DATA: _tiny_pairs(),
    )


def _make_source_sweep(
    monkeypatch,
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (0, 1),
) -> Path:
    _patch_sweep_inputs(monkeypatch, tmp_path)
    source = tmp_path / "source_sweep"
    specs = tuple(
        ANALYZE.sweep.FullBatchRunSpec(
            run_id=f"seed_{seed:02d}_test",
            seed=seed,
            epochs=1,
            learning_rate=1e-3,
        )
        for seed in seeds
    )
    summary = ANALYZE.sweep.run_sweep(
        output_dir=source,
        device_name="cpu",
        run_specs=specs,
        verbose=False,
    )
    assert summary["all_runs_completed"] is True
    return source


def _dummy_candidate(tmp_path: Path) -> tuple[object, Path]:
    snapshot_root = tmp_path / "diagnostics" / "source_snapshot"
    snapshot = snapshot_root / "run_001_seed_00_test"
    files = {
        "run_spec": snapshot / "run_spec.json",
        "completion": snapshot / "completed.json",
        "manifest": snapshot / "artifact_manifest.json",
        "checkpoint": snapshot / "models" / "autoencoder.pt",
        "sidecar": snapshot / "models" / "autoencoder.json",
        "history": snapshot / "logs" / "history.json",
        "summary": snapshot / "training_summary.json",
    }
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    spec = ANALYZE.sweep.FullBatchRunSpec(
        run_id="seed_00_test",
        seed=0,
        epochs=1,
        learning_rate=1e-3,
    )
    candidate = ANALYZE.FrozenCandidate(
        plan_index=1,
        spec=spec,
        run_root=tmp_path,
        attempt=1,
        run_spec=tmp_path / "run_spec.json",
        run_spec_sha256=ANALYZE.study.sha256_file(files["run_spec"]),
        completion=tmp_path / "completed.json",
        completion_sha256=ANALYZE.study.sha256_file(files["completion"]),
        artifact_manifest=tmp_path / "artifact_manifest.json",
        artifact_manifest_sha256=ANALYZE.study.sha256_file(files["manifest"]),
        checkpoint=tmp_path / "autoencoder.pt",
        checkpoint_sha256=ANALYZE.study.sha256_file(files["checkpoint"]),
        sidecar=tmp_path / "autoencoder.json",
        sidecar_sha256=ANALYZE.study.sha256_file(files["sidecar"]),
        history=tmp_path / "history.json",
        history_sha256=ANALYZE.study.sha256_file(files["history"]),
        training_summary=tmp_path / "training_summary.json",
        training_summary_sha256=ANALYZE.study.sha256_file(files["summary"]),
        final_train={
            "loss_reconstruction": 0.1,
            "loss_prediction": 0.2,
            "loss_total": 0.3,
        },
        history_final_learning_rate=1e-3,
        optimizer_final_learning_rate=1e-3,
    )
    return candidate, snapshot_root


def test_safe_target_requires_fresh_sibling(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="isolated sibling"):
        ANALYZE._assert_safe_target(source, tmp_path / "nested" / "diagnostics")
    with pytest.raises(ValueError, match="overlaps protected"):
        ANALYZE._assert_safe_target(source, source)

    target = tmp_path / "diagnostics"
    ANALYZE._assert_safe_target(source, target)
    target.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ANALYZE._assert_safe_target(source, target)


def test_source_sweep_verifies_completed_artifact_chain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _make_source_sweep(monkeypatch, tmp_path)
    frozen = ANALYZE._verify_source_sweep(source)

    assert [item.source_status for item in frozen.inventory] == [
        "completed_valid",
        "completed_valid",
    ]
    assert [item.candidate.spec.seed for item in frozen.inventory] == [0, 1]
    assert all(
        item.candidate.checkpoint_sha256
        for item in frozen.inventory
        if item.candidate is not None
    )


def test_corrupt_candidate_is_excluded_without_hiding_later_valid_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _make_source_sweep(monkeypatch, tmp_path, seeds=(0, 1, 2))
    checkpoint = (
        source
        / "runs"
        / "seed_01_test"
        / "attempts"
        / "attempt_001"
        / "models"
        / "autoencoder.pt"
    )
    with checkpoint.open("ab") as stream:
        stream.write(b"corrupt")

    frozen = ANALYZE._verify_source_sweep(source)

    assert [item.source_status for item in frozen.inventory] == [
        "completed_valid",
        "source_invalid",
        "completed_valid",
    ]
    assert [candidate.spec.seed for candidate in frozen.candidates] == [0, 2]
    assert "failed hash validation" in frozen.inventory[1].error_message


def test_statistics_and_reference_deltas_use_conditioned_denominator() -> None:
    payload = {
        "statistics": {
            "conditioned_trajectories": 7_862,
            "counts": {
                "correctly_classified_in_negative_basin": 1_524,
                "correctly_classified_in_positive_basin": 2_408,
                "misclassified_in_negative_basin": 2,
                "misclassified_in_positive_basin": 0,
                "outside_both_basins": 3_928,
            },
            "percentages": {
                "correctly_classified_in_negative_basin": 19.384380564741797,
                "correctly_classified_in_positive_basin": 30.62833884507759,
                "misclassified_in_negative_basin": 0.02543881963876876,
                "misclassified_in_positive_basin": 0.0,
                "outside_both_basins": 49.961841770541845,
            },
        },
        "uniform_is_bistable": True,
        "adaptive_graph": {"is_bistable": True},
        "roots_define_two_distinct_attractor_basins": True,
        "eligible_for_bistable_dimension_table": True,
    }
    row = ANALYZE._stats_fields(payload)
    references = {
        "full_batch_10000": {
            "correct_combined_percent": (
                ANALYZE.FULL_BATCH_10K_COMBINED_PERCENT
            )
        },
        "marcio_archived": {
            "correct_combined_percent": (
                ANALYZE.MARCIO_ARCHIVED_COMBINED_PERCENT
            )
        },
    }
    ANALYZE._add_reference_deltas(row, references)

    assert row["correct_combined_count"] == 3_932
    assert row["correct_combined_percent"] == pytest.approx(
        ANALYZE.FULL_BATCH_10K_COMBINED_PERCENT
    )
    assert row["misc_count"] == 2
    assert row["delta_vs_full_batch_10000_percentage_points"] == pytest.approx(
        0.0
    )
    assert row["beats_full_batch_10000"] is False
    assert row["delta_vs_marcio_archived_percentage_points"] < -28.0


def test_failed_candidate_is_recorded_without_reraising(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate, snapshot_root = _dummy_candidate(tmp_path)
    target = tmp_path / "diagnostics"
    references = {
        name: {"correct_combined_percent": value}
        for name, value in (
            ("full_batch_10000", 50.0),
            ("marcio_archived", 78.0),
        )
    }
    monkeypatch.setattr(
        ANALYZE.study,
        "_run_bounds",
        lambda *args, **kwargs: None,
    )

    def fail_coarse(*_args, **_kwargs):
        raise RuntimeError("synthetic coarse failure")

    monkeypatch.setattr(
        ANALYZE.study,
        "_run_precompute_coarse",
        fail_coarse,
    )
    row = ANALYZE._analyze_candidate(
        target=target,
        snapshot_root=snapshot_root,
        candidate=candidate,
        analysis_plan_sha256="plan",
        inputs=object(),
        device=ANALYZE.torch.device("cpu"),
        batch_points="auto",
        references=references,
    )

    assert row["status"] == "failed"
    assert row["error_stage"] == "precompute-coarse"
    assert row["error_message"] == "synthetic coarse failure"
    manifest = json.loads(
        (
            target
            / "by_run"
            / candidate.snapshot_name
            / "analysis_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["completed_stages"] == ["bounds"]
    assert manifest["status"] == "failed"


def test_final_results_preserve_plan_order_and_rank_only_post_hoc(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "plan_index": 1,
            "run_id": "seed_00",
            "seed": 0,
            "source_status": "completed_valid",
            "status": "complete",
            "correct_combined_percent": 60.0,
            "delta_vs_full_batch_10000_percentage_points": 10.0,
            "delta_vs_marcio_archived_percentage_points": -18.0,
        },
        {
            "plan_index": 2,
            "run_id": "seed_01",
            "seed": 1,
            "source_status": "completed_valid",
            "status": "complete",
            "correct_combined_percent": 70.0,
            "delta_vs_full_batch_10000_percentage_points": 20.0,
            "delta_vs_marcio_archived_percentage_points": -8.0,
        },
    ]
    payload = ANALYZE._write_results(
        tmp_path,
        analysis_plan_sha256="plan",
        references={},
        rows=rows,
        expected_runs=2,
        final=True,
    )

    assert [
        row["run_id"] for row in payload["results_in_frozen_plan_order"]
    ] == ["seed_00", "seed_01"]
    ranking = payload["post_hoc_exploratory_ranking"]
    assert ranking["used_for_computation_candidate_inclusion_or_checkpoint_selection"] is False
    assert [row["run_id"] for row in ranking["rows"]] == [
        "seed_01",
        "seed_00",
    ]
    assert (tmp_path / "results_by_run.json").is_file()
    assert (tmp_path / "results_by_run.csv").is_file()

    with pytest.raises(ValueError, match="frozen plan order"):
        ANALYZE._write_results(
            tmp_path,
            analysis_plan_sha256="plan",
            references={},
            rows=list(reversed(rows)),
            expected_runs=2,
            final=False,
        )
