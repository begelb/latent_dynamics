from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    script = scripts / "analyze_chafee_d1_grad_accum_milestones.py"
    spec = importlib.util.spec_from_file_location(
        "analyze_chafee_d1_grad_accum_milestones",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZE = _load_module()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_safe_target_requires_fresh_isolated_sibling(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="isolated sibling"):
        ANALYZE._assert_safe_target(source, tmp_path / "nested" / "analysis")
    with pytest.raises(ValueError, match="overlaps protected"):
        ANALYZE._assert_safe_target(source, source)

    target = tmp_path / "analysis"
    ANALYZE._assert_safe_target(source, target)
    target.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ANALYZE._assert_safe_target(source, target)


def test_run_plan_requires_all_predeclared_milestones() -> None:
    plan = {
        "schema_version": 1,
        "status": "frozen_before_training",
        "purpose": "long fixed-horizon 1-D gradient-accumulation experiment",
        "settings": dict(ANALYZE.EXPECTED_SETTINGS),
        "milestone_epochs": list(ANALYZE.MILESTONE_EPOCHS),
        "architecture": {"high_dims": 64, "low_dims": 1},
        "data": {
            "sha256": ANALYZE.single.EXPECTED_TRAINING_SHA256,
            "shape": [30_000, 128],
            "scaling": "none",
            "shuffle": False,
            "drop_last": False,
        },
        "objective": {
            "formula": "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)",
            "weights": [1.0, 1.0, 0.0],
        },
        "early_stopping": False,
        "validation_used": False,
        "basin_or_cmgdb_inputs_allowed": False,
        "accumulation": {
            "microbatches_per_update": 30,
            "last_microbatch_rows": 304,
            "optimizer_steps_per_epoch": 1,
        },
    }
    ANALYZE._verify_run_plan_payload(plan)

    plan["milestone_epochs"] = plan["milestone_epochs"][:-1]
    with pytest.raises(ValueError, match="exact 20 milestones"):
        ANALYZE._verify_run_plan_payload(plan)


def test_milestone_inventory_rejects_hash_or_path_drift(
    tmp_path: Path,
) -> None:
    run = tmp_path / "source"
    epoch = 1_000
    milestone = run / "milestones" / "epoch_01000"
    checkpoint = milestone / "models" / "autoencoder.pt"
    sidecar = milestone / "models" / "autoencoder.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    sidecar.write_bytes(b"sidecar")
    manifest = {
        "schema_version": 1,
        "epoch": epoch,
        "optimizer_updates": epoch,
        "run_plan_sha256": "plan",
        "basin_artifacts_accessed": False,
        "learning_rate": 0.003,
        "train": {
            "loss_reconstruction": 0.1,
            "loss_prediction": 0.2,
            "loss_total": 0.3,
        },
        "checkpoint": {
            "path": "milestones/epoch_01000/models/autoencoder.pt",
            "sha256": ANALYZE.study.sha256_file(checkpoint),
            "sidecar_path": (
                "milestones/epoch_01000/models/autoencoder.json"
            ),
            "sidecar_sha256": ANALYZE.study.sha256_file(sidecar),
        },
    }
    manifest_path = milestone / "manifest.json"
    _write_json(manifest_path, manifest)

    frozen = ANALYZE._verify_milestone(
        run.resolve(),
        epoch=epoch,
        run_plan_sha256="plan",
        verify_model=False,
    )
    assert frozen.epoch == epoch
    assert frozen.checkpoint_sha256 == manifest["checkpoint"]["sha256"]

    manifest["checkpoint"]["path"] = "../outside.pt"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="escapes the run"):
        ANALYZE._verify_milestone(
            run.resolve(),
            epoch=epoch,
            run_plan_sha256="plan",
            verify_model=False,
        )


def test_statistics_fields_combine_both_correct_basins_and_misc() -> None:
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
                "correctly_classified_in_negative_basin": 19.3843805647,
                "correctly_classified_in_positive_basin": 30.6283388451,
                "misclassified_in_negative_basin": 0.0254388196,
                "misclassified_in_positive_basin": 0.0,
                "outside_both_basins": 49.9618417705,
            },
        },
        "uniform_is_bistable": True,
        "adaptive_graph": {"is_bistable": True},
        "roots_define_two_distinct_attractor_basins": True,
        "eligible_for_bistable_dimension_table": True,
    }

    fields = ANALYZE._stats_fields(payload)

    assert fields["correct_combined_count"] == 3_932
    assert fields["correct_combined_percent"] == pytest.approx(
        50.0127194098
    )
    assert fields["misc_count"] == 2


def test_failed_milestone_is_recorded_without_reraising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "diagnostics"
    snapshot = target / "source_snapshot" / "epoch_01000"
    snapshot_checkpoint = snapshot / "models" / "autoencoder.pt"
    snapshot_sidecar = snapshot / "models" / "autoencoder.json"
    snapshot_checkpoint.parent.mkdir(parents=True)
    snapshot_checkpoint.write_bytes(b"checkpoint")
    snapshot_sidecar.write_bytes(b"sidecar")
    snapshot_manifest = snapshot / "manifest.json"
    snapshot_manifest.write_bytes(b"manifest")
    milestone = ANALYZE.FrozenMilestone(
        epoch=1_000,
        manifest=tmp_path / "manifest.json",
        manifest_sha256=ANALYZE.study.sha256_file(snapshot_manifest),
        checkpoint=tmp_path / "autoencoder.pt",
        checkpoint_sha256=ANALYZE.study.sha256_file(snapshot_checkpoint),
        sidecar=tmp_path / "autoencoder.json",
        sidecar_sha256=ANALYZE.study.sha256_file(snapshot_sidecar),
        train={
            "loss_reconstruction": 0.1,
            "loss_prediction": 0.2,
            "loss_total": 0.3,
        },
        learning_rate=0.003,
    )

    monkeypatch.setattr(
        ANALYZE.study,
        "_run_bounds",
        lambda *args, **kwargs: None,
    )

    def fail_coarse(*args, **kwargs):
        raise ValueError("synthetic graph failure")

    monkeypatch.setattr(ANALYZE.study, "_run_precompute_coarse", fail_coarse)
    row = ANALYZE._analyze_milestone(
        target=target,
        snapshot_root=target / "source_snapshot",
        milestone=milestone,
        analysis_plan_sha256="plan",
        inputs=object(),
        device=ANALYZE.torch.device("cpu"),
        batch_points="auto",
        references={},
    )

    assert row["status"] == "failed"
    assert row["error_stage"] == "precompute-coarse"
    assert row["error_message"] == "synthetic graph failure"
    manifest = json.loads(
        (
            target / "epoch_01000" / "analysis_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["completed_stages"] == ["bounds"]
    assert manifest["status"] == "failed"


def test_final_comparison_requires_chronological_complete_inventory(
    tmp_path: Path,
) -> None:
    rows = [
        {"epoch": 2_000, "status": "failed"},
        {"epoch": 1_000, "status": "complete"},
    ]
    with pytest.raises(ValueError, match="chronological"):
        ANALYZE._write_comparison(
            tmp_path,
            analysis_plan_sha256="plan",
            references={},
            rows=rows,
            final=False,
        )

    with pytest.raises(ValueError, match="all 20 milestones"):
        ANALYZE._write_comparison(
            tmp_path,
            analysis_plan_sha256="plan",
            references={},
            rows=[{"epoch": 1_000, "status": "complete"}],
            final=True,
        )
