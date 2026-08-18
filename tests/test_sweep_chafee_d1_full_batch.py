"""Focused tests for the exploratory direct-full-batch sweep runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from latentdynamics.config import ArchConfig


def _load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sweep_chafee_d1_full_batch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sweep_chafee_d1_full_batch",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SWEEP = _load_module()


def _tiny_arch() -> ArchConfig:
    return ArchConfig(
        high_dims=3,
        low_dims=1,
        encoder={
            "hidden_shapes": [4],
            "activation": "tanh",
            "out_activation": "none",
        },
        latent_map={
            "hidden_shapes": [4],
            "activation": "tanh",
            "out_activation": "none",
        },
        decoder={
            "hidden_shapes": [4],
            "activation": "tanh",
            "out_activation": "none",
        },
    )


def _tiny_pairs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(101)
    return (
        rng.normal(size=(8, 3)).astype(np.float64),
        rng.normal(size=(8, 3)).astype(np.float64),
    )


def _patch_external_inputs(monkeypatch, tmp_path: Path) -> None:
    sources: dict[str, dict[str, object]] = {}
    for name in (
        "train_data",
        "canonical_checkpoint",
        "canonical_architecture_sidecar",
        "reference_training_implementation",
        "sweep_runner_implementation",
    ):
        path = tmp_path / "frozen_sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        sources[name] = {
            "path": str(path),
            "sha256": SWEEP._sha256(path),
            "size_bytes": path.stat().st_size,
        }
    monkeypatch.setattr(
        SWEEP,
        "_current_source_provenance",
        lambda _train_data=SWEEP.TRAIN_DATA: sources,
    )
    monkeypatch.setattr(SWEEP, "_load_canonical_arch", _tiny_arch)
    monkeypatch.setattr(
        SWEEP,
        "_load_training_pairs",
        lambda _train_data=SWEEP.TRAIN_DATA: _tiny_pairs(),
    )


def test_plan_hash_detects_a_modified_frozen_matrix() -> None:
    plan = {"runs": [{"run_id": "seed_00", "seed": 0}]}
    envelope = SWEEP._plan_envelope(plan)
    assert SWEEP._validate_plan_envelope(envelope) == plan

    envelope["plan"]["runs"][0]["seed"] = 9
    with pytest.raises(ValueError, match="plan hash mismatch"):
        SWEEP._validate_plan_envelope(envelope)


def test_completed_run_is_hash_verified_and_skipped_on_resume(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_external_inputs(monkeypatch, tmp_path)
    output = tmp_path / "sweep"
    run = SWEEP.FullBatchRunSpec(
        run_id="seed_04_test",
        seed=4,
        epochs=2,
        learning_rate=1e-3,
    )

    first = SWEEP.run_sweep(
        output_dir=output,
        device_name="cpu",
        run_specs=(run,),
        verbose=False,
    )
    assert first["all_runs_completed"] is True
    assert first["runs"][0]["status"] == "completed"

    def fail_if_called(**_kwargs):
        raise AssertionError("verified completed run should not retrain")

    monkeypatch.setattr(SWEEP, "train_reference_full_batch", fail_if_called)
    second = SWEEP.run_sweep(output_dir=output, verbose=False)

    assert second["all_runs_completed"] is True
    assert second["runs"][0]["status"] == "already_completed"
    attempts = output / "runs" / run.run_id / "attempts"
    assert [path.name for path in attempts.iterdir()] == ["attempt_001"]
    assert (output / "summaries" / "invocation_0001.json").is_file()
    assert (output / "summaries" / "invocation_0002.json").is_file()


def test_one_failure_does_not_prevent_later_runs_and_resume_retries_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_external_inputs(monkeypatch, tmp_path)
    output = tmp_path / "sweep"
    runs = tuple(
        SWEEP.FullBatchRunSpec(
            run_id=f"seed_{seed:02d}_test",
            seed=seed,
            epochs=1,
            learning_rate=1e-3,
        )
        for seed in range(3)
    )
    real_train = SWEEP.train_reference_full_batch

    def fail_seed_one(**kwargs):
        if kwargs["seed"] == 1:
            raise RuntimeError("synthetic per-run failure")
        return real_train(**kwargs)

    monkeypatch.setattr(SWEEP, "train_reference_full_batch", fail_seed_one)
    first = SWEEP.run_sweep(
        output_dir=output,
        device_name="cpu",
        run_specs=runs,
        verbose=False,
    )
    assert [row["status"] for row in first["runs"]] == [
        "completed",
        "failed",
        "completed",
    ]
    assert first["counts"] == {
        "completed": 2,
        "failed": 1,
        "invalid_existing_run": 0,
        "total": 3,
    }

    monkeypatch.setattr(SWEEP, "train_reference_full_batch", real_train)
    second = SWEEP.run_sweep(output_dir=output, verbose=False)
    assert [row["status"] for row in second["runs"]] == [
        "already_completed",
        "completed",
        "already_completed",
    ]
    retry_root = output / "runs" / runs[1].run_id / "attempts"
    assert sorted(path.name for path in retry_root.iterdir()) == [
        "attempt_001",
        "attempt_002",
    ]


def test_corrupted_completed_artifact_is_never_overwritten_or_retrained(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_external_inputs(monkeypatch, tmp_path)
    output = tmp_path / "sweep"
    run = SWEEP.FullBatchRunSpec(
        run_id="seed_07_test",
        seed=7,
        epochs=1,
        learning_rate=1e-3,
    )
    SWEEP.run_sweep(
        output_dir=output,
        device_name="cpu",
        run_specs=(run,),
        verbose=False,
    )
    checkpoint = (
        output
        / "runs"
        / run.run_id
        / "attempts"
        / "attempt_001"
        / "models"
        / "autoencoder.pt"
    )
    original_size = checkpoint.stat().st_size
    with checkpoint.open("ab") as stream:
        stream.write(b"corruption")

    def fail_if_called(**_kwargs):
        raise AssertionError("an invalid completed run must not be retrained")

    monkeypatch.setattr(SWEEP, "train_reference_full_batch", fail_if_called)
    summary = SWEEP.run_sweep(output_dir=output, verbose=False)

    assert summary["runs"][0]["status"] == "invalid_existing_run"
    assert summary["counts"]["invalid_existing_run"] == 1
    assert checkpoint.stat().st_size == original_size + len(b"corruption")
    attempts = checkpoint.parents[2]
    assert [path.name for path in attempts.iterdir()] == ["attempt_001"]


def test_refuses_existing_unmanaged_output_and_mismatched_resume_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_external_inputs(monkeypatch, tmp_path)
    unmanaged = tmp_path / "unmanaged"
    unmanaged.mkdir()
    with pytest.raises(FileExistsError, match="without sweep_plan"):
        SWEEP.run_sweep(output_dir=unmanaged, verbose=False)

    output = tmp_path / "managed"
    original = SWEEP.FullBatchRunSpec(
        run_id="seed_01_test",
        seed=1,
        epochs=1,
    )
    SWEEP.run_sweep(
        output_dir=output,
        device_name="cpu",
        run_specs=(original,),
        verbose=False,
    )
    changed = SWEEP.FullBatchRunSpec(
        run_id="seed_01_test",
        seed=2,
        epochs=1,
    )
    with pytest.raises(ValueError, match="differs from the frozen"):
        SWEEP.run_sweep(
            output_dir=output,
            run_specs=(changed,),
            verbose=False,
        )


def test_matrix_json_allows_defaults_and_rejects_unknown_keys(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [{"run_id": "seed_12", "seed": 12}],
            }
        ),
        encoding="utf-8",
    )
    specs = SWEEP._load_matrix(valid)
    assert specs == (
        SWEEP.FullBatchRunSpec(run_id="seed_12", seed=12),
    )

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {"run_id": "seed_12", "seed": 12, "batch_size": 10}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown run-spec"):
        SWEEP._load_matrix(invalid)
