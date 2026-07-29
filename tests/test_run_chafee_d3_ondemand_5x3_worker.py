from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
SCRIPT = SCRIPTS / "run_chafee_d3_ondemand_5x3_worker.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_chafee_d3_ondemand_5x3_worker",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORKER
SPEC.loader.exec_module(WORKER)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _install_synthetic_training_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_attempt: int = 2,
    checkpoint_attempt: int | None = None,
) -> tuple[Path, Path, Path]:
    training_root = tmp_path / "training"
    sources: dict[str, dict[str, Any]] = {}
    for dataset in range(1, 6):
        path = tmp_path / "archive" / f"dataset_{dataset}" / "train_data.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"dataset-{dataset}".encode())
        sources[f"train_data_dataset_{dataset}"] = {
            "path": str(path.resolve()),
            "sha256": WORKER.base._sha256(path),
            "size_bytes": path.stat().st_size,
        }

    architecture = {
        "high_dims": 64,
        "low_dims": 3,
        "encoder": {"hidden_shapes": [64, 32]},
        "latent_map": {"hidden_shapes": [32, 32]},
        "decoder": {"hidden_shapes": [32, 64]},
    }
    trials = [
        {
            "plan_index": index,
            "dataset": dataset,
            "training_seed": seed,
            "run_id": f"dataset_{dataset:02d}_seed_{seed:02d}_lr3e3_e4000",
            "training_spec": {
                "run_id": (
                    f"dataset_{dataset:02d}_seed_{seed:02d}_lr3e3_e4000"
                ),
                "seed": seed,
                "epochs": 4_000,
                "learning_rate": 0.003,
            },
        }
        for index, (dataset, seed) in enumerate(
            (
                (dataset, seed)
                for dataset in range(1, 6)
                for seed in range(3)
            ),
            start=1,
        )
    ]
    plan = {
        "architecture": architecture,
        "trials": trials,
        "sources": sources,
    }
    plan_sha256 = WORKER.base._payload_sha256(plan)
    _write_json(
        training_root / "experiment_plan.json",
        {
            "schema_version": 1,
            "plan_sha256": plan_sha256,
            "plan": plan,
        },
    )

    target_run_id = "dataset_01_seed_00_lr3e3_e4000"
    selected_attempt = (
        target_attempt if checkpoint_attempt is None else checkpoint_attempt
    )
    target_run_root = training_root / "runs" / target_run_id
    checkpoint = (
        target_run_root
        / "attempts"
        / f"attempt_{selected_attempt:03d}"
        / "models"
        / "autoencoder.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"frozen-checkpoint")
    checkpoint.with_suffix(".json").write_text(
        json.dumps({"version": 1, "arch": architecture}),
        encoding="utf-8",
    )

    for trial in trials:
        run_id = str(trial["run_id"])
        run_root = training_root / "runs" / run_id
        run_spec = {
            "schema_version": 1,
            "plan_sha256": plan_sha256,
            "run": trial["training_spec"],
        }
        _write_json(run_root / "run_spec.json", run_spec)
        completion: dict[str, Any] = {
            "schema_version": 1,
            "status": "completed",
            "plan_sha256": plan_sha256,
            "attempt": target_attempt if run_id == target_run_id else 1,
            "run": trial["training_spec"],
        }
        if run_id == target_run_id:
            attempt_root = (
                run_root
                / "attempts"
                / f"attempt_{target_attempt:03d}"
            )
            attempt_root.mkdir(parents=True, exist_ok=True)
            training_summary = {
                "arch": architecture,
                "seed": 0,
                "epochs_completed": 4_000,
            }
            _write_json(
                attempt_root / "training_summary.json",
                training_summary,
            )
            artifact_manifest = {
                "plan_sha256": plan_sha256,
                "matched_d3_trial": trial,
                "architecture": architecture,
            }
            artifact_manifest_path = attempt_root / "artifact_manifest.json"
            _write_json(artifact_manifest_path, artifact_manifest)
            completion["checkpoint"] = {
                "path": str(checkpoint.relative_to(run_root)),
                "sha256": WORKER.base._sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            }
            completion["artifact_manifest"] = {
                "path": str(artifact_manifest_path.relative_to(run_root)),
                "sha256": WORKER.base._sha256(artifact_manifest_path),
            }
        _write_json(run_root / "completed.json", completion)

    trajectories = tmp_path / "reference" / "traj_attractors.pkl"
    roots = tmp_path / "reference" / "stable_solutions.csv"
    trajectories.parent.mkdir(parents=True, exist_ok=True)
    trajectories.write_bytes(b"trusted-trajectories")
    roots.write_bytes(b"trusted-roots")
    monkeypatch.setattr(WORKER, "TRAINING_ROOT", training_root)
    monkeypatch.setattr(WORKER, "TRAJECTORIES", trajectories)
    monkeypatch.setattr(WORKER, "STABLE_ROOTS", roots)
    monkeypatch.setattr(
        WORKER,
        "TRAJECTORIES_SHA256",
        WORKER.base._sha256(trajectories),
    )
    monkeypatch.setattr(
        WORKER,
        "STABLE_ROOTS_SHA256",
        WORKER.base._sha256(roots),
    )
    return training_root, target_run_root, checkpoint


@pytest.mark.parametrize(
    "run_id",
    (
        "dataset_1_seed_00_lr3e3_e4000",
        "dataset_01_seed_03_lr3e3_e4000",
        "dataset_06_seed_00_lr3e3_e4000",
        "../dataset_01_seed_00_lr3e3_e4000",
    ),
)
def test_resolver_rejects_run_ids_outside_frozen_matrix(run_id: str) -> None:
    with pytest.raises(ValueError, match="run id"):
        WORKER._resolve_inputs(run_id)


def test_resolver_binds_checkpoint_to_declared_source_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, run_root, checkpoint = _install_synthetic_training_archive(
        tmp_path,
        monkeypatch,
        target_attempt=2,
    )

    resolved = WORKER._resolve_inputs("dataset_01_seed_00_lr3e3_e4000")

    assert resolved["dataset"] == 1
    assert resolved["training_seed"] == 0
    assert resolved["run_root"] == run_root.resolve()
    assert resolved["attempt_root"] == (
        run_root / "attempts" / "attempt_002"
    ).resolve()
    assert resolved["checkpoint"] == checkpoint.resolve()
    assert resolved["attempt_root"] in resolved["checkpoint"].parents
    assert resolved["sources"]["checkpoint"]["sha256"] == (
        WORKER.base._sha256(checkpoint)
    )


def test_resolver_rejects_checkpoint_from_a_different_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_synthetic_training_archive(
        tmp_path,
        monkeypatch,
        target_attempt=2,
        checkpoint_attempt=1,
    )

    with pytest.raises(ValueError, match=r"attempt|checkpoint"):
        WORKER._resolve_inputs("dataset_01_seed_00_lr3e3_e4000")


def _minimal_inputs(tmp_path: Path, run_id: str) -> dict[str, Any]:
    training_root = tmp_path / "training"
    run_root = training_root / "runs" / run_id
    attempt_root = run_root / "attempts" / "attempt_001"
    return {
        "run_id": run_id,
        "run_root": run_root,
        "attempt_root": attempt_root,
    }


def test_terminal_pointer_prevents_new_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "dataset_01_seed_00_lr3e3_e4000"
    output_root = tmp_path / "analysis"
    inputs = _minimal_inputs(tmp_path, run_id)
    monkeypatch.setattr(WORKER, "TRAINING_ROOT", tmp_path / "training")
    terminal = output_root / "runs" / run_id / "completed.json"
    _write_json(terminal, {"status": "complete"})

    with pytest.raises(FileExistsError, match="terminal marker"):
        WORKER._allocate_attempt(output_root, inputs)


def test_failed_attempt_is_immutable_and_retry_gets_next_number(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "dataset_01_seed_00_lr3e3_e4000"
    output_root = tmp_path / "analysis"
    inputs = _minimal_inputs(tmp_path, run_id)
    monkeypatch.setattr(WORKER, "TRAINING_ROOT", tmp_path / "training")

    run_root, first, first_number = WORKER._allocate_attempt(
        output_root,
        inputs,
    )
    (first / "failure.json").write_text('{"status":"failed"}')
    same_failure = (first / "failure.json").read_bytes()
    second_run_root, second, second_number = WORKER._allocate_attempt(
        output_root,
        inputs,
    )

    assert run_root == second_run_root
    assert first_number == 1 and first.name == "attempt_001"
    assert second_number == 2 and second.name == "attempt_002"
    assert first.is_dir() and (first / "failure.json").read_bytes() == same_failure


def test_analysis_plan_publish_is_atomic_under_concurrent_bootstrap(
    tmp_path: Path,
) -> None:
    source_names = (
        "training_plan",
        "trajectories",
        "stable_roots",
        "worker",
        "ondemand_backend",
        "study_helpers",
        "basin_statistics_implementation",
        "morse_implementation",
    )
    inputs = {
        "plan_sha256": "frozen-training-plan",
        "sources": {
            name: {"path": f"/source/{name}", "sha256": name, "size_bytes": 1}
            for name in source_names
        },
    }

    def publish() -> tuple[Path, str]:
        return WORKER._ensure_analysis_plan(
            tmp_path / "analysis",
            inputs=inputs,
            runtime={"cmgdb_native": {"sha256": "native"}},
            device="mps",
            max_edges=1_200_000_000,
            max_forward_points=800_000,
            rss_sample_seconds=0.1,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: publish(), range(16)))

    assert len({path for path, _ in results}) == 1
    assert len({digest for _, digest in results}) == 1
    envelope = json.loads(results[0][0].read_text(encoding="utf-8"))
    assert envelope["plan_sha256"] == results[0][1]
    assert not list((tmp_path / "analysis").glob(".analysis_plan.*.tmp"))


def test_worker_rejects_output_nested_in_target_training_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "dataset_01_seed_00_lr3e3_e4000"
    inputs = _minimal_inputs(tmp_path, run_id)
    monkeypatch.setattr(WORKER, "TRAINING_ROOT", tmp_path / "training")

    with pytest.raises(ValueError, match="overlap"):
        WORKER._assert_safe_output_root(
            inputs["run_root"] / "analysis",
            inputs,
        )


def test_worker_rejects_output_anywhere_inside_training_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "dataset_01_seed_00_lr3e3_e4000"
    training_root = tmp_path / "training"
    inputs = _minimal_inputs(tmp_path, run_id)
    monkeypatch.setattr(WORKER, "TRAINING_ROOT", training_root)

    with pytest.raises(ValueError, match="overlap"):
        WORKER._assert_safe_output_root(
            training_root / "analysis",
            inputs,
        )


def _reference_truth_and_prediction() -> tuple[np.ndarray, np.ndarray]:
    truth = np.concatenate(
        (
            np.full(3_909, -1, dtype=np.int64),
            np.zeros(2_138, dtype=np.int64),
            np.full(3_953, 1, dtype=np.int64),
        )
    )
    predicted = np.full(10_000, -4, dtype=np.int32)
    predicted[truth == -1] = 11
    predicted[truth == 1] = 29
    return truth, predicted


def test_classification_payload_preserves_strict_conditioning() -> None:
    truth, predicted = _reference_truth_and_prediction()

    payload = WORKER._classification_payload(
        truth=truth,
        predicted=predicted,
        negative_attractor=11,
        positive_attractor=29,
    )

    assert payload["total_trajectories"] == 10_000
    assert payload["excluded_zero_trajectories"] == 2_138
    assert payload["conditioned_trajectories"] == 7_862
    assert payload["counts"] == {
        "outside_both_basins": 0,
        "misclassified_in_negative_basin": 0,
        "misclassified_in_positive_basin": 0,
        "correctly_classified_in_negative_basin": 3_909,
        "correctly_classified_in_positive_basin": 3_953,
    }
    assert sum(payload["percentages"].values()) == pytest.approx(100.0)


def test_classification_payload_rejects_scientifically_invalid_conditioning() -> None:
    truth, predicted = _reference_truth_and_prediction()
    truth[3_909] = 1

    with pytest.raises(ValueError, match="7,862"):
        WORKER._classification_payload(
            truth=truth,
            predicted=predicted,
            negative_attractor=11,
            positive_attractor=29,
        )


def test_classification_payload_rejects_one_attractor_for_both_roots() -> None:
    truth, predicted = _reference_truth_and_prediction()

    with pytest.raises(ValueError, match="distinct"):
        WORKER._classification_payload(
            truth=truth,
            predicted=predicted,
            negative_attractor=11,
            positive_attractor=11,
        )
