"""Focused tests for the training-only matched Chafee d=3 runner."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from latentdynamics.config import ArchConfig


def _load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_chafee_d3_matched_5x3_training.py"
    )
    scripts = str(script.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(
        "run_chafee_d3_matched_5x3_training",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D3 = _load_module()


def _tiny_arch() -> ArchConfig:
    return ArchConfig(
        high_dims=3,
        low_dims=3,
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
    rng = np.random.default_rng(811)
    return (
        rng.normal(size=(7, 3)).astype(np.float64),
        rng.normal(size=(7, 3)).astype(np.float64),
    )


def _patch_external_sources(monkeypatch, tmp_path: Path) -> None:
    sources = {}
    for dataset in D3.DATASETS:
        path = tmp_path / "sources" / f"dataset_{dataset}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"dataset {dataset}\n", encoding="utf-8")
        sources[f"train_data_dataset_{dataset}"] = {
            "path": str(path.resolve()),
            "sha256": D3._sha256(path),
            "size_bytes": path.stat().st_size,
        }
    monkeypatch.setattr(D3, "_current_sources", lambda: sources)
    monkeypatch.setattr(D3, "_d3_architecture", _tiny_arch)
    monkeypatch.setattr(D3, "_load_training_pairs", lambda _path: _tiny_pairs())
    monkeypatch.setattr(D3, "EPOCHS", 1)


def test_fixed_matrix_and_architecture_are_exact() -> None:
    trials = D3._trial_matrix()
    assert len(trials) == 15
    assert [(trial.dataset, trial.training_seed) for trial in trials] == [
        (dataset, seed)
        for dataset in range(1, 6)
        for seed in range(3)
    ]
    assert all(trial.training_spec.epochs == 4_000 for trial in trials)
    assert all(
        trial.training_spec.learning_rate == 0.003 for trial in trials
    )
    try:
        arch = D3._d3_architecture()
    except FileNotFoundError as exc:
        pytest.skip(f"chafee artifacts not present: {exc}")
    assert arch.high_dims == 64
    assert arch.low_dims == 3
    assert arch.encoder.hidden_shapes == [64, 32]
    assert arch.latent_map.hidden_shapes == [32, 32]
    assert arch.decoder.hidden_shapes == [32, 64]


def test_runner_has_no_analysis_or_cmgdb_import() -> None:
    source = D3.RUNNER_IMPLEMENTATION.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert "CMGDB" not in imported
    assert "chafee_latent_dimension_study" not in imported
    assert "analyze_chafee_d1_full_batch_sweep" not in imported
    assert "repeat_chafee_d1_full_batch" not in imported


def test_plan_hash_rejects_mutation() -> None:
    plan = {
        "resolved_device": "cpu",
        "training_semantics": {"latent_dimension": 3},
        "scope_guards": {
            "training_only": True,
            "cmgdb_imported_or_invoked": False,
        },
        "sources": {},
        "architecture": _tiny_arch().model_dump(mode="json"),
        "trials": [],
    }
    envelope = D3._plan_envelope(plan)
    envelope["plan"]["training_semantics"]["latent_dimension"] = 2
    try:
        D3._validate_plan_envelope(envelope)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("modified plan was accepted")


def test_training_is_resumable_and_preserves_one_attempt_per_completed_trial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_external_sources(monkeypatch, tmp_path)
    output = tmp_path / "matched_d3"
    first = D3.run_experiment(
        output_root=output,
        stage="train",
        verbose=False,
    )
    assert first["status"] == "complete"
    assert first["counts"] == {
        "expected": 15,
        "complete": 15,
        "not_completed": 0,
        "invalid": 0,
    }

    def fail_if_called(**_kwargs):
        raise AssertionError("verified completed trials must not retrain")

    monkeypatch.setattr(D3, "train_reference_full_batch", fail_if_called)
    second = D3.run_experiment(
        output_root=output,
        stage="train",
        verbose=False,
    )
    assert second["status"] == "complete"
    assert all(
        row["status"] == "already_completed"
        for row in second["invocation_rows"]
    )
    for trial in D3._trial_matrix():
        attempts = output / "runs" / trial.run_id / "attempts"
        assert [path.name for path in attempts.iterdir()] == ["attempt_001"]

    plan = D3._read_json(output / "experiment_plan.json")["plan"]
    assert plan["scope_guards"] == {
        "training_only": True,
        "cmgdb_imported_or_invoked": False,
        "morse_graph_or_roa_analysis_performed": False,
        "existing_d3_artifacts_mutated": False,
    }
