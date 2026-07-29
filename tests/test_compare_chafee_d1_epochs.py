from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


def _load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_chafee_d1_epochs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_chafee_d1_epochs",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMPARE = _load_module()


class _IdentityPart(torch.nn.Module):
    def forward(self, values):
        return values


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _IdentityPart()
        self.latent_map = _IdentityPart()
        self.decoder = _IdentityPart()


def test_evaluate_model_reports_exact_zero_identity_errors() -> None:
    values = torch.tensor([[1.0], [2.0], [3.0]])
    metrics = COMPARE._evaluate_model(
        _ToyModel(),
        values,
        values,
        batch_size=2,
    )

    assert metrics == {
        "L1_reconstruction_mse": 0.0,
        "L2_decoded_one_step_prediction_mse": 0.0,
        "L1_plus_L2": 0.0,
        "L3_unconditioned_latent_semiconjugacy_mse": 0.0,
        "global_max_euclidean_latent_residual": 0.0,
    }


def test_prefix_comparison_detects_exact_and_changed_history() -> None:
    current = {
        key: [float(index) for index in range(COMPARE.CURRENT_EPOCHS)]
        for key in (
            "loss_reconstruction",
            "loss_prediction",
            "loss_total",
            "learning_rate",
        )
    }
    extended = {
        key: [*values, 99.0]
        for key, values in current.items()
    }
    exact = COMPARE._prefix_comparison(current, extended)
    assert exact["all_history_arrays_exactly_equal"] is True

    extended["loss_total"][10] += 0.25
    changed = COMPARE._prefix_comparison(current, extended)
    assert changed["all_history_arrays_exactly_equal"] is False
    assert (
        changed["metrics"]["loss_total"]["max_absolute_difference"]
        == pytest.approx(0.25)
    )


def test_load_history_rejects_wrong_epoch_count(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "training_method": "marcio_full_batch",
                "train": {
                    "loss_reconstruction": [1.0],
                    "loss_prediction": [1.0],
                    "loss_total": [2.0],
                    "learning_rate": [0.1],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 2"):
        COMPARE._load_history(path, 2)


def test_current_artifact_guards_match_canonical_files() -> None:
    assert COMPARE._sha256(COMPARE.TRAIN_DATA) == COMPARE.TRAIN_DATA_SHA256
    assert (
        COMPARE._sha256(
            COMPARE.CURRENT_RUN / "models" / "autoencoder.pt"
        )
        == COMPARE.CURRENT_CHECKPOINT_SHA256
    )
    assert (
        COMPARE._sha256(
            COMPARE.CURRENT_RUN / "logs" / "history.json"
        )
        == COMPARE.CURRENT_HISTORY_SHA256
    )


def test_comparison_reports_improvement_direction() -> None:
    result = COMPARE._comparison(
        {"metric": 4.0},
        {"metric": 1.0},
    )["metric"]

    assert result["absolute_change"] == -3.0
    assert result["percent_change"] == -75.0
    assert result["improvement_factor"] == 4.0
