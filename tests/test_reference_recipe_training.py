"""Focused tests for the paper-faithful Chafee--Infante training loop."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import Adam

from latentdynamics.config import ArchConfig
from latentdynamics.models import build_autoencoder
from latentdynamics.training import load_checkpoint, train_reference_full_batch


def _tiny_arch() -> ArchConfig:
    return ArchConfig(
        high_dims=3,
        low_dims=1,
        encoder={"hidden_shapes": [4], "activation": "tanh", "out_activation": "none"},
        latent_map={"hidden_shapes": [4], "activation": "tanh", "out_activation": "none"},
        decoder={"hidden_shapes": [4], "activation": "tanh", "out_activation": "none"},
    )


def _tiny_pairs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    return (
        rng.normal(size=(7, 3)).astype(np.float64),
        rng.normal(size=(7, 3)).astype(np.float64),
    )


def test_one_epoch_matches_the_archived_two_term_adam_update(tmp_path):
    arch = _tiny_arch()
    x, y = _tiny_pairs()
    torch.manual_seed(17)
    initial = build_autoencoder(arch)
    expected = copy.deepcopy(initial)

    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    optimizer = Adam(expected.parameters(), lr=3e-3)
    optimizer.zero_grad()
    z = expected.encoder(x_tensor)
    manual_reconstruction = nn.functional.mse_loss(expected.decoder(z), x_tensor)
    manual_prediction = nn.functional.mse_loss(
        expected.decoder(expected.latent_map(z)),
        y_tensor,
    )
    manual_total = manual_reconstruction + manual_prediction
    manual_total.backward()
    optimizer.step()

    result = train_reference_full_batch(
        arch=arch,
        model=initial,
        x=x,
        y=y,
        epochs=1,
        learning_rate=3e-3,
        seed=91,
        device="cpu",
        output_dir=tmp_path,
    )

    assert result.history["loss_reconstruction"] == pytest.approx(
        [float(manual_reconstruction.detach())]
    )
    assert result.history["loss_prediction"] == pytest.approx(
        [float(manual_prediction.detach())]
    )
    assert result.history["loss_total"] == pytest.approx([float(manual_total.detach())])
    for name, expected_value in expected.state_dict().items():
        torch.testing.assert_close(result.model.state_dict()[name], expected_value)


def test_runs_fixed_epochs_schedules_on_training_loss_and_saves_final_state(
    tmp_path,
    monkeypatch,
):
    import latentdynamics.training.reference_recipe as reference_module

    arch = _tiny_arch()
    x, y = _tiny_pairs()
    seen_metrics: list[float] = []
    real_scheduler = reference_module.ReduceLROnPlateau

    class RecordingScheduler:
        def __init__(self, *args, **kwargs):
            self.delegate = real_scheduler(*args, **kwargs)

        def step(self, metric):
            seen_metrics.append(float(metric))
            self.delegate.step(metric)

    monkeypatch.setattr(reference_module, "ReduceLROnPlateau", RecordingScheduler)
    result = train_reference_full_batch(
        arch=arch,
        x=x,
        y=y,
        epochs=4,
        learning_rate=1e-3,
        seed=5,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        scheduler_patience=1,
    )

    assert len(result.history["loss_total"]) == 4
    assert seen_metrics == pytest.approx(result.history["loss_total"])
    assert result.summary["epochs_completed"] == 4
    assert result.summary["checkpoint_epoch"] == 4
    assert result.summary["checkpoint_selection"] == "final_epoch"
    assert result.summary["validation_used"] is False
    assert result.summary["early_stopping_used"] is False
    assert result.summary["best_weight_restoration_used"] is False

    loaded, loaded_arch = load_checkpoint(tmp_path / "models")
    assert loaded_arch == arch
    for name, live_value in result.model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], live_value.cpu())


def test_same_seed_reproduces_weights_history_and_compact_metadata(tmp_path):
    arch = _tiny_arch()
    x, y = _tiny_pairs()

    results = [
        train_reference_full_batch(
            arch=arch,
            x=x,
            y=y,
            epochs=3,
            learning_rate=1e-3,
            seed=77,
            device="cpu",
            output_dir=tmp_path / label,
        )
        for label in ("a", "b")
    ]

    assert results[0].history == results[1].history
    for (name_a, value_a), (name_b, value_b) in zip(
        results[0].model.state_dict().items(),
        results[1].model.state_dict().items(),
        strict=True,
    ):
        assert name_a == name_b
        torch.testing.assert_close(value_a, value_b, rtol=0, atol=0)

    history_payload = json.loads(results[0].history_path.read_text())
    summary_payload = json.loads(results[0].summary_path.read_text())
    assert history_payload["train"] == results[0].history
    assert history_payload["training_method"] == "reference_full_batch"
    assert summary_payload["objective"] == "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)"
    assert summary_payload["scheduler"]["monitor"] == "train.loss_total"
    assert summary_payload["data"] == {
        "n_pairs": 7,
        "high_dims": 3,
        "dtype": "float32",
        "full_batch": True,
    }
    assert summary_payload["model_initialized_by_helper"] is True
    reproducibility = summary_payload["reproducibility"]
    assert reproducibility["resolved_backend"] == "cpu"
    assert isinstance(reproducibility["deterministic_algorithms_enforced"], bool)
    assert reproducibility[
        "bitwise_reproducible_across_backends_or_runtime_versions"
    ] is False
    assert "may produce numerically different" in reproducibility["limitation"]
    assert reproducibility["runtime"]["torch"] == str(torch.__version__)
    assert results[0].checkpoint_path.exists()
    assert results[0].checkpoint_metadata_path.exists()


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        (np.zeros((2, 2)), np.zeros((2, 2)), "arch.high_dims"),
        (np.zeros((2, 3)), np.zeros((3, 3)), "same shape"),
        (np.zeros(3), np.zeros(3), "rank-2"),
    ],
)
def test_rejects_invalid_full_batch_shapes(tmp_path, x, y, message):
    with pytest.raises(ValueError, match=message):
        train_reference_full_batch(
            arch=_tiny_arch(),
            x=x,
            y=y,
            epochs=1,
            learning_rate=1e-3,
            seed=0,
            device="cpu",
            output_dir=tmp_path,
        )
