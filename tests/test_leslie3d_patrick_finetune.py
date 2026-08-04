"""Contract tests for the isolated Patrick Leslie3D warm-start experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from latentdynamics.cli import scale_data
from latentdynamics.cli import train as train_cli
from latentdynamics.cli.pipeline import _check_read_only, plan_cells
from latentdynamics.config import ArchConfig, TrainingConfig, load_config
from latentdynamics.models import build_autoencoder
from latentdynamics.training import LossHistory, Trainer, save_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "leslie3d_example2_patrick_finetune_4x"
PATRICK_MODELS = REPO_ROOT / "replay_sources" / "leslie3d_example2" / "models"
PATRICK_SCALER = (
    REPO_ROOT / "replay_sources" / "leslie3d_example2" / "data" / "scalers" / "scaler"
)

EXPECTED_SOURCE_HASHES = {
    "encoder.pt": "e581773b1ab0dfdb1002ffc1542331b71398b4c7cb37e323c653f47c4fb67255",
    "dynamics.pt": "b062ae69cd855f3ff304a46a3532b45048f9628f5990032df400831821d92d60",
    "decoder.pt": "855a1eee3bfa6f57935cd58b9241725c70eca04ef8c1aadee267b04fbff0b57f",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_finetune_config_isolated_contract():
    cfg = load_config(CONFIG_NAME)

    assert cfg.system.params == {
        "th1": 28.9,
        "th2": 29.8,
        "th3": 22.0,
        "survival_p1": 0.7,
        "survival_p2": 0.7,
    }
    assert cfg.data.n_samples_train == 32_000
    assert cfg.data.n_samples_val == 8_000
    assert cfg.data.n_iterations == 20
    assert cfg.training.learning_rate == pytest.approx(1e-4)
    assert cfg.training.epochs == 300
    assert cfg.training.patience == 50
    assert cfg.training.warm_start_checkpoint_dir == Path(
        "replay_sources/leslie3d_example2/models"
    )
    assert cfg.paths.output_dir == Path("output/leslie3d_example2_patrick_finetune_4x")
    assert cfg.paths.scaler_read_only
    assert cfg.paths.flat_scaler
    assert cfg.seeds == [0, 1, 2]
    assert [cell["output_dir"] for cell in plan_cells(cfg)] == [
        "output/leslie3d_example2_patrick_finetune_4x/seed_0",
        "output/leslie3d_example2_patrick_finetune_4x/seed_1",
        "output/leslie3d_example2_patrick_finetune_4x/seed_2",
    ]


def test_archived_source_payloads_have_expected_hashes():
    assert {name: _sha256(PATRICK_MODELS / name) for name in EXPECTED_SOURCE_HASHES} == (
        EXPECTED_SOURCE_HASHES
    )
    assert _sha256(PATRICK_SCALER) == (
        "bb908b946d259fd6aa6a716cc003f789631e21bc7c9aa0a6a64c09ac629aa5e1"
    )


def test_protected_scaler_fails_closed():
    cfg = load_config(CONFIG_NAME)
    before = _sha256(PATRICK_SCALER)

    with pytest.raises(RuntimeError, match="scaler_read_only"):
        scale_data.run(cfg, "train", verbose=False)
    with pytest.raises(RuntimeError, match="scaler_read_only"):
        _check_read_only(cfg, ["scale"], force_overwrite=True)

    assert _sha256(PATRICK_SCALER) == before


def test_prefit_validation_runs_on_trainer_device_without_updates():
    arch = ArchConfig(num_layers=1, hidden_shape=4, high_dims=3, low_dims=2)
    training = TrainingConfig(
        learning_rate=1e-4,
        batch_size=4,
        epochs=2,
        patience=2,
        lr_patience=1,
        loss_weights=[100, 10, 20],
    )
    inputs = torch.linspace(0.0, 1.0, 24).reshape(8, 3)
    dataset = TensorDataset(inputs, torch.flip(inputs, dims=(0,)))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = build_autoencoder(arch)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        training_cfg=training,
        arch_cfg=arch,
        device=torch.device("cpu"),
        verbose=False,
    )

    losses = trainer.evaluate_validation()

    assert set(losses) == {
        "loss_reconstruction",
        "loss_prediction",
        "loss_semiconjugacy",
        "loss_total",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert next(trainer.model.parameters()).device == torch.device("cpu")
    assert not trainer.model.training
    assert trainer.history.train == {}
    assert trainer.history.val == {}
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0.0, atol=0.0)


def _one_epoch_trainer() -> Trainer:
    arch = ArchConfig(num_layers=1, hidden_shape=4, high_dims=3, low_dims=2)
    training = TrainingConfig(
        learning_rate=1e-4,
        batch_size=4,
        epochs=1,
        patience=2,
        lr_patience=1,
        loss_weights=[1, 1, 1],
    )
    values = torch.zeros(4, 3)
    loader = DataLoader(TensorDataset(values, values), batch_size=4)
    return Trainer(
        model=build_autoencoder(arch),
        train_loader=loader,
        val_loader=loader,
        training_cfg=training,
        arch_cfg=arch,
        device=torch.device("cpu"),
        verbose=False,
    )


def test_registered_baseline_is_restored_when_update_is_worse(monkeypatch):
    trainer = _one_epoch_trainer()
    original = {key: value.detach().clone() for key, value in trainer.model.state_dict().items()}
    trainer.register_baseline({"loss_total": 1.0})

    def fake_epoch(_loader, *, training):
        if training:
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.add_(1.0)
            return {"loss_total": 1.0}
        return {"loss_total": 2.0}

    monkeypatch.setattr(trainer, "_run_epoch", fake_epoch)
    trainer.fit()

    assert trainer.best_epoch == -1
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0.0, atol=0.0)


def test_training_epoch_replaces_registered_baseline_when_improved(monkeypatch):
    trainer = _one_epoch_trainer()
    original = {key: value.detach().clone() for key, value in trainer.model.state_dict().items()}
    trainer.register_baseline({"loss_total": 1.0})

    def fake_epoch(_loader, *, training):
        if training:
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.add_(1.0)
            return {"loss_total": 1.0}
        return {"loss_total": 0.5}

    monkeypatch.setattr(trainer, "_run_epoch", fake_epoch)
    trainer.fit()

    assert trainer.best_epoch == 0
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, original[key] + 1.0, rtol=0.0, atol=0.0)


def test_train_warm_starts_weights_and_records_source(monkeypatch, tmp_path):
    cfg = load_config(CONFIG_NAME).model_copy(deep=True)
    cfg.training.warm_start_checkpoint_dir = PATRICK_MODELS
    cfg.paths.output_dir = tmp_path / "fine_tune"

    monkeypatch.setattr(train_cli, "_build_loaders", lambda *_args: (object(), object()))

    captured: dict[str, object] = {}
    events: list[str] = []

    class CapturingTrainer:
        def __init__(self, *, model, training_cfg, arch_cfg, **_kwargs):
            captured["model"] = model
            captured["learning_rate"] = training_cfg.learning_rate
            self.model = model
            self.arch = arch_cfg
            self.best_epoch = 0

        def evaluate_validation(self):
            events.append("initial_val")
            return {"loss_total": 0.75, "loss_semiconjugacy": 0.25}

        def register_baseline(self, breakdown):
            events.append("register_baseline")
            assert breakdown == {"loss_total": 0.75, "loss_semiconjugacy": 0.25}

        def fit(self):
            events.append("fit")
            history = LossHistory()
            history.append_train({"loss_total": 1.0})
            history.append_val({"loss_total": 0.5})
            return history

        def save(self, output_dir):
            save_checkpoint(self.model, self.arch, Path(output_dir) / "models")

    monkeypatch.setattr(train_cli, "Trainer", CapturingTrainer)

    before = {name: _sha256(PATRICK_MODELS / name) for name in EXPECTED_SOURCE_HASHES}
    train_cli.run(cfg, seed=1, device=None, verbose=False)
    after = {name: _sha256(PATRICK_MODELS / name) for name in EXPECTED_SOURCE_HASHES}

    assert captured["learning_rate"] == pytest.approx(1e-4)
    assert captured["model"].__class__.__name__ == "LatentDynamicsAutoencoder"
    assert events == ["initial_val", "register_baseline", "fit"]
    assert before == after == EXPECTED_SOURCE_HASHES

    summary = json.loads((tmp_path / "fine_tune" / "training_summary.json").read_text())
    assert summary["initialization"] == {
        "type": "warm_start_weights",
        "checkpoint_dir": str(PATRICK_MODELS),
        "checkpoint_sha256": EXPECTED_SOURCE_HASHES,
        "optimizer_state_restored": False,
        "scheduler_state_restored": False,
    }
    assert summary["initial_val"] == {
        "loss_total": 0.75,
        "loss_semiconjugacy": 0.25,
    }
    assert summary["best_source"] == "training_epoch"
    assert summary["selected_val"] == {"loss_total": 0.5}


def test_train_summary_selects_warm_start_when_no_epoch_improves(monkeypatch, tmp_path):
    cfg = load_config(CONFIG_NAME).model_copy(deep=True)
    cfg.training.warm_start_checkpoint_dir = PATRICK_MODELS
    cfg.paths.output_dir = tmp_path / "baseline_selected"
    monkeypatch.setattr(train_cli, "_build_loaders", lambda *_args: (object(), object()))

    class BaselineSelectedTrainer:
        def __init__(self, *, model, arch_cfg, **_kwargs):
            self.model = model
            self.arch = arch_cfg
            self.best_epoch = -1

        def evaluate_validation(self):
            return {"loss_total": 0.75, "loss_semiconjugacy": 0.25}

        def register_baseline(self, breakdown):
            assert breakdown["loss_total"] == pytest.approx(0.75)

        def fit(self):
            history = LossHistory()
            history.append_train({"loss_total": 1.25, "loss_semiconjugacy": 0.4})
            history.append_val({"loss_total": 1.0, "loss_semiconjugacy": 0.3})
            return history

        def save(self, output_dir):
            save_checkpoint(self.model, self.arch, Path(output_dir) / "models")

    monkeypatch.setattr(train_cli, "Trainer", BaselineSelectedTrainer)
    train_cli.run(cfg, seed=0, device=None, verbose=False)

    summary = json.loads(
        (tmp_path / "baseline_selected" / "training_summary.json").read_text()
    )
    assert summary["best_epoch"] == -1
    assert summary["best_source"] == "warm_start_initial"
    assert summary["selected_val"] == {
        "loss_total": 0.75,
        "loss_semiconjugacy": 0.25,
    }
    final_losses = (tmp_path / "baseline_selected" / "final_losses.txt").read_text()
    assert "best_epoch: -1" in final_losses
    assert "best_source: warm_start_initial" in final_losses
    assert "val_loss_total: 7.500000e-01" in final_losses
    assert "val_loss_semiconjugacy: 2.500000e-01" in final_losses
