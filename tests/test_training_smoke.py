"""End-to-end smoke tests for the Trainer."""

from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from latentdynamics.config.schema import ArchConfig, TrainingConfig
from latentdynamics.models import build_autoencoder
from latentdynamics.training import Trainer, load_checkpoint
from latentdynamics.training.trainer import _select_device


def _make_loaders(seed: int = 0) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(64, 4, generator=g)
    Y = torch.rand(64, 4, generator=g)
    train_ds = TensorDataset(X[:48], Y[:48])
    val_ds = TensorDataset(X[48:], Y[48:])
    return DataLoader(train_ds, batch_size=8, shuffle=True), DataLoader(val_ds, batch_size=8)


class TestDeviceSelection:
    def test_trainer_prefers_mps_over_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        assert _select_device() == torch.device("mps")


@pytest.mark.slow
class TestTrainingSmoke:
    def test_loss_decreases_after_training(self):
        torch.manual_seed(0)
        arch = ArchConfig(num_layers=2, hidden_shape=16, high_dims=4, low_dims=2)
        train_cfg = TrainingConfig(
            learning_rate=1e-2,
            batch_size=8,
            epochs=25,
            patience=100,
            loss_weights=[1.0, 1.0, 1.0],
        )
        train_loader, val_loader = _make_loaders()
        model = build_autoencoder(arch)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            training_cfg=train_cfg,
            arch_cfg=arch,
            verbose=False,
            device=torch.device("cpu"),
        )
        history = trainer.fit()
        first = history.train["loss_total"][0]
        last = history.train["loss_total"][-1]
        assert last < first

    def test_save_and_load_checkpoint_roundtrip(self, tmp_path):
        torch.manual_seed(0)
        arch = ArchConfig(num_layers=1, hidden_shape=8, high_dims=4, low_dims=2)
        train_cfg = TrainingConfig(
            learning_rate=1e-2,
            batch_size=8,
            epochs=3,
            patience=100,
            loss_weights=[1, 1, 1],
        )
        train_loader, val_loader = _make_loaders()
        model = build_autoencoder(arch)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            training_cfg=train_cfg,
            arch_cfg=arch,
            verbose=False,
            device=torch.device("cpu"),
        )
        trainer.fit()
        trainer.save(tmp_path)

        ckpt_dir = tmp_path / "models"
        assert (ckpt_dir / "autoencoder.pt").exists()
        assert (ckpt_dir / "autoencoder.json").exists()
        assert (tmp_path / "logs" / "history.json").exists()

        # The loaded model produces the same outputs as the live one.
        loaded, _arch = load_checkpoint(ckpt_dir)
        x = torch.randn(5, 4)
        torch.testing.assert_close(model(x, x).x_t_hat, loaded(x, x).x_t_hat)

    def test_history_json_is_well_formed(self, tmp_path):
        arch = ArchConfig(num_layers=1, hidden_shape=4, high_dims=4, low_dims=2)
        train_cfg = TrainingConfig(
            learning_rate=1e-2,
            batch_size=8,
            epochs=2,
            patience=100,
            loss_weights=[1, 1, 1],
        )
        train_loader, val_loader = _make_loaders()
        model = build_autoencoder(arch)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            training_cfg=train_cfg,
            arch_cfg=arch,
            verbose=False,
            device=torch.device("cpu"),
        )
        trainer.fit()
        trainer.save(tmp_path)
        h = json.loads((tmp_path / "logs" / "history.json").read_text())
        assert set(h.keys()) == {"train", "val"}
        assert set(h["train"].keys()) == {
            "loss_ae1",
            "loss_ae2",
            "loss_dyn",
            "loss_cycle_pred",
            "loss_total",
        }
        assert len(h["train"]["loss_total"]) == 2
