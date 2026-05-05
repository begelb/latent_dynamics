"""Determinism check: same seed -> identical final loss across runs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from latentdynamics.cli import make_data, scale_data, train
from latentdynamics.config import (
    ArchConfig,
    CMGDBConfig,
    DataConfig,
    ExperimentConfig,
    PathsConfig,
    SystemConfig,
    TrainingConfig,
)
from latentdynamics.training import load_checkpoint


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tiny_cfg(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        system=SystemConfig(name="coral"),
        arch=ArchConfig(num_layers=2, hidden_shape=8, high_dims=13, low_dims=1),
        training=TrainingConfig(
            learning_rate=1e-2, batch_size=16, epochs=5, patience=100, loss_weights=[1, 1, 1]
        ),
        data=DataConfig(
            sampling_method="uniform",
            n_samples_train=32,
            n_samples_test=32,
            n_iterations=2,
        ),
        cmgdb=CMGDBConfig(),
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
    )


def _final_loss(cfg: ExperimentConfig) -> float:
    text = (cfg.paths.output_dir / "final_losses.txt").read_text()
    for line in text.strip().splitlines():
        if line.startswith("loss_total:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError("loss_total not found in final_losses.txt")


@pytest.mark.slow
class TestReproducibility:
    def test_same_seed_yields_same_final_loss(self, tmp_path):
        cfg_a = _tiny_cfg(tmp_path / "a")
        cfg_b = _tiny_cfg(tmp_path / "b")

        for cfg in (cfg_a, cfg_b):
            _seed_all(0)
            make_data.run(cfg, verbose=False)
            scale_data.run(cfg, "train", verbose=False)
            train.run(cfg, train_file="train", seed=0, device=torch.device("cpu"), verbose=False)

        loss_a = _final_loss(cfg_a)
        loss_b = _final_loss(cfg_b)
        assert loss_a == pytest.approx(loss_b, abs=1e-6), (
            f"non-deterministic training: {loss_a} vs {loss_b}"
        )

    def test_same_seed_yields_same_checkpoint_weights(self, tmp_path):
        cfg_a = _tiny_cfg(tmp_path / "a")
        cfg_b = _tiny_cfg(tmp_path / "b")

        for cfg in (cfg_a, cfg_b):
            _seed_all(0)
            make_data.run(cfg, verbose=False)
            scale_data.run(cfg, "train", verbose=False)
            train.run(cfg, train_file="train", seed=0, device=torch.device("cpu"), verbose=False)

        m_a, _ = load_checkpoint(cfg_a.paths.output_dir / "models")
        m_b, _ = load_checkpoint(cfg_b.paths.output_dir / "models")
        for (k1, v1), (k2, v2) in zip(
            m_a.state_dict().items(), m_b.state_dict().items(), strict=True
        ):
            assert k1 == k2
            torch.testing.assert_close(v1, v2)

    def test_different_seeds_yield_different_weights(self, tmp_path):
        cfg_a = _tiny_cfg(tmp_path / "a")
        cfg_b = _tiny_cfg(tmp_path / "b")
        for cfg, seed in ((cfg_a, 0), (cfg_b, 1)):
            _seed_all(seed)
            make_data.run(cfg, verbose=False)
            scale_data.run(cfg, "train", verbose=False)
            train.run(cfg, train_file="train", seed=seed, device=torch.device("cpu"), verbose=False)

        m_a, _ = load_checkpoint(cfg_a.paths.output_dir / "models")
        m_b, _ = load_checkpoint(cfg_b.paths.output_dir / "models")
        any_diff = any(
            not torch.allclose(va, vb)
            for (_, va), (_, vb) in zip(
                m_a.state_dict().items(), m_b.state_dict().items(), strict=True
            )
        )
        assert any_diff, "different seeds produced identical state_dicts"
