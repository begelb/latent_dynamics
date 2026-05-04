"""Tiny end-to-end pipeline: make_data -> scale_data -> train -> load checkpoint."""

from __future__ import annotations

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


@pytest.mark.slow
def test_pipeline_coral(tmp_path):
    cfg = ExperimentConfig(
        system=SystemConfig(name="coral"),
        arch=ArchConfig(num_layers=1, hidden_shape=8, high_dims=13, low_dims=1),
        training=TrainingConfig(
            learning_rate=1e-2, batch_size=8, epochs=3, patience=100, loss_weights=[1, 1, 1]
        ),
        data=DataConfig(
            sampling_method="uniform",
            n_samples_train=8,
            n_samples_test=8,
            n_iterations=2,
        ),
        cmgdb=CMGDBConfig(),
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
    )

    make_data.run(cfg, verbose=False)
    scale_data.run(cfg, "train", verbose=False)
    train.run(cfg, train_file="train", seed=0, verbose=False)

    ckpt_dir = tmp_path / "out" / "models"
    model, arch = load_checkpoint(ckpt_dir)
    assert arch.high_dims == 13
    assert arch.low_dims == 1

    x = torch.zeros(2, 13)
    fp = model(x, x)
    assert fp.x_t_hat.shape == (2, 13)
    assert fp.z_t.shape == (2, 1)

    # final_losses.txt was emitted.
    assert (tmp_path / "out" / "final_losses.txt").exists()
