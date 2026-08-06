"""Tests for paper-faithful fixed ambient-box scaling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latentdynamics.cli import scale_data
from latentdynamics.config import (
    ArchConfig,
    CMGDBConfig,
    DataConfig,
    ExperimentConfig,
    PathsConfig,
    SystemConfig,
    TrainingConfig,
)
from latentdynamics.sampling import FixedBoundsScaler, fit_fixed_bounds_scaler, load_scaler


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        system=SystemConfig(name="ives"),
        arch=ArchConfig(num_layers=1, hidden_shape=4, high_dims=3, low_dims=2),
        training=TrainingConfig(
            learning_rate=1e-3,
            batch_size=2,
            epochs=1,
            patience=2,
            lr_patience=1,
            loss_weights=[1.0, 1.0, 1.0],
        ),
        data=DataConfig(
            sampling_method="uniform",
            scaling="fixed_bounds",
            n_samples_train=2,
            n_samples_val=1,
            n_iterations=1,
        ),
        cmgdb=CMGDBConfig(),
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
    )


def _write_train_csv(cfg: ExperimentConfig) -> None:
    cfg.paths.data_dir.mkdir(parents=True)
    pairs = np.array(
        [
            [-2.0, -5.0, -1.0, -1.8, -4.0, -1.4],
            [0.0, 0.0, 0.0, 0.3, 0.6, 0.2],
        ]
    )
    np.savetxt(
        cfg.paths.data_dir / "train.csv",
        pairs,
        delimiter=",",
        header="x0,x1,x2,y0,y1,y2",
        comments="",
    )


def test_fixed_bounds_scaler_matches_archived_normalization() -> None:
    lower = np.array([-3.0, -7.5, -3.0])
    upper = np.array([1.5, 1.5, 1.5])
    epsilon = 1e-6
    scaler = fit_fixed_bounds_scaler(lower, upper, epsilon=epsilon)
    points = np.vstack([lower, upper, [-2.0, -5.0, -1.0]])

    scaled = scaler.transform(points)

    np.testing.assert_allclose(scaled, (points - lower) / (upper - lower + epsilon))
    np.testing.assert_allclose(scaler.inverse_transform(scaled), points, rtol=1e-14, atol=1e-14)


def test_fixed_bounds_mode_records_and_checks_all_scaling_inputs(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _write_train_csv(cfg)

    scale_data.run(cfg, verbose=False)

    scaler = load_scaler(cfg.paths.scaler_path("train"))
    assert isinstance(scaler, FixedBoundsScaler)
    metadata = json.loads(
        scale_data.scaler_metadata_path(cfg.paths.scaler_path("train")).read_text()
    )
    assert metadata["lower_bounds"] == [-3.0, -7.5, -3.0]
    assert metadata["upper_bounds"] == [1.5, 1.5, 1.5]
    assert metadata["scaling_epsilon"] == 1e-6
    assert scale_data.scaler_is_current(cfg, "train")

    cfg.data.scaling_epsilon = 2e-6
    assert not scale_data.scaler_is_current(cfg, "train")

    cfg.data.scaling_epsilon = 1e-6
    cfg.system.params["upper_bounds"] = [1.6, 1.5, 1.5]
    assert not scale_data.scaler_is_current(cfg, "train")


def test_scaling_epsilon_default_preserves_archived_value() -> None:
    data = DataConfig(
        sampling_method="uniform",
        scaling="fixed_bounds",
        n_samples_train=1,
        n_samples_val=1,
        n_iterations=1,
    )

    assert data.scaling_epsilon == 1e-6
