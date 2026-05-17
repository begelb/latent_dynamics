"""End-to-end smoke test for the make_data and scale_data CLI pipelines."""

from __future__ import annotations

import json

import numpy as np
import pytest

from latentdynamics.cli import make_data, scale_data
from latentdynamics.config import (
    ArchConfig,
    CMGDBConfig,
    DataConfig,
    ExperimentConfig,
    PathsConfig,
    SystemConfig,
    TrainingConfig,
)
from latentdynamics.sampling import load_scaler
from latentdynamics.systems import build_system


def _tiny_cfg(system_name: str, tmp_path, **overrides) -> ExperimentConfig:
    base = {
        "system": SystemConfig(name=system_name),
        "arch": ArchConfig(
            num_layers=1, hidden_shape=4, high_dims=overrides.pop("high_dims", 13), low_dims=1
        ),
        "training": TrainingConfig(
            learning_rate=1e-3, batch_size=8, epochs=2, patience=2, lr_patience=1, loss_weights=[1, 1, 1]
        ),
        "data": DataConfig(
            sampling_method="uniform",
            n_samples_train=overrides.pop("n_samples_train", 4),
            n_samples_val=4,
            n_iterations=2,
        ),
        "cmgdb": CMGDBConfig(),
        "paths": PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
    }
    return ExperimentConfig(**base)


class TestMakeDataRun:
    def test_coral_single_train_file(self, tmp_path):
        cfg = _tiny_cfg("coral", tmp_path, high_dims=13, n_samples_train=4)
        make_data.run(cfg, verbose=False)

        train_csv = tmp_path / "data" / "train.csv"
        val_csv = tmp_path / "data" / "val.csv"
        train_meta = tmp_path / "data" / "train_metadata.json"

        assert train_csv.exists() and val_csv.exists() and train_meta.exists()

        data = np.loadtxt(train_csv, delimiter=",", skiprows=1)
        assert data.shape == (4 * 2, 26)

        meta = json.loads(train_meta.read_text())
        assert meta["system"] == "RedCoralModel"
        assert meta["dataset_name"] == "train"

    def test_leslie3d_train_size_list(self, tmp_path):
        cfg = _tiny_cfg("leslie3d", tmp_path, high_dims=3, n_samples_train=[3, 5])
        make_data.run(cfg, verbose=False)

        for n in (3, 5):
            assert (tmp_path / "data" / f"train_{n}.csv").exists()
            assert (tmp_path / "data" / f"train_{n}_metadata.json").exists()

    def test_matching_existing_data_is_preserved(self, tmp_path):
        cfg = _tiny_cfg("coral", tmp_path, high_dims=13, n_samples_train=4)
        make_data.run(cfg, verbose=False)
        data_dir = cfg.paths.data_dir
        train_csv = data_dir / "train.csv"
        train_csv.write_text("sentinel\n")

        make_data.run(cfg, verbose=False)

        assert train_csv.read_text() == "sentinel\n"

    def test_existing_train_data_with_wrong_n_samples_raises(self, tmp_path):
        cfg = _tiny_cfg("coral", tmp_path, high_dims=13, n_samples_train=4)
        make_data.run(cfg, verbose=False)
        stale_cfg = _tiny_cfg("coral", tmp_path, high_dims=13, n_samples_train=5)

        with pytest.raises(ValueError, match=r"stale existing dataset.*n_samples"):
            make_data.run(stale_cfg, verbose=False)

    def test_existing_train_data_with_wrong_seed_raises(self, tmp_path):
        cfg = _tiny_cfg("coral", tmp_path, high_dims=13, n_samples_train=4)
        make_data.run(cfg, verbose=False)
        stale_cfg = _tiny_cfg("coral", tmp_path, high_dims=13, n_samples_train=4)
        stale_cfg.data.train_seed = cfg.data.train_seed + 1

        with pytest.raises(ValueError, match=r"stale existing dataset.*sampling_seed"):
            make_data.run(stale_cfg, verbose=False)

    def test_partial_existing_data_refuses_overwrite(self, tmp_path):
        cfg = _tiny_cfg("coral", tmp_path, high_dims=13, n_samples_train=4)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("sentinel\n")

        with pytest.raises(FileExistsError):
            make_data.run(cfg, verbose=False)

    def test_adaptive_data_stage_validates_precomputed_files(self, tmp_path):
        cfg = ExperimentConfig(
            system=SystemConfig(name="coral"),
            arch=ArchConfig(num_layers=1, hidden_shape=4, high_dims=13, low_dims=1),
            training=TrainingConfig(
                learning_rate=1e-3, batch_size=8, epochs=2, patience=2, lr_patience=1, loss_weights=[1, 1, 1]
            ),
            data=DataConfig(
                sampling_method="adaptive",
                n_samples_train=500,
                n_samples_val=4,
                n_iterations=2,
                train_files=["train_500_100_adaptive"],
            ),
            cmgdb=CMGDBConfig(),
            paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
        )
        cfg.paths.data_dir.mkdir()
        coral = build_system("coral", {})
        train_meta = {
            "dataset_name": "train_500_100_adaptive",
            "role": "train",
            "system": "RedCoralModel",
            "dimension": coral.dim,
            "model_params": coral.params,
            "n_samples": 500,
            "n_iterations": 2,
            "skip_initial_steps": 0,
            "sampling_method": "adaptive",
        }
        val_meta = {
            "dataset_name": "val",
            "role": "val",
            "system": "RedCoralModel",
            "dimension": coral.dim,
            "model_params": coral.params,
            "n_samples": 4,
            "n_iterations": 2,
            "skip_initial_steps": 0,
            "sampling_method": "adaptive",
        }
        for label in ("train_500_100_adaptive", "val"):
            (cfg.paths.data_dir / f"{label}.csv").write_text("sentinel\n")
        (cfg.paths.data_dir / "train_500_100_adaptive_metadata.json").write_text(
            json.dumps(train_meta)
        )
        (cfg.paths.data_dir / "val_metadata.json").write_text(json.dumps(val_meta))

        make_data.run(cfg, verbose=False)

        assert (cfg.paths.data_dir / "train_500_100_adaptive.csv").read_text() == "sentinel\n"

    def test_scale_data_after_make_data(self, tmp_path):
        cfg = _tiny_cfg("coral", tmp_path, high_dims=13, n_samples_train=4)
        make_data.run(cfg, verbose=False)
        scale_data.run(cfg, "train", verbose=False)
        assert (tmp_path / "out" / "scalers" / "train" / "scaler.gz").exists()
        assert (tmp_path / "out" / "scalers" / "train" / "scaler_metadata.json").exists()

    def test_scale_data_identity_scaler(self, tmp_path):
        cfg = _tiny_cfg("coral", tmp_path, high_dims=13, n_samples_train=4)
        cfg.data.scaling = "none"
        make_data.run(cfg, verbose=False)
        scale_data.run(cfg, "train", verbose=False)
        scaler = load_scaler(cfg.paths.scaler_path("train"))
        data = np.loadtxt(tmp_path / "data" / "train.csv", delimiter=",", skiprows=1)
        np.testing.assert_allclose(scaler.transform(data[:, :13]), data[:, :13])
        assert scale_data.scaler_is_current(cfg, "train")

    @pytest.mark.slow
    def test_chafee_infante_with_solve_ivp(self, tmp_path):
        cfg = ExperimentConfig(
            system=SystemConfig(name="chafee_infante", params={"N": 8, "tau": 0.05}),
            arch=ArchConfig(num_layers=1, hidden_shape=4, high_dims=8, low_dims=2),
            training=TrainingConfig(
                learning_rate=1e-3, batch_size=8, epochs=2, patience=2, lr_patience=1, loss_weights=[1, 1, 0]
            ),
            data=DataConfig(
                sampling_method="uniform",
                n_samples_train=3,
                n_samples_val=3,
                n_iterations=2,
            ),
            cmgdb=CMGDBConfig(),
            paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
        )
        make_data.run(cfg, verbose=False)
        data = np.loadtxt(tmp_path / "data" / "train.csv", delimiter=",", skiprows=1)
        assert data.shape == (3 * 2, 16)
