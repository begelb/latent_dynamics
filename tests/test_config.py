"""Tests for the pydantic config schema and YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from latentdynamics.config import (
    ArchConfig,
    CMGDBConfig,
    ExperimentConfig,
    PathsConfig,
    TrainingConfig,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


class TestSchema:
    def test_arch_lowercases_activation(self):
        cfg = ArchConfig(num_layers=2, hidden_shape=32, high_dims=3, low_dims=2, activation="ReLU")  # type: ignore[arg-type]
        assert cfg.activation == "relu"

    def test_loss_weights_must_be_three(self):
        with pytest.raises(ValueError):
            TrainingConfig(
                learning_rate=1e-3,
                batch_size=32,
                epochs=10,
                patience=5,
                loss_weights=[1.0, 1.0],
            )

    def test_cmgdb_subdivision_ordering(self):
        with pytest.raises(ValueError):
            CMGDBConfig(subdiv_init=10, subdiv_min=8, subdiv_max=12)

    def test_cmgdb_fixed_bounds_must_be_paired(self):
        with pytest.raises(ValueError):
            CMGDBConfig(lower_bounds=[-1.0, -1.0])

    def test_training_clip_norm_can_be_disabled(self):
        cfg = TrainingConfig(
            learning_rate=1e-3,
            batch_size=32,
            epochs=10,
            patience=5,
            loss_weights=[1.0, 1.0, 0.0],
            gradient_clip_norm=None,
        )
        assert cfg.gradient_clip_norm is None

    def test_paths_derived_subdirectories(self, tmp_path):
        p = PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out")
        assert p.model_dir == tmp_path / "out" / "models"
        assert p.scaler_dir == tmp_path / "out" / "scalers"
        assert p.figures_dir == tmp_path / "out" / "figures"
        assert p.morse_dir == tmp_path / "out" / "MG"

    def test_high_dims_must_exceed_low_dims(self):
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate(
                {
                    "system": {"name": "coral"},
                    "arch": {
                        "num_layers": 2,
                        "hidden_shape": 16,
                        "high_dims": 1,
                        "low_dims": 13,
                    },
                    "training": {
                        "learning_rate": 1e-3,
                        "batch_size": 32,
                        "epochs": 1,
                        "patience": 1,
                        "loss_weights": [1, 1, 1],
                    },
                    "data": {
                        "sampling_method": "uniform",
                        "n_samples_train": 10,
                        "n_samples_test": 10,
                        "n_iterations": 5,
                    },
                    "paths": {"data_dir": "data", "output_dir": "out"},
                }
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ArchConfig.model_validate(
                {"num_layers": 2, "hidden_shape": 16, "high_dims": 3, "low_dims": 1, "extra": True}
            )

    def test_component_arch_resolves_hidden_shapes(self):
        cfg = ArchConfig(
            num_layers=2,
            hidden_shape=16,
            high_dims=64,
            low_dims=2,
            activation="ReLU",  # type: ignore[arg-type]
            encoder={"hidden_shapes": [64, 32], "activation": "Tanh", "out_activation": "none"},
            latent_map={"hidden_shapes": [32, 32], "out_activation": "none"},
        )
        enc = cfg.component("encoder")
        dyn = cfg.component("latent_map")
        dec = cfg.component("decoder")
        assert enc.hidden_shapes == (64, 32)
        assert enc.activation == "tanh"
        assert enc.out_activation == "none"
        assert dyn.hidden_shapes == (32, 32)
        assert dyn.out_activation == "none"
        assert dec.hidden_shapes == (16, 16)

    def test_component_hidden_shapes_must_match_num_layers_when_both_set(self):
        with pytest.raises(ValueError):
            ArchConfig(
                num_layers=2,
                hidden_shape=16,
                high_dims=4,
                low_dims=2,
                encoder={"num_layers": 3, "hidden_shapes": [8, 8]},
            )

    def test_arch_accepts_per_component_only(self):
        """If every component supplies hidden_shapes, shared num_layers/hidden_shape
        must not be required."""
        arch = ArchConfig.model_validate(
            {
                "high_dims": 4,
                "low_dims": 2,
                "encoder": {"hidden_shapes": [16, 8]},
                "latent_map": {"hidden_shapes": [8, 8]},
                "decoder": {"hidden_shapes": [8, 16]},
            }
        )
        assert arch.component("encoder").hidden_shapes == (16, 8)
        assert arch.component("latent_map").hidden_shapes == (8, 8)
        assert arch.component("decoder").hidden_shapes == (8, 16)

    def test_arch_rejects_unresolvable_component(self):
        """If a component is missing hidden_shapes and there is no shared
        num_layers/hidden_shape, validation must fail with a clear message
        naming the unresolvable component."""
        with pytest.raises(ValueError, match=r"encoder.*unresolvable"):
            ArchConfig.model_validate(
                {
                    "high_dims": 4,
                    "low_dims": 2,
                    "latent_map": {"hidden_shapes": [8, 8]},
                    "decoder": {"hidden_shapes": [8, 16]},
                }
            )


class TestLoader:
    def test_coral_basic_yaml_loads(self):
        cfg = load_config(CONFIGS_DIR / "coral_basic.yaml")
        assert cfg.system.name == "coral"
        assert cfg.arch.high_dims == 13
        assert cfg.arch.low_dims == 1
        assert cfg.training.loss_weights == [10.0, 10.0, 1.0]
        assert cfg.data.sampling_method == "uniform"
        assert cfg.data.n_iterations == 20
        assert cfg.training.epochs == 1000  # from defaults
        assert cfg.data.scaling == "minmax"

    def test_chafee_yaml_uses_archived_asymmetric_network(self):
        cfg = load_config(CONFIGS_DIR / "chafee_infante.yaml")
        assert cfg.data.scaling == "none"
        assert cfg.arch.component("encoder").hidden_shapes == (64, 32)
        assert cfg.arch.component("latent_map").hidden_shapes == (32, 32)
        assert cfg.arch.component("decoder").hidden_shapes == (32, 64)
        assert cfg.arch.component("latent_map").out_activation == "none"
        assert cfg.cmgdb.lower_bounds == [-3.0, -2.0]
        assert cfg.cmgdb.padding is False

    def test_leslie3d_yaml_loads(self):
        cfg = load_config(CONFIGS_DIR / "leslie3d.yaml")
        assert cfg.system.name == "leslie3d"
        assert cfg.system.params["th1"] == pytest.approx(28.9)
        assert cfg.arch.high_dims == 3
        assert cfg.arch.low_dims == 2
        assert cfg.data.n_samples_train == 2000
        assert cfg.cmgdb.subdiv_max == 10  # from defaults

    def test_defaults_are_overridable(self, tmp_path):
        shared = tmp_path / "_shared"
        shared.mkdir()
        (shared / "defaults.yaml").write_text(
            "training:\n  epochs: 5000\n  patience: 200\n  loss_mode: weighted\n"
        )
        cfg_path = tmp_path / "x.yaml"
        cfg_path.write_text(
            "system:\n"
            "  name: coral\n"
            "arch:\n"
            "  num_layers: 1\n"
            "  hidden_shape: 8\n"
            "  high_dims: 13\n"
            "  low_dims: 1\n"
            "training:\n"
            "  learning_rate: 1e-3\n"
            "  batch_size: 32\n"
            "  epochs: 10\n"
            "  patience: 3\n"
            "  loss_weights: [1, 1, 1]\n"
            "data:\n"
            "  sampling_method: uniform\n"
            "  n_samples_train: 4\n"
            "  n_samples_test: 4\n"
            "  n_iterations: 2\n"
            "paths:\n"
            "  data_dir: data/x\n"
            "  output_dir: out/x\n"
        )
        cfg = load_config(cfg_path)
        assert cfg.training.epochs == 10  # override beats default
        assert cfg.training.patience == 3
