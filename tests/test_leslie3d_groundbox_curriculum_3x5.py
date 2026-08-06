"""Protocol tests for the Leslie3D ground-box curriculum 3x5 experiment."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from latentdynamics.config import (
    CurriculumLBFGSPolishConfig,
    CurriculumOptimizerConfig,
    load_config,
)
from latentdynamics.systems import build_system

CONFIG_STEM = "leslie3d_groundbox_curriculum_wide"
EXAMPLE_ALIAS = "groundbox_curriculum"


def _load_seed_sweep():
    script = Path(__file__).resolve().parents[1] / "scripts" / "retrain_seed_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "retrain_seed_sweep_groundbox_curriculum_test",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SWEEP = _load_seed_sweep()


def test_groundbox_curriculum_config_freezes_requested_training_protocol() -> None:
    cfg = load_config(CONFIG_STEM)
    system = build_system(cfg.system.name, cfg.system.params)

    assert cfg.system.name == "leslie3d"
    assert cfg.system.params == {
        "th1": 28.9,
        "th2": 29.8,
        "th3": 22.0,
        "survival_p1": 0.7,
        "survival_p2": 0.7,
        "lower_bounds": [0.0, 0.0, 0.0],
        "upper_bounds": [110.0, 77.0, 54.0],
    }
    assert system.lower_bounds.tolist() == [0.0, 0.0, 0.0]
    assert system.upper_bounds.tolist() == [110.0, 77.0, 54.0]

    assert (cfg.arch.high_dims, cfg.arch.low_dims) == (3, 2)
    assert cfg.arch.component("encoder").hidden_shapes == (128, 64)
    assert cfg.arch.component("latent_map").hidden_shapes == (64, 64)
    assert cfg.arch.component("decoder").hidden_shapes == (64, 128)
    for name in ("encoder", "latent_map", "decoder"):
        component = cfg.arch.component(name)
        assert component.activation == "tanh"
        assert component.out_activation == "none"

    assert (cfg.data.n_samples_train, cfg.data.n_samples_val) == (1_000, 200)
    assert (cfg.data.n_iterations, cfg.data.skip) == (20, 0)
    assert cfg.data.sampling_method == "uniform"
    assert cfg.data.scaling == "minmax"
    assert (cfg.data.train_seed, cfg.data.val_seed) == (2_158, 9_999)
    assert cfg.training.batch_size == 20_000
    assert cfg.training.epochs == 12_000
    assert cfg.training.gradient_clip_norm is None
    assert cfg.seeds == [0, 1, 2]

    optimizer = cfg.training.curriculum_optimizer
    assert isinstance(optimizer, CurriculumOptimizerConfig)
    assert optimizer.model_dump(mode="json") == {
        "name": "adamw",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
    }

    stages = cfg.training.curriculum
    assert stages is not None
    assert [stage.model_dump(mode="json") for stage in stages] == [
        {
            "name": "autoencoder",
            "epochs": 4_000,
            "learning_rate": 0.003,
            "loss_weights": [1.0, 0.0, 0.0],
            "trainable_components": ["encoder", "decoder"],
        },
        {
            "name": "decoded_prediction",
            "epochs": 4_000,
            "learning_rate": 0.003,
            "loss_weights": [1.0, 1.0, 0.0],
            "trainable_components": ["encoder", "latent_map", "decoder"],
        },
        {
            "name": "semiconjugacy",
            "epochs": 4_000,
            "learning_rate": 0.003,
            "loss_weights": [1.0, 1.0, 1.0],
            "trainable_components": ["encoder", "latent_map", "decoder"],
        },
    ]
    assert sum(stage.epochs for stage in stages) == cfg.training.epochs

    polish = cfg.training.curriculum_polish
    assert isinstance(polish, CurriculumLBFGSPolishConfig)
    assert polish.model_dump(mode="json") == {
        "name": "lbfgs",
        "device": "cpu",
        "dtype": "float64",
        "outer_steps": 12,
        "learning_rate": 0.25,
        "max_iter": 10,
        "max_eval": 25,
        "history_size": 50,
        "tolerance_grad": 1e-9,
        "tolerance_change": 1e-12,
        "line_search_fn": "strong_wolfe",
        "loss_weights": [1.0, 1.0, 1.0],
        "trainable_components": ["encoder", "latent_map", "decoder"],
    }

    assert (
        cfg.cmgdb.subdiv_init,
        cfg.cmgdb.subdiv_min,
        cfg.cmgdb.subdiv_max,
        cfg.cmgdb.subdiv_limit,
    ) == (25, 28, 29, 10_000)
    assert cfg.cmgdb.box_map_backend == "adaptive_precomputed"
    assert cfg.cmgdb.bounds_data_role == "train_pairs"
    assert cfg.cmgdb.adaptive_precompute_subdiv == "init"
    assert cfg.cmgdb.bounds_epsilon_frac == pytest.approx(0.01)
    assert cfg.cmgdb.padding is True
    assert cfg.cmgdb.compute_roa is False
    assert cfg.cmgdb.lower_bounds is None
    assert cfg.cmgdb.upper_bounds is None


def test_groundbox_pair_counts_and_full_batch_are_consistent() -> None:
    cfg = load_config(CONFIG_STEM)
    sizes = SWEEP._data_size_summary(
        cfg,
        requested_total_initial_conditions=None,
        dataset_count=5,
    )

    assert sizes["effective_initial_conditions_per_dataset"] == {
        "train": 1_000,
        "validation": 200,
        "total": 1_200,
    }
    assert sizes["trajectory"] == {
        "generated_steps": 20,
        "discarded_steps": 0,
        "retained_steps": 20,
    }
    assert sizes["transition_pairs_per_dataset"] == {
        "train": 20_000,
        "validation": 4_000,
        "total": 24_000,
    }
    assert sizes["transition_pairs_across_dataset_trees"] == {
        "train": 100_000,
        "validation": 20_000,
        "total": 120_000,
    }
    assert cfg.training.batch_size == sizes["transition_pairs_per_dataset"]["train"]


def test_seed_sweep_alias_dry_plan_has_frozen_five_by_three_grid(capsys) -> None:
    result = SWEEP.main(
        [
            "--example",
            EXAMPLE_ALIAS,
            "--ic-seeds",
            "2158,4792,3174,688,5727",
            "--model-seeds",
            "0,1,2",
            "--tag",
            "3x5_v1",
            "--trajectory-length",
            "20",
            "--cmgdb-subdiv",
            "25,28,29",
            "--box-map-backend",
            "adaptive_precomputed",
            "--bounds-data-role",
            "train_pairs",
            "--adaptive-precompute-subdiv",
            "init",
            "--full-batch",
            "--dry-run",
        ]
    )

    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["examples"] == [CONFIG_STEM]
    assert plan["ic_seeds"] == [2_158, 4_792, 3_174, 688, 5_727]
    assert plan["model_seeds"] == [0, 1, 2]
    assert plan["n_cells"] == 15
    assert plan["shared_val_seeds"] == {CONFIG_STEM: 9_999}
    assert plan["data_sizes"][CONFIG_STEM]["transition_pairs_per_dataset"] == {
        "train": 20_000,
        "validation": 4_000,
        "total": 24_000,
    }

    output_dirs = [cell["output_dir"] for cell in plan["cells"]]
    assert len(output_dirs) == len(set(output_dirs)) == 15
    assert output_dirs[0].endswith(
        "output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_2158/seed_0"
    )
    assert output_dirs[-1].endswith(
        "output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_5727/seed_2"
    )
    expected_suffixes = {
        f"dataset_{data_seed}/seed_{model_seed}"
        for data_seed in (2_158, 4_792, 3_174, 688, 5_727)
        for model_seed in range(3)
    }
    assert {
        "/".join(Path(output_dir).parts[-2:]) for output_dir in output_dirs
    } == expected_suffixes
