"""Smoke tests for the unified pipeline + reproduce_paper.py wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from latentdynamics.cli.pipeline import (
    ALL_STAGES,
    _config_for_seed,
    _normalise_stages,
    _select_cells,
    _stage_complete,
    _train_files_for,
    iter_cells,
    plan_cells,
    run,
)
from latentdynamics.config import (
    ArchConfig,
    CMGDBConfig,
    DataConfig,
    ExperimentConfig,
    PathsConfig,
    SystemConfig,
    TrainingConfig,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


def _make_minimal_cfg(*, n_samples_train, output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        system=SystemConfig(name="coral"),
        arch=ArchConfig(num_layers=1, hidden_shape=4, high_dims=13, low_dims=1),
        training=TrainingConfig(
            learning_rate=1e-3, batch_size=8, epochs=1, patience=1, loss_weights=[1, 1, 1]
        ),
        data=DataConfig(
            sampling_method="uniform",
            n_samples_train=n_samples_train,
            n_samples_test=4,
            n_iterations=2,
        ),
        cmgdb=CMGDBConfig(),
        paths=PathsConfig(data_dir=output_dir / "data", output_dir=output_dir / "out"),
    )


class TestPipelineHelpers:
    def test_config_for_seed_routes_single_train_file(self, tmp_path):
        cfg = _make_minimal_cfg(n_samples_train=500, output_dir=tmp_path)
        seed_cfg = _config_for_seed(cfg, train_file="train", seed=7)
        assert seed_cfg.paths.output_dir == tmp_path / "out" / "seed_7"
        # original cfg unchanged
        assert cfg.paths.output_dir == tmp_path / "out"

    def test_config_for_seed_routes_sweep(self, tmp_path):
        cfg = _make_minimal_cfg(n_samples_train=[100, 500], output_dir=tmp_path)
        seed_cfg = _config_for_seed(cfg, train_file="train_500", seed=3)
        assert seed_cfg.paths.output_dir == tmp_path / "out" / "train_500" / "seed_3"

    def test_train_files_int(self, tmp_path):
        cfg = _make_minimal_cfg(n_samples_train=500, output_dir=tmp_path)
        assert _train_files_for(cfg) == ["train"]

    def test_train_files_list(self, tmp_path):
        cfg = _make_minimal_cfg(n_samples_train=[100, 500], output_dir=tmp_path)
        assert _train_files_for(cfg) == ["train_100", "train_500"]

    def test_normalise_stages_canonical_order(self):
        assert _normalise_stages(["metrics", "data"]) == ["data", "metrics"]

    def test_normalise_stages_unknown_rejected(self):
        with pytest.raises(ValueError):
            _normalise_stages(["bogus"])

    def test_normalise_stages_none_returns_all(self):
        assert _normalise_stages(None) == list(ALL_STAGES)

    def test_iter_cells_enumerates_train_file_seed_grid(self, tmp_path):
        cfg = _make_minimal_cfg(n_samples_train=[100, 500], output_dir=tmp_path)
        cfg.seeds = [0, 1]
        cells = iter_cells(cfg)
        assert [(c.index, c.train_file, c.seed) for c in cells] == [
            (0, "train_100", 0),
            (1, "train_100", 1),
            (2, "train_500", 0),
            (3, "train_500", 1),
        ]
        assert cells[2].output_dir.endswith("out/train_500/seed_0")

    def test_plan_cells_is_json_serialisable(self, tmp_path):
        cfg = _make_minimal_cfg(n_samples_train=500, output_dir=tmp_path)
        plan = plan_cells(cfg)
        assert plan == [
            {
                "index": 0,
                "train_file": "train",
                "seed": 0,
                "output_dir": str(tmp_path / "out" / "seed_0"),
            }
        ]

    def test_select_cells_single_index(self, tmp_path):
        cfg = _make_minimal_cfg(n_samples_train=[100, 500], output_dir=tmp_path)
        cfg.seeds = [0, 1]
        selected = _select_cells(cfg, max_seeds=None, cell_index=3, expected_cells=4)
        assert len(selected) == 1
        assert selected[0].train_file == "train_500"
        assert selected[0].seed == 1

    def test_select_cells_rejects_stale_array_size(self, tmp_path):
        cfg = _make_minimal_cfg(n_samples_train=[100, 500], output_dir=tmp_path)
        with pytest.raises(ValueError):
            _select_cells(cfg, max_seeds=None, cell_index=0, expected_cells=99)

    def test_stage_complete_checks_expected_artefacts(self, tmp_path):
        cfg = _make_minimal_cfg(n_samples_train=500, output_dir=tmp_path)
        seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
        assert not _stage_complete("scale", cfg, seed_cfg, train_file="train")

        scaler_path = cfg.paths.scaler_path("train")
        scaler_path.parent.mkdir(parents=True)
        scaler_path.write_text("x")
        assert not _stage_complete("scale", cfg, seed_cfg, train_file="train")

    def test_run_writes_manifest_even_for_empty_stage_plan(self, tmp_path):
        cfg = _make_minimal_cfg(n_samples_train=500, output_dir=tmp_path)
        results = run(cfg, stages=[], max_seeds=1, device="cpu", verbose=False)
        manifest_path = Path(results[0]["manifest"])
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["cell"]["train_file"] == "train"
        assert manifest["requested_stages"] == []
        assert manifest["config_hash"]


class TestConfigsLoadable:
    @pytest.mark.parametrize(
        "config_name",
        [
            "coral_basic.yaml",
            "coral_data_scaling.yaml",
            "coral_adaptive.yaml",
            "leslie_contraction.yaml",
            "leslie3d_spurious.yaml",
            "leslie3d_success.yaml",
            "chafee_infante.yaml",
            "leslie3d.yaml",
        ],
    )
    def test_each_config_validates(self, config_name: str):
        cfg = load_config(CONFIGS_DIR / config_name)
        assert cfg.system.name in {
            "leslie_contraction",
            "leslie3d",
            "leslie4d",
            "coral",
            "chafee_infante",
        }


class TestReproducePaperScript:
    def test_module_imports_and_lists_experiments(self):
        sys.path.insert(0, str(REPO_ROOT))
        try:
            import importlib

            mod = importlib.import_module("reproduce_paper")
            assert isinstance(mod.EXPERIMENTS, dict)
            assert len(mod.EXPERIMENTS) >= 7
            for config_name in mod.EXPERIMENTS.values():
                assert (CONFIGS_DIR / config_name).exists(), config_name
        finally:
            sys.path.pop(0)
