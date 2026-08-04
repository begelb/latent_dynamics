"""Focused tests for the generic paper-example seed sweep runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "retrain_seed_sweep.py"
    spec = importlib.util.spec_from_file_location("retrain_seed_sweep", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SWEEP = _load_module()


def test_leslie_example2_t40_uses_explicit_precomputation_and_isolated_paths() -> None:
    cfg = SWEEP._dataset_config(
        "leslie3d_example2",
        ic_seed=3,
        model_seeds=[0, 1, 2],
        tag="t40",
        trajectory_length=40,
        box_map_backend="adaptive_precomputed",
    )

    assert cfg.data.n_samples_train == 8000
    assert cfg.data.n_samples_val == 2000
    assert cfg.data.n_iterations == 40
    assert cfg.data.skip == 0
    assert cfg.data.train_seed == 3
    assert cfg.data.val_seed == 9999
    assert cfg.seeds == [0, 1, 2]
    assert cfg.cmgdb.box_map_backend == "adaptive_precomputed"
    assert (cfg.cmgdb.subdiv_init, cfg.cmgdb.subdiv_min, cfg.cmgdb.subdiv_max) == (
        25,
        28,
        29,
    )
    assert str(cfg.paths.data_dir).endswith("data/leslie3d_example2_seedsweep_t40/dataset_3")
    assert str(cfg.paths.output_dir).endswith("output/leslie3d_example2_seedsweep_t40/dataset_3")


def test_dataset_axis_keeps_one_shared_validation_holdout() -> None:
    configs = [
        SWEEP._dataset_config(
            "leslie3d_example2",
            ic_seed=seed,
            model_seeds=[0, 1, 2],
            tag="t40",
            trajectory_length=40,
            box_map_backend="adaptive_precomputed",
        )
        for seed in SWEEP.DEFAULT_IC_SEEDS
    ]

    assert [cfg.data.train_seed for cfg in configs] == [1, 2, 3, 4, 5]
    assert {cfg.data.val_seed for cfg in configs} == {9999}


def test_t40_dry_run_records_full_5x3_plan(capsys) -> None:
    result = SWEEP.main(
        [
            "--example",
            "leslie3d_example2",
            "--trajectory-length",
            "40",
            "--box-map-backend",
            "adaptive_precomputed",
            "--tag",
            "t40",
            "--dry-run",
        ]
    )

    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["ic_seeds"] == [1, 2, 3, 4, 5]
    assert plan["model_seeds"] == [0, 1, 2]
    assert plan["n_cells"] == 15
    assert plan["trajectory_length"] == 40
    assert plan["box_map_backend"] == "adaptive_precomputed"
    assert plan["cmgdb_subdiv"] is None
    assert plan["shared_val_seeds"] == {"leslie3d_example2": 9999}
    assert plan["cells"][0]["output_dir"].endswith(
        "output/leslie3d_example2_seedsweep_t40/dataset_1/seed_0"
    )
    assert plan["cells"][-1]["output_dir"].endswith(
        "output/leslie3d_example2_seedsweep_t40/dataset_5/seed_2"
    )


def test_t25_n50000_preserves_packaged_split_and_counts_pairs() -> None:
    cfg = SWEEP._dataset_config(
        "leslie3d_example2",
        ic_seed=5,
        model_seeds=[0, 1, 2],
        tag="t25_n50000",
        trajectory_length=25,
        total_initial_conditions=50_000,
        box_map_backend="adaptive_precomputed",
    )

    assert cfg.data.n_samples_train == 40_000
    assert cfg.data.n_samples_val == 10_000
    assert cfg.data.n_iterations == 25
    assert cfg.data.skip == 0
    assert cfg.data.train_seed == 5
    assert cfg.data.val_seed == 9999
    assert cfg.cmgdb.box_map_backend == "adaptive_precomputed"
    assert str(cfg.paths.data_dir).endswith(
        "data/leslie3d_example2_seedsweep_t25_n50000/dataset_5"
    )

    sizes = SWEEP._data_size_summary(
        cfg,
        requested_total_initial_conditions=50_000,
        dataset_count=5,
    )
    assert sizes["effective_initial_conditions_per_dataset"] == {
        "train": 40_000,
        "validation": 10_000,
        "total": 50_000,
    }
    assert sizes["transition_pairs_per_dataset"] == {
        "train": 1_000_000,
        "validation": 250_000,
        "total": 1_250_000,
    }
    assert sizes["transition_pairs_across_dataset_trees"]["total"] == 6_250_000


def test_t25_n50000_dry_run_records_full_5x3_data_plan(capsys) -> None:
    result = SWEEP.main(
        [
            "--example",
            "leslie3d_example2",
            "--trajectory-length",
            "25",
            "--total-initial-conditions",
            "50000",
            "--box-map-backend",
            "adaptive_precomputed",
            "--tag",
            "t25_n50000",
            "--dry-run",
        ]
    )

    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["n_cells"] == 15
    assert plan["trajectory_length"] == 25
    assert plan["total_initial_conditions"] == 50_000
    assert plan["shared_val_seeds"] == {"leslie3d_example2": 9999}
    assert plan["box_map_backend"] == "adaptive_precomputed"
    sizes = plan["data_sizes"]["leslie3d_example2"]
    assert sizes["requested_total_initial_conditions_per_dataset"] == 50_000
    assert sizes["effective_initial_conditions_per_dataset"]["train"] == 40_000
    assert sizes["effective_initial_conditions_per_dataset"]["validation"] == 10_000
    assert sizes["transition_pairs_per_dataset"]["train"] == 1_000_000
    assert sizes["transition_pairs_per_dataset"]["validation"] == 250_000
    assert plan["cells"][0]["output_dir"].endswith(
        "output/leslie3d_example2_seedsweep_t25_n50000/dataset_1/seed_0"
    )
    assert plan["cells"][-1]["output_dir"].endswith(
        "output/leslie3d_example2_seedsweep_t25_n50000/dataset_5/seed_2"
    )


def test_figures_selection_is_recorded_in_dry_run(capsys) -> None:
    result = SWEEP.main(
        [
            "--example",
            "leslie3d_example2",
            "--figures",
            "morse,overlay,extras",
            "--dry-run",
        ]
    )

    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["figures"] == ["extras", "morse", "overlay"]


def test_figures_default_preserves_render_all_contract(capsys) -> None:
    result = SWEEP.main(
        [
            "--example",
            "leslie3d_example2",
            "--max-datasets",
            "1",
            "--max-seeds",
            "1",
            "--dry-run",
        ]
    )

    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["figures"] is None


def test_figures_selection_rejects_unknown_group() -> None:
    try:
        SWEEP._parse_figure_set("morse,basins")
    except ValueError as exc:
        assert "unknown --figures group" in str(exc)
    else:
        raise AssertionError("expected an unknown render group to fail")


def test_total_initial_conditions_rejects_degenerate_total() -> None:
    try:
        SWEEP._dataset_config(
            "leslie3d_example2",
            ic_seed=1,
            model_seeds=[0],
            total_initial_conditions=1,
        )
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a one-IC total")
