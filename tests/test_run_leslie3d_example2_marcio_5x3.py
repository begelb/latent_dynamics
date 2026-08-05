"""Focused contracts for the Leslie3D Example 2 Marcio-style 5x3 runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latentdynamics.config import load_config


def _load_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_leslie3d_example2_marcio_5x3.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_leslie3d_example2_marcio_5x3",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def test_packaged_config_is_exact_marcio_architecture_and_patrick_cmgdb() -> None:
    cfg = load_config(RUNNER.CONFIG_STEM)
    RUNNER._validate_protocol(cfg)

    assert (cfg.arch.high_dims, cfg.arch.low_dims) == (3, 2)
    assert cfg.arch.component("encoder").hidden_shapes == (64, 32)
    assert cfg.arch.component("latent_map").hidden_shapes == (32, 32)
    assert cfg.arch.component("decoder").hidden_shapes == (32, 64)
    for name in ("encoder", "latent_map", "decoder"):
        component = cfg.arch.component(name)
        assert component.activation == "tanh"
        assert component.out_activation == "none"

    assert cfg.training.learning_rate == pytest.approx(0.003)
    assert cfg.training.batch_size == 30_000
    assert cfg.training.epochs == 4_000
    assert cfg.training.loss_weights == [1.0, 1.0, 0.0]
    assert cfg.training.gradient_clip_norm is None
    assert cfg.training.scheduler_factor == pytest.approx(0.5)
    assert cfg.training.lr_patience == 100
    assert cfg.training.scheduler_threshold == pytest.approx(1e-4)
    assert cfg.training.scheduler_min_lr == pytest.approx(1e-6)

    assert (cfg.data.n_samples_train, cfg.data.n_samples_val) == (1_000, 200)
    assert (cfg.data.n_iterations, cfg.data.skip) == (30, 0)
    assert cfg.data.scaling == "minmax"
    assert (
        cfg.cmgdb.subdiv_init,
        cfg.cmgdb.subdiv_min,
        cfg.cmgdb.subdiv_max,
        cfg.cmgdb.subdiv_limit,
    ) == (25, 28, 29, 10_000)
    assert cfg.cmgdb.padding is True
    assert cfg.cmgdb.bounds_epsilon_frac == pytest.approx(0.01)
    assert cfg.cmgdb.lower_bounds is None
    assert cfg.cmgdb.upper_bounds is None
    assert cfg.cmgdb.box_map_backend == "adaptive_precomputed"
    assert cfg.cmgdb.compute_roa is False


def test_frozen_5x3_plan_uses_explicit_data_seeds_and_isolated_paths(tmp_path) -> None:
    cells = RUNNER.plan_cells(
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
    )

    assert len(cells) == 15
    assert [cell.initial_condition_seed for cell in cells[::3]] == [
        2_158,
        4_792,
        3_174,
        688,
        5_727,
    ]
    assert [cell.model_seed for cell in cells[:3]] == [0, 1, 2]
    assert Path(cells[0].data_dir).name == "dataset_01"
    assert Path(cells[0].output_dir).parts[-2:] == ("dataset_01", "seed_0")
    assert Path(cells[-1].data_dir).name == "dataset_05"
    assert Path(cells[-1].output_dir).parts[-2:] == ("dataset_05", "seed_2")

    configs = [
        RUNNER._dataset_config(
            dataset,
            data_root=tmp_path / "data",
            output_root=tmp_path / "output",
        )
        for dataset in range(1, 6)
    ]
    assert [cfg.data.train_seed for cfg in configs] == [2_158, 4_792, 3_174, 688, 5_727]
    assert {cfg.data.val_seed for cfg in configs} == {9_999}
    assert all(cfg.paths.scaler_dir.name == "scalers" for cfg in configs)
    plan = RUNNER._build_plan(
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
    )
    assert plan["cmgdb_bounds_inference"] == {
        "data_role": "train_pairs",
        "source": "encoded_train_pairs",
        "included_arrays": ["train.x", "train.y"],
        "validation_pairs_included": False,
        "epsilon_frac": pytest.approx(0.01),
    }


def test_dry_run_reports_selected_canonical_stages_without_writing(tmp_path, capsys) -> None:
    output_root = tmp_path / "output"
    result = RUNNER.main(
        [
            "--stages",
            "metrics,data,train",
            "--datasets",
            "2,5",
            "--model-seeds",
            "0,2",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == ["data", "train", "metrics"]
    assert payload["datasets"] == [2, 5]
    assert payload["dataset_initial_condition_seeds"] == {
        "dataset_02": 4_792,
        "dataset_05": 5_727,
    }
    assert payload["model_seeds"] == [0, 2]
    assert payload["n_cells"] == 4
    assert payload["cells"][0]["output_dir"].endswith("dataset_02/seed_0")
    assert payload["cells"][-1]["output_dir"].endswith("dataset_05/seed_2")
    assert not output_root.exists()


def test_train_stage_calls_marcio_on_full_scaled_pairs_and_writes_completion(
    tmp_path,
    monkeypatch,
) -> None:
    dataset_cfg = RUNNER._dataset_config(
        1,
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
    )
    seed_cfg = RUNNER._seed_config(dataset_cfg, 2)
    scaled_x = np.full((RUNNER.TRAIN_PAIRS, 3), 0.25, dtype=np.float64)
    scaled_y = np.full((RUNNER.TRAIN_PAIRS, 3), 0.75, dtype=np.float64)

    def fake_load_scaled_pairs(cfg, *, role):
        assert cfg is dataset_cfg
        assert role == "train"
        return scaled_x, scaled_y, cfg.paths.data_dir / "train.csv"

    observed = {}

    def fake_train_marcio_full_batch(**kwargs):
        observed.update(kwargs)
        root = Path(kwargs["output_dir"])
        (root / "models").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "models" / "autoencoder.pt").write_bytes(b"checkpoint")
        (root / "models" / "autoencoder.json").write_text("{}")
        (root / "logs" / "history.json").write_text("{}")
        (root / "training_summary.json").write_text("{}")
        return SimpleNamespace(model=torch.nn.Identity())

    def fake_write_final_evaluation(**kwargs):
        root = kwargs["seed_cfg"].paths.output_dir
        evaluation = root / "holdout_evaluation.json"
        losses = root / "final_losses.txt"
        evaluation.write_text('{"holdout":{"n_pairs":6000}}\n')
        losses.write_text("val_loss_total: 1.000000000e-02\n")
        return evaluation, losses

    monkeypatch.setattr(RUNNER, "_load_scaled_pairs", fake_load_scaled_pairs)
    monkeypatch.setattr(
        RUNNER,
        "train_marcio_full_batch",
        fake_train_marcio_full_batch,
    )
    monkeypatch.setattr(
        RUNNER,
        "_write_final_evaluation",
        fake_write_final_evaluation,
    )

    RUNNER._run_train(
        dataset_cfg=dataset_cfg,
        seed_cfg=seed_cfg,
        dataset=1,
        model_seed=2,
        device=torch.device("cpu"),
        verbose=False,
        force_overwrite=False,
    )

    assert observed["x"] is scaled_x
    assert observed["y"] is scaled_y
    assert observed["epochs"] == 4_000
    assert observed["learning_rate"] == pytest.approx(0.003)
    assert observed["seed"] == 2
    assert observed["scheduler_factor"] == pytest.approx(0.5)
    assert observed["scheduler_patience"] == 100
    assert observed["scheduler_threshold"] == pytest.approx(1e-4)
    assert observed["scheduler_min_lr"] == pytest.approx(1e-6)
    assert observed["output_dir"] == seed_cfg.paths.output_dir
    assert RUNNER._training_complete(seed_cfg)
    contract = json.loads((seed_cfg.paths.output_dir / "training_contract.json").read_text())
    assert contract["training_entrypoint"].endswith("train_marcio_full_batch")
    assert contract["training"]["gradient_clip_norm"] is None
    assert contract["data"]["holdout_used_for_training_or_selection"] is False


def test_post_update_holdout_objective_matches_decoded_two_term_mse() -> None:
    class IdentityAutoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Identity()
            self.latent_map = torch.nn.Identity()
            self.decoder = torch.nn.Identity()

    x = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    y = np.asarray([[1.0, 1.0], [0.0, 3.0]], dtype=np.float64)
    metrics = RUNNER._evaluate_two_term_objective(
        IdentityAutoencoder(),
        x,
        y,
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert metrics["loss_reconstruction"] == pytest.approx(0.0)
    assert metrics["loss_prediction"] == pytest.approx(1.25)
    assert metrics["loss_total"] == pytest.approx(1.25)


def test_morse_stage_requests_train_pair_bounds_only(tmp_path, monkeypatch) -> None:
    dataset_cfg = RUNNER._dataset_config(
        1,
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
    )
    seed_cfg = RUNNER._seed_config(dataset_cfg, 0)
    observed = {}

    def fake_morse_run(cfg, **kwargs):
        observed["cfg"] = cfg
        observed.update(kwargs)

    monkeypatch.setattr(RUNNER.morse_stage, "run", fake_morse_run)

    RUNNER._execute_stage(
        "morse",
        dataset=1,
        model_seed=0,
        dataset_cfg=dataset_cfg,
        seed_cfg=seed_cfg,
        device=torch.device("cpu"),
        verbose=False,
        force_overwrite=False,
    )

    assert observed["cfg"] is seed_cfg
    assert observed["train_file"] == "train"
    assert observed["bounds_data_role"] == "train_pairs"


def test_morse_completion_requires_train_pair_bounds_source(tmp_path) -> None:
    dataset_cfg = RUNNER._dataset_config(
        1,
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
    )
    seed_cfg = RUNNER._seed_config(dataset_cfg, 0)
    root = seed_cfg.paths.output_dir
    morse_dir = root / "MG"
    morse_dir.mkdir(parents=True)
    (morse_dir / "morse_graph").write_text("digraph {}\n")
    (morse_dir / "morse_sets").write_text("0,0,1,1,0\n")
    common_log = "\n".join(
        [
            "subdiv_init: 25",
            "subdiv_min: 28",
            "subdiv_max: 29",
            "subdiv_limit: 10000",
            "bounds_epsilon_frac: 0.01",
            "padding: True",
            "box_map_backend: adaptive_precomputed",
            "compute_roa: False",
        ]
    )
    log_path = root / "mg_params_log.txt"
    log_path.write_text(common_log + "\nbounds_source: encoded_data\n")
    assert not RUNNER._morse_complete(seed_cfg)

    log_path.write_text(common_log + "\nbounds_source: encoded_train_pairs\n")
    assert RUNNER._morse_complete(seed_cfg)
