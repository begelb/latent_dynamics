"""Frozen protocol tests for the Ives Lake Mývatn 3x5 replication."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from latentdynamics.analysis.morse import infer_latent_bounds
from latentdynamics.cli import morse_graph
from latentdynamics.cli.pipeline import _config_for_seed, _stage_complete
from latentdynamics.config import ExperimentConfig, load_config
from latentdynamics.models import build_autoencoder
from latentdynamics.sampling import fit_fixed_bounds_scaler
from latentdynamics.systems import build_system
from latentdynamics.training import save_checkpoint

CONFIG_STEM = "ives_myvatn"
DATA_SEEDS = (2_158, 4_792, 3_174, 688, 5_727)
MODEL_SEEDS = (0, 1, 2)
CODE_ROOT = Path(__file__).resolve().parents[1]


def _load_seed_sweep():
    script = CODE_ROOT / "scripts" / "retrain_seed_sweep.py"
    spec = importlib.util.spec_from_file_location("retrain_seed_sweep_ives_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SWEEP = _load_seed_sweep()


def test_packaged_config_freezes_the_archived_ives_recipe() -> None:
    cfg = load_config(CONFIG_STEM)
    system = build_system(cfg.system.name, cfg.system.params)

    assert cfg.experiment_name == CONFIG_STEM
    assert cfg.system.name == "ives"
    assert cfg.system.params == {
        "coordinate_mode": "log",
        "r1": 3.873,
        "r2": 11.746,
        "c": 0.000000367282300498085,
        "d": 0.5517,
        "p": 0.06659,
        "q": 0.9026,
        "lower_bounds": [-3.0, -7.5, -3.0],
        "upper_bounds": [1.5, 1.5, 1.5],
    }
    np.testing.assert_array_equal(system.lower_bounds, [-3.0, -7.5, -3.0])
    np.testing.assert_array_equal(system.upper_bounds, [1.5, 1.5, 1.5])
    np.testing.assert_allclose(
        system.step(np.zeros(3)),
        [0.3287694948708851, 0.5881807125380104, 0.15185979147616613],
        rtol=0.0,
        atol=2e-15,
    )

    assert (cfg.arch.high_dims, cfg.arch.low_dims) == (3, 2)
    assert cfg.arch.component("encoder").hidden_shapes == (32,)
    assert cfg.arch.component("latent_map").hidden_shapes == (64, 64, 64, 64, 64)
    assert cfg.arch.component("decoder").hidden_shapes == (32,)
    assert cfg.arch.component("encoder").out_activation == "tanh"
    assert cfg.arch.component("latent_map").out_activation == "tanh"
    assert cfg.arch.component("decoder").out_activation == "sigmoid"

    assert cfg.training.learning_rate == 0.001
    assert cfg.training.batch_size == 1_024
    assert (cfg.training.epochs, cfg.training.patience, cfg.training.lr_patience) == (
        500,
        300,
        20,
    )
    assert cfg.training.loss_weights == [1.0, 1.0, 1.0]
    assert cfg.training.gradient_clip_norm is None
    assert cfg.training.scheduler_factor == 0.5
    assert cfg.training.scheduler_threshold == 0.0001
    assert cfg.training.scheduler_min_lr == 0.000001
    assert cfg.training.curriculum is None
    assert cfg.training.warm_start_checkpoint_dir is None

    assert cfg.data.sampling_method == "uniform"
    assert cfg.data.scaling == "fixed_bounds"
    assert cfg.data.scaling_epsilon == 0.000001
    assert (cfg.data.n_samples_train, cfg.data.n_samples_val) == (1_000, 200)
    assert (cfg.data.n_iterations, cfg.data.skip) == (70, 50)
    assert (cfg.data.train_seed, cfg.data.val_seed) == (2_158, 9_999)
    assert cfg.data.n_samples_train * (cfg.data.n_iterations - cfg.data.skip) == 20_000
    assert cfg.data.n_samples_val * (cfg.data.n_iterations - cfg.data.skip) == 4_000

    assert (
        cfg.cmgdb.subdiv_init,
        cfg.cmgdb.subdiv_min,
        cfg.cmgdb.subdiv_max,
        cfg.cmgdb.subdiv_limit,
    ) == (18, 22, 30, 100_000)
    assert cfg.cmgdb.box_map_backend == "adaptive_precomputed"
    assert cfg.cmgdb.bounds_data_role == "system_grid"
    assert cfg.cmgdb.bounds_grid_resolution == 64
    assert cfg.cmgdb.bounds_include_latent_image is True
    assert cfg.cmgdb.bounds_epsilon_frac == 0.1
    assert cfg.cmgdb.bounds_clip_lower == [-1.0, -1.0]
    assert cfg.cmgdb.bounds_clip_upper == [1.0, 1.0]
    assert cfg.cmgdb.adaptive_precompute_subdiv == "init"
    assert cfg.cmgdb.padding is True
    assert cfg.cmgdb.compute_roa is False
    assert cfg.seeds == [0, 1, 2]


def test_seed_sweep_dry_plan_is_exact_five_by_three_morse_only_grid(capsys) -> None:
    result = SWEEP.main(
        [
            "--example",
            "myvatn",
            "--ic-seeds",
            "2158,4792,3174,688,5727",
            "--model-seeds",
            "0,1,2",
            "--stages",
            "data,scale,train,diagnose,morse,render,metrics",
            "--figures",
            "morse",
            "--tag",
            "3x5_v1",
            "--trajectory-length",
            "70",
            "--cmgdb-subdiv",
            "18,22,30",
            "--box-map-backend",
            "adaptive_precomputed",
            "--bounds-data-role",
            "system_grid",
            "--adaptive-precompute-subdiv",
            "init",
            "--dry-run",
        ]
    )

    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["examples"] == [CONFIG_STEM]
    assert plan["ic_seeds"] == list(DATA_SEEDS)
    assert plan["model_seeds"] == list(MODEL_SEEDS)
    assert plan["n_cells"] == 15
    assert plan["figures"] == ["morse"]
    assert plan["shared_val_seeds"] == {CONFIG_STEM: 9_999}
    assert plan["cmgdb_subdiv"] == [18, 22, 30]
    assert plan["box_map_backend"] == "adaptive_precomputed"
    assert plan["bounds_data_role"] == "system_grid"
    assert plan["adaptive_precompute_subdiv"] == "init"
    assert plan["trajectory_length"] == 70
    assert plan["total_initial_conditions"] is None
    assert plan["full_batch"] is False
    assert plan["data_sizes"][CONFIG_STEM]["effective_initial_conditions_per_dataset"] == {
        "train": 1_000,
        "validation": 200,
        "total": 1_200,
    }
    assert plan["data_sizes"][CONFIG_STEM]["transition_pairs_per_dataset"] == {
        "train": 20_000,
        "validation": 4_000,
        "total": 24_000,
    }

    observed = [
        (cell["ic_seed"], cell["model_seed"], Path(cell["output_dir"]))
        for cell in plan["cells"]
    ]
    expected = [
        (
            data_seed,
            model_seed,
            CODE_ROOT
            / "output"
            / "ives_myvatn_seedsweep_3x5_v1"
            / f"dataset_{data_seed}"
            / f"seed_{model_seed}",
        )
        for data_seed in DATA_SEEDS
        for model_seed in MODEL_SEEDS
    ]
    assert observed == expected
    assert len({path for _, _, path in observed}) == 15


def test_launcher_contract_records_exclusions_and_safe_persistent_outputs() -> None:
    launcher = (CODE_ROOT / "scripts" / "run_ives_myvatn_3x5.sh").read_text()

    for fragment in (
        'readonly SWEEP_STEM="ives_myvatn_seedsweep_${SWEEP_TAG}"',
        "readonly -a DATA_SEEDS=(2158 4792 3174 688 5727)",
        "readonly -a MODEL_SEEDS=(0 1 2)",
        "--trajectory-length 70",
        "--cmgdb-subdiv 18,22,30",
        "--box-map-backend adaptive_precomputed",
        "--bounds-data-role system_grid",
        "--adaptive-precompute-subdiv init",
        "--figures morse",
        'readonly STATUS_FILE="${SWEEP_ROOT}/run_status.txt"',
        'readonly RUN_LOG="${SWEEP_ROOT}/run.log"',
        'readonly PID_FILE="${SWEEP_ROOT}/controller.pid"',
        'readonly SESSION_FILE="${SWEEP_ROOT}/session.txt"',
        "scripts/recover_ives_myvatn_3x5.py",
        "scripts/summarize_ives_myvatn_3x5.py",
        '--sweep-root "${SWEEP_ROOT}" --verify',
    ):
        assert fragment in launcher

    for excluded in (
        "regions_of_attraction",
        "basin_plots",
        "training_data_plots",
        "latent_evolution_snapshots_or_animations",
        "density_overlays",
        "invariant_overlays",
        "separation_training_extras",
        "unrelated_paper_figures",
    ):
        assert f'"{excluded}"' in launcher

    assert "--full-batch" not in launcher
    assert "--total-initial-conditions" not in launcher
    assert "--allow-incomplete" not in launcher
    assert "--force-overwrite" not in launcher
    assert launcher.index('CURRENT_PHASE="recovery"') < launcher.index(
        'CURRENT_PHASE="data_scale"'
    )
    assert launcher.index('CURRENT_PHASE="data_scale"') < launcher.index(
        'CURRENT_PHASE="train_diagnose"'
    )
    assert launcher.index('CURRENT_PHASE="train_diagnose"') < launcher.index(
        'CURRENT_PHASE="morse dataset=${data_seed} model=${model_seed}"'
    )
    assert 'system = build_system(cfg.system.name, cfg.system.params)' in launcher


def test_system_grid_bounds_source_builds_scaled_full_box_without_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config(CONFIG_STEM).model_copy(deep=True)
    cfg.paths.data_dir = tmp_path / "unused-data"
    cfg.paths.output_dir = tmp_path / "output"
    cfg.cmgdb.bounds_grid_resolution = 2
    lower = np.asarray(cfg.system.params["lower_bounds"], dtype=np.float64)
    upper = np.asarray(cfg.system.params["upper_bounds"], dtype=np.float64)
    scaler = fit_fixed_bounds_scaler(lower, upper, epsilon=cfg.data.scaling_epsilon)
    loaded_paths: list[Path] = []

    def fake_load_scaler(path: Path):
        loaded_paths.append(Path(path))
        return scaler

    monkeypatch.setattr(morse_graph, "load_scaler", fake_load_scaler)

    observed = morse_graph._load_data_and_scale(
        cfg,
        "train",
        bounds_data_role="system_grid",
    )

    ambient_corners = np.asarray(
        list(itertools.product(*zip(lower, upper, strict=True))),
        dtype=np.float64,
    )
    expected = scaler.transform(ambient_corners)
    assert loaded_paths == [cfg.paths.scaler_path("train")]
    assert observed.shape == (2**3, 3)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    assert not cfg.paths.data_dir.exists()


def test_latent_bound_inference_includes_image_then_expands_and_clips() -> None:
    encoder = torch.nn.Linear(2, 2, bias=False)
    latent_map = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        encoder.weight.copy_(torch.eye(2))
        latent_map.weight.copy_(4.0 * torch.eye(2))
    data = np.asarray([[-0.2, -0.1], [0.3, 0.4]], dtype=np.float64)

    encoded_only = infer_latent_bounds(
        encoder,
        data,
        epsilon_frac=0.1,
        device=torch.device("cpu"),
    )
    with_image = infer_latent_bounds(
        encoder,
        data,
        epsilon_frac=0.1,
        device=torch.device("cpu"),
        latent_map=latent_map,
        clip_lower=[-0.9, -0.9],
        clip_upper=[0.9, 0.9],
    )

    np.testing.assert_allclose(encoded_only.lower, [-0.25, -0.15], atol=2e-8)
    np.testing.assert_allclose(encoded_only.upper, [0.35, 0.45], atol=2e-8)
    np.testing.assert_allclose(with_image.lower, [-0.9, -0.6], atol=3e-8)
    np.testing.assert_allclose(with_image.upper, [0.9, 0.9], atol=3e-8)


def _write_complete_ives_mg_artifacts(cfg: ExperimentConfig, output_dir: Path) -> None:
    morse_dir = output_dir / "MG"
    morse_dir.mkdir(parents=True)
    (morse_dir / "morse_graph").write_text("digraph { 0 -> 1 }\n")
    (morse_dir / "morse_sets").write_text("0,0,1,1,0\n")
    (output_dir / "mg_params_log.txt").write_text(
        "\n".join(
            [
                "Lower bounds: [-1.0, -1.0]",
                "Upper bounds: [1.0, 1.0]",
                f"subdiv_init: {cfg.cmgdb.subdiv_init}",
                f"subdiv_min: {cfg.cmgdb.subdiv_min}",
                f"subdiv_max: {cfg.cmgdb.subdiv_max}",
                f"subdiv_limit: {cfg.cmgdb.subdiv_limit}",
                f"bounds_epsilon_frac: {cfg.cmgdb.bounds_epsilon_frac}",
                f"padding: {cfg.cmgdb.padding}",
                f"box_map_backend: {cfg.cmgdb.box_map_backend}",
                f"bounds_data_role: {cfg.cmgdb.bounds_data_role}",
                f"bounds_grid_resolution: {cfg.cmgdb.bounds_grid_resolution}",
                f"bounds_include_latent_image: {cfg.cmgdb.bounds_include_latent_image}",
                f"bounds_clip_lower: {cfg.cmgdb.bounds_clip_lower}",
                f"bounds_clip_upper: {cfg.cmgdb.bounds_clip_upper}",
                f"adaptive_precompute_subdiv: {cfg.cmgdb.adaptive_precompute_subdiv}",
                f"compute_roa: {cfg.cmgdb.compute_roa}",
                "bounds_source: encoded_system_grid_and_latent_image",
            ]
        )
        + "\n"
    )


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("bounds_grid_resolution", 65),
        ("bounds_include_latent_image", False),
        ("bounds_clip_lower", [-0.95, -1.0]),
        ("bounds_clip_upper", [0.95, 1.0]),
    ],
)
def test_morse_resume_rejects_each_changed_system_grid_recipe_field(
    tmp_path: Path,
    field: str,
    changed_value: object,
) -> None:
    cfg = load_config(CONFIG_STEM).model_copy(deep=True)
    cfg.paths.data_dir = tmp_path / "data"
    cfg.paths.output_dir = tmp_path / "output"
    seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
    _write_complete_ives_mg_artifacts(cfg, seed_cfg.paths.output_dir)

    assert _stage_complete("morse", cfg, seed_cfg, train_file="train")

    stale_cfg = cfg.model_copy(deep=True)
    setattr(stale_cfg.cmgdb, field, changed_value)
    assert not _stage_complete("morse", stale_cfg, seed_cfg, train_file="train")


def test_generic_training_resume_requires_a_coherent_final_artifact_set(
    tmp_path: Path,
) -> None:
    cfg = load_config(CONFIG_STEM).model_copy(deep=True)
    cfg.paths.data_dir = tmp_path / "data"
    cfg.paths.output_dir = tmp_path / "output"
    seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
    save_checkpoint(
        build_autoencoder(cfg.arch),
        cfg.arch,
        seed_cfg.paths.model_dir,
    )

    assert not _stage_complete("train", cfg, seed_cfg, train_file="train")

    epochs_run = 2
    seed_cfg.paths.log_dir.mkdir(parents=True)
    (seed_cfg.paths.log_dir / "history.json").write_text(
        json.dumps(
            {
                "train": {"loss_total": [1.0, 0.5]},
                "val": {"loss_total": [1.1, 0.6]},
            }
        )
    )
    (seed_cfg.paths.output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "n_epochs_run": epochs_run,
                "loss_weights": list(cfg.training.loss_weights),
            }
        )
    )
    (seed_cfg.paths.output_dir / "final_losses.txt").write_text("loss_total: 0.5\n")

    assert _stage_complete("train", cfg, seed_cfg, train_file="train")


@pytest.mark.parametrize(
    ("cmgdb_updates", "message"),
    [
        ({"bounds_grid_resolution": 1}, "greater than or equal to 2"),
        (
            {"bounds_clip_lower": [-1.0, -1.0], "bounds_clip_upper": None},
            "must be set together",
        ),
        (
            {"bounds_clip_lower": [-1.0], "bounds_clip_upper": [1.0, 1.0]},
            "same length",
        ),
        (
            {"bounds_clip_lower": [-1.0, 0.0], "bounds_clip_upper": [1.0, 0.0]},
            "less than",
        ),
        (
            {"bounds_clip_lower": [-1.0], "bounds_clip_upper": [1.0]},
            "match arch.low_dims",
        ),
    ],
)
def test_schema_rejects_invalid_system_grid_bound_edges(
    cmgdb_updates: dict[str, object], message: str
) -> None:
    payload = load_config(CONFIG_STEM).model_dump(mode="python")
    payload["cmgdb"].update(cmgdb_updates)

    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(payload)
