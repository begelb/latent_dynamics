"""Staleness guards for resumable pipeline stages."""

from __future__ import annotations

from pathlib import Path

from latentdynamics.cli.pipeline import _config_for_seed, _stage_complete
from latentdynamics.config import (
    ArchConfig,
    CMGDBConfig,
    DataConfig,
    ExperimentConfig,
    PathsConfig,
    SystemConfig,
    TrainingConfig,
)


def _tiny_cfg(tmp_path: Path, *, cmgdb: CMGDBConfig | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        system=SystemConfig(name="leslie3d"),
        arch=ArchConfig(num_layers=1, hidden_shape=4, high_dims=3, low_dims=2),
        training=TrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            epochs=2,
            patience=2,
            lr_patience=1,
            loss_weights=[1, 1, 1],
        ),
        data=DataConfig(
            sampling_method="uniform",
            n_samples_train=4,
            n_samples_val=4,
            n_iterations=2,
        ),
        cmgdb=cmgdb or CMGDBConfig(),
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
    )


def _seed_morse_artifacts(morse_dir: Path) -> None:
    morse_dir.mkdir(parents=True)
    (morse_dir / "morse_graph").write_text("digraph { 0 -> 1 }\n")
    (morse_dir / "morse_sets").write_text("0,0,1,1,0\n")


def _write_mg_log(path: Path, cfg: ExperimentConfig) -> None:
    path.write_text(
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
                f"compute_roa: {cfg.cmgdb.compute_roa}",
                f"roa_max_vertices: {cfg.cmgdb.roa_max_vertices}",
                "bounds_source: encoded_data",
            ]
        )
        + "\n"
    )


def test_morse_stage_incomplete_when_log_missing(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
    _seed_morse_artifacts(seed_cfg.paths.morse_dir)

    assert not _stage_complete("morse", cfg, seed_cfg, train_file="train")


def test_morse_stage_incomplete_when_log_mismatches_config(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
    _seed_morse_artifacts(seed_cfg.paths.morse_dir)
    _write_mg_log(seed_cfg.paths.output_dir / "mg_params_log.txt", cfg)

    stale_cfg = _tiny_cfg(tmp_path, cmgdb=CMGDBConfig(subdiv_init=5, subdiv_min=8, subdiv_max=10))

    assert not _stage_complete("morse", stale_cfg, seed_cfg, train_file="train")


def test_morse_stage_complete_when_log_matches_config(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
    _seed_morse_artifacts(seed_cfg.paths.morse_dir)
    _write_mg_log(seed_cfg.paths.output_dir / "mg_params_log.txt", cfg)

    assert _stage_complete("morse", cfg, seed_cfg, train_file="train")


def test_morse_stage_incomplete_when_exact_roa_enabled_but_artifact_missing(tmp_path):
    cfg = _tiny_cfg(tmp_path, cmgdb=CMGDBConfig(compute_roa=True))
    seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
    _seed_morse_artifacts(seed_cfg.paths.morse_dir)
    _write_mg_log(seed_cfg.paths.output_dir / "mg_params_log.txt", cfg)

    assert not _stage_complete("morse", cfg, seed_cfg, train_file="train")


def test_morse_stage_complete_when_exact_roa_enabled_and_artifact_present(tmp_path):
    cfg = _tiny_cfg(tmp_path, cmgdb=CMGDBConfig(compute_roa=True))
    seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
    _seed_morse_artifacts(seed_cfg.paths.morse_dir)
    _write_mg_log(seed_cfg.paths.output_dir / "mg_params_log.txt", cfg)
    (seed_cfg.paths.morse_dir / "regions_of_attraction_exact.npz").write_bytes(b"npz")

    assert _stage_complete("morse", cfg, seed_cfg, train_file="train")


def test_morse_stage_validates_fixed_bounds_when_configured(tmp_path):
    cfg = _tiny_cfg(
        tmp_path,
        cmgdb=CMGDBConfig(lower_bounds=[-2.0, -1.0], upper_bounds=[2.0, 1.0]),
    )
    seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
    _seed_morse_artifacts(seed_cfg.paths.morse_dir)
    _write_mg_log(seed_cfg.paths.output_dir / "mg_params_log.txt", cfg)

    assert not _stage_complete("morse", cfg, seed_cfg, train_file="train")


def test_train_stage_incomplete_when_legacy_checkpoint_files_are_empty(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
    model_dir = seed_cfg.paths.model_dir
    model_dir.mkdir(parents=True)
    for name in ("encoder.pt", "dynamics.pt", "decoder.pt"):
        (model_dir / name).write_bytes(b"")

    assert not _stage_complete("train", cfg, seed_cfg, train_file="train")
