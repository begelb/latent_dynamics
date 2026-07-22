"""Tests for the notebook-facing modes: config overrides, recompute_morse, retrain."""

from __future__ import annotations

import json

import pytest
import yaml
from pydantic import ValidationError

from latentdynamics.config import load_config
from latentdynamics.config.loader import deep_merge
from latentdynamics.replay import (
    DEFAULT_PLAYGROUND_ROOT,
    REPO_ROOT,
    _check_playground_dir,
    retrain,
)


def test_deep_merge_nested_and_replace():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}
    out = deep_merge(base, {"a": {"y": 20, "z": 30}, "b": [9]})
    assert out == {"a": {"x": 1, "y": 20, "z": 30}, "b": [9], "c": 3}
    # neither input mutated
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}


def test_load_config_with_overrides():
    cfg = load_config(
        "leslie3d_example1",
        overrides={"cmgdb": {"subdiv_max": 30}, "training": {"epochs": 7}},
    )
    assert cfg.cmgdb.subdiv_max == 30
    assert cfg.training.epochs == 7
    # untouched fields keep their YAML values
    base = load_config("leslie3d_example1")
    assert cfg.cmgdb.subdiv_init == base.cmgdb.subdiv_init
    assert cfg.arch.high_dims == base.arch.high_dims


def test_load_config_overrides_unknown_key_rejected():
    with pytest.raises(ValidationError):
        load_config("leslie3d_example1", overrides={"cmgdb": {"not_a_field": 1}})


def test_load_config_overrides_validated():
    # subdiv ordering validator must run against the merged config
    with pytest.raises(ValidationError):
        load_config("leslie3d_example1", overrides={"cmgdb": {"subdiv_max": 1}})


def test_playground_guard_rejects_preserved_trees(tmp_path):
    with pytest.raises(ValueError):
        _check_playground_dir(REPO_ROOT / "replay_sources" / "anything")
    with pytest.raises(ValueError):
        _check_playground_dir(REPO_ROOT / "paper_figures")
    # ordinary locations pass through
    assert _check_playground_dir(tmp_path / "ok") == tmp_path / "ok"
    assert _check_playground_dir(DEFAULT_PLAYGROUND_ROOT / "x") is not None


TINY_CONFIG = {
    "system": {"name": "coral"},
    "arch": {"num_layers": 1, "hidden_shape": 8, "high_dims": 13, "low_dims": 1},
    "training": {
        "learning_rate": 0.01,
        "batch_size": 8,
        "epochs": 2,
        "patience": 100,
        "loss_weights": [1, 1, 1],
    },
    "data": {
        "sampling_method": "uniform",
        "n_samples_train": 8,
        "n_samples_val": 8,
        "n_iterations": 2,
    },
    "cmgdb": {"subdiv_init": 2, "subdiv_min": 3, "subdiv_max": 4},
    "paths": {"data_dir": "data/_tiny", "output_dir": "output/_tiny"},
    "seeds": [0],
}


@pytest.mark.slow
def test_retrain_and_recompute_morse(tmp_path):
    cfg_yaml = tmp_path / "tiny_coral.yaml"
    cfg_yaml.write_text(yaml.safe_dump(TINY_CONFIG))
    run_root = tmp_path / "run"

    exp = retrain(
        cfg_yaml,
        out_root=run_root,
        stages=("data", "scale", "train", "morse"),
        verbose=False,
    )

    # the run is isolated under out_root: data, scaler, model, Morse artifacts
    assert (run_root / "data" / "train.csv").is_file()
    assert exp.seed_dir == run_root / "seed_0"
    assert (exp.morse_dir / "morse_graph").stat().st_size > 0
    assert (exp.morse_dir / "morse_sets").stat().st_size > 0
    lower, upper = exp.morse_bounds()
    assert lower is not None and upper is not None and len(lower) == 1

    manifest = json.loads((exp.seed_dir / "run_manifest.json").read_text())
    assert manifest["cmgdb_version"]

    # recompute the Morse graph of the saved model at different subdivisions
    play_dir = tmp_path / "play"
    redo = exp.recompute_morse(subdiv=(2, 2, 5), out_dir=play_dir, verbose=False)
    assert redo.seed_dir == play_dir
    assert (redo.morse_dir / "morse_graph").stat().st_size > 0
    redo_lower, _redo_upper = redo.morse_bounds()
    assert redo_lower == pytest.approx(lower)

    figs = redo.render_morse(out_dir=tmp_path / "render")
    assert figs.morse_graph_png.exists()

    # the playground guard refuses the preserved trees before any compute
    with pytest.raises(ValueError):
        exp.recompute_morse(subdiv=(2, 2, 4), out_dir=REPO_ROOT / "replay_sources" / "_guard")
