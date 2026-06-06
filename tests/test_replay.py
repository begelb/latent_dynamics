"""Tests for ``latentdynamics.replay`` (the notebook-facing figure replay API).

Pure-logic tests run unconditionally. Tests that need on-disk paper artifacts
skip gracefully when those artifacts are not present (e.g. a fresh checkout
without the replay_sources/output trees), so the suite stays green on CI while
still exercising the real replay path locally.
"""

from __future__ import annotations

import numpy as np
import pytest

from latentdynamics import replay
from latentdynamics.replay import (
    ReplayExperiment,
    available_experiments,
    load_experiment,
    repo_path,
    resolve_config_path,
)


def test_available_experiments_lists_known_configs():
    names = available_experiments()
    for expected in ("leslie_2gen_contraction", "leslie3d_example1", "coral_data_scaling"):
        assert expected in names


def test_resolve_config_path_by_name_and_missing():
    path = resolve_config_path("leslie3d_example1")
    assert path.name == "leslie3d_example1.yaml"
    assert path.exists()
    with pytest.raises(FileNotFoundError):
        resolve_config_path("definitely_not_a_config_xyz")


def test_repo_path_is_absolute_under_root():
    path = repo_path("configs", "leslie3d_example1.yaml")
    assert path.is_absolute()
    assert path == replay.REPO_ROOT / "configs" / "leslie3d_example1.yaml"


def test_blocked_cell_raises_filenotfound(tmp_path):
    # A cell whose checkpoints exist but are 0-byte must fail clearly rather than
    # returning a broken model. coral_basic now has real models in replay_sources
    # (train_500 was populated), so synthesize the blocked state under an
    # output_dir override, which resolves to <output_dir>/train_500/seed_0/models.
    models = tmp_path / "train_500" / "seed_0" / "models"
    models.mkdir(parents=True)
    for fname in ("encoder.pt", "decoder.pt", "dynamics.pt"):
        (models / fname).touch()  # 0-byte stub
    with pytest.raises(FileNotFoundError):
        load_experiment("coral_basic", output_dir=tmp_path)


def _load_or_skip(name: str, **kwargs) -> ReplayExperiment:
    try:
        return load_experiment(name, **kwargs)
    except FileNotFoundError as exc:
        pytest.skip(f"artifacts for {name} not present: {exc}")


def test_load_replay_ready_leslie3d_example1():
    exp = _load_or_skip("leslie3d_example1_replay")
    assert isinstance(exp, ReplayExperiment)
    assert (exp.arch.high_dims, exp.arch.low_dims) == (3, 2)
    assert exp.seed_dir.exists()
    lower, upper = exp.morse_bounds()
    assert lower is not None and upper is not None
    assert len(lower) == 2 and len(upper) == 2


def test_render_morse_produces_pngs(tmp_path):
    # coral train_2000 is a tiny 1D Morse set, fast to render.
    exp = _load_or_skip("coral_data_scaling", train_file="train_2000", seed=0)
    figs = exp.render_morse(out_dir=tmp_path)
    assert figs.morse_graph_png.exists()
    assert any(p.suffix == ".png" and p.exists() for p in figs.morse_sets_paths)


def test_encode_advance_shapes():
    exp = _load_or_skip("coral_data_scaling", train_file="train_2000", seed=0)
    x = np.zeros((5, exp.arch.high_dims), dtype=np.float64)
    z = exp.encode(x)
    assert z.shape == (5, exp.arch.low_dims)
    assert exp.advance(z).shape == z.shape


def test_diagnostics_has_core_keys():
    exp = _load_or_skip("leslie3d_example1_replay")
    diag = exp.diagnostics()
    for key in ("experiment", "seed", "train_file", "dims", "seed_dir"):
        assert key in diag
