from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from latentdynamics.cli import morse_graph
from latentdynamics.config import (
    ArchConfig,
    CMGDBConfig,
    DataConfig,
    ExperimentConfig,
    PathsConfig,
    SystemConfig,
    TrainingConfig,
)


def _cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        system=SystemConfig(name="leslie_contraction"),
        arch=ArchConfig(num_layers=1, hidden_shape=4, high_dims=2, low_dims=1),
        training=TrainingConfig(
            learning_rate=1e-3,
            batch_size=2,
            epochs=1,
            patience=2,
            lr_patience=1,
        ),
        data=DataConfig(
            sampling_method="uniform",
            n_samples_train=2,
            n_samples_val=2,
            n_iterations=1,
        ),
        cmgdb=CMGDBConfig(lower_bounds=[-1.0], upper_bounds=[1.0]),
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
    )


def test_run_uses_any_checkpoint_loader_without_autoencoder_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    models = cfg.paths.output_dir / "models"
    models.mkdir(parents=True)
    for name in ("encoder.pt", "dynamics.pt", "decoder.pt"):
        (models / name).write_bytes(b"legacy")

    calls: list[tuple[Path, ArchConfig]] = []
    model = torch.nn.Identity()

    def fake_load_any_checkpoint(path: Path, *, arch: ArchConfig):
        calls.append((path, arch))
        return model, arch

    monkeypatch.setattr(morse_graph, "load_any_checkpoint", fake_load_any_checkpoint)
    monkeypatch.setattr(
        morse_graph,
        "_load_data_and_scale",
        lambda _cfg, _train_file: np.zeros((2, _cfg.arch.high_dims)),
    )
    monkeypatch.setattr(
        morse_graph,
        "compute_morse_graph",
        lambda actual_model, bounds, cmgdb, device: ("graph", None),
    )

    def fake_save_morse_graph_artifacts(morse, morse_dir):
        morse_dir.mkdir(parents=True)
        dot = morse_dir / "morse_graph"
        csv = morse_dir / "morse_sets"
        dot.write_text("digraph {}")
        csv.write_text("0,0")
        return dot, csv

    monkeypatch.setattr(morse_graph, "save_morse_graph_artifacts", fake_save_morse_graph_artifacts)

    morse_graph.run(cfg, device="cpu", verbose=False)

    assert calls == [(models, cfg.arch)]
    assert (cfg.paths.morse_dir / "morse_graph").read_text() == "digraph {}"


def test_run_keeps_no_overwrite_guard_before_loading_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    cfg.paths.morse_dir.mkdir(parents=True)
    (cfg.paths.morse_dir / "morse_graph").write_text("existing")

    def fail_load_any_checkpoint(*args, **kwargs):
        raise AssertionError("checkpoint loader should not run when artifacts exist")

    monkeypatch.setattr(morse_graph, "load_any_checkpoint", fail_load_any_checkpoint)

    with pytest.raises(RuntimeError, match="prior Morse artifacts present"):
        morse_graph.run(cfg, device="cpu", verbose=False)


def test_run_writes_exact_roa_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path).model_copy(
        update={
            "arch": ArchConfig(num_layers=1, hidden_shape=4, high_dims=2, low_dims=2),
            "cmgdb": CMGDBConfig(
                lower_bounds=[-1.0, -1.0],
                upper_bounds=[1.0, 1.0],
                compute_roa=True,
                collapse_roa_to_lca=False,
            ),
        }
    )
    models = cfg.paths.output_dir / "models"
    models.mkdir(parents=True)
    for name in ("encoder.pt", "dynamics.pt", "decoder.pt"):
        (models / name).write_bytes(b"legacy")

    model = torch.nn.Identity()
    morse = object()
    map_graph = object()
    calls = []

    monkeypatch.setattr(
        morse_graph,
        "load_any_checkpoint",
        lambda _path, *, arch: (model, arch),
    )
    monkeypatch.setattr(
        morse_graph,
        "_load_data_and_scale",
        lambda _cfg, _train_file: np.zeros((2, _cfg.arch.high_dims)),
    )
    monkeypatch.setattr(
        morse_graph,
        "compute_morse_graph",
        lambda actual_model, bounds, cmgdb, device: (morse, map_graph),
    )

    def fake_save_morse_graph_artifacts(_morse, morse_dir):
        morse_dir.mkdir(parents=True)
        dot = morse_dir / "morse_graph"
        csv = morse_dir / "morse_sets"
        dot.write_text('digraph { 0 [label="0"]; }\n')
        csv.write_text("0,1,0\n")
        return dot, csv

    def fake_compute_and_save_exact_roa(
        *,
        map_graph,
        cmgdb_morse_graph,
        morse_graph_dot,
        out_dir,
        lower_bounds,
        upper_bounds,
        max_vertices,
        collapse_to_lca,
    ):
        calls.append(
            (
                map_graph,
                cmgdb_morse_graph,
                Path(morse_graph_dot),
                Path(out_dir),
                (lower_bounds, upper_bounds),
                max_vertices,
                collapse_to_lca,
            )
        )
        out = Path(out_dir) / "regions_of_attraction_exact.npz"
        out.write_bytes(b"npz")
        return out

    monkeypatch.setattr(morse_graph, "save_morse_graph_artifacts", fake_save_morse_graph_artifacts)
    monkeypatch.setattr(morse_graph, "compute_and_save_exact_roa", fake_compute_and_save_exact_roa)

    morse_graph.run(cfg, device="cpu", verbose=False)

    assert len(calls) == 1
    assert calls[0][0] is map_graph
    assert calls[0][1] is morse
    assert calls[0][2] == cfg.paths.morse_dir / "morse_graph"
    assert calls[0][3] == cfg.paths.morse_dir
    assert calls[0][6] is False
