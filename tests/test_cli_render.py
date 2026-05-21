from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from latentdynamics.cli import render as render_mod
from latentdynamics.config import (
    ArchConfig,
    CMGDBConfig,
    DataConfig,
    ExperimentConfig,
    PathsConfig,
    SystemConfig,
    TrainingConfig,
)


@dataclass(frozen=True)
class _RenderedMorse:
    morse_graph_pdf: Path
    morse_graph_png: Path
    morse_sets_paths: list[Path]


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.latent_map = torch.nn.Identity()


def _cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        system=SystemConfig(name="leslie_contraction"),
        arch=ArchConfig(num_layers=1, hidden_shape=4, high_dims=2, low_dims=2),
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
        cmgdb=CMGDBConfig(lower_bounds=[-1.0, -1.0], upper_bounds=[1.0, 1.0]),
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
    )


def test_render_stage_passes_requested_device_to_roa_overlay(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.paths.morse_dir.mkdir(parents=True)
    (cfg.paths.morse_dir / "morse_graph").write_text(
        'digraph { 0 [label="0", fillcolor="#111111"]; }\n'
    )
    (cfg.paths.morse_dir / "morse_sets").write_text("0,0,1,1,0\n")

    monkeypatch.setattr(
        render_mod,
        "render_morse_from_files",
        lambda _morse_dir, **_kwargs: _RenderedMorse(
            morse_graph_pdf=tmp_path / "morse_graph.pdf",
            morse_graph_png=tmp_path / "morse_graph.png",
            morse_sets_paths=[tmp_path / "morse_sets.png"],
        ),
    )
    monkeypatch.setattr(render_mod, "has_legacy_checkpoint", lambda _path: True)
    monkeypatch.setattr(render_mod, "has_new_checkpoint", lambda _path: False)
    monkeypatch.setattr(
        render_mod,
        "load_any_checkpoint",
        lambda _path, *, arch: (_Model(), arch),
    )

    captured: dict[str, str] = {}

    def fake_render_cell_graph_roa(
        _dot, _csv, _latent_map, out_path, *, resolution, device, title
    ):
        captured["device"] = device
        captured["resolution"] = resolution
        captured["title"] = title
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(render_mod, "render_cell_graph_roa", fake_render_cell_graph_roa)

    result = render_mod.render_stage(cfg, device="mps", verbose=False)

    assert captured["device"] == "mps"
    assert captured["resolution"] == 128
    assert "diagnostic regions of attraction" in captured["title"]
    assert str(cfg.paths.output_dir / "MG" / "regions_of_attraction.png") in result["figures"]


def test_render_stage_prefers_exact_roa_artifact(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.paths.morse_dir.mkdir(parents=True)
    (cfg.paths.morse_dir / "morse_graph").write_text(
        'digraph { 0 [label="0", fillcolor="#111111"]; }\n'
    )
    (cfg.paths.morse_dir / "morse_sets").write_text("0,0,1,1,0\n")
    (cfg.paths.morse_dir / "regions_of_attraction_exact.npz").write_bytes(b"npz")

    monkeypatch.setattr(
        render_mod,
        "render_morse_from_files",
        lambda _morse_dir, **_kwargs: _RenderedMorse(
            morse_graph_pdf=tmp_path / "morse_graph.pdf",
            morse_graph_png=tmp_path / "morse_graph.png",
            morse_sets_paths=[tmp_path / "morse_sets.png"],
        ),
    )

    called: dict[str, Path] = {}

    def fake_render_exact_roa_artifact(_artifact, _dot, out_path, *, title):
        called["artifact"] = Path(_artifact)
        called["out_path"] = Path(out_path)
        called["title"] = title
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"png")
        return Path(out_path)

    def fail_diagnostic(*_args, **_kwargs):
        raise AssertionError("diagnostic RoA renderer should not run when exact artifact exists")

    monkeypatch.setattr(render_mod, "render_exact_roa_artifact", fake_render_exact_roa_artifact)
    monkeypatch.setattr(render_mod, "render_cell_graph_roa", fail_diagnostic)

    result = render_mod.render_stage(cfg, device="cpu", verbose=False)

    assert called["artifact"] == cfg.paths.morse_dir / "regions_of_attraction_exact.npz"
    assert called["out_path"] == cfg.paths.output_dir / "MG" / "regions_of_attraction_exact.png"
    assert "exact regions of attraction" in called["title"]
    assert str(called["out_path"]) in result["figures"]
