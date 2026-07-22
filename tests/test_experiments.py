"""Smoke tests for the unified pipeline + reproduce_paper.py wiring."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from latentdynamics.cli.pipeline import (
    ALL_STAGES,
    _config_for_seed,
    _derived_output_dir,
    _normalize_stages,
    _resolve_replay_root,
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
CONFIGS_DIR = REPO_ROOT / "src" / "latentdynamics" / "configs"


def _make_minimal_cfg(*, n_samples_train, output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        system=SystemConfig(name="coral"),
        arch=ArchConfig(num_layers=1, hidden_shape=4, high_dims=13, low_dims=1),
        training=TrainingConfig(
            learning_rate=1e-3, batch_size=8, epochs=1, patience=2, lr_patience=1, loss_weights=[1, 1, 1]
        ),
        data=DataConfig(
            sampling_method="uniform",
            n_samples_train=n_samples_train,
            n_samples_val=4,
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

    def test_normalize_stages_canonical_order(self):
        assert _normalize_stages(["metrics", "data"]) == ["data", "metrics"]

    def test_normalize_stages_unknown_rejected(self):
        with pytest.raises(ValueError):
            _normalize_stages(["bogus"])

    def test_normalize_stages_none_returns_all(self):
        assert _normalize_stages(None) == list(ALL_STAGES)

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

    def test_plan_cells_is_json_serializable(self, tmp_path):
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

    def test_stage_complete_checks_expected_artifacts(self, tmp_path):
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
            "leslie_2gen_contraction.yaml",
            "leslie3d_example1.yaml",
            "leslie3d_example2.yaml",
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

    @pytest.mark.parametrize(
        ("config_name", "output_dir"),
        [
            ("leslie3d_example2_replay.yaml", "replay_sources/leslie3d_example2"),
            ("leslie_2gen_contraction_replay.yaml", "replay_sources/leslie_2gen_contraction"),
            (
                "leslie3d_example1_replay.yaml",
                "replay_sources/leslie3d_example1/spurious_attractor_ex",
            ),
            ("coral_basic.yaml", "replay_sources/coral"),
            ("coral_data_scaling.yaml", "replay_sources/coral"),
            ("coral_adaptive.yaml", "replay_sources/coral"),
        ],
    )
    def test_paper_replay_configs_are_read_only(self, config_name: str, output_dir: str):
        cfg = load_config(CONFIGS_DIR / config_name)
        assert cfg.paths.read_only
        assert cfg.paths.output_dir == Path(output_dir)

    def test_chafee_infante_cmgdb_parameters(self):
        cfg = load_config(CONFIGS_DIR / "chafee_infante_replay.yaml")
        assert (cfg.cmgdb.subdiv_init, cfg.cmgdb.subdiv_min, cfg.cmgdb.subdiv_max) == (
            10,
            14,
            28,
        )
        assert cfg.cmgdb.lower_bounds == [-3.0, -2.0]
        assert cfg.cmgdb.upper_bounds == [3.0, 2.0]
        assert not cfg.cmgdb.padding

    @pytest.mark.parametrize(
        "config_name",
        ["coral_basic.yaml", "coral_data_scaling.yaml", "coral_adaptive.yaml"],
    )
    def test_coral_configs_cmgdb_parameters(self, config_name: str):
        cfg = load_config(CONFIGS_DIR / config_name)
        assert (cfg.cmgdb.subdiv_init, cfg.cmgdb.subdiv_min, cfg.cmgdb.subdiv_max) == (
            8,
            8,
            12,
        )


_MINIMAL_DOT = (
    "digraph {\n"
    '  0 [label="0", style=filled, fillcolor="#ff7f00"];\n'
    '  1 [label="1", style=filled, fillcolor="#1f77b4"];\n'
    "  0 -> 1;\n"
    "}\n"
)


def _seed_morse_artifacts(morse_dir: Path) -> None:
    """Write minimal but valid ``morse_graph`` (DOT) + ``morse_sets`` (CSV).

    The CSV is in the 1-D box format ``(lower, upper, label)`` consumed by
    :func:`render_morse_sets_from_csv`.
    """
    morse_dir.mkdir(parents=True, exist_ok=True)
    (morse_dir / "morse_graph").write_text(_MINIMAL_DOT)
    np.savetxt(
        morse_dir / "morse_sets",
        np.array([[0.0, 1.0, 0], [1.0, 2.0, 1]], dtype=np.float64),
        delimiter=",",
    )


def _read_only_cfg(*, tmp_path: Path) -> ExperimentConfig:
    cfg = ExperimentConfig(
        system=SystemConfig(name="coral"),
        arch=ArchConfig(num_layers=1, hidden_shape=4, high_dims=13, low_dims=1),
        training=TrainingConfig(
            learning_rate=1e-3, batch_size=8, epochs=1, patience=2, lr_patience=1, loss_weights=[1, 1, 1]
        ),
        data=DataConfig(
            sampling_method="uniform",
            n_samples_train=4,
            n_samples_val=4,
            n_iterations=2,
        ),
        cmgdb=CMGDBConfig(),
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "source" / "expt",
            read_only=True,
        ),
        seeds=[0],
        experiment_name="replay_smoke",
    )
    return cfg


class TestReplayRouting:
    def test_derived_output_dir_passthrough_when_not_read_only(self, tmp_path):
        cfg = _make_minimal_cfg(n_samples_train=500, output_dir=tmp_path)
        seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
        # Non-read-only configs ignore replay_root entirely.
        replay_root = _resolve_replay_root(cfg, tmp_path / "replay")
        assert replay_root is None
        assert (
            _derived_output_dir(cfg, seed_cfg, replay_root=replay_root) == seed_cfg.paths.output_dir
        )

    def test_derived_output_dir_redirects_under_read_only(self, tmp_path):
        cfg = _read_only_cfg(tmp_path=tmp_path)
        cfg.seeds = [0, 1]
        seed_cfg = _config_for_seed(cfg, train_file="train", seed=1)
        replay_root = _resolve_replay_root(cfg, tmp_path / "replay")
        assert replay_root == tmp_path / "replay"
        derived = _derived_output_dir(cfg, seed_cfg, replay_root=replay_root)
        # Must mirror seed substructure under <replay_root>/<experiment_name>/.
        assert derived == tmp_path / "replay" / "replay_smoke" / "seed_1"

    def test_force_overwrite_disables_replay_redirect(self, tmp_path):
        cfg = _read_only_cfg(tmp_path=tmp_path)
        replay_root = _resolve_replay_root(
            cfg,
            tmp_path / "replay",
            force_overwrite=True,
        )
        assert replay_root is None

    def test_render_replay_does_not_dirty_source_pdfs(self, tmp_path):
        cfg = _read_only_cfg(tmp_path=tmp_path)
        seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
        morse_dir = seed_cfg.paths.morse_dir
        _seed_morse_artifacts(morse_dir)

        # Simulate tracked, preserved source PDFs (the file content the
        # safety patch must not touch).
        source_pdfs = {
            morse_dir / "morse_graph.pdf": b"PRESERVED-GRAPH-PDF",
            morse_dir / "morse_graph.png": b"PRESERVED-GRAPH-PNG",
            morse_dir / "morse_sets.pdf": b"PRESERVED-SETS-PDF",
            morse_dir / "morse_sets.png": b"PRESERVED-SETS-PNG",
        }
        for path, blob in source_pdfs.items():
            path.write_bytes(blob)
        source_hashes = {p: hashlib.sha256(b).hexdigest() for p, b in source_pdfs.items()}

        replay_root = tmp_path / "replay"
        results = run(
            cfg,
            stages=["render"],
            device="cpu",
            verbose=False,
            replay_root=replay_root,
        )

        # Source PDFs are byte-identical to what we wrote.
        for path, want in source_hashes.items():
            assert path.exists()
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            assert got == want, f"source artifact mutated: {path}"

        # Replay tree mirrors the seed substructure and is populated.
        replay_seed_dir = replay_root / "replay_smoke" / "seed_0"
        replay_morse = replay_seed_dir / "MG"
        for name in ("morse_graph.pdf", "morse_graph.png", "morse_sets.pdf", "morse_sets.png"):
            target = replay_morse / name
            assert target.exists() and target.stat().st_size > 0, (
                f"missing replay artifact: {target}"
            )

        # Manifest also lives in the replay tree, not the source tree.
        manifest_path = Path(results[0]["manifest"])
        assert manifest_path == replay_seed_dir / "run_manifest.json"
        assert not (seed_cfg.paths.output_dir / "run_manifest.json").exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["cell"]["output_dir"] == str(seed_cfg.paths.output_dir)
        assert manifest["cell"]["replay_dir"] == str(replay_seed_dir)

    def test_skip_completed_under_replay_checks_replay_not_source(self, tmp_path):
        cfg = _read_only_cfg(tmp_path=tmp_path)
        seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
        _seed_morse_artifacts(seed_cfg.paths.morse_dir)
        replay_root = tmp_path / "replay"
        replay_seed_dir = replay_root / "replay_smoke" / "seed_0"
        replay_morse = replay_seed_dir / "MG"
        replay_morse.mkdir(parents=True)

        # Pre-populate the replay tree with sentinel content. If
        # skip_completed correctly checks the replay tree, render should
        # short-circuit and leave the sentinels in place.
        sentinel = b"REPLAY-ALREADY-DONE"
        replay_pdfs = [
            replay_morse / "morse_graph.pdf",
            replay_morse / "morse_graph.png",
            replay_morse / "morse_sets.pdf",
            replay_morse / "morse_sets.png",
        ]
        for p in replay_pdfs:
            p.write_bytes(sentinel)

        run(
            cfg,
            stages=["render"],
            device="cpu",
            verbose=False,
            skip_completed=True,
            replay_root=replay_root,
        )

        for p in replay_pdfs:
            assert p.read_bytes() == sentinel, (
                f"render did not honour skip_completed under replay: {p}"
            )

    def test_read_only_blocks_train_stage_even_with_replay_root(self, tmp_path):
        cfg = _read_only_cfg(tmp_path=tmp_path)
        with pytest.raises(RuntimeError, match="read_only"):
            run(
                cfg,
                stages=["train"],
                device="cpu",
                verbose=False,
                replay_root=tmp_path / "replay",
            )


class TestTopLevelPipelineScript:
    def test_cli_routes_summary_to_replay_for_read_only_config(self, tmp_path):
        cfg = _read_only_cfg(tmp_path=tmp_path)
        seed_cfg = _config_for_seed(cfg, train_file="train", seed=0)
        _seed_morse_artifacts(seed_cfg.paths.morse_dir)

        cfg_path = tmp_path / "read_only_config.json"
        cfg_path.write_text(json.dumps(cfg.model_dump(mode="json")))

        sys.path.insert(0, str(REPO_ROOT))
        previous = sys.modules.pop("pipeline", None)
        try:
            import importlib

            mod = importlib.import_module("pipeline")
            replay_root = tmp_path / "replay"
            rc = mod.main(
                [
                    "--config",
                    str(cfg_path),
                    "--stages",
                    "render",
                    "--replay-root",
                    str(replay_root),
                    "--quiet",
                ]
            )
        finally:
            sys.modules.pop("pipeline", None)
            if previous is not None:
                sys.modules["pipeline"] = previous
            sys.path.pop(0)

        assert rc == 0
        replay_expt = replay_root / "replay_smoke"
        summary_path = replay_expt / "pipeline_summary.json"
        assert summary_path.exists()
        assert not (cfg.paths.output_dir / "pipeline_summary.json").exists()

        summary = json.loads(summary_path.read_text())
        assert summary[0]["replay_dir"] == str(replay_expt / "seed_0")
        assert (replay_expt / "seed_0" / "MG" / "morse_sets.pdf").exists()


class TestReproducePaperScript:
    def test_module_imports_and_lists_experiments(self):
        sys.path.insert(0, str(REPO_ROOT))
        try:
            import importlib

            mod = importlib.import_module("reproduce_paper")
            assert isinstance(mod.EXPERIMENTS, dict)
            assert len(mod.EXPERIMENTS) >= 7
            # EXPERIMENTS values are packaged config stems (resolved by load_config).
            for stem in mod.EXPERIMENTS.values():
                assert (CONFIGS_DIR / f"{stem}.yaml").exists(), stem
        finally:
            sys.path.pop(0)
