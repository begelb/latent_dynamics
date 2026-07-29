"""Focused contract tests for the literal-repeat Chafee d=1 driver."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    script = scripts / "repeat_chafee_d1_full_batch.py"
    spec = importlib.util.spec_from_file_location(
        "repeat_chafee_d1_full_batch",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPEAT = _load_module()


def test_matrix_is_three_literal_process_repeats_of_five_seed_plan() -> None:
    trials = REPEAT.all_trials()

    assert len(trials) == 15
    assert [trial.repeat_index for trial in trials] == [
        repeat for repeat in range(3) for _ in range(5)
    ]
    for repeat_index in REPEAT.REPEAT_INDICES:
        replicate = REPEAT.repeat_trials(repeat_index)
        assert [trial.base_seed for trial in replicate] == [0, 1, 2, 3, 4]
        assert [trial.training_seed for trial in replicate] == [0, 1, 2, 3, 4]
        specs = [trial.training_spec() for trial in replicate]
        assert all(spec.epochs == 4_000 for spec in specs)
        assert all(spec.learning_rate == pytest.approx(0.003) for spec in specs)
        assert all(spec.scheduler_factor == pytest.approx(0.5) for spec in specs)
        assert all(spec.scheduler_patience == 100 for spec in specs)
        assert all(spec.scheduler_threshold == pytest.approx(1e-4) for spec in specs)
        assert all(spec.scheduler_min_lr == pytest.approx(1e-6) for spec in specs)

    by_seed = {
        seed: [trial.training_seed for trial in trials if trial.base_seed == seed]
        for seed in REPEAT.BASE_SEEDS
    }
    assert by_seed == {seed: [seed, seed, seed] for seed in range(5)}
    assert REPEAT.CANONICAL_DEVICE == "mps"


def test_replicate_commands_use_three_fresh_python_process_targets(
    tmp_path: Path,
) -> None:
    commands = [
        REPEAT._replicate_command(
            repeat_index=repeat,
            replicate_root=tmp_path / f"repeat_{repeat:02d}",
            quiet=True,
        )
        for repeat in REPEAT.REPEAT_INDICES
    ]

    assert len(commands) == 3
    assert all(command[0] == sys.executable for command in commands)
    assert len({command[command.index("--_replicate-root") + 1] for command in commands}) == 3
    assert [int(command[command.index("--_replicate-index") + 1]) for command in commands] == [
        0,
        1,
        2,
    ]


def test_outer_target_is_fail_if_present_and_protected(tmp_path: Path) -> None:
    target = tmp_path / "fresh"
    assert REPEAT._assert_fresh_outer(target) == target.resolve()
    target.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        REPEAT._assert_fresh_outer(target)

    with pytest.raises(ValueError, match="overlaps protected"):
        REPEAT._assert_fresh_outer(REPEAT.sweep.CANONICAL_RUN)


def test_strict_subset_validation_rejects_any_query_disagreement() -> None:
    full = np.asarray([3, -1, 8, 8], dtype=np.int32)
    ids = np.asarray([0, 2, 3], dtype=np.int64)
    REPEAT._validate_strict_query_subset(
        full,
        ids,
        np.asarray([3, 8, 8], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="disagrees"):
        REPEAT._validate_strict_query_subset(
            full,
            ids,
            np.asarray([3, 8, 9], dtype=np.int32),
        )
    with pytest.raises(ValueError, match="outside"):
        REPEAT._validate_strict_query_subset(
            full,
            np.asarray([4], dtype=np.int64),
            np.asarray([8], dtype=np.int32),
        )


class _FakeMapGraph:
    def num_vertices(self) -> int:
        return 256

    def adjacencies(self, vertex: int) -> tuple[int]:
        return (vertex,)


class _FakeCmgdbMorseGraph:
    def morse_set(self, node: int) -> tuple[int]:
        return (0,) if node == 4 else (255,)


class _FakeDag:
    nodes = (4, 9)
    minimal = frozenset((4, 9))

    def lca_of_minimals(self, _nodes):
        return None


def test_full_uniform_roa_persists_all_256_cells_and_distinct_semantics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = tmp_path / "run"
    paths = REPEAT.single.ExactRunPaths(output_root=run, dimension=1)
    paths.uniform.mkdir(parents=True)
    (paths.uniform / "morse_graph").write_text("digraph {}\n", encoding="utf-8")
    queried_ids = np.asarray([0, 2, 255], dtype=np.int64)

    def strict_lookup(_map_graph, _morse_graph, ids):
        labels = np.full(np.asarray(ids).shape, -1, dtype=np.int32)
        labels[np.asarray(ids) == 0] = 4
        labels[np.asarray(ids) == 255] = 9
        return np.ascontiguousarray(labels)

    np.savez_compressed(
        paths.uniform / "marcio_singleton_reachability_queries.npz",
        queried_cell_ids=queried_ids,
        singleton_node_by_queried_cell=strict_lookup(None, None, queried_ids),
        point_candidate_cell_ids=queried_ids[:2],
        point_singleton_nodes=strict_lookup(None, None, queried_ids[:2]),
        point_basin_labels=np.asarray([4, -1], dtype=np.int32),
        root_candidate_cell_ids=queried_ids[2:],
        root_singleton_nodes=strict_lookup(None, None, queried_ids[2:]),
        encoded_stable_roots=np.asarray([[-1.0], [1.0]], dtype=np.float64),
    )
    np.save(
        run / "trajectory_basin_labels.npy",
        np.asarray([4, -1], dtype=np.int32),
    )
    np.save(
        run / "encoded_stable_roots.npy",
        np.asarray([[-1.0], [1.0]], dtype=np.float64),
    )
    monkeypatch.setattr(
        REPEAT.study,
        "_native_singleton_reachability",
        strict_lookup,
    )
    monkeypatch.setattr(
        REPEAT.study,
        "_morse_attractors",
        lambda _graph: [4, 9],
    )
    monkeypatch.setattr(
        REPEAT.MorseGraph,
        "from_dot",
        staticmethod(lambda _path: _FakeDag()),
    )

    record = REPEAT._save_full_roa_artifacts(
        paths=paths,
        bounds=SimpleNamespace(
            lower=np.asarray([-2.0]),
            upper=np.asarray([2.0]),
        ),
        morse_graph=_FakeCmgdbMorseGraph(),
        map_graph=_FakeMapGraph(),
    )

    assert record["status"] == "complete"
    assert record["uniform_cells"] == 256
    strict_path = paths.uniform / "regions_of_attraction_strict_singleton.npz"
    exact_path = paths.uniform / "regions_of_attraction_exact.npz"
    with np.load(strict_path) as strict:
        assert np.array_equal(strict["cell_ids"], np.arange(256))
        assert strict["singleton_node_by_cell"].shape == (256,)
        assert strict["singleton_node_by_cell"].dtype == np.int32
    with np.load(exact_path) as exact:
        assert exact["box_roa"].shape == (256,)
        assert set(exact.files) >= {"box_roa", "reach_mask", "minimal_order"}
    strict_metadata = json.loads(
        (paths.uniform / "regions_of_attraction_strict_singleton.json").read_text(encoding="utf-8")
    )
    exact_metadata = json.loads(
        (paths.uniform / "regions_of_attraction_exact.json").read_text(encoding="utf-8")
    )
    assert strict_metadata["queried_subset_validation"]["status"] == ("validated_exact_subset")
    assert strict_metadata["not_equivalent_to_exact_blocker_lca_roa"] is True
    assert exact_metadata["used_for_marcio_trajectory_statistics"] is False
    assert exact_metadata["different_from_strict_singleton_lookup"] is True


def test_topology_inventory_does_not_require_basin_statistics(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    paths = REPEAT.single.ExactRunPaths(output_root=run, dimension=1)
    required = (
        paths.models / "autoencoder.pt",
        paths.uniform / "morse_graph",
        paths.uniform / "morse_sets",
        paths.adaptive / "morse_graph",
        paths.adaptive / "morse_sets",
        paths.uniform / "regions_of_attraction_strict_singleton.npz",
        paths.uniform / "regions_of_attraction_strict_singleton.json",
        paths.uniform / "regions_of_attraction_exact.npz",
        paths.uniform / "regions_of_attraction_exact.json",
        run / "analysis_manifest.json",
        run / "topology_roa_augmentation.json",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())

    inventory = REPEAT._trial_artifact_inventory(
        run,
        {"status": "complete"},
    )

    assert inventory["topology_status"] == "complete"
    assert inventory["full_uniform_roa_status"] == "complete"
    assert inventory["basin_statistics_status"] == "unavailable_or_invalid"
    assert inventory["files"]["basin_statistics"]["status"] == "unavailable"


def test_seed_repeat_diagnostics_keep_unavailable_statistics_explicit() -> None:
    manifests = []
    for repeat in range(3):
        manifests.append(
            {
                "trials": [
                    {
                        "repeat_index": repeat,
                        "base_seed": 0,
                        "correct_combined_percent": (None if repeat == 1 else 40.0 + 10.0 * repeat),
                        "artifacts": {"files": {"checkpoint": {"sha256": "same-checkpoint"}}},
                    }
                ]
            }
        )

    row = REPEAT._seed_repeat_diagnostics(manifests)[0]

    assert row["repeats_reported"] == [0, 1, 2]
    assert row["checkpoints_bitwise_identical_across_available_repeats"] is True
    assert row["statistics_available_repeats"] == 2
    assert row["correct_combined_percent"]["values"] == [40.0, 60.0]
    assert row["correct_combined_percent"]["mean"] == pytest.approx(50.0)
    assert row["correct_combined_percent"]["population_std"] == pytest.approx(10.0)
