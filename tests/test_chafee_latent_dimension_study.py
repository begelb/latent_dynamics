from __future__ import annotations

import json

import numpy as np
import pytest
from scripts import chafee_latent_dimension_study as study


def _exact_inputs(tmp_path):
    return study.ExactInputs(
        archive_dir=tmp_path,
        train_data=tmp_path / "train_data.csv",
        trajectory_labels=tmp_path / "traj_attractors.pkl",
        stable_roots=tmp_path / "stable_solutions.csv",
        hashes={
            "train_data.csv": study.TRAIN_DATA_SHA256,
            "traj_attractors.pkl": study.TRAJECTORY_LABELS_SHA256,
            "stable_solutions.csv": study.STABLE_ROOTS_SHA256,
        },
        sizes_bytes={
            "train_data.csv": 1,
            "traj_attractors.pkl": 2,
            "stable_solutions.csv": 3,
        },
    )


def test_matched_resolution_and_reference_architecture_are_exact():
    one = study.RESOLUTIONS[1]
    three = study.RESOLUTIONS[3]

    assert (one.adaptive_init, one.adaptive_min, one.adaptive_max) == (7, 8, 11)
    assert (one.uniform_init, one.uniform_min, one.uniform_max) == (8, 8, 8)
    assert (three.adaptive_init, three.adaptive_min, three.adaptive_max) == (21, 24, 33)
    assert (three.uniform_init, three.uniform_min, three.uniform_max) == (24, 24, 24)
    assert one.uniform_cells == 256
    assert three.uniform_cells == 256**3

    arch = study.reference_architecture(3)
    assert arch.high_dims == 64
    assert arch.low_dims == 3
    assert arch.component("encoder").hidden_shapes == (64, 32)
    assert arch.component("latent_map").hidden_shapes == (32, 32)
    assert arch.component("decoder").hidden_shapes == (32, 64)
    for component in ("encoder", "latent_map", "decoder"):
        resolved = arch.component(component)
        assert resolved.activation == "tanh"
        assert resolved.out_activation == "none"


def test_stage_parser_is_resumable_and_dependency_ordered():
    assert study._parse_stages(["adaptive,precompute-fine", "bounds"]) == (
        "bounds",
        "precompute-fine",
        "adaptive",
    )
    assert study._parse_stages(["all"]) == study.STAGE_ORDER
    assert study._parse_stages(["validate"]) == ()
    with pytest.raises(ValueError, match="unknown stages"):
        study._parse_stages(["guess"])


def test_one_dimensional_fine_selection_covers_every_level_8_cell():
    bounds = study.LatentBounds(
        lower=np.asarray([-2.0]),
        upper=np.asarray([3.0]),
    )
    boxes = study._all_uniform_boxes(bounds, study.RESOLUTIONS[1])

    assert boxes.shape == (256, 2)
    assert boxes[0, 0] == -2.0
    assert boxes[-1, 1] == 3.0
    np.testing.assert_allclose(boxes[1:, 0], boxes[:-1, 1], rtol=0, atol=0)


def test_three_dimensional_fine_selection_uses_all_saved_morse_sets(tmp_path):
    bounds = study.LatentBounds(lower=np.zeros(3), upper=np.ones(3))
    width = 1.0 / 256.0
    rows = np.asarray(
        [
            [0.0, 0.0, 0.0, width, width, width, 7.0],
            [width, 0.0, 0.0, 2 * width, width, width, 9.0],
            [0.0, 0.0, 0.0, width, width, width, 7.0],
        ]
    )
    path = tmp_path / "morse_sets"
    np.savetxt(path, rows, delimiter=",")

    boxes, counts = study._recurrent_uniform_boxes(
        path,
        dimension=3,
        bounds=bounds,
        resolution=study.RESOLUTIONS[3],
    )

    assert boxes.shape == (2, 6)
    assert counts == {7: 2, 9: 1}


def test_hierarchical_stage_requires_persisted_dense_and_sparse_arrays(tmp_path):
    paths = study.DimensionPaths(output_root=tmp_path, dimension=3)
    paths.hierarchical_table.mkdir(parents=True)
    for name in (
        "metadata.json",
        "coarse_values.npy",
        "active_coarse_indices.npy",
        "fine_block_values.npy",
        "active_coarse_boxes.npy",
    ):
        (paths.hierarchical_table / name).touch()

    assert study._stage_outputs_exist(paths, "precompute-fine")
    (paths.hierarchical_table / "fine_block_values.npy").unlink()
    assert not study._stage_outputs_exist(paths, "precompute-fine")


def test_manual_level24_table_and_bounds_are_resumed_in_place(tmp_path):
    paths = study.DimensionPaths(output_root=tmp_path, dimension=3)
    manual = paths.run / "precomputed_level24_to33"
    manual.mkdir(parents=True)
    assert paths.coarse_table == manual
    assert paths.hierarchical_table == manual

    bounds_path = paths.run / "bounds.json"
    bounds_path.write_text(
        json.dumps(
            {
                "lower": [-3.0, -2.0, -1.0],
                "upper": [1.0, 2.0, 3.0],
                "epsilon_fraction": 0.1,
            }
        )
    )
    bounds = study._load_bounds(bounds_path, 3)
    np.testing.assert_array_equal(bounds.lower, [-3.0, -2.0, -1.0])
    np.testing.assert_array_equal(bounds.upper, [1.0, 2.0, 3.0])


def test_uniform_point_cells_include_both_sides_of_an_exact_boundary():
    bounds = study.LatentBounds(lower=np.asarray([0.0]), upper=np.asarray([1.0]))
    points = np.asarray([[0.1], [0.25], [-0.1], [1.0]])

    cells = study._uniform_point_cells(points, bounds, study.RESOLUTIONS[1])

    np.testing.assert_array_equal(cells.candidates(0), [25])
    np.testing.assert_array_equal(cells.candidates(1), [63, 64])
    assert cells.candidates(2).size == 0
    np.testing.assert_array_equal(cells.candidates(3), [255])


def test_native_singleton_query_contract_and_negative_basin_priority(monkeypatch):
    def fake_query(_map_graph, _morse_graph, query):
        return np.asarray([5, 7, -2], dtype=np.int32)[: len(query)]

    monkeypatch.setattr(study.CMGDB, "MorseSingletonReachability", fake_query)
    queried = study._native_singleton_reachability(
        object(),
        object(),
        np.asarray([10, 11, 12]),
    )
    np.testing.assert_array_equal(queried, [5, 7, -2])

    cells = study.UniformPointCells(
        flat_cell_ids=np.asarray([10, 11, 12, 13]),
        offsets=np.asarray([0, 2, 3, 4]),
    )
    labels = study._point_basin_labels(
        np.asarray([7, 5, 5, -2], dtype=np.int32),
        cells,
        negative_attractor=5,
        positive_attractor=7,
    )
    # The first point touches both basins; the archived loop tests the
    # negative basin first.
    np.testing.assert_array_equal(labels, [5, 5, study.OUTSIDE])


def test_morse_attractors_reads_unreduced_edges_once_without_adjacency_queries():
    class FakeMorseGraph:
        edge_calls = 0

        @staticmethod
        def vertices():
            return (0, 1, 2, 3)

        def edges_unreduced(self):
            self.edge_calls += 1
            return ((3, 2), (3, 1), (2, 1), (0, 0))

        @staticmethod
        def edges():
            raise AssertionError("reduced edges must not be computed")

        @staticmethod
        def adjacencies(_node):
            raise AssertionError("per-node adjacency queries must not be used")

    graph = FakeMorseGraph()

    assert study._morse_attractors(graph) == [0, 1]
    assert graph.edge_calls == 1


def test_morse_attractors_uses_one_reduced_edge_fallback():
    class LegacyMorseGraph:
        edge_calls = 0

        @staticmethod
        def vertices():
            return (0, 1, 2)

        def edges(self):
            self.edge_calls += 1
            return ((2, 1),)

        @staticmethod
        def adjacencies(_node):
            raise AssertionError("per-node adjacency queries must not be used")

    graph = LegacyMorseGraph()

    assert study._morse_attractors(graph) == [0, 1]
    assert graph.edge_calls == 1


def test_live_statistics_requires_exactly_two_uniform_attractors(
    monkeypatch,
    tmp_path,
):
    class FakeMorseGraph:
        def num_vertices(self):
            return 3

        def edges_unreduced(self):
            return ()

    paths = study.DimensionPaths(output_root=tmp_path, dimension=1)
    paths.run.mkdir(parents=True)
    inputs = _exact_inputs(tmp_path)
    monkeypatch.setattr(
        study,
        "_load_trajectory_labels",
        lambda _path: pytest.fail("trajectory data loaded before bistability check"),
    )
    monkeypatch.setattr(
        study,
        "_native_singleton_reachability",
        lambda *_args: pytest.fail("reachability queried before bistability check"),
    )

    with pytest.raises(ValueError, match="exactly two minimal attracting Morse nodes"):
        study._compute_live_reference_statistics(
            paths,
            inputs,
            device=study.torch.device("cpu"),
            bounds=study.LatentBounds(
                lower=np.asarray([-1.0]),
                upper=np.asarray([1.0]),
            ),
            resolution=study.RESOLUTIONS[1],
            map_graph=object(),
            morse_graph=FakeMorseGraph(),
        )
    assert not paths.stats.exists()


def test_study_config_records_zero_neural_evaluations_during_cmgdb(tmp_path):
    inputs = _exact_inputs(tmp_path)
    paths = study.DimensionPaths(output_root=tmp_path / "out", dimension=3)

    payload = study._study_config(
        paths,
        inputs,
        device=study.torch.device("cpu"),
        batch_points="auto",
        adaptive_topology_only=False,
    )

    assert payload["cmgdb"]["callback_policy"].startswith("persisted lookup only")
    assert payload["cmgdb"]["conley_policy"] == {
        "uniform": "deferred_to_adaptive",
        "adaptive": "attempt_conley_with_topology_only_error_fallback",
    }
    assert payload["cmgdb"]["uniform_cells"] == 256**3
    reproducibility = payload["training"]["reproducibility"]
    assert reproducibility["canonical_artifact_backend"] == "mps"
    assert reproducibility["seed"] == 0
    assert isinstance(reproducibility["deterministic_algorithms_enforced"], bool)
    assert not reproducibility[
        "bitwise_reproducible_across_backends_or_runtime_versions"
    ]
    assert "may produce numerically different" in reproducibility["limitation"]
    # Ensure the payload is stable JSON rather than containing Path/NumPy values.
    json.dumps(payload)


def test_lookup_cmgdb_topology_only_never_calls_conley(monkeypatch):
    calls: list[str] = []

    class FakeModel:
        def set_batch_map(self, callback):
            assert callable(callback)

    class FakeMapGraph:
        def has_cache(self):
            return True

    class FakeBoxMap:
        @staticmethod
        def batch(_rectangles):
            return []

    monkeypatch.setattr(study.CMGDB, "Model", lambda *_args: FakeModel())
    monkeypatch.setattr(
        study.CMGDB,
        "ComputeMorseGraph",
        lambda _model: (calls.append("topology") or object(), FakeMapGraph()),
    )
    monkeypatch.setattr(
        study.CMGDB,
        "ComputeConleyMorseGraph",
        lambda _model: (calls.append("conley") or object(), FakeMapGraph()),
    )

    _morse, _map, _duration, status = study._run_lookup_cmgdb(
        FakeBoxMap(),
        study.LatentBounds(lower=np.asarray([-1.0]), upper=np.asarray([1.0])),
        subdiv_init=8,
        subdiv_min=8,
        subdiv_max=8,
        compute_conley=False,
    )

    assert calls == ["topology"]
    assert status["routine"] == "CMGDB.ComputeMorseGraph"
    assert not status["requested"]
    assert not status["computed"]
    assert status["topology_morse_sets_map_graph_and_basins_invariant"]


def test_lookup_cmgdb_records_conley_error_and_falls_back(monkeypatch):
    calls: list[str] = []

    class FakeModel:
        def set_batch_map(self, callback):
            assert callable(callback)

    class FakeMapGraph:
        def has_cache(self):
            return True

    class FakeBoxMap:
        @staticmethod
        def batch(_rectangles):
            return []

    def fail_conley(_model):
        calls.append("conley")
        raise RuntimeError("smith normal form failed")

    monkeypatch.setattr(study.CMGDB, "Model", lambda *_args: FakeModel())
    monkeypatch.setattr(study.CMGDB, "ComputeConleyMorseGraph", fail_conley)
    monkeypatch.setattr(
        study.CMGDB,
        "ComputeMorseGraph",
        lambda _model: (calls.append("topology") or object(), FakeMapGraph()),
    )

    _morse, _map, _duration, status = study._run_lookup_cmgdb(
        FakeBoxMap(),
        study.LatentBounds(lower=np.asarray([-1.0]), upper=np.asarray([1.0])),
        subdiv_init=7,
        subdiv_min=8,
        subdiv_max=11,
        compute_conley=True,
        fallback_to_topology_on_conley_error=True,
    )

    assert calls == ["conley", "topology"]
    assert status["status"] == "failed_then_topology_only"
    assert status["routine"] == "CMGDB.ComputeMorseGraph"
    assert status["error_type"] == "RuntimeError"
    assert "smith normal form failed" in status["error_message"]


def _write_minimal_training_artifacts(paths):
    paths.models.mkdir(parents=True)
    (paths.models / "autoencoder.pt").write_bytes(b"checkpoint-v1")
    (paths.models / "autoencoder.json").write_text("{}\n")
    (paths.run / "training_summary.json").write_text("{}\n")
    (paths.run / "logs").mkdir()
    (paths.run / "logs" / "history.json").write_text("{}\n")


def _stage_reusable(paths, inputs, stage, *, device=None):
    if device is None:
        device = study.torch.device("cpu")
    return study._stage_is_reusable(
        paths,
        inputs,
        stage,
        device=device,
        batch_points="auto",
        min_box_side_frac=0.0025,
        adaptive_topology_only=False,
        cache={},
        memo={},
    )


def test_stage_provenance_rejects_changed_output_and_training_backend(tmp_path):
    paths = study.DimensionPaths(output_root=tmp_path, dimension=1)
    paths.run.mkdir(parents=True)
    inputs = _exact_inputs(tmp_path)
    _write_minimal_training_artifacts(paths)
    direct = study._stage_direct_inputs(
        paths,
        inputs,
        "train",
        device=study.torch.device("cpu"),
        batch_points="auto",
        min_box_side_frac=0.0025,
        adaptive_topology_only=False,
        cache={},
    )
    study._stamp_stage_provenance(paths, "train", direct, cache={})

    assert _stage_reusable(paths, inputs, "train")
    assert not _stage_reusable(
        paths,
        inputs,
        "train",
        device=study.torch.device("mps"),
    )

    (paths.models / "autoencoder.pt").write_bytes(b"checkpoint-v2")
    assert not _stage_reusable(paths, inputs, "train")


def test_changed_checkpoint_invalidates_provenance_of_bounds_descendant(tmp_path):
    paths = study.DimensionPaths(output_root=tmp_path, dimension=1)
    paths.run.mkdir(parents=True)
    inputs = _exact_inputs(tmp_path)
    _write_minimal_training_artifacts(paths)
    training_direct = study._stage_direct_inputs(
        paths,
        inputs,
        "train",
        device=study.torch.device("cpu"),
        batch_points="auto",
        min_box_side_frac=0.0025,
        adaptive_topology_only=False,
        cache={},
    )
    study._stamp_stage_provenance(paths, "train", training_direct, cache={})
    study._write_json(
        paths.bounds,
        {
            "lower": [-1.0],
            "upper": [1.0],
            "epsilon_frac": study.BOUNDS_EPSILON_FRAC,
        },
    )
    bounds_direct = study._stage_direct_inputs(
        paths,
        inputs,
        "bounds",
        device=study.torch.device("cpu"),
        batch_points="auto",
        min_box_side_frac=0.0025,
        adaptive_topology_only=False,
        cache={},
    )
    study._stamp_stage_provenance(paths, "bounds", bounds_direct, cache={})

    assert _stage_reusable(paths, inputs, "bounds")
    (paths.models / "autoencoder.pt").write_bytes(b"retrained-checkpoint")
    assert not _stage_reusable(paths, inputs, "bounds")


def test_upstream_recompute_marks_every_descendant_non_reusable(tmp_path):
    paths = study.DimensionPaths(output_root=tmp_path, dimension=3)
    for stage in study.STAGE_ORDER:
        study._write_json(
            paths.stage_marker(stage),
            {"stage": stage, "schema_version": study.STAGE_PROVENANCE_SCHEMA_VERSION},
        )

    study._invalidate_stage_and_descendants(paths, "uniform")

    assert "invalidated" not in json.loads(paths.stage_marker("train").read_text())
    assert "invalidated" not in json.loads(
        paths.stage_marker("precompute-coarse").read_text()
    )
    for stage in ("uniform", "precompute-fine", "adaptive", "stats", "render"):
        marker = json.loads(paths.stage_marker(stage).read_text())
        assert marker["invalidated"]["by_stage"] == "uniform"


def test_matching_legacy_training_artifacts_are_adopted_without_retraining(tmp_path):
    paths = study.DimensionPaths(output_root=tmp_path, dimension=1)
    paths.run.mkdir(parents=True)
    inputs = _exact_inputs(tmp_path)
    _write_minimal_training_artifacts(paths)
    summary = {
        "training_method": "marcio_full_batch",
        "seed": study.SEED,
        "device": "cpu",
        "epochs_completed": study.EPOCHS,
        "checkpoint_epoch": study.EPOCHS,
        "checkpoint_selection": "final_epoch",
        "objective": "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)",
        "optimizer": {"name": "Adam", "learning_rate": study.LEARNING_RATE},
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": 100,
            "threshold": 1e-4,
            "min_lr": 1e-6,
        },
        "arch": study.reference_architecture(1).model_dump(),
        "data": {
            "n_pairs": study.TRAINING_ROWS,
            "high_dims": study.HIGH_DIMENSION,
            "dtype": "float32",
            "full_batch": True,
            "sha256": study.TRAIN_DATA_SHA256,
        },
    }
    study._write_json(paths.run / "training_summary.json", summary)

    assert _stage_reusable(paths, inputs, "train")
    marker = json.loads(paths.stage_marker("train").read_text())
    assert marker["provenance"]["legacy_artifacts_adopted_after_validation"] is True
    assert marker["provenance"]["outputs"]["fingerprint"]


def test_obsolete_minimal_lca_roa_archive_is_quarantined(tmp_path):
    paths = study.DimensionPaths(output_root=tmp_path, dimension=1)
    paths.uniform.mkdir(parents=True)
    obsolete = paths.uniform / "regions_of_attraction_exact.npz"
    obsolete.write_bytes(b"old-minimal-lca-labels")

    quarantined = study._quarantine_legacy_roa_artifact(paths)

    assert quarantined == paths.uniform / "regions_of_attraction_legacy_minimal_lca.npz"
    assert not obsolete.exists()
    assert quarantined.read_bytes() == b"old-minimal-lca-labels"
    manifest = json.loads(
        (
            paths.uniform / "regions_of_attraction_legacy_minimal_lca.json"
        ).read_text()
    )
    assert manifest["status"] == "legacy_not_used_by_dimension_study"
    assert "not the strict" in manifest["legacy_method"]
    assert manifest["authoritative_replacement"].startswith(
        "reference_singleton_reachability_queries.npz"
    )


def test_three_dimensional_render_manifest_includes_cubical_and_all_projections(
    tmp_path,
):
    paths = study.DimensionPaths(output_root=tmp_path, dimension=3)

    outputs = study._stage_output_paths(paths, "render")

    for i, j in ((1, 2), (1, 3), (2, 3)):
        for extension in ("pdf", "png"):
            assert f"morse_sets_z{i}_z{j}_{extension}" in outputs
    assert outputs["morse_sets_cubical_3d_pdf"].name == "morse_sets_cubical_3d.pdf"
    assert outputs["morse_sets_cubical_3d_png"].name == "morse_sets_cubical_3d.png"


def test_three_dimensional_render_stage_invokes_and_records_cubical_view(
    tmp_path,
    monkeypatch,
):
    paths = study.DimensionPaths(output_root=tmp_path, dimension=3)
    paths.adaptive.mkdir(parents=True)
    (paths.adaptive / "morse_graph").write_text("digraph {}\n")
    (paths.adaptive / "morse_sets").write_text("0,0,0,1,1,1,0\n")
    study._write_json(
        paths.bounds,
        {
            "lower": [0.0, 0.0, 0.0],
            "upper": [1.0, 1.0, 1.0],
            "epsilon_frac": study.BOUNDS_EPSILON_FRAC,
        },
    )
    graph_paths = [
        paths.adaptive / "morse_graph.pdf",
        paths.adaptive / "morse_graph.png",
    ]
    cubical_paths = [
        paths.adaptive / "morse_sets_cubical_3d.pdf",
        paths.adaptive / "morse_sets_cubical_3d.png",
    ]
    called = {}
    monkeypatch.setattr(
        study,
        "render_morse_graph_from_dot",
        lambda *_args, **_kwargs: graph_paths,
    )
    monkeypatch.setattr(
        study,
        "render_morse_set_projections_from_csv",
        lambda *_args, **_kwargs: {
            (0, 1): [paths.adaptive / "morse_sets_z1_z2.pdf"]
        },
    )

    def fake_cubical(*_args, **kwargs):
        called["basename"] = kwargs["basename"]
        called["formats"] = kwargs["formats"]
        return cubical_paths

    monkeypatch.setattr(
        study,
        "render_morse_sets_3d_cubical_from_csv",
        fake_cubical,
    )

    study._run_render(paths, min_box_side_frac=0.0025)

    assert called == {
        "basename": "morse_sets_cubical_3d",
        "formats": ("pdf", "png"),
    }
    marker = json.loads(paths.stage_marker("render").read_text())
    assert marker["morse_sets_cubical_3d_outputs"] == [
        str(path) for path in cubical_paths
    ]
