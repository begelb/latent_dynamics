"""Focused synthetic-fixture tests for the Leslie3D T=40 sweep aggregator."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analyze_leslie3d_example2_t40_sweep.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_leslie3d_example2_t40_sweep", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZER = _load_module()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pair_csv(path: Path, n_pairs: int, *, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["x0,x1,x2,y0,y1,y2"]
    rows.extend(f"{i + offset},0,0,{i + offset + 1},0,0" for i in range(n_pairs))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _training_summary(seed: int) -> dict:
    def split(offset: float) -> dict:
        return {
            "loss_reconstruction": {
                "final": 0.01 + offset,
                "best_epoch_value": 0.009 + offset,
                "min": 0.008 + offset,
                "mean": 0.02 + offset,
                "max": 0.1 + offset,
            },
            "loss_prediction": {
                "final": 0.02 + offset,
                "best_epoch_value": 0.019 + offset,
                "min": 0.018 + offset,
                "mean": 0.03 + offset,
                "max": 0.2 + offset,
            },
            "loss_semiconjugacy": {
                "final": 0.03 + offset,
                "best_epoch_value": 0.029 + offset,
                "min": 0.028 + offset,
                "mean": 0.04 + offset,
                "max": 0.3 + offset,
            },
            "loss_total": {
                "final": 0.4 + offset,
                "best_epoch_value": 0.39 + offset,
                "min": 0.38 + offset,
                "mean": 0.5 + offset,
                "max": 1.0 + offset,
            },
        }

    return {
        "best_epoch": 7,
        "n_epochs_run": 12,
        "train_duration_seconds": 60.0 + seed,
        "train": split(seed / 1000),
        "val": split(0.1 + seed / 1000),
    }


def _write_complete_fixture(tmp_path: Path) -> tuple[Path, Path]:
    sweep_root = tmp_path / "output" / "leslie3d_example2_seedsweep_t40"
    data_root = tmp_path / "data" / "leslie3d_example2_seedsweep_t40"
    for dataset_id in range(1, 6):
        dataset_dir = data_root / f"dataset_{dataset_id}"
        train_csv = dataset_dir / "train.csv"
        validation_csv = dataset_dir / "val.csv"
        _write_pair_csv(train_csv, 40, offset=dataset_id * 100)
        _write_pair_csv(validation_csv, 40)
        common = {
            "system": "LeslieModel3D",
            "dimension": 3,
            "n_samples": 1,
            "n_iterations": 40,
            "skip_initial_steps": 0,
            "sampling_method": "uniform",
        }
        _write_json(
            dataset_dir / "train_metadata.json",
            {**common, "sampling_seed": dataset_id, "role": "train"},
        )
        _write_json(
            dataset_dir / "val_metadata.json",
            {**common, "sampling_seed": 9999, "role": "val"},
        )

        for model_seed in range(3):
            cell = sweep_root / f"dataset_{dataset_id}" / f"seed_{model_seed}"
            graph = """digraph {
0 [label="0 : (x^4-1, 0, 0)"];
1 [label="1 : (x^4-1, 0, 0)"];
2 [label="2 : (0, x-1, 0)"];
2 -> 0;
2 -> 1;
}
"""
            (cell / "MG").mkdir(parents=True, exist_ok=True)
            (cell / "MG" / "morse_graph").write_text(graph, encoding="utf-8")
            (cell / "MG" / "morse_sets").write_text(
                "0,0,1,1,0\n1,0,2,1,0\n3,0,4,1,1\n5,0,6,1,2\n",
                encoding="utf-8",
            )
            _write_json(
                cell / "metrics.json",
                {
                    "minimal_morse_labels": [0, 1],
                    "minimal_morse_sets": {
                        "0": {
                            "n_boxes": 2,
                            "tau_bar": 0.1,
                            "n_semiconjugacy_samples": 5,
                            "max_semiconjugacy_error": 0.01,
                            "is_spurious_attractor": False,
                        },
                        "1": {
                            "n_boxes": 1,
                            "tau_bar": 0.1,
                            "n_semiconjugacy_samples": 5,
                            "max_semiconjugacy_error": 0.02,
                            "is_spurious_attractor": False,
                        },
                    },
                    "morse_graph_consistency": {
                        "n_morse_sets": 3,
                        "n_minimal_attractors": 2,
                        "consistent": True,
                    },
                },
            )
            _write_json(cell / "training_summary.json", _training_summary(model_seed))
            _write_json(
                cell / "diagnose.json",
                {
                    "diagnostic": "ok",
                    "hard_flags": {
                        "encoder_collapsed": False,
                        "latent_map_overcontracted": False,
                    },
                    "encoder": {"max_extent_relative": 0.8},
                    "latent_map": {"contraction_ratio": 0.9},
                    "bounds": {"source": "encoded_data"},
                },
            )
            (cell / "mg_params_log.txt").write_text(
                "Lower bounds: [-1.0, -1.0]\n"
                "Upper bounds: [1.0, 1.0]\n"
                "subdiv_init: 25\n"
                "subdiv_min: 28\n"
                "subdiv_max: 29\n"
                "adaptive_precompute_subdiv: init\n"
                "box_map_backend: adaptive_precomputed\n"
                "duration_minutes: 1.5\n",
                encoding="utf-8",
            )
            _write_json(
                cell / "run_manifest.json",
                {
                    "created_at_utc": "2026-08-03T00:00:00+00:00",
                    "latentdynamics_version": "0.1.0",
                    "python": "3.12",
                    "platform": "test",
                    "torch": {"version": "test"},
                    "cmgdb_version": "test",
                    "requested_stages": ["all"],
                    "cell": {"seed": model_seed},
                    "config_hash": f"config-{dataset_id}-{model_seed}",
                    "config": {
                        "data": {
                            "n_samples_train": 1,
                            "n_samples_val": 1,
                            "n_iterations": 40,
                            "skip": 0,
                            "train_seed": dataset_id,
                            "val_seed": 9999,
                        },
                        "cmgdb": {
                            "box_map_backend": "adaptive_precomputed",
                        },
                    },
                    "artifacts": {"train_csv_sha256": _sha256(train_csv)},
                },
            )
            (cell / "models").mkdir(parents=True, exist_ok=True)
            (cell / "models" / "autoencoder.pt").write_bytes(
                f"model-{dataset_id}-{model_seed}".encode()
            )
            _write_json(cell / "models" / "autoencoder.json", {"version": 1})
    return sweep_root, data_root


def test_complete_sweep_writes_detailed_and_aggregate_reports(tmp_path: Path) -> None:
    sweep_root, data_root = _write_complete_fixture(tmp_path)

    outputs = ANALYZER.analyze_sweep(sweep_root=sweep_root, data_root=data_root)

    assert set(outputs) == {"cells_csv", "cells_json", "aggregate_summary"}
    with outputs["cells_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert rows[0]["trajectory_length_T"] == "40"
    assert rows[0]["train_transition_pairs_observed"] == "40"
    assert rows[0]["n_graph_sinks"] == "2"
    assert rows[0]["sink_conley_indices"] == '[["x^4-1","0","0"],["x^4-1","0","0"]]'
    assert rows[0]["n_attractor_type_nodes"] == "2"
    assert rows[0]["attractor_type_labels"] == "[0,1]"
    assert rows[0]["attractor_type_conley_indices"] == (
        '[["x^4-1","0","0"],["x^4-1","0","0"]]'
    )
    assert rows[0]["marcio_style_success"] == "True"
    assert rows[0]["minimal_node_success"] == "True"
    assert rows[0]["morse_boxes_by_label"] == '{"0":2,"1":1,"2":1}'
    assert rows[0]["exact_conley_success"] == "True"
    assert rows[0]["box_map_backend_is_explicit"] == "True"
    assert rows[0]["precompute_lattice_dimension"] == "2"
    assert rows[0]["precompute_subdiv_role"] == "init"
    assert rows[0]["precompute_subdiv"] == "25"
    assert rows[0]["precompute_axis_depth_M"] == "13"
    assert rows[0]["precompute_axis_depths"] == "[13,12]"
    assert rows[0]["precompute_corners_per_axis"] == "[8193,4097]"
    assert rows[0]["precompute_table_points"] == "33566721"
    assert rows[0]["model_sha256"]

    details = json.loads(outputs["cells_json"].read_text())
    assert details["provisional"] is False
    assert len(details["datasets"]) == 5
    assert len(details["cells"]) == 15
    first = details["cells"][0]
    assert details["primary_success_criterion"]["name"] == (
        "exactly_two_nonzero_degree0_nodes"
    )
    assert first["marcio_style_success"] is True
    assert first["minimal_node_success"] is True
    assert first["morse_graph"]["n_attractor_type_nodes"] == 2
    assert first["morse_graph"]["nodes"][0]["n_boxes"] == 2
    assert first["metrics"]["all_minimal_tolerance_pass"] is True
    assert first["metrics"]["tolerance_status"] == "pass"
    assert first["durations"]["cmgdb_seconds"] == 90.0
    assert first["cmgdb"]["adaptive_precompute_lattice"] == {
        "axis_depth_M": 13,
        "axis_depths": [13, 12],
        "axis_depth_formula": "ceil((precompute_subdiv - axis_index) / dimension)",
        "backend": "adaptive_precomputed",
        "cells_per_axis": [8192, 4096],
        "corners_per_axis": [8193, 4097],
        "dimension": 2,
        "lattice_shape": [8193, 4097],
        "precompute_subdiv": 25,
        "precompute_subdiv_role": "init",
        "subdiv_max": 29,
        "table_points": 33_566_721,
        "table_points_formula": "product(2^axis_depth + 1)",
    }
    assert first["verification_passed"] is True

    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    assert aggregate["inventory"]["n_verified_cells"] == 15
    assert aggregate["exact_conley_success"]["n_successes"] == 15
    assert aggregate["exact_conley_success"]["rate_over_expected_15_cells"] == 1.0
    assert aggregate["marcio_style_success"]["n_successes"] == 15
    assert aggregate["marcio_style_success"]["rate_over_expected_15_cells"] == 1.0
    assert aggregate["minimal_node_success"]["n_successes"] == 15
    assert aggregate["minimal_node_success"]["rate_over_expected_15_cells"] == 1.0
    assert aggregate["primary_success_criterion"]["uses_all_morse_nodes"] is True
    assert aggregate["topology"]["n_distinct_signatures"] == 1
    assert aggregate["losses"]["validation_total_final"]["count"] == 15
    assert aggregate["durations_seconds"]["cmgdb"]["mean"] == 90.0
    assert aggregate["dataset_design"]["verification_passed"] is True
    assert aggregate["dataset_design"]["training"]["n_distinct_csv_sha256"] == 5
    assert aggregate["dataset_design"]["validation"]["n_distinct_csv_sha256"] == 1
    assert aggregate["dataset_design"]["validation"]["n_distinct_seeds"] == 1
    assert aggregate["dataset_design"]["validation"]["shared_seed"] == 9999
    assert aggregate["dataset_design"]["validation"]["shared_hash_and_seed"] is True
    lattice = aggregate["adaptive_precompute_lattice"]
    assert lattice["n_cells_recorded"] == 15
    assert lattice["n_distinct_lattices"] == 1
    assert lattice["lattices"][0]["corners_per_axis"] == [8193, 4097]
    assert lattice["lattices"][0]["table_points"] == 33_566_721
    assert lattice["lattices"][0]["n_cells"] == 15
    assert aggregate["success_criterion"]["required_sink_conley_index"] == [
        "x^4-1",
        "0",
        "0",
    ]


def test_strict_mode_rejects_missing_cell_but_allow_incomplete_inventory_works(
    tmp_path: Path,
) -> None:
    sweep_root, data_root = _write_complete_fixture(tmp_path)
    (sweep_root / "dataset_5" / "seed_2" / "metrics.json").unlink()

    with pytest.raises(ANALYZER.SweepValidationError, match="15 verified cells"):
        ANALYZER.analyze_sweep(sweep_root=sweep_root, data_root=data_root)
    assert not (sweep_root / "analysis").exists()

    outputs = ANALYZER.analyze_sweep(
        sweep_root=sweep_root,
        data_root=data_root,
        allow_incomplete=True,
    )
    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    details = json.loads(outputs["cells_json"].read_text())
    assert aggregate["provisional"] is True
    assert aggregate["inventory"]["n_complete_cells"] == 14
    assert aggregate["inventory"]["n_verified_cells"] == 14
    missing = details["cells"][-1]
    assert missing["complete"] is False
    assert missing["verification_passed"] is False
    assert {error["code"] for error in missing["errors"]} == {"missing_cell_artifact"}


def test_strict_mode_accepts_read_only_metrics_replay_overlay(tmp_path: Path) -> None:
    sweep_root, data_root = _write_complete_fixture(tmp_path)
    metrics_root = tmp_path / "metrics_replay"
    for dataset_id in range(1, 6):
        for model_seed in range(3):
            source = (
                sweep_root
                / f"dataset_{dataset_id}"
                / f"seed_{model_seed}"
                / "metrics.json"
            )
            payload = json.loads(source.read_text(encoding="utf-8"))
            replay = (
                metrics_root
                / f"{sweep_root.name}_dataset_{dataset_id}"
                / f"seed_{model_seed}"
                / "metrics.json"
            )
            _write_json(replay, payload)
            source.unlink()

    outputs = ANALYZER.analyze_sweep(
        sweep_root=sweep_root,
        data_root=data_root,
        metrics_root=metrics_root,
    )

    details = json.loads(outputs["cells_json"].read_text())
    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    assert details["metrics_are_derived_replay"] is True
    assert aggregate["metrics_are_derived_replay"] is True
    assert aggregate["inventory"]["n_verified_cells"] == 15
    assert not list(sweep_root.glob("dataset_*/seed_*/metrics.json"))


def test_zero_sample_tolerance_is_inconclusive_not_pass(tmp_path: Path) -> None:
    sweep_root, data_root = _write_complete_fixture(tmp_path)
    metrics_path = sweep_root / "dataset_1" / "seed_0" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["minimal_morse_sets"]["0"].update(
        {"n_semiconjugacy_samples": 0, "is_spurious_attractor": False}
    )
    _write_json(metrics_path, metrics)

    outputs = ANALYZER.analyze_sweep(sweep_root=sweep_root, data_root=data_root)

    details = json.loads(outputs["cells_json"].read_text())
    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    first = details["cells"][0]
    assert first["metrics"]["tolerance_status"] == "inconclusive"
    assert first["metrics"]["all_minimal_tolerance_pass"] is None
    assert aggregate["tolerance"]["status_counts"] == {
        "inconclusive": 1,
        "pass": 14,
    }


def test_strict_mode_requires_five_distinct_training_csv_hashes(tmp_path: Path) -> None:
    sweep_root, data_root = _write_complete_fixture(tmp_path)
    source = data_root / "dataset_1" / "train.csv"
    duplicate = data_root / "dataset_5" / "train.csv"
    duplicate.write_bytes(source.read_bytes())
    duplicate_hash = _sha256(duplicate)
    for model_seed in range(3):
        manifest_path = sweep_root / "dataset_5" / f"seed_{model_seed}" / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["train_csv_sha256"] = duplicate_hash
        _write_json(manifest_path, manifest)

    with pytest.raises(
        ANALYZER.SweepValidationError, match="training_datasets_not_distinct"
    ):
        ANALYZER.analyze_sweep(sweep_root=sweep_root, data_root=data_root)

    outputs = ANALYZER.analyze_sweep(
        sweep_root=sweep_root,
        data_root=data_root,
        allow_incomplete=True,
    )
    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    design = aggregate["dataset_design"]
    assert aggregate["provisional"] is True
    assert aggregate["inventory"]["n_verified_cells"] == 15
    assert design["verification_passed"] is False
    assert design["training"]["n_distinct_csv_sha256"] == 4
    assert design["training"]["all_training_csvs_distinct"] is False
    assert {error["code"] for error in design["errors"]} == {
        "training_datasets_not_distinct"
    }


def test_strict_mode_requires_one_shared_validation_hash_and_seed(tmp_path: Path) -> None:
    sweep_root, data_root = _write_complete_fixture(tmp_path)
    dataset_dir = data_root / "dataset_5"
    _write_pair_csv(dataset_dir / "val.csv", 40, offset=50_000)
    metadata_path = dataset_dir / "val_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["sampling_seed"] = 8888
    _write_json(metadata_path, metadata)
    for model_seed in range(3):
        manifest_path = sweep_root / "dataset_5" / f"seed_{model_seed}" / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config"]["data"]["val_seed"] = 8888
        _write_json(manifest_path, manifest)

    with pytest.raises(
        ANALYZER.SweepValidationError, match="validation_holdout_hash_not_shared"
    ):
        ANALYZER.analyze_sweep(sweep_root=sweep_root, data_root=data_root)

    outputs = ANALYZER.analyze_sweep(
        sweep_root=sweep_root,
        data_root=data_root,
        allow_incomplete=True,
    )
    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    validation = aggregate["dataset_design"]["validation"]
    assert aggregate["inventory"]["n_verified_cells"] == 15
    assert validation["n_distinct_csv_sha256"] == 2
    assert validation["n_distinct_seeds"] == 2
    assert validation["shared_csv_sha256"] is None
    assert validation["shared_seed"] is None
    assert validation["shared_hash_and_seed"] is False
    assert {error["code"] for error in aggregate["dataset_design"]["errors"]} == {
        "validation_holdout_hash_not_shared",
        "validation_holdout_seed_not_shared",
        "validation_holdout_hash_seed_pair_not_shared",
    }


def test_alternate_t25_40k_10k_design_is_enforced_in_metadata_and_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep_root, data_root = _write_complete_fixture(tmp_path)
    for dataset_id in range(1, 6):
        dataset_dir = data_root / f"dataset_{dataset_id}"
        for filename, n_samples in (("train_metadata.json", 40_000), ("val_metadata.json", 10_000)):
            metadata_path = dataset_dir / filename
            metadata = json.loads(metadata_path.read_text())
            metadata["n_samples"] = n_samples
            metadata["n_iterations"] = 25
            _write_json(metadata_path, metadata)
        for model_seed in range(3):
            manifest_path = (
                sweep_root / f"dataset_{dataset_id}" / f"seed_{model_seed}" / "run_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text())
            manifest["config"]["data"].update(
                {
                    "n_samples_train": 40_000,
                    "n_samples_val": 10_000,
                    "n_iterations": 25,
                }
            )
            _write_json(manifest_path, manifest)

    real_scan = ANALYZER._scan_dataset_csv

    def design_sized_scan(path: Path) -> dict:
        scanned = real_scan(path)
        scanned["transition_pairs_observed"] = (
            1_000_000 if path.name == "train.csv" else 250_000
        )
        return scanned

    monkeypatch.setattr(ANALYZER, "_scan_dataset_csv", design_sized_scan)
    result = ANALYZER.main(
        [
            "--sweep-root",
            str(sweep_root),
            "--data-root",
            str(data_root),
            "--expected-t",
            "25",
            "--expected-train-initial-conditions",
            "40000",
            "--expected-validation-initial-conditions",
            "10000",
        ]
    )
    assert result == 0
    capsys.readouterr()
    aggregate = json.loads((sweep_root / "analysis" / "aggregate_summary.json").read_text())
    assert aggregate["expected_design"]["trajectory_length_T"] == 25
    assert aggregate["expected_design"]["train_initial_conditions"] == 40_000
    assert aggregate["expected_design"]["validation_initial_conditions"] == 10_000
    details = json.loads((sweep_root / "analysis" / "cells.json").read_text())
    assert details["expected_design"]["train_initial_conditions"] == 40_000
    assert details["expected_design"]["validation_initial_conditions"] == 10_000
    assert details["cells"][0]["dataset"]["train"]["initial_conditions"] == 40_000
    assert details["cells"][0]["dataset"]["train"]["transition_pairs_expected"] == 1_000_000
    assert details["cells"][0]["dataset"]["validation"]["initial_conditions"] == 10_000
    assert (
        details["cells"][0]["dataset"]["validation"]["transition_pairs_expected"]
        == 250_000
    )

    metadata_path = data_root / "dataset_5" / "train_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["n_samples"] = 39_999
    _write_json(metadata_path, metadata)
    with pytest.raises(ANALYZER.SweepValidationError, match="unexpected_initial_condition_count"):
        ANALYZER.analyze_sweep(
            sweep_root=sweep_root,
            data_root=data_root,
            expected_t=25,
            expected_train_initial_conditions=40_000,
            expected_validation_initial_conditions=10_000,
        )
    metadata["n_samples"] = 40_000
    _write_json(metadata_path, metadata)

    manifest_path = sweep_root / "dataset_5" / "seed_2" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["data"]["n_samples_val"] = 9_999
    _write_json(manifest_path, manifest)
    with pytest.raises(ANALYZER.SweepValidationError, match="manifest_mismatch"):
        ANALYZER.analyze_sweep(
            sweep_root=sweep_root,
            data_root=data_root,
            expected_t=25,
            expected_train_initial_conditions=40_000,
            expected_validation_initial_conditions=10_000,
        )


def test_exact_success_and_tolerance_are_reported_as_distinct_outcomes(tmp_path: Path) -> None:
    sweep_root, data_root = _write_complete_fixture(tmp_path)
    target = sweep_root / "dataset_2" / "seed_1"
    graph_path = target / "MG" / "morse_graph"
    graph_path.write_text(
        graph_path.read_text().replace(
            '1 [label="1 : (x^4-1, 0, 0)"]',
            '1 [label="1 : (x^3-1, 0, 0)"]',
        ),
        encoding="utf-8",
    )
    metrics_path = target / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["minimal_morse_sets"]["0"]["is_spurious_attractor"] = True
    _write_json(metrics_path, metrics)

    outputs = ANALYZER.analyze_sweep(sweep_root=sweep_root, data_root=data_root)

    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    assert aggregate["inventory"]["n_verified_cells"] == 15
    assert aggregate["exact_conley_success"]["n_successes"] == 14
    assert aggregate["tolerance"]["n_cells_all_minimal_pass"] == 14
    assert aggregate["tolerance"]["total_failed_minimal_sets"] == 1
    details = json.loads(outputs["cells_json"].read_text())
    changed = next(
        cell
        for cell in details["cells"]
        if cell["dataset_id"] == 2 and cell["model_seed"] == 1
    )
    assert changed["exact_conley_success"] is False
    assert changed["metrics"]["all_minimal_tolerance_pass"] is False
    assert changed["verification_passed"] is True


def test_minimal_attractor_count_excludes_trivial_index_graph_sink(tmp_path: Path) -> None:
    sweep_root, data_root = _write_complete_fixture(tmp_path)
    target = sweep_root / "dataset_2" / "seed_2"
    graph_path = target / "MG" / "morse_graph"
    graph_path.write_text(
        graph_path.read_text().replace(
            '1 [label="1 : (x^4-1, 0, 0)"]',
            '1 [label="1 : (0, 0, 0)"]',
        ),
        encoding="utf-8",
    )
    metrics_path = target / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["morse_graph_consistency"].update(
        {"n_minimal_attractors": 1, "n_trivial_index": 1}
    )
    _write_json(metrics_path, metrics)

    outputs = ANALYZER.analyze_sweep(sweep_root=sweep_root, data_root=data_root)

    details = json.loads(outputs["cells_json"].read_text())
    changed = next(
        cell
        for cell in details["cells"]
        if cell["dataset_id"] == 2 and cell["model_seed"] == 2
    )
    assert changed["morse_graph"]["sinks"] == [0, 1]
    assert changed["sink_conley_indices"] == [["x^4-1", "0", "0"], ["0", "0", "0"]]
    assert changed["metrics"]["morse_graph_consistency"]["n_minimal_attractors"] == 1
    assert changed["exact_conley_success"] is False
    assert changed["verification_passed"] is True
