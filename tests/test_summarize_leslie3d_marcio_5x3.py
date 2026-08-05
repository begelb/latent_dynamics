from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib
import pytest
from PIL import Image


def _load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_leslie3d_marcio_5x3.py"
    spec = importlib.util.spec_from_file_location("summarize_leslie3d_marcio_5x3", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = _load_module()


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (["x^4-1", "0", "0"], True),
        (["x-1", "x-1", "0"], True),
        (["0", "x-1", "0"], False),
        (["x-1", "0"], False),
        (None, False),
    ],
)
def test_stable_conley_index_uses_nonzero_h0_only(index, expected: bool) -> None:
    assert REPORT._is_stable_conley_index(index) is expected


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (["x-1", "0", "0"], True),
        (["x^4-1", "0", "0"], True),
        (["x^12-1", "0", "0"], True),
        (["x^0-1", "0", "0"], False),
        (["x^4-1", "x-1", "0"], False),
        (["x^4-1", "0"], False),
        (None, False),
    ],
)
def test_periodic_bistability_index_requires_pure_positive_period(
    index, expected: bool
) -> None:
    assert REPORT._is_periodic_bistability_index(index) is expected


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    sweep = tmp_path / "output" / "leslie3d_example2_marcio_5x3_v1"
    data = tmp_path / "data" / "leslie3d_example2_marcio_5x3_v1"
    arch = {
        "high_dims": 3,
        "low_dims": 2,
        "encoder": {"hidden_shapes": [64, 64], "activation": "relu", "out_activation": "tanh"},
        "latent_map": {"hidden_shapes": [64, 64], "activation": "relu", "out_activation": "tanh"},
        "decoder": {"hidden_shapes": [64, 64], "activation": "relu", "out_activation": "sigmoid"},
    }
    for dataset_id in range(1, 6):
        dataset = data / f"dataset_{dataset_id:02d}"
        dataset.mkdir(parents=True)
        (dataset / "train.csv").write_text(
            f"x0,x1,x2,y0,y1,y2\n{dataset_id},0,0,{dataset_id + 1},0,0\n",
            encoding="utf-8",
        )
        (dataset / "val.csv").write_text(
            "x0,x1,x2,y0,y1,y2\n0,0,0,1,0,0\n", encoding="utf-8"
        )
        _write_json(dataset / "train_metadata.json", {"sampling_seed": 100 + dataset_id, "role": "train"})
        _write_json(dataset / "val_metadata.json", {"sampling_seed": 9999, "role": "val"})
        train_hash = REPORT._sha256(dataset / "train.csv")

        for seed in range(3):
            cell = sweep / f"dataset_{dataset_id:02d}" / f"seed_{seed}"
            (cell / "models").mkdir(parents=True)
            (cell / "logs").mkdir()
            (cell / "MG").mkdir()
            (cell / "models" / "autoencoder.pt").write_bytes(
                f"checkpoint-{dataset_id}-{seed}".encode()
            )
            _write_json(cell / "models" / "autoencoder.json", {"version": 1, "arch": arch})
            history = {
                "schema_version": 1,
                "training_method": "marcio_full_batch",
                "train": {
                    "loss_reconstruction": [0.2, 0.01],
                    "loss_prediction": [0.3, 0.02],
                    "loss_total": [0.5, 0.03],
                    "learning_rate": [0.001, 0.001],
                },
            }
            _write_json(cell / "logs" / "history.json", history)
            _write_json(
                cell / "training_summary.json",
                {
                    "schema_version": 1,
                    "training_method": "marcio_full_batch",
                    "objective": REPORT.MARCIO_OBJECTIVE,
                    "seed": seed,
                    "data": {"n_pairs": 1, "high_dims": 3, "dtype": "float32", "full_batch": True},
                    "arch": arch,
                    "epochs_requested": 2,
                    "epochs_completed": 2,
                    "checkpoint_epoch": 2,
                    "checkpoint_selection": "final_epoch",
                    "validation_used": False,
                    "early_stopping_used": False,
                    "best_weight_restoration_used": False,
                    "final_epoch_train": {
                        "loss_reconstruction": 0.01,
                        "loss_prediction": 0.02,
                        "loss_total": 0.03,
                    },
                    "artifacts": {
                        "checkpoint": "models/autoencoder.pt",
                        "checkpoint_metadata": "models/autoencoder.json",
                        "history": "logs/history.json",
                    },
                },
            )
            (cell / "final_losses.txt").write_text(
                "loss_reconstruction: 0.01\nloss_prediction: 0.02\nloss_total: 0.03\nval_loss_total: 0.04\n",
                encoding="utf-8",
            )
            _write_json(
                cell / "diagnose.json",
                {
                    "diagnostic": "ok",
                    "hard_flags": {"encoder_collapsed": False, "latent_map_overcontracted": False},
                    "bounds": {"source": "encoded_data"},
                },
            )
            (cell / "MG" / "morse_graph").write_text(
                """digraph {
0 [label="0 : (x^{4} - 1, 0, 0)"];
1 [label="1 : (x^4 - 1, 0, 0)"];
2 [label="2 : (0, x-1, 0)"];
2 -> 0;
2 -> 1;
}
""",
                encoding="utf-8",
            )
            (cell / "MG" / "morse_sets").write_text(
                "0,0,1,1,0\n1,0,2,1,0\n2,0,3,1,1\n3,0,4,1,2\n",
                encoding="utf-8",
            )
            for filename, color in (("morse_graph.png", "white"), ("morse_sets.png", "lightblue")):
                Image.new("RGB", (20, 12), color).save(cell / "MG" / filename)
            for filename in ("morse_graph.pdf", "morse_sets.pdf"):
                (cell / "MG" / filename).write_bytes(b"%PDF-1.4\n%%EOF\n")
            (cell / "mg_params_log.txt").write_text(
                "Lower bounds: [-1.0, -1.0]\n"
                "Upper bounds: [1.0, 1.0]\n"
                "subdiv_init: 25\nsubdiv_min: 28\nsubdiv_max: 29\n"
                "padding: True\nbox_map_backend: adaptive_precomputed\n"
                "bounds_source: encoded_train_pairs\nduration_minutes: 1.5\n",
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
                },
            )
            _write_json(
                cell / "run_manifest.json",
                {
                    "cell": {"seed": seed},
                    "config_hash": f"config-{dataset_id}-{seed}",
                    "config": {
                        "data": {"train_seed": 100 + dataset_id, "val_seed": 9999},
                        "cmgdb": {"subdiv_init": 25, "subdiv_min": 28, "subdiv_max": 29},
                    },
                    "artifacts": {"train_csv_sha256": train_hash},
                },
            )
    return sweep, data


def _pdf_page_count(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))


def test_complete_report_writes_all_outputs_and_six_page_pdf(tmp_path: Path) -> None:
    sweep, data = _write_fixture(tmp_path)

    outputs = REPORT.build_summary(sweep_root=sweep, data_root=data)

    assert set(outputs) == {
        "cells_csv",
        "cells_json",
        "aggregate_summary",
        "summary_markdown",
        "summary_pdf",
    }
    assert all(path.is_file() for path in outputs.values())
    with outputs["cells_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert rows[0]["bistability_pass"] == "True"
    assert rows[0]["marcio_style_success"] == "True"
    assert rows[0]["minimal_node_success"] == "True"
    assert rows[0]["periodic_bistability_success"] == "True"
    assert rows[0]["exact_conley_success"] == "True"
    assert rows[0]["tolerance_status"] == "pass"
    assert rows[0]["n_stable_conley_index_nodes"] == "2"
    assert rows[0]["stable_index_labels"] == "[0,1]"
    assert rows[0]["stable_conley_indices"] == (
        '[["x^{4} - 1","0","0"],["x^4 - 1","0","0"]]'
    )
    assert rows[0]["sink_degree0_normalized"] == '["x^4-1","x^4-1"]'
    assert rows[0]["checkpoint_sha256"]
    assert rows[0]["morse_graph_sha256"]
    assert rows[0]["morse_sets_sha256"]
    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    assert aggregate["schema_version"] == 4
    assert aggregate["provisional"] is False
    assert aggregate["inventory"]["n_verified_cells"] == 15
    assert aggregate["bistability"]["n_successes"] == 15
    assert aggregate["marcio_style_success"]["n_successes"] == 15
    assert aggregate["minimal_node_success"]["n_successes"] == 15
    assert aggregate["periodic_bistability_success"]["n_successes"] == 15
    assert aggregate["exact_conley_success"]["n_successes"] == 15
    assert aggregate["success_criterion"] == {
        "conley_index_affects_classification": True,
        "definition": REPORT.SUCCESS_CRITERION["definition"],
        "expected_conley_index_components": 3,
        "graph_edges_affect_classification": False,
        "graph_minimality_affects_classification": False,
        "name": "bistability_exactly_two_stable_conley_index_nodes",
        "required_stable_conley_index_node_count": 2,
        "stable_conley_index_pattern": ["nonzero", "any", "any"],
        "tolerance_affects_classification": False,
    }
    assert aggregate["bistability"]["stable_conley_index_node_count_distribution"] == {
        "2": 15,
    }
    assert aggregate["tolerance"]["status_counts"] == {"pass": 15}
    assert aggregate["training"]["methods"] == {"marcio_full_batch": 15}
    assert aggregate["durations_seconds"]["cmgdb"]["mean"] == 90.0
    assert aggregate["durations_seconds"]["sum_cmgdb"] == 1350.0
    assert aggregate["durations_seconds"]["sum_training"] is None
    assert aggregate["durations_seconds"]["sum_combined"] is None
    assert aggregate["visualization"]["summary_morse_sets"] == {
        "box_scale": "auto",
        "changes_scientific_artifacts": False,
        "display_only": True,
        "min_box_side_frac": 0.0075,
        "source": "raw MG/morse_sets CSV",
    }
    markdown = outputs["summary_markdown"].read_text(encoding="utf-8")
    assert "display-only minimum box side of 0.75%" in markdown
    assert "## Topology criteria" in markdown
    assert "## Main findings" in markdown
    assert "## Run profile and timing" in markdown
    assert "| 2 H0 | 2 min | Periodic | Exact |" in markdown
    assert _pdf_page_count(outputs["summary_pdf"]) == 6
    assert not list((sweep / "summary").glob(".morse_sets_summary_*"))


def test_any_nonzero_h0_conley_index_can_satisfy_bistability(tmp_path: Path) -> None:
    sweep, data = _write_fixture(tmp_path)
    graph = sweep / "dataset_02" / "seed_1" / "MG" / "morse_graph"
    graph.write_text(
        graph.read_text().replace(
            "1 : (x^4 - 1, 0, 0)",
            "1 : (x-1, x-1, 0)",
        ),
        encoding="utf-8",
    )

    outputs = REPORT.build_summary(sweep_root=sweep, data_root=data)

    details = json.loads(outputs["cells_json"].read_text())
    target = next(cell for cell in details["cells"] if cell["dataset_id"] == 2 and cell["model_seed"] == 1)
    assert target["verification_passed"] is True
    assert target["bistability_pass"] is True
    assert target["marcio_style_success"] is True
    assert target["minimal_node_success"] is True
    assert target["periodic_bistability_success"] is False
    assert target["exact_conley_success"] is False
    assert target["stable_conley_index_nodes"][1]["conley_index_normalized"] == [
        "x-1",
        "x-1",
        "0",
    ]
    assert target["cell_status"] == "verified_success"
    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    assert aggregate["provisional"] is False
    assert aggregate["bistability"]["n_successes"] == 15
    assert aggregate["marcio_style_success"]["n_successes"] == 15
    assert aggregate["minimal_node_success"]["n_successes"] == 15
    assert aggregate["periodic_bistability_success"]["n_successes"] == 14
    assert aggregate["exact_conley_success"]["n_successes"] == 14


def test_adaptive_grid_edge_does_not_change_index_bistability(tmp_path: Path) -> None:
    sweep, data = _write_fixture(tmp_path)
    cell = sweep / "dataset_02" / "seed_1"
    graph = cell / "MG" / "morse_graph"
    graph.write_text(
        graph.read_text(encoding="utf-8").replace("2 -> 1;\n}", "2 -> 1;\n1 -> 0;\n}"),
        encoding="utf-8",
    )
    metrics_path = cell / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["minimal_morse_labels"] = [0]
    metrics["minimal_morse_sets"] = {"0": metrics["minimal_morse_sets"]["0"]}
    _write_json(metrics_path, metrics)

    outputs = REPORT.build_summary(sweep_root=sweep, data_root=data)

    details = json.loads(outputs["cells_json"].read_text())
    target = next(
        item for item in details["cells"]
        if item["dataset_id"] == 2 and item["model_seed"] == 1
    )
    assert target["verification_passed"] is True
    assert target["bistability_pass"] is True
    assert target["n_stable_conley_index_nodes"] == 2
    assert target["stable_index_labels"] == [0, 1]
    assert target["morse_graph"]["sinks"] == [0]
    assert target["marcio_style_success"] is True
    assert target["minimal_node_success"] is False
    assert target["periodic_bistability_success"] is False
    assert target["exact_conley_success"] is False
    assert target["cell_status"] == "verified_success"
    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    assert aggregate["bistability"]["n_successes"] == 15
    assert aggregate["minimal_node_success"]["n_successes"] == 14
    assert aggregate["periodic_bistability_success"]["n_successes"] == 14
    assert aggregate["exact_conley_success"]["n_successes"] == 14


def test_only_one_nonzero_h0_node_fails_bistability_without_invalidating_cell(
    tmp_path: Path,
) -> None:
    sweep, data = _write_fixture(tmp_path)
    graph = sweep / "dataset_02" / "seed_1" / "MG" / "morse_graph"
    graph.write_text(
        graph.read_text().replace(
            "1 : (x^4 - 1, 0, 0)",
            "1 : (0, x^4 - 1, 0)",
        ),
        encoding="utf-8",
    )

    outputs = REPORT.build_summary(sweep_root=sweep, data_root=data)

    details = json.loads(outputs["cells_json"].read_text())
    target = next(
        item for item in details["cells"]
        if item["dataset_id"] == 2 and item["model_seed"] == 1
    )
    assert target["verification_passed"] is True
    assert target["bistability_pass"] is False
    assert target["marcio_style_success"] is False
    assert target["minimal_node_success"] is True
    assert target["periodic_bistability_success"] is False
    assert target["exact_conley_success"] is False
    assert target["n_stable_conley_index_nodes"] == 1
    assert target["stable_index_labels"] == [0]
    assert target["cell_status"] == "verified_criterion_failure"
    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    assert aggregate["bistability"]["n_successes"] == 14
    assert aggregate["minimal_node_success"]["n_successes"] == 15
    assert aggregate["periodic_bistability_success"]["n_successes"] == 14
    assert aggregate["exact_conley_success"]["n_successes"] == 14


def test_three_nonzero_h0_nodes_fail_exact_bistability(tmp_path: Path) -> None:
    sweep, data = _write_fixture(tmp_path)
    graph = sweep / "dataset_02" / "seed_1" / "MG" / "morse_graph"
    graph.write_text(
        graph.read_text().replace(
            "2 : (0, x-1, 0)",
            "2 : (x-1, 0, 0)",
        ),
        encoding="utf-8",
    )

    outputs = REPORT.build_summary(sweep_root=sweep, data_root=data)

    details = json.loads(outputs["cells_json"].read_text())
    target = next(
        item for item in details["cells"]
        if item["dataset_id"] == 2 and item["model_seed"] == 1
    )
    assert target["verification_passed"] is True
    assert target["n_stable_conley_index_nodes"] == 3
    assert target["stable_index_labels"] == [0, 1, 2]
    assert target["bistability_pass"] is False
    assert target["marcio_style_success"] is False
    assert target["minimal_node_success"] is True
    assert target["periodic_bistability_success"] is True
    assert target["exact_conley_success"] is True


def test_tolerance_failure_does_not_change_bistability_pass(tmp_path: Path) -> None:
    sweep, data = _write_fixture(tmp_path)
    metrics_path = sweep / "dataset_03" / "seed_2" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for item in metrics["minimal_morse_sets"].values():
        item["is_spurious_attractor"] = True
        item["max_semiconjugacy_error"] = 0.2
    _write_json(metrics_path, metrics)

    outputs = REPORT.build_summary(sweep_root=sweep, data_root=data)

    details = json.loads(outputs["cells_json"].read_text())
    target = next(
        item for item in details["cells"]
        if item["dataset_id"] == 3 and item["model_seed"] == 2
    )
    assert target["metrics"]["all_minimal_tolerance_pass"] is False
    assert target["metrics"]["tolerance_status"] == "fail"
    assert target["bistability_pass"] is True
    assert target["periodic_bistability_success"] is True
    assert target["cell_status"] == "verified_success"


def test_strict_rejects_missing_cell_artifact_and_allow_incomplete_is_honest(tmp_path: Path) -> None:
    sweep, data = _write_fixture(tmp_path)
    (sweep / "dataset_05" / "seed_2" / "metrics.json").unlink()

    with pytest.raises(REPORT.SweepValidationError, match="all 15 cells"):
        REPORT.build_summary(sweep_root=sweep, data_root=data)
    assert not (sweep / "summary").exists()

    outputs = REPORT.build_summary(sweep_root=sweep, data_root=data, allow_incomplete=True)
    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    assert aggregate["provisional"] is True
    assert aggregate["inventory"]["n_complete_cells"] == 14
    assert aggregate["inventory"]["n_verified_cells"] == 14
    assert aggregate["inventory"]["error_counts_by_code"]["missing_cell_artifact"] == 1
    assert "PROVISIONAL / INCOMPLETE" in outputs["summary_markdown"].read_text()
    assert _pdf_page_count(outputs["summary_pdf"]) == 6


def test_strict_rejects_validation_inclusive_cmgdb_bounds(tmp_path: Path) -> None:
    sweep, data = _write_fixture(tmp_path)
    log_path = sweep / "dataset_03" / "seed_1" / "mg_params_log.txt"
    log_path.write_text(
        log_path.read_text(encoding="utf-8").replace(
            "bounds_source: encoded_train_pairs",
            "bounds_source: encoded_data",
        ),
        encoding="utf-8",
    )

    with pytest.raises(REPORT.SweepValidationError, match="all 15 cells"):
        REPORT.build_summary(sweep_root=sweep, data_root=data)

    outputs = REPORT.build_summary(
        sweep_root=sweep,
        data_root=data,
        allow_incomplete=True,
    )
    aggregate = json.loads(outputs["aggregate_summary"].read_text())
    assert aggregate["inventory"]["n_verified_cells"] == 14
    assert aggregate["inventory"]["error_counts_by_code"] == {
        "invalid_cmgdb_bounds_source": 1,
    }


def test_known_tolerance_failure_dominates_an_unknown_sink() -> None:
    summary, errors = REPORT._metrics_summary(
        {
            "minimal_morse_labels": [0, 1],
            "minimal_morse_sets": {
                "0": {
                    "n_boxes": 2,
                    "tau_bar": 0.01,
                    "n_semiconjugacy_samples": 1,
                    "max_semiconjugacy_error": 0.1,
                    "is_spurious_attractor": True,
                },
                "1": {
                    "n_boxes": 3,
                    "tau_bar": 0.01,
                    "n_semiconjugacy_samples": 0,
                    "max_semiconjugacy_error": None,
                    "is_spurious_attractor": None,
                },
            },
        },
        [0, 1],
        {"0": 2, "1": 3},
    )

    assert errors == []
    assert summary["minimal_sets"]["0"]["tolerance_pass"] is False
    assert summary["minimal_sets"]["1"]["tolerance_pass"] is None
    assert summary["all_minimal_tolerance_pass"] is False
    assert summary["tolerance_status"] == "fail"
    assert summary["n_minimal_tolerance_failures"] == 1


def test_zero_sample_metric_is_inconclusive_even_with_stale_boolean() -> None:
    summary, errors = REPORT._metrics_summary(
        {
            "minimal_morse_labels": [0],
            "minimal_morse_sets": {
                "0": {
                    "n_boxes": 2,
                    "tau_bar": 0.01,
                    "n_semiconjugacy_samples": 0,
                    "max_semiconjugacy_error": 0.0,
                    "is_spurious_attractor": False,
                    "tolerance_pass": True,
                },
            },
        },
        [0],
        {"0": 2},
    )

    assert errors == []
    assert summary["minimal_sets"]["0"]["is_spurious_attractor"] is None
    assert summary["minimal_sets"]["0"]["tolerance_pass"] is None
    assert summary["all_minimal_tolerance_pass"] is None
    assert summary["tolerance_status"] == "inconclusive"


def test_summary_morse_sets_use_display_only_minimum_box_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell_dir = tmp_path / "dataset_01" / "seed_2"
    morse_dir = cell_dir / "MG"
    morse_dir.mkdir(parents=True)
    (morse_dir / "morse_sets").write_text("0,0,0.01,0.01,0\n", encoding="utf-8")
    calls: list[dict] = []

    def fake_render(csv_path: Path, out_dir: Path, **kwargs):
        matplotlib.rcParams["savefig.bbox"] = "tight"
        calls.append({"csv_path": Path(csv_path), "out_dir": Path(out_dir), **kwargs})
        output = Path(out_dir) / "morse_sets_summary.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), "magenta").save(output)
        return [output]

    monkeypatch.setattr(
        "latentdynamics.viz.morse_plots.render_morse_sets_from_csv",
        fake_render,
    )
    cells = [{
        "dataset_id": 1,
        "model_seed": 2,
        "resolved_cell_path": cell_dir,
        "cmgdb": {"bounds": {"lower": [-1.0, -2.0], "upper": [3.0, 4.0]}},
    }]

    original_bbox = matplotlib.rcParams["savefig.bbox"]
    rendered = REPORT._render_summary_morse_set_images(
        cells,
        tmp_path / "summary-assets",
        min_box_side_frac=0.0025,
    )

    assert rendered[(1, 2)].is_file()
    assert matplotlib.rcParams["savefig.bbox"] == original_bbox
    assert calls == [{
        "csv_path": morse_dir / "morse_sets",
        "out_dir": tmp_path / "summary-assets" / "dataset_01" / "seed_2",
        "bounds_lower": [-1.0, -2.0],
        "bounds_upper": [3.0, 4.0],
        "basename": "morse_sets_summary",
        "formats": ("png",),
        "box_scale": "auto",
        "min_box_side_frac": 0.0025,
    }]

    with pytest.raises(ValueError, match="finite and nonnegative"):
        REPORT._render_summary_morse_set_images(
            cells,
            tmp_path / "invalid",
            min_box_side_frac=-0.01,
        )


def test_invalid_summary_display_floor_writes_nothing(tmp_path: Path) -> None:
    sweep, data = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        REPORT.build_summary(
            sweep_root=sweep,
            data_root=data,
            summary_min_box_side_frac=float("nan"),
        )

    assert not (sweep / "summary").exists()
