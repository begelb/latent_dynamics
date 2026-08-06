"""Focused end-to-end tests for the ground-box curriculum 3x5 summary."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_report():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_leslie3d_groundbox_curriculum_3x5.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_leslie3d_groundbox_curriculum_3x5",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = _load_report()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _training_summary(model_seed: int) -> dict:
    stages = []
    for index, weights in enumerate(REPORT.EXPECTED_STAGE_WEIGHTS, start=1):
        losses = {
            "loss_reconstruction": index * 1.0e-4,
            "loss_prediction": index * 2.0e-4,
            "loss_semiconjugacy": index * 3.0e-4,
        }
        losses["loss_total"] = sum(
            weight * losses[key] for weight, key in zip(weights, REPORT.LOSS_KEYS[:3], strict=True)
        )
        name = REPORT.EXPECTED_STAGE_NAMES[index - 1]
        stages.append(
            {
                "index": index,
                "name": name,
                "start_epoch_one_based": (index - 1) * 4_000 + 1,
                "end_epoch_one_based": index * 4_000,
                "epochs": 4_000,
                "learning_rate": 3.0e-3,
                "loss_weights": list(weights),
                "trainable_components": list(REPORT.EXPECTED_STAGE_TRAINABLE[index - 1]),
                "optimizer_state_continued_from_previous_stage": index > 1,
                "train_endpoint_post_update": losses,
                "holdout_endpoint_post_update": losses,
                "checkpoint": (f"stage_checkpoints/{index:02d}_{name}/models/autoencoder.pt"),
                "checkpoint_metadata": (
                    f"stage_checkpoints/{index:02d}_{name}/models/autoencoder.json"
                ),
            }
        )
    final_losses = {
        "loss_reconstruction": 1.0e-5,
        "loss_prediction": 2.0e-5,
        "loss_semiconjugacy": 3.0e-5,
        "loss_total": 6.0e-5,
    }
    adamw_losses = stages[-1]["train_endpoint_post_update"]
    polish_delta = {key: final_losses[key] - adamw_losses[key] for key in REPORT.LOSS_KEYS}
    return {
        "training_method": "curriculum_full_batch",
        "optimizer": {
            "sequence": ["AdamW", "LBFGS"],
            "first_order": {
                "name": "AdamW",
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "foreach": False,
                "fused": False,
                "state_continues_across_stages": True,
                "stage_learning_rates": [3.0e-3, 3.0e-3, 3.0e-3],
                "updates_completed": 12_000,
                "device": "mps",
                "dtype": "float32",
            },
            "polish": {
                "name": "LBFGS",
                "starts_with_fresh_optimizer_state": True,
                "device": "cpu",
                "dtype": "float64",
                "outer_steps_requested": 12,
                "outer_steps_completed": 12,
                "learning_rate": 0.25,
                "max_iter": 10,
                "max_eval": 25,
                "history_size": 50,
                "tolerance_grad": 1.0e-9,
                "tolerance_change": 1.0e-12,
                "line_search_fn": "strong_wolfe",
                "loss_weights": [1.0, 1.0, 1.0],
                "trainable_components": ["encoder", "latent_map", "decoder"],
                "closure_evaluations": 137,
                "internal_iterations": 91,
            },
        },
        "scheduler": None,
        "seed": model_seed,
        "arch": {
            "high_dims": 3,
            "low_dims": 2,
            "encoder": {
                "hidden_shapes": [128, 64],
                "activation": "tanh",
                "out_activation": "none",
            },
            "latent_map": {
                "hidden_shapes": [64, 64],
                "activation": "tanh",
                "out_activation": "none",
            },
            "decoder": {
                "hidden_shapes": [64, 128],
                "activation": "tanh",
                "out_activation": "none",
            },
        },
        "model_initialized_by_helper": True,
        "data": {
            "n_training_pairs": 20_000,
            "n_validation_pairs": 4_000,
            "high_dims": 3,
            "dtype": "float32",
            "full_batch": True,
        },
        "curriculum": stages,
        "loss_weights": [1.0, 1.0, 1.0],
        "n_epochs_run": 12_000,
        "epochs_requested": 12_000,
        "epochs_completed": 12_000,
        "first_order_epochs_completed": 12_000,
        "checkpoint_selection": "final_lbfgs_float32_endpoint",
        "checkpoint_source": "lbfgs_float32_endpoint",
        "best_epoch": None,
        "validation_evaluated": True,
        "validation_used_for_optimization": False,
        "validation_used_for_checkpoint_selection": False,
        "early_stopping_used": False,
        "patience_used": False,
        "scheduler_used": False,
        "gradient_clipping_used": False,
        "best_weight_restoration_used": False,
        "train_duration_seconds": 1.0,
        "adamw_endpoint_train": adamw_losses,
        "adamw_endpoint_holdout": adamw_losses,
        "final_checkpoint_train": final_losses,
        "final_holdout": final_losses,
        "polish_delta_train": polish_delta,
        "polish_delta_holdout": polish_delta,
        "final_learning_rate": 3.0e-3,
        "artifacts": {
            "checkpoint": "models/autoencoder.pt",
            "checkpoint_metadata": "models/autoencoder.json",
            "adamw_checkpoint": "adamw_endpoint/models/autoencoder.pt",
            "adamw_checkpoint_metadata": "adamw_endpoint/models/autoencoder.json",
            "history": "logs/history.json",
        },
    }


def _write_complete_sweep(tmp_path: Path) -> Path:
    sweep = tmp_path / "sweep"
    cells = []
    sink_nodes = [
        {"node": 0, "index": "(x^4-1, 0, 0)", "period": 4},
        {"node": 1, "index": "(x-1, 0, 0)", "period": 1},
    ]
    graph = """digraph {
0 [label="0 : (x^4-1, 0, 0)"];
1 [label="1 : (x-1, 0, 0)"];
2 [label="2 : (0, x-1, 0)"];
2 -> 0;
2 -> 1;
}
"""
    for data_seed in REPORT.DATA_SEEDS:
        for model_seed in REPORT.MODEL_SEEDS:
            cell = sweep / f"dataset_{data_seed}" / f"seed_{model_seed}"
            _write_json(cell / "training_summary.json", _training_summary(model_seed))
            (cell / "models").mkdir(parents=True)
            (cell / "models" / "autoencoder.pt").write_bytes(b"checkpoint")
            _write_json(cell / "models" / "autoencoder.json", {})
            (cell / "adamw_endpoint" / "models").mkdir(parents=True)
            (cell / "adamw_endpoint" / "models" / "autoencoder.pt").write_bytes(b"adamw-checkpoint")
            _write_json(cell / "adamw_endpoint" / "models" / "autoencoder.json", {})
            _write_json(cell / "logs" / "history.json", {})
            for index, name in enumerate(REPORT.EXPECTED_STAGE_NAMES, start=1):
                stage_models = cell / "stage_checkpoints" / f"{index:02d}_{name}" / "models"
                stage_models.mkdir(parents=True)
                (stage_models / "autoencoder.pt").write_bytes(b"stage-checkpoint")
                _write_json(stage_models / "autoencoder.json", {})
            (cell / "MG").mkdir()
            (cell / "MG" / "morse_graph").write_text(graph, encoding="utf-8")
            (cell / "MG" / "morse_sets").write_text("0,0,1,1,0\n", encoding="utf-8")
            (cell / "mg_params_log.txt").write_text(
                "\n".join(
                    (
                        "Lower bounds: [-1.0, -2.0]",
                        "Upper bounds: [1.0, 2.0]",
                        "subdiv_init: 25",
                        "subdiv_min: 28",
                        "subdiv_max: 29",
                        "subdiv_limit: 10000",
                        "bounds_epsilon_frac: 0.01",
                        "padding: True",
                        "box_map_backend: adaptive_precomputed",
                        "bounds_data_role: train_pairs",
                        "adaptive_precompute_subdiv: init",
                        "bounds_source: encoded_train_pairs",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(
                cell / "metrics.json",
                {
                    "minimal_morse_labels": [0, 1],
                    "minimal_morse_sets": {
                        "0": {
                            "n_boxes": 10,
                            "tau_bar": 1.0e-3,
                            "n_semiconjugacy_samples": 4,
                            "max_semiconjugacy_error": 5.0e-4,
                        },
                        "1": {
                            "n_boxes": 8,
                            "tau_bar": 1.0e-3,
                            "n_semiconjugacy_samples": 0,
                            "max_semiconjugacy_error": None,
                        },
                    },
                },
            )
            _write_json(
                cell / "diagnose.json",
                {
                    "diagnostic": "ok",
                    "hard_flags": {
                        "encoder_collapsed": False,
                        "latent_map_overcontracted": False,
                    },
                },
            )
            _write_json(cell / "run_manifest.json", {"status": "synthetic-test"})
            cells.append(
                {
                    "ic_seed": data_seed,
                    "model_seed": model_seed,
                    "output_dir": str(cell),
                    "n_morse_nodes": 3,
                    "n_morse_edges": 2,
                    "n_sinks": 2,
                    "n_attractor_type_nodes": 2,
                    "n_periodic_attractor_nodes": 2,
                    "sink_nodes": sink_nodes,
                    "bistability_pass": True,
                }
            )
    _write_json(
        sweep / "sweep_summary.json",
        {
            "example": "leslie3d_groundbox_curriculum_wide",
            "ic_seeds": list(REPORT.DATA_SEEDS),
            "model_seeds": list(REPORT.MODEL_SEEDS),
            "cells": cells,
        },
    )
    return sweep


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_strict_complete_sweep_writes_nonprovisional_15_cell_summary(
    tmp_path: Path,
) -> None:
    sweep = _write_complete_sweep(tmp_path)

    assert REPORT.main(["--sweep-root", str(sweep)]) == 0

    rows = _csv_rows(sweep / "summary" / "cells.csv")
    aggregate = json.loads((sweep / "summary" / "aggregate_summary.json").read_text())
    assert len(rows) == 15
    assert {row["cell_status"] for row in rows} == {"complete"}
    assert {row["training_contract_valid"] for row in rows} == {"True"}
    assert aggregate["provisional"] is False
    assert aggregate["inventory"]["n_complete_cells"] == 15
    assert {row["optimizer_sequence"] for row in rows} == {'["AdamW","LBFGS"]'}
    assert {row["first_order_epochs_completed"] for row in rows} == {"12000"}
    assert {row["lbfgs_internal_iterations"] for row in rows} == {"91"}
    assert {row["lbfgs_closure_evaluations"] for row in rows} == {"137"}
    assert {float(row["adamw_train_total"]) for row in rows} == {1.8e-3}
    assert {float(row["final_train_total"]) for row in rows} == {6.0e-5}
    assert all(
        float(row["lbfgs_delta_train_total"]) == pytest.approx(6.0e-5 - 1.8e-3) for row in rows
    )
    assert aggregate["training_contract"]["optimizer"]["sequence"] == ["AdamW", "LBFGS"]
    assert aggregate["optimizer_accounting"]["lbfgs_internal_iterations"]["mean"] == 91
    assert aggregate["optimizer_accounting"]["lbfgs_closure_evaluations"]["mean"] == 137
    assert aggregate["optimizer_accounting"]["lbfgs_train_total_nonincrease"] == {
        "n_evaluated": 15,
        "n_pass": 15,
    }
    assert aggregate["topology"]["bistability_pass"]["n_pass"] == 15
    assert aggregate["topology"]["exact_period4_bistability_pass"]["n_pass"] == 0
    assert aggregate["sampled_tolerance_diagnostic"] == {
        "interpretation": (
            "Counts only the saved sampled residual-versus-tau inequality; "
            "it is not a classifier of spuriousness or invariant-set correspondence."
        ),
        "n_minimal_components": 30,
        "n_evaluable": 15,
        "n_pass": 15,
    }


def test_incomplete_sweep_strict_refuses_then_provisional_writes_15_rows(
    tmp_path: Path,
) -> None:
    sweep = tmp_path / "incomplete"
    _write_json(
        sweep / "sweep_summary.json",
        {
            "ic_seeds": list(REPORT.DATA_SEEDS),
            "model_seeds": list(REPORT.MODEL_SEEDS),
            "cells": [],
        },
    )

    with pytest.raises(REPORT.SweepValidationError, match="strict summary refused"):
        REPORT.main(["--sweep-root", str(sweep)])
    assert not (sweep / "summary").exists()

    assert REPORT.main(["--sweep-root", str(sweep), "--allow-incomplete"]) == 0
    rows = _csv_rows(sweep / "summary" / "cells.csv")
    aggregate = json.loads((sweep / "summary" / "aggregate_summary.json").read_text())
    assert len(rows) == 15
    assert {row["cell_status"] for row in rows} == {"missing"}
    assert aggregate["provisional"] is True
    assert aggregate["inventory"]["n_missing_cells"] == 15


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda summary: summary["optimizer"]["first_order"].__setitem__("weight_decay", 1.0e-4),
            "optimizer.first_order.weight_decay",
        ),
        (
            lambda summary: summary["optimizer"]["polish"].__setitem__("closure_evaluations", 301),
            "optimizer.polish.closure_evaluations",
        ),
        (
            lambda summary: summary["final_checkpoint_train"].update(
                {
                    "loss_reconstruction": 1.0e-3,
                    "loss_prediction": 1.0e-3,
                    "loss_semiconjugacy": 1.0e-3,
                    "loss_total": 3.0e-3,
                }
            ),
            "increased the training objective",
        ),
    ),
)
def test_strict_rejects_optimizer_protocol_or_polish_regression(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    sweep = _write_complete_sweep(tmp_path)
    target = sweep / f"dataset_{REPORT.DATA_SEEDS[0]}" / "seed_0" / "training_summary.json"
    summary = json.loads(target.read_text(encoding="utf-8"))
    mutate(summary)
    _write_json(target, summary)

    with pytest.raises(REPORT.SweepValidationError, match="invalid_training_contract"):
        REPORT.main(["--sweep-root", str(sweep)])
    assert not (sweep / "summary").exists()

    assert REPORT.main(["--sweep-root", str(sweep), "--allow-incomplete"]) == 0
    rows = _csv_rows(sweep / "summary" / "cells.csv")
    invalid = [row for row in rows if row["cell_status"] == "invalid"]
    assert len(invalid) == 1
    assert message in invalid[0]["validation_errors"]
