"""Focused tests for the strict Ives Lake Mývatn 3x5 summarizer."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image


def _load_report():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_ives_myvatn_3x5.py"
    spec = importlib.util.spec_from_file_location("summarize_ives_myvatn_3x5", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = _load_report()

from latentdynamics.cli.provenance import hash_config_dict  # noqa: E402
from latentdynamics.config import load_config  # noqa: E402
from latentdynamics.config.schema import ArchConfig  # noqa: E402
from latentdynamics.models.autoencoder import build_autoencoder  # noqa: E402
from latentdynamics.sampling import fit_fixed_bounds_scaler  # noqa: E402
from latentdynamics.training.checkpoints import save_checkpoint  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_reference(path: Path) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    cycle = np.asarray(
        [
            [-2.8 + 0.35 * phase, -7.0 + 0.45 * (phase % 5), -2.7 + 0.3 * (phase % 7)]
            for phase in range(12)
        ],
        dtype=np.float64,
    )
    fixed = np.asarray([[1.4, 1.4, 1.4]], dtype=np.float64)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["vertex", "component_id", "barycenter_x", "barycenter_y", "barycenter_z"]
        )
        for phase, point in enumerate(cycle):
            writer.writerow([0, phase, *point])
        writer.writerow([1, 0, *fixed[0]])
    return np.vstack([fixed, cycle])


def _make_checkpoint(path: Path) -> tuple[object, ArchConfig]:
    arch = load_config("ives_myvatn").arch
    model = build_autoencoder(arch)
    with torch.no_grad():
        first = model.encoder.net[0]
        last = model.encoder.net[2]
        first.weight.zero_()
        first.weight[:3].copy_(torch.eye(3))
        first.bias.zero_()
        last.weight.zero_()
        last.weight[0, 0] = 1.0
        last.weight[1, 1] = 1.0
        last.bias.zero_()
    save_checkpoint(model, arch, path)
    return model, arch


def _graph_text(*, with_periods: bool = True) -> str:
    fixed_index = "(x-1, 0, 0)" if with_periods else "(0, 0, 0)"
    cycle_index = "(x^12-1, 0, 0)" if with_periods else "(0, 0, 0)"
    return f'''digraph {{
"10" [label="10 : {fixed_index}"];
"20" [label="20 : (0, x-1, 0)"];
"30" [label="30 : {cycle_index}"];
"40" [label="40 : (0, 0, x-1)"];
"40" -> "30";
"40" -> "20";
"20" -> "10";
}}
'''


def _write_morse_sets(path: Path, encoded: np.ndarray) -> None:
    pairwise = np.max(np.abs(encoded[:, None, :] - encoded[None, :, :]), axis=2)
    positive = pairwise[pairwise > 0]
    epsilon = float(positive.min() / 4.0)
    rows: list[list[float | int]] = []
    fixed = encoded[0]
    rows.append([fixed[0] - epsilon, fixed[1] - epsilon, fixed[0] + epsilon, fixed[1] + epsilon, 10])
    for point in encoded[1:]:
        rows.append(
            [point[0] - epsilon, point[1] - epsilon, point[0] + epsilon, point[1] + epsilon, 30]
        )
    rows.extend(
        [
            [-5.0, -5.0, -4.0, -4.0, 20],
            [4.0, 4.0, 5.0, 5.0, 40],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _write_renders(morse_dir: Path) -> None:
    for name in ("morse_graph.png", "morse_sets.png"):
        Image.new("RGB", (2, 2), "white").save(morse_dir / name)
    for name in ("morse_graph.pdf", "morse_sets.pdf"):
        (morse_dir / name).write_bytes(b"%PDF-1.4\n%%EOF\n")


def _write_transition_csv(path: Path, *, n_rows: int, marker: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "x0,x1,x2,y0,y1,y2\n"
    row = ",".join(f"{marker + offset * 1.0e-6:.8f}" for offset in range(6)) + "\n"
    path.write_text(header + row * n_rows, encoding="utf-8")


def _dataset_metadata(*, role: str, data_seed: int) -> dict:
    is_train = role == "train"
    return {
        "system": "IvesModel",
        "dimension": 3,
        "n_samples": 1000 if is_train else 200,
        "n_iterations": 70,
        "skip_initial_steps": 50,
        "lower_bounds": list(REPORT.EXPECTED_AMBIENT_LOWER),
        "upper_bounds": list(REPORT.EXPECTED_AMBIENT_UPPER),
        "model_params": dict(REPORT.EXPECTED_MODEL_PARAMS),
        "dataset_name": "train" if is_train else "val",
        "sampling_method": "uniform",
        "sampling_seed": data_seed if is_train else REPORT.EXPECTED_VALIDATION_SEED,
        "role": role if is_train else "val",
    }


def _write_complete_sweep(tmp_path: Path) -> tuple[Path, Path, Path]:
    sweep = tmp_path / "sweep"
    data_root = tmp_path / "data"
    reference = tmp_path / "ives_reference.csv"
    reference_points = _write_reference(reference)

    template_models = tmp_path / "template_models"
    model, _arch = _make_checkpoint(template_models)
    scaler = fit_fixed_bounds_scaler(
        REPORT.EXPECTED_AMBIENT_LOWER,
        REPORT.EXPECTED_AMBIENT_UPPER,
        epsilon=REPORT.EXPECTED_SCALING_EPSILON,
    )
    scaled = scaler.transform(reference_points)
    with torch.no_grad():
        encoded = model.encoder(torch.as_tensor(scaled, dtype=torch.float32)).numpy()

    for data_seed in REPORT.DATA_SEEDS:
        dataset = sweep / f"dataset_{data_seed}"
        dataset_data = data_root / f"dataset_{data_seed}"
        train_csv = dataset_data / "train.csv"
        validation_csv = dataset_data / "val.csv"
        _write_transition_csv(
            train_csv,
            n_rows=REPORT.EXPECTED_TRAIN_PAIRS,
            marker=-2.0 + data_seed * 1.0e-5,
        )
        _write_transition_csv(
            validation_csv,
            n_rows=REPORT.EXPECTED_VALIDATION_PAIRS,
            marker=-1.25,
        )
        _write_json(
            dataset_data / "train_metadata.json",
            _dataset_metadata(role="train", data_seed=data_seed),
        )
        _write_json(
            dataset_data / "val_metadata.json",
            _dataset_metadata(role="validation", data_seed=data_seed),
        )
        scaler_path = dataset / "scalers" / "train" / "scaler.gz"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        _write_json(
            scaler_path.with_name("scaler_metadata.json"),
            {
                "created_at_utc": "2026-08-06T00:00:00+00:00",
                "train_file": "train",
                "train_csv": str(train_csv.resolve()),
                "train_csv_sha256": REPORT._sha256(train_csv),
                "scaling": "fixed_bounds",
                "high_dims": 3,
                "lower_bounds": list(REPORT.EXPECTED_AMBIENT_LOWER),
                "upper_bounds": list(REPORT.EXPECTED_AMBIENT_UPPER),
                "scaling_epsilon": REPORT.EXPECTED_SCALING_EPSILON,
            },
        )
        expected_config = REPORT._expected_manifest_config(
            data_seed=data_seed,
            data_root=data_root,
            sweep_root=sweep,
        )
        for model_seed in REPORT.MODEL_SEEDS:
            cell = dataset / f"seed_{model_seed}"
            models = cell / "models"
            models.mkdir(parents=True, exist_ok=True)
            shutil.copy2(template_models / "autoencoder.pt", models / "autoencoder.pt")
            shutil.copy2(template_models / "autoencoder.json", models / "autoencoder.json")
            _write_json(
                cell / "logs" / "history.json",
                {
                    "train": {"loss_total": [0.4, 0.2]},
                    "val": {"loss_total": [0.5, 0.25]},
                },
            )
            _write_json(
                cell / "training_summary.json",
                {
                    "best_epoch": 1,
                    "best_source": "training_epoch",
                    "selected_val": {
                        "loss_reconstruction": 0.1,
                        "loss_prediction": 0.1,
                        "loss_semiconjugacy": 0.05,
                        "loss_total": 0.25,
                    },
                    "loss_weights": [1.0, 1.0, 1.0],
                    "n_epochs_run": 2,
                    "train_duration_seconds": 1.5,
                },
            )
            (cell / "final_losses.txt").write_text(
                "best_epoch: 1\nbest_source: training_epoch\nval_loss_total: 2.5e-1\n",
                encoding="utf-8",
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
            _write_json(cell / "metrics.json", {"morse_graph_consistency": {"consistent": True}})
            _write_json(
                cell / "run_manifest.json",
                {
                    "config": expected_config,
                    "config_hash": hash_config_dict(expected_config),
                    "cell": {
                        "seed": model_seed,
                        "train_file": "train",
                        "output_dir": str(cell.resolve()),
                    },
                    "artifacts": {
                        "train_csv": str(train_csv.resolve()),
                        "train_csv_sha256": REPORT._sha256(train_csv),
                        "scaler": str(scaler_path.resolve()),
                        "scaler_sha256": REPORT._sha256(scaler_path),
                        "model_dir": str((cell / "models").resolve()),
                        "morse_dir": str((cell / "MG").resolve()),
                        "metrics": str((cell / "metrics.json").resolve()),
                    },
                },
            )
            morse_dir = cell / "MG"
            morse_dir.mkdir(parents=True, exist_ok=True)
            (morse_dir / "morse_graph").write_text(_graph_text(), encoding="utf-8")
            _write_morse_sets(morse_dir / "morse_sets", encoded)
            _write_renders(morse_dir)
            (cell / "mg_params_log.txt").write_text(
                "\n".join(
                    (
                        "Lower bounds: [-6.0, -6.0]",
                        "Upper bounds: [6.0, 6.0]",
                        "subdiv_init: 18",
                        "subdiv_min: 22",
                        "subdiv_max: 30",
                        "subdiv_limit: 100000",
                        "bounds_epsilon_frac: 0.1",
                        "padding: True",
                        "box_map_backend: adaptive_precomputed",
                        "bounds_data_role: system_grid",
                        "bounds_grid_resolution: 64",
                        "bounds_include_latent_image: True",
                        "bounds_clip_lower: [-1.0, -1.0]",
                        "bounds_clip_upper: [1.0, 1.0]",
                        "adaptive_precompute_subdiv: init",
                        "precompute_batch_points: auto",
                        "compute_roa: False",
                        "roa_max_vertices: 50000000",
                        "collapse_roa_to_lca: True",
                        "bounds_source: encoded_system_grid_and_latent_image",
                        "duration_minutes: 0.1",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
    return sweep, data_root, reference


def _memberships(
    *,
    target_count: int,
    final_membership: list[str],
) -> dict:
    phases = []
    for phase in range(12):
        sink_memberships = ["30"] if phase < target_count else list(final_membership)
        phases.append({"phase": phase, "sink_memberships": sink_memberships})
    return {
        "fixed_point": {"sink_memberships": ["10"]},
        "period_12_phases": phases,
    }


def test_strict_complete_sweep_writes_15_passes_and_full_evidence(tmp_path: Path) -> None:
    sweep, data_root, reference = _write_complete_sweep(tmp_path)

    assert (
        REPORT.main(
            [
                "--sweep-root",
                str(sweep),
                "--data-root",
                str(data_root),
                "--reference-csv",
                str(reference),
            ]
        )
        == 0
    )

    aggregate = json.loads((sweep / "summary" / "aggregate_summary.json").read_text())
    detailed = json.loads((sweep / "summary" / "cells.json").read_text())
    with (sweep / "summary" / "cells.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert aggregate["provisional"] is False
    assert aggregate["inventory"]["n_verified_cells"] == 15
    assert aggregate["classification"]["n_pass"] == 15
    assert aggregate["classification"]["n_exact_conley_periods_1_12"] == 15
    assert aggregate["training"]["selected_validation_losses"]["loss_total"]["mean"] == 0.25
    assert {row["machine_pass"] for row in rows} == {"True"}
    first = detailed["cells"][0]
    assert {node["id"] for node in first["morse_graph"]["nodes"]} == {"10", "20", "30", "40"}
    assert first["morse_graph"]["nodes"][0]["conley_tuple"] == "(x-1, 0, 0)"
    assert first["reference_memberships"]["fixed_point"]["sink_memberships"] == ["10"]
    assert all(
        phase["sink_memberships"] == ["30"]
        for phase in first["reference_memberships"]["period_12_phases"]
    )
    assert len(first["artifacts"]) == len(REPORT.REQUIRED_FILES) + 1
    assert all(artifact["sha256"] for artifact in first["artifacts"].values())


def test_conflicting_cycle_sink_fails_even_with_eleven_target_phases(tmp_path: Path) -> None:
    dot = tmp_path / "morse_graph"
    dot.write_text(_graph_text(), encoding="utf-8")
    graph = REPORT._parse_dot(dot)

    result = REPORT._classify(
        graph,
        _memberships(target_count=11, final_membership=["10"]),
    )

    assert result["archive_graph_isomorphic"] is True
    assert result["cycle_unique_target_count"] == 11
    assert result["cycle_conflicting_phase_count"] == 1
    assert result["machine_pass"] is False


def test_eleven_cycle_memberships_and_one_unassigned_pass_without_conley_periods(
    tmp_path: Path,
) -> None:
    dot = tmp_path / "morse_graph"
    dot.write_text(_graph_text(with_periods=False), encoding="utf-8")
    graph = REPORT._parse_dot(dot)

    result = REPORT._classify(
        graph,
        _memberships(target_count=11, final_membership=[]),
    )

    assert result["cycle_unique_target_count"] == 11
    assert result["cycle_unassigned_count"] == 1
    assert result["cycle_conflicting_phase_count"] == 0
    assert result["exact_conley_periods_1_12"] is False
    assert result["machine_pass"] is True


def test_strict_incomplete_refuses_to_write_but_progress_mode_is_provisional(
    tmp_path: Path,
) -> None:
    sweep, data_root, reference = _write_complete_sweep(tmp_path)
    missing = sweep / f"dataset_{REPORT.DATA_SEEDS[0]}" / "seed_0" / "metrics.json"
    missing.unlink()

    assert (
        REPORT.main(
            [
                "--sweep-root",
                str(sweep),
                "--data-root",
                str(data_root),
                "--reference-csv",
                str(reference),
            ]
        )
        == 2
    )
    assert not (sweep / "summary").exists()

    assert (
        REPORT.main(
            [
                "--sweep-root",
                str(sweep),
                "--data-root",
                str(data_root),
                "--reference-csv",
                str(reference),
                "--allow-incomplete",
            ]
        )
        == 0
    )
    aggregate = json.loads((sweep / "summary" / "aggregate_summary.json").read_text())
    assert aggregate["provisional"] is True
    assert aggregate["inventory"]["n_verified_cells"] == 14
    assert aggregate["inventory"]["n_incomplete_cells"] == 1
    assert aggregate["classification"]["n_evaluated"] == 15


def test_verify_is_read_only(tmp_path: Path) -> None:
    sweep, data_root, reference = _write_complete_sweep(tmp_path)

    assert (
        REPORT.main(
            [
                "--sweep-root",
                str(sweep),
                "--data-root",
                str(data_root),
                "--reference-csv",
                str(reference),
                "--verify",
            ]
        )
        == 0
    )
    assert not (sweep / "summary").exists()


def test_verify_accepts_persisted_cells_csv_integer_fields(tmp_path: Path) -> None:
    sweep, data_root, reference = _write_complete_sweep(tmp_path)
    common_args = [
        "--sweep-root",
        str(sweep),
        "--data-root",
        str(data_root),
        "--reference-csv",
        str(reference),
    ]

    assert REPORT.main(common_args) == 0
    assert REPORT.main([*common_args, "--verify"]) == 0
