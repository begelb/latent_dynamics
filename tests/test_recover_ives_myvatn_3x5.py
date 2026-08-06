from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "recover_ives_myvatn_3x5.py"
    spec = importlib.util.spec_from_file_location("recover_ives_myvatn_3x5", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RECOVER = _load_module()


def _write_complete_training(cell: Path) -> None:
    (cell / "models").mkdir(parents=True, exist_ok=True)
    (cell / "models" / "autoencoder.pt").write_bytes(b"checkpoint")
    (cell / "models" / "autoencoder.json").write_text("{}")
    (cell / "logs").mkdir(parents=True, exist_ok=True)
    (cell / "logs" / "history.json").write_text(
        json.dumps(
            {
                "train": {"loss_total": [1.0, 0.5]},
                "val": {"loss_total": [1.1, 0.6]},
            }
        )
    )
    (cell / "training_summary.json").write_text(
        json.dumps({"n_epochs_run": 2, "loss_weights": [1.0, 1.0, 1.0]})
    )
    (cell / "final_losses.txt").write_text("val_loss_total: 0.6\n")


def test_recover_moves_partial_training_but_keeps_complete_training(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    sweep_root = tmp_path / "output"
    partial = sweep_root / "dataset_2158" / "seed_0"
    (partial / "models").mkdir(parents=True)
    (partial / "models" / "autoencoder.pt").write_bytes(b"partial")

    complete = sweep_root / "dataset_2158" / "seed_1"
    _write_complete_training(complete)

    records = RECOVER.recover_partials(data_root, sweep_root)

    assert not partial.exists()
    assert complete.exists()
    assert len(records) == 1
    recovered = Path(records[0]["recovered_to"])
    assert (recovered / "models" / "autoencoder.pt").read_bytes() == b"partial"
    manifest = next((sweep_root / "recovery").glob("*/recovery_manifest.json"))
    assert len(json.loads(manifest.read_text())["moves"]) == 1


def test_recover_leaves_empty_future_topology_and_complete_render_alone(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    sweep_root = tmp_path / "output"
    cell = sweep_root / "dataset_2158" / "seed_0"
    _write_complete_training(cell)

    assert RECOVER.recover_partials(data_root, sweep_root) == []
    assert cell.exists()
    assert not (sweep_root / "recovery").exists()


def test_recover_moves_complete_looking_but_truncated_training(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    sweep_root = tmp_path / "output"
    cell = sweep_root / "dataset_2158" / "seed_0"
    _write_complete_training(cell)
    (cell / "training_summary.json").write_text('{"n_epochs_run":')

    records = RECOVER.recover_partials(data_root, sweep_root)

    assert not cell.exists()
    assert len(records) == 1
    recovered = Path(records[0]["recovered_to"])
    assert (recovered / "training_summary.json").read_text() == '{"n_epochs_run":'
