"""Move interrupted Ives 3x5 artifacts aside before a resumable launch.

Only genuinely partial artifact groups are recovered. A completed training
cell that is merely waiting for topology is left in place, as is an empty
future cell. Recovery is a rename into a timestamped tree; nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data" / "ives_myvatn_seedsweep_3x5_v1"
DEFAULT_SWEEP_ROOT = CODE_ROOT / "output" / "ives_myvatn_seedsweep_3x5_v1"
DATA_SEEDS = (2158, 4792, 3174, 688, 5727)
MODEL_SEEDS = (0, 1, 2)


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _partial(paths: tuple[Path, ...]) -> bool:
    present = [_nonempty(path) for path in paths]
    return any(present) and not all(present)


def _valid_training_group(cell: Path) -> bool:
    """Mirror the generic trainer's resumable-completion contract.

    An interrupted process can leave all expected filenames behind while a
    JSON report is truncated.  Move that whole cell aside too, so the next
    launch never overwrites a checkpoint that only *looks* complete by file
    presence.
    """

    checkpoint = cell / "models" / "autoencoder.pt"
    sidecar = cell / "models" / "autoencoder.json"
    history_path = cell / "logs" / "history.json"
    summary_path = cell / "training_summary.json"
    final_losses = cell / "final_losses.txt"
    if not all(
        _nonempty(path)
        for path in (checkpoint, sidecar, history_path, summary_path, final_losses)
    ):
        return False
    try:
        sidecar_payload = json.loads(sidecar.read_text())
        history = json.loads(history_path.read_text())
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not all(isinstance(payload, dict) for payload in (sidecar_payload, history, summary)):
        return False
    epochs_run = summary.get("n_epochs_run")
    if (
        isinstance(epochs_run, bool)
        or not isinstance(epochs_run, int)
        or not 1 <= epochs_run <= 500
        or summary.get("loss_weights") != [1.0, 1.0, 1.0]
    ):
        return False
    for split in ("train", "val"):
        block = history.get(split)
        totals = block.get("loss_total") if isinstance(block, dict) else None
        if not isinstance(totals, list) or len(totals) != epochs_run:
            return False
    return True


def recover_partials(data_root: Path, sweep_root: Path) -> list[dict[str, str]]:
    """Recover partial groups and return an audit record for every move."""

    pending: list[tuple[Path, str]] = []
    for data_seed in DATA_SEEDS:
        dataset_dir = data_root / f"dataset_{data_seed}"
        dataset_core = tuple(
            dataset_dir / name
            for name in ("train.csv", "train_metadata.json", "val.csv", "val_metadata.json")
        )
        if _partial(dataset_core):
            pending.append((dataset_dir, f"data/dataset_{data_seed}"))
            continue

        dataset_output = sweep_root / f"dataset_{data_seed}"
        scaler_root = dataset_output / "scalers" / "train"
        scaler_core = (
            scaler_root / "scaler.gz",
            scaler_root / "scaler_metadata.json",
        )
        if _partial(scaler_core):
            pending.append((dataset_output / "scalers", f"scalers/dataset_{data_seed}"))

        for model_seed in MODEL_SEEDS:
            cell = dataset_output / f"seed_{model_seed}"
            training_core = (
                cell / "models" / "autoencoder.pt",
                cell / "models" / "autoencoder.json",
                cell / "logs" / "history.json",
                cell / "training_summary.json",
                cell / "final_losses.txt",
            )
            if any(_nonempty(path) for path in training_core) and not _valid_training_group(cell):
                pending.append(
                    (cell, f"cells/dataset_{data_seed}/seed_{model_seed}")
                )
                continue

            topology_core = (
                cell / "MG" / "morse_graph",
                cell / "MG" / "morse_sets",
                cell / "mg_params_log.txt",
            )
            if _partial(topology_core):
                if (cell / "MG").exists():
                    pending.append(
                        (cell / "MG", f"topology/dataset_{data_seed}/seed_{model_seed}/MG")
                    )
                if (cell / "mg_params_log.txt").exists():
                    pending.append(
                        (
                            cell / "mg_params_log.txt",
                            f"topology/dataset_{data_seed}/seed_{model_seed}/mg_params_log.txt",
                        )
                    )
                continue

            render_paths = tuple(
                cell / "MG" / name
                for name in (
                    "morse_graph.pdf",
                    "morse_graph.png",
                    "morse_sets.pdf",
                    "morse_sets.png",
                )
            )
            if _partial(render_paths):
                for path in render_paths:
                    if path.exists():
                        pending.append(
                            (
                                path,
                                f"render/dataset_{data_seed}/seed_{model_seed}/{path.name}",
                            )
                        )

    if not pending:
        return []

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    recovery_root = sweep_root / "recovery" / stamp
    records: list[dict[str, str]] = []
    for source, relative_destination in pending:
        if not source.exists():
            continue
        destination = recovery_root / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"recovery destination already exists: {destination}")
        shutil.move(str(source), str(destination))
        records.append({"source": str(source), "recovered_to": str(destination)})

    recovery_root.mkdir(parents=True, exist_ok=True)
    (recovery_root / "recovery_manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "moves": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    args = parser.parse_args(argv)
    records = recover_partials(args.data_root.resolve(), args.sweep_root.resolve())
    if records:
        print(json.dumps({"recovered": records}, indent=2))
    else:
        print("no partial Ives artifacts require recovery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
