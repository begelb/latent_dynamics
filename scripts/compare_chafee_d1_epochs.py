"""Train and compare the Chafee--Infante d=1 model at a longer epoch count.

The canonical 4,000-epoch artifact is read-only.  A fresh seed-0 run is trained
from the same exact 30,000 archived pairs, architecture, optimizer, scheduler,
and backend into a separate output directory.  Both final checkpoints are then
evaluated on CPU float32 over all stored pairs, and the full-batch training
history is rendered as a residual-by-epoch curve.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from latentdynamics.training import load_checkpoint, train_marcio_full_batch
from latentdynamics.viz.style import PALETTE, apply_paper_style, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA = PROJECT_ROOT / "archive" / "marcio" / "scripts" / "train_data.csv"
CURRENT_RUN = (
    CODE_ROOT
    / "output"
    / "chafee_latent_dimension_study"
    / "latent_1d"
    / "seed_0"
)
DEFAULT_OUTPUT = CURRENT_RUN.parent / "seed_0_epoch_10000"

TRAIN_DATA_SHA256 = "890fea29a2d26b31b44bbc4fbd773af0ebf742b08893c27ddaf1bdbf87e30329"
CURRENT_CHECKPOINT_SHA256 = (
    "f2d1ad7dcc094e4565f25446e613d4b528261012810bb493ef70d1a3977c0f91"
)
CURRENT_HISTORY_SHA256 = (
    "793ba0b58e9caa18fc162f192d3cb3094692b9e996054c537c2ca78445a57067"
)

N_PAIRS = 30_000
HIGH_DIMENSION = 64
CURRENT_EPOCHS = 4_000
DEFAULT_EPOCHS = 10_000
LEARNING_RATE = 0.003
EVALUATION_BATCH_SIZE = 4_096


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_sha256(path: Path, expected: str, *, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{description} SHA256 mismatch for {path}: "
            f"expected {expected}, observed {actual}"
        )
    return actual


def _load_training_pairs(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pairs = np.loadtxt(path, delimiter=",", dtype=np.float64)
    expected_shape = (N_PAIRS, 2 * HIGH_DIMENSION)
    if pairs.shape != expected_shape:
        raise ValueError(f"{path} has shape {pairs.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(pairs)):
        raise ValueError(f"{path} contains non-finite values")
    return (
        np.ascontiguousarray(pairs[:, :HIGH_DIMENSION]),
        np.ascontiguousarray(pairs[:, HIGH_DIMENSION:]),
    )


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_history(path: Path, expected_epochs: int) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("training_method") != "marcio_full_batch":
        raise ValueError(f"unexpected training method in {path}")
    history = payload.get("train")
    expected_keys = {
        "loss_reconstruction",
        "loss_prediction",
        "loss_total",
        "learning_rate",
    }
    if not isinstance(history, dict) or set(history) != expected_keys:
        raise ValueError(f"malformed training history in {path}")
    lengths = {key: len(values) for key, values in history.items()}
    if set(lengths.values()) != {expected_epochs}:
        raise ValueError(
            f"history lengths in {path} are {lengths}; expected {expected_epochs}"
        )
    if any(
        not np.all(np.isfinite(np.asarray(values, dtype=np.float64)))
        for values in history.values()
    ):
        raise ValueError(f"history contains non-finite values in {path}")
    return history


def _evaluate_model(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    reconstruction_errors: list[Tensor] = []
    prediction_errors: list[Tensor] = []
    latent_errors: list[Tensor] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            current = x[start:stop]
            next_state = y[start:stop]
            encoded_current = model.encoder(current)
            encoded_next = model.encoder(next_state)
            predicted_latent = model.latent_map(encoded_current)
            reconstructed = model.decoder(encoded_current)
            predicted_next = model.decoder(predicted_latent)

            reconstruction_errors.append((reconstructed - current).cpu())
            prediction_errors.append((predicted_next - next_state).cpu())
            latent_errors.append((predicted_latent - encoded_next).cpu())

    reconstruction_error = torch.cat(reconstruction_errors)
    prediction_error = torch.cat(prediction_errors)
    latent_error = torch.cat(latent_errors)
    reconstruction = float(torch.mean(reconstruction_error.square()))
    prediction = float(torch.mean(prediction_error.square()))
    return {
        "L1_reconstruction_mse": reconstruction,
        "L2_decoded_one_step_prediction_mse": prediction,
        "L1_plus_L2": reconstruction + prediction,
        "L3_unconditioned_latent_semiconjugacy_mse": (
            float(torch.mean(latent_error.square()))
        ),
        "global_max_euclidean_latent_residual": float(
            torch.linalg.vector_norm(latent_error, dim=1).max()
        ),
    }


def _evaluate_checkpoint(
    model_dir: Path,
    x: Tensor,
    y: Tensor,
) -> tuple[dict[str, float], dict[str, Any]]:
    model, arch = load_checkpoint(model_dir, map_location="cpu")
    if arch.high_dims != HIGH_DIMENSION or arch.low_dims != 1:
        raise ValueError(
            f"checkpoint in {model_dir} has architecture "
            f"{arch.high_dims}->{arch.low_dims}, expected {HIGH_DIMENSION}->1"
        )
    metrics = _evaluate_model(
        model,
        x,
        y,
        batch_size=EVALUATION_BATCH_SIZE,
    )
    return metrics, arch.model_dump(mode="json")


def _comparison(
    current: dict[str, float],
    extended: dict[str, float],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in current:
        old = current[metric]
        new = extended[metric]
        result[metric] = {
            "current_epoch_4000": old,
            "extended_epoch_10000": new,
            "absolute_change": new - old,
            "percent_change": 100.0 * (new - old) / old,
            "improvement_factor": old / new,
        }
    return result


def _prefix_comparison(
    current: dict[str, list[float]],
    extended: dict[str, list[float]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    all_exact = True
    for key in (
        "loss_reconstruction",
        "loss_prediction",
        "loss_total",
        "learning_rate",
    ):
        left = np.asarray(current[key], dtype=np.float64)
        right = np.asarray(extended[key][:CURRENT_EPOCHS], dtype=np.float64)
        exact = bool(np.array_equal(left, right))
        all_exact &= exact
        metrics[key] = {
            "exactly_equal": exact,
            "max_absolute_difference": float(np.max(np.abs(left - right))),
        }
    return {
        "epochs_compared": CURRENT_EPOCHS,
        "all_history_arrays_exactly_equal": all_exact,
        "metrics": metrics,
    }


def _write_history_csv(
    path: Path,
    history: dict[str, list[float]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "epoch",
                "loss_reconstruction",
                "loss_prediction",
                "loss_total",
                "learning_rate",
            )
        )
        for index in range(len(history["loss_total"])):
            writer.writerow(
                (
                    index + 1,
                    history["loss_reconstruction"][index],
                    history["loss_prediction"][index],
                    history["loss_total"][index],
                    history["learning_rate"][index],
                )
            )
    return path


def _render_history(
    current: dict[str, list[float]],
    extended: dict[str, list[float]],
    output_dir: Path,
) -> list[Path]:
    apply_paper_style()
    epochs = np.arange(1, len(extended["loss_total"]) + 1)
    current_epochs = np.arange(1, CURRENT_EPOCHS + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    ax.semilogy(
        epochs,
        extended["loss_total"],
        color=PALETTE[4],
        linewidth=1.55,
        label="Total objective",
    )
    ax.semilogy(
        epochs,
        extended["loss_reconstruction"],
        color=PALETTE[2],
        linewidth=1.0,
        label="Reconstruction",
    )
    ax.semilogy(
        epochs,
        extended["loss_prediction"],
        color=PALETTE[1],
        linewidth=1.0,
        label="Decoded one-step",
    )
    ax.semilogy(
        current_epochs,
        current["loss_total"],
        color="#555555",
        linewidth=0.9,
        linestyle=(0, (4, 3)),
        label="Current 4,000-epoch total",
    )
    ax.axvline(
        CURRENT_EPOCHS,
        color="#777777",
        linewidth=0.8,
        linestyle=(0, (2, 3)),
    )

    total = np.asarray(extended["loss_total"], dtype=np.float64)
    for epoch in (CURRENT_EPOCHS, len(total)):
        value = float(total[epoch - 1])
        ax.scatter(
            [epoch],
            [value],
            s=22,
            color=PALETTE[4],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        ax.annotate(
            f"{epoch:,}: {value:.3e}",
            xy=(epoch, value),
            xytext=(-8 if epoch == len(total) else 7, 10),
            textcoords="offset points",
            ha="right" if epoch == len(total) else "left",
            fontsize=9,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Full-batch MSE")
    ax.set_xlim(1, len(total))
    ax.grid(axis="y", which="major", color="#777777", alpha=0.20, linewidth=0.45)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    return save_figure(
        fig,
        output_dir / "d1_residual_by_epoch",
        formats=("pdf", "png"),
        close=True,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )


def run_comparison(
    *,
    epochs: int,
    device_name: str,
    output_dir: Path,
    verbose: bool,
    reuse_existing: bool,
) -> dict[str, Any]:
    if epochs <= CURRENT_EPOCHS:
        raise ValueError(f"epochs must exceed the current {CURRENT_EPOCHS} epochs")
    output_dir = output_dir.resolve()
    if output_dir == CURRENT_RUN.resolve():
        raise ValueError("output directory must not overwrite the canonical current run")
    extended_checkpoint_path = output_dir / "models" / "autoencoder.pt"
    extended_history_path = output_dir / "logs" / "history.json"
    extended_summary_path = output_dir / "training_summary.json"
    if extended_checkpoint_path.exists() and not reuse_existing:
        raise FileExistsError(
            f"{output_dir} already contains a checkpoint; refusing to overwrite"
        )

    data_sha = _checked_sha256(
        TRAIN_DATA,
        TRAIN_DATA_SHA256,
        description="exact Marcio training data",
    )
    current_checkpoint = CURRENT_RUN / "models" / "autoencoder.pt"
    current_history_path = CURRENT_RUN / "logs" / "history.json"
    current_checkpoint_sha = _checked_sha256(
        current_checkpoint,
        CURRENT_CHECKPOINT_SHA256,
        description="canonical d=1 checkpoint",
    )
    current_history_sha = _checked_sha256(
        current_history_path,
        CURRENT_HISTORY_SHA256,
        description="canonical d=1 history",
    )
    current_history = _load_history(current_history_path, CURRENT_EPOCHS)
    x_numpy, y_numpy = _load_training_pairs(TRAIN_DATA)

    _, arch = load_checkpoint(CURRENT_RUN / "models", map_location="cpu")
    device = _resolve_device(device_name)
    if reuse_existing:
        if not extended_checkpoint_path.is_file():
            raise FileNotFoundError(extended_checkpoint_path)
        if not extended_summary_path.is_file():
            raise FileNotFoundError(extended_summary_path)
        summary = json.loads(extended_summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("epochs_completed") != epochs
            or summary.get("seed") != 0
            or summary.get("arch") != arch.model_dump(mode="json")
        ):
            raise ValueError(
                f"existing output in {output_dir} is not the requested run"
            )
    else:
        result = train_marcio_full_batch(
            arch=arch,
            x=x_numpy,
            y=y_numpy,
            epochs=epochs,
            learning_rate=LEARNING_RATE,
            seed=0,
            device=device,
            output_dir=output_dir,
            scheduler_factor=0.5,
            scheduler_patience=100,
            scheduler_threshold=1e-4,
            scheduler_min_lr=1e-6,
            verbose=verbose,
        )
        extended_checkpoint_path = result.checkpoint_path
        extended_history_path = result.history_path
    extended_history = _load_history(extended_history_path, epochs)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x = torch.as_tensor(x_numpy, dtype=torch.float32)
    y = torch.as_tensor(y_numpy, dtype=torch.float32)
    current_metrics, current_arch = _evaluate_checkpoint(
        CURRENT_RUN / "models",
        x,
        y,
    )
    extended_metrics, extended_arch = _evaluate_checkpoint(
        output_dir / "models",
        x,
        y,
    )
    if current_arch != extended_arch:
        raise ValueError("current and extended checkpoint architectures differ")

    history_csv = _write_history_csv(
        output_dir / "logs" / "history.csv",
        extended_history,
    )
    figure_outputs = _render_history(
        current_history,
        extended_history,
        output_dir,
    )

    comparison = _comparison(current_metrics, extended_metrics)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "fresh d=1 epoch-count comparison against the canonical 4000-epoch run",
        "training_protocol": {
            "fresh_initialization": True,
            "continued_from_epoch_4000_checkpoint": False,
            "seed": 0,
            "device": str(device),
            "full_batch": True,
            "n_pairs": N_PAIRS,
            "dtype": "float32",
            "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE},
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "factor": 0.5,
                "patience": 100,
                "threshold": 1e-4,
                "threshold_mode": "rel",
                "min_lr": 1e-6,
            },
            "objective": "MSE(D(E(x)), x) + MSE(D(g(E(x))), f(x))",
            "current_epochs": CURRENT_EPOCHS,
            "extended_epochs": epochs,
        },
        "sources": {
            "train_data": {
                "path": str(TRAIN_DATA.resolve()),
                "sha256": data_sha,
            },
            "current_checkpoint": {
                "path": str(current_checkpoint.resolve()),
                "sha256": current_checkpoint_sha,
            },
            "current_history": {
                "path": str(current_history_path.resolve()),
                "sha256": current_history_sha,
            },
        },
        "extended_artifacts": {
            "checkpoint": {
                "path": str(extended_checkpoint_path.resolve()),
                "sha256": _sha256(extended_checkpoint_path),
            },
            "history_json": {
                "path": str(extended_history_path.resolve()),
                "sha256": _sha256(extended_history_path),
            },
            "history_csv": {
                "path": str(history_csv.resolve()),
                "sha256": _sha256(history_csv),
            },
            "figures": {
                path.name: {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
                for path in figure_outputs
            },
        },
        "history_prefix_comparison": _prefix_comparison(
            current_history,
            extended_history,
        ),
        "post_checkpoint_cpu_float32_evaluation_on_all_30000_pairs": {
            "current_epoch_4000": current_metrics,
            "extended_epoch_10000": extended_metrics,
            "comparison": comparison,
        },
        "history_endpoint_pre_update_forward_pass": {
            "current_epoch_4000": {
                key: current_history[key][-1]
                for key in (
                    "loss_reconstruction",
                    "loss_prediction",
                    "loss_total",
                    "learning_rate",
                )
            },
            "extended_epoch_10000": {
                key: extended_history[key][-1]
                for key in (
                    "loss_reconstruction",
                    "loss_prediction",
                    "loss_total",
                    "learning_rate",
                )
            },
        },
        "software": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
        },
    }
    summary_path = output_dir / "epoch_comparison.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="skip training and refresh the audit/plot from a matching saved run",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = run_comparison(
        epochs=args.epochs,
        device_name=args.device,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        reuse_existing=args.reuse_existing,
    )
    evaluation = payload[
        "post_checkpoint_cpu_float32_evaluation_on_all_30000_pairs"
    ]
    print(
        json.dumps(
            {
                "current_epoch_4000": evaluation["current_epoch_4000"],
                "extended_epoch_10000": evaluation["extended_epoch_10000"],
                "comparison": evaluation["comparison"],
                "history_prefix_comparison": payload[
                    "history_prefix_comparison"
                ],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
