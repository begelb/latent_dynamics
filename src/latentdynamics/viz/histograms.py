"""Histogram of total final populations across trajectories (paper coral Fig. 1.469)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import PALETTE, apply_paper_style


def plot_final_population_histogram(
    csv_path: str | Path,
    save_path: str | Path,
    *,
    steps_per_trajectory: int,
    ymax: float | None = None,
    style: bool = True,
) -> Path:
    """Histogram of summed populations at the last step of each trajectory block."""
    csv_path = Path(csv_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    y_cols = [col for col in df.columns if col.startswith("y")]
    final_states = df.iloc[steps_per_trajectory - 1 :: steps_per_trajectory]
    final_populations = final_states[y_cols].sum(axis=1).to_numpy()

    if style:
        apply_paper_style()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_yscale("log")
    ax.hist(
        final_populations,
        bins=10,
        color=PALETTE[2],
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_xlabel("Total final population")
    ax.set_ylabel("Number of trajectories")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
    if ymax is not None:
        ax.set_ylim(top=ymax)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def steps_per_trajectory_from_metadata(metadata_path: str | Path) -> int:
    """Read ``n_iterations`` from the metadata JSON sidecar."""
    with Path(metadata_path).open() as f:
        meta = json.load(f)
    return int(meta["n_iterations"])


def _resolved_save_path(out_dir: Path, train_file: str) -> Path:
    return out_dir / f"histogram_{train_file}.pdf"


def emit_population_histogram(
    data_dir: str | Path,
    train_file: str,
    out_dir: str | Path,
    ymax: float | None = None,
) -> Path:
    """Convenience wrapper used by scripts: read metadata, render histogram."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    csv_path = data_dir / f"{train_file}.csv"
    metadata_path = data_dir / f"{train_file}_metadata.json"
    steps = steps_per_trajectory_from_metadata(metadata_path)
    save_path = _resolved_save_path(out_dir, train_file)
    return plot_final_population_histogram(
        csv_path, save_path, steps_per_trajectory=steps, ymax=ymax
    )


# silence unused-import lint for np (kept for future use)
_ = np
