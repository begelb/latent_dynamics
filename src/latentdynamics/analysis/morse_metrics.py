"""Unique-membership metric for paper-named coral fixed points (a0, a1, r)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydot
import torch

from ..systems.coral import RedCoralModel


def get_minimal_labels(morse_graph_dot: str | Path) -> set[int]:
    """Sink labels (no outgoing edges) in a Morse graph stored as graphviz DOT."""
    graphs = pydot.graph_from_dot_file(str(morse_graph_dot))
    if not graphs:
        return set()
    G = graphs[0]
    nodes = {n.get_name() for n in G.get_nodes() if n.get_name().lstrip("-").isdigit()}
    sources = {e.get_source() for e in G.get_edges()}
    return {int(n) for n in nodes - sources}


def find_morse_label_1d(z: float, morse_df: pd.DataFrame) -> int | None:
    """Return the label of the 1-D Morse interval containing ``z``, or None."""
    for _, row in morse_df.iterrows():
        if row["a"] <= z <= row["b"]:
            return int(row["label"])
    return None


def _read_morse_sets_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "a" not in df.columns:
        df = pd.read_csv(path, names=["a", "b", "label"])
    return df


def check_unique_membership(
    encoder: torch.nn.Module,
    scaler,
    morse_sets_path: str | Path,
    morse_graph_path: str | Path,
    *,
    fixed_points: dict[str, np.ndarray] | None = None,
    device: torch.device | None = None,
) -> tuple[dict[str, int | None], dict[str, bool]]:
    """For each named fixed point, decide whether ``E(p)`` lands uniquely in a Morse set
    of the expected sink/source type.

    For ``a0`` and ``a1`` (sinks), the assigned label must be a sink in the
    Morse graph; for ``r`` (saddle), the label must NOT be a sink.
    """
    fixed_points = fixed_points if fixed_points is not None else RedCoralModel.FIXED_POINTS
    device = device or next(encoder.parameters()).device
    morse_df = _read_morse_sets_csv(morse_sets_path)
    minimal = get_minimal_labels(morse_graph_path)

    encoder.eval()
    labels: dict[str, int | None] = {}
    with torch.no_grad():
        for name, pt in fixed_points.items():
            scaled = scaler.transform([pt])
            z = encoder(torch.as_tensor(scaled, dtype=torch.float32, device=device))
            z_val = float(z.cpu().numpy().flatten()[0])
            labels[name] = find_morse_label_1d(z_val, morse_df)

    metrics: dict[str, bool] = {}
    for name in fixed_points:
        my = labels[name]
        if my is None:
            metrics[name] = False
            continue
        unique = not any(labels[other] == my for other in fixed_points if other != name)
        if name in ("a0", "a1"):
            metrics[name] = unique and (my in minimal)
        else:
            metrics[name] = unique and (my not in minimal)
    return labels, metrics


def find_seed_subdirs(base_output_dir: str | Path) -> list[str]:
    """List subdirectories under ``base_output_dir`` that look like complete runs."""
    base = Path(base_output_dir)
    out: list[str] = []
    if not base.is_dir():
        return out
    for entry in sorted(base.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        encoder_old = (entry / "models" / "encoder.pt").exists()
        encoder_new = (entry / "models" / "autoencoder.pt").exists()
        morse_sets = (entry / "MG" / "morse_sets").exists()
        morse_graph = (entry / "MG" / "morse_graph").exists()
        if (encoder_old or encoder_new) and morse_sets and morse_graph:
            out.append(entry.name)
    return out
