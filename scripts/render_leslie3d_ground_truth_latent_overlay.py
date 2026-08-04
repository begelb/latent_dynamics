#!/usr/bin/env python3
"""Render direct Leslie ground truth over the accepted max-30 latent Morse sets.

The learned max-30 Morse boxes are the background. Two direct-system objects
are superimposed after applying the archived scaler and accepted encoder:

* centers of the level-24 display cover derived from the saved level-33 direct
  CMGDB Morse boxes; and
* the exact fixed and periodic points used by the invariant-aware audit.

Encoding display-cover centers is a visualization, not an enclosing image of
the full three-dimensional boxes and not a Conley-index certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import torch
from matplotlib.lines import Line2D
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from latentdynamics.config import load_config
from latentdynamics.training import load_any_checkpoint
from latentdynamics.viz.morse_plots import plot_morse_sets_from_csv

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "leslie3d_invariant_aware_v2_smooth_max30"
MAX30_ROOT = (
    CODE_ROOT
    / "output"
    / "notebooks"
    / "leslie3d_invariant_aware_v2_smooth_max30"
    / "seed_20260809"
)
LEARNED_MORSE_SETS = MAX30_ROOT / "MG" / "morse_sets"
CHECKPOINT = MAX30_ROOT / "models" / "autoencoder.pt"
ENCODED_INVARIANTS = MAX30_ROOT / "analysis" / "encoded_invariant_points.csv"
DIRECT_DISPLAY_COVER = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "ground_truth"
    / "absorbing_B_uniform_level33_recurrent_closure"
    / "cubical_3d_level24_display_cover"
    / "morse_sets_level24_display_cover.csv"
)
DIRECT_DISPLAY_MANIFEST = DIRECT_DISPLAY_COVER.parent / "manifest.json"
DIRECT_RAW_MORSE_SETS = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "ground_truth"
    / "absorbing_B_i29_m33_M36_L10000"
    / "screen"
    / "MG"
    / "morse_sets"
)
SCALER = CODE_ROOT / "replay_sources" / "leslie3d_example2" / "data" / "scalers" / "scaler"
DEFAULT_OUTPUT = (
    CODE_ROOT.parent / "output" / "pdf" / "leslie3d_morse_report" / "assets"
)

EXPECTED_HASHES = {
    LEARNED_MORSE_SETS: "faf4de21228ccf331d69e37b7e22de073569135a3d41b35b5cd86feb11b2fd63",
    CHECKPOINT: "9fbee2cde690d58d2413c0d3521763838abaeb493736120f7173035612da3f3d",
    ENCODED_INVARIANTS: "9b3f511a495b427c82e8daa3f2185f8cfd4e28125b1fd7149e698df325097c16",
    DIRECT_DISPLAY_COVER: "1726db63abffbc4a0984d61591f1fbacd0291187301271e71fdbed474c3c2b29",
    SCALER: "bb908b946d259fd6aa6a716cc003f789631e21bc7c9aa0a6a64c09ac629aa5e1",
}

OBJECT_ORDER = ("P0", "P1", "S2", "S4", "p_star", "origin")
DIRECT_NODE_TO_OBJECT = dict(enumerate(OBJECT_ORDER))
OBJECT_STYLE = {
    "P0": {"color": "#FFB000", "marker": "o", "label": r"direct $P_0$"},
    "P1": {"color": "#DC267F", "marker": "s", "label": r"direct $P_1$"},
    "S2": {"color": "#FE6100", "marker": "^", "label": r"direct $S_2$"},
    "S4": {"color": "#648FFF", "marker": "D", "label": r"direct $S_4$"},
    "p_star": {"color": "#785EF0", "marker": "*", "label": r"direct $p_*$"},
    "origin": {"color": "#008080", "marker": "X", "label": r"direct origin"},
}
LATENT_ASSIGNMENT = {
    "P0": 0,
    "P1": 4,
    "S2": 2,
    "S4": 5,
    "p_star": 3,
    "origin": 7,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"input hash mismatch for {path}: {actual} != {expected}")
        observed[str(path)] = actual
    direct_manifest = json.loads(DIRECT_DISPLAY_MANIFEST.read_text(encoding="utf-8"))
    raw_expected = direct_manifest["source"]["morse_sets"]["sha256"]
    raw_actual = _sha256(DIRECT_RAW_MORSE_SETS)
    if raw_actual != raw_expected:
        raise ValueError(
            f"direct raw Morse-set hash mismatch: {raw_actual} != {raw_expected}"
        )
    observed[str(DIRECT_RAW_MORSE_SETS)] = raw_actual
    return observed


def _load_exact_invariants() -> dict[str, NDArray[np.float64]]:
    grouped: dict[str, list[tuple[int, tuple[float, float]]]] = {
        name: [] for name in OBJECT_ORDER
    }
    with ENCODED_INVARIANTS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["object"]
            if name not in grouped:
                continue
            grouped[name].append(
                (int(row["phase"]), (float(row["z0"]), float(row["z1"])))
            )
    result: dict[str, NDArray[np.float64]] = {}
    for name in OBJECT_ORDER:
        rows = sorted(grouped[name])
        if not rows:
            raise ValueError(f"missing encoded direct invariant object {name}")
        result[name] = np.asarray([point for _phase, point in rows], dtype=np.float64)
    return result


@torch.no_grad()
def _encode_display_cover() -> tuple[dict[str, NDArray[np.float64]], dict[str, int]]:
    cfg = load_config(CONFIG_NAME)
    model, _ = load_any_checkpoint(
        MAX30_ROOT / "models",
        arch=cfg.arch,
        basename="autoencoder",
    )
    model.eval().cpu()
    scaler = joblib.load(SCALER)

    cover = np.loadtxt(DIRECT_DISPLAY_COVER, delimiter=",", ndmin=2)
    if cover.shape[1] != 7:
        raise ValueError(f"expected seven direct-cover columns, got {cover.shape}")
    labels = np.rint(cover[:, 6]).astype(np.int64)
    if not np.array_equal(cover[:, 6], labels):
        raise ValueError("direct-cover labels are not integral")
    centers = 0.5 * (cover[:, :3] + cover[:, 3:6])
    scaled = scaler.transform(centers)
    encoded_chunks: list[NDArray[np.float64]] = []
    for start in range(0, len(scaled), 16384):
        values = torch.as_tensor(scaled[start : start + 16384], dtype=torch.float32)
        encoded_chunks.append(model.encoder(values).cpu().numpy().astype(np.float64))
    encoded = np.vstack(encoded_chunks)
    grouped = {
        DIRECT_NODE_TO_OBJECT[node]: encoded[labels == node]
        for node in sorted(DIRECT_NODE_TO_OBJECT)
    }
    counts = {name: int(points.shape[0]) for name, points in grouped.items()}
    return grouped, counts


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=OBJECT_STYLE[name]["color"],
            marker=OBJECT_STYLE[name]["marker"],
            markerfacecolor="white" if name != "origin" else OBJECT_STYLE[name]["color"],
            markeredgecolor=OBJECT_STYLE[name]["color"],
            markeredgewidth=1.2,
            linewidth=0.0,
            label=OBJECT_STYLE[name]["label"],
        )
        for name in OBJECT_ORDER
    ]


def _render(output_dir: Path) -> dict[str, Any]:
    input_hashes = _verify_inputs()
    cover_encoded, cover_counts = _encode_display_cover()
    exact_encoded = _load_exact_invariants()

    plot = plot_morse_sets_from_csv(
        LEARNED_MORSE_SETS,
        box_scale="auto",
        paper_style=True,
    )
    fig, ax = plot.fig, plot.ax
    fig.set_size_inches(9.7, 7.0)
    for collection in ax.collections:
        collection.set_alpha(0.64)

    for name in OBJECT_ORDER:
        style = OBJECT_STYLE[name]
        cloud = cover_encoded[name]
        ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            s=6.0 if name == "P1" else 13.0,
            marker=".",
            color=style["color"],
            alpha=0.22 if name == "P1" else 0.48,
            linewidths=0.0,
            rasterized=True,
            zorder=18,
        )
        exact = exact_encoded[name]
        ax.scatter(
            exact[:, 0],
            exact[:, 1],
            s=58 if name != "p_star" else 110,
            marker=style["marker"],
            facecolor="white" if name != "origin" else style["color"],
            edgecolor=style["color"],
            linewidth=1.5,
            zorder=26,
        )

    all_overlay = np.vstack(
        [*cover_encoded.values(), *exact_encoded.values()]
    )
    boxes = plot.data[:, :4]
    lower = np.minimum(boxes[:, :2].min(axis=0), all_overlay.min(axis=0))
    upper = np.maximum(boxes[:, 2:4].max(axis=0), all_overlay.max(axis=0))
    padding = 0.035 * np.maximum(upper - lower, 1e-6)
    ax.set_xlim(float(lower[0] - padding[0]), float(upper[0] + padding[0]))
    ax.set_ylim(float(lower[1] - padding[1]), float(upper[1] + padding[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Max-30 latent Morse sets with encoded direct-system ground truth",
        pad=12,
    )
    ax.legend(
        handles=_legend_handles(),
        title="Direct objects encoded by E",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=1,
    )
    ax.text(
        0.01,
        0.015,
        "Background: learned max-30 Morse boxes\n"
        "Dots: encoded centers of direct-map display-cover cells\n"
        "Outlined symbols: exact direct fixed/periodic phases",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3.0},
        zorder=40,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "direct_ground_truth_on_max30_morse_sets"
    outputs: dict[str, dict[str, Any]] = {}
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", dpi=320)
        outputs[path.name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    plt.close(fig)

    provenance = {
        "schema_version": 1,
        "purpose": "direct Leslie ground truth over accepted max-30 latent Morse sets",
        "configuration": CONFIG_NAME,
        "input_sha256": input_hashes,
        "direct_display_cover_counts": cover_counts,
        "direct_display_cover_source_box_count": 1_955_948,
        "latent_role_assignment": LATENT_ASSIGNMENT,
        "method": {
            "learned_background": "all saved max-30 latent Morse boxes",
            "direct_set_overlay": (
                "centers of level-24 parent cells derived from saved level-33 "
                "direct-map Morse boxes, scaled and encoded by E"
            ),
            "exact_object_overlay": (
                "the 16 direct-system fixed/periodic phases from the invariant-aware manifest"
            ),
        },
        "limitations": [
            "encoded display-cover centers are a sampled visualization, not an enclosing image of each 3-D box under E",
            "the display cover is render-only and is not a recomputed level-24 Morse decomposition",
            "the overlay is numerical evidence and not a semiconjugacy or Conley-index certificate",
        ],
        "outputs": outputs,
    }
    provenance_path = output_dir / "direct_ground_truth_overlay_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else CODE_ROOT / args.output_dir
    print(json.dumps(_render(output_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
