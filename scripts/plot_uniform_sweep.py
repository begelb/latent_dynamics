"""Render morse_graph + morse_sets PDFs/PNGs for every (mode, smax) in the
uniform-grid sweep at output/leslie2d_to_2d_uniform_nopad.

For each leaf directory containing a saved `morse_graph` (DOT, no extension)
and `morse_sets` (CSV with .csv extension), we produce sibling `.pdf` and
`.png` files via:
  - graphviz `dot -Tpdf`/`-Tpng` for the morse_graph
  - CMGDB.LoadMorseSetFile + matplotlib scatter for the morse_sets
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import CMGDB

CODE_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = CODE_ROOT / "output/leslie2d_to_2d_uniform_nopad"
PROFILE_JSON = SWEEP_ROOT / "profile_results.json"


def render_morse_graph_dot(dot_path: Path) -> None:
    """Render `dot_path` (graphviz dot source) to .pdf and .png siblings."""
    for fmt in ("pdf", "png"):
        out = dot_path.with_suffix(f".{fmt}")
        subprocess.run(
            ["dot", f"-T{fmt}", str(dot_path), "-o", str(out)],
            check=True,
        )


def render_morse_sets(csv_path: Path, bounds_lower, bounds_upper, title: str) -> None:
    # CMGDB stores morse_sets as a no-extension CSV file. LoadMorseSetFile
    # appends ".csv" internally, so we strip any trailing extension before passing.
    csv_stem = str(csv_path)
    if not Path(csv_stem).exists() and not Path(csv_stem + ".csv").exists():
        return
    morse_sets = CMGDB.LoadMorseSetFile(csv_stem) if Path(csv_stem).exists() else CMGDB.LoadMorseSetFile(csv_stem)
    if not morse_sets:
        return
    fig_path_pdf = Path(csv_stem + ".pdf")
    fig_path_png = Path(csv_stem + ".png")

    fig, ax = plt.subplots(figsize=(7, 6))
    clist = [
        "#1f77b4", "#e6550d", "#31a354", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#80b1d3",
    ]
    rect0 = morse_sets[0]
    dim = (len(rect0) - 1) // 2
    assert dim == 2, f"Plot script assumes 2D, got dim={dim}"

    for box in morse_sets:
        x_lo, y_lo, x_hi, y_hi, node = box
        cx = 0.5 * (x_lo + x_hi)
        cy = 0.5 * (y_lo + y_hi)
        ax.add_patch(
            plt.Rectangle(
                (x_lo, y_lo),
                x_hi - x_lo,
                y_hi - y_lo,
                facecolor=clist[int(node) % len(clist)],
                edgecolor="none",
                alpha=0.85,
            )
        )
    ax.set_xlim(bounds_lower[0], bounds_upper[0])
    ax.set_ylim(bounds_lower[1], bounds_upper[1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(fig_path_pdf, dpi=160, bbox_inches="tight")
    fig.savefig(fig_path_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    profile = json.loads(PROFILE_JSON.read_text())
    a_lower = profile["ambient_bounds"]["lower"]
    a_upper = profile["ambient_bounds"]["upper"]
    l_lower = profile["latent_bounds"]["lower"]
    l_upper = profile["latent_bounds"]["upper"]

    n_rendered = 0
    for row in profile["sweep"]:
        smax = row["subdiv_max"]
        for mode in ("analytic", "latent"):
            entry = row[mode]
            n_nodes = entry["morse_nodes"]
            n_min = entry["morse_minimal_nodes"]
            n_boxes = entry["morse_total_boxes"]
            dot_path = Path(entry["dot"])
            csv_path = Path(entry["csv"])

            if dot_path.exists():
                render_morse_graph_dot(dot_path)
                n_rendered += 1
            else:
                print(f"missing DOT: {dot_path}")

            if mode == "analytic":
                bounds_l, bounds_u = a_lower, a_upper
            else:
                bounds_l, bounds_u = l_lower, l_upper

            title = (
                f"{mode}, uniform smax={smax}, nodes={n_nodes}, "
                f"minimal={n_min}, boxes={n_boxes}"
            )
            render_morse_sets(csv_path, bounds_l, bounds_u, title)
            n_rendered += 1

    print(f"Rendered {n_rendered} artefacts under {SWEEP_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
