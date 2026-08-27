#!/usr/bin/env python3
"""Plot the coarse leslie3d_example1 Morse decomposition, as PDFs.

The coarse recomputation can resolve more Morse sets than the adaptive run --
a grid pinned at (22,22,22) reports 24, most of them one-cell components of
trivial index. What the manuscript shows either way is the *nontrivial
skeleton*: the sets carrying a non-trivial Conley index and the edges between
them. That skeleton is written as its own DOT by the grid driver, so the nodes
to draw and the colour each carries are read from there, not hard-coded here.

This exists because the packaged bundle that used to produce these two panels
needs ``ComputeConleyIndexForCells``, which upstream CMGDB does not expose. The
sets and the skeleton graph need no homology -- both were already computed and
written by the coarse-grid step -- so plotting them directly makes the panels
available on any build.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from latentdynamics.viz import plot_morse_sets_2d_cmgdb  # noqa: E402

DEFAULT_DIR = REPO_ROOT / "output" / "leslie3d_example1_study" / "fixed22"
DEFAULT_OUT = REPO_ROOT / "output" / "leslie3d_example1_figures"

#: Fallback when a skeleton node carries no explicit fillcolor.
FALLBACK_COLOR = "#999999ff"

_NODE_RE = re.compile(
    r'^\s*(\d+)\s*\[label="[^"]*"[^]]*fillcolor="(#[0-9A-Fa-f]{6,8})"', re.MULTILINE
)


def skeleton_nodes(dot_path: Path) -> dict[int, str]:
    """Map node -> fill colour for every node declared in a skeleton DOT."""
    text = dot_path.read_text(encoding="utf-8")
    found = {int(n): c for n, c in _NODE_RE.findall(text)}
    if not found:
        raise SystemExit(f"no node declarations parsed from {dot_path}")
    return found


def palette_for(nodes: dict[int, str]) -> list[str]:
    """Colour list indexed by Morse node, as CMGDB indexes it.

    CMGDB builds a ListedColormap from the list and looks up entry ``k`` for
    Morse set ``k``, so the list must be dense up to the highest node drawn --
    gaps are nodes that exist in the CSV but are not plotted.
    """
    palette = [FALLBACK_COLOR] * (max(nodes) + 1)
    for node, colour in nodes.items():
        palette[node] = colour if len(colour) == 9 else colour + "ff"
    return palette


def plot_graph(dot_path: Path, out_pdf: Path) -> Path:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["dot", "-Tpdf", str(dot_path), "-o", str(out_pdf)], capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"dot failed: {completed.stderr.strip()}")
    return out_pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--depth", type=int, default=22)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--box-scale", type=float, nargs="*", default=None, metavar="FACTOR")
    parser.add_argument("--zoom-nodes", type=int, nargs="*", default=None, metavar="NODE")
    parser.add_argument("--xlabel", default="$z_1$")
    parser.add_argument("--ylabel", default="$z_2$")
    args = parser.parse_args(argv)

    stem = f"fixed{args.depth}"
    sets_csv = args.input_dir / f"morse_sets_{stem}_connection_complete.csv"
    graph_dot = args.input_dir / f"morse_graph_{stem}_nontrivial_skeleton.dot"
    missing = [p for p in (sets_csv, graph_dot) if not p.is_file()]
    if missing:
        raise SystemExit(
            "missing coarse-grid output: " + ", ".join(str(p) for p in missing)
            + f"\nRun scripts/leslie3d_example1_uniform_grid.py --depth {args.depth} first."
        )

    nodes = skeleton_nodes(graph_dot)
    drawn = sorted(nodes)
    print(f"nontrivial nodes: {drawn}")

    out_sets = args.out_dir / f"morse_sets_coarse{args.depth}.pdf"
    out_graph = args.out_dir / f"morse_graph_coarse{args.depth}.pdf"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = plot_morse_sets_2d_cmgdb(
        sets_csv,
        out_sets,
        scale_factor=args.box_scale,
        morse_nodes=drawn,
        zoom_nodes=args.zoom_nodes,
        palette=palette_for(nodes),
        xlabel=args.xlabel,
        ylabel=args.ylabel,
    )
    print(f"sets  {sets_csv} -> {written}")
    print(f"graph {graph_dot} -> {plot_graph(graph_dot, out_graph)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
