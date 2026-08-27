#!/usr/bin/env python3
"""Plot a coarsened Morse decomposition with CMGDB, as PDFs.

The coarsening driver leaves two things behind: a box CSV of the coarsened
Morse sets and a Graphviz DOT of the coarsened graph. This turns both into
PDFs -- the sets through ``CMGDB.PlotMorseSets``, the graph through ``dot``.

Splitting this out matters on a CMGDB without ``ComputeConleyIndexForCells``.
The coarsened *sets* are exact on any build, because connection completion is
reachability on the cell graph and needs no homology; only the Conley index
labelling the merged node is unavailable. Plotting from the CSV and the DOT
therefore yields correct set figures either way, and the DOT already carries
whatever label the coarsening recorded -- a real index, or an explicit
"unavailable" marker.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from latentdynamics.viz import plot_morse_sets_2d_cmgdb  # noqa: E402

DEFAULT_DIR = REPO_ROOT / "output" / "leslie3d_example1_study" / "coarsened_45"

#: Match the palette the packaged panels use, so a figure produced here is
#: not distinguishable from one produced by the bundle by colour alone.
PALETTE = ["#ffb000ff", "#dc267fff", "#648fffff", "#fe6100ff", "#785ef0ff", "#008080ff"]


def plot_sets(
    csv_path: Path,
    out_pdf: Path,
    *,
    labels: tuple[str, str],
    scale_factor: list[float] | None = None,
    zoom_nodes: list[int] | None = None,
) -> Path:
    """Draw the Morse-set boxes as rectangles, coloured by node.

    ``zoom_nodes`` adds a magnified inset over those sets instead of inflating
    them, which keeps every box at its true size in both views.
    """
    return plot_morse_sets_2d_cmgdb(
        csv_path,
        out_pdf,
        scale_factor=scale_factor,
        zoom_nodes=zoom_nodes,
        palette=PALETTE,
        xlabel=labels[0],
        ylabel=labels[1],
    )


def plot_graph(dot_path: Path, out_pdf: Path) -> Path:
    """Render the coarsened Morse graph DOT."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["dot", "-Tpdf", str(dot_path), "-o", str(out_pdf)], capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"dot failed: {completed.stderr.strip()}")
    return out_pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR,
                        help="directory holding the coarsening outputs")
    parser.add_argument("--sets-csv", type=Path, default=None)
    parser.add_argument("--graph-dot", type=Path, default=None)
    parser.add_argument("--out-sets", type=Path, default=None)
    parser.add_argument("--out-graph", type=Path, default=None)
    parser.add_argument("--box-scale", type=float, nargs="*", default=None,
                        metavar="FACTOR",
                        help="CMGDB scale_factor, one entry per Morse node")
    parser.add_argument("--zoom-nodes", type=int, nargs="*", default=None,
                        metavar="NODE",
                        help="magnify the region holding these Morse sets in an "
                             "inset, with a box and connector lines")
    parser.add_argument("--xlabel", default="$z_1$")
    parser.add_argument("--ylabel", default="$z_2$")
    args = parser.parse_args(argv)

    sets_csv = args.sets_csv or args.input_dir / "morse_sets_connection_complete.csv"
    graph_dot = args.graph_dot or args.input_dir / "morse_graph_coarse.dot"
    out_sets = args.out_sets or args.input_dir / "morse_sets_coarse.pdf"
    out_graph = args.out_graph or args.input_dir / "morse_graph_coarse.pdf"

    missing = [p for p in (sets_csv, graph_dot) if not p.is_file()]
    if missing:
        raise SystemExit(
            "missing coarsening output: "
            + ", ".join(str(p) for p in missing)
            + "\nRun scripts/leslie3d_example1_coarsen_morse_graph.py first "
              "(add --allow-placeholder-index on a CMGDB without "
              "ComputeConleyIndexForCells)."
        )

    print(f"sets  {sets_csv} -> "
          f"{plot_sets(sets_csv, out_sets, labels=(args.xlabel, args.ylabel), scale_factor=args.box_scale, zoom_nodes=args.zoom_nodes)}")
    print(f"graph {graph_dot} -> {plot_graph(graph_dot, out_graph)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
