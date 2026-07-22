"""Collapse the non-attracting Chafee--Infante graph nodes toward M(1).

This is a graph and visualization postprocessing, not a new CMGDB computation
of a coarse Morse set. The two minimal nodes stay distinct; nodes 2--6 form one fiber.
The resulting map from the seven-node graph onto the three-node graph is a
surjective, order-preserving poset map.  Following the manuscript notation,
``M(1)`` represents the unstable equilibria and their connecting orbits in the
target Morse representation. The saved box visualization contains only the
union of the fine recurrent boxes, not an enclosure of those connections. The
merged node is intentionally not given a Conley index, because that index
cannot be inferred by combining the fine-node annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latentdynamics.analysis.morse_coarsening import (
    coarsen_morse_graph,
    write_morse_graph_dot,
    write_quotient_morse_sets,
)
from latentdynamics.analysis.morse_graph_parser import MorseGraph
from latentdynamics.viz import render_morse_from_files

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = CODE_ROOT / "replay_sources" / "chafee_infante" / "replay" / "MG"
DEFAULT_OUTPUT = CODE_ROOT / "paper_figures" / "coarsened" / "chafee_infante"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--collapse",
        default="2,3,4,5,6",
        help="comma-separated fine Morse nodes to merge (default: 2,3,4,5,6)",
    )
    args = parser.parse_args()

    merged = frozenset(int(value) for value in args.collapse.split(",") if value.strip())
    graph = MorseGraph.from_dot(args.source / "morse_graph")
    quotient = coarsen_morse_graph(
        graph,
        [merged],
        labels={
            frozenset({0}): "M(0-) : (x-1, 0, 0)",
            frozenset({1}): "M(0+) : (x-1, 0, 0)",
            merged: "M(1)",
        },
    )

    morse_dir = args.output / "MG"
    write_morse_graph_dot(quotient.graph, morse_dir / "morse_graph")
    write_quotient_morse_sets(
        args.source / "morse_sets",
        morse_dir / "morse_sets",
        quotient.projection,
    )
    render_morse_from_files(
        morse_dir,
        bounds_lower=[-3.0, -2.0],
        bounds_upper=[3.0, 2.0],
        out_dir=args.output,
        box_scale="auto",
        min_box_side_frac=0.0025,
    )

    try:
        source_description = str(args.source.resolve().relative_to(CODE_ROOT))
    except ValueError:
        source_description = str(args.source)
    manifest = {
        "source": source_description,
        "projection": {str(k): v for k, v in sorted(quotient.projection.items())},
        "fibers": {str(k): sorted(v) for k, v in quotient.fibers.items()},
        "quotient_edges": quotient.graph.edges,
        "merged_morse_set_name": "M(1)",
        "attracting_morse_set_names": {"0": "M(0^-)", "1": "M(0^+)"},
        "merged_morse_set_description": (
            "the unstable equilibria represented by fine nodes 2--6 and their connecting orbits"
        ),
        "box_union_note": (
            "The plotted region is only the union of saved fine Morse-set boxes; "
            "it does not enclose connecting orbits and is not a newly computed "
            "coarse Morse set."
        ),
        "conley_index_note": (
            "No Conley index is assigned to the merged fiber; it must be "
            "recomputed from an index pair for the union."
        ),
        "rendering": {
            "min_box_side_frac": 0.0025,
            "display_only": True,
        },
    }
    (args.output / "quotient.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"projection: {manifest['projection']}")
    print(f"quotient edges: {quotient.graph.edges}")
    print(f"artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
