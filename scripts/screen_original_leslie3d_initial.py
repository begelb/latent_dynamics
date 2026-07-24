"""Screen the initial subdivision for the paper's original 3D Leslie map.

This deliberately computes only the Morse graph.  Once an initial subdivision
separates the two minimal components, the more expensive Conley-index
calculation can be rerun for that level.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

import CMGDB
import matplotlib


CODE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CMGDB_ROOT = (CODE_ROOT.parent / "archive" / "CMGDB").resolve()
LOWER = [-0.01, -0.01, -0.01]
UPPER = [200.0, 200.0, 200.0]
DEFAULT_SUBDIV_MIN = 33
DEFAULT_SUBDIV_MAX = 39
SUBDIV_LIMIT = 10_000


def f(x: list[float]) -> list[float]:
    return [
        (28.9 * x[0] + 29.8 * x[1] + 22.0 * x[2])
        * math.exp(-0.1 * (x[0] + x[1] + x[2])),
        0.7 * x[0],
        0.7 * x[1],
    ]


def box_map(rect: list[float]) -> list[float]:
    return CMGDB.BoxMap(f, rect, padding=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial", type=int)
    parser.add_argument("--subdiv-min", type=int, default=DEFAULT_SUBDIV_MIN)
    parser.add_argument("--subdiv-max", type=int, default=DEFAULT_SUBDIV_MAX)
    parser.add_argument(
        "--conley",
        action="store_true",
        help="Compute Conley indices and save Morse-set artifacts.",
    )
    args = parser.parse_args()

    module_path = Path(CMGDB.__file__).resolve()
    if LOCAL_CMGDB_ROOT not in module_path.parents:
        raise RuntimeError(
            f"Expected CMGDB below {LOCAL_CMGDB_ROOT}; imported {module_path}"
        )
    if not 1 <= args.initial <= args.subdiv_min <= args.subdiv_max:
        raise ValueError(
            "subdivision levels must satisfy "
            f"1 <= initial <= min <= max, received "
            f"{args.initial}/{args.subdiv_min}/{args.subdiv_max}"
        )

    run_root = (
        CODE_ROOT
        / "output"
        / "original_leslie"
        / (
            "leslie_3d_original_exact_"
            f"s{args.initial}_{args.subdiv_min}_{args.subdiv_max}_bounds_m0p01_200"
        )
    )
    output = run_root / ("conley" if args.conley else "screen")
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = CMGDB.Model(
        args.subdiv_min,
        args.subdiv_max,
        args.initial,
        SUBDIV_LIMIT,
        LOWER,
        UPPER,
        box_map,
    )
    if args.conley:
        morse_graph, _ = CMGDB.ComputeConleyMorseGraph(model)
    else:
        morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    compute_seconds = time.perf_counter() - started

    vertices = list(morse_graph.vertices())
    edges = [
        [int(source), int(target)]
        for source in vertices
        for target in morse_graph.adjacencies(source)
    ]
    minimal = [int(node) for node in vertices if not morse_graph.adjacencies(node)]
    graph = CMGDB.PlotMorseGraph(morse_graph)
    (output / "morse_graph").write_text(graph.source)
    graph.render(str(output / "morse_graph"), format="pdf", view=False, cleanup=False)
    graph.render(str(output / "morse_graph"), format="png", view=False, cleanup=False)
    if args.conley:
        morse_dir = output / "MG"
        morse_dir.mkdir(parents=True, exist_ok=True)
        CMGDB.SaveMorseSets(morse_graph, str(morse_dir / "morse_sets"))
        for proj_dims, suffix, labels in [
            ([0, 1], "x1_x2", ("$x_1$", "$x_2$")),
            ([0, 2], "x1_x3", ("$x_1$", "$x_3$")),
            ([1, 2], "x2_x3", ("$x_2$", "$x_3$")),
        ]:
            CMGDB.PlotMorseSets(
                morse_graph,
                proj_dims=proj_dims,
                cmap=matplotlib.cm.cool,
                axis_labels=True,
                xlabel=labels[0],
                ylabel=labels[1],
                fontsize=18,
                fig_fname=str(output / f"morse_sets_{suffix}"),
                dpi=300,
            )

    manifest = {
        "system": "original 3D Leslie",
        "theta": [28.9, 29.8, 22.0],
        "survival": [0.7, 0.7],
        "bounds": {"lower": LOWER, "upper": UPPER},
        "subdivision": {
            "init": args.initial,
            "min": args.subdiv_min,
            "max": args.subdiv_max,
            "limit": SUBDIV_LIMIT,
        },
        "box_map": "CMGDB.BoxMap(f, rect, padding=False)",
        "cmgdb": {
            "version": version("CMGDB"),
            "module_path": str(module_path),
        },
        "algorithm": (
            "ComputeConleyMorseGraph" if args.conley else "ComputeMorseGraph"
        ),
        "compute_seconds": round(compute_seconds, 3),
        "morse_nodes": len(vertices),
        "edges": edges,
        "minimal_nodes": minimal,
        "minimal_node_count": len(minimal),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
