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
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]

import CMGDB  # noqa: E402
import matplotlib  # noqa: E402

from latentdynamics.analysis.cmgdb_fork import (  # noqa: E402
    cmgdb_provenance,
    require_fork_cmgdb,
)

WIDE_CUBE_LOWER = [-0.01, -0.01, -0.01]
WIDE_CUBE_UPPER = [200.0, 200.0, 200.0]
PAPER_LOWER = [0.0, 0.0, 0.0]
PAPER_UPPER = [220.0, 154.0, 108.0]
ABSORBING_LOWER = [0.0, 0.0, 0.0]
ABSORBING_UPPER = [110.0, 77.0, 54.0]
DEFAULT_SUBDIV_MIN = 33
DEFAULT_SUBDIV_MAX = 39
DEFAULT_SUBDIV_LIMIT = 10_000


def f(x: list[float]) -> list[float]:
    return [
        (28.9 * x[0] + 29.8 * x[1] + 22.0 * x[2])
        * math.exp(-0.1 * (x[0] + x[1] + x[2])),
        0.7 * x[0],
        0.7 * x[1],
    ]


def box_map(rect: list[float]) -> list[float]:
    return CMGDB.BoxMap(f, rect, padding=False)



def count_saved_morse_boxes(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            node = str(int(line.rsplit(",", 1)[1]))
            counts[node] = counts.get(node, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial", type=int)
    parser.add_argument("--subdiv-min", type=int, default=DEFAULT_SUBDIV_MIN)
    parser.add_argument("--subdiv-max", type=int, default=DEFAULT_SUBDIV_MAX)
    parser.add_argument("--subdiv-limit", type=int, default=DEFAULT_SUBDIV_LIMIT)
    parser.add_argument(
        "--domain",
        choices=("wide_cube", "paper", "absorbing"),
        default="wide_cube",
    )
    parser.add_argument("--lower-bounds", type=float, nargs=3)
    parser.add_argument("--upper-bounds", type=float, nargs=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--conley",
        action="store_true",
        help="Compute Conley indices and save Morse-set artifacts.",
    )
    args = parser.parse_args()

    require_fork_cmgdb()
    if not 1 <= args.initial <= args.subdiv_min <= args.subdiv_max:
        raise ValueError(
            "subdivision levels must satisfy "
            f"1 <= initial <= min <= max, received "
            f"{args.initial}/{args.subdiv_min}/{args.subdiv_max}"
        )
    if args.subdiv_limit < 1:
        raise ValueError("subdiv-limit must be positive")
    if (args.lower_bounds is None) != (args.upper_bounds is None):
        raise ValueError("lower-bounds and upper-bounds must be supplied together")

    if args.lower_bounds is not None and args.upper_bounds is not None:
        lower = args.lower_bounds
        upper = args.upper_bounds
        domain_suffix = "bounds_custom"
    elif args.domain == "absorbing":
        lower = ABSORBING_LOWER
        upper = ABSORBING_UPPER
        domain_suffix = "bounds_absorbing_B"
    elif args.domain == "paper":
        lower = PAPER_LOWER
        upper = PAPER_UPPER
        domain_suffix = "bounds_paper_X"
    else:
        lower = WIDE_CUBE_LOWER
        upper = WIDE_CUBE_UPPER
        domain_suffix = "bounds_m0p01_200"
    if any(
        lower_value >= upper_value
        for lower_value, upper_value in zip(lower, upper, strict=True)
    ):
        raise ValueError(f"each lower bound must be below its upper bound: {lower} -> {upper}")

    if args.output_dir is None:
        limit_suffix = (
            ""
            if args.subdiv_limit == DEFAULT_SUBDIV_LIMIT
            else f"_limit{args.subdiv_limit}"
        )
        run_root = (
            CODE_ROOT
            / "output"
            / "original_leslie"
            / (
                "leslie_3d_original_exact_"
                f"s{args.initial}_{args.subdiv_min}_{args.subdiv_max}_"
                f"{domain_suffix}{limit_suffix}"
            )
        )
    else:
        run_root = (
            args.output_dir
            if args.output_dir.is_absolute()
            else (CODE_ROOT / args.output_dir).resolve()
        )
    output = run_root / ("conley" if args.conley else "screen")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)

    resolved_config = {
        "system": "original 3D Leslie",
        "theta": [28.9, 29.8, 22.0],
        "survival": [0.7, 0.7],
        "bounds": {"lower": lower, "upper": upper},
        "subdivision": {
            "init": args.initial,
            "min": args.subdiv_min,
            "max": args.subdiv_max,
            "limit": args.subdiv_limit,
        },
        "box_map": "CMGDB.BoxMap(f, rect, padding=False)",
        "algorithm": (
            "ComputeConleyMorseGraph" if args.conley else "ComputeMorseGraph"
        ),
        "output": str(output),
        "cmgdb": cmgdb_provenance(),
    }
    (output / "run_config.json").write_text(
        json.dumps(resolved_config, indent=2) + "\n"
    )
    print(json.dumps(resolved_config, indent=2), flush=True)

    started = time.perf_counter()
    model = CMGDB.Model(
        args.subdiv_min,
        args.subdiv_max,
        args.initial,
        args.subdiv_limit,
        lower,
        upper,
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
    morse_dir = output / "MG"
    morse_dir.mkdir(parents=True)
    morse_sets_path = morse_dir / "morse_sets"
    CMGDB.SaveMorseSets(morse_graph, str(morse_sets_path))
    boxes_per_node = count_saved_morse_boxes(morse_sets_path)
    guaranteed_first_post_min_limit_stops = (
        sorted(
            [
                node
                for node, count in boxes_per_node.items()
                if 2 * count > args.subdiv_limit
            ],
            key=int,
        )
        if args.subdiv_max > args.subdiv_min
        else []
    )
    if args.conley:
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
        **resolved_config,
        "compute_seconds": round(compute_seconds, 3),
        "morse_nodes": len(vertices),
        "edges": edges,
        "minimal_nodes": minimal,
        "minimal_node_count": len(minimal),
        "morse_boxes_per_node": boxes_per_node,
        "morse_boxes_total": sum(boxes_per_node.values()),
        "subdivision_diagnostics": {
            "requested_post_min_levels": args.subdiv_max - args.subdiv_min,
            "nodes_guaranteed_to_exceed_limit_before_first_post_min_decomposition": (
                guaranteed_first_post_min_limit_stops
            ),
        },
        "conley_indices": (
            {str(int(node)): list(morse_graph.annotations(node)) for node in vertices}
            if args.conley
            else None
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
