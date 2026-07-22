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


CODE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CMGDB_ROOT = (CODE_ROOT.parent / "archive" / "CMGDB").resolve()
LOWER = [-0.01, -0.01, -0.01]
UPPER = [200.0, 200.0, 200.0]
SUBDIV_MIN = 33
SUBDIV_MAX = 39
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
    args = parser.parse_args()

    module_path = Path(CMGDB.__file__).resolve()
    if LOCAL_CMGDB_ROOT not in module_path.parents:
        raise RuntimeError(
            f"Expected CMGDB below {LOCAL_CMGDB_ROOT}; imported {module_path}"
        )
    if not 1 <= args.initial <= SUBDIV_MIN:
        raise ValueError(f"initial must lie in [1,{SUBDIV_MIN}]")

    output = (
        CODE_ROOT
        / "output"
        / "original_leslie"
        / f"leslie_3d_original_exact_s{args.initial}_33_39_bounds_m0p01_200"
        / "screen"
    )
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = CMGDB.Model(
        SUBDIV_MIN,
        SUBDIV_MAX,
        args.initial,
        SUBDIV_LIMIT,
        LOWER,
        UPPER,
        box_map,
    )
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

    manifest = {
        "system": "original 3D Leslie",
        "theta": [28.9, 29.8, 22.0],
        "survival": [0.7, 0.7],
        "bounds": {"lower": LOWER, "upper": UPPER},
        "subdivision": {
            "init": args.initial,
            "min": SUBDIV_MIN,
            "max": SUBDIV_MAX,
            "limit": SUBDIV_LIMIT,
        },
        "box_map": "CMGDB.BoxMap(f, rect, padding=False)",
        "cmgdb": {
            "version": version("CMGDB"),
            "module_path": str(module_path),
        },
        "algorithm": "ComputeMorseGraph",
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
