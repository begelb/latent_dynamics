"""Fine original-system Leslie computation requested on 2026-07-22."""

from __future__ import annotations

import json
import math
import time
from importlib.metadata import version
from pathlib import Path

import CMGDB
import matplotlib


OUTPUT = Path(__file__).resolve().parent
LOWER = [-0.01, -0.01, -0.01]
UPPER = [200.0, 200.0, 200.0]
SUBDIV_INIT = 24
SUBDIV_MIN = 33
SUBDIV_MAX = 39
SUBDIV_LIMIT = 10_000


def f(x):
    theta_1, theta_2, theta_3 = 28.9, 29.8, 22.0
    return [
        (theta_1 * x[0] + theta_2 * x[1] + theta_3 * x[2])
        * math.exp(-0.1 * (x[0] + x[1] + x[2])),
        0.7 * x[0],
        0.7 * x[1],
    ]


def F(rect):
    return CMGDB.BoxMap(f, rect, padding=False)


def main() -> None:
    module_path = Path(CMGDB.__file__).resolve()
    expected_root = Path(__file__).resolve().parents[4] / "archive" / "CMGDB"
    if expected_root not in module_path.parents:
        raise RuntimeError(f"Expected local CMGDB below {expected_root}; imported {module_path}")

    print(f"CMGDB module: {module_path}", flush=True)
    print(f"CMGDB version: {version('CMGDB')}", flush=True)
    print(f"bounds: {LOWER} -> {UPPER}", flush=True)
    print(
        f"subdivisions: init={SUBDIV_INIT}, min={SUBDIV_MIN}, max={SUBDIV_MAX}, "
        f"limit={SUBDIV_LIMIT}",
        flush=True,
    )
    started = time.perf_counter()
    model = CMGDB.Model(
        SUBDIV_MIN,
        SUBDIV_MAX,
        SUBDIV_INIT,
        SUBDIV_LIMIT,
        LOWER,
        UPPER,
        F,
    )
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    compute_seconds = time.perf_counter() - started
    print(
        f"CMGDB complete after {compute_seconds / 60:.2f} minutes; "
        f"nodes={morse_graph.num_vertices()}",
        flush=True,
    )

    graph = CMGDB.PlotMorseGraph(morse_graph, cmap=matplotlib.cm.cool)
    graph.render(str(OUTPUT / "morse_graph"), format="pdf", view=False, cleanup=False)
    graph.render(str(OUTPUT / "morse_graph"), format="png", view=False, cleanup=False)
    CMGDB.SaveMorseSets(morse_graph, str(OUTPUT / "MG" / "morse_sets"))
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
            fig_fname=str(OUTPUT / f"morse_sets_{suffix}"),
            dpi=300,
        )

    manifest = {
        "system": "original 3D Leslie",
        "theta": [28.9, 29.8, 22.0],
        "survival": [0.7, 0.7],
        "bounds": {"lower": LOWER, "upper": UPPER},
        "subdivision": {
            "init": SUBDIV_INIT,
            "min": SUBDIV_MIN,
            "max": SUBDIV_MAX,
            "limit": SUBDIV_LIMIT,
        },
        "box_map": "CMGDB.BoxMap(f, rect, padding=False)",
        "cmgdb": {"version": version("CMGDB"), "module_path": str(module_path)},
        "compute_seconds": round(compute_seconds, 3),
        "morse_nodes": morse_graph.num_vertices(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
