"""Conley-index variant of the exact fine Leslie computation."""

from __future__ import annotations

import json
import time
from importlib.metadata import version
from pathlib import Path

import CMGDB
import matplotlib

from run_exact import F, LOWER, SUBDIV_INIT, SUBDIV_LIMIT, SUBDIV_MAX, SUBDIV_MIN, UPPER


OUTPUT = Path(__file__).resolve().parent / "conley"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "MG").mkdir(parents=True, exist_ok=True)
    module_path = Path(CMGDB.__file__).resolve()
    print(f"CMGDB module: {module_path}", flush=True)
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
    morse_graph, _ = CMGDB.ComputeConleyMorseGraph(model)
    compute_seconds = time.perf_counter() - started
    print(
        f"CMGDB Conley computation complete after {compute_seconds / 60:.2f} minutes; "
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
        "algorithm": "ComputeConleyMorseGraph",
        "compute_seconds": round(compute_seconds, 3),
        "morse_nodes": morse_graph.num_vertices(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
