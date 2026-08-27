#!/usr/bin/env python3
"""Compute the leslie3d_example1 coarse Morse graph at a chosen subdivision.

The uniform-grid driver pins ``subdiv_init = subdiv_min = subdiv_max``, which
answers "what does one fixed resolution see". This answers a different
question: what happens when the *adaptive* bracket is opened, so CMGDB starts
coarse and refines only where the dynamics demands it. ``subdiv_init`` sets the
grid the first pass runs on, ``subdiv_min`` the floor every recurrent cell is
carried to, and ``subdiv_max`` the depth refinement may reach.

Every Morse set is reported, including the one-cell components with trivial
Conley index that the manuscript's nontrivial skeleton omits -- the point here
is to see how the decomposition itself changes with resolution, so nothing is
filtered.

(22, 22, 24) is the paper's coarse example. Conley indices come from CMGDB's
own ``ComputeConleyMorseGraph``, which handles the adaptive grid: the study
script's per-cell substitute cannot express an index pair spanning depths
22-24 on a build without ``ComputeConleyIndexForCells`` and reports a trivial
index instead, which would drop two of the four sets from the figure.

Each run writes its own directory holding the CMGDB Morse graph DOT, the box
CSV, a rendered graph and set plot, and a JSON summary. The model checkpoint is
the shipped one throughout: only the subdivision changes between runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import CMGDB  # noqa: E402

from latentdynamics.analysis.morse import LatentBounds, _build_box_map  # noqa: E402
from latentdynamics.config.schema import CMGDBConfig  # noqa: E402
from latentdynamics.replay import load_experiment  # noqa: E402
from latentdynamics.viz import plot_morse_sets_2d_cmgdb  # noqa: E402
from latentdynamics.viz.morse_plots import save_morse_graph_artifacts  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "output" / "leslie3d_example1_study" / "coarse_subdiv"
PALETTE = ["#ffb000ff", "#dc267fff", "#648fffff", "#fe6100ff", "#785ef0ff", "#008080ff"]

#: (init, min, max) triples to run when none are given on the command line.
DEFAULT_SCAN = [(22, 22, 24), (22, 22, 26), (18, 22, 24), (18, 22, 26)]


def scan_config(experiment, init: int, minimum: int, maximum: int) -> CMGDBConfig:
    """The shipped CMGDB settings with only the subdivision bracket replaced."""
    raw = experiment.seed_cfg.cmgdb.model_dump()
    raw.update(
        subdiv_init=init,
        subdiv_min=minimum,
        subdiv_max=maximum,
        compute_roa=False,
    )
    return CMGDBConfig.model_validate(raw)


def conley_indices(morse_graph) -> dict[int, list[str]]:
    """Conley index polynomials per node, as CMGDB reports them."""
    out: dict[int, list[str]] = {}
    for node in range(morse_graph.num_vertices()):
        try:
            out[node] = [str(v) for v in morse_graph.annotations(node)]
        except Exception:
            out[node] = []
    return out


def run_one(experiment, bounds, init, minimum, maximum, out_root: Path) -> dict:
    label = f"{init}_{minimum}_{maximum}"
    out = out_root / label
    out.mkdir(parents=True, exist_ok=True)
    config = scan_config(experiment, init, minimum, maximum)

    print(f"\n===== subdiv init={init} min={minimum} max={maximum} =====", flush=True)
    started = time.perf_counter()
    box_map = _build_box_map(
        experiment.model.latent_map, bounds, config, device=torch.device("cpu")
    )
    model = CMGDB.Model(
        config.subdiv_min,
        config.subdiv_max,
        config.subdiv_init,
        config.subdiv_limit,
        bounds.lower.tolist(),
        bounds.upper.tolist(),
        box_map,
    )
    batch_map = getattr(box_map, "batch", None)
    if callable(batch_map) and hasattr(model, "set_batch_map"):
        model.set_batch_map(batch_map)
    morse_graph, _map_graph = CMGDB.ComputeConleyMorseGraph(model)
    seconds = time.perf_counter() - started

    # Same two artifacts the pipeline persists, so these runs are re-renderable
    # with exactly the tools the paper figures use.
    save_morse_graph_artifacts(morse_graph, out, palette=PALETTE)
    graph_pdf = out / "morse_graph.pdf"
    subprocess.run(["dot", "-Tpdf", str(out / "morse_graph"), "-o", str(graph_pdf)],
                   check=False)

    rows = CMGDB.LoadMorseSetFile(str(out / "morse_sets"))
    counts: dict[int, int] = {}
    for row in rows:
        node = int(row[-1])
        counts[node] = counts.get(node, 0) + 1

    plot_morse_sets_2d_cmgdb(
        out / "morse_sets",
        out / "morse_sets.pdf",
        palette=PALETTE,
        xlabel="$z_1$",
        ylabel="$z_2$",
    )

    indices = conley_indices(morse_graph)
    summary = {
        "subdiv": {"init": init, "min": minimum, "max": maximum},
        "num_morse_sets": morse_graph.num_vertices(),
        "total_boxes": len(rows),
        "boxes_per_node": {str(k): counts.get(k, 0) for k in sorted(counts)},
        "conley_indices": {str(k): v for k, v in indices.items()},
        "seconds": seconds,
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"  {morse_graph.num_vertices()} Morse sets, {len(rows):,} boxes, {seconds:.1f}s",
          flush=True)
    for node in sorted(counts):
        idx = ", ".join(indices.get(node, [])) or "?"
        print(f"    node {node:>2}: {counts[node]:>7,} boxes   index ({idx})", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subdiv", action="append", nargs=3, type=int,
                        metavar=("INIT", "MIN", "MAX"),
                        help="a subdivision triple; repeatable "
                             f"(default: {DEFAULT_SCAN})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    triples = [tuple(t) for t in args.subdiv] if args.subdiv else list(DEFAULT_SCAN)
    experiment = load_experiment("leslie3d_example1_replay", device="cpu")
    lower_raw, upper_raw = experiment.morse_bounds()
    bounds = LatentBounds(
        lower=np.asarray(lower_raw, dtype=np.float64),
        upper=np.asarray(upper_raw, dtype=np.float64),
    )
    print(f"latent bounds: lower={bounds.lower.tolist()} upper={bounds.upper.tolist()}")

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = [run_one(experiment, bounds, *t, args.output) for t in triples]
    (args.output / "scan.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("\n===== summary =====")
    print(f"{'subdiv':>14}  {'sets':>5}  {'boxes':>10}  {'time':>8}")
    for s in summaries:
        d = s["subdiv"]
        print(f"{d['init']:>4},{d['min']:>3},{d['max']:>3}  {s['num_morse_sets']:>5}  "
              f"{s['total_boxes']:>10,}  {s['seconds']:>7.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
