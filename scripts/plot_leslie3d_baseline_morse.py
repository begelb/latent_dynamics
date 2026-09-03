#!/usr/bin/env python3
"""Baseline 3-D Leslie at a machine-sized screen: the Morse-sets panel.

The same computation as ``plot_leslie3d_baseline.py`` -- same map, same
absorbing box, same camera, same display inflations, and the Conley index of
every Morse set -- run at a subdivision that fits an ordinary machine. Three
things differ, and all three follow from that:

* **Subdivision (24, 33, 36)** instead of the published (29, 33, 36). The
  Morse SETS at a coarse initial grid are those of the published screen, which
  is what makes this a legitimate source for the sets panel. The Morse-graph
  EDGES are not: a coarse init leaves the transient region under-resolved, a
  spurious connection appears, and the two attractors collapse into a chain --
  the published graph is bistable. The Morse graph rendered here is therefore
  a diagnostic, not the paper's graph panel, and the figures driver collects
  only the sets from this script; the run says so on stderr whenever it is
  below the published init. The Conley indices are still computed, so each
  set's index can be checked against the published annotations.
* **Published color order.** The paper's Morse graph draws node 3 blue and
  node 2 orange, the reverse of the index-ordered palette, so those two
  entries are exchanged before anything is drawn and both renders share the
  exchanged list.
* **Its own output directory**, so a run here can never overwrite the
  published screen's results next door in ``output/leslie3d_baseline``.

    python scripts/plot_leslie3d_baseline_morse.py
    python scripts/plot_leslie3d_baseline_morse.py --subdiv 21 30 33   # cheaper
    python scripts/plot_leslie3d_baseline_morse.py --replot            # redraw only
    python scripts/render_paper_figures.py --only leslie3d_baseline_morse
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import CMGDB  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

from latentdynamics.analysis.cmgdb_features import cmgdb_provenance  # noqa: E402
from latentdynamics.viz import render_morse_graph  # noqa: E402
from latentdynamics.viz.style import PALETTE, save_figure  # noqa: E402

# The map, the absorbing box, the per-set display inflations and the z-label
# placement are the reference script's, imported rather than copied so the two
# baselines cannot drift apart.
from plot_leslie3d_baseline import (  # noqa: E402
    ABSORBING_LOWER,
    ABSORBING_UPPER,
    DEFAULT_BOX_SCALE,
    _box_scale_for,
    _resolve_backend,
    leslie_3d,
    leslie_3d_points,
)

DEFAULT_SUBDIV = (24, 33, 36)      # init, min, max -- fits a 16 GB machine
PAPER_INIT = 29                    # the published screen's initial subdivision
DEFAULT_OUTPUT = REPO_ROOT / "output" / "leslie3d_baseline_morse"

#: Node -> color with the published figure's assignment for nodes 2 and 3.
PALETTE_PUBLISHED = list(PALETTE)
PALETTE_PUBLISHED[2], PALETTE_PUBLISHED[3] = PALETTE_PUBLISHED[3], PALETTE_PUBLISHED[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subdiv", type=int, nargs=3, default=list(DEFAULT_SUBDIV),
                        metavar=("INIT", "MIN", "MAX"),
                        help=f"default {DEFAULT_SUBDIV}; the published screen is "
                             f"(29, 33, 36) and lives in plot_leslie3d_baseline.py")
    parser.add_argument("--subdiv-limit", type=int, default=10_000)
    parser.add_argument("--box-map-backend", choices=("on_demand", "precomputed"),
                        default="on_demand",
                        help="on_demand matches the published reference and is the "
                             "only option at published subdivisions")
    parser.add_argument("--padding", action="store_true", default=False,
                        help="pad box images (the reference screen used no padding)")
    parser.add_argument("--transition-cache", dest="transition_cache",
                        action="store_true", default=False,
                        help="cache the per-level transition graph: faster, but the "
                             "edge cache is far larger than the box data and does "
                             "not fit a small machine at these depths")
    parser.add_argument("--no-transition-cache", dest="transition_cache",
                        action="store_false")
    parser.add_argument("--lower-bounds", type=float, nargs=3, default=ABSORBING_LOWER)
    parser.add_argument("--upper-bounds", type=float, nargs=3, default=ABSORBING_UPPER)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    parser.add_argument("--box-scale", type=float, nargs="*", default=DEFAULT_BOX_SCALE,
                        metavar="FACTOR",
                        help="per-Morse-set display inflation, indexed by label; "
                             "one value scales every set")
    parser.add_argument("--legend", dest="legend", action="store_true", default=False,
                        help="draw the Morse-set legend below the axes (off by default)")
    parser.add_argument("--zlabel-x", type=float, default=None,
                        help="pin the $z_3$ label to this axes fraction; by default "
                             "CMGDB measures it into place beside the z tick numbers")
    parser.add_argument("--zlabel-y", type=float, default=None,
                        help="the other half of a pinned position; pass both or neither")
    parser.add_argument("--rasterized", dest="rasterized", action="store_true", default=None,
                        help="force a rasterized Morse-set collection (default: "
                             "vector below the face threshold, raster above)")
    parser.add_argument("--vector", dest="rasterized", action="store_false",
                        help="force vector output regardless of face count")
    parser.add_argument("--dpi", type=int, default=600,
                        help="raster resolution when the collection is rasterized")
    parser.add_argument("--replot", action="store_true",
                        help="skip the computation and redraw the sets panel from "
                             "the boxes an earlier run saved under --output")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if (args.zlabel_x is None) != (args.zlabel_y is None):
        parser.error("--zlabel-x and --zlabel-y pin the label together: both or neither")
    zlabel_pos = None if args.zlabel_x is None else (args.zlabel_x, args.zlabel_y)

    lower, upper = args.lower_bounds, args.upper_bounds
    init, minimum, maximum = args.subdiv
    print(f"3-D Leslie baseline (sets panel screen) on {lower} - {upper}")
    if args.replot:
        # Drawing costs minutes and the decomposition tens of them, so work on
        # the figure itself reads back the boxes an earlier run saved. Only the
        # sets panel is rewritten: the Morse graph and result.json belong to
        # the run that computed them.
        saved_boxes = args.output / "morse_sets"
        if not saved_boxes.is_file():
            parser.error(f"--replot needs {saved_boxes}, written by an earlier run")
        morse_sets = CMGDB.LoadMorseSetFile(str(saved_boxes))
        num_sets = max(int(rect[-1]) for rect in morse_sets) + 1
        n_boxes = len(morse_sets)
        graph_paths: tuple = ()
        print(f"replotting {num_sets} Morse sets from {saved_boxes}")
    else:
        print(f"CMGDB subdiv init={init} min={minimum} max={maximum} "
              f"padding={args.padding}")
        if init < PAPER_INIT:
            print(f"  note: init={init} is below the published {PAPER_INIT}; the Morse "
                  f"sets are the published ones but the Morse-graph edges are not "
                  f"(the two attractors collapse into a chain)", file=sys.stderr)

        backend, backend_note = _resolve_backend(args.box_map_backend, maximum, 3)
        print(f"box map: {backend_note}")

        def box_map(rect):
            return CMGDB.BoxMap(leslie_3d, rect, padding=args.padding)

        model = CMGDB.Model(minimum, maximum, init, args.subdiv_limit, lower, upper,
                            box_map)
        if backend == "on_demand":
            # Batched evaluation: one vectorized call per level instead of one
            # Python callback per box.
            def batch_map(rects):
                return CMGDB.BoxMapBatch(leslie_3d_points, rects, padding=args.padding)

            model.set_batch_map(batch_map)
        started = time.perf_counter()
        morse_graph, _map_graph = CMGDB.ComputeConleyMorseGraph(
            model, cache_transition_graph=args.transition_cache)
        seconds = time.perf_counter() - started
        print(f"{morse_graph.num_vertices()} Morse sets in {seconds / 60:.2f} min")

        args.output.mkdir(parents=True, exist_ok=True)
        CMGDB.SaveMorseSets(morse_graph, str(args.output / "morse_sets"))
        graph_paths = render_morse_graph(morse_graph, args.output,
                                         palette=PALETTE_PUBLISHED)
        morse_sets = morse_graph
        num_sets = morse_graph.num_vertices()
        n_boxes = sum(len(morse_graph.morse_set_boxes(v)) for v in morse_graph.vertices())

    print(f"drawing {n_boxes:,} boxes as exposed cubical faces")
    box_scale = _box_scale_for(args.box_scale, num_sets)
    fig, ax = CMGDB.PlotMorseSets3D(
        morse_sets,
        clist=PALETTE_PUBLISHED,
        scale_factor=box_scale,
        elev=args.elev,
        azim=args.azim,
        grid=True,
        xlabel="$z_1$",
        ylabel="$z_2$",
        zlabel="$z_3$",
        zlabel_pos=zlabel_pos,
        rasterize=args.rasterized,
        show=False,
    )
    # The paper's camera: the domain at its true proportions under an
    # orthographic projection, as in plot_leslie3d_baseline.py.
    ax.set_box_aspect((upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]))
    ax.set_proj_type("ortho")
    if args.legend:
        handles = [
            mpatches.Patch(
                facecolor=PALETTE_PUBLISHED[int(node) % len(PALETTE_PUBLISHED)],
                edgecolor=(0.08, 0.08, 0.08, 0.25),
                label=f"$M_{{{int(node)}}}$",
            )
            for node in range(num_sets)
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                  ncol=6, frameon=False, handlelength=1.0, columnspacing=0.9)
    # Cropped like the published panel: the plot sits inside a 1:1.25 axes box,
    # so the full page would carry a third of its width as white margin and
    # shrink the figure by that much wherever it is placed at a set width.
    sets_paths = save_figure(fig, args.output / "morse_sets_3d", dpi=args.dpi,
                             bbox_inches="tight")
    if args.replot:
        for path in sets_paths:
            print(f"  {path}")
        return 0

    summary = {
        "system": "3-class overcompensatory Leslie, theta=(28.9,29.8,22.0), s=0.7 "
                  "(direct; no autoencoder)",
        "bounds": {"lower": lower, "upper": upper},
        "subdivision": {"init": init, "min": minimum, "max": maximum,
                        "limit": args.subdiv_limit},
        "graph_edges_published": init >= PAPER_INIT,
        "padding": args.padding,
        "box_map_backend": backend,
        "box_map_note": backend_note,
        "box_scale": box_scale,
        "palette": PALETTE_PUBLISHED[:morse_graph.num_vertices()],
        "morse_sets": morse_graph.num_vertices(),
        "boxes": int(n_boxes),
        "conley_indices": {
            str(int(node)): list(morse_graph.annotations(int(node)))
            for node in morse_graph.vertices()
        },
        "seconds": seconds,
        "cmgdb": cmgdb_provenance(),
    }
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in (*graph_paths, *sets_paths):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
