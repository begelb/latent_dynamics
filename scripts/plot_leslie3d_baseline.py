#!/usr/bin/env python3
"""Baseline 3-D Leslie: run CMGDB on the true map and plot the result.

The direct computation the 3-D latent example is compared against -- CMGDB on
the dynamical system itself, no autoencoder. Unlike the latent runs, whose
Morse sets live in a 2-D latent space, these sets are genuinely
three-dimensional, so they are drawn with CMGDB's cubical surface renderer
(``CMGDB.PlotMorseSets3D``) straight from the computed Morse graph: only faces
on the boundary of a labelled region are emitted, which makes the cost scale
with the surface of the Morse sets rather than their volume.

The map is the overcompensatory three-class Leslie model at
theta = (28.9, 29.8, 22.0) with survival 0.7, on the absorbing box the paper
uses. The default is the published screen (29, 33, 36). The full init=29 grid
is essential: the Morse SETS are the same from a coarse init, but the
Morse-graph EDGES are not -- a coarser initial grid leaves the transient
region under-resolved and a spurious connection collapses the two attractors
into a chain, losing the bistability. This run does not fit a 16 GB machine:
budget roughly 100 GB with the default uncached path (slow, ~6-8 h), or
250 GB+ with --transition-cache (fast, edge cache of order 2e10 edges).

    python scripts/plot_leslie3d_baseline.py                     # published screen
    python scripts/plot_leslie3d_baseline.py --subdiv 24 30 33   # small-machine preview
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import CMGDB  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

from latentdynamics.analysis.cmgdb_features import cmgdb_provenance  # noqa: E402
from latentdynamics.viz import render_morse_graph  # noqa: E402
from latentdynamics.viz.style import PALETTE, save_figure  # noqa: E402

# The absorbing box: forward-invariant, so the Morse decomposition inside it is
# the whole recurrent structure rather than an artefact of a clipped domain.
ABSORBING_LOWER = [0.0, 0.0, 0.0]
ABSORBING_UPPER = [110.0, 77.0, 54.0]

DEFAULT_SUBDIV = (29, 33, 36)      # init, min, max -- the published screen
# Per-Morse-set display inflation, indexed by label -- the 3-D counterpart of
# the 2-D box scaling. A period-4 orbit is a handful of cells beside a set that
# spans the domain, so faithful sizes hide it. Each factor enlarges its set
# about each cell's own centre; culling still runs on the unscaled grid, so a
# scaled set stays a closed surface.
# The published level-33 reference resolves six Morse sets whose sizes differ by
# four orders of magnitude -- its level-24 display cover holds (141, 10125, 81,
# 66, 84, 1) cells. Drawn faithfully everything but node 1 disappears, so each
# small set is inflated about its own cells' centres; node 1 stays at 1.
DEFAULT_BOX_SCALE: list[float] = [10.0, 1.0, 20.0, 20.0, 20.0, 50.0]

# Clear of the z tick numbers, which run to three digits on this domain.
DEFAULT_ZLABEL_POS = (1.06, 0.60)
PAPER_SUBDIV = (29, 33, 36)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "leslie3d_baseline"


def leslie_3d(point):
    """One step of the three-class overcompensatory Leslie map."""
    total = point[0] + point[1] + point[2]
    return [
        (28.9 * point[0] + 29.8 * point[1] + 22.0 * point[2]) * math.exp(-0.1 * total),
        0.7 * point[0],
        0.7 * point[1],
    ]


def leslie_3d_points(points):
    """The same map, vectorized over an (m, 3) array of points."""
    points = np.asarray(points, dtype=np.float64)
    total = points.sum(axis=1)
    return np.column_stack((
        (28.9 * points[:, 0] + 29.8 * points[:, 1] + 22.0 * points[:, 2])
        * np.exp(-0.1 * total),
        0.7 * points[:, 0],
        0.7 * points[:, 1],
    ))



def _box_scale_for(factors: list[float], node_count: int) -> list[float] | None:
    """Per-node display factors as the list CMGDB's scale_factor expects."""
    if not factors:
        return None
    if len(factors) == 1:
        return [float(factors[0])] * node_count
    if len(factors) != node_count:
        print(
            f"  note: {len(factors)} scale factors for {node_count} Morse sets; "
            f"{'extra entries ignored' if len(factors) > node_count else 'unlisted sets drawn at 1.0'}"
        )
    return [float(value) for value in factors[:node_count]]


def _lattice_nodes(subdiv_max: int, dim: int) -> int:
    """Corner-lattice size CMGDB would precompute at ``subdiv_max``.

    CMGDB bisects coordinate ``depth % dim`` at each depth, so axis ``j`` is
    split ``ceil((subdiv_max - j) / dim)`` times.
    """
    depths = [(subdiv_max - j + dim - 1) // dim for j in range(dim)]
    nodes = 1
    for depth in depths:
        nodes *= 2 ** depth + 1
    return nodes


def _resolve_backend(name: str, subdiv_max: int, dim: int):
    """Return ``(box_map_factory_kind, note)`` for the requested backend.

    On-demand is the default because these baselines evaluate an analytic map:
    a handful of flops, far cheaper than the table that would replace it. A
    precomputed corner lattice pays for a costly map -- a network, a GP, an ODE
    solve -- which is why the latent runs use one and these do not. At the
    published subdivisions the table is not merely wasteful but impossible: the
    lattice reaches 1.1e12 nodes in 2-D at depth 40 and 6.9e10 in 3-D at depth
    36, well past the 2**31 nodes CMGDB itself refuses.
    """
    nodes = _lattice_nodes(subdiv_max, dim)
    table_bytes = nodes * dim * 8
    if name == "precomputed":
        if nodes > 2 ** 31:
            raise SystemExit(
                f"precomputed backend needs a {nodes:,}-node corner lattice "
                f"({table_bytes / 2**40:.1f} TiB) at subdiv_max={subdiv_max}; "
                f"CMGDB refuses above 2**31 nodes. Use --box-map-backend "
                f"on_demand, or lower --subdiv."
            )
        return "precomputed", f"corner lattice {nodes:,} nodes (~{table_bytes / 2**30:.2f} GiB)"
    return "on_demand", f"on-demand (a precomputed lattice here would be {nodes:,} nodes)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subdiv", type=int, nargs=3, default=list(DEFAULT_SUBDIV),
                        metavar=("INIT", "MIN", "MAX"),
                        help=f"default is the published screen {PAPER_SUBDIV}")
    parser.add_argument("--subdiv-limit", type=int, default=10_000)
    parser.add_argument("--box-map-backend", choices=("on_demand", "precomputed"),
                        default="on_demand",
                        help="on_demand matches the published reference and is the "
                             "only option at published subdivisions; see "
                             "_resolve_backend for why")
    parser.add_argument("--padding", action="store_true", default=False,
                        help="pad box images (the reference screen used no padding)")
    parser.add_argument("--transition-cache", dest="transition_cache",
                        action="store_true", default=False,
                        help="cache the per-level transition graph: much faster "
                             "but needs ~250 GB+ at the published init=29 "
                             "(default off: re-evaluate the cheap batched map "
                             "during the SCC passes, ~100 GB peak)")
    parser.add_argument("--no-transition-cache", dest="transition_cache",
                        action="store_false")
    parser.add_argument("--lower-bounds", type=float, nargs=3, default=ABSORBING_LOWER)
    parser.add_argument("--upper-bounds", type=float, nargs=3, default=ABSORBING_UPPER)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    parser.add_argument("--box-scale", type=float, nargs="*", default=DEFAULT_BOX_SCALE,
                        metavar="FACTOR",
                        help="per-Morse-set display inflation, indexed by label; "
                             "one value scales every set, 'auto' is available "
                             "through the library API")
    parser.add_argument("--legend", dest="legend", action="store_true", default=False,
                        help="draw the Morse-set legend below the axes (off by default)")
    parser.add_argument("--zlabel-x", type=float, default=DEFAULT_ZLABEL_POS[0],
                        help="axes-fraction x of the $z_3$ label; raise it to clear "
                             "wider tick numbers")
    parser.add_argument("--zlabel-y", type=float, default=DEFAULT_ZLABEL_POS[1])
    parser.add_argument("--rasterized", dest="rasterized", action="store_true", default=None,
                        help="force a rasterized Morse-set collection (default: "
                             "vector below the face threshold, raster above)")
    parser.add_argument("--vector", dest="rasterized", action="store_false",
                        help="force vector output regardless of face count")
    parser.add_argument("--dpi", type=int, default=600,
                        help="raster resolution when the collection is rasterized")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    lower, upper = args.lower_bounds, args.upper_bounds
    init, minimum, maximum = args.subdiv
    print(f"3-D Leslie baseline on {lower} - {upper}")
    print(f"CMGDB subdiv init={init} min={minimum} max={maximum} padding={args.padding}")

    backend, backend_note = _resolve_backend(args.box_map_backend, maximum, 3)
    print(f"box map: {backend_note}")

    def box_map(rect):
        return CMGDB.BoxMap(leslie_3d, rect, padding=args.padding)

    model = CMGDB.Model(minimum, maximum, init, args.subdiv_limit, lower, upper, box_map)
    if backend == "on_demand":
        # Batched evaluation (see plot_leslie2d_baseline.py): one vectorized
        # call per level instead of one Python callback per box.
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
    graph_paths = render_morse_graph(morse_graph, args.output)

    n_boxes = sum(len(morse_graph.morse_set_boxes(v)) for v in morse_graph.vertices())
    print(f"drawing {n_boxes:,} boxes as exposed cubical faces")
    box_scale = _box_scale_for(args.box_scale, morse_graph.num_vertices())
    # Drawn straight from the live Morse graph by CMGDB's cubical renderer;
    # the z label is placed as an unclipped 2-D annotation via zlabel_pos.
    fig, ax = CMGDB.PlotMorseSets3D(
        morse_graph,
        clist=list(PALETTE),
        scale_factor=box_scale,
        elev=args.elev,
        azim=args.azim,
        grid=True,
        xlabel="$z_1$",
        ylabel="$z_2$",
        zlabel="$z_3$",
        zlabel_pos=(args.zlabel_x, args.zlabel_y),
        rasterize=bool(args.rasterized),
        show=False,
    )
    # The paper's camera: the domain at its true proportions under an
    # orthographic projection. The default cube aspect foreshortens z_1 and
    # visually stacks the interior sets (p* over S2); the true aspect keeps
    # them separated as in the published panel.
    ax.set_box_aspect((upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]))
    ax.set_proj_type("ortho")
    if args.legend:
        handles = [
            mpatches.Patch(
                facecolor=PALETTE[int(node) % len(PALETTE)],
                edgecolor=(0.08, 0.08, 0.08, 0.25),
                label=f"$M_{{{int(node)}}}$",
            )
            for node in sorted(morse_graph.vertices())
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                  ncol=6, frameon=False, handlelength=1.0, columnspacing=0.9)
    sets_paths = save_figure(fig, args.output / "morse_sets_3d", dpi=args.dpi)

    summary = {
        "system": "3-class overcompensatory Leslie, theta=(28.9,29.8,22.0), s=0.7 "
                  "(direct; no autoencoder)",
        "bounds": {"lower": lower, "upper": upper},
        "subdivision": {"init": init, "min": minimum, "max": maximum,
                        "limit": args.subdiv_limit},
        "padding": args.padding,
        "box_map_backend": backend,
        "box_map_note": backend_note,
        "box_scale": box_scale,
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
