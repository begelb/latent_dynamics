#!/usr/bin/env python3
"""Baseline 2-D Leslie: run CMGDB on the true map and plot the result.

This is the direct computation the latent model is compared against -- CMGDB
applied to the dynamical system itself, with no autoencoder anywhere. It is the
control: whatever the latent Morse graph shows has to be read against what the
true map does on the same domain.

The map is the exact two-dimensional restriction of the maintained 10-D
``leslie_2gen_contraction`` system, so the baseline and the latent example
describe the same dynamics rather than merely similar ones.

The default subdivisions (24, 27, 28) are the paper's baseline run; the
earlier deep screen used (26, 30, 40) and took roughly an hour.

    python scripts/plot_leslie2d_baseline.py
    python scripts/plot_leslie2d_baseline.py --subdiv 26 30 40 --output output/leslie2d_deep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import CMGDB  # noqa: E402

from latentdynamics.analysis.cmgdb_features import cmgdb_provenance  # noqa: E402
from latentdynamics.config import load_config  # noqa: E402
from latentdynamics.systems import build_system  # noqa: E402
from latentdynamics.systems.leslie import LeslieContraction  # noqa: E402
from latentdynamics.viz import (  # noqa: E402
    plot_morse_sets_2d_cmgdb,
    render_morse_graph,
    save_morse_graph_artifacts,
)

DEFAULT_SUBDIV = (24, 27, 28)      # init, min, max -- the paper's baseline run
# Per-Morse-set display inflation, by label. The recurrent sets here differ by
# orders of magnitude in extent -- a period-3 orbit is a handful of boxes next
# to an invariant circle -- so drawn faithfully the small ones vanish. Each
# factor enlarges its set about its own centre; positions and relative
# geometry are unchanged, only the drawn size. The list is indexed by Morse
# label, so entry i scales node i.
DEFAULT_BOX_SCALE = [20.0, 100.0, 20.0, 100.0, 100.0, 20.0]
PAPER_SUBDIV = (24, 27, 28)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "leslie2d_baseline"


def leslie_2d_map():
    """The exact 2-D restriction of the 10-D Leslie contraction, and its box."""
    config = load_config("leslie_2gen_contraction")
    ambient = build_system(config.system.name, config.system.params)
    if not isinstance(ambient, LeslieContraction) or ambient.dim != 10:
        raise TypeError("leslie_2gen_contraction must configure a 10-D LeslieContraction")

    def step(points):
        array = np.asarray(points, dtype=np.float64)
        single = array.ndim == 1
        batch = np.atleast_2d(array)
        embedded = np.zeros((batch.shape[0], ambient.dim), dtype=np.float64)
        embedded[:, :2] = batch
        image = ambient.step(embedded)[:, :2]
        return image[0] if single else image

    lower = [float(v) for v in ambient.lower_bounds[:2]]
    upper = [float(v) for v in ambient.upper_bounds[:2]]
    return step, lower, upper



def _box_scale_for(factors: list[float], node_count: int) -> dict[int, float] | float:
    """Map a per-label factor list onto the Morse sets this run produced.

    A dict is passed through the renderer unclamped, unlike the ``"auto"``
    mode, so the factors are honoured exactly as given. The list length is tied
    to how many Morse sets the subdivision resolves: a coarser grid merges
    recurrent structure and yields fewer, so surplus entries are dropped and a
    short list leaves the remaining sets at 1.0. Either way the mismatch is
    reported rather than silently reinterpreted.
    """
    if not factors:
        return 1.0
    if len(factors) == 1:
        return float(factors[0])
    if len(factors) != node_count:
        print(
            f"  note: {len(factors)} scale factors for {node_count} Morse sets; "
            f"{'extra entries ignored' if len(factors) > node_count else 'unlisted sets drawn at 1.0'}"
        )
    return {index: float(value) for index, value in enumerate(factors) if index < node_count}


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
                        help=f"default is the published reference {PAPER_SUBDIV}")
    parser.add_argument("--subdiv-limit", type=int, default=10_000)
    parser.add_argument("--box-map-backend", choices=("on_demand", "precomputed"),
                        default="on_demand",
                        help="on_demand matches the published reference and is the "
                             "only option at published subdivisions; see "
                             "_resolve_backend for why")
    # No padding, matching the published reference. Padded box images are a
    # more conservative over-approximation: recurrent sets inflate and marginal
    # connections drop out, so the Morse graph is not the published one.
    parser.add_argument("--padding", action="store_true", default=False,
                        help="pad box images (the published reference did not)")
    parser.add_argument("--no-padding", dest="padding", action="store_false")
    parser.add_argument("--box-scale", type=float, nargs="*", default=DEFAULT_BOX_SCALE,
                        metavar="FACTOR",
                        help="per-Morse-set display inflation, indexed by label "
                             f"(default {DEFAULT_BOX_SCALE}); pass a single value "
                             "for a uniform factor, or nothing for faithful sizes")
    parser.add_argument("--xlabel", default="$z_1$")
    parser.add_argument("--ylabel", default="$z_2$")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    step, lower, upper = leslie_2d_map()
    init, minimum, maximum = args.subdiv
    print(f"2-D Leslie baseline on [{lower[0]}, {upper[0]}] x [{lower[1]}, {upper[1]}]")
    print(f"CMGDB subdiv init={init} min={minimum} max={maximum} padding={args.padding}")

    backend, backend_note = _resolve_backend(args.box_map_backend, maximum, 2)
    print(f"box map: {backend_note}")
    if backend == "precomputed":
        # CMGDB's corner-lattice class, matching the preference order
        # compute_original_leslie.py uses.
        precomputed = CMGDB.PrecomputedBoxMap(
            step, lower, upper, maximum, mode="corners", padding=args.padding
        )

        def box_map(rect):
            return precomputed(rect)
    else:
        def box_map(rect):
            return CMGDB.BoxMap(step, rect, padding=args.padding)

    model = CMGDB.Model(minimum, maximum, init, args.subdiv_limit, lower, upper, box_map)
    if backend == "on_demand":
        # Batched evaluation: CMGDB hands over every rectangle of a level at
        # once and one vectorized call computes all corner images, removing
        # the per-box Python callback that dominates on-demand runs. Row-for-
        # row equivalent to the per-rect BoxMap above.
        def batch_map(rects):
            return CMGDB.BoxMapBatch(step, rects, padding=args.padding)

        model.set_batch_map(batch_map)
    started = time.perf_counter()
    morse_graph, _map_graph = CMGDB.ComputeConleyMorseGraph(model)
    seconds = time.perf_counter() - started
    print(f"{morse_graph.num_vertices()} Morse sets in {seconds / 60:.2f} min")

    dot_path, csv_path = save_morse_graph_artifacts(morse_graph, args.output)
    graph_paths = render_morse_graph(morse_graph, args.output)
    sets_pdf = plot_morse_sets_2d_cmgdb(
        csv_path,
        args.output / "morse_sets.pdf",
        scale_factor=args.box_scale or None,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
    )
    written = {
        "morse_graph_dot": dot_path,
        "morse_sets_csv": csv_path,
        "morse_sets_pdf": sets_pdf,
        **{f"morse_graph_{i}": path for i, path in enumerate(graph_paths)},
    }
    summary = {
        "system": "leslie_2gen_contraction, 2-D restriction (direct; no autoencoder)",
        "bounds": {"lower": lower, "upper": upper},
        "subdivision": {"init": init, "min": minimum, "max": maximum,
                        "limit": args.subdiv_limit},
        "padding": args.padding,
        "box_map_backend": backend,
        "box_map_note": backend_note,
        "morse_sets": morse_graph.num_vertices(),
        "conley_indices": {
            str(int(node)): list(morse_graph.annotations(int(node)))
            for node in morse_graph.vertices()
        },
        "seconds": seconds,
        "cmgdb": cmgdb_provenance(),
        "artifacts": {k: str(v) for k, v in written.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, path in written.items():
        print(f"  {name:12} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
