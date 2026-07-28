"""Collapse the non-attracting Chafee--Infante graph nodes toward M(1).

The two minimal nodes stay distinct; nodes 2--6 form one fiber. The resulting
map from the seven-node graph onto the three-node graph is a surjective,
order-preserving poset map. Following the manuscript notation, ``M(1)``
represents the unstable equilibria and their connecting orbits in the target
Morse representation.

The production default reconstructs Marcio's original computation from his raw
weights, training data, data-derived bounds, padding convention, and
subdivisions. ``--computation reference`` retains the later package
recomputation as an explicitly labelled comparison. The merged node is
intentionally not given a Conley index, because that index cannot be inferred
by combining the fine-node annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import CMGDB
import numpy as np
import torch

from latentdynamics.analysis.morse import LatentBounds, compute_morse_graph
from latentdynamics.analysis.morse_coarsening import (
    coarsen_morse_graph,
    compute_connection_complete_morse_sets,
    write_connection_complete_morse_sets,
    write_morse_graph_dot,
    write_quotient_morse_sets,
)
from latentdynamics.analysis.morse_graph_parser import MorseGraph
from latentdynamics.config import load_config
from latentdynamics.models import build_autoencoder
from latentdynamics.replay import load_experiment
from latentdynamics.viz import plot_morse_sets_from_csv, render_morse_from_files
from latentdynamics.viz.style import save_latent_figure

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
MARCIO_ROOT = PROJECT_ROOT / "archive" / "marcio" / "scripts"
MARCIO_WEIGHTS = MARCIO_ROOT / "ci_model_weights.pth"
MARCIO_DATA = MARCIO_ROOT / "train_data.csv"
MARCIO_SUBDIVISIONS = (14, 16, 22)
MARCIO_PALETTE = (
    "#1f77b4",
    "#e6550d",
    "#31a354",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
)
MARCIO_COARSE_PALETTE = (
    MARCIO_PALETTE[0],
    MARCIO_PALETTE[1],
    "#7f7f7f",
)
MARCIO_EXPECTED_EDGES = {
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (4, 2),
    (4, 3),
    (5, 4),
    (6, 5),
}
DEFAULT_REFERENCE_SOURCE = (
    CODE_ROOT / "replay_sources" / "chafee_infante" / "replay" / "MG"
)
DEFAULT_OUTPUT = CODE_ROOT / "paper_figures" / "coarsened" / "chafee_infante"


def _parsed_graph_from_live(cmgdb_morse_graph) -> MorseGraph:
    nodes = list(range(int(cmgdb_morse_graph.num_vertices())))
    edges: dict[int, list[int]] = {}
    for raw_source, raw_target in cmgdb_morse_graph.edges():
        source, target = int(raw_source), int(raw_target)
        edges.setdefault(source, []).append(target)
    return MorseGraph(
        nodes=nodes,
        edges={source: sorted(targets) for source, targets in edges.items()},
        colors={
            node: f"{MARCIO_PALETTE[node % len(MARCIO_PALETTE)]}ff"
            for node in nodes
        },
        labels={
            node: f"{node} : ({', '.join(cmgdb_morse_graph.annotations(node))})"
            for node in nodes
        },
    )


def _load_marcio_model(device: str):
    config = load_config(
        CODE_ROOT / "src" / "latentdynamics" / "configs" / "chafee_infante_replay.yaml"
    )
    model = build_autoencoder(config.arch)
    raw = torch.load(MARCIO_WEIGHTS, map_location="cpu", weights_only=True)
    remapped = {}
    for key, value in raw.items():
        component, rest = key.split(".", 1)
        remapped[f"{component}.net.{rest}"] = value
    model.load_state_dict(remapped, strict=True)
    return model.to(torch.device(device)).eval()


def _marcio_bounds(model, device: str) -> LatentBounds:
    table = np.loadtxt(MARCIO_DATA, delimiter=",")
    states = np.concatenate((table[:, :64], table[:, 64:]), axis=0).astype(
        np.float32,
        copy=False,
    )
    encoded_chunks = []
    with torch.no_grad():
        for start in range(0, len(states), 8192):
            chunk = torch.from_numpy(states[start : start + 8192]).to(device)
            encoded_chunks.append(model.encoder(chunk).cpu().numpy())
    encoded = np.concatenate(encoded_chunks, axis=0)
    lower = encoded.min(axis=0).astype(np.float64)
    upper = encoded.max(axis=0).astype(np.float64)
    span = upper - lower
    return LatentBounds(
        lower=lower - 0.1 * span,
        upper=upper + 0.1 * span,
    )


def _compute_marcio_graph(device: str):
    model = _load_marcio_model(device)
    bounds = _marcio_bounds(model, device)

    @torch.no_grad()
    def latent_map(point):
        tensor = torch.tensor([point], dtype=torch.float32, device=device)
        return model.latent_map(tensor).cpu().numpy()[0]

    def box_map(rect):
        return CMGDB.BoxMap(latent_map, rect, padding=True)

    subdiv_init, subdiv_min, subdiv_max = MARCIO_SUBDIVISIONS
    cmgdb_model = CMGDB.Model(
        subdiv_min,
        subdiv_max,
        subdiv_init,
        10000,
        bounds.lower.tolist(),
        bounds.upper.tolist(),
        box_map,
    )
    cmgdb_morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(cmgdb_model)
    live_edges = {
        (int(source), int(target))
        for source, target in cmgdb_morse_graph.edges()
    }
    if int(cmgdb_morse_graph.num_vertices()) != 7 or live_edges != MARCIO_EXPECTED_EDGES:
        raise ValueError(
            "Marcio reconstruction does not match his archived seven-node graph: "
            f"nodes={int(cmgdb_morse_graph.num_vertices())}, edges={sorted(live_edges)}"
        )
    return cmgdb_morse_graph, map_graph, bounds


def _compute_reference_graph(config: str, device: str):
    experiment = load_experiment(config, device=device)
    cmgdb_config = experiment.seed_cfg.cmgdb
    if cmgdb_config.lower_bounds is not None and cmgdb_config.upper_bounds is not None:
        lower = cmgdb_config.lower_bounds
        upper = cmgdb_config.upper_bounds
    else:
        lower, upper = experiment.morse_bounds()
        if lower is None or upper is None:
            raise ValueError(f"{config}: no CMGDB bounds in the config or saved run")
    bounds = LatentBounds(
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
    )
    cmgdb_morse_graph, map_graph = compute_morse_graph(
        experiment.model,
        bounds,
        cmgdb_config,
        device=experiment.device,
    )
    return cmgdb_morse_graph, map_graph, bounds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--computation",
        choices=("marcio", "reference"),
        default="marcio",
        help="cell graph to use (default: Marcio's original computation)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_REFERENCE_SOURCE,
        help="saved DOT/CSV source used only by the reference computation",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--collapse",
        default="2,3,4,5,6",
        help="comma-separated fine Morse nodes to merge (default: 2,3,4,5,6)",
    )
    parser.add_argument(
        "--with-connections",
        action="store_true",
        help=(
            "add cells on paths between fine Morse nodes in each quotient fiber; "
            "always enabled for the Marcio production computation"
        ),
    )
    parser.add_argument(
        "--config",
        default="chafee_infante_replay",
        help="config/checkpoint used only by --computation reference",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device used with --with-connections (default: cpu)",
    )
    args = parser.parse_args()

    merged = frozenset(int(value) for value in args.collapse.split(",") if value.strip())
    connection_completion = args.with_connections or args.computation == "marcio"
    if args.computation == "marcio":
        live_morse_graph, map_graph, bounds = _compute_marcio_graph(args.device)
        graph = _parsed_graph_from_live(live_morse_graph)
        source_description = "archive/marcio/scripts"
        computation_metadata = {
            "name": "marcio_original",
            "weights": str(MARCIO_WEIGHTS.relative_to(PROJECT_ROOT)),
            "data": str(MARCIO_DATA.relative_to(PROJECT_ROOT)),
            "bounds": {
                "lower": bounds.lower.tolist(),
                "upper": bounds.upper.tolist(),
                "rule": "encoded train_data extrema plus 10 percent per-axis padding",
            },
            "subdivisions": {
                "init": MARCIO_SUBDIVISIONS[0],
                "min": MARCIO_SUBDIVISIONS[1],
                "max": MARCIO_SUBDIVISIONS[2],
            },
            "padding": True,
        }
    else:
        graph = MorseGraph.from_dot(args.source / "morse_graph")
        live_morse_graph = map_graph = bounds = None
        reference_config = load_config(args.config)
        if (
            reference_config.cmgdb.lower_bounds is not None
            and reference_config.cmgdb.upper_bounds is not None
        ):
            bounds = LatentBounds(
                lower=np.asarray(reference_config.cmgdb.lower_bounds, dtype=np.float64),
                upper=np.asarray(reference_config.cmgdb.upper_bounds, dtype=np.float64),
            )
        try:
            source_description = str(args.source.resolve().relative_to(CODE_ROOT))
        except ValueError:
            source_description = str(args.source)
        computation_metadata = {
            "name": "package_reference",
            "config": args.config,
        }
    quotient = coarsen_morse_graph(
        graph,
        [merged],
        labels={
            frozenset({0}): "M(0⁺)",
            frozenset({1}): "M(0⁻)",
            merged: "M(1)",
        },
    )

    morse_dir = args.output / "MG"
    write_morse_graph_dot(quotient.graph, morse_dir / "morse_graph")
    connection_counts: dict[str, int] | None = None
    recurrent_counts: dict[str, int] | None = None
    if connection_completion:
        if live_morse_graph is None or map_graph is None:
            live_morse_graph, map_graph, bounds = _compute_reference_graph(
                args.config,
                args.device,
            )
            saved_edges = {
                (source, target)
                for source, targets in graph.edges.items()
                for target in targets
            }
            live_edges = {
                (int(source), int(target))
                for source, target in live_morse_graph.edges()
            }
            if (
                int(live_morse_graph.num_vertices()) != len(graph.nodes)
                or live_edges != saved_edges
            ):
                raise ValueError(
                    "recomputed reference graph does not match its saved source graph; "
                    "refusing to apply the saved node projection"
                )
        completed = compute_connection_complete_morse_sets(
            map_graph,
            live_morse_graph,
            quotient.projection,
        )
        write_connection_complete_morse_sets(
            live_morse_graph,
            completed,
            morse_dir / "morse_sets",
        )
        connection_counts = {
            str(coarse): int(cells.size)
            for coarse, cells in completed.connection_cells.items()
        }
        recurrent_counts = {
            str(coarse): int(completed.cells[coarse].size - cells.size)
            for coarse, cells in completed.connection_cells.items()
        }
    else:
        write_quotient_morse_sets(
            args.source / "morse_sets",
            morse_dir / "morse_sets",
            quotient.projection,
        )
    render_morse_from_files(
        morse_dir,
        bounds_lower=None if bounds is None else bounds.lower.tolist(),
        bounds_upper=None if bounds is None else bounds.upper.tolist(),
        out_dir=args.output,
        palette=MARCIO_COARSE_PALETTE,
        box_scale="auto",
        min_box_side_frac=0.0025,
    )
    if bounds is not None:
        coarse_plot = plot_morse_sets_from_csv(
            morse_dir / "morse_sets",
            bounds_lower=bounds.lower.tolist(),
            bounds_upper=bounds.upper.tolist(),
            palette=MARCIO_COARSE_PALETTE,
            box_scale="auto",
            min_box_side_frac=0.0025,
        )
        coarse_plot.ax.set_xticks([])
        coarse_plot.ax.set_yticks([])
        save_latent_figure(
            coarse_plot.fig,
            args.output / "morse_sets",
            close=True,
        )

    manifest = {
        "source": source_description,
        "computation": computation_metadata,
        "projection": {str(k): v for k, v in sorted(quotient.projection.items())},
        "fibers": {str(k): sorted(v) for k, v in quotient.fibers.items()},
        "quotient_edges": quotient.graph.edges,
        "merged_morse_set_name": "M(1)",
        "attracting_morse_set_names": {"0": "M(0^+)", "1": "M(0^-)"},
        "merged_morse_set_description": (
            "the unstable equilibria represented by fine nodes 2--6 and their connecting orbits"
        ),
        "connection_completion": connection_completion,
        "recurrent_cell_counts": recurrent_counts,
        "connection_cell_counts": connection_counts,
        "box_union_note": (
            "The plotted regions include the directed cell-graph connections "
            "within each quotient fiber."
            if connection_completion
            else "The plotted region is only the union of saved fine Morse-set boxes; "
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
    if recurrent_counts is not None:
        print(f"recurrent cells: {recurrent_counts}")
        print(f"connection cells: {connection_counts}")
    print(f"artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
