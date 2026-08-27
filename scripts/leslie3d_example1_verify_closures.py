#!/usr/bin/env python3
"""Verify fixed-depth minimal Morse sets equal their forward closures.

Rebuilds the leslie3d_example1 fixed-depth cell graph produced by
``scripts/leslie3d_example1_uniform_grid.py`` and checks, for every minimal
node and for the distinguished fixed-point/period-two node, that the
recurrent component equals its exact map-graph forward closure and that the
saved boxes match the live boxes exactly.

Inputs: ``<--output>/fixed<depth>`` plus the fetched
``replay_sources/leslie3d_example1/`` artifacts.  Writes
``<--output>/fixed<depth>/residual_tolerance/forward_closure_verification.json``
(default study root: ``output/leslie3d_example1_study``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import CMGDB
from latentdynamics._paths import get_repo_root
from latentdynamics.analysis.cmgdb_roa import attractor_cells
from latentdynamics.analysis.morse import LatentBounds, _build_box_map
from latentdynamics.config.schema import CMGDBConfig

from leslie3d_example1_uniform_grid import SUBDIV_MAX  # noqa: E402
from latentdynamics.replay import load_experiment


REPO_ROOT = get_repo_root()
DEFAULT_OUTPUT = REPO_ROOT / "output" / "leslie3d_example1_study"
_EXPERIMENT = None


def cached_experiment():
    global _EXPERIMENT
    if _EXPERIMENT is None:
        _EXPERIMENT = load_experiment("leslie3d_example1_replay", device="cpu")
    return _EXPERIMENT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_boxes_hash(values: np.ndarray) -> str:
    rows = np.asarray(values, dtype=np.float64)
    order = np.lexsort(tuple(rows[:, index] for index in reversed(range(rows.shape[1]))))
    canonical = np.ascontiguousarray(rows[order], dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def sorted_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    order = np.lexsort(tuple(rows[:, index] for index in reversed(range(rows.shape[1]))))
    return np.ascontiguousarray(rows[order])


def main(depth: int, output_root: Path) -> None:
    fixed = output_root / f"fixed{depth}"
    raw_boxes = fixed / f"morse_sets_fixed{depth}_raw.csv"
    box_csv = (
        raw_boxes
        if raw_boxes.is_file()
        else fixed / f"morse_sets_fixed{depth}_connection_complete.csv"
    )
    result_json = fixed / "result.json"
    out = fixed / "residual_tolerance" / "forward_closure_verification.json"
    fixed_result = json.loads(result_json.read_text(encoding="utf-8"))
    minimal_nodes = tuple(int(node) for node in fixed_result["fixed_graph"]["minimal"])
    nodes = tuple(
        sorted(
            set(minimal_nodes)
            | {
                int(fixed_result["distinguished_objects"]["fixed_point"]["node"]),
                int(fixed_result["distinguished_objects"]["period_two"]["node"]),
            }
        )
    )
    experiment = cached_experiment()
    lower_raw, upper_raw = experiment.morse_bounds()
    lower = np.asarray(lower_raw, dtype=np.float64)
    upper = np.asarray(upper_raw, dtype=np.float64)
    bounds = LatentBounds(lower=lower, upper=upper)
    raw = experiment.seed_cfg.cmgdb.model_dump()
    # Imported from the grid step so the two can never drift: verifying under a
    # different bracket would check closures of a graph that was never built.
    raw.update(
        subdiv_init=depth,
        subdiv_min=depth,
        subdiv_max=SUBDIV_MAX,
        compute_roa=False,
    )
    config = CMGDBConfig.model_validate(raw)
    box_map = _build_box_map(
        experiment.model.latent_map,
        bounds,
        config,
        device=torch.device("cpu"),
    )
    model = CMGDB.Model(
        config.subdiv_min,
        config.subdiv_max,
        config.subdiv_init,
        config.subdiv_limit,
        lower.tolist(),
        upper.tolist(),
        box_map,
    )
    if hasattr(model, "set_batch_map"):
        model.set_batch_map(box_map.batch)

    started = time.perf_counter()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    elapsed = time.perf_counter() - started
    saved = np.loadtxt(box_csv, delimiter=",", dtype=np.float64, ndmin=2)
    saved_labels = saved[:, -1].astype(np.int64)

    node_results: dict[str, Any] = {}
    for node in nodes:
        recurrent = {int(cell) for cell in morse_graph.morse_set(node)}
        closure = attractor_cells(map_graph, morse_graph, [node])
        live_boxes = sorted_rows(
            np.asarray(morse_graph.morse_set_boxes(node), dtype=np.float64)
        )
        saved_boxes = sorted_rows(saved[saved_labels == node, :-1])
        exact_match = live_boxes.shape == saved_boxes.shape and np.array_equal(
            live_boxes,
            saved_boxes,
        )
        max_abs = (
            float(np.max(np.abs(live_boxes - saved_boxes)))
            if live_boxes.shape == saved_boxes.shape and live_boxes.size
            else None
        )
        node_results[str(node)] = {
            "recurrent_cells": len(recurrent),
            "forward_closure_cells": len(closure),
            "added_cells": len(closure - recurrent),
            "closure_equals_recurrent_set": closure == recurrent,
            "saved_live_boxes_exact": exact_match,
            "saved_live_max_absolute_difference": max_abs,
            "canonical_boxes_sha256": canonical_boxes_hash(live_boxes),
        }

    edges = sorted(
        [int(source), int(target)]
        for source, target in morse_graph.edges()
    )
    output = {
        "status": "complete",
        "purpose": (
            f"verify exact fixed-{depth} minimal recurrent sets are "
            "forward-invariant cell blocks"
        ),
        "checkpoint": "author-provided leslie3d_example1 replay checkpoint",
        "subdivision": [depth, depth, config.subdiv_max],
        "padding": config.padding,
        "subdivision_limit": config.subdiv_limit,
        "cmgdb_module": str(Path(CMGDB.__file__).resolve()),
        "elapsed_seconds": elapsed,
        "morse_node_count": int(morse_graph.num_vertices()),
        "map_cell_count": int(map_graph.num_vertices()),
        "edges": edges,
        "nodes": node_results,
        "source_hashes": {
            "morse_sets_csv": sha256(box_csv),
            "fixed_result_json": sha256(result_json),
        },
        "interpretation": {
            "minimal_nodes": (
                f"nodes {list(minimal_nodes)} each equal their full forward closure"
            ),
            "additional_checked_nodes": sorted(set(nodes) - set(minimal_nodes)),
        },
    }
    if not all(
        node_results[str(node)]["closure_equals_recurrent_set"]
        and node_results[str(node)]["saved_live_boxes_exact"]
        for node in minimal_nodes
    ):
        raise RuntimeError(f"fixed-{depth} minimal closure verification failed")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--depth", type=int, default=22)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "study root holding fixed<depth>/ from "
            "leslie3d_example1_uniform_grid.py "
            "(default: output/leslie3d_example1_study)"
        ),
    )
    arguments = parser.parse_args()
    main(arguments.depth, arguments.output)
