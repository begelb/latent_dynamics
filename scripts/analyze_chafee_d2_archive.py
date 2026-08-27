"""Post-process the archived 5-dataset x 3-model Chafee--Infante d=2 study.

The archived computation retains trained 2-D checkpoints and rendered adaptive
Morse PDFs, but not the live CMGDB graphs needed for basin queries.  This script
does not retrain.  For each checkpoint it reproduces the separate uniform
level-16 computation used by the archived ``compute_att_basins_statistics.py``:

* model- and dataset-specific E(X)/E(Y) bounds with a 10% range margin;
* a 256 x 256 uniform CMGDB grid, padding=True;
* strict singleton-all-reachable-Morse-set basin semantics; and
* the common archived 10,000-trajectory truth set.

Every input and output is hashed.  Results are written only below a new output
root; source checkpoints, source data, archived PDFs, and all 3-D artifacts are
read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import re
import statistics
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import CMGDB

from latentdynamics.analysis.morse_reachability import (
    morse_singleton_reachability,
)
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba
from numpy.typing import NDArray
from torch import nn

from latentdynamics._paths import get_repo_root
from latentdynamics.analysis.basin_statistics import (
    OUTSIDE,
    cmgdb_morton_cell_indices,
    compute_chafee_basin_statistics,
)
from latentdynamics.analysis.morse import (
    LatentBounds,
    make_box_map_uniform_precomputed,
)
from latentdynamics.analysis.morse_graph_parser import MorseGraph
from latentdynamics.viz import save_morse_graph_artifacts

plt.switch_backend("Agg")


REPO_ROOT = get_repo_root()
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "replay_sources" / "chafee_infante" / "reference_inputs"
# Rebound in main() when --reference-root is supplied.
REFERENCE_ROOT = DEFAULT_REFERENCE_ROOT
SOURCE_ROOT = DEFAULT_REFERENCE_ROOT / "computations"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "chafee_d2_archive_5x3_roa_v1"

HIGH_DIMENSION = 64
LATENT_DIMENSION = 2
TRAINING_ROWS = 30_000
TRAJECTORY_ROWS = 10_000
SUBDIVISION = 16
SUBDIV_LIMIT = 10_000
GRID_SIDE = 2 ** (SUBDIVISION // LATENT_DIMENSION)
UNIFORM_CELLS = 2**SUBDIVISION
BOUNDS_MARGIN = 0.1
PADDING = True

DATASET_IC_SEEDS = {1: 2158, 2: 4792, 3: 3174, 4: 688, 5: 5727}
EXPECTED_TRAJECTORY_COUNTS = {-1: 3_909, 0: 2_138, 1: 3_953}
TRAJECTORY_SHA256 = "f163b7427e50a4e4d08ab54c87cb5bd16592768edfe8432f019842416afbb145"
STABLE_ROOTS_SHA256 = "cae0222acb37ae9688e54cb2a1f42ac3777360e49b3919403ec1433363de0586"

# CMGDB.MorseSingletonReachability uses these values.  They are deliberately
# not conflated with latentdynamics.analysis.cmgdb_roa's sentinel values.
NO_REACHABLE_MORSE_NODE = -1
MULTIPLE_REACHABLE_MORSE_NODES = -2
ARCHIVED_COMBINED_CORRECT_PERCENTAGE = 78.38972271686593

STATISTIC_COUNT_FIELDS = (
    "outside_both_basins",
    "misclassified_in_negative_basin",
    "misclassified_in_positive_basin",
    "correctly_classified_in_negative_basin",
    "correctly_classified_in_positive_basin",
)


class ArchivedDynamicsAutoencoder(nn.Module):
    """Exact architecture used by the archived state_dict checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )
        self.latent_map = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
        )


@dataclass(frozen=True)
class ClosedCellCandidates:
    flat_cell_ids: NDArray[np.int64]
    offsets: NDArray[np.int64]

    @property
    def n_points(self) -> int:
        return int(self.offsets.size - 1)

    def candidates(self, point_index: int) -> NDArray[np.int64]:
        start = int(self.offsets[point_index])
        stop = int(self.offsets[point_index + 1])
        return self.flat_cell_ids[start:stop]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_reference(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display = resolved
    if relative_to is not None:
        try:
            display = resolved.relative_to(relative_to.resolve())
        except ValueError:
            pass
    return {
        "path": str(display),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def label_counts(values: NDArray[np.integer[Any]]) -> dict[str, int]:
    return {
        str(int(label)): int(count)
        for label, count in sorted(Counter(np.asarray(values).tolist()).items())
    }


def load_reference_inputs() -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    dict[str, Any],
]:
    trajectory_path = REFERENCE_ROOT / "traj_attractors.pkl"
    roots_path = REFERENCE_ROOT / "stable_solutions.csv"
    if sha256_file(trajectory_path) != TRAJECTORY_SHA256:
        raise ValueError(f"trajectory archive hash mismatch: {trajectory_path}")
    if sha256_file(roots_path) != STABLE_ROOTS_SHA256:
        raise ValueError(f"stable-root archive hash mismatch: {roots_path}")

    with trajectory_path.open("rb") as source:
        # Trusted local pickle, accepted only after matching its frozen SHA256.
        archived = pickle.load(source)
    if not isinstance(archived, dict) or len(archived) != TRAJECTORY_ROWS:
        raise ValueError("trajectory archive must be a 10,000-entry dictionary")
    points = np.asarray([tuple(point) for point in archived], dtype=np.float64)
    truth = np.asarray(list(archived.values()), dtype=np.int64)
    counts = {
        int(label): int(count)
        for label, count in Counter(truth.tolist()).items()
    }
    if points.shape != (TRAJECTORY_ROWS, HIGH_DIMENSION):
        raise ValueError(f"trajectory point shape changed: {points.shape}")
    if counts != EXPECTED_TRAJECTORY_COUNTS:
        raise ValueError(f"trajectory truth counts changed: {counts}")

    roots = np.loadtxt(roots_path, delimiter=",", dtype=np.float64)
    if roots.shape != (2, HIGH_DIMENSION):
        raise ValueError(f"stable-root shape changed: {roots.shape}")
    if roots[0, 0] >= 0.0 or roots[1, 0] <= 0.0:
        raise ValueError("stable roots are not ordered negative then positive")

    provenance = {
        "trajectory_archive": file_reference(trajectory_path),
        "stable_roots": file_reference(roots_path),
        "trajectory_label_counts": {str(k): v for k, v in counts.items()},
        "conditioned_nonzero_trajectories": int(np.count_nonzero(truth)),
    }
    return points, truth, roots, provenance


def load_training_pairs(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    expected = (TRAINING_ROWS, 2 * HIGH_DIMENSION)
    if values.shape != expected:
        raise ValueError(f"{path} has shape {values.shape}; expected {expected}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite values")
    return values[:, :HIGH_DIMENSION], values[:, HIGH_DIMENSION:]


def load_model(path: Path, device: torch.device) -> ArchivedDynamicsAutoencoder:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{path} did not contain a state_dict")
    model = ArchivedDynamicsAutoencoder()
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def encode(
    encoder: nn.Module,
    values: NDArray[np.float64],
    *,
    device: torch.device,
    batch_size: int = 16_384,
) -> NDArray[np.float64]:
    chunks: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            tensor = torch.as_tensor(
                values[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            chunks.append(
                encoder(tensor).detach().cpu().numpy().astype(np.float64)
            )
    return np.concatenate(chunks, axis=0)


def infer_bounds(
    model: ArchivedDynamicsAutoencoder,
    current: NDArray[np.float64],
    forward: NDArray[np.float64],
    *,
    device: torch.device,
) -> LatentBounds:
    current_encoded = encode(model.encoder, current, device=device)
    forward_encoded = encode(model.encoder, forward, device=device)
    combined_lower = np.minimum(current_encoded.min(axis=0), forward_encoded.min(axis=0))
    combined_upper = np.maximum(current_encoded.max(axis=0), forward_encoded.max(axis=0))
    span = combined_upper - combined_lower
    if np.any(span <= 0.0):
        raise ValueError("encoded training bounds have zero width")
    return LatentBounds(
        lower=combined_lower - BOUNDS_MARGIN * span,
        upper=combined_upper + BOUNDS_MARGIN * span,
    )


def closed_cell_candidates(
    points: NDArray[np.float64],
    bounds: LatentBounds,
) -> ClosedCellCandidates:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != LATENT_DIMENSION:
        raise ValueError(f"encoded points must have shape (n, 2); got {values.shape}")
    shape = np.asarray([GRID_SIDE, GRID_SIDE], dtype=np.int64)
    span = bounds.upper - bounds.lower
    outside = np.any(
        (values < bounds.lower[None, :]) | (values > bounds.upper[None, :]),
        axis=1,
    )
    rows: list[NDArray[np.int64]] = []
    offsets = np.zeros(values.shape[0] + 1, dtype=np.int64)
    for point_index, point in enumerate(values):
        if outside[point_index]:
            offsets[point_index + 1] = offsets[point_index]
            continue
        clipped = np.clip(point, bounds.lower, bounds.upper)
        scaled = (clipped - bounds.lower) / span * shape
        base = np.minimum(np.floor(scaled).astype(np.int64), shape - 1)
        nearest = np.rint(scaled).astype(np.int64)
        reconstructed = bounds.lower + nearest * span / shape
        internal_boundary = (
            (nearest > 0)
            & (nearest < shape)
            & (clipped == reconstructed)
        )
        choices = [
            (int(nearest[axis] - 1), int(nearest[axis]))
            if internal_boundary[axis]
            else (int(base[axis]),)
            for axis in range(LATENT_DIMENSION)
        ]
        bins = np.asarray(list(product(*choices)), dtype=np.int64)
        cell_ids = np.unique(cmgdb_morton_cell_indices(bins, shape))
        rows.append(cell_ids)
        offsets[point_index + 1] = offsets[point_index] + cell_ids.size
    flat = (
        np.concatenate(rows).astype(np.int64, copy=False)
        if rows
        else np.empty(0, dtype=np.int64)
    )
    return ClosedCellCandidates(flat_cell_ids=flat, offsets=offsets)


def morse_attractors(morse_graph: Any) -> list[int]:
    vertices_method = getattr(morse_graph, "vertices", None)
    vertices = (
        [int(value) for value in vertices_method()]
        if callable(vertices_method)
        else list(range(int(morse_graph.num_vertices())))
    )
    edge_method = getattr(morse_graph, "edges_unreduced", None)
    if not callable(edge_method):
        edge_method = getattr(morse_graph, "edges", None)
    if not callable(edge_method):
        raise TypeError("CMGDB Morse graph has no edge iterator")
    non_attractors = {
        int(source)
        for source, target in edge_method()
        if int(source) != int(target)
    }
    return [vertex for vertex in vertices if vertex not in non_attractors]


def associate_root(
    singleton_by_candidate: NDArray[np.int32],
    candidates: ClosedCellCandidates,
    root_index: int,
    attractors: Sequence[int],
) -> int:
    start = int(candidates.offsets[root_index])
    stop = int(candidates.offsets[root_index + 1])
    matches = sorted(
        {
            int(value)
            for value in singleton_by_candidate[start:stop]
            if int(value) in attractors
        }
    )
    if len(matches) != 1:
        raise ValueError(
            f"encoded stable root {root_index} matched attractors {matches}; "
            "expected exactly one strict singleton basin"
        )
    return matches[0]


def classify_points_negative_first(
    singleton_by_candidate: NDArray[np.int32],
    candidates: ClosedCellCandidates,
    *,
    negative_attractor: int,
    positive_attractor: int,
) -> NDArray[np.int32]:
    predicted = np.full(candidates.n_points, OUTSIDE, dtype=np.int32)
    for point_index in range(candidates.n_points):
        start = int(candidates.offsets[point_index])
        stop = int(candidates.offsets[point_index + 1])
        values = singleton_by_candidate[start:stop]
        if np.any(values == negative_attractor):
            predicted[point_index] = negative_attractor
        elif np.any(values == positive_attractor):
            predicted[point_index] = positive_attractor
    return predicted


def render_strict_roa(
    output_path: Path,
    *,
    singleton_by_cell: NDArray[np.int32],
    bounds: LatentBounds,
    negative_attractor: int,
    positive_attractor: int,
    encoded_roots: NDArray[np.float64],
) -> None:
    shape = np.asarray([GRID_SIDE, GRID_SIDE], dtype=np.int64)
    axes = np.indices((GRID_SIDE, GRID_SIDE), dtype=np.int64)
    bins = np.column_stack((axes[0].ravel(), axes[1].ravel()))
    cell_ids = cmgdb_morton_cell_indices(bins, shape)
    regular = singleton_by_cell[cell_ids].reshape(GRID_SIDE, GRID_SIDE)
    image = np.zeros((GRID_SIDE, GRID_SIDE, 4), dtype=np.float64)
    image[regular == negative_attractor] = to_rgba("#2166ac", alpha=0.72)
    image[regular == positive_attractor] = to_rgba("#b2182b", alpha=0.72)
    image[regular == MULTIPLE_REACHABLE_MORSE_NODES] = to_rgba(
        "#bdbdbd", alpha=0.28
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.imshow(
        np.transpose(image, (1, 0, 2)),
        origin="lower",
        extent=(
            float(bounds.lower[0]),
            float(bounds.upper[0]),
            float(bounds.lower[1]),
            float(bounds.upper[1]),
        ),
        interpolation="nearest",
        aspect="equal",
    )
    ax.scatter(
        encoded_roots[:, 0],
        encoded_roots[:, 1],
        c=["#2166ac", "#b2182b"],
        edgecolors="black",
        linewidths=0.8,
        s=38,
        zorder=3,
    )
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title("Strict singleton regions of attraction (uniform level 16)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def source_record(dataset: int, trial: int) -> dict[str, Any]:
    dataset_dir = SOURCE_ROOT / f"run_dataset_{dataset}"
    training_path = dataset_dir / "train_data.csv"
    weights_path = dataset_dir / f"ci_model_weights_{trial}.pth"
    graph_pdf = dataset_dir / f"ci_morse_graph_{trial}.pdf"
    sets_pdf = dataset_dir / f"ci_morse_sets_{trial}.pdf"
    readme = dataset_dir / "Readme.txt"
    for path in (training_path, weights_path, graph_pdf, sets_pdf, readme):
        if not path.is_file():
            raise FileNotFoundError(path)
    numbers = [int(value) for value in re.findall(r"\d+", readme.read_text())]
    if DATASET_IC_SEEDS[dataset] not in numbers:
        raise ValueError(f"{readme} does not record IC seed {DATASET_IC_SEEDS[dataset]}")
    return {
        "dataset": dataset,
        "dataset_initial_condition_seed": DATASET_IC_SEEDS[dataset],
        "training_trial": trial,
        "training_trial_is_not_a_recorded_rng_seed": True,
        "training_data": file_reference(training_path),
        "checkpoint": file_reference(weights_path),
        "archived_adaptive_morse_graph_pdf": file_reference(graph_pdf),
        "archived_adaptive_morse_sets_pdf": file_reference(sets_pdf),
        "archived_readme": file_reference(readme),
    }


def run_one(
    *,
    dataset: int,
    trial: int,
    current: NDArray[np.float64],
    forward: NDArray[np.float64],
    trajectory_points: NDArray[np.float64],
    trajectory_truth: NDArray[np.int64],
    stable_roots: NDArray[np.float64],
    reference_provenance: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    final_dir = output_root / f"dataset_{dataset}" / f"trial_{trial}"
    temporary_dir = final_dir.with_name(f".{final_dir.name}.in_progress")
    if final_dir.exists() or temporary_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing run output: {final_dir} or {temporary_dir}"
        )
    temporary_dir.mkdir(parents=True)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    source = source_record(dataset, trial)
    weights_path = SOURCE_ROOT / f"run_dataset_{dataset}" / f"ci_model_weights_{trial}.pth"

    try:
        step = time.perf_counter()
        model = load_model(weights_path, device)
        bounds = infer_bounds(model, current, forward, device=device)
        timings["load_model_and_infer_bounds_seconds"] = time.perf_counter() - step
        write_json(
            temporary_dir / "bounds.json",
            {
                "lower": bounds.lower.tolist(),
                "upper": bounds.upper.tolist(),
                "epsilon_frac": BOUNDS_MARGIN,
                "source": "bbox of checkpoint encoder applied to dataset E(X) and E(Y)",
                "training_data_sha256": source["training_data"]["sha256"],
                "checkpoint_sha256": source["checkpoint"]["sha256"],
                "encoding_device": str(device),
            },
        )

        step = time.perf_counter()
        box_map = make_box_map_uniform_precomputed(
            model.latent_map,
            bounds,
            SUBDIVISION,
            padding=PADDING,
            device=device,
            precompute_batch_points="auto",
        )
        timings["precompute_corner_lattice_seconds"] = time.perf_counter() - step

        cmgdb_model = CMGDB.Model(
            SUBDIVISION,
            SUBDIVISION,
            SUBDIVISION,
            SUBDIV_LIMIT,
            bounds.lower.tolist(),
            bounds.upper.tolist(),
            box_map,
        )
        if not hasattr(cmgdb_model, "set_batch_map"):
            raise RuntimeError("CMGDB.Model.set_batch_map is required")
        cmgdb_model.set_batch_map(box_map.batch)
        step = time.perf_counter()
        morse_graph, map_graph = CMGDB.ComputeMorseGraph(cmgdb_model)
        timings["uniform_cmgdb_seconds"] = time.perf_counter() - step
        if int(map_graph.num_vertices()) != UNIFORM_CELLS:
            raise ValueError(
                f"uniform graph has {map_graph.num_vertices()} cells; "
                f"expected {UNIFORM_CELLS}"
            )
        has_cache = getattr(map_graph, "has_cache", None)
        if callable(has_cache) and not bool(has_cache()):
            raise RuntimeError("CMGDB did not retain its batched MapGraph cache")

        mg_dir = temporary_dir / "MG_uniform_s16"
        dot_path, sets_path = save_morse_graph_artifacts(morse_graph, mg_dir)
        dag = MorseGraph.from_dot(dot_path)
        attractors = sorted(morse_attractors(morse_graph))
        if sorted(dag.minimal) != attractors:
            raise ValueError(
                f"live attractors {attractors} disagree with DOT sinks {sorted(dag.minimal)}"
            )
        if len(attractors) != 2:
            raise ValueError(
                f"strict basin statistics require two attractors; found {attractors}"
            )

        step = time.perf_counter()
        all_cell_ids = np.arange(UNIFORM_CELLS, dtype=np.int64)
        singleton_by_cell = morse_singleton_reachability(
            map_graph,
            morse_graph,
            all_cell_ids,
        )
        timings["full_strict_singleton_query_seconds"] = time.perf_counter() - step
        if (
            not isinstance(singleton_by_cell, np.ndarray)
            or singleton_by_cell.dtype != np.int32
            or singleton_by_cell.shape != (UNIFORM_CELLS,)
            or not singleton_by_cell.flags.c_contiguous
        ):
            raise TypeError("native full-grid singleton query violated its array contract")

        step = time.perf_counter()
        encoded_points = encode(model.encoder, trajectory_points, device=device)
        encoded_roots = encode(model.encoder, stable_roots, device=device)
        point_cells = closed_cell_candidates(encoded_points, bounds)
        root_cells = closed_cell_candidates(encoded_roots, bounds)
        point_singletons = singleton_by_cell[point_cells.flat_cell_ids]
        root_singletons = singleton_by_cell[root_cells.flat_cell_ids]
        negative_attractor = associate_root(
            root_singletons, root_cells, 0, attractors
        )
        positive_attractor = associate_root(
            root_singletons, root_cells, 1, attractors
        )
        if negative_attractor == positive_attractor:
            raise ValueError("negative and positive roots associated to one attractor")
        predicted = classify_points_negative_first(
            point_singletons,
            point_cells,
            negative_attractor=negative_attractor,
            positive_attractor=positive_attractor,
        )
        statistics_result = compute_chafee_basin_statistics(
            trajectory_truth,
            predicted,
            negative_basin_label=negative_attractor,
            positive_basin_label=positive_attractor,
        )
        timings["encode_and_classify_seconds"] = time.perf_counter() - step

        strict_path = mg_dir / "regions_of_attraction_strict_singleton.npz"
        with strict_path.open("xb") as destination:
            np.savez_compressed(
                destination,
                cell_ids=all_cell_ids,
                singleton_node_by_cell=singleton_by_cell,
                minimal_attractor_nodes=np.asarray(attractors, dtype=np.int32),
                grid_shape=np.asarray([GRID_SIDE, GRID_SIDE], dtype=np.int64),
                bounds_lower=np.asarray(bounds.lower, dtype=np.float64),
                bounds_upper=np.asarray(bounds.upper, dtype=np.float64),
            )
        strict_metadata_path = (
            mg_dir / "regions_of_attraction_strict_singleton.json"
        )
        write_json(
            strict_metadata_path,
            {
                "schema_version": 1,
                "status": "complete",
                "method": "CMGDB.MorseSingletonReachability",
                "semantics": (
                    "complete reachable Morse-node set must equal exactly one "
                    "singleton Morse node; authoritative archive-equivalent basin grid"
                ),
                "uniform_cells": UNIFORM_CELLS,
                "grid_shape": [GRID_SIDE, GRID_SIDE],
                "minimal_attractor_nodes": attractors,
                "counts_by_singleton_label": label_counts(singleton_by_cell),
                "sentinels": {
                    str(NO_REACHABLE_MORSE_NODE): "complete reachable set is empty",
                    str(MULTIPLE_REACHABLE_MORSE_NODES): (
                        "complete reachable set has multiple Morse nodes"
                    ),
                },
                "not_lca_collapsed": True,
                "used_for_headline_trajectory_statistics": True,
            },
        )

        query_ids = np.unique(
            np.concatenate((point_cells.flat_cell_ids, root_cells.flat_cell_ids))
        )
        query_path = mg_dir / "reference_singleton_reachability_queries.npz"
        np.savez_compressed(
            query_path,
            queried_cell_ids=query_ids,
            singleton_node_by_queried_cell=singleton_by_cell[query_ids],
            point_candidate_cell_ids=point_cells.flat_cell_ids,
            point_candidate_offsets=point_cells.offsets,
            point_singleton_nodes=point_singletons,
            point_basin_labels=predicted,
            root_candidate_cell_ids=root_cells.flat_cell_ids,
            root_candidate_offsets=root_cells.offsets,
            root_singleton_nodes=root_singletons,
            encoded_stable_roots=encoded_roots,
        )
        np.save(temporary_dir / "trajectory_basin_labels.npy", predicted)
        np.save(temporary_dir / "encoded_stable_roots.npy", encoded_roots)

        render_strict_roa(
            mg_dir / "regions_of_attraction_strict_singleton.png",
            singleton_by_cell=singleton_by_cell,
            bounds=bounds,
            negative_attractor=negative_attractor,
            positive_attractor=positive_attractor,
            encoded_roots=encoded_roots,
        )

        counts = statistics_result.counts()
        percentages = statistics_result.percentages()
        combined_correct_count = (
            counts["correctly_classified_in_negative_basin"]
            + counts["correctly_classified_in_positive_basin"]
        )
        combined_correct_percentage = (
            100.0
            * combined_correct_count
            / statistics_result.conditioned_trajectories
        )
        edge_count = int(sum(len(values) for values in dag.edges.values()))
        cached_edges_method = getattr(map_graph, "num_cached_edges", None)
        cached_edges = (
            int(cached_edges_method()) if callable(cached_edges_method) else None
        )
        total_seconds = time.perf_counter() - started
        timings["total_seconds"] = total_seconds
        statistics_payload = {
            "schema_version": 1,
            "status": "complete",
            "dataset": dataset,
            "dataset_initial_condition_seed": DATASET_IC_SEEDS[dataset],
            "training_trial": trial,
            "training_trial_is_not_a_recorded_rng_seed": True,
            "method": (
                "Exact archived singleton-all-reachable-Morse-set basin semantics "
                "on a uniform level-16 CMGDB graph"
            ),
            "source": source,
            "reference_inputs": reference_provenance,
            "bounds": {
                "lower": bounds.lower.tolist(),
                "upper": bounds.upper.tolist(),
                "epsilon_frac": BOUNDS_MARGIN,
            },
            "cmgdb": {
                "subdiv_init": SUBDIVISION,
                "subdiv_min": SUBDIVISION,
                "subdiv_max": SUBDIVISION,
                "subdiv_limit": SUBDIV_LIMIT,
                "padding": PADDING,
                "uniform_cells": UNIFORM_CELLS,
                "grid_shape": [GRID_SIDE, GRID_SIDE],
                "morse_nodes": len(dag.nodes),
                "morse_edges": edge_count,
                "attractor_nodes": attractors,
                "cached_edges": cached_edges,
                "topology_only": True,
                "conley_indices_required_for_basin_statistics": False,
            },
            "root_association": {
                "status": "valid",
                "encoded_roots_negative_then_positive": encoded_roots.tolist(),
                "negative_root_candidate_cells": root_cells.candidates(0).tolist(),
                "positive_root_candidate_cells": root_cells.candidates(1).tolist(),
                "negative_basin_label": negative_attractor,
                "positive_basin_label": positive_attractor,
                "distinct_attractor_basins": True,
            },
            "classification": {
                "rule": (
                    "complete reachable Morse-node set equals the corresponding "
                    "singleton attractor"
                ),
                "closed_cell_boundary_policy": (
                    "negative basin first, then positive basin, matching archive"
                ),
                "counts_by_predicted_point_label": label_counts(predicted),
                "queried_unique_cells": int(query_ids.size),
            },
            "statistics": {
                "total_trajectories": statistics_result.total_trajectories,
                "excluded_zero_trajectories": (
                    statistics_result.excluded_zero_trajectories
                ),
                "conditioned_trajectories": (
                    statistics_result.conditioned_trajectories
                ),
                "counts": counts,
                "percentages": percentages,
                "combined_correct_count": combined_correct_count,
                "combined_correct_percentage": combined_correct_percentage,
            },
            "timings": timings,
        }
        write_json(temporary_dir / "basin_statistics.json", statistics_payload)

        output_names = {
            "bounds": temporary_dir / "bounds.json",
            "morse_graph": dot_path,
            "morse_sets": sets_path,
            "strict_roa": strict_path,
            "strict_roa_metadata": strict_metadata_path,
            "strict_roa_png": (
                mg_dir / "regions_of_attraction_strict_singleton.png"
            ),
            "query_artifact": query_path,
            "trajectory_basin_labels": (
                temporary_dir / "trajectory_basin_labels.npy"
            ),
            "encoded_stable_roots": temporary_dir / "encoded_stable_roots.npy",
            "basin_statistics": temporary_dir / "basin_statistics.json",
        }
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "completed_at_utc": utc_now(),
            "source": source,
            "protocol": {
                "no_retraining": True,
                "uniform_statistics_grid": True,
                "adaptive_archive_pdfs_not_consumed_for_roa": True,
                "subdivisions": [SUBDIVISION, SUBDIVISION, SUBDIVISION],
                "padding": PADDING,
                "strict_singleton_semantics": True,
            },
            "root_association_status": "valid",
            "statistics": statistics_payload["statistics"],
            "topology": statistics_payload["cmgdb"],
            "timings": timings,
            "outputs": {
                name: file_reference(path, relative_to=temporary_dir)
                for name, path in output_names.items()
            },
        }
        write_json(temporary_dir / "run_manifest.json", manifest)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir.rename(final_dir)
        print(
            f"[dataset {dataset} trial {trial}] "
            f"correct={combined_correct_percentage:.6f}% "
            f"nodes={len(dag.nodes)} attractors={attractors} "
            f"elapsed={total_seconds:.2f}s",
            flush=True,
        )
        return {
            "status": "complete",
            "dataset": dataset,
            "dataset_initial_condition_seed": DATASET_IC_SEEDS[dataset],
            "training_trial": trial,
            "checkpoint_sha256": source["checkpoint"]["sha256"],
            "training_data_sha256": source["training_data"]["sha256"],
            "output_dir": str(final_dir.resolve()),
            "morse_nodes": len(dag.nodes),
            "morse_edges": edge_count,
            "attractor_count": len(attractors),
            "attractor_nodes": attractors,
            "root_association_status": "valid",
            "negative_basin_label": negative_attractor,
            "positive_basin_label": positive_attractor,
            "conditioned_trajectories": statistics_result.conditioned_trajectories,
            **counts,
            **{
                f"{name}_percentage": value
                for name, value in percentages.items()
            },
            "combined_correct_count": combined_correct_count,
            "combined_correct_percentage": combined_correct_percentage,
            "elapsed_seconds": total_seconds,
        }
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "dataset": dataset,
            "training_trial": trial,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "failed_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "source": source,
        }
        write_json(temporary_dir / "failure.json", failure)
        failed_dir = final_dir.with_name(f"{final_dir.name}_failed")
        if failed_dir.exists():
            raise FileExistsError(failed_dir) from error
        temporary_dir.rename(failed_dir)
        print(
            f"[dataset {dataset} trial {trial}] FAILED: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return {
            "status": "failed",
            "dataset": dataset,
            "dataset_initial_condition_seed": DATASET_IC_SEEDS[dataset],
            "training_trial": trial,
            "output_dir": str(failed_dir.resolve()),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }


def load_completed_row(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "basin_statistics.json").read_text())
    source = payload["source"]
    counts = payload["statistics"]["counts"]
    percentages = payload["statistics"]["percentages"]
    return {
        "status": "complete",
        "dataset": int(payload["dataset"]),
        "dataset_initial_condition_seed": int(
            payload["dataset_initial_condition_seed"]
        ),
        "training_trial": int(payload["training_trial"]),
        "checkpoint_sha256": source["checkpoint"]["sha256"],
        "training_data_sha256": source["training_data"]["sha256"],
        "output_dir": str(run_dir.resolve()),
        "morse_nodes": int(payload["cmgdb"]["morse_nodes"]),
        "morse_edges": int(payload["cmgdb"]["morse_edges"]),
        "attractor_count": len(payload["cmgdb"]["attractor_nodes"]),
        "attractor_nodes": payload["cmgdb"]["attractor_nodes"],
        "root_association_status": payload["root_association"]["status"],
        "negative_basin_label": int(
            payload["root_association"]["negative_basin_label"]
        ),
        "positive_basin_label": int(
            payload["root_association"]["positive_basin_label"]
        ),
        "conditioned_trajectories": int(
            payload["statistics"]["conditioned_trajectories"]
        ),
        **{name: int(value) for name, value in counts.items()},
        **{
            f"{name}_percentage": float(value)
            for name, value in percentages.items()
        },
        "combined_correct_count": int(
            payload["statistics"]["combined_correct_count"]
        ),
        "combined_correct_percentage": float(
            payload["statistics"]["combined_correct_percentage"]
        ),
        "elapsed_seconds": float(payload["timings"]["total_seconds"]),
    }


def describe(values: Sequence[float]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("cannot describe an empty metric")
    return {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "sample_standard_deviation": (
            statistics.stdev(numeric) if len(numeric) > 1 else 0.0
        ),
        "median": statistics.median(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def metric_descriptives(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "combined_correct_percentage": describe(
            [float(row["combined_correct_percentage"]) for row in rows]
        ),
        "outside_both_basins_percentage": describe(
            [float(row["outside_both_basins_percentage"]) for row in rows]
        ),
        "total_misclassified_percentage": describe(
            [
                float(row["misclassified_in_negative_basin_percentage"])
                + float(row["misclassified_in_positive_basin_percentage"])
                for row in rows
            ]
        ),
        "morse_nodes": describe([float(row["morse_nodes"]) for row in rows]),
    }


def grouped_summary(
    completed: Sequence[dict[str, Any]],
    *,
    field: str,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for value in sorted({int(row[field]) for row in completed}):
        selected = [row for row in completed if int(row[field]) == value]
        entry: dict[str, Any] = {
            field: value,
            "n_runs": len(selected),
            "descriptive": metric_descriptives(selected),
        }
        if field == "dataset":
            entry["dataset_initial_condition_seed"] = DATASET_IC_SEEDS[value]
        else:
            entry["training_trial_is_not_a_recorded_rng_seed"] = True
        groups.append(entry)
    return groups


def aggregate_payload(
    rows: list[dict[str, Any]],
    *,
    pipeline_wall_clock_seconds: float | None = None,
) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "complete"]
    scores = [float(row["combined_correct_percentage"]) for row in completed]
    pooled_conditioned = sum(
        int(row["conditioned_trajectories"]) for row in completed
    )
    pooled_counts = {
        field: sum(int(row[field]) for row in completed)
        for field in STATISTIC_COUNT_FIELDS
    }
    pooled_percentages = (
        {
            field: 100.0 * count / pooled_conditioned
            for field, count in pooled_counts.items()
        }
        if pooled_conditioned
        else {}
    )
    pooled_correct = (
        pooled_counts.get("correctly_classified_in_negative_basin", 0)
        + pooled_counts.get("correctly_classified_in_positive_basin", 0)
    )
    retrospective_exceedances = sum(
        score > ARCHIVED_COMBINED_CORRECT_PERCENTAGE for score in scores
    )
    return {
        "schema_version": 1,
        "status": (
            "complete"
            if len(completed) == 15 and len(rows) == 15
            else "complete_with_failures_or_subset"
        ),
        "design": {
            "datasets": 5,
            "training_trials_per_dataset": 3,
            "total_checkpoints": 15,
            "dataset_initial_condition_seeds": {
                str(key): value for key, value in DATASET_IC_SEEDS.items()
            },
            "training_trials_are_not_recorded_rng_seeds": True,
            "shared_evaluation_trajectories": TRAJECTORY_ROWS,
            "shared_conditioned_nonzero_trajectories": 7_862,
            "hierarchical_dependence_warning": (
                "the 15 rows reuse one evaluation archive and are grouped three "
                "training initializations within each of five training datasets"
            ),
        },
        "run_counts": {
            "requested": len(rows),
            "complete": len(completed),
            "failed": len(rows) - len(completed),
            "exactly_two_attractors": sum(
                int(row.get("attractor_count", 0) == 2) for row in completed
            ),
            "valid_distinct_root_association": sum(
                int(row.get("root_association_status") == "valid")
                for row in completed
            ),
        },
        "descriptive_across_completed_runs": (
            metric_descriptives(completed) if completed else None
        ),
        # Retain the original convenience key for downstream readers.
        "combined_correct_percentage_descriptive": (
            describe(scores) if scores else None
        ),
        "pooled_conditioned_statistics": {
            "conditioned_rows": pooled_conditioned,
            "counts": pooled_counts,
            "percentages": pooled_percentages,
            "combined_correct_count": pooled_correct,
            "combined_correct_percentage": (
                100.0 * pooled_correct / pooled_conditioned
                if pooled_conditioned
                else None
            ),
            "interpretation": (
                "descriptive only: the same 7,862 conditioned evaluation "
                "trajectories are reused for every checkpoint, so these are "
                "not independent pooled observations"
            ),
        },
        "by_dataset": (
            grouped_summary(completed, field="dataset") if completed else []
        ),
        "by_training_trial": (
            grouped_summary(completed, field="training_trial")
            if completed
            else []
        ),
        "retrospective_archived_benchmark_comparison": {
            "archived_combined_correct_percentage": (
                ARCHIVED_COMBINED_CORRECT_PERCENTAGE
            ),
            "runs_strictly_exceeding": retrospective_exceedances,
            "completed_runs": len(completed),
            "fraction_strictly_exceeding": (
                retrospective_exceedances / len(completed) if completed else None
            ),
            "warning": (
                "retrospective descriptive comparison only; the benchmark was "
                "known before this archive audit and the 15 rows are hierarchical"
            ),
        },
        "runtime": {
            "pipeline_wall_clock_seconds": pipeline_wall_clock_seconds,
            "sum_of_per_run_elapsed_seconds": sum(
                float(row["elapsed_seconds"]) for row in completed
            ),
            "per_run_elapsed_seconds": (
                describe([float(row["elapsed_seconds"]) for row in completed])
                if completed
                else None
            ),
        },
        "results": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def write_readme(output_root: Path, aggregate: dict[str, Any]) -> None:
    rows = [row for row in aggregate["results"] if row["status"] == "complete"]
    lines = [
        "# Archived Chafee-Infante 2-D 5x3 basin audit",
        "",
        "No model was retrained. Each row recomputes the archived separate uniform",
        "level-16 strict-singleton basin analysis from one archived checkpoint.",
        "The archived adaptive PDFs are preserved by reference but are not used",
        "to infer regions of attraction.",
        "",
        "| Dataset | IC seed | Trial | Correct (%) | Outside (%) | Misclassified | Morse nodes |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        misclassified = int(row["misclassified_in_negative_basin"]) + int(
            row["misclassified_in_positive_basin"]
        )
        lines.append(
            f"| {row['dataset']} | {row['dataset_initial_condition_seed']} | "
            f"{row['training_trial']} | {row['combined_correct_percentage']:.6f} | "
            f"{row['outside_both_basins_percentage']:.6f} | {misclassified} | "
            f"{row['morse_nodes']} |"
        )
    descriptive = aggregate["combined_correct_percentage_descriptive"]
    if descriptive is not None:
        all_metrics = aggregate["descriptive_across_completed_runs"]
        outside = all_metrics["outside_both_basins_percentage"]
        misclassified = all_metrics["total_misclassified_percentage"]
        nodes = all_metrics["morse_nodes"]
        pooled = aggregate["pooled_conditioned_statistics"]
        benchmark = aggregate["retrospective_archived_benchmark_comparison"]
        runtime = aggregate["runtime"]
        lines.extend(
            [
                "",
                "Across the 15 archived checkpoints (descriptive, not an",
                "independence claim):",
                "",
                "| Metric | Mean | Sample SD | Median | Range |",
                "|---|---:|---:|---:|---:|",
                (
                    "| Correct (%) | "
                    f"{descriptive['mean']:.6f} | "
                    f"{descriptive['sample_standard_deviation']:.6f} | "
                    f"{descriptive['median']:.6f} | "
                    f"{descriptive['minimum']:.6f}-{descriptive['maximum']:.6f} |"
                ),
                (
                    "| Outside (%) | "
                    f"{outside['mean']:.6f} | "
                    f"{outside['sample_standard_deviation']:.6f} | "
                    f"{outside['median']:.6f} | "
                    f"{outside['minimum']:.6f}-{outside['maximum']:.6f} |"
                ),
                (
                    "| Total misclassified (%) | "
                    f"{misclassified['mean']:.6f} | "
                    f"{misclassified['sample_standard_deviation']:.6f} | "
                    f"{misclassified['median']:.6f} | "
                    f"{misclassified['minimum']:.6f}-{misclassified['maximum']:.6f} |"
                ),
                (
                    "| Uniform Morse nodes | "
                    f"{nodes['mean']:.3f} | "
                    f"{nodes['sample_standard_deviation']:.3f} | "
                    f"{nodes['median']:.3f} | "
                    f"{nodes['minimum']:.0f}-{nodes['maximum']:.0f} |"
                ),
                "",
                "Pooled descriptive counts (the same evaluation archive is",
                "reused 15 times, so this is not an independent-observation pool):",
                "",
                (
                    f"- Conditioned rows: {pooled['conditioned_rows']:,}; "
                    f"outside {pooled['counts']['outside_both_basins']:,}, "
                    f"misclassified-negative "
                    f"{pooled['counts']['misclassified_in_negative_basin']:,}, "
                    f"misclassified-positive "
                    f"{pooled['counts']['misclassified_in_positive_basin']:,}, "
                    f"correct-negative "
                    f"{pooled['counts']['correctly_classified_in_negative_basin']:,}, "
                    f"correct-positive "
                    f"{pooled['counts']['correctly_classified_in_positive_basin']:,}."
                ),
                (
                    f"- Pooled correct: "
                    f"{pooled['combined_correct_percentage']:.6f}%; "
                    f"pooled outside: "
                    f"{pooled['percentages']['outside_both_basins']:.6f}%; "
                    f"pooled total misclassified: "
                    f"{pooled['percentages']['misclassified_in_negative_basin'] + pooled['percentages']['misclassified_in_positive_basin']:.6f}%."
                ),
                "",
                (
                    f"Retrospectively, {benchmark['runs_strictly_exceeding']}/"
                    f"{benchmark['completed_runs']} checkpoints exceed the known "
                    f"archived reference score of "
                    f"{benchmark['archived_combined_correct_percentage']:.6f}%. "
                    "This is a descriptive post-hoc comparison, not a prospective test."
                ),
                "",
                (
                    f"Initial pipeline wall time: "
                    f"{runtime['pipeline_wall_clock_seconds']:.2f} seconds; "
                    f"sum of per-run times: "
                    f"{runtime['sum_of_per_run_elapsed_seconds']:.2f} seconds."
                ),
                "",
                "Grouped metric summaries by dataset and by unseeded training",
                "trial are recorded in `aggregate_statistics.json`.",
                "",
                "Each `dataset_N/trial_M/` directory contains the uniform Morse",
                "DOT/CSV, the full strict-singleton RoA grid, root association,",
                "trajectory labels, detailed statistics, hashes, and timings.",
                "",
            ]
        )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def validate_package(output_root: Path, *, require_all: bool = True) -> dict[str, Any]:
    results_path = output_root / "results.json"
    aggregate_path = output_root / "aggregate_statistics.json"
    if not results_path.is_file() or not aggregate_path.is_file():
        raise FileNotFoundError("aggregate result files are missing")
    rows = json.loads(results_path.read_text())["results"]
    if require_all and len(rows) != 15:
        raise ValueError(f"expected 15 rows; found {len(rows)}")
    checkpoint_hashes: set[str] = set()
    errors: list[str] = []
    validated = 0
    for row in rows:
        if row["status"] != "complete":
            errors.append(
                f"dataset {row['dataset']} trial {row['training_trial']} failed"
            )
            continue
        run_dir = Path(row["output_dir"])
        try:
            payload = json.loads((run_dir / "basin_statistics.json").read_text())
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            checkpoint_hashes.add(payload["source"]["checkpoint"]["sha256"])
            if manifest["status"] != "complete":
                raise ValueError("run manifest is not complete")
            if payload["root_association"]["status"] != "valid":
                raise ValueError("root association is not valid")
            if len(payload["cmgdb"]["attractor_nodes"]) != 2:
                raise ValueError("run does not have exactly two attractors")
            counts = payload["statistics"]["counts"]
            if sum(int(value) for value in counts.values()) != 7_862:
                raise ValueError("conditioned table counts do not conserve 7,862")
            strict_path = (
                run_dir
                / "MG_uniform_s16"
                / "regions_of_attraction_strict_singleton.npz"
            )
            query_path = (
                run_dir
                / "MG_uniform_s16"
                / "reference_singleton_reachability_queries.npz"
            )
            with np.load(strict_path) as strict, np.load(query_path) as query:
                full = np.asarray(
                    strict["singleton_node_by_cell"], dtype=np.int32
                )
                ids = np.asarray(query["queried_cell_ids"], dtype=np.int64)
                subset = np.asarray(
                    query["singleton_node_by_queried_cell"], dtype=np.int32
                )
                if full.shape != (UNIFORM_CELLS,):
                    raise ValueError("strict RoA does not cover 65,536 cells")
                if not np.array_equal(full[ids], subset):
                    raise ValueError("query subset disagrees with full strict RoA")
                point_ids = np.asarray(
                    query["point_candidate_cell_ids"], dtype=np.int64
                )
                point_values = np.asarray(
                    query["point_singleton_nodes"], dtype=np.int32
                )
                if not np.array_equal(full[point_ids], point_values):
                    raise ValueError("point queries disagree with full strict RoA")
            for name, reference in manifest["outputs"].items():
                artifact = run_dir / reference["path"]
                if not artifact.is_file():
                    raise FileNotFoundError(f"{name}: {artifact}")
                if sha256_file(artifact) != reference["sha256"]:
                    raise ValueError(f"{name} hash mismatch")
            validated += 1
        except Exception as error:
            errors.append(
                f"dataset {row['dataset']} trial {row['training_trial']}: {error}"
            )
    if require_all and len(checkpoint_hashes) != 15:
        errors.append(
            f"expected 15 unique checkpoint hashes; found {len(checkpoint_hashes)}"
        )
    if errors:
        raise ValueError("package validation failed: " + "; ".join(errors))
    report = {
        "schema_version": 1,
        "status": "validated",
        "validated_at_utc": utc_now(),
        "runs_validated": validated,
        "unique_checkpoint_hashes": len(checkpoint_hashes),
        "checks": [
            "all requested run manifests complete",
            "exactly two uniform attractors per run",
            "valid distinct encoded-root association per run",
            "conditioned statistics conserve 7,862 trajectories",
            "strict RoA covers all 65,536 uniform cells",
            "persisted query values equal the corresponding full-grid values",
            "all named artifact hashes match run manifests",
        ],
    }
    write_json(output_root / "validation_report.json", report)
    return report


def parse_int_selection(raw: Sequence[int] | None, allowed: set[int]) -> list[int]:
    values = sorted(allowed if not raw else set(raw))
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"selection contains unsupported values: {sorted(unknown)}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help=(
            "root of the archived reference inputs "
            "(computations/run_dataset_N/, traj_attractors.pkl, stable_solutions.csv)"
        ),
    )
    parser.add_argument("--datasets", type=int, nargs="+")
    parser.add_argument("--trials", type=int, nargs="+")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    global REFERENCE_ROOT, SOURCE_ROOT
    REFERENCE_ROOT = args.reference_root.resolve()
    SOURCE_ROOT = REFERENCE_ROOT / "computations"

    output_root = args.output_root.resolve()
    if args.validate_only:
        report = validate_package(output_root, require_all=True)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    datasets = parse_int_selection(args.datasets, set(range(1, 6)))
    trials = parse_int_selection(args.trials, {1, 2, 3})
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"output root exists; use --resume to reuse completed runs: {output_root}"
        )
    prior_manifest_path = output_root / "package_manifest.json"
    prior_pipeline_wall_clock: float | None = None
    if args.resume and prior_manifest_path.is_file():
        prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        prior_elapsed = prior_manifest.get("elapsed_seconds")
        if isinstance(prior_elapsed, (int, float)) and float(prior_elapsed) > 0.0:
            prior_pipeline_wall_clock = float(prior_elapsed)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trajectory_points, trajectory_truth, stable_roots, reference_provenance = (
        load_reference_inputs()
    )
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    rows: list[dict[str, Any]] = []
    new_runs = 0
    reused_runs = 0
    for dataset in datasets:
        dataset_dir = SOURCE_ROOT / f"run_dataset_{dataset}"
        current, forward = load_training_pairs(dataset_dir / "train_data.csv")
        for trial in trials:
            run_dir = output_root / f"dataset_{dataset}" / f"trial_{trial}"
            if run_dir.is_dir() and args.resume:
                rows.append(load_completed_row(run_dir))
                reused_runs += 1
                print(
                    f"[dataset {dataset} trial {trial}] reused completed output",
                    flush=True,
                )
                continue
            new_runs += 1
            rows.append(
                run_one(
                    dataset=dataset,
                    trial=trial,
                    current=current,
                    forward=forward,
                    trajectory_points=trajectory_points,
                    trajectory_truth=trajectory_truth,
                    stable_roots=stable_roots,
                    reference_provenance=reference_provenance,
                    output_root=output_root,
                    device=device,
                )
            )
        del current, forward

    rows.sort(key=lambda row: (int(row["dataset"]), int(row["training_trial"])))
    results_payload = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "results": rows,
    }
    write_json(output_root / "results.json", results_payload)
    write_csv(output_root / "results.csv", rows)
    refresh_elapsed = time.perf_counter() - started
    reported_pipeline_wall_clock = (
        prior_pipeline_wall_clock
        if new_runs == 0 and prior_pipeline_wall_clock is not None
        else refresh_elapsed
    )
    aggregate = aggregate_payload(
        rows,
        pipeline_wall_clock_seconds=reported_pipeline_wall_clock,
    )
    write_json(output_root / "aggregate_statistics.json", aggregate)
    write_readme(output_root, aggregate)
    refresh_elapsed = time.perf_counter() - started
    package_manifest = {
        "schema_version": 1,
        "status": aggregate["status"],
        "generated_at_utc": utc_now(),
        "elapsed_seconds": reported_pipeline_wall_clock,
        "aggregation_refresh_seconds": refresh_elapsed,
        "new_runs_in_this_invocation": new_runs,
        "reused_runs_in_this_invocation": reused_runs,
        "source_root": str(SOURCE_ROOT.resolve()),
        "reference_root": str(REFERENCE_ROOT.resolve()),
        "script": file_reference(Path(__file__)),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "device": str(device),
            "cmgdb_module": str(Path(CMGDB.__file__).resolve()),
        },
        "reference_inputs": reference_provenance,
        "outputs": {
            name: file_reference(output_root / name, relative_to=output_root)
            for name in (
                "results.json",
                "results.csv",
                "aggregate_statistics.json",
                "README.md",
            )
        },
    }
    write_json(output_root / "package_manifest.json", package_manifest)
    if len(rows) == 15:
        validate_package(output_root, require_all=True)
    print(
        f"Completed {sum(row['status'] == 'complete' for row in rows)}/{len(rows)} "
        f"runs; initial pipeline wall time "
        f"{reported_pipeline_wall_clock:.2f}s, this invocation "
        f"{refresh_elapsed:.2f}s: {output_root}",
        flush=True,
    )
    return 0 if all(row["status"] == "complete" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
