"""Run three isolated literal repeats of the canonical Chafee d=1 sweep.

This is a one-shot exploratory driver.  The outer process freezes a plan and
launches three fresh Python subprocesses.  Each subprocess trains the same five
canonical full-batch candidates (seeds 0 through 4), analyzes every completed
checkpoint, and then augments every available uniform topology with:

* the complete 256-cell CMGDB ``MorseSingletonReachability`` lookup used by
  the coauthor reference strict basin rule; and
* the distinct blocker/LCA exact-RoA representation.

The augmentation also recovers and persists uniform and adaptive Morse
topology when the established analyzer cannot form a two-root basin statistic.
Consequently, an unavailable statistic does not discard otherwise valid
topological output.  All output roots are fail-if-present and no checkpoint is
selected or ranked during computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import analyze_chafee_d1_full_batch_sweep as analyze
import chafee_latent_dimension_study as study
import sweep_chafee_d1_full_batch as sweep
from latentdynamics.analysis.cmgdb_roa import compute_exact_roa, save_exact_roa
from latentdynamics.analysis.morse_graph_parser import MorseGraph

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = study.DEFAULT_REFERENCE_ROOT
# Rebound in main() when --reference-root is supplied.
REFERENCE_ROOT = DEFAULT_REFERENCE_ROOT
DEFAULT_OUTPUT = (
    sweep.LATENT_1D_OUTPUT_ROOT / "exploratory_full_batch_literal_repeats_v1"
)
DEFAULT_EXISTING_OUTPUT = (
    sweep.LATENT_1D_OUTPUT_ROOT
    / "exploratory_full_batch_repeatability_5seed_analysis_package_v1"
)
DEFAULT_EXISTING_SOURCES = tuple(
    sweep.LATENT_1D_OUTPUT_ROOT
    / f"exploratory_full_batch_repeatability_5seed_rerun_{index:02d}"
    for index in (1, 2, 3)
)
EXPECTED_EXISTING_PLAN_SHA256 = (
    "d515de0063b91370447a368615e67738f398e8574c91b242e93949ddf20a0081"
)
DRIVER_IMPLEMENTATION = Path(__file__).resolve()

BASE_SEEDS = (0, 1, 2, 3, 4)
REPEAT_INDICES = (0, 1, 2)
EPOCHS = 4_000
LEARNING_RATE = 0.003
CANONICAL_DEVICE = study.CANONICAL_TRAINING_BACKEND
EXPECTED_UNIFORM_CELLS = study.RESOLUTIONS[1].uniform_cells

PLAN_SCHEMA_VERSION = 1
REPLICATE_MANIFEST_SCHEMA_VERSION = 1
AGGREGATE_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LiteralRepeatTrial:
    """One seed inside one process-isolated repeat."""

    repeat_index: int
    plan_index: int
    base_seed: int
    run_id: str

    @property
    def training_seed(self) -> int:
        """The literal training seed; the repeat index never changes it."""

        return self.base_seed

    @property
    def snapshot_name(self) -> str:
        return f"run_{self.plan_index:03d}_{self.run_id}"

    def training_spec(self) -> sweep.FullBatchRunSpec:
        return sweep.FullBatchRunSpec(
            run_id=self.run_id,
            seed=self.training_seed,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_fresh_outer(output_root: Path) -> Path:
    target = output_root.resolve()
    protected = (
        sweep.CANONICAL_RUN.resolve(),
        sweep.DEFAULT_OUTPUT.resolve(),
        analyze.DEFAULT_ANALYSIS.resolve(),
        analyze.CANONICAL_4K.resolve(),
        analyze.CANONICAL_10K.resolve(),
    )
    for root in protected:
        if target == root or _is_within(target, root) or _is_within(root, target):
            raise ValueError(f"repeat-sweep target {target} overlaps protected directory {root}")
    if output_root.is_symlink():
        raise ValueError(f"repeat-sweep target must not be a symlink: {output_root}")
    if output_root.exists():
        raise FileExistsError(
            f"repeat-sweep target already exists; refusing to overwrite: {output_root}"
        )
    return target


def repeat_trials(
    repeat_index: int,
) -> tuple[LiteralRepeatTrial, ...]:
    if repeat_index not in REPEAT_INDICES:
        raise ValueError(f"repeat_index must be one of {REPEAT_INDICES}")
    return tuple(
        LiteralRepeatTrial(
            repeat_index=repeat_index,
            plan_index=plan_index,
            base_seed=seed,
            run_id=f"seed_{seed:02d}_lr3e3_e4000",
        )
        for plan_index, seed in enumerate(BASE_SEEDS, start=1)
    )


def all_trials() -> tuple[LiteralRepeatTrial, ...]:
    return tuple(trial for repeat_index in REPEAT_INDICES for trial in repeat_trials(repeat_index))


def _file_reference(path: Path, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": (
            str(resolved.relative_to(root.resolve()))
            if root is not None and _is_within(resolved, root.resolve())
            else str(resolved)
        ),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _implementation_inventory() -> dict[str, dict[str, Any]]:
    paths = {
        "repeat_driver": DRIVER_IMPLEMENTATION,
        "training_sweep": Path(sweep.__file__).resolve(),
        "batch_analyzer": Path(analyze.__file__).resolve(),
        "dimension_study": Path(study.__file__).resolve(),
    }
    return {name: _file_reference(path) for name, path in paths.items()}


def _copy_verified(source: Path, destination: Path, expected_hash: str) -> Path:
    if _sha256(source) != expected_hash:
        raise ValueError(f"input changed before copy: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as target, source.open("rb") as src:
        shutil.copyfileobj(src, target)
    shutil.copystat(source, destination)
    if _sha256(destination) != expected_hash:
        raise ValueError(f"copied input hash mismatch: {destination}")
    return destination


def _copy_reference_inputs(
    inputs: study.ExactInputs,
    destination: Path,
) -> dict[str, dict[str, Any]]:
    files = {
        "train_data.csv": inputs.train_data,
        "traj_attractors.pkl": inputs.trajectory_labels,
        "stable_solutions.csv": inputs.stable_roots,
    }
    records: dict[str, dict[str, Any]] = {}
    for name, source in files.items():
        copied = _copy_verified(source, destination / name, inputs.hashes[name])
        records[name] = _file_reference(copied, destination.parent)
    return records


def _copy_truth_inputs(
    inputs: study.ExactInputs,
    destination: Path,
) -> dict[str, dict[str, Any]]:
    files = {
        "traj_attractors.pkl": inputs.trajectory_labels,
        "stable_solutions.csv": inputs.stable_roots,
    }
    records: dict[str, dict[str, Any]] = {}
    for name, source in files.items():
        copied = _copy_verified(source, destination / name, inputs.hashes[name])
        records[name] = _file_reference(copied, destination.parent)
    return records


def _outer_plan(inputs: study.ExactInputs) -> dict[str, Any]:
    trials = all_trials()
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "frozen_before_training",
        "created_at_utc": _utc_now(),
        "purpose": "exploratory literal-repeat robustness sweep",
        "process_isolation": {
            "outer_process_launches": len(REPEAT_INDICES),
            "one_fresh_python_process_per_repeat": True,
            "one_fresh_output_root_per_repeat": True,
            "repeat_indices": list(REPEAT_INDICES),
        },
        "literal_repeat_semantics": {
            "training_seed_equals_base_seed": True,
            "repeat_index_changes_seed": False,
            "same_five_seed_plan_in_each_process": True,
        },
        "training_protocol": {
            "backend": CANONICAL_DEVICE,
            "optimizer": "Adam",
            "full_batch": True,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "scheduler": {
                "factor": 0.5,
                "patience": 100,
                "threshold": 1e-4,
                "min_lr": 1e-6,
            },
            "objective": ("reference decoded reconstruction plus decoded one-step prediction"),
            "training_rows": sweep.TRAINING_ROWS,
            "trajectory_truth_or_roa_used_for_training": False,
            "early_stopping": False,
        },
        "analysis_protocol": {
            "all_completed_checkpoints_analyzed": True,
            "uniform_subdivision": 8,
            "uniform_cells": EXPECTED_UNIFORM_CELLS,
            "adaptive_subdivision": 11,
            "persist_full_strict_singleton_lookup": True,
            "persist_exact_blocker_lca_roa_separately": True,
            "recover_topology_when_basin_statistics_unavailable": True,
            "statistics_used_for_training_or_checkpoint_selection": False,
        },
        "replicates": [
            {
                "repeat_index": repeat_index,
                "directory": f"repeat_{repeat_index:02d}",
                "trials": [
                    {
                        **asdict(trial),
                        "training_seed": trial.training_seed,
                        "training_spec": asdict(trial.training_spec()),
                    }
                    for trial in repeat_trials(repeat_index)
                ],
            }
            for repeat_index in REPEAT_INDICES
        ],
        "trial_count": len(trials),
        "archived_inputs": inputs.provenance(),
        "implementations": _implementation_inventory(),
        "canonical_training_sources": sweep._current_source_provenance(),
    }
    payload["frozen_payload_sha256"] = _payload_sha256(payload)
    return payload


def _validate_existing_frozen_sweep(
    frozen: analyze.FrozenSweep,
    *,
    expected_plan_sha256: str | None,
) -> None:
    if (
        expected_plan_sha256 is not None
        and frozen.plan_sha256 != expected_plan_sha256
    ):
        raise ValueError(
            "existing sweep plan hash mismatch: "
            f"expected {expected_plan_sha256}, observed {frozen.plan_sha256}"
        )
    expected_specs = tuple(trial.training_spec() for trial in repeat_trials(0))
    observed_specs = tuple(row.spec for row in frozen.inventory)
    if observed_specs != expected_specs:
        raise ValueError(
            f"{frozen.root} does not contain the exact canonical five-seed plan"
        )
    if len(frozen.candidates) != len(expected_specs):
        raise ValueError(
            f"{frozen.root} does not have five completed, valid candidates"
        )


def _verify_existing_sweeps(
    source_sweeps: Sequence[Path],
    *,
    expected_plan_sha256: str | None,
) -> tuple[analyze.FrozenSweep, ...]:
    if len(source_sweeps) != len(REPEAT_INDICES):
        raise ValueError("existing analysis mode requires exactly three sweep roots")
    resolved = tuple(Path(path).resolve() for path in source_sweeps)
    if len(set(resolved)) != len(resolved):
        raise ValueError("existing sweep roots must be distinct")
    frozen = tuple(analyze._verify_source_sweep(path) for path in resolved)
    observed_hashes = {item.plan_sha256 for item in frozen}
    if len(observed_hashes) != 1:
        raise ValueError(
            f"existing sweeps do not share one frozen plan hash: {observed_hashes}"
        )
    for item in frozen:
        _validate_existing_frozen_sweep(
            item,
            expected_plan_sha256=expected_plan_sha256,
        )
    return frozen


def _existing_analysis_plan(
    *,
    frozen_sweeps: Sequence[analyze.FrozenSweep],
    inputs: study.ExactInputs,
) -> dict[str, Any]:
    sources = []
    for repeat_index, frozen in zip(
        REPEAT_INDICES,
        frozen_sweeps,
        strict=True,
    ):
        sources.append(
            {
                "repeat_index": repeat_index,
                "source_root": str(frozen.root),
                "source_plan": _file_reference(frozen.plan_path),
                "plan_payload_sha256": frozen.plan_sha256,
                "source_read_only": True,
                "candidates": [
                    {
                        "plan_index": candidate.plan_index,
                        "run_id": candidate.spec.run_id,
                        "seed": candidate.spec.seed,
                        "checkpoint_sha256": candidate.checkpoint_sha256,
                        "history_sha256": candidate.history_sha256,
                        "completion_sha256": candidate.completion_sha256,
                        "artifact_manifest_sha256": (
                            candidate.artifact_manifest_sha256
                        ),
                    }
                    for candidate in frozen.candidates
                ],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "frozen_before_analysis",
        "created_at_utc": _utc_now(),
        "mode": "analyze_verified_existing_training_sweeps",
        "training_performed_by_this_driver": False,
        "source_mutation_permitted": False,
        "source_copy_policy": (
            "copy each verified 1.8 MB frozen sweep into a fresh replicate root; "
            "analyze only the verified copy"
        ),
        "shared_source_plan_sha256": frozen_sweeps[0].plan_sha256,
        "process_isolation": {
            "one_fresh_python_process_per_repeat_analysis": True,
            "one_fresh_output_root_per_repeat": True,
            "repeat_indices": list(REPEAT_INDICES),
        },
        "literal_repeat_semantics": {
            "training_seed_equals_base_seed": True,
            "repeat_index_changes_seed": False,
            "same_five_seed_plan_in_each_source": True,
        },
        "analysis_protocol": {
            "all_completed_checkpoints_analyzed": True,
            "uniform_subdivision": 8,
            "uniform_cells": EXPECTED_UNIFORM_CELLS,
            "adaptive_subdivision": 11,
            "persist_full_strict_singleton_lookup": True,
            "persist_exact_blocker_lca_roa_separately": True,
            "recover_topology_when_basin_statistics_unavailable": True,
            "statistics_used_for_training_or_checkpoint_selection": False,
        },
        "sources": sources,
        "trial_count": len(BASE_SEEDS) * len(REPEAT_INDICES),
        "archived_inputs": inputs.provenance(),
        "implementations": _implementation_inventory(),
    }
    payload["frozen_payload_sha256"] = _payload_sha256(payload)
    return payload


def _require_canonical_backend() -> None:
    if CANONICAL_DEVICE != "mps":
        raise ValueError(f"expected canonical backend 'mps', observed {CANONICAL_DEVICE!r}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("canonical repeat sweep requires an available MPS backend")


def _validate_strict_query_subset(
    singleton_by_cell: np.ndarray,
    queried_cell_ids: np.ndarray,
    singleton_by_queried_cell: np.ndarray,
) -> None:
    full = np.asarray(singleton_by_cell, dtype=np.int32)
    ids = np.asarray(queried_cell_ids, dtype=np.int64)
    queried = np.asarray(singleton_by_queried_cell, dtype=np.int32)
    if full.ndim != 1 or ids.ndim != 1 or queried.ndim != 1:
        raise ValueError("strict singleton arrays must be one-dimensional")
    if ids.shape != queried.shape:
        raise ValueError("queried cell ids and singleton labels differ in shape")
    if np.any(ids < 0) or np.any(ids >= full.size):
        raise ValueError("queried cell id lies outside the full lookup")
    if not np.array_equal(full[ids], queried):
        raise ValueError("queried singleton reachability disagrees with full-grid lookup")


def _label_counts(values: np.ndarray) -> dict[str, int]:
    return {
        str(int(label)): int(count)
        for label, count in sorted(Counter(np.asarray(values).tolist()).items())
    }


def _verify_persisted_uniform_topology(
    *,
    paths: analyze.ExactRunPaths,
    recovered_morse_graph: Any,
) -> dict[str, Any]:
    """Require persisted node ids, edges, and labelled boxes to match recovery."""

    persisted_dot = paths.uniform / "morse_graph"
    persisted_sets = paths.uniform / "morse_sets"
    if not persisted_dot.is_file() or not persisted_sets.is_file():
        raise FileNotFoundError("persisted uniform Morse graph/set pair is incomplete")
    with tempfile.TemporaryDirectory(
        prefix=".uniform_recovery_check_",
        dir=paths.run,
    ) as temporary:
        check_dir = Path(temporary)
        check_dot, check_sets = study.save_morse_graph_artifacts(
            recovered_morse_graph,
            check_dir,
        )
        persisted_dag = MorseGraph.from_dot(persisted_dot)
        recovered_dag = MorseGraph.from_dot(check_dot)
        persisted_edges = {
            (int(source), int(target))
            for source, targets in persisted_dag.edges.items()
            for target in targets
        }
        recovered_edges = {
            (int(source), int(target))
            for source, targets in recovered_dag.edges.items()
            for target in targets
        }
        if persisted_dag.nodes != recovered_dag.nodes or persisted_edges != recovered_edges:
            raise ValueError(
                "persisted uniform Morse graph node ids/edges disagree with "
                "the reconstructed CMGDB graph"
            )
        persisted_rows = np.loadtxt(
            persisted_sets,
            delimiter=",",
            ndmin=2,
            dtype=np.float64,
        )
        recovered_rows = np.loadtxt(
            check_sets,
            delimiter=",",
            ndmin=2,
            dtype=np.float64,
        )
        persisted_sorted = persisted_rows[
            np.lexsort(
                tuple(persisted_rows[:, i] for i in reversed(range(persisted_rows.shape[1])))
            )
        ]
        recovered_sorted = recovered_rows[
            np.lexsort(
                tuple(recovered_rows[:, i] for i in reversed(range(recovered_rows.shape[1])))
            )
        ]
        if not np.array_equal(persisted_sorted, recovered_sorted):
            raise ValueError(
                "persisted uniform Morse-set boxes/labels disagree with "
                "the reconstructed CMGDB graph"
            )
    return {
        "status": "validated",
        "node_ids": persisted_dag.nodes,
        "edges": sorted([list(edge) for edge in persisted_edges]),
        "morse_set_rows": int(persisted_rows.shape[0]),
    }


def _save_full_roa_artifacts(
    *,
    paths: analyze.ExactRunPaths,
    bounds: Any,
    morse_graph: Any,
    map_graph: Any,
) -> dict[str, Any]:
    uniform = paths.uniform
    strict_path = uniform / "regions_of_attraction_strict_singleton.npz"
    strict_metadata = uniform / "regions_of_attraction_strict_singleton.json"
    exact_path = uniform / "regions_of_attraction_exact.npz"
    exact_metadata = uniform / "regions_of_attraction_exact.json"
    for target in (strict_path, strict_metadata, exact_path, exact_metadata):
        if target.exists():
            raise FileExistsError(f"full-RoA artifact already exists; refusing overwrite: {target}")

    n_cells = int(map_graph.num_vertices())
    if n_cells != EXPECTED_UNIFORM_CELLS:
        raise ValueError(f"uniform map has {n_cells} cells; expected {EXPECTED_UNIFORM_CELLS}")
    cell_ids = np.arange(n_cells, dtype=np.int64)
    strict = study._native_singleton_reachability(
        map_graph,
        morse_graph,
        cell_ids,
    )
    if strict.shape != (n_cells,) or strict.dtype != np.int32:
        raise TypeError("full strict singleton lookup has an invalid array contract")
    attractors = np.asarray(
        study._morse_attractors(morse_graph),
        dtype=np.int32,
    )

    # Analyses persisted before the artifact rename carry the older
    # "marcio_..." basename; both hold the same query payload.
    query_candidates = (
        uniform / "reference_singleton_reachability_queries.npz",
        uniform / "marcio_singleton_reachability_queries.npz",
    )
    query_path = next(
        (path for path in query_candidates if path.is_file()),
        query_candidates[0],
    )
    query_validation: dict[str, Any]
    if query_path.is_file():
        with np.load(query_path) as query:
            queried_ids = np.asarray(query["queried_cell_ids"], dtype=np.int64)
            queried_values = np.asarray(
                query["singleton_node_by_queried_cell"],
                dtype=np.int32,
            )
            point_candidate_ids = np.asarray(
                query["point_candidate_cell_ids"],
                dtype=np.int64,
            )
            point_singletons = np.asarray(
                query["point_singleton_nodes"],
                dtype=np.int32,
            )
            root_candidate_ids = np.asarray(
                query["root_candidate_cell_ids"],
                dtype=np.int64,
            )
            root_singletons = np.asarray(
                query["root_singleton_nodes"],
                dtype=np.int32,
            )
            point_labels = np.asarray(
                query["point_basin_labels"],
                dtype=np.int32,
            )
            query_encoded_roots = np.asarray(
                query["encoded_stable_roots"],
                dtype=np.float64,
            )
        _validate_strict_query_subset(strict, queried_ids, queried_values)
        _validate_strict_query_subset(
            strict,
            point_candidate_ids,
            point_singletons,
        )
        _validate_strict_query_subset(
            strict,
            root_candidate_ids,
            root_singletons,
        )
        trajectory_labels_path = paths.run / "trajectory_basin_labels.npy"
        encoded_roots_path = paths.run / "encoded_stable_roots.npy"
        if not trajectory_labels_path.is_file() or not encoded_roots_path.is_file():
            raise FileNotFoundError(
                "queried-cell artifact exists without its trajectory/root arrays"
            )
        if not np.array_equal(np.load(trajectory_labels_path), point_labels):
            raise ValueError("trajectory_basin_labels.npy disagrees with queried-cell artifact")
        if not np.array_equal(np.load(encoded_roots_path), query_encoded_roots):
            raise ValueError("encoded_stable_roots.npy disagrees with queried-cell artifact")
        query_validation = {
            "status": "validated_exact_subset",
            "queried_cells": int(queried_ids.size),
            "point_candidate_entries": int(point_candidate_ids.size),
            "root_candidate_entries": int(root_candidate_ids.size),
            "trajectory_and_root_arrays_match_query_artifact": True,
            "artifact": str(query_path.relative_to(paths.run)),
            "sha256": _sha256(query_path),
        }
    else:
        query_validation = {
            "status": "unavailable",
            "reason": (
                "root/trajectory query artifact was not produced because "
                "reference basin association/statistics were unavailable"
            ),
        }

    uniform.mkdir(parents=True, exist_ok=True)
    with strict_path.open("xb") as destination:
        np.savez_compressed(
            destination,
            cell_ids=cell_ids,
            singleton_node_by_cell=strict,
            minimal_attractor_nodes=attractors,
            grid_shape=np.asarray([n_cells], dtype=np.int64),
            bounds_lower=np.asarray(bounds.lower, dtype=np.float64),
            bounds_upper=np.asarray(bounds.upper, dtype=np.float64),
        )
    strict_payload = {
        "schema_version": 1,
        "status": "complete",
        "method": "CMGDB.MorseSingletonReachability",
        "semantics": (
            "complete reachable Morse-node set must equal exactly one singleton "
            "Morse node; this is the authoritative full-grid lookup for "
            "reference-equivalent strict basin classification"
        ),
        "uniform_cells": n_cells,
        "cell_ids_complete_zero_based_range": True,
        "minimal_attractor_nodes": attractors.tolist(),
        "counts_by_singleton_label": _label_counts(strict),
        "sentinels": {
            "-1": "NO_MORSE_NODE: complete reachable set is empty",
            "-2": "MULTIPLE_MORSE_NODES: complete reachable set is not a singleton",
        },
        "queried_subset_validation": query_validation,
        "not_equivalent_to_exact_blocker_lca_roa": True,
        "negative_sentinels_are_not_cmgdb_roa_sentinels": True,
    }
    _write_json_exclusive(strict_metadata, strict_payload)

    dag = MorseGraph.from_dot(uniform / "morse_graph")
    exact = compute_exact_roa(
        map_graph,
        morse_graph,
        dag,
        lower_bounds=np.asarray(bounds.lower, dtype=np.float64),
        upper_bounds=np.asarray(bounds.upper, dtype=np.float64),
        collapse_to_lca=True,
    )
    if np.asarray(exact.box_roa).shape != (n_cells,):
        raise ValueError("exact blocker/LCA RoA does not cover all 256 cells")
    if exact.reach_mask is None or np.asarray(exact.reach_mask).shape != (n_cells,):
        raise ValueError("exact blocker/LCA reach mask does not cover all cells")
    if exact.minimal_order is None or not np.array_equal(
        np.asarray(exact.minimal_order, dtype=np.int32),
        np.asarray(sorted(dag.minimal), dtype=np.int32),
    ):
        raise ValueError("exact blocker/LCA minimal order disagrees with Morse DAG")
    if np.any(np.asarray(exact.box_roa) == -3):
        raise ValueError("LCA-collapsed exact RoA unexpectedly retains MULTI=-3")
    saved_exact = save_exact_roa(exact, uniform)
    if saved_exact.resolve() != exact_path.resolve():
        raise ValueError("exact-RoA writer returned an unexpected destination")
    exact_payload = {
        "schema_version": 1,
        "status": "complete",
        "method": "compute_exact_roa(collapse_to_lca=True)",
        "semantics": (
            "other recurrent Morse sets block reverse reachability; cells "
            "reaching multiple minimal sets collapse to their Morse-poset LCA"
        ),
        "uniform_cells": n_cells,
        "counts_by_box_roa_label": _label_counts(exact.box_roa),
        "minimal_order": (
            []
            if exact.minimal_order is None
            else np.asarray(exact.minimal_order, dtype=np.int32).tolist()
        ),
        "reach_mask_persisted": exact.reach_mask is not None,
        "used_for_reference_trajectory_statistics": False,
        "different_from_strict_singleton_lookup": True,
        "negative_sentinels": {
            "-1": "BOUNDARY",
            "-2": "ESCAPE",
            "-3": "MULTI (must be absent after LCA collapse)",
        },
    }
    _write_json_exclusive(exact_metadata, exact_payload)
    return {
        "status": "complete",
        "uniform_cells": n_cells,
        "strict_singleton": _file_reference(strict_path, paths.run),
        "strict_singleton_metadata": _file_reference(strict_metadata, paths.run),
        "exact_blocker_lca": _file_reference(exact_path, paths.run),
        "exact_blocker_lca_metadata": _file_reference(exact_metadata, paths.run),
        "queried_subset_validation": query_validation,
    }


def _uniform_topology_available(paths: analyze.ExactRunPaths) -> bool:
    return all((paths.uniform / name).is_file() for name in ("morse_graph", "morse_sets"))


def _adaptive_topology_available(paths: analyze.ExactRunPaths) -> bool:
    return all((paths.adaptive / name).is_file() for name in ("morse_graph", "morse_sets"))


def _recover_topology_and_roa(
    *,
    run: Path,
    device: torch.device,
    batch_points: int | str,
) -> dict[str, Any]:
    """Recover topology independently of two-root statistical validity."""

    paths = analyze.ExactRunPaths(output_root=run, dimension=1)
    manifest_path = run / "topology_roa_augmentation.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": _utc_now(),
        "uniform_topology": {"status": "unavailable"},
        "full_uniform_roa": {"status": "unavailable"},
        "adaptive_topology": {"status": "unavailable"},
        "basin_statistics": {
            "status": ("available" if paths.stats.is_file() else "unavailable_or_invalid"),
            "path": (str(paths.stats.relative_to(paths.run)) if paths.stats.is_file() else None),
        },
    }
    started = time.perf_counter()
    bounds = None
    morse_graph = None
    map_graph = None
    try:
        if not paths.bounds.is_file():
            raise FileNotFoundError("bounds unavailable; standard analysis did not reach topology")
        if not (paths.coarse_table / "coarse_values.npy").is_file():
            raise FileNotFoundError(
                "coarse lookup unavailable; standard analysis did not reach topology"
            )
        bounds = study._load_bounds(paths.bounds, 1)
        loaded_box_map = study.HierarchicalPrecomputedBoxMap.load(
            paths.coarse_table,
            mmap_mode="r",
        )
        # Fine-block precomputation updates this same persisted table later in
        # the standard analysis.  Re-wrap only its immutable level-8 values so
        # uniform recovery cannot accidentally depend on any level-11 block.
        box_map = study.HierarchicalPrecomputedBoxMap(
            lower=loaded_box_map.lower,
            upper=loaded_box_map.upper,
            coarse_subdiv=loaded_box_map.coarse_subdiv,
            fine_subdiv=loaded_box_map.fine_subdiv,
            coarse_values=loaded_box_map.coarse_values,
            padding=loaded_box_map.padding,
        )
        resolution = study.RESOLUTIONS[1]
        morse_graph, map_graph, duration, lookup_status = study._run_lookup_cmgdb(
            box_map,
            bounds,
            subdiv_init=resolution.uniform_init,
            subdiv_min=resolution.uniform_min,
            subdiv_max=resolution.uniform_max,
            compute_conley=False,
        )
        n_cells = int(map_graph.num_vertices())
        if n_cells != EXPECTED_UNIFORM_CELLS:
            raise ValueError(
                f"uniform CMGDB returned {n_cells} cells; expected {EXPECTED_UNIFORM_CELLS}"
            )
        recovered = not _uniform_topology_available(paths)
        if recovered:
            study.save_morse_graph_artifacts(morse_graph, paths.uniform)
        if not _uniform_topology_available(paths):
            raise FileNotFoundError("uniform Morse topology was not persisted")
        topology_validation = _verify_persisted_uniform_topology(
            paths=paths,
            recovered_morse_graph=morse_graph,
        )
        cache_metadata = study._map_graph_cache_metadata(map_graph)
        uniform_marker = paths.stage_marker("uniform")
        marker_validation: dict[str, Any]
        if uniform_marker.is_file():
            marker = _read_json(uniform_marker)
            if int(marker.get("map_cells", -1)) != n_cells:
                raise ValueError("uniform stage marker map-cell count disagrees with recovery")
            marker_edges = marker.get("cached_edges")
            recovered_edges = cache_metadata.get("cached_edges")
            if (
                marker_edges is not None
                and recovered_edges is not None
                and int(marker_edges) != int(recovered_edges)
            ):
                raise ValueError("uniform stage marker cached-edge count disagrees with recovery")
            marker_validation = {
                "status": "validated",
                "map_cells": n_cells,
                "cached_edges": recovered_edges,
            }
        else:
            marker_validation = {
                "status": "unavailable",
                "reason": (
                    "standard uniform stage did not complete its statistical "
                    "precondition; recovered topology is independently validated"
                ),
            }
        uniform_record = {
            "status": "complete",
            "recovered_after_standard_analysis_failure": recovered,
            "lookup_duration_seconds": duration,
            "map_cells": n_cells,
            "morse_nodes": int(morse_graph.num_vertices()),
            "minimal_attractor_nodes": study._morse_attractors(morse_graph),
            "lookup": lookup_status,
            "cache": cache_metadata,
            "persisted_topology_validation": topology_validation,
            "standard_uniform_marker_validation": marker_validation,
            "morse_graph": _file_reference(
                paths.uniform / "morse_graph",
                paths.run,
            ),
            "morse_sets": _file_reference(
                paths.uniform / "morse_sets",
                paths.run,
            ),
        }
        record["uniform_topology"] = uniform_record
    except Exception as error:
        record["uniform_topology"] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    if (
        record["uniform_topology"]["status"] == "complete"
        and bounds is not None
        and morse_graph is not None
        and map_graph is not None
    ):
        try:
            record["full_uniform_roa"] = _save_full_roa_artifacts(
                paths=paths,
                bounds=bounds,
                morse_graph=morse_graph,
                map_graph=map_graph,
            )
        except Exception as error:
            record["full_uniform_roa"] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }

    try:
        if not paths.bounds.is_file():
            raise FileNotFoundError("bounds unavailable for adaptive recovery")
        if not (paths.coarse_table / "coarse_values.npy").is_file():
            raise FileNotFoundError("coarse lookup unavailable for adaptive recovery")
        if not paths.stage_marker("precompute-fine").is_file():
            study._run_precompute_fine(
                paths,
                device=device,
                batch_points=batch_points,
            )
        if not _adaptive_topology_available(paths):
            study._run_adaptive(paths, topology_only=False)
        if not _adaptive_topology_available(paths):
            raise FileNotFoundError("adaptive Morse topology was not persisted")
        record["adaptive_topology"] = {
            "status": "complete",
            "morse_graph": _file_reference(
                paths.adaptive / "morse_graph",
                paths.run,
            ),
            "morse_sets": _file_reference(
                paths.adaptive / "morse_sets",
                paths.run,
            ),
        }
    except Exception as error:
        record["adaptive_topology"] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }

    phase_statuses = {
        name: record[name]["status"]
        for name in (
            "uniform_topology",
            "full_uniform_roa",
            "adaptive_topology",
        )
    }
    record["status"] = (
        "complete"
        if all(status == "complete" for status in phase_statuses.values())
        else "complete_with_failures"
    )
    record["phase_statuses"] = phase_statuses
    record["completed_at_utc"] = _utc_now()
    record["elapsed_seconds"] = time.perf_counter() - started
    _write_json_exclusive(manifest_path, record)
    return record


def _analysis_rows(analysis_root: Path) -> list[dict[str, Any]]:
    path = analysis_root / "results_by_run.json"
    if not path.is_file():
        return []
    payload = _read_json(path)
    rows = payload.get("results_in_frozen_plan_order", [])
    if not isinstance(rows, list):
        raise ValueError(f"{path} has an invalid results row list")
    return [row for row in rows if isinstance(row, dict)]


def _trial_artifact_inventory(
    run: Path,
    augmentation: dict[str, Any] | None,
) -> dict[str, Any]:
    paths = analyze.ExactRunPaths(output_root=run, dimension=1)
    named = {
        "checkpoint": paths.models / "autoencoder.pt",
        "uniform_morse_graph": paths.uniform / "morse_graph",
        "uniform_morse_sets": paths.uniform / "morse_sets",
        "adaptive_morse_graph": paths.adaptive / "morse_graph",
        "adaptive_morse_sets": paths.adaptive / "morse_sets",
        "strict_singleton_full": (paths.uniform / "regions_of_attraction_strict_singleton.npz"),
        "strict_singleton_metadata": (
            paths.uniform / "regions_of_attraction_strict_singleton.json"
        ),
        "exact_blocker_lca": paths.uniform / "regions_of_attraction_exact.npz",
        "exact_blocker_lca_metadata": (paths.uniform / "regions_of_attraction_exact.json"),
        # Analyses persisted before the artifact rename carry the older
        # "marcio_..." basename; both hold the same query payload.
        "strict_singleton_queries": next(
            (
                path
                for path in (
                    paths.uniform / "reference_singleton_reachability_queries.npz",
                    paths.uniform / "marcio_singleton_reachability_queries.npz",
                )
                if path.is_file()
            ),
            paths.uniform / "reference_singleton_reachability_queries.npz",
        ),
        "trajectory_basin_labels": run / "trajectory_basin_labels.npy",
        "encoded_stable_roots": run / "encoded_stable_roots.npy",
        "basin_statistics": paths.stats,
        "analysis_manifest": run / "analysis_manifest.json",
        "augmentation_manifest": run / "topology_roa_augmentation.json",
    }
    files = {
        name: (
            {"status": "available", **_file_reference(path, run)}
            if path.is_file()
            else {"status": "unavailable", "path": str(path.relative_to(run))}
        )
        for name, path in named.items()
    }
    topology_complete = all(
        files[name]["status"] == "available"
        for name in (
            "uniform_morse_graph",
            "uniform_morse_sets",
            "adaptive_morse_graph",
            "adaptive_morse_sets",
        )
    )
    full_roa_complete = all(
        files[name]["status"] == "available"
        for name in (
            "strict_singleton_full",
            "strict_singleton_metadata",
            "exact_blocker_lca",
            "exact_blocker_lca_metadata",
        )
    )
    stats_available = files["basin_statistics"]["status"] == "available"
    return {
        "topology_status": "complete" if topology_complete else "unavailable",
        "full_uniform_roa_status": ("complete" if full_roa_complete else "unavailable"),
        "basin_statistics_status": ("available" if stats_available else "unavailable_or_invalid"),
        "augmentation_status": (None if augmentation is None else augmentation.get("status")),
        "files": files,
    }


def _copy_verified_sweep(
    source: analyze.FrozenSweep,
    destination: Path,
) -> analyze.FrozenSweep:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    shutil.copytree(
        source.root,
        destination,
        copy_function=shutil.copy2,
        symlinks=False,
    )
    copied = analyze._verify_source_sweep(destination)
    if copied.plan_sha256 != source.plan_sha256:
        raise ValueError("copied sweep plan payload hash changed")
    source_chain = [
        (
            candidate.spec,
            candidate.checkpoint_sha256,
            candidate.history_sha256,
            candidate.completion_sha256,
            candidate.artifact_manifest_sha256,
        )
        for candidate in source.candidates
    ]
    copied_chain = [
        (
            candidate.spec,
            candidate.checkpoint_sha256,
            candidate.history_sha256,
            candidate.completion_sha256,
            candidate.artifact_manifest_sha256,
        )
        for candidate in copied.candidates
    ]
    if copied_chain != source_chain:
        raise ValueError("copied sweep candidate artifact chain changed")
    return copied


def _run_replicate(
    *,
    repeat_index: int,
    replicate_root: Path,
    quiet: bool,
    source_sweep: Path | None = None,
    expected_source_plan_sha256: str | None = None,
) -> dict[str, Any]:
    if repeat_index not in REPEAT_INDICES:
        raise ValueError(f"invalid repeat index {repeat_index}")
    if replicate_root.is_symlink() or replicate_root.exists():
        raise FileExistsError(
            f"replicate root already exists; refusing overwrite: {replicate_root}"
        )
    _require_canonical_backend()
    replicate_root.mkdir(parents=True, exist_ok=False)
    trials = repeat_trials(repeat_index)
    inputs = study.verify_exact_inputs(REFERENCE_ROOT)
    references = _copy_reference_inputs(inputs, replicate_root / "reference_inputs")
    manifest_path = replicate_root / "replicate_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": REPLICATE_MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "repeat_index": repeat_index,
        "process_id": os.getpid(),
        "started_at_utc": _utc_now(),
        "training_seed_rule": "training_seed == base_seed",
        "trials_expected": len(trials),
        "reference_inputs": references,
    }
    _write_json_exclusive(manifest_path, manifest)
    training_root = replicate_root / "training_sweep"
    analysis_root = replicate_root / "analysis"
    training_summary: dict[str, Any] | None = None
    analysis_summary: dict[str, Any] | None = None
    top_level_errors: list[dict[str, str]] = []

    if source_sweep is None:
        try:
            training_summary = sweep.run_sweep(
                output_dir=training_root,
                device_name=CANONICAL_DEVICE,
                run_specs=tuple(trial.training_spec() for trial in trials),
                verbose=not quiet,
            )
        except Exception as error:
            top_level_errors.append(
                {
                    "stage": "training_sweep",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    else:
        try:
            source = analyze._verify_source_sweep(source_sweep)
            _validate_existing_frozen_sweep(
                source,
                expected_plan_sha256=expected_source_plan_sha256,
            )
        except Exception as error:
            top_level_errors.append(
                {
                    "stage": "verify_existing_source",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            source = None
        if source is not None:
            try:
                copied = _copy_verified_sweep(source, training_root)
                training_summary = {
                    "mode": "copied_existing_verified_sweep",
                    "training_performed": False,
                    "source_root": str(source.root),
                    "source_plan_sha256": source.plan_sha256,
                    "copied_plan_sha256": copied.plan_sha256,
                    "source_plan": _file_reference(source.plan_path),
                    "copied_plan": _file_reference(
                        copied.plan_path,
                        replicate_root,
                    ),
                    "all_runs_completed": True,
                    "runs": [
                        {
                            "run_id": candidate.spec.run_id,
                            "seed": candidate.spec.seed,
                            "status": "copied_completed_valid",
                            "checkpoint_sha256": candidate.checkpoint_sha256,
                        }
                        for candidate in copied.candidates
                    ],
                }
            except Exception as error:
                top_level_errors.append(
                    {
                        "stage": "copy_existing_source",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )

    if training_root.is_dir():
        try:
            analysis_summary = analyze.run_batch_analysis(
                source_sweep=training_root,
                analysis_root=analysis_root,
                device_name=CANONICAL_DEVICE,
                batch_points="auto",
            )
        except Exception as error:
            top_level_errors.append(
                {
                    "stage": "batch_analysis",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

    analysis_truth_inputs: dict[str, dict[str, Any]] = {}
    if analysis_root.is_dir():
        try:
            analysis_truth_inputs = _copy_truth_inputs(
                inputs,
                analysis_root / "reference_inputs",
            )
        except Exception as error:
            top_level_errors.append(
                {
                    "stage": "copy_analysis_truth_inputs",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

    rows = _analysis_rows(analysis_root) if analysis_root.is_dir() else []
    row_by_run = {str(row.get("run_id")): row for row in rows if isinstance(row.get("run_id"), str)}
    augmentation_rows: list[dict[str, Any]] = []
    device = torch.device(CANONICAL_DEVICE)
    for trial in trials:
        run = analysis_root / "by_run" / trial.snapshot_name
        augmentation: dict[str, Any] | None = None
        if run.is_dir():
            augmentation = _recover_topology_and_roa(
                run=run,
                device=device,
                batch_points="auto",
            )
        row = row_by_run.get(trial.run_id, {})
        augmentation_rows.append(
            {
                **asdict(trial),
                "training_seed": trial.training_seed,
                "training_status": next(
                    (
                        item.get("status")
                        for item in (training_summary or {}).get("runs", [])
                        if item.get("run_id") == trial.run_id
                    ),
                    "unavailable",
                ),
                "analysis_status": row.get("status", "unavailable"),
                "analysis_error_stage": row.get("error_stage", ""),
                "analysis_error_type": row.get("error_type", ""),
                "analysis_error_message": row.get("error_message", ""),
                "correct_combined_percent": row.get("correct_combined_percent"),
                "analysis_directory": (
                    str(run.relative_to(replicate_root)) if run.is_dir() else None
                ),
                "artifacts": (
                    _trial_artifact_inventory(run, augmentation)
                    if run.is_dir()
                    else {
                        "topology_status": "unavailable",
                        "full_uniform_roa_status": "unavailable",
                        "basin_statistics_status": "unavailable_or_invalid",
                        "augmentation_status": None,
                        "files": {},
                    }
                ),
            }
        )
    _write_json_exclusive(
        analysis_root / "topology_roa_augmentation.json",
        {
            "schema_version": 1,
            "repeat_index": repeat_index,
            "trials": augmentation_rows,
        },
    ) if analysis_root.is_dir() else None

    topology_complete = sum(
        row["artifacts"]["topology_status"] == "complete" for row in augmentation_rows
    )
    full_roa_complete = sum(
        row["artifacts"]["full_uniform_roa_status"] == "complete" for row in augmentation_rows
    )
    stats_available = sum(
        row["artifacts"]["basin_statistics_status"] == "available" for row in augmentation_rows
    )
    source_unchanged: dict[str, Any] | None = None
    if source_sweep is not None:
        try:
            final_source = analyze._verify_source_sweep(source_sweep)
            _validate_existing_frozen_sweep(
                final_source,
                expected_plan_sha256=expected_source_plan_sha256,
            )
            source_unchanged = {
                "status": "verified_unchanged_after_analysis",
                "root": str(final_source.root),
                "plan": _file_reference(final_source.plan_path),
                "plan_payload_sha256": final_source.plan_sha256,
            }
        except Exception as error:
            source_unchanged = {
                "status": "verification_failed",
                "root": str(Path(source_sweep).resolve()),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            top_level_errors.append(
                {
                    "stage": "verify_source_unchanged_after_analysis",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    manifest.update(
        {
            "status": (
                "complete"
                if (
                    not top_level_errors
                    and topology_complete == len(trials)
                    and full_roa_complete == len(trials)
                )
                else "complete_with_failures"
            ),
            "completed_at_utc": _utc_now(),
            "training_summary": training_summary,
            "analysis_summary_status": (
                None if analysis_summary is None else analysis_summary.get("status")
            ),
            "analysis_truth_inputs": analysis_truth_inputs,
            "read_only_source_post_analysis_verification": source_unchanged,
            "top_level_errors": top_level_errors,
            "counts": {
                "trials": len(trials),
                "topology_complete": topology_complete,
                "full_uniform_roa_complete": full_roa_complete,
                "basin_statistics_available": stats_available,
                "basin_statistics_unavailable_or_invalid": (len(trials) - stats_available),
            },
            "trials": augmentation_rows,
        }
    )
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _replicate_command(
    *,
    repeat_index: int,
    replicate_root: Path,
    quiet: bool,
    source_sweep: Path | None = None,
    expected_source_plan_sha256: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(DRIVER_IMPLEMENTATION),
        "--_replicate-index",
        str(repeat_index),
        "--_replicate-root",
        str(replicate_root),
        "--reference-root",
        str(REFERENCE_ROOT),
    ]
    if source_sweep is not None:
        command.extend(("--_source-sweep", str(source_sweep.resolve())))
    if expected_source_plan_sha256 is not None:
        command.extend(
            ("--_expected-source-plan-sha256", expected_source_plan_sha256)
        )
    if quiet:
        command.append("--quiet")
    return command


def _seed_repeat_diagnostics(
    replicate_manifests: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in BASE_SEEDS}
    for replicate in replicate_manifests:
        for trial in replicate.get("trials", []):
            if isinstance(trial, dict) and trial.get("base_seed") in by_seed:
                by_seed[int(trial["base_seed"])].append(trial)
    diagnostics: list[dict[str, Any]] = []
    for seed in BASE_SEEDS:
        trials = sorted(
            by_seed[seed],
            key=lambda row: int(row.get("repeat_index", -1)),
        )
        scores = [
            float(row["correct_combined_percent"])
            for row in trials
            if isinstance(row.get("correct_combined_percent"), (int, float))
            and math.isfinite(float(row["correct_combined_percent"]))
        ]
        checkpoint_hashes: list[str] = []
        for row in trials:
            checkpoint = row.get("artifacts", {}).get("files", {}).get("checkpoint", {})
            digest = checkpoint.get("sha256")
            if isinstance(digest, str):
                checkpoint_hashes.append(digest)
        diagnostics.append(
            {
                "base_seed": seed,
                "repeats_reported": [row.get("repeat_index") for row in trials],
                "checkpoint_sha256_by_repeat": checkpoint_hashes,
                "unique_checkpoint_hashes": len(set(checkpoint_hashes)),
                "checkpoints_bitwise_identical_across_available_repeats": (
                    len(checkpoint_hashes) == len(REPEAT_INDICES)
                    and len(set(checkpoint_hashes)) == 1
                ),
                "statistics_available_repeats": len(scores),
                "correct_combined_percent": {
                    "values": scores,
                    "mean": statistics.fmean(scores) if scores else None,
                    "population_std": (statistics.pstdev(scores) if scores else None),
                    "minimum": min(scores) if scores else None,
                    "maximum": max(scores) if scores else None,
                    "range": (max(scores) - min(scores)) if scores else None,
                },
            }
        )
    return diagnostics


def _aggregate_manifest(
    *,
    output_root: Path,
    plan_path: Path,
    process_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    replicate_manifests: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    for repeat_index in REPEAT_INDICES:
        root = output_root / f"repeat_{repeat_index:02d}"
        path = root / "replicate_manifest.json"
        if path.is_file():
            manifest = _read_json(path)
            replicate_manifests.append(manifest)
            replicate_rows.append(
                {
                    "repeat_index": repeat_index,
                    "status": manifest.get("status"),
                    "manifest": _file_reference(path, output_root),
                    "counts": manifest.get("counts", {}),
                }
            )
        else:
            replicate_rows.append(
                {
                    "repeat_index": repeat_index,
                    "status": "missing_manifest",
                    "manifest": None,
                    "counts": {},
                }
            )
    trials = [
        trial
        for manifest in replicate_manifests
        for trial in manifest.get("trials", [])
        if isinstance(trial, dict)
    ]
    topology_complete = sum(
        row.get("artifacts", {}).get("topology_status") == "complete" for row in trials
    )
    full_roa_complete = sum(
        row.get("artifacts", {}).get("full_uniform_roa_status") == "complete" for row in trials
    )
    stats_available = sum(
        row.get("artifacts", {}).get("basin_statistics_status") == "available" for row in trials
    )
    expected = len(BASE_SEEDS) * len(REPEAT_INDICES)
    status = (
        "complete"
        if (
            len(trials) == expected
            and topology_complete == expected
            and full_roa_complete == expected
            and all(int(row.get("returncode", 1)) == 0 for row in process_results)
        )
        else "complete_with_failures"
    )
    return {
        "schema_version": AGGREGATE_MANIFEST_SCHEMA_VERSION,
        "status": status,
        "completed_at_utc": _utc_now(),
        "plan": _file_reference(plan_path, output_root),
        "ordering": "repeat index, then frozen seed-plan order; not score sorted",
        "statistics_used_for_training_selection_or_order": False,
        "process_results": list(process_results),
        "replicates": replicate_rows,
        "counts": {
            "trials_expected": expected,
            "trials_reported": len(trials),
            "topology_complete": topology_complete,
            "full_uniform_roa_complete": full_roa_complete,
            "basin_statistics_available": stats_available,
            "basin_statistics_unavailable_or_invalid": len(trials) - stats_available,
        },
        "seed_repeat_diagnostics": _seed_repeat_diagnostics(replicate_manifests),
        "trials_in_frozen_order": sorted(
            trials,
            key=lambda row: (
                int(row.get("repeat_index", -1)),
                int(row.get("plan_index", -1)),
            ),
        ),
    }


def run_all_repeats(
    *,
    output_root: Path,
    quiet: bool = False,
) -> dict[str, Any]:
    target = _assert_fresh_outer(output_root)
    _require_canonical_backend()
    inputs = study.verify_exact_inputs(REFERENCE_ROOT)
    plan = _outer_plan(inputs)
    target.mkdir(parents=True, exist_ok=False)
    plan_path = _write_json_exclusive(target / "repeat_plan.json", plan)
    _copy_reference_inputs(inputs, target / "reference_inputs")

    process_results: list[dict[str, Any]] = []
    for repeat_index in REPEAT_INDICES:
        replicate_root = target / f"repeat_{repeat_index:02d}"
        command = _replicate_command(
            repeat_index=repeat_index,
            replicate_root=replicate_root,
            quiet=quiet,
        )
        started = time.perf_counter()
        completed = subprocess.run(command, check=False, cwd=CODE_ROOT)
        process_results.append(
            {
                "repeat_index": repeat_index,
                "returncode": int(completed.returncode),
                "elapsed_seconds": time.perf_counter() - started,
                "replicate_root": str(replicate_root.relative_to(target)),
                "fresh_python_process": True,
            }
        )
    aggregate = _aggregate_manifest(
        output_root=target,
        plan_path=plan_path,
        process_results=process_results,
    )
    _write_json_exclusive(target / "aggregate_manifest.json", aggregate)
    return aggregate


def run_existing_sweeps(
    *,
    output_root: Path,
    source_sweeps: Sequence[Path],
    expected_plan_sha256: str | None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Analyze three verified existing sweeps without mutating or retraining them."""

    target = _assert_fresh_outer(output_root)
    _require_canonical_backend()
    frozen = _verify_existing_sweeps(
        source_sweeps,
        expected_plan_sha256=expected_plan_sha256,
    )
    for source in frozen:
        if (
            target == source.root
            or _is_within(target, source.root)
            or _is_within(source.root, target)
        ):
            raise ValueError(
                f"package target {target} overlaps source sweep {source.root}"
            )
    inputs = study.verify_exact_inputs(REFERENCE_ROOT)
    plan = _existing_analysis_plan(frozen_sweeps=frozen, inputs=inputs)
    target.mkdir(parents=True, exist_ok=False)
    plan_path = _write_json_exclusive(target / "repeat_plan.json", plan)
    _copy_reference_inputs(inputs, target / "reference_inputs")

    process_results: list[dict[str, Any]] = []
    for repeat_index, source in zip(REPEAT_INDICES, frozen, strict=True):
        replicate_root = target / f"repeat_{repeat_index:02d}"
        command = _replicate_command(
            repeat_index=repeat_index,
            replicate_root=replicate_root,
            quiet=quiet,
            source_sweep=source.root,
            expected_source_plan_sha256=source.plan_sha256,
        )
        started = time.perf_counter()
        completed = subprocess.run(command, check=False, cwd=CODE_ROOT)
        process_results.append(
            {
                "repeat_index": repeat_index,
                "returncode": int(completed.returncode),
                "elapsed_seconds": time.perf_counter() - started,
                "replicate_root": str(replicate_root.relative_to(target)),
                "fresh_python_process": True,
                "training_performed": False,
                "read_only_source_sweep": str(source.root),
                "source_plan_sha256": source.plan_sha256,
            }
        )
    aggregate = _aggregate_manifest(
        output_root=target,
        plan_path=plan_path,
        process_results=process_results,
    )
    aggregate["mode"] = "analyzed_verified_existing_training_sweeps"
    aggregate["training_performed"] = False
    aggregate["shared_source_plan_sha256"] = frozen[0].plan_sha256
    _write_json_exclusive(target / "aggregate_manifest.json", aggregate)
    return aggregate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help=(
            "root of the archived reference inputs "
            "(train_data.csv, traj_attractors.pkl, stable_solutions.csv)"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--source-sweep",
        type=Path,
        action="append",
        help=(
            "analyze an existing read-only training sweep; provide exactly "
            "three times in literal-repeat order"
        ),
    )
    parser.add_argument(
        "--analyze-existing-defaults",
        action="store_true",
        help=(
            "analyze the verified rerun_01, rerun_02, and rerun_03 sweeps "
            "without retraining"
        ),
    )
    parser.add_argument(
        "--expected-source-plan-sha256",
        help="optional required shared frozen source-plan payload hash",
    )
    parser.add_argument(
        "--_replicate-index",
        type=int,
        choices=REPEAT_INDICES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_replicate-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_source-sweep",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_expected-source-plan-sha256",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    global REFERENCE_ROOT
    REFERENCE_ROOT = args.reference_root.resolve()
    internal = args._replicate_index is not None or args._replicate_root is not None
    if internal:
        if args._replicate_index is None or args._replicate_root is None:
            raise ValueError("internal replicate index and root must be supplied together")
        try:
            _run_replicate(
                repeat_index=args._replicate_index,
                replicate_root=args._replicate_root,
                quiet=args.quiet,
                source_sweep=args._source_sweep,
                expected_source_plan_sha256=(
                    args._expected_source_plan_sha256
                ),
            )
        except Exception:
            traceback.print_exc()
            return 1
        return 0

    if args.analyze_existing_defaults and args.source_sweep:
        raise ValueError(
            "--analyze-existing-defaults and --source-sweep are mutually exclusive"
        )
    sources = (
        DEFAULT_EXISTING_SOURCES
        if args.analyze_existing_defaults
        else tuple(args.source_sweep or ())
    )
    if sources:
        expected_hash = (
            args.expected_source_plan_sha256
            or (
                EXPECTED_EXISTING_PLAN_SHA256
                if args.analyze_existing_defaults
                else None
            )
        )
        output = args.output_dir or DEFAULT_EXISTING_OUTPUT
        aggregate = run_existing_sweeps(
            output_root=output,
            source_sweeps=sources,
            expected_plan_sha256=expected_hash,
            quiet=args.quiet,
        )
    else:
        if args.expected_source_plan_sha256 is not None:
            raise ValueError(
                "--expected-source-plan-sha256 requires existing-sweep mode"
            )
        output = args.output_dir or DEFAULT_OUTPUT
        aggregate = run_all_repeats(
            output_root=output,
            quiet=args.quiet,
        )
    print(
        f"repeat_sweep_status={aggregate['status']} "
        f"topology_complete={aggregate['counts']['topology_complete']}/"
        f"{aggregate['counts']['trials_expected']} "
        f"full_roa_complete={aggregate['counts']['full_uniform_roa_complete']}/"
        f"{aggregate['counts']['trials_expected']} "
        f"output={output.resolve()}",
        flush=True,
    )
    return 0 if aggregate["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
