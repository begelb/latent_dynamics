"""Run one matched D3 uniform-s24 on-demand CMGDB + strict-RoA analysis.

This is the single-process production worker used by the matched five-dataset
by three-initialization study.  The cached MapGraph is intentionally not
serializable, so the strict singleton-reachability query is completed and
persisted before the process exits.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import CMGDB
import numpy as np
import torch

import chafee_latent_dimension_study as study
import run_chafee_d3_ondemand_s24 as base
from latentdynamics.analysis.basin_statistics import (
    compute_chafee_basin_statistics,
)
from latentdynamics.analysis.morse import LatentBounds
from latentdynamics.training import load_checkpoint
from latentdynamics.viz import save_morse_graph_artifacts

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
TRAINING_ROOT = (
    CODE_ROOT / "output" / "chafee_d3_matched_d2_archive_5x3_training_v1"
)
DEFAULT_OUTPUT_ROOT = (
    CODE_ROOT / "output" / "chafee_d3_matched_d2_archive_5x3_ondemand_v2"
)
TRAJECTORIES = PROJECT_ROOT / "archive" / "marcio" / "scripts" / "traj_attractors.pkl"
STABLE_ROOTS = PROJECT_ROOT / "archive" / "marcio" / "scripts" / "stable_solutions.csv"
TRAJECTORIES_SHA256 = (
    "f163b7427e50a4e4d08ab54c87cb5bd16592768edfe8432f019842416afbb145"
)
STABLE_ROOTS_SHA256 = (
    "cae0222acb37ae9688e54cb2a1f42ac3777360e49b3919403ec1433363de0586"
)
RUN_ID = re.compile(r"^dataset_(\d{2})_seed_(\d{2})_lr3e3_e4000$")
EXPECTED_RUNS = 15
MAX_CONCURRENCY = 2


class ScientificInvalidError(RuntimeError):
    """Terminal topology/root-association outcome, not a retryable run failure."""


def _assert_safe_output_root(output_root: Path, inputs: dict[str, Any]) -> Path:
    raw = output_root.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError("analysis output root must not be a symlink")
    output = raw.resolve()
    broad_roots = (CODE_ROOT.resolve(), (CODE_ROOT / "output").resolve())
    if output in broad_roots:
        raise ValueError("analysis output must be a dedicated subdirectory")
    protected = (
        TRAINING_ROOT.resolve(),
        inputs["run_root"],
        inputs["attempt_root"],
        base.LEGACY_D3_ROOT.resolve(),
    )
    for path in protected:
        if base._paths_overlap(output, path):
            raise ValueError(
                f"analysis output {output} overlaps protected input {path}"
            )
    return output


def _allocate_attempt(
    output_root: Path,
    inputs: dict[str, Any],
) -> tuple[Path, Path, int]:
    safe_root = _assert_safe_output_root(output_root, inputs)
    run_root = safe_root / "runs" / inputs["run_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    for marker_name in ("completed.json", "completed_invalid.json"):
        marker = run_root / marker_name
        if marker.exists():
            raise FileExistsError(
                f"{inputs['run_id']} already has terminal marker {marker}"
            )
    attempts_root = run_root / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(match.group(1))
        for path in attempts_root.iterdir()
        if path.is_dir()
        and (match := re.fullmatch(r"attempt_(\d{3})", path.name)) is not None
    ]
    attempt_number = max(existing, default=0) + 1
    while True:
        attempt = attempts_root / f"attempt_{attempt_number:03d}"
        try:
            attempt.mkdir(exist_ok=False)
            return run_root, attempt, attempt_number
        except FileExistsError:
            attempt_number += 1


def _ensure_analysis_plan(
    analysis_root: Path,
    *,
    inputs: dict[str, Any],
    runtime: dict[str, Any],
    device: str,
    max_edges: int,
    max_forward_points: int,
    rss_sample_seconds: float,
) -> tuple[Path, str]:
    common_source_names = (
        "training_plan",
        "trajectories",
        "stable_roots",
        "worker",
        "ondemand_backend",
        "study_helpers",
        "basin_statistics_implementation",
        "morse_implementation",
    )
    plan = {
        "schema_version": 1,
        "purpose": "matched D3 5x3 uniform-s24 on-demand CMGDB plus strict RoA",
        "training_plan_sha256": inputs["plan_sha256"],
        "matrix": {
            "datasets": [1, 2, 3, 4, 5],
            "training_seeds": [0, 1, 2],
            "required_runs": EXPECTED_RUNS,
        },
        "computation": {
            "backend": "batched_on_demand_neural",
            "precomputed": False,
            "subdivisions": [24, 24, 24],
            "padding": True,
            "expected_cells": base.EXPECTED_CELLS,
            "expected_callback_rectangles": base.EXPECTED_CALLBACK_RECTANGLES,
            "expected_neural_corner_points": base.EXPECTED_NEURAL_CORNER_POINTS,
            "max_edges": max_edges,
            "max_forward_points": max_forward_points,
            "device": device,
            "trajectory_and_root_encoding_device": "cpu",
            "rss_sample_seconds": rss_sample_seconds,
        },
        "strict_roa": {
            "semantics": "singleton-all-reachable-Morse-set",
            "one_combined_unique_query": True,
            "negative_root_first": True,
            "conditioned_trajectories": 7_862,
        },
        "concurrency": {
            "maximum_processes": MAX_CONCURRENCY,
            "shared_mps_device": True,
            "timing_caveat": (
                "Concurrent wall times are operational throughput measurements "
                "and are not directly comparable to the isolated s24 benchmark."
            ),
        },
        "runtime": runtime,
        "common_sources": {
            name: inputs["sources"][name] for name in common_source_names
        },
    }
    plan_sha256 = base._payload_sha256(plan)
    envelope = {
        "schema_version": 1,
        "created_at_utc": base._utc_now(),
        "plan_sha256": plan_sha256,
        "plan": plan,
    }
    plan_path = analysis_root / "analysis_plan.json"
    temporary = analysis_root / (
        f".analysis_plan.{os.getpid()}.{time.time_ns()}.tmp"
    )
    base._write_json_exclusive(temporary, envelope)
    try:
        os.link(temporary, plan_path)
    except FileExistsError as publish_error:
        existing = base._read_json(plan_path)
        if (
            existing.get("plan_sha256") != plan_sha256
            or existing.get("plan") != plan
        ):
            raise ValueError(
                f"analysis configuration differs from frozen plan {plan_path}"
            ) from publish_error
    finally:
        temporary.unlink(missing_ok=True)
    return plan_path, plan_sha256


def _resolve_inputs(run_id: str) -> dict[str, Any]:
    match = RUN_ID.fullmatch(run_id)
    if match is None:
        raise ValueError(f"invalid matched D3 run id: {run_id!r}")
    dataset = int(match.group(1))
    training_seed = int(match.group(2))
    if dataset not in range(1, 6) or training_seed not in range(3):
        raise ValueError(f"run id is outside the fixed 5x3 matrix: {run_id}")

    plan_path = TRAINING_ROOT / "experiment_plan.json"
    envelope = base._read_json(plan_path)
    plan = envelope.get("plan")
    plan_sha256 = str(envelope.get("plan_sha256", ""))
    if not isinstance(plan, dict) or base._payload_sha256(plan) != plan_sha256:
        raise ValueError("matched D3 training plan hash is invalid")
    matrix = base._matrix_completion_status(
        TRAINING_ROOT,
        envelope,
        plan_sha256=plan_sha256,
    )
    if not matrix["complete"] or int(matrix["completed_runs"]) != EXPECTED_RUNS:
        raise RuntimeError(
            "all 15 matched D3 trainings must complete before analysis; "
            f"missing {matrix['incomplete_run_ids']}"
        )

    records = [
        row
        for row in plan.get("trials", [])
        if isinstance(row, dict) and row.get("run_id") == run_id
    ]
    if len(records) != 1:
        raise ValueError(f"training plan does not identify {run_id} exactly once")
    trial = records[0]
    if (
        int(trial.get("dataset", -1)) != dataset
        or int(trial.get("training_seed", -1)) != training_seed
    ):
        raise ValueError("run id and frozen trial metadata disagree")

    run_root = TRAINING_ROOT / "runs" / run_id
    run_spec_path = run_root / "run_spec.json"
    run_spec = base._read_json(run_spec_path)
    if (
        run_spec.get("plan_sha256") != plan_sha256
        or run_spec.get("run") != trial.get("training_spec")
    ):
        raise ValueError(f"{run_id} run specification differs from the frozen plan")

    completed_path = run_root / "completed.json"
    completed = base._read_json(completed_path)
    if (
        completed.get("status") != "completed"
        or completed.get("plan_sha256") != plan_sha256
        or completed.get("run") != run_spec.get("run")
    ):
        raise ValueError(f"invalid completion marker for {run_id}")
    attempt_number = int(completed.get("attempt", 0))
    attempt_root = (
        run_root / "attempts" / f"attempt_{attempt_number:03d}"
    ).resolve()
    if run_root.resolve() not in attempt_root.parents:
        raise ValueError(f"{run_id} selected training attempt escapes its run root")
    checkpoint_entry = completed.get("checkpoint")
    if not isinstance(checkpoint_entry, dict):
        raise ValueError(f"{run_id} completion marker has no checkpoint")
    checkpoint = base._safe_relative_file(
        run_root,
        str(checkpoint_entry.get("path")),
    )
    expected_checkpoint = attempt_root / "models" / "autoencoder.pt"
    if checkpoint != expected_checkpoint:
        raise ValueError(
            f"{run_id} completion marker selected {checkpoint}, "
            f"expected {expected_checkpoint}"
        )
    checkpoint_record = base._file_record(
        checkpoint,
        expected_sha256=str(checkpoint_entry.get("sha256")),
    )
    if checkpoint_record["size_bytes"] != int(checkpoint_entry.get("size_bytes", -1)):
        raise ValueError(f"{run_id} checkpoint size differs from completion marker")

    checkpoint_sidecar = checkpoint.with_suffix(".json")
    sidecar = base._read_json(checkpoint_sidecar)
    sidecar_arch = sidecar.get("arch")
    if (
        sidecar.get("version") != 1
        or not isinstance(sidecar_arch, dict)
        or sidecar_arch != plan.get("architecture")
        or int(sidecar_arch.get("low_dims", -1)) != 3
        or int(sidecar_arch.get("high_dims", -1)) != 64
    ):
        raise ValueError(f"{run_id} checkpoint architecture is not frozen D3")

    training_summary_path = attempt_root / "training_summary.json"
    training_summary = base._read_json(training_summary_path)
    if (
        training_summary.get("arch") != sidecar_arch
        or int(training_summary.get("seed", -1)) != training_seed
        or int(training_summary.get("epochs_completed", -1)) != 4_000
    ):
        raise ValueError(f"{run_id} training summary differs from its checkpoint")

    artifact_entry = completed.get("artifact_manifest")
    if not isinstance(artifact_entry, dict):
        raise ValueError(f"{run_id} completion marker has no artifact manifest")
    artifact_manifest_path = base._safe_relative_file(
        run_root,
        str(artifact_entry.get("path")),
    )
    if attempt_root not in artifact_manifest_path.parents:
        raise ValueError(f"{run_id} artifact manifest is outside its attempt")
    artifact_manifest = base._read_json(artifact_manifest_path)
    if (
        artifact_manifest.get("plan_sha256") != plan_sha256
        or artifact_manifest.get("matched_d3_trial") != trial
        or artifact_manifest.get("architecture") != sidecar_arch
    ):
        raise ValueError(f"{run_id} artifact manifest differs from frozen inputs")

    data_record = plan.get("sources", {}).get(f"train_data_dataset_{dataset}")
    if not isinstance(data_record, dict):
        raise ValueError(f"training plan has no source for dataset {dataset}")
    train_data = Path(str(data_record.get("path"))).resolve()
    sources = {
        "training_plan": base._file_record(plan_path),
        "run_spec": base._file_record(run_spec_path),
        "completion_marker": base._file_record(completed_path),
        "artifact_manifest": base._file_record(
            artifact_manifest_path,
            expected_sha256=str(artifact_entry.get("sha256")),
        ),
        "training_summary": base._file_record(training_summary_path),
        "checkpoint": checkpoint_record,
        "checkpoint_sidecar": base._file_record(checkpoint_sidecar),
        "train_data": base._file_record(
            train_data,
            expected_sha256=str(data_record.get("sha256")),
        ),
        "trajectories": base._file_record(
            TRAJECTORIES,
            expected_sha256=TRAJECTORIES_SHA256,
        ),
        "stable_roots": base._file_record(
            STABLE_ROOTS,
            expected_sha256=STABLE_ROOTS_SHA256,
        ),
        "worker": base._file_record(SCRIPT_PATH),
        "ondemand_backend": base._file_record(Path(base.__file__)),
        "study_helpers": base._file_record(Path(study.__file__)),
        "basin_statistics_implementation": base._file_record(
            CODE_ROOT / "src" / "latentdynamics" / "analysis" / "basin_statistics.py"
        ),
        "morse_implementation": base._file_record(
            CODE_ROOT / "src" / "latentdynamics" / "analysis" / "morse.py"
        ),
    }
    return {
        "run_id": run_id,
        "dataset": dataset,
        "training_seed": training_seed,
        "trial": trial,
        "plan_sha256": plan_sha256,
        "matrix": matrix,
        "run_root": run_root.resolve(),
        "attempt_root": attempt_root,
        "checkpoint": checkpoint,
        "train_data": train_data,
        "sources": sources,
    }


def _classification_payload(
    *,
    truth: np.ndarray,
    predicted: np.ndarray,
    negative_attractor: int,
    positive_attractor: int,
) -> dict[str, Any]:
    statistics = compute_chafee_basin_statistics(
        truth,
        predicted,
        negative_basin_label=negative_attractor,
        positive_basin_label=positive_attractor,
    )
    percentages = statistics.percentages()
    if statistics.conditioned_trajectories != 7_862:
        raise ValueError("strict statistics no longer condition on 7,862 trajectories")
    if not np.isclose(sum(percentages.values()), 100.0, rtol=0.0, atol=1e-12):
        raise ValueError("strict-stat percentages do not sum to 100")
    return {
        "total_trajectories": statistics.total_trajectories,
        "excluded_zero_trajectories": statistics.excluded_zero_trajectories,
        "conditioned_trajectories": statistics.conditioned_trajectories,
        "counts": statistics.counts(),
        "percentages": percentages,
    }


def run_worker(
    *,
    run_id: str,
    output_root: Path,
    device_name: str,
    max_edges: int,
    max_forward_points: int,
    rss_sample_seconds: float,
) -> dict[str, Any]:
    inputs = _resolve_inputs(run_id)
    device = base._resolve_device(device_name)
    runtime = base._runtime_provenance(device)
    safe_output_root = _assert_safe_output_root(output_root, inputs)
    analysis_plan_path, analysis_plan_sha256 = _ensure_analysis_plan(
        safe_output_root,
        inputs=inputs,
        runtime=runtime,
        device=str(device),
        max_edges=max_edges,
        max_forward_points=max_forward_points,
        rss_sample_seconds=rss_sample_seconds,
    )
    analysis_run_root, output, attempt_number = _allocate_attempt(
        safe_output_root,
        inputs,
    )

    started_wall = time.perf_counter()
    started_cpu = base._usage_snapshot()
    status_path = output / "status.json"
    launch = {
        "schema_version": 1,
        "created_at_utc": base._utc_now(),
        "purpose": "matched D3 uniform-s24 on-demand graph plus strict RoA query",
        "run_id": run_id,
        "dataset": inputs["dataset"],
        "training_seed": inputs["training_seed"],
        "analysis_attempt": attempt_number,
        "analysis_run_root": str(analysis_run_root),
        "analysis_attempt_root": str(output),
        "analysis_plan_sha256": analysis_plan_sha256,
        "training_plan_sha256": inputs["plan_sha256"],
        "backend": "batched_on_demand_neural",
        "precomputed": False,
        "subdivisions": [24, 24, 24],
        "padding": True,
        "device": device_name,
        "trajectory_and_root_encoding_device": "cpu",
        "max_edges": max_edges,
        "max_forward_points": max_forward_points,
        "runtime": runtime,
        "analysis_plan": base._file_record(analysis_plan_path),
        "sources": inputs["sources"],
    }
    launch_path = base._write_json_exclusive(output / "launch_manifest.json", launch)
    base._write_json_atomic(
        analysis_run_root / "latest_attempt.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "attempt": attempt_number,
            "status": "running",
            "attempt_root": str(output),
            "launch_manifest": base._file_record(launch_path),
        },
    )
    base._write_json_exclusive(
        status_path,
        {
            "schema_version": 1,
            "status": "running",
            "phase": "bounds",
            "started_at_utc": base._utc_now(),
            "launch_manifest": base._file_record(launch_path),
        },
    )

    sampler = base.PeakRSSSampler(rss_sample_seconds)
    sampler.start()
    terminal_artifacts = {
        "launch_manifest": base._file_record(launch_path),
        "analysis_plan": base._file_record(analysis_plan_path),
    }
    scientific_context: dict[str, Any] = {}
    try:
        model, arch = load_checkpoint(inputs["checkpoint"].parent, map_location="cpu")
        if int(arch.low_dims) != 3 or int(arch.high_dims) != 64:
            raise ValueError("checkpoint is not the frozen 64-to-3 architecture")
        model.eval()
        x, y = base._load_training_pairs(inputs["train_data"])
        bounds_started = time.perf_counter()
        lower, upper = base.compute_cpu_bounds(model.encoder, x, y)
        bounds_seconds = time.perf_counter() - bounds_started
        bounds_payload = {
            "schema_version": 1,
            "dimension": 3,
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "epsilon_frac": 0.1,
            "source": "encoder(concatenate(current,next))",
            "n_encoded_states": 60_000,
            "encoder_device": "cpu",
            "checkpoint_sha256": inputs["sources"]["checkpoint"]["sha256"],
            "train_data_sha256": inputs["sources"]["train_data"]["sha256"],
        }
        bounds_path = base._write_json_exclusive(output / "bounds.json", bounds_payload)
        terminal_artifacts["bounds"] = base._file_record(bounds_path)
        bounds = LatentBounds(lower=lower, upper=upper)

        trajectory_points, truth = study._load_trajectory_labels(TRAJECTORIES)
        roots = study._load_stable_roots(STABLE_ROOTS)
        cpu = torch.device("cpu")
        encoded_points = study._encode_numpy(model.encoder, trajectory_points, device=cpu)
        encoded_roots = study._encode_numpy(model.encoder, roots, device=cpu)
        point_cells = study._uniform_point_cells(
            encoded_points,
            bounds,
            study.RESOLUTIONS[3],
        )
        root_cells = study._uniform_point_cells(
            encoded_roots,
            bounds,
            study.RESOLUTIONS[3],
        )
        all_candidate_ids = np.concatenate(
            (point_cells.flat_cell_ids, root_cells.flat_cell_ids)
        )
        unique_cell_ids, inverse = np.unique(all_candidate_ids, return_inverse=True)

        evaluator = base.NeuralEvaluator(model.latent_map, device)
        evaluator(
            np.stack(
                (
                    lower,
                    upper,
                    0.5 * (lower + upper),
                    lower + 0.25 * (upper - lower),
                ),
                axis=0,
            )
        )
        evaluator.reset_counters()
        box_map = base.OnDemandNeuralBoxMap(
            evaluator,
            max_forward_points=max_forward_points,
            padding=True,
        )
        os.environ["CMGDB_MAPGRAPH_MAX_VERTICES"] = str(base.EXPECTED_CELLS)
        os.environ["CMGDB_MAPGRAPH_MAX_EDGES"] = str(max_edges)
        os.environ["CMGDB_MAPGRAPH_RESERVE_EDGES"] = str(max_edges)
        os.environ["CMGDB_MAPGRAPH_RESERVE_MIN_VERTICES"] = str(base.EXPECTED_CELLS)

        base._write_json_atomic(
            status_path,
            {
                "schema_version": 1,
                "status": "running",
                "phase": "CMGDB.ComputeMorseGraph",
                "run_id": run_id,
                "device": str(device),
                "bounds": base._file_record(bounds_path),
            },
        )
        print(f"{run_id}: launching exact on-demand s24 on {device}", flush=True)
        cmgdb_model = CMGDB.Model(
            24,
            24,
            24,
            10_000,
            lower.tolist(),
            upper.tolist(),
            box_map,
        )
        if not hasattr(cmgdb_model, "set_batch_map"):
            raise RuntimeError("CMGDB.Model.set_batch_map is required")
        cmgdb_model.set_batch_map(box_map.batch)
        graph_started = time.perf_counter()
        morse_graph, map_graph = CMGDB.ComputeMorseGraph(cmgdb_model)
        graph_seconds = time.perf_counter() - graph_started

        if (
            int(map_graph.num_vertices()) != base.EXPECTED_CELLS
            or not bool(map_graph.has_cache())
        ):
            raise RuntimeError("s24 MapGraph is incomplete or lacks its native cache")
        cached_edges = int(map_graph.num_cached_edges())
        callback = box_map.stats()
        if (
            int(callback["scalar_calls"]) != 0
            or int(callback["rectangles"]) != base.EXPECTED_CALLBACK_RECTANGLES
            or int(callback["neural_corner_points"])
            != base.EXPECTED_NEURAL_CORNER_POINTS
        ):
            raise RuntimeError(f"on-demand callback completeness failure: {callback}")

        graph_dir = output / "MG_uniform_s24"
        artifact_started = time.perf_counter()
        dot_path, morse_sets_path = save_morse_graph_artifacts(
            morse_graph,
            graph_dir,
        )
        artifact_seconds = time.perf_counter() - artifact_started
        terminal_artifacts.update(
            {
                "morse_graph": base._file_record(dot_path),
                "morse_sets": base._file_record(morse_sets_path),
            }
        )
        scientific_context.update(
            {
                "cmgdb_seconds": graph_seconds,
                "morse_artifact_seconds": artifact_seconds,
                "cached_edges": cached_edges,
                "morse_nodes": int(morse_graph.num_vertices()),
            }
        )
        try:
            attractors = study._require_exactly_two_minimal_attractors(morse_graph)
        except ValueError as error:
            raise ScientificInvalidError(str(error)) from error
        scientific_context["attractor_nodes"] = attractors

        base._write_json_atomic(
            status_path,
            {
                "schema_version": 1,
                "status": "running",
                "phase": "strict_singleton_reachability",
                "run_id": run_id,
                "cmgdb_seconds": graph_seconds,
                "cached_edges": cached_edges,
                "attractors": attractors,
            },
        )
        query_started = time.perf_counter()
        singleton_by_unique = study._native_singleton_reachability(
            map_graph,
            morse_graph,
            unique_cell_ids,
        )
        query_seconds = time.perf_counter() - query_started
        singleton_by_candidate = singleton_by_unique[inverse]
        split = point_cells.flat_cell_ids.size
        point_singletons = np.asarray(singleton_by_candidate[:split], dtype=np.int32)
        root_singletons = np.asarray(singleton_by_candidate[split:], dtype=np.int32)
        raw_query_path = graph_dir / "singleton_reachability_raw.npz"
        np.savez_compressed(
            raw_query_path,
            queried_cell_ids=unique_cell_ids,
            singleton_node_by_queried_cell=singleton_by_unique,
            point_candidate_cell_ids=point_cells.flat_cell_ids,
            point_candidate_offsets=point_cells.offsets,
            point_singleton_nodes=point_singletons,
            root_candidate_cell_ids=root_cells.flat_cell_ids,
            root_candidate_offsets=root_cells.offsets,
            root_singleton_nodes=root_singletons,
            encoded_stable_roots=encoded_roots,
        )
        terminal_artifacts["raw_singleton_query"] = base._file_record(raw_query_path)
        scientific_context.update(
            {
                "strict_query_seconds": query_seconds,
                "queried_uniform_cells": int(unique_cell_ids.size),
            }
        )
        try:
            negative_attractor = study._root_attractor_label(
                root_singletons,
                root_cells,
                0,
                attractors,
            )
            positive_attractor = study._root_attractor_label(
                root_singletons,
                root_cells,
                1,
                attractors,
            )
        except ValueError as error:
            raise ScientificInvalidError(str(error)) from error
        if negative_attractor == positive_attractor:
            raise ScientificInvalidError(
                "encoded stable roots map to the same attractor"
            )
        predicted = study._point_basin_labels(
            point_singletons,
            point_cells,
            negative_attractor=negative_attractor,
            positive_attractor=positive_attractor,
        )
        statistics_payload = _classification_payload(
            truth=truth,
            predicted=predicted,
            negative_attractor=negative_attractor,
            positive_attractor=positive_attractor,
        )

        query_path = graph_dir / "marcio_singleton_reachability_queries.npz"
        np.savez_compressed(
            query_path,
            queried_cell_ids=unique_cell_ids,
            singleton_node_by_queried_cell=singleton_by_unique,
            point_candidate_cell_ids=point_cells.flat_cell_ids,
            point_candidate_offsets=point_cells.offsets,
            point_singleton_nodes=point_singletons,
            point_basin_labels=predicted,
            root_candidate_cell_ids=root_cells.flat_cell_ids,
            root_candidate_offsets=root_cells.offsets,
            root_singleton_nodes=root_singletons,
            encoded_stable_roots=encoded_roots,
        )
        trajectory_labels_path = output / "trajectory_basin_labels.npy"
        roots_path = output / "encoded_stable_roots.npy"
        np.save(trajectory_labels_path, predicted)
        np.save(roots_path, encoded_roots)
        basin_payload = {
            "schema_version": 2,
            "run_id": run_id,
            "dataset": inputs["dataset"],
            "training_seed": inputs["training_seed"],
            "method": (
                "Exact Marcio singleton-all-reachable-Morse-set basin semantics "
                "on uniform CMGDB graph"
            ),
            "stable_roots": {
                "encoded": encoded_roots.tolist(),
                "negative_basin_label": negative_attractor,
                "positive_basin_label": positive_attractor,
                "candidate_cell_ids": [
                    root_cells.candidates(index).tolist() for index in range(2)
                ],
            },
            "trajectory_data": {
                "total": 10_000,
                "sha256": TRAJECTORIES_SHA256,
            },
            "classification": {
                "rule": "complete reachable Morse-node set equals one singleton",
                "counts_by_singleton_label": {
                    str(label): int(count)
                    for label, count in sorted(
                        Counter(predicted.tolist()).items()
                    )
                },
                "queried_uniform_cells": int(unique_cell_ids.size),
            },
            "cmgdb": {
                "subdivisions": [24, 24, 24],
                "uniform_cells": base.EXPECTED_CELLS,
                "cached_edges": cached_edges,
                "morse_nodes": int(morse_graph.num_vertices()),
                "attractor_nodes": attractors,
            },
            "statistics": statistics_payload,
        }
        basin_path = base._write_json_exclusive(
            output / "basin_statistics.json",
            basin_payload,
        )

        graph_summary = base._graph_summary(morse_graph, map_graph)
        sampler.stop()
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at_utc": base._utc_now(),
            "run_id": run_id,
            "dataset": inputs["dataset"],
            "training_seed": inputs["training_seed"],
            "analysis_attempt": attempt_number,
            "analysis_plan_sha256": analysis_plan_sha256,
            "backend": "batched_on_demand_neural",
            "precomputed": False,
            "subdivisions": [24, 24, 24],
            "timings": {
                "bounds_seconds": bounds_seconds,
                "cmgdb_seconds": graph_seconds,
                "morse_artifact_seconds": artifact_seconds,
                "strict_query_seconds": query_seconds,
                "total_seconds": time.perf_counter() - started_wall,
                "total_cpu": base._usage_delta(
                    started_cpu,
                    base._usage_snapshot(),
                ),
            },
            "memory": sampler.stats(),
            "callback": callback,
            "graph": graph_summary,
            "statistics": statistics_payload,
            "artifacts": {
                "launch_manifest": base._file_record(launch_path),
                "bounds": base._file_record(bounds_path),
                "morse_graph": base._file_record(dot_path),
                "morse_sets": base._file_record(morse_sets_path),
                "raw_singleton_query": base._file_record(raw_query_path),
                "query": base._file_record(query_path),
                "trajectory_basin_labels": base._file_record(
                    trajectory_labels_path
                ),
                "encoded_stable_roots": base._file_record(roots_path),
                "basin_statistics": base._file_record(basin_path),
            },
            "runtime": runtime,
            "trajectory_and_root_encoding_device": "cpu",
            "sources": inputs["sources"],
        }
        summary_path = base._write_json_exclusive(output / "summary.json", summary)
        base._write_json_atomic(
            status_path,
            {
                "schema_version": 1,
                "status": "complete",
                "completed_at_utc": base._utc_now(),
                "summary": base._file_record(summary_path),
            },
        )
        completion = {
            "schema_version": 1,
            "status": "completed",
            "completed_at_utc": base._utc_now(),
            "run_id": run_id,
            "dataset": inputs["dataset"],
            "training_seed": inputs["training_seed"],
            "attempt": attempt_number,
            "analysis_plan_sha256": analysis_plan_sha256,
            "summary": base._file_record(summary_path),
        }
        completion_path = base._write_json_exclusive(
            analysis_run_root / "completed.json",
            completion,
        )
        base._write_json_atomic(
            analysis_run_root / "latest_attempt.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "attempt": attempt_number,
                "status": "completed",
                "attempt_root": str(output),
                "terminal_marker": base._file_record(completion_path),
            },
        )
        print(
            f"{run_id}: complete in {graph_seconds:.3f}s; "
            f"strict correct="
            f"{statistics_payload['percentages']['correctly_classified_in_negative_basin'] + statistics_payload['percentages']['correctly_classified_in_positive_basin']:.3f}%",
            flush=True,
        )
        return summary
    except ScientificInvalidError as error:
        sampler.stop()
        invalid = {
            "schema_version": 1,
            "status": "completed_invalid",
            "completed_at_utc": base._utc_now(),
            "run_id": run_id,
            "dataset": inputs["dataset"],
            "training_seed": inputs["training_seed"],
            "attempt": attempt_number,
            "analysis_plan_sha256": analysis_plan_sha256,
            "reason_type": type(error).__name__,
            "reason": str(error),
            "scientific_context": scientific_context,
            "elapsed_seconds": time.perf_counter() - started_wall,
            "memory": sampler.stats(),
            "artifacts": terminal_artifacts,
            "runtime": runtime,
            "sources": inputs["sources"],
        }
        invalid_path = base._write_json_exclusive(
            output / "completed_invalid.json",
            invalid,
        )
        base._write_json_atomic(
            status_path,
            {
                "schema_version": 1,
                "status": "completed_invalid",
                "completed_at_utc": base._utc_now(),
                "result": base._file_record(invalid_path),
            },
        )
        terminal_path = base._write_json_exclusive(
            analysis_run_root / "completed_invalid.json",
            {
                "schema_version": 1,
                "status": "completed_invalid",
                "completed_at_utc": base._utc_now(),
                "run_id": run_id,
                "dataset": inputs["dataset"],
                "training_seed": inputs["training_seed"],
                "attempt": attempt_number,
                "analysis_plan_sha256": analysis_plan_sha256,
                "result": base._file_record(invalid_path),
            },
        )
        base._write_json_atomic(
            analysis_run_root / "latest_attempt.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "attempt": attempt_number,
                "status": "completed_invalid",
                "attempt_root": str(output),
                "terminal_marker": base._file_record(terminal_path),
            },
        )
        print(f"{run_id}: completed_invalid: {error}", flush=True)
        return invalid
    except BaseException as error:
        sampler.stop()
        failure = {
            "schema_version": 1,
            "status": "failed",
            "failed_at_utc": base._utc_now(),
            "run_id": run_id,
            "dataset": inputs["dataset"],
            "training_seed": inputs["training_seed"],
            "attempt": attempt_number,
            "analysis_plan_sha256": analysis_plan_sha256,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started_wall,
            "memory": sampler.stats(),
            "runtime": runtime,
            "sources": inputs["sources"],
        }
        failure_path = output / "failure.json"
        base._write_json_atomic(failure_path, failure)
        base._write_json_atomic(
            status_path,
            {
                "schema_version": 1,
                "status": "failed",
                "failure": base._file_record(failure_path),
            },
        )
        base._write_json_atomic(
            analysis_run_root / "last_failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at_utc": base._utc_now(),
                "run_id": run_id,
                "attempt": attempt_number,
                "analysis_plan_sha256": analysis_plan_sha256,
                "failure": base._file_record(failure_path),
            },
        )
        base._write_json_atomic(
            analysis_run_root / "latest_attempt.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "attempt": attempt_number,
                "status": "failed",
                "attempt_root": str(output),
                "failure": base._file_record(failure_path),
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--max-edges", type=int, default=1_200_000_000)
    parser.add_argument("--max-forward-points", type=int, default=800_000)
    parser.add_argument("--rss-sample-seconds", type=float, default=0.1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_worker(
            run_id=args.run_id,
            output_root=args.output_root,
            device_name=args.device,
            max_edges=args.max_edges,
            max_forward_points=args.max_forward_points,
            rss_sample_seconds=args.rss_sample_seconds,
        )
        return 0
    except Exception as error:
        traceback.print_exc()
        print(
            f"{args.run_id} failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
