#!/usr/bin/env python3
"""Dense sampled residual/tolerance audit for the max-30 Leslie3D sinks.

This is a finite numerical search.  It deliberately reports an unsquared
Euclidean semiconjugacy residual in the stored latent coordinates and assigns
zero tolerance clearance whenever a sampled image leaves the candidate block.

Run from ``code/`` with Shapely available, for example::

    uv run --with shapely python \
      scripts/compute_leslie3d_minimal_node_residual_tolerance.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import shapely
import sklearn
import torch
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from shapely import box as shapely_box
from shapely import contains_xy, coverage_union_all, covers, points

from latentdynamics.config import load_config
from latentdynamics.systems import build_system
from latentdynamics.training import load_any_checkpoint

CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
RUN_ROOT = (
    CODE_ROOT
    / "output"
    / "notebooks"
    / "leslie3d_invariant_aware_v2_smooth_max30"
    / "seed_20260809"
)
CONFIG_PATH = (
    CODE_ROOT
    / "src"
    / "latentdynamics"
    / "configs"
    / "leslie3d_invariant_aware_v2_smooth_max30.yaml"
)
MODEL_PATH = RUN_ROOT / "models" / "autoencoder.pt"
MODEL_METADATA_PATH = RUN_ROOT / "models" / "autoencoder.json"
MORSE_SETS_PATH = RUN_ROOT / "MG" / "morse_sets"
MORSE_GRAPH_PATH = RUN_ROOT / "MG" / "morse_graph"
RUN_MANIFEST_PATH = RUN_ROOT / "run_manifest.json"
SCALER_PATH = CODE_ROOT / "replay_sources" / "leslie3d_example2" / "data" / "scalers" / "scaler"
PAIR_PATHS = (
    CODE_ROOT / "data" / "leslie3d_invariant_aware_v2" / "train.csv",
    CODE_ROOT / "data" / "leslie3d_invariant_aware_v2" / "val.csv",
)
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "output" / "leslie3d_morse_report" / "analysis"
PHYSICAL_LOWER = np.array([0.0, 0.0, 0.0], dtype=np.float64)
PHYSICAL_UPPER = np.array([110.0, 77.0, 54.0], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def minimal_labels(dot_path: Path) -> list[int]:
    text = dot_path.read_text(encoding="utf-8")
    nodes = {int(value) for value in re.findall(r"^(\d+)\s+\[label=", text, re.M)}
    has_outgoing = {int(value) for value in re.findall(r"^(\d+)\s*->", text, re.M)}
    return sorted(nodes - has_outgoing)


class BlockGeometry:
    """Robust union geometry for a two-dimensional cubical block."""

    def __init__(self, boxes: np.ndarray) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64)
        if self.boxes.ndim != 2 or self.boxes.shape[1] != 4:
            raise ValueError(f"expected 2D boxes with four columns, got {self.boxes.shape}")
        rectangles = shapely_box(
            self.boxes[:, 0],
            self.boxes[:, 1],
            self.boxes[:, 2],
            self.boxes[:, 3],
        )
        try:
            geometry = coverage_union_all(rectangles)
        except shapely.errors.GEOSException:
            geometry = shapely.unary_union(rectangles)
        if not geometry.is_valid:
            geometry = shapely.make_valid(geometry)
        self.geometry = geometry
        self.boundary = shapely.boundary(geometry)
        self.component_count = len(geometry.geoms) if hasattr(geometry, "geoms") else 1

    def membership(self, values: np.ndarray, *, interior: bool = False) -> np.ndarray:
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if interior:
            return np.asarray(contains_xy(self.geometry, values[:, 0], values[:, 1]))
        return np.asarray(covers(self.geometry, points(values)))

    def clearance(self, values: np.ndarray) -> np.ndarray:
        """Distance to the relative complement; zero outside the interior."""
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        result = np.zeros(values.shape[0], dtype=np.float64)
        interior = self.membership(values, interior=True)
        if np.any(interior):
            result[interior] = shapely.distance(points(values[interior]), self.boundary)
        return result


@torch.no_grad()
def apply_module(
    module: torch.nn.Module,
    values: np.ndarray,
    *,
    chunk_size: int = 65_536,
) -> np.ndarray:
    values = np.asarray(values)
    outputs: list[np.ndarray] = []
    for start in range(0, values.shape[0], chunk_size):
        tensor = torch.as_tensor(values[start : start + chunk_size], dtype=torch.float32)
        outputs.append(module(tensor).cpu().numpy())
    if not outputs:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(outputs)


@dataclass
class ToleranceAccumulator:
    value: float = math.inf
    witness: dict[str, Any] | None = None
    sample_count: int = 0
    outside_or_boundary_images: int = 0


def evaluate_tolerance_points(
    *,
    model: torch.nn.Module,
    block: BlockGeometry,
    samples: np.ndarray,
    box_ids: np.ndarray,
    source: str,
    accumulator: ToleranceAccumulator,
    per_box_best: np.ndarray,
    source_metadata: dict[str, Any] | None = None,
) -> None:
    batch_size = 131_072
    for start in range(0, samples.shape[0], batch_size):
        stop = min(start + batch_size, samples.shape[0])
        z = samples[start:stop]
        ids = box_ids[start:stop]
        mapped = apply_module(model.latent_map, z)
        clearance = block.clearance(mapped)
        accumulator.sample_count += int(z.shape[0])
        accumulator.outside_or_boundary_images += int(np.count_nonzero(clearance == 0.0))
        np.minimum.at(per_box_best, ids, clearance)
        local = int(np.argmin(clearance))
        value = float(clearance[local])
        if value < accumulator.value:
            witness: dict[str, Any] = {
                "value": value,
                "input_z": z[local].tolist(),
                "image_g_z": mapped[local].tolist(),
                "box_index_zero_based": int(ids[local]),
                "source": source,
            }
            if source_metadata:
                witness["source_metadata"] = source_metadata
            accumulator.value = value
            accumulator.witness = witness


def deterministic_box_points(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = boxes[:, :2]
    upper = boxes[:, 2:]
    centers = 0.5 * (lower + upper)
    samples = np.vstack(
        (
            lower,
            np.column_stack((upper[:, 0], lower[:, 1])),
            upper,
            np.column_stack((lower[:, 0], upper[:, 1])),
            centers,
        )
    )
    box_ids = np.tile(np.arange(boxes.shape[0], dtype=np.int64), 5)
    return samples, box_ids


def sample_tolerance(
    *,
    model: torch.nn.Module,
    block: BlockGeometry,
    target_points: int,
    sobol_scrambles: int,
    local_boxes: int,
    seed: int,
) -> dict[str, Any]:
    started = time.time()
    boxes = block.boxes
    lower = boxes[:, :2]
    upper = boxes[:, 2:]
    span = upper - lower
    per_box_best = np.full(boxes.shape[0], math.inf, dtype=np.float64)
    accumulator = ToleranceAccumulator()

    deterministic, deterministic_ids = deterministic_box_points(boxes)
    evaluate_tolerance_points(
        model=model,
        block=block,
        samples=deterministic,
        box_ids=deterministic_ids,
        source="all_box_corners_and_centers",
        accumulator=accumulator,
        per_box_best=per_box_best,
    )

    requested_sobol = max(0, target_points - deterministic.shape[0])
    points_per_box_per_scramble = max(
        1,
        math.ceil(requested_sobol / (sobol_scrambles * boxes.shape[0])),
    )
    sobol_power = math.ceil(math.log2(points_per_box_per_scramble))
    sobol_seeds: list[int] = []
    for scramble in range(sobol_scrambles):
        sobol_seed = seed + scramble
        sobol_seeds.append(sobol_seed)
        unit = qmc.Sobol(d=2, scramble=True, seed=sobol_seed).random_base2(sobol_power)
        unit = unit[:points_per_box_per_scramble]
        max_chunk_points = 262_144
        box_batch = max(1, max_chunk_points // unit.shape[0])
        for box_start in range(0, boxes.shape[0], box_batch):
            box_stop = min(box_start + box_batch, boxes.shape[0])
            chunk_lower = lower[box_start:box_stop]
            chunk_span = span[box_start:box_stop]
            samples = (chunk_lower[None, :, :] + unit[:, None, :] * chunk_span[None, :, :]).reshape(
                -1, 2
            )
            ids = np.tile(
                np.arange(box_start, box_stop, dtype=np.int64),
                unit.shape[0],
            )
            evaluate_tolerance_points(
                model=model,
                block=block,
                samples=samples,
                box_ids=ids,
                source=f"boxwise_scrambled_sobol_{scramble}",
                accumulator=accumulator,
                per_box_best=per_box_best,
                source_metadata={"sobol_seed": sobol_seed},
            )

    local_search_evaluations = 0
    local_results: list[dict[str, Any]] = []
    if accumulator.value > 0.0 and local_boxes > 0:
        promising = np.argsort(per_box_best)[: min(local_boxes, boxes.shape[0])]

        def objective(z: np.ndarray) -> float:
            image = apply_module(model.latent_map, np.atleast_2d(z), chunk_size=1)
            return float(block.clearance(image)[0])

        for box_index in promising:
            row = boxes[int(box_index)]
            result = differential_evolution(
                objective,
                [(row[0], row[2]), (row[1], row[3])],
                seed=seed + 10_000 + int(box_index),
                popsize=12,
                maxiter=120,
                tol=1e-10,
                polish=True,
                workers=1,
                updating="immediate",
            )
            local_search_evaluations += int(result.nfev)
            local_record = {
                "box_index_zero_based": int(box_index),
                "minimum": float(result.fun),
                "evaluations": int(result.nfev),
                "success": bool(result.success),
            }
            local_results.append(local_record)
            if float(result.fun) < accumulator.value:
                mapped = apply_module(
                    model.latent_map,
                    np.atleast_2d(result.x),
                    chunk_size=1,
                )[0]
                accumulator.value = float(result.fun)
                accumulator.witness = {
                    "value": float(result.fun),
                    "input_z": result.x.tolist(),
                    "image_g_z": mapped.tolist(),
                    "box_index_zero_based": int(box_index),
                    "source": "local_differential_evolution",
                }

    return {
        "formula": "min_{z in T_q} dist_2(g(z), Z \\ Int(N_q))",
        "sampled_minimum": accumulator.value,
        "interpretation": "finite sampled upper estimate of the exact tolerance infimum",
        "witness": accumulator.witness,
        "explicit_latent_samples": accumulator.sample_count,
        "local_search_function_evaluations": local_search_evaluations,
        "total_point_evaluations": accumulator.sample_count + local_search_evaluations,
        "all_explicit_sample_images_in_interior": accumulator.outside_or_boundary_images == 0,
        "outside_or_boundary_image_count": accumulator.outside_or_boundary_images,
        "sampling": {
            "target_points": target_points,
            "actual_explicit_points": accumulator.sample_count,
            "deterministic_corners_and_centers": int(deterministic.shape[0]),
            "sobol_scrambles": sobol_scrambles,
            "sobol_seeds": sobol_seeds,
            "sobol_points_per_box_per_scramble": points_per_box_per_scramble,
            "sobol_generation_power": sobol_power,
            "locally_searched_boxes": len(local_results),
            "local_search_results": local_results,
        },
        "elapsed_seconds": time.time() - started,
    }


def empty_residual_stats() -> dict[str, Any]:
    return {
        "formula": "max_{x in S_q, E(x) in N_q} ||g(E(x)) - E(f(x))||_2",
        "sampled_maximum": None,
        "squared_value_diagnostic": None,
        "evaluated_candidates": 0,
        "accepted_samples": 0,
        "source_summaries": {},
        "witness": None,
    }


def source_summary() -> dict[str, Any]:
    return {
        "evaluated_candidates": 0,
        "accepted_samples": 0,
        "max_euclidean_residual": None,
    }


def update_residual_stats(
    *,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    source: str,
    source_offset: int,
    model: torch.nn.Module,
    scaler: Any,
    blocks: dict[int, BlockGeometry],
    stats: dict[int, dict[str, Any]],
) -> None:
    raw_x = np.atleast_2d(np.asarray(raw_x, dtype=np.float64))
    raw_y = np.atleast_2d(np.asarray(raw_y, dtype=np.float64))
    finite = np.all(np.isfinite(raw_x), axis=1) & np.all(np.isfinite(raw_y), axis=1)
    raw_x = raw_x[finite]
    raw_y = raw_y[finite]
    if raw_x.shape[0] == 0:
        return
    scaled_x = scaler.transform(raw_x)
    scaled_y = scaler.transform(raw_y)
    encoded_x = apply_module(model.encoder, scaled_x)
    encoded_y = apply_module(model.encoder, scaled_y)
    predicted = apply_module(model.latent_map, encoded_x)
    errors = np.linalg.norm(predicted - encoded_y, axis=1)

    for label, block in blocks.items():
        node = stats[label]
        summary = node["source_summaries"].setdefault(source, source_summary())
        node["evaluated_candidates"] += int(raw_x.shape[0])
        summary["evaluated_candidates"] += int(raw_x.shape[0])
        membership = block.membership(encoded_x)
        accepted = int(np.count_nonzero(membership))
        node["accepted_samples"] += accepted
        summary["accepted_samples"] += accepted
        if accepted == 0:
            continue
        selected_rows = np.flatnonzero(membership)
        local_index = int(np.argmax(errors[membership]))
        row = int(selected_rows[local_index])
        value = float(errors[row])
        source_best = summary["max_euclidean_residual"]
        if source_best is None or value > float(source_best):
            summary["max_euclidean_residual"] = value
        node_best = node["sampled_maximum"]
        if node_best is None or value > float(node_best):
            node["sampled_maximum"] = value
            node["squared_value_diagnostic"] = value**2
            node["witness"] = {
                "source": source,
                "row_index_zero_based": source_offset + row,
                "x_raw": raw_x[row].tolist(),
                "E_x": encoded_x[row].tolist(),
                "g_E_x": predicted[row].tolist(),
                "E_f_x": encoded_y[row].tolist(),
                "residual_vector": (predicted[row] - encoded_y[row]).tolist(),
                "euclidean_residual": value,
            }


def sample_stored_pairs(
    *,
    model: torch.nn.Module,
    scaler: Any,
    blocks: dict[int, BlockGeometry],
    stats: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    started = time.time()
    counts: dict[str, int] = {}
    for path in PAIR_PATHS:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        counts[relative(path)] = int(data.shape[0])
        for start in range(0, data.shape[0], 8192):
            chunk = data[start : start + 8192]
            update_residual_stats(
                raw_x=chunk[:, :3],
                raw_y=chunk[:, 3:],
                source=relative(path),
                source_offset=start,
                model=model,
                scaler=scaler,
                blocks=blocks,
                stats=stats,
            )
    return {"files": counts, "elapsed_seconds": time.time() - started}


def sample_global_physical_sobol(
    *,
    system: Any,
    model: torch.nn.Module,
    scaler: Any,
    blocks: dict[int, BlockGeometry],
    stats: dict[int, dict[str, Any]],
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    """Sample independent one-step pairs throughout the report's absorbing box."""
    started = time.time()
    if sample_count <= 0:
        return {
            "sample_count": 0,
            "seed": seed,
            "elapsed_seconds": 0.0,
        }
    power = math.ceil(math.log2(sample_count))
    unit = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(power)
    unit = unit[:sample_count]
    raw = PHYSICAL_LOWER + unit * (PHYSICAL_UPPER - PHYSICAL_LOWER)
    source = "global_physical_absorbing_box_sobol"
    for start in range(0, raw.shape[0], 8192):
        raw_x = raw[start : start + 8192]
        update_residual_stats(
            raw_x=raw_x,
            raw_y=system.step(raw_x),
            source=source,
            source_offset=start,
            model=model,
            scaler=scaler,
            blocks=blocks,
            stats=stats,
        )
    return {
        "sample_count": sample_count,
        "sobol_generation_power": power,
        "seed": seed,
        "elapsed_seconds": time.time() - started,
    }


def sample_fresh_trajectories(
    *,
    system: Any,
    model: torch.nn.Module,
    scaler: Any,
    blocks: dict[int, BlockGeometry],
    stats: dict[int, dict[str, Any]],
    initial_conditions: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    started = time.time()
    if initial_conditions <= 0 or steps <= 0:
        return {
            "initial_conditions": 0,
            "steps": 0,
            "candidate_transitions": 0,
            "seed": seed,
            "elapsed_seconds": 0.0,
        }
    power = math.ceil(math.log2(initial_conditions))
    unit = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(power)
    unit = unit[:initial_conditions]
    lower = PHYSICAL_LOWER
    upper = PHYSICAL_UPPER
    initials = lower + unit * (upper - lower)
    source = "fresh_physical_sobol_trajectories"
    batch_size = 4096
    for batch_start in range(0, initial_conditions, batch_size):
        state = initials[batch_start : batch_start + batch_size]
        for step_index in range(steps):
            next_state = system.step(state)
            update_residual_stats(
                raw_x=state,
                raw_y=next_state,
                source=source,
                source_offset=step_index * initial_conditions + batch_start,
                model=model,
                scaler=scaler,
                blocks=blocks,
                stats=stats,
            )
            state = next_state
    return {
        "initial_conditions": initial_conditions,
        "steps": steps,
        "candidate_transitions": initial_conditions * steps,
        "sobol_generation_power": power,
        "seed": seed,
        "elapsed_seconds": time.time() - started,
    }


def refine_residual_witnesses(
    *,
    system: Any,
    model: torch.nn.Module,
    scaler: Any,
    blocks: dict[int, BlockGeometry],
    stats: dict[int, dict[str, Any]],
    samples_per_scale: int,
    seed: int,
) -> dict[str, Any]:
    """Search Gaussian neighborhoods around the best physical witness per node."""
    started = time.time()
    if samples_per_scale <= 0:
        return {
            "samples_per_scale_per_node": 0,
            "candidate_states": 0,
            "elapsed_seconds": 0.0,
        }
    relative_scales = (
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
    )
    span = PHYSICAL_UPPER - PHYSICAL_LOWER
    rng = np.random.default_rng(seed)
    total = 0
    centers: dict[str, Any] = {}
    for label in sorted(blocks):
        witness = stats[label]["witness"]
        if witness is None:
            centers[str(label)] = {"status": "no_seed_witness"}
            continue
        center = np.asarray(witness["x_raw"], dtype=np.float64)
        centers[str(label)] = {
            "seed_witness_source": witness["source"],
            "seed_x_raw": center.tolist(),
        }
        for scale in relative_scales:
            raw_x = center + scale * span * rng.standard_normal((samples_per_scale, 3))
            raw_x = np.clip(raw_x, PHYSICAL_LOWER, PHYSICAL_UPPER)
            source = f"local_physical_refinement_node_{label}_scale_{scale:g}"
            for start in range(0, raw_x.shape[0], 8192):
                chunk = raw_x[start : start + 8192]
                update_residual_stats(
                    raw_x=chunk,
                    raw_y=system.step(chunk),
                    source=source,
                    source_offset=start,
                    model=model,
                    scaler=scaler,
                    blocks=blocks,
                    stats=stats,
                )
            total += int(raw_x.shape[0])
    return {
        "samples_per_scale_per_node": samples_per_scale,
        "relative_scales": list(relative_scales),
        "candidate_states": total,
        "seed": seed,
        "centers": centers,
        "elapsed_seconds": time.time() - started,
    }


def latent_points_in_block(
    block: BlockGeometry,
    *,
    target: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    boxes = block.boxes
    lower = boxes[:, :2]
    upper = boxes[:, 2:]
    centers = 0.5 * (lower + upper)
    remaining = max(0, target - centers.shape[0])
    points_per_box = max(1, math.ceil(remaining / boxes.shape[0]))
    power = math.ceil(math.log2(points_per_box))
    unit = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(power)
    unit = unit[:points_per_box]
    samples = (lower[None, :, :] + unit[:, None, :] * (upper - lower)[None, :, :]).reshape(-1, 2)
    samples = np.vstack((centers, samples))
    return samples, {
        "target": target,
        "actual": int(samples.shape[0]),
        "box_centers": int(centers.shape[0]),
        "sobol_points_per_box": points_per_box,
        "sobol_generation_power": power,
        "seed": seed,
    }


def sample_decoder_guided_preimages(
    *,
    system: Any,
    model: torch.nn.Module,
    scaler: Any,
    blocks: dict[int, BlockGeometry],
    stats: dict[int, dict[str, Any]],
    target_per_node: int,
    seed: int,
) -> dict[str, Any]:
    started = time.time()
    if target_per_node <= 0:
        return {
            "target_per_node": 0,
            "candidate_states": 0,
            "states_in_physical_domain": 0,
            "elapsed_seconds": 0.0,
        }
    noise_scales = (0.0, 1e-4, 1e-3, 1e-2, 5e-2)
    rng = np.random.default_rng(seed + 100)
    lower = PHYSICAL_LOWER
    upper = PHYSICAL_UPPER
    total_candidates = 0
    total_in_domain = 0
    per_node: dict[str, Any] = {}
    for label, block in blocks.items():
        latent, sampling = latent_points_in_block(
            block,
            target=target_per_node,
            seed=seed + 1000 + label,
        )
        per_node[str(label)] = {"latent_sampling": sampling}
        for start in range(0, latent.shape[0], 4096):
            z = latent[start : start + 4096]
            decoded_scaled = apply_module(model.decoder, z)
            for sigma in noise_scales:
                if sigma == 0.0:
                    candidate_scaled = decoded_scaled
                else:
                    candidate_scaled = decoded_scaled + sigma * rng.standard_normal(
                        decoded_scaled.shape
                    )
                    candidate_scaled = np.clip(candidate_scaled, 0.0, 1.0)
                raw_x = scaler.inverse_transform(candidate_scaled)
                in_domain = np.all((raw_x >= lower) & (raw_x <= upper), axis=1)
                total_candidates += int(raw_x.shape[0])
                total_in_domain += int(np.count_nonzero(in_domain))
                raw_x = raw_x[in_domain]
                if raw_x.shape[0] == 0:
                    continue
                raw_y = system.step(raw_x)
                source = f"decoder_guided_from_node_{label}_noise_{sigma:g}"
                update_residual_stats(
                    raw_x=raw_x,
                    raw_y=raw_y,
                    source=source,
                    source_offset=start,
                    model=model,
                    scaler=scaler,
                    blocks=blocks,
                    stats=stats,
                )
    return {
        "target_per_node": target_per_node,
        "noise_scales_in_scaled_physical_coordinates": list(noise_scales),
        "candidate_states": total_candidates,
        "states_in_physical_domain": total_in_domain,
        "seed": seed + 100,
        "per_node": per_node,
        "elapsed_seconds": time.time() - started,
    }


def optimize_encoder_preimages(
    *,
    system: Any,
    model: torch.nn.Module,
    scaler: Any,
    blocks: dict[int, BlockGeometry],
    stats: dict[int, dict[str, Any]],
    targets_per_node: int,
    restarts: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    """Search physical preimages of interior latent targets by gradient descent."""
    started = time.time()
    if targets_per_node <= 0 or restarts <= 0 or steps <= 0:
        return {
            "targets_per_node": 0,
            "restarts": 0,
            "optimization_steps": 0,
            "candidate_states": 0,
            "elapsed_seconds": 0.0,
        }
    rng = np.random.default_rng(seed + 200)
    torch.manual_seed(seed + 200)
    scaled_lower = torch.as_tensor(
        scaler.transform(PHYSICAL_LOWER[None, :])[0],
        dtype=torch.float32,
    )
    scaled_upper = torch.as_tensor(
        scaler.transform(PHYSICAL_UPPER[None, :])[0],
        dtype=torch.float32,
    )
    total_candidates = 0
    per_node: dict[str, Any] = {}
    for label, block in blocks.items():
        latent, sampling = latent_points_in_block(
            block,
            target=targets_per_node,
            seed=seed + 2000 + label,
        )
        if latent.shape[0] > targets_per_node:
            selection = np.linspace(
                0,
                latent.shape[0] - 1,
                targets_per_node,
                dtype=np.int64,
            )
            latent = latent[selection]
        accepted_before = int(stats[label]["accepted_samples"])
        final_target_errors: list[float] = []
        for start in range(0, latent.shape[0], 512):
            target_np = latent[start : start + 512]
            target = torch.as_tensor(target_np, dtype=torch.float32)
            with torch.no_grad():
                decoded = model.decoder(target).detach()
            for restart in range(restarts):
                if restart == 0:
                    initial = decoded.clone()
                else:
                    sigma = 0.03 if restart == 1 else 0.12
                    noise = torch.as_tensor(
                        rng.standard_normal(decoded.shape),
                        dtype=torch.float32,
                    )
                    initial = torch.clamp(
                        decoded + sigma * noise,
                        min=scaled_lower,
                        max=scaled_upper,
                    )
                initial = torch.clamp(initial, min=scaled_lower, max=scaled_upper)
                physical_scaled = initial.detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([physical_scaled], lr=0.025)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[max(1, steps // 2), max(2, 3 * steps // 4)],
                    gamma=0.2,
                )
                for _ in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    encoded = model.encoder(physical_scaled)
                    loss = torch.sum((encoded - target) ** 2)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        physical_scaled.clamp_(min=scaled_lower, max=scaled_upper)
                    scheduler.step()
                with torch.no_grad():
                    encoded = model.encoder(physical_scaled)
                    errors = torch.linalg.vector_norm(encoded - target, dim=1)
                    final_target_errors.extend(errors.cpu().numpy().astype(float).tolist())
                    candidate_scaled = physical_scaled.detach().cpu().numpy()
                raw_x = scaler.inverse_transform(candidate_scaled)
                raw_y = system.step(raw_x)
                source = f"optimized_encoder_preimages_node_{label}_restart_{restart}"
                update_residual_stats(
                    raw_x=raw_x,
                    raw_y=raw_y,
                    source=source,
                    source_offset=start,
                    model=model,
                    scaler=scaler,
                    blocks=blocks,
                    stats=stats,
                )
                total_candidates += int(raw_x.shape[0])
        per_node[str(label)] = {
            "latent_sampling": sampling,
            "targets_used": int(latent.shape[0]),
            "accepted_into_source_node": int(stats[label]["accepted_samples"]) - accepted_before,
            "target_error_min": min(final_target_errors),
            "target_error_median": float(np.median(final_target_errors)),
            "target_error_max": max(final_target_errors),
        }
    return {
        "targets_per_node": targets_per_node,
        "restarts": restarts,
        "optimization_steps": steps,
        "candidate_states": total_candidates,
        "seed": seed + 200,
        "per_node": per_node,
        "elapsed_seconds": time.time() - started,
    }


def format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "infinity"
    return f"{value:.10g}"


def render_markdown(result: dict[str, Any]) -> str:
    rows = []
    tolerance_counts = []
    for label in sorted(result["nodes"], key=int):
        node = result["nodes"][label]
        tolerance_counts.append(node["tolerance"]["explicit_latent_samples"])
        ratio = node["comparison"]["sampled_residual_over_sampled_tolerance"]
        rows.append(
            "| {label} | {role} | {boxes:,} | {components} | {accepted:,} | {residual} | "
            "{tolerance} | {ratio} | {conclusion} |".format(
                label=label,
                role=node["role"],
                boxes=node["n_boxes"],
                components=node["n_components"],
                accepted=node["residual"]["accepted_samples"],
                residual=format_number(node["residual"]["sampled_maximum"]),
                tolerance=format_number(node["tolerance"]["sampled_minimum"]),
                ratio=format_number(ratio),
                conclusion=node["comparison"]["conclusion"],
            )
        )
    return """# Sampled residual and tolerance - Leslie3D max-30 minimal nodes

Generated {generated}. This is a finite numerical diagnostic, not a certified
uniform bound or a Conley-index certificate.

For the candidate block `N_q` formed by the saved cells of minimal node `q`,

```
R_hat(q)   = max ||g(E(x)) - E(f(x))||_2,  over sampled x with E(x) in N_q
tau_hat(q) = min dist_2(g(z), Z \\ Int(N_q)), over sampled z in N_q.
```

Both quantities use Euclidean distance in the stored latent coordinates. The
residual is not squared. Any sampled `g(z)` outside the block interior receives
zero clearance.

| Node | Role | Boxes | Components | Accepted physical samples | R_hat | tau_hat | Ratio | Diagnostic |
|---:|---|---:|---:|---:|---:|---:|---:|---|
{rows}

The tolerance searches used {tolerance_counts} explicit latent samples,
including all box corners and centers and two independently scrambled boxwise
Sobol samples, followed by local minimization in the lowest-clearance boxes.
Residual candidates combine stored exact pairs, global physical Sobol points,
fresh trajectories, local physical refinement, decoder-guided states, and
optimized encoder preimages.

Interpretation: `R_hat > tau_hat` is a numerical witness contradicting the
sufficient lifting inequality for this candidate block. It does not imply that
the corresponding attractor is spurious. Conversely, `R_hat < tau_hat` would
only mean that this finite search found no violation.

Node 0 corresponds to the direct `P0` period-four attractor. Node 1 is the extra
learned minimal component; it is not the direct `P1` orbit, which belongs to
nonminimal learned node 4.

See `sampled_residual_tolerance.json` for witnesses, source counts, sampling
seeds, timings, software versions, and SHA-256 provenance.
""".format(
        generated=result["generated_at_utc"],
        rows="\n".join(rows),
        tolerance_counts=" and ".join(f"{value:,}" for value in tolerance_counts),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tolerance-target", type=int, default=2**23)
    parser.add_argument("--tolerance-scrambles", type=int, default=2)
    parser.add_argument("--local-boxes", type=int, default=24)
    parser.add_argument("--global-physical-samples", type=int, default=2**22)
    parser.add_argument("--fresh-initials", type=int, default=2**17)
    parser.add_argument("--fresh-steps", type=int, default=64)
    parser.add_argument("--refinement-samples-per-scale", type=int, default=2**16)
    parser.add_argument("--decoder-target", type=int, default=2**18)
    parser.add_argument("--preimage-targets", type=int, default=4096)
    parser.add_argument("--preimage-restarts", type=int, default=3)
    parser.add_argument("--preimage-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    cfg = load_config(CONFIG_PATH)
    model, _ = load_any_checkpoint(RUN_ROOT / "models", arch=cfg.arch)
    model.to(torch.device("cpu"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = joblib.load(SCALER_PATH)
    scaler_warnings = [str(item.message) for item in caught]
    system = build_system(cfg.system.name, cfg.system.params)

    labels = minimal_labels(MORSE_GRAPH_PATH)
    if labels != [0, 1]:
        raise ValueError(f"expected max-30 minimal nodes [0, 1], observed {labels}")
    saved = np.loadtxt(MORSE_SETS_PATH, delimiter=",", ndmin=2)
    boxes = {
        label: np.asarray(saved[saved[:, -1] == label, :-1], dtype=np.float64) for label in labels
    }
    blocks = {label: BlockGeometry(rows) for label, rows in boxes.items()}

    tolerance: dict[int, dict[str, Any]] = {}
    for label in labels:
        tolerance[label] = sample_tolerance(
            model=model,
            block=blocks[label],
            target_points=args.tolerance_target,
            sobol_scrambles=args.tolerance_scrambles,
            local_boxes=args.local_boxes,
            seed=args.seed + 10 * label,
        )

    residual = {label: empty_residual_stats() for label in labels}
    stored_protocol = sample_stored_pairs(
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=residual,
    )
    global_protocol = sample_global_physical_sobol(
        system=system,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=residual,
        sample_count=args.global_physical_samples,
        seed=args.seed,
    )
    refinement_protocol = refine_residual_witnesses(
        system=system,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=residual,
        samples_per_scale=args.refinement_samples_per_scale,
        seed=args.seed + 1,
    )
    fresh_protocol = sample_fresh_trajectories(
        system=system,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=residual,
        initial_conditions=args.fresh_initials,
        steps=args.fresh_steps,
        seed=args.seed + 100,
    )
    decoder_protocol = sample_decoder_guided_preimages(
        system=system,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=residual,
        target_per_node=args.decoder_target,
        seed=args.seed + 200,
    )
    optimized_protocol = optimize_encoder_preimages(
        system=system,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=residual,
        targets_per_node=args.preimage_targets,
        restarts=args.preimage_restarts,
        steps=args.preimage_steps,
        seed=args.seed + 300,
    )

    input_paths = (
        CONFIG_PATH,
        MODEL_PATH,
        MODEL_METADATA_PATH,
        MORSE_SETS_PATH,
        MORSE_GRAPH_PATH,
        RUN_MANIFEST_PATH,
        SCALER_PATH,
        *PAIR_PATHS,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "finite_sample_diagnostic_not_a_certificate",
        "metric": "Euclidean distance in stored latent coordinates",
        "candidate_block": (
            "union of saved recurrent cells for each graph-minimal node; "
            "treated as a candidate attracting block, not a certified block for g"
        ),
        "definitions": {
            "sample_residual": ("max_{x in S_q, E(x) in N_q} ||g(E(x)) - E(f(x))||_2"),
            "sample_tolerance": (
                "min_{z in T_q} dist_2(g(z), Z \\ Int(N_q)); zero outside the interior"
            ),
            "logic": (
                "sample residual is a lower bound on the supremum; sampled tolerance "
                "is an upper estimate of the infimum; R_hat >= tau_hat contradicts "
                "the strict sufficient inequality for the evaluated candidate block"
            ),
        },
        "minimal_nodes": labels,
        "roles": {
            "0": "direct P0 period-four attractor",
            "1": "extra learned period-two attractor",
        },
        "sampling_protocol": {
            "physical_sampling_domain": {
                "lower": PHYSICAL_LOWER.tolist(),
                "upper": PHYSICAL_UPPER.tolist(),
                "note": "report absorbing box; the reusable Leslie system class has doubled plotting bounds",
            },
            "tolerance": {
                "target_explicit_points_per_node": args.tolerance_target,
                "sobol_scrambles": args.tolerance_scrambles,
                "local_boxes": args.local_boxes,
            },
            "residual_stored_pairs": stored_protocol,
            "residual_global_physical_sobol": global_protocol,
            "residual_fresh_trajectories": fresh_protocol,
            "residual_local_physical_refinement": refinement_protocol,
            "residual_decoder_guided": decoder_protocol,
            "residual_optimized_preimages": optimized_protocol,
        },
        "nodes": {},
        "provenance": {
            "inputs": {
                relative(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                for path in input_paths
            },
            "versions": {
                "python": __import__("sys").version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "shapely": shapely.__version__,
                "torch": torch.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "scaler_load_warnings": scaler_warnings,
            "script": relative(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
    }
    for label in labels:
        r_value = residual[label]["sampled_maximum"]
        tau_value = tolerance[label]["sampled_minimum"]
        residual_witness = residual[label]["witness"]
        if residual_witness is not None:
            encoded_witness = np.asarray(residual_witness["E_x"], dtype=np.float64)
            residual_witness["E_x_clearance_inside_block"] = float(
                blocks[label].clearance(encoded_witness[None, :])[0]
            )
            x_witness = np.asarray(residual_witness["x_raw"], dtype=np.float64)
            residual_witness["x_inside_report_absorbing_box"] = bool(
                np.all((x_witness >= PHYSICAL_LOWER) & (x_witness <= PHYSICAL_UPPER))
            )
        if r_value is None:
            conclusion = "residual_unavailable_no_accepted_physical_samples"
            ratio = None
        elif tau_value == 0.0:
            conclusion = "sampled_violation"
            ratio = None
        else:
            ratio = float(r_value) / float(tau_value)
            conclusion = (
                "sampled_violation"
                if float(r_value) >= float(tau_value)
                else "no_sampled_violation_found"
            )
        result["nodes"][str(label)] = {
            "role": result["roles"][str(label)],
            "n_boxes": int(boxes[label].shape[0]),
            "n_components": blocks[label].component_count,
            "set_kind": "saved_minimal_morse_cell_union_candidate_attracting_block",
            "tolerance": tolerance[label],
            "residual": residual[label],
            "comparison": {
                "sampled_residual_over_sampled_tolerance": ratio,
                "ratio_is_infinite": bool(r_value is not None and tau_value == 0.0),
                "conclusion": conclusion,
                "theorem_verified": False,
                "reason_not_verified": (
                    "finite residual sampling provides no uniform residual upper bound"
                    if conclusion != "sampled_violation"
                    else "sampled witnesses contradict the strict sufficient inequality"
                ),
            },
        }
    result["total_elapsed_seconds"] = time.time() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "sampled_residual_tolerance.json"
    markdown_path = args.output_dir / "sampled_residual_tolerance.md"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
