#!/usr/bin/env python3
"""Paper-comparable sampled residual/tolerance audit for the successful Ives run.

The candidate blocks are the saved cell unions of the two graph-minimal nodes
in data seed 2158 / model seed 2.  Both quantities use the unsquared Euclidean
metric in stored latent coordinates.  This is a finite numerical search, not a
uniform bound or a certificate.

Run from ``code/`` with Shapely available::

    uv run --with shapely python \
      scripts/compute_ives_myvatn_sampled_residual_tolerance.py

The defaults reproduce the sampling convention behind Table 8 of
``paper/main_KM2.tex``: at least ``2**23`` explicit latent samples per minimal
node, two independently scrambled boxwise Sobol designs, local search in the
12 lowest-clearance boxes, stored state pairs, 131072 fresh Sobol trajectories
of length 24, and decoder-guided states.
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
DEFAULT_SWEEP_ROOT = CODE_ROOT / "output" / "ives_myvatn_seedsweep_3x5_v1"
DEFAULT_DATA_ROOT = CODE_ROOT / "data" / "ives_myvatn_seedsweep_3x5_v1"
DEFAULT_DATA_SEED = 2158
DEFAULT_MODEL_SEED = 2
DEFAULT_PROTOCOL_SEED = 20260727
TOLERANCE_SOBOL_SEED = 20260725
PHYSICAL_LOWER = np.asarray([-3.0, -7.5, -3.0], dtype=np.float64)
PHYSICAL_UPPER = np.asarray([1.5, 1.5, 1.5], dtype=np.float64)
EXPECTED_MINIMAL_LABELS = [0, 2]
ROLES = {
    0: "stable period-12 orbit",
    2: "stable fixed point",
}

# Values as displayed in Table 8 of paper/main_KM2.tex.  Ratios are derived
# here so the cross-example comparison is dimensionless.
PAPER_TABLE_ROWS = (
    ("1st Leslie 3D", 0, 5.68e5, 1.07, 4.25e-4),
    ("1st Leslie 3D", 1, 3.78e6, 6.97e-1, 4.06e-4),
    ("1st Leslie 3D", 4, 2.84e4, 2.31e-1, 4.62e-4),
    ("2nd Leslie 3D", 0, 4.46e2, 8.48e-2, 4.57e-5),
    ("2nd Leslie 3D", 1, 6.70e3, 9.75e-2, 4.48e-5),
    ("2D Leslie in 10D", 0, 2.24e6, 6.80e-2, 5.20e-5),
    ("2D Leslie in 10D", 1, 1.31e5, 5.31e-2, 5.41e-5),
    ("Red coral", 0, 1.94e6, 5.40e-2, 7.79e-3),
    ("Red coral", 1, 7.16e5, 2.48e-1, 7.96e-3),
    ("Chafee--Infante d=1", 0, 1.33e5, 6.58, 1.04e-1),
    ("Chafee--Infante d=1", 1, 1.31e5, 6.11, 6.58e-2),
    ("Chafee--Infante d=2", 0, 1.11e5, 3.52e-2, 3.95e-2),
    ("Chafee--Infante d=2", 1, 1.25e5, 1.60e-2, 4.25e-2),
    ("Chafee--Infante d=3", 0, 1.19e5, 4.31e-3, 2.34e-2),
    ("Chafee--Infante d=3", 1, 1.18e5, 4.73e-3, 2.36e-2),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path.resolve())


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
        if self.boxes.shape[0] == 0:
            raise ValueError("block must contain at least one box")
        if np.any(self.boxes[:, 2:] <= self.boxes[:, :2]):
            raise ValueError("block contains a non-positive-width box")

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
        if values.shape[1] != 2:
            raise ValueError(f"expected points with two coordinates, got {values.shape}")
        if interior:
            return np.asarray(contains_xy(self.geometry, values[:, 0], values[:, 1]))
        return np.asarray(covers(self.geometry, points(values)))

    def clearance(self, values: np.ndarray) -> np.ndarray:
        """Distance to the relative complement, zero outside/on the boundary."""
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
    values = np.atleast_2d(np.asarray(values))
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
    explicit_sample_count: int = 0
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
    source_summaries: dict[str, dict[str, Any]],
    source_metadata: dict[str, Any] | None = None,
) -> None:
    summary = source_summaries.setdefault(
        source,
        {"explicit_samples": 0, "sampled_minimum": None},
    )
    batch_size = 131_072
    for start in range(0, samples.shape[0], batch_size):
        stop = min(start + batch_size, samples.shape[0])
        z = samples[start:stop]
        ids = box_ids[start:stop]
        mapped = apply_module(model.latent_map, z)
        clearance = block.clearance(mapped)
        count = int(z.shape[0])
        accumulator.explicit_sample_count += count
        accumulator.outside_or_boundary_images += int(np.count_nonzero(clearance == 0.0))
        summary["explicit_samples"] += count
        np.minimum.at(per_box_best, ids, clearance)
        local = int(np.argmin(clearance))
        value = float(clearance[local])
        old_source_value = summary["sampled_minimum"]
        if old_source_value is None or value < float(old_source_value):
            summary["sampled_minimum"] = value
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
    sobol_seed: int,
    label: int | None = None,
) -> dict[str, Any]:
    """Estimate tau with the paper's boxwise dense-sampling convention."""
    started = time.time()
    boxes = block.boxes
    lower = boxes[:, :2]
    span = boxes[:, 2:] - lower
    per_box_best = np.full(boxes.shape[0], math.inf, dtype=np.float64)
    accumulator = ToleranceAccumulator()
    source_summaries: dict[str, dict[str, Any]] = {}

    deterministic, deterministic_ids = deterministic_box_points(boxes)
    evaluate_tolerance_points(
        model=model,
        block=block,
        samples=deterministic,
        box_ids=deterministic_ids,
        source="all_box_corners_and_centers",
        accumulator=accumulator,
        per_box_best=per_box_best,
        source_summaries=source_summaries,
    )
    prefix = "tolerance" if label is None else f"tolerance M{label}"
    print(
        f"{prefix}: deterministic {deterministic.shape[0]:,}, "
        f"minimum={accumulator.value:.12g}",
        flush=True,
    )

    requested_sobol = max(0, target_points - deterministic.shape[0])
    points_per_box_per_scramble = max(
        1,
        math.ceil(requested_sobol / (sobol_scrambles * boxes.shape[0])),
    )
    sobol_power = math.ceil(math.log2(points_per_box_per_scramble))
    sobol_seeds: list[int] = []
    minima_after_sources = [
        {
            "source": "all_box_corners_and_centers",
            "cumulative_explicit_samples": accumulator.explicit_sample_count,
            "sampled_minimum": accumulator.value,
        }
    ]
    for scramble in range(sobol_scrambles):
        current_seed = sobol_seed + scramble
        sobol_seeds.append(current_seed)
        unit = qmc.Sobol(d=2, scramble=True, seed=current_seed).random_base2(sobol_power)
        unit = unit[:points_per_box_per_scramble]
        max_chunk_points = 262_144
        box_batch = max(1, max_chunk_points // unit.shape[0])
        source = f"boxwise_scrambled_sobol_{scramble}"
        for box_start in range(0, boxes.shape[0], box_batch):
            box_stop = min(box_start + box_batch, boxes.shape[0])
            chunk_lower = lower[box_start:box_stop]
            chunk_span = span[box_start:box_stop]
            samples = (
                chunk_lower[None, :, :] + unit[:, None, :] * chunk_span[None, :, :]
            ).reshape(-1, 2)
            ids = np.tile(
                np.arange(box_start, box_stop, dtype=np.int64),
                unit.shape[0],
            )
            evaluate_tolerance_points(
                model=model,
                block=block,
                samples=samples,
                box_ids=ids,
                source=source,
                accumulator=accumulator,
                per_box_best=per_box_best,
                source_summaries=source_summaries,
                source_metadata={"sobol_seed": current_seed},
            )
        minima_after_sources.append(
            {
                "source": source,
                "cumulative_explicit_samples": accumulator.explicit_sample_count,
                "sampled_minimum": accumulator.value,
            }
        )
        print(
            f"{prefix}: Sobol {scramble + 1}/{sobol_scrambles}, "
            f"samples={accumulator.explicit_sample_count:,}, "
            f"minimum={accumulator.value:.12g}",
            flush=True,
        )

    local_search_evaluations = 0
    local_results: list[dict[str, Any]] = []
    if accumulator.value > 0.0 and local_boxes > 0:
        promising = np.argsort(per_box_best)[: min(local_boxes, boxes.shape[0])]

        def objective(z: np.ndarray) -> float:
            image = apply_module(model.latent_map, np.atleast_2d(z), chunk_size=1)
            return float(block.clearance(image)[0])

        for index, box_index_value in enumerate(promising):
            box_index = int(box_index_value)
            row = boxes[box_index]
            result = differential_evolution(
                objective,
                [(row[0], row[2]), (row[1], row[3])],
                seed=box_index,
                popsize=10,
                maxiter=80,
                tol=1e-9,
                polish=True,
                workers=1,
                updating="immediate",
            )
            local_search_evaluations += int(result.nfev)
            local_record = {
                "box_index_zero_based": box_index,
                "sampled_box_minimum": float(per_box_best[box_index]),
                "optimized_minimum": float(result.fun),
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
                    "box_index_zero_based": box_index,
                    "source": "local_differential_evolution",
                }
            print(
                f"{prefix}: local {index + 1}/{len(promising)}, "
                f"minimum={accumulator.value:.12g}",
                flush=True,
            )

    return {
        "formula": "min_{z in T_q} dist_2(g(z), Z \\ Int(N_q))",
        "sampled_minimum": accumulator.value,
        "interpretation": (
            "finite sampled minimum and therefore an upper estimate of the exact "
            "tolerance infimum, not a certified lower bound"
        ),
        "witness": accumulator.witness,
        "explicit_latent_samples": accumulator.explicit_sample_count,
        "local_search_function_evaluations": local_search_evaluations,
        "total_point_evaluations": (
            accumulator.explicit_sample_count + local_search_evaluations
        ),
        "all_explicit_sample_images_in_interior": (
            accumulator.outside_or_boundary_images == 0
        ),
        "outside_or_boundary_image_count": accumulator.outside_or_boundary_images,
        "sampling": {
            "target_points": target_points,
            "actual_explicit_points": accumulator.explicit_sample_count,
            "deterministic_corners_and_centers": int(deterministic.shape[0]),
            "sobol_scrambles": sobol_scrambles,
            "sobol_seeds": sobol_seeds,
            "sobol_points_per_box_per_scramble": points_per_box_per_scramble,
            "sobol_generation_power": sobol_power,
            "locally_searched_boxes": len(local_results),
            "local_search_results": local_results,
            "source_summaries": source_summaries,
            "minimum_after_each_source": minima_after_sources,
        },
        "elapsed_seconds": time.time() - started,
    }


def empty_residual_stats() -> dict[str, Any]:
    return {
        "formula": "max_{x in S_q} ||g(E(x)) - E(f(x))||_2",
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


def residual_source_maximum(
    source_summaries: dict[str, dict[str, Any]],
    prefixes: tuple[str, ...],
) -> float | None:
    values = [
        float(summary["max_euclidean_residual"])
        for source, summary in source_summaries.items()
        if source.startswith(prefixes) and summary["max_euclidean_residual"] is not None
    ]
    return max(values) if values else None


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
        count = int(raw_x.shape[0])
        node["evaluated_candidates"] += count
        summary["evaluated_candidates"] += count
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
    pair_paths: tuple[Path, ...],
    model: torch.nn.Module,
    scaler: Any,
    blocks: dict[int, BlockGeometry],
    stats: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    started = time.time()
    counts: dict[str, int] = {}
    for path in pair_paths:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        if data.ndim != 2 or data.shape[1] != 6:
            raise ValueError(f"expected six-column Ives pairs in {path}, got {data.shape}")
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
    initials = PHYSICAL_LOWER + unit[:initial_conditions] * (PHYSICAL_UPPER - PHYSICAL_LOWER)
    source = "fresh_full_domain_sobol_trajectories"
    batch_size = 4096
    batch_count = math.ceil(initial_conditions / batch_size)
    for batch_index, batch_start in enumerate(range(0, initial_conditions, batch_size)):
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
        if (batch_index + 1) % 4 == 0 or batch_index + 1 == batch_count:
            maxima = ", ".join(
                f"M{label}={stats[label]['sampled_maximum']:.8g}"
                for label in sorted(stats)
                if stats[label]["sampled_maximum"] is not None
            )
            print(
                f"residual trajectories: batch {batch_index + 1}/{batch_count}; {maxima}",
                flush=True,
            )
    return {
        "initial_conditions": initial_conditions,
        "steps": steps,
        "candidate_transitions": initial_conditions * steps,
        "sobol_generation_power": power,
        "seed": seed,
        "elapsed_seconds": time.time() - started,
    }


def latent_points_in_block(
    block: BlockGeometry,
    *,
    target: int,
    seed: int,
    sobol_scrambles: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    deterministic, _ = deterministic_box_points(block.boxes)
    remaining = max(0, target - deterministic.shape[0])
    points_per_box_per_scramble = max(
        1,
        math.ceil(remaining / (sobol_scrambles * block.boxes.shape[0])),
    )
    power = math.ceil(math.log2(points_per_box_per_scramble))
    lower = block.boxes[:, :2]
    span = block.boxes[:, 2:] - lower
    samples = [deterministic]
    seeds = []
    for scramble in range(sobol_scrambles):
        current_seed = seed + scramble
        seeds.append(current_seed)
        unit = qmc.Sobol(d=2, scramble=True, seed=current_seed).random_base2(power)
        unit = unit[:points_per_box_per_scramble]
        samples.append((lower[None, :, :] + unit[:, None, :] * span[None, :, :]).reshape(-1, 2))
    values = np.vstack(samples)
    return values, {
        "target": target,
        "actual": int(values.shape[0]),
        "deterministic_corners_and_centers": int(deterministic.shape[0]),
        "sobol_scrambles": sobol_scrambles,
        "sobol_seeds": seeds,
        "sobol_points_per_box_per_scramble": points_per_box_per_scramble,
        "sobol_generation_power": power,
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
            for noise_index, sigma in enumerate(noise_scales):
                if sigma == 0.0:
                    candidate_scaled = decoded_scaled
                else:
                    candidate_scaled = decoded_scaled + sigma * rng.standard_normal(
                        decoded_scaled.shape
                    )
                    candidate_scaled = np.clip(candidate_scaled, 0.0, 1.0)
                raw_x = scaler.inverse_transform(candidate_scaled)
                in_domain = np.all(
                    (raw_x >= PHYSICAL_LOWER) & (raw_x <= PHYSICAL_UPPER),
                    axis=1,
                )
                total_candidates += int(raw_x.shape[0])
                total_in_domain += int(np.count_nonzero(in_domain))
                raw_x = raw_x[in_domain]
                if raw_x.shape[0] == 0:
                    continue
                update_residual_stats(
                    raw_x=raw_x,
                    raw_y=system.step(raw_x),
                    source=f"decoder_guided_from_node_{label}_noise_{sigma:g}",
                    source_offset=noise_index * latent.shape[0] + start,
                    model=model,
                    scaler=scaler,
                    blocks=blocks,
                    stats=stats,
                )
        maxima = ", ".join(
            f"M{node}={stats[node]['sampled_maximum']:.8g}"
            for node in sorted(stats)
            if stats[node]["sampled_maximum"] is not None
        )
        print(f"residual decoder source M{label} complete; {maxima}", flush=True)
    return {
        "target_per_node": target_per_node,
        "noise_scales_in_scaled_physical_coordinates": list(noise_scales),
        "candidate_states": total_candidates,
        "states_in_physical_domain": total_in_domain,
        "seed": seed + 100,
        "per_node": per_node,
        "elapsed_seconds": time.time() - started,
    }


def format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "infinity"
    return f"{value:.10g}"


def paper_crosscheck() -> list[dict[str, Any]]:
    return [
        {
            "example": example,
            "node": node,
            "accepted_samples_as_displayed": accepted,
            "sampled_residual_as_displayed": residual,
            "sampled_tolerance_as_displayed": tolerance,
            "ratio_from_displayed_values": residual / tolerance,
            "sampled_inequality_holds_as_displayed": residual < tolerance,
        }
        for example, node, accepted, residual, tolerance in PAPER_TABLE_ROWS
    ]


def render_markdown(result: dict[str, Any]) -> str:
    result_rows = []
    source_rows = []
    for label in sorted(result["nodes"], key=int):
        node = result["nodes"][label]
        result_rows.append(
            "| M{label} | {role} | {boxes:,} | {components} | {accepted:,} | "
            "{candidates:,} | {latent:,} | {residual} | {tolerance} | {ratio} | {diagnostic} |".format(
                label=label,
                role=node["role"],
                boxes=node["n_boxes"],
                components=node["n_components"],
                accepted=node["residual"]["accepted_samples"],
                candidates=node["residual"]["evaluated_candidates"],
                latent=node["tolerance"]["explicit_latent_samples"],
                residual=format_number(node["residual"]["sampled_maximum"]),
                tolerance=format_number(node["tolerance"]["sampled_minimum"]),
                ratio=format_number(
                    node["comparison"]["sampled_residual_over_sampled_tolerance"]
                ),
                diagnostic=node["comparison"]["conclusion"].replace("_", " "),
            )
        )
        robustness = node["comparison"]["source_robustness"]
        source_rows.append(
            "| M{label} | {legacy} | {stored} | {fresh} | {decoder} | {ratio} |".format(
                label=label,
                legacy=format_number(robustness["legacy_lightweight_residual"]),
                stored=format_number(robustness["stored_pairs_maximum"]),
                fresh=format_number(robustness["fresh_trajectories_maximum"]),
                decoder=format_number(robustness["decoder_guided_maximum"]),
                ratio=format_number(robustness["non_decoder_residual_over_tolerance"]),
            )
        )

    comparison_rows = []
    for row in result["paper_crosscheck"]["table_8_rows"]:
        comparison_rows.append(
            "| {example} | {node} | {accepted:.3g} | {residual:.3g} | {tolerance:.3g} | "
            "{ratio:.3g} | {status} |".format(
                example=row["example"],
                node=row["node"],
                accepted=row["accepted_samples_as_displayed"],
                residual=row["sampled_residual_as_displayed"],
                tolerance=row["sampled_tolerance_as_displayed"],
                ratio=row["ratio_from_displayed_values"],
                status=(
                    "no sampled violation"
                    if row["sampled_inequality_holds_as_displayed"]
                    else "sampled violation"
                ),
            )
        )

    return """# Ives--Myvatn sampled residual and tolerance

Generated {generated} for the successful learned run: data seed 2158, model
seed 2. This is a finite numerical diagnostic, not a certified uniform bound.

For each graph-minimal node, the saved cell union is used as the candidate
block `N_q`:

```
R_hat(q)   = max ||g(E(x)) - E(f(x))||_2,  sampled x with E(x) in N_q
tau_hat(q) = min dist_2(g(z), Z \\ Int(N_q)), sampled z in N_q
```

Both are unsquared Euclidean distances in the stored two-dimensional latent
coordinates. Images outside or on the block boundary receive zero clearance.

| Set | Role | Boxes | Components | Accepted S_q | Residual candidates | Explicit latent samples | R_hat | tau_hat | Ratio | Diagnostic |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{result_rows}

`R_hat >= tau_hat` is a sampled witness against the strict sufficient lifting
inequality for that candidate block. It does not mean the attractor is
spurious. Conversely, `R_hat < tau_hat` would only mean no sampled violation
was found.

## Residual-source robustness

| Set | Legacy lightweight R_hat | Stored pairs max | Fresh trajectories max | Decoder-guided max | Best non-decoder / tau_hat |
|---|---:|---:|---:|---:|---:|
{source_rows}

The final maxima come from decoder-guided states, as in the paper protocol,
but that source is not needed for the conclusion: stored pairs plus fresh
trajectories already give `R_hat > tau_hat` for both sets. The legacy metric
used only 4096 post-transient physical samples and a corner-based tolerance;
the dense search covers the full experiment domain and every retained
trajectory time.

## Cross-check against paper Table 8

| Paper example | q | |S_q| | R_hat | tau_hat | Ratio | Diagnostic |
|---|---:|---:|---:|---:|---:|---|
{comparison_rows}

The useful cross-example comparison is the inequality and dimensionless ratio,
not the raw latent distances: independently trained latent coordinates have
different geometry. The Leslie-family rows have ratios about 500--2516; the
two-dimensional Chafee--Infante rows with no sampled violation have ratios
about 0.376 and 0.891.

The tolerance computation evaluates all corners and centers of every cell,
two independently scrambled boxwise Sobol designs with at least `2^23`
explicit latent points per node, and local differential-evolution searches in
the lowest-clearance cells. The residual pool combines stored train/validation
pairs, 131072 fresh Sobol initial conditions followed for 24 steps, and
decoder-guided states at five noise scales.

See `sampled_residual_tolerance.json` for witnesses, per-source counts, seeds,
timings, software versions, and SHA-256 provenance. The paper definitions and
comparison table are at `paper/main_KM2.tex:2940-2979`.
""".format(
        generated=result["generated_at_utc"],
        result_rows="\n".join(result_rows),
        source_rows="\n".join(source_rows),
        comparison_rows="\n".join(comparison_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage", choices=("all", "tolerance", "residual"), default="all")
    parser.add_argument("--tolerance-target", type=int, default=2**23)
    parser.add_argument("--tolerance-scrambles", type=int, default=2)
    parser.add_argument("--local-boxes", type=int, default=12)
    parser.add_argument("--fresh-initials", type=int, default=2**17)
    parser.add_argument("--fresh-steps", type=int, default=24)
    parser.add_argument("--decoder-target", type=int, default=2**16)
    parser.add_argument("--seed", type=int, default=DEFAULT_PROTOCOL_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    run_root = (
        args.sweep_root.resolve()
        / f"dataset_{args.data_seed}"
        / f"seed_{args.model_seed}"
    )
    data_root = args.data_root.resolve() / f"dataset_{args.data_seed}"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_root / "analysis" / "sampled_residual_tolerance"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = CODE_ROOT / "src" / "latentdynamics" / "configs" / "ives_myvatn.yaml"
    model_path = run_root / "models" / "autoencoder.pt"
    model_metadata_path = run_root / "models" / "autoencoder.json"
    morse_sets_path = run_root / "MG" / "morse_sets"
    morse_graph_path = run_root / "MG" / "morse_graph"
    run_manifest_path = run_root / "run_manifest.json"
    metrics_path = run_root / "metrics.json"
    scaler_path = run_root.parent / "scalers" / "train" / "scaler.gz"
    pair_paths = (data_root / "train.csv", data_root / "val.csv")

    required = (
        config_path,
        model_path,
        model_metadata_path,
        morse_sets_path,
        morse_graph_path,
        run_manifest_path,
        metrics_path,
        scaler_path,
        *pair_paths,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(map(str, missing)))

    cfg = load_config(config_path)
    model, _ = load_any_checkpoint(run_root / "models", arch=cfg.arch)
    model.to(torch.device("cpu"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = joblib.load(scaler_path)
    scaler_warnings = [str(item.message) for item in caught]
    system = build_system(cfg.system.name, cfg.system.params)
    np.testing.assert_allclose(system.lower_bounds, PHYSICAL_LOWER, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(system.upper_bounds, PHYSICAL_UPPER, rtol=0.0, atol=0.0)

    labels = minimal_labels(morse_graph_path)
    if labels != EXPECTED_MINIMAL_LABELS:
        raise ValueError(
            f"expected successful-run minimal nodes {EXPECTED_MINIMAL_LABELS}, observed {labels}"
        )
    saved = np.loadtxt(morse_sets_path, delimiter=",", ndmin=2)
    boxes = {
        label: np.asarray(saved[saved[:, -1] == label, :-1], dtype=np.float64)
        for label in labels
    }
    blocks = {label: BlockGeometry(rows) for label, rows in boxes.items()}

    tolerance_path = output_dir / "tolerance_sampling.json"
    if args.stage in {"all", "tolerance"}:
        tolerance: dict[int, dict[str, Any]] = {}
        for label in labels:
            print(
                f"tolerance M{label}: {boxes[label].shape[0]:,} boxes, "
                f"target {args.tolerance_target:,}",
                flush=True,
            )
            tolerance[label] = sample_tolerance(
                model=model,
                block=blocks[label],
                target_points=args.tolerance_target,
                sobol_scrambles=args.tolerance_scrambles,
                local_boxes=args.local_boxes,
                sobol_seed=TOLERANCE_SOBOL_SEED,
                label=label,
            )
        tolerance_payload = {
            "status": "complete",
            "metric": "Euclidean distance in stored latent coordinates",
            "minimal_nodes": labels,
            "nodes": {str(label): tolerance[label] for label in labels},
        }
        tolerance_path.write_text(
            json.dumps(tolerance_payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:
        tolerance_payload = json.loads(tolerance_path.read_text(encoding="utf-8"))
        tolerance = {int(label): node for label, node in tolerance_payload["nodes"].items()}

    if args.stage == "tolerance":
        print(tolerance_path, flush=True)
        return

    residual = {label: empty_residual_stats() for label in labels}
    print("residual: stored train and validation pairs", flush=True)
    stored_protocol = sample_stored_pairs(
        pair_paths=pair_paths,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=residual,
    )
    print("residual: fresh Sobol trajectories", flush=True)
    fresh_protocol = sample_fresh_trajectories(
        system=system,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=residual,
        initial_conditions=args.fresh_initials,
        steps=args.fresh_steps,
        seed=args.seed,
    )
    print("residual: decoder-guided states", flush=True)
    decoder_protocol = sample_decoder_guided_preimages(
        system=system,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=residual,
        target_per_node=args.decoder_target,
        seed=args.seed,
    )

    baseline = json.loads(metrics_path.read_text(encoding="utf-8"))
    input_paths = required
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "finite_sample_diagnostic_not_a_certificate",
        "run": {
            "system": "Ives Lake Myvatn map",
            "data_seed": args.data_seed,
            "model_seed": args.model_seed,
            "run_root": relative(run_root),
        },
        "metric": "Euclidean distance in stored latent coordinates",
        "candidate_block": (
            "union of saved recurrent cells for each graph-minimal node; candidate "
            "attracting block for the learned map, not a certified block"
        ),
        "definitions": {
            "sampled_residual": "max_{x in S_q} ||g(E(x)) - E(f(x))||_2",
            "sampled_tolerance": (
                "min_{z in T_q} dist_2(g(z), Z \\ Int(N_q)); zero outside/on boundary"
            ),
            "logic": (
                "R_hat is a sampled lower bound on the supremum and tau_hat is a "
                "sampled upper estimate of the infimum. R_hat >= tau_hat contradicts "
                "the strict sufficient inequality for the evaluated candidate block; "
                "the reverse comparison is inconclusive."
            ),
        },
        "minimal_nodes": labels,
        "roles": {str(label): ROLES[label] for label in labels},
        "sampling_protocol": {
            "physical_sampling_domain_log10": {
                "lower": PHYSICAL_LOWER.tolist(),
                "upper": PHYSICAL_UPPER.tolist(),
            },
            "tolerance": {
                "target_explicit_points_per_node": args.tolerance_target,
                "sobol_scrambles": args.tolerance_scrambles,
                "sobol_seed_base": TOLERANCE_SOBOL_SEED,
                "local_boxes": args.local_boxes,
            },
            "residual_stored_pairs": stored_protocol,
            "residual_fresh_trajectories": fresh_protocol,
            "residual_decoder_guided": decoder_protocol,
        },
        "paper_crosscheck": {
            "source": "paper/main_KM2.tex:2940-2979, Table 8",
            "note": (
                "raw latent distances depend on learned-coordinate geometry; compare "
                "the inequality and dimensionless ratio across examples"
            ),
            "table_8_rows": paper_crosscheck(),
        },
        "legacy_lightweight_metrics": baseline,
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
            residual_witness["x_inside_sampling_domain"] = bool(
                np.all((x_witness >= PHYSICAL_LOWER) & (x_witness <= PHYSICAL_UPPER))
            )
        if r_value is None:
            conclusion = "residual_unavailable_no_accepted_samples"
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
        source_summaries = residual[label]["source_summaries"]

        stored_maximum = residual_source_maximum(source_summaries, ("code/data/",))
        fresh_maximum = residual_source_maximum(
            source_summaries,
            ("fresh_full_domain_sobol_trajectories",),
        )
        decoder_maximum = residual_source_maximum(source_summaries, ("decoder_guided_",))
        non_decoder_values = [
            value for value in (stored_maximum, fresh_maximum) if value is not None
        ]
        non_decoder_maximum = max(non_decoder_values) if non_decoder_values else None
        non_decoder_ratio = (
            None
            if non_decoder_maximum is None or tau_value == 0.0
            else non_decoder_maximum / float(tau_value)
        )
        legacy_node = baseline["minimal_morse_sets"][str(label)]
        result["nodes"][str(label)] = {
            "role": ROLES[label],
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
                    "sampled witnesses contradict the strict sufficient inequality"
                    if conclusion == "sampled_violation"
                    else "finite residual sampling gives no uniform residual upper bound"
                ),
                "source_robustness": {
                    "legacy_lightweight_residual": legacy_node[
                        "max_semiconjugacy_error"
                    ],
                    "stored_pairs_maximum": stored_maximum,
                    "fresh_trajectories_maximum": fresh_maximum,
                    "decoder_guided_maximum": decoder_maximum,
                    "non_decoder_maximum": non_decoder_maximum,
                    "non_decoder_residual_over_tolerance": non_decoder_ratio,
                    "sampled_violation_without_decoder_guidance": bool(
                        non_decoder_maximum is not None
                        and non_decoder_maximum >= float(tau_value)
                    ),
                },
            },
        }
    result["total_elapsed_seconds"] = time.time() - started

    json_path = output_dir / "sampled_residual_tolerance.json"
    markdown_path = output_dir / "sampled_residual_tolerance.md"
    json_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json_path, flush=True)


if __name__ == "__main__":
    main()
