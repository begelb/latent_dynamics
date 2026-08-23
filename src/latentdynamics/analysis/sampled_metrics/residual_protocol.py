"""Dense, reproducible sampling of the semiconjugacy residual.

Supplements every stored transition with fresh Sobol-seeded trajectories and
latent-guided decoder samples.  This remains a numerical search, not a uniform
bound on the residual over an entire encoder preimage.

Entry point: :func:`run_dense_sampling`, or the
``scripts/compute_sampled_residual_tolerance.py`` driver.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from ...config import load_config
from ...systems import build_system
from ...training import load_any_checkpoint
from .tolerance_protocol import (
    EXAMPLES,
    REPO_ROOT,
    BlockGeometry,
    apply_module,
    candidate_points,
    default_output_root,
    load_block_boxes,
    load_scaler,
    reference_results_root,
    residual_lower_bound,
    scale,
    split_pair_files,
)

DISCRETE_TRAJECTORY_INITIALS = {
    "leslie3d_example1": 131_072,
    "leslie_2gen_contraction": 131_072,
    "coral_candidate_train500_seed16": 16_384,
}

DISCRETE_TRAJECTORY_STEPS = {
    "leslie3d_example1": 24,
    "leslie_2gen_contraction": 24,
    "coral_candidate_train500_seed16": 24,
}

CORAL_INITIAL_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
DECODER_TARGET_DISCRETE = 65_536
DECODER_TARGET_CHAFEE = 2_048
DECODER_NOISE_SCALES = (0.0, 1e-4, 1e-3, 1e-2, 5e-2)
CHAFEE_INITIALS = 1_024
CHAFEE_STEPS = 30
BASE_SEED = 20260727


def inverse_scale(scaler, values: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(values, dtype=np.float64)
    return scaler.inverse_transform(values)


def source_template() -> dict[str, object]:
    return {
        "evaluated_samples": 0,
        "accepted_samples": 0,
        "max_euclidean_residual": None,
        "max_squared_residual_diagnostic": None,
    }


def initialize_stats(
    example_name: str,
    spec,
    cfg,
    model,
    scaler,
    blocks: dict[int, object],
) -> dict[int, dict[str, object]]:
    stats: dict[int, dict[str, object]] = {}
    for label, block in blocks.items():
        baseline = residual_lower_bound(
            example_name,
            spec,
            cfg,
            model,
            scaler,
            block,
            label,
        )
        stats[label] = {
            "formula": baseline["formula"],
            "sampled_maximum": baseline["lower_bound"],
            "squared_value_diagnostic": baseline["squared_value_diagnostic"],
            "accepted_samples": int(baseline["accepted_samples"]),
            "evaluated_samples": int(baseline["evaluated_samples"]),
            "source_summaries": copy.deepcopy(baseline["source_summaries"]),
            "witness": copy.deepcopy(baseline["witness"]),
        }
    return stats


def update_stats(
    *,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    source: str,
    source_offset: int,
    model,
    scaler,
    blocks: dict[int, object],
    stats: dict[int, dict[str, object]],
) -> None:
    raw_x = np.atleast_2d(np.asarray(raw_x, dtype=np.float64))
    raw_y = np.atleast_2d(np.asarray(raw_y, dtype=np.float64))
    finite = np.all(np.isfinite(raw_x), axis=1) & np.all(np.isfinite(raw_y), axis=1)
    raw_x = raw_x[finite]
    raw_y = raw_y[finite]
    if raw_x.shape[0] == 0:
        return

    scaled_x = scale(scaler, raw_x)
    scaled_y = scale(scaler, raw_y)
    encoded_x = apply_module(model.encoder, scaled_x)
    encoded_y = apply_module(model.encoder, scaled_y)
    predicted = apply_module(model.latent_map, encoded_x)
    errors = np.linalg.norm(predicted - encoded_y, axis=1)

    for label, block in blocks.items():
        node = stats[label]
        node["evaluated_samples"] = int(node["evaluated_samples"]) + raw_x.shape[0]
        summary = node["source_summaries"].setdefault(source, source_template())
        summary["evaluated_samples"] = int(summary["evaluated_samples"]) + raw_x.shape[0]
        mask = block.membership(encoded_x)
        accepted = int(np.count_nonzero(mask))
        node["accepted_samples"] = int(node["accepted_samples"]) + accepted
        summary["accepted_samples"] = int(summary["accepted_samples"]) + accepted
        if accepted == 0:
            continue

        selected_indices = np.flatnonzero(mask)
        local = int(np.argmax(errors[mask]))
        row = int(selected_indices[local])
        value = float(errors[row])
        previous_source = summary["max_euclidean_residual"]
        if previous_source is None or value > float(previous_source):
            summary["max_euclidean_residual"] = value
            summary["max_squared_residual_diagnostic"] = value**2
        previous = node["sampled_maximum"]
        if previous is None or value > float(previous):
            node["sampled_maximum"] = value
            node["squared_value_diagnostic"] = value**2
            node["witness"] = {
                "source": source,
                "row_index_zero_based": source_offset + row,
                "x_raw": raw_x[row].tolist(),
                "E_x": encoded_x[row].tolist(),
                "g_E_x": predicted[row].tolist(),
                "E_f_x": encoded_y[row].tolist(),
            }


def sobol_points(
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    power = int(math.ceil(math.log2(count)))
    unit = qmc.Sobol(
        d=lower.shape[0],
        scramble=True,
        seed=seed,
    ).random_base2(power)[:count]
    return lower + unit * (upper - lower)


def sample_fresh_trajectories(
    *,
    example_name: str,
    system,
    model,
    scaler,
    blocks: dict[int, object],
    stats: dict[int, dict[str, object]],
    seed_base: int,
    chafee_initials: int,
) -> dict[str, object]:
    started = time.time()
    if example_name == "chafee_infante_current":
        counts_and_scales = [(chafee_initials, 1.0)]
        steps = CHAFEE_STEPS
        batch_size = 32
    elif example_name == "coral_candidate_train500_seed16":
        counts_and_scales = [
            (DISCRETE_TRAJECTORY_INITIALS[example_name], scale)
            for scale in CORAL_INITIAL_SCALES
        ]
        steps = DISCRETE_TRAJECTORY_STEPS[example_name]
        batch_size = 2_048
    else:
        counts_and_scales = [(DISCRETE_TRAJECTORY_INITIALS[example_name], 1.0)]
        steps = DISCRETE_TRAJECTORY_STEPS[example_name]
        batch_size = 4_096

    total_initials = 0
    total_transitions = 0
    for scale_index, (count, scale) in enumerate(counts_and_scales):
        points = sobol_points(
            np.asarray(system.lower_bounds),
            np.asarray(system.upper_bounds),
            count,
            seed=seed_base + scale_index,
        )
        if example_name == "coral_candidate_train500_seed16":
            points = scale * points
        total_initials += count
        source = f"fresh_sobol_trajectories_scale_{scale:g}"
        source_offset = 0
        for start in range(0, count, batch_size):
            state = points[start : start + batch_size]
            for _ in range(steps):
                next_state = system.step(state)
                update_stats(
                    raw_x=state,
                    raw_y=next_state,
                    source=source,
                    source_offset=source_offset,
                    model=model,
                    scaler=scaler,
                    blocks=blocks,
                    stats=stats,
                )
                source_offset += state.shape[0]
                total_transitions += state.shape[0]
                state = next_state
    return {
        "initial_conditions": total_initials,
        "steps_per_initial_condition": steps,
        "candidate_transitions": total_transitions,
        "seed_base": seed_base,
        "elapsed_seconds": time.time() - started,
    }


def points_inside_system(system, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.all(
        (points >= np.asarray(system.lower_bounds))
        & (points <= np.asarray(system.upper_bounds)),
        axis=1,
    )


def sample_decoder_preimages(
    *,
    example_name: str,
    system,
    model,
    scaler,
    blocks: dict[int, object],
    stats: dict[int, dict[str, object]],
    seed_base: int,
) -> dict[str, object]:
    started = time.time()
    target = (
        DECODER_TARGET_CHAFEE
        if example_name == "chafee_infante_current"
        else DECODER_TARGET_DISCRETE
    )
    batch_size = 32 if example_name == "chafee_infante_current" else 4_096
    rng = np.random.default_rng(seed_base + 100)
    total_candidates = 0
    total_in_domain = 0
    per_node: dict[str, object] = {}

    for label, block in blocks.items():
        latent, _, metadata = candidate_points(
            block.boxes,
            target_points=target,
            sobol_scrambles=2,
        )
        per_node[str(label)] = {
            "latent_samples": int(latent.shape[0]),
            "latent_sampling": metadata,
            "noise_scales": list(DECODER_NOISE_SCALES),
        }
        node_offset = 0
        for start in range(0, latent.shape[0], batch_size):
            z = latent[start : start + batch_size]
            decoded_scaled = apply_module(model.decoder, z)
            for sigma in DECODER_NOISE_SCALES:
                if sigma == 0.0:
                    candidate_scaled = decoded_scaled
                elif scaler is None:
                    span = np.asarray(system.upper_bounds) - np.asarray(system.lower_bounds)
                    candidate_scaled = decoded_scaled + sigma * span * rng.standard_normal(
                        decoded_scaled.shape
                    )
                else:
                    candidate_scaled = decoded_scaled + sigma * rng.standard_normal(
                        decoded_scaled.shape
                    )
                    candidate_scaled = np.clip(candidate_scaled, 0.0, 1.0)
                if scaler is None:
                    candidate_scaled = np.clip(
                        candidate_scaled,
                        np.asarray(system.lower_bounds),
                        np.asarray(system.upper_bounds),
                    )
                raw_x = inverse_scale(scaler, candidate_scaled)
                in_domain = points_inside_system(system, raw_x)
                raw_x = raw_x[in_domain]
                total_candidates += candidate_scaled.shape[0]
                total_in_domain += raw_x.shape[0]
                if raw_x.shape[0] == 0:
                    continue
                raw_y = system.step(raw_x)
                source = f"decoder_guided_node_{label}_noise_{sigma:g}"
                update_stats(
                    raw_x=raw_x,
                    raw_y=raw_y,
                    source=source,
                    source_offset=node_offset,
                    model=model,
                    scaler=scaler,
                    blocks=blocks,
                    stats=stats,
                )
                node_offset += raw_x.shape[0]
    return {
        "target_latent_points_per_node": target,
        "noise_scales": list(DECODER_NOISE_SCALES),
        "candidate_states": total_candidates,
        "states_in_system_domain": total_in_domain,
        "seed": seed_base + 100,
        "per_node": per_node,
        "elapsed_seconds": time.time() - started,
    }


def sample_coral_fixed_point_clouds(
    *,
    system,
    model,
    scaler,
    blocks: dict[int, object],
    stats: dict[int, dict[str, object]],
    seed_base: int,
) -> dict[str, object]:
    started = time.time()
    rng = np.random.default_rng(seed_base + 200)
    count_per_cloud = 8_192
    scales = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2)
    total = 0
    for name in ("a0", "a1"):
        fixed = np.asarray(system.FIXED_POINTS[name], dtype=np.float64)
        for sigma in scales:
            if sigma == 0.0:
                raw_x = fixed[None, :]
            elif name == "a0":
                raw_x = (
                    sigma
                    * np.asarray(system.upper_bounds)
                    * rng.random((count_per_cloud, system.dim))
                )
            else:
                raw_x = fixed + (
                    sigma
                    * np.asarray(system.upper_bounds)
                    * rng.standard_normal((count_per_cloud, system.dim))
                )
                raw_x = np.clip(
                    raw_x,
                    np.asarray(system.lower_bounds),
                    np.asarray(system.upper_bounds),
                )
            raw_y = system.step(raw_x)
            update_stats(
                raw_x=raw_x,
                raw_y=raw_y,
                source=f"coral_fixed_point_cloud_{name}_scale_{sigma:g}",
                source_offset=0,
                model=model,
                scaler=scaler,
                blocks=blocks,
                stats=stats,
            )
            total += raw_x.shape[0]
    return {
        "fixed_points": ["a0", "a1"],
        "cloud_size_per_nonzero_scale": count_per_cloud,
        "relative_scales": list(scales),
        "candidate_states": total,
        "seed": seed_base + 200,
        "elapsed_seconds": time.time() - started,
    }


def run_dense_sampling(
    example: str,
    *,
    seed: int = BASE_SEED,
    chafee_initials: int = CHAFEE_INITIALS,
    output_suffix: str | None = None,
    skip_fresh_trajectories: bool = False,
    output_root: Path | None = None,
    blocks_root: Path | None = None,
) -> Path:
    """Run the dense residual search for one example and write dense_sampling.json.

    Requires the example's ``tolerance_evaluation.json``, looked up first under
    ``<output_root>/<example>/`` and then in the shipped reference results.
    Returns the written path.
    """
    spec = EXAMPLES[example]
    cfg = load_config(spec.config)
    source_root = REPO_ROOT / spec.root
    model, _ = load_any_checkpoint(source_root / "models", arch=cfg.arch)
    model.to(torch.device("cpu"))
    model.eval()
    scaler = load_scaler(spec.scaler)
    system = build_system(cfg.system.name, cfg.system.params)
    output_root = Path(output_root) if output_root is not None else default_output_root()
    blocks_root = Path(blocks_root) if blocks_root is not None else reference_results_root()

    tolerance_path = output_root / example / "tolerance_evaluation.json"
    if not tolerance_path.is_file():
        tolerance_path = blocks_root / example / "tolerance_evaluation.json"
    tolerance_result = json.loads(tolerance_path.read_text())
    labels = sorted(int(label) for label in tolerance_result["nodes"])
    blocks: dict[int, object] = {}
    n_boxes: dict[int, int] = {}
    for label in labels:
        boxes, _ = load_block_boxes(example, source_root, label, blocks_root)
        blocks[label] = BlockGeometry(boxes)
        n_boxes[label] = int(boxes.shape[0])

    stats = initialize_stats(
        example,
        spec,
        cfg,
        model,
        scaler,
        blocks,
    )
    if skip_fresh_trajectories:
        fresh = {
            "initial_conditions": 0,
            "steps_per_initial_condition": 0,
            "candidate_transitions": 0,
            "seed_base": seed,
            "elapsed_seconds": 0.0,
        }
    else:
        fresh = sample_fresh_trajectories(
            example_name=example,
            system=system,
            model=model,
            scaler=scaler,
            blocks=blocks,
            stats=stats,
            seed_base=seed,
            chafee_initials=chafee_initials,
        )
    decoder = sample_decoder_preimages(
        example_name=example,
        system=system,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=stats,
        seed_base=seed,
    )
    coral_clouds = None
    if example == "coral_candidate_train500_seed16":
        coral_clouds = sample_coral_fixed_point_clouds(
            system=system,
            model=model,
            scaler=scaler,
            blocks=blocks,
            stats=stats,
            seed_base=seed,
        )

    stored_present, stored_missing = split_pair_files(spec.pair_files)

    output: dict[str, object] = {
        "example": example,
        "metric": "Euclidean distance in stored latent coordinates",
        "interpretation": (
            "Both quantities are dense sampled estimates. "
            "R_hat < tau_hat means that no sampled violation was found, not that "
            "the uniform theorem hypothesis was proved."
        ),
        "sampling_protocol": {
            "stored_transitions": stored_present,
            "stored_transitions_missing": stored_missing,
            "fresh_trajectories": fresh,
            "decoder_guided_preimages": decoder,
            "coral_fixed_point_clouds": coral_clouds,
        },
        "nodes": {},
    }
    for label in labels:
        tolerance_node = tolerance_result["nodes"][str(label)]["tolerance"]
        tau = float(tolerance_node["upper_bound"])
        residual = stats[label]
        residual_value = residual["sampled_maximum"]
        ratio = None if residual_value is None or tau == 0.0 else residual_value / tau
        sampled_conclusion = (
            "sampled_violation"
            if residual_value is not None and residual_value >= tau
            else "no_sampled_violation_found"
        )
        output["nodes"][str(label)] = {
            "n_boxes": n_boxes[label],
            "tolerance": {
                "sampled_minimum": tau,
                "sample_count": int(tolerance_node["sample_count"]),
                "sampling": tolerance_node.get("sampling"),
                "all_sample_images_in_interior": tolerance_node[
                    "all_sample_images_in_interior"
                ],
                "witness": tolerance_node["upper_bound_witness"],
            },
            "residual": residual,
            "comparison": {
                "sampled_residual_over_sampled_tolerance": ratio,
                "sampled_conclusion": sampled_conclusion,
            },
        }

    out_dir = output_root / example
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_suffix:
        out_path = out_dir / f"dense_sampling_{output_suffix}.json"
    else:
        out_path = out_dir / "dense_sampling.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path
