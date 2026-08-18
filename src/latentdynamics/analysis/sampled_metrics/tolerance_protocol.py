"""Numerically evaluate the attracting-block tolerance criterion.

The reported residual is always the Euclidean norm

    ||g(E(x)) - E(f(x))||_2,

not the squared training loss.  Finite residual searches provide lower bounds
on the supremum.  Pointwise tolerance searches provide upper bounds on the
infimum clearance.  Interval-bound propagation supplies a numerical lower
enclosure for the tolerance when every image enclosure lies inside the block.
It is not outward-rounded and is therefore not labelled a rigorous certificate.

The dense tolerance search targets at least ``2**23`` explicit latent samples
per minimal node: every box corner and center, two independently scrambled
boxwise Sobol sequences, and local differential-evolution refinement in the
most promising cells.

Entry point: :func:`run_tolerance_evaluation`, or the
``scripts/compute_sampled_residual_tolerance.py`` driver.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from ..._paths import get_repo_root
from ...config import load_config
from ...systems import build_system
from ...training import load_any_checkpoint

REPO_ROOT = get_repo_root()

DEFAULT_SAMPLE_TARGET = 2**23
DEFAULT_SOBOL_SCRAMBLES = 2
DEFAULT_LOCAL_BOXES = 12
TOLERANCE_SOBOL_SEED_BASE = 20260725


def reference_results_root() -> Path:
    """Frozen published results shipped with the repository."""
    return REPO_ROOT / "artifacts" / "reference_results" / "sampled_residual_tolerance"


def default_output_root() -> Path:
    return REPO_ROOT / "output" / "sampled_residual_tolerance"


def _shapely():
    """Import shapely lazily; only 2-D block geometry needs it."""
    try:
        import shapely
    except ImportError as error:
        raise ImportError(
            "two-dimensional attracting-block geometry requires the optional "
            "'shapely' dependency; install it with 'pip install shapely'"
        ) from error
    return shapely


@dataclass(frozen=True)
class Example:
    """Inputs for one experiment, as paths relative to the repo root.

    ``pair_files`` are one-step transition CSVs (header row, then
    ``2 * high_dims`` columns).  Fetched artifact bundles place them under
    ``replay_sources/<experiment>/``.
    """

    config: str
    root: str
    pair_files: tuple[str, ...]
    scaler: str | None
    train_file: str = "train"


EXAMPLES = {
    "chafee_infante_current": Example(
        config="chafee_infante_replay",
        root="replay_sources/chafee_infante/replay",
        pair_files=("replay_sources/chafee_infante/data/train.csv",),
        scaler=None,
    ),
    "leslie3d_example1": Example(
        config="leslie3d_example1_replay",
        root="replay_sources/leslie3d_example1/spurious_attractor_ex",
        pair_files=(
            "replay_sources/leslie3d_example1/data/Leslie_3D_larger_domain_tail_only/28.9_29.8_22.0/2train.csv",
            "replay_sources/leslie3d_example1/data/Leslie_3D_larger_domain_tail_only/28.9_29.8_22.0/2test.csv",
            "replay_sources/leslie3d_example1/data_pairs/train.csv",
            "replay_sources/leslie3d_example1/data_pairs/val.csv",
        ),
        scaler="replay_sources/leslie3d_example1/28.9_29.8_22.0/scalers/scaler.gz",
    ),
    "leslie_2gen_contraction": Example(
        config="leslie_2gen_contraction_replay",
        root="replay_sources/leslie_2gen_contraction",
        pair_files=(
            "replay_sources/leslie_2gen_contraction/data_pairs/train.csv",
            "replay_sources/leslie_2gen_contraction/data_pairs/val.csv",
        ),
        scaler="replay_sources/leslie_2gen_contraction/scalers/train/scaler.gz",
    ),
    "coral_candidate_train500_seed16": Example(
        config="coral_basic",
        root="replay_sources/coral/train_500/seed_16",
        pair_files=(
            "replay_sources/coral/data/coral/train_500.csv",
            "replay_sources/coral/data/coral/test.csv",
        ),
        scaler="replay_sources/coral/data/scalers/train_500/scaler.gz",
        train_file="train_500",
    ),
}


def minimal_labels(dot_path: Path) -> list[int]:
    text = dot_path.read_text()
    nodes = {int(x) for x in re.findall(r"^(\d+)\s+\[label=", text, re.M)}
    has_out = {int(x) for x in re.findall(r"^(\d+)\s*->", text, re.M)}
    return sorted(nodes - has_out)


def merge_intervals(boxes: np.ndarray) -> np.ndarray:
    rows = np.asarray(boxes, dtype=np.float64)
    rows = rows[np.argsort(rows[:, 0])]
    merged: list[list[float]] = []
    for lo, hi in rows:
        if not merged or lo > merged[-1][1] + 1e-13:
            merged.append([float(lo), float(hi)])
        else:
            merged[-1][1] = max(merged[-1][1], float(hi))
    return np.asarray(merged, dtype=np.float64)


class BlockGeometry:
    """Union-of-boxes geometry for a 1-D or 2-D latent attracting block."""

    def __init__(self, boxes: np.ndarray) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64)
        self.dimension = self.boxes.shape[1] // 2
        if self.dimension == 1:
            self.intervals = merge_intervals(self.boxes)
            self.geometry = None
            self.boundary = None
        elif self.dimension == 2:
            shapely = _shapely()
            self._shapely = shapely
            rects = shapely.box(
                self.boxes[:, 0],
                self.boxes[:, 1],
                self.boxes[:, 2],
                self.boxes[:, 3],
            )
            try:
                self.geometry = shapely.coverage_union_all(rects)
            except shapely.errors.GEOSException:
                self.geometry = shapely.unary_union(rects)
            if not self.geometry.is_valid:
                self.geometry = shapely.make_valid(self.geometry)
            self.boundary = shapely.boundary(self.geometry)
            self.intervals = None
        else:
            raise ValueError(f"only 1D and 2D latent blocks are supported, got {self.dimension}D")

    def membership(self, values: np.ndarray, *, interior: bool = False) -> np.ndarray:
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if self.dimension == 1:
            x = values[:, 0]
            mask = np.zeros(x.shape[0], dtype=bool)
            for lo, hi in self.intervals:
                if interior:
                    mask |= (lo < x) & (x < hi)
                else:
                    mask |= (lo <= x) & (x <= hi)
            return mask
        shapely = self._shapely
        if interior:
            return np.asarray(shapely.contains_xy(self.geometry, values[:, 0], values[:, 1]))
        return np.asarray(shapely.covers(self.geometry, shapely.points(values)))

    def clearance(self, values: np.ndarray) -> np.ndarray:
        """Distance to the relative complement, returning zero outside the interior."""
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        result = np.zeros(values.shape[0], dtype=np.float64)
        if self.dimension == 1:
            x = values[:, 0]
            for lo, hi in self.intervals:
                mask = (lo < x) & (x < hi)
                result[mask] = np.minimum(x[mask] - lo, hi - x[mask])
            return result
        shapely = self._shapely
        interior = self.membership(values, interior=True)
        if np.any(interior):
            result[interior] = shapely.distance(shapely.points(values[interior]), self.boundary)
        return result

    def enclosure_clearance(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clearance of whole interval enclosures and a containment mask."""
        lower = np.atleast_2d(np.asarray(lower, dtype=np.float64))
        upper = np.atleast_2d(np.asarray(upper, dtype=np.float64))
        if self.dimension == 1:
            contained = np.zeros(lower.shape[0], dtype=bool)
            clearance = np.zeros(lower.shape[0], dtype=np.float64)
            for lo, hi in self.intervals:
                mask = (lo < lower[:, 0]) & (upper[:, 0] < hi)
                contained |= mask
                clearance[mask] = np.minimum(lower[mask, 0] - lo, hi - upper[mask, 0])
            return clearance, contained
        shapely = self._shapely
        output_rects = shapely.box(lower[:, 0], lower[:, 1], upper[:, 0], upper[:, 1])
        contained = np.asarray(shapely.covers(self.geometry, output_rects))
        clearance = np.zeros(lower.shape[0], dtype=np.float64)
        if np.any(contained):
            clearance[contained] = shapely.distance(output_rects[contained], self.boundary)
        contained &= clearance > 0.0
        return clearance, contained


@torch.no_grad()
def apply_module(module: torch.nn.Module, values: np.ndarray, chunk: int = 65536) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, values.shape[0], chunk):
        tensor = torch.as_tensor(values[start : start + chunk], dtype=torch.float32)
        output.append(module(tensor).cpu().numpy())
    return np.vstack(output)


def interval_bound_network(
    network: torch.nn.Module, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary float64 interval-bound propagation through a sequential MLP."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    children = list(network.net.children())
    for layer in children:
        if isinstance(layer, torch.nn.Linear):
            weight = layer.weight.detach().cpu().numpy().astype(np.float64)
            bias = layer.bias.detach().cpu().numpy().astype(np.float64)
            positive = np.maximum(weight, 0.0)
            negative = np.minimum(weight, 0.0)
            next_lo = lo @ positive.T + hi @ negative.T + bias
            next_hi = hi @ positive.T + lo @ negative.T + bias
            lo, hi = next_lo, next_hi
        elif isinstance(layer, torch.nn.ReLU):
            lo, hi = np.maximum(lo, 0.0), np.maximum(hi, 0.0)
        elif isinstance(layer, torch.nn.Tanh):
            lo, hi = np.tanh(lo), np.tanh(hi)
        elif isinstance(layer, torch.nn.Sigmoid):
            lo = 1.0 / (1.0 + np.exp(-lo))
            hi = 1.0 / (1.0 + np.exp(-hi))
        elif isinstance(layer, (torch.nn.Identity, torch.nn.Dropout)):
            continue
        else:
            raise TypeError(f"unsupported interval layer: {type(layer).__name__}")
    return lo, hi


def candidate_points(
    boxes: np.ndarray,
    *,
    target_points: int,
    sobol_scrambles: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    dimension = boxes.shape[1] // 2
    lower = boxes[:, :dimension]
    upper = boxes[:, dimension:]
    centers = (lower + upper) / 2.0
    if dimension == 1:
        base_samples = [lower, upper, centers]
    else:
        base_samples = [
            lower,
            np.column_stack((upper[:, 0], lower[:, 1])),
            upper,
            np.column_stack((lower[:, 0], upper[:, 1])),
            centers,
        ]
    deterministic_count = sum(rows.shape[0] for rows in base_samples)
    requested_sobol = max(0, target_points - deterministic_count)
    points_per_box_per_scramble = max(
        1,
        math.ceil(requested_sobol / (sobol_scrambles * boxes.shape[0])),
    )
    sobol_power = int(math.ceil(math.log2(points_per_box_per_scramble)))
    sobol_count = 2**sobol_power
    sobol_seeds = [TOLERANCE_SOBOL_SEED_BASE + index for index in range(sobol_scrambles)]
    for seed in sobol_seeds:
        unit = qmc.Sobol(
            d=dimension,
            scramble=True,
            seed=seed,
        ).random_base2(sobol_power)[:points_per_box_per_scramble]
        samples = (
            lower[None, :, :]
            + unit[:, None, :] * (upper - lower)[None, :, :]
        ).reshape(-1, dimension)
        base_samples.append(samples)
    samples = np.vstack(base_samples)
    deterministic_repetitions = 3 if dimension == 1 else 5
    box_ids = np.concatenate(
        [
            np.tile(np.arange(boxes.shape[0]), deterministic_repetitions),
            *[
                np.tile(np.arange(boxes.shape[0]), points_per_box_per_scramble)
                for _ in sobol_seeds
            ],
        ]
    )
    metadata = {
        "target_points": int(target_points),
        "actual_points": int(samples.shape[0]),
        "deterministic_points": int(deterministic_count),
        "sobol_scrambles": int(sobol_scrambles),
        "sobol_seeds": sobol_seeds,
        "sobol_points_per_box_per_scramble": int(points_per_box_per_scramble),
        "sobol_generation_power": sobol_power,
        "sobol_generated_then_truncated_per_scramble": sobol_count,
    }
    return samples, box_ids, metadata


def uniformly_refine_boxes(boxes: np.ndarray, rounds: int) -> np.ndarray:
    """Split every box in half along every latent coordinate per round."""
    refined = np.asarray(boxes, dtype=np.float64)
    dimension = refined.shape[1] // 2
    for _ in range(rounds):
        lower = refined[:, :dimension]
        upper = refined[:, dimension:]
        middle = (lower + upper) / 2.0
        children = []
        for corner_bits in np.ndindex(*(2,) * dimension):
            bits = np.asarray(corner_bits, dtype=bool)
            child_lower = np.where(bits, middle, lower)
            child_upper = np.where(bits, upper, middle)
            children.append(np.hstack((child_lower, child_upper)))
        refined = np.vstack(children)
    return refined


def ibp_tolerance_lower_bound(
    model: torch.nn.Module,
    block: BlockGeometry,
    boxes: np.ndarray,
    *,
    refinement_target: float,
) -> dict[str, object]:
    dimension = block.dimension
    frozen_lower_bound = math.inf
    pending = np.asarray(boxes, dtype=np.float64)
    total_evaluated = 0
    depth_counts: list[dict[str, int]] = []
    batch_size = 4096
    max_total_evaluated = 5_000_000
    max_adaptive_rounds = 6
    remaining_failures = pending.shape[0]
    remaining_low_clearance = 0
    complete = False
    for depth in range(max_adaptive_rounds + 1):
        contained_parts = []
        clearance_parts = []
        for start in range(0, pending.shape[0], batch_size):
            rows = pending[start : start + batch_size]
            lo, hi = rows[:, :dimension], rows[:, dimension:]
            out_lo, out_hi = interval_bound_network(model.latent_map, lo, hi)
            enclosed_clearance, contained = block.enclosure_clearance(out_lo, out_hi)
            contained_parts.append(contained)
            clearance_parts.append(enclosed_clearance)
        contained = np.concatenate(contained_parts)
        enclosed_clearance = np.concatenate(clearance_parts)
        total_evaluated += int(pending.shape[0])
        low_clearance = contained & (enclosed_clearance < refinement_target)
        refine_mask = (~contained) | low_clearance
        frozen = contained & (~low_clearance)
        if np.any(frozen):
            frozen_lower_bound = min(
                frozen_lower_bound,
                float(np.min(enclosed_clearance[frozen])),
            )
        remaining_failures = int(np.count_nonzero(~contained))
        remaining_low_clearance = int(np.count_nonzero(low_clearance))
        depth_counts.append(
            {
                "adaptive_depth": depth,
                "evaluated": int(pending.shape[0]),
                "frozen_with_clearance_at_least_target": int(np.count_nonzero(frozen)),
                "refine_low_clearance": remaining_low_clearance,
                "refine_not_strictly_enclosed": remaining_failures,
            }
        )
        if not np.any(refine_mask):
            complete = True
            break
        if depth == max_adaptive_rounds:
            if remaining_failures == 0:
                frozen_lower_bound = min(
                    frozen_lower_bound,
                    float(np.min(enclosed_clearance[low_clearance])),
                )
                remaining_low_clearance = 0
                complete = True
            break
        boxes_to_refine = pending[refine_mask]
        next_count = boxes_to_refine.shape[0] * (2**dimension)
        if (
            total_evaluated + next_count > max_total_evaluated
        ):
            break
        pending = uniformly_refine_boxes(boxes_to_refine, 1)
    if not complete or not np.isfinite(frozen_lower_bound):
        lower_bound = 0.0
    else:
        lower_bound = frozen_lower_bound
    return {
        "lower_bound": lower_bound,
        "complete": complete,
        "remaining_failures": remaining_failures,
        "remaining_low_clearance": remaining_low_clearance,
        "refinement_target": refinement_target,
        "total_evaluated": total_evaluated,
        "depth_counts": depth_counts,
    }


def tolerance_bounds(
    model: torch.nn.Module,
    block: BlockGeometry,
    *,
    local_boxes: int,
    sample_target: int,
    sobol_scrambles: int,
) -> dict[str, object]:
    samples, box_ids, sampling = candidate_points(
        block.boxes,
        target_points=sample_target,
        sobol_scrambles=sobol_scrambles,
    )
    mapped = apply_module(model.latent_map, samples)
    clearance = block.clearance(mapped)
    best_index = int(np.argmin(clearance))
    best = {
        "value": float(clearance[best_index]),
        "input": samples[best_index].tolist(),
        "image": mapped[best_index].tolist(),
        "box_index": int(box_ids[best_index]),
        "source": "box_corners_centers_and_sobol",
    }

    def objective(z: np.ndarray) -> float:
        image = apply_module(model.latent_map, np.atleast_2d(z), chunk=1)
        return float(block.clearance(image)[0])

    promising = []
    for index in np.argsort(clearance):
        box_index = int(box_ids[index])
        if box_index not in promising:
            promising.append(box_index)
        if len(promising) >= local_boxes:
            break
    dimension = block.dimension
    for box_index in promising:
        row = block.boxes[box_index]
        lower, upper = row[:dimension], row[dimension:]
        result = differential_evolution(
            objective,
            list(zip(lower, upper)),
            seed=box_index,
            popsize=10,
            maxiter=80,
            tol=1e-9,
            polish=True,
            workers=1,
            updating="immediate",
        )
        if float(result.fun) < best["value"]:
            image = apply_module(model.latent_map, np.atleast_2d(result.x), chunk=1)[0]
            best = {
                "value": float(result.fun),
                "input": result.x.tolist(),
                "image": image.tolist(),
                "box_index": box_index,
                "source": "local_differential_evolution",
            }

    max_refined_boxes = 20_000
    children_per_round = 2**dimension
    refinement_rounds = 0
    while (
        refinement_rounds < 6
        and block.boxes.shape[0] * children_per_round ** (refinement_rounds + 1)
        <= max_refined_boxes
    ):
        refinement_rounds += 1
    ibp_boxes = uniformly_refine_boxes(block.boxes, refinement_rounds)
    ibp = ibp_tolerance_lower_bound(
        model,
        block,
        ibp_boxes,
        refinement_target=max(1e-8, 0.25 * float(best["value"])),
    )

    return {
        "formula": "inf_{z in N} dist_2(g(z), Z \\\\ Int(N))",
        "upper_bound": best["value"],
        "upper_bound_witness": best,
        "sample_count": int(samples.shape[0]),
        "sampling": sampling,
        "all_sample_images_in_interior": bool(np.all(clearance > 0.0)),
        "numerical_ibp_lower_bound": ibp["lower_bound"],
        "ibp_complete": ibp["complete"],
        "ibp_remaining_failed_enclosures": ibp["remaining_failures"],
        "ibp_remaining_low_clearance_enclosures": ibp["remaining_low_clearance"],
        "ibp_refinement_target": ibp["refinement_target"],
        "ibp_total_evaluated_enclosures": ibp["total_evaluated"],
        "ibp_adaptive_depth_counts": ibp["depth_counts"],
        "ibp_initial_enclosures": int(ibp_boxes.shape[0]),
        "ibp_uniform_refinement_rounds": refinement_rounds,
        "ibp_is_outward_rounded": False,
    }


def load_scaler(path: str | None):
    if path is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(REPO_ROOT / path)


def scale(scaler, values: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(values, dtype=np.float64)
    return scaler.transform(values)


def pair_batches(
    files: Iterable[str],
    high_dimension: int,
    *,
    batch_size: int = 8192,
) -> Iterable[tuple[str, int, np.ndarray, np.ndarray]]:
    for relative in files:
        path = REPO_ROOT / relative
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        for start in range(0, data.shape[0], batch_size):
            chunk = data[start : start + batch_size]
            yield relative, start, chunk[:, :high_dimension], chunk[:, high_dimension:]


def residual_lower_bound(
    example_name: str,
    spec: Example,
    cfg,
    model: torch.nn.Module,
    scaler,
    block: BlockGeometry,
    label: int,
) -> dict[str, object]:
    best_error = -math.inf
    witness: dict[str, object] | None = None
    accepted = 0
    evaluated = 0
    source_summaries: dict[str, dict[str, object]] = {}

    for source, offset, raw_x, raw_y in pair_batches(spec.pair_files, cfg.arch.high_dims):
        source_summary = source_summaries.setdefault(
            source,
            {
                "evaluated_samples": 0,
                "accepted_samples": 0,
                "max_euclidean_residual": None,
                "max_squared_residual_diagnostic": None,
            },
        )
        scaled_x = scale(scaler, raw_x)
        scaled_y = scale(scaler, raw_y)
        encoded_x = apply_module(model.encoder, scaled_x)
        mask = block.membership(encoded_x)
        evaluated += int(raw_x.shape[0])
        accepted += int(np.count_nonzero(mask))
        source_summary["evaluated_samples"] += int(raw_x.shape[0])
        source_summary["accepted_samples"] += int(np.count_nonzero(mask))
        if not np.any(mask):
            continue
        selected_x = scaled_x[mask]
        selected_y = scaled_y[mask]
        selected_z = encoded_x[mask]
        encoded_next = apply_module(model.encoder, selected_y)
        predicted = apply_module(model.latent_map, selected_z)
        errors = np.linalg.norm(predicted - encoded_next, axis=1)
        local_index = int(np.argmax(errors))
        value = float(errors[local_index])
        old_source_max = source_summary["max_euclidean_residual"]
        if old_source_max is None or value > old_source_max:
            source_summary["max_euclidean_residual"] = value
            source_summary["max_squared_residual_diagnostic"] = value**2
        if value > best_error:
            original_indices = np.flatnonzero(mask)
            row_index = offset + int(original_indices[local_index])
            best_error = value
            witness = {
                "source": source,
                "row_index_zero_based": row_index,
                "x_raw": raw_x[original_indices[local_index]].tolist(),
                "E_x": selected_z[local_index].tolist(),
                "g_E_x": predicted[local_index].tolist(),
                "E_f_x": encoded_next[local_index].tolist(),
            }

    if example_name == "leslie3d_example1":
        indexed_path = (
            REPO_ROOT
            / spec.root
            / "plot_data"
            / "preimage_plot_data_indexed.pkl"
        )
        with indexed_path.open("rb") as stream:
            indexed = pickle.load(stream)
        indexed_labels = np.asarray(indexed["labels"], dtype=int)
        label_mask = indexed_labels == label
        if np.any(label_mask):
            raw_x = np.column_stack(
                (
                    np.asarray(indexed["x"])[label_mask],
                    np.asarray(indexed["y"])[label_mask],
                    np.asarray(indexed["z"])[label_mask],
                )
            )
            system = build_system(cfg.system.name, cfg.system.params)
            raw_y = system.step(raw_x)
            scaled_x = scale(scaler, raw_x)
            scaled_y = scale(scaler, raw_y)
            encoded_x = apply_module(model.encoder, scaled_x)
            mask = block.membership(encoded_x)
            indexed_source = str(indexed_path.relative_to(REPO_ROOT))
            indexed_summary = source_summaries.setdefault(
                indexed_source,
                {
                    "evaluated_samples": 0,
                    "accepted_samples": 0,
                    "max_euclidean_residual": None,
                    "max_squared_residual_diagnostic": None,
                },
            )
            evaluated += int(raw_x.shape[0])
            accepted += int(np.count_nonzero(mask))
            indexed_summary["evaluated_samples"] += int(raw_x.shape[0])
            indexed_summary["accepted_samples"] += int(np.count_nonzero(mask))
            if np.any(mask):
                encoded_next = apply_module(model.encoder, scaled_y[mask])
                predicted = apply_module(model.latent_map, encoded_x[mask])
                errors = np.linalg.norm(predicted - encoded_next, axis=1)
                local_index = int(np.argmax(errors))
                value = float(errors[local_index])
                indexed_summary["max_euclidean_residual"] = value
                indexed_summary["max_squared_residual_diagnostic"] = value**2
                if value > best_error:
                    original_index = int(np.flatnonzero(mask)[local_index])
                    best_error = value
                    witness = {
                        "source": indexed_source,
                        "row_index_zero_based": original_index,
                        "x_raw": raw_x[original_index].tolist(),
                        "E_x": encoded_x[mask][local_index].tolist(),
                        "g_E_x": predicted[local_index].tolist(),
                        "E_f_x": encoded_next[local_index].tolist(),
                    }

    if example_name == "leslie3d_example1" and label == 4:
        special = REPO_ROOT / spec.root / "plot_data" / "preimage_samples_k4_20pts.pkl"
        with special.open("rb") as stream:
            raw_x = np.asarray(pickle.load(stream), dtype=np.float64)
        special_source = str(special.relative_to(REPO_ROOT))
        special_summary = source_summaries.setdefault(
            special_source,
            {
                "evaluated_samples": 0,
                "accepted_samples": 0,
                "max_euclidean_residual": None,
                "max_squared_residual_diagnostic": None,
            },
        )
        system = build_system(cfg.system.name, cfg.system.params)
        raw_y = system.step(raw_x)
        scaled_x = scale(scaler, raw_x)
        scaled_y = scale(scaler, raw_y)
        encoded_x = apply_module(model.encoder, scaled_x)
        mask = block.membership(encoded_x)
        evaluated += int(raw_x.shape[0])
        accepted += int(np.count_nonzero(mask))
        special_summary["evaluated_samples"] += int(raw_x.shape[0])
        special_summary["accepted_samples"] += int(np.count_nonzero(mask))
        if np.any(mask):
            encoded_next = apply_module(model.encoder, scaled_y[mask])
            predicted = apply_module(model.latent_map, encoded_x[mask])
            errors = np.linalg.norm(predicted - encoded_next, axis=1)
            local_index = int(np.argmax(errors))
            value = float(errors[local_index])
            special_summary["max_euclidean_residual"] = value
            special_summary["max_squared_residual_diagnostic"] = value**2
            if value > best_error:
                original_index = int(np.flatnonzero(mask)[local_index])
                best_error = value
                witness = {
                    "source": special_source,
                    "row_index_zero_based": original_index,
                    "x_raw": raw_x[original_index].tolist(),
                    "E_x": encoded_x[mask][local_index].tolist(),
                    "g_E_x": predicted[local_index].tolist(),
                    "E_f_x": encoded_next[local_index].tolist(),
                }

    return {
        "formula": "sup_{x in E^{-1}(N)} ||g(E(x)) - E(f(x))||_2",
        "lower_bound": None if witness is None else best_error,
        "squared_value_diagnostic": None if witness is None else best_error**2,
        "accepted_samples": accepted,
        "evaluated_samples": evaluated,
        "source_summaries": source_summaries,
        "witness": witness,
        "upper_bound": None,
        "note": "finite samples give only a lower bound on the uniform residual",
    }


def load_block_boxes(
    example_name: str,
    source_root: Path,
    label: int,
    blocks_root: Path,
) -> tuple[np.ndarray, str]:
    """Load one node's block boxes, preferring verified attracting-block artifacts.

    ``blocks_root`` holds per-example directories with ``attracting_blocks.json``
    and ``block_<label>.npz`` from the forward-closure verification; the shipped
    reference copies live under :func:`reference_results_root`.
    """
    artifact = blocks_root / example_name / f"block_{label}.npz"
    if artifact.is_file():
        if example_name != "chafee_infante_current":
            saved_path = source_root / "MG" / "morse_sets"
            if saved_path.is_file():
                saved = np.loadtxt(saved_path, delimiter=",", ndmin=2)
                rows = saved[saved[:, -1] == label, :-1]
                if rows.size:
                    manifest_path = blocks_root / example_name / "attracting_blocks.json"
                    manifest = json.loads(manifest_path.read_text())
                    block_info = manifest["minimal_blocks"].get(str(label), {})
                    live_node = block_info.get("live_node")
                    live_info = manifest["live_nodes"].get(str(live_node), {})
                    match_fraction = live_info.get("best_saved_match_fraction")
                    if match_fraction == 1.0:
                        kind = "saved_morse_set_with_exact_live_forward_invariance_check"
                    else:
                        kind = (
                            "saved_morse_set_with_near_geometry_live_forward_invariance_check"
                        )
                    return np.asarray(rows, dtype=np.float64), kind
        with np.load(artifact) as data:
            return np.asarray(data["block_boxes"], dtype=np.float64), "forward_closure_from_live_map_graph"
    saved = np.loadtxt(source_root / "MG" / "morse_sets", delimiter=",", ndmin=2)
    rows = saved[saved[:, -1] == label, :-1]
    return np.asarray(rows, dtype=np.float64), "saved_recurrent_morse_set_unchecked"


def run_tolerance_evaluation(
    example: str,
    *,
    labels: list[int] | None = None,
    local_boxes: int = DEFAULT_LOCAL_BOXES,
    sample_target: int = DEFAULT_SAMPLE_TARGET,
    sobol_scrambles: int = DEFAULT_SOBOL_SCRAMBLES,
    output_root: Path | None = None,
    blocks_root: Path | None = None,
) -> Path:
    """Run the dense tolerance search for every minimal node of one example.

    Writes ``<output_root>/<example>/tolerance_evaluation.json`` (or a
    ``tolerance_evaluation_labels_<...>.json`` variant when explicit labels are
    given) and returns the written path.
    """
    spec = EXAMPLES[example]
    cfg = load_config(spec.config)
    source_root = REPO_ROOT / spec.root
    model, _ = load_any_checkpoint(source_root / "models", arch=cfg.arch)
    model.to(torch.device("cpu"))
    model.eval()
    scaler = load_scaler(spec.scaler)
    output_root = Path(output_root) if output_root is not None else default_output_root()
    blocks_root = Path(blocks_root) if blocks_root is not None else reference_results_root()

    if labels:
        selected = list(labels)
    elif example == "chafee_infante_current":
        manifest_path = blocks_root / example / "attracting_blocks.json"
        manifest = json.loads(manifest_path.read_text())
        selected = sorted(int(label) for label in manifest["minimal_blocks"])
    else:
        selected = minimal_labels(source_root / "MG" / "morse_graph")
    output: dict[str, object] = {
        "example": example,
        "source_root": spec.root,
        "metric": "Euclidean distance in stored latent coordinates",
        "criterion": "R(N) < tau(N)",
        "nodes": {},
    }

    for label in selected:
        boxes, set_kind = load_block_boxes(example, source_root, label, blocks_root)
        block = BlockGeometry(boxes)
        tolerance = tolerance_bounds(
            model,
            block,
            local_boxes=local_boxes,
            sample_target=sample_target,
            sobol_scrambles=sobol_scrambles,
        )
        if (
            set_kind == "saved_recurrent_morse_set_unchecked"
            and tolerance["ibp_complete"]
            and tolerance["numerical_ibp_lower_bound"] > 0.0
        ):
            set_kind = "saved_recurrent_morse_set_numerically_verified_as_attracting_block"
        residual = residual_lower_bound(
            example,
            spec,
            cfg,
            model,
            scaler,
            block,
            label,
        )
        residual_lower = residual["lower_bound"]
        tolerance_upper = tolerance["upper_bound"]
        if residual_lower is not None and residual_lower >= tolerance_upper:
            conclusion = "criterion_contradicted_by_numerical_witnesses"
        else:
            conclusion = "inconclusive"
        output["nodes"][str(label)] = {
            "set_kind": set_kind,
            "n_boxes": int(boxes.shape[0]),
            "tolerance": tolerance,
            "residual": residual,
            "comparison": {
                "residual_lower_over_tolerance_upper": (
                    None
                    if residual_lower is None or tolerance_upper == 0.0
                    else residual_lower / tolerance_upper
                ),
                "conclusion": conclusion,
                "theorem_verified": False,
                "reason_not_verified": (
                    "no uniform residual upper bound"
                    if conclusion == "inconclusive"
                    else "strict theorem inequality is contradicted for the evaluated block"
                ),
            },
        }

    out_dir = output_root / example
    out_dir.mkdir(parents=True, exist_ok=True)
    if labels:
        suffix = "_".join(str(label) for label in sorted(labels))
        path = out_dir / f"tolerance_evaluation_labels_{suffix}.json"
    else:
        path = out_dir / "tolerance_evaluation.json"
    path.write_text(json.dumps(output, indent=2))
    return path
