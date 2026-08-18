"""Appendix residual/tolerance estimates for Chafee--Infante latent d=1 and d=3.

The persisted minimal Morse sets for both dimensions are complete unions of
uniform cells filling one axis-aligned hyperrectangle.  This module validates
that property before using exact hyperrectangle membership and clearance.

The tolerance protocol matches the d=2 computation in
:mod:`.tolerance_protocol`:

* Euclidean distance in the stored latent coordinates;
* at least 2^23 boxwise corner/center/Sobol samples per minimal node;
* local differential-evolution searches in the most promising cells; and
* float64 interval-bound propagation as a non-rigorous enclosure diagnostic.

The residual protocol matches the dense d=2 search in
:mod:`.residual_protocol`:

* all 30,000 persisted one-step pairs;
* 10,216 deterministic Sobol-seeded fresh trajectories in total; and
* four independently perturbed, domain-clipped decoder searches.

The expensive residual batches are independent and may be run concurrently;
:func:`merge_partials` combines them only after every expected artifact is
present.

Inputs are the fetched latent-dimension-study bundle at
``replay_sources/chafee_infante/latent_dimension_study/latent_{1,3}d/seed_0/``
(``models/autoencoder.pt`` + sidecar JSON, ``MG_adaptive/morse_sets``) and the
30,000 one-step training pairs at
``replay_sources/chafee_infante/reference_inputs/train_data.csv``.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from numpy.typing import NDArray
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from ...systems import build_system
from ...training import load_checkpoint
from .tolerance_protocol import REPO_ROOT, default_output_root

STUDY_ROOT = REPO_ROOT / "replay_sources" / "chafee_infante" / "latent_dimension_study"
TRAIN_DATA = REPO_ROOT / "replay_sources" / "chafee_infante" / "reference_inputs" / "train_data.csv"
TRAIN_DATA_SHA256 = "890fea29a2d26b31b44bbc4fbd773af0ebf742b08893c27ddaf1bdbf87e30329"

DIMENSIONS = (1, 3)
LABELS = (0, 1)
PHYSICAL_SIGN = {0: "negative", 1: "positive"}
SAMPLE_TARGET = 2**23
SOBOL_SCRAMBLES = 2
TOLERANCE_SOBOL_SEEDS = (20260725, 20260726)
FRESH_RUNS = (
    (20260727, 1024),
    (20260728, 2048),
    (20260729, 2048),
    (20260730, 2048),
    (20260731, 2048),
)
DECODER_SEEDS = (20260732, 20260733, 20260734, 20260735)
DECODER_TARGET = 2048
DECODER_NOISE_SCALES = (0.0, 1e-4, 1e-3, 1e-2, 5e-2)
FRESH_STEPS = 30
HIGH_DIMENSION = 64
INTEGRATION_BATCH_SIZE = 32
MODEL_BATCH_SIZE = 65536


def default_result_root() -> Path:
    return default_output_root() / "chafee_latent_dimensions"


def _result_root(result_root: Path | None) -> Path:
    return Path(result_root) if result_root is not None else default_result_root()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def dimension_root(dimension: int) -> Path:
    return STUDY_ROOT / f"latent_{dimension}d" / "seed_0"


def tolerance_path(dimension: int, result_root: Path | None = None) -> Path:
    return _result_root(result_root) / f"chafee_infante_d{dimension}" / "tolerance_evaluation.json"


def partial_path(kind: str, suffix: str, result_root: Path | None = None) -> Path:
    return _result_root(result_root) / "residual_partials" / f"{kind}_{suffix}.json"


@dataclass(frozen=True)
class HyperrectangleBlock:
    """A cell decomposition whose union is exactly one hyperrectangle."""

    boxes: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    cell_width: NDArray[np.float64]
    grid_shape: tuple[int, ...]

    @property
    def dimension(self) -> int:
        return int(self.lower.shape[0])

    @classmethod
    def from_boxes(cls, boxes: NDArray[np.float64]) -> HyperrectangleBlock:
        rows = np.asarray(boxes, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] % 2:
            raise ValueError(f"invalid box array shape {rows.shape}")
        dimension = rows.shape[1] // 2
        lower_rows = rows[:, :dimension]
        upper_rows = rows[:, dimension:]
        widths = upper_rows - lower_rows
        if np.any(widths <= 0.0):
            raise ValueError("block contains a non-positive-width cell")
        cell_width = widths[0]
        if not np.allclose(widths, cell_width, rtol=1e-11, atol=1e-13):
            raise ValueError("block cells do not have uniform widths")

        coordinate_values = [np.unique(lower_rows[:, axis]) for axis in range(dimension)]
        grid_shape = tuple(int(values.shape[0]) for values in coordinate_values)
        expected_count = math.prod(grid_shape)
        if expected_count != rows.shape[0]:
            raise ValueError(
                "block is not a complete rectangular cell union: "
                f"{rows.shape[0]} cells versus grid shape {grid_shape}"
            )
        observed = {tuple(row.tolist()) for row in lower_rows}
        expected = {
            tuple(float(value) for value in values) for values in product(*coordinate_values)
        }
        if observed != expected:
            raise ValueError("block has a gap or duplicate in its rectangular grid")
        lower = np.min(lower_rows, axis=0)
        upper = np.max(upper_rows, axis=0)
        expected_upper = lower_rows + cell_width
        if not np.allclose(upper_rows, expected_upper, rtol=1e-11, atol=1e-13):
            raise ValueError("cell upper endpoints do not match the uniform grid")
        union_volume = rows.shape[0] * float(np.prod(cell_width))
        bounding_volume = float(np.prod(upper - lower))
        if not math.isclose(union_volume, bounding_volume, rel_tol=1e-10, abs_tol=1e-13):
            raise ValueError("cell union volume does not equal bounding-box volume")
        return cls(
            boxes=np.ascontiguousarray(rows),
            lower=np.ascontiguousarray(lower),
            upper=np.ascontiguousarray(upper),
            cell_width=np.ascontiguousarray(cell_width),
            grid_shape=grid_shape,
        )

    def membership(
        self,
        values: NDArray[np.float64],
        *,
        interior: bool = False,
    ) -> NDArray[np.bool_]:
        points = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if interior:
            return np.all((self.lower < points) & (points < self.upper), axis=1)
        return np.all((self.lower <= points) & (points <= self.upper), axis=1)

    def clearance(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Euclidean distance to the complement, zero outside the interior."""
        points = np.atleast_2d(np.asarray(values, dtype=np.float64))
        face_clearances = np.minimum(points - self.lower, self.upper - points)
        clearance = np.min(face_clearances, axis=1)
        return np.where(clearance > 0.0, clearance, 0.0)

    def enclosure_clearance(
        self,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        lo = np.atleast_2d(np.asarray(lower, dtype=np.float64))
        hi = np.atleast_2d(np.asarray(upper, dtype=np.float64))
        contained = np.all((self.lower < lo) & (hi < self.upper), axis=1)
        face_clearances = np.minimum(lo - self.lower, self.upper - hi)
        clearance = np.min(face_clearances, axis=1)
        clearance = np.where(contained, clearance, 0.0)
        return clearance, contained

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "validated_complete_uniform_cell_union_hyperrectangle",
            "dimension": self.dimension,
            "cell_count": int(self.boxes.shape[0]),
            "grid_shape": list(self.grid_shape),
            "cell_width": self.cell_width.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
        }


def load_blocks(dimension: int) -> tuple[dict[int, HyperrectangleBlock], Path]:
    source = dimension_root(dimension) / "MG_adaptive" / "morse_sets"
    rows = np.loadtxt(source, delimiter=",", ndmin=2, dtype=np.float64)
    if rows.shape[1] != 2 * dimension + 1:
        raise ValueError(f"{source} has {rows.shape[1]} columns; expected {2 * dimension + 1}")
    labels = np.rint(rows[:, -1]).astype(np.int64)
    if not np.array_equal(labels.astype(np.float64), rows[:, -1]):
        raise ValueError(f"{source} contains nonintegral node labels")
    blocks = {}
    for label in LABELS:
        selected = rows[labels == label, :-1]
        if selected.shape[0] == 0:
            raise ValueError(f"{source} contains no cells for node {label}")
        blocks[label] = HyperrectangleBlock.from_boxes(selected)
    return blocks, source


def load_model(dimension: int) -> torch.nn.Module:
    model, arch = load_checkpoint(dimension_root(dimension) / "models")
    if arch.high_dims != HIGH_DIMENSION or arch.low_dims != dimension:
        raise ValueError(
            f"d={dimension} checkpoint architecture is "
            f"{arch.high_dims}->{arch.low_dims}, expected {HIGH_DIMENSION}->{dimension}"
        )
    model.to(torch.device("cpu"))
    model.eval()
    return model


@torch.no_grad()
def apply_module(
    module: torch.nn.Module,
    values: NDArray[np.float64],
    *,
    chunk: int = MODEL_BATCH_SIZE,
) -> NDArray[np.float32]:
    rows = np.atleast_2d(np.asarray(values))
    output: list[NDArray[np.float32]] = []
    for start in range(0, rows.shape[0], chunk):
        tensor = torch.as_tensor(rows[start : start + chunk], dtype=torch.float32)
        output.append(module(tensor).cpu().numpy())
    return np.vstack(output)


def interval_bound_network(
    network: torch.nn.Module,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    for layer in network.net.children():
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
            raise TypeError(f"unsupported interval layer {type(layer).__name__}")
    return lo, hi


def uniformly_refine_boxes(
    boxes: NDArray[np.float64],
    rounds: int,
) -> NDArray[np.float64]:
    refined = np.asarray(boxes, dtype=np.float64)
    dimension = refined.shape[1] // 2
    for _ in range(rounds):
        lower = refined[:, :dimension]
        upper = refined[:, dimension:]
        middle = (lower + upper) / 2.0
        children = []
        for corner_bits in product((0, 1), repeat=dimension):
            bits = np.asarray(corner_bits, dtype=bool)
            child_lower = np.where(bits, middle, lower)
            child_upper = np.where(bits, upper, middle)
            children.append(np.hstack((child_lower, child_upper)))
        refined = np.vstack(children)
    return refined


def ibp_tolerance_lower_bound(
    model: torch.nn.Module,
    block: HyperrectangleBlock,
    boxes: NDArray[np.float64],
    *,
    refinement_target: float,
) -> dict[str, Any]:
    dimension = block.dimension
    frozen_lower_bound = math.inf
    pending = np.asarray(boxes, dtype=np.float64)
    total_evaluated = 0
    depth_counts: list[dict[str, int]] = []
    max_total_evaluated = 5_000_000
    max_adaptive_rounds = 6
    remaining_failures = pending.shape[0]
    remaining_low_clearance = 0
    complete = False

    for depth in range(max_adaptive_rounds + 1):
        contained_parts = []
        clearance_parts = []
        for start in range(0, pending.shape[0], 4096):
            rows = pending[start : start + 4096]
            out_lo, out_hi = interval_bound_network(
                model.latent_map,
                rows[:, :dimension],
                rows[:, dimension:],
            )
            clearance, contained = block.enclosure_clearance(out_lo, out_hi)
            contained_parts.append(contained)
            clearance_parts.append(clearance)
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
        next_count = int(np.count_nonzero(refine_mask)) * (2**dimension)
        if total_evaluated + next_count > max_total_evaluated:
            break
        pending = uniformly_refine_boxes(pending[refine_mask], 1)

    lower_bound = float(frozen_lower_bound) if complete and np.isfinite(frozen_lower_bound) else 0.0
    return {
        "lower_bound": lower_bound,
        "complete": complete,
        "remaining_failures": remaining_failures,
        "remaining_low_clearance": remaining_low_clearance,
        "refinement_target": refinement_target,
        "total_evaluated": total_evaluated,
        "depth_counts": depth_counts,
    }


def deterministic_box_points(
    block: HyperrectangleBlock,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    dimension = block.dimension
    lower = block.boxes[:, :dimension]
    upper = block.boxes[:, dimension:]
    parts = []
    for corner_bits in product((0, 1), repeat=dimension):
        bits = np.asarray(corner_bits, dtype=bool)
        parts.append(np.where(bits, upper, lower))
    parts.append((lower + upper) / 2.0)
    return (
        np.vstack(parts),
        np.tile(np.arange(block.boxes.shape[0], dtype=np.int64), len(parts)),
    )


def tolerance_sampling_plan(block: HyperrectangleBlock) -> dict[str, Any]:
    deterministic_count = (2**block.dimension + 1) * block.boxes.shape[0]
    requested_sobol = max(0, SAMPLE_TARGET - deterministic_count)
    denominator = SOBOL_SCRAMBLES * block.boxes.shape[0]
    points_per_box_per_scramble = max(1, math.ceil(requested_sobol / denominator))
    sobol_power = math.ceil(math.log2(points_per_box_per_scramble))
    actual = (
        deterministic_count + SOBOL_SCRAMBLES * points_per_box_per_scramble * block.boxes.shape[0]
    )
    return {
        "target_points": SAMPLE_TARGET,
        "actual_points": actual,
        "deterministic_points": deterministic_count,
        "sobol_scrambles": SOBOL_SCRAMBLES,
        "sobol_seeds": list(TOLERANCE_SOBOL_SEEDS),
        "sobol_points_per_box_per_scramble": points_per_box_per_scramble,
        "sobol_generation_power": sobol_power,
        "sobol_generated_then_truncated_per_scramble": 2**sobol_power,
    }


def evaluate_tolerance_samples(
    model: torch.nn.Module,
    block: HyperrectangleBlock,
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    plan = tolerance_sampling_plan(block)
    dimension = block.dimension
    lower = block.boxes[:, :dimension]
    widths = block.boxes[:, dimension:] - lower
    per_box_minimum = np.full(block.boxes.shape[0], math.inf, dtype=np.float64)
    best_value = math.inf
    best_input: NDArray[np.float64] | None = None
    best_image: NDArray[np.float64] | None = None
    best_box = -1
    all_inside = True
    evaluated = 0

    def consume(points: NDArray[np.float64], box_ids: NDArray[np.int64]) -> None:
        nonlocal best_value, best_input, best_image, best_box, all_inside, evaluated
        for start in range(0, points.shape[0], MODEL_BATCH_SIZE):
            rows = points[start : start + MODEL_BATCH_SIZE]
            ids = box_ids[start : start + MODEL_BATCH_SIZE]
            mapped = apply_module(model.latent_map, rows)
            clearance = block.clearance(mapped)
            all_inside = all_inside and bool(np.all(clearance > 0.0))
            evaluated += int(rows.shape[0])
            np.minimum.at(per_box_minimum, ids, clearance)
            local = int(np.argmin(clearance))
            value = float(clearance[local])
            if value < best_value:
                best_value = value
                best_input = rows[local].copy()
                best_image = mapped[local].astype(np.float64, copy=True)
                best_box = int(ids[local])

    deterministic, deterministic_ids = deterministic_box_points(block)
    consume(deterministic, deterministic_ids)
    points_per_box = int(plan["sobol_points_per_box_per_scramble"])
    power = int(plan["sobol_generation_power"])
    unit_chunk = max(1, MODEL_BATCH_SIZE // block.boxes.shape[0])
    for seed in TOLERANCE_SOBOL_SEEDS:
        unit = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(power)
        unit = unit[:points_per_box]
        for start in range(0, unit.shape[0], unit_chunk):
            selected = unit[start : start + unit_chunk]
            points = (lower[None, :, :] + selected[:, None, :] * widths[None, :, :]).reshape(
                -1, dimension
            )
            box_ids = np.tile(
                np.arange(block.boxes.shape[0], dtype=np.int64),
                selected.shape[0],
            )
            consume(points, box_ids)

    if best_input is None or best_image is None:
        raise RuntimeError("tolerance sampling produced no candidate")
    if evaluated != int(plan["actual_points"]):
        raise RuntimeError(
            f"evaluated {evaluated} tolerance samples, expected {plan['actual_points']}"
        )
    return (
        {
            "value": best_value,
            "input": best_input.tolist(),
            "image": best_image.tolist(),
            "box_index": best_box,
            "source": "box_corners_centers_and_sobol",
            "all_sample_images_in_interior": all_inside,
            "sample_count": evaluated,
            "sampling": plan,
        },
        per_box_minimum,
    )


def tolerance_for_node(
    model: torch.nn.Module,
    block: HyperrectangleBlock,
) -> dict[str, Any]:
    best, per_box_minimum = evaluate_tolerance_samples(model, block)

    def objective(z: NDArray[np.float64]) -> float:
        mapped = apply_module(model.latent_map, np.atleast_2d(z), chunk=1)
        return float(block.clearance(mapped)[0])

    promising = np.argsort(per_box_minimum)[: min(12, block.boxes.shape[0])]
    dimension = block.dimension
    for box_index_value in promising:
        box_index = int(box_index_value)
        row = block.boxes[box_index]
        result = differential_evolution(
            objective,
            list(zip(row[:dimension], row[dimension:], strict=False)),
            seed=box_index,
            popsize=10,
            maxiter=80,
            tol=1e-9,
            polish=True,
            workers=1,
            updating="immediate",
        )
        if float(result.fun) < float(best["value"]):
            image = apply_module(
                model.latent_map,
                np.atleast_2d(result.x),
                chunk=1,
            )[0]
            best.update(
                {
                    "value": float(result.fun),
                    "input": result.x.tolist(),
                    "image": image.tolist(),
                    "box_index": box_index,
                    "source": "local_differential_evolution",
                }
            )

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
        "upper_bound": float(best["value"]),
        "upper_bound_witness": {
            key: value
            for key, value in best.items()
            if key not in {"all_sample_images_in_interior", "sample_count", "sampling"}
        },
        "sample_count": int(best["sample_count"]),
        "sampling": best["sampling"],
        "all_sample_images_in_interior": bool(best["all_sample_images_in_interior"]),
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


def run_tolerance(dimension: int, *, result_root: Path | None = None) -> Path:
    """Dense tolerance search for one latent dimension; writes tolerance_evaluation.json."""
    started = time.time()
    model = load_model(dimension)
    blocks, block_source = load_blocks(dimension)
    output: dict[str, Any] = {
        "example": f"chafee_infante_d{dimension}",
        "dimension": dimension,
        "metric": "Euclidean distance in stored latent coordinates",
        "criterion": "R_hat(N) < tau_hat(N)",
        "checkpoint": {
            "path": relative(dimension_root(dimension) / "models" / "autoencoder.pt"),
            "sha256": sha256_file(dimension_root(dimension) / "models" / "autoencoder.pt"),
        },
        "block_source": {
            "path": relative(block_source),
            "sha256": sha256_file(block_source),
        },
        "nodes": {},
    }
    for label in LABELS:
        result = tolerance_for_node(model, blocks[label])
        output["nodes"][str(label)] = {
            "physical_attractor": PHYSICAL_SIGN[label],
            "block_geometry": blocks[label].metadata(),
            "tolerance": result,
        }
        print(
            f"d={dimension} node={label} tau_hat={result['upper_bound']:.12g} "
            f"samples={result['sample_count']} ibp={result['numerical_ibp_lower_bound']:.12g}"
        )
    output["elapsed_seconds"] = time.time() - started
    path = tolerance_path(dimension, result_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    return path


def empty_partial(kind: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "metadata": metadata,
        "dimensions": {
            str(dimension): {
                "nodes": {
                    str(label): {
                        "evaluated_samples": 0,
                        "accepted_samples": 0,
                        "max_euclidean_residual": None,
                        "max_squared_residual_diagnostic": None,
                        "witness": None,
                        "source_summaries": {},
                    }
                    for label in LABELS
                }
            }
            for dimension in DIMENSIONS
        },
    }


def update_residual_dimension(
    *,
    partial: dict[str, Any],
    dimension: int,
    model: torch.nn.Module,
    blocks: dict[int, HyperrectangleBlock],
    raw_x: NDArray[np.float64],
    raw_y: NDArray[np.float64],
    source: str,
    source_offset: int,
) -> None:
    x = np.atleast_2d(np.asarray(raw_x, dtype=np.float64))
    y = np.atleast_2d(np.asarray(raw_y, dtype=np.float64))
    encoded_x = apply_module(model.encoder, x)
    encoded_y = apply_module(model.encoder, y)
    predicted = apply_module(model.latent_map, encoded_x)
    errors = np.linalg.norm(predicted - encoded_y, axis=1)

    for label in LABELS:
        node = partial["dimensions"][str(dimension)]["nodes"][str(label)]
        summary = node["source_summaries"].setdefault(
            source,
            {
                "evaluated_samples": 0,
                "accepted_samples": 0,
                "max_euclidean_residual": None,
                "max_squared_residual_diagnostic": None,
            },
        )
        mask = blocks[label].membership(encoded_x)
        accepted = int(np.count_nonzero(mask))
        evaluated = int(x.shape[0])
        node["evaluated_samples"] += evaluated
        node["accepted_samples"] += accepted
        summary["evaluated_samples"] += evaluated
        summary["accepted_samples"] += accepted
        if accepted == 0:
            continue
        indices = np.flatnonzero(mask)
        local = int(np.argmax(errors[mask]))
        row = int(indices[local])
        value = float(errors[row])
        if summary["max_euclidean_residual"] is None or value > float(
            summary["max_euclidean_residual"]
        ):
            summary["max_euclidean_residual"] = value
            summary["max_squared_residual_diagnostic"] = value**2
        if node["max_euclidean_residual"] is None or value > float(node["max_euclidean_residual"]):
            node["max_euclidean_residual"] = value
            node["max_squared_residual_diagnostic"] = value**2
            node["witness"] = {
                "source": source,
                "row_index_zero_based": source_offset + row,
                "x_raw": x[row].tolist(),
                "E_x": encoded_x[row].tolist(),
                "g_E_x": predicted[row].tolist(),
                "E_f_x": encoded_y[row].tolist(),
            }


def loaded_models_and_blocks() -> tuple[
    dict[int, torch.nn.Module],
    dict[int, dict[int, HyperrectangleBlock]],
]:
    models = {dimension: load_model(dimension) for dimension in DIMENSIONS}
    blocks = {dimension: load_blocks(dimension)[0] for dimension in DIMENSIONS}
    return models, blocks


def run_stored(*, result_root: Path | None = None) -> Path:
    """Residual search over the 30,000 stored one-step pairs."""
    started = time.time()
    models, blocks = loaded_models_and_blocks()
    partial = empty_partial(
        "stored_pairs",
        {
            "source": relative(TRAIN_DATA),
            "sha256": sha256_file(TRAIN_DATA),
            "n_pairs": 30_000,
        },
    )
    data = np.loadtxt(TRAIN_DATA, delimiter=",", dtype=np.float64)
    if data.shape != (30_000, 2 * HIGH_DIMENSION):
        raise ValueError(f"stored pair array has unexpected shape {data.shape}")
    source = relative(TRAIN_DATA)
    for start in range(0, data.shape[0], 4096):
        rows = data[start : start + 4096]
        for dimension in DIMENSIONS:
            update_residual_dimension(
                partial=partial,
                dimension=dimension,
                model=models[dimension],
                blocks=blocks[dimension],
                raw_x=rows[:, :HIGH_DIMENSION],
                raw_y=rows[:, HIGH_DIMENSION:],
                source=source,
                source_offset=start,
            )
    partial["metadata"]["elapsed_seconds"] = time.time() - started
    path = partial_path("stored", "pairs", result_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(partial, indent=2))
    return path


def chafee_system():
    return build_system(
        "chafee_infante",
        {
            "N": 64,
            "alpha": 28.0,
            "tau": 0.1,
            "amplitude": 2.0,
            "decay": 0.5,
        },
    )


def sobol_initials(system, count: int, seed: int) -> NDArray[np.float64]:
    power = math.ceil(math.log2(count))
    unit = qmc.Sobol(d=HIGH_DIMENSION, scramble=True, seed=seed).random_base2(power)
    unit = unit[:count]
    return np.asarray(system.lower_bounds) + unit * (
        np.asarray(system.upper_bounds) - np.asarray(system.lower_bounds)
    )


def run_fresh(seed: int, count: int, *, result_root: Path | None = None) -> Path:
    """Residual search over one deterministic fresh-trajectory batch."""
    started = time.time()
    models, blocks = loaded_models_and_blocks()
    system = chafee_system()
    partial = empty_partial(
        "fresh_trajectories",
        {
            "seed": seed,
            "initial_conditions": count,
            "steps_per_initial_condition": FRESH_STEPS,
            "candidate_transitions": count * FRESH_STEPS,
        },
    )
    points = sobol_initials(system, count, seed)
    source = f"fresh_sobol_trajectories_seed_{seed}"
    source_offset = 0
    for start in range(0, count, INTEGRATION_BATCH_SIZE):
        state = points[start : start + INTEGRATION_BATCH_SIZE]
        for _ in range(FRESH_STEPS):
            next_state = system.step(state)
            for dimension in DIMENSIONS:
                update_residual_dimension(
                    partial=partial,
                    dimension=dimension,
                    model=models[dimension],
                    blocks=blocks[dimension],
                    raw_x=state,
                    raw_y=next_state,
                    source=source,
                    source_offset=source_offset,
                )
            source_offset += int(state.shape[0])
            state = next_state
    partial["metadata"]["elapsed_seconds"] = time.time() - started
    path = partial_path("fresh", f"seed{seed}", result_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(partial, indent=2))
    return path


def decoder_latent_points(
    block: HyperrectangleBlock,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    dimension = block.dimension
    deterministic, _ = deterministic_box_points(block)
    deterministic_count = deterministic.shape[0]
    requested_sobol = max(0, DECODER_TARGET - deterministic_count)
    points_per_box_per_scramble = max(
        1,
        math.ceil(requested_sobol / (SOBOL_SCRAMBLES * block.boxes.shape[0])),
    )
    power = math.ceil(math.log2(points_per_box_per_scramble))
    lower = block.boxes[:, :dimension]
    widths = block.boxes[:, dimension:] - lower
    parts = [deterministic]
    for seed in TOLERANCE_SOBOL_SEEDS:
        unit = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(power)
        unit = unit[:points_per_box_per_scramble]
        parts.append(
            (lower[None, :, :] + unit[:, None, :] * widths[None, :, :]).reshape(-1, dimension)
        )
    points = np.vstack(parts)
    metadata = {
        "target_points": DECODER_TARGET,
        "actual_points": int(points.shape[0]),
        "deterministic_points": deterministic_count,
        "sobol_scrambles": SOBOL_SCRAMBLES,
        "sobol_seeds": list(TOLERANCE_SOBOL_SEEDS),
        "sobol_points_per_box_per_scramble": points_per_box_per_scramble,
        "sobol_generation_power": power,
        "sobol_generated_then_truncated_per_scramble": 2**power,
    }
    return points, metadata


def run_decoder(seed: int, *, result_root: Path | None = None) -> Path:
    """Residual search over one perturbed, domain-clipped decoder batch."""
    started = time.time()
    models, blocks = loaded_models_and_blocks()
    system = chafee_system()
    rng = np.random.default_rng(seed + 100)
    partial = empty_partial(
        "domain_clipped_decoder_search",
        {
            "seed": seed,
            "noise_seed": seed + 100,
            "target_latent_points_per_node": DECODER_TARGET,
            "noise_scales": list(DECODER_NOISE_SCALES),
            "candidate_states_by_dimension": {},
            "per_dimension_node_sampling": {},
        },
    )
    lower = np.asarray(system.lower_bounds)
    upper = np.asarray(system.upper_bounds)
    span = upper - lower
    for dimension in DIMENSIONS:
        candidate_count = 0
        partial["metadata"]["per_dimension_node_sampling"][str(dimension)] = {}
        for source_label in LABELS:
            latent, sampling = decoder_latent_points(blocks[dimension][source_label])
            partial["metadata"]["per_dimension_node_sampling"][str(dimension)][
                str(source_label)
            ] = sampling
            decoded = apply_module(models[dimension].decoder, latent)
            for sigma in DECODER_NOISE_SCALES:
                if sigma == 0.0:
                    candidates = decoded.astype(np.float64)
                else:
                    candidates = decoded + sigma * span * rng.standard_normal(decoded.shape)
                candidates = np.clip(candidates, lower, upper)
                source = (
                    f"decoder_guided_dim_{dimension}_node_{source_label}_"
                    f"noise_{sigma:g}_seed_{seed}"
                )
                source_offset = 0
                for start in range(0, candidates.shape[0], INTEGRATION_BATCH_SIZE):
                    raw_x = candidates[start : start + INTEGRATION_BATCH_SIZE]
                    raw_y = system.step(raw_x)
                    update_residual_dimension(
                        partial=partial,
                        dimension=dimension,
                        model=models[dimension],
                        blocks=blocks[dimension],
                        raw_x=raw_x,
                        raw_y=raw_y,
                        source=source,
                        source_offset=source_offset,
                    )
                    source_offset += int(raw_x.shape[0])
                candidate_count += int(candidates.shape[0])
        partial["metadata"]["candidate_states_by_dimension"][str(dimension)] = candidate_count
    partial["metadata"]["elapsed_seconds"] = time.time() - started
    path = partial_path("decoder", f"seed{seed}", result_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(partial, indent=2))
    return path


def expected_partial_paths(result_root: Path | None = None) -> list[Path]:
    paths = [partial_path("stored", "pairs", result_root)]
    paths.extend(partial_path("fresh", f"seed{seed}", result_root) for seed, _ in FRESH_RUNS)
    paths.extend(partial_path("decoder", f"seed{seed}", result_root) for seed in DECODER_SEEDS)
    return paths


def merge_partials(*, result_root: Path | None = None) -> Path:
    """Combine all residual partials with both tolerance runs into dense_sampling.json."""
    missing = [path for path in expected_partial_paths(result_root) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing residual partials:\n" + "\n".join(str(path) for path in missing)
        )
    partials = [json.loads(path.read_text()) for path in expected_partial_paths(result_root)]
    tolerances = {
        dimension: json.loads(tolerance_path(dimension, result_root).read_text())
        for dimension in DIMENSIONS
    }
    output: dict[str, Any] = {
        "example": "chafee_infante_latent_dimensions",
        "metric": "Euclidean distance in each model's stored latent coordinates",
        "interpretation": (
            "Both quantities are dense sampled estimates. R_hat < tau_hat means "
            "that no sampled violation was found, not that the uniform theorem "
            "hypothesis was proved."
        ),
        "protocol": {
            "stored_pairs": 30_000,
            "fresh_runs": [
                {"seed": seed, "initial_conditions": count, "steps": FRESH_STEPS}
                for seed, count in FRESH_RUNS
            ],
            "fresh_initial_conditions_total": sum(count for _, count in FRESH_RUNS),
            "fresh_candidate_transitions_total": sum(
                count * FRESH_STEPS for _, count in FRESH_RUNS
            ),
            "decoder_seeds": list(DECODER_SEEDS),
            "decoder_noise_scales": list(DECODER_NOISE_SCALES),
            "tolerance_target_samples_per_node": SAMPLE_TARGET,
            "tolerance_sobol_seeds": list(TOLERANCE_SOBOL_SEEDS),
        },
        "provenance": {
            "script": relative(Path(__file__)),
            "script_sha256": sha256_file(Path(__file__)),
            "train_data": {
                "path": relative(TRAIN_DATA),
                "sha256": sha256_file(TRAIN_DATA),
            },
            "software": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
            },
            "partials": [
                {"path": relative(path), "sha256": sha256_file(path)}
                for path in expected_partial_paths(result_root)
            ],
        },
        "dimensions": {},
    }
    for dimension in DIMENSIONS:
        blocks, block_source = load_blocks(dimension)
        dimension_result: dict[str, Any] = {
            "checkpoint": {
                "path": relative(dimension_root(dimension) / "models" / "autoencoder.pt"),
                "sha256": sha256_file(dimension_root(dimension) / "models" / "autoencoder.pt"),
            },
            "block_source": {
                "path": relative(block_source),
                "sha256": sha256_file(block_source),
            },
            "nodes": {},
        }
        for label in LABELS:
            merged = {
                "evaluated_samples": 0,
                "accepted_samples": 0,
                "sampled_maximum": None,
                "squared_value_diagnostic": None,
                "witness": None,
                "source_summaries": {},
            }
            for partial in partials:
                node = partial["dimensions"][str(dimension)]["nodes"][str(label)]
                merged["evaluated_samples"] += int(node["evaluated_samples"])
                merged["accepted_samples"] += int(node["accepted_samples"])
                for source, summary in node["source_summaries"].items():
                    if source in merged["source_summaries"]:
                        raise ValueError(f"duplicate residual source {source}")
                    merged["source_summaries"][source] = summary
                value = node["max_euclidean_residual"]
                if value is not None and (
                    merged["sampled_maximum"] is None
                    or float(value) > float(merged["sampled_maximum"])
                ):
                    merged["sampled_maximum"] = float(value)
                    merged["squared_value_diagnostic"] = float(value) ** 2
                    merged["witness"] = node["witness"]
            tolerance = tolerances[dimension]["nodes"][str(label)]["tolerance"]
            tau = float(tolerance["upper_bound"])
            residual = float(merged["sampled_maximum"])
            ratio = residual / tau if tau else math.inf
            dimension_result["nodes"][str(label)] = {
                "physical_attractor": PHYSICAL_SIGN[label],
                "block_geometry": blocks[label].metadata(),
                "tolerance": {
                    "sampled_minimum": tau,
                    "sample_count": int(tolerance["sample_count"]),
                    "all_sample_images_in_interior": bool(
                        tolerance["all_sample_images_in_interior"]
                    ),
                    "numerical_ibp_lower_bound": float(tolerance["numerical_ibp_lower_bound"]),
                    "ibp_complete": bool(tolerance["ibp_complete"]),
                    "witness": tolerance["upper_bound_witness"],
                },
                "residual": merged,
                "comparison": {
                    "sampled_residual_over_sampled_tolerance": ratio,
                    "sampled_conclusion": (
                        "sampled_violation" if residual >= tau else "no_sampled_violation_found"
                    ),
                },
            }
        output["dimensions"][str(dimension)] = dimension_result
    path = _result_root(result_root) / "dense_sampling.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    for dimension in DIMENSIONS:
        for label in LABELS:
            node = output["dimensions"][str(dimension)]["nodes"][str(label)]
            print(
                f"d={dimension} node={label} accepted="
                f"{node['residual']['accepted_samples']} "
                f"R_hat={node['residual']['sampled_maximum']:.12g} "
                f"tau_hat={node['tolerance']['sampled_minimum']:.12g} "
                f"ratio={node['comparison']['sampled_residual_over_sampled_tolerance']:.6g} "
                f"{node['comparison']['sampled_conclusion']}"
            )
    return path


def validate_inputs() -> None:
    """Check the fetched checkpoints and block artifacts against audited hashes."""
    if sha256_file(TRAIN_DATA) != TRAIN_DATA_SHA256:
        raise ValueError("training-pair hash does not match the audited artifact")
    expected_hashes = {
        1: {
            "checkpoint": "f2d1ad7dcc094e4565f25446e613d4b528261012810bb493ef70d1a3977c0f91",
            "blocks": "ff7b5b704974153e5d2c082c09407d437a488df7f3d5639e5df93816f34e6154",
        },
        3: {
            "checkpoint": "bdb0f8a69fe1358ab3d7f3bb2e69f6e6883f92fe83fe61e365bed3e04e1e2bab",
            "blocks": "14979bd3f3cf526e24a7a486822e0c48328b93bfc57d374cf0709682c2370919",
        },
    }
    for dimension in DIMENSIONS:
        blocks, block_source = load_blocks(dimension)
        checkpoint = dimension_root(dimension) / "models" / "autoencoder.pt"
        if sha256_file(checkpoint) != expected_hashes[dimension]["checkpoint"]:
            raise ValueError(f"d={dimension} checkpoint hash mismatch")
        if sha256_file(block_source) != expected_hashes[dimension]["blocks"]:
            raise ValueError(f"d={dimension} block hash mismatch")
        load_model(dimension)
        for label in LABELS:
            print(
                f"d={dimension} node={label} {PHYSICAL_SIGN[label]} "
                f"{json.dumps(blocks[label].metadata(), sort_keys=True)}"
            )
