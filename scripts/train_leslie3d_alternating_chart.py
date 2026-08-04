#!/usr/bin/env python3
"""Safely refine the Leslie3D latent chart, then repair its latent map.

The accepted invariant-aware v2 checkpoint was trained with a fixed encoder
and decoder.  This opt-in continuation makes a deliberately small chart
update: by default only the encoder's final affine layer and the decoder's
first affine layer are trainable.  Reconstruction, frozen-map consistency,
reference-chart trust, cross-object margins, local-secant anti-folding, and
latent inverse consistency prevent the usual constant/folded chart failure.

Only after the chart candidate passes held-out numerical gates are the chart
parameters frozen again.  Every encoded transition, recurrent anchor, anchor
scale, and trajectory-shadowing target is then recomputed before optimizing
the latent map with the existing exact anchor projection and exact-autograd
periodic-orbit characteristic/topology objectives.

An explicit continuation mode can restore a completed map training-state
bundle.  It preserves hidden-layer Adam/scheduler/RNG state while giving the
final affine layer a separate no-momentum gradient step in the nullspace of
the current anchor-feature design, followed by the same exact projection.

This is a numerical continuation, not a Conley-index certificate.  In
particular, the finite anti-fold bank cannot prove injectivity, and this first
alternating path does not yet feed independently discovered extra recurrent
roots back as moving hard negatives.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from latentdynamics.config import load_config
from latentdynamics.training import load_any_checkpoint, save_checkpoint

if __package__:
    from scripts import train_leslie3d_smooth_topology as topology
else:  # Direct ``python scripts/<name>.py`` execution.
    import train_leslie3d_smooth_topology as topology

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "leslie3d_invariant_aware_v2_alternating"
PRIMARY_V2_SHA256 = "9fbee2cde690d58d2413c0d3521763838abaeb493736120f7173035612da3f3d"
PRESERVED_CHART_REFINED_SHA256 = (
    "7f528f001f66652689aa9b3b31b8c46909084f2c82a31de739a59877679b24fc"
)
PRESERVED_MAP_RESUME_SUMMARY_SHA256 = (
    "773ab21449d2694ee86996661f2dd33be8bf6119fdf44760f1e1282bdcab88e6"
)
# 1.5 times the independently audited accepted/source-chart 320-step loss
# 0.00464711385.  This absolute ceiling prevents a changed chart from making
# promotion easier merely by degrading its own comparison baseline.
SOURCE_ROLLOUT_ABSOLUTE_LIMIT = 0.006970670775
CONTINUATION_BOUNDARY_PREACTIVATION_TOLERANCE = 1.0e-3
PRIMARY_REPLAY_COMPONENT_WEIGHTS = {
    "saddle_tangent_transition_tubes": 4.0,
    "origin_positive_cone_transition_fan": 3.0,
    "audited_origin_p_star_s2_transition_tubes": 8.0,
}
# Keep the previously audited through-64 training targets in the rebuilt cache,
# but the safety retry filters optimization to a <=4 then <=16 curriculum.
# The 31/32 and 63/64 targets remain explicit evaluation-only inventory, and
# promotion independently gates the complete 320-step held-out inventory.
TRAINING_SHADOWING_HORIZON_GROUPS = {
    "short": (1, 2, 3, 4),
    "medium": (7, 8, 15, 16, 31, 32),
    "long": (63, 64),
}


@dataclass(frozen=True)
class SecantBank:
    """Deterministic local pairs and their reference-chart separations."""

    left: Tensor
    right: Tensor
    reference_distance: Tensor


@dataclass(frozen=True)
class LatentCache:
    """All chart-dependent tensors consumed by latent-map optimization."""

    z_train: Tensor
    z_train_next: Tensor
    z_val: Tensor
    z_val_next: Tensor
    targets: dict[str, Tensor]
    scales: dict[str, Tensor]
    train_shadowing_blocks: list[dict[str, Any]]
    val_shadowing_blocks: list[dict[str, Any]]
    shadowing_report: dict[str, Any]


@dataclass(frozen=True)
class MapTrainingSettings:
    """Numerical policy for one fresh latent-map attempt."""

    epochs: int
    learning_rate: float
    rollout_learning_rate: float
    rollout_min_topology_epochs: int
    anchor_weight: float
    characteristic_weight: float
    topology_weight: float
    trust_weight: float
    rollout_weight: float
    rollout_ratio_limit: float
    rollout_absolute_limit: float
    rollout_backprop_steps: int
    rollout_short_epochs: int
    rollout_medium_max_horizon: int
    spectral_start_epoch: int
    spectral_ramp_epochs: int
    per_term_gradient_clip_norm: float
    gradient_diagnostics_every: int
    gradient_diagnostics_threshold: float
    eval_every: int
    stable_ceiling: float
    unstable_floor: float
    jury_buffer: float
    validation_ratio_limit: float
    source_reconstruction_ratio_limit: float
    source_prediction_ratio_limit: float
    anchor_acceptance: float
    characteristic_acceptance: float


@dataclass(frozen=True)
class OutputConstraintGeometry:
    """Tangent geometry for the exact final-affine anchor constraints."""

    output_layer: nn.Linear
    design: Tensor
    desired_preactivation: Tensor
    nullspace_basis: Tensor
    rank: int
    tolerance: float
    largest_singular_value: float
    smallest_retained_singular_value: float | None


def _validate_map_settings(settings: MapTrainingSettings) -> None:
    if (
        settings.epochs < 1
        or settings.eval_every < 1
        or settings.rollout_backprop_steps < 1
        or settings.rollout_short_epochs < 0
        or not 4 <= settings.rollout_medium_max_horizon <= 16
        or settings.rollout_min_topology_epochs < 0
        or settings.spectral_start_epoch < 0
        or settings.spectral_ramp_epochs < 1
        or settings.gradient_diagnostics_every < 1
    ):
        raise ValueError("invalid map epoch, rollout-curriculum, or diagnostic settings")
    if (
        settings.learning_rate <= 0.0
        or settings.rollout_learning_rate <= 0.0
        or settings.per_term_gradient_clip_norm <= 0.0
        or settings.gradient_diagnostics_threshold <= 0.0
    ):
        raise ValueError("map learning rates and gradient limits must be positive")
    if any(
        weight < 0.0
        for weight in (
            settings.anchor_weight,
            settings.characteristic_weight,
            settings.topology_weight,
            settings.trust_weight,
            settings.rollout_weight,
        )
    ):
        raise ValueError("map loss weights must be nonnegative")
    if (
        settings.rollout_ratio_limit <= 0.0
        or settings.rollout_absolute_limit <= 0.0
        or settings.validation_ratio_limit <= 0.0
        or settings.source_reconstruction_ratio_limit <= 0.0
        or settings.source_prediction_ratio_limit <= 0.0
        or settings.anchor_acceptance <= 0.0
        or settings.characteristic_acceptance <= 0.0
        or settings.stable_ceiling <= 0.0
        or settings.unstable_floor <= settings.stable_ceiling
        or settings.jury_buffer < 0.0
    ):
        raise ValueError("map gate limits are inconsistent")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else CODE_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _capture_rng_state() -> dict[str, Any]:
    """Capture every RNG stream used by this deterministic trainer."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a checkpoint boundary without silently reseeding continuation."""

    required = {"python", "numpy", "torch"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"training state is missing RNG streams: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("training state contains CUDA RNG streams but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


def _configure_chart_parameters(model: nn.Module, scope: str = "edge") -> list[str]:
    """Freeze the map and expose a small, auditable E/D parameter subset."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if scope == "edge":
        encoder_layers = _linear_layers(model.encoder)
        decoder_layers = _linear_layers(model.decoder)
        if len(encoder_layers) < 1 or len(decoder_layers) < 1:
            raise ValueError("chart refinement requires affine encoder/decoder layers")
        selected = (encoder_layers[-1], decoder_layers[0])
        for layer in selected:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    elif scope == "all":
        for component in (model.encoder, model.decoder):
            for parameter in component.parameters():
                parameter.requires_grad_(True)
    else:
        raise ValueError("chart scope must be 'edge' or 'all'")
    model.encoder.train()
    model.decoder.train()
    model.latent_map.eval()
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _configure_map_parameters(
    model: nn.Module,
    *,
    constrained_output_update: bool = False,
) -> list[str]:
    """Expose map parameters, optionally including the constrained final affine."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.latent_map.parameters():
        parameter.requires_grad_(True)
    affine_layers = _linear_layers(model.latent_map)
    if not affine_layers:
        raise ValueError("latent-map repair requires a final affine layer")
    if not constrained_output_update:
        for parameter in affine_layers[-1].parameters():
            parameter.requires_grad_(False)
    model.encoder.eval()
    model.decoder.eval()
    model.latent_map.train()
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


@torch.no_grad()
def _output_constraint_geometry(
    latent_map: nn.Module,
    targets: dict[str, Tensor],
) -> OutputConstraintGeometry:
    """Build the exact nullspace of the current anchor-feature design matrix.

    If ``Theta = [W.T; b]`` is the final affine parameter matrix, the anchor
    constraints are ``F Theta = atanh(successor)``.  Therefore every feasible
    first-order output-only update has the form ``delta = N q``, where the
    columns of ``N`` span ``null(F)``.  The SVD is tiny (16 by 65 in this
    experiment), so recomputing it avoids a stale tangent space after hidden
    features change.
    """

    children = list(latent_map.net.children())
    if len(children) < 2 or not isinstance(children[-2], nn.Linear):
        raise TypeError("expected the penultimate latent-map module to be Linear")
    if not isinstance(children[-1], nn.Tanh):
        raise TypeError("constrained output update currently requires a Tanh output")
    output_layer = children[-2]
    points = torch.cat([targets[name] for name in topology.OBJECT_ORDER])
    successors = torch.cat(
        [
            torch.roll(targets[name], shifts=-1, dims=0)
            for name in topology.OBJECT_ORDER
        ]
    )
    features = points
    for layer in children[:-2]:
        features = layer(features)
    design = torch.cat(
        [features, torch.ones((len(features), 1), device=features.device)],
        dim=1,
    ).double()
    desired = torch.atanh(torch.clamp(successors, -0.999999, 0.999999)).double()
    _left, singular_values, right_transpose = torch.linalg.svd(
        design,
        full_matrices=True,
    )
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    tolerance = max(design.shape) * torch.finfo(design.dtype).eps * largest
    rank = int(torch.count_nonzero(singular_values > tolerance))
    nullspace = right_transpose[rank:].T.contiguous()
    smallest_retained = (
        float(singular_values[rank - 1]) if rank > 0 else None
    )
    return OutputConstraintGeometry(
        output_layer=output_layer,
        design=design,
        desired_preactivation=desired,
        nullspace_basis=nullspace,
        rank=rank,
        tolerance=tolerance,
        largest_singular_value=largest,
        smallest_retained_singular_value=smallest_retained,
    )


@torch.no_grad()
def _anchor_preactivation_report(
    geometry: OutputConstraintGeometry,
) -> dict[str, float | int | None]:
    """Measure the installed float parameters against ``F Theta = y``."""

    layer = geometry.output_layer
    parameters = torch.cat(
        [layer.weight.T, layer.bias.unsqueeze(0)],
        dim=0,
    ).double()
    residual = geometry.design @ parameters - geometry.desired_preactivation
    return {
        "preactivation_residual_max_abs": float(torch.max(torch.abs(residual))),
        "preactivation_residual_l2": float(torch.linalg.vector_norm(residual)),
        "design_rank": geometry.rank,
        "design_nullity": int(geometry.design.shape[1] - geometry.rank),
        "design_largest_singular_value": geometry.largest_singular_value,
        "design_smallest_retained_singular_value": (
            geometry.smallest_retained_singular_value
        ),
        "svd_rank_tolerance": geometry.tolerance,
    }


def _validate_continuation_boundary(
    geometry: OutputConstraintGeometry,
    report: dict[str, float | int | None],
    training_state: dict[str, Any],
    *,
    tolerance: float = CONTINUATION_BOUNDARY_PREACTIVATION_TOLERANCE,
) -> None:
    """Reject a state that is not a feasible, full-row-rank anchor boundary."""

    if tolerance <= 0.0:
        raise ValueError("continuation-boundary tolerance must be positive")
    row_count = int(geometry.design.shape[0])
    if geometry.rank != row_count:
        raise ValueError(
            "continuation anchor design is rank deficient: "
            f"rank {geometry.rank}, required {row_count}"
        )
    numeric_values = (
        report["preactivation_residual_max_abs"],
        report["preactivation_residual_l2"],
        report["design_largest_singular_value"],
        report["design_smallest_retained_singular_value"],
    )
    if any(value is None or not math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("continuation anchor geometry contains non-finite diagnostics")
    residual = float(report["preactivation_residual_max_abs"])
    if residual > tolerance:
        raise ValueError(
            "continuation boundary violates anchor preactivation constraints: "
            f"max residual {residual:.6g} exceeds {tolerance:.6g}"
        )
    saved_projection = training_state.get("last_anchor_projection")
    if not isinstance(saved_projection, dict):
        raise ValueError("training state lacks its last exact-anchor projection report")
    saved_residual = saved_projection.get("preactivation_residual_after_max_abs")
    if saved_residual is None or not math.isfinite(float(saved_residual)):
        raise ValueError("saved exact-anchor projection residual is missing or non-finite")
    if float(saved_residual) > tolerance:
        raise ValueError(
            "saved exact-anchor projection exceeded the continuation tolerance: "
            f"{float(saved_residual):.6g} > {tolerance:.6g}"
        )


def _resolve_continuation_output_learning_rate(
    training_state: dict[str, Any],
    requested_learning_rate: float | None,
) -> tuple[float | None, str]:
    """Inherit a format-v2 output-step policy unless the caller overrides it."""

    if requested_learning_rate is not None:
        if requested_learning_rate <= 0.0:
            raise ValueError("output-layer learning rate must be positive")
        return requested_learning_rate, "explicit_cli_override"
    if int(training_state.get("format_version", 1)) < 2:
        return None, "format_v1_default_follow_hidden_adam"
    continuation = training_state.get("continuation")
    if not isinstance(continuation, dict):
        raise ValueError("format-v2 state lacks continuation optimizer policy")
    policy = continuation.get("output_learning_rate_policy")
    if policy == "follow_restored_hidden_adam_learning_rate":
        return None, "inherited_follow_hidden_adam"
    if policy == "fixed_explicit":
        fixed = continuation.get("fixed_output_learning_rate")
        if fixed is None or float(fixed) <= 0.0:
            raise ValueError("format-v2 state has an invalid fixed output learning rate")
        return float(fixed), "inherited_fixed_from_format_v2_state"
    raise ValueError(f"unsupported format-v2 output learning-rate policy {policy!r}")


def _project_final_affine_gradients(
    named_parameters: list[tuple[str, nn.Parameter]],
    gradients: list[Tensor | None],
    geometry: OutputConstraintGeometry,
) -> tuple[list[Tensor | None], dict[str, Any]]:
    """Orthogonally project final-affine gradients into ``null(F)``."""

    weight_index = next(
        (
            index
            for index, (_name, parameter) in enumerate(named_parameters)
            if parameter is geometry.output_layer.weight
        ),
        None,
    )
    bias_index = next(
        (
            index
            for index, (_name, parameter) in enumerate(named_parameters)
            if parameter is geometry.output_layer.bias
        ),
        None,
    )
    if weight_index is None or bias_index is None:
        raise ValueError("constrained final-affine parameters are absent from gradients")
    weight_gradient = gradients[weight_index]
    bias_gradient = gradients[bias_index]
    if weight_gradient is None:
        weight_gradient = torch.zeros_like(geometry.output_layer.weight)
    if bias_gradient is None:
        bias_gradient = torch.zeros_like(geometry.output_layer.bias)
    raw = torch.cat(
        [weight_gradient.T, bias_gradient.unsqueeze(0)],
        dim=0,
    ).double()
    basis = geometry.nullspace_basis
    projected = basis @ (basis.T @ raw)
    constraint_gradient = geometry.design @ projected
    updated = list(gradients)
    updated[weight_index] = projected[:-1].T.to(weight_gradient.dtype)
    updated[bias_index] = projected[-1].to(bias_gradient.dtype)
    raw_norm = float(torch.linalg.vector_norm(raw))
    projected_norm = float(torch.linalg.vector_norm(projected))
    return updated, {
        "raw_gradient_l2": raw_norm,
        "projected_gradient_l2": projected_norm,
        "removed_constraint_normal_l2": float(torch.linalg.vector_norm(raw - projected)),
        "retained_gradient_fraction": projected_norm / max(raw_norm, 1e-300),
        "F_projected_gradient_max_abs": float(
            torch.max(torch.abs(constraint_gradient))
        ),
        "F_projected_gradient_l2": float(
            torch.linalg.vector_norm(constraint_gradient)
        ),
    }


@torch.no_grad()
def _apply_constrained_output_sgd(
    geometry: OutputConstraintGeometry,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    """Apply one no-momentum tangent SGD step and report ``F * delta``."""

    if learning_rate <= 0.0:
        raise ValueError("constrained output learning rate must be positive")
    layer = geometry.output_layer
    if layer.weight.grad is None or layer.bias.grad is None:
        raise ValueError("constrained output gradients are missing")
    gradient = torch.cat(
        [layer.weight.grad.T, layer.bias.grad.unsqueeze(0)],
        dim=0,
    ).double()
    # Reprojection is intentionally repeated after per-term accumulation and
    # global clipping.  This only removes floating-point leakage.
    basis = geometry.nullspace_basis
    tangent_gradient = basis @ (basis.T @ gradient)
    requested_delta = -learning_rate * tangent_gradient
    before = torch.cat(
        [layer.weight.T, layer.bias.unsqueeze(0)],
        dim=0,
    ).double().clone()
    layer.weight.add_(requested_delta[:-1].T.to(layer.weight.dtype))
    layer.bias.add_(requested_delta[-1].to(layer.bias.dtype))
    after = torch.cat(
        [layer.weight.T, layer.bias.unsqueeze(0)],
        dim=0,
    ).double()
    applied_delta = after - before
    ideal_constraint_delta = geometry.design @ requested_delta
    applied_constraint_delta = geometry.design @ applied_delta
    return {
        "method": "explicit_nullspace_projected_sgd_without_momentum",
        "learning_rate": learning_rate,
        "design_rank": geometry.rank,
        "design_nullity": int(geometry.design.shape[1] - geometry.rank),
        "requested_delta_l2": float(torch.linalg.vector_norm(requested_delta)),
        "applied_float_parameter_delta_l2": float(
            torch.linalg.vector_norm(applied_delta)
        ),
        "ideal_F_delta_max_abs": float(
            torch.max(torch.abs(ideal_constraint_delta))
        ),
        "ideal_F_delta_l2": float(torch.linalg.vector_norm(ideal_constraint_delta)),
        "F_delta_max_abs": float(
            torch.max(torch.abs(applied_constraint_delta))
        ),
        "F_delta_l2": float(torch.linalg.vector_norm(applied_constraint_delta)),
        "tangent_gradient_l2": float(torch.linalg.vector_norm(tangent_gradient)),
    }


def _known_object_batch(
    scaler: Any,
    manifest: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, tuple[str, ...]]:
    points: list[np.ndarray] = []
    labels: list[str] = []
    for name in topology.OBJECT_ORDER:
        values = np.asarray(manifest["known_objects"][name]["points"], dtype=np.float64)
        points.append(values)
        labels.extend([name] * len(values))
    scaled = torch.tensor(scaler.transform(np.vstack(points)), dtype=torch.float32, device=device)
    return scaled, tuple(labels)


def _cross_role_margin_loss(
    current: Tensor,
    reference: Tensor,
    labels: tuple[str, ...],
    *,
    expansion: float,
    minimum_margin: float,
) -> Tensor:
    if expansion < 1.0 or minimum_margin <= 0.0:
        raise ValueError("margin expansion must be >= 1 and minimum margin positive")
    pairs = [
        (left, right)
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
        if labels[left] != labels[right]
    ]
    if not pairs:
        return torch.zeros((), dtype=current.dtype, device=current.device)
    left = torch.tensor([pair[0] for pair in pairs], device=current.device)
    right = torch.tensor([pair[1] for pair in pairs], device=current.device)
    current_distance = torch.linalg.vector_norm(current[left] - current[right], dim=1)
    with torch.no_grad():
        reference_distance = torch.linalg.vector_norm(reference[left] - reference[right], dim=1)
        target = torch.maximum(
            expansion * reference_distance,
            torch.full_like(reference_distance, minimum_margin),
        )
    return torch.mean((torch.relu(target - current_distance) / target) ** 2)


def _make_local_secant_bank(
    physical: Tensor,
    reference_latent: Tensor,
    *,
    sample_count: int,
    neighbors: int,
) -> SecantBank:
    """Select deterministic physical nearest-neighbor secants.

    The bank is intentionally finite and is only an anti-collapse diagnostic;
    it is not treated as an injectivity proof.
    """

    if sample_count < 2 or neighbors < 1:
        raise ValueError("secant sample count must be >=2 and neighbors positive")
    count = min(int(sample_count), len(physical))
    indices = (
        torch.linspace(0, len(physical) - 1, steps=count, device=physical.device).round().long()
    )
    sampled = physical[indices]
    distances = torch.cdist(sampled, sampled)
    distances.fill_diagonal_(float("inf"))
    k = min(int(neighbors), count - 1)
    local = torch.topk(distances, k=k, largest=False).indices
    left = indices[:, None].expand(-1, k).reshape(-1)
    right = indices[local.reshape(-1)]
    canonical_left = torch.minimum(left, right)
    right = torch.maximum(left, right)
    left = canonical_left
    if len(left) == 0:
        raise ValueError("local secant construction produced no distinct pairs")
    pair_codes = left * len(physical) + right
    sorted_codes, order = torch.sort(pair_codes)
    keep_unique = torch.ones_like(sorted_codes, dtype=torch.bool)
    keep_unique[1:] = sorted_codes[1:] != sorted_codes[:-1]
    order = order[keep_unique]
    left = left[order]
    right = right[order]
    reference_distance = torch.linalg.vector_norm(
        reference_latent[left] - reference_latent[right], dim=1
    )
    nondegenerate = reference_distance > 1e-7
    if not bool(torch.any(nondegenerate)):
        raise ValueError("reference chart collapses every selected local secant")
    return SecantBank(
        left=left[nondegenerate].detach(),
        right=right[nondegenerate].detach(),
        reference_distance=reference_distance[nondegenerate].detach(),
    )


def _anti_fold_loss(
    encoded: Tensor,
    bank: SecantBank,
    *,
    retained_fraction: float,
) -> Tensor:
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("anti-fold retained fraction must lie in (0,1]")
    current = torch.linalg.vector_norm(encoded[bank.left] - encoded[bank.right], dim=1)
    target = retained_fraction * bank.reference_distance
    return torch.mean((torch.relu(target - current) / bank.reference_distance) ** 2)


def _chart_terms(
    model: nn.Module,
    reference: nn.Module,
    x: Tensor,
    y: Tensor,
    reference_x: Tensor,
    reference_y: Tensor,
    known_physical: Tensor,
    known_reference: Tensor,
    known_labels: tuple[str, ...],
    secants: SecantBank,
    sample_weights: Tensor | None,
    *,
    margin_expansion: float,
    minimum_margin: float,
    anti_fold_fraction: float,
) -> dict[str, Tensor]:
    z = model.encoder(x)
    z_next = model.encoder(y)
    x_hat = model.decoder(z)
    y_hat = model.decoder(z_next)
    # Reference-map parameters are frozen, but gradients must still flow
    # through its input to the trainable encoder.
    frozen_map_value = reference.latent_map(z)
    predicted_y = model.decoder(frozen_map_value)
    known = model.encoder(known_physical)
    inverse_x = model.encoder(model.decoder(reference_x))
    inverse_y = model.encoder(model.decoder(reference_y))

    def mse(predicted: Tensor, expected: Tensor) -> Tensor:
        if sample_weights is None:
            return nn.functional.mse_loss(predicted, expected)
        per_row = torch.mean((predicted - expected) ** 2, dim=1)
        return torch.sum(sample_weights * per_row) / torch.sum(sample_weights)

    return {
        "reconstruction": 0.5 * (mse(x_hat, x) + mse(y_hat, y)),
        "prediction": mse(predicted_y, y),
        "semiconjugacy": mse(frozen_map_value, z_next),
        "reference": 0.5 * (mse(z, reference_x) + mse(z_next, reference_y)),
        "margin": _cross_role_margin_loss(
            known,
            known_reference,
            known_labels,
            expansion=margin_expansion,
            minimum_margin=minimum_margin,
        ),
        "anti_fold": _anti_fold_loss(z, secants, retained_fraction=anti_fold_fraction),
        "inverse": 0.5 * (mse(inverse_x, reference_x) + mse(inverse_y, reference_y)),
    }


def _weighted_chart_score(terms: dict[str, Tensor], weights: dict[str, float]) -> Tensor:
    return sum(weights[name] * value for name, value in terms.items())


def _cross_role_statistics(
    current: Tensor,
    reference: Tensor,
    labels: tuple[str, ...],
) -> dict[str, float]:
    ratios: list[Tensor] = []
    distances: list[Tensor] = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            if labels[left] == labels[right]:
                continue
            current_distance = torch.linalg.vector_norm(current[left] - current[right])
            reference_distance = torch.linalg.vector_norm(reference[left] - reference[right])
            distances.append(current_distance)
            ratios.append(current_distance / torch.clamp(reference_distance, min=1e-8))
    return {
        "minimum_distance": float(torch.min(torch.stack(distances)).detach()),
        "minimum_ratio_to_reference": float(torch.min(torch.stack(ratios)).detach()),
    }


@torch.no_grad()
def _evaluate_chart(
    model: nn.Module,
    reference: nn.Module,
    x: Tensor,
    y: Tensor,
    reference_x: Tensor,
    reference_y: Tensor,
    known_physical: Tensor,
    known_reference: Tensor,
    known_labels: tuple[str, ...],
    secants: SecantBank,
    weights: dict[str, float],
    sample_weights: Tensor | None,
    *,
    margin_expansion: float,
    minimum_margin: float,
    anti_fold_fraction: float,
) -> dict[str, Any]:
    model.eval()
    terms = _chart_terms(
        model,
        reference,
        x,
        y,
        reference_x,
        reference_y,
        known_physical,
        known_reference,
        known_labels,
        secants,
        sample_weights,
        margin_expansion=margin_expansion,
        minimum_margin=minimum_margin,
        anti_fold_fraction=anti_fold_fraction,
    )
    encoded_x = model.encoder(x)
    encoded_y = model.encoder(y)
    known = model.encoder(known_physical)
    secant_distance = torch.linalg.vector_norm(
        encoded_x[secants.left] - encoded_x[secants.right], dim=1
    )
    secant_ratio = secant_distance / secants.reference_distance
    combined = torch.cat((encoded_x, encoded_y))
    return {
        "terms": {name: float(value) for name, value in terms.items()},
        "score": float(_weighted_chart_score(terms, weights)),
        "encoder_drift_rmse": float(
            torch.sqrt(
                0.5
                * (
                    nn.functional.mse_loss(encoded_x, reference_x)
                    + nn.functional.mse_loss(encoded_y, reference_y)
                )
            )
        ),
        "cross_role": _cross_role_statistics(known, known_reference, known_labels),
        "local_secant_ratio": {
            "minimum": float(torch.min(secant_ratio)),
            "p01": float(torch.quantile(secant_ratio, 0.01)),
            "median": float(torch.median(secant_ratio)),
        },
        "encoded_bounds": {
            "lower": torch.min(combined, dim=0).values.cpu().tolist(),
            "upper": torch.max(combined, dim=0).values.cpu().tolist(),
        },
        "maximum_absolute_latent_coordinate": float(torch.max(torch.abs(combined))),
    }


def _chart_gate_report(
    result: dict[str, Any],
    baseline: dict[str, Any],
    *,
    reconstruction_ratio_limit: float,
    prediction_ratio_limit: float,
    semiconjugacy_ratio_limit: float,
    drift_limit: float,
    cross_role_ratio_floor: float,
    secant_p01_ratio_floor: float,
) -> dict[str, Any]:
    reconstruction_ratio = result["terms"]["reconstruction"] / max(
        baseline["terms"]["reconstruction"], 1e-15
    )
    prediction_ratio = result["terms"]["prediction"] / max(baseline["terms"]["prediction"], 1e-15)
    semiconjugacy_ratio = result["terms"]["semiconjugacy"] / max(
        baseline["terms"]["semiconjugacy"], 1e-15
    )
    finite = all(
        math.isfinite(value)
        for value in (
            result["score"],
            reconstruction_ratio,
            prediction_ratio,
            semiconjugacy_ratio,
            result["encoder_drift_rmse"],
            result["cross_role"]["minimum_ratio_to_reference"],
            result["local_secant_ratio"]["p01"],
        )
    )
    gates = {
        "finite": finite,
        "heldout_reconstruction": reconstruction_ratio <= reconstruction_ratio_limit,
        "heldout_prediction": prediction_ratio <= prediction_ratio_limit,
        "heldout_semiconjugacy": semiconjugacy_ratio <= semiconjugacy_ratio_limit,
        "reference_chart_trust": result["encoder_drift_rmse"] <= drift_limit,
        "cross_role_separation": (
            result["cross_role"]["minimum_ratio_to_reference"] >= cross_role_ratio_floor
        ),
        "local_secant_anti_fold": (result["local_secant_ratio"]["p01"] >= secant_p01_ratio_floor),
    }
    violations = {
        "heldout_reconstruction": max(0.0, reconstruction_ratio / reconstruction_ratio_limit - 1.0),
        "heldout_prediction": max(0.0, prediction_ratio / prediction_ratio_limit - 1.0),
        "heldout_semiconjugacy": max(0.0, semiconjugacy_ratio / semiconjugacy_ratio_limit - 1.0),
        "reference_chart_trust": max(0.0, result["encoder_drift_rmse"] / drift_limit - 1.0),
        "cross_role_separation": max(
            0.0,
            cross_role_ratio_floor / max(result["cross_role"]["minimum_ratio_to_reference"], 1e-15)
            - 1.0,
        ),
        "local_secant_anti_fold": max(
            0.0,
            secant_p01_ratio_floor / max(result["local_secant_ratio"]["p01"], 1e-15) - 1.0,
        ),
    }
    return {
        "accepted": all(gates.values()),
        "gates": gates,
        "violations": violations,
        "reconstruction_ratio": reconstruction_ratio,
        "prediction_ratio": prediction_ratio,
        "semiconjugacy_ratio": semiconjugacy_ratio,
    }


def _chart_rank(result: dict[str, Any]) -> tuple[float, float, float, float]:
    report = result["gates"]
    violations = report["violations"]
    failed = sum(not passed for passed in report["gates"].values())
    return (
        float(failed),
        max(violations.values(), default=0.0),
        sum(violations.values()),
        result["score"],
    )


def _rebuild_latent_cache(
    model: nn.Module,
    scaler: Any,
    manifest: dict[str, Any],
    train_metadata: dict[str, Any],
    val_metadata: dict[str, Any],
    x_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    y_val: Tensor,
) -> LatentCache:
    """Recompute every chart-dependent target after an E/D update."""

    device = x_train.device
    model.encoder.eval()
    with torch.no_grad():
        z_train = model.encoder(x_train).detach()
        z_train_next = model.encoder(y_train).detach()
        z_val = model.encoder(x_val).detach()
        z_val_next = model.encoder(y_val).detach()
    targets = topology._phase_latents(model, scaler, manifest, device)
    scales = topology._phase_scales(targets)
    training_groups = topology._validate_trajectory_shadowing_horizon_groups(
        TRAINING_SHADOWING_HORIZON_GROUPS
    )
    validation_groups = topology._validate_trajectory_shadowing_horizon_groups(
        topology.DEFAULT_TRAJECTORY_SHADOWING_HORIZON_GROUPS
    )
    train_blocks, train_report = topology._prepare_trajectory_shadowing_blocks(
        x_train,
        y_train,
        z_train,
        z_train_next,
        train_metadata,
        training_groups,
        split_name="train",
    )
    val_blocks, val_report = topology._prepare_trajectory_shadowing_blocks(
        x_val,
        y_val,
        z_val,
        z_val_next,
        val_metadata,
        validation_groups,
        split_name="validation",
    )
    return LatentCache(
        z_train=z_train,
        z_train_next=z_train_next,
        z_val=z_val,
        z_val_next=z_val_next,
        targets=targets,
        scales=scales,
        train_shadowing_blocks=train_blocks,
        val_shadowing_blocks=val_blocks,
        shadowing_report={"train": train_report, "validation": val_report},
    )


def _cache_report(cache: LatentCache) -> dict[str, Any]:
    return {
        "train_transition_count": len(cache.z_train),
        "validation_transition_count": len(cache.z_val),
        "recurrent_phase_count": sum(len(points) for points in cache.targets.values()),
        "object_phase_counts": {name: len(cache.targets[name]) for name in topology.OBJECT_ORDER},
        "anchor_scales": {
            name: cache.scales[name].detach().cpu().tolist() for name in topology.OBJECT_ORDER
        },
        "trajectory_shadowing": cache.shadowing_report,
        "semantics": (
            "all transition latents, invariant anchors, scales, and rollout targets "
            "were recomputed after the selected chart state and detached"
        ),
        "rollout_horizon_policy": (
            "cache targets through step 64 for auditability; map optimization uses a "
            "short-to-medium curriculum capped at step 16, while promotion independently "
            "evaluates held-out targets through step 320"
        ),
    }


def _limit_shadowing_horizons(
    blocks: list[dict[str, Any]],
    maximum_horizon: int,
) -> list[dict[str, Any]]:
    """Return shallow block copies containing only horizons at or below a cap."""

    if maximum_horizon < 1:
        raise ValueError("maximum rollout horizon must be positive")
    limited: list[dict[str, Any]] = []
    for block in blocks:
        groups = {
            name: tuple(horizon for horizon in horizons if horizon <= maximum_horizon)
            for name, horizons in block["horizon_groups"].items()
        }
        groups = {name: horizons for name, horizons in groups.items() if horizons}
        if groups:
            limited.append({**block, "horizon_groups": groups})
    if not limited:
        raise ValueError(f"no trajectory-shadowing targets at horizon <= {maximum_horizon}")
    return limited


def _trajectory_shadowing_loss_truncated(
    latent_map: nn.Module,
    blocks: list[dict[str, Any]],
    *,
    backprop_steps: int,
    diagnostics: bool,
) -> tuple[Tensor, dict[str, Any]]:
    """Evaluate exact forward rollouts with bounded reverse-mode segments.

    Detaching a prediction after every ``backprop_steps`` forward applications
    does not change any predicted value or loss.  It only prevents unstable
    64-step Jacobian products from dominating all local replay/topology
    gradients.  Full, untruncated forward losses remain the promotion gates.
    """

    if backprop_steps < 1:
        raise ValueError("rollout backpropagation steps must be positive")
    if not blocks:
        raise ValueError("trajectory-shadowing loss requires at least one block")
    component_losses: list[Tensor] = []
    component_report: dict[str, Any] = {}
    for block in blocks:
        latent_states = block["latent_states"]
        horizon_groups = block["horizon_groups"]
        selected_horizons = {
            horizon for horizons in horizon_groups.values() for horizon in horizons
        }
        maximum_horizon = max(selected_horizons)
        prediction = latent_states[:-1]
        losses_by_horizon: dict[int, Tensor] = {}
        for horizon in range(1, maximum_horizon + 1):
            prediction = latent_map(prediction)
            if horizon in selected_horizons:
                losses_by_horizon[horizon] = nn.functional.mse_loss(
                    prediction, latent_states[horizon:]
                )
            if horizon < maximum_horizon:
                prediction = prediction[:-1]
                if horizon % backprop_steps == 0:
                    prediction = prediction.detach()
        group_losses = {
            group_name: torch.stack(
                [losses_by_horizon[horizon] for horizon in horizons]
            ).mean()
            for group_name, horizons in horizon_groups.items()
        }
        component_loss = torch.stack(list(group_losses.values())).mean()
        component_losses.append(component_loss)
        if diagnostics:
            component_report[block["name"]] = {
                "balanced_mse": float(component_loss.detach()),
                "group_mse": {
                    name: float(loss.detach()) for name, loss in group_losses.items()
                },
            }
    total = torch.stack(component_losses).mean()
    report: dict[str, Any] = {}
    if diagnostics:
        report = {
            "balanced_mse": float(total.detach()),
            "backpropagation_segment_steps": backprop_steps,
            "forward_values_are_unmodified": True,
            "components": component_report,
        }
    return total, report


def _gradient_tensor_statistics(
    named_gradients: list[tuple[str, Tensor | None]],
) -> dict[str, Any]:
    """Compute a global norm with float64 accumulation.

    Map gradients are float32, so the largest possible squared finite element
    remains representable in float64.  This avoids the false ``inf`` reported
    by the standard float32 norm reduction at the failed epoch 58 attempt.
    """

    present = [(name, gradient.detach()) for name, gradient in named_gradients if gradient is not None]
    if not present:
        return {
            "finite": True,
            "total_l2": 0.0,
            "maximum_absolute_element": 0.0,
            "nonfinite_parameters": [],
            "norm_accumulator_dtype": "float64",
        }
    nonfinite = [
        name for name, gradient in present if not bool(torch.all(torch.isfinite(gradient)))
    ]
    if nonfinite:
        finite_maxima = [
            float(torch.max(torch.abs(gradient[torch.isfinite(gradient)])))
            for _, gradient in present
            if bool(torch.any(torch.isfinite(gradient)))
        ]
        return {
            "finite": False,
            "total_l2": None,
            "maximum_absolute_element": max(finite_maxima, default=None),
            "nonfinite_parameters": nonfinite,
            "norm_accumulator_dtype": "float64",
        }
    maximum = max(float(torch.max(torch.abs(gradient))) for _, gradient in present)
    square_sum = sum(float(torch.sum(gradient.to(torch.float64) ** 2)) for _, gradient in present)
    total = math.sqrt(square_sum)
    return {
        "finite": True,
        "total_l2": total,
        "maximum_absolute_element": maximum,
        "nonfinite_parameters": [],
        "norm_accumulator_dtype": "float64",
    }


def _safe_clip_grad_norm_(
    named_parameters: list[tuple[str, nn.Parameter]],
    max_norm: float,
) -> dict[str, Any]:
    """Clip finite gradients using a scale-safe norm and return diagnostics."""

    if max_norm <= 0.0:
        raise ValueError("gradient clip norm must be positive")
    statistics = _gradient_tensor_statistics(
        [(name, parameter.grad) for name, parameter in named_parameters]
    )
    if not statistics["finite"]:
        raise FloatingPointError(
            "non-finite gradient elements in parameters "
            f"{statistics['nonfinite_parameters']}"
        )
    total = float(statistics["total_l2"])
    coefficient = min(1.0, max_norm / (total + 1e-12))
    if coefficient < 1.0:
        for _, parameter in named_parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    return {**statistics, "clip_coefficient": coefficient, "max_norm": max_norm}


def _term_gradient_diagnostics(
    terms: dict[str, Tensor],
    named_parameters: list[tuple[str, nn.Parameter]],
) -> dict[str, Any]:
    """Report each scalar term's independent gradient without mutating ``.grad``."""

    parameters = [parameter for _, parameter in named_parameters]
    report: dict[str, Any] = {}
    for name, term in terms.items():
        if not term.requires_grad:
            report[name] = {
                "loss": float(term.detach()),
                "finite": True,
                "total_l2": 0.0,
                "maximum_absolute_element": 0.0,
                "nonfinite_parameters": [],
            }
            continue
        gradients = torch.autograd.grad(
            term,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        report[name] = {
            "loss": float(term.detach()),
            **_gradient_tensor_statistics(
                [
                    (parameter_name, gradient)
                    for (parameter_name, _), gradient in zip(
                        named_parameters, gradients, strict=True
                    )
                ]
            ),
        }
    return report


def _backward_terms_with_caps(
    terms: dict[str, Tensor],
    named_parameters: list[tuple[str, nn.Parameter]],
    *,
    per_term_max_norm: float,
    output_constraint_geometry: OutputConstraintGeometry | None = None,
) -> dict[str, Any]:
    """Accumulate independently clipped term gradients into ``parameter.grad``.

    Each configured loss weight is applied before this function.  The cap is a
    guardrail against one unstable objective erasing every other direction;
    a final global clip is still applied to the accumulated gradient.
    """

    if per_term_max_norm <= 0.0:
        raise ValueError("per-term gradient norm cap must be positive")
    parameters = [parameter for _, parameter in named_parameters]
    for parameter in parameters:
        parameter.grad = None
    report: dict[str, Any] = {}
    for name, term in terms.items():
        if not term.requires_grad:
            report[name] = {
                "loss": float(term.detach()),
                "finite": True,
                "total_l2": 0.0,
                "maximum_absolute_element": 0.0,
                "nonfinite_parameters": [],
                "clip_coefficient": 1.0,
                "max_norm": per_term_max_norm,
            }
            continue
        gradients = list(
            torch.autograd.grad(
                term,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
        )
        tangent_report: dict[str, Any] | None = None
        if output_constraint_geometry is not None:
            gradients, tangent_report = _project_final_affine_gradients(
                named_parameters,
                gradients,
                output_constraint_geometry,
            )
        statistics = _gradient_tensor_statistics(
            [
                (parameter_name, gradient)
                for (parameter_name, _), gradient in zip(
                    named_parameters, gradients, strict=True
                )
            ]
        )
        if not statistics["finite"]:
            report[name] = {
                "loss": float(term.detach()),
                **statistics,
                "clip_coefficient": None,
                "max_norm": per_term_max_norm,
            }
            raise FloatingPointError(
                f"non-finite gradient elements in map objective term {name!r}"
            )
        total = float(statistics["total_l2"])
        coefficient = min(1.0, per_term_max_norm / (total + 1e-12))
        for parameter, gradient in zip(parameters, gradients, strict=True):
            if gradient is None:
                continue
            contribution = gradient.detach() * coefficient
            if parameter.grad is None:
                parameter.grad = contribution.clone()
            else:
                parameter.grad.add_(contribution)
        report[name] = {
            "loss": float(term.detach()),
            **statistics,
            "clip_coefficient": coefficient,
            "max_norm": per_term_max_norm,
            "output_tangent_projection": tangent_report,
        }
    return report


def _map_evaluation(
    model: nn.Module,
    reference_map: nn.Module,
    cache: LatentCache,
    x_val: Tensor,
    y_val: Tensor,
    data_weights: Tensor,
    *,
    stable_ceiling: float,
    unstable_floor: float,
    jury_buffer: float,
    rollout_weight: float,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        replay = topology._replay_losses(
            model,
            cache.z_val,
            cache.z_val_next,
            x_val,
            y_val,
            data_weights,
        )
        anchors, _ = topology._anchor_residuals(model.latent_map, cache.targets, cache.scales)
        trust = nn.functional.mse_loss(model.latent_map(cache.z_val), reference_map(cache.z_val))
        if rollout_weight > 0.0:
            rollout, rollout_report = topology._trajectory_shadowing_loss(
                model.latent_map,
                cache.val_shadowing_blocks,
                diagnostics=True,
            )
        else:
            rollout = torch.zeros((), device=x_val.device)
            rollout_report = {"enabled": False, "targets_recomputed": True}
    with torch.enable_grad():
        characteristic, topological, spectra = topology._spectral_terms(
            model.latent_map,
            cache.targets,
            stable_ceiling=stable_ceiling,
            unstable_floor=unstable_floor,
            jury_buffer=jury_buffer,
            diagnostics=True,
        )
    characteristic_errors = {
        name: max(
            spectrum["trace_relative_error"],
            spectrum["determinant_relative_error"],
        )
        for name, spectrum in spectra.items()
    }
    role_violations = {
        name: topology._role_violation(
            name,
            spectrum["eigenvalues"],
            stable_ceiling,
            unstable_floor,
        )
        for name, spectrum in spectra.items()
    }
    model.latent_map.train()
    return {
        "replay": topology._float_losses(replay),
        "max_anchor_normalized_l2": topology._max_anchor_normalized_l2(anchors),
        "anchor_quadratic": float(torch.mean(anchors**2)),
        "characteristic_loss": float(characteristic.detach()),
        "topology_loss": float(topological.detach()),
        "max_characteristic_relative_error": max(characteristic_errors.values()),
        "characteristic_relative_errors": characteristic_errors,
        "max_role_margin_violation": max(role_violations.values()),
        "role_margin_violations": role_violations,
        "spectra": spectra,
        "trust_rmse": float(torch.sqrt(trust)),
        "rollout_loss": float(rollout),
        "rollout": rollout_report,
    }


@torch.no_grad()
def _source_physical_replay_baseline(
    source_model: nn.Module,
    x_val: Tensor,
    y_val: Tensor,
    data_weights: Tensor,
) -> dict[str, float]:
    """Return source-chart physical losses that remain comparable after chart surgery."""

    source_model.eval()
    source_z = source_model.encoder(x_val)
    source_z_next = source_model.encoder(y_val)
    replay = topology._replay_losses(
        source_model,
        source_z,
        source_z_next,
        x_val,
        y_val,
        data_weights,
    )
    return {
        "reconstruction": float(replay["reconstruction"]),
        "prediction": float(replay["prediction"]),
    }


def _map_gate_report(
    result: dict[str, Any],
    baseline_replay_total: float,
    baseline_rollout_loss: float,
    *,
    rollout_enabled: bool,
    validation_ratio_limit: float,
    rollout_ratio_limit: float,
    rollout_absolute_limit: float,
    anchor_acceptance: float,
    characteristic_acceptance: float,
    source_physical_baseline: dict[str, float] | None = None,
    source_reconstruction_ratio_limit: float = 1.02,
    source_prediction_ratio_limit: float = 1.05,
) -> dict[str, Any]:
    validation_ratio = result["replay"]["total"] / max(baseline_replay_total, 1e-15)
    rollout_ratio = (
        result["rollout_loss"] / max(baseline_rollout_loss, 1e-15) if rollout_enabled else 1.0
    )
    source_reconstruction_ratio = (
        result["replay"]["reconstruction"]
        / max(source_physical_baseline["reconstruction"], 1e-15)
        if source_physical_baseline is not None
        else 1.0
    )
    source_prediction_ratio = (
        result["replay"]["prediction"]
        / max(source_physical_baseline["prediction"], 1e-15)
        if source_physical_baseline is not None
        else 1.0
    )
    finite = all(
        math.isfinite(value)
        for value in (
            result["replay"]["total"],
            result["max_anchor_normalized_l2"],
            result["max_characteristic_relative_error"],
            result["max_role_margin_violation"],
            result["trust_rmse"],
            result["rollout_loss"],
            validation_ratio,
            rollout_ratio,
            source_reconstruction_ratio,
            source_prediction_ratio,
        )
    )
    gates = {
        "finite_diagnostics": finite,
        "validation_replay": validation_ratio <= validation_ratio_limit,
        "validation_rollout": (not rollout_enabled or rollout_ratio <= rollout_ratio_limit),
        "validation_rollout_absolute": (
            not rollout_enabled or result["rollout_loss"] <= rollout_absolute_limit
        ),
        "source_chart_reconstruction": (
            source_reconstruction_ratio <= source_reconstruction_ratio_limit
        ),
        "source_chart_prediction": (
            source_prediction_ratio <= source_prediction_ratio_limit
        ),
        "fixed_anchor_closure": (result["max_anchor_normalized_l2"] <= anchor_acceptance),
        "characteristic_polynomials": (
            result["max_characteristic_relative_error"] <= characteristic_acceptance
        ),
        "periodic_role_margins": result["max_role_margin_violation"] == 0.0,
    }
    violations = {
        "validation_replay": max(0.0, validation_ratio / validation_ratio_limit - 1.0),
        "validation_rollout": (
            max(0.0, rollout_ratio / rollout_ratio_limit - 1.0) if rollout_enabled else 0.0
        ),
        "validation_rollout_absolute": (
            max(0.0, result["rollout_loss"] / rollout_absolute_limit - 1.0)
            if rollout_enabled
            else 0.0
        ),
        "source_chart_reconstruction": max(
            0.0,
            source_reconstruction_ratio / source_reconstruction_ratio_limit - 1.0,
        ),
        "source_chart_prediction": max(
            0.0,
            source_prediction_ratio / source_prediction_ratio_limit - 1.0,
        ),
        "fixed_anchor_closure": max(
            0.0,
            result["max_anchor_normalized_l2"] / anchor_acceptance - 1.0,
        ),
        "characteristic_polynomials": max(
            0.0,
            result["max_characteristic_relative_error"] / characteristic_acceptance - 1.0,
        ),
        "periodic_role_margins": result["max_role_margin_violation"],
    }
    return {
        "accepted": all(gates.values()),
        "gates": gates,
        "violations": violations,
        "validation_ratio": validation_ratio,
        "rollout_ratio": rollout_ratio,
        "rollout_absolute_limit": rollout_absolute_limit,
        "source_reconstruction_ratio": source_reconstruction_ratio,
        "source_prediction_ratio": source_prediction_ratio,
    }


def _map_recoverability_report(
    result: dict[str, Any],
    *,
    baseline_replay_total: float,
    baseline_rollout_loss: float,
    anchor_limit: float = 0.01,
    characteristic_limit: float = 1.0,
    role_margin_limit: float = 0.1,
    replay_ratio_limit: float = 1.25,
    rollout_ratio_limit: float = 10.0,
) -> dict[str, Any]:
    """Screen whether topology-first repair is numerically plausible.

    This deliberately loose screen is not a promotion gate.  It prevents a
    long attempt from starting when re-encoding plus exact anchor projection
    has already produced a non-finite or grossly displaced map.  The strict
    gate in :func:`_map_gate_report` remains the only promotion criterion.
    """

    replay_ratio = result["replay"]["total"] / max(baseline_replay_total, 1e-15)
    rollout_ratio = result["rollout_loss"] / max(baseline_rollout_loss, 1e-15)
    values = (
        result["max_anchor_normalized_l2"],
        result["max_characteristic_relative_error"],
        result["max_role_margin_violation"],
        replay_ratio,
        rollout_ratio,
    )
    gates = {
        "finite_diagnostics": all(math.isfinite(value) for value in values),
        "anchor_projection_near_closed": (
            result["max_anchor_normalized_l2"] <= anchor_limit
        ),
        "characteristic_error_repairable": (
            result["max_characteristic_relative_error"] <= characteristic_limit
        ),
        "role_margin_repairable": (
            result["max_role_margin_violation"] <= role_margin_limit
        ),
        "heldout_replay_repairable": replay_ratio <= replay_ratio_limit,
        "heldout_rollout_repairable": rollout_ratio <= rollout_ratio_limit,
    }
    return {
        "recoverable": all(gates.values()),
        "purpose": "pretraining_screen_not_promotion_gate",
        "gates": gates,
        "observed": {
            "max_anchor_normalized_l2": result["max_anchor_normalized_l2"],
            "max_characteristic_relative_error": result[
                "max_characteristic_relative_error"
            ],
            "max_role_margin_violation": result["max_role_margin_violation"],
            "heldout_replay_ratio": replay_ratio,
            "heldout_rollout_ratio": rollout_ratio,
        },
        "limits": {
            "max_anchor_normalized_l2": anchor_limit,
            "max_characteristic_relative_error": characteristic_limit,
            "max_role_margin_violation": role_margin_limit,
            "heldout_replay_ratio": replay_ratio_limit,
            "heldout_rollout_ratio": rollout_ratio_limit,
        },
    }


def _map_rank(result: dict[str, Any]) -> tuple[float, float, float, float, float]:
    report = result["gates"]
    failed = sum(not passed for passed in report["gates"].values())
    if report["accepted"]:
        return (
            0.0,
            max(
                result["max_anchor_normalized_l2"],
                result["max_characteristic_relative_error"],
                result["max_role_margin_violation"],
            ),
            result["max_anchor_normalized_l2"]
            + result["max_characteristic_relative_error"]
            + result["max_role_margin_violation"],
            report["rollout_ratio"],
            result["replay"]["total"] + result["rollout_loss"],
        )
    return (
        1.0 + float(failed),
        max(report["violations"].values(), default=0.0),
        sum(report["violations"].values()),
        report["rollout_ratio"],
        result["replay"]["total"]
        + result["anchor_quadratic"]
        + result["characteristic_loss"]
        + result["topology_loss"]
        + result["rollout_loss"],
    )


def _derived_bounds(cache: LatentCache, epsilon_fraction: float) -> dict[str, list[float]]:
    encoded = torch.cat((cache.z_train, cache.z_train_next, cache.z_val, cache.z_val_next))
    lower = torch.min(encoded, dim=0).values
    upper = torch.max(encoded, dim=0).values
    padding = epsilon_fraction * (upper - lower)
    return {
        "lower_bounds": (lower - padding).detach().cpu().tolist(),
        "upper_bounds": (upper + padding).detach().cpu().tolist(),
    }


def _run_map_phase(
    model: nn.Module,
    cfg: Any,
    cache: LatentCache,
    x_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    train_sample_weights: Tensor,
    models_dir: Path,
    logs_dir: Path,
    settings: MapTrainingSettings,
    source_physical_baseline: dict[str, float] | None = None,
    *,
    resume_state: dict[str, Any] | None = None,
    reference_model: nn.Module | None = None,
    continuation_epochs: int | None = None,
    constrained_output_learning_rate: float | None = None,
    source_training_state: dict[str, Any] | None = None,
    input_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a fresh map attempt or an exact-state constrained continuation."""

    _validate_map_settings(settings)
    device = x_train.device
    is_continuation = resume_state is not None
    if is_continuation:
        if continuation_epochs is None or continuation_epochs < 1:
            raise ValueError("continuation_epochs must be positive when resuming")
        if reference_model is None:
            raise ValueError("continuation requires the preserved reference model")
        if input_provenance is None:
            raise ValueError("continuation requires pinned input provenance")
        if constrained_output_learning_rate is not None and constrained_output_learning_rate <= 0:
            raise ValueError("constrained output learning rate must be positive")
    elif continuation_epochs is not None or constrained_output_learning_rate is not None:
        raise ValueError("continuation-only settings were supplied to a fresh map attempt")
    baseline_model = reference_model if reference_model is not None else model
    reference_map = copy.deepcopy(baseline_model.latent_map).eval()
    for parameter in reference_map.parameters():
        parameter.requires_grad_(False)
    trainable_map_parameters = _configure_map_parameters(
        model,
        constrained_output_update=is_continuation,
    )
    output_layer = _linear_layers(model.latent_map)[-1]
    optimized_parameters = [
        parameter
        for parameter in model.latent_map.parameters()
        if parameter.requires_grad
        and parameter is not output_layer.weight
        and parameter is not output_layer.bias
    ]
    named_map_parameters = [
        (name, parameter)
        for name, parameter in model.latent_map.named_parameters()
        if parameter.requires_grad
    ]
    data_weights = torch.tensor(cfg.training.loss_weights, dtype=torch.float32, device=device)
    reference_map_baseline = _map_evaluation(
        baseline_model,
        reference_map,
        cache,
        x_val,
        y_val,
        data_weights,
        stable_ceiling=settings.stable_ceiling,
        unstable_floor=settings.unstable_floor,
        jury_buffer=settings.jury_buffer,
        rollout_weight=settings.rollout_weight,
    )
    baseline_replay_total = reference_map_baseline["replay"]["total"]
    baseline_rollout_loss = reference_map_baseline["rollout_loss"]
    if is_continuation:
        assert resume_state is not None
        boundary_geometry = _output_constraint_geometry(
            model.latent_map,
            cache.targets,
        )
        boundary_report = _anchor_preactivation_report(boundary_geometry)
        _validate_continuation_boundary(
            boundary_geometry,
            boundary_report,
            resume_state,
        )
        projection_initial = {
            "continuation_boundary_was_not_mutated": True,
            "acceptance_tolerance": CONTINUATION_BOUNDARY_PREACTIVATION_TOLERANCE,
            **boundary_report,
        }
    else:
        projection_initial = topology._project_anchor_equalities(
            model.latent_map,
            cache.targets,
        )
    initial_map = _map_evaluation(
        model,
        reference_map,
        cache,
        x_val,
        y_val,
        data_weights,
        stable_ceiling=settings.stable_ceiling,
        unstable_floor=settings.unstable_floor,
        jury_buffer=settings.jury_buffer,
        rollout_weight=settings.rollout_weight,
    )
    if (
        is_continuation
        and initial_map["max_anchor_normalized_l2"] > settings.anchor_acceptance
    ):
        raise ValueError(
            "continuation boundary fails normalized anchor acceptance: "
            f"{initial_map['max_anchor_normalized_l2']:.6g} > "
            f"{settings.anchor_acceptance:.6g}"
        )
    initial_map["gates"] = _map_gate_report(
        initial_map,
        baseline_replay_total,
        baseline_rollout_loss,
        rollout_enabled=settings.rollout_weight > 0.0,
        validation_ratio_limit=settings.validation_ratio_limit,
        rollout_ratio_limit=settings.rollout_ratio_limit,
        rollout_absolute_limit=settings.rollout_absolute_limit,
        anchor_acceptance=settings.anchor_acceptance,
        characteristic_acceptance=settings.characteristic_acceptance,
        source_physical_baseline=source_physical_baseline,
        source_reconstruction_ratio_limit=settings.source_reconstruction_ratio_limit,
        source_prediction_ratio_limit=settings.source_prediction_ratio_limit,
    )
    recoverability = _map_recoverability_report(
        initial_map,
        baseline_replay_total=baseline_replay_total,
        baseline_rollout_loss=baseline_rollout_loss,
    )
    recoverability_path = logs_dir / "map_recoverability.json"
    recoverability_path.write_text(json.dumps(recoverability, indent=2, allow_nan=False) + "\n")
    if not recoverability["recoverable"]:
        return {
            "status": "post_chart_map_not_recoverable",
            "trainable_parameters": trainable_map_parameters,
            "projected_output_layer_excluded_from_adam": True,
            "projected_output_layer_excluded_from_optimizer": not is_continuation,
            "constrained_output_update": is_continuation,
            "reference_before_anchor_projection": reference_map_baseline,
            "initial_anchor_projection": projection_initial,
            "initial": initial_map,
            "recoverability": recoverability,
            "recoverability_report": str(recoverability_path),
            "candidate_checkpoint": None,
            "promoted_checkpoint": None,
        }

    if is_continuation:
        assert resume_state is not None
        best_map = copy.deepcopy(resume_state["best_map"])
        best_map_state = {
            key: value.detach().cpu().clone()
            for key, value in resume_state["best_model_state_dict"].items()
        }
        best_map_epoch = int(resume_state["best_map_epoch"])
    else:
        best_map = copy.deepcopy(initial_map)
        best_map_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_map_epoch = -1
    optimizer = Adam(optimized_parameters, lr=settings.learning_rate)

    def make_scheduler(active_optimizer: Adam, learning_rate: float) -> ReduceLROnPlateau:
        return ReduceLROnPlateau(
            active_optimizer,
            mode="min",
            factor=cfg.training.scheduler_factor,
            patience=max(
                1,
                min(cfg.training.lr_patience, max(1, settings.epochs // 5))
                // settings.eval_every,
            ),
            threshold=cfg.training.scheduler_threshold,
            min_lr=min(cfg.training.scheduler_min_lr, learning_rate),
        )

    scheduler = make_scheduler(optimizer, settings.learning_rate)
    if is_continuation:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
    prior_map_history: list[dict[str, Any]] = (
        copy.deepcopy(resume_state["map_history"]) if is_continuation else []
    )
    map_history: list[dict[str, Any]] = list(prior_map_history)
    run_history: list[dict[str, Any]] = []
    map_gradient_history: list[dict[str, Any]] = []
    output_constraint_history: list[dict[str, Any]] = []
    history_path = logs_dir / "map_history.json"
    gradient_history_path = logs_dir / "map_gradient_diagnostics.json"
    output_constraint_history_path = logs_dir / "output_constraint_history.json"
    state_path = models_dir / "map_training_state.pt"
    rollout_stage_start_epoch: int | None = (
        resume_state["rollout_stage_start_epoch"] if is_continuation else None
    )
    projection_latest = projection_initial
    if settings.rollout_weight > 0.0:
        short_rollout_blocks = _limit_shadowing_horizons(cache.train_shadowing_blocks, 4)
        medium_rollout_blocks = _limit_shadowing_horizons(
            cache.train_shadowing_blocks,
            settings.rollout_medium_max_horizon,
        )
    else:
        short_rollout_blocks = []
        medium_rollout_blocks = []
    start_epoch = int(resume_state["next_epoch"]) if is_continuation else 0
    end_epoch = (
        start_epoch + int(continuation_epochs)
        if is_continuation
        else settings.epochs
    )
    if is_continuation:
        _restore_rng_state(resume_state["rng"])
    map_start = time.perf_counter()
    iterator = tqdm(
        range(start_epoch, end_epoch),
        desc="map-continuation" if is_continuation else "map",
    )
    for epoch in iterator:
        output_geometry = (
            _output_constraint_geometry(model.latent_map, cache.targets)
            if is_continuation
            else None
        )
        replay = topology._replay_losses(
            model,
            cache.z_train,
            cache.z_train_next,
            x_train,
            y_train,
            data_weights,
            sample_weights=train_sample_weights,
        )
        anchors, _ = topology._anchor_residuals(model.latent_map, cache.targets, cache.scales)
        trust = nn.functional.mse_loss(
            model.latent_map(cache.z_train), reference_map(cache.z_train)
        )
        rollout_active = (
            settings.rollout_weight > 0.0
            and rollout_stage_start_epoch is not None
            and epoch >= rollout_stage_start_epoch
        )
        rollout_max_horizon: int | None = None
        if rollout_active:
            rollout_age = epoch - rollout_stage_start_epoch
            if rollout_age < settings.rollout_short_epochs:
                rollout_blocks = short_rollout_blocks
                rollout_max_horizon = 4
            else:
                rollout_blocks = medium_rollout_blocks
                rollout_max_horizon = settings.rollout_medium_max_horizon
            rollout, _ = _trajectory_shadowing_loss_truncated(
                model.latent_map,
                rollout_blocks,
                backprop_steps=settings.rollout_backprop_steps,
                diagnostics=False,
            )
        else:
            rollout = torch.zeros((), device=device)
        if epoch >= settings.spectral_start_epoch:
            spectral_ramp = min(
                1.0,
                (epoch - settings.spectral_start_epoch + 1)
                / settings.spectral_ramp_epochs,
            )
            characteristic, topological, _ = topology._spectral_terms(
                model.latent_map,
                cache.targets,
                stable_ceiling=settings.stable_ceiling,
                unstable_floor=settings.unstable_floor,
                jury_buffer=settings.jury_buffer,
                diagnostics=False,
            )
        else:
            spectral_ramp = 0.0
            characteristic = torch.zeros((), device=device)
            topological = torch.zeros((), device=device)
        trust_fraction = max(
            0.0,
            1.0 - epoch / max(1, settings.epochs - 1),
        )
        scaled_terms = {
            "replay": replay["total"],
            "anchor": settings.anchor_weight * torch.mean(anchors**2),
            "characteristic": (
                spectral_ramp * settings.characteristic_weight * characteristic
            ),
            "topology": spectral_ramp * settings.topology_weight * topological,
            "trust": settings.trust_weight * trust_fraction * trust,
            "rollout": settings.rollout_weight * rollout,
        }
        objective = sum(scaled_terms.values())
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite map objective at epoch {epoch}")
        objective_value = float(objective.detach())
        try:
            term_gradient_report = _backward_terms_with_caps(
                scaled_terms,
                named_map_parameters,
                per_term_max_norm=settings.per_term_gradient_clip_norm,
                output_constraint_geometry=output_geometry,
            )
        except FloatingPointError:
            failure = {
                "epoch": epoch,
                "stage": "rollout" if rollout_active else "topology_repair",
                "terms": _term_gradient_diagnostics(scaled_terms, named_map_parameters),
            }
            map_gradient_history.append(failure)
            gradient_history_path.write_text(
                json.dumps(map_gradient_history, indent=2, allow_nan=False) + "\n"
            )
            raise
        total_gradient = _gradient_tensor_statistics(
            [(name, parameter.grad) for name, parameter in named_map_parameters]
        )
        diagnostic_due = (
            epoch % settings.gradient_diagnostics_every == 0
            or not total_gradient["finite"]
            or (
                total_gradient["total_l2"] is not None
                and total_gradient["total_l2"]
                >= settings.gradient_diagnostics_threshold
            )
        )
        if not total_gradient["finite"]:
            raise FloatingPointError(
                "non-finite accumulated map gradient elements; diagnostics were written to "
                f"{gradient_history_path}"
            )
        if cfg.training.gradient_clip_norm is not None:
            clip_report = _safe_clip_grad_norm_(
                named_map_parameters,
                cfg.training.gradient_clip_norm,
            )
        else:
            clip_report = {**total_gradient, "clip_coefficient": 1.0, "max_norm": None}
        output_step_report: dict[str, Any] | None = None
        if output_geometry is not None:
            active_output_learning_rate = (
                constrained_output_learning_rate
                if constrained_output_learning_rate is not None
                else float(optimizer.param_groups[0]["lr"])
            )
            output_step_report = _apply_constrained_output_sgd(
                output_geometry,
                learning_rate=active_output_learning_rate,
            )
        optimizer.step()
        projection_latest = topology._project_anchor_equalities(model.latent_map, cache.targets)
        if output_step_report is not None:
            output_constraint_record = {
                "epoch": epoch,
                **output_step_report,
                "preactivation_residual_before_exact_projection_max_abs": (
                    projection_latest.get("preactivation_residual_before_max_abs")
                ),
                "preactivation_residual_after_exact_projection_max_abs": (
                    projection_latest.get("preactivation_residual_after_max_abs")
                ),
                "exact_projection_output_parameter_correction_l2": (
                    projection_latest.get("output_parameter_correction_l2")
                ),
            }
            output_constraint_history.append(output_constraint_record)
        if diagnostic_due:
            map_gradient_history.append(
                {
                    "epoch": epoch,
                    "stage": "rollout" if rollout_active else "topology_repair",
                    "total_before_global_clip": total_gradient,
                    "terms_before_per_term_clip": term_gradient_report,
                    "rollout_backprop_steps": (
                        settings.rollout_backprop_steps if rollout_active else None
                    ),
                    "rollout_training_max_horizon": rollout_max_horizon,
                    "spectral_ramp": spectral_ramp,
                    "constrained_output_step": output_step_report,
                    "exact_projection_after_hidden_and_output_steps": projection_latest,
                }
            )
            gradient_history_path.write_text(
                json.dumps(map_gradient_history, indent=2, allow_nan=False) + "\n"
            )
        del objective, scaled_terms, replay, anchors, trust, rollout, characteristic, topological
        if (epoch + 1) % settings.eval_every != 0 and epoch + 1 != end_epoch:
            iterator.set_postfix(loss=f"{objective_value:.3e}", trust=f"{trust_fraction:.2f}")
            continue
        current = _map_evaluation(
            model,
            reference_map,
            cache,
            x_val,
            y_val,
            data_weights,
            stable_ceiling=settings.stable_ceiling,
            unstable_floor=settings.unstable_floor,
            jury_buffer=settings.jury_buffer,
            rollout_weight=settings.rollout_weight,
        )
        current["gates"] = _map_gate_report(
            current,
            baseline_replay_total,
            baseline_rollout_loss,
            rollout_enabled=settings.rollout_weight > 0.0,
            validation_ratio_limit=settings.validation_ratio_limit,
            rollout_ratio_limit=settings.rollout_ratio_limit,
            rollout_absolute_limit=settings.rollout_absolute_limit,
            anchor_acceptance=settings.anchor_acceptance,
            characteristic_acceptance=settings.characteristic_acceptance,
            source_physical_baseline=source_physical_baseline,
            source_reconstruction_ratio_limit=settings.source_reconstruction_ratio_limit,
            source_prediction_ratio_limit=settings.source_prediction_ratio_limit,
        )
        selection_score = _map_rank(current)[-1]
        scheduler.step(selection_score)
        topology_recovered = (
            current["max_anchor_normalized_l2"] <= settings.anchor_acceptance
            and current["max_characteristic_relative_error"]
            <= settings.characteristic_acceptance
            and current["max_role_margin_violation"] == 0.0
        )
        start_rollout_after_this_epoch = (
            settings.rollout_weight > 0.0
            and rollout_stage_start_epoch is None
            and epoch + 1 >= settings.rollout_min_topology_epochs
            and topology_recovered
        )
        history_entry = {
            "epoch": epoch,
            "stage": "rollout" if rollout_active else "topology_repair",
            "selection_score": selection_score,
            "accepted": current["gates"]["accepted"],
            "validation_ratio": current["gates"]["validation_ratio"],
            "rollout_ratio": current["gates"]["rollout_ratio"],
            "max_anchor_normalized_l2": current["max_anchor_normalized_l2"],
            "max_characteristic_relative_error": current[
                "max_characteristic_relative_error"
            ],
            "max_role_margin_violation": current["max_role_margin_violation"],
            "trust_anneal_fraction": trust_fraction,
            "spectral_ramp": spectral_ramp,
            "trust_rmse": current["trust_rmse"],
            "rollout_loss": current["rollout_loss"],
            "topology_recovered": topology_recovered,
            "rollout_starts_next_epoch": start_rollout_after_this_epoch,
            "rollout_training_max_horizon": rollout_max_horizon,
            "gradient_total_l2_before_global_clip": clip_report["total_l2"],
            "gradient_global_clip_coefficient": clip_report["clip_coefficient"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "constrained_output_learning_rate": (
                output_step_report["learning_rate"]
                if output_step_report is not None
                else None
            ),
            "F_delta_max_abs": (
                output_step_report["F_delta_max_abs"]
                if output_step_report is not None
                else None
            ),
        }
        map_history.append(history_entry)
        run_history.append(history_entry)
        if _map_rank(current) < _map_rank(best_map):
            best_map = copy.deepcopy(current)
            best_map_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_map_epoch = epoch
        history_path.write_text(json.dumps(run_history, indent=2, allow_nan=False) + "\n")
        if is_continuation:
            output_constraint_history_path.write_text(
                json.dumps(output_constraint_history, indent=2, allow_nan=False) + "\n"
            )
        save_checkpoint(model, cfg.arch, models_dir, basename="map_last_good")
        if start_rollout_after_this_epoch:
            rollout_stage_start_epoch = epoch + 1
            optimizer = Adam(optimized_parameters, lr=settings.rollout_learning_rate)
            scheduler = make_scheduler(optimizer, settings.rollout_learning_rate)
        _atomic_torch_save(
            {
                "format_version": 2 if is_continuation else 1,
                "next_epoch": epoch + 1,
                "next_stage": (
                    "rollout"
                    if rollout_stage_start_epoch is not None
                    and epoch + 1 >= rollout_stage_start_epoch
                    else "topology_repair"
                ),
                "rollout_stage_start_epoch": rollout_stage_start_epoch,
                "settings": asdict(settings),
                "architecture": cfg.arch.model_dump(mode="json"),
                "input_provenance": input_provenance,
                "last_anchor_projection": projection_latest,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_model_state_dict": best_map_state,
                "best_map": best_map,
                "best_map_epoch": best_map_epoch,
                "last_map": current,
                "map_history": map_history,
                "rng": _capture_rng_state(),
                "continuation": (
                    {
                        "source_training_state": source_training_state,
                        "source_next_epoch": start_epoch,
                        "requested_additional_epochs": continuation_epochs,
                        "constrained_output_update": True,
                        "output_learning_rate_policy": (
                            "fixed_explicit"
                            if constrained_output_learning_rate is not None
                            else "follow_restored_hidden_adam_learning_rate"
                        ),
                        "fixed_output_learning_rate": constrained_output_learning_rate,
                        "source_history_length": len(prior_map_history),
                        "output_constraint_history": output_constraint_history,
                    }
                    if is_continuation
                    else None
                ),
            },
            state_path,
        )
        iterator.set_postfix(
            val=f"{current['gates']['validation_ratio']:.3f}",
            anchor=f"{current['max_anchor_normalized_l2']:.2e}",
            char=f"{current['max_characteristic_relative_error']:.2e}",
            role=f"{current['max_role_margin_violation']:.2e}",
        )
    map_duration = time.perf_counter() - map_start
    model.load_state_dict(best_map_state)
    selected_map = _map_evaluation(
        model,
        reference_map,
        cache,
        x_val,
        y_val,
        data_weights,
        stable_ceiling=settings.stable_ceiling,
        unstable_floor=settings.unstable_floor,
        jury_buffer=settings.jury_buffer,
        rollout_weight=settings.rollout_weight,
    )
    selected_map["gates"] = _map_gate_report(
        selected_map,
        baseline_replay_total,
        baseline_rollout_loss,
        rollout_enabled=settings.rollout_weight > 0.0,
        validation_ratio_limit=settings.validation_ratio_limit,
        rollout_ratio_limit=settings.rollout_ratio_limit,
        rollout_absolute_limit=settings.rollout_absolute_limit,
        anchor_acceptance=settings.anchor_acceptance,
        characteristic_acceptance=settings.characteristic_acceptance,
        source_physical_baseline=source_physical_baseline,
        source_reconstruction_ratio_limit=settings.source_reconstruction_ratio_limit,
        source_prediction_ratio_limit=settings.source_prediction_ratio_limit,
    )
    candidate_paths = save_checkpoint(
        model.cpu(), cfg.arch, models_dir, basename="alternating_candidate"
    )
    promoted_paths: tuple[Path, Path] | None = None
    if selected_map["gates"]["accepted"]:
        promoted_paths = save_checkpoint(model, cfg.arch, models_dir)
    history_path.write_text(json.dumps(run_history, indent=2, allow_nan=False) + "\n")
    return {
        "status": (
            "accepted_numerical_candidate_not_a_conley_certificate"
            if promoted_paths is not None
            else "map_rejected_candidate_only"
        ),
        "trainable_parameters": trainable_map_parameters,
        "projected_output_layer_excluded_from_adam": True,
        "projected_output_layer_excluded_from_optimizer": not is_continuation,
        "constrained_output_update": is_continuation,
        "initial_anchor_projection": projection_initial,
        "last_anchor_projection": projection_latest,
        "reference_before_anchor_projection": reference_map_baseline,
        "initial": initial_map,
        "recoverability": recoverability,
        "last": current,
        "last_epoch": end_epoch - 1,
        "selected": selected_map,
        "best_epoch": best_map_epoch,
        "epochs": settings.epochs,
        "start_epoch": start_epoch,
        "end_epoch_exclusive": end_epoch,
        "epochs_executed": end_epoch - start_epoch,
        "source_history_length": len(prior_map_history),
        "duration_seconds": map_duration,
        "rollout_stage_start_epoch": rollout_stage_start_epoch,
        "rollout_gradient_policy": {
            "method": "truncated_gradient_surrogate_with_exact_forward_values",
            "backpropagation_segment_steps": settings.rollout_backprop_steps,
            "short_curriculum_epochs": settings.rollout_short_epochs,
            "short_max_horizon": 4,
            "medium_max_horizon": settings.rollout_medium_max_horizon,
            "long_horizons": "evaluation_only",
            "heldout_promotion_max_horizon": 320,
        },
        "gradient_policy": {
            "norm_accumulator_dtype": "float64",
            "per_term_max_norm": settings.per_term_gradient_clip_norm,
            "global_max_norm": cfg.training.gradient_clip_norm,
            "nonfinite_elements_are_rejected": True,
            "final_affine": (
                {
                    "method": "explicit_nullspace_projected_sgd_without_momentum",
                    "design_recomputed_every_step": True,
                    "exact_projection_after_hidden_and_output_steps": True,
                    "F_delta_logged_every_step": True,
                    "learning_rate_policy": (
                        "fixed_explicit"
                        if constrained_output_learning_rate is not None
                        else "follow_restored_hidden_adam_learning_rate"
                    ),
                    "fixed_learning_rate": constrained_output_learning_rate,
                }
                if is_continuation
                else "excluded_from_training"
            ),
        },
        "optimizer_checkpoint": str(state_path),
        "output_constraint_history": (
            str(output_constraint_history_path) if is_continuation else None
        ),
        "maximum_F_delta_residual": (
            max(record["F_delta_max_abs"] for record in output_constraint_history)
            if output_constraint_history
            else None
        ),
        "candidate_checkpoint": [str(path) for path in candidate_paths],
        "candidate_checkpoint_sha256": {
            path.name: _sha256(path) for path in candidate_paths
        },
        "promoted_checkpoint": (
            [str(path) for path in promoted_paths] if promoted_paths is not None else None
        ),
        "promoted_checkpoint_sha256": (
            {path.name: _sha256(path) for path in promoted_paths}
            if promoted_paths is not None
            else None
        ),
    }


def run(
    config_name: str,
    *,
    device_name: str,
    chart_epochs: int,
    map_epochs: int,
    chart_scope: str,
    chart_learning_rate: float,
    map_learning_rate: float,
    secant_sample_count: int,
    secant_neighbors: int,
    margin_expansion: float,
    minimum_margin: float,
    anti_fold_fraction: float,
    chart_weights: dict[str, float],
    reconstruction_ratio_limit: float,
    prediction_ratio_limit: float,
    semiconjugacy_ratio_limit: float,
    drift_limit: float,
    cross_role_ratio_floor: float,
    secant_p01_ratio_floor: float,
    chart_improvement_fraction: float,
    map_anchor_weight: float,
    map_characteristic_weight: float,
    map_topology_weight: float,
    map_trust_weight: float,
    rollout_weight: float,
    rollout_ratio_limit: float,
    rollout_absolute_limit: float,
    rollout_backprop_steps: int,
    rollout_short_epochs: int,
    rollout_medium_max_horizon: int,
    rollout_learning_rate: float,
    rollout_min_topology_epochs: int,
    spectral_start_epoch: int,
    spectral_ramp_epochs: int,
    per_term_gradient_clip_norm: float,
    gradient_diagnostics_every: int,
    gradient_diagnostics_threshold: float,
    eval_every: int,
    stable_ceiling: float,
    unstable_floor: float,
    jury_buffer: float,
    validation_ratio_limit: float,
    anchor_acceptance: float,
    characteristic_acceptance: float,
    expected_source_sha256: str,
    validate_only: bool,
) -> dict[str, Any]:
    cfg = load_config(config_name)
    if len(cfg.seeds) != 1:
        raise ValueError("alternating continuation requires exactly one seed")
    if (
        chart_epochs < 0
        or map_epochs < 0
        or eval_every < 1
        or spectral_start_epoch < 0
        or spectral_ramp_epochs < 1
        or rollout_backprop_steps < 1
        or rollout_short_epochs < 0
        or rollout_medium_max_horizon < 4
        or rollout_medium_max_horizon > 16
        or rollout_min_topology_epochs < 0
        or per_term_gradient_clip_norm <= 0.0
        or gradient_diagnostics_every < 1
        or gradient_diagnostics_threshold <= 0.0
    ):
        raise ValueError("epoch counts must be nonnegative and eval_every positive")
    if not validate_only and (chart_epochs < 1 or map_epochs < 1):
        raise ValueError("training requires positive chart and map epoch counts")
    if (
        chart_learning_rate <= 0.0
        or map_learning_rate <= 0.0
        or rollout_learning_rate <= 0.0
    ):
        raise ValueError("learning rates must be positive")
    if not 0.0 <= chart_improvement_fraction < 1.0:
        raise ValueError("chart improvement fraction must lie in [0,1)")
    if any(weight < 0.0 for weight in chart_weights.values()):
        raise ValueError("chart loss weights must be nonnegative")
    if any(
        weight < 0.0
        for weight in (
            map_anchor_weight,
            map_characteristic_weight,
            map_topology_weight,
            map_trust_weight,
            rollout_weight,
        )
    ):
        raise ValueError("map loss weights must be nonnegative")
    if (
        reconstruction_ratio_limit <= 0.0
        or prediction_ratio_limit <= 0.0
        or semiconjugacy_ratio_limit <= 0.0
        or drift_limit <= 0.0
        or cross_role_ratio_floor <= 0.0
        or secant_p01_ratio_floor <= 0.0
        or anchor_acceptance <= 0.0
        or characteristic_acceptance <= 0.0
        or validation_ratio_limit <= 0.0
        or rollout_ratio_limit <= 0.0
        or rollout_absolute_limit <= 0.0
        or stable_ceiling <= 0.0
        or unstable_floor <= stable_ceiling
        or jury_buffer < 0.0
    ):
        raise ValueError("all numerical gate limits must be positive")
    seed = int(cfg.seeds[0])
    _seed_everything(seed)
    device = torch.device(device_name)
    output_dir = _resolve(cfg.paths.output_dir) / f"seed_{seed}"
    if not validate_only and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty alternating output directory {output_dir}"
        )
    source_dir = _resolve(cfg.training.warm_start_checkpoint_dir)
    source_path = source_dir / "autoencoder.pt"
    source_sha = _sha256(source_path)
    if source_sha != expected_source_sha256:
        raise ValueError(
            "source checkpoint hash does not match the accepted primary-v2 checkpoint: "
            f"expected {expected_source_sha256}, got {source_sha}"
        )
    model, source_arch = load_any_checkpoint(source_dir, map_location=device)
    if source_arch.model_dump() != cfg.arch.model_dump():
        raise ValueError("primary-v2 checkpoint architecture does not match config")
    model = model.to(device)
    reference = copy.deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    data_dir = _resolve(cfg.paths.data_dir)
    scaler_path = _resolve(cfg.paths.scaler_path("train"))
    scaler = joblib.load(scaler_path)
    manifest_path = data_dir / "dataset_manifest.json"
    train_metadata_path = data_dir / "train_metadata.json"
    val_metadata_path = data_dir / "val_metadata.json"
    manifest = json.loads(manifest_path.read_text())
    train_metadata = json.loads(train_metadata_path.read_text())
    val_metadata = json.loads(val_metadata_path.read_text())
    x_train, y_train = topology._load_pairs(data_dir / "train.csv", scaler, device)
    x_val, y_val = topology._load_pairs(data_dir / "val.csv", scaler, device)
    train_sample_weights, replay_component_provenance = topology._manifest_component_sample_weights(
        train_metadata,
        row_count=len(x_train),
        overrides=PRIMARY_REPLAY_COMPONENT_WEIGHTS,
        device=device,
    )
    run_hyperparameters = {
        "chart_epochs": chart_epochs,
        "map_epochs": map_epochs,
        "chart_scope": chart_scope,
        "chart_learning_rate": chart_learning_rate,
        "map_learning_rate": map_learning_rate,
        "secant_sample_count": secant_sample_count,
        "secant_neighbors": secant_neighbors,
        "margin_expansion": margin_expansion,
        "minimum_margin": minimum_margin,
        "anti_fold_fraction": anti_fold_fraction,
        "chart_weights": chart_weights,
        "primary_replay_component_weight_overrides": PRIMARY_REPLAY_COMPONENT_WEIGHTS,
        "map_anchor_weight": map_anchor_weight,
        "map_characteristic_weight": map_characteristic_weight,
        "map_topology_weight": map_topology_weight,
        "map_trust_weight": map_trust_weight,
        "rollout_weight": rollout_weight,
        "rollout_ratio_limit": rollout_ratio_limit,
        "rollout_absolute_limit": rollout_absolute_limit,
        "rollout_backprop_steps": rollout_backprop_steps,
        "rollout_short_epochs": rollout_short_epochs,
        "rollout_medium_max_horizon": rollout_medium_max_horizon,
        "rollout_learning_rate": rollout_learning_rate,
        "rollout_min_topology_epochs": rollout_min_topology_epochs,
        "spectral_start_epoch": spectral_start_epoch,
        "spectral_ramp_epochs": spectral_ramp_epochs,
        "per_term_gradient_clip_norm": per_term_gradient_clip_norm,
        "gradient_diagnostics_every": gradient_diagnostics_every,
        "gradient_diagnostics_threshold": gradient_diagnostics_threshold,
        "eval_every": eval_every,
        "stable_ceiling": stable_ceiling,
        "unstable_floor": unstable_floor,
        "jury_buffer": jury_buffer,
    }
    acceptance_limits = {
        "chart_reconstruction_ratio": reconstruction_ratio_limit,
        "chart_prediction_ratio": prediction_ratio_limit,
        "chart_semiconjugacy_ratio": semiconjugacy_ratio_limit,
        "chart_encoder_drift_rmse": drift_limit,
        "chart_cross_role_ratio_floor": cross_role_ratio_floor,
        "chart_secant_p01_ratio_floor": secant_p01_ratio_floor,
        "chart_improvement_fraction": chart_improvement_fraction,
        "map_validation_replay_ratio": validation_ratio_limit,
        "map_validation_rollout_ratio": rollout_ratio_limit,
        "map_validation_rollout_absolute": rollout_absolute_limit,
        "map_source_reconstruction_ratio": reconstruction_ratio_limit,
        "map_source_prediction_ratio": prediction_ratio_limit,
        "map_anchor_normalized_l2": anchor_acceptance,
        "map_characteristic_relative_error": characteristic_acceptance,
    }
    with torch.no_grad():
        train_reference_x = reference.encoder(x_train).detach()
        train_reference_y = reference.encoder(y_train).detach()
        val_reference_x = reference.encoder(x_val).detach()
        val_reference_y = reference.encoder(y_val).detach()
    source_physical_baseline = _source_physical_replay_baseline(
        reference,
        x_val,
        y_val,
        torch.tensor(cfg.training.loss_weights, dtype=torch.float32, device=device),
    )
    known_physical, known_labels = _known_object_batch(scaler, manifest, device)
    with torch.no_grad():
        known_reference = reference.encoder(known_physical).detach()
    train_secants = _make_local_secant_bank(
        x_train,
        train_reference_x,
        sample_count=secant_sample_count,
        neighbors=secant_neighbors,
    )
    val_secants = _make_local_secant_bank(
        x_val,
        val_reference_x,
        sample_count=min(secant_sample_count, len(x_val)),
        neighbors=secant_neighbors,
    )

    trainable_chart_parameters = _configure_chart_parameters(model, chart_scope)
    baseline_chart = _evaluate_chart(
        model,
        reference,
        x_val,
        y_val,
        val_reference_x,
        val_reference_y,
        known_physical,
        known_reference,
        known_labels,
        val_secants,
        chart_weights,
        None,
        margin_expansion=margin_expansion,
        minimum_margin=minimum_margin,
        anti_fold_fraction=anti_fold_fraction,
    )
    baseline_chart["gates"] = _chart_gate_report(
        baseline_chart,
        baseline_chart,
        reconstruction_ratio_limit=reconstruction_ratio_limit,
        prediction_ratio_limit=prediction_ratio_limit,
        semiconjugacy_ratio_limit=semiconjugacy_ratio_limit,
        drift_limit=drift_limit,
        cross_role_ratio_floor=cross_role_ratio_floor,
        secant_p01_ratio_floor=secant_p01_ratio_floor,
    )

    if validate_only:
        cache = _rebuild_latent_cache(
            model,
            scaler,
            manifest,
            train_metadata,
            val_metadata,
            x_train,
            y_train,
            x_val,
            y_val,
        )
        reference_map = copy.deepcopy(reference.latent_map).eval()
        data_weights = torch.tensor(cfg.training.loss_weights, dtype=torch.float32, device=device)
        map_reference_result = _map_evaluation(
            model,
            reference_map,
            cache,
            x_val,
            y_val,
            data_weights,
            stable_ceiling=stable_ceiling,
            unstable_floor=unstable_floor,
            jury_buffer=jury_buffer,
            rollout_weight=rollout_weight,
        )
        projection = topology._project_anchor_equalities(model.latent_map, cache.targets)
        map_result = _map_evaluation(
            model,
            reference_map,
            cache,
            x_val,
            y_val,
            data_weights,
            stable_ceiling=stable_ceiling,
            unstable_floor=unstable_floor,
            jury_buffer=jury_buffer,
            rollout_weight=rollout_weight,
        )
        return {
            "mode": "read_only_validation",
            "source_checkpoint_sha256": source_sha,
            "trainer_sha256": _sha256(Path(__file__).resolve()),
            "configuration": cfg.model_dump(mode="json"),
            "hyperparameters": run_hyperparameters,
            "acceptance_limits": acceptance_limits,
            "trainable_chart_parameters": trainable_chart_parameters,
            "chart_baseline": baseline_chart,
            "training_replay_component_weights": replay_component_provenance,
            "source_physical_replay_baseline": source_physical_baseline,
            "cache": _cache_report(cache),
            "map_reference_before_reprojection": map_reference_result,
            "anchor_projection": projection,
            "map_baseline_after_reprojection": map_result,
            "derived_cmgdb_bounds": _derived_bounds(cache, cfg.cmgdb.bounds_epsilon_frac),
            "writes_performed": False,
        }

    chart_optimizer = Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=chart_learning_rate,
    )
    chart_scheduler = ReduceLROnPlateau(
        chart_optimizer,
        mode="min",
        factor=cfg.training.scheduler_factor,
        patience=min(cfg.training.lr_patience, max(1, chart_epochs // 5)),
        threshold=cfg.training.scheduler_threshold,
        min_lr=min(cfg.training.scheduler_min_lr, chart_learning_rate),
    )
    best_chart = copy.deepcopy(baseline_chart)
    best_chart_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_chart_epoch = -1
    chart_history: list[dict[str, Any]] = []
    chart_start = time.perf_counter()
    iterator = tqdm(range(chart_epochs), desc="chart")
    for epoch in iterator:
        model.encoder.train()
        model.decoder.train()
        model.latent_map.eval()
        chart_optimizer.zero_grad(set_to_none=True)
        terms = _chart_terms(
            model,
            reference,
            x_train,
            y_train,
            train_reference_x,
            train_reference_y,
            known_physical,
            known_reference,
            known_labels,
            train_secants,
            train_sample_weights,
            margin_expansion=margin_expansion,
            minimum_margin=minimum_margin,
            anti_fold_fraction=anti_fold_fraction,
        )
        objective = _weighted_chart_score(terms, chart_weights)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite chart objective at epoch {epoch}")
        objective.backward()
        if cfg.training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                cfg.training.gradient_clip_norm,
                error_if_nonfinite=True,
            )
        chart_optimizer.step()
        current = _evaluate_chart(
            model,
            reference,
            x_val,
            y_val,
            val_reference_x,
            val_reference_y,
            known_physical,
            known_reference,
            known_labels,
            val_secants,
            chart_weights,
            None,
            margin_expansion=margin_expansion,
            minimum_margin=minimum_margin,
            anti_fold_fraction=anti_fold_fraction,
        )
        current["gates"] = _chart_gate_report(
            current,
            baseline_chart,
            reconstruction_ratio_limit=reconstruction_ratio_limit,
            prediction_ratio_limit=prediction_ratio_limit,
            semiconjugacy_ratio_limit=semiconjugacy_ratio_limit,
            drift_limit=drift_limit,
            cross_role_ratio_floor=cross_role_ratio_floor,
            secant_p01_ratio_floor=secant_p01_ratio_floor,
        )
        chart_scheduler.step(current["score"])
        chart_history.append(
            {
                "epoch": epoch,
                "train_score": float(objective.detach()),
                "validation_score": current["score"],
                "accepted_by_safety_gates": current["gates"]["accepted"],
                "reconstruction_ratio": current["gates"]["reconstruction_ratio"],
                "prediction_ratio": current["gates"]["prediction_ratio"],
                "semiconjugacy_ratio": current["gates"]["semiconjugacy_ratio"],
                "encoder_drift_rmse": current["encoder_drift_rmse"],
                "minimum_cross_role_ratio": current["cross_role"]["minimum_ratio_to_reference"],
                "p01_local_secant_ratio": current["local_secant_ratio"]["p01"],
                "learning_rate": chart_optimizer.param_groups[0]["lr"],
            }
        )
        if _chart_rank(current) < _chart_rank(best_chart):
            best_chart = copy.deepcopy(current)
            best_chart_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_chart_epoch = epoch
        iterator.set_postfix(
            score=f"{current['score']:.3e}",
            recon=f"{current['gates']['reconstruction_ratio']:.3f}",
            drift=f"{current['encoder_drift_rmse']:.2e}",
            fold=f"{current['local_secant_ratio']['p01']:.2f}",
        )
    chart_duration = time.perf_counter() - chart_start
    model.load_state_dict(best_chart_state)
    selected_chart = _evaluate_chart(
        model,
        reference,
        x_val,
        y_val,
        val_reference_x,
        val_reference_y,
        known_physical,
        known_reference,
        known_labels,
        val_secants,
        chart_weights,
        None,
        margin_expansion=margin_expansion,
        minimum_margin=minimum_margin,
        anti_fold_fraction=anti_fold_fraction,
    )
    selected_chart["gates"] = _chart_gate_report(
        selected_chart,
        baseline_chart,
        reconstruction_ratio_limit=reconstruction_ratio_limit,
        prediction_ratio_limit=prediction_ratio_limit,
        semiconjugacy_ratio_limit=semiconjugacy_ratio_limit,
        drift_limit=drift_limit,
        cross_role_ratio_floor=cross_role_ratio_floor,
        secant_p01_ratio_floor=secant_p01_ratio_floor,
    )
    chart_improved = (
        selected_chart["score"] <= (1.0 - chart_improvement_fraction) * baseline_chart["score"]
    )
    chart_accepted = selected_chart["gates"]["accepted"] and chart_improved

    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir()
    chart_candidate_paths = save_checkpoint(
        model.cpu(), cfg.arch, models_dir, basename="chart_candidate"
    )
    model = model.to(device)
    (logs_dir / "chart_history.json").write_text(
        json.dumps(chart_history, indent=2, allow_nan=False) + "\n"
    )
    if not chart_accepted:
        summary = {
            "experiment": cfg.experiment_name,
            "status": "chart_rejected_map_phase_not_run",
            "source_checkpoint_sha256": source_sha,
            "trainer_sha256": _sha256(Path(__file__).resolve()),
            "configuration": cfg.model_dump(mode="json"),
            "hyperparameters": run_hyperparameters,
            "acceptance_limits": acceptance_limits,
            "training_replay_component_weights": replay_component_provenance,
            "trainable_chart_parameters": trainable_chart_parameters,
            "baseline_chart": baseline_chart,
            "selected_chart": selected_chart,
            "chart_score_improved": chart_improved,
            "best_chart_epoch": best_chart_epoch,
            "chart_duration_seconds": chart_duration,
            "candidate_checkpoint": [str(path) for path in chart_candidate_paths],
            "limitation": (
                "strict chart gates rejected the candidate; no map optimization or "
                "promotion was attempted"
            ),
        }
        (output_dir / "alternating_refinement_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        return summary

    cache = _rebuild_latent_cache(
        model,
        scaler,
        manifest,
        train_metadata,
        val_metadata,
        x_train,
        y_train,
        x_val,
        y_val,
    )
    cache_report = _cache_report(cache)
    chart_refined_paths = save_checkpoint(
        model.cpu(), cfg.arch, models_dir, basename="chart_refined"
    )
    model = model.to(device)
    map_settings = MapTrainingSettings(
        epochs=map_epochs,
        learning_rate=map_learning_rate,
        rollout_learning_rate=rollout_learning_rate,
        rollout_min_topology_epochs=rollout_min_topology_epochs,
        anchor_weight=map_anchor_weight,
        characteristic_weight=map_characteristic_weight,
        topology_weight=map_topology_weight,
        trust_weight=map_trust_weight,
        rollout_weight=rollout_weight,
        rollout_ratio_limit=rollout_ratio_limit,
        rollout_absolute_limit=rollout_absolute_limit,
        rollout_backprop_steps=rollout_backprop_steps,
        rollout_short_epochs=rollout_short_epochs,
        rollout_medium_max_horizon=rollout_medium_max_horizon,
        spectral_start_epoch=spectral_start_epoch,
        spectral_ramp_epochs=spectral_ramp_epochs,
        per_term_gradient_clip_norm=per_term_gradient_clip_norm,
        gradient_diagnostics_every=gradient_diagnostics_every,
        gradient_diagnostics_threshold=gradient_diagnostics_threshold,
        eval_every=eval_every,
        stable_ceiling=stable_ceiling,
        unstable_floor=unstable_floor,
        jury_buffer=jury_buffer,
        validation_ratio_limit=validation_ratio_limit,
        source_reconstruction_ratio_limit=reconstruction_ratio_limit,
        source_prediction_ratio_limit=prediction_ratio_limit,
        anchor_acceptance=anchor_acceptance,
        characteristic_acceptance=characteristic_acceptance,
    )
    map_phase = _run_map_phase(
        model,
        cfg,
        cache,
        x_train,
        y_train,
        x_val,
        y_val,
        train_sample_weights,
        models_dir,
        logs_dir,
        map_settings,
        source_physical_baseline,
    )
    summary = {
        "experiment": cfg.experiment_name,
        "method": "guarded_chart_edge_update_then_reencoded_exact_anchor_topology_map_repair",
        "status": map_phase["status"],
        "seed": seed,
        "deterministic_algorithms": True,
        "configuration": cfg.model_dump(mode="json"),
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "hyperparameters": run_hyperparameters,
        "acceptance_limits": acceptance_limits,
        "source": {
            "path": str(source_dir),
            "checkpoint_sha256": source_sha,
            "architecture_sidecar_sha256": _sha256(source_dir / "autoencoder.json"),
            "expected_primary_v2_sha256": expected_source_sha256,
            "physical_replay_baseline": source_physical_baseline,
        },
        "data_provenance": {
            "train_csv_sha256": _sha256(data_dir / "train.csv"),
            "validation_csv_sha256": _sha256(data_dir / "val.csv"),
            "manifest_sha256": _sha256(manifest_path),
            "train_metadata_sha256": _sha256(train_metadata_path),
            "validation_metadata_sha256": _sha256(val_metadata_path),
            "scaler_sha256": _sha256(scaler_path),
            "training_replay_component_weights": replay_component_provenance,
        },
        "chart_phase": {
            "scope": chart_scope,
            "trainable_parameters": trainable_chart_parameters,
            "weights": chart_weights,
            "baseline": baseline_chart,
            "selected": selected_chart,
            "best_epoch": best_chart_epoch,
            "epochs": chart_epochs,
            "duration_seconds": chart_duration,
            "candidate_checkpoint": [str(path) for path in chart_candidate_paths],
            "candidate_checkpoint_sha256": {
                path.name: _sha256(path) for path in chart_candidate_paths
            },
            "accepted_checkpoint": [str(path) for path in chart_refined_paths],
            "accepted_checkpoint_sha256": {
                path.name: _sha256(path) for path in chart_refined_paths
            },
            "finite_secant_bank_not_injectivity_proof": True,
        },
        "latent_cache_rebuild": cache_report,
        "map_phase": {
            **map_phase,
            "trust_policy": "linear_anneal_from_configured_weight_to_zero",
            "weights": {
                "replay": list(cfg.training.loss_weights),
                "anchor": map_anchor_weight,
                "characteristic": map_characteristic_weight,
                "topology": map_topology_weight,
                "trust_initial": map_trust_weight,
                "rollout": rollout_weight,
            },
        },
        "derived_cmgdb_bounds": _derived_bounds(cache, cfg.cmgdb.bounds_epsilon_frac),
        "candidate_checkpoint": map_phase["candidate_checkpoint"],
        "candidate_checkpoint_sha256": map_phase.get("candidate_checkpoint_sha256"),
        "promoted_checkpoint": map_phase["promoted_checkpoint"],
        "promoted_checkpoint_sha256": map_phase.get("promoted_checkpoint_sha256"),
        "limitations": [
            "the local finite secant bank cannot certify global encoder injectivity",
            "a 3-to-2 chart cannot embed an open three-dimensional neighborhood",
            "periodic anchor and multiplier supervision is prior information, not a Conley proof",
            "extra recurrent roots are not yet supplied as moving hard negatives",
            "CMGDB bounds must use the derived post-chart bounds, not the frozen-chart rectangle",
        ],
    }
    (output_dir / "alternating_refinement_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def resume_map_from_chart_refined(
    config_name: str,
    *,
    device_name: str,
    settings: MapTrainingSettings,
    expected_source_sha256: str,
    expected_chart_sha256: str,
) -> dict[str, Any]:
    """Start a new map attempt from the preserved accepted chart checkpoint.

    The failed epoch-58 process left no valid map/optimizer checkpoint.  This
    entry point therefore rebuilds all chart-dependent caches and begins at map
    epoch zero with a fresh optimizer and schedules; it is never described as
    continuation at epoch 59.
    """

    cfg = load_config(config_name)
    if len(cfg.seeds) != 1:
        raise ValueError("map-only continuation requires exactly one seed")
    _validate_map_settings(settings)
    seed = int(cfg.seeds[0])
    _seed_everything(seed)
    device = torch.device(device_name)
    output_dir = _resolve(cfg.paths.output_dir) / f"seed_{seed}"
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    chart_path = models_dir / "chart_refined.pt"
    chart_sidecar_path = models_dir / "chart_refined.json"
    chart_history_path = logs_dir / "chart_history.json"
    for path in (chart_path, chart_sidecar_path, chart_history_path):
        if not path.is_file():
            raise FileNotFoundError(f"preserved chart artifact is missing: {path}")
    chart_sha = _sha256(chart_path)
    if chart_sha != expected_chart_sha256:
        raise ValueError(
            "chart_refined checkpoint hash mismatch: "
            f"expected {expected_chart_sha256}, got {chart_sha}"
        )
    source_dir = _resolve(cfg.training.warm_start_checkpoint_dir)
    source_path = source_dir / "autoencoder.pt"
    source_sha = _sha256(source_path)
    if source_sha != expected_source_sha256:
        raise ValueError(
            "primary-v2 source hash mismatch: "
            f"expected {expected_source_sha256}, got {source_sha}"
        )
    source_model, source_arch = load_any_checkpoint(source_dir, map_location=device)
    if source_arch.model_dump() != cfg.arch.model_dump():
        raise ValueError("primary-v2 source architecture does not match continuation config")
    source_model = source_model.to(device).eval()
    model, chart_arch = load_any_checkpoint(
        models_dir,
        basename="chart_refined",
        map_location=device,
    )
    if chart_arch.model_dump() != cfg.arch.model_dump():
        raise ValueError("chart_refined architecture does not match continuation config")
    model = model.to(device)

    collision_paths = (
        logs_dir / "map_history.json",
        logs_dir / "map_gradient_diagnostics.json",
        logs_dir / "map_attempt_manifest.json",
        models_dir / "map_training_state.pt",
        models_dir / "map_last_good.pt",
        models_dir / "alternating_candidate.pt",
        models_dir / "autoencoder.pt",
    )
    collisions = [str(path) for path in collision_paths if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite an existing map attempt; found " + ", ".join(collisions)
        )

    data_dir = _resolve(cfg.paths.data_dir)
    scaler_path = _resolve(cfg.paths.scaler_path("train"))
    scaler = joblib.load(scaler_path)
    manifest_path = data_dir / "dataset_manifest.json"
    train_metadata_path = data_dir / "train_metadata.json"
    val_metadata_path = data_dir / "val_metadata.json"
    manifest = json.loads(manifest_path.read_text())
    train_metadata = json.loads(train_metadata_path.read_text())
    val_metadata = json.loads(val_metadata_path.read_text())
    x_train, y_train = topology._load_pairs(data_dir / "train.csv", scaler, device)
    x_val, y_val = topology._load_pairs(data_dir / "val.csv", scaler, device)
    source_physical_baseline = _source_physical_replay_baseline(
        source_model,
        x_val,
        y_val,
        torch.tensor(cfg.training.loss_weights, dtype=torch.float32, device=device),
    )
    train_sample_weights, replay_component_provenance = (
        topology._manifest_component_sample_weights(
            train_metadata,
            row_count=len(x_train),
            overrides=PRIMARY_REPLAY_COMPONENT_WEIGHTS,
            device=device,
        )
    )
    cache = _rebuild_latent_cache(
        model,
        scaler,
        manifest,
        train_metadata,
        val_metadata,
        x_train,
        y_train,
        x_val,
        y_val,
    )
    trainer_sha = _sha256(Path(__file__).resolve())
    attempt_manifest_path = logs_dir / "map_attempt_manifest.json"
    attempt_manifest: dict[str, Any] = {
        "status": "running",
        "semantics": "fresh_map_epoch_zero_attempt_not_epoch_59_resume",
        "seed": seed,
        "deterministic_algorithms": True,
        "chart_refined_path": str(chart_path),
        "chart_refined_sha256": chart_sha,
        "chart_history_sha256": _sha256(chart_history_path),
        "primary_v2_source_sha256": source_sha,
        "trainer_sha256": trainer_sha,
        "settings": asdict(settings),
    }
    attempt_manifest_path.write_text(
        json.dumps(attempt_manifest, indent=2, allow_nan=False) + "\n"
    )
    try:
        map_phase = _run_map_phase(
            model,
            cfg,
            cache,
            x_train,
            y_train,
            x_val,
            y_val,
            train_sample_weights,
            models_dir,
            logs_dir,
            settings,
            source_physical_baseline,
        )
    except Exception as error:
        attempt_manifest.update(
            {
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        attempt_manifest_path.write_text(
            json.dumps(attempt_manifest, indent=2, allow_nan=False) + "\n"
        )
        raise
    attempt_manifest["status"] = map_phase["status"]
    attempt_manifest_path.write_text(
        json.dumps(attempt_manifest, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        "experiment": cfg.experiment_name,
        "mode": "fresh_map_attempt_from_preserved_chart_refined",
        "status": map_phase["status"],
        "seed": seed,
        "configuration": cfg.model_dump(mode="json"),
        "trainer_sha256": trainer_sha,
        "chart_source": {
            "path": str(models_dir),
            "checkpoint_sha256": chart_sha,
            "sidecar_sha256": _sha256(chart_sidecar_path),
            "chart_history_sha256": _sha256(chart_history_path),
        },
        "primary_v2_source": {
            "path": str(source_dir),
            "checkpoint_sha256": source_sha,
            "physical_replay_baseline": source_physical_baseline,
        },
        "data_provenance": {
            "train_csv_sha256": _sha256(data_dir / "train.csv"),
            "validation_csv_sha256": _sha256(data_dir / "val.csv"),
            "manifest_sha256": _sha256(manifest_path),
            "train_metadata_sha256": _sha256(train_metadata_path),
            "validation_metadata_sha256": _sha256(val_metadata_path),
            "scaler_sha256": _sha256(scaler_path),
            "training_replay_component_weights": replay_component_provenance,
        },
        "latent_cache_rebuild": _cache_report(cache),
        "map_phase": map_phase,
        "derived_cmgdb_bounds": _derived_bounds(cache, cfg.cmgdb.bounds_epsilon_frac),
        "attempt_manifest": str(attempt_manifest_path),
        "limitations": [
            "this is a numerical map repair, not a Conley-index certificate",
            "long rollout horizons are evaluation-only during this safety attempt",
            "CMGDB must use the derived post-chart bounds",
        ],
    }
    (output_dir / "alternating_map_resume_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def _validated_map_training_state(
    path: Path,
    cfg: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Load a complete post-projection state boundary for true continuation."""

    if not path.is_file():
        raise FileNotFoundError(f"map training state is missing: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    required = {
        "format_version",
        "next_epoch",
        "next_stage",
        "rollout_stage_start_epoch",
        "settings",
        "architecture",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "last_anchor_projection",
        "best_model_state_dict",
        "best_map",
        "best_map_epoch",
        "map_history",
        "rng",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"map training state is incomplete: missing {missing}")
    if int(state["format_version"]) not in (1, 2):
        raise ValueError(f"unsupported map training state format {state['format_version']}")
    if state["architecture"] != cfg.arch.model_dump(mode="json"):
        raise ValueError("map training state architecture does not match config")
    settings = MapTrainingSettings(**state["settings"])
    _validate_map_settings(settings)
    next_epoch = int(state["next_epoch"])
    if next_epoch < 1:
        raise ValueError("map training state must follow at least one completed epoch")
    history = state["map_history"]
    if not history or int(history[-1]["epoch"]) != next_epoch - 1:
        raise ValueError("map history does not end at the saved continuation boundary")
    rollout_start = state["rollout_stage_start_epoch"]
    expected_stage = (
        "rollout"
        if rollout_start is not None and next_epoch >= int(rollout_start)
        else "topology_repair"
    )
    if state["next_stage"] != expected_stage:
        raise ValueError("saved next_stage conflicts with rollout-stage boundary")
    return state


def _input_provenance(
    cfg: Any,
    *,
    data_dir: Path,
    scaler_path: Path,
    manifest_path: Path,
    train_metadata_path: Path,
    val_metadata_path: Path,
    replay_component_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Fingerprint every mutable input that affects continuation gradients."""

    return {
        "schema_version": 1,
        "data_provenance": {
            "train_csv_sha256": _sha256(data_dir / "train.csv"),
            "validation_csv_sha256": _sha256(data_dir / "val.csv"),
            "manifest_sha256": _sha256(manifest_path),
            "train_metadata_sha256": _sha256(train_metadata_path),
            "validation_metadata_sha256": _sha256(val_metadata_path),
            "scaler_sha256": _sha256(scaler_path),
            "training_replay_component_weights": replay_component_provenance,
        },
        "training_configuration": cfg.training.model_dump(mode="json"),
    }


def _source_input_provenance(
    state: dict[str, Any],
    state_path: Path,
    base_output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover pinned inputs from v2 state or the completed v1 summary."""

    if int(state["format_version"]) >= 2:
        provenance = state.get("input_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("format-v2 state lacks pinned input provenance")
        return provenance, {"source": "format_v2_atomic_training_state"}
    candidates = (
        state_path.parent.parent / "alternating_map_resume_summary.json",
        base_output_dir / "alternating_map_resume_summary.json",
    )
    summary_path = next((path for path in candidates if path.is_file()), None)
    if summary_path is None:
        raise FileNotFoundError(
            "format-v1 continuation requires its alternating_map_resume_summary.json "
            "to pin mutable data and training inputs"
        )
    summary = json.loads(summary_path.read_text())
    summary_sha = _sha256(summary_path)
    if summary_sha != PRESERVED_MAP_RESUME_SUMMARY_SHA256:
        raise ValueError(
            "format-v1 source summary hash mismatch: "
            f"expected {PRESERVED_MAP_RESUME_SUMMARY_SHA256}, got {summary_sha}"
        )
    if "data_provenance" not in summary or "configuration" not in summary:
        raise ValueError("format-v1 source summary lacks input provenance")
    provenance = {
        "schema_version": 1,
        "data_provenance": summary["data_provenance"],
        "training_configuration": summary["configuration"]["training"],
    }
    return provenance, {
        "source": "format_v1_completed_attempt_summary",
        "path": str(summary_path),
        "sha256": summary_sha,
    }


def _validate_input_provenance(
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> None:
    """Refuse continuation if any gradient-defining input has changed."""

    if expected == observed:
        return
    mismatched_sections = sorted(
        key
        for key in set(expected) | set(observed)
        if expected.get(key) != observed.get(key)
    )
    raise ValueError(
        "continuation input provenance mismatch in sections "
        f"{mismatched_sections}; data/training changes require a fresh attempt"
    )


def continue_map_from_training_state(
    config_name: str,
    *,
    device_name: str,
    additional_epochs: int,
    output_layer_learning_rate: float | None,
    training_state_path: str | Path | None,
    continuation_output_dir: str | Path | None,
    expected_source_sha256: str,
    expected_chart_sha256: str,
    expected_training_state_sha256: str | None = None,
) -> dict[str, Any]:
    """Continue a valid map state with a constrained final-affine update.

    The source bundle is read-only.  Adam, its scheduler, the last model (not
    the independently selected best model), the historical best candidate,
    rollout-stage state, and every RNG stream are restored from the same
    atomic checkpoint boundary.  New artifacts live under a collision-proof
    continuation directory.
    """

    if additional_epochs < 1:
        raise ValueError("additional continuation epochs must be positive")
    if output_layer_learning_rate is not None and output_layer_learning_rate <= 0.0:
        raise ValueError("output-layer learning rate must be positive")
    cfg = load_config(config_name)
    if len(cfg.seeds) != 1:
        raise ValueError("map continuation requires exactly one seed")
    seed = int(cfg.seeds[0])
    device = torch.device(device_name)
    torch.use_deterministic_algorithms(True)
    base_output_dir = _resolve(cfg.paths.output_dir) / f"seed_{seed}"
    base_models_dir = base_output_dir / "models"
    base_logs_dir = base_output_dir / "logs"
    state_path = (
        _resolve(training_state_path)
        if training_state_path is not None
        else base_models_dir / "map_training_state.pt"
    )
    state_sha_before = _sha256(state_path)
    if (
        expected_training_state_sha256 is not None
        and state_sha_before != expected_training_state_sha256
    ):
        raise ValueError(
            "map training state hash mismatch: "
            f"expected {expected_training_state_sha256}, got {state_sha_before}"
        )
    state = _validated_map_training_state(state_path, cfg, device)
    settings = MapTrainingSettings(**state["settings"])
    effective_output_learning_rate, output_learning_rate_source = (
        _resolve_continuation_output_learning_rate(
            state,
            output_layer_learning_rate,
        )
    )
    start_epoch = int(state["next_epoch"])
    end_epoch = start_epoch + additional_epochs

    chart_path = base_models_dir / "chart_refined.pt"
    chart_sidecar_path = base_models_dir / "chart_refined.json"
    chart_history_path = base_logs_dir / "chart_history.json"
    for path in (chart_path, chart_sidecar_path, chart_history_path):
        if not path.is_file():
            raise FileNotFoundError(f"preserved chart artifact is missing: {path}")
    chart_sha = _sha256(chart_path)
    if chart_sha != expected_chart_sha256:
        raise ValueError(
            "chart_refined checkpoint hash mismatch: "
            f"expected {expected_chart_sha256}, got {chart_sha}"
        )

    source_dir = _resolve(cfg.training.warm_start_checkpoint_dir)
    source_path = source_dir / "autoencoder.pt"
    source_sha = _sha256(source_path)
    if source_sha != expected_source_sha256:
        raise ValueError(
            "primary-v2 source hash mismatch: "
            f"expected {expected_source_sha256}, got {source_sha}"
        )
    source_model, source_arch = load_any_checkpoint(source_dir, map_location=device)
    if source_arch.model_dump() != cfg.arch.model_dump():
        raise ValueError("primary-v2 source architecture does not match continuation config")
    source_model = source_model.to(device).eval()
    reference_model, chart_arch = load_any_checkpoint(
        base_models_dir,
        basename="chart_refined",
        map_location=device,
    )
    if chart_arch.model_dump() != cfg.arch.model_dump():
        raise ValueError("chart_refined architecture does not match continuation config")
    reference_model = reference_model.to(device).eval()
    model = copy.deepcopy(reference_model).to(device)
    model.load_state_dict(state["model_state_dict"])
    reference_state = reference_model.state_dict()
    for name, value in model.state_dict().items():
        if name.startswith("latent_map."):
            continue
        if not torch.equal(value, reference_state[name]):
            raise ValueError(
                "training state changed the frozen chart parameter "
                f"{name!r}; refusing to reuse chart-dependent caches"
            )

    if continuation_output_dir is None:
        continuation_dir = (
            base_output_dir
            / "continuations"
            / f"map_state_e{start_epoch}_to_e{end_epoch}_nullspace"
        )
    else:
        continuation_dir = _resolve(continuation_output_dir)
    if continuation_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite continuation output: {continuation_dir}"
        )

    data_dir = _resolve(cfg.paths.data_dir)
    scaler_path = _resolve(cfg.paths.scaler_path("train"))
    scaler = joblib.load(scaler_path)
    manifest_path = data_dir / "dataset_manifest.json"
    train_metadata_path = data_dir / "train_metadata.json"
    val_metadata_path = data_dir / "val_metadata.json"
    manifest = json.loads(manifest_path.read_text())
    train_metadata = json.loads(train_metadata_path.read_text())
    val_metadata = json.loads(val_metadata_path.read_text())
    x_train, y_train = topology._load_pairs(data_dir / "train.csv", scaler, device)
    x_val, y_val = topology._load_pairs(data_dir / "val.csv", scaler, device)
    data_weights = torch.tensor(
        cfg.training.loss_weights,
        dtype=torch.float32,
        device=device,
    )
    source_physical_baseline = _source_physical_replay_baseline(
        source_model,
        x_val,
        y_val,
        data_weights,
    )
    train_sample_weights, replay_component_provenance = (
        topology._manifest_component_sample_weights(
            train_metadata,
            row_count=len(x_train),
            overrides=PRIMARY_REPLAY_COMPONENT_WEIGHTS,
            device=device,
        )
    )
    current_input_provenance = _input_provenance(
        cfg,
        data_dir=data_dir,
        scaler_path=scaler_path,
        manifest_path=manifest_path,
        train_metadata_path=train_metadata_path,
        val_metadata_path=val_metadata_path,
        replay_component_provenance=replay_component_provenance,
    )
    expected_input_provenance, input_provenance_source = _source_input_provenance(
        state,
        state_path,
        base_output_dir,
    )
    _validate_input_provenance(
        expected_input_provenance,
        current_input_provenance,
    )
    cache = _rebuild_latent_cache(
        model,
        scaler,
        manifest,
        train_metadata,
        val_metadata,
        x_train,
        y_train,
        x_val,
        y_val,
    )

    continuation_dir.mkdir(parents=True, exist_ok=False)
    models_dir = continuation_dir / "models"
    logs_dir = continuation_dir / "logs"
    models_dir.mkdir()
    logs_dir.mkdir()
    trainer_sha = _sha256(Path(__file__).resolve())
    state_provenance = {
        "path": str(state_path),
        "sha256": state_sha_before,
        "format_version": int(state["format_version"]),
        "next_epoch": start_epoch,
    }
    attempt_manifest_path = logs_dir / "continuation_attempt_manifest.json"
    attempt_manifest: dict[str, Any] = {
        "status": "running",
        "semantics": "exact_last_state_continuation_with_new_constrained_output_update",
        "seed": seed,
        "deterministic_algorithms": True,
        "source_training_state": state_provenance,
        "source_best_epoch": int(state["best_map_epoch"]),
        "restores": [
            "last_model_state",
            "hidden_adam_state",
            "scheduler_state",
            "rollout_stage_state",
            "historical_best_candidate",
            "python_numpy_torch_rng_state",
        ],
        "start_epoch": start_epoch,
        "additional_epochs": additional_epochs,
        "end_epoch_exclusive": end_epoch,
        "requested_output_layer_learning_rate": output_layer_learning_rate,
        "effective_output_layer_learning_rate": effective_output_learning_rate,
        "output_learning_rate_source": output_learning_rate_source,
        "input_provenance": current_input_provenance,
        "input_provenance_source": input_provenance_source,
        "trainer_sha256": trainer_sha,
    }
    attempt_manifest_path.write_text(
        json.dumps(attempt_manifest, indent=2, allow_nan=False) + "\n"
    )
    try:
        map_phase = _run_map_phase(
            model,
            cfg,
            cache,
            x_train,
            y_train,
            x_val,
            y_val,
            train_sample_weights,
            models_dir,
            logs_dir,
            settings,
            source_physical_baseline,
            resume_state=state,
            reference_model=reference_model,
            continuation_epochs=additional_epochs,
            constrained_output_learning_rate=effective_output_learning_rate,
            source_training_state=state_provenance,
            input_provenance=current_input_provenance,
        )
    except Exception as error:
        attempt_manifest.update(
            {
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        attempt_manifest_path.write_text(
            json.dumps(attempt_manifest, indent=2, allow_nan=False) + "\n"
        )
        raise
    finally:
        state_sha_after = _sha256(state_path)
        if state_sha_after != state_sha_before:
            raise RuntimeError("source map training state changed during continuation")

    attempt_manifest.update(
        {
            "status": map_phase["status"],
            "source_training_state_sha256_after": state_sha_after,
            "source_training_state_unchanged": True,
        }
    )
    attempt_manifest_path.write_text(
        json.dumps(attempt_manifest, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        "experiment": cfg.experiment_name,
        "mode": "exact_map_training_state_continuation_with_nullspace_output_sgd",
        "status": map_phase["status"],
        "seed": seed,
        "trainer_sha256": trainer_sha,
        "source_training_state": {
            **state_provenance,
            "sha256_after": state_sha_after,
            "unchanged": True,
        },
        "input_provenance": current_input_provenance,
        "input_provenance_source": input_provenance_source,
        "output_learning_rate": {
            "requested": output_layer_learning_rate,
            "effective_fixed": effective_output_learning_rate,
            "source": output_learning_rate_source,
        },
        "chart_source": {
            "path": str(base_models_dir),
            "checkpoint_sha256": chart_sha,
            "sidecar_sha256": _sha256(chart_sidecar_path),
            "chart_history_sha256": _sha256(chart_history_path),
        },
        "primary_v2_source": {
            "path": str(source_dir),
            "checkpoint_sha256": source_sha,
            "physical_replay_baseline": source_physical_baseline,
        },
        "data_provenance": current_input_provenance["data_provenance"],
        "latent_cache_rebuild": _cache_report(cache),
        "map_phase": map_phase,
        "derived_cmgdb_bounds": _derived_bounds(cache, cfg.cmgdb.bounds_epsilon_frac),
        "attempt_manifest": str(attempt_manifest_path),
        "limitations": [
            (
                "the continuation starts from the last state at epoch "
                f"{start_epoch - 1}, not the independently selected epoch-"
                f"{int(state['best_map_epoch'])} best model"
            ),
            "the constrained output step changes the optimizer beginning at the continuation boundary",
            "nullspace feasibility and exact projection are numerical constraints, not a Conley-index certificate",
            "the final candidate still requires a recurrent-root census and a new CMGDB audit",
        ],
    }
    summary_path = continuation_dir / "map_continuation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chart-epochs", type=int, default=1500)
    parser.add_argument("--map-epochs", type=int, default=4000)
    parser.add_argument("--chart-scope", choices=("edge", "all"), default="edge")
    parser.add_argument("--chart-learning-rate", type=float, default=2e-6)
    parser.add_argument("--map-learning-rate", type=float, default=5e-8)
    parser.add_argument("--secant-sample-count", type=int, default=2048)
    parser.add_argument("--secant-neighbors", type=int, default=4)
    parser.add_argument("--margin-expansion", type=float, default=1.05)
    parser.add_argument("--minimum-margin", type=float, default=0.03)
    parser.add_argument("--anti-fold-fraction", type=float, default=0.8)
    parser.add_argument("--reconstruction-weight", type=float, default=100.0)
    parser.add_argument("--prediction-weight", type=float, default=20.0)
    parser.add_argument("--semiconjugacy-weight", type=float, default=20.0)
    parser.add_argument("--reference-weight", type=float, default=50.0)
    parser.add_argument("--margin-weight", type=float, default=5.0)
    parser.add_argument("--anti-fold-weight", type=float, default=20.0)
    parser.add_argument("--inverse-weight", type=float, default=10.0)
    parser.add_argument("--reconstruction-ratio-limit", type=float, default=1.02)
    parser.add_argument("--prediction-ratio-limit", type=float, default=1.05)
    parser.add_argument("--semiconjugacy-ratio-limit", type=float, default=1.05)
    parser.add_argument("--drift-limit", type=float, default=0.03)
    parser.add_argument("--cross-role-ratio-floor", type=float, default=0.95)
    parser.add_argument("--secant-p01-ratio-floor", type=float, default=0.8)
    parser.add_argument("--chart-improvement-fraction", type=float, default=0.001)
    parser.add_argument("--map-anchor-weight", type=float, default=10.0)
    parser.add_argument("--map-characteristic-weight", type=float, default=5.0)
    parser.add_argument("--map-topology-weight", type=float, default=20.0)
    parser.add_argument("--map-trust-weight", type=float, default=10.0)
    parser.add_argument("--rollout-weight", type=float, default=0.001)
    parser.add_argument("--rollout-ratio-limit", type=float, default=1.5)
    parser.add_argument(
        "--rollout-absolute-limit",
        type=float,
        default=SOURCE_ROLLOUT_ABSOLUTE_LIMIT,
        help="absolute held-out long-rollout promotion ceiling from the source-chart audit",
    )
    parser.add_argument("--rollout-backprop-steps", type=int, default=8)
    parser.add_argument("--rollout-short-epochs", type=int, default=250)
    parser.add_argument("--rollout-medium-max-horizon", type=int, default=16)
    parser.add_argument("--rollout-learning-rate", type=float, default=1e-8)
    parser.add_argument("--rollout-min-topology-epochs", type=int, default=250)
    parser.add_argument("--spectral-start-epoch", type=int, default=100)
    parser.add_argument("--spectral-ramp-epochs", type=int, default=2000)
    parser.add_argument("--per-term-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--gradient-diagnostics-every", type=int, default=50)
    parser.add_argument("--gradient-diagnostics-threshold", type=float, default=1000000.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--stable-ceiling", type=float, default=0.98)
    parser.add_argument("--unstable-floor", type=float, default=1.05)
    parser.add_argument("--jury-buffer", type=float, default=0.005)
    parser.add_argument("--validation-ratio-limit", type=float, default=1.05)
    parser.add_argument("--anchor-acceptance", type=float, default=0.001)
    parser.add_argument("--characteristic-acceptance", type=float, default=0.05)
    parser.add_argument("--expected-source-sha256", default=PRIMARY_V2_SHA256)
    parser.add_argument(
        "--expected-chart-sha256",
        default=PRESERVED_CHART_REFINED_SHA256,
        help="required hash for the preserved chart_refined checkpoint",
    )
    parser.add_argument(
        "--resume-chart-refined",
        action="store_true",
        help=(
            "start a fresh map epoch-zero attempt from the preserved chart_refined "
            "checkpoint without repeating chart training"
        ),
    )
    parser.add_argument(
        "--continue-map-training-state",
        action="store_true",
        help=(
            "restore the complete saved map state and add constrained-output "
            "continuation epochs in a new isolated output directory"
        ),
    )
    parser.add_argument(
        "--continuation-epochs",
        type=int,
        default=1000,
        help="number of epochs to add after the saved next_epoch boundary",
    )
    parser.add_argument(
        "--map-training-state",
        default=None,
        help="optional source map_training_state.pt; defaults to this run's models directory",
    )
    parser.add_argument(
        "--continuation-output-dir",
        default=None,
        help="optional new output directory; an existing path is always rejected",
    )
    parser.add_argument(
        "--output-layer-learning-rate",
        type=float,
        default=None,
        help=(
            "fixed no-momentum nullspace-SGD rate; by default it follows the "
            "restored hidden-Adam learning rate"
        ),
    )
    parser.add_argument(
        "--expected-training-state-sha256",
        default=None,
        help="optional source training-state hash pin",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="load and rebuild all caches, run gates, and perform no writes or training",
    )
    args = parser.parse_args()
    selected_modes = sum(
        bool(mode)
        for mode in (
            args.resume_chart_refined,
            args.continue_map_training_state,
            args.validate_only,
        )
    )
    if selected_modes > 1:
        parser.error(
            "--resume-chart-refined, --continue-map-training-state, and "
            "--validate-only are mutually exclusive"
        )
    if args.continue_map_training_state:
        result = continue_map_from_training_state(
            args.config,
            device_name=args.device,
            additional_epochs=args.continuation_epochs,
            output_layer_learning_rate=args.output_layer_learning_rate,
            training_state_path=args.map_training_state,
            continuation_output_dir=args.continuation_output_dir,
            expected_source_sha256=args.expected_source_sha256,
            expected_chart_sha256=args.expected_chart_sha256,
            expected_training_state_sha256=args.expected_training_state_sha256,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    if args.resume_chart_refined:
        settings = MapTrainingSettings(
            epochs=args.map_epochs,
            learning_rate=args.map_learning_rate,
            rollout_learning_rate=args.rollout_learning_rate,
            rollout_min_topology_epochs=args.rollout_min_topology_epochs,
            anchor_weight=args.map_anchor_weight,
            characteristic_weight=args.map_characteristic_weight,
            topology_weight=args.map_topology_weight,
            trust_weight=args.map_trust_weight,
            rollout_weight=args.rollout_weight,
            rollout_ratio_limit=args.rollout_ratio_limit,
            rollout_absolute_limit=args.rollout_absolute_limit,
            rollout_backprop_steps=args.rollout_backprop_steps,
            rollout_short_epochs=args.rollout_short_epochs,
            rollout_medium_max_horizon=args.rollout_medium_max_horizon,
            spectral_start_epoch=args.spectral_start_epoch,
            spectral_ramp_epochs=args.spectral_ramp_epochs,
            per_term_gradient_clip_norm=args.per_term_gradient_clip_norm,
            gradient_diagnostics_every=args.gradient_diagnostics_every,
            gradient_diagnostics_threshold=args.gradient_diagnostics_threshold,
            eval_every=args.eval_every,
            stable_ceiling=args.stable_ceiling,
            unstable_floor=args.unstable_floor,
            jury_buffer=args.jury_buffer,
            validation_ratio_limit=args.validation_ratio_limit,
            source_reconstruction_ratio_limit=args.reconstruction_ratio_limit,
            source_prediction_ratio_limit=args.prediction_ratio_limit,
            anchor_acceptance=args.anchor_acceptance,
            characteristic_acceptance=args.characteristic_acceptance,
        )
        result = resume_map_from_chart_refined(
            args.config,
            device_name=args.device,
            settings=settings,
            expected_source_sha256=args.expected_source_sha256,
            expected_chart_sha256=args.expected_chart_sha256,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    result = run(
        args.config,
        device_name=args.device,
        chart_epochs=args.chart_epochs,
        map_epochs=args.map_epochs,
        chart_scope=args.chart_scope,
        chart_learning_rate=args.chart_learning_rate,
        map_learning_rate=args.map_learning_rate,
        secant_sample_count=args.secant_sample_count,
        secant_neighbors=args.secant_neighbors,
        margin_expansion=args.margin_expansion,
        minimum_margin=args.minimum_margin,
        anti_fold_fraction=args.anti_fold_fraction,
        chart_weights={
            "reconstruction": args.reconstruction_weight,
            "prediction": args.prediction_weight,
            "semiconjugacy": args.semiconjugacy_weight,
            "reference": args.reference_weight,
            "margin": args.margin_weight,
            "anti_fold": args.anti_fold_weight,
            "inverse": args.inverse_weight,
        },
        reconstruction_ratio_limit=args.reconstruction_ratio_limit,
        prediction_ratio_limit=args.prediction_ratio_limit,
        semiconjugacy_ratio_limit=args.semiconjugacy_ratio_limit,
        drift_limit=args.drift_limit,
        cross_role_ratio_floor=args.cross_role_ratio_floor,
        secant_p01_ratio_floor=args.secant_p01_ratio_floor,
        chart_improvement_fraction=args.chart_improvement_fraction,
        map_anchor_weight=args.map_anchor_weight,
        map_characteristic_weight=args.map_characteristic_weight,
        map_topology_weight=args.map_topology_weight,
        map_trust_weight=args.map_trust_weight,
        rollout_weight=args.rollout_weight,
        rollout_ratio_limit=args.rollout_ratio_limit,
        rollout_absolute_limit=args.rollout_absolute_limit,
        rollout_backprop_steps=args.rollout_backprop_steps,
        rollout_short_epochs=args.rollout_short_epochs,
        rollout_medium_max_horizon=args.rollout_medium_max_horizon,
        rollout_learning_rate=args.rollout_learning_rate,
        rollout_min_topology_epochs=args.rollout_min_topology_epochs,
        spectral_start_epoch=args.spectral_start_epoch,
        spectral_ramp_epochs=args.spectral_ramp_epochs,
        per_term_gradient_clip_norm=args.per_term_gradient_clip_norm,
        gradient_diagnostics_every=args.gradient_diagnostics_every,
        gradient_diagnostics_threshold=args.gradient_diagnostics_threshold,
        eval_every=args.eval_every,
        stable_ceiling=args.stable_ceiling,
        unstable_floor=args.unstable_floor,
        jury_buffer=args.jury_buffer,
        validation_ratio_limit=args.validation_ratio_limit,
        anchor_acceptance=args.anchor_acceptance,
        characteristic_acceptance=args.characteristic_acceptance,
        expected_source_sha256=args.expected_source_sha256,
        validate_only=args.validate_only,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
