#!/usr/bin/env python3
"""Train a smooth, topology-supervised Leslie3D latent map.

This experiment keeps the deterministic invariant-aware encoder and decoder
fixed, replaces only the 2x64 ReLU latent map by a 2x64 GELU map, and copies
all compatible linear weights.  The new map is fit against every curated
transition plus fixed encoded recurrent-phase anchors.  Anchor equalities use
an augmented Lagrangian; only after they close does the script ramp in exact
``create_graph=True`` autograd monodromy constraints.

The recurrent catalogue and multiplier targets are supervised prior
information.  Passing the numerical gates is deliberately not described as a
computer-assisted Conley-index proof.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from latentdynamics.config import ArchConfig, load_config
from latentdynamics.models import build_autoencoder
from latentdynamics.training import load_any_checkpoint, save_checkpoint

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "leslie3d_invariant_aware_smooth"
OBJECT_ORDER = ("P0", "P1", "S2", "S4", "p_star", "origin")
PERIODS = {"P0": 4, "P1": 4, "S2": 2, "S4": 4, "p_star": 1, "origin": 1}
EXPECTED_UNSTABLE = {"P0": 0, "P1": 0, "S2": 1, "S4": 1, "p_star": 1, "origin": 1}
EXPECTED_UNSTABLE_SIGN = {"S2": -1.0, "S4": 1.0, "p_star": -1.0, "origin": 1.0}
RECURRENT_EXCLUSION_PERIODS = (1, 2, 4)
DEFAULT_RECURRENT_EXCLUSION_WEIGHT = 0.0
TRAJECTORY_SHADOWING_COMPONENTS = (
    "saddle_tangent_transition_tubes",
    "origin_positive_cone_transition_fan",
    "audited_origin_p_star_s2_transition_tubes",
)
DEFAULT_TRAJECTORY_SHADOWING_WEIGHT = 0.0
DEFAULT_TRAJECTORY_SHADOWING_HORIZON_GROUPS = {
    "short": (1, 2, 3, 4),
    "medium": (7, 8, 15, 16, 31, 32),
    "long": (63, 64, 127, 128, 255, 256, 319, 320),
}

# Trace and determinant of the desired two-dimensional return-map
# characteristic polynomials.  P0/P1/S2/S4 come from the dynamically relevant
# physical multipliers; p_star/origin use the useful two-dimensional quotient
# signatures established by the deterministic base fit.
TARGET_CHARACTERISTIC: dict[str, dict[str, float]] = {
    "P0": {"trace": -0.9303681694, "determinant": -0.0371570324, "weight": 1.0},
    "P1": {"trace": 1.5798299874, "determinant": 0.9218607202, "weight": 5.0},
    "S2": {"trace": -0.6485571575, "determinant": -0.8161446641, "weight": 5.0},
    "S4": {"trace": 2.1757975675, "determinant": 0.4541942545, "weight": 1.0},
    "p_star": {"trace": -0.28901877, "determinant": -0.90460031, "weight": 0.5},
    "origin": {"trace": 2.58249596, "determinant": -0.34064272, "weight": 0.5},
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else CODE_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _load_pairs(path: Path, scaler: Any, device: torch.device) -> tuple[Tensor, Tensor]:
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    x = torch.tensor(scaler.transform(values[:, :3]), dtype=torch.float32, device=device)
    y = torch.tensor(scaler.transform(values[:, 3:6]), dtype=torch.float32, device=device)
    return x, y


def _assert_smooth_transition(source: ArchConfig, target: ArchConfig) -> None:
    if (source.high_dims, source.low_dims) != (target.high_dims, target.low_dims):
        raise ValueError("source and smooth dimensions do not match")
    for component in ("encoder", "decoder"):
        if source.component(component) != target.component(component):
            raise ValueError(f"smooth config changes the frozen {component} architecture")
    source_map = source.component("latent_map")
    target_map = target.component("latent_map")
    if source_map.hidden_shapes != (64, 64) or target_map.hidden_shapes != (64, 64):
        raise ValueError("smooth experiment requires the same 2x64 latent MLP")
    if source_map.out_activation != target_map.out_activation:
        raise ValueError("smooth experiment must preserve the latent output activation")
    if source_map.activation != "relu" or target_map.activation != "gelu":
        raise ValueError("expected a ReLU-to-GELU latent hidden-activation transition")


def _transfer_components(
    source: nn.Module,
    target: nn.Module,
    *,
    gelu_sharpness: float = 1.0,
) -> list[str]:
    target.encoder.load_state_dict(source.encoder.state_dict(), strict=True)
    target.decoder.load_state_dict(source.decoder.state_dict(), strict=True)
    # ReLU and GELU have no parameters, so strict loading first transfers every
    # compatible Linear tensor while retaining the target activations.
    target.latent_map.load_state_dict(source.latent_map.state_dict(), strict=True)
    if gelu_sharpness <= 0.0:
        raise ValueError("gelu_sharpness must be positive")
    if gelu_sharpness != 1.0:
        # GELU(k a)/k converges to ReLU(a). For a two-hidden-layer MLP, these
        # exact rescalings make the smooth map a sharp, function-preserving
        # approximation to the ReLU warm start:
        #
        #   W0,b0 <- k(W0,b0); b1 <- k b1; W2 <- W2/k.
        #
        # This sharply reduces the initial mismatch of an unscaled GELU
        # transplant; the run summary records the actual independent
        # validation ratio rather than baking an empirical claim into code.
        source_layers = list(source.latent_map.net.children())
        target_layers = list(target.latent_map.net.children())
        source_linear = [layer for layer in source_layers if isinstance(layer, nn.Linear)]
        target_linear = [layer for layer in target_layers if isinstance(layer, nn.Linear)]
        if len(source_linear) != 3 or len(target_linear) != 3:
            raise ValueError("sharp GELU transfer requires exactly three Linear layers")
        k = float(gelu_sharpness)
        with torch.no_grad():
            target_linear[0].weight.copy_(k * source_linear[0].weight)
            target_linear[0].bias.copy_(k * source_linear[0].bias)
            target_linear[1].weight.copy_(source_linear[1].weight)
            target_linear[1].bias.copy_(k * source_linear[1].bias)
            target_linear[2].weight.copy_(source_linear[2].weight / k)
            target_linear[2].bias.copy_(source_linear[2].bias)
    return sorted(target.latent_map.state_dict())


def _assert_frozen_chart_matches(source: nn.Module, candidate: nn.Module) -> None:
    """Reject a continuation checkpoint that silently changes the fixed chart."""
    for component_name in ("encoder", "decoder"):
        source_state = getattr(source, component_name).state_dict()
        candidate_state = getattr(candidate, component_name).state_dict()
        if source_state.keys() != candidate_state.keys():
            raise ValueError(
                f"initial smooth checkpoint changes the frozen {component_name} state keys"
            )
        changed = [
            name
            for name in source_state
            if not torch.equal(source_state[name], candidate_state[name])
        ]
        if changed:
            raise ValueError(
                f"initial smooth checkpoint changes frozen {component_name} tensors: {changed}"
            )


def _replay_losses(
    model: nn.Module,
    z: Tensor,
    z_next: Tensor,
    x: Tensor,
    y: Tensor,
    weights: Tensor,
    sample_weights: Tensor | None = None,
) -> dict[str, Tensor]:
    z_pred = model.latent_map(z)
    x_hat = model.decoder(z)
    y_hat = model.decoder(z_pred)
    cycle = model.encoder(y_hat)

    def weighted_mse(predicted: Tensor, expected: Tensor) -> Tensor:
        if sample_weights is None:
            return nn.functional.mse_loss(predicted, expected)
        per_row = torch.mean((predicted - expected) ** 2, dim=1)
        return torch.sum(sample_weights * per_row) / torch.sum(sample_weights)

    losses = {
        "reconstruction": weighted_mse(x_hat, x),
        "prediction": weighted_mse(y_hat, y),
        "semiconjugacy": weighted_mse(z_pred, z_next),
        "cycle": weighted_mse(cycle, z_pred),
    }
    losses["total"] = sum(weights[index] * losses[name] for index, name in enumerate(losses))
    return losses


def _parse_replay_component_weights(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        name, separator, raw_weight = value.partition("=")
        if not separator or not name or not raw_weight:
            raise ValueError(
                f"invalid replay component weight {value!r}; expected NAME=POSITIVE_WEIGHT"
            )
        if name in parsed:
            raise ValueError(f"duplicate replay component weight for {name!r}")
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(f"invalid replay component weight {value!r}") from error
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"replay component weight must be finite and positive: {value!r}")
        parsed[name] = weight
    return parsed


def _manifest_component_sample_weights(
    metadata: dict[str, Any],
    *,
    row_count: int,
    overrides: dict[str, float],
    device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    components = metadata.get("components", [])
    names = {str(component["name"]) for component in components}
    unknown = sorted(set(overrides) - names)
    if unknown:
        raise ValueError(f"unknown replay component weight names: {unknown}")
    weights = torch.ones(row_count, dtype=torch.float32, device=device)
    covered = torch.zeros(row_count, dtype=torch.bool, device=device)
    report: dict[str, Any] = {}
    for component in components:
        name = str(component["name"])
        start = int(component["row_start_inclusive"])
        stop = int(component["row_stop_exclusive"])
        if start < 0 or stop <= start or stop > row_count:
            raise ValueError(f"invalid row interval for replay component {name!r}")
        if bool(torch.any(covered[start:stop])):
            raise ValueError(f"overlapping row interval for replay component {name!r}")
        weight = float(overrides.get(name, 1.0))
        weights[start:stop] = weight
        covered[start:stop] = True
        report[name] = {
            "row_start_inclusive": start,
            "row_stop_exclusive": stop,
            "row_count": stop - start,
            "weight": weight,
            "explicit_override": name in overrides,
        }
    if not bool(torch.all(covered)):
        raise ValueError("replay component metadata does not cover every training row exactly once")
    return weights, report


def _validate_trajectory_shadowing_horizon_groups(
    horizon_groups: dict[str, list[int] | tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    if not horizon_groups:
        raise ValueError("trajectory-shadowing horizon groups cannot be empty")
    validated: dict[str, tuple[int, ...]] = {}
    claimed: dict[int, str] = {}
    for raw_name, raw_horizons in horizon_groups.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("trajectory-shadowing horizon group names cannot be empty")
        horizons = tuple(raw_horizons)
        if not horizons:
            raise ValueError(f"trajectory-shadowing horizon group {name!r} cannot be empty")
        if any(isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) for horizon in horizons):
            raise ValueError(
                f"trajectory-shadowing horizons in group {name!r} must be integers"
            )
        normalized = tuple(sorted(int(horizon) for horizon in horizons))
        if normalized[0] <= 0:
            raise ValueError("trajectory-shadowing horizons must be positive")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"duplicate trajectory-shadowing horizon in group {name!r}")
        overlaps = sorted(set(normalized) & set(claimed))
        if overlaps:
            other = sorted({claimed[horizon] for horizon in overlaps})
            raise ValueError(
                f"trajectory-shadowing horizons {overlaps} occur in multiple groups: "
                f"{[*other, name]}"
            )
        for horizon in normalized:
            claimed[horizon] = name
        validated[name] = normalized
    return validated


def _prepare_trajectory_shadowing_blocks(
    x: Tensor,
    y: Tensor,
    encoded_x: Tensor,
    encoded_y: Tensor,
    metadata: dict[str, Any],
    horizon_groups: dict[str, list[int] | tuple[int, ...]],
    *,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild and encode selected time-major trajectory blocks exactly once."""
    groups = _validate_trajectory_shadowing_horizon_groups(horizon_groups)
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError(f"{split_name} trajectory-shadowing pairs must be equal 2-D tensors")
    if encoded_x.ndim != 2 or encoded_y.shape != encoded_x.shape or len(encoded_x) != len(x):
        raise ValueError(
            f"{split_name} encoded trajectory-shadowing pairs must be equal 2-D tensors"
        )
    components_by_name: dict[str, dict[str, Any]] = {}
    for component in metadata.get("components", []):
        name = str(component.get("name", ""))
        if name not in TRAJECTORY_SHADOWING_COMPONENTS:
            continue
        if name in components_by_name:
            raise ValueError(
                f"duplicate {split_name} trajectory-shadowing component metadata for {name!r}"
            )
        components_by_name[name] = component

    blocks: list[dict[str, Any]] = []
    component_report: dict[str, Any] = {}
    for name in TRAJECTORY_SHADOWING_COMPONENTS:
        component = components_by_name.get(name)
        if component is None:
            continue
        start = int(component["row_start_inclusive"])
        stop = int(component["row_stop_exclusive"])
        trajectories = int(component["trajectories"])
        steps = int(component["steps"])
        row_count = stop - start
        if start < 0 or stop <= start or stop > len(x):
            raise ValueError(
                f"invalid {split_name} trajectory-shadowing row interval for {name!r}"
            )
        if trajectories <= 0 or steps <= 0 or row_count != trajectories * steps:
            raise ValueError(
                f"{split_name} trajectory-shadowing component {name!r} has inconsistent "
                "rows, trajectories, or steps"
            )
        if "rows" in component and int(component["rows"]) != row_count:
            raise ValueError(
                f"{split_name} trajectory-shadowing component {name!r} has a stale row count"
            )

        current = x[start:stop].reshape(steps, trajectories, x.shape[1])
        successor = y[start:stop].reshape(steps, trajectories, y.shape[1])
        continuity = torch.zeros((), dtype=x.dtype, device=x.device)
        if steps > 1:
            continuity = torch.max(torch.abs(successor[:-1] - current[1:]))
        if not torch.isfinite(continuity) or float(continuity) > 1e-6:
            raise ValueError(
                f"{split_name} trajectory-shadowing component {name!r} is not a "
                "continuous time-major trajectory block"
            )
        encoded_current = encoded_x[start:stop].reshape(steps, trajectories, -1)
        encoded_successor = encoded_y[start:stop].reshape(steps, trajectories, -1)
        latent_states = torch.cat((encoded_current, encoded_successor[-1:]), dim=0).detach()
        valid_groups = {
            group_name: tuple(horizon for horizon in horizons if horizon <= steps)
            for group_name, horizons in groups.items()
        }
        valid_groups = {
            group_name: horizons
            for group_name, horizons in valid_groups.items()
            if horizons
        }
        if not valid_groups:
            raise ValueError(
                f"no configured trajectory-shadowing horizon fits {split_name} component {name!r}"
            )
        blocks.append(
            {
                "name": name,
                "steps": steps,
                "trajectories": trajectories,
                "latent_states": latent_states,
                "horizon_groups": valid_groups,
            }
        )
        component_report[name] = {
            "row_start_inclusive": start,
            "row_stop_exclusive": stop,
            "pair_rows": row_count,
            "trajectories": trajectories,
            "steps": steps,
            "encoded_true_state_count": (steps + 1) * trajectories,
            "scaled_state_continuity_max_abs": float(continuity),
            "horizon_groups": {
                group_name: list(horizons)
                for group_name, horizons in valid_groups.items()
            },
        }
    if not blocks:
        raise ValueError(
            f"{split_name} metadata contains none of the supported trajectory-shadowing components"
        )
    return blocks, {
        "split": split_name,
        "row_order": "time_major_step_then_trajectory",
        "components": component_report,
        "missing_supported_components": [
            name for name in TRAJECTORY_SHADOWING_COMPONENTS if name not in components_by_name
        ],
    }


def _trajectory_shadowing_loss(
    latent_map: nn.Module,
    blocks: list[dict[str, Any]],
    *,
    diagnostics: bool,
) -> tuple[Tensor, dict[str, Any]]:
    """Uniformly average rollout MSE over components, groups, and horizons."""
    if not blocks:
        raise ValueError("trajectory-shadowing loss requires at least one trajectory block")
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
                "steps": block["steps"],
                "trajectories": block["trajectories"],
                "balanced_mse": float(component_loss.detach()),
                "group_mse": {
                    group_name: float(loss.detach())
                    for group_name, loss in group_losses.items()
                },
                "horizons": {
                    str(horizon): {
                        "mse": float(losses_by_horizon[horizon].detach()),
                        "starting_windows": (
                            (block["steps"] + 1 - horizon) * block["trajectories"]
                        ),
                    }
                    for horizon in sorted(selected_horizons)
                },
            }
    total = torch.stack(component_losses).mean()
    report: dict[str, Any] = {}
    if diagnostics:
        report = {
            "enabled": True,
            "balanced_mse": float(total.detach()),
            "component_weighting": "uniform",
            "horizon_group_weighting_within_component": "uniform",
            "horizon_weighting_within_group": "uniform",
            "sample_weighting_within_horizon": "uniform_mean",
            "components": component_report,
        }
    return total, report


def _phase_latents(
    model: nn.Module,
    scaler: Any,
    manifest: dict[str, Any],
    device: torch.device,
) -> dict[str, Tensor]:
    targets: dict[str, Tensor] = {}
    with torch.no_grad():
        for name in OBJECT_ORDER:
            points = np.asarray(manifest["known_objects"][name]["points"], dtype=np.float64)
            scaled = torch.tensor(scaler.transform(points), dtype=torch.float32, device=device)
            targets[name] = model.encoder(scaled).detach()
    return targets


def _phase_scales(targets: dict[str, Tensor]) -> dict[str, Tensor]:
    """Use each anchor's nearest distinct recurrent anchor as its fixed scale."""
    flat = torch.cat([targets[name] for name in OBJECT_ORDER])
    distances = torch.cdist(flat, flat)
    distances.fill_diagonal_(float("inf"))
    nearest = torch.clamp(torch.min(distances, dim=1).values, min=0.02)
    scales: dict[str, Tensor] = {}
    cursor = 0
    for name in OBJECT_ORDER:
        count = len(targets[name])
        scales[name] = nearest[cursor : cursor + count].detach()
        cursor += count
    return scales


def _anchor_residuals(
    latent_map: nn.Module,
    targets: dict[str, Tensor],
    scales: dict[str, Tensor],
) -> tuple[Tensor, dict[str, list[dict[str, Tensor]]]]:
    normalized: list[Tensor] = []
    details: dict[str, list[dict[str, Tensor]]] = {}
    for name in OBJECT_ORDER:
        points = targets[name]
        object_details: list[dict[str, Tensor]] = []
        for phase in range(len(points)):
            successor = points[(phase + 1) % len(points)]
            residual = latent_map(points[phase]) - successor
            scaled = residual / scales[name][phase]
            normalized.append(scaled)
            object_details.append(
                {"residual": residual, "normalized": scaled, "scale": scales[name][phase]}
            )
        details[name] = object_details
    return torch.cat(normalized), details


@torch.no_grad()
def _project_anchor_equalities(
    latent_map: nn.Module,
    targets: dict[str, Tensor],
) -> dict[str, float]:
    """Project the output layer onto every fixed phase-successor equality.

    With 64 penultimate features, the final affine layer has 130 coefficients
    and the catalogue supplies only 32 scalar equalities.  The Moore--Penrose
    correction below is the minimum-Frobenius-change affine update satisfying
    those equations for the current hidden features.  Repeating it after each
    optimizer step is projected-gradient training on the exact anchor
    manifold, rather than a large-penalty approximation.
    """

    children = list(latent_map.net.children())
    if len(children) < 2 or not isinstance(children[-2], nn.Linear):
        raise TypeError("expected the penultimate latent-map module to be Linear")
    if not isinstance(children[-1], nn.Tanh):
        raise TypeError("exact anchor projection currently requires a Tanh output")
    output_layer = children[-2]
    points = torch.cat([targets[name] for name in OBJECT_ORDER])
    successors = torch.cat([torch.roll(targets[name], shifts=-1, dims=0) for name in OBJECT_ORDER])
    features = points
    for layer in children[:-2]:
        features = layer(features)
    design = torch.cat([features, torch.ones((len(features), 1), device=features.device)], dim=1)
    desired_preactivation = torch.atanh(torch.clamp(successors, -0.999999, 0.999999))
    current = torch.cat([output_layer.weight.T, output_layer.bias.unsqueeze(0)], dim=0)

    design64 = design.double()
    current64 = current.double()
    desired64 = desired_preactivation.double()
    residual_before = design64 @ current64 - desired64
    correction = torch.linalg.pinv(design64) @ (-residual_before)
    projected = current64 + correction
    output_layer.weight.copy_(projected[:-1].T.to(output_layer.weight.dtype))
    output_layer.bias.copy_(projected[-1].to(output_layer.bias.dtype))
    # Report the residual of the float32 parameters actually installed, rather
    # than the idealized float64 projection before its final cast.
    applied = torch.cat([output_layer.weight.T, output_layer.bias.unsqueeze(0)], dim=0).double()
    residual_after = design64 @ applied - desired64
    singular_values = torch.linalg.svdvals(design64)
    return {
        "preactivation_residual_before_max_abs": float(torch.max(torch.abs(residual_before))),
        "preactivation_residual_after_max_abs": float(torch.max(torch.abs(residual_after))),
        "output_parameter_correction_l2": float(torch.linalg.vector_norm(correction)),
        "design_largest_singular_value": float(singular_values[0]),
        "design_smallest_singular_value": float(singular_values[-1]),
    }


def _max_anchor_normalized_l2(residuals: Tensor) -> float:
    return float(torch.max(torch.linalg.vector_norm(residuals.reshape(-1, 2), dim=1)).detach())


def _point_jacobian(latent_map: nn.Module, point: Tensor) -> Tensor:
    """Return D(latent_map)(point) with a graph back to the map parameters."""
    differentiable_point = point.detach().clone().requires_grad_(True)
    value = latent_map(differentiable_point)
    rows = [
        torch.autograd.grad(
            value[index],
            differentiable_point,
            create_graph=True,
            retain_graph=True,
        )[0]
        for index in range(value.numel())
    ]
    return torch.stack(rows)


def _anchored_monodromy(latent_map: nn.Module, phases: Tensor) -> Tensor:
    """Exact AD product DG(z[p-1]) ... DG(z[0]) at fixed phase anchors."""
    monodromy = torch.eye(phases.shape[1], dtype=phases.dtype, device=phases.device)
    for phase in phases:
        monodromy = _point_jacobian(latent_map, phase) @ monodromy
    return monodromy


def _topology_term(
    name: str,
    trace: Tensor,
    determinant: Tensor,
    *,
    stable_ceiling: float,
    unstable_floor: float,
    jury_buffer: float,
) -> Tensor:
    if EXPECTED_UNSTABLE[name] == 0:
        scaled_trace = trace / stable_ceiling
        scaled_det = determinant / stable_ceiling**2
        # Strict second-order Jury conditions after scaling the disk radius.
        jury = torch.stack(
            (
                1.0 - scaled_det,
                1.0 - scaled_trace + scaled_det,
                1.0 + scaled_trace + scaled_det,
            )
        )
        return torch.mean(torch.relu(jury_buffer - jury) ** 2)

    discriminant = trace**2 - 4.0 * determinant
    root_gap = torch.sqrt(torch.clamp(discriminant, min=1e-8))
    root_minus = 0.5 * (trace - root_gap)
    root_plus = 0.5 * (trace + root_gap)
    if EXPECTED_UNSTABLE_SIGN[name] < 0.0:
        stable, oriented_unstable = root_plus, -root_minus
    else:
        stable, oriented_unstable = root_minus, root_plus
    return (
        torch.relu(1e-4 - discriminant) ** 2
        + torch.relu(torch.abs(stable) - stable_ceiling) ** 2
        + torch.relu(unstable_floor - oriented_unstable) ** 2
    )


def _spectral_terms(
    latent_map: nn.Module,
    targets: dict[str, Tensor],
    *,
    stable_ceiling: float,
    unstable_floor: float,
    jury_buffer: float,
    diagnostics: bool,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    characteristic: list[Tensor] = []
    topology: list[Tensor] = []
    report: dict[str, Any] = {}
    for name in OBJECT_ORDER:
        monodromy = _anchored_monodromy(latent_map, targets[name])
        trace = torch.trace(monodromy)
        determinant = torch.linalg.det(monodromy)
        target = TARGET_CHARACTERISTIC[name]
        char_term = ((trace - target["trace"]) / max(1.0, abs(target["trace"]))) ** 2 + (
            (determinant - target["determinant"]) / max(1.0, abs(target["determinant"]))
        ) ** 2
        characteristic.append(target["weight"] * char_term)
        topology.append(
            _topology_term(
                name,
                trace,
                determinant,
                stable_ceiling=stable_ceiling,
                unstable_floor=unstable_floor,
                jury_buffer=jury_buffer,
            )
        )
        if diagnostics:
            eigenvalues = torch.linalg.eigvals(monodromy.detach())
            order = torch.argsort(torch.abs(eigenvalues))
            eigenvalues = eigenvalues[order]
            report[name] = {
                "monodromy": monodromy.detach().cpu().tolist(),
                "trace": float(trace.detach()),
                "determinant": float(determinant.detach()),
                "trace_relative_error": float(
                    abs(trace.detach() - target["trace"]) / max(1.0, abs(target["trace"]))
                ),
                "determinant_relative_error": float(
                    abs(determinant.detach() - target["determinant"])
                    / max(1.0, abs(target["determinant"]))
                ),
                "eigenvalues": [
                    {
                        "real": float(value.real.cpu()),
                        "imag": float(value.imag.cpu()),
                        "modulus": float(abs(value).cpu()),
                    }
                    for value in eigenvalues
                ],
            }
    return torch.stack(characteristic).mean(), torch.stack(topology).mean(), report


def _role_violation(
    name: str,
    eigenvalues: list[dict[str, float]],
    stable_ceiling: float,
    unstable_floor: float,
) -> float:
    moduli = [value["modulus"] for value in eigenvalues]
    if EXPECTED_UNSTABLE[name] == 0:
        return max(0.0, max(moduli) - stable_ceiling)
    stable, unstable = eigenvalues
    sign = EXPECTED_UNSTABLE_SIGN[name]
    orientation = unstable_floor - sign * unstable["real"]
    return max(
        0.0,
        stable["modulus"] - stable_ceiling,
        unstable_floor - unstable["modulus"],
        orientation,
        abs(unstable["imag"]),
    )


def _global_trust_points(count: int, seed: int, device: torch.device) -> Tensor:
    if count < 4:
        raise ValueError("global_trust_points must be at least four")
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    points = 2.0 * engine.draw(count).to(device=device, dtype=torch.float32) - 1.0
    points[:4] = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=points.dtype,
        device=device,
    )
    return points


def _cmgdb_latent_bounds(
    cfg: Any,
    encoded_pairs: tuple[Tensor, ...],
) -> tuple[Tensor, Tensor, str]:
    """Reproduce the default CMGDB bounds from the frozen encoded data pairs."""
    reference = encoded_pairs[0]
    if cfg.cmgdb.lower_bounds is not None and cfg.cmgdb.upper_bounds is not None:
        lower = torch.tensor(
            cfg.cmgdb.lower_bounds,
            dtype=reference.dtype,
            device=reference.device,
        )
        upper = torch.tensor(
            cfg.cmgdb.upper_bounds,
            dtype=reference.dtype,
            device=reference.device,
        )
        source = "config"
    else:
        encoded = torch.cat(encoded_pairs)
        lower = torch.min(encoded, dim=0).values
        upper = torch.max(encoded, dim=0).values
        buffer = cfg.cmgdb.bounds_epsilon_frac * (upper - lower)
        lower = lower - buffer
        upper = upper + buffer
        source = "encoded_train_and_validation_pairs"
    if not torch.all(torch.isfinite(lower)) or not torch.all(torch.isfinite(upper)):
        raise ValueError("non-finite recurrent-exclusion/CMGDB bounds")
    if torch.any(upper <= lower):
        raise ValueError("collapsed recurrent-exclusion/CMGDB bounds")
    return lower.detach(), upper.detach(), source


def _sample_recurrent_exclusion_probes(
    lower: Tensor,
    upper: Tensor,
    anchors: Tensor,
    *,
    global_count: int,
    local_radius_count: int,
    local_direction_count: int,
    local_min_radius: float,
    local_max_radius: float,
    seed: int,
) -> tuple[Tensor, dict[str, Any]]:
    """Combine global Sobol probes with log-radius probes around every anchor."""
    if global_count < 0:
        raise ValueError("global recurrent-exclusion probe count cannot be negative")
    if local_radius_count < 1 or local_direction_count < 1:
        raise ValueError("local recurrent-exclusion radius/direction counts must be positive")
    if not 0.0 < local_min_radius <= local_max_radius:
        raise ValueError("invalid recurrent-exclusion local radius interval")
    if lower.numel() != 2:
        raise ValueError("local recurrent-exclusion probes currently require 2-D latent space")
    span = upper - lower
    batches: list[Tensor] = []
    if global_count:
        engine = torch.quasirandom.SobolEngine(lower.numel(), scramble=True, seed=seed)
        global_unit = engine.draw(global_count).to(device=lower.device, dtype=lower.dtype)
        batches.append(lower + global_unit * span)

    radii = torch.logspace(
        np.log10(local_min_radius),
        np.log10(local_max_radius),
        local_radius_count,
        device=lower.device,
        dtype=lower.dtype,
    )
    base_angles = (
        2.0
        * torch.pi
        * torch.arange(local_direction_count, device=lower.device, dtype=lower.dtype)
        / local_direction_count
    )
    rotation_engine = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed + 1)
    rotations = (
        2.0
        * torch.pi
        * rotation_engine.draw(len(anchors)).to(device=lower.device, dtype=lower.dtype).squeeze(1)
    )
    local_retained: list[Tensor] = []
    local_candidate_count = 0
    local_counts: list[int] = []
    for anchor, rotation in zip(anchors, rotations, strict=True):
        angles = base_angles + rotation
        directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        offsets = radii[:, None, None] * directions[None, :, :]
        candidates = anchor + offsets.reshape(-1, 2)
        local_candidate_count += len(candidates)
        inside = torch.all(
            (candidates >= lower.unsqueeze(0)) & (candidates <= upper.unsqueeze(0)),
            dim=1,
        )
        retained = candidates[inside]
        local_retained.append(retained)
        local_counts.append(len(retained))
    if local_retained:
        batches.append(torch.cat(local_retained))
    if any(count == 0 for count in local_counts):
        raise ValueError(
            "at least one intended phase has no local recurrent-exclusion probes "
            "inside the CMGDB bounds"
        )
    if not batches:
        raise ValueError("recurrent-exclusion probe set is empty")
    probes = torch.cat(batches)
    if not len(probes):
        raise ValueError("all recurrent-exclusion local probes fell outside the CMGDB bounds")
    return probes.detach(), {
        "samplers": ["scrambled_sobol_global", "log_radius_equiangular_local"],
        "seed": seed,
        "total_retained_count": len(probes),
        "global_count": global_count,
        "local_radius_count": local_radius_count,
        "local_direction_count": local_direction_count,
        "local_min_radius": local_min_radius,
        "local_max_radius": local_max_radius,
        "local_candidate_count": local_candidate_count,
        "local_retained_count": sum(local_counts),
        "local_retained_count_by_anchor": local_counts,
        "local_rejected_outside_bounds": local_candidate_count - sum(local_counts),
    }


def _load_extra_recurrent_probes(
    path: Path,
    *,
    device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    """Load fixed census negatives from a small JSON or CSV point file."""
    if not path.is_file():
        raise FileNotFoundError(f"recurrent-exclusion extra probe file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        census_selection: dict[str, Any] | None = None
        if isinstance(payload, dict):
            if "cycles" in payload:
                selected_cycles = [
                    cycle
                    for cycle in payload["cycles"]
                    if bool(cycle.get("inside_cmgdb_bounds"))
                    and not bool(cycle.get("association", {}).get("is_intended_catalogue_cycle"))
                ]
                selected_points = [
                    point for cycle in selected_cycles for point in cycle.get("phase_points", [])
                ]
                census_selection = {
                    "schema_version": payload.get("schema_version"),
                    "source_checkpoint": payload.get("checkpoint"),
                    "source_cmgdb_bounds": payload.get("cmgdb_bounds"),
                    "selected_cycle_count": len(selected_cycles),
                    "selected_cycle_ids": [cycle.get("cycle_id") for cycle in selected_cycles],
                    "selected_least_periods": [
                        cycle.get("least_period") for cycle in selected_cycles
                    ],
                    "selection_rule": (
                        "inside_cmgdb_bounds and not association.is_intended_catalogue_cycle"
                    ),
                }
                payload = selected_points
            elif "points" in payload:
                payload = payload["points"]
            else:
                raise ValueError("extra-probe JSON object must contain a 'points' list")
        if not isinstance(payload, list):
            raise ValueError("extra-probe JSON must be a point list or {'points': [...]}")
        rows: list[list[float]] = []
        for item in payload:
            if isinstance(item, dict):
                if "z0" in item and "z1" in item:
                    rows.append([float(item["z0"]), float(item["z1"])])
                elif "point" in item:
                    rows.append([float(value) for value in item["point"]])
                else:
                    raise ValueError("extra-probe JSON rows need z0/z1 or point")
            else:
                rows.append([float(value) for value in item])
        values = np.asarray(rows, dtype=np.float64)
        file_format = "json"
    elif path.suffix.lower() == ".csv":
        lines = path.read_text().splitlines()
        if not lines:
            raise ValueError("recurrent-exclusion extra probe CSV is empty")
        try:
            [float(value) for value in lines[0].split(",")]
            has_header = False
        except ValueError:
            has_header = True
        values = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1 if has_header else 0,
            ndmin=2,
        )
        file_format = "csv_with_header" if has_header else "csv"
    else:
        raise ValueError("recurrent-exclusion extra probes must use .json or .csv")
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("recurrent-exclusion extra probes must be a nonempty N x 2 array")
    if not np.all(np.isfinite(values)):
        raise ValueError("recurrent-exclusion extra probes contain NaN/Inf")
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return tensor, {
        "path": str(path),
        "sha256": _sha256(path),
        "format": file_format,
        "count": len(tensor),
        "census_selection": census_selection if file_format == "json" else None,
    }


def _allowed_recurrent_anchor_sets(
    targets: dict[str, Tensor],
) -> tuple[dict[int, Tensor], dict[str, list[str]]]:
    """For G^p, allow exactly the phases whose least period divides p."""
    allowed: dict[int, Tensor] = {}
    labels: dict[str, list[str]] = {}
    for period in RECURRENT_EXCLUSION_PERIODS:
        names = [name for name in OBJECT_ORDER if period % PERIODS[name] == 0]
        allowed[period] = torch.cat([targets[name] for name in names])
        labels[str(period)] = names
    return allowed, labels


def _iterate_probe_batch(latent_map: nn.Module, probes: Tensor, period: int) -> Tensor:
    value = probes
    for _ in range(period):
        value = latent_map(value)
    return value


def _recurrent_exclusion_term(
    latent_map: nn.Module,
    probes: Tensor,
    allowed_anchors: dict[int, Tensor],
    *,
    core_radius: float,
    distance_epsilon: float,
    score_margin: float,
    temperature: float,
    optimization_probe_count: int | None,
    diagnostics: bool,
) -> tuple[Tensor, dict[str, Any]]:
    """Penalize low return/distance scores away from allowed intended roots."""
    period_values: dict[int, dict[str, Tensor]] = {}
    score_columns: list[Tensor] = []
    valid_columns: list[Tensor] = []
    for period in RECURRENT_EXCLUSION_PERIODS:
        returned = _iterate_probe_batch(latent_map, probes, period)
        return_squared = torch.sum((returned - probes) ** 2, dim=1)
        return_l2 = torch.sqrt(torch.clamp(return_squared, min=0.0))
        smooth_return_l2 = torch.sqrt(return_squared + 1e-20)
        anchor_distance = torch.min(torch.cdist(probes, allowed_anchors[period]), dim=1).values
        valid = anchor_distance >= core_radius
        denominator = torch.clamp(anchor_distance, min=distance_epsilon)
        smooth_score = smooth_return_l2 / denominator
        exact_score = return_l2 / denominator
        period_values[period] = {
            "return_l2": return_l2,
            "anchor_distance": anchor_distance,
            "score": exact_score,
            "valid": valid,
        }
        score_columns.append(smooth_score)
        valid_columns.append(valid)
    scores = torch.stack(score_columns, dim=1)
    valid = torch.stack(valid_columns, dim=1)
    if optimization_probe_count is None:
        optimization_probe_count = len(probes)
    if not 0 < optimization_probe_count <= len(probes):
        raise ValueError("invalid recurrent-exclusion optimization probe count")
    optimization_rows = torch.arange(len(probes), device=probes.device) < optimization_probe_count
    optimization_valid = valid & optimization_rows.unsqueeze(1)
    valid_scores = scores[optimization_valid]
    if not len(valid_scores):
        raise ValueError("no recurrent-exclusion probe/period pairs remain outside cores")
    # A global differentiable soft minimum concentrates the gradient on the
    # worst sampled return score instead of diluting a nearby extra root across
    # tens of thousands of safe probes. The unnormalized logsumexp is a
    # conservative lower approximation to the sampled minimum.
    soft_min = -temperature * torch.logsumexp(-valid_scores / temperature, dim=0)
    soft_hinge = temperature * nn.functional.softplus((score_margin - soft_min) / temperature)
    loss = soft_hinge**2
    if not diagnostics:
        return loss, {}

    report_periods: dict[str, Any] = {}
    for period, values in period_values.items():
        period_valid = values["valid"]
        period_scores = values["score"][period_valid].detach()
        returns = values["return_l2"][period_valid].detach()
        anchor_distances = values["anchor_distance"][period_valid].detach()
        if not len(period_scores):
            raise ValueError(f"no recurrent-exclusion probes remain outside p={period} cores")
        report_periods[str(period)] = {
            "evaluated_probe_count": len(period_scores),
            "excluded_inside_intended_core_count": len(probes) - len(period_scores),
            "minimum_return_distance_score": float(torch.min(period_scores)),
            "q01_return_distance_score": float(torch.quantile(period_scores, 0.01)),
            "q05_return_distance_score": float(torch.quantile(period_scores, 0.05)),
            "median_return_distance_score": float(torch.median(period_scores)),
            "mean_return_distance_score": float(torch.mean(period_scores)),
            "minimum_raw_return_l2": float(torch.min(returns)),
            "q01_raw_return_l2": float(torch.quantile(returns, 0.01)),
            "q05_raw_return_l2": float(torch.quantile(returns, 0.05)),
            "minimum_distance_to_allowed_anchor": float(torch.min(anchor_distances)),
            "fraction_below_training_score_margin": float(
                torch.mean((period_scores < score_margin).to(torch.float32))
            ),
        }
    exact_minimum = min(value["minimum_return_distance_score"] for value in report_periods.values())
    return loss, {
        "enabled": True,
        "periods": list(RECURRENT_EXCLUSION_PERIODS),
        "probe_count": len(probes),
        "optimization_probe_count": optimization_probe_count,
        "strict_only_fixed_probe_count": len(probes) - optimization_probe_count,
        "intended_core_radius": core_radius,
        "distance_normalization_epsilon": distance_epsilon,
        "training_score_margin": score_margin,
        "softmin_temperature": temperature,
        "soft_min_return_distance_score": float(soft_min.detach()),
        "loss": float(loss.detach()),
        "minimum_return_distance_score_over_all_periods": exact_minimum,
        "by_period": report_periods,
    }


def _recurrent_exclusion_gate(
    report: dict[str, Any],
    *,
    enabled: bool,
    acceptance_score: float,
) -> tuple[bool, float]:
    if not enabled:
        return True, 0.0
    minimum = float(report["minimum_return_distance_score_over_all_periods"])
    passed = np.isfinite(minimum) and minimum >= acceptance_score
    violation = max(0.0, acceptance_score / max(minimum, 1e-15) - 1.0)
    return passed, violation


def _float_losses(losses: dict[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach()) for name, value in losses.items()}


def train(
    config_name: str,
    *,
    device_name: str,
    epochs: int | None,
    learning_rate: float | None,
    transfer_epochs: int,
    transfer_anchor_weight: float,
    constraint_learning_rate: float,
    initial_smooth_checkpoint_dir: str | None,
    initial_smooth_basename: str,
    anchor_projection: bool,
    gelu_sharpness: float,
    global_trust_count: int,
    trust_weight: float,
    replay_component_weights: dict[str, float],
    characteristic_weight: float,
    topology_weight: float,
    trajectory_shadowing_weight: float,
    trajectory_shadowing_horizon_groups: dict[str, list[int] | tuple[int, ...]],
    recurrent_exclusion_weight: float,
    recurrent_exclusion_global_count: int,
    recurrent_exclusion_local_radius_count: int,
    recurrent_exclusion_local_direction_count: int,
    recurrent_exclusion_local_min_radius: float,
    recurrent_exclusion_local_max_radius: float,
    recurrent_exclusion_core_radius: float,
    recurrent_exclusion_distance_epsilon: float,
    recurrent_exclusion_score_margin: float,
    recurrent_exclusion_temperature: float,
    recurrent_exclusion_acceptance_score: float,
    recurrent_exclusion_seed_offset: int,
    recurrent_exclusion_extra_probes: str | None,
    min_closure_epochs: int,
    spectral_start_anchor: float,
    spectral_ramp_epochs: int,
    anchor_rho: float,
    anchor_rho_growth: float,
    anchor_rho_max: float,
    dual_update_every: int,
    rho_update_every: int,
    dual_clip: float,
    stable_ceiling: float,
    unstable_floor: float,
    jury_buffer: float,
    eval_every: int,
    validation_ratio_limit: float,
    anchor_acceptance: float,
    characteristic_acceptance: float,
    global_trust_rmse_limit: float,
) -> dict[str, Any]:
    cfg = load_config(config_name)
    if len(cfg.seeds) != 1:
        raise ValueError("smooth topology training requires exactly one configured seed")
    seed = int(cfg.seeds[0])
    _seed_everything(seed)
    device = torch.device(device_name)
    epoch_limit = cfg.training.epochs if epochs is None else int(epochs)
    step_size = cfg.training.learning_rate if learning_rate is None else float(learning_rate)
    if epoch_limit <= 0 or step_size <= 0.0:
        raise ValueError("epochs and learning rate must be positive")
    if (
        transfer_epochs < 0
        or transfer_epochs >= epoch_limit
        or transfer_anchor_weight < 0.0
        or constraint_learning_rate <= 0.0
        or min_closure_epochs < 0
        or spectral_ramp_epochs < 1
        or eval_every < 1
        or dual_update_every < 1
        or rho_update_every < 1
    ):
        raise ValueError("invalid staging/evaluation interval")
    if anchor_rho <= 0.0 or anchor_rho_growth < 1.0 or anchor_rho_max < anchor_rho:
        raise ValueError("invalid augmented-Lagrangian penalty settings")
    if (
        trust_weight < 0.0
        or characteristic_weight < 0.0
        or topology_weight < 0.0
        or trajectory_shadowing_weight < 0.0
        or recurrent_exclusion_weight < 0.0
        or dual_clip <= 0.0
        or stable_ceiling <= 0.0
        or unstable_floor <= stable_ceiling
        or jury_buffer < 0.0
        or validation_ratio_limit <= 0.0
        or anchor_acceptance <= 0.0
        or characteristic_acceptance <= 0.0
        or global_trust_rmse_limit <= 0.0
    ):
        raise ValueError("invalid loss weight, spectral margin, or acceptance limit")
    trajectory_shadowing_enabled = trajectory_shadowing_weight > 0.0
    validated_shadowing_horizon_groups = _validate_trajectory_shadowing_horizon_groups(
        trajectory_shadowing_horizon_groups
    )
    recurrent_exclusion_enabled = recurrent_exclusion_weight > 0.0
    if recurrent_exclusion_extra_probes is not None and not recurrent_exclusion_enabled:
        raise ValueError(
            "recurrent-exclusion extra probes require a positive recurrent-exclusion weight"
        )
    if recurrent_exclusion_enabled and (
        recurrent_exclusion_global_count < 0
        or recurrent_exclusion_local_radius_count < 1
        or recurrent_exclusion_local_direction_count < 1
        or recurrent_exclusion_core_radius <= 0.0
        or recurrent_exclusion_distance_epsilon <= 0.0
        or recurrent_exclusion_distance_epsilon > recurrent_exclusion_core_radius
        or recurrent_exclusion_local_min_radius <= recurrent_exclusion_core_radius
        or recurrent_exclusion_local_max_radius < recurrent_exclusion_local_min_radius
        or recurrent_exclusion_score_margin <= 0.0
        or recurrent_exclusion_temperature <= 0.0
        or recurrent_exclusion_acceptance_score <= 0.0
        or recurrent_exclusion_acceptance_score > recurrent_exclusion_score_margin
    ):
        raise ValueError("invalid off-anchor recurrent-exclusion settings")

    warm_dir = _resolve(cfg.training.warm_start_checkpoint_dir)
    output_dir = _resolve(cfg.paths.output_dir) / f"seed_{seed}"
    preexisting_promoted = {
        path.name: _sha256(path)
        for path in (
            output_dir / "models" / "autoencoder.pt",
            output_dir / "models" / "autoencoder.json",
        )
        if path.is_file()
    }
    data_dir = _resolve(cfg.paths.data_dir)
    scaler_path = _resolve(cfg.paths.scaler_path("train"))
    manifest_path = data_dir / "dataset_manifest.json"
    train_metadata_path = data_dir / "train_metadata.json"
    validation_metadata_path = data_dir / "val_metadata.json"
    scaler = joblib.load(scaler_path)
    manifest = json.loads(manifest_path.read_text())
    train_metadata = json.loads(train_metadata_path.read_text())
    validation_metadata = (
        json.loads(validation_metadata_path.read_text())
        if trajectory_shadowing_enabled
        else None
    )

    source_model, source_arch = load_any_checkpoint(warm_dir, map_location=device)
    _assert_smooth_transition(source_arch, cfg.arch)
    source_model = source_model.to(device).eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    model = build_autoencoder(cfg.arch).to(device)
    transferred_keys = _transfer_components(
        source_model,
        model,
        gelu_sharpness=gelu_sharpness,
    )
    smooth_initialization: dict[str, Any] | None = None
    if initial_smooth_checkpoint_dir is not None:
        initial_dir = _resolve(initial_smooth_checkpoint_dir)
        initial_model, initial_arch = load_any_checkpoint(
            initial_dir,
            basename=initial_smooth_basename,
            map_location=device,
        )
        if initial_arch.model_dump() != cfg.arch.model_dump():
            raise ValueError("initial smooth checkpoint architecture does not match config")
        _assert_frozen_chart_matches(source_model, initial_model)
        model.load_state_dict(initial_model.state_dict(), strict=True)
        initial_path = initial_dir / f"{initial_smooth_basename}.pt"
        initial_sidecar = initial_dir / f"{initial_smooth_basename}.json"
        smooth_initialization = {
            "semantics": "model_state_only_optimizer_scheduler_dual_and_stage_reset",
            "basename": initial_smooth_basename,
            "path": str(initial_path),
            "sha256": _sha256(initial_path),
            "architecture_sidecar_path": str(initial_sidecar),
            "architecture_sidecar_sha256": _sha256(initial_sidecar),
        }
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    model.decoder.eval()
    model.latent_map.train()

    x_train, y_train = _load_pairs(data_dir / "train.csv", scaler, device)
    x_val, y_val = _load_pairs(data_dir / "val.csv", scaler, device)
    train_sample_weights, replay_component_provenance = _manifest_component_sample_weights(
        train_metadata,
        row_count=len(x_train),
        overrides=replay_component_weights,
        device=device,
    )
    effective_train_sample_weights = train_sample_weights if replay_component_weights else None
    with torch.no_grad():
        z_train, z_train_next = model.encoder(x_train), model.encoder(y_train)
        z_val, z_val_next = model.encoder(x_val), model.encoder(y_val)
    train_shadowing_blocks: list[dict[str, Any]] = []
    validation_shadowing_blocks: list[dict[str, Any]] = []
    trajectory_shadowing_provenance: dict[str, Any] = {
        "enabled": trajectory_shadowing_enabled,
        "weight": trajectory_shadowing_weight,
        "supported_components": list(TRAJECTORY_SHADOWING_COMPONENTS),
        "configured_horizon_groups": {
            name: list(horizons)
            for name, horizons in validated_shadowing_horizon_groups.items()
        },
        "true_state_latents": (
            "fixed encoder outputs computed once before optimization and detached"
        ),
        "optimized_component": "latent_map_only",
        "balancing": (
            "uniform mean over components; within each component, uniform mean over "
            "nonempty horizon groups and then over horizons; each horizon MSE is a "
            "uniform mean over its valid starting windows"
        ),
    }
    if trajectory_shadowing_enabled:
        assert validation_metadata is not None
        train_shadowing_blocks, train_shadowing_report = (
            _prepare_trajectory_shadowing_blocks(
                x_train,
                y_train,
                z_train,
                z_train_next,
                train_metadata,
                validated_shadowing_horizon_groups,
                split_name="train",
            )
        )
        validation_shadowing_blocks, validation_shadowing_report = (
            _prepare_trajectory_shadowing_blocks(
                x_val,
                y_val,
                z_val,
                z_val_next,
                validation_metadata,
                validated_shadowing_horizon_groups,
                split_name="validation",
            )
        )
        trajectory_shadowing_provenance.update(
            {
                "train": train_shadowing_report,
                "validation": validation_shadowing_report,
            }
        )
    else:
        trajectory_shadowing_provenance["reason"] = "trajectory_shadowing_weight_is_zero"
    targets = _phase_latents(model, scaler, manifest, device)
    scales = _phase_scales(targets)
    recurrent_anchors = torch.cat([targets[name] for name in OBJECT_ORDER])
    allowed_recurrent_anchors, allowed_recurrent_labels = _allowed_recurrent_anchor_sets(targets)
    recurrent_probes: Tensor | None = None
    recurrent_optimization_probe_count: int | None = None
    recurrent_probe_provenance: dict[str, Any] = {
        "enabled": recurrent_exclusion_enabled,
        "periods": list(RECURRENT_EXCLUSION_PERIODS),
        "allowed_object_roles_by_period": allowed_recurrent_labels,
    }
    if recurrent_exclusion_enabled:
        recurrent_lower, recurrent_upper, recurrent_bounds_source = _cmgdb_latent_bounds(
            cfg,
            (z_train, z_val, z_train_next, z_val_next),
        )
        extra_probes: Tensor | None = None
        extra_provenance: dict[str, Any] | None = None
        if recurrent_exclusion_extra_probes is not None:
            extra_probe_path = _resolve(recurrent_exclusion_extra_probes)
            extra_probes, extra_provenance = _load_extra_recurrent_probes(
                extra_probe_path,
                device=device,
            )
            census_selection = extra_provenance.get("census_selection")
            census_checkpoint = (
                census_selection.get("source_checkpoint") if census_selection is not None else None
            )
            census_checkpoint_sha = (
                census_checkpoint.get("sha256") if isinstance(census_checkpoint, dict) else None
            )
            initialization_sha = (
                smooth_initialization.get("sha256") if smooth_initialization is not None else None
            )
            extra_provenance["matches_initial_smooth_checkpoint"] = (
                census_checkpoint_sha == initialization_sha
                if census_checkpoint_sha is not None and initialization_sha is not None
                else None
            )
            inside = torch.all(
                (extra_probes >= recurrent_lower.unsqueeze(0))
                & (extra_probes <= recurrent_upper.unsqueeze(0)),
                dim=1,
            )
            if not bool(torch.all(inside)):
                outside_rows = torch.nonzero(~inside, as_tuple=False).flatten().cpu().tolist()
                raise ValueError(
                    "recurrent-exclusion extra probes must lie inside the actual "
                    f"CMGDB bounds; outside rows={outside_rows}"
                )
        local_probe_centers = (
            torch.cat((recurrent_anchors, extra_probes))
            if extra_probes is not None
            else recurrent_anchors
        )
        recurrent_probes, sampling_provenance = _sample_recurrent_exclusion_probes(
            recurrent_lower,
            recurrent_upper,
            local_probe_centers,
            global_count=recurrent_exclusion_global_count,
            local_radius_count=recurrent_exclusion_local_radius_count,
            local_direction_count=recurrent_exclusion_local_direction_count,
            local_min_radius=recurrent_exclusion_local_min_radius,
            local_max_radius=recurrent_exclusion_local_max_radius,
            seed=seed + recurrent_exclusion_seed_offset,
        )
        recurrent_optimization_probe_count = len(recurrent_probes)
        if extra_probes is not None:
            recurrent_probes = torch.cat((recurrent_probes, extra_probes))
            sampling_provenance["fixed_extra_probes"] = extra_provenance
            sampling_provenance["total_retained_count"] = len(recurrent_probes)
            sampling_provenance["local_extra_center_count"] = len(extra_probes)
        else:
            sampling_provenance["fixed_extra_probes"] = None
            sampling_provenance["local_extra_center_count"] = 0
        recurrent_probe_provenance.update(
            {
                "bounds_source": recurrent_bounds_source,
                "bounds_lower": recurrent_lower.cpu().tolist(),
                "bounds_upper": recurrent_upper.cpu().tolist(),
                "sampling": sampling_provenance,
                "intended_phase_anchor_count": len(recurrent_anchors),
                "optimization_probe_count": recurrent_optimization_probe_count,
                "intended_phase_anchor_order": [
                    f"{name}[{phase}]"
                    for name in OBJECT_ORDER
                    for phase in range(len(targets[name]))
                ],
                "local_probe_center_order": [
                    *[
                        f"{name}[{phase}]"
                        for name in OBJECT_ORDER
                        for phase in range(len(targets[name]))
                    ],
                    *[
                        f"fixed_extra[{index}]"
                        for index in range(0 if extra_probes is None else len(extra_probes))
                    ],
                ],
            }
        )
    projection_initial = (
        _project_anchor_equalities(model.latent_map, targets) if anchor_projection else None
    )
    projection_latest = projection_initial
    trust_global = _global_trust_points(global_trust_count, seed + 17, device)
    with torch.no_grad():
        trust_train_reference = source_model.latent_map(z_train)
        trust_global_reference = source_model.latent_map(trust_global)

    weights = torch.tensor(cfg.training.loss_weights, dtype=torch.float32, device=device)
    optimizer = Adam(model.latent_map.parameters(), lr=step_size)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.scheduler_factor,
        patience=cfg.training.lr_patience,
        threshold=cfg.training.scheduler_threshold,
        min_lr=cfg.training.scheduler_min_lr,
    )
    with torch.no_grad():
        baseline_val = _replay_losses(source_model, z_val, z_val_next, x_val, y_val, weights)
    baseline_val_total = float(baseline_val["total"])

    anchor_count = sum(len(targets[name]) for name in OBJECT_ORDER)
    dual = torch.zeros(anchor_count * cfg.arch.low_dims, dtype=torch.float32, device=device)
    rho = float(anchor_rho)
    last_rho_anchor: float | None = None
    spectral_start_epoch: int | None = None

    def trust_losses() -> tuple[Tensor, Tensor]:
        replay = nn.functional.mse_loss(model.latent_map(z_train), trust_train_reference)
        global_value = nn.functional.mse_loss(
            model.latent_map(trust_global), trust_global_reference
        )
        return replay, global_value

    def evaluate(epoch: int) -> dict[str, Any]:
        model.latent_map.eval()
        with torch.no_grad():
            data = _replay_losses(model, z_val, z_val_next, x_val, y_val, weights)
            residuals, details = _anchor_residuals(model.latent_map, targets, scales)
            replay_trust, global_trust = trust_losses()
            if trajectory_shadowing_enabled:
                trajectory_shadowing, trajectory_shadowing_report = (
                    _trajectory_shadowing_loss(
                        model.latent_map,
                        validation_shadowing_blocks,
                        diagnostics=True,
                    )
                )
            else:
                trajectory_shadowing = torch.zeros((), device=device)
                trajectory_shadowing_report = {
                    "enabled": False,
                    "reason": "trajectory_shadowing_weight_is_zero",
                }
            if recurrent_exclusion_enabled:
                assert recurrent_probes is not None
                recurrent_loss, recurrent_report = _recurrent_exclusion_term(
                    model.latent_map,
                    recurrent_probes,
                    allowed_recurrent_anchors,
                    core_radius=recurrent_exclusion_core_radius,
                    distance_epsilon=recurrent_exclusion_distance_epsilon,
                    score_margin=recurrent_exclusion_score_margin,
                    temperature=recurrent_exclusion_temperature,
                    optimization_probe_count=recurrent_optimization_probe_count,
                    diagnostics=True,
                )
            else:
                recurrent_loss = torch.zeros((), device=device)
                recurrent_report = {
                    "enabled": False,
                    "periods": list(RECURRENT_EXCLUSION_PERIODS),
                    "reason": "recurrent_exclusion_weight_is_zero",
                }
        with torch.enable_grad():
            characteristic, topology, spectra = _spectral_terms(
                model.latent_map,
                targets,
                stable_ceiling=stable_ceiling,
                unstable_floor=unstable_floor,
                jury_buffer=jury_buffer,
                diagnostics=True,
            )
        anchor_report: dict[str, list[dict[str, float]]] = {}
        for name in OBJECT_ORDER:
            anchor_report[name] = [
                {
                    "phase": phase,
                    "scale": float(item["scale"]),
                    "one_step_l2": float(torch.linalg.vector_norm(item["residual"])),
                    "normalized_one_step_l2": float(torch.linalg.vector_norm(item["normalized"])),
                }
                for phase, item in enumerate(details[name])
            ]
        role_violations = {
            name: _role_violation(
                name,
                spectra[name]["eigenvalues"],
                stable_ceiling,
                unstable_floor,
            )
            for name in OBJECT_ORDER
        }
        max_characteristic_error = max(
            max(
                spectra[name]["trace_relative_error"],
                spectra[name]["determinant_relative_error"],
            )
            for name in OBJECT_ORDER
        )
        validation_ratio = float(data["total"]) / baseline_val_total
        max_anchor = _max_anchor_normalized_l2(residuals)
        global_trust_rmse = float(torch.sqrt(global_trust))
        recurrent_gate, recurrent_violation = _recurrent_exclusion_gate(
            recurrent_report,
            enabled=recurrent_exclusion_enabled,
            acceptance_score=recurrent_exclusion_acceptance_score,
        )
        finite = all(
            np.isfinite(value)
            for value in (
                float(data["total"]),
                max_anchor,
                max_characteristic_error,
                global_trust_rmse,
                float(trajectory_shadowing),
                float(recurrent_loss),
                *role_violations.values(),
            )
        )
        gates = {
            "finite_diagnostics": finite,
            "spectral_stage_started": spectral_start_epoch is not None,
            "validation_ratio": validation_ratio <= validation_ratio_limit,
            "fixed_anchor_closure": max_anchor <= anchor_acceptance,
            "characteristic_polynomials": (max_characteristic_error <= characteristic_acceptance),
            "orientation_and_stability_roles": max(role_violations.values()) == 0.0,
            "global_distillation": global_trust_rmse <= global_trust_rmse_limit,
            "off_anchor_recurrent_exclusion": recurrent_gate,
        }
        violations = {
            "validation_ratio": max(0.0, validation_ratio / validation_ratio_limit - 1.0),
            "fixed_anchor_closure": max(0.0, max_anchor / anchor_acceptance - 1.0),
            "characteristic_polynomials": max(
                0.0, max_characteristic_error / characteristic_acceptance - 1.0
            ),
            "orientation_and_stability_roles": max(role_violations.values()),
            "global_distillation": max(0.0, global_trust_rmse / global_trust_rmse_limit - 1.0),
            "spectral_stage": 0.0 if spectral_start_epoch is not None else 1.0,
            "off_anchor_recurrent_exclusion": recurrent_violation,
        }
        model.latent_map.train()
        return {
            "epoch": epoch,
            "replay": _float_losses(data),
            "baseline_validation_total": baseline_val_total,
            "validation_ratio_to_relu_base": validation_ratio,
            "anchor_quadratic": float(torch.mean(residuals**2)),
            "max_anchor_normalized_l2": max_anchor,
            "anchors": anchor_report,
            "characteristic_loss": float(characteristic.detach()),
            "topology_loss": float(topology.detach()),
            "max_characteristic_relative_error": max_characteristic_error,
            "role_margin_violations": role_violations,
            "max_role_margin_violation": max(role_violations.values()),
            "monodromies": spectra,
            "trust_replay_mse": float(replay_trust),
            "trust_global_mse": float(global_trust),
            "trust_global_rmse": global_trust_rmse,
            "trajectory_shadowing_loss": float(trajectory_shadowing),
            "trajectory_shadowing_validation": trajectory_shadowing_report,
            "recurrent_exclusion_loss": float(recurrent_loss),
            "off_anchor_recurrent_exclusion": recurrent_report,
            "spectral_stage_start_epoch": spectral_start_epoch,
            "acceptance_gates": gates,
            "gate_violations": violations,
            "accepted": all(gates.values()),
        }

    def rank(result: dict[str, Any]) -> tuple[float, float, float, float, float]:
        violations = result["gate_violations"]
        failed = sum(not passed for passed in result["acceptance_gates"].values())
        if result["accepted"]:
            # Once every hard gate is satisfied, prefer the checkpoint that
            # most nearly realizes the anchored periodic equations and target
            # characteristic polynomials. Validation loss is only the final
            # tie-breaker; it must not select a looser topological candidate.
            normalized_anchor = result["max_anchor_normalized_l2"] / anchor_acceptance
            normalized_characteristic = (
                result["max_characteristic_relative_error"] / characteristic_acceptance
            )
            return (
                0.0,
                max(normalized_anchor, normalized_characteristic),
                normalized_anchor + normalized_characteristic,
                result["trust_global_rmse"] / global_trust_rmse_limit,
                result["replay"]["total"]
                + trajectory_shadowing_weight * result["trajectory_shadowing_loss"],
            )
        return (
            1.0,
            float(failed),
            max(violations.values()),
            sum(violations.values()),
            result["replay"]["total"]
            + trajectory_shadowing_weight * result["trajectory_shadowing_loss"],
        )

    initial = evaluate(-1)
    best = copy.deepcopy(initial)
    best_epoch = -1
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    no_improve = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    iterator = tqdm(range(epoch_limit))
    for epoch in iterator:
        if epoch == transfer_epochs:
            # Start the constrained phase with fresh Adam moments and a much
            # smaller step. Carrying high-step transfer moments into the
            # augmented Lagrangian caused the deliberately preserved rejected
            # probe to oscillate across the anchor manifold.
            optimizer = Adam(model.latent_map.parameters(), lr=constraint_learning_rate)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=cfg.training.scheduler_factor,
                patience=cfg.training.lr_patience,
                threshold=cfg.training.scheduler_threshold,
                min_lr=cfg.training.scheduler_min_lr,
            )
            # Transfer is an initialization phase, not evidence that the
            # constrained objective has stalled. Start its patience budget
            # afresh when the augmented-Lagrangian phase begins.
            no_improve = 0
        optimizer.zero_grad(set_to_none=True)
        replay = _replay_losses(
            model,
            z_train,
            z_train_next,
            x_train,
            y_train,
            weights,
            sample_weights=effective_train_sample_weights,
        )
        anchor_vector, _ = _anchor_residuals(model.latent_map, targets, scales)
        replay_trust, global_trust = trust_losses()
        if trajectory_shadowing_enabled:
            trajectory_shadowing, _ = _trajectory_shadowing_loss(
                model.latent_map,
                train_shadowing_blocks,
                diagnostics=False,
            )
        else:
            trajectory_shadowing = torch.zeros((), device=device)
        in_transfer = epoch < transfer_epochs
        recurrent_exclusion = torch.zeros((), device=device)
        if in_transfer:
            spectral_ramp = 0.0
            characteristic = torch.zeros((), device=device)
            topology = torch.zeros((), device=device)
            objective = (
                replay["total"]
                + trust_weight * (replay_trust + global_trust)
                + transfer_anchor_weight * torch.mean(anchor_vector**2)
            )
        else:
            if anchor_projection:
                anchor_lagrangian = torch.zeros((), device=device)
            else:
                anchor_lagrangian = torch.mean(dual * anchor_vector) + 0.5 * rho * torch.mean(
                    anchor_vector**2
                )
            if spectral_start_epoch is None:
                spectral_ramp = 0.0
                characteristic = torch.zeros((), device=device)
                topology = torch.zeros((), device=device)
            else:
                spectral_ramp = min(1.0, (epoch - spectral_start_epoch + 1) / spectral_ramp_epochs)
                characteristic, topology, _ = _spectral_terms(
                    model.latent_map,
                    targets,
                    stable_ceiling=stable_ceiling,
                    unstable_floor=unstable_floor,
                    jury_buffer=jury_buffer,
                    diagnostics=False,
                )
                if recurrent_exclusion_enabled:
                    assert recurrent_probes is not None
                    recurrent_exclusion, _ = _recurrent_exclusion_term(
                        model.latent_map,
                        recurrent_probes,
                        allowed_recurrent_anchors,
                        core_radius=recurrent_exclusion_core_radius,
                        distance_epsilon=recurrent_exclusion_distance_epsilon,
                        score_margin=recurrent_exclusion_score_margin,
                        temperature=recurrent_exclusion_temperature,
                        optimization_probe_count=recurrent_optimization_probe_count,
                        diagnostics=False,
                    )
            objective = (
                replay["total"]
                + anchor_lagrangian
                + trust_weight * (replay_trust + global_trust)
                + spectral_ramp
                * (
                    characteristic_weight * characteristic
                    + topology_weight * topology
                    + recurrent_exclusion_weight * recurrent_exclusion
                )
            )
        if trajectory_shadowing_enabled:
            objective = objective + trajectory_shadowing_weight * trajectory_shadowing
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite smooth topology objective at epoch {epoch}")
        objective.backward()
        if cfg.training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.latent_map.parameters(), cfg.training.gradient_clip_norm
            )
        optimizer.step()
        if anchor_projection:
            projection_latest = _project_anchor_equalities(model.latent_map, targets)

        with torch.no_grad():
            updated_anchor, _ = _anchor_residuals(model.latent_map, targets, scales)
            current_anchor = _max_anchor_normalized_l2(updated_anchor)
            constraint_epoch = epoch - transfer_epochs
            if (
                not anchor_projection
                and not in_transfer
                and (constraint_epoch + 1) % dual_update_every == 0
            ):
                dual.add_(rho * updated_anchor).clamp_(-dual_clip, dual_clip)
            if (
                not anchor_projection
                and not in_transfer
                and (constraint_epoch + 1) % rho_update_every == 0
            ):
                if last_rho_anchor is not None and current_anchor > 0.75 * last_rho_anchor:
                    rho = min(anchor_rho_max, anchor_rho_growth * rho)
                last_rho_anchor = current_anchor
            if (
                spectral_start_epoch is None
                and not in_transfer
                and constraint_epoch + 1 >= min_closure_epochs
                and current_anchor <= spectral_start_anchor
            ):
                spectral_start_epoch = epoch + 1

        if (epoch + 1) % eval_every != 0 and epoch + 1 != epoch_limit:
            iterator.set_postfix(
                stage=(
                    "transfer"
                    if in_transfer
                    else "closure"
                    if spectral_start_epoch is None
                    else "spectral"
                ),
                anchor=f"{current_anchor:.2e}",
                rho=f"{rho:.1e}",
            )
            continue

        current = evaluate(epoch)
        if in_transfer:
            selection_score = (
                current["replay"]["total"]
                + transfer_anchor_weight * current["anchor_quadratic"]
                + trust_weight * (current["trust_replay_mse"] + current["trust_global_mse"])
            )
        else:
            selection_score = (
                current["replay"]["total"]
                + current["anchor_quadratic"]
                + trust_weight * (current["trust_replay_mse"] + current["trust_global_mse"])
                + characteristic_weight * current["characteristic_loss"]
                + topology_weight * current["topology_loss"]
                + (
                    recurrent_exclusion_weight * current["recurrent_exclusion_loss"]
                    if spectral_start_epoch is not None
                    else 0.0
                )
            )
        if trajectory_shadowing_enabled:
            selection_score += (
                trajectory_shadowing_weight * current["trajectory_shadowing_loss"]
            )
        scheduler.step(selection_score)
        history.append(
            {
                "epoch": epoch,
                "stage": (
                    "transfer"
                    if in_transfer
                    else "closure"
                    if spectral_start_epoch is None
                    else "spectral"
                ),
                "spectral_ramp": spectral_ramp,
                "selection_score": selection_score,
                "validation_total": current["replay"]["total"],
                "validation_ratio_to_relu_base": current["validation_ratio_to_relu_base"],
                "max_anchor_normalized_l2": current["max_anchor_normalized_l2"],
                "max_characteristic_relative_error": current["max_characteristic_relative_error"],
                "max_role_margin_violation": current["max_role_margin_violation"],
                "trust_global_rmse": current["trust_global_rmse"],
                "trajectory_shadowing_loss": current["trajectory_shadowing_loss"],
                "recurrent_exclusion_loss": current["recurrent_exclusion_loss"],
                "minimum_recurrent_exclusion_score": (
                    current["off_anchor_recurrent_exclusion"].get(
                        "minimum_return_distance_score_over_all_periods"
                    )
                ),
                "accepted": current["accepted"],
                "rho": rho,
                "dual_max_abs": float(torch.max(torch.abs(dual))),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if rank(current) < rank(best):
            best = copy.deepcopy(current)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            no_improve = 0
        elif not in_transfer:
            no_improve += eval_every
            if no_improve >= cfg.training.patience:
                break
        iterator.set_postfix(
            stage=(
                "transfer"
                if in_transfer
                else "closure"
                if spectral_start_epoch is None
                else "spectral"
            ),
            val=f"{current['validation_ratio_to_relu_base']:.3f}",
            anchor=f"{current['max_anchor_normalized_l2']:.2e}",
            char=f"{current['max_characteristic_relative_error']:.2e}",
            role=f"{current['max_role_margin_violation']:.2e}",
        )

    duration = time.perf_counter() - start
    model.load_state_dict(best_state)
    selected = evaluate(best_epoch)
    if best["spectral_stage_start_epoch"] is None:
        # ``evaluate`` closes over the final run stage. Do not retroactively
        # claim that a transfer/initial checkpoint passed through the spectral
        # phase merely because a later checkpoint did.
        selected["spectral_stage_start_epoch"] = None
        selected["acceptance_gates"]["spectral_stage_started"] = False
        selected["gate_violations"]["spectral_stage"] = 1.0
        selected["accepted"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    candidate_paths = save_checkpoint(
        model.cpu(), cfg.arch, models_dir, basename="smooth_candidate"
    )
    promoted_paths: tuple[Path, Path] | None = None
    if selected["accepted"]:
        promoted_paths = save_checkpoint(model, cfg.arch, models_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "smooth_topology_history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        "experiment": cfg.experiment_name,
        "method": (
            "frozen_chart_sharp_gelu_exact_anchor_projection_exact_autograd_monodromy"
            if anchor_projection
            else "frozen_chart_sharp_gelu_augmented_lagrangian_exact_autograd_monodromy"
        ),
        "seed": seed,
        "deterministic_algorithms": True,
        "warm_start": {
            "path": str(warm_dir),
            "autoencoder_sha256": _sha256(warm_dir / "autoencoder.pt"),
            "architecture_sidecar_sha256": _sha256(warm_dir / "autoencoder.json"),
            "source_latent_activation": source_arch.component("latent_map").activation,
        },
        "configuration": cfg.model_dump(mode="json"),
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "target_latent_activation": cfg.arch.component("latent_map").activation,
        "gelu_sharpness": gelu_sharpness,
        "gelu_sharpness_applied_to_effective_initialization": (smooth_initialization is None),
        "transferred_latent_state_keys": transferred_keys,
        "initial_smooth_checkpoint": smooth_initialization,
        "frozen_components": ["encoder", "decoder"],
        "optimized_components": ["latent_map"],
        "train_csv_sha256": _sha256(data_dir / "train.csv"),
        "validation_csv_sha256": _sha256(data_dir / "val.csv"),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "train_metadata_sha256": _sha256(train_metadata_path),
        "validation_metadata_sha256": (
            _sha256(validation_metadata_path) if trajectory_shadowing_enabled else None
        ),
        "scaler_sha256": _sha256(scaler_path),
        "full_batch_train_rows": len(x_train),
        "validation_rows": len(x_val),
        "global_trust_points": global_trust_count,
        "target_characteristic_polynomials": TARGET_CHARACTERISTIC,
        "training_replay_component_weights": replay_component_provenance,
        "trajectory_shadowing": trajectory_shadowing_provenance,
        "off_anchor_recurrent_exclusion": {
            **recurrent_probe_provenance,
            "weight": recurrent_exclusion_weight,
            "core_radius": recurrent_exclusion_core_radius,
            "distance_normalization_epsilon": recurrent_exclusion_distance_epsilon,
            "training_score_margin": recurrent_exclusion_score_margin,
            "softmin_temperature": recurrent_exclusion_temperature,
            "strict_acceptance_score": recurrent_exclusion_acceptance_score,
            "fixed_extra_probe_policy": (
                "exact census points are strict-gate diagnostics; deterministic local "
                "rings centered on them supply optimization gradients"
            ),
            "limitations": [
                "finite probes cannot certify absence of recurrent points between probes",
                "only periods 1, 2, and 4 are tested",
                "the tiny intended-root cores can hide additional roots inside those cores",
                "the objective may move roots outside the sampled bounds rather than destroy them",
                "fixed census negatives do not track an extra root after that root moves",
                "census negatives can be stale when their source checkpoint differs from the cleanup initialization",
                "the radial return norm has zero derivative at an exact root, so fixed roots rely on their local probe rings for the first optimization step",
                "small scores can also flag slow or near-neutral returns that are not roots",
                "small return-distance score is only a numerical recurrence screen, not a Conley certificate",
            ],
        },
        "exact_anchor_projection": {
            "enabled": anchor_projection,
            "initial": projection_initial,
            "last_iteration": projection_latest,
        },
        "hyperparameters": {
            "epochs_requested": epoch_limit,
            "transfer_epochs": transfer_epochs,
            "transfer_learning_rate": step_size,
            "transfer_anchor_weight": transfer_anchor_weight,
            "constraint_learning_rate": constraint_learning_rate,
            "loss_weights": list(cfg.training.loss_weights),
            "replay_component_weight_overrides": replay_component_weights,
            "trust_weight": trust_weight,
            "characteristic_weight": characteristic_weight,
            "topology_weight": topology_weight,
            "trajectory_shadowing_weight": trajectory_shadowing_weight,
            "trajectory_shadowing_horizon_groups": {
                name: list(horizons)
                for name, horizons in validated_shadowing_horizon_groups.items()
            },
            "recurrent_exclusion_weight": recurrent_exclusion_weight,
            "recurrent_exclusion_global_count": recurrent_exclusion_global_count,
            "recurrent_exclusion_local_radius_count": (recurrent_exclusion_local_radius_count),
            "recurrent_exclusion_local_direction_count": (
                recurrent_exclusion_local_direction_count
            ),
            "recurrent_exclusion_local_min_radius": (recurrent_exclusion_local_min_radius),
            "recurrent_exclusion_local_max_radius": (recurrent_exclusion_local_max_radius),
            "recurrent_exclusion_seed_offset": recurrent_exclusion_seed_offset,
            "recurrent_exclusion_extra_probes": recurrent_exclusion_extra_probes,
            "min_closure_epochs": min_closure_epochs,
            "spectral_start_anchor": spectral_start_anchor,
            "spectral_ramp_epochs": spectral_ramp_epochs,
            "anchor_rho_initial": anchor_rho,
            "anchor_rho_final": rho,
            "anchor_rho_growth": anchor_rho_growth,
            "anchor_rho_max": anchor_rho_max,
            "dual_update_every": dual_update_every,
            "rho_update_every": rho_update_every,
            "dual_clip": dual_clip,
            "eval_every": eval_every,
            "stable_ceiling": stable_ceiling,
            "unstable_floor": unstable_floor,
            "jury_buffer": jury_buffer,
        },
        "acceptance_limits": {
            "validation_ratio": validation_ratio_limit,
            "max_anchor_normalized_l2": anchor_acceptance,
            "max_characteristic_relative_error": characteristic_acceptance,
            "global_trust_rmse": global_trust_rmse_limit,
            "minimum_off_anchor_return_distance_score": (recurrent_exclusion_acceptance_score),
        },
        "baseline_relu_validation": _float_losses(baseline_val),
        "initial": initial,
        "selected": selected,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"] + 1 if history else 0,
        "spectral_stage_start_epoch": spectral_start_epoch,
        "duration_seconds": duration,
        "candidate_checkpoint": [str(path) for path in candidate_paths],
        "candidate_checkpoint_sha256": {path.name: _sha256(path) for path in candidate_paths},
        "promoted_checkpoint": (
            [str(path) for path in promoted_paths] if promoted_paths is not None else None
        ),
        "promoted_checkpoint_sha256": (
            {path.name: _sha256(path) for path in promoted_paths}
            if promoted_paths is not None
            else None
        ),
        "preexisting_promoted_checkpoint_at_start": preexisting_promoted or None,
        "status": (
            "accepted_numerical_candidate_not_a_conley_certificate"
            if selected["accepted"]
            else "rejected_by_strict_numerical_gates_candidate_only"
        ),
    }
    (output_dir / "smooth_topology_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--transfer-epochs", type=int, default=3000)
    parser.add_argument("--transfer-anchor-weight", type=float, default=1.0)
    parser.add_argument("--constraint-learning-rate", type=float, default=0.00001)
    parser.add_argument(
        "--initial-smooth-checkpoint-dir",
        default=None,
        help=(
            "model-state-only initialization; optimizer/dual/stage reset, and use "
            "--transfer-epochs 0 to begin constrained continuation immediately"
        ),
    )
    parser.add_argument("--initial-smooth-basename", default="smooth_candidate")
    parser.add_argument(
        "--anchor-projection",
        action="store_true",
        help="use the experimental exact output-layer projection",
    )
    parser.add_argument("--gelu-sharpness", type=float, default=100.0)
    parser.add_argument("--global-trust-points", type=int, default=16384)
    parser.add_argument("--trust-weight", type=float, default=10.0)
    parser.add_argument(
        "--replay-component-weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help=(
            "repeatable positive training-row weight from train_metadata.json; "
            "unspecified manifest components remain 1.0"
        ),
    )
    parser.add_argument("--characteristic-weight", type=float, default=5.0)
    parser.add_argument("--topology-weight", type=float, default=20.0)
    parser.add_argument(
        "--trajectory-shadowing-weight",
        type=float,
        default=DEFAULT_TRAJECTORY_SHADOWING_WEIGHT,
        help="opt-in weight for balanced multi-step latent trajectory shadowing",
    )
    parser.add_argument(
        "--trajectory-shadowing-short-horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAJECTORY_SHADOWING_HORIZON_GROUPS["short"]),
    )
    parser.add_argument(
        "--trajectory-shadowing-medium-horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAJECTORY_SHADOWING_HORIZON_GROUPS["medium"]),
    )
    parser.add_argument(
        "--trajectory-shadowing-long-horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAJECTORY_SHADOWING_HORIZON_GROUPS["long"]),
        help="long odd/even rollout horizons are filtered to each component's length",
    )
    parser.add_argument(
        "--recurrent-exclusion-weight",
        type=float,
        default=DEFAULT_RECURRENT_EXCLUSION_WEIGHT,
        help="opt-in weight for the sampled off-anchor p=1,2,4 exclusion loss",
    )
    parser.add_argument("--recurrent-exclusion-global-probes", type=int, default=16384)
    parser.add_argument("--recurrent-exclusion-local-radii", type=int, default=16)
    parser.add_argument("--recurrent-exclusion-local-directions", type=int, default=16)
    parser.add_argument("--recurrent-exclusion-local-min-radius", type=float, default=2e-5)
    parser.add_argument("--recurrent-exclusion-local-max-radius", type=float, default=5e-3)
    parser.add_argument(
        "--recurrent-exclusion-core-radius",
        type=float,
        default=1e-5,
        help="absolute latent radius excluded around each p-dividing intended phase",
    )
    parser.add_argument("--recurrent-exclusion-distance-epsilon", type=float, default=1e-8)
    parser.add_argument("--recurrent-exclusion-score-margin", type=float, default=0.02)
    parser.add_argument("--recurrent-exclusion-temperature", type=float, default=0.0001)
    parser.add_argument("--recurrent-exclusion-acceptance-score", type=float, default=0.005)
    parser.add_argument("--recurrent-exclusion-seed-offset", type=int, default=101)
    parser.add_argument(
        "--recurrent-exclusion-extra-probes",
        default=None,
        metavar="PATH",
        help="optional N x 2 JSON/CSV census roots or cycle phases used as fixed negatives",
    )
    parser.add_argument("--min-closure-epochs", type=int, default=250)
    parser.add_argument("--spectral-start-anchor", type=float, default=0.1)
    parser.add_argument("--spectral-ramp-epochs", type=int, default=750)
    parser.add_argument("--anchor-rho", type=float, default=10.0)
    parser.add_argument("--anchor-rho-growth", type=float, default=2.0)
    parser.add_argument("--anchor-rho-max", type=float, default=100000.0)
    parser.add_argument("--dual-update-every", type=int, default=5)
    parser.add_argument("--rho-update-every", type=int, default=250)
    parser.add_argument("--dual-clip", type=float, default=10000.0)
    parser.add_argument("--stable-ceiling", type=float, default=0.98)
    parser.add_argument("--unstable-floor", type=float, default=1.05)
    parser.add_argument("--jury-buffer", type=float, default=0.005)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--validation-ratio-limit", type=float, default=1.05)
    parser.add_argument("--anchor-acceptance", type=float, default=0.001)
    parser.add_argument("--characteristic-acceptance", type=float, default=0.05)
    parser.add_argument("--global-trust-rmse-limit", type=float, default=0.05)
    args = parser.parse_args()
    replay_component_weights = _parse_replay_component_weights(args.replay_component_weight)
    trajectory_shadowing_horizon_groups = {
        "short": args.trajectory_shadowing_short_horizons,
        "medium": args.trajectory_shadowing_medium_horizons,
        "long": args.trajectory_shadowing_long_horizons,
    }
    summary = train(
        args.config,
        device_name=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        transfer_epochs=args.transfer_epochs,
        transfer_anchor_weight=args.transfer_anchor_weight,
        constraint_learning_rate=args.constraint_learning_rate,
        initial_smooth_checkpoint_dir=args.initial_smooth_checkpoint_dir,
        initial_smooth_basename=args.initial_smooth_basename,
        anchor_projection=args.anchor_projection,
        gelu_sharpness=args.gelu_sharpness,
        global_trust_count=args.global_trust_points,
        trust_weight=args.trust_weight,
        replay_component_weights=replay_component_weights,
        characteristic_weight=args.characteristic_weight,
        topology_weight=args.topology_weight,
        trajectory_shadowing_weight=args.trajectory_shadowing_weight,
        trajectory_shadowing_horizon_groups=trajectory_shadowing_horizon_groups,
        recurrent_exclusion_weight=args.recurrent_exclusion_weight,
        recurrent_exclusion_global_count=args.recurrent_exclusion_global_probes,
        recurrent_exclusion_local_radius_count=args.recurrent_exclusion_local_radii,
        recurrent_exclusion_local_direction_count=(args.recurrent_exclusion_local_directions),
        recurrent_exclusion_local_min_radius=(args.recurrent_exclusion_local_min_radius),
        recurrent_exclusion_local_max_radius=(args.recurrent_exclusion_local_max_radius),
        recurrent_exclusion_core_radius=args.recurrent_exclusion_core_radius,
        recurrent_exclusion_distance_epsilon=(args.recurrent_exclusion_distance_epsilon),
        recurrent_exclusion_score_margin=args.recurrent_exclusion_score_margin,
        recurrent_exclusion_temperature=args.recurrent_exclusion_temperature,
        recurrent_exclusion_acceptance_score=(args.recurrent_exclusion_acceptance_score),
        recurrent_exclusion_seed_offset=args.recurrent_exclusion_seed_offset,
        recurrent_exclusion_extra_probes=args.recurrent_exclusion_extra_probes,
        min_closure_epochs=args.min_closure_epochs,
        spectral_start_anchor=args.spectral_start_anchor,
        spectral_ramp_epochs=args.spectral_ramp_epochs,
        anchor_rho=args.anchor_rho,
        anchor_rho_growth=args.anchor_rho_growth,
        anchor_rho_max=args.anchor_rho_max,
        dual_update_every=args.dual_update_every,
        rho_update_every=args.rho_update_every,
        dual_clip=args.dual_clip,
        stable_ceiling=args.stable_ceiling,
        unstable_floor=args.unstable_floor,
        jury_buffer=args.jury_buffer,
        eval_every=args.eval_every,
        validation_ratio_limit=args.validation_ratio_limit,
        anchor_acceptance=args.anchor_acceptance,
        characteristic_acceptance=args.characteristic_acceptance,
        global_trust_rmse_limit=args.global_trust_rmse_limit,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
