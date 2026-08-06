#!/usr/bin/env python3
"""Topology-gated AdamW then L-BFGS polish of the accepted Leslie3D map.

This is deliberately a new experiment rather than an edit of the historical
smooth-topology trainer.  It loads the accepted GELU checkpoint, keeps the
encoder and decoder bitwise frozen, and optimizes only the latent map.  The
static post-ramp objective from the accepted run is retained:

    weighted replay
    + 0.5 * rho * normalized-anchor MSE
    + trust_weight * (replay trust + global trust)
    + characteristic_weight * characteristic loss
    + topology_weight * topology loss.

AdamW supplies an adaptive first-order polish.  CPU float64 L-BFGS then uses a
pure, deterministic, full-batch strong-Wolfe closure.  Every reported candidate
is cast back to float32 before held-out and topology-gate evaluation.  The
accepted source checkpoint is always an eligible rollback candidate, and this
script never modifies its directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import LBFGS, AdamW

from latentdynamics.config import ExperimentConfig, load_config
from latentdynamics.training import load_any_checkpoint, save_checkpoint

CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_leslie3d_smooth_topology as smooth  # noqa: E402

DEFAULT_CONFIG = "leslie3d_invariant_aware_v2_smooth"
DEFAULT_SOURCE = (
    CODE_ROOT
    / "output"
    / "leslie3d_invariant_aware_v2_smooth"
    / "seed_20260809"
    / "models"
)
DEFAULT_OUTPUT = (
    CODE_ROOT
    / "output"
    / "leslie3d_invariant_aware_v2_smooth_adamw_lbfgs"
    / "seed_20260809"
)
EXPECTED_SOURCE_SHA256 = "9fbee2cde690d58d2413c0d3521763838abaeb493736120f7173035612da3f3d"


@dataclass(frozen=True)
class FixedData:
    """All deterministic tensors used by an optimizer objective/evaluation."""

    x_train: Tensor
    y_train: Tensor
    z_train: Tensor
    z_train_next: Tensor
    x_val: Tensor
    y_val: Tensor
    z_val: Tensor
    z_val_next: Tensor
    sample_weights: Tensor
    loss_weights: Tensor
    targets: dict[str, Tensor]
    scales: dict[str, Tensor]
    trust_train_reference: Tensor
    trust_global_points: Tensor
    trust_global_reference: Tensor

    def to(self, *, dtype: torch.dtype, device: torch.device) -> FixedData:
        def cast(value: Tensor) -> Tensor:
            return value.detach().to(device=device, dtype=dtype).contiguous()

        return FixedData(
            x_train=cast(self.x_train),
            y_train=cast(self.y_train),
            z_train=cast(self.z_train),
            z_train_next=cast(self.z_train_next),
            x_val=cast(self.x_val),
            y_val=cast(self.y_val),
            z_val=cast(self.z_val),
            z_val_next=cast(self.z_val_next),
            sample_weights=cast(self.sample_weights),
            loss_weights=cast(self.loss_weights),
            targets={name: cast(value) for name, value in self.targets.items()},
            scales={name: cast(value) for name, value in self.scales.items()},
            trust_train_reference=cast(self.trust_train_reference),
            trust_global_points=cast(self.trust_global_points),
            trust_global_reference=cast(self.trust_global_reference),
        )


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (CODE_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _float32_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().float().clone() for name, value in model.state_dict().items()}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _freeze_chart(model: nn.Module) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.latent_map.parameters():
        parameter.requires_grad_(True)
    model.encoder.eval()
    model.decoder.eval()
    model.latent_map.train()


def _assert_chart_equal(
    baseline_state: dict[str, Tensor],
    candidate: nn.Module,
) -> None:
    candidate_state = candidate.state_dict()
    for prefix in ("encoder.", "decoder."):
        changed = [
            name
            for name, value in baseline_state.items()
            if name.startswith(prefix) and not torch.equal(value, candidate_state[name].cpu())
        ]
        if changed:
            raise RuntimeError(f"fine-tune changed frozen chart tensors: {changed}")


def _objective(
    model: nn.Module,
    data: FixedData,
    *,
    rho: float,
    trust_weight: float,
    characteristic_weight: float,
    topology_weight: float,
    stable_ceiling: float,
    unstable_floor: float,
    jury_buffer: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    replay = smooth._replay_losses(
        model,
        data.z_train,
        data.z_train_next,
        data.x_train,
        data.y_train,
        data.loss_weights,
        sample_weights=data.sample_weights,
    )
    anchor, _ = smooth._anchor_residuals(model.latent_map, data.targets, data.scales)
    trust_replay = nn.functional.mse_loss(
        model.latent_map(data.z_train), data.trust_train_reference
    )
    trust_global = nn.functional.mse_loss(
        model.latent_map(data.trust_global_points), data.trust_global_reference
    )
    characteristic, topology, _ = smooth._spectral_terms(
        model.latent_map,
        data.targets,
        stable_ceiling=stable_ceiling,
        unstable_floor=unstable_floor,
        jury_buffer=jury_buffer,
        diagnostics=False,
    )
    anchor_quadratic = torch.mean(anchor**2)
    total = (
        replay["total"]
        + 0.5 * float(rho) * anchor_quadratic
        + float(trust_weight) * (trust_replay + trust_global)
        + float(characteristic_weight) * characteristic
        + float(topology_weight) * topology
    )
    return total, {
        "objective": total,
        "weighted_replay": replay["total"],
        "anchor_quadratic": anchor_quadratic,
        "trust_replay": trust_replay,
        "trust_global": trust_global,
        "characteristic": characteristic,
        "topology": topology,
    }


def _evaluate(
    model: nn.Module,
    data: FixedData,
    *,
    relu_validation_total: float,
    validation_ratio_limit: float,
    anchor_acceptance: float,
    characteristic_acceptance: float,
    global_trust_rmse_limit: float,
    stable_ceiling: float,
    unstable_floor: float,
    jury_buffer: float,
    strict_thresholds: dict[str, float] | None,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        validation = smooth._replay_losses(
            model,
            data.z_val,
            data.z_val_next,
            data.x_val,
            data.y_val,
            data.loss_weights,
        )
        train_weighted = smooth._replay_losses(
            model,
            data.z_train,
            data.z_train_next,
            data.x_train,
            data.y_train,
            data.loss_weights,
            sample_weights=data.sample_weights,
        )
        train_unweighted = smooth._replay_losses(
            model,
            data.z_train,
            data.z_train_next,
            data.x_train,
            data.y_train,
            data.loss_weights,
        )
        anchor, _ = smooth._anchor_residuals(model.latent_map, data.targets, data.scales)
        trust_replay = nn.functional.mse_loss(
            model.latent_map(data.z_train), data.trust_train_reference
        )
        trust_global = nn.functional.mse_loss(
            model.latent_map(data.trust_global_points), data.trust_global_reference
        )
    with torch.enable_grad():
        characteristic, topology, spectra = smooth._spectral_terms(
            model.latent_map,
            data.targets,
            stable_ceiling=stable_ceiling,
            unstable_floor=unstable_floor,
            jury_buffer=jury_buffer,
            diagnostics=True,
        )

    validation_values = smooth._float_losses(validation)
    train_weighted_values = smooth._float_losses(train_weighted)
    train_unweighted_values = smooth._float_losses(train_unweighted)
    max_anchor = smooth._max_anchor_normalized_l2(anchor)
    max_characteristic_error = max(
        max(
            spectra[name]["trace_relative_error"],
            spectra[name]["determinant_relative_error"],
        )
        for name in smooth.OBJECT_ORDER
    )
    role_violations = {
        name: smooth._role_violation(
            name,
            spectra[name]["eigenvalues"],
            stable_ceiling,
            unstable_floor,
        )
        for name in smooth.OBJECT_ORDER
    }
    max_role_violation = max(role_violations.values())
    trust_global_rmse = float(torch.sqrt(trust_global).detach())
    validation_ratio = validation_values["total"] / float(relu_validation_total)
    numeric_values = (
        *validation_values.values(),
        *train_weighted_values.values(),
        *train_unweighted_values.values(),
        max_anchor,
        max_characteristic_error,
        max_role_violation,
        trust_global_rmse,
        float(characteristic.detach()),
        float(topology.detach()),
    )
    finite = all(math.isfinite(value) for value in numeric_values)
    gates = {
        "finite": finite,
        "validation_ratio": validation_ratio <= validation_ratio_limit,
        "fixed_anchor_closure": max_anchor <= anchor_acceptance,
        "characteristic_polynomials": max_characteristic_error
        <= characteristic_acceptance,
        "orientation_and_stability_roles": max_role_violation == 0.0,
        "global_distillation": trust_global_rmse <= global_trust_rmse_limit,
    }
    accepted = all(gates.values())
    if strict_thresholds is None:
        strict_checks = {
            "max_anchor_nonregression": True,
            "max_characteristic_nonregression": True,
            "global_trust_nonregression": True,
            "topology_loss_nonregression": True,
        }
    else:
        strict_checks = {
            "max_anchor_nonregression": max_anchor
            <= strict_thresholds["max_anchor_normalized_l2"],
            "max_characteristic_nonregression": max_characteristic_error
            <= strict_thresholds["max_characteristic_relative_error"],
            "global_trust_nonregression": trust_global_rmse
            <= strict_thresholds["trust_global_rmse"],
            "topology_loss_nonregression": float(topology.detach())
            <= strict_thresholds["topology_loss"],
        }
    strict = accepted and all(strict_checks.values())
    model.train()
    model.encoder.eval()
    model.decoder.eval()
    return {
        "validation": validation_values,
        "train_weighted": train_weighted_values,
        "train_unweighted": train_unweighted_values,
        "validation_ratio_to_relu_base": validation_ratio,
        "max_anchor_normalized_l2": max_anchor,
        "characteristic_loss": float(characteristic.detach()),
        "topology_loss": float(topology.detach()),
        "max_characteristic_relative_error": max_characteristic_error,
        "role_margin_violations": role_violations,
        "max_role_margin_violation": max_role_violation,
        "trust_replay_mse": float(trust_replay.detach()),
        "trust_global_mse": float(trust_global.detach()),
        "trust_global_rmse": trust_global_rmse,
        "monodromies": spectra,
        "acceptance_gates": gates,
        "accepted": accepted,
        "strict_nonregression_checks": strict_checks,
        "strict_nonregression": strict,
    }


def _cosine_learning_rate(start: float, end: float, step: int, total: int) -> float:
    if total <= 1:
        return float(end)
    fraction = (step - 1) / (total - 1)
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * fraction)))


def _clone_as_float32(template: nn.Module, state: dict[str, Tensor]) -> nn.Module:
    candidate = copy.deepcopy(template).cpu().float()
    candidate.load_state_dict(
        {name: value.detach().cpu().float() for name, value in state.items()}, strict=True
    )
    _freeze_chart(candidate)
    return candidate


def _record_is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    return candidate["validation"]["total"] < incumbent["validation"]["total"]


def _lbfgs_step(
    optimizer: LBFGS,
    model: nn.Module,
    data: FixedData,
    *,
    rho: float,
    outer_step: int,
    trust_weight: float,
    characteristic_weight: float,
    topology_weight: float,
    stable_ceiling: float,
    unstable_floor: float,
    jury_buffer: float,
) -> tuple[Tensor, int]:
    """Run one L-BFGS outer step with a side-effect-free full-batch closure."""
    closure_calls = 0

    def closure() -> Tensor:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        total, _ = _objective(
            model,
            data,
            rho=rho,
            trust_weight=trust_weight,
            characteristic_weight=characteristic_weight,
            topology_weight=topology_weight,
            stable_ceiling=stable_ceiling,
            unstable_floor=unstable_floor,
            jury_buffer=jury_buffer,
        )
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"non-finite L-BFGS objective at rho={rho}, outer={outer_step}"
            )
        total.backward()
        return total

    returned = optimizer.step(closure)
    return returned, closure_calls


def _prepare(
    cfg: ExperimentConfig,
    accepted_model: nn.Module,
    relu_model: nn.Module,
    source_summary: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[FixedData, dict[str, Any]]:
    data_dir = _resolve(cfg.paths.data_dir)
    scaler_path = _resolve(cfg.paths.scaler_path("train"))
    scaler = joblib.load(scaler_path)
    x_train, y_train = smooth._load_pairs(data_dir / "train.csv", scaler, device)
    x_val, y_val = smooth._load_pairs(data_dir / "val.csv", scaler, device)
    with torch.no_grad():
        z_train = accepted_model.encoder(x_train).detach()
        z_train_next = accepted_model.encoder(y_train).detach()
        z_val = accepted_model.encoder(x_val).detach()
        z_val_next = accepted_model.encoder(y_val).detach()
    manifest_path = data_dir / "dataset_manifest.json"
    metadata_path = data_dir / "train_metadata.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    overrides = {
        name: float(value)
        for name, value in source_summary["hyperparameters"][
            "replay_component_weight_overrides"
        ].items()
    }
    sample_weights, weight_report = smooth._manifest_component_sample_weights(
        metadata,
        row_count=len(x_train),
        overrides=overrides,
        device=device,
    )
    targets = smooth._phase_latents(accepted_model, scaler, manifest, device)
    scales = smooth._phase_scales(targets)
    global_points = smooth._global_trust_points(16_384, seed + 17, device)
    with torch.no_grad():
        trust_train_reference = relu_model.latent_map(z_train).detach()
        trust_global_reference = relu_model.latent_map(global_points).detach()
    data = FixedData(
        x_train=x_train,
        y_train=y_train,
        z_train=z_train,
        z_train_next=z_train_next,
        x_val=x_val,
        y_val=y_val,
        z_val=z_val,
        z_val_next=z_val_next,
        sample_weights=sample_weights,
        loss_weights=torch.tensor(
            cfg.training.loss_weights, dtype=torch.float32, device=device
        ),
        targets=targets,
        scales=scales,
        trust_train_reference=trust_train_reference,
        trust_global_points=global_points,
        trust_global_reference=trust_global_reference,
    )
    provenance = {
        "train_csv": {"path": str(data_dir / "train.csv"), "sha256": _sha256(data_dir / "train.csv")},
        "validation_csv": {
            "path": str(data_dir / "val.csv"),
            "sha256": _sha256(data_dir / "val.csv"),
        },
        "dataset_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "train_metadata": {"path": str(metadata_path), "sha256": _sha256(metadata_path)},
        "scaler": {"path": str(scaler_path), "sha256": _sha256(scaler_path)},
        "train_rows": len(x_train),
        "validation_rows": len(x_val),
        "sample_weight_overrides": overrides,
        "sample_weight_sum": float(sample_weights.sum()),
        "sample_weight_report": weight_report,
    }
    return data, provenance


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise ValueError("this deterministic L-BFGS experiment currently requires --device cpu")
    if args.adamw_steps < 0 or args.adamw_eval_every < 1:
        raise ValueError("invalid AdamW step/evaluation count")
    if args.adamw_lr_start <= 0 or args.adamw_lr_end <= 0:
        raise ValueError("AdamW learning rates must be positive")
    if args.adamw_weight_decay < 0:
        raise ValueError("AdamW weight decay must be nonnegative")
    if any(rho <= 0 for rho in args.lbfgs_rho):
        raise ValueError("L-BFGS rho values must be positive")

    cfg = load_config(args.config)
    if len(cfg.seeds) != 1:
        raise ValueError("fine-tune config must contain exactly one seed")
    seed = int(cfg.seeds[0])
    _seed_everything(seed)
    device = torch.device("cpu")
    source_dir = _resolve(args.source_checkpoint_dir)
    source_path = source_dir / "autoencoder.pt"
    source_sidecar = source_dir / "autoencoder.json"
    source_hash = _sha256(source_path)
    if source_hash != args.expected_source_sha256:
        raise RuntimeError(
            f"source checkpoint hash mismatch: expected {args.expected_source_sha256}, got {source_hash}"
        )
    output_dir = _resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force_overwrite:
        raise FileExistsError(
            f"refusing to write nonempty output directory {output_dir}; use --force-overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_summary_path = _resolve(args.source_summary)
    source_summary = json.loads(source_summary_path.read_text())
    accepted_model, accepted_arch = load_any_checkpoint(source_dir, map_location=device)
    if accepted_arch.model_dump() != cfg.arch.model_dump():
        raise ValueError("accepted checkpoint architecture does not match the config")
    relu_dir = _resolve(cfg.training.warm_start_checkpoint_dir)
    relu_model, relu_arch = load_any_checkpoint(relu_dir, map_location=device)
    smooth._assert_smooth_transition(relu_arch, cfg.arch)
    accepted_model = accepted_model.to(device).float()
    relu_model = relu_model.to(device).float().eval()
    for parameter in relu_model.parameters():
        parameter.requires_grad_(False)
    _freeze_chart(accepted_model)
    baseline_state = _state_cpu(accepted_model)

    data, data_provenance = _prepare(
        cfg,
        accepted_model,
        relu_model,
        source_summary,
        seed=seed,
        device=device,
    )
    hp = source_summary["hyperparameters"]
    limits = source_summary["acceptance_limits"]
    stable_ceiling = float(hp["stable_ceiling"])
    unstable_floor = float(hp["unstable_floor"])
    jury_buffer = float(hp["jury_buffer"])
    trust_weight = float(hp["trust_weight"])
    characteristic_weight = float(hp["characteristic_weight"])
    topology_weight = float(hp["topology_weight"])
    relu_validation_total = float(source_summary["baseline_relu_validation"]["total"])

    baseline = _evaluate(
        accepted_model,
        data,
        relu_validation_total=relu_validation_total,
        validation_ratio_limit=float(limits["validation_ratio"]),
        anchor_acceptance=float(limits["max_anchor_normalized_l2"]),
        characteristic_acceptance=float(limits["max_characteristic_relative_error"]),
        global_trust_rmse_limit=float(limits["global_trust_rmse"]),
        stable_ceiling=stable_ceiling,
        unstable_floor=unstable_floor,
        jury_buffer=jury_buffer,
        strict_thresholds=None,
    )
    strict_thresholds = {
        "max_anchor_normalized_l2": baseline["max_anchor_normalized_l2"]
        * (1.0 + args.strict_relative_slack)
        + args.strict_absolute_slack,
        "max_characteristic_relative_error": baseline[
            "max_characteristic_relative_error"
        ]
        * (1.0 + args.strict_relative_slack)
        + args.strict_absolute_slack,
        "trust_global_rmse": baseline["trust_global_rmse"]
        * (1.0 + args.strict_relative_slack)
        + args.strict_absolute_slack,
        "topology_loss": baseline["topology_loss"] + args.strict_absolute_slack,
    }
    baseline = _evaluate(
        accepted_model,
        data,
        relu_validation_total=relu_validation_total,
        validation_ratio_limit=float(limits["validation_ratio"]),
        anchor_acceptance=float(limits["max_anchor_normalized_l2"]),
        characteristic_acceptance=float(limits["max_characteristic_relative_error"]),
        global_trust_rmse_limit=float(limits["global_trust_rmse"]),
        stable_ceiling=stable_ceiling,
        unstable_floor=unstable_floor,
        jury_buffer=jury_buffer,
        strict_thresholds=strict_thresholds,
    )
    if not baseline["accepted"] or not baseline["strict_nonregression"]:
        raise RuntimeError("accepted source checkpoint does not pass its reconstructed gates")

    best_accepted = copy.deepcopy(baseline)
    best_accepted_state = copy.deepcopy(baseline_state)
    best_accepted_source = "accepted_source"
    best_strict = copy.deepcopy(baseline)
    best_strict_state = copy.deepcopy(baseline_state)
    best_strict_source = "accepted_source"
    adamw_history: list[dict[str, Any]] = []
    started = time.perf_counter()

    adamw = AdamW(
        accepted_model.latent_map.parameters(),
        lr=args.adamw_lr_start,
        betas=(args.adamw_beta1, args.adamw_beta2),
        eps=args.adamw_eps,
        weight_decay=args.adamw_weight_decay,
    )
    for step in range(1, args.adamw_steps + 1):
        lr = _cosine_learning_rate(
            args.adamw_lr_start, args.adamw_lr_end, step, args.adamw_steps
        )
        for group in adamw.param_groups:
            group["lr"] = lr
        adamw.zero_grad(set_to_none=True)
        objective, components = _objective(
            accepted_model,
            data,
            rho=args.adamw_rho,
            trust_weight=trust_weight,
            characteristic_weight=characteristic_weight,
            topology_weight=topology_weight,
            stable_ceiling=stable_ceiling,
            unstable_floor=unstable_floor,
            jury_buffer=jury_buffer,
        )
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite AdamW objective at step {step}")
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            accepted_model.latent_map.parameters(), args.adamw_gradient_clip
        )
        adamw.step()
        if step % args.adamw_eval_every != 0 and step != args.adamw_steps:
            continue
        _assert_chart_equal(baseline_state, accepted_model)
        evaluated = _evaluate(
            accepted_model,
            data,
            relu_validation_total=relu_validation_total,
            validation_ratio_limit=float(limits["validation_ratio"]),
            anchor_acceptance=float(limits["max_anchor_normalized_l2"]),
            characteristic_acceptance=float(limits["max_characteristic_relative_error"]),
            global_trust_rmse_limit=float(limits["global_trust_rmse"]),
            stable_ceiling=stable_ceiling,
            unstable_floor=unstable_floor,
            jury_buffer=jury_buffer,
            strict_thresholds=strict_thresholds,
        )
        record = {
            "step": step,
            "learning_rate": lr,
            "gradient_norm_before_clip": float(grad_norm),
            "pre_update_objective": {
                name: float(value.detach()) for name, value in components.items()
            },
            "evaluation": evaluated,
        }
        adamw_history.append(record)
        if evaluated["accepted"] and _record_is_better(evaluated, best_accepted):
            best_accepted = copy.deepcopy(evaluated)
            best_accepted_state = _state_cpu(accepted_model)
            best_accepted_source = f"adamw_step_{step}"
        if evaluated["strict_nonregression"] and _record_is_better(evaluated, best_strict):
            best_strict = copy.deepcopy(evaluated)
            best_strict_state = _state_cpu(accepted_model)
            best_strict_source = f"adamw_step_{step}"
        print(
            f"AdamW {step}/{args.adamw_steps} lr={lr:.3e} "
            f"val={evaluated['validation']['total']:.10e} "
            f"anchor={evaluated['max_anchor_normalized_l2']:.3e} "
            f"char={evaluated['max_characteristic_relative_error']:.3e} "
            f"accepted={evaluated['accepted']} strict={evaluated['strict_nonregression']}",
            flush=True,
        )

    adamw_start_state = copy.deepcopy(best_accepted_state)
    lbfgs_histories: list[dict[str, Any]] = []
    for rho in args.lbfgs_rho:
        double_model = _clone_as_float32(accepted_model, adamw_start_state).double()
        _freeze_chart(double_model)
        double_data = data.to(dtype=torch.float64, device=device)
        lbfgs = LBFGS(
            double_model.latent_map.parameters(),
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_max_iter,
            max_eval=args.lbfgs_max_eval,
            tolerance_grad=args.lbfgs_tolerance_grad,
            tolerance_change=args.lbfgs_tolerance_change,
            history_size=args.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
        branch_records: list[dict[str, Any]] = []
        for outer in range(1, args.lbfgs_outer_steps + 1):
            returned, closure_calls = _lbfgs_step(
                lbfgs,
                double_model,
                double_data,
                rho=rho,
                outer_step=outer,
                trust_weight=trust_weight,
                characteristic_weight=characteristic_weight,
                topology_weight=topology_weight,
                stable_ceiling=stable_ceiling,
                unstable_floor=unstable_floor,
                jury_buffer=jury_buffer,
            )
            candidate_state = _float32_state(double_model)
            candidate = _clone_as_float32(accepted_model, candidate_state)
            _assert_chart_equal(baseline_state, candidate)
            evaluated = _evaluate(
                candidate,
                data,
                relu_validation_total=relu_validation_total,
                validation_ratio_limit=float(limits["validation_ratio"]),
                anchor_acceptance=float(limits["max_anchor_normalized_l2"]),
                characteristic_acceptance=float(limits["max_characteristic_relative_error"]),
                global_trust_rmse_limit=float(limits["global_trust_rmse"]),
                stable_ceiling=stable_ceiling,
                unstable_floor=unstable_floor,
                jury_buffer=jury_buffer,
                strict_thresholds=strict_thresholds,
            )
            branch_records.append(
                {
                    "outer_step": outer,
                    "closure_calls": closure_calls,
                    "optimizer_returned_objective": float(returned.detach()),
                    "evaluation_float32": evaluated,
                }
            )
            if evaluated["accepted"] and _record_is_better(evaluated, best_accepted):
                best_accepted = copy.deepcopy(evaluated)
                best_accepted_state = copy.deepcopy(candidate_state)
                best_accepted_source = f"lbfgs_rho_{rho:g}_outer_{outer}"
            if evaluated["strict_nonregression"] and _record_is_better(evaluated, best_strict):
                best_strict = copy.deepcopy(evaluated)
                best_strict_state = copy.deepcopy(candidate_state)
                best_strict_source = f"lbfgs_rho_{rho:g}_outer_{outer}"
            print(
                f"L-BFGS rho={rho:g} outer={outer}/{args.lbfgs_outer_steps} "
                f"closures={closure_calls} val={evaluated['validation']['total']:.10e} "
                f"anchor={evaluated['max_anchor_normalized_l2']:.3e} "
                f"char={evaluated['max_characteristic_relative_error']:.3e} "
                f"accepted={evaluated['accepted']} strict={evaluated['strict_nonregression']}",
                flush=True,
            )
        lbfgs_histories.append({"rho": rho, "records": branch_records})

    elapsed = time.perf_counter() - started
    models_dir = output_dir / "models"
    accepted_candidate_model = _clone_as_float32(accepted_model, best_accepted_state)
    strict_candidate_model = _clone_as_float32(accepted_model, best_strict_state)
    accepted_paths = save_checkpoint(
        accepted_candidate_model, cfg.arch, models_dir, basename="loss_candidate"
    )
    strict_paths = save_checkpoint(
        strict_candidate_model, cfg.arch, models_dir, basename="strict_candidate"
    )
    strict_improvement = (
        best_strict["validation"]["total"]
        < baseline["validation"]["total"] - args.minimum_validation_improvement
    )
    promoted_paths: tuple[Path, Path] | None = None
    if strict_improvement:
        promoted_paths = save_checkpoint(strict_candidate_model, cfg.arch, models_dir)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    history_payload = {
        "schema_version": 1,
        "adamw": adamw_history,
        "lbfgs": lbfgs_histories,
    }
    (logs_dir / "optimizer_history.json").write_text(
        json.dumps(history_payload, indent=2, allow_nan=False) + "\n"
    )
    checkpoint_hashes = {
        path.name: _sha256(path)
        for path in (*accepted_paths, *strict_paths, *(promoted_paths or ()))
    }
    summary = {
        "schema_version": 1,
        "status": (
            "strict_validation_improvement_promoted"
            if strict_improvement
            else "no_strict_validation_improvement_source_retained"
        ),
        "scientific_status": (
            "optimizer experiment only; requires fresh periodic-root and Morse checks"
        ),
        "source": {
            "checkpoint_dir": str(source_dir),
            "checkpoint_sha256": source_hash,
            "architecture_sidecar_sha256": _sha256(source_sidecar),
            "summary": str(source_summary_path),
            "summary_sha256": _sha256(source_summary_path),
        },
        "config": args.config,
        "configuration": cfg.model_dump(mode="json"),
        "seed": seed,
        "device": str(device),
        "deterministic_algorithms": True,
        "frozen_components": ["encoder", "decoder"],
        "optimized_components": ["latent_map"],
        "parameter_counts": {
            "latent_map": sum(p.numel() for p in accepted_model.latent_map.parameters()),
            "all": sum(p.numel() for p in accepted_model.parameters()),
        },
        "objective": {
            "formula": (
                "weighted replay + 0.5*rho*anchor_mse + trust_weight*(train_trust+global_trust) "
                "+ characteristic_weight*characteristic + topology_weight*topology"
            ),
            "loss_weights": list(cfg.training.loss_weights),
            "trust_weight": trust_weight,
            "characteristic_weight": characteristic_weight,
            "topology_weight": topology_weight,
        },
        "optimizer": {
            "adamw": {
                "steps": args.adamw_steps,
                "learning_rate_start": args.adamw_lr_start,
                "learning_rate_end": args.adamw_lr_end,
                "schedule": "cosine",
                "weight_decay": args.adamw_weight_decay,
                "betas": [args.adamw_beta1, args.adamw_beta2],
                "eps": args.adamw_eps,
                "gradient_clip_norm": args.adamw_gradient_clip,
                "rho": args.adamw_rho,
                "evaluation_every": args.adamw_eval_every,
            },
            "lbfgs": {
                "dtype": "float64 closure; float32 candidate evaluation",
                "rho_branches": args.lbfgs_rho,
                "outer_steps": args.lbfgs_outer_steps,
                "lr": args.lbfgs_lr,
                "max_iter": args.lbfgs_max_iter,
                "max_eval": args.lbfgs_max_eval,
                "history_size": args.lbfgs_history_size,
                "tolerance_grad": args.lbfgs_tolerance_grad,
                "tolerance_change": args.lbfgs_tolerance_change,
                "line_search_fn": "strong_wolfe",
                "closure_policy": "pure full batch; no clipping, projection, scheduler, or state mutation",
            },
        },
        "acceptance_limits": limits,
        "strict_nonregression_thresholds": strict_thresholds,
        "baseline": baseline,
        "best_accepted": {
            "source": best_accepted_source,
            "evaluation": best_accepted,
        },
        "best_strict": {
            "source": best_strict_source,
            "evaluation": best_strict,
        },
        "strict_validation_improvement": strict_improvement,
        "minimum_validation_improvement": args.minimum_validation_improvement,
        "checkpoints": {
            "loss_candidate": [str(path) for path in accepted_paths],
            "strict_candidate": [str(path) for path in strict_paths],
            "promoted_autoencoder": (
                [str(path) for path in promoted_paths] if promoted_paths is not None else None
            ),
            "sha256": checkpoint_hashes,
        },
        "data": data_provenance,
        "duration_seconds": elapsed,
        "duration_minutes": elapsed / 60.0,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "historical_trainer": str(Path(smooth.__file__).resolve()),
            "historical_trainer_current_sha256": _sha256(Path(smooth.__file__).resolve()),
        },
    }
    summary_path = output_dir / "adamw_lbfgs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    final_lines = [
        f"status: {summary['status']}",
        f"baseline_val_loss_total: {baseline['validation']['total']:.12e}",
        f"best_accepted_source: {best_accepted_source}",
        f"best_accepted_val_loss_total: {best_accepted['validation']['total']:.12e}",
        f"best_strict_source: {best_strict_source}",
        f"best_strict_val_loss_total: {best_strict['validation']['total']:.12e}",
        f"strict_validation_improvement: {strict_improvement}",
    ]
    (output_dir / "final_losses.txt").write_text("\n".join(final_lines) + "\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-checkpoint-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument(
        "--source-summary",
        default=str(DEFAULT_SOURCE.parent / "smooth_topology_summary.json"),
    )
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--adamw-steps", type=int, default=5000)
    parser.add_argument("--adamw-lr-start", type=float, default=1e-7)
    parser.add_argument("--adamw-lr-end", type=float, default=2e-8)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.0)
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.999)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--adamw-gradient-clip", type=float, default=1.0)
    parser.add_argument("--adamw-rho", type=float, default=5000.0)
    parser.add_argument("--adamw-eval-every", type=int, default=25)
    parser.add_argument("--lbfgs-rho", type=float, nargs="+", default=[5000.0, 20000.0])
    parser.add_argument("--lbfgs-outer-steps", type=int, default=12)
    parser.add_argument("--lbfgs-lr", type=float, default=0.25)
    parser.add_argument("--lbfgs-max-iter", type=int, default=10)
    parser.add_argument("--lbfgs-max-eval", type=int, default=25)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-9)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-12)
    parser.add_argument("--strict-relative-slack", type=float, default=0.02)
    parser.add_argument("--strict-absolute-slack", type=float, default=1e-8)
    parser.add_argument("--minimum-validation-improvement", type=float, default=1e-9)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "baseline_val": summary["baseline"]["validation"]["total"],
                "best_accepted_source": summary["best_accepted"]["source"],
                "best_accepted_val": summary["best_accepted"]["evaluation"]["validation"][
                    "total"
                ],
                "best_strict_source": summary["best_strict"]["source"],
                "best_strict_val": summary["best_strict"]["evaluation"]["validation"][
                    "total"
                ],
                "output_dir": args.output_dir,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
