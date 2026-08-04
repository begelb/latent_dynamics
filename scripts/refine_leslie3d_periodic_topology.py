#!/usr/bin/env python3
"""Refine the Leslie3D latent map around explicitly learned periodic cycles.

Value losses alone can place a nearby latent periodic orbit on the wrong side
of a unit-modulus multiplier.  This stage freezes the learned encoder/decoder
chart, optimizes only the 2-D latent map, and represents each catalogued orbit
by a trainable latent base point.  It combines the full transition replay loss
with cycle closure, tethering to E(x), proper-period barriers, and finite-
difference characteristic-polynomial constraints on G^p.

The recurrent labels and physical multiplier targets are supervised numerical
prior information.  They make the topology test sharper but do not turn the
result into a computer-assisted Conley-index proof.
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

from latentdynamics.config import load_config
from latentdynamics.training import load_any_checkpoint, save_checkpoint

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "leslie3d_invariant_aware_refined"
OBJECT_ORDER = ("P0", "P1", "S2", "S4", "p_star", "origin")
PERIODS = {"P0": 4, "P1": 4, "S2": 2, "S4": 4, "p_star": 1, "origin": 1}
EXPECTED_UNSTABLE = {"P0": 0, "P1": 0, "S2": 1, "S4": 1, "p_star": 1, "origin": 1}
EXPECTED_UNSTABLE_SIGN = {"S2": -1.0, "S4": 1.0, "p_star": -1.0, "origin": 1.0}

# Characteristic-polynomial targets for the 2-D monodromy. P0/P1/S2/S4 use
# the two dynamically relevant physical multipliers. A real 2-D quotient
# cannot retain p_star's/origin's unstable direction plus their full stable
# complex plane, so those two preserve the useful pre-refinement signatures.
TARGET_CHARACTERISTIC = {
    "P0": {"trace": -0.9303681694, "determinant": -0.0371570324, "weight": 1.0},
    "P1": {"trace": 1.5798299874, "determinant": 0.9218607202, "weight": 5.0},
    "S2": {"trace": -0.6485571575, "determinant": -0.8161446641, "weight": 5.0},
    "S4": {"trace": 2.1757975675, "determinant": 0.4541942545, "weight": 1.0},
    "p_star": {"trace": -0.28901877, "determinant": -0.90460031, "weight": 0.5},
    "origin": {"trace": 2.58249596, "determinant": -0.34064272, "weight": 0.5},
}
DEFAULT_FD_RADII = (1e-3, 3e-3, 1e-2)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else CODE_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pairs(path: Path, scaler: Any, device: torch.device) -> tuple[Tensor, Tensor]:
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    x = torch.tensor(scaler.transform(values[:, :3]), dtype=torch.float32, device=device)
    y = torch.tensor(scaler.transform(values[:, 3:6]), dtype=torch.float32, device=device)
    return x, y


def _data_losses(
    model: nn.Module,
    z: Tensor,
    z_next: Tensor,
    x: Tensor,
    y: Tensor,
    weights: Tensor,
) -> dict[str, Tensor]:
    z_pred = model.latent_map(z)
    x_hat = model.decoder(z)
    y_hat = model.decoder(z_pred)
    cycle = model.encoder(y_hat)
    mse = nn.functional.mse_loss
    losses = {
        "reconstruction": mse(x_hat, x),
        "prediction": mse(y_hat, y),
        "semiconjugacy": mse(z_pred, z_next),
        "cycle": mse(cycle, z_pred),
    }
    losses["total"] = (
        weights[0] * losses["reconstruction"]
        + weights[1] * losses["prediction"]
        + weights[2] * losses["semiconjugacy"]
        + weights[3] * losses["cycle"]
    )
    return losses


def _phase_latents(
    model: nn.Module,
    scaler: Any,
    manifest: dict[str, Any],
    device: torch.device,
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    with torch.no_grad():
        for name in OBJECT_ORDER:
            points = np.asarray(manifest["known_objects"][name]["points"], dtype=np.float64)
            scaled = torch.tensor(scaler.transform(points), dtype=torch.float32, device=device)
            out[name] = model.encoder(scaled).detach()
    return out


def _iterate(latent_map: nn.Module, point: Tensor, steps: int) -> Tensor:
    value = point
    for _ in range(steps):
        value = latent_map(value)
    return value


def _rollout(latent_map: nn.Module, base: Tensor, period: int) -> tuple[Tensor, Tensor]:
    phases: list[Tensor] = []
    value = base
    for _ in range(period):
        phases.append(value)
        value = latent_map(value)
    return torch.stack(phases), value


def _finite_difference_monodromy(
    latent_map: nn.Module,
    base: Tensor,
    period: int,
    radius: float,
) -> Tensor:
    columns: list[Tensor] = []
    eye = torch.eye(base.numel(), dtype=base.dtype, device=base.device)
    for direction in eye:
        plus = _iterate(latent_map, base + radius * direction, period)
        minus = _iterate(latent_map, base - radius * direction, period)
        columns.append((plus - minus) / (2.0 * radius))
    return torch.stack(columns, dim=1)


def _phase_scale(points: Tensor) -> Tensor:
    if len(points) == 1:
        return torch.tensor(0.05, dtype=points.dtype, device=points.device)
    distances = torch.cdist(points, points)
    positive = distances[distances > 0]
    return torch.clamp(torch.min(positive), min=0.02)


def _constraint_terms(
    latent_map: nn.Module,
    centers: nn.ParameterDict,
    targets: dict[str, Tensor],
    reference_map: nn.Module,
    trust_points: Tensor,
    *,
    fd_radii: tuple[float, ...],
    stable_ceiling: float,
    unstable_floor: float,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    closure_terms: list[Tensor] = []
    tether_terms: list[Tensor] = []
    divisor_terms: list[Tensor] = []
    characteristic_terms: list[Tensor] = []
    topology_terms: list[Tensor] = []
    diagnostics: dict[str, Any] = {}

    for name in OBJECT_ORDER:
        target = targets[name]
        period = PERIODS[name]
        scale = _phase_scale(target)
        rollout, returned = _rollout(latent_map, centers[name], period)
        closure_vector = returned - centers[name]
        closure_terms.append(torch.mean(closure_vector**2) / scale**2)
        tether_terms.append(torch.mean((rollout - target) ** 2) / scale**2)

        proper_divisors = [step for step in range(1, period) if period % step == 0]
        per_divisor: dict[str, float] = {}
        for divisor in proper_divisors:
            returned_early = _iterate(latent_map, centers[name], divisor)
            distance = torch.linalg.vector_norm(returned_early - centers[name])
            margin = 0.5 * torch.linalg.vector_norm(target[divisor] - target[0])
            divisor_terms.append((torch.relu(margin - distance) / scale) ** 2)
            per_divisor[str(divisor)] = float(distance.detach().cpu())

        spectra: dict[str, Any] = {}
        target_char = TARGET_CHARACTERISTIC[name]
        for radius in fd_radii:
            monodromy = _finite_difference_monodromy(
                latent_map, centers[name], period, radius
            )
            trace = torch.trace(monodromy)
            determinant = torch.linalg.det(monodromy)
            eigenvalues = torch.linalg.eigvals(monodromy)
            order = torch.argsort(torch.abs(eigenvalues))
            eigenvalues = eigenvalues[order]
            moduli = torch.abs(eigenvalues)

            char_loss = (
                (trace - target_char["trace"]) / max(1.0, abs(target_char["trace"]))
            ) ** 2 + (
                (determinant - target_char["determinant"])
                / max(1.0, abs(target_char["determinant"]))
            ) ** 2
            characteristic_terms.append(target_char["weight"] * char_loss)

            if EXPECTED_UNSTABLE[name] == 0:
                topology_terms.append(torch.mean(torch.relu(moduli - stable_ceiling) ** 2))
            else:
                unstable = eigenvalues[-1]
                sign = EXPECTED_UNSTABLE_SIGN[name]
                orientation = (
                    torch.relu(unstable.real + unstable_floor)
                    if sign < 0.0
                    else torch.relu(unstable_floor - unstable.real)
                )
                topology_terms.append(
                    torch.relu(moduli[0] - stable_ceiling) ** 2
                    + torch.relu(unstable_floor - moduli[-1]) ** 2
                    + orientation**2
                    + unstable.imag**2
                )
            spectra[f"{radius:.1e}"] = {
                "trace": float(trace.detach().cpu()),
                "determinant": float(determinant.detach().cpu()),
                "eigenvalues": [
                    {
                        "real": float(value.real.detach().cpu()),
                        "imag": float(value.imag.detach().cpu()),
                        "modulus": float(abs(value).detach().cpu()),
                    }
                    for value in eigenvalues
                ],
            }

        diagnostics[name] = {
            "base": [float(value.detach().cpu()) for value in centers[name]],
            "scale": float(scale.detach().cpu()),
            "closure_l2": float(torch.linalg.vector_norm(closure_vector).detach().cpu()),
            "max_tether_l2": float(
                torch.max(torch.linalg.vector_norm(rollout - target, dim=1)).detach().cpu()
            ),
            "proper_divisor_return_l2": per_divisor,
            "finite_difference_spectra": spectra,
        }

    with torch.no_grad():
        reference_values = reference_map(trust_points)
    trust = nn.functional.mse_loss(latent_map(trust_points), reference_values)
    zero = torch.zeros((), dtype=trust.dtype, device=trust.device)
    terms = {
        "closure": torch.stack(closure_terms).mean(),
        "tether": torch.stack(tether_terms).mean(),
        "proper_divisor": torch.stack(divisor_terms).mean() if divisor_terms else zero,
        "characteristic": torch.stack(characteristic_terms).mean(),
        "topology": torch.stack(topology_terms).mean(),
        "trust": trust,
    }
    return terms, diagnostics


def _role_violation(
    name: str,
    eigenvalues: list[dict[str, float]],
    stable_ceiling: float,
    unstable_floor: float,
) -> float:
    moduli = [value["modulus"] for value in eigenvalues]
    if EXPECTED_UNSTABLE[name] == 0:
        return max(0.0, max(moduli) - stable_ceiling)
    unstable = eigenvalues[-1]
    sign = EXPECTED_UNSTABLE_SIGN[name]
    orientation = (
        unstable["real"] + unstable_floor
        if sign < 0.0
        else unstable_floor - unstable["real"]
    )
    return max(
        0.0,
        moduli[0] - stable_ceiling,
        unstable_floor - moduli[-1],
        orientation,
        abs(unstable["imag"]),
    )


def _float_losses(losses: dict[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def refine(
    config_name: str,
    *,
    device_name: str,
    epochs: int | None,
    learning_rate: float | None,
    center_learning_rate: float,
    closure_weight: float,
    tether_weight: float,
    divisor_weight: float,
    characteristic_weight: float,
    topology_weight: float,
    trust_weight: float,
    stable_ceiling: float,
    unstable_floor: float,
    fd_radii: tuple[float, ...],
) -> dict[str, Any]:
    cfg = load_config(config_name)
    if len(cfg.seeds) != 1:
        raise ValueError("topology refinement requires exactly one configured seed")
    seed = int(cfg.seeds[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    epoch_limit = cfg.training.epochs if epochs is None else int(epochs)
    step_size = cfg.training.learning_rate if learning_rate is None else float(learning_rate)
    if epoch_limit <= 0 or step_size <= 0.0 or center_learning_rate <= 0.0:
        raise ValueError("epochs and learning rates must be positive")

    output_dir = _resolve(cfg.paths.output_dir) / f"seed_{seed}"
    warm_dir = _resolve(cfg.training.warm_start_checkpoint_dir)
    data_dir = _resolve(cfg.paths.data_dir)
    scaler = joblib.load(_resolve(cfg.paths.scaler_path("train")))
    manifest = json.loads((data_dir / "dataset_manifest.json").read_text())
    model, source_arch = load_any_checkpoint(warm_dir, arch=cfg.arch)
    if source_arch.model_dump() != cfg.arch.model_dump():
        raise ValueError("warm-start architecture does not match refinement config")
    model = model.to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    model.decoder.eval()
    model.latent_map.train()
    reference_map = copy.deepcopy(model.latent_map).eval()
    for parameter in reference_map.parameters():
        parameter.requires_grad_(False)

    x_train, y_train = _load_pairs(data_dir / "train.csv", scaler, device)
    x_val, y_val = _load_pairs(data_dir / "val.csv", scaler, device)
    with torch.no_grad():
        z_train, z_train_next = model.encoder(x_train), model.encoder(y_train)
        z_val, z_val_next = model.encoder(x_val), model.encoder(y_val)
    targets = _phase_latents(model, scaler, manifest, device)
    centers = nn.ParameterDict(
        {name: nn.Parameter(targets[name][0].clone()) for name in OBJECT_ORDER}
    ).to(device)
    data_weights = torch.tensor(cfg.training.loss_weights, dtype=torch.float32, device=device)
    optimizer = Adam(
        [
            {"params": model.latent_map.parameters(), "lr": step_size},
            {"params": centers.parameters(), "lr": center_learning_rate},
        ]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.scheduler_factor,
        patience=cfg.training.lr_patience,
        threshold=cfg.training.scheduler_threshold,
        min_lr=cfg.training.scheduler_min_lr,
    )

    def weighted_score(data: dict[str, Tensor], constraints: dict[str, Tensor]) -> Tensor:
        return (
            data["total"]
            + closure_weight * constraints["closure"]
            + tether_weight * constraints["tether"]
            + divisor_weight * constraints["proper_divisor"]
            + characteristic_weight * constraints["characteristic"]
            + topology_weight * constraints["topology"]
            + trust_weight * constraints["trust"]
        )

    def evaluate() -> dict[str, Any]:
        model.eval()
        data = _data_losses(model, z_val, z_val_next, x_val, y_val, data_weights)
        constraints, diagnostics = _constraint_terms(
            model.latent_map,
            centers,
            targets,
            reference_map,
            z_train,
            fd_radii=fd_radii,
            stable_ceiling=stable_ceiling,
            unstable_floor=unstable_floor,
        )
        score = weighted_score(data, constraints)
        violations = {
            name: max(
                _role_violation(
                    name,
                    spectrum["eigenvalues"],
                    stable_ceiling,
                    unstable_floor,
                )
                for spectrum in diagnostics[name]["finite_difference_spectra"].values()
            )
            for name in OBJECT_ORDER
        }
        closure_max = max(diagnostics[name]["closure_l2"] for name in OBJECT_ORDER)
        model.latent_map.train()
        return {
            "data": _float_losses(data),
            "constraints": _float_losses(constraints),
            "score": float(score.detach().cpu()),
            "role_margin_violations": violations,
            "max_role_margin_violation": max(violations.values()),
            "all_role_margins_satisfied": max(violations.values()) == 0.0,
            "max_cycle_closure_l2": closure_max,
            "cycles": diagnostics,
        }

    initial = evaluate()
    initial_val_total = initial["data"]["total"]

    def selection_rank(result: dict[str, Any]) -> tuple[float, float, float, float]:
        validation_ratio = result["data"]["total"] / initial_val_total
        gates = float(not result["all_role_margins_satisfied"]) + float(validation_ratio > 1.05)
        return (
            gates,
            result["max_role_margin_violation"],
            result["max_cycle_closure_l2"],
            result["score"],
        )

    best = copy.deepcopy(initial)
    best_epoch = -1
    best_model_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_center_state = {
        key: value.detach().cpu().clone() for key, value in centers.state_dict().items()
    }
    no_improve = 0
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()
    iterator = tqdm(range(epoch_limit))
    for epoch in iterator:
        optimizer.zero_grad(set_to_none=True)
        data = _data_losses(model, z_train, z_train_next, x_train, y_train, data_weights)
        constraints, _ = _constraint_terms(
            model.latent_map,
            centers,
            targets,
            reference_map,
            z_train,
            fd_radii=fd_radii,
            stable_ceiling=stable_ceiling,
            unstable_floor=unstable_floor,
        )
        objective = weighted_score(data, constraints)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite topology objective at epoch {epoch}")
        objective.backward()
        if cfg.training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(model.latent_map.parameters()) + list(centers.parameters()),
                cfg.training.gradient_clip_norm,
            )
        optimizer.step()
        current = evaluate()
        scheduler.step(current["score"])
        history.append(
            {
                "epoch": epoch,
                "train_data_total": float(data["total"].detach().cpu()),
                "selection_score": current["score"],
                "val_data_total": current["data"]["total"],
                "max_role_margin_violation": current["max_role_margin_violation"],
                "max_cycle_closure_l2": current["max_cycle_closure_l2"],
                "latent_learning_rate": optimizer.param_groups[0]["lr"],
                "center_learning_rate": optimizer.param_groups[1]["lr"],
                **{
                    f"constraint_{name}": value
                    for name, value in current["constraints"].items()
                },
            }
        )
        if selection_rank(current) < selection_rank(best):
            best = copy.deepcopy(current)
            best_epoch = epoch
            best_model_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_center_state = {
                key: value.detach().cpu().clone() for key, value in centers.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.training.patience:
                break
        iterator.set_postfix(
            score=f"{current['score']:.3e}",
            val=f"{current['data']['total']:.3e}",
            margin=f"{current['max_role_margin_violation']:.2e}",
            close=f"{current['max_cycle_closure_l2']:.2e}",
        )

    duration = time.perf_counter() - start_time
    model.load_state_dict(best_model_state)
    centers.load_state_dict(best_center_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model.cpu(), cfg.arch, output_dir / "models")
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs" / "topology_refinement_history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        "experiment": cfg.experiment_name,
        "method": "frozen_chart_trainable_cycles_finite_difference_monodromy",
        "seed": seed,
        "warm_start": {
            "path": str(warm_dir),
            "autoencoder_sha256": _sha256(warm_dir / "autoencoder.pt"),
        },
        "train_csv_sha256": _sha256(data_dir / "train.csv"),
        "validation_csv_sha256": _sha256(data_dir / "val.csv"),
        "optimized_parameters": "latent_map_and_auxiliary_cycle_basepoints",
        "learning_rates": {"latent_map": step_size, "cycle_basepoints": center_learning_rate},
        "weights": {
            "data": list(cfg.training.loss_weights),
            "closure": closure_weight,
            "tether": tether_weight,
            "proper_divisor": divisor_weight,
            "characteristic": characteristic_weight,
            "topology": topology_weight,
            "trust": trust_weight,
        },
        "finite_difference_radii": fd_radii,
        "spectral_margins": {
            "stable_ceiling": stable_ceiling,
            "unstable_floor": unstable_floor,
        },
        "target_characteristic_polynomials": TARGET_CHARACTERISTIC,
        "initial": initial,
        "selected": best,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "duration_seconds": duration,
        "status": "numerical_role_supervision_not_a_conley_certificate",
    }
    (output_dir / "topology_refinement_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--center-learning-rate", type=float, default=2e-5)
    parser.add_argument("--closure-weight", type=float, default=20.0)
    parser.add_argument("--tether-weight", type=float, default=1.0)
    parser.add_argument("--divisor-weight", type=float, default=1.0)
    parser.add_argument("--characteristic-weight", type=float, default=0.1)
    parser.add_argument("--topology-weight", type=float, default=1.0)
    parser.add_argument("--trust-weight", type=float, default=20.0)
    parser.add_argument("--stable-ceiling", type=float, default=0.98)
    parser.add_argument("--unstable-floor", type=float, default=1.05)
    parser.add_argument(
        "--fd-radii",
        type=float,
        nargs="+",
        default=DEFAULT_FD_RADII,
    )
    args = parser.parse_args()
    result = refine(
        args.config,
        device_name=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        center_learning_rate=args.center_learning_rate,
        closure_weight=args.closure_weight,
        tether_weight=args.tether_weight,
        divisor_weight=args.divisor_weight,
        characteristic_weight=args.characteristic_weight,
        topology_weight=args.topology_weight,
        trust_weight=args.trust_weight,
        stable_ceiling=args.stable_ceiling,
        unstable_floor=args.unstable_floor,
        fd_radii=tuple(args.fd_radii),
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
