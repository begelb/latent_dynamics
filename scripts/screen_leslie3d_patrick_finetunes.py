"""Fast, non-rigorous screen for Patrick-warm-start Leslie3D checkpoints.

This script is deliberately *not* a Conley-index or Morse-graph computation.
It ranks completed fine-tuning repeats before an expensive CMGDB run using:

1. losses recomputed on the experiment's fixed validation CSV and archived
   Patrick scaler (plus the corresponding values in ``training_summary.json``),
2. forward iteration of many encoded validation initial conditions, followed
   by phase-invariant clustering of apparent attracting cycles, and
3. bounded numerical searches for roots of ``G^p(z)-z`` for p in {1, 2, 4},
   with eigenvalues of ``D(G^p)`` at the recovered cycles.

The root search and orbit clustering are numerical diagnostics. They can miss
objects, merge nearby objects, or report non-isolating recurrent structure.
Only a subsequent CMGDB computation can answer the Conley-index question.
Input artifacts are hashed before and after the screen and are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import least_squares

from latentdynamics.config import ExperimentConfig, load_config
from latentdynamics.training import load_any_checkpoint

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "leslie3d_example2_patrick_finetune_4x"
DEFAULT_PERIODS = (1, 2, 4)
EXPECTED_ATTRACTING_PERIOD = 4
EXPECTED_ATTRACTING_CYCLES = 2


@dataclass(frozen=True)
class FixedValidationData:
    x_scaled: NDArray[np.float64]
    y_scaled: NDArray[np.float64]
    initial_scaled: NDArray[np.float64]
    metadata: dict[str, Any]
    csv_path: Path
    metadata_path: Path
    scaler_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_code_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (CODE_ROOT / candidate).resolve()


def _resolve_config_file(config_arg: str | Path) -> Path:
    candidate = Path(config_arg)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_absolute():
        raise FileNotFoundError(candidate)
    stem = candidate.stem if candidate.suffix else candidate.name
    packaged = CODE_ROOT / "src" / "latentdynamics" / "configs" / f"{stem}.yaml"
    if not packaged.is_file():
        raise FileNotFoundError(packaged)
    return packaged.resolve()


def _load_pair(path: Path, high_dims: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2 * high_dims:
        raise ValueError(
            f"{path} must have {2 * high_dims} numeric columns; found shape {data.shape}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"{path} contains non-finite values")
    return data[:, :high_dims], data[:, high_dims:]


def _same_numeric_mapping(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            if not math.isclose(float(actual_value), float(expected_value), rel_tol=0.0, abs_tol=1e-12):
                return False
        elif actual_value != expected_value:
            return False
    return True


def load_fixed_validation(cfg: ExperimentConfig) -> FixedValidationData:
    """Load and validate the exact holdout/scaler contract used for fine-tuning."""
    csv_path = _resolve_code_path(cfg.paths.val_csv())
    metadata_path = _resolve_code_path(cfg.paths.val_metadata())
    scaler_dir = _resolve_code_path(cfg.paths.scaler_dir)
    if cfg.paths.flat_scaler:
        modern_scaler = scaler_dir / "scaler.gz"
        legacy_scaler = scaler_dir / "scaler"
        scaler_path = legacy_scaler if not modern_scaler.is_file() and legacy_scaler.is_file() else modern_scaler
    else:
        scaler_path = scaler_dir / "train" / "scaler.gz"
    for path in (csv_path, metadata_path, scaler_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")
    expected_pairs = cfg.data.n_samples_val * (cfg.data.n_iterations - cfg.data.skip)
    checks = {
        "dimension": metadata.get("dimension") == cfg.arch.high_dims,
        "n_samples": metadata.get("n_samples") == cfg.data.n_samples_val,
        "n_iterations": metadata.get("n_iterations") == cfg.data.n_iterations,
        "skip_initial_steps": metadata.get("skip_initial_steps") == cfg.data.skip,
        "sampling_seed": metadata.get("sampling_seed") == cfg.data.val_seed,
        "model_params": isinstance(metadata.get("model_params"), dict)
        and _same_numeric_mapping(metadata["model_params"], cfg.system.params),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"fixed validation metadata disagrees with config: {failed}")

    x_raw, y_raw = _load_pair(csv_path, cfg.arch.high_dims)
    if x_raw.shape[0] != expected_pairs:
        raise ValueError(
            f"{csv_path} has {x_raw.shape[0]} pairs; expected {expected_pairs} "
            "from n_samples_val * retained trajectory steps"
        )
    scaler = joblib.load(scaler_path)
    x_scaled = np.asarray(scaler.transform(x_raw), dtype=np.float64)
    y_scaled = np.asarray(scaler.transform(y_raw), dtype=np.float64)
    # sample_trajectories stores one n_samples block per time step. The first
    # block is therefore the independent validation initial-condition set.
    initial_scaled = x_scaled[: cfg.data.n_samples_val].copy()
    return FixedValidationData(
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        initial_scaled=initial_scaled,
        metadata=metadata,
        csv_path=csv_path,
        metadata_path=metadata_path,
        scaler_path=scaler_path,
    )


def recompute_validation_losses(
    model: torch.nn.Module,
    data: FixedValidationData,
    *,
    loss_weights: Sequence[float],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Compute sample-weighted loss terms over every fixed validation pair."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    sums = {
        "loss_reconstruction": 0.0,
        "loss_prediction": 0.0,
        "loss_semiconjugacy": 0.0,
        "loss_cycle": 0.0,
    }
    counts = dict.fromkeys(sums, 0)
    use_cycle = len(loss_weights) == 4 and float(loss_weights[3]) != 0.0
    with torch.inference_mode():
        for start in range(0, data.x_scaled.shape[0], batch_size):
            stop = min(start + batch_size, data.x_scaled.shape[0])
            x = torch.as_tensor(data.x_scaled[start:stop], dtype=torch.float32, device=device)
            y = torch.as_tensor(data.y_scaled[start:stop], dtype=torch.float32, device=device)
            fp = model(x, y)
            terms = {
                "loss_reconstruction": (fp.x_t_hat - fp.x_t).square(),
                "loss_prediction": (fp.x_tau_hat - fp.x_tau).square(),
                "loss_semiconjugacy": (fp.z_tau_pred - fp.z_tau).square(),
                "loss_cycle": (fp.z_tau_pred_cycle - fp.z_tau_pred).square(),
            }
            for name, values in terms.items():
                if name == "loss_cycle" and not use_cycle:
                    continue
                sums[name] += float(values.double().sum().cpu())
                counts[name] += int(values.numel())

    losses = {
        name: sums[name] / counts[name]
        for name in ("loss_reconstruction", "loss_prediction", "loss_semiconjugacy")
    }
    if use_cycle:
        losses["loss_cycle"] = sums["loss_cycle"] / counts["loss_cycle"]
    total = sum(
        float(weight) * losses[name]
        for weight, name in zip(
            loss_weights[:3],
            ("loss_reconstruction", "loss_prediction", "loss_semiconjugacy"),
            strict=True,
        )
    )
    if use_cycle:
        total += float(loss_weights[3]) * losses["loss_cycle"]
    return {
        **losses,
        "loss_total": float(total),
        "n_transition_pairs": int(data.x_scaled.shape[0]),
        "aggregation": "sample-weighted mean squared error over the complete fixed holdout",
    }


def reported_validation_losses(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    reported: dict[str, float] = {}
    selected = payload.get("selected_val")
    if isinstance(selected, dict) and selected:
        for name, value in selected.items():
            if isinstance(value, (int, float)):
                reported[str(name)] = float(value)
        reported_source = "selected_val"
    else:
        # Compatibility with summaries written before selected_val existed.
        val = payload.get("val")
        if isinstance(val, dict):
            for name, block in val.items():
                if isinstance(block, dict) and isinstance(
                    block.get("best_epoch_value"), (int, float)
                ):
                    reported[str(name)] = float(block["best_epoch_value"])
        reported_source = "val.*.best_epoch_value" if reported else None
    initial = payload.get("initial_val")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "best_epoch": payload.get("best_epoch"),
        "best_source": payload.get("best_source"),
        "initial_validation": initial if isinstance(initial, dict) else None,
        "saved_checkpoint_validation": reported or None,
        "saved_checkpoint_validation_source": reported_source,
    }


def _canonical_cycle(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a phase-invariant cycle signature while retaining point coordinates."""
    if points.ndim != 2 or points.shape[0] < 1:
        raise ValueError("cycle points must have shape (period, latent_dim)")
    order = np.lexsort(tuple(points[:, axis] for axis in reversed(range(points.shape[1]))))
    return points[order].reshape(-1)


def _cluster_signatures(
    signatures: list[NDArray[np.float64]],
    *,
    rms_tolerance: float,
) -> list[dict[str, Any]]:
    """Deterministic online clustering for already-converged cycle signatures."""
    clusters: list[dict[str, Any]] = []
    for sample_index, signature in enumerate(signatures):
        choices = [
            float(np.sqrt(np.mean((signature - cluster["center"]) ** 2)))
            for cluster in clusters
        ]
        nearest = int(np.argmin(choices)) if choices else -1
        if nearest >= 0 and choices[nearest] <= rms_tolerance:
            cluster = clusters[nearest]
            old_count = int(cluster["support"])
            cluster["support"] = old_count + 1
            cluster["center"] = (cluster["center"] * old_count + signature) / (old_count + 1)
            cluster["member_indices"].append(sample_index)
            cluster["max_assignment_rms"] = max(
                float(cluster["max_assignment_rms"]), choices[nearest]
            )
        else:
            clusters.append(
                {
                    "center": signature.copy(),
                    "support": 1,
                    "member_indices": [sample_index],
                    "max_assignment_rms": 0.0,
                }
            )
    return sorted(clusters, key=lambda cluster: (-int(cluster["support"]), cluster["center"].tolist()))


@torch.inference_mode()
def probe_attracting_cycles(
    model: torch.nn.Module,
    initial_scaled: NDArray[np.float64],
    *,
    device: torch.device,
    n_orbits: int,
    burn_in: int,
    max_period: int,
    cycle_tolerance_relative: float,
    cluster_tolerance_relative: float,
    min_basin_fraction: float,
) -> tuple[dict[str, Any], NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Iterate encoded ICs and cluster numerically closed tail cycles."""
    if not 1 <= n_orbits <= initial_scaled.shape[0]:
        raise ValueError(f"n_orbits must be in [1, {initial_scaled.shape[0]}]")
    if burn_in < 1 or max_period < 1:
        raise ValueError("burn_in and max_period must be positive")
    if not 0.0 <= min_basin_fraction <= 1.0:
        raise ValueError("min_basin_fraction must lie in [0, 1]")
    selected = np.linspace(0, initial_scaled.shape[0] - 1, n_orbits, dtype=np.int64)
    x = torch.as_tensor(initial_scaled[selected], dtype=torch.float32, device=device)
    z = model.encoder(x)
    encoded_initial = z.detach().cpu().numpy().astype(np.float64)
    lower = encoded_initial.min(axis=0)
    upper = encoded_initial.max(axis=0)
    span = upper - lower
    margin = 0.01 * np.maximum(span, 1e-6)
    bounds = (lower - margin, upper + margin)
    diameter = max(float(np.linalg.norm(bounds[1] - bounds[0])), 1e-12)
    closure_tolerance = cycle_tolerance_relative * diameter
    cluster_tolerance = cluster_tolerance_relative * diameter

    for _ in range(burn_in):
        z = model.latent_map(z)
    tail = [z.detach().cpu().numpy().astype(np.float64)]
    for _ in range(max_period):
        z = model.latent_map(z)
        tail.append(z.detach().cpu().numpy().astype(np.float64))

    periods = np.zeros(n_orbits, dtype=np.int64)
    closure_errors = np.full(n_orbits, np.nan, dtype=np.float64)
    for period in range(1, max_period + 1):
        error = np.linalg.norm(tail[period] - tail[0], axis=1)
        newly_closed = (periods == 0) & (error <= closure_tolerance)
        periods[newly_closed] = period
        closure_errors[newly_closed] = error[newly_closed]

    cluster_rows: list[dict[str, Any]] = []
    supported_threshold = max(1, math.ceil(min_basin_fraction * n_orbits))
    for period in sorted(int(value) for value in np.unique(periods) if value > 0):
        orbit_indices = np.flatnonzero(periods == period)
        signatures = [
            _canonical_cycle(np.stack([tail[phase][index] for phase in range(period)]))
            for index in orbit_indices
        ]
        clusters = _cluster_signatures(signatures, rms_tolerance=cluster_tolerance)
        for cluster in clusters:
            members = orbit_indices[np.asarray(cluster["member_indices"], dtype=np.int64)]
            center = np.asarray(cluster["center"], dtype=np.float64).reshape(
                period, encoded_initial.shape[1]
            )
            support = int(cluster["support"])
            cluster_rows.append(
                {
                    "period": period,
                    "support": support,
                    "basin_fraction": support / n_orbits,
                    "supported": support >= supported_threshold,
                    "cycle_points_lexicographic": center.tolist(),
                    "median_closure_error": float(np.median(closure_errors[members])),
                    "max_closure_error": float(np.max(closure_errors[members])),
                    "max_assignment_rms": float(cluster["max_assignment_rms"]),
                }
            )
    cluster_rows.sort(key=lambda row: (-row["support"], row["period"]))
    detected = int(np.count_nonzero(periods))
    return (
        {
            "method": "forward latent iteration from encoded fixed-validation initial conditions",
            "n_orbits": n_orbits,
            "burn_in": burn_in,
            "tested_periods": list(range(1, max_period + 1)),
            "latent_bounds_from_encoded_initial_conditions": {
                "lower": bounds[0].tolist(),
                "upper": bounds[1].tolist(),
            },
            "latent_box_diameter": diameter,
            "closure_tolerance_absolute": closure_tolerance,
            "cluster_tolerance_rms_absolute": cluster_tolerance,
            "minimum_supported_basin_count": supported_threshold,
            "period_counts": {
                str(period): int(np.count_nonzero(periods == period))
                for period in range(1, max_period + 1)
                if np.any(periods == period)
            },
            "unclassified_count": n_orbits - detected,
            "unclassified_fraction": (n_orbits - detected) / n_orbits,
            "cycle_clusters": cluster_rows,
        },
        encoded_initial,
        bounds,
    )


class _ReturnMapEvaluator:
    """Cache residual and exact PyTorch Jacobian for one scipy evaluation point."""

    def __init__(self, latent_map: torch.nn.Module, period: int, device: torch.device) -> None:
        self.latent_map = latent_map
        self.period = period
        self.device = device
        self._point: NDArray[np.float64] | None = None
        self._residual: NDArray[np.float64] | None = None
        self._jacobian: NDArray[np.float64] | None = None

    def _compute(self, point: NDArray[np.float64]) -> None:
        point = np.asarray(point, dtype=np.float64)
        if self._point is not None and np.array_equal(point, self._point):
            return
        parameter = next(self.latent_map.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.float32
        z = torch.tensor(point, dtype=dtype, device=self.device, requires_grad=True)

        def residual(value: torch.Tensor) -> torch.Tensor:
            image = value
            for _ in range(self.period):
                image = self.latent_map(image)
            return image - value

        with torch.enable_grad():
            value = residual(z)
            jacobian = torch.autograd.functional.jacobian(residual, z, vectorize=True)
        self._point = point.copy()
        self._residual = value.detach().cpu().double().numpy()
        self._jacobian = jacobian.detach().cpu().double().numpy()

    def fun(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        self._compute(point)
        assert self._residual is not None
        return self._residual

    def jac(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        self._compute(point)
        assert self._jacobian is not None
        return self._jacobian


@torch.inference_mode()
def _iterate_point(
    latent_map: torch.nn.Module,
    point: NDArray[np.float64],
    period: int,
    *,
    device: torch.device,
) -> NDArray[np.float64]:
    parameter = next(latent_map.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.float32
    value = torch.as_tensor(point, dtype=dtype, device=device)
    for _ in range(period):
        value = latent_map(value)
    return value.detach().cpu().double().numpy()


def _return_jacobian(
    latent_map: torch.nn.Module,
    point: NDArray[np.float64],
    period: int,
    *,
    device: torch.device,
) -> NDArray[np.float64]:
    parameter = next(latent_map.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.float32
    z = torch.tensor(point, dtype=dtype, device=device, requires_grad=True)

    def return_map(value: torch.Tensor) -> torch.Tensor:
        image = value
        for _ in range(period):
            image = latent_map(image)
        return image

    with torch.enable_grad():
        jacobian = torch.autograd.functional.jacobian(return_map, z, vectorize=True)
    return jacobian.detach().cpu().double().numpy()


def _primitive_period(
    latent_map: torch.nn.Module,
    point: NDArray[np.float64],
    return_period: int,
    *,
    device: torch.device,
    residual_tolerance: float,
) -> int | None:
    for divisor in range(1, return_period + 1):
        if return_period % divisor != 0:
            continue
        residual = np.linalg.norm(_iterate_point(latent_map, point, divisor, device=device) - point)
        if residual <= residual_tolerance:
            return divisor
    return None


def _grid_starts(
    lower: NDArray[np.float64], upper: NDArray[np.float64], n_starts: int
) -> NDArray[np.float64]:
    if lower.shape != upper.shape:
        raise ValueError("root-search lower/upper bounds have different shapes")
    if lower.size != 2:
        raise ValueError("this Leslie3D screen expects a two-dimensional latent map")
    side = max(2, math.ceil(math.sqrt(n_starts)))
    mesh = np.meshgrid(
        np.linspace(lower[0], upper[0], side),
        np.linspace(lower[1], upper[1], side),
        indexing="ij",
    )
    complete = np.column_stack([axis.ravel() for axis in mesh])
    indices = np.linspace(0, complete.shape[0] - 1, min(n_starts, complete.shape[0]), dtype=int)
    return complete[indices]


def find_periodic_roots(
    latent_map: torch.nn.Module,
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    *,
    periods: Sequence[int],
    n_grid_starts: int,
    extra_starts: NDArray[np.float64] | None,
    device: torch.device,
    residual_tolerance_relative: float,
    dedupe_tolerance_relative: float,
    hyperbolicity_margin: float,
    max_nfev: int,
) -> dict[str, Any]:
    """Bounded numerical roots of G^p-id, deduplicated as primitive cycles."""
    lower, upper = (np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64))
    diameter = max(float(np.linalg.norm(upper - lower)), 1e-12)
    residual_tolerance = residual_tolerance_relative * diameter
    dedupe_tolerance = dedupe_tolerance_relative * diameter
    starts = _grid_starts(lower, upper, n_grid_starts)
    if extra_starts is not None and extra_starts.size:
        extra = np.asarray(extra_starts, dtype=np.float64)
        inside = np.all((extra >= lower) & (extra <= upper), axis=1)
        starts = np.vstack([starts, extra[inside]])

    raw: list[dict[str, Any]] = []
    attempted = 0
    converged_by_return_period: dict[str, int] = {}
    for return_period in periods:
        evaluator = _ReturnMapEvaluator(latent_map, int(return_period), device)
        converged = 0
        for start in starts:
            attempted += 1
            result = least_squares(
                evaluator.fun,
                start,
                jac=evaluator.jac,
                bounds=(lower, upper),
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                max_nfev=max_nfev,
            )
            root = np.asarray(result.x, dtype=np.float64)
            residual = float(np.linalg.norm(evaluator.fun(root)))
            if not result.success or residual > residual_tolerance:
                continue
            primitive = _primitive_period(
                latent_map,
                root,
                int(return_period),
                device=device,
                residual_tolerance=residual_tolerance,
            )
            if primitive is None:
                continue
            orbit = np.stack(
                [_iterate_point(latent_map, root, phase, device=device) for phase in range(primitive)]
            )
            if not np.all((orbit >= lower - residual_tolerance) & (orbit <= upper + residual_tolerance)):
                continue
            raw.append(
                {
                    "root": root,
                    "signature": _canonical_cycle(orbit),
                    "primitive_period": primitive,
                    "searched_return_period": int(return_period),
                    "residual": residual,
                }
            )
            converged += 1
        converged_by_return_period[str(return_period)] = converged

    groups: list[dict[str, Any]] = []
    for record in sorted(raw, key=lambda row: (row["primitive_period"], row["residual"])):
        compatible: list[tuple[float, int]] = []
        for index, group in enumerate(groups):
            if group["primitive_period"] != record["primitive_period"]:
                continue
            distance = float(np.sqrt(np.mean((record["signature"] - group["center"]) ** 2)))
            compatible.append((distance, index))
        distance, match = min(compatible, default=(math.inf, -1))
        if match >= 0 and distance <= dedupe_tolerance:
            group = groups[match]
            support = int(group["support"])
            group["center"] = (group["center"] * support + record["signature"]) / (support + 1)
            group["support"] = support + 1
            group["searched_return_periods"].add(record["searched_return_period"])
            if record["residual"] < group["best"]["residual"]:
                group["best"] = record
        else:
            groups.append(
                {
                    "primitive_period": record["primitive_period"],
                    "center": record["signature"].copy(),
                    "support": 1,
                    "searched_return_periods": {record["searched_return_period"]},
                    "best": record,
                }
            )

    cycles: list[dict[str, Any]] = []
    for group in groups:
        best = group["best"]
        period = int(group["primitive_period"])
        root = np.asarray(best["root"], dtype=np.float64)
        orbit = np.stack(
            [_iterate_point(latent_map, root, phase, device=device) for phase in range(period)]
        )
        jacobian = _return_jacobian(latent_map, root, period, device=device)
        eigenvalues = np.linalg.eigvals(jacobian)
        moduli = np.abs(eigenvalues)
        near_unit = np.abs(moduli - 1.0) <= hyperbolicity_margin
        stable_dimension = int(np.count_nonzero(moduli < 1.0 - hyperbolicity_margin))
        unstable_dimension = int(np.count_nonzero(moduli > 1.0 + hyperbolicity_margin))
        if np.any(near_unit):
            stability = "numerically_nonhyperbolic"
        elif unstable_dimension == 0:
            stability = "attractor"
        elif stable_dimension == 0:
            stability = "repeller"
        else:
            stability = "saddle"
        cycles.append(
            {
                "primitive_period": period,
                "representative": root.tolist(),
                "cycle_points": orbit.tolist(),
                "return_residual": float(best["residual"]),
                "solver_support": int(group["support"]),
                "searched_return_periods": sorted(int(value) for value in group["searched_return_periods"]),
                "return_jacobian": jacobian.tolist(),
                "eigenvalues": [
                    {"real": float(value.real), "imag": float(value.imag), "modulus": float(abs(value))}
                    for value in eigenvalues
                ],
                "stable_dimension": stable_dimension,
                "unstable_dimension": unstable_dimension,
                "stability": stability,
            }
        )
    cycles.sort(key=lambda row: (row["primitive_period"], row["stability"], row["representative"]))
    return {
        "method": "bounded scipy least-squares roots of G^p-id with PyTorch Jacobians",
        "searched_return_periods": [int(value) for value in periods],
        "grid_starts": int(n_grid_starts),
        "extra_starts": 0 if extra_starts is None else int(extra_starts.shape[0]),
        "attempted_solver_calls": attempted,
        "converged_solver_calls_by_return_period": converged_by_return_period,
        "residual_tolerance_absolute": residual_tolerance,
        "cycle_deduplication_rms_absolute": dedupe_tolerance,
        "hyperbolicity_margin_about_unit_modulus": hyperbolicity_margin,
        "cycles": cycles,
        "limitations": (
            "A bounded multi-start root search is incomplete: absent cycles may have been missed, "
            "and numerical hyperbolicity is not an isolating-neighborhood proof."
        ),
    }


def _checkpoint_payloads(model_dir: Path) -> list[Path]:
    names = ("autoencoder.pt", "autoencoder.json", "encoder.pt", "dynamics.pt", "decoder.pt")
    paths = [model_dir / name for name in names if (model_dir / name).is_file()]
    if not paths:
        raise FileNotFoundError(f"no checkpoint payloads found in {model_dir}")
    return paths


def _screening_criteria(record: dict[str, Any]) -> dict[str, Any]:
    supported = [
        cluster
        for cluster in record["recurrent_orbit_probe"]["cycle_clusters"]
        if cluster["supported"]
    ]
    p4_attractors = sum(cluster["period"] == EXPECTED_ATTRACTING_PERIOD for cluster in supported)
    other_attractors = sum(cluster["period"] != EXPECTED_ATTRACTING_PERIOD for cluster in supported)
    basin_penalty = abs(p4_attractors - EXPECTED_ATTRACTING_CYCLES) + other_attractors

    roots = record["periodic_root_probe"]["cycles"]
    root_p4_attractors = sum(
        cycle["primitive_period"] == 4 and cycle["stability"] == "attractor" for cycle in roots
    )
    root_has_p2_saddle = any(
        cycle["primitive_period"] == 2 and cycle["stability"] == "saddle" for cycle in roots
    )
    root_has_p1_saddle = any(
        cycle["primitive_period"] == 1 and cycle["stability"] == "saddle" for cycle in roots
    )
    root_penalty = (
        abs(root_p4_attractors - EXPECTED_ATTRACTING_CYCLES)
        + int(not root_has_p2_saddle)
        + int(not root_has_p1_saddle)
    )
    unclassified = float(record["recurrent_orbit_probe"]["unclassified_fraction"])
    heldout = float(record["validation_loss_recomputed"]["loss_total"])
    return {
        "expected_ground_truth_heuristic": {
            "supported_attracting_period_4_cycles": EXPECTED_ATTRACTING_CYCLES,
            "period_2_saddle_found": True,
            "period_1_saddle_found": True,
        },
        "observed": {
            "supported_attracting_period_4_cycles": p4_attractors,
            "other_supported_attracting_cycles": other_attractors,
            "root_period_4_attractors": root_p4_attractors,
            "root_period_2_saddle_found": root_has_p2_saddle,
            "root_period_1_saddle_found": root_has_p1_saddle,
        },
        "basin_pattern_penalty": basin_penalty,
        "root_pattern_penalty": root_penalty,
        "unclassified_orbit_fraction": unclassified,
        "heldout_total_loss": heldout,
        "lexicographic_sort_key": [basin_penalty, root_penalty, unclassified, heldout],
        "ranking_rule": (
            "prefer the expected attracting-cycle pattern; then the expected local saddle "
            "pattern; then fewer unclassified forward orbits; then lower fixed-holdout loss"
        ),
    }


def _analyze_checkpoint(
    *,
    label: str,
    run_dir: Path,
    model_dir: Path,
    cfg: ExperimentConfig,
    data: FixedValidationData,
    device: torch.device,
    eval_batch_size: int,
    n_orbits: int,
    burn_in: int,
    max_period: int,
    cycle_tolerance_relative: float,
    cluster_tolerance_relative: float,
    min_basin_fraction: float,
    root_periods: Sequence[int],
    root_starts: int,
    root_residual_tolerance_relative: float,
    root_dedupe_tolerance_relative: float,
    root_max_nfev: int,
) -> dict[str, Any]:
    payloads = _checkpoint_payloads(model_dir)
    model, arch = load_any_checkpoint(model_dir, arch=cfg.arch, map_location=device)
    if arch.model_dump() != cfg.arch.model_dump():
        raise ValueError(f"checkpoint architecture mismatch in {model_dir}")
    model.to(device).eval()
    validation = recompute_validation_losses(
        model,
        data,
        loss_weights=cfg.training.loss_weights,
        device=device,
        batch_size=eval_batch_size,
    )
    summary = reported_validation_losses(run_dir / "training_summary.json")
    reported_total = None
    if summary is not None and summary["saved_checkpoint_validation"] is not None:
        reported_total = summary["saved_checkpoint_validation"].get("loss_total")
    comparison = None
    if reported_total is not None:
        absolute_difference = float(abs(reported_total - validation["loss_total"]))
        relative_difference = float(
            absolute_difference / max(abs(validation["loss_total"]), 1e-15)
        )
        comparison = {
            "reported_minus_recomputed_total": float(reported_total - validation["loss_total"]),
            "absolute_difference": absolute_difference,
            "relative_difference": relative_difference,
            "agrees_within_floating_tolerance": bool(
                absolute_difference <= 1e-8 + 1e-5 * abs(validation["loss_total"])
            ),
            "comparison_tolerance": {"absolute": 1e-8, "relative": 1e-5},
            "note": (
                "The trainer and this screen both use sample-weighted means and should agree up "
                "to floating-point accumulation. A material discrepancy indicates a checkpoint, "
                "data, scaler, or loss-definition mismatch."
            ),
        }
    recurrent, encoded_initial, bounds = probe_attracting_cycles(
        model,
        data.initial_scaled,
        device=device,
        n_orbits=n_orbits,
        burn_in=burn_in,
        max_period=max_period,
        cycle_tolerance_relative=cycle_tolerance_relative,
        cluster_tolerance_relative=cluster_tolerance_relative,
        min_basin_fraction=min_basin_fraction,
    )
    extra_starts = []
    for cluster in recurrent["cycle_clusters"]:
        extra_starts.extend(cluster["cycle_points_lexicographic"])
    extra_array = (
        np.asarray(extra_starts, dtype=np.float64)
        if extra_starts
        else encoded_initial[np.linspace(0, encoded_initial.shape[0] - 1, min(16, encoded_initial.shape[0]), dtype=int)]
    )
    roots = find_periodic_roots(
        model.latent_map,
        bounds,
        periods=root_periods,
        n_grid_starts=root_starts,
        extra_starts=extra_array,
        device=device,
        residual_tolerance_relative=root_residual_tolerance_relative,
        dedupe_tolerance_relative=root_dedupe_tolerance_relative,
        hyperbolicity_margin=1e-3,
        max_nfev=root_max_nfev,
    )
    record = {
        "label": label,
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "checkpoint_payloads": {
            str(path.name): {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in payloads
        },
        "training_summary": summary,
        "validation_loss_recomputed": validation,
        "training_summary_comparison": comparison,
        "recurrent_orbit_probe": recurrent,
        "periodic_root_probe": roots,
    }
    record["screening_criteria"] = _screening_criteria(record)
    return record


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def _snapshot_inputs(paths: Sequence[Path]) -> dict[str, str]:
    unique = sorted({path.resolve() for path in paths})
    return {str(path): _sha256(path) for path in unique}


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve_config_file(args.config)
    cfg = load_config(config_path)
    if cfg.arch.low_dims != 2:
        raise ValueError("the Patrick Leslie3D screen requires latent dimension 2")
    if not cfg.paths.scaler_read_only:
        raise ValueError("screen requires a config with scaler_read_only=true")
    data = load_fixed_validation(cfg)
    device = _resolve_device(args.device)
    experiment_root = _resolve_code_path(cfg.paths.output_dir)

    candidates: list[tuple[str, Path, Path]] = []
    seeds = cfg.seeds if args.seeds is None else args.seeds
    for seed in seeds:
        run_dir = experiment_root / f"seed_{seed}"
        candidates.append((f"seed_{seed}", run_dir, run_dir / "models"))
    baseline: tuple[str, Path, Path] | None = None
    if args.include_baseline:
        source = cfg.training.warm_start_checkpoint_dir
        if source is None:
            raise ValueError("--include-baseline requires training.warm_start_checkpoint_dir")
        model_dir = _resolve_code_path(source)
        baseline = ("patrick_archived_baseline", model_dir.parent, model_dir)

    input_paths = [config_path, data.csv_path, data.metadata_path, data.scaler_path]
    for _, run_dir, model_dir in candidates + ([] if baseline is None else [baseline]):
        input_paths.extend(_checkpoint_payloads(model_dir))
        summary = run_dir / "training_summary.json"
        if summary.is_file():
            input_paths.append(summary)
    before = _snapshot_inputs(input_paths)

    settings = {
        "device": str(device),
        "eval_batch_size": args.eval_batch_size,
        "n_orbits": args.orbits,
        "burn_in": args.burn_in,
        "max_period": args.max_period,
        "cycle_tolerance_relative": args.cycle_tolerance,
        "cluster_tolerance_relative": args.cluster_tolerance,
        "minimum_basin_fraction": args.min_basin_fraction,
        "root_periods": list(args.root_periods),
        "root_grid_starts": args.root_starts,
        "root_residual_tolerance_relative": args.root_residual_tolerance,
        "root_deduplication_tolerance_relative": args.root_dedupe_tolerance,
        "root_max_function_evaluations": args.root_max_nfev,
    }

    def analyze(spec: tuple[str, Path, Path]) -> dict[str, Any]:
        return _analyze_checkpoint(
            label=spec[0],
            run_dir=spec[1],
            model_dir=spec[2],
            cfg=cfg,
            data=data,
            device=device,
            eval_batch_size=args.eval_batch_size,
            n_orbits=args.orbits,
            burn_in=args.burn_in,
            max_period=args.max_period,
            cycle_tolerance_relative=args.cycle_tolerance,
            cluster_tolerance_relative=args.cluster_tolerance,
            min_basin_fraction=args.min_basin_fraction,
            root_periods=args.root_periods,
            root_starts=args.root_starts,
            root_residual_tolerance_relative=args.root_residual_tolerance,
            root_dedupe_tolerance_relative=args.root_dedupe_tolerance,
            root_max_nfev=args.root_max_nfev,
        )

    reference = analyze(baseline) if baseline is not None else None
    records = [analyze(spec) for spec in candidates]
    ordered = sorted(records, key=lambda row: tuple(row["screening_criteria"]["lexicographic_sort_key"]))
    ranking = []
    for rank, record in enumerate(ordered, start=1):
        ranking.append(
            {
                "rank": rank,
                "label": record["label"],
                "lexicographic_sort_key": record["screening_criteria"]["lexicographic_sort_key"],
                "heldout_total_loss": record["validation_loss_recomputed"]["loss_total"],
            }
        )
    if reference is not None:
        baseline_loss = float(reference["validation_loss_recomputed"]["loss_total"])
        for record in records:
            current = float(record["validation_loss_recomputed"]["loss_total"])
            record["comparison_to_archived_patrick"] = {
                "heldout_total_loss_difference": current - baseline_loss,
                "heldout_total_loss_fractional_change": (current - baseline_loss) / baseline_loss,
                "basin_pattern_penalty_difference": (
                    record["screening_criteria"]["basin_pattern_penalty"]
                    - reference["screening_criteria"]["basin_pattern_penalty"]
                ),
                "root_pattern_penalty_difference": (
                    record["screening_criteria"]["root_pattern_penalty"]
                    - reference["screening_criteria"]["root_pattern_penalty"]
                ),
            }

    after = _snapshot_inputs(input_paths)
    if before != after:
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        raise RuntimeError(f"screen input changed while it was being read: {changed}")

    return {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "analysis_kind": "non-rigorous pre-CMGDB numerical screen",
        "is_conley_index_computation": False,
        "is_morse_graph_computation": False,
        "warning": (
            "Do not interpret cycle counts or Jacobian classes as Conley indices. "
            "Use this report only to choose which checkpoint receives a rigorous CMGDB run."
        ),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "fixed_validation": {
            "csv": {"path": str(data.csv_path), "sha256": _sha256(data.csv_path)},
            "metadata": {"path": str(data.metadata_path), "sha256": _sha256(data.metadata_path)},
            "scaler": {"path": str(data.scaler_path), "sha256": _sha256(data.scaler_path)},
            "n_transition_pairs": int(data.x_scaled.shape[0]),
            "n_initial_conditions": int(data.initial_scaled.shape[0]),
            "sampling_seed": data.metadata.get("sampling_seed"),
            "model_params": data.metadata.get("model_params"),
        },
        "settings": settings,
        "reference": reference,
        "candidates": records,
        "ranking": ranking,
        "input_hashes_verified_unchanged": True,
    }


def _parse_seeds(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("seed list cannot be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=_parse_seeds, help="comma-separated subset; default config seeds")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--orbits", type=int, default=4096)
    parser.add_argument("--burn-in", type=int, default=600)
    parser.add_argument("--max-period", type=int, default=8)
    parser.add_argument("--cycle-tolerance", type=float, default=1e-5)
    parser.add_argument("--cluster-tolerance", type=float, default=2e-3)
    parser.add_argument("--min-basin-fraction", type=float, default=0.005)
    parser.add_argument("--root-periods", type=_parse_seeds, default=list(DEFAULT_PERIODS))
    parser.add_argument("--root-starts", type=int, default=169)
    parser.add_argument("--root-residual-tolerance", type=float, default=2e-5)
    parser.add_argument("--root-dedupe-tolerance", type=float, default=2e-4)
    parser.add_argument("--root-max-nfev", type=int, default=100)
    parser.add_argument("--output", type=Path, help="optional JSON report; refuses to overwrite")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("eval_batch_size", "orbits", "burn_in", "max_period", "root_starts", "root_max_nfev"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "cycle_tolerance",
        "cluster_tolerance",
        "root_residual_tolerance",
        "root_dedupe_tolerance",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if any(period < 1 for period in args.root_periods):
        raise ValueError("--root-periods entries must be positive")
    if not 0.0 <= args.min_basin_fraction <= 1.0:
        raise ValueError("--min-basin-fraction must lie in [0, 1]")
    report = run_screen(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        output = _resolve_code_path(args.output)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
