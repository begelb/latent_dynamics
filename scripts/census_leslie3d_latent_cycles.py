#!/usr/bin/env python3
"""Dense numerical census of low-period cycles in a learned Leslie3D latent map.

The search solves ``G^p(z) - z = 0`` from a tensor grid, uniform random
starts, and dense starts around the encoded catalogue phases.  Candidate
roots are polished, reduced to their least period, deduplicated up to cyclic
phase shift, and classified by exact PyTorch-autograd monodromy multipliers.

This is a reproducible numerical lower bound on the recurrent inventory, not
an interval-arithmetic proof that no additional cycles exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import scipy
import torch
from numpy.typing import NDArray
from scipy.optimize import least_squares, root
from scipy.special import erf
from torch import nn

from latentdynamics.config import load_config
from latentdynamics.training import load_checkpoint

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = "output/leslie3d_invariant_aware_smooth/seed_20260805/models"
DEFAULT_CHECKPOINT_BASENAME = "autoencoder"
DEFAULT_CONFIG = "leslie3d_invariant_aware_smooth"
KNOWN_OBJECT_ORDER = ("P0", "P1", "S2", "S4", "p_star", "origin")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else CODE_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_paths(checkpoint_dir: Path, basename: str) -> tuple[Path, Path]:
    if not basename or Path(basename).name != basename or basename in {".", ".."}:
        raise ValueError("checkpoint basename must be a plain non-empty file basename")
    return checkpoint_dir / f"{basename}.pt", checkpoint_dir / f"{basename}.json"


def _default_output_name(*, map_mode: str, checkpoint_basename: str) -> str:
    basename_suffix = (
        "" if checkpoint_basename == DEFAULT_CHECKPOINT_BASENAME else f"_{checkpoint_basename}"
    )
    mode_suffix = "" if map_mode == "latent" else "_decoder_closed"
    return f"dense_periodic_root_census{basename_suffix}{mode_suffix}.json"


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not parsed or parsed[0] <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def _parse_floats(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected comma-separated floating-point values")
    return parsed


def _parse_pair(value: str) -> tuple[float, float]:
    parsed = _parse_floats(value)
    if len(parsed) != 2:
        raise argparse.ArgumentTypeError("expected exactly two comma-separated values")
    return parsed


@dataclass(frozen=True)
class _Layer:
    kind: str
    weight: NDArray[np.float64] | None = None
    bias: NDArray[np.float64] | None = None


class NumpyIteratedMap(Protocol):
    """Structural interface consumed by the root census."""

    def value_and_jacobian(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def iterate(
        self, points: NDArray[np.float64], period: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        values = np.atleast_2d(np.asarray(points, dtype=np.float64)).copy()
        total = np.broadcast_to(np.eye(2), (len(values), 2, 2)).copy()
        for _ in range(period):
            values, local = self.value_and_jacobian(values)
            total = np.einsum("nij,njk->nik", local, total)
        return values, total

    def residual(self, points: NDArray[np.float64], period: int) -> NDArray[np.float64]:
        starts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        values, _ = self.iterate(starts, period)
        return values - starts


class NumpySequentialMap(NumpyIteratedMap):
    """NumPy value/Jacobian evaluator for the repository's smooth 2-D MLPs."""

    def __init__(self, module: nn.Sequential) -> None:
        layers: list[_Layer] = []
        for item in module.children():
            if isinstance(item, nn.Linear):
                layers.append(
                    _Layer(
                        "linear",
                        item.weight.detach().cpu().double().numpy().copy(),
                        item.bias.detach().cpu().double().numpy().copy(),
                    )
                )
            elif isinstance(item, nn.GELU):
                if item.approximate != "none":
                    raise ValueError("only exact nn.GELU(approximate='none') is supported")
                layers.append(_Layer("gelu"))
            elif isinstance(item, nn.Tanh):
                layers.append(_Layer("tanh"))
            elif isinstance(item, nn.Sigmoid):
                layers.append(_Layer("sigmoid"))
            elif isinstance(item, nn.ReLU):
                layers.append(_Layer("relu"))
            else:
                raise TypeError(f"unsupported latent-map layer {type(item).__name__}")
        if not layers or layers[0].kind != "linear" or layers[0].weight is None:
            raise ValueError("latent map must be a non-empty sequential MLP")
        self.layers = tuple(layers)
        self.input_dimension = int(layers[0].weight.shape[1])

    def value_and_jacobian(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        values = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if values.shape[1] != self.input_dimension:
            raise ValueError(f"expected shape (n, {self.input_dimension}), got {values.shape}")
        jacobian = np.broadcast_to(
            np.eye(self.input_dimension),
            (len(values), self.input_dimension, self.input_dimension),
        ).copy()
        for layer in self.layers:
            if layer.kind == "linear":
                assert layer.weight is not None and layer.bias is not None
                values = values @ layer.weight.T + layer.bias
                jacobian = np.einsum("ij,njk->nik", layer.weight, jacobian)
            elif layer.kind == "gelu":
                derivative = 0.5 * (1.0 + erf(values / math.sqrt(2.0)))
                derivative += values * np.exp(-0.5 * values * values) / math.sqrt(2.0 * math.pi)
                values = 0.5 * values * (1.0 + erf(values / math.sqrt(2.0)))
                jacobian *= derivative[:, :, None]
            elif layer.kind == "tanh":
                values = np.tanh(values)
                jacobian *= (1.0 - values * values)[:, :, None]
            elif layer.kind == "sigmoid":
                values = 1.0 / (1.0 + np.exp(-values))
                jacobian *= (values * (1.0 - values))[:, :, None]
            elif layer.kind == "relu":
                derivative = (values > 0.0).astype(np.float64)
                values = np.maximum(values, 0.0)
                jacobian *= derivative[:, :, None]
            else:  # pragma: no cover - constructor validates this invariant.
                raise AssertionError(layer.kind)
        return values, jacobian


class DecoderClosedLeslie3DMap(NumpyIteratedMap):
    """Evaluate ``z -> E(scale(f(unscale(D(z)))))`` and its Jacobian."""

    def __init__(
        self,
        *,
        decoder: nn.Sequential,
        encoder: nn.Sequential,
        scaler: Any,
        params: dict[str, Any],
    ) -> None:
        self.decoder = NumpySequentialMap(decoder)
        self.encoder = NumpySequentialMap(encoder)
        self.scale = np.asarray(scaler.scale_, dtype=np.float64)
        self.offset = np.asarray(scaler.min_, dtype=np.float64)
        if self.scale.shape != (3,) or self.offset.shape != (3,):
            raise ValueError("decoder-closed Leslie3D mode requires a three-coordinate scaler")
        self.theta = np.asarray([params["th1"], params["th2"], params["th3"]], dtype=np.float64)
        self.survival_p1 = float(params["survival_p1"])
        self.survival_p2 = float(params["survival_p2"])

    def value_and_jacobian(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        decoded_scaled, decoder_jacobian = self.decoder.value_and_jacobian(points)
        physical = (decoded_scaled - self.offset) / self.scale
        total_population = physical.sum(axis=1)
        weighted_population = physical @ self.theta
        decay = np.exp(-0.1 * total_population)
        next_physical = np.column_stack(
            (
                weighted_population * decay,
                self.survival_p1 * physical[:, 0],
                self.survival_p2 * physical[:, 1],
            )
        )
        physical_jacobian = np.zeros((len(physical), 3, 3), dtype=np.float64)
        physical_jacobian[:, 0, :] = decay[:, None] * (
            self.theta[None, :] - 0.1 * weighted_population[:, None]
        )
        physical_jacobian[:, 1, 0] = self.survival_p1
        physical_jacobian[:, 2, 1] = self.survival_p2
        scaled_physical_jacobian = (
            self.scale[None, :, None] * physical_jacobian / self.scale[None, None, :]
        )
        next_scaled = next_physical * self.scale + self.offset
        encoded, encoder_jacobian = self.encoder.value_and_jacobian(next_scaled)
        jacobian = np.einsum(
            "nij,njk,nkl->nil",
            encoder_jacobian,
            scaled_physical_jacobian,
            decoder_jacobian,
        )
        return encoded, jacobian


def _decoder_closed_torch_map(model: nn.Module, scaler: Any, params: dict[str, Any]) -> Any:
    scale = torch.tensor(scaler.scale_, dtype=torch.float64)
    offset = torch.tensor(scaler.min_, dtype=torch.float64)
    theta = torch.tensor([params["th1"], params["th2"], params["th3"]], dtype=torch.float64)
    survival_p1 = float(params["survival_p1"])
    survival_p2 = float(params["survival_p2"])

    def map_value(latent: torch.Tensor) -> torch.Tensor:
        decoded_scaled = model.decoder(latent)
        physical = (decoded_scaled - offset) / scale
        decay = torch.exp(-0.1 * physical.sum(dim=-1))
        head = (physical * theta).sum(dim=-1) * decay
        next_physical = torch.stack(
            (head, survival_p1 * physical[..., 0], survival_p2 * physical[..., 1]),
            dim=-1,
        )
        return model.encoder(next_physical * scale + offset)

    return map_value


def canonical_cycle(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Choose a deterministic representative among cyclic phase rotations."""

    cycle = np.asarray(points, dtype=np.float64)
    rotations = [np.roll(cycle, -shift, axis=0) for shift in range(len(cycle))]
    return min(rotations, key=lambda item: tuple(np.round(item.ravel(), 12)))


def cycles_equivalent(
    left: NDArray[np.float64], right: NDArray[np.float64], tolerance: float
) -> bool:
    if len(left) != len(right):
        return False
    return bool(np.max(np.linalg.norm(left - right, axis=1)) < tolerance)


def _symmetric_hausdorff(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def _cyclic_alignment(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> dict[str, float] | None:
    if len(left) != len(right):
        return None
    values: list[tuple[float, float]] = []
    for shift in range(len(left)):
        distances = np.linalg.norm(np.roll(left, -shift, axis=0) - right, axis=1)
        values.append((float(distances.max()), float(distances.mean())))
    maximum, mean = min(values)
    return {"max_l2": maximum, "mean_l2": mean}


def _build_starts(
    known: dict[str, NDArray[np.float64]],
    *,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    grid_points: int,
    random_starts: int,
    local_starts_per_phase: int,
    seed: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    if grid_points < 2 or random_starts < 0 or local_starts_per_phase < 1:
        raise ValueError("invalid start-count arguments")
    rng = np.random.default_rng(seed)
    inset = np.minimum(1e-3, 1e-3 * (upper - lower))
    axes = [np.linspace(lower[i] + inset[i], upper[i] - inset[i], grid_points) for i in range(2)]
    first, second = np.meshgrid(*axes, indexing="xy")
    blocks = [np.column_stack((first.ravel(), second.ravel()))]
    if random_starts:
        blocks.append(rng.uniform(lower, upper, size=(random_starts, 2)))
    phase_count = 0
    for points in known.values():
        if len(points) > 1:
            scale = min(
                np.linalg.norm(points[i] - points[j])
                for i in range(len(points))
                for j in range(i + 1, len(points))
            )
        else:
            scale = 0.05
        radius = max(0.03, min(0.09, 0.8 * scale))
        for point in points:
            phase_count += 1
            blocks.append(point[None, :])
            remaining = local_starts_per_phase - 1
            if remaining:
                angles = rng.uniform(0.0, 2.0 * math.pi, remaining)
                radii = radius * np.sqrt(rng.random(remaining))
                local = point + np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
                blocks.append(np.clip(local, lower, upper))
    starts = np.vstack(blocks)
    metadata = {
        "total_unique_starts_per_period_equation": len(starts),
        "global_tensor_grid": {
            "points_per_axis": grid_points,
            "count": grid_points * grid_points,
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "boundary_inset": inset.tolist(),
        },
        "global_uniform_random": {
            "count": random_starts,
            "seed": seed,
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
        "local_random": {
            "count_per_encoded_phase_including_exact_phase": local_starts_per_phase,
            "encoded_phase_count": phase_count,
            "count": local_starts_per_phase * phase_count,
            "seed": seed,
            "radius_formula": (
                "max(0.03, min(0.09, 0.8 * minimum within-object encoded phase "
                "separation)); fixed-point separation defaults to 0.05"
            ),
        },
    }
    return starts, metadata


def _batched_newton(
    evaluator: NumpyIteratedMap,
    starts: NDArray[np.float64],
    period: int,
    *,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    step_cap: float,
    iterations: int,
    convergence: float,
    candidate_tolerance: float,
) -> NDArray[np.float64]:
    points = np.asarray(starts, dtype=np.float64).copy()
    active = np.arange(len(points))
    converged: list[int] = []
    identity = np.eye(2)
    for _ in range(iterations):
        if not len(active):
            break
        current = points[active]
        values, derivative = evaluator.iterate(current, period)
        residual = values - current
        norms = np.linalg.norm(residual, axis=1)
        done = norms < convergence
        if done.any():
            converged.extend(active[done].tolist())
            active = active[~done]
            current = current[~done]
            residual = residual[~done]
            derivative = derivative[~done]
        if not len(active):
            break
        jacobian = derivative - identity
        determinant = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
        regular = np.abs(determinant) > 1e-11
        step = np.zeros_like(residual)
        step[regular, 0] = (
            jacobian[regular, 1, 1] * residual[regular, 0]
            - jacobian[regular, 0, 1] * residual[regular, 1]
        ) / determinant[regular]
        step[regular, 1] = (
            -jacobian[regular, 1, 0] * residual[regular, 0]
            + jacobian[regular, 0, 0] * residual[regular, 1]
        ) / determinant[regular]
        if (~regular).any():
            singular = jacobian[~regular]
            hessian = np.einsum("nji,njk->nik", singular, singular) + 1e-5 * identity
            gradient = np.einsum("nji,nj->ni", singular, residual[~regular])
            step[~regular] = np.linalg.solve(hessian, gradient)
        lengths = np.linalg.norm(step, axis=1)
        step *= np.minimum(1.0, step_cap / np.maximum(lengths, 1e-15))[:, None]
        points[active] = np.clip(current - step, lower, upper)
    if len(active):
        residual = np.linalg.norm(evaluator.residual(points[active], period), axis=1)
        converged.extend(active[residual < candidate_tolerance].tolist())
    return points[np.unique(converged)]


def _polish_root(
    evaluator: NumpyIteratedMap, point: NDArray[np.float64], period: int
) -> tuple[NDArray[np.float64], float]:
    def residual(value: NDArray[np.float64]) -> NDArray[np.float64]:
        return evaluator.residual(value, period)[0]

    def jacobian(value: NDArray[np.float64]) -> NDArray[np.float64]:
        _, derivative = evaluator.iterate(value, period)
        return derivative[0] - np.eye(2)

    solution = root(
        residual,
        point,
        jac=jacobian,
        method="hybr",
        options={"xtol": 1e-11, "maxfev": 1000},
    )
    norm = float(np.linalg.norm(residual(solution.x)))
    if norm > 1e-11:
        fallback = least_squares(
            residual,
            point,
            jac=jacobian,
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
            max_nfev=2000,
        )
        solution_x = fallback.x
        norm = float(np.linalg.norm(residual(solution_x)))
    else:
        solution_x = solution.x
    return np.asarray(solution_x, dtype=np.float64), norm


def _orbit(
    evaluator: NumpyIteratedMap, point: NDArray[np.float64], period: int
) -> NDArray[np.float64]:
    phases: list[NDArray[np.float64]] = []
    value = np.asarray(point, dtype=np.float64)[None, :]
    for _ in range(period):
        phases.append(value[0].copy())
        value, _ = evaluator.iterate(value, 1)
    return np.asarray(phases)


def _least_period(
    evaluator: NumpyIteratedMap,
    point: NDArray[np.float64],
    equation_period: int,
    tolerance: float,
) -> int:
    for divisor in range(1, equation_period):
        if equation_period % divisor == 0:
            residual = np.linalg.norm(evaluator.residual(point, divisor))
            if residual < tolerance:
                return divisor
    return equation_period


def _torch_monodromy(
    map_value: Any, point: NDArray[np.float64], period: int
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    value = torch.tensor(point, dtype=torch.float64, requires_grad=True)

    def iterate(item: torch.Tensor) -> torch.Tensor:
        for _ in range(period):
            item = map_value(item)
        return item

    matrix = (
        torch.autograd.functional.jacobian(iterate, value, vectorize=True).detach().cpu().numpy()
    )
    return matrix, np.linalg.eigvals(matrix).astype(np.complex128)


def census(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = _resolve(args.checkpoint_dir)
    checkpoint_path, sidecar_path = _checkpoint_paths(checkpoint_dir, args.checkpoint_basename)
    model, arch = load_checkpoint(checkpoint_dir, basename=args.checkpoint_basename)
    model = model.double().eval()
    if arch.low_dims != 2:
        raise ValueError("cycle census currently requires a two-dimensional latent map")
    cfg = load_config(args.config)
    scaler = joblib.load(_resolve(cfg.paths.scaler_path("train")))
    if args.map_mode == "latent":
        evaluator: NumpyIteratedMap = NumpySequentialMap(model.latent_map.net)
        torch_map = model.latent_map
        map_formula = "G(z)"
    elif args.map_mode == "decoder_closed":
        if cfg.system.name != "leslie3d":
            raise ValueError("decoder-closed map mode currently supports only system.name=leslie3d")
        evaluator = DecoderClosedLeslie3DMap(
            decoder=model.decoder.net,
            encoder=model.encoder.net,
            scaler=scaler,
            params=cfg.system.params,
        )
        torch_map = _decoder_closed_torch_map(model, scaler, cfg.system.params)
        map_formula = "E(scale(f(unscale(D(z)))))"
    else:  # pragma: no cover - argparse validates the choice.
        raise AssertionError(args.map_mode)
    manifest_path = (
        _resolve(args.manifest)
        if args.manifest
        else _resolve(Path(cfg.paths.data_dir) / "dataset_manifest.json")
    )
    manifest = json.loads(manifest_path.read_text())
    known: dict[str, NDArray[np.float64]] = {}
    with torch.no_grad():
        for name in KNOWN_OBJECT_ORDER:
            physical = np.asarray(manifest["known_objects"][name]["points"], dtype=np.float64)
            scaled = torch.tensor(scaler.transform(physical), dtype=torch.float64)
            known[name] = model.encoder(scaled).cpu().numpy()

    search_lower = np.asarray(args.search_lower, dtype=np.float64)
    search_upper = np.asarray(args.search_upper, dtype=np.float64)
    if np.any(search_lower >= search_upper):
        raise ValueError("search lower bounds must be below upper bounds")
    starts, starts_metadata = _build_starts(
        known,
        lower=search_lower,
        upper=search_upper,
        grid_points=args.grid_points,
        random_starts=args.random_starts,
        local_starts_per_phase=args.local_starts_per_phase,
        seed=args.seed,
    )

    raw: list[tuple[int, NDArray[np.float64]]] = []
    for period in args.periods:
        candidates: list[NDArray[np.float64]] = []
        for step_cap in args.newton_step_caps:
            chunks = np.array_split(starts, math.ceil(len(starts) / args.chunk_size))
            for chunk in chunks:
                found = _batched_newton(
                    evaluator,
                    chunk,
                    period,
                    lower=search_lower,
                    upper=search_upper,
                    step_cap=step_cap,
                    iterations=args.newton_iterations,
                    convergence=args.newton_convergence,
                    candidate_tolerance=args.candidate_tolerance,
                )
                if len(found):
                    candidates.append(found)
        if not candidates:
            continue
        stacked = np.vstack(candidates)
        _, indices = np.unique(
            np.round(stacked, args.candidate_round_decimals), axis=0, return_index=True
        )
        for candidate in stacked[np.sort(indices)]:
            polished, residual = _polish_root(evaluator, candidate, period)
            if (
                residual <= args.root_acceptance
                and np.all(polished >= search_lower)
                and np.all(polished <= search_upper)
            ):
                raw.append((period, polished))

    orbit_records: list[dict[str, Any]] = []
    for equation_period, point in raw:
        least_period = _least_period(evaluator, point, equation_period, args.lower_period_tolerance)
        polished, residual = _polish_root(evaluator, point, least_period)
        phases = canonical_cycle(_orbit(evaluator, polished, least_period))
        existing = next(
            (
                record
                for record in orbit_records
                if record["least_period"] == least_period
                and cycles_equivalent(
                    phases, record["phase_points_array"], args.orbit_deduplication
                )
            ),
            None,
        )
        if existing is not None:
            existing["discovered_from_period_equations"].add(equation_period)
            continue
        orbit_records.append(
            {
                "least_period": least_period,
                "phase_points_array": phases,
                "period_equation_residual_l2": residual,
                "discovered_from_period_equations": {equation_period},
            }
        )

    diagnose_path = (
        _resolve(args.diagnose) if args.diagnose else checkpoint_dir.parent / "diagnose.json"
    )
    if diagnose_path.exists():
        diagnose = json.loads(diagnose_path.read_text())
        cmgdb_lower = np.asarray(diagnose["bounds"]["lower"], dtype=np.float64)
        cmgdb_upper = np.asarray(diagnose["bounds"]["upper"], dtype=np.float64)
        cmgdb_bounds: dict[str, Any] | None = {
            "lower": cmgdb_lower.tolist(),
            "upper": cmgdb_upper.tolist(),
            "source": diagnose["bounds"]["source"],
            "diagnose_path": str(diagnose_path),
        }
    elif cfg.cmgdb.lower_bounds is not None and cfg.cmgdb.upper_bounds is not None:
        cmgdb_lower = np.asarray(cfg.cmgdb.lower_bounds, dtype=np.float64)
        cmgdb_upper = np.asarray(cfg.cmgdb.upper_bounds, dtype=np.float64)
        cmgdb_bounds = {
            "lower": cmgdb_lower.tolist(),
            "upper": cmgdb_upper.tolist(),
            "source": "explicit config.cmgdb bounds (diagnose.json absent)",
            "config": args.config,
        }
    else:
        cmgdb_lower = cmgdb_upper = None
        cmgdb_bounds = None

    for record in orbit_records:
        phases = record["phase_points_array"]
        distances = {
            name: {
                "symmetric_hausdorff_l2": _symmetric_hausdorff(phases, encoded),
                "cyclic_alignment": _cyclic_alignment(phases, encoded),
            }
            for name, encoded in known.items()
        }
        nearest = min(distances, key=lambda name: distances[name]["symmetric_hausdorff_l2"])
        matrix, multipliers = _torch_monodromy(torch_map, phases[0], record["least_period"])
        lower_returns = {
            str(step): float(np.linalg.norm(evaluator.residual(phases[0], step)))
            for step in range(1, record["least_period"])
        }
        unstable = int(np.sum(np.abs(multipliers) > 1.0))
        record.update(
            {
                "representative": phases[0].tolist(),
                "phase_points": phases.tolist(),
                "lower_iterate_returns_l2": lower_returns,
                "least_period_verified": all(
                    value > args.lower_period_tolerance for value in lower_returns.values()
                ),
                "monodromy_matrix": matrix.tolist(),
                "monodromy_trace": float(np.trace(matrix)),
                "monodromy_determinant": float(np.linalg.det(matrix)),
                "multipliers": [
                    {
                        "real": float(value.real),
                        "imag": float(value.imag),
                        "modulus": float(abs(value)),
                    }
                    for value in multipliers
                ],
                "unstable_dimension": unstable,
                "role": {0: "sink", 1: "saddle", 2: "repeller"}.get(
                    unstable, f"{unstable}_unstable_directions"
                ),
                "association": {
                    "nearest_known_object": nearest,
                    "nearest_symmetric_hausdorff_l2": distances[nearest]["symmetric_hausdorff_l2"],
                    "distances_to_known_objects": distances,
                    "is_intended_catalogue_cycle": False,
                    "intended_object": None,
                },
                "inside_cmgdb_bounds": (
                    bool(np.all(phases >= cmgdb_lower) and np.all(phases <= cmgdb_upper))
                    if cmgdb_lower is not None and cmgdb_upper is not None
                    else None
                ),
            }
        )

    for name, encoded in known.items():
        period = len(encoded)
        candidates = [record for record in orbit_records if record["least_period"] == period]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda record: record["association"]["distances_to_known_objects"][name][
                "symmetric_hausdorff_l2"
            ],
        )
        distance = selected["association"]["distances_to_known_objects"][name][
            "symmetric_hausdorff_l2"
        ]
        if distance <= args.intended_distance_threshold:
            selected["association"]["is_intended_catalogue_cycle"] = True
            selected["association"]["intended_object"] = name

    order = {name: index for index, name in enumerate(KNOWN_OBJECT_ORDER)}
    orbit_records.sort(
        key=lambda record: (
            record["least_period"],
            order[record["association"]["nearest_known_object"]],
            record["association"]["nearest_symmetric_hausdorff_l2"],
            record["representative"],
        )
    )
    for index, record in enumerate(orbit_records, 1):
        record["cycle_id"] = f"C{index:02d}"
        record["discovered_from_period_equations"] = sorted(
            record["discovered_from_period_equations"]
        )
        del record["phase_points_array"]

    intended = sum(record["association"]["is_intended_catalogue_cycle"] for record in orbit_records)
    inside = sum(record["inside_cmgdb_bounds"] is True for record in orbit_records)
    outside = sum(record["inside_cmgdb_bounds"] is False for record in orbit_records)
    unclassified = sum(record["inside_cmgdb_bounds"] is None for record in orbit_records)
    extra_inside = sum(
        record["inside_cmgdb_bounds"] is True
        and not record["association"]["is_intended_catalogue_cycle"]
        for record in orbit_records
    )
    extra_outside = sum(
        record["inside_cmgdb_bounds"] is False
        and not record["association"]["is_intended_catalogue_cycle"]
        for record in orbit_records
    )
    extra_unclassified = sum(
        record["inside_cmgdb_bounds"] is None
        and not record["association"]["is_intended_catalogue_cycle"]
        for record in orbit_records
    )
    by_period = {
        str(period): sum(record["least_period"] == period for record in orbit_records)
        for period in args.periods
    }
    inside_by_period = {
        str(period): sum(
            record["least_period"] == period and record["inside_cmgdb_bounds"] is True
            for record in orbit_records
        )
        for period in args.periods
    }
    artifact = {
        "schema_version": 1,
        "experiment": cfg.experiment_name,
        "map_mode": {"name": args.map_mode, "formula": map_formula},
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "basename": args.checkpoint_basename,
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "architecture_sidecar": str(sidecar_path),
            "architecture_sidecar_sha256": _sha256(sidecar_path),
        },
        "status": "dense_numerical_census_lower_bound_not_a_proof_of_exhaustiveness",
        "method": {
            "period_equations": list(args.periods),
            "global_search_domain": {
                "lower": search_lower.tolist(),
                "upper": search_upper.tolist(),
                "rationale": (
                    "For this terminal-tanh latent map, every periodic point lies in [-1,1]^2."
                ),
            },
            "initial_starts": starts_metadata,
            "solver": {
                "batched_newton_step_caps": list(args.newton_step_caps),
                "batched_newton_max_iterations": args.newton_iterations,
                "batched_convergence_l2": args.newton_convergence,
                "candidate_acceptance_l2": args.candidate_tolerance,
                "candidate_round_decimals": args.candidate_round_decimals,
                "polish": "scipy.optimize.root(hybr), then least_squares fallback",
                "final_root_acceptance_l2": args.root_acceptance,
                "cyclic_orbit_deduplication_l2": args.orbit_deduplication,
            },
            "classification": {
                "lower_period_tolerance_l2": args.lower_period_tolerance,
                "multipliers": (
                    f"Exact PyTorch autograd Jacobian of ({map_formula})^p at the polished "
                    "root, using checkpoint coefficients evaluated in float64."
                ),
                "association_metric": (
                    "Symmetric Hausdorff L2 distance between learned and encoded phase sets; "
                    "cyclic alignment is also reported when periods match."
                ),
                "intended_distance_threshold_l2": args.intended_distance_threshold,
                "cmgdb_inside_rule": (
                    "Every learned phase lies componentwise inside the diagnose.json bounds."
                ),
            },
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
            },
            "reproduction_script": str(Path(__file__).resolve()),
            "reproduction_checkpoint_basename": args.checkpoint_basename,
        },
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "cmgdb_bounds": cmgdb_bounds,
        "counts": {
            "total_cycles": len(orbit_records),
            "by_least_period": by_period,
            "inside_cmgdb_bounds": inside,
            "outside_cmgdb_bounds": outside,
            "unclassified_cmgdb_bounds": unclassified,
            "inside_cmgdb_bounds_by_least_period": inside_by_period,
            "intended_catalogue_cycles": intended,
            "extra_cycles_total": len(orbit_records) - intended,
            "extra_cycles_inside_cmgdb_bounds": extra_inside,
            "extra_cycles_outside_cmgdb_bounds": extra_outside,
            "extra_cycles_unclassified_cmgdb_bounds": extra_unclassified,
        },
        "cycles": orbit_records,
    }
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-basename",
        default=DEFAULT_CHECKPOINT_BASENAME,
        help="checkpoint pair basename, for example autoencoder or smooth_candidate",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--diagnose", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--map-mode", choices=("latent", "decoder_closed"), default="latent")
    parser.add_argument("--periods", type=_parse_ints, default=(1, 2, 4))
    parser.add_argument("--search-lower", type=_parse_pair, default=(-1.0, -1.0))
    parser.add_argument("--search-upper", type=_parse_pair, default=(1.0, 1.0))
    parser.add_argument("--grid-points", type=int, default=81)
    parser.add_argument("--random-starts", type=int, default=8000)
    parser.add_argument("--local-starts-per-phase", type=int, default=400)
    parser.add_argument("--seed", type=int, default=202608051)
    parser.add_argument("--newton-step-caps", type=_parse_floats, default=(0.12, 0.35))
    parser.add_argument("--newton-iterations", type=int, default=45)
    parser.add_argument("--newton-convergence", type=float, default=1e-10)
    parser.add_argument("--candidate-tolerance", type=float, default=1e-8)
    parser.add_argument("--candidate-round-decimals", type=int, default=6)
    parser.add_argument("--root-acceptance", type=float, default=1e-9)
    parser.add_argument("--lower-period-tolerance", type=float, default=1e-8)
    parser.add_argument("--orbit-deduplication", type=float, default=1e-6)
    parser.add_argument("--intended-distance-threshold", type=float, default=1e-3)
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args()
    checkpoint_dir = _resolve(args.checkpoint_dir)
    default_name = _default_output_name(
        map_mode=args.map_mode,
        checkpoint_basename=args.checkpoint_basename,
    )
    output = (
        _resolve(args.output) if args.output else checkpoint_dir.parent / "analysis" / default_name
    )
    result = census(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
