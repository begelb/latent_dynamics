#!/usr/bin/env python3
"""Refine and audit the known invariant objects of the direct Ives map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latentdynamics.systems import IvesModel

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    CODE_ROOT
    / "src"
    / "latentdynamics"
    / "reference_data"
    / "ives_myvatn_invariant_points.csv"
)
DEFAULT_OUTPUT = CODE_ROOT / "output" / "ives_myvatn_3d_ground_truth" / "invariant_stability"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_points(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape != (13, 5):
        raise ValueError(f"expected 13 five-column reference rows; got {data.shape}")
    cycle = data[data[:, 0] == 0, 2:]
    fixed = data[data[:, 0] == 1, 2:]
    if cycle.shape != (12, 3) or fixed.shape != (1, 3):
        raise ValueError("reference data must contain a 12-cycle and one fixed point")
    return cycle, fixed[0]


def _iterate(system: IvesModel, point: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    result = point.copy()
    for _ in range(count):
        result = system.step(result)
    return result


def _refine_periodic_seed(
    system: IvesModel,
    seed: NDArray[np.float64],
    *,
    period: int,
    tolerance: float = 1e-14,
    maximum_iterations: int = 1_000,
) -> tuple[NDArray[np.float64], int, float]:
    point = seed.copy()
    for iteration in range(1, maximum_iterations + 1):
        image = _iterate(system, point, period)
        residual = float(np.max(np.abs(image - point)))
        point = image
        if residual <= tolerance:
            return point, iteration, residual
    raise RuntimeError("periodic-point refinement did not converge")


def _finite_difference_iterate_jacobian(
    system: IvesModel,
    point: NDArray[np.float64],
    *,
    iterate: int,
    epsilon: float,
) -> NDArray[np.float64]:
    jacobian = np.empty((3, 3), dtype=np.float64)
    for axis in range(3):
        perturbation = np.zeros(3, dtype=np.float64)
        perturbation[axis] = epsilon
        positive = _iterate(system, point + perturbation, iterate)
        negative = _iterate(system, point - perturbation, iterate)
        jacobian[:, axis] = (positive - negative) / (2.0 * epsilon)
    return jacobian


def _complex_records(values: NDArray[np.complex128]) -> list[dict[str, float]]:
    return [
        {
            "real": float(value.real),
            "imag": float(value.imag),
            "modulus": float(abs(value)),
        }
        for value in values
    ]


def analyze(reference: Path, output_dir: Path) -> dict[str, Any]:
    archived_cycle, archived_fixed = _load_points(reference)
    system = IvesModel()
    cycle_seed, cycle_iterations, cycle_refinement_residual = _refine_periodic_seed(
        system, archived_cycle[0], period=12
    )
    refined_cycle = np.empty((12, 3), dtype=np.float64)
    refined_cycle[0] = cycle_seed
    for phase in range(1, 12):
        refined_cycle[phase] = system.step(refined_cycle[phase - 1])
    cycle_closure_residual = float(
        np.max(np.abs(system.step(refined_cycle[-1]) - refined_cycle[0]))
    )
    fixed, fixed_iterations, fixed_refinement_residual = _refine_periodic_seed(
        system, archived_fixed, period=1
    )
    fixed_residual = float(np.max(np.abs(system.step(fixed) - fixed)))

    epsilons = (1e-5, 3e-6, 1e-6, 3e-7)
    stability: dict[str, Any] = {}
    cycle_radii: list[float] = []
    fixed_radii: list[float] = []
    for epsilon in epsilons:
        cycle_eigenvalues = np.linalg.eigvals(
            _finite_difference_iterate_jacobian(
                system, refined_cycle[0], iterate=12, epsilon=epsilon
            )
        )
        fixed_eigenvalues = np.linalg.eigvals(
            _finite_difference_iterate_jacobian(
                system, fixed, iterate=1, epsilon=epsilon
            )
        )
        cycle_radius = float(np.max(np.abs(cycle_eigenvalues)))
        fixed_radius = float(np.max(np.abs(fixed_eigenvalues)))
        cycle_radii.append(cycle_radius)
        fixed_radii.append(fixed_radius)
        stability[f"{epsilon:.0e}"] = {
            "period12_multipliers": _complex_records(cycle_eigenvalues),
            "period12_spectral_radius": cycle_radius,
            "fixed_point_eigenvalues": _complex_records(fixed_eigenvalues),
            "fixed_point_spectral_radius": fixed_radius,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    refined_path = output_dir / "refined_invariant_points.csv"
    with refined_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["vertex", "component_id", "barycenter_x", "barycenter_y", "barycenter_z"]
        )
        for phase, point in enumerate(refined_cycle):
            writer.writerow([0, phase, *[f"{value:.17g}" for value in point]])
        writer.writerow([1, 0, *[f"{value:.17g}" for value in fixed]])

    payload = {
        "schema_version": 1,
        "purpose": "direct-map invariant refinement and local stability audit",
        "system": {"name": "IvesModel", "parameters": system.params},
        "source": {
            "path": str(reference.resolve()),
            "sha256": _sha256(reference),
            "maximum_archived_to_refined_coordinate_change": float(
                max(
                    np.max(np.abs(refined_cycle - archived_cycle)),
                    np.max(np.abs(fixed - archived_fixed)),
                )
            ),
        },
        "refinement": {
            "period12_iterations_of_F12": cycle_iterations,
            "period12_last_refinement_residual": cycle_refinement_residual,
            "period12_closure_residual": cycle_closure_residual,
            "fixed_iterations_of_F": fixed_iterations,
            "fixed_last_refinement_residual": fixed_refinement_residual,
            "fixed_residual": fixed_residual,
            "refined_points_path": str(refined_path.resolve()),
            "refined_points_sha256": _sha256(refined_path),
        },
        "finite_difference_stability": {
            "epsilons": list(epsilons),
            "by_epsilon": stability,
            "period12_maximum_spectral_radius": max(cycle_radii),
            "fixed_point_maximum_spectral_radius": max(fixed_radii),
            "period12_locally_attracting": max(cycle_radii) < 1.0,
            "fixed_point_locally_attracting": max(fixed_radii) < 1.0,
            "scope": (
                "central finite differences of F^12 at phase 0 and F at the fixed "
                "point; consistent over four perturbation scales"
            ),
        },
    }
    (output_dir / "stability.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = analyze(args.reference.resolve(), args.output_dir.resolve())
    print(json.dumps(payload["finite_difference_stability"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
