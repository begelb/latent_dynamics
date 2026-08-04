#!/usr/bin/env python3
"""Analyze the recurrent-skeleton-aware Leslie3D latent experiment.

The script is useful both before and after CMGDB.  It always evaluates the
trained model and Patrick warm-start baseline on the curated validation set
and on every exact recurrent phase.  When ``MG/morse_graph`` and
``MG/morse_sets`` exist, it additionally:

* assigns every actual encoded phase ``E(x)`` to a recurrent box (or reports
  its exact nearest-cover distance);
* compares role-aligned indices and reachability with the direct-map target;
* renders ``E(P0), E(P1), E(S2), E(S4), E(p_*), E(0)`` over the Morse sets.

These are numerical diagnostics.  A finite loss, point-sampled box map, or
matching graph is not promoted to a rigorous semiconjugacy/Conley certificate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from scipy.optimize import root

from latentdynamics.analysis.morse_graph_parser import MorseGraph
from latentdynamics.config import load_config
from latentdynamics.training import load_any_checkpoint
from latentdynamics.viz.morse_plots import plot_morse_sets_from_csv

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "leslie3d_invariant_aware"
DEFAULT_CHECKPOINT_BASENAME = "autoencoder"
PATRICK_BASELINE = CODE_ROOT / "replay_sources" / "leslie3d_example2" / "models"
OBJECT_ORDER = ("P0", "P1", "S2", "S4", "p_star", "origin")
OBJECT_STYLE = {
    "P0": {"color": "#FFB000", "marker": "o", "label": r"$E(P_0)$"},
    "P1": {"color": "#DC267F", "marker": "s", "label": r"$E(P_1)$"},
    "S2": {"color": "#FE6100", "marker": "^", "label": r"$E(S_2)$"},
    "S4": {"color": "#648FFF", "marker": "D", "label": r"$E(S_4)$"},
    "p_star": {"color": "#785EF0", "marker": "*", "label": r"$E(p_*)$"},
    "origin": {"color": "#008080", "marker": "X", "label": r"$E(0)$"},
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else CODE_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_smooth_promotion(output_dir: Path) -> dict[str, Any] | None:
    """Prevent a rejected latest run from being paired with a stale promotion."""
    summary_path = output_dir / "smooth_topology_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "accepted_numerical_candidate_not_a_conley_certificate":
        raise RuntimeError(
            "latest smooth topology run was not accepted; refusing to analyze a "
            "possibly stale promoted checkpoint"
        )
    if not summary.get("promoted_checkpoint"):
        raise RuntimeError("accepted smooth summary does not identify a promoted checkpoint")
    expected_hashes = summary.get("promoted_checkpoint_sha256")
    verified_hashes: dict[str, str] | None = None
    if expected_hashes:
        verified_hashes = {}
        for name, expected in expected_hashes.items():
            path = output_dir / "models" / name
            if not path.is_file():
                raise FileNotFoundError(f"promoted smooth checkpoint file is missing: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise RuntimeError(
                    f"promoted smooth checkpoint hash mismatch for {path}: "
                    f"expected {expected}, observed {actual}"
                )
            verified_hashes[name] = actual
    return {
        "summary_path": str(summary_path),
        "status": summary["status"],
        "checkpoint_role": "promoted",
        "promoted_checkpoint_sha256": verified_hashes,
    }


def _validate_smooth_candidate(output_dir: Path) -> dict[str, Any] | None:
    """Verify an explicitly requested candidate without requiring promotion."""

    summary_path = output_dir / "smooth_topology_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    expected_hashes = summary.get("candidate_checkpoint_sha256")
    if not expected_hashes:
        raise RuntimeError("smooth summary does not identify hashed candidate checkpoint files")
    verified_hashes: dict[str, str] = {}
    for name in ("smooth_candidate.pt", "smooth_candidate.json"):
        if name not in expected_hashes:
            raise RuntimeError(f"smooth summary is missing the candidate hash for {name}")
        path = output_dir / "models" / name
        if not path.is_file():
            raise FileNotFoundError(f"smooth candidate checkpoint file is missing: {path}")
        actual = _sha256(path)
        if actual != expected_hashes[name]:
            raise RuntimeError(
                f"smooth candidate checkpoint hash mismatch for {path}: "
                f"expected {expected_hashes[name]}, observed {actual}"
            )
        verified_hashes[name] = actual
    return {
        "summary_path": str(summary_path),
        "status": summary.get("status"),
        "checkpoint_role": "candidate_not_assumed_promoted",
        "candidate_checkpoint_sha256": verified_hashes,
    }


def _checkpoint_artifact_suffix(checkpoint_basename: str) -> str:
    if (
        not checkpoint_basename
        or Path(checkpoint_basename).name != checkpoint_basename
        or checkpoint_basename in {".", ".."}
    ):
        raise ValueError("checkpoint basename must be a plain non-empty file basename")
    return "" if checkpoint_basename == DEFAULT_CHECKPOINT_BASENAME else f"_{checkpoint_basename}"


def _load_pairs(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[:, :3], data[:, 3:6]


def _object_arrays(manifest: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
    return {
        name: np.asarray(manifest["known_objects"][name]["points"], dtype=np.float64)
        for name in OBJECT_ORDER
    }


@torch.no_grad()
def _evaluate_pairs(
    model: torch.nn.Module,
    scaler: Any,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    batch_size: int = 16384,
) -> dict[str, NDArray[np.float64]]:
    model.eval().cpu()
    x_scaled = scaler.transform(x)
    y_scaled = scaler.transform(y)
    values: dict[str, list[NDArray[np.float64]]] = {
        "reconstruction": [],
        "prediction": [],
        "semiconjugacy": [],
        "cycle": [],
    }
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        xt = torch.as_tensor(x_scaled[start:stop], dtype=torch.float32)
        yt = torch.as_tensor(y_scaled[start:stop], dtype=torch.float32)
        fp = model(xt, yt)
        values["reconstruction"].append(torch.mean((fp.x_t_hat - fp.x_t) ** 2, dim=1).cpu().numpy())
        values["prediction"].append(torch.mean((fp.x_tau_hat - fp.x_tau) ** 2, dim=1).cpu().numpy())
        values["semiconjugacy"].append(
            torch.mean((fp.z_tau_pred - fp.z_tau) ** 2, dim=1).cpu().numpy()
        )
        values["cycle"].append(
            torch.mean((fp.z_tau_pred_cycle - fp.z_tau_pred) ** 2, dim=1).cpu().numpy()
        )
    return {name: np.concatenate(chunks) for name, chunks in values.items()}


def _summarize(values: NDArray[np.float64]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _component_summaries(
    losses: dict[str, NDArray[np.float64]], metadata: dict[str, Any]
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for component in metadata["components"]:
        start = int(component["row_start_inclusive"])
        stop = int(component["row_stop_exclusive"])
        out[component["name"]] = {
            name: _summarize(values[start:stop]) for name, values in losses.items()
        }
    return out


@torch.no_grad()
def _encode(
    model: torch.nn.Module, scaler: Any, points: NDArray[np.float64]
) -> NDArray[np.float64]:
    scaled = scaler.transform(points)
    return model.encoder(torch.as_tensor(scaled, dtype=torch.float32)).cpu().numpy()


@torch.no_grad()
def _advance(model: torch.nn.Module, points: NDArray[np.float64]) -> NDArray[np.float64]:
    return model.latent_map(torch.as_tensor(points, dtype=torch.float32)).cpu().numpy()


def _latent_monodromy(
    model: torch.nn.Module, points: NDArray[np.float64]
) -> NDArray[np.complex128]:
    monodromy = np.eye(points.shape[1], dtype=np.float64)
    for point in points:
        with torch.enable_grad():
            value = torch.tensor(point, dtype=torch.float32, requires_grad=True)
            jacobian = torch.autograd.functional.jacobian(
                lambda item: model.latent_map(item.unsqueeze(0)).squeeze(0),
                value,
                vectorize=True,
            )
        monodromy = jacobian.detach().cpu().numpy().astype(np.float64) @ monodromy
    return np.linalg.eigvals(monodromy).astype(np.complex128)


def _learned_cycle_diagnostics(
    model: torch.nn.Module,
    scaler: Any,
    objects: dict[str, NDArray[np.float64]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray[np.float64]]]:
    encoded: dict[str, NDArray[np.float64]] = {}
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for name, points in objects.items():
        z = _encode(model, scaler, points)
        z_true_next = np.roll(z, -1, axis=0)
        z_pred = _advance(model, z)
        decoded_scaled = model.decoder(torch.as_tensor(z, dtype=torch.float32)).detach().numpy()
        decoded = scaler.inverse_transform(decoded_scaled)
        encoded[name] = z
        one_step = np.linalg.norm(z_pred - z_true_next, axis=1)
        reconstruction = np.linalg.norm((decoded - points) / np.maximum(UPPER, 1.0), axis=1)

        rolled = z.copy()
        for _ in range(len(points)):
            rolled = _advance(model, rolled)
        period_return = np.linalg.norm(rolled - z, axis=1)
        proper_divisors = [step for step in range(1, len(points)) if len(points) % step == 0]
        divisor_returns: dict[str, dict[str, float]] = {}
        for divisor in proper_divisors:
            divisor_rolled = z.copy()
            for _ in range(divisor):
                divisor_rolled = _advance(model, divisor_rolled)
            distances = np.linalg.norm(divisor_rolled - z, axis=1)
            divisor_returns[str(divisor)] = {
                "min": float(distances.min()),
                "mean": float(distances.mean()),
                "max": float(distances.max()),
            }
        multipliers = _latent_monodromy(model, z)
        if len(points) > 1:
            phase_separation = float(
                min(
                    np.linalg.norm(z[i] - z[j]) for i in range(len(z)) for j in range(i + 1, len(z))
                )
            )
        else:
            phase_separation = None
        summary[name] = {
            "period": len(points),
            "max_one_step_latent_l2": float(one_step.max()),
            "mean_one_step_latent_l2": float(one_step.mean()),
            "learned_period_return_l2": float(period_return.max()),
            "mean_learned_period_return_l2": float(period_return.mean()),
            "proper_divisor_returns_l2": divisor_returns,
            "minimum_proper_divisor_return_l2": (
                min(item["min"] for item in divisor_returns.values()) if divisor_returns else None
            ),
            "latent_monodromy_multipliers": [
                {
                    "real": float(value.real),
                    "imag": float(value.imag),
                    "modulus": float(abs(value)),
                }
                for value in multipliers
            ],
            "latent_unstable_multiplier_count": int(np.sum(np.abs(multipliers) > 1.0)),
            "min_encoded_phase_separation": phase_separation,
            "max_scaled_reconstruction_l2": float(reconstruction.max()),
        }
        for phase, (x, zi, pred, err, rec) in enumerate(
            zip(points, z, z_pred, one_step, reconstruction, strict=True)
        ):
            rows.append(
                {
                    "object": name,
                    "phase": phase,
                    "period": len(points),
                    "x0": float(x[0]),
                    "x1": float(x[1]),
                    "x2": float(x[2]),
                    "z0": float(zi[0]),
                    "z1": float(zi[1]),
                    "g_z0": float(pred[0]),
                    "g_z1": float(pred[1]),
                    "one_step_latent_l2": float(err),
                    "scaled_reconstruction_l2": float(rec),
                }
            )
    all_named = list(objects)
    between: dict[str, float] = {}
    for i, left in enumerate(all_named):
        for right in all_named[i + 1 :]:
            distances = np.linalg.norm(
                encoded[left][:, None, :] - encoded[right][None, :, :], axis=2
            )
            between[f"{left}__{right}"] = float(distances.min())
    summary["minimum_between_object_separation"] = between
    return summary, rows, encoded


# The scaler is fixed to Patrick's full Example-2 physical coordinates.  This
# domain normalizer is only used to report a dimensionless reconstruction norm.
UPPER = np.array([220.0, 154.0, 108.0], dtype=np.float64)


def _iterate_tensor(latent_map: torch.nn.Module, point: torch.Tensor, steps: int) -> torch.Tensor:
    value = point
    for _ in range(steps):
        value = latent_map(value.unsqueeze(0)).squeeze(0)
    return value


def _root_refined_cycle_diagnostics(
    model: torch.nn.Module,
    encoded: dict[str, NDArray[np.float64]],
    *,
    refinement_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Independently solve ``G^p(q)=q`` near every encoded physical orbit.

    The ordinary invariant audit differentiates at ``E(x_i)``. Those points
    need not be a closed learned orbit and, for a ReLU map, can occupy different
    activation cells from the nearby invariant cycle. This audit instead
    converts a copy of ``G`` to float64, root-solves the periodic-point equation,
    and differentiates ``G^p`` at the solution. It remains a numerical local
    diagnostic, not an existence proof.
    """

    latent_map = copy.deepcopy(model.latent_map).cpu().double().eval()
    extra_bases: dict[str, NDArray[np.float64]] = {}
    if refinement_summary_path is not None and refinement_summary_path.is_file():
        refinement = json.loads(refinement_summary_path.read_text())
        cycles = refinement.get("selected", {}).get("cycles", {})
        for name in OBJECT_ORDER:
            if name in cycles and "base" in cycles[name]:
                extra_bases[name] = np.asarray(cycles[name]["base"], dtype=np.float64)

    expected_unstable = {
        "P0": 0,
        "P1": 0,
        "S2": 1,
        "S4": 1,
        "p_star": 1,
        "origin": 1,
    }
    expected_unstable_sign = {"S2": -1, "S4": 1, "p_star": -1, "origin": 1}
    result: dict[str, Any] = {}

    for name in OBJECT_ORDER:
        target = encoded[name].astype(np.float64)
        period = len(target)

        def period_iterate(item: torch.Tensor, steps: int = period) -> torch.Tensor:
            return _iterate_tensor(latent_map, item, steps)

        def residual_array(point: NDArray[np.float64]) -> NDArray[np.float64]:
            value = torch.as_tensor(point, dtype=torch.float64)
            with torch.no_grad():
                residual = period_iterate(value) - value
            return residual.cpu().numpy()

        def residual_jacobian(point: NDArray[np.float64]) -> NDArray[np.float64]:
            value = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            with torch.enable_grad():
                jacobian = torch.autograd.functional.jacobian(
                    lambda item: period_iterate(item) - item,
                    value,
                    vectorize=True,
                )
            return jacobian.detach().cpu().numpy()

        seeds = list(target)
        if name in extra_bases:
            seeds.append(extra_bases[name])
        candidates: list[dict[str, Any]] = []
        for seed in seeds:
            solution = root(
                residual_array,
                seed,
                jac=residual_jacobian,
                method="hybr",
                options={"xtol": 1e-11, "maxfev": 500},
            )
            base = np.asarray(solution.x, dtype=np.float64)
            residual_l2 = float(np.linalg.norm(residual_array(base)))
            if not np.all(np.isfinite(base)) or residual_l2 > 1e-8:
                continue

            orbit = [base]
            current = torch.as_tensor(base, dtype=torch.float64)
            with torch.no_grad():
                for _ in range(1, period):
                    current = _iterate_tensor(latent_map, current, 1)
                    orbit.append(current.cpu().numpy())
            orbit_array = np.asarray(orbit, dtype=np.float64)
            cyclic_rms = []
            cyclic_max = []
            for shift in range(period):
                aligned = np.roll(target, -shift, axis=0)
                distances = np.linalg.norm(orbit_array - aligned, axis=1)
                cyclic_rms.append(float(np.sqrt(np.mean(distances**2))))
                cyclic_max.append(float(np.max(distances)))
            best_shift = int(np.argmin(cyclic_rms))
            candidates.append(
                {
                    "solver_success": bool(solution.success),
                    "solver_message": str(solution.message),
                    "base": base,
                    "orbit": orbit_array,
                    "residual_l2": residual_l2,
                    "cyclic_shift": best_shift,
                    "cyclic_rms_distance_to_encoded_phases": cyclic_rms[best_shift],
                    "cyclic_max_distance_to_encoded_phases": cyclic_max[best_shift],
                }
            )

        if not candidates:
            result[name] = {
                "period": period,
                "root_found": False,
                "expected_role_matches": False,
                "attempted_seed_count": len(seeds),
            }
            continue

        candidate = min(
            candidates,
            key=lambda item: (
                item["cyclic_rms_distance_to_encoded_phases"],
                item["residual_l2"],
            ),
        )
        base = candidate["base"]
        value = torch.tensor(base, dtype=torch.float64, requires_grad=True)
        with torch.enable_grad():
            monodromy = torch.autograd.functional.jacobian(
                period_iterate,
                value,
                vectorize=True,
            )
        multipliers = np.linalg.eigvals(monodromy.detach().cpu().numpy()).astype(np.complex128)
        divisor_returns: dict[str, float] = {}
        for divisor in range(1, period):
            if period % divisor != 0:
                continue
            with torch.no_grad():
                divisor_value = _iterate_tensor(
                    latent_map, torch.as_tensor(base, dtype=torch.float64), divisor
                )
            divisor_returns[str(divisor)] = float(
                np.linalg.norm(divisor_value.cpu().numpy() - base)
            )

        if period > 1:
            phase_scale = min(
                np.linalg.norm(target[i] - target[j])
                for i in range(period)
                for j in range(i + 1, period)
            )
        else:
            phase_scale = 0.05
        association_limit = max(1e-4, 0.5 * phase_scale)
        divisor_limit = max(1e-5, 0.05 * phase_scale)
        proper_period = all(value > divisor_limit for value in divisor_returns.values())
        unstable = [value for value in multipliers if abs(value) > 1.0 + 1e-6]
        neutral = [value for value in multipliers if abs(abs(value) - 1.0) <= 1e-6]
        orientation_matches = True
        if name in expected_unstable_sign:
            orientation_matches = (
                len(unstable) == 1
                and abs(unstable[0].imag) <= 1e-6
                and int(np.sign(unstable[0].real)) == expected_unstable_sign[name]
            )
        associated = candidate["cyclic_max_distance_to_encoded_phases"] <= association_limit
        role_matches = (
            proper_period
            and associated
            and not neutral
            and len(unstable) == expected_unstable[name]
            and orientation_matches
        )
        result[name] = {
            "period": period,
            "root_found": True,
            "solver_reported_success": candidate["solver_success"],
            "solver_message": candidate["solver_message"],
            "attempted_seed_count": len(seeds),
            "accepted_candidate_count": len(candidates),
            "base": candidate["base"].tolist(),
            "orbit": candidate["orbit"].tolist(),
            "root_residual_l2": candidate["residual_l2"],
            "cyclic_shift": candidate["cyclic_shift"],
            "cyclic_rms_distance_to_encoded_phases": candidate[
                "cyclic_rms_distance_to_encoded_phases"
            ],
            "cyclic_max_distance_to_encoded_phases": candidate[
                "cyclic_max_distance_to_encoded_phases"
            ],
            "association_limit": association_limit,
            "proper_divisor_returns_l2": divisor_returns,
            "proper_divisor_limit": divisor_limit,
            "proper_period": proper_period,
            "monodromy_multipliers": [
                {
                    "real": float(value.real),
                    "imag": float(value.imag),
                    "modulus": float(abs(value)),
                }
                for value in multipliers
            ],
            "unstable_multiplier_count": len(unstable),
            "neutral_multiplier_count": len(neutral),
            "expected_unstable_multiplier_count": expected_unstable[name],
            "unstable_orientation_matches": orientation_matches,
            "expected_role_matches": role_matches,
        }

    result["all_expected_roles_match"] = all(
        result[name].get("expected_role_matches", False) for name in OBJECT_ORDER
    )
    result["status"] = "independent_numerical_root_audit_not_an_existence_proof"
    return result


def _box_distance(point: NDArray[np.float64], boxes: NDArray[np.float64]) -> float:
    delta = np.maximum(np.maximum(boxes[:, :2] - point, point - boxes[:, 2:4]), 0.0)
    return float(np.linalg.norm(delta, axis=1).min())


def _add_membership(
    rows: list[dict[str, Any]],
    encoded: dict[str, NDArray[np.float64]],
    morse_sets_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = np.loadtxt(morse_sets_path, delimiter=",", ndmin=2)
    if data.shape[1] != 5:
        raise ValueError(f"expected 2-D Morse-set boxes, got {data.shape}")
    labels = data[:, 4].astype(np.int64)
    by_label = {int(label): data[labels == label, :4] for label in np.unique(labels)}
    row_lookup = {(row["object"], row["phase"]): row for row in rows}
    object_summary: dict[str, Any] = {}
    for name, points in encoded.items():
        phase_nodes: list[list[int]] = []
        phase_nearest: list[int] = []
        for phase, point in enumerate(points):
            distances = {label: _box_distance(point, boxes) for label, boxes in by_label.items()}
            containing = sorted(label for label, distance in distances.items() if distance <= 1e-14)
            nearest = min(distances, key=distances.get)
            phase_nodes.append(containing)
            phase_nearest.append(nearest)
            row = row_lookup[(name, phase)]
            row.update(
                {
                    "containing_morse_nodes": ";".join(map(str, containing)),
                    "nearest_morse_node": nearest,
                    "nearest_cover_distance": distances[nearest],
                }
            )
        unique_exact = {nodes[0] for nodes in phase_nodes if len(nodes) == 1}
        all_unique = all(len(nodes) == 1 for nodes in phase_nodes)
        assigned = next(iter(unique_exact)) if all_unique and len(unique_exact) == 1 else None
        object_summary[name] = {
            "assigned_morse_node": assigned,
            "all_phases_in_one_unique_morse_set": assigned is not None,
            "phase_containing_nodes": phase_nodes,
            "phase_nearest_nodes": phase_nearest,
        }
    return rows, object_summary


def _parse_index(label: str) -> list[str] | None:
    match = re.search(r":\s*\(([^)]*)\)", label)
    return [part.strip() for part in match.group(1).split(",")] if match else None


def _graph_comparison(
    graph_path: Path,
    assignments: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    graph = MorseGraph.from_dot(graph_path)
    indices = {node: _parse_index(graph.labels.get(node, "")) for node in graph.nodes}
    object_checks: dict[str, Any] = {}
    for name in OBJECT_ORDER:
        node = assignments[name]["assigned_morse_node"]
        expected_direct = manifest["expected_direct_indices"][name]
        expected_latent = expected_direct[:3]
        observed = indices.get(node) if node is not None else None
        object_checks[name] = {
            "assigned_node": node,
            "expected_latent_index": expected_latent,
            "observed_index": observed,
            "index_matches": observed == expected_latent,
            "is_minimal": node in graph.minimal if node is not None else None,
        }
    edge_checks: list[dict[str, Any]] = []
    for source, target in manifest["orbit_manifold_informed_reduced_edges"]:
        src = assignments[source]["assigned_morse_node"]
        dst = assignments[target]["assigned_morse_node"]
        # ``descendants`` includes the source itself.  A role collapse must not
        # masquerade as the requested heteroclinic ordering relation.
        reached = (
            src is not None and dst is not None and src != dst and dst in graph.descendants[src]
        )
        edge_checks.append(
            {
                "source_object": source,
                "target_object": target,
                "source_node": src,
                "target_node": dst,
                "reachable": reached,
            }
        )
    assigned_nodes = [
        value["assigned_morse_node"]
        for value in assignments.values()
        if value["assigned_morse_node"] is not None
    ]
    all_assigned = len(assigned_nodes) == len(OBJECT_ORDER)
    all_distinct = len(set(assigned_nodes)) == len(OBJECT_ORDER)

    expected_reach = {name: {name} for name in OBJECT_ORDER}
    for source, target in manifest["orbit_manifold_informed_reduced_edges"]:
        expected_reach[source].add(target)
    changed = True
    while changed:
        changed = False
        for source in OBJECT_ORDER:
            expanded = set(expected_reach[source])
            for target in tuple(expected_reach[source]):
                expanded.update(expected_reach[target])
            if expanded != expected_reach[source]:
                expected_reach[source] = expanded
                changed = True

    reachability_matrix: list[dict[str, Any]] = []
    for source in OBJECT_ORDER:
        for target in OBJECT_ORDER:
            if source == target:
                continue
            src = assignments[source]["assigned_morse_node"]
            dst = assignments[target]["assigned_morse_node"]
            observed = (
                src is not None and dst is not None and src != dst and dst in graph.descendants[src]
            )
            expected = target in expected_reach[source]
            reachability_matrix.append(
                {
                    "source_object": source,
                    "target_object": target,
                    "expected": expected,
                    "observed": observed,
                    "matches": observed == expected,
                }
            )

    expected_minimal = {"P0", "P1"}
    for name, check in object_checks.items():
        check["expected_minimal"] = name in expected_minimal
        check["minimality_matches"] = check["is_minimal"] == check["expected_minimal"]

    node_count_matches = len(graph.nodes) == len(OBJECT_ORDER)
    all_indices_match = all(item["index_matches"] for item in object_checks.values())
    all_minimality_matches = all(item["minimality_matches"] for item in object_checks.values())
    all_reachability_matches = all(item["matches"] for item in reachability_matrix)
    return {
        "nodes": graph.nodes,
        "edges": [
            [source, target] for source, targets in graph.edges.items() for target in targets
        ],
        "minimal_nodes": sorted(graph.minimal),
        "node_indices": {str(node): index for node, index in indices.items()},
        "object_checks": object_checks,
        "orbit_manifold_reachability_checks": edge_checks,
        "role_aligned_reachability_matrix": reachability_matrix,
        "all_objects_uniquely_assigned": all_assigned,
        "all_objects_in_distinct_nodes": all_distinct,
        "node_count_matches_six_roles": node_count_matches,
        "all_object_indices_match": all_indices_match,
        "all_object_minimality_matches": all_minimality_matches,
        "all_expected_relations_reachable": all(item["reachable"] for item in edge_checks),
        "all_role_reachability_and_nonreachability_match": all_reachability_matches,
        "exact_role_aligned_morse_graph_match": (
            all_assigned
            and all_distinct
            and node_count_matches
            and all_indices_match
            and all_minimality_matches
            and all_reachability_matches
        ),
    }


def _plot_overlay(
    morse_sets_path: Path,
    encoded: dict[str, NDArray[np.float64]],
    output_base: Path,
) -> list[Path]:
    plot = plot_morse_sets_from_csv(morse_sets_path, box_scale="auto", paper_style=True)
    ax = plot.ax
    for name in OBJECT_ORDER:
        points = encoded[name]
        style = OBJECT_STYLE[name]
        if len(points) > 1:
            closed = np.vstack([points, points[0]])
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                color=style["color"],
                linewidth=1.25,
                alpha=0.85,
                zorder=20,
            )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=52 if name != "p_star" else 100,
            marker=style["marker"],
            facecolor="white" if name != "origin" else style["color"],
            edgecolor=style["color"],
            linewidth=1.4,
            zorder=21,
        )
    handles = [
        Line2D(
            [0],
            [0],
            color=OBJECT_STYLE[name]["color"],
            marker=OBJECT_STYLE[name]["marker"],
            markerfacecolor="white" if name != "origin" else OBJECT_STYLE[name]["color"],
            markeredgewidth=1.2,
            linewidth=1.0 if len(encoded[name]) > 1 else 0.0,
            label=OBJECT_STYLE[name]["label"],
        )
        for name in OBJECT_ORDER
    ]
    # Keep the recurrent-set legend outside the phase portrait.  Several of
    # the fine-grid Morse components are small and sit exactly where a
    # data-dependent ``loc="best"`` legend used to obscure them.
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=1,
    )
    all_points = np.vstack(list(encoded.values()))
    boxes = plot.data[:, :4]
    lower = np.minimum(boxes[:, :2].min(axis=0), all_points.min(axis=0))
    upper = np.maximum(boxes[:, 2:4].max(axis=0), all_points.max(axis=0))
    padding = 0.04 * np.maximum(upper - lower, 1e-6)
    ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    ax.set_aspect("equal", adjustable="box")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in (".png", ".pdf"):
        path = output_base.with_suffix(suffix)
        plot.fig.savefig(path, bbox_inches="tight", dpi=300)
        written.append(path)
    plt.close(plot.fig)
    return written


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def analyze(
    config_name: str,
    *,
    morse_dir: Path | None = None,
    checkpoint_basename: str = DEFAULT_CHECKPOINT_BASENAME,
) -> dict[str, Any]:
    cfg = load_config(config_name)
    artifact_suffix = _checkpoint_artifact_suffix(checkpoint_basename)
    output_dir = _resolve(cfg.paths.output_dir)
    if len(cfg.seeds) == 1:
        output_dir = output_dir / f"seed_{cfg.seeds[0]}"
    elif len(cfg.seeds) > 1:
        raise ValueError("analyzer requires a config with at most one seed")
    data_dir = _resolve(cfg.paths.data_dir)
    resolved_morse_dir = output_dir / "MG" if morse_dir is None else _resolve(Path(morse_dir))
    analysis_dir = (
        output_dir / "analysis" if morse_dir is None else resolved_morse_dir.parent / "analysis"
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_basename == DEFAULT_CHECKPOINT_BASENAME:
        smooth_checkpoint_provenance = _validate_smooth_promotion(output_dir)
    elif checkpoint_basename == "smooth_candidate":
        smooth_checkpoint_provenance = _validate_smooth_candidate(output_dir)
    else:
        smooth_checkpoint_provenance = None
    manifest = json.loads((data_dir / "dataset_manifest.json").read_text())
    val_metadata = json.loads((data_dir / "val_metadata.json").read_text())
    scaler = joblib.load(_resolve(cfg.paths.scaler_path("train")))
    model, _ = load_any_checkpoint(
        output_dir / "models",
        arch=cfg.arch,
        basename=checkpoint_basename,
    )
    # Patrick's preserved checkpoint is legacy and therefore has no
    # architecture sidecar. Always reconstruct it with its original ReLU
    # architecture, even when the trained candidate uses a smooth latent map.
    baseline_arch = load_config(DEFAULT_CONFIG).arch
    baseline, _ = load_any_checkpoint(PATRICK_BASELINE, arch=baseline_arch)
    x_val, y_val = _load_pairs(data_dir / "val.csv")

    trained_losses = _evaluate_pairs(model, scaler, x_val, y_val)
    baseline_losses = _evaluate_pairs(baseline, scaler, x_val, y_val)
    losses_summary = {
        "trained": {name: _summarize(values) for name, values in trained_losses.items()},
        "patrick_baseline": {name: _summarize(values) for name, values in baseline_losses.items()},
        "trained_by_component": _component_summaries(trained_losses, val_metadata),
        "patrick_baseline_by_component": _component_summaries(baseline_losses, val_metadata),
    }
    losses_summary["mean_relative_change_from_patrick"] = {
        name: (
            losses_summary["trained"][name]["mean"]
            / losses_summary["patrick_baseline"][name]["mean"]
            - 1.0
        )
        for name in trained_losses
    }

    objects = _object_arrays(manifest)
    invariant_summary, invariant_rows, encoded = _learned_cycle_diagnostics(model, scaler, objects)
    root_refined_cycles = _root_refined_cycle_diagnostics(
        model,
        encoded,
        refinement_summary_path=output_dir / "topology_refinement_summary.json",
    )
    baseline_invariant_summary, _baseline_rows, _baseline_encoded = _learned_cycle_diagnostics(
        baseline, scaler, objects
    )
    result: dict[str, Any] = {
        "experiment": cfg.experiment_name,
        "status": "numerical_experiment_not_a_topological_certificate",
        "data_manifest": str(data_dir / "dataset_manifest.json"),
        "morse_directory": str(resolved_morse_dir),
        "checkpoint": {
            "basename": checkpoint_basename,
            "path": str(output_dir / "models" / f"{checkpoint_basename}.pt"),
            "architecture_sidecar": str(
                output_dir / "models" / f"{checkpoint_basename}.json"
            ),
        },
        "configured_cmgdb_bounds": {
            "lower": cfg.cmgdb.lower_bounds,
            "upper": cfg.cmgdb.upper_bounds,
            "source": f"config:{config_name}",
        },
        "smooth_training_provenance": smooth_checkpoint_provenance,
        "validation_losses": losses_summary,
        "invariant_objects": invariant_summary,
        "root_refined_invariant_objects": root_refined_cycles,
        "patrick_baseline_invariant_objects": baseline_invariant_summary,
    }

    graph_path = resolved_morse_dir / "morse_graph"
    morse_sets_path = resolved_morse_dir / "morse_sets"
    if graph_path.is_file() and morse_sets_path.is_file():
        invariant_rows, assignments = _add_membership(invariant_rows, encoded, morse_sets_path)
        result["morse_membership"] = assignments
        result["morse_graph_comparison"] = _graph_comparison(graph_path, assignments, manifest)
        result["overlay_paths"] = [
            str(path)
            for path in _plot_overlay(
                morse_sets_path,
                encoded,
                resolved_morse_dir / f"encoded_invariants_on_morse_sets{artifact_suffix}",
            )
        ]
    else:
        result["morse_status"] = "not_yet_computed"

    _write_csv(
        analysis_dir / f"encoded_invariant_points{artifact_suffix}.csv",
        invariant_rows,
    )
    (analysis_dir / f"invariant_aware_summary{artifact_suffix}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint-basename",
        default=DEFAULT_CHECKPOINT_BASENAME,
        help=(
            "checkpoint pair basename; use autoencoder for an accepted promotion "
            "or smooth_candidate to audit a rejected/unpromoted run"
        ),
    )
    parser.add_argument(
        "--morse-dir",
        type=Path,
        default=None,
        help="optional MG directory from an alternate-resolution replay",
    )
    args = parser.parse_args()
    result = analyze(
        args.config,
        morse_dir=args.morse_dir,
        checkpoint_basename=args.checkpoint_basename,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
