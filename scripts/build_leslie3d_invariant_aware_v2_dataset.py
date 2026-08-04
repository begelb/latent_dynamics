#!/usr/bin/env python3
"""Build Leslie3D invariant-aware v2 data with audited heteroclinic witnesses.

Version 2 preserves every component of the original invariant-aware builder
and adds deterministic positive-cone trajectories that explicitly traverse

``origin neighborhood -> p_star neighborhood -> S2 neighborhood -> onward``.

The training and validation witness banks use different deterministic Sobol
discoveries.  Every stored row remains an analytic pair ``(x, f(x))``; no
interpolated or hand-authored successor is used.  The default output directory
is separate from the original data so an existing run cannot be overwritten.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

if __package__:
    from scripts import build_leslie3d_invariant_aware_dataset as base
else:  # Support ``python scripts/build_leslie3d_invariant_aware_v2_dataset.py``.
    import build_leslie3d_invariant_aware_dataset as base

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = CODE_ROOT / "data" / "leslie3d_invariant_aware_v2"
DEFAULT_MORSE_SETS = base.DEFAULT_MORSE_SETS

DATASET_VERSION = 2
WITNESS_COMPONENT = "audited_origin_p_star_s2_transition_tubes"
WITNESS_STEPS = 320
WITNESS_JITTER_RELATIVE = 1e-3
WITNESSES_PER_BASE = 9

# These six centers were selected by the deterministic search documented in
# WITNESS_DISCOVERY below.  Each one starts inside the direct-map origin cell,
# enters the direct p_star Morse set, leaves it, and subsequently enters and
# leaves the direct S2 Morse set.
TRAIN_WITNESS_BASES = np.asarray(
    [
        [1.830407331139335e-05, 2.0654845899705045e-05, 9.917609527972234e-06],
        [7.431093169870681e-04, 4.065216258161918e-04, 3.059463997896831e-04],
        [5.521308537495305e-04, 3.854965771805111e-04, 5.920548828810364e-04],
    ],
    dtype=np.float64,
)
VALIDATION_WITNESS_BASES = np.asarray(
    [
        [5.473316498012153e-10, 8.961049455474447e-10, 4.542378552898324e-10],
        [5.454142765816238e-04, 4.329780877582068e-04, 5.356687593607635e-04],
        [9.992785605681323e-09, 3.422013946111370e-08, 1.123657078496704e-08],
    ],
    dtype=np.float64,
)

WITNESS_DISCOVERY: dict[str, Any] = {
    "method": "deterministic_scrambled_sobol_positive_cone_search",
    "sobol_dimension": 4,
    "sobol_scramble": True,
    "sobol_seed": 90317,
    "sobol_power": 19,
    "candidate_count": 2**19,
    "direction_rule": (
        "clip the first three Sobol coordinates below at 1e-15, then normalize "
        "to Euclidean unit length"
    ),
    "scaled_radius_rule": ("10**(-12 + u3*(log10(5e-3)+12)); x0=direction*scaled_radius*UPPER"),
    "candidate_ranking_horizon": 160,
    "postselection_itinerary_audit_horizon": WITNESS_STEPS,
    "ranking_metric": "minimum ||(f^t(x)-p_star)/UPPER||_2",
    "selection_rule": (
        "start in direct origin cell, enter direct p_star box union, leave it, "
        "then enter and leave direct S2 box union"
    ),
    "selected_sample_indices": {
        "train": [151783, 234911, 64934],
        "validation": [482705, 232929, 386002],
    },
}


def _witness_jitter_factors() -> NDArray[np.float64]:
    """Return center plus the eight coordinatewise +/-0.1% corners."""

    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)), dtype=np.float64)
    return np.vstack((np.ones((1, 3)), 1.0 + WITNESS_JITTER_RELATIVE * signs))


def reproduce_discovery_bases(
    role: Literal["train", "validation"],
) -> NDArray[np.float64]:
    """Recompute the selected centers from their Sobol indices and search rule."""

    if role not in ("train", "validation"):
        raise ValueError(f"unknown witness role {role!r}")
    starts: list[NDArray[np.float64]] = []
    for index in WITNESS_DISCOVERY["selected_sample_indices"][role]:
        sampler = qmc.Sobol(d=4, scramble=True, seed=WITNESS_DISCOVERY["sobol_seed"])
        sampler.fast_forward(int(index))
        sample = sampler.random(1)[0]
        direction = np.maximum(sample[:3], 1e-15)
        direction /= np.linalg.norm(direction)
        radius = 10.0 ** (-12.0 + sample[3] * (np.log10(5e-3) + 12.0))
        starts.append(direction * radius * base.UPPER)
    return np.asarray(starts, dtype=np.float64)


def audited_witness_starts(role: Literal["train", "validation"]) -> NDArray[np.float64]:
    """Return the 27 deterministic, strictly positive starts for one split."""

    if role == "train":
        bases = TRAIN_WITNESS_BASES
    elif role == "validation":
        bases = VALIDATION_WITNESS_BASES
    else:
        raise ValueError(f"unknown witness role {role!r}")
    factors = _witness_jitter_factors()
    starts = np.vstack([point * factors for point in bases])
    if starts.shape != (len(bases) * WITNESSES_PER_BASE, 3):
        raise AssertionError("unexpected audited witness shape")
    if np.any(starts <= 0.0):
        raise AssertionError("audited witnesses must remain in the strict positive cone")
    return starts


def trajectory_states(
    starts: NDArray[np.float64], *, steps: int = WITNESS_STEPS
) -> NDArray[np.float64]:
    """Return states at times 0 through ``steps`` for each initial condition."""

    initial = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
    states = np.empty((steps + 1, len(initial), 3), dtype=np.float64)
    states[0] = initial
    for step in range(steps):
        states[step + 1] = base.leslie(states[step])
    return states


def _trajectory_pair_rows(states: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.hstack((states[:-1].reshape(-1, 3), states[1:].reshape(-1, 3)))


def _load_transition_neighborhoods(path: Path) -> dict[str, NDArray[np.float64]]:
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] != 7:
        raise ValueError(f"expected 7-column direct Morse boxes, got {data.shape}")
    labels = data[:, 6].astype(np.int64)
    boxes = {
        "S2": data[labels == int(base.KNOWN_OBJECTS["S2"]["expected_direct_node"]), :6],
        "p_star": data[labels == int(base.KNOWN_OBJECTS["p_star"]["expected_direct_node"]), :6],
        "origin": data[labels == int(base.KNOWN_OBJECTS["origin"]["expected_direct_node"]), :6],
    }
    empty = [name for name, values in boxes.items() if len(values) == 0]
    if empty:
        raise ValueError(f"direct Morse source has no boxes for {empty}")
    return boxes


def _inside_box_union(
    points: NDArray[np.float64], boxes: NDArray[np.float64], *, chunk_size: int = 128
) -> NDArray[np.bool_]:
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    result = np.zeros(len(flat), dtype=bool)
    lower = boxes[:, :3]
    upper = boxes[:, 3:]
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        chunk = flat[start:stop, None, :]
        result[start:stop] = np.any(
            np.all((chunk >= lower[None, :, :]) & (chunk <= upper[None, :, :]), axis=2),
            axis=1,
        )
    return result.reshape(np.asarray(points).shape[:-1])


def _nearest_phase_distances(
    states: NDArray[np.float64], phases: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    differences = states[:, :, None, :] - phases[None, None, :, :]
    scaled = np.linalg.norm(differences / base.UPPER[None, None, None, :], axis=3)
    physical = np.linalg.norm(differences, axis=3)
    return scaled.min(axis=2), physical.min(axis=2)


def audit_witnesses(
    *,
    role: Literal["train", "validation"],
    starts: NDArray[np.float64],
    states: NDArray[np.float64],
    neighborhoods: dict[str, NDArray[np.float64]],
) -> dict[str, Any]:
    """Validate the directed itinerary and return manifest-ready diagnostics."""

    if states.shape != (WITNESS_STEPS + 1, len(starts), 3):
        raise ValueError(f"unexpected witness state shape {states.shape}")
    origin_membership = _inside_box_union(starts, neighborhoods["origin"])
    p_star_membership = _inside_box_union(states, neighborhoods["p_star"])
    s2_membership = _inside_box_union(states, neighborhoods["S2"])
    p_star = np.asarray(base.KNOWN_OBJECTS["p_star"]["points"], dtype=np.float64)
    s2 = np.asarray(base.KNOWN_OBJECTS["S2"]["points"], dtype=np.float64)
    p_scaled, p_physical = _nearest_phase_distances(states, p_star)
    s2_scaled, s2_physical = _nearest_phase_distances(states, s2)
    factors = _witness_jitter_factors()
    sample_indices = WITNESS_DISCOVERY["selected_sample_indices"][role]

    records: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        p_times = np.flatnonzero(p_star_membership[:, index])
        s2_times = np.flatnonzero(s2_membership[:, index])
        if not origin_membership[index]:
            raise ValueError(f"{role} witness {index} does not start in the origin cell")
        if not len(p_times) or not len(s2_times):
            raise ValueError(f"{role} witness {index} misses p_star or S2")
        if int(p_times[-1]) >= int(s2_times[0]):
            raise ValueError(f"{role} witness {index} does not leave p_star before S2")
        if int(s2_times[-1]) >= WITNESS_STEPS:
            raise ValueError(f"{role} witness {index} has not continued beyond S2")
        p_min_time = int(np.argmin(p_scaled[:, index]))
        s2_min_time = int(np.argmin(s2_scaled[:, index]))
        base_index, jitter_index = divmod(index, WITNESSES_PER_BASE)
        records.append(
            {
                "witness_id": f"{role}_b{base_index + 1}_j{jitter_index}",
                "discovery_sample_index": int(sample_indices[base_index]),
                "base_index": base_index,
                "jitter_index": jitter_index,
                "multiplicative_factor": factors[jitter_index].tolist(),
                "start": start.tolist(),
                "scaled_start_radius": float(np.linalg.norm(start / base.UPPER)),
                "origin_cell_member": True,
                "p_star": {
                    "first_entry_time": int(p_times[0]),
                    "last_member_time": int(p_times[-1]),
                    "closest_time": p_min_time,
                    "minimum_scaled_l2": float(p_scaled[p_min_time, index]),
                    "minimum_physical_l2": float(p_physical[p_min_time, index]),
                },
                "S2": {
                    "first_entry_time": int(s2_times[0]),
                    "last_member_time": int(s2_times[-1]),
                    "closest_time": s2_min_time,
                    "minimum_scaled_l2": float(s2_scaled[s2_min_time, index]),
                    "minimum_physical_l2": float(s2_physical[s2_min_time, index]),
                },
            }
        )

    return {
        "role": role,
        "trajectory_count": len(starts),
        "steps": WITNESS_STEPS,
        "pair_rows": len(starts) * WITNESS_STEPS,
        "base_starts": (
            TRAIN_WITNESS_BASES.tolist() if role == "train" else VALIDATION_WITNESS_BASES.tolist()
        ),
        "all_start_in_origin_cell": True,
        "all_enter_and_leave_p_star_before_S2": True,
        "all_enter_and_leave_S2_within_horizon": True,
        "worst_minimum_p_star_scaled_l2": max(
            record["p_star"]["minimum_scaled_l2"] for record in records
        ),
        "worst_minimum_S2_scaled_l2": max(record["S2"]["minimum_scaled_l2"] for record in records),
        "records": records,
    }


def _base_split(
    role: Literal["train", "validation"], morse_sets_path: Path
) -> tuple[base.PairAccumulator, dict[str, Any]]:
    if role == "train":
        return base.build_split(
            role=role,
            morse_sets_path=morse_sets_path,
            exact_repeats=512,
            local_per_phase=256,
            morse_per_node=1024,
            saddle_eps=16,
            saddle_log_cell_offset=0.25,
            transition_steps=64,
            origin_rays=16,
            origin_steps=64,
            sobol_initial=1024,
            sobol_steps=12,
            include_corners=True,
            seed=20260803,
        )
    return base.build_split(
        role=role,
        morse_sets_path=morse_sets_path,
        exact_repeats=0,
        local_per_phase=64,
        morse_per_node=256,
        saddle_eps=8,
        saddle_log_cell_offset=0.75,
        transition_steps=64,
        origin_rays=8,
        origin_steps=64,
        sobol_initial=512,
        sobol_steps=12,
        include_corners=False,
        seed=20260804,
    )


def _add_witness_component(
    accumulator: base.PairAccumulator,
    states: NDArray[np.float64],
    audit: dict[str, Any],
) -> None:
    accumulator.add(
        WITNESS_COMPONENT,
        states[:-1].reshape(-1, 3),
        trajectories=audit["trajectory_count"],
        steps=WITNESS_STEPS,
        route=["origin", "p_star", "S2", "onward"],
        successor_policy="analytic Leslie f(x)",
        jitter_relative=WITNESS_JITTER_RELATIVE,
        base_starts=audit["base_starts"],
        witness_ids=[record["witness_id"] for record in audit["records"]],
        audit_summary={
            "all_start_in_origin_cell": audit["all_start_in_origin_cell"],
            "all_enter_and_leave_p_star_before_S2": audit["all_enter_and_leave_p_star_before_S2"],
            "all_enter_and_leave_S2_within_horizon": audit["all_enter_and_leave_S2_within_horizon"],
            "worst_minimum_p_star_scaled_l2": audit["worst_minimum_p_star_scaled_l2"],
            "worst_minimum_S2_scaled_l2": audit["worst_minimum_S2_scaled_l2"],
        },
    )


def _row_disjointness(
    training_rows: NDArray[np.float64], validation_rows: NDArray[np.float64]
) -> dict[str, int]:
    train_keys = [base._csv_row_key(row) for row in training_rows]
    validation_keys = [base._csv_row_key(row) for row in validation_rows]
    train_set = set(train_keys)
    validation_set = set(validation_keys)
    return {
        "training_rows": len(train_keys),
        "training_unique_rows": len(train_set),
        "validation_rows": len(validation_keys),
        "validation_unique_rows": len(validation_set),
        "exact_overlap_rows_at_15_digit_csv_precision": len(train_set & validation_set),
    }


def _metadata(
    role: str, accumulator: base.PairAccumulator, details: dict[str, Any]
) -> dict[str, Any]:
    metadata = base._metadata(role, accumulator, details)
    metadata["dataset_kind"] = "invariant_aware_curated_true_transitions_v2"
    metadata["dataset_version"] = DATASET_VERSION
    return metadata


def build(output_dir: Path, morse_sets_path: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir == base.DEFAULT_OUTPUT.resolve():
        raise ValueError("v2 builder refuses to overwrite the original invariant-aware data")
    output_dir.mkdir(parents=True, exist_ok=True)

    train, train_details = _base_split("train", morse_sets_path)
    validation, validation_details = _base_split("validation", morse_sets_path)
    neighborhoods = _load_transition_neighborhoods(morse_sets_path)

    train_starts = audited_witness_starts("train")
    validation_starts = audited_witness_starts("validation")
    train_states = trajectory_states(train_starts)
    validation_states = trajectory_states(validation_starts)
    train_audit = audit_witnesses(
        role="train",
        starts=train_starts,
        states=train_states,
        neighborhoods=neighborhoods,
    )
    validation_audit = audit_witnesses(
        role="validation",
        starts=validation_starts,
        states=validation_states,
        neighborhoods=neighborhoods,
    )
    witness_disjointness = _row_disjointness(
        _trajectory_pair_rows(train_states), _trajectory_pair_rows(validation_states)
    )
    if witness_disjointness["exact_overlap_rows_at_15_digit_csv_precision"] != 0:
        raise ValueError("audited train and validation witness rows overlap")

    _add_witness_component(train, train_states, train_audit)
    _add_witness_component(validation, validation_states, validation_audit)
    validation, overlap_audit = base._remove_validation_overlap(validation, train.array())
    witness_overlap = overlap_audit["components"][WITNESS_COMPONENT]
    if witness_overlap["kept_rows"] != WITNESS_STEPS * len(validation_starts):
        raise ValueError("validation witness rows collided with the training data")
    validation_details["overlap_audit"] = overlap_audit
    final_disjointness = _row_disjointness(train.array(), validation.array())
    if final_disjointness["exact_overlap_rows_at_15_digit_csv_precision"] != 0:
        raise AssertionError("validation overlap filter did not produce disjoint splits")

    train_details["audited_transition_witness_summary"] = {
        key: value for key, value in train_audit.items() if key != "records"
    }
    validation_details["audited_transition_witness_summary"] = {
        key: value for key, value in validation_audit.items() if key != "records"
    }

    train_path = output_dir / "train.csv"
    validation_path = output_dir / "val.csv"
    base._write_pairs(train_path, train.array())
    base._write_pairs(validation_path, validation.array())
    (output_dir / "train_metadata.json").write_text(
        json.dumps(_metadata("train", train, train_details), indent=2) + "\n"
    )
    (output_dir / "val_metadata.json").write_text(
        json.dumps(_metadata("validation", validation, validation_details), indent=2) + "\n"
    )
    base._write_invariants(output_dir / "invariant_objects.csv")

    builder_path = Path(__file__).resolve()
    base_builder_path = Path(base.__file__).resolve()
    manifest = {
        "schema_version": 2,
        "dataset_version": DATASET_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "builder": str(builder_path),
        "builder_sha256": base.sha256(builder_path),
        "base_builder": {
            "path": str(base_builder_path),
            "sha256": base.sha256(base_builder_path),
            "policy": "all v1 components are rebuilt unchanged before adding v2 witnesses",
        },
        "purpose": (
            "Leslie3D Example2 invariant-aware v2 training with explicit audited "
            "origin-to-p_star-to-S2 transition trajectories"
        ),
        "scientific_status": (
            "The direct Morse neighborhoods and heteroclinic itineraries are strong "
            "reproducible numerical evidence, not a Conley-index proof."
        ),
        "transition_policy": "Every row is (x, analytic_f(x)); no interpolated transitions.",
        "validation_policy": (
            "The validation witness bank uses different Sobol discoveries from training; "
            "all final validation rows are disjoint from training at 15-digit CSV precision."
        ),
        "domain": {"lower": base.LOWER.tolist(), "upper": base.UPPER.tolist()},
        "parameters": {
            "theta": base.THETA.tolist(),
            "survival": [base.SURVIVAL, base.SURVIVAL],
        },
        "known_objects": base.KNOWN_OBJECTS,
        "expected_direct_indices": base.EXPECTED_DIRECT_INDICES,
        "orbit_manifold_informed_reduced_edges": base.EXPECTED_REDUCED_EDGES,
        "direct_morse_sets_source": {
            "path": str(morse_sets_path.resolve()),
            "sha256": base.sha256(morse_sets_path),
            "membership_rule": "point lies in the exact union of closed boxes for the node label",
            "labels_used_for_witness_audit": {
                name: int(base.KNOWN_OBJECTS[name]["expected_direct_node"])
                for name in ("origin", "p_star", "S2")
            },
        },
        "audited_origin_p_star_s2_witnesses": {
            "component_name": WITNESS_COMPONENT,
            "steps": WITNESS_STEPS,
            "jitter": {
                "relative": WITNESS_JITTER_RELATIVE,
                "rule": "center plus all eight coordinatewise +/- relative corners",
                "factors": _witness_jitter_factors().tolist(),
            },
            "discovery": WITNESS_DISCOVERY,
            "train": train_audit,
            "validation": validation_audit,
            "witness_bank_disjointness": witness_disjointness,
        },
        "final_split_disjointness": final_disjointness,
        "splits": {
            "train": {
                "rows": train.n_rows,
                "components": train.components,
                "csv_sha256": base.sha256(train_path),
            },
            "validation": {
                "rows": validation.n_rows,
                "components": validation.components,
                "csv_sha256": base.sha256(validation_path),
            },
        },
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--direct-morse-sets", type=Path, default=DEFAULT_MORSE_SETS)
    args = parser.parse_args()
    manifest = build(args.output_dir, args.direct_morse_sets)
    print(f"wrote {manifest['splits']['train']['rows']:,} training pairs")
    print(f"wrote {manifest['splits']['validation']['rows']:,} validation pairs")
    print(f"output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
