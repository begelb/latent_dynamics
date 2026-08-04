#!/usr/bin/env python3
"""Build a recurrent-skeleton-aware data set for Leslie3D Example 2.

The ordinary paper data set begins from uniformly sampled initial conditions.
That makes the stable period-four orbits common late in trajectories, but it
almost never observes the origin, the positive saddle, or the saddle cycles.

This builder writes the same flat ``(x, f(x))`` CSV schema used by the normal
pipeline, but balances six complementary sources:

* every phase of every numerically catalogued recurrent object;
* multiscale local clouds around those phases;
* balanced samples from the saved direct-map Morse-set boxes;
* true forward trajectories from tangent-direction probes on both saddle sides;
* a positive-cone fan leaving the unstable origin; and
* a modest Sobol background cover of the absorbing box.

Every successor is evaluated with the analytic Leslie map.  No linearly
interpolated pseudo-transition is used.  The direct Morse sets and periodic
orbit census are numerical inputs rather than proofs of completeness, and the
manifest says so explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = CODE_ROOT / "data" / "leslie3d_invariant_aware"
DEFAULT_MORSE_SETS = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "ground_truth"
    / "absorbing_B_i29_m33_M36_L10000"
    / "screen"
    / "MG"
    / "morse_sets"
)

THETA = np.array([28.9, 29.8, 22.0], dtype=np.float64)
SURVIVAL = 0.7
LOWER = np.zeros(3, dtype=np.float64)
UPPER = np.array([110.0, 77.0, 54.0], dtype=np.float64)

# Coordinates are the high-accuracy inventory used by the direct-map
# validation script.  Each list is ordered by the true Leslie dynamics.
KNOWN_OBJECTS: dict[str, dict[str, Any]] = {
    "P0": {
        "description": "stable period-four orbit",
        "expected_direct_node": 0,
        "points": [
            [0.06476966192518428, 71.81567937129047, 3.2375662571576385],
            [1.2097281778002897, 0.04533876334762899, 50.27097555990333],
            [6.607278075831916, 0.8468097244602028, 0.031737134343340294],
            [102.59382767327212, 4.625094653082341, 0.5927668071221419],
        ],
    },
    "P1": {
        "description": "stable period-four orbit",
        "expected_direct_node": 1,
        "points": [
            [3.231447513714454, 30.156899956353254, 7.0621449189907235],
            [20.09019988641001, 2.2620132596001175, 21.109829969447276],
            [14.412540651001478, 14.063139920487005, 1.5834092817200822],
            [43.08128565193322, 10.088778455701034, 9.844197944340904],
        ],
    },
    "S2": {
        "description": "saddle period-two orbit",
        "expected_direct_node": 2,
        "points": [
            [4.995002957976051, 25.272089741366145, 2.4475514494082646],
            [36.10298534480878, 3.4965020705832353, 17.6904628189563],
        ],
    },
    "S4": {
        "description": "saddle period-four orbit",
        "expected_direct_node": 3,
        "points": [
            [0.6595601552884892, 46.017276427535535, 9.204435860581222],
            [5.960579974197539, 0.46169210870194244, 32.21209349927487],
            [18.78456298077801, 4.172405981938277, 0.3231844760913597],
            [65.73896632505077, 13.149194086544606, 2.9206841873567937],
        ],
    },
    "p_star": {
        "description": "positive saddle fixed point",
        "expected_direct_node": 4,
        "points": [[18.73654933147751, 13.115584532034255, 9.180909172423979]],
    },
    "origin": {
        "description": "unstable boundary fixed point",
        "expected_direct_node": 5,
        "points": [[0.0, 0.0, 0.0]],
    },
}

EXPECTED_REDUCED_EDGES = [
    ["S2", "P1"],
    ["S4", "P0"],
    ["S4", "P1"],
    ["p_star", "S2"],
    ["origin", "S4"],
    ["origin", "p_star"],
]

EXPECTED_DIRECT_INDICES = {
    "P0": ["x^4-1", "0", "0", "0"],
    "P1": ["x^4-1", "0", "0", "0"],
    "S2": ["0", "x^2+1", "0", "0"],
    "S4": ["0", "x^4-1", "0", "0"],
    "p_star": ["0", "x+1", "0", "0"],
    "origin": ["0", "0", "0", "0"],
}


def leslie(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized Leslie map at the Example-2 parameter tuple."""

    x = np.asarray(points, dtype=np.float64)
    total = x.sum(axis=-1)
    head = (x @ THETA) * np.exp(-0.1 * total)
    return np.stack([head, SURVIVAL * x[..., 0], SURVIVAL * x[..., 1]], axis=-1)


def jacobian(point: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.asarray(point, dtype=np.float64)
    linear_births = float(THETA @ x)
    decay = float(np.exp(-0.1 * x.sum()))
    out = np.zeros((3, 3), dtype=np.float64)
    out[0] = decay * (THETA - 0.1 * linear_births)
    out[1, 0] = SURVIVAL
    out[2, 1] = SURVIVAL
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class PairAccumulator:
    chunks: list[NDArray[np.float64]]
    components: list[dict[str, Any]]
    n_rows: int = 0

    @classmethod
    def empty(cls) -> PairAccumulator:
        return cls(chunks=[], components=[])

    def add(self, name: str, points: NDArray[np.float64], **metadata: Any) -> None:
        x = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(x) == 0:
            return
        if not np.all(np.isfinite(x)):
            raise ValueError(f"component {name!r} contains non-finite points")
        if np.any(x < LOWER - 1e-10) or np.any(x > UPPER + 1e-10):
            raise ValueError(f"component {name!r} leaves the absorbing box")
        y = leslie(x)
        if np.any(y < LOWER - 1e-10) or np.any(y > UPPER + 1e-8):
            raise ValueError(f"successors of component {name!r} leave the absorbing box")
        rows = np.hstack([x, y])
        start = self.n_rows
        self.chunks.append(rows)
        self.n_rows += len(rows)
        self.components.append(
            {
                "name": name,
                "row_start_inclusive": start,
                "row_stop_exclusive": self.n_rows,
                "rows": len(rows),
                **metadata,
            }
        )

    def array(self) -> NDArray[np.float64]:
        return np.vstack(self.chunks)


def _object_arrays() -> dict[str, NDArray[np.float64]]:
    return {
        name: np.asarray(record["points"], dtype=np.float64)
        for name, record in KNOWN_OBJECTS.items()
    }


def _phase_step_error(objects: dict[str, NDArray[np.float64]]) -> dict[str, float]:
    return {
        name: float(np.max(np.abs(leslie(points) - np.roll(points, -1, axis=0))))
        for name, points in objects.items()
    }


def _unstable_direction(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    monodromy = np.eye(3, dtype=np.float64)
    for point in points:
        monodromy = jacobian(point) @ monodromy
    values, vectors = np.linalg.eig(monodromy)
    index = int(np.argmax(np.abs(values)))
    direction = np.real_if_close(vectors[:, index]).real.astype(np.float64)
    scaled_norm = float(np.linalg.norm(direction / UPPER))
    if scaled_norm == 0.0:
        raise ValueError("zero unstable eigendirection")
    direction /= scaled_norm
    return direction, float(np.real_if_close(values[index]).real)


def _local_cloud(
    objects: dict[str, NDArray[np.float64]],
    *,
    samples_per_phase: int,
    seed: int,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    clouds: list[NDArray[np.float64]] = []
    for points in objects.values():
        for point in points:
            directions = rng.normal(size=(samples_per_phase, 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            # Log-uniform scaled-space radii resolve the index-pair neighborhood
            # from essentially exact phases out to a few percent of the box.
            radii = 10.0 ** rng.uniform(-7.0, np.log10(0.035), size=(samples_per_phase, 1))
            perturbed = point + directions * radii * UPPER
            clouds.append(np.clip(perturbed, LOWER, UPPER))
    return np.vstack(clouds)


def _sample_direct_morse_neighborhoods(
    path: Path,
    *,
    samples_per_node: int,
    seed: int,
) -> tuple[NDArray[np.float64], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] != 7:
        raise ValueError(f"expected 7-column direct Morse boxes, got {data.shape}")
    rng = np.random.default_rng(seed)
    samples: list[NDArray[np.float64]] = []
    available: dict[str, int] = {}
    for label in range(5):
        boxes = data[data[:, 6].astype(np.int64) == label, :6]
        available[str(label)] = len(boxes)
        indices = rng.choice(
            len(boxes), size=samples_per_node, replace=len(boxes) < samples_per_node
        )
        chosen = boxes[indices]
        lower, upper = chosen[:, :3], chosen[:, 3:6]
        samples.append(rng.uniform(lower, upper))
    # Node 5 is the exact origin cell; the exact-origin component supplies it
    # with much greater weight than random sampling inside a boundary cube.
    available["5"] = int(np.sum(data[:, 6].astype(np.int64) == 5))
    return np.vstack(samples), available


def _iterate_starts(starts: NDArray[np.float64], steps: int) -> NDArray[np.float64]:
    points = np.asarray(starts, dtype=np.float64)
    chunks: list[NDArray[np.float64]] = []
    for _ in range(steps):
        chunks.append(points)
        points = leslie(points)
    return np.vstack(chunks)


def _saddle_transition_tubes(
    objects: dict[str, NDArray[np.float64]],
    *,
    eps_count: int,
    steps: int,
    log_cell_offset: float,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    if not 0.0 < log_cell_offset < 1.0:
        raise ValueError("log_cell_offset must lie strictly between zero and one")
    starts: list[NDArray[np.float64]] = []
    diagnostics: dict[str, Any] = {}
    log_edges = np.linspace(np.log(2e-7), np.log(3e-2), eps_count + 1)
    log_epsilons = log_edges[:-1] + log_cell_offset * np.diff(log_edges)
    epsilons = np.exp(log_epsilons)
    for name in ("S2", "S4", "p_star"):
        points = objects[name]
        direction, multiplier = _unstable_direction(points)
        object_starts: list[NDArray[np.float64]] = []
        for sign in (-1.0, 1.0):
            perturbed = points[0] + sign * epsilons[:, None] * direction
            perturbed = np.clip(perturbed, LOWER, UPPER)
            object_starts.append(perturbed)
            starts.append(perturbed)
        diagnostics[name] = {
            "leading_multiplier": multiplier,
            "scaled_unstable_direction": direction.tolist(),
            "branches": 2,
            "epsilons": epsilons.tolist(),
            "log_cell_offset": log_cell_offset,
            "starts": int(sum(len(chunk) for chunk in object_starts)),
            "steps": steps,
        }
    all_starts = np.vstack(starts)
    return _iterate_starts(all_starts, steps), diagnostics


def _origin_fan(*, rays: int, steps: int, seed: int) -> tuple[NDArray[np.float64], dict[str, Any]]:
    # Positive directions are essential at the boundary origin: the opposite
    # branch of the Perron direction immediately leaves population space.
    unit = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(rays))))[:rays]
    unit = np.maximum(unit, 1e-12)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    radii = np.geomspace(1e-10, 2e-3, rays)[:, None]
    starts = unit * radii * UPPER
    return _iterate_starts(starts, steps), {
        "rays": rays,
        "steps": steps,
        "scaled_radius_min": float(radii.min()),
        "scaled_radius_max": float(radii.max()),
    }


def _sobol_trajectories(*, initial_conditions: int, steps: int, seed: int) -> NDArray[np.float64]:
    exponent = int(np.ceil(np.log2(initial_conditions)))
    unit = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(exponent)[:initial_conditions]
    return _iterate_starts(unit * UPPER, steps)


def _box_corners() -> NDArray[np.float64]:
    return np.asarray(
        [
            [UPPER[0] if i & 4 else 0.0, UPPER[1] if i & 2 else 0.0, UPPER[2] if i & 1 else 0.0]
            for i in range(8)
        ],
        dtype=np.float64,
    )


def build_split(
    *,
    role: str,
    morse_sets_path: Path,
    exact_repeats: int,
    local_per_phase: int,
    morse_per_node: int,
    saddle_eps: int,
    saddle_log_cell_offset: float,
    transition_steps: int,
    origin_rays: int,
    origin_steps: int,
    sobol_initial: int,
    sobol_steps: int,
    include_corners: bool,
    seed: int,
) -> tuple[PairAccumulator, dict[str, Any]]:
    objects = _object_arrays()
    acc = PairAccumulator.empty()

    exact = np.vstack(list(objects.values()))
    acc.add(
        "exact_recurrent_phases",
        np.repeat(exact, exact_repeats, axis=0),
        unique_phases=len(exact),
        repeats_per_phase=exact_repeats,
    )
    acc.add(
        "multiscale_recurrent_neighborhoods",
        _local_cloud(objects, samples_per_phase=local_per_phase, seed=seed + 1),
        samples_per_phase=local_per_phase,
        scaled_radius_range=[1e-7, 0.035],
    )
    morse_points, available = _sample_direct_morse_neighborhoods(
        morse_sets_path,
        samples_per_node=morse_per_node,
        seed=seed + 2,
    )
    acc.add(
        "balanced_direct_morse_neighborhoods",
        morse_points,
        samples_per_nonorigin_node=morse_per_node,
        source_boxes_per_node=available,
    )
    saddle_tubes, saddle_diagnostics = _saddle_transition_tubes(
        objects,
        eps_count=saddle_eps,
        steps=transition_steps,
        log_cell_offset=saddle_log_cell_offset,
    )
    acc.add(
        "saddle_tangent_transition_tubes",
        saddle_tubes,
        trajectories=3 * 2 * saddle_eps,
        steps=transition_steps,
    )
    origin_tube, origin_diagnostics = _origin_fan(
        rays=origin_rays,
        steps=origin_steps,
        seed=seed + 3,
    )
    acc.add(
        "origin_positive_cone_transition_fan",
        origin_tube,
        trajectories=origin_rays,
        steps=origin_steps,
    )
    acc.add(
        "sobol_background_trajectories",
        _sobol_trajectories(
            initial_conditions=sobol_initial,
            steps=sobol_steps,
            seed=seed + 4,
        ),
        initial_conditions=sobol_initial,
        steps=sobol_steps,
    )
    if include_corners:
        # These eight points make the scaler's physical coordinate box explicit.
        acc.add("absorbing_box_corners", _box_corners(), corners=8)

    details = {
        "role": role,
        "seed": seed,
        "saddle_transition_diagnostics": saddle_diagnostics,
        "origin_transition_diagnostics": origin_diagnostics,
        "phase_step_error_linf": _phase_step_error(objects),
    }
    return acc, details


def _csv_row_key(row: NDArray[np.float64]) -> tuple[str, ...]:
    """Return the exact key produced by the on-disk 15-digit CSV format."""

    return tuple(f"{value:.15e}" for value in row)


def _remove_validation_overlap(
    validation: PairAccumulator,
    training_rows: NDArray[np.float64],
) -> tuple[PairAccumulator, dict[str, Any]]:
    """Remove train/validation collisions and accidental validation duplicates."""

    training_keys = {_csv_row_key(row) for row in training_rows}
    seen: set[tuple[str, ...]] = set()
    filtered = PairAccumulator.empty()
    audit: dict[str, Any] = {"removed_against_train": 0, "removed_duplicates": 0, "components": {}}
    for chunk, component in zip(validation.chunks, validation.components, strict=True):
        keep: list[bool] = []
        removed_train = 0
        removed_duplicate = 0
        for row in chunk:
            key = _csv_row_key(row)
            if key in training_keys:
                keep.append(False)
                removed_train += 1
            elif key in seen:
                keep.append(False)
                removed_duplicate += 1
            else:
                keep.append(True)
                seen.add(key)
        metadata = {
            key: value
            for key, value in component.items()
            if key not in {"name", "row_start_inclusive", "row_stop_exclusive", "rows"}
        }
        kept = chunk[np.asarray(keep, dtype=bool)]
        filtered.add(component["name"], kept[:, :3], **metadata)
        audit["components"][component["name"]] = {
            "raw_rows": len(chunk),
            "kept_rows": len(kept),
            "removed_against_train": removed_train,
            "removed_duplicates": removed_duplicate,
        }
        audit["removed_against_train"] += removed_train
        audit["removed_duplicates"] += removed_duplicate
    audit["remaining_exact_overlap_rows"] = 0
    return filtered, audit


def _write_pairs(path: Path, pairs: NDArray[np.float64]) -> None:
    header = "x0,x1,x2,y0,y1,y2"
    np.savetxt(path, pairs, delimiter=",", header=header, comments="", fmt="%.15e")


def _write_invariants(path: Path) -> None:
    rows: list[list[Any]] = []
    for name, record in KNOWN_OBJECTS.items():
        points = np.asarray(record["points"], dtype=np.float64)
        successors = leslie(points)
        for phase, (point, successor) in enumerate(zip(points, successors, strict=True)):
            rows.append(
                [
                    name,
                    phase,
                    len(points),
                    record["description"],
                    record["expected_direct_node"],
                    *point,
                    *successor,
                ]
            )
    header = "object,phase,period,description,expected_direct_node,x0,x1,x2,y0,y1,y2"
    with path.open("w") as handle:
        handle.write(header + "\n")
        for row in rows:
            text = [
                str(value) if not isinstance(value, float) else f"{value:.15e}" for value in row
            ]
            handle.write(",".join(text) + "\n")


def _metadata(role: str, acc: PairAccumulator, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": "LeslieModel3D",
        "dimension": 3,
        "dataset_kind": "invariant_aware_curated_true_transitions",
        "role": role,
        "n_samples": acc.n_rows,
        "n_iterations": 1,
        "skip_initial_steps": 0,
        "lower_bounds": LOWER.tolist(),
        "upper_bounds": UPPER.tolist(),
        "model_params": {
            "th1": float(THETA[0]),
            "th2": float(THETA[1]),
            "th3": float(THETA[2]),
            "survival_p1": SURVIVAL,
            "survival_p2": SURVIVAL,
        },
        "components": acc.components,
        **details,
    }


def build(output_dir: Path, morse_sets_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, train_details = build_split(
        role="train",
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
    val, val_details = build_split(
        role="validation",
        morse_sets_path=morse_sets_path,
        # Exact recurrent phases are training constraints and are audited
        # separately; putting them here would make model selection optimistic.
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
    val, overlap_audit = _remove_validation_overlap(val, train.array())
    val_details["overlap_audit"] = overlap_audit

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    _write_pairs(train_path, train.array())
    _write_pairs(val_path, val.array())
    (output_dir / "train_metadata.json").write_text(
        json.dumps(_metadata("train", train, train_details), indent=2) + "\n"
    )
    (output_dir / "val_metadata.json").write_text(
        json.dumps(_metadata("validation", val, val_details), indent=2) + "\n"
    )
    _write_invariants(output_dir / "invariant_objects.csv")

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "builder": str(Path(__file__).resolve()),
        "purpose": (
            "Leslie3D Example2 training with exact recurrent phases, local isolating "
            "neighborhoods, and true forward transition probes"
        ),
        "scientific_status": (
            "The orbit census, direct Morse boxes, and manifold connections are strong "
            "numerical evidence, not a proof that the recurrent inventory is complete."
        ),
        "transition_policy": "Every row is (x, analytic_f(x)); no interpolated transitions.",
        "validation_policy": (
            "Validation is disjoint from training at the 15-digit CSV representation. "
            "Exact recurrent phases are evaluated separately as target-defining audits."
        ),
        "domain": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
        "parameters": {"theta": THETA.tolist(), "survival": [SURVIVAL, SURVIVAL]},
        "known_objects": KNOWN_OBJECTS,
        "expected_direct_indices": EXPECTED_DIRECT_INDICES,
        "orbit_manifold_informed_reduced_edges": EXPECTED_REDUCED_EDGES,
        "direct_morse_sets_source": {
            "path": str(morse_sets_path.resolve()),
            "sha256": sha256(morse_sets_path),
        },
        "splits": {
            "train": {
                "rows": train.n_rows,
                "components": train.components,
                "csv_sha256": sha256(train_path),
            },
            "validation": {
                "rows": val.n_rows,
                "components": val.components,
                "csv_sha256": sha256(val_path),
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
