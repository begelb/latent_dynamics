#!/usr/bin/env python3
"""Dense sampled residual and tolerance for leslie3d_example1 uniform runs.

The computation intentionally matches the manuscript protocol
(:mod:`latentdynamics.analysis.sampled_metrics`) while changing only the
recurrent Morse boxes: blocks are the minimal components of a fixed-depth
uniform graph computed by ``scripts/leslie3d_example1_uniform_grid.py``.  By
default it reproduces the exact uniform (22,22,22) calculation; ``--depth``
selects another saved fixed-depth graph under the same study root.  The
quantities are finite-sample estimates, not uniform or rigorous bounds.

Inputs: the fixed-depth results under ``<--output>/fixed<depth>`` plus the
fetched ``replay_sources/leslie3d_example1/`` artifacts.  Results are written
to ``<--output>/fixed<depth>/residual_tolerance`` (default study root:
``output/leslie3d_example1_study``).  The two-dimensional block geometry
requires the optional ``shapely`` dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import differential_evolution

from latentdynamics._paths import get_repo_root
from latentdynamics.analysis.sampled_metrics import residual_protocol
from latentdynamics.analysis.sampled_metrics import tolerance_protocol
from latentdynamics.analysis.sampled_metrics.residual_protocol import BASE_SEED
from latentdynamics.analysis.sampled_metrics.tolerance_protocol import (
    DEFAULT_LOCAL_BOXES,
    DEFAULT_SAMPLE_TARGET,
    DEFAULT_SOBOL_SCRAMBLES,
)
from latentdynamics.config import load_config
from latentdynamics.systems import build_system
from latentdynamics.training import load_any_checkpoint


REPO_ROOT = get_repo_root()
DEFAULT_OUTPUT = REPO_ROOT / "output" / "leslie3d_example1_study"
DEPTH = 22
FIXED = DEFAULT_OUTPUT / f"fixed{DEPTH}"
OUT = FIXED / "residual_tolerance"
BOX_CSV = FIXED / f"morse_sets_fixed{DEPTH}_connection_complete.csv"
FIXED_RESULT = FIXED / "result.json"
LABELS = (0, 1)


def configure_depth(depth: int, output_root: Path) -> None:
    global DEPTH, FIXED, OUT, BOX_CSV, FIXED_RESULT, LABELS
    DEPTH = depth
    FIXED = output_root / f"fixed{depth}"
    OUT = FIXED / "residual_tolerance"
    raw_boxes = FIXED / f"morse_sets_fixed{depth}_raw.csv"
    BOX_CSV = (
        raw_boxes
        if raw_boxes.is_file()
        else FIXED / f"morse_sets_fixed{depth}_connection_complete.csv"
    )
    FIXED_RESULT = FIXED / "result.json"
    fixed = json.loads(FIXED_RESULT.read_text(encoding="utf-8"))
    if int(fixed["subdivision"]["init"]) != depth:
        raise RuntimeError(f"saved result is not fixed depth {depth}")
    LABELS = tuple(int(node) for node in fixed["fixed_graph"]["minimal"])
    if not LABELS:
        raise RuntimeError(f"fixed depth {depth} has no minimal Morse nodes")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_path(path: Path) -> str:
    """Repo-relative path when possible, otherwise the path as given."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_boxes() -> dict[int, np.ndarray]:
    values = np.loadtxt(BOX_CSV, delimiter=",", dtype=np.float64, ndmin=2)
    labels = values[:, -1].astype(np.int64)
    boxes = {label: values[labels == label, :-1] for label in LABELS}
    fixed = json.loads(FIXED_RESULT.read_text(encoding="utf-8"))
    expected = {
        label: int(fixed["fixed_graph"]["components"][str(label)]["cells"])
        for label in LABELS
    }
    actual = {label: int(rows.shape[0]) for label, rows in boxes.items()}
    if actual != expected:
        raise RuntimeError(f"unexpected fixed-{DEPTH} box counts: {actual}")
    return boxes


def sampled_tolerance(
    model: torch.nn.Module,
    block,
    *,
    target_points: int,
    sobol_scrambles: int,
    local_boxes: int,
) -> dict[str, Any]:
    started = time.time()
    samples, box_ids, sampling = tolerance_protocol.candidate_points(
        block.boxes,
        target_points=target_points,
        sobol_scrambles=sobol_scrambles,
    )
    per_box_min = np.full(block.boxes.shape[0], np.inf, dtype=np.float64)
    best_value = math.inf
    best: dict[str, Any] | None = None
    all_inside = True
    chunk = 262_144
    for start in range(0, samples.shape[0], chunk):
        stop = min(start + chunk, samples.shape[0])
        mapped = tolerance_protocol.apply_module(model.latent_map, samples[start:stop])
        clearance = block.clearance(mapped)
        all_inside &= bool(np.all(clearance > 0.0))
        np.minimum.at(per_box_min, box_ids[start:stop], clearance)
        local_index = int(np.argmin(clearance))
        value = float(clearance[local_index])
        if value < best_value:
            absolute = start + local_index
            best_value = value
            best = {
                "value": value,
                "input": samples[absolute].tolist(),
                "image": mapped[local_index].tolist(),
                "box_index": int(box_ids[absolute]),
                "source": "box_corners_centers_and_sobol",
            }
    if best is None:
        raise RuntimeError("tolerance sampler evaluated no points")

    def objective(z: np.ndarray) -> float:
        image = tolerance_protocol.apply_module(
            model.latent_map, np.atleast_2d(z), chunk=1
        )
        return float(block.clearance(image)[0])

    promising = np.argsort(per_box_min)[:local_boxes]
    dimension = block.dimension
    local_runs: list[dict[str, Any]] = []
    for box_index_raw in promising:
        box_index = int(box_index_raw)
        row = block.boxes[box_index]
        lower, upper = row[:dimension], row[dimension:]
        result = differential_evolution(
            objective,
            list(zip(lower, upper)),
            seed=box_index,
            popsize=10,
            maxiter=80,
            tol=1e-9,
            polish=True,
            workers=1,
            updating="immediate",
        )
        local_runs.append(
            {
                "box_index": box_index,
                "sampled_box_minimum": float(per_box_min[box_index]),
                "optimized_value": float(result.fun),
                "evaluations": int(result.nfev),
                "success": bool(result.success),
            }
        )
        if float(result.fun) < best_value:
            image = tolerance_protocol.apply_module(
                model.latent_map,
                np.atleast_2d(result.x),
                chunk=1,
            )[0]
            best_value = float(result.fun)
            best = {
                "value": best_value,
                "input": result.x.tolist(),
                "image": image.tolist(),
                "box_index": box_index,
                "source": "local_differential_evolution",
            }

    return {
        "formula": "min_{z in T_q} dist_2(g(z), Z \\ Int(N_q))",
        "sampled_minimum": best_value,
        "witness": best,
        "sample_count": int(samples.shape[0]),
        "sampling": sampling,
        "all_sample_images_in_interior": all_inside,
        "local_optimization": {
            "boxes_searched": int(len(promising)),
            "selection": "boxes with the lowest boxwise sampled clearance",
            "runs": local_runs,
        },
        "elapsed_seconds": time.time() - started,
        "interpretation": (
            "finite sampled minimum; an upper estimate of the true infimum "
            "clearance, not a certified lower bound"
        ),
    }


def load_model_and_context():
    spec = tolerance_protocol.EXAMPLES["leslie3d_example1"]
    cfg = load_config(spec.config)
    source_root = REPO_ROOT / spec.root
    model, _ = load_any_checkpoint(source_root / "models", arch=cfg.arch)
    model.to(torch.device("cpu"))
    model.eval()
    scaler = tolerance_protocol.load_scaler(spec.scaler)
    system = build_system(cfg.system.name, cfg.system.params)
    return spec, cfg, source_root, model, scaler, system


def provenance(spec, source_root: Path) -> dict[str, Any]:
    fixed = json.loads(FIXED_RESULT.read_text(encoding="utf-8"))
    result = {
        "checkpoint_source": recorded_path(source_root),
        "model_sha256": {
            name: sha256(source_root / "models" / name)
            for name in (
                "autoencoder.pt",
                "autoencoder.json",
                "encoder.pt",
                "dynamics.pt",
                "decoder.pt",
            )
            if (source_root / "models" / name).is_file()
        },
        "uniform_morse_sets": recorded_path(BOX_CSV),
        "uniform_morse_sets_sha256": sha256(BOX_CSV),
        "uniform_result_sha256": sha256(FIXED_RESULT),
        "subdivision": fixed["subdivision"],
        "minimal_nodes": fixed["fixed_graph"]["minimal"],
        "pair_files": list(spec.pair_files),
        "protocol_scripts": {
            "tolerance": "src/latentdynamics/analysis/sampled_metrics/tolerance_protocol.py",
            "dense_residual": "src/latentdynamics/analysis/sampled_metrics/residual_protocol.py",
        },
        "metric_script": "scripts/leslie3d_example1_uniform_sampled_metrics.py",
    }
    closure_path = OUT / "forward_closure_verification.json"
    if closure_path.is_file():
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        result["forward_closure_verification"] = {
            "path": recorded_path(closure_path),
            "sha256": sha256(closure_path),
            "nodes": closure["nodes"],
            "elapsed_seconds": closure["elapsed_seconds"],
            "cmgdb_module": closure["cmgdb_module"],
        }
    return result


def compute_tolerances(
    model,
    boxes: dict[int, np.ndarray],
    *,
    target_points: int,
    sobol_scrambles: int,
    local_boxes: int,
) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    for label in LABELS:
        print(
            f"tolerance node {label}: {len(boxes[label]):,} boxes, "
            f"target {target_points:,}",
            flush=True,
        )
        block = tolerance_protocol.BlockGeometry(boxes[label])
        result = sampled_tolerance(
            model,
            block,
            target_points=target_points,
            sobol_scrambles=sobol_scrambles,
            local_boxes=local_boxes,
        )
        nodes[str(label)] = {
            "n_boxes": int(len(boxes[label])),
            "set_kind": (
                "saved recurrent Morse component from exact uniform "
                f"({DEPTH},{DEPTH},{DEPTH}) graph"
            ),
            "tolerance": result,
        }
        print(
            f"tolerance node {label}: tau_hat={result['sampled_minimum']:.12g}, "
            f"samples={result['sample_count']:,}, "
            f"all_inside={result['all_sample_images_in_interior']}",
            flush=True,
        )
    return nodes


def compute_residuals(
    spec,
    cfg,
    model,
    scaler,
    system,
    boxes: dict[int, np.ndarray],
    *,
    seed: int,
    trajectory_initials: int | None = None,
    decoder_target: int | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    blocks = {
        label: tolerance_protocol.BlockGeometry(boxes[label])
        for label in LABELS
    }
    if trajectory_initials is not None:
        residual_protocol.DISCRETE_TRAJECTORY_INITIALS["leslie3d_example1"] = (
            trajectory_initials
        )
    if decoder_target is not None:
        residual_protocol.DECODER_TARGET_DISCRETE = decoder_target
    print("residual: stored transitions and archived preimage samples", flush=True)
    stats: dict[int, dict[str, Any]] = {
        label: {
            "formula": "max_{x in S_q} ||g(E(x)) - E(f(x))||_2",
            "sampled_maximum": None,
            "squared_value_diagnostic": None,
            "accepted_samples": 0,
            "evaluated_samples": 0,
            "source_summaries": {},
            "witness": None,
        }
        for label in LABELS
    }
    for source, offset, raw_x, raw_y in tolerance_protocol.pair_batches(
        spec.pair_files,
        cfg.arch.high_dims,
    ):
        residual_protocol.update_stats(
            raw_x=raw_x,
            raw_y=raw_y,
            source=source,
            source_offset=offset,
            model=model,
            scaler=scaler,
            blocks=blocks,
            stats=stats,
        )

    # The archived labels in this file belong to the adaptive graph. For the
    # fixed-depth calculation, deliberately ignore those labels and reclassify
    # every state by membership in the selected fixed-depth block geometry.
    indexed_path = (
        REPO_ROOT
        / spec.root
        / "plot_data"
        / "preimage_plot_data_indexed.pkl"
    )
    with indexed_path.open("rb") as stream:
        indexed = pickle.load(stream)
    raw_x = np.column_stack(
        (
            np.asarray(indexed["x"]),
            np.asarray(indexed["y"]),
            np.asarray(indexed["z"]),
        )
    )
    raw_y = system.step(raw_x)
    residual_protocol.update_stats(
        raw_x=raw_x,
        raw_y=raw_y,
        source=(
            recorded_path(indexed_path)
            + f" (adaptive labels ignored; fixed{DEPTH} geometry reclassification)"
        ),
        source_offset=0,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=stats,
    )
    print("residual: fresh Sobol trajectories", flush=True)
    fresh = residual_protocol.sample_fresh_trajectories(
        example_name="leslie3d_example1",
        system=system,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=stats,
        seed_base=seed,
        chafee_initials=residual_protocol.CHAFEE_INITIALS,
    )
    print("residual: decoder-guided preimage samples", flush=True)
    decoder = residual_protocol.sample_decoder_preimages(
        example_name="leslie3d_example1",
        system=system,
        model=model,
        scaler=scaler,
        blocks=blocks,
        stats=stats,
        seed_base=seed,
    )
    protocol = {
        "stored_transitions": list(spec.pair_files),
        "archived_preimage_samples": {
            "path": recorded_path(indexed_path),
            "adaptive_labels_ignored": True,
            "classification": f"fixed{DEPTH} block geometry",
        },
        "fresh_trajectories": fresh,
        "decoder_guided_preimages": decoder,
        "seed": seed,
    }
    return stats, protocol


def write_summary(result: dict[str, Any]) -> None:
    fixed = json.loads(FIXED_RESULT.read_text(encoding="utf-8"))
    distinguished = {
        int(fixed["distinguished_objects"]["fixed_point"]["node"]),
        int(fixed["distinguished_objects"]["period_two"]["node"]),
    }
    additional = sorted(distinguished - set(LABELS))
    lines = [
        f"# Uniform ({DEPTH},{DEPTH},{DEPTH}) sampled residual and tolerance",
        "",
        f"The minimal components of the exact fixed-depth Morse graph are nodes {list(LABELS)}.",
        *(
            [
                f"Distinguished nonminimal nodes {additional} are not included in the attracting-block comparison."
            ]
            if additional
            else []
        ),
        "",
        "| Node | Boxes | Accepted residual samples | Residual candidates | R_hat | Tolerance samples | tau_hat | R_hat/tau_hat |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        node = result["nodes"][str(label)]
        residual = node["residual"]
        tolerance = node["tolerance"]
        residual_value = residual["sampled_maximum"]
        ratio = node["comparison"]["sampled_residual_over_sampled_tolerance"]
        residual_text = "n/a" if residual_value is None else f"{residual_value:.12g}"
        ratio_text = "n/a" if ratio is None else f"{ratio:.6g}"
        lines.append(
            f"| {label} | {node['n_boxes']:,} | "
            f"{residual['accepted_samples']:,} | {residual['evaluated_samples']:,} | "
            f"{residual_text} | {tolerance['sample_count']:,} | "
            f"{tolerance['sampled_minimum']:.12g} | "
            f"{ratio_text} |"
        )
    lines.extend(
        [
            "",
            "R_hat is a finite sampled maximum and therefore a lower bound on the true supremum residual.",
            "tau_hat is a finite sampled minimum and therefore an upper estimate of the true infimum clearance.",
            f"An exact replay of the fixed-{DEPTH} cell graph verifies that each minimal recurrent component equals its full forward closure.",
            "Thus R_hat >= tau_hat is a numerical witness against the strict sufficient inequality for the sampled recurrent set; the calculation does not classify an attractor as spurious.",
            "",
        ]
    )
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--depth", type=int, default=22)
    parser.add_argument(
        "--stage",
        choices=("tolerance", "residual", "all"),
        default="all",
    )
    parser.add_argument("--sample-target", type=int, default=DEFAULT_SAMPLE_TARGET)
    parser.add_argument(
        "--sobol-scrambles",
        type=int,
        default=DEFAULT_SOBOL_SCRAMBLES,
    )
    parser.add_argument("--local-boxes", type=int, default=DEFAULT_LOCAL_BOXES)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--trajectory-initials", type=int)
    parser.add_argument("--decoder-target", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "study root holding fixed<depth>/ from "
            "leslie3d_example1_uniform_grid.py "
            "(default: output/leslie3d_example1_study)"
        ),
    )
    args = parser.parse_args()
    configure_depth(args.depth, args.output)

    spec, cfg, source_root, model, scaler, system = load_model_and_context()
    boxes = load_boxes()
    OUT.mkdir(parents=True, exist_ok=True)
    tolerance_path = OUT / "tolerance_sampling.json"

    if args.stage in ("tolerance", "all"):
        tolerance_nodes = compute_tolerances(
            model,
            boxes,
            target_points=args.sample_target,
            sobol_scrambles=args.sobol_scrambles,
            local_boxes=args.local_boxes,
        )
        tolerance_output = {
            "status": "complete",
            "metric": "Euclidean distance in stored latent coordinates",
            "sample_target_per_node": args.sample_target,
            "sobol_scrambles": args.sobol_scrambles,
            "local_boxes": args.local_boxes,
            "provenance": provenance(spec, source_root),
            "nodes": tolerance_nodes,
        }
        tolerance_path.write_text(
            json.dumps(tolerance_output, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        tolerance_output = json.loads(tolerance_path.read_text(encoding="utf-8"))

    if args.stage == "tolerance":
        print(tolerance_path, flush=True)
        return

    stats, residual_protocol_record = compute_residuals(
        spec,
        cfg,
        model,
        scaler,
        system,
        boxes,
        seed=args.seed,
        trajectory_initials=args.trajectory_initials,
        decoder_target=args.decoder_target,
    )
    nodes: dict[str, Any] = {}
    closure_path = OUT / "forward_closure_verification.json"
    closure_result = (
        json.loads(closure_path.read_text(encoding="utf-8"))
        if closure_path.is_file()
        else None
    )
    for label in LABELS:
        tolerance = tolerance_output["nodes"][str(label)]["tolerance"]
        residual = stats[label]
        residual_value = residual["sampled_maximum"]
        tau = float(tolerance["sampled_minimum"])
        ratio = None if residual_value is None or tau == 0.0 else residual_value / tau
        if residual_value is None:
            conclusion = "no_accepted_residual_samples"
        elif residual_value >= tau:
            conclusion = "sampled_violation"
        else:
            conclusion = "no_sampled_violation_found"
        nodes[str(label)] = {
            "n_boxes": int(len(boxes[label])),
            "set_kind": (
                "saved recurrent Morse component; exact map-graph forward closure equals recurrent set"
                if closure_result is not None
                and closure_result["nodes"][str(label)][
                    "closure_equals_recurrent_set"
                ]
                else tolerance_output["nodes"][str(label)]["set_kind"]
            ),
            "conley_index": json.loads(FIXED_RESULT.read_text(encoding="utf-8"))[
                "fixed_graph"
            ]["components"][str(label)]["conley_index"],
            "tolerance": tolerance,
            "residual": residual,
            "comparison": {
                "sampled_residual_over_sampled_tolerance": ratio,
                "sampled_conclusion": conclusion,
            },
        }
        residual_text = "n/a" if residual_value is None else f"{residual_value:.12g}"
        ratio_text = "n/a" if ratio is None else f"{ratio:.6g}"
        print(
            f"node {label}: R_hat={residual_text}, tau_hat={tau:.12g}, "
            f"ratio={ratio_text}, accepted={residual['accepted_samples']:,}",
            flush=True,
        )

    output = {
        "status": "complete",
        "run": (
            "leslie3d_example1 author-provided checkpoint, exact uniform "
            f"({DEPTH},{DEPTH},{DEPTH}) Morse graph"
        ),
        "metric": "Euclidean distance in stored latent coordinates",
        "definitions": {
            "sampled_residual": "max_{x in S_q} ||g(E(x)) - E(f(x))||_2",
            "sampled_tolerance": "min_{z in T_q} dist_2(g(z), Z \\ Int(N_q))",
        },
        "interpretation": (
            "R_hat is a sampled lower bound on the supremum residual and tau_hat "
            "is a sampled upper estimate of the infimum clearance. R_hat >= tau_hat "
            "numerically contradicts the strict sufficient inequality for the "
            "evaluated recurrent set; it does not classify spuriousness."
        ),
        "provenance": provenance(spec, source_root),
        "sampling_protocol": {
            "tolerance": {
                "target_per_node": args.sample_target,
                "sobol_scrambles": args.sobol_scrambles,
                "local_boxes": args.local_boxes,
            },
            "residual": residual_protocol_record,
        },
        "nodes": nodes,
    }
    result_path = OUT / "sampled_residual_tolerance.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_summary(output)
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
