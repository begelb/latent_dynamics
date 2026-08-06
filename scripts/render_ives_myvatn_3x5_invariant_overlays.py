#!/usr/bin/env python3
"""Render every Ives 3x5 latent Morse set with refined invariant overlays.

The period-12 points come from the refined direct-system orbit.  Each phase is
encoded independently with the corresponding learned encoder; the connecting
line therefore shows the encoded physical orbit, not an orbit obtained by
iterating the learned latent map.  The fixed point is included as a reference.

All saved Ives Morse boxes lie on a collision-free 2048 x 2048 terminal grid.
Rendering that grid as a single indexed raster is equivalent to drawing every
saved box and avoids constructing millions of individual matplotlib patches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.colors import to_rgba
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from latentdynamics.replay import load_experiment
from latentdynamics.viz.style import PALETTE, apply_paper_style, style_latent_axes

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_ROOT = CODE_ROOT / "output" / "ives_myvatn_seedsweep_3x5_v1"
DEFAULT_REFINED_POINTS = (
    CODE_ROOT
    / "output"
    / "ives_myvatn_3d_ground_truth"
    / "invariant_stability"
    / "refined_invariant_points.csv"
)
DEFAULT_STABILITY = DEFAULT_REFINED_POINTS.with_name("stability.json")
DEFAULT_OUTPUT_DIR = DEFAULT_SWEEP_ROOT / "summary" / "figures" / "invariant_overlays"
DEFAULT_CELLS_JSON = DEFAULT_SWEEP_ROOT / "summary" / "cells.json"

DATA_SEEDS = (2158, 4792, 3174, 688, 5727)
MODEL_SEEDS = (0, 1, 2)
EXPECTED_GRID_SHAPE = (2048, 2048)
INK = "#172B3A"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_refined_points(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cycle: list[tuple[int, list[float]]] = []
    fixed: list[list[float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "vertex",
            "component_id",
            "barycenter_x",
            "barycenter_y",
            "barycenter_z",
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"{path} does not have the required invariant columns")
        for row in reader:
            vertex = int(row["vertex"])
            component = int(row["component_id"])
            point = [
                float(row["barycenter_x"]),
                float(row["barycenter_y"]),
                float(row["barycenter_z"]),
            ]
            if not all(math.isfinite(value) for value in point):
                raise ValueError(f"non-finite invariant point in {path}")
            if vertex == 0:
                cycle.append((component, point))
            elif vertex == 1:
                fixed.append(point)
            else:
                raise ValueError(f"unexpected invariant vertex {vertex}")

    cycle.sort(key=lambda item: item[0])
    if [phase for phase, _point in cycle] != list(range(12)):
        raise ValueError("refined invariant file must contain cycle phases 0 through 11")
    if len(fixed) != 1:
        raise ValueError("refined invariant file must contain exactly one fixed point")
    return (
        np.asarray([point for _phase, point in cycle], dtype=np.float64),
        np.asarray(fixed, dtype=np.float64),
    )


def _load_morse_boxes(path: Path) -> NDArray[np.float64]:
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.ndim != 2 or data.shape[1] != 5 or data.shape[0] < 1:
        raise ValueError(f"expected non-empty 2D Morse CSV with five columns: {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite Morse data in {path}")
    labels = np.rint(data[:, 4])
    if not np.array_equal(labels, data[:, 4]) or np.any(labels < 0):
        raise ValueError(f"non-integral or negative Morse label in {path}")
    if np.any(data[:, :2] >= data[:, 2:4]):
        raise ValueError(f"non-positive Morse box width in {path}")
    return data


def _rasterize_uniform_cells(
    data: NDArray[np.float64],
    bounds_lower: NDArray[np.float64],
    bounds_upper: NDArray[np.float64],
    *,
    palette: tuple[str, ...] | list[str] = PALETTE,
) -> tuple[NDArray[np.uint8], tuple[int, int]]:
    lower = np.asarray(bounds_lower, dtype=np.float64)
    upper = np.asarray(bounds_upper, dtype=np.float64)
    if lower.shape != (2,) or upper.shape != (2,) or np.any(lower >= upper):
        raise ValueError("Morse bounds must be two strictly ordered 2D points")

    widths = data[:, 2:4] - data[:, :2]
    cell_width = np.median(widths, axis=0)
    tolerance = 128.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(cell_width))
    if np.any(np.abs(widths - cell_width) > tolerance):
        raise ValueError("Morse boxes are not on one uniform terminal grid")

    shape_xy = np.rint((upper - lower) / cell_width).astype(np.int64)
    if np.any(shape_xy < 1):
        raise ValueError("invalid inferred Morse raster shape")
    reconstructed_width = (upper - lower) / shape_xy
    if np.any(np.abs(reconstructed_width - cell_width) > tolerance):
        raise ValueError("Morse cell width does not tile the saved bounds")

    indices = np.rint((data[:, :2] - lower) / reconstructed_width).astype(np.int64)
    reconstructed_lower = lower + indices * reconstructed_width
    alignment_tolerance = 256.0 * np.finfo(np.float64).eps * np.maximum(
        1.0, np.abs(data[:, :2])
    )
    if np.any(np.abs(reconstructed_lower - data[:, :2]) > alignment_tolerance):
        raise ValueError("Morse lower bounds are not aligned to the inferred grid")
    if np.any(indices < 0) or np.any(indices >= shape_xy):
        raise ValueError("Morse cell index lies outside the saved bounds")

    flat = indices[:, 1] * shape_xy[0] + indices[:, 0]
    if np.unique(flat).size != flat.size:
        raise ValueError("Morse CSV contains duplicate terminal-grid cells")

    image = np.zeros((int(shape_xy[1]), int(shape_xy[0]), 4), dtype=np.uint8)
    labels = np.rint(data[:, 4]).astype(np.int64)
    palette_rgba = np.asarray(
        [np.rint(255.0 * np.asarray(to_rgba(color))).astype(np.uint8) for color in palette]
    )
    image[indices[:, 1], indices[:, 0]] = palette_rgba[labels % len(palette_rgba)]
    return image, (int(shape_xy[0]), int(shape_xy[1]))


def _memberships(
    data: NDArray[np.float64], points: NDArray[np.float64]
) -> list[list[str]]:
    labels = np.rint(data[:, 4]).astype(np.int64)
    memberships: list[list[str]] = []
    for point in points:
        inside = np.all((data[:, :2] <= point) & (point <= data[:, 2:4]), axis=1)
        memberships.append([str(label) for label in sorted(set(labels[inside].tolist()))])
    return memberships


def _view_limits(
    data: NDArray[np.float64],
    points: NDArray[np.float64],
    bounds_lower: NDArray[np.float64],
    bounds_upper: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    occupied_lower = np.minimum(data[:, :2].min(axis=0), points.min(axis=0))
    occupied_upper = np.maximum(data[:, 2:4].max(axis=0), points.max(axis=0))
    span = np.maximum(occupied_upper - occupied_lower, 1.0e-12)
    median_width = np.median(data[:, 2:4] - data[:, :2], axis=0)
    margin = np.maximum(0.035 * span, 2.0 * median_width)
    lower = np.maximum(bounds_lower, occupied_lower - margin)
    upper = np.minimum(bounds_upper, occupied_upper + margin)
    if np.any(lower >= upper):
        raise ValueError("invalid plotted Morse-set limits")
    return lower, upper


def _draw_overlay(
    ax: plt.Axes,
    cycle: NDArray[np.float64],
    fixed: NDArray[np.float64],
    *,
    phase_labels: bool,
) -> None:
    closed = np.vstack((cycle, cycle[0]))
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color="white",
        linewidth=3.8,
        solid_joinstyle="round",
        zorder=20,
    )
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color=INK,
        linewidth=1.35,
        solid_joinstyle="round",
        zorder=21,
    )
    ax.scatter(
        cycle[:, 0],
        cycle[:, 1],
        s=62 if phase_labels else 44,
        marker="o",
        facecolor="white",
        edgecolor=INK,
        linewidth=1.15,
        zorder=23,
    )
    if phase_labels:
        for phase, point in enumerate(cycle):
            ax.text(
                float(point[0]),
                float(point[1]),
                str(phase),
                ha="center",
                va="center",
                color=INK,
                fontsize=4.8,
                fontweight="bold",
                zorder=24,
            )
    ax.scatter(
        fixed[:, 0],
        fixed[:, 1],
        s=145 if phase_labels else 105,
        marker="*",
        facecolor="white",
        edgecolor=INK,
        linewidth=1.2,
        zorder=25,
    )


def _rasterize_graph_pdf(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = destination.with_suffix("")
    subprocess.run(
        [
            "pdftoppm",
            "-singlefile",
            "-png",
            "-r",
            "300",
            str(source),
            str(prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"pdftoppm did not create {destination}")


def _render_one(
    *,
    data_seed: int,
    model_seed: int,
    cell: dict[str, Any],
    sweep_root: Path,
    output_dir: Path,
    cycle_ambient: NDArray[np.float64],
    fixed_ambient: NDArray[np.float64],
    detailed: bool,
) -> dict[str, Any]:
    dataset_root = sweep_root / f"dataset_{data_seed}"
    run_root = dataset_root / f"seed_{model_seed}"
    morse_path = run_root / "MG" / "morse_sets"
    graph_path = run_root / "MG" / "morse_graph.png"
    graph_pdf_path = run_root / "MG" / "morse_graph.pdf"
    expected_morse_hash = cell["artifacts"]["morse_sets"]["sha256"]
    expected_graph_hash = cell["artifacts"]["morse_graph_png"]["sha256"]
    if _sha256(morse_path) != expected_morse_hash:
        raise ValueError(f"Morse-set hash mismatch for {data_seed}/{model_seed}")
    if _sha256(graph_path) != expected_graph_hash:
        raise ValueError(f"Morse-graph PNG hash mismatch for {data_seed}/{model_seed}")
    if _sha256(graph_pdf_path) != cell["artifacts"]["morse_graph_pdf"]["sha256"]:
        raise ValueError(f"Morse-graph PDF hash mismatch for {data_seed}/{model_seed}")

    experiment = load_experiment(
        "ives_myvatn",
        seed=model_seed,
        device="cpu",
        output_dir=dataset_root.resolve(),
    )
    cycle = np.asarray(experiment.encode(cycle_ambient), dtype=np.float64)
    fixed = np.asarray(experiment.encode(fixed_ambient), dtype=np.float64)
    if cycle.shape != (12, 2) or fixed.shape != (1, 2):
        raise ValueError(f"unexpected encoded invariant shape for {data_seed}/{model_seed}")
    if not np.isfinite(cycle).all() or not np.isfinite(fixed).all():
        raise ValueError(f"non-finite encoded invariant for {data_seed}/{model_seed}")

    bounds_lower_raw, bounds_upper_raw = experiment.morse_bounds()
    if bounds_lower_raw is None or bounds_upper_raw is None:
        raise ValueError(f"missing Morse bounds for {data_seed}/{model_seed}")
    bounds_lower = np.asarray(bounds_lower_raw, dtype=np.float64)
    bounds_upper = np.asarray(bounds_upper_raw, dtype=np.float64)
    data = _load_morse_boxes(morse_path)
    raster, grid_shape = _rasterize_uniform_cells(data, bounds_lower, bounds_upper)

    all_points = np.vstack((cycle, fixed))
    refined_memberships = _memberships(data, all_points)
    saved_cycle = cell["reference_memberships"]["period_12_phases"]
    saved_fixed = cell["reference_memberships"]["fixed_point"]
    saved_encoded = np.asarray(
        [phase["encoded_coordinates"] for phase in saved_cycle]
        + [saved_fixed["encoded_coordinates"]],
        dtype=np.float64,
    )
    saved_memberships = [
        list(phase["morse_node_memberships"]) for phase in saved_cycle
    ] + [list(saved_fixed["morse_node_memberships"])]
    if refined_memberships != saved_memberships:
        raise ValueError(
            f"refined invariant changed saved Morse membership for {data_seed}/{model_seed}"
        )

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.1, 4.1), layout="constrained")
    ax.imshow(
        raster,
        extent=[bounds_lower[0], bounds_upper[0], bounds_lower[1], bounds_upper[1]],
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        zorder=1,
    )
    lower, upper = _view_limits(data, all_points, bounds_lower, bounds_upper)
    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    style_latent_axes(ax, two_d=True)
    _draw_overlay(ax, cycle, fixed, phase_labels=detailed)

    stem = f"dataset_{data_seed}_seed_{model_seed}_morse_sets_refined_orbit"
    if detailed:
        stem += "_detailed"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    graph_report_path = output_dir / f"dataset_{data_seed}_seed_{model_seed}_morse_graph_300dpi.png"
    fig.savefig(png_path, dpi=500, bbox_inches=None, pad_inches=0)
    fig.savefig(pdf_path, dpi=500, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    if not detailed:
        _rasterize_graph_pdf(graph_pdf_path, graph_report_path)
    elif not graph_report_path.is_file():
        raise FileNotFoundError(graph_report_path)

    return {
        "data_seed": data_seed,
        "model_seed": model_seed,
        "detailed": detailed,
        "machine_pass": bool(cell["machine_pass"]),
        "grid_shape_xy": list(grid_shape),
        "saved_morse_box_count": int(data.shape[0]),
        "maximum_archived_to_refined_encoded_change": float(
            np.max(np.abs(all_points - saved_encoded))
        ),
        "refined_memberships_equal_saved_memberships": True,
        "encoded_refined_cycle": cycle.tolist(),
        "encoded_refined_fixed_point": fixed[0].tolist(),
        "inputs": {
            "morse_sets": _file_record(morse_path),
            "morse_graph_png": _file_record(graph_path),
            "morse_graph_pdf": _file_record(graph_pdf_path),
            "checkpoint_sha256": cell["artifacts"]["checkpoint"]["sha256"],
            "scaler_sha256": cell["artifacts"]["scaler"]["sha256"],
        },
        "outputs": {
            "png": _file_record(png_path),
            "pdf": _file_record(pdf_path),
            "morse_graph_300dpi_png": _file_record(graph_report_path),
        },
    }


def render_all(
    *,
    sweep_root: Path,
    cells_json: Path,
    refined_points: Path,
    stability_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    detailed = json.loads(cells_json.read_text(encoding="utf-8"))
    cells = detailed.get("cells")
    if not isinstance(cells, list) or len(cells) != 15:
        raise ValueError("cells JSON must contain the complete 15-cell Ives design")
    cell_by_key = {
        (int(cell["data_seed"]), int(cell["model_seed"])): cell for cell in cells
    }
    expected_keys = {(data, model) for data in DATA_SEEDS for model in MODEL_SEEDS}
    if set(cell_by_key) != expected_keys:
        raise ValueError("cells JSON does not contain exactly the frozen Ives 5 x 3 grid")

    cycle_ambient, fixed_ambient = _load_refined_points(refined_points)
    stability = json.loads(stability_json.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for data_seed in DATA_SEEDS:
        for model_seed in MODEL_SEEDS:
            records.append(
                _render_one(
                    data_seed=data_seed,
                    model_seed=model_seed,
                    cell=cell_by_key[(data_seed, model_seed)],
                    sweep_root=sweep_root,
                    output_dir=output_dir,
                    cycle_ambient=cycle_ambient,
                    fixed_ambient=fixed_ambient,
                    detailed=False,
                )
            )

    winner = cell_by_key[(2158, 2)]
    records.append(
        _render_one(
            data_seed=2158,
            model_seed=2,
            cell=winner,
            sweep_root=sweep_root,
            output_dir=output_dir,
            cycle_ambient=cycle_ambient,
            fixed_ambient=fixed_ambient,
            detailed=True,
        )
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "All 15 learned latent Morse sets with independently encoded phases "
            "of the refined stable direct-system period-12 orbit"
        ),
        "interpretation": (
            "Connected markers are E(x_k) for the refined direct orbit phases; "
            "they are not iterates of the learned latent map G."
        ),
        "sources": {
            "cells_json": _file_record(cells_json),
            "refined_invariant_points": _file_record(refined_points),
            "stability_json": _file_record(stability_json),
            "period12_closure_residual": stability["refinement"][
                "period12_closure_residual"
            ],
        },
        "design": {
            "data_seeds": list(DATA_SEEDS),
            "model_seeds": list(MODEL_SEEDS),
            "gallery_run_count": 15,
            "detailed_run": {"data_seed": 2158, "model_seed": 2},
        },
        "maximum_archived_to_refined_encoded_change": max(
            record["maximum_archived_to_refined_encoded_change"] for record in records
        ),
        "all_refined_memberships_equal_saved_memberships": all(
            record["refined_memberships_equal_saved_memberships"] for record in records
        ),
        "renders": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "manifest": _file_record(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--cells-json", type=Path, default=DEFAULT_CELLS_JSON)
    parser.add_argument("--refined-points", type=Path, default=DEFAULT_REFINED_POINTS)
    parser.add_argument("--stability-json", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = render_all(
        sweep_root=args.sweep_root.resolve(),
        cells_json=args.cells_json.resolve(),
        refined_points=args.refined_points.resolve(),
        stability_json=args.stability_json.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "manifest": result["manifest"],
                "gallery_run_count": result["design"]["gallery_run_count"],
                "maximum_archived_to_refined_encoded_change": result[
                    "maximum_archived_to_refined_encoded_change"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
