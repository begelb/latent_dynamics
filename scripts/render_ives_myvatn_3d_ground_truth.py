#!/usr/bin/env python3
"""Render the saved direct three-dimensional Ives Morse sets.

The scientific source remains the exact CMGDB ``MG/morse_sets`` file.  This
utility maps its uniform fine cells to the coarsest collision-free parent grid
at or above ``--display-level`` and renders that labeled outer display cover.
The display cover is never presented as a recomputed coarse decomposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from numpy.typing import NDArray

from latentdynamics.viz import PALETTE
from latentdynamics.viz.morse_plots import plot_morse_sets_3d_cubical_from_csv
from latentdynamics.viz.style import save_figure

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    CODE_ROOT
    / "output"
    / "ives_myvatn_3d_ground_truth"
    / "absorbing_v3_i18_m33_M39_L30000000_morse"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _axis_splits(level: int, dimension: int = 3) -> NDArray[np.int64]:
    return np.asarray(
        [(level - axis + dimension - 1) // dimension for axis in range(dimension)],
        dtype=np.int64,
    )


def _load_source(
    run_dir: Path,
) -> tuple[
    dict[str, Any],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    manifest_path = run_dir / "manifest.json"
    membership_path = run_dir / "reference_membership.json"
    source_path = run_dir / "MG" / "morse_sets"
    for path in (manifest_path, membership_path, source_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("artifacts", {}).get("morse_sets", {})
    if recorded.get("sha256") != _sha256(source_path):
        raise ValueError("Morse-set hash does not match the run manifest")
    source_level = int(manifest["morse_sets"]["source_level"])
    source_splits = _axis_splits(source_level)
    recorded_splits = np.asarray(
        manifest["morse_sets"]["source_axis_splits"], dtype=np.int64
    )
    if not np.array_equal(source_splits, recorded_splits):
        raise ValueError("manifest source-axis splits are inconsistent")

    bounds = manifest["system"]["bounds"]
    domain_lower = np.asarray(bounds["lower"], dtype=np.float64)
    domain_upper = np.asarray(bounds["upper"], dtype=np.float64)
    if domain_lower.shape != (3,) or domain_upper.shape != (3,):
        raise ValueError("run manifest must contain three-dimensional bounds")

    data = np.loadtxt(source_path, delimiter=",", ndmin=2, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 7 or data.shape[0] == 0:
        raise ValueError(f"expected nonempty seven-column Morse data; got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("Morse data contain non-finite values")
    raw_labels = data[:, 6]
    labels = np.rint(raw_labels).astype(np.int64)
    if not np.array_equal(raw_labels, labels):
        raise ValueError("Morse labels must be integral")
    unique_labels = np.unique(labels)
    if np.any(unique_labels < 0) or int(unique_labels.max()) >= len(PALETTE):
        raise ValueError("Morse-node IDs exceed the repository palette")

    source_counts = np.left_shift(1, source_splits)
    widths = (domain_upper - domain_lower) / source_counts
    if not np.allclose(data[:, 3:6] - data[:, :3], widths, rtol=0.0, atol=1e-11):
        raise ValueError(f"not every source box is a uniform level-{source_level} cell")
    indices_float = (data[:, :3] - domain_lower) / widths
    indices = np.rint(indices_float).astype(np.int64)
    if not np.allclose(indices_float, indices, rtol=0.0, atol=1e-9):
        raise ValueError("source Morse boxes do not align with the manifest grid")
    if np.any(indices < 0) or np.any(indices >= source_counts):
        raise ValueError("source Morse box lies outside the manifest grid")

    expected_counts = {
        str(label): int(np.count_nonzero(labels == label)) for label in unique_labels
    }
    if expected_counts != manifest["morse_sets"]["boxes_per_node"]:
        raise ValueError("source label counts do not match the run manifest")

    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    manifest["_membership"] = membership
    manifest["_source_path"] = str(source_path)
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_membership_path"] = str(membership_path)
    return manifest, labels, indices, domain_lower, domain_upper


def _display_cover(
    labels: NDArray[np.int64],
    source_indices: NDArray[np.int64],
    domain_lower: NDArray[np.float64],
    domain_upper: NDArray[np.float64],
    *,
    source_level: int,
    display_level: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    source_splits = _axis_splits(source_level)
    display_splits = _axis_splits(display_level)
    delta = source_splits - display_splits
    if display_level < 0 or np.any(delta < 0):
        raise ValueError("display level must not exceed the source level")
    factors = np.left_shift(1, delta)
    display_counts = np.left_shift(1, display_splits)
    display_indices = source_indices // factors
    volume = int(np.prod(display_counts, dtype=np.int64))
    geometry = (
        (display_indices[:, 0] * display_counts[1] + display_indices[:, 1])
        * display_counts[2]
        + display_indices[:, 2]
    )
    unique_keys = np.unique(labels * volume + geometry)
    cover_labels = (unique_keys // volume).astype(np.int64)
    cover_geometry = unique_keys % volume
    geometry_count = int(np.unique(cover_geometry).size)
    collision_count = int(cover_geometry.size - geometry_count)

    ix = cover_geometry // (display_counts[1] * display_counts[2])
    remainder = cover_geometry % (display_counts[1] * display_counts[2])
    iy = remainder // display_counts[2]
    iz = remainder % display_counts[2]
    cover_indices = np.column_stack((ix, iy, iz)).astype(np.int64)
    widths = (domain_upper - domain_lower) / display_counts
    lower = domain_lower + cover_indices * widths
    upper = lower + widths
    cover = np.column_stack((lower, upper, cover_labels)).astype(np.float64)

    source_unique = np.unique(labels)
    diagnostics = {
        "source_level": source_level,
        "source_axis_splits": source_splits.tolist(),
        "display_level": display_level,
        "display_axis_splits": display_splits.tolist(),
        "coarsening_factor_by_axis": factors.tolist(),
        "display_axis_counts": display_counts.tolist(),
        "display_box_widths": widths.tolist(),
        "source_box_count": int(labels.size),
        "display_cell_count": int(cover.shape[0]),
        "cross_label_display_cell_overlaps": collision_count,
        "source_boxes_per_node": {
            str(label): int(np.count_nonzero(labels == label))
            for label in source_unique
        },
        "display_cells_per_node": {
            str(label): int(np.count_nonzero(cover_labels == label))
            for label in source_unique
        },
    }
    return cover, diagnostics


def _choose_display_cover(
    labels: NDArray[np.int64],
    source_indices: NDArray[np.int64],
    domain_lower: NDArray[np.float64],
    domain_upper: NDArray[np.float64],
    *,
    source_level: int,
    minimum_display_level: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    if minimum_display_level > source_level:
        raise ValueError("minimum display level exceeds the source level")
    candidates = list(range(minimum_display_level, source_level + 1))
    for display_level in candidates:
        cover, diagnostics = _display_cover(
            labels,
            source_indices,
            domain_lower,
            domain_upper,
            source_level=source_level,
            display_level=display_level,
        )
        if diagnostics["cross_label_display_cell_overlaps"] == 0:
            diagnostics["candidate_levels_tested"] = candidates[
                : candidates.index(display_level) + 1
            ]
            return cover, diagnostics
    raise RuntimeError("no collision-free display level was found")


def _roles_and_points(
    membership: dict[str, Any],
) -> tuple[dict[int, str], NDArray[np.float64], NDArray[np.float64]]:
    rows = membership.get("rows", [])
    if len(rows) != 13:
        raise ValueError("reference membership must contain 13 invariant points")
    cycle_rows = sorted(
        (row for row in rows if int(row["vertex"]) == 0),
        key=lambda row: int(row["component_id"]),
    )
    fixed_rows = [row for row in rows if int(row["vertex"]) == 1]
    if len(cycle_rows) != 12 or len(fixed_rows) != 1:
        raise ValueError("expected a period-12 orbit and one fixed point")
    if any(len(row["morse_node_memberships"]) != 1 for row in rows):
        raise ValueError("every invariant point must have one raw Morse-node membership")

    cycle_nodes = {int(row["morse_node_memberships"][0]) for row in cycle_rows}
    fixed_node = int(fixed_rows[0]["morse_node_memberships"][0])
    if len(cycle_nodes) != 1:
        raise ValueError("the 12 cycle phases do not occupy one Morse component")
    cycle_node = next(iter(cycle_nodes))
    if cycle_node == fixed_node:
        raise ValueError("cycle and fixed point occupy the same Morse component")
    roles = {
        cycle_node: "period-12 component",
        fixed_node: "fixed-point attractor",
    }
    cycle = np.asarray([row["point"] for row in cycle_rows], dtype=np.float64)
    fixed = np.asarray(fixed_rows[0]["point"], dtype=np.float64)
    return roles, cycle, fixed


def _assert_points_in_role_cells(
    cover: NDArray[np.float64],
    roles: dict[int, str],
    cycle: NDArray[np.float64],
    fixed: NDArray[np.float64],
) -> None:
    inverse = {role: node for node, role in roles.items()}
    checks = (
        (cycle, inverse["period-12 component"]),
        (fixed.reshape(1, 3), inverse["fixed-point attractor"]),
    )
    tolerance = 1e-11
    for points, node in checks:
        boxes = cover[cover[:, 6].astype(np.int64) == node]
        for point in points:
            inside = np.all(point >= boxes[:, :3] - tolerance, axis=1) & np.all(
                point <= boxes[:, 3:6] + tolerance, axis=1
            )
            if not np.any(inside):
                raise ValueError(f"invariant point is absent from display role node {node}")


def _render_view(
    cover_path: Path,
    output_dir: Path,
    *,
    basename: str,
    source_level: int,
    roles: dict[int, str],
    cycle: NDArray[np.float64],
    fixed: NDArray[np.float64],
    elev: float,
    azim: float,
) -> list[Path]:
    plot = plot_morse_sets_3d_cubical_from_csv(
        cover_path,
        palette=PALETTE,
        paper_style=True,
        elev=elev,
        azim=azim,
        alpha=1.0,
        shade=True,
        shade_strength=0.28,
        highlight_strength=0.10,
        edge_alpha=0.14,
        edge_linewidth=0.055,
        minimal_frame=True,
        show_ticks=True,
        show_axis_labels=False,
        show_legend=False,
    )
    # Keep invariant markers above the rasterized cubical surface.  Axes3D's
    # default depth-based artist reordering can otherwise partially hide them.
    plot.ax.computed_zorder = False
    plot.ax.scatter(
        cycle[:, 0],
        cycle[:, 1],
        cycle[:, 2],
        marker="o",
        s=28,
        facecolors="white",
        edgecolors="#111111",
        linewidths=0.8,
        depthshade=False,
        zorder=20,
    )
    plot.ax.scatter(
        [fixed[0]],
        [fixed[1]],
        [fixed[2]],
        marker="*",
        s=135,
        facecolors="#111111",
        edgecolors="white",
        linewidths=0.8,
        depthshade=False,
        zorder=21,
    )
    plot.ax.set_xlabel(r"$\log_{10}(\mathrm{midge})$", labelpad=7)
    plot.ax.set_ylabel(r"$\log_{10}(\mathrm{algae})$", labelpad=7)
    plot.ax.text2D(
        0.96,
        0.60,
        r"$\log_{10}(\mathrm{detritus})$",
        transform=plot.ax.transAxes,
        rotation=90,
        rotation_mode="anchor",
        ha="center",
        va="center",
        clip_on=False,
    )
    plot.ax.set_title(f"Direct Ives map: level-{source_level} Morse sets", pad=8)

    labels = sorted(np.unique(plot.data[:, 6].astype(np.int64)).tolist())
    node_handles = [
        mpatches.Patch(
            facecolor=PALETTE[node],
            edgecolor=(0.08, 0.08, 0.08, 0.25),
            label=f"$M_{{{node}}}$ ({roles.get(node, 'unassigned')})",
        )
        for node in labels
    ]
    invariant_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor="#111111",
            label="period-12 phases",
        ),
        mlines.Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=10,
            markerfacecolor="#111111",
            markeredgecolor="white",
            label="fixed point",
        ),
    ]
    plot.ax.legend(
        handles=[*node_handles, *invariant_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.0,
    )
    return save_figure(
        plot.fig,
        output_dir / basename,
        formats=("pdf", "png"),
        close=True,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )


def render(
    run_dir: Path,
    output_dir: Path,
    *,
    minimum_display_level: int,
) -> dict[str, Any]:
    manifest, labels, indices, lower, upper = _load_source(run_dir)
    source_level = int(manifest["morse_sets"]["source_level"])
    cover, diagnostics = _choose_display_cover(
        labels,
        indices,
        lower,
        upper,
        source_level=source_level,
        minimum_display_level=minimum_display_level,
    )
    roles, cycle, fixed = _roles_and_points(manifest["_membership"])
    _assert_points_in_role_cells(cover, roles, cycle, fixed)

    output_dir.mkdir(parents=True, exist_ok=True)
    cover_path = output_dir / (
        f"morse_sets_level{diagnostics['display_level']}_display_cover.csv"
    )
    np.savetxt(cover_path, cover, delimiter=",", fmt="%.16g")
    outputs = [
        *_render_view(
            cover_path,
            output_dir,
            basename="morse_sets_cubical_3d",
            source_level=source_level,
            roles=roles,
            cycle=cycle,
            fixed=fixed,
            elev=18.0,
            azim=-112.0,
        ),
        *_render_view(
            cover_path,
            output_dir,
            basename="morse_sets_cubical_3d_alternate",
            source_level=source_level,
            roles=roles,
            cycle=cycle,
            fixed=fixed,
            elev=22.0,
            azim=-55.0,
        ),
    ]
    payload = {
        "schema_version": 1,
        "purpose": "3-D cubical rendering of the direct analytic Ives Morse sets",
        "render_only": True,
        "scientific_scope": (
            f"Each displayed cell is a level-{diagnostics['display_level']} parent "
            f"of at least one saved level-{source_level} Morse cell. The display "
            "cover is not a recomputed coarser Morse decomposition."
        ),
        "source": {
            "run_manifest": _file_record(Path(manifest["_manifest_path"])),
            "reference_membership": _file_record(
                Path(manifest["_membership_path"])
            ),
            "morse_sets": _file_record(Path(manifest["_source_path"])),
        },
        "cover": {**diagnostics, "csv": _file_record(cover_path)},
        "roles": {str(node): role for node, role in sorted(roles.items())},
        "display": {
            "primary_camera": {"elev": 18.0, "azim": -112.0},
            "alternate_camera": {"elev": 22.0, "azim": -55.0},
            "period12_marker": "white circle with dark edge",
            "fixed_point_marker": "dark star with white edge",
        },
        "outputs": {path.name: _file_record(path) for path in outputs},
    }
    output_manifest = output_dir / "manifest.json"
    output_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--display-level", type=int, default=24)
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "render_3d"
    )
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1")
    payload = render(
        run_dir,
        output_dir,
        minimum_display_level=args.display_level,
    )
    print(json.dumps(payload["cover"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
