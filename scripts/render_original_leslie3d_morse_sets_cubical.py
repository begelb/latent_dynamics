"""Render a Chafee--Infante-style 3-D view of the saved Leslie Morse sets.

The exact saved sets contain almost two million aligned level-33 cells.  This
render-only utility maps each occupied cell to its containing level-24 cell and
draws the resulting labeled outer display cover with the repository's cubical
surface renderer.  It does not rerun CMGDB, alter the Morse graph, or claim a
new level-24 Morse decomposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray

from latentdynamics.viz import plot_morse_sets_3d_cubical_from_csv
from latentdynamics.viz.style import save_figure

CODE_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "ground_truth"
    / "absorbing_B_i29_m33_M36_L10000"
)
UNIFORM_ROOT = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "ground_truth"
    / "absorbing_B_uniform_level33_recurrent_closure"
)
DEFAULT_SOURCE = RUN_ROOT / "screen" / "MG" / "morse_sets"
DEFAULT_MANIFEST = UNIFORM_ROOT / "manifest.json"
DEFAULT_OUTPUT = UNIFORM_ROOT / "cubical_3d_level24_display_cover"

SOURCE_LEVEL = 33
DISPLAY_LEVEL = 24
SOURCE_AXIS_SPLITS = (11, 11, 11)
DISPLAY_AXIS_SPLITS = (8, 8, 8)
PALETTE = (
    "#FFB000",
    "#DC267F",
    "#FE6100",
    "#648FFF",
    "#785EF0",
    "#008080",
)
LEGEND_LABELS = {
    0: r"$M_0\;(P_0)$",
    1: r"$M_1\;(P_1)$",
    2: r"$M_2\;(S_2)$",
    3: r"$M_3\;(S_4)$",
    4: r"$M_4\;(p_*)$",
    5: r"$M_5\;(0)$",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    grid = manifest.get("uniform_grid", {})
    if grid.get("level") != SOURCE_LEVEL:
        raise ValueError(f"expected a level-{SOURCE_LEVEL} manifest: {path}")
    if tuple(grid.get("axis_splits", ())) != SOURCE_AXIS_SPLITS:
        raise ValueError(f"unexpected source axis splits in {path}")
    return manifest


def _load_and_validate_source(
    source: Path,
    manifest: dict[str, Any],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    recorded_hash = manifest.get("source", {}).get("morse_sets_sha256")
    observed_hash = _sha256(source)
    if observed_hash != recorded_hash:
        raise ValueError(
            f"saved-set hash differs from the uniform computation: "
            f"{observed_hash} != {recorded_hash}"
        )

    data = np.loadtxt(source, delimiter=",", ndmin=2, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(f"expected seven Morse-set columns; got {data.shape}")
    raw_labels = data[:, 6]
    labels = np.rint(raw_labels).astype(np.int64)
    if not np.array_equal(raw_labels, labels):
        raise ValueError("Morse-set labels are not integral")
    if set(np.unique(labels)) != set(range(len(PALETTE))):
        raise ValueError(f"expected labels 0--{len(PALETTE) - 1}")

    domain = manifest.get("domain", {})
    lower = np.asarray(domain.get("lower"), dtype=np.float64)
    upper = np.asarray(domain.get("upper"), dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("uniform manifest has invalid domain bounds")

    grid = manifest["uniform_grid"]
    widths = np.asarray(grid.get("box_widths"), dtype=np.float64)
    counts = np.asarray(grid.get("axis_counts"), dtype=np.int64)
    if widths.shape != (3,) or tuple(counts) != (2048, 2048, 2048):
        raise ValueError("uniform manifest has unexpected grid geometry")
    if not np.allclose(data[:, 3:6] - data[:, :3], widths, rtol=0.0, atol=1e-12):
        raise ValueError("not every saved box is a level-33 cell")

    indices_float = (data[:, :3] - lower) / widths
    indices = np.rint(indices_float).astype(np.int64)
    if not np.allclose(indices_float, indices, rtol=0.0, atol=1e-9):
        raise ValueError("saved boxes do not align with the level-33 grid")
    if np.any(indices < 0) or np.any(indices >= counts):
        raise ValueError("saved box lies outside the level-33 domain")

    observed_counts = {
        str(label): int(np.count_nonzero(labels == label))
        for label in range(len(PALETTE))
    }
    recorded_counts = manifest.get("morse_boxes_per_node")
    if observed_counts != recorded_counts:
        raise ValueError(
            f"saved-set counts differ from the manifest: "
            f"{observed_counts} != {recorded_counts}"
        )
    return labels, indices, lower, upper


def _build_display_cover(
    labels: NDArray[np.int64],
    source_indices: NDArray[np.int64],
    domain_lower: NDArray[np.float64],
    domain_upper: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    factors = np.asarray(
        [
            1 << (source_split - display_split)
            for source_split, display_split in zip(
                SOURCE_AXIS_SPLITS,
                DISPLAY_AXIS_SPLITS,
                strict=True,
            )
        ],
        dtype=np.int64,
    )
    if np.any(factors <= 0):
        raise ValueError("display grid must be coarser than the source grid")
    display_counts = np.asarray(
        [1 << split for split in DISPLAY_AXIS_SPLITS],
        dtype=np.int64,
    )
    display_indices = source_indices // factors
    volume = int(np.prod(display_counts))
    geometry = (
        (display_indices[:, 0] * display_counts[1] + display_indices[:, 1])
        * display_counts[2]
        + display_indices[:, 2]
    )
    labeled_keys = labels * volume + geometry
    unique_keys = np.unique(labeled_keys)
    cover_labels = (unique_keys // volume).astype(np.int64)
    cover_geometry = unique_keys % volume
    if np.unique(cover_geometry).size != cover_geometry.size:
        raise ValueError("two Morse sets occupy the same display-cover cell")

    ix = cover_geometry // (display_counts[1] * display_counts[2])
    remainder = cover_geometry % (display_counts[1] * display_counts[2])
    iy = remainder // display_counts[2]
    iz = remainder % display_counts[2]
    cover_indices = np.column_stack((ix, iy, iz)).astype(np.int64)
    display_widths = (domain_upper - domain_lower) / display_counts
    cover_lower = domain_lower + cover_indices * display_widths
    cover_upper = cover_lower + display_widths
    cover = np.column_stack((cover_lower, cover_upper, cover_labels)).astype(
        np.float64
    )

    source_counts = {
        str(label): int(np.count_nonzero(labels == label))
        for label in range(len(PALETTE))
    }
    cover_counts = {
        str(label): int(np.count_nonzero(cover_labels == label))
        for label in range(len(PALETTE))
    }
    diagnostics = {
        "source_level": SOURCE_LEVEL,
        "source_axis_splits": list(SOURCE_AXIS_SPLITS),
        "display_level": DISPLAY_LEVEL,
        "display_axis_splits": list(DISPLAY_AXIS_SPLITS),
        "coarsening_factor_by_axis": factors.tolist(),
        "display_axis_counts": display_counts.tolist(),
        "display_box_widths": display_widths.tolist(),
        "source_boxes_per_node": source_counts,
        "display_cells_per_node": cover_counts,
        "source_box_count": int(labels.size),
        "display_cell_count": int(cover.shape[0]),
        "cross_label_display_cell_overlaps": 0,
    }
    return cover, diagnostics


def _emphasize_origin_cube(plot: Any) -> None:
    origin_rows = plot.data[plot.data[:, 6].astype(np.int64) == 5]
    if origin_rows.shape[0] != 1:
        raise ValueError("expected one level-24 origin display cell")
    x0, y0, z0, x1, y1, z1 = origin_rows[0, :6]
    faces = np.asarray(
        [
            [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
            [[x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0]],
            [[x0, y0, z0], [x0, y0, z1], [x1, y0, z1], [x1, y0, z0]],
            [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
            [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
            [[x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1]],
        ],
        dtype=np.float64,
    )
    plot.ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors=PALETTE[5],
            edgecolors=(0.03, 0.03, 0.03, 0.95),
            linewidths=0.65,
            alpha=1.0,
            rasterized=True,
            zsort="average",
            shade=False,
        )
    )


def _render_view(
    cover_path: Path,
    output_dir: Path,
    *,
    basename: str,
    elev: float,
    azim: float,
    labeled: bool,
    emphasize_origin: bool,
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
        edge_alpha=0.16,
        edge_linewidth=0.065,
        minimal_frame=True,
        show_ticks=labeled,
        show_axis_labels=False,
        show_legend=True,
        legend_labels=LEGEND_LABELS,
    )
    if labeled:
        plot.ax.set_xlabel("$x_1$", labelpad=5)
        plot.ax.set_ylabel("$x_2$", labelpad=5)
        plot.ax.text2D(
            0.95,
            0.60,
            "$x_3$",
            transform=plot.ax.transAxes,
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="center",
            clip_on=False,
        )
    if emphasize_origin:
        _emphasize_origin_cube(plot)
    return save_figure(
        plot.fig,
        output_dir / basename,
        formats=("pdf", "png"),
        close=True,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def render(
    source: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    labels, indices, lower, upper = _load_and_validate_source(source, manifest)
    cover, diagnostics = _build_display_cover(labels, indices, lower, upper)

    output_dir.mkdir(parents=True, exist_ok=True)
    cover_path = output_dir / "morse_sets_level24_display_cover.csv"
    np.savetxt(cover_path, cover, delimiter=",", fmt="%.16g")

    outputs = [
        *_render_view(
            cover_path,
            output_dir,
            basename="morse_sets_cubical_3d",
            elev=22.0,
            azim=-55.0,
            labeled=False,
            emphasize_origin=False,
        ),
        *_render_view(
            cover_path,
            output_dir,
            basename="morse_sets_cubical_3d_labeled",
            elev=22.0,
            azim=-55.0,
            labeled=True,
            emphasize_origin=True,
        ),
        *_render_view(
            cover_path,
            output_dir,
            basename="morse_sets_cubical_3d_x1_x3_view",
            elev=18.0,
            azim=-90.0,
            labeled=False,
            emphasize_origin=True,
        ),
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "3-D cubical display of the six saved original-Leslie Morse sets",
        "render_only": True,
        "scientific_scope": (
            "Each displayed cell is the level-24 parent of at least one exact "
            "saved level-33 Morse cell. This is a labeled outer display cover, "
            "not a recomputed level-24 Morse decomposition."
        ),
        "source": {
            "morse_sets": _file_record(source),
            "uniform_manifest": _file_record(manifest_path),
        },
        "cover": {
            **diagnostics,
            "csv": _file_record(cover_path),
        },
        "display": {
            "palette_by_node": {
                str(label): color for label, color in enumerate(PALETTE)
            },
            "paper_camera": {"elev": 22.0, "azim": -55.0},
            "x1_x3_camera": {"elev": 18.0, "azim": -90.0},
            "shade_strength": 0.28,
            "highlight_strength": 0.10,
            "origin_cube_emphasis": (
                "The labeled and x1-x3 views redraw the single M5 display "
                "cell at its exact bounds with a stronger edge. No marker or "
                "geometric enlargement is used."
            ),
        },
        "outputs": {path.name: _file_record(path) for path in outputs},
    }
    manifest_output = output_dir / "manifest.json"
    manifest_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1")
    payload = render(
        args.source.resolve(),
        args.manifest.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(payload["cover"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
