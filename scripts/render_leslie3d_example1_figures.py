#!/usr/bin/env python3
"""Render paper-ready leslie3d_example1 comparison figures from saved Morse artifacts.

This is display-only. It does not recompute a Morse graph, alter a Morse set,
or change a Conley index. Small boxes receive a visibility floor only in the
full-domain plots so they remain legible at one-half text width.

Reads and writes inside the bundle directory assembled by
``scripts/leslie3d_example1_package_bundle.py`` (``--output``; default
``output/leslie3d_example1_study/fixed22_vs_adaptive``): rendered panels are
placed next to their source CSVs and collected under
``<bundle>/paper_ready_no_legend``.  Copying finished panels into the
manuscript is performed outside this repository.  Requires the Graphviz
``dot`` executable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from latentdynamics._paths import get_repo_root


REPO_ROOT = get_repo_root()
DEFAULT_BUNDLE = (
    REPO_ROOT / "output" / "leslie3d_example1_study" / "fixed22_vs_adaptive"
)

# The manuscript uses amsart at 12 pt with a text width near 6.14 inches.
# These square panels are authored at their intended 0.5\textwidth placement.
TEXTWIDTH_IN = 6.14
PANEL_IN = 0.5 * TEXTWIDTH_IN
MIN_BOX_SIDE_FRAC = 0.005

COLORS = {
    0: "#FFB000",
    1: "#DC267F",
    2: "#648FFF",
    3: "#FE6100",
    4: "#785EF0",
    5: "#008080",
}
FIXED_COLORS = {
    0: COLORS[0],
    1: COLORS[1],
    9: COLORS[2],
    23: COLORS[5],
}
MERGED_COLORS = {
    0: COLORS[0],
    1: COLORS[1],
    2: COLORS[2],
    3: COLORS[3],
    4: COLORS[4],
}
ZOOM_COLORS = {
    0: COLORS[4],  # original node 4
    1: COLORS[5],  # original node 5
    2: "#D62728",  # added connection cells
}

PAPER_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
}


def load_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if values.shape[1] != 5:
        raise ValueError(f"expected 2-D Morse boxes in {path}, got {values.shape}")
    return values


def common_limits(
    arrays: list[np.ndarray],
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    occupied_lower = np.min(
        np.vstack([values[:, :2].min(axis=0) for values in arrays]), axis=0
    )
    occupied_upper = np.max(
        np.vstack([values[:, 2:4].max(axis=0) for values in arrays]), axis=0
    )
    occupied_span = occupied_upper - occupied_lower
    widths = np.vstack([values[:, 2:4] - values[:, :2] for values in arrays])
    median_width = np.median(widths, axis=0)
    margin = np.maximum(2.0 * median_width, 0.03 * occupied_span)
    lower = np.maximum(occupied_lower - margin, lower_bound)
    upper = np.minimum(occupied_upper + margin, upper_bound)
    return (float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))


def local_limits(
    values: np.ndarray,
    *,
    pad_fraction: float = 0.14,
) -> tuple[tuple[float, float], tuple[float, float]]:
    lower = values[:, :2].min(axis=0)
    upper = values[:, 2:4].max(axis=0)
    span = upper - lower
    median_width = np.median(values[:, 2:4] - values[:, :2], axis=0)
    pad = np.maximum(pad_fraction * span, 2.0 * median_width)
    return (
        (float(lower[0] - pad[0]), float(upper[0] + pad[0])),
        (float(lower[1] - pad[1]), float(upper[1] + pad[1])),
    )


def add_boxes(
    ax: plt.Axes,
    values: np.ndarray,
    colors: dict[int, str],
    limits: tuple[tuple[float, float], tuple[float, float]],
    *,
    minimum_side_fraction: float,
    zorder: int = 2,
) -> None:
    xlim, ylim = limits
    min_width = minimum_side_fraction * (xlim[1] - xlim[0])
    min_height = minimum_side_fraction * (ylim[1] - ylim[0])
    centers = 0.5 * (values[:, :2] + values[:, 2:4])
    widths = np.maximum(values[:, 2] - values[:, 0], min_width)
    heights = np.maximum(values[:, 3] - values[:, 1], min_height)
    left = centers[:, 0] - 0.5 * widths
    right = centers[:, 0] + 0.5 * widths
    bottom = centers[:, 1] - 0.5 * heights
    top = centers[:, 1] + 0.5 * heights

    vertices = np.empty((len(values), 4, 2), dtype=np.float64)
    vertices[:, 0, :] = np.column_stack((left, bottom))
    vertices[:, 1, :] = np.column_stack((right, bottom))
    vertices[:, 2, :] = np.column_stack((right, top))
    vertices[:, 3, :] = np.column_stack((left, top))

    labels = values[:, -1].astype(np.int64)
    unknown = sorted(set(labels.tolist()) - set(colors))
    if unknown:
        raise ValueError(f"no plot color configured for labels {unknown}")
    facecolors = np.empty((len(values), 4), dtype=np.float64)
    for label, color in colors.items():
        facecolors[labels == label] = to_rgba(color)
    ax.add_collection(
        PolyCollection(
            vertices,
            facecolors=facecolors,
            edgecolors="none",
            rasterized=True,
            zorder=zorder,
        )
    )


def style_full_axes(
    ax: plt.Axes,
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    xlim, ylim = limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)


def save_pair(fig: plt.Figure, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, dpi=300, bbox_inches=None)
    fig.savefig(png, dpi=300, bbox_inches=None)
    plt.close(fig)
    return pdf, png


def render_full(
    values: np.ndarray,
    colors: dict[int, str],
    limits: tuple[tuple[float, float], tuple[float, float]],
    stem: Path,
) -> tuple[Path, Path]:
    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(
            figsize=(PANEL_IN, PANEL_IN),
            layout="constrained",
        )
        add_boxes(
            ax,
            values,
            colors,
            limits,
            minimum_side_fraction=MIN_BOX_SIDE_FRAC,
        )
        style_full_axes(ax, limits)
        return save_pair(fig, stem)


def draw_distinguished_markers(
    ax: plt.Axes,
    fixed_point: np.ndarray,
    period_two: np.ndarray,
    *,
    compact: bool,
) -> None:
    ax.plot(
        period_two[:, 0],
        period_two[:, 1],
        linestyle="--",
        linewidth=0.75 if compact else 1.0,
        color="#333333",
        alpha=0.8,
        zorder=5,
    )
    ax.scatter(
        period_two[:, 0],
        period_two[:, 1],
        marker="o",
        s=14 if compact else 28,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
    )
    ax.scatter(
        [fixed_point[0]],
        [fixed_point[1]],
        marker="*",
        s=34 if compact else 65,
        facecolor="black",
        edgecolor="white",
        linewidth=0.5,
        zorder=7,
    )


def render_fixed_with_zoom(
    values: np.ndarray,
    limits: tuple[tuple[float, float], tuple[float, float]],
    fixed_point: np.ndarray,
    period_two: np.ndarray,
    stem: Path,
) -> tuple[Path, Path]:
    node23 = values[values[:, -1].astype(np.int64) == 23]
    zoom_limits = local_limits(node23, pad_fraction=0.16)
    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(
            figsize=(PANEL_IN, PANEL_IN),
            layout="constrained",
        )
        add_boxes(
            ax,
            values,
            FIXED_COLORS,
            limits,
            minimum_side_fraction=MIN_BOX_SIDE_FRAC,
        )
        style_full_axes(ax, limits)

        inset = ax.inset_axes([0.58, 0.47, 0.36, 0.48])
        add_boxes(
            inset,
            node23,
            {23: FIXED_COLORS[23]},
            zoom_limits,
            minimum_side_fraction=0.0,
        )
        inset.set_xlim(*zoom_limits[0])
        inset.set_ylim(*zoom_limits[1])
        inset.set_aspect("equal", adjustable="box")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.tick_params(axis="both", which="both", length=0)
        draw_distinguished_markers(
            inset,
            fixed_point,
            period_two,
            compact=True,
        )
        box_lower = node23[:, :2].min(axis=0)
        box_upper = node23[:, 2:4].max(axis=0)
        ax.add_patch(
            Rectangle(
                box_lower,
                *(box_upper - box_lower),
                fill=False,
                edgecolor=FIXED_COLORS[23],
                linewidth=0.8,
                zorder=8,
            )
        )
        return save_pair(fig, stem)


def render_adaptive_with_zoom(
    values: np.ndarray,
    detail_values: np.ndarray,
    limits: tuple[tuple[float, float], tuple[float, float]],
    stem: Path,
) -> tuple[Path, Path]:
    zoom_limits = local_limits(detail_values, pad_fraction=0.12)
    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(
            figsize=(PANEL_IN, PANEL_IN),
            layout="constrained",
        )
        add_boxes(
            ax,
            values,
            COLORS,
            limits,
            minimum_side_fraction=MIN_BOX_SIDE_FRAC,
        )
        style_full_axes(ax, limits)

        inset = ax.inset_axes([0.58, 0.47, 0.36, 0.48])
        add_boxes(
            inset,
            detail_values,
            ZOOM_COLORS,
            zoom_limits,
            minimum_side_fraction=0.0,
        )
        inset.set_xlim(*zoom_limits[0])
        inset.set_ylim(*zoom_limits[1])
        inset.set_aspect("equal", adjustable="box")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.tick_params(axis="both", which="both", length=0)
        return save_pair(fig, stem)


def render_zoom(
    values: np.ndarray,
    colors: dict[int, str],
    fixed_point: np.ndarray,
    period_two: np.ndarray,
    stem: Path,
    *,
    show_distinguished_markers: bool = True,
) -> tuple[Path, Path]:
    limits = local_limits(values, pad_fraction=0.12)
    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(2.1, PANEL_IN), layout="constrained")
        add_boxes(
            ax,
            values,
            colors,
            limits,
            minimum_side_fraction=0.0,
        )
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", length=0)
        if show_distinguished_markers:
            draw_distinguished_markers(
                ax,
                fixed_point,
                period_two,
                compact=False,
            )
        return save_pair(fig, stem)


def box_key(row: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in np.round(row[:4], decimals=14))


def merged_zoom_values(raw: np.ndarray, merged: np.ndarray) -> np.ndarray:
    labels = raw[:, -1].astype(np.int64)
    raw4 = raw[labels == 4].copy()
    raw5 = raw[labels == 5].copy()
    merged4 = merged[merged[:, -1].astype(np.int64) == 4].copy()
    fine_keys = {box_key(row) for row in np.vstack((raw4, raw5))}
    connections = np.asarray(
        [row for row in merged4 if box_key(row) not in fine_keys],
        dtype=np.float64,
    )
    if (len(raw4), len(raw5), len(connections), len(merged4)) != (174, 123, 25, 322):
        raise RuntimeError(
            "unexpected merged zoom counts: "
            f"{len(raw4)}, {len(raw5)}, {len(connections)}, {len(merged4)}"
        )
    raw4[:, -1] = 0
    raw5[:, -1] = 1
    connections[:, -1] = 2
    return np.vstack((raw4, raw5, connections))


def render_dot_pdf(dot: Path, pdf: Path) -> Path:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["dot", "-Tpdf", str(dot), "-o", str(pdf)], check=True)
    return pdf


def copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_paper_ready_folder(bundle: Path, sources: dict[str, Path]) -> Path:
    paper_ready = bundle / "paper_ready_no_legend"
    if paper_ready.exists():
        shutil.rmtree(paper_ready)
    paper_ready.mkdir(parents=True)
    for name, source in sorted(sources.items()):
        copy_artifact(source, paper_ready / name)

    manifest = {
        "display_only": True,
        "legends": False,
        "titles": False,
        "numeric_ticks": False,
        "intended_latex_width": "0.5\\textwidth",
        "minimum_box_side_fraction_full_views": MIN_BOX_SIDE_FRAC,
        "minimum_box_side_percent_full_views": 100.0 * MIN_BOX_SIDE_FRAC,
        "zoom_views_use_exact_box_sizes": True,
        "marker_semantics": {
            "star": "stable fixed point",
            "open_circles": "two phases of the unstable period-2 orbit",
            "dashed_segment": "joins the two period-2 phases for visual reference",
        },
        "marker_views": [
            "uniform_22_nontrivial_morse_sets_with_node23_zoom_no_legend.pdf",
            "uniform_22_nontrivial_morse_sets_with_node23_zoom_no_legend.png",
            "uniform_22_node23_zoom_no_legend.pdf",
            "uniform_22_node23_zoom_no_legend.png",
        ],
        "marker_free_views": [
            "adaptive_original_morse_sets_with_separate_4_5_zoom_no_legend.pdf",
            "adaptive_original_morse_sets_with_separate_4_5_zoom_no_legend.png",
            "adaptive_merged_4_5_zoom_no_legend.pdf",
            "adaptive_merged_4_5_zoom_no_legend.png",
        ],
        "display_palette": {
            "merged_4_5": "#785EF0",
            "merged_4_5_semantic": "ground-truth period-2 saddle color",
        },
        "files": sorted(sources),
    }
    (paper_ready / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = """# Paper-ready leslie3d_example1 comparison figures

All figures in this folder are display-only rerenders of the saved Morse data.
No Morse boxes, graph edges, node memberships, or Conley indices were changed.

The full-domain Morse-set panels use a minimum displayed box side of 0.5% of
the corresponding axis span. This is tuned for placement at 0.5\\textwidth.
The local zoom panels draw the computed boxes at their exact sizes because the
boxes are already visible at that scale.

There are no legends, titles, or numeric tick labels. The axes retain z1 and
z2. The uniform node-23 zooms use a star for the stable fixed point and two
open circles for the phases of the unstable period-2 orbit. The adaptive
merge-detail zoom contains only the Morse-set cover: purple for original node
4, teal for original node 5, and red for the 25 added connection cells. It has
no fixed-point or period-2 marker layer; the color mapping should be stated in
the caption. The adaptive original-with-zoom panel places that same marker-free
detail view as an inset over the full original Morse-set panel.

The connection-complete merged node [4,5] is purple (#785EF0) in both the
coarse Morse graph and the full merged Morse-set view, matching the
ground-truth period-2 saddle palette.

Copying panels from this folder into the manuscript is performed outside
this repository.
"""
    (paper_ready / "README.md").write_text(readme, encoding="utf-8")
    return paper_ready


def main(bundle: Path) -> None:
    fixed_dir = bundle / "uniform_22_22_22" / "nontrivial"
    raw_dir = bundle / "adaptive_23_23_27" / "raw"
    merged_dir = bundle / "adaptive_23_23_27" / "merged_4_5"

    fixed_values = load_csv(fixed_dir / "morse_sets.csv")
    raw_values = load_csv(raw_dir / "morse_sets.csv")
    merged_values = load_csv(merged_dir / "morse_sets.csv")
    result = json.loads(
        (bundle / "uniform_22_22_22" / "result.json").read_text(encoding="utf-8")
    )
    lower = np.asarray(result["bounds"]["lower"], dtype=np.float64)
    upper = np.asarray(result["bounds"]["upper"], dtype=np.float64)
    fixed_point = np.asarray(
        result["distinguished_objects"]["fixed_point"]["point"],
        dtype=np.float64,
    )
    period_two = np.asarray(
        result["distinguished_objects"]["period_two"]["phases"],
        dtype=np.float64,
    )
    limits = common_limits([fixed_values, raw_values, merged_values], lower, upper)

    fixed_graph_pdf = render_dot_pdf(
        fixed_dir / "morse_graph.dot",
        fixed_dir / "morse_graph.pdf",
    )
    merged_graph_pdf = render_dot_pdf(
        merged_dir / "morse_graph.dot",
        merged_dir / "morse_graph.pdf",
    )

    fixed_full_pdf, fixed_full_png = render_full(
        fixed_values,
        FIXED_COLORS,
        limits,
        fixed_dir / "morse_sets_full_no_legend",
    )
    fixed_combined_pdf, fixed_combined_png = render_fixed_with_zoom(
        fixed_values,
        limits,
        fixed_point,
        period_two,
        fixed_dir / "morse_sets",
    )
    fixed23 = fixed_values[fixed_values[:, -1].astype(np.int64) == 23].copy()
    fixed_zoom_pdf, fixed_zoom_png = render_zoom(
        fixed23,
        {23: FIXED_COLORS[23]},
        fixed_point,
        period_two,
        fixed_dir / "morse_set_23_zoom_no_legend",
    )

    raw_pdf, raw_png = render_full(
        raw_values,
        COLORS,
        limits,
        raw_dir / "morse_sets_no_legend",
    )
    merged_full_pdf, merged_full_png = render_full(
        merged_values,
        MERGED_COLORS,
        limits,
        merged_dir / "morse_sets_no_legend",
    )
    detail_values = merged_zoom_values(raw_values, merged_values)
    raw_combined_pdf, raw_combined_png = render_adaptive_with_zoom(
        raw_values,
        detail_values,
        limits,
        raw_dir / "morse_sets_with_separate_4_5_zoom_no_legend",
    )
    merged_zoom_pdf, merged_zoom_png = render_zoom(
        detail_values,
        ZOOM_COLORS,
        fixed_point,
        period_two,
        merged_dir / "morse_set_4_5",
        show_distinguished_markers=False,
    )

    sources = {
        "adaptive_original_morse_graph.pdf": raw_dir / "morse_graph.pdf",
        "adaptive_original_morse_sets_no_legend.pdf": raw_pdf,
        "adaptive_original_morse_sets_no_legend.png": raw_png,
        "adaptive_original_morse_sets_with_separate_4_5_zoom_no_legend.pdf": raw_combined_pdf,
        "adaptive_original_morse_sets_with_separate_4_5_zoom_no_legend.png": raw_combined_png,
        "adaptive_merged_4_5_morse_graph.pdf": merged_graph_pdf,
        "adaptive_merged_4_5_morse_sets_no_legend.pdf": merged_full_pdf,
        "adaptive_merged_4_5_morse_sets_no_legend.png": merged_full_png,
        "adaptive_merged_4_5_zoom_no_legend.pdf": merged_zoom_pdf,
        "adaptive_merged_4_5_zoom_no_legend.png": merged_zoom_png,
        "uniform_22_nontrivial_morse_graph.pdf": fixed_graph_pdf,
        "uniform_22_nontrivial_morse_sets_no_legend.pdf": fixed_full_pdf,
        "uniform_22_nontrivial_morse_sets_no_legend.png": fixed_full_png,
        "uniform_22_nontrivial_morse_sets_with_node23_zoom_no_legend.pdf": fixed_combined_pdf,
        "uniform_22_nontrivial_morse_sets_with_node23_zoom_no_legend.png": fixed_combined_png,
        "uniform_22_node23_zoom_no_legend.pdf": fixed_zoom_pdf,
        "uniform_22_node23_zoom_no_legend.png": fixed_zoom_png,
    }
    paper_ready = build_paper_ready_folder(bundle, sources)
    print(paper_ready)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BUNDLE,
        help=(
            "bundle directory assembled by leslie3d_example1_package_bundle.py "
            "(default: output/leslie3d_example1_study/fixed22_vs_adaptive)"
        ),
    )
    arguments = parser.parse_args()
    main(arguments.output)
