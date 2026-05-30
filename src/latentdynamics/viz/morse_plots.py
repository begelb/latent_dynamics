"""Render Morse graphs and Morse sets, either from a CMGDB object or from saved files.

The pipeline writes every Morse computation to two persistent artifacts:

- ``morse_graph``   : graphviz DOT file describing the Hasse diagram
- ``morse_sets``    : CSV file of boxes ``(lx, ly, ux, uy, label)`` for 2-D
                      latents, or intervals ``(a, b, label)`` for 1-D

The :func:`render_*_from_dot` and :func:`render_*_from_csv` helpers reload
those artifacts and re-emit PDF/PNG plots without invoking CMGDB.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import CMGDB
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pydot
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .style import (
    LATENT_ARROW_MUTATION_SCALE,
    LATENT_FIG_WIDTH_IN,
    PALETTE,
    apply_paper_style,
    save_latent_figure,
    style_latent_axes,
)


@dataclass(frozen=True)
class RenderedMorseFigures:
    """Paths to every figure produced by :func:`render_morse_outputs`."""

    morse_graph_pdf: Path
    morse_graph_png: Path
    morse_sets_paths: list[Path]


@dataclass(frozen=True)
class MorseSetsPlot:
    """Live matplotlib Morse-set plot ready for postprocessing overlays."""

    fig: Figure
    ax: Axes
    data: NDArray[np.float64]
    dim: int
    label_to_y: dict[int, float]


# --------------------------------------------------------------------------- #
# CMGDB-object renderers (used by the morse stage when the object is in hand) #
# --------------------------------------------------------------------------- #


def render_morse_graph(
    morse_graph,
    out_dir: str | Path,
    *,
    basename: str = "morse_graph",
    formats: Sequence[str] = ("pdf", "png"),
    palette: Sequence[str] = PALETTE,
) -> list[Path]:
    """Render the Morse-graph Hasse diagram via graphviz once per format."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot = CMGDB.PlotMorseGraph(morse_graph, clist=list(palette))
    rendered: list[Path] = []
    for fmt in formats:
        plot.render(str(out / basename), format=fmt, view=False, cleanup=False)
        rendered.append(out / f"{basename}.{fmt}")
    return rendered


def render_morse_sets(
    morse_graph,
    out_dir: str | Path,
    *,
    bounds_lower: Sequence[float] | None = None,
    bounds_upper: Sequence[float] | None = None,
    formats: Sequence[str] = ("pdf", "png"),
    fontsize: int = 20,
    palette: Sequence[str] = PALETTE,
    labels: tuple[str, str] = ("$z_1$", "$z_2$"),
) -> list[Path]:
    """Render the Morse sets to one file per format using CMGDB's plotter."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    is_2d = bounds_lower is not None and bounds_upper is not None and len(bounds_lower) == 2

    rendered: list[Path] = []
    for fmt in formats:
        path = out / f"morse_sets.{fmt}"
        if is_2d:
            CMGDB.PlotMorseSets(
                morse_graph,
                clist=list(palette),
                xlim=[bounds_lower[0], bounds_upper[0]],
                ylim=[bounds_lower[1], bounds_upper[1]],
                xlabel=labels[0],
                ylabel=labels[1],
                fontsize=fontsize,
                fig_fname=str(path),
            )
        else:
            CMGDB.PlotMorseSets(
                morse_graph,
                clist=list(palette),
                fontsize=fontsize,
                fig_fname=str(path),
            )
        rendered.append(path)
    return rendered


def save_morse_graph_artifacts(
    morse_graph,
    out_dir: str | Path,
    *,
    palette: Sequence[str] = PALETTE,
) -> tuple[Path, Path]:
    """Persist the Morse computation as a DOT file + a Morse-sets CSV (no PDFs).

    These two files are sufficient to re-render every Morse plot offline:

    - ``morse_graph``  : graphviz DOT source written via ``Digraph.save``.
    - ``morse_sets``   : box CSV emitted by :func:`CMGDB.SaveMorseSets`.

    Rendering is deferred to the ``render`` stage / :func:`render_morse_from_files`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dot_path = out / "morse_graph"
    csv_path = out / "morse_sets"
    plot = CMGDB.PlotMorseGraph(morse_graph, clist=list(palette))
    plot.save(filename="morse_graph", directory=str(out))
    CMGDB.SaveMorseSets(morse_graph, str(csv_path))
    return dot_path, csv_path


def render_morse_outputs(
    morse_graph,
    out_dir: str | Path,
    *,
    bounds_lower: Sequence[float] | None = None,
    bounds_upper: Sequence[float] | None = None,
    palette: Sequence[str] = PALETTE,
) -> RenderedMorseFigures:
    """Render Hasse diagram + Morse-set plot in PDF and PNG, plus the CSV.

    Kept for callers that want one-shot compute+render. The pipeline's morse
    stage now uses :func:`save_morse_graph_artifacts` to keep computation and
    rendering decoupled.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph_paths = render_morse_graph(morse_graph, out, palette=palette)
    set_paths = render_morse_sets(
        morse_graph,
        out,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        palette=palette,
    )
    CMGDB.SaveMorseSets(morse_graph, str(out / "morse_sets"))
    pdf = next((p for p in graph_paths if p.suffix == ".pdf"), graph_paths[0])
    png = next((p for p in graph_paths if p.suffix == ".png"), graph_paths[-1])
    return RenderedMorseFigures(
        morse_graph_pdf=pdf,
        morse_graph_png=png,
        morse_sets_paths=set_paths,
    )


# --------------------------------------------------------------------------- #
# From-file renderers (used by the render stage; no CMGDB recompute)          #
# --------------------------------------------------------------------------- #


def render_morse_graph_from_dot(
    dot_path: str | Path,
    out_dir: str | Path,
    *,
    basename: str = "morse_graph",
    formats: Sequence[str] = ("pdf", "png"),
    palette: Sequence[str] | None = None,
) -> list[Path]:
    """Read a ``morse_graph`` DOT file and emit it as PDF/PNG via graphviz.

    If ``palette`` is given, the node ``fillcolor`` attributes are overridden
    by ``palette[label % len(palette)]``; otherwise the file's own colors
    are preserved.
    """
    dot_path = Path(dot_path)
    if not dot_path.exists():
        raise FileNotFoundError(dot_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    graphs = pydot.graph_from_dot_file(str(dot_path))
    if not graphs:
        raise ValueError(f"empty DOT file: {dot_path}")
    graph = graphs[0]

    if palette is not None:
        for node in graph.get_nodes():
            name = node.get_name()
            stripped = name.lstrip("-")
            if not stripped.isdigit():
                continue
            label_idx = int(stripped)
            node.set("fillcolor", palette[label_idx % len(palette)])

    rendered: list[Path] = []
    for fmt in formats:
        out_path = out / f"{basename}.{fmt}"
        method = getattr(graph, f"write_{fmt}", None)
        if method is None:
            raise ValueError(f"unsupported format {fmt!r} for graphviz output")
        method(str(out_path))
        rendered.append(out_path)
    return rendered


def _load_morse_sets(csv_path: Path) -> tuple[NDArray[np.float64], int]:
    data = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    n_cols = int(data.shape[1])
    if n_cols == 3:
        return data, 1
    if n_cols == 5:
        return data, 2
    raise ValueError(f"unrecognised morse_sets CSV (got {n_cols} cols, expected 3 or 5)")


def plot_morse_sets_from_csv(
    csv_path: str | Path,
    *,
    bounds_lower: Sequence[float] | None = None,
    bounds_upper: Sequence[float] | None = None,
    ax: Axes | None = None,
    palette: Sequence[str] = PALETTE,
    labels_2d: tuple[str, str] = ("$z_1$", "$z_2$"),
    paper_style: bool = True,
    box_scale: float | dict[int, float] | str = 1.0,
) -> MorseSetsPlot:
    """Read a saved ``morse_sets`` CSV and return a live matplotlib canvas.

    1-D format: rows ``(a, b, label)``. Each interval is drawn as a filled
    horizontal band, one row per Morse-set label.

    2-D format: rows ``(lower_x, lower_y, upper_x, upper_y, label)``. Each box
    is drawn as a filled rectangle.

    The returned :class:`MorseSetsPlot` keeps the figure open so callers can add
    postprocessing overlays before saving, e.g. encoded fixed points,
    trajectories, highlighted regions, or annotations.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if paper_style:
        apply_paper_style()

    data, dim = _load_morse_sets(csv_path)
    if dim == 1:
        fig, ax, label_to_y = _plot_morse_sets_1d(data, palette, ax=ax)
    else:
        fig, ax = _plot_morse_sets_2d(
            data,
            palette,
            bounds_lower,
            bounds_upper,
            labels_2d,
            ax=ax,
            box_scale=box_scale,
        )
        label_to_y = {}
    return MorseSetsPlot(fig=fig, ax=ax, data=data, dim=dim, label_to_y=label_to_y)


def render_morse_sets_from_csv(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    bounds_lower: Sequence[float] | None = None,
    bounds_upper: Sequence[float] | None = None,
    basename: str = "morse_sets",
    formats: Sequence[str] = ("pdf", "png"),
    palette: Sequence[str] = PALETTE,
    labels_2d: tuple[str, str] = ("$z_1$", "$z_2$"),
    paper_style: bool = True,
    box_scale: float | dict[int, float] | str = 1.0,
) -> list[Path]:
    """Read a saved ``morse_sets`` CSV and write rendered figure files.

    Existing pipeline callers use this saving wrapper. Overlay callers should
    use :func:`plot_morse_sets_from_csv` and save the returned figure after
    drawing their additions.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot = plot_morse_sets_from_csv(
        csv_path,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        palette=palette,
        labels_2d=labels_2d,
        paper_style=paper_style,
        box_scale=box_scale,
    )

    # save_latent_figure neutralises the global tight-bbox so the saved canvas
    # is exactly the figure's on-page width (1:1 \includegraphics placement).
    return save_latent_figure(plot.fig, out / basename, formats=tuple(formats), close=True)


def _plot_morse_sets_1d(
    data: NDArray[np.float64],
    palette: Sequence[str],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, dict[int, float]]:
    a, b, lbls = data[:, 0], data[:, 1], data[:, 2].astype(int)
    unique = sorted(np.unique(lbls).tolist())

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(LATENT_FIG_WIDTH_IN, round(LATENT_FIG_WIDTH_IN * 0.2, 3)),
            layout="constrained",
        )
    else:
        fig = ax.figure

    for ai, bi, lbl in zip(a, b, lbls, strict=False):
        ax.plot(
            [ai, bi],
            [0.0, 0.0],
            color=palette[int(lbl) % len(palette)],
            linewidth=12,
            solid_capstyle="projecting",
            zorder=0,
        )

    for lbl in unique:
        mask = lbls == lbl
        midpoint = 0.5 * (float(a[mask].min()) + float(b[mask].max()))
        ax.text(
            midpoint,
            0.25,
            f"M$_{{{lbl}}}$",
            ha="center",
            va="bottom",
            color="black",
        )

    ax.set_xlabel("$z_1$")
    ax.set_xlim(a.min(), b.max())
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.xaxis.set_ticks_position("bottom")
    style_latent_axes(ax, two_d=False)

    label_to_y = {lbl: 0.0 for lbl in unique}
    return fig, ax, label_to_y


def _resolve_box_scales(
    box_scale: float | dict[int, float] | str,
    lx: NDArray[np.float64],
    ly: NDArray[np.float64],
    ux: NDArray[np.float64],
    uy: NDArray[np.float64],
    lbls: NDArray[np.int_],
    *,
    min_frac: float = 0.025,
    max_scale: float = 10.0,
) -> Callable[[int], float]:
    """Return a ``label -> box-scale`` function from a float, dict, or ``"auto"``.

    ``"auto"`` inflates only Morse sets whose larger extent falls below
    ``min_frac`` of the occupied view span, scaling each of their boxes up to
    that floor (capped at ``max_scale``) so single-box / tiny sets stay visible;
    all other sets keep scale 1.0 (faithful).
    """
    if isinstance(box_scale, dict):
        per = {int(k): float(v) for k, v in box_scale.items()}
        return lambda lbl: per.get(int(lbl), 1.0)
    if isinstance(box_scale, str):
        if box_scale != "auto":
            raise ValueError(f"box_scale string must be 'auto', got {box_scale!r}")
        span = max(float(ux.max() - lx.min()), float(uy.max() - ly.min()), 1e-12)
        target = min_frac * span
        per = {}
        for label in np.unique(lbls):
            m = lbls == label
            ext = max(float(ux[m].max() - lx[m].min()), float(uy[m].max() - ly[m].min()))
            if 0.0 < ext < target:
                per[int(label)] = min(target / ext, max_scale)
        return lambda lbl: per.get(int(lbl), 1.0)
    bs = float(box_scale)
    return lambda lbl: bs


def _plot_morse_sets_2d(
    data: NDArray[np.float64],
    palette: Sequence[str],
    bounds_lower: Sequence[float] | None,
    bounds_upper: Sequence[float] | None,
    labels_2d: tuple[str, str],
    *,
    ax: Axes | None = None,
    box_scale: float | dict[int, float] | str = 1.0,
) -> tuple[Figure, Axes]:
    lx, ly, ux, uy, lbls = (data[:, i] for i in range(5))
    lbls = lbls.astype(int)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(LATENT_FIG_WIDTH_IN, LATENT_FIG_WIDTH_IN),
            layout="constrained",
        )
    else:
        fig = ax.figure
    # ``box_scale`` inflates each box about its own center so tiny attractor
    # sets stay visible. It may be a float (global), a {label: scale} dict
    # (per-set control), or "auto" (inflate only sets below a visibility floor).
    scale_for = _resolve_box_scales(box_scale, lx, ly, ux, uy, lbls)
    rects = []
    facecolors = []
    for box_lx, box_ly, box_ux, box_uy, lbl in zip(lx, ly, ux, uy, lbls, strict=False):
        s = scale_for(int(lbl))
        width = (box_ux - box_lx) * s
        height = (box_uy - box_ly) * s
        cx = 0.5 * (box_lx + box_ux)
        cy = 0.5 * (box_ly + box_uy)
        rects.append(mpatches.Rectangle((cx - 0.5 * width, cy - 0.5 * height), width, height))
        facecolors.append(palette[int(lbl) % len(palette)])
    # Draw all boxes as one rasterized collection (fast and small PDF even when a
    # Morse set has 10^5+ boxes), rather than adding each as a separate patch.
    ax.add_collection(
        PatchCollection(rects, facecolors=facecolors, edgecolors="none", rasterized=True)
    )

    xlim, ylim = _adaptive_2d_morse_set_limits(lx, ly, ux, uy, bounds_lower, bounds_upper)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(labels_2d[0])
    ax.set_ylabel(labels_2d[1])
    style_latent_axes(ax, two_d=True)
    return fig, ax


def _adaptive_2d_morse_set_limits(
    lx: NDArray[np.float64],
    ly: NDArray[np.float64],
    ux: NDArray[np.float64],
    uy: NDArray[np.float64],
    bounds_lower: Sequence[float] | None,
    bounds_upper: Sequence[float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    occupied_lower = np.asarray([lx.min(), ly.min()], dtype=np.float64)
    occupied_upper = np.asarray([ux.max(), uy.max()], dtype=np.float64)
    occupied_span = occupied_upper - occupied_lower

    widths = np.column_stack((ux - lx, uy - ly))
    median_width = np.zeros(2, dtype=np.float64)
    for axis in range(2):
        positive = widths[:, axis][widths[:, axis] > 0.0]
        if positive.size:
            median_width[axis] = float(np.median(positive))

    margin = np.maximum(2.0 * median_width, 0.03 * occupied_span)
    lower = occupied_lower - margin
    upper = occupied_upper + margin

    if bounds_lower is not None and bounds_upper is not None:
        cmgdb_lower = np.asarray(bounds_lower[:2], dtype=np.float64)
        cmgdb_upper = np.asarray(bounds_upper[:2], dtype=np.float64)
        lower = np.maximum(lower, cmgdb_lower)
        upper = np.minimum(upper, cmgdb_upper)

    return (float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))


def render_morse_from_files(
    morse_dir: str | Path,
    *,
    bounds_lower: Sequence[float] | None = None,
    bounds_upper: Sequence[float] | None = None,
    palette: Sequence[str] = PALETTE,
    out_dir: str | Path | None = None,
    basename_graph: str = "morse_graph",
    basename_sets: str = "morse_sets",
    paper_style: bool = True,
    box_scale: float | dict[int, float] | str = 1.0,
) -> RenderedMorseFigures:
    """Re-render both Morse outputs from saved ``morse_graph`` + ``morse_sets``."""
    morse_dir = Path(morse_dir)
    out = Path(out_dir) if out_dir is not None else morse_dir

    graph_paths = render_morse_graph_from_dot(
        morse_dir / "morse_graph",
        out,
        basename=basename_graph,
        palette=palette,
    )
    set_paths = render_morse_sets_from_csv(
        morse_dir / "morse_sets",
        out,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        basename=basename_sets,
        palette=palette,
        paper_style=paper_style,
        box_scale=box_scale,
    )
    pdf = next((p for p in graph_paths if p.suffix == ".pdf"), graph_paths[0])
    png = next((p for p in graph_paths if p.suffix == ".png"), graph_paths[-1])
    return RenderedMorseFigures(
        morse_graph_pdf=pdf,
        morse_graph_png=png,
        morse_sets_paths=set_paths,
    )


# --------------------------------------------------------------------------- #
# Morse-set + latent-trajectory overlay (grey arrows showing orbit direction) #
# --------------------------------------------------------------------------- #

_ARROW_GREY = "#6e6e6e"


def _draw_grey_trajectory(
    ax: Axes,
    traj: NDArray[np.float64],
) -> None:
    """Overlay one latent orbit as short grey arrows between iterates.

    Matches the shortened-arrow style of :func:`plot_latent_trajectory`: each
    connecting segment is trimmed 8% at both ends and the arrow heads are
    shrunk so they sit between, not over, the Morse-set boxes. No point markers
    are drawn -- the colored Morse-set boxes already mark each orbit point, so
    overlaying solid markers on them is redundant and distracting.
    """
    for start, end in zip(traj[:-1], traj[1:], strict=False):
        delta = end - start
        ax.plot(
            [start[0] + 0.08 * delta[0], end[0] - 0.08 * delta[0]],
            [start[1] + 0.08 * delta[1], end[1] - 0.08 * delta[1]],
            color=_ARROW_GREY,
            alpha=0.35,
            linewidth=0.8,
            zorder=5,
        )
    for i in range(len(traj) - 1):
        ax.annotate(
            "",
            xy=(traj[i + 1, 0], traj[i + 1, 1]),
            xytext=(traj[i, 0], traj[i, 1]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": _ARROW_GREY,
                "lw": 0.8,
                "alpha": 1.0,
                "mutation_scale": LATENT_ARROW_MUTATION_SCALE,
                "shrinkA": 7.0,
                "shrinkB": 7.0,
            },
            zorder=100,
        )


def _draw_grey_arrows(
    ax: Axes,
    arrows: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> None:
    """Draw explicit grey arrows, each spanning the gap between two components.

    Each ``(p0, p1)`` runs from a box on one Morse-set component to the closest
    box of the next component in orbit order. The head is shrunk so it stops at
    the target component's near edge instead of landing over its boxes.
    """
    for p0, p1 in arrows:
        ax.annotate(
            "",
            xy=(float(p1[0]), float(p1[1])),
            xytext=(float(p0[0]), float(p0[1])),
            arrowprops={
                "arrowstyle": "-|>",
                "color": _ARROW_GREY,
                "lw": 0.8,
                "alpha": 1.0,
                "mutation_scale": LATENT_ARROW_MUTATION_SCALE,
                "shrinkA": 3.0,
                "shrinkB": 6.0,
            },
            zorder=100,
        )


def render_morse_sets_with_overlay(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    latent_starts: NDArray[np.float64] | None = None,
    advance_latent: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    arrows: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]] | None = None,
    bounds_lower: Sequence[float] | None = None,
    bounds_upper: Sequence[float] | None = None,
    trajectory_steps: int | Sequence[int] = 6,
    box_scale: float | dict[int, float] | str = 1.0,
    basename: str = "morse_sets_with_overlay",
    formats: tuple[str, ...] = ("pdf", "png"),
    palette: Sequence[str] = PALETTE,
    labels_2d: tuple[str, str] = ("$z_1$", "$z_2$"),
    paper_style: bool = True,
) -> list[Path]:
    """Render filled Morse sets with grey latent-orbit arrows overlaid.

    ``latent_starts`` are ``(N, 2)`` seed points already in latent space; each
    is advanced ``trajectory_steps`` times through ``advance_latent`` and drawn
    as a short grey arrow chain. ``box_scale`` inflates the Morse boxes so tiny
    attractor sets remain visible under the overlay.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot = plot_morse_sets_from_csv(
        csv_path,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        palette=palette,
        labels_2d=labels_2d,
        paper_style=paper_style,
        box_scale=box_scale,
    )
    if plot.dim != 2:
        raise ValueError("morse_sets_with_overlay requires a 2-D latent space")

    if arrows is not None:
        _draw_grey_arrows(plot.ax, arrows)
    elif latent_starts is not None and advance_latent is not None:
        starts = np.atleast_2d(np.asarray(latent_starts, dtype=np.float64))
        if isinstance(trajectory_steps, int):
            steps_per_start = [trajectory_steps] * len(starts)
        else:
            steps_per_start = [int(s) for s in trajectory_steps]
        for z0, steps in zip(starts, steps_per_start, strict=False):
            traj = [z0]
            z = z0[None, :]
            for _ in range(steps):
                z = np.asarray(advance_latent(z), dtype=np.float64)
                traj.append(z[0])
            _draw_grey_trajectory(plot.ax, np.asarray(traj))

    written = save_latent_figure(
        plot.fig, out / basename, formats=tuple(formats), close=True
    )
    return written
