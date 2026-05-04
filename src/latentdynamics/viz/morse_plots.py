"""Render Morse graphs and Morse sets, either from a CMGDB object or from saved files.

The pipeline writes every Morse computation to two persistent artefacts:

- ``morse_graph``   : graphviz DOT file describing the Hasse diagram
- ``morse_sets``    : CSV file of boxes ``(lx, ly, ux, uy, label)`` for 2-D
                      latents, or intervals ``(a, b, label)`` for 1-D

The :func:`render_*_from_dot` and :func:`render_*_from_csv` helpers reload
those artefacts and re-emit PDF/PNG plots without invoking CMGDB.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import CMGDB
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pydot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .style import PALETTE, apply_paper_style


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
    labels: tuple[str, str] = ("$x_1$", "$x_2$"),
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


def save_morse_graph_artefacts(
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
    stage now uses :func:`save_morse_graph_artefacts` to keep computation and
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
    by ``palette[label % len(palette)]``; otherwise the file's own colours
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
    )

    rendered: list[Path] = []
    for fmt in formats:
        out_path = out / f"{basename}.{fmt}"
        plot.fig.savefig(out_path)
        rendered.append(out_path)
    plt.close(plot.fig)
    return rendered


def _plot_morse_sets_1d(
    data: NDArray[np.float64],
    palette: Sequence[str],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, dict[int, float]]:
    a, b, lbls = data[:, 0], data[:, 1], data[:, 2].astype(int)
    unique = sorted(np.unique(lbls).tolist())

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.6 * len(unique) + 1.5)))
    else:
        fig = ax.figure
    label_to_y = {lbl: float(row) for row, lbl in enumerate(unique)}
    for row, lbl in enumerate(unique):
        mask = lbls == lbl
        for ai, bi in zip(a[mask], b[mask], strict=False):
            ax.fill_betweenx(
                [row - 0.4, row + 0.4],
                ai,
                bi,
                color=palette[lbl % len(palette)],
                edgecolor="none",
            )
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels([f"M$_{{{lbl}}}$" for lbl in unique])
    ax.set_xlabel("$z_1$")
    ax.set_xlim(a.min(), b.max())
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig, ax, label_to_y


def _plot_morse_sets_2d(
    data: NDArray[np.float64],
    palette: Sequence[str],
    bounds_lower: Sequence[float] | None,
    bounds_upper: Sequence[float] | None,
    labels_2d: tuple[str, str],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    lx, ly, ux, uy, lbls = (data[:, i] for i in range(5))
    lbls = lbls.astype(int)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    seen: set[int] = set()
    for box_lx, box_ly, box_ux, box_uy, lbl in zip(lx, ly, ux, uy, lbls, strict=False):
        rect = mpatches.Rectangle(
            (box_lx, box_ly),
            box_ux - box_lx,
            box_uy - box_ly,
            facecolor=palette[lbl % len(palette)],
            edgecolor="none",
            label=f"Morse set {lbl}" if lbl not in seen else None,
        )
        ax.add_patch(rect)
        seen.add(lbl)

    if bounds_lower is not None and bounds_upper is not None:
        ax.set_xlim(bounds_lower[0], bounds_upper[0])
        ax.set_ylim(bounds_lower[1], bounds_upper[1])
    else:
        ax.set_xlim(lx.min(), ux.max())
        ax.set_ylim(ly.min(), uy.max())

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(labels_2d[0])
    ax.set_ylabel(labels_2d[1])
    fig.tight_layout()
    return fig, ax


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
    )
    pdf = next((p for p in graph_paths if p.suffix == ".pdf"), graph_paths[0])
    png = next((p for p in graph_paths if p.suffix == ".png"), graph_paths[-1])
    return RenderedMorseFigures(
        morse_graph_pdf=pdf,
        morse_graph_png=png,
        morse_sets_paths=set_paths,
    )
