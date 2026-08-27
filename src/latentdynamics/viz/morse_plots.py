"""Render Morse graphs and Morse sets, either from a CMGDB object or from saved files.

The pipeline writes every Morse computation to two persistent artifacts:

- ``morse_graph``   : graphviz DOT file describing the Hasse diagram
- ``morse_sets``    : CSV file of boxes
                      ``(lower_1, ..., lower_d, upper_1, ..., upper_d, label)``

The :func:`render_*_from_dot` and :func:`render_*_from_csv` helpers reload
those artifacts and re-emit PDF/PNG plots without invoking CMGDB.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import CMGDB
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pydot
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import LightSource, to_rgb
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray

from .style import (
    LATENT_ARROW_MUTATION_SCALE,
    LATENT_FIG_WIDTH_IN,
    PALETTE,
    apply_paper_style,
    save_figure,
    save_latent_figure,
    style_latent_axes,
)

#: Above this many exposed faces a vector PDF stops being the smaller option,
#: and the 3-D collection is rasterized instead. Below it, face culling has
#: already reduced the geometry enough that vector wins on both size and
#: fidelity.
RASTERIZE_FACE_THRESHOLD = 200_000


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



def plot_morse_sets_2d_cmgdb(
    morse_sets: Any,
    out_path: str | Path,
    *,
    scale_factor: Sequence[float] | None = None,
    morse_nodes: Sequence[int] | None = None,
    zoom_nodes: Sequence[int] | None = None,
    zoom_pos: Sequence[float] | None = None,
    zoom_pad: float = 0.25,
    palette: Sequence[str] = PALETTE,
    xlabel: str = "$z_1$",
    ylabel: str = "$z_2$",
    fig_w: float = 8.0,
    fig_h: float = 8.0,
    margin: float = 0.02,
    dpi: int = 300,
) -> Path:
    """Draw 2-D Morse sets with ``CMGDB.PlotMorseSets`` and save one figure.

    CMGDB renders each box as a filled rectangle at its true extent, and takes
    ``scale_factor`` as a list indexed by Morse node --
    entry ``i`` enlarges set ``i``. That is CMGDB's own emphasis mechanism, so
    figures produced here match what CMGDB draws elsewhere rather than
    reproducing it through this package's rectangle renderer.

    ``morse_sets`` is a saved box CSV path, a live ``MorseGraph``, or a list of
    rows. A short ``scale_factor`` is padded with ones, since CMGDB indexes it
    by node and would otherwise raise on a run that resolved more sets than the
    caller listed.

    ``margin`` pads the axes by that fraction of the drawn span on each side so
    the sets clear the bounding box. ``zoom_nodes`` magnifies the region holding
    those sets in an inset joined to its source by connector lines -- the honest
    alternative to a large ``scale_factor`` for a set only a few boxes across,
    since it preserves true box size. ``morse_nodes`` restricts which sets are
    drawn at all.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if scale_factor is not None:
        rows = (
            CMGDB.LoadMorseSetFile(str(morse_sets))
            if isinstance(morse_sets, (str, Path))
            else None
        )
        if rows is not None:
            node_count = max(int(row[-1]) for row in rows) + 1
        elif hasattr(morse_sets, "num_vertices"):
            node_count = int(morse_sets.num_vertices())
        else:
            node_count = max(int(row[-1]) for row in morse_sets) + 1
        factors = [float(v) for v in scale_factor]
        if len(factors) < node_count:
            factors = factors + [1.0] * (node_count - len(factors))
        scale_factor = factors

    CMGDB.PlotMorseSets(
        str(morse_sets) if isinstance(morse_sets, Path) else morse_sets,
        clist=list(palette),
        scale_factor=scale_factor,
        morse_nodes=list(morse_nodes) if morse_nodes is not None else None,
        zoom_nodes=list(zoom_nodes) if zoom_nodes else None,
        zoom_pos=list(zoom_pos) if zoom_pos else None,
        zoom_pad=zoom_pad,
        xlabel=xlabel,
        ylabel=ylabel,
        fig_w=fig_w,
        fig_h=fig_h,
        margin=margin,
        fig_fname=str(out_path),
        dpi=dpi,
    )
    return out_path


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
    if n_cols < 3 or n_cols % 2 == 0:
        raise ValueError(
            "unrecognised morse_sets CSV "
            f"(got {n_cols} cols, expected 2*d+1 columns for some d >= 1)"
        )
    return data, (n_cols - 1) // 2


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
    box_scale_min_frac: float = 0.025,
    box_scale_max: float = 10.0,
    min_box_side_frac: float = 0.0,
    label_draw_order: Sequence[int] | None = None,
) -> MorseSetsPlot:
    """Read a saved ``morse_sets`` CSV and return a live matplotlib canvas.

    1-D format: rows ``(a, b, label)``. Each interval is drawn as a filled
    horizontal band, one row per Morse-set label.

    2-D format: rows ``(lower_x, lower_y, upper_x, upper_y, label)``. Each box
    is drawn as a filled rectangle.

    ``min_box_side_frac`` is a display-only visibility floor: after applying
    ``box_scale``, each side is drawn at least this fraction of the plotted
    span on the corresponding axis.  The default zero preserves the computed
    box dimensions exactly.

    For 2-D data, ``label_draw_order`` may list every plotted label from
    back to front. By default, boxes retain their CSV row order.

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
        if label_draw_order is not None:
            raise ValueError("label_draw_order is supported only for 2-D Morse-set plots")
        fig, ax, label_to_y = _plot_morse_sets_1d(data, palette, ax=ax)
    elif dim == 2:
        fig, ax = _plot_morse_sets_2d(
            data,
            palette,
            bounds_lower,
            bounds_upper,
            labels_2d,
            ax=ax,
            box_scale=box_scale,
            box_scale_min_frac=box_scale_min_frac,
            box_scale_max=box_scale_max,
            min_box_side_frac=min_box_side_frac,
            label_draw_order=label_draw_order,
        )
        label_to_y = {}
    else:
        raise ValueError(
            f"{csv_path} contains {dim}-D Morse boxes; use "
            "plot_morse_set_projections_from_csv for 2-D coordinate-pair projections"
        )
    return MorseSetsPlot(fig=fig, ax=ax, data=data, dim=dim, label_to_y=label_to_y)


def plot_morse_set_projections_from_csv(
    csv_path: str | Path,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    bounds_lower: Sequence[float] | None = None,
    bounds_upper: Sequence[float] | None = None,
    palette: Sequence[str] = PALETTE,
    paper_style: bool = True,
    box_scale: float | dict[int, float] | str = 1.0,
    box_scale_min_frac: float = 0.025,
    box_scale_max: float = 10.0,
    min_box_side_frac: float = 0.0,
    label_draw_order: Sequence[int] | None = None,
) -> dict[tuple[int, int], MorseSetsPlot]:
    """Plot requested 2-D coordinate-pair projections of saved Morse boxes.

    CMGDB writes a ``d``-dimensional box as
    ``(lower_0, ..., lower_{d-1}, upper_0, ..., upper_{d-1}, label)``.
    ``pairs`` uses those zero-based coordinate indices. If it is omitted, all
    ``d choose 2`` pairs are plotted in lexicographic order. Axis labels remain
    one-based mathematical labels (``z_1``, ..., ``z_d``).

    Each projection is a separate paper-sized figure and is returned under its
    coordinate pair. ``bounds_lower`` and ``bounds_upper``, when supplied, are
    the full-dimensional CMGDB bounds; the matching coordinates are selected
    for every panel. ``min_box_side_frac`` is the same optional display-only
    visibility floor supported by :func:`plot_morse_sets_from_csv`.
    ``label_draw_order`` lists every projected label from back to front; the
    default preserves CSV row order.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if paper_style:
        apply_paper_style()

    data, dim = _load_morse_sets(csv_path)
    requested_pairs = _resolve_projection_pairs(pairs, dim)
    projected_bounds = _validate_projection_bounds(bounds_lower, bounds_upper, dim)

    plots: dict[tuple[int, int], MorseSetsPlot] = {}
    for pair in requested_pairs:
        i, j = pair
        projected = data[:, [i, j, dim + i, dim + j, 2 * dim]]
        if projected_bounds is None:
            lower_2d = upper_2d = None
        else:
            full_lower, full_upper = projected_bounds
            lower_2d = [full_lower[i], full_lower[j]]
            upper_2d = [full_upper[i], full_upper[j]]

        fig, ax = _plot_morse_sets_2d(
            projected,
            palette,
            lower_2d,
            upper_2d,
            (f"$z_{{{i + 1}}}$", f"$z_{{{j + 1}}}$"),
            box_scale=box_scale,
            box_scale_min_frac=box_scale_min_frac,
            box_scale_max=box_scale_max,
            min_box_side_frac=min_box_side_frac,
            label_draw_order=label_draw_order,
        )
        plots[pair] = MorseSetsPlot(
            fig=fig,
            ax=ax,
            data=projected,
            dim=2,
            label_to_y={},
        )
    return plots


def render_morse_set_projections_from_csv(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    bounds_lower: Sequence[float] | None = None,
    bounds_upper: Sequence[float] | None = None,
    basename: str = "morse_sets",
    formats: Sequence[str] = ("pdf", "png"),
    palette: Sequence[str] = PALETTE,
    paper_style: bool = True,
    box_scale: float | dict[int, float] | str = 1.0,
    box_scale_min_frac: float = 0.025,
    box_scale_max: float = 10.0,
    min_box_side_frac: float = 0.0,
    label_draw_order: Sequence[int] | None = None,
) -> dict[tuple[int, int], list[Path]]:
    """Render saved Morse-box projections to one file per pair and format.

    Files are named ``{basename}_z{i}_z{j}.{format}``, with one-based coordinate
    numbers in the filename. The returned mapping uses the zero-based pairs
    accepted by ``pairs``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots = plot_morse_set_projections_from_csv(
        csv_path,
        pairs=pairs,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        palette=palette,
        paper_style=paper_style,
        box_scale=box_scale,
        box_scale_min_frac=box_scale_min_frac,
        box_scale_max=box_scale_max,
        min_box_side_frac=min_box_side_frac,
        label_draw_order=label_draw_order,
    )

    rendered: dict[tuple[int, int], list[Path]] = {}
    for (i, j), plot in plots.items():
        pair_basename = f"{basename}_z{i + 1}_z{j + 1}"
        rendered[(i, j)] = save_latent_figure(
            plot.fig,
            out / pair_basename,
            formats=tuple(formats),
            close=True,
        )
    return rendered


def _exposed_cubical_faces(
    data: NDArray[np.float64],
    scale_of: Callable[[int], float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Return exposed quadrilateral faces of aligned three-dimensional boxes.

    ``scale_of`` maps a Morse label to a display inflation factor. Culling runs
    on the *unscaled* grid, where adjacency is what it means, and each surviving
    face is then scaled about its own cell's centre. Scaling the cells first
    would defeat the culling: inflated cells no longer sit on one aligned
    lattice, which is the invariant this face test depends on.
    """

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"3-D cubical Morse data must have shape (n, 7); got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("3-D cubical Morse data must contain at least one box")

    lower = values[:, :3]
    upper = values[:, 3:6]
    labels = values[:, 6].astype(int)
    widths = upper - lower
    if np.any(widths <= 0.0):
        raise ValueError("every 3-D Morse box must have positive side lengths")
    cell_width = np.median(widths, axis=0)
    if not np.allclose(widths, cell_width, rtol=1e-8, atol=1e-11):
        raise ValueError(
            "cubical surface rendering requires Morse boxes on one aligned terminal grid"
        )

    origin = lower.min(axis=0)
    indices = np.rint((lower - origin) / cell_width).astype(np.int64)
    if not np.allclose(lower, origin + indices * cell_width, rtol=1e-8, atol=1e-10):
        raise ValueError("3-D Morse boxes are not aligned to a common cubical grid")

    occupied = {
        (int(label), int(index[0]), int(index[1]), int(index[2]))
        for label, index in zip(labels, indices, strict=True)
    }
    if len(occupied) != values.shape[0]:
        raise ValueError("3-D Morse-box data contain duplicate labeled grid cells")

    faces: list[list[list[float]]] = []
    face_labels: list[int] = []
    for lo, hi, label, index in zip(lower, upper, labels, indices, strict=True):
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        vertices = (
            [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
            [[x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0]],
            [[x0, y0, z0], [x0, y0, z1], [x1, y0, z1], [x1, y0, z0]],
            [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
            [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
            [[x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1]],
        )
        neighbors = (
            (index[0] - 1, index[1], index[2]),
            (index[0] + 1, index[1], index[2]),
            (index[0], index[1] - 1, index[2]),
            (index[0], index[1] + 1, index[2]),
            (index[0], index[1], index[2] - 1),
            (index[0], index[1], index[2] + 1),
        )
        factor = 1.0 if scale_of is None else float(scale_of(int(label)))
        centre = ((lo + hi) / 2.0) if factor != 1.0 else None
        for face, neighbor in zip(vertices, neighbors, strict=True):
            if (int(label), *map(int, neighbor)) not in occupied:
                if centre is not None:
                    face = (
                        centre + (np.asarray(face, dtype=np.float64) - centre) * factor
                    ).tolist()
                faces.append(face)
                face_labels.append(int(label))

    return np.asarray(faces, dtype=np.float64), np.asarray(face_labels, dtype=int)


def _subtly_shaded_cubical_facecolors(
    faces: NDArray[np.float64],
    face_labels: NDArray[np.int_],
    palette: Sequence[str],
    *,
    light_azdeg: float,
    light_altdeg: float,
    strength: float,
    highlight_strength: float,
) -> NDArray[np.float64]:
    """Shade axis-aligned faces without obscuring their Morse-set colors."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError("shade_strength must lie in [0, 1]")
    if not 0.0 <= highlight_strength <= 1.0:
        raise ValueError("highlight_strength must lie in [0, 1]")

    base_colors = np.asarray(
        [to_rgb(palette[int(label) % len(palette)]) for label in face_labels],
        dtype=np.float64,
    )
    if strength == 0.0:
        return base_colors

    # The vertices in ``_exposed_cubical_faces`` wind inward, so reverse the
    # cross-product normals before applying directional diffuse illumination.
    normals = -np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0.0)
    light_direction = LightSource(
        azdeg=light_azdeg,
        altdeg=light_altdeg,
    ).direction
    diffuse = np.clip(normals @ light_direction, 0.0, 1.0)
    intensity = 1.0 - strength * (1.0 - diffuse)
    shaded = base_colors * intensity[:, None]
    highlight = highlight_strength * np.clip((diffuse - 0.4) / 0.6, 0.0, 1.0)
    shaded += (1.0 - shaded) * highlight[:, None]
    return np.clip(shaded, 0.0, 1.0)


def _cubical_edgecolors(
    face_labels: NDArray[np.int_],
    palette: Sequence[str],
    *,
    alpha: float,
    light_edges_on_dark_faces: bool,
) -> NDArray[np.float64]:
    """Return restrained grid edges, optionally lightened on dark set colors."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("edge_alpha must lie in [0, 1]")

    base_colors = np.asarray(
        [to_rgb(palette[int(label) % len(palette)]) for label in face_labels],
        dtype=np.float64,
    )
    edge_rgb = np.full_like(base_colors, 0.08)
    if light_edges_on_dark_faces:
        luminance = base_colors @ np.asarray([0.2126, 0.7152, 0.0722])
        dark = luminance < 0.48
        edge_rgb[dark] = 0.72 + 0.28 * base_colors[dark]
    return np.column_stack((edge_rgb, np.full(face_labels.shape[0], alpha)))


def plot_morse_sets_3d_cubical_from_csv(
    csv_path: str | Path,
    *,
    palette: Sequence[str] = PALETTE,
    paper_style: bool = True,
    elev: float = 22.0,
    azim: float = -55.0,
    alpha: float = 0.98,
    shade: bool = True,
    light_azdeg: float = 300.0,
    light_altdeg: float = 55.0,
    shade_strength: float = 0.32,
    highlight_strength: float = 0.12,
    edge_alpha: float = 0.16,
    edge_linewidth: float = 0.065,
    light_edges_on_dark_faces: bool = True,
    minimal_frame: bool = True,
    show_ticks: bool = True,
    show_axis_labels: bool = True,
    show_legend: bool = True,
    legend_labels: Mapping[int, str] | None = None,
    zlabel_pos: tuple[float, float] | None = None,
    rasterized: bool | None = None,
) -> MorseSetsPlot:
    """Render saved three-dimensional Morse boxes as exposed cubical cell faces.

    ``legend_labels`` optionally replaces the default ``M_i`` text for selected
    integer labels; omitted entries retain their default text.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    if not 0.0 <= shade_strength <= 1.0:
        raise ValueError("shade_strength must lie in [0, 1]")
    if not 0.0 <= highlight_strength <= 1.0:
        raise ValueError("highlight_strength must lie in [0, 1]")
    if not 0.0 <= edge_alpha <= 1.0:
        raise ValueError("edge_alpha must lie in [0, 1]")
    if edge_linewidth < 0.0:
        raise ValueError("edge_linewidth must be nonnegative")
    if paper_style:
        apply_paper_style()

    data, dim = _load_morse_sets(csv_path)
    if dim != 3:
        raise ValueError(f"cubical 3-D rendering requires dim=3 Morse boxes; got dim={dim}")
    return plot_morse_sets_3d_cubical(
        data,
        palette=palette,
        paper_style=False,  # already applied above
        elev=elev,
        azim=azim,
        alpha=alpha,
        shade=shade,
        light_azdeg=light_azdeg,
        light_altdeg=light_altdeg,
        shade_strength=shade_strength,
        highlight_strength=highlight_strength,
        edge_alpha=edge_alpha,
        edge_linewidth=edge_linewidth,
        light_edges_on_dark_faces=light_edges_on_dark_faces,
        minimal_frame=minimal_frame,
        show_ticks=show_ticks,
        show_axis_labels=show_axis_labels,
        show_legend=show_legend,
        zlabel_pos=zlabel_pos,
        rasterized=rasterized,
        legend_labels=legend_labels,
    )


def morse_boxes_from_graph(morse_graph: Any) -> NDArray[np.float64]:
    """Labelled box array ``[lo..., hi..., label]`` from a live CMGDB Morse graph.

    ``morse_set_boxes(node)`` yields the same rows ``SaveMorseSets`` writes, so
    a freshly computed decomposition can be plotted without a round trip
    through a CSV.
    """
    rows: list[list[float]] = []
    for node in morse_graph.vertices():
        node = int(node)
        for box in morse_graph.morse_set_boxes(node):
            rows.append([*(float(v) for v in box), float(node)])
    if not rows:
        raise ValueError("the Morse graph has no boxes to plot")
    return np.asarray(rows, dtype=np.float64)



def _resolve_box_scales_3d(
    data: NDArray[np.float64],
    box_scale: float | dict[int, float] | str,
    *,
    min_frac: float,
    max_scale: float,
) -> Callable[[int], float] | None:
    """``label -> display inflation`` for 3-D Morse sets, or ``None`` for none.

    Mirrors the 2-D :func:`_resolve_box_scales`: a float scales every set, a
    dict scales by label and is honoured exactly as given, and ``"auto"``
    enlarges only sets whose extent falls below ``min_frac`` of the domain,
    capped at ``max_scale``. Inflation is about each cell's own centre, so
    positions and relative geometry are unchanged -- only the drawn size.
    """
    if isinstance(box_scale, dict):
        per = {int(k): float(v) for k, v in box_scale.items()}
        return (lambda label: per.get(int(label), 1.0)) if per else None
    if isinstance(box_scale, str):
        if box_scale != "auto":
            raise ValueError(f"box_scale string must be 'auto', got {box_scale!r}")
        lower, upper, labels = data[:, :3], data[:, 3:6], data[:, 6].astype(int)
        span = max(
            float(upper[:, axis].max() - lower[:, axis].min()) for axis in range(3)
        )
        target = min_frac * span
        per = {}
        for label in np.unique(labels):
            mask = labels == label
            extent = max(
                float(upper[mask, axis].max() - lower[mask, axis].min())
                for axis in range(3)
            )
            if 0.0 < extent < target:
                per[int(label)] = min(target / extent, max_scale)
        return (lambda label: per.get(int(label), 1.0)) if per else None
    uniform = float(box_scale)
    if uniform == 1.0:
        return None
    return lambda label: uniform


def plot_morse_sets_3d_cubical(
    boxes: Any,
    *,
    palette: Sequence[str] = PALETTE,
    paper_style: bool = True,
    box_scale: float | dict[int, float] | str = 1.0,
    box_scale_min_frac: float = 0.025,
    box_scale_max: float = 10.0,
    elev: float = 22.0,
    azim: float = -55.0,
    alpha: float = 0.98,
    shade: bool = True,
    light_azdeg: float = 300.0,
    light_altdeg: float = 55.0,
    shade_strength: float = 0.32,
    highlight_strength: float = 0.12,
    edge_alpha: float = 0.16,
    edge_linewidth: float = 0.065,
    light_edges_on_dark_faces: bool = True,
    minimal_frame: bool = True,
    show_ticks: bool = True,
    show_axis_labels: bool = True,
    show_legend: bool = True,
    legend_labels: Mapping[int, str] | None = None,
    zlabel_pos: tuple[float, float] | None = None,
    rasterized: bool | None = None,
) -> MorseSetsPlot:
    """Render three-dimensional Morse boxes as exposed cubical cell faces.

    ``boxes`` is an ``(n, 7)`` array of ``[x_lo, y_lo, z_lo, x_hi, y_hi, z_hi,
    label]`` rows -- what :func:`morse_boxes_from_graph` returns and what
    ``SaveMorseSets`` writes. Only faces on the boundary of a labelled region
    are drawn, so the cost scales with the surface of the Morse sets rather
    than their volume; a solid block of cells renders as a shell.
    """
    data = np.asarray(boxes, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(
            "cubical 3-D rendering needs (n, 7) rows [lo x3, hi x3, label]; "
            f"got shape {data.shape}"
        )
    if paper_style:
        apply_paper_style()
    scale_of = _resolve_box_scales_3d(
        data, box_scale, min_frac=box_scale_min_frac, max_scale=box_scale_max
    )
    faces, face_labels = _exposed_cubical_faces(data, scale_of)
    facecolors = [palette[int(label) % len(palette)] for label in face_labels]
    if shade:
        facecolors = _subtly_shaded_cubical_facecolors(
            faces,
            face_labels,
            palette,
            light_azdeg=light_azdeg,
            light_altdeg=light_altdeg,
            strength=shade_strength,
            highlight_strength=highlight_strength,
        )
    edgecolors = _cubical_edgecolors(
        face_labels,
        palette,
        alpha=edge_alpha,
        light_edges_on_dark_faces=light_edges_on_dark_faces,
    )

    # Vector output is exact at any zoom and, once face-culling has run, is
    # usually the smaller file too: a 28k-box set reduces to ~16k faces, giving
    # a 313 KB vector PDF against 1.3 MB rasterized at 1200 dpi. That reverses
    # for a set dense enough to survive culling in bulk, so the choice is made
    # from the face count rather than fixed.
    if rasterized is None:
        rasterized = len(faces) > RASTERIZE_FACE_THRESHOLD
    fig = plt.figure(figsize=(6.14, 5.25), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        faces,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=edge_linewidth,
        alpha=alpha,
        # Rasterizing keeps the PDF small when a Morse set has 10^5+ faces, at
        # the cost of a resolution ceiling. Vector output is exact at any zoom
        # but grows with the face count.
        rasterized=rasterized,
        zsort="average",
        shade=False,
    )
    ax.add_collection3d(collection)

    lower = data[:, :3].min(axis=0)
    upper = data[:, 3:6].max(axis=0)
    span = upper - lower
    margin = np.maximum(0.035 * span, np.median(data[:, 3:6] - data[:, :3], axis=0))
    ax.set_xlim(float(lower[0] - margin[0]), float(upper[0] + margin[0]))
    ax.set_ylim(float(lower[1] - margin[1]), float(upper[1] + margin[1]))
    ax.set_zlim(float(lower[2] - margin[2]), float(upper[2] + margin[2]))
    ax.set_box_aspect(span)
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("$z_1$" if show_axis_labels else "", labelpad=5)
    ax.set_ylabel("$z_2$" if show_axis_labels else "", labelpad=5)
    ax.set_zlabel("$z_3$" if show_axis_labels else "", labelpad=5)
    for label in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        label.set_clip_on(False)
    # Matplotlib's projected z-axis label can be clipped even when the saved
    # bounding box is tight. Keep its semantic label on the axis and draw an
    # unclipped 2-D copy at a stable paper-layout position.
    ax.zaxis.label.set_visible(False)
    if show_axis_labels:
        # Default sits inside the tick numbers; ``zlabel_pos`` places it
        # clear of them, which matters once the z ticks carry three digits.
        default_pos = (0.95 if show_ticks else 0.90, 0.60 if show_ticks else 0.55)
        label_x, label_y = default_pos if zlabel_pos is None else zlabel_pos
        ax.text2D(
            label_x,
            label_y,
            "$z_3$",
            transform=ax.transAxes,
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="center",
            clip_on=False,
        )

    from matplotlib.ticker import MaxNLocator

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
        axis.set_tick_params(labelsize=8)
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor(
            (0.25, 0.25, 0.25, 0.0 if minimal_frame else 0.35)
        )
        axis.line.set_color((0.12, 0.12, 0.12, 0.72))
        if minimal_frame:
            axis.pane.set_visible(False)
        if not show_ticks:
            axis.set_ticks([])
    ax.grid(False)

    if show_legend:
        unique_labels = sorted(np.unique(data[:, 6].astype(int)).tolist())
        handles = [
            mpatches.Patch(
                facecolor=palette[label % len(palette)],
                edgecolor=(0.08, 0.08, 0.08, 0.25),
                label=(
                    legend_labels.get(label, f"$M_{{{label}}}$")
                    if legend_labels is not None
                    else f"$M_{{{label}}}$"
                ),
            )
            for label in unique_labels
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=6,
            frameon=False,
            handlelength=1.0,
            columnspacing=0.9,
        )

    return MorseSetsPlot(fig=fig, ax=ax, data=data, dim=3, label_to_y={})


def render_morse_sets_3d_cubical_from_csv(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    basename: str = "morse_sets_cubical",
    formats: Sequence[str] = ("pdf", "png"),
    palette: Sequence[str] = PALETTE,
    paper_style: bool = True,
    elev: float = 22.0,
    azim: float = -55.0,
    alpha: float = 0.98,
    shade: bool = True,
    light_azdeg: float = 300.0,
    light_altdeg: float = 55.0,
    shade_strength: float = 0.32,
    highlight_strength: float = 0.12,
    edge_alpha: float = 0.16,
    edge_linewidth: float = 0.065,
    light_edges_on_dark_faces: bool = True,
    minimal_frame: bool = True,
    show_ticks: bool = True,
    show_axis_labels: bool = True,
    show_legend: bool = True,
    zlabel_pos: tuple[float, float] | None = None,
    rasterized: bool | None = None,
    legend_labels: Mapping[int, str] | None = None,
) -> list[Path]:
    """Write a cubical three-dimensional Morse-set view in each requested format.

    ``zlabel_pos`` places the z-axis label in axes fractions, for the cases where
    the default sits under the tick numbers; ``None`` keeps the plotter's own
    placement.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot = plot_morse_sets_3d_cubical_from_csv(
        csv_path,
        palette=palette,
        paper_style=paper_style,
        elev=elev,
        azim=azim,
        alpha=alpha,
        shade=shade,
        light_azdeg=light_azdeg,
        light_altdeg=light_altdeg,
        shade_strength=shade_strength,
        highlight_strength=highlight_strength,
        edge_alpha=edge_alpha,
        edge_linewidth=edge_linewidth,
        light_edges_on_dark_faces=light_edges_on_dark_faces,
        minimal_frame=minimal_frame,
        show_ticks=show_ticks,
        show_axis_labels=show_axis_labels,
        show_legend=show_legend,
        zlabel_pos=zlabel_pos,
        rasterized=rasterized,
        legend_labels=legend_labels,
    )
    return save_figure(
        plot.fig,
        out / basename,
        formats=tuple(formats),
        close=True,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )


def _resolve_projection_pairs(
    pairs: Sequence[tuple[int, int]] | None,
    dim: int,
) -> list[tuple[int, int]]:
    if dim < 2:
        raise ValueError(f"coordinate-pair projections require dim >= 2, got dim={dim}")
    if pairs is None:
        return list(combinations(range(dim), 2))

    resolved: list[tuple[int, int]] = []
    for raw_pair in pairs:
        if len(raw_pair) != 2:
            raise ValueError(f"projection pair must contain exactly two indices, got {raw_pair!r}")
        pair = (int(raw_pair[0]), int(raw_pair[1]))
        if pair[0] == pair[1]:
            raise ValueError(f"projection pair must use distinct coordinates, got {pair}")
        if not all(0 <= coordinate < dim for coordinate in pair):
            raise ValueError(f"projection pair {pair} is out of range for dim={dim}")
        if pair in resolved:
            raise ValueError(f"duplicate projection pair: {pair}")
        resolved.append(pair)
    if not resolved:
        raise ValueError("at least one projection pair is required")
    return resolved


def _validate_projection_bounds(
    bounds_lower: Sequence[float] | None,
    bounds_upper: Sequence[float] | None,
    dim: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    if (bounds_lower is None) != (bounds_upper is None):
        raise ValueError("bounds_lower and bounds_upper must be provided together")
    if bounds_lower is None or bounds_upper is None:
        return None

    lower = np.asarray(bounds_lower, dtype=np.float64)
    upper = np.asarray(bounds_upper, dtype=np.float64)
    if lower.shape != (dim,) or upper.shape != (dim,):
        raise ValueError(
            f"projection bounds must each contain dim={dim} coordinates, "
            f"got shapes {lower.shape} and {upper.shape}"
        )
    if not np.all(lower < upper):
        raise ValueError("every lower projection bound must be less than its upper bound")
    return lower, upper


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
    box_scale_min_frac: float = 0.025,
    box_scale_max: float = 10.0,
    min_box_side_frac: float = 0.0,
    label_draw_order: Sequence[int] | None = None,
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
        box_scale_min_frac=box_scale_min_frac,
        box_scale_max=box_scale_max,
        min_box_side_frac=min_box_side_frac,
        label_draw_order=label_draw_order,
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

    label_to_y = dict.fromkeys(unique, 0.0)
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
    box_scale_min_frac: float = 0.025,
    box_scale_max: float = 10.0,
    min_box_side_frac: float = 0.0,
    label_draw_order: Sequence[int] | None = None,
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
    xlim, ylim = _adaptive_2d_morse_set_limits(lx, ly, ux, uy, bounds_lower, bounds_upper)
    if min_box_side_frac < 0.0:
        raise ValueError("min_box_side_frac must be nonnegative")
    min_width = min_box_side_frac * (xlim[1] - xlim[0])
    min_height = min_box_side_frac * (ylim[1] - ylim[0])

    # ``box_scale`` inflates each box about its own center so tiny attractor
    # sets stay visible. It may be a float (global), a {label: scale} dict
    # (per-set control), or "auto" (inflate only sets below a visibility floor;
    # ``box_scale_min_frac`` sets the floor, ``box_scale_max`` caps the factor).
    scale_for = _resolve_box_scales(
        box_scale, lx, ly, ux, uy, lbls, min_frac=box_scale_min_frac, max_scale=box_scale_max
    )
    if label_draw_order is None:
        draw_indices = np.arange(lbls.size)
    else:
        requested = [int(label) for label in label_draw_order]
        if len(requested) != len(set(requested)):
            raise ValueError("label_draw_order must not contain duplicate labels")
        present = set(np.unique(lbls).tolist())
        missing = sorted(present - set(requested))
        unknown = sorted(set(requested) - present)
        if missing or unknown:
            raise ValueError(
                "label_draw_order must contain every plotted label exactly once; "
                f"missing={missing}, unknown={unknown}"
            )
        draw_indices = np.concatenate(
            [np.flatnonzero(lbls == label) for label in requested]
        )

    rects = []
    facecolors = []
    for index in draw_indices:
        box_lx, box_ly, box_ux, box_uy, lbl = data[int(index)]
        lbl = int(lbl)
        s = scale_for(int(lbl))
        width = max((box_ux - box_lx) * s, min_width)
        height = max((box_uy - box_ly) * s, min_height)
        cx = 0.5 * (box_lx + box_ux)
        cy = 0.5 * (box_ly + box_uy)
        rects.append(mpatches.Rectangle((cx - 0.5 * width, cy - 0.5 * height), width, height))
        facecolors.append(palette[int(lbl) % len(palette)])
    # One collection rather than a patch per box, and vector unless the box count
    # would make the PDF unreasonable: a rasterized Morse-set layer cannot be
    # zoomed, which is exactly what these figures are read for.
    # Edge in each box's own face colour, as CMGDB.PlotMorseSets draws them:
    # closes the antialiasing seam between neighbouring boxes without
    # outlining any of them.
    ax.add_collection(
        PatchCollection(
            rects,
            facecolors=facecolors,
            edgecolors=facecolors,
            linewidths=0.4,
            rasterized=len(rects) > RASTERIZE_FACE_THRESHOLD,
        )
    )

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
    labels_2d: tuple[str, str] = ("$z_1$", "$z_2$"),
    paper_style: bool = True,
    box_scale: float | dict[int, float] | str = 1.0,
    box_scale_min_frac: float = 0.025,
    box_scale_max: float = 10.0,
    min_box_side_frac: float = 0.0,
    label_draw_order: Sequence[int] | None = None,
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
        labels_2d=labels_2d,
        paper_style=paper_style,
        box_scale=box_scale,
        box_scale_min_frac=box_scale_min_frac,
        box_scale_max=box_scale_max,
        min_box_side_frac=min_box_side_frac,
        label_draw_order=label_draw_order,
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
    for start, end in pairwise(traj):
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
    box_scale_min_frac: float = 0.025,
    box_scale_max: float = 10.0,
    min_box_side_frac: float = 0.0,
    basename: str = "morse_sets_with_overlay",
    formats: tuple[str, ...] = ("pdf", "png"),
    palette: Sequence[str] = PALETTE,
    labels_2d: tuple[str, str] = ("$z_1$", "$z_2$"),
    paper_style: bool = True,
    label_draw_order: Sequence[int] | None = None,
) -> list[Path]:
    """Render filled Morse sets with grey latent-orbit arrows overlaid.

    ``latent_starts`` are ``(N, 2)`` seed points already in latent space; each
    is advanced ``trajectory_steps`` times through ``advance_latent`` and drawn
    as a short grey arrow chain. ``box_scale`` inflates the Morse boxes so tiny
    attractor sets remain visible under the overlay. ``min_box_side_frac``
    applies the same display-only lower bound as :func:`plot_morse_sets_from_csv`.
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
        box_scale_min_frac=box_scale_min_frac,
        box_scale_max=box_scale_max,
        min_box_side_frac=min_box_side_frac,
        label_draw_order=label_draw_order,
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
