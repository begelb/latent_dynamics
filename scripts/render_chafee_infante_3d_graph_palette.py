"""Render the saved three-dimensional Morse sets with a hierarchy palette.

This is a render-only utility.  It computes each node's height above the two
minimal Morse sets and applies the same reference-style level colors to the
Morse graph, cubical Morse sets, and pairwise projections.  The persisted
Graphviz ``morse_graph`` file remains unchanged.  This utility does not load or
evaluate a neural network and does not run CMGDB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pydot
from matplotlib.colors import to_hex, to_rgba

from latentdynamics.viz import (
    plot_morse_set_projections_from_csv,
    render_morse_graph_from_dot,
    render_morse_sets_3d_cubical_from_csv,
)
from latentdynamics.viz.style import save_latent_figure

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    CODE_ROOT
    / "output"
    / "chafee_latent_dimension_study"
    / "latent_3d"
    / "seed_0"
    / "MG_adaptive"
)
DEFAULT_OUTPUT = (
    CODE_ROOT
    / "paper_figures"
    / "standardized"
    / "chafee_infante"
    / "latent_3d_level_palette"
)

ATTRACTOR_COLORS = {0: "#FFB000", 1: "#DC267F"}
COLOR_BY_POSITIVE_HEIGHT = {
    1: "#648FFF",
    2: "#FE6100",
    3: "#785EF0",
    4: "#008080",
    5: "#FCC2E8",
}
EXPECTED_HEIGHT_GROUPS = {
    0: frozenset({0, 1}),
    1: frozenset({2, 3}),
    2: frozenset({4, 7}),
    3: frozenset({5, 8}),
    4: frozenset({6, 9}),
    5: frozenset({10}),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_opaque_color(value: str) -> str:
    raw = value.strip().strip('"')
    rgba = to_rgba(raw)
    if not np.isclose(rgba[3], 1.0):
        raise ValueError(f"Morse-graph color must be opaque; got {value!r}")
    return to_hex(rgba, keep_alpha=False).upper()


def graph_palette_from_dot(
    dot_path: Path,
    *,
    expected_labels: frozenset[int] | None = None,
) -> tuple[str, ...]:
    """Return the contiguous node-indexed palette stored in ``dot_path``."""

    graphs = pydot.graph_from_dot_file(str(dot_path))
    if not graphs:
        raise ValueError(f"empty DOT file: {dot_path}")

    colors: dict[int, str] = {}
    for node in graphs[0].get_nodes():
        raw_name = node.get_name().strip('"')
        if not raw_name.lstrip("-").isdigit():
            continue
        label = int(raw_name)
        fillcolor = node.get("fillcolor")
        if fillcolor is None:
            raise ValueError(f"node {label} has no fillcolor in {dot_path}")
        colors[label] = _normalize_opaque_color(fillcolor)

    actual_labels = frozenset(colors)
    if expected_labels is not None and actual_labels != expected_labels:
        raise ValueError(
            "Morse-graph labels do not match the saved Morse sets: "
            f"graph={sorted(actual_labels)}, sets={sorted(expected_labels)}"
        )
    contiguous = frozenset(range(len(colors)))
    if actual_labels != contiguous:
        raise ValueError(
            f"Morse-graph labels must be contiguous from zero; got {sorted(colors)}"
        )
    return tuple(colors[label] for label in range(len(colors)))


def _numeric_dot_name(value: str) -> int | None:
    stripped = value.strip().strip('"')
    return int(stripped) if stripped.lstrip("-").isdigit() else None


def morse_heights_from_dot(dot_path: Path) -> dict[int, int]:
    """Return longest-path heights above the minimal nodes of a Morse DAG."""

    graphs = pydot.graph_from_dot_file(str(dot_path))
    if not graphs:
        raise ValueError(f"empty DOT file: {dot_path}")
    graph = graphs[0]

    labels = {
        label
        for node in graph.get_nodes()
        if (label := _numeric_dot_name(node.get_name())) is not None
    }
    successors: dict[int, set[int]] = {label: set() for label in labels}
    for edge in graph.get_edges():
        source = _numeric_dot_name(edge.get_source())
        target = _numeric_dot_name(edge.get_destination())
        if source is None or target is None:
            raise ValueError(f"non-numeric Morse edge in {dot_path}")
        if source not in successors or target not in successors:
            raise ValueError(
                f"Morse edge {source}->{target} references an undeclared node"
            )
        successors[source].add(target)

    heights: dict[int, int] = {}
    visiting: set[int] = set()

    def height(label: int) -> int:
        if label in heights:
            return heights[label]
        if label in visiting:
            raise ValueError(f"Morse graph contains a directed cycle at node {label}")
        visiting.add(label)
        children = successors[label]
        value = 0 if not children else 1 + max(height(child) for child in children)
        visiting.remove(label)
        heights[label] = value
        return value

    for label in sorted(labels):
        height(label)
    return heights


def chafee_level_palette_from_dot(
    dot_path: Path,
    *,
    expected_labels: frozenset[int] | None = None,
) -> tuple[str, ...]:
    """Return the reference-style palette keyed by Morse-graph height."""

    heights = morse_heights_from_dot(dot_path)
    actual_labels = frozenset(heights)
    if expected_labels is not None and actual_labels != expected_labels:
        raise ValueError(
            "Morse-graph labels do not match the saved Morse sets: "
            f"graph={sorted(actual_labels)}, sets={sorted(expected_labels)}"
        )
    if actual_labels != frozenset(range(len(actual_labels))):
        raise ValueError(
            f"Morse-graph labels must be contiguous from zero; got {sorted(actual_labels)}"
        )

    height_groups = {
        height: frozenset(
            label for label, actual_height in heights.items()
            if actual_height == height
        )
        for height in sorted(set(heights.values()))
    }
    if height_groups != EXPECTED_HEIGHT_GROUPS:
        raise ValueError(
            "saved d=3 Morse hierarchy differs from the audited reference structure: "
            f"{height_groups}"
        )

    colors: dict[int, str] = dict(ATTRACTOR_COLORS)
    for label, height in heights.items():
        if height > 0:
            colors[label] = COLOR_BY_POSITIVE_HEIGHT[height]
    return tuple(colors[label] for label in range(len(colors)))


def _validate_morse_sets(path: Path) -> tuple[frozenset[int], int]:
    data = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(
            f"three-dimensional morse_sets must have seven columns; got {data.shape}"
        )
    labels = data[:, 6]
    if not np.all(np.isfinite(data)) or not np.all(labels == np.rint(labels)):
        raise ValueError("morse_sets contains non-finite or non-integer labels")
    unique = frozenset(labels.astype(np.int64).tolist())
    if unique != frozenset(range(len(unique))):
        raise ValueError(f"morse_sets labels are not contiguous: {sorted(unique)}")
    return unique, int(data.shape[0])


def _output_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_bounds(path: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lower = tuple(float(value) for value in payload["lower"])
    upper = tuple(float(value) for value in payload["upper"])
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError(f"expected three-dimensional bounds in {path}")
    if any(left >= right for left, right in zip(lower, upper, strict=True)):
        raise ValueError(f"non-increasing bounds in {path}")
    return lower, upper


def _render_tick_free_projections(
    sets_path: Path,
    bounds_path: Path,
    output_dir: Path,
    palette: tuple[str, ...],
) -> list[Path]:
    lower, upper = _load_bounds(bounds_path)
    plots = plot_morse_set_projections_from_csv(
        sets_path,
        bounds_lower=lower,
        bounds_upper=upper,
        palette=palette,
        paper_style=True,
        min_box_side_frac=0.0025,
    )
    outputs: list[Path] = []
    for (left, right), plot in plots.items():
        plot.ax.set_xlabel("")
        plot.ax.set_ylabel("")
        plot.ax.set_xticks([])
        plot.ax.set_yticks([])
        plot.ax.grid(False)
        outputs.extend(
            save_latent_figure(
                plot.fig,
                output_dir
                / (
                    "ci_latent_3d_morse_sets_"
                    f"z{left + 1}_z{right + 1}_level_palette"
                ),
                formats=("pdf", "png"),
                close=True,
            )
        )
    return outputs


def render_level_palette_variants(
    source_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Render graph, shaded cubes, flat cubes, and projections by graph height."""

    graph_path = source_dir / "morse_graph"
    sets_path = source_dir / "morse_sets"
    if not graph_path.is_file() or not sets_path.is_file():
        raise FileNotFoundError(
            f"expected persisted morse_graph and morse_sets in {source_dir}"
        )

    set_labels, set_rows = _validate_morse_sets(sets_path)
    stored_palette = graph_palette_from_dot(
        graph_path,
        expected_labels=set_labels,
    )
    palette = chafee_level_palette_from_dot(
        graph_path,
        expected_labels=set_labels,
    )
    heights = morse_heights_from_dot(graph_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_outputs = render_morse_graph_from_dot(
        graph_path,
        output_dir,
        basename="ci_latent_3d_morse_graph_level_palette",
        formats=("pdf", "png"),
        palette=palette,
    )
    shaded_outputs = render_morse_sets_3d_cubical_from_csv(
        sets_path,
        output_dir,
        basename="ci_latent_3d_morse_sets_cubical_level_palette",
        formats=("pdf", "png"),
        palette=palette,
        elev=22.0,
        azim=-55.0,
        alpha=1.0,
        shade=True,
        shade_strength=0.28,
        highlight_strength=0.10,
        edge_alpha=0.16,
        edge_linewidth=0.065,
        minimal_frame=True,
        show_ticks=False,
        show_axis_labels=False,
        show_legend=True,
    )
    flat_outputs = render_morse_sets_3d_cubical_from_csv(
        sets_path,
        output_dir,
        basename="ci_latent_3d_morse_sets_cubical_level_palette_flat",
        formats=("pdf", "png"),
        palette=palette,
        elev=22.0,
        azim=-55.0,
        alpha=1.0,
        shade=False,
        edge_alpha=0.14,
        edge_linewidth=0.055,
        minimal_frame=True,
        show_ticks=False,
        show_axis_labels=False,
        show_legend=True,
    )
    projection_outputs = _render_tick_free_projections(
        sets_path,
        source_dir.parent / "bounds.json",
        output_dir,
        palette,
    )

    outputs = [
        *graph_outputs,
        *shaded_outputs,
        *flat_outputs,
        *projection_outputs,
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "reference-style graph-height colors shared by the saved 3-D Morse graph and Morse-set views",
        "render_only": True,
        "neural_network_evaluations": 0,
        "cmgdb_model_evaluations": 0,
        "source": {
            "directory": str(source_dir.resolve()),
            "morse_graph_sha256": _sha256(graph_path),
            "morse_sets_sha256": _sha256(sets_path),
            "morse_sets_rows": set_rows,
        },
        "palette_source": "height above the two minimal nodes in the persisted Morse DAG",
        "palette_by_node": {
            str(label): color for label, color in enumerate(palette)
        },
        "height_by_node": {
            str(label): height for label, height in sorted(heights.items())
        },
        "persisted_palette_by_node": {
            str(label): color for label, color in enumerate(stored_palette)
        },
        "display": {
            "ticks": False,
            "latent_coordinate_labels": False,
            "legend": True,
            "camera": {"elev": 22.0, "azim": -55.0},
            "shaded_variant": {
                "shade_strength": 0.28,
                "highlight_strength": 0.10,
            },
            "flat_variant": {
                "face_colors_are_literal_level_palette_colors": True,
            },
            "projection_minimum_display_side_fraction": 0.0025,
        },
        "outputs": {path.name: _output_record(path) for path in outputs},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    manifest = render_level_palette_variants(
        args.source_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
