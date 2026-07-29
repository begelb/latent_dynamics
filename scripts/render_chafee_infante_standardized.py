"""Render standardized Chafee--Infante figures from persisted DOT/CSV files.

This script is deliberately render-only.  It never loads a neural network and
never constructs or evaluates a CMGDB model.  The two stable physical states
are colored by meaning rather than by the local CMGDB node numbering:

* negative stable state: repository yellow;
* positive stable state: repository magenta;
* explicitly coarsened unstable/connecting class: neutral gray.

The one-dimensional computation is a fine result, so its unstable node retains
the repository's normal third palette color.  Gray is reserved for the
explicitly coarsened two-dimensional connecting class.

Marcio's exact adaptive two-dimensional fine computation is currently
preserved as PDFs but not as a DOT/CSV pair.  Consequently it is omitted by
default.  ``--d2-fine-dir`` may be used once an exact persisted snapshot is
available; strict graph, annotation, label, and cell-count checks prevent a
different replay or uniform-grid computation from being substituted silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latentdynamics.analysis.morse_graph_parser import MorseGraph
from latentdynamics.viz import (
    chafee_semantic_palette,
    plot_morse_sets_from_csv,
    render_morse_graph_from_dot,
)
from latentdynamics.viz.style import save_latent_figure

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent

DEFAULT_D1_DIR = (
    CODE_ROOT
    / "output"
    / "chafee_latent_dimension_study"
    / "latent_1d"
    / "seed_0"
    / "MG_adaptive"
)
DEFAULT_D1_BOUNDS = DEFAULT_D1_DIR.parent / "bounds.json"
DEFAULT_D2_COARSE_DIR = (
    CODE_ROOT / "paper_figures" / "coarsened" / "chafee_infante" / "MG"
)
DEFAULT_D2_MANIFEST = DEFAULT_D2_COARSE_DIR.parent / "quotient.json"
DEFAULT_OUTPUT = (
    CODE_ROOT / "paper_figures" / "standardized" / "chafee_infante"
)

MARCIO_D2_FINE_EDGES = frozenset(
    {
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (4, 2),
        (4, 3),
        (5, 4),
        (6, 5),
    }
)
MARCIO_D2_FINE_ANNOTATIONS = {
    0: "(x-1, 0, 0)",
    1: "(x-1, 0, 0)",
    2: "(0, x-1, 0)",
    3: "(0, x-1, 0)",
    4: "(0, x-1, 0)",
    5: "(0, 0, x-1)",
    6: "(0, 0, x-1)",
}


@dataclass(frozen=True)
class RenderSpec:
    """One persisted Morse computation and its display semantics."""

    key: str
    source_dir: Path
    dimension: int
    negative_label: int
    positive_label: int
    connecting_labels: tuple[int, ...]
    expected_nodes: frozenset[int]
    expected_edges: frozenset[tuple[int, int]]
    expected_annotations: dict[int, str]
    expected_rows: int
    expected_set_labels: frozenset[int]
    graph_basename: str
    sets_basename: str
    bounds_lower: tuple[float, ...]
    bounds_upper: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _edges(graph: MorseGraph) -> frozenset[tuple[int, int]]:
    return frozenset(
        (int(source), int(target))
        for source, targets in graph.edges.items()
        for target in targets
    )


def _annotation(label: str) -> str:
    """Return only the Conley-index tuple from a saved graph label."""

    _, separator, suffix = label.partition(":")
    return suffix.strip() if separator else label.strip()


def _load_bounds(path: Path, *, nested: tuple[str, ...] = ()) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    for key in nested:
        if not isinstance(payload, dict) or key not in payload:
            raise ValueError(f"{path} does not contain nested bounds key {nested!r}")
        payload = payload[key]
    if not isinstance(payload, dict):
        raise ValueError(f"{path} bounds payload is not a JSON object")
    lower = tuple(float(value) for value in payload.get("lower", ()))
    upper = tuple(float(value) for value in payload.get("upper", ()))
    if not lower or len(lower) != len(upper):
        raise ValueError(f"{path} contains malformed lower/upper bounds")
    if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
        raise ValueError(f"{path} contains non-increasing bounds")
    return lower, upper


def _validate_spec(spec: RenderSpec) -> tuple[MorseGraph, np.ndarray]:
    dot_path = spec.source_dir / "morse_graph"
    sets_path = spec.source_dir / "morse_sets"
    if not dot_path.is_file() or not sets_path.is_file():
        raise FileNotFoundError(
            f"{spec.key} requires persisted morse_graph and morse_sets in "
            f"{spec.source_dir}"
        )

    graph = MorseGraph.from_dot(dot_path)
    actual_nodes = frozenset(graph.nodes)
    actual_edges = _edges(graph)
    if actual_nodes != spec.expected_nodes:
        raise ValueError(
            f"{spec.key} nodes do not match the expected persisted computation: "
            f"expected={sorted(spec.expected_nodes)}, actual={sorted(actual_nodes)}"
        )
    if actual_edges != spec.expected_edges:
        missing = sorted(spec.expected_edges - actual_edges)
        extra = sorted(actual_edges - spec.expected_edges)
        raise ValueError(
            f"{spec.key} edges do not match the expected persisted computation: "
            f"missing={missing}, extra={extra}"
        )

    actual_annotations = {
        node: _annotation(graph.labels.get(node, ""))
        for node in sorted(spec.expected_nodes)
    }
    if actual_annotations != spec.expected_annotations:
        differences = {
            node: {
                "expected": spec.expected_annotations[node],
                "actual": actual_annotations[node],
            }
            for node in sorted(spec.expected_nodes)
            if actual_annotations[node] != spec.expected_annotations[node]
        }
        raise ValueError(
            f"{spec.key} graph annotations do not match the expected computation: "
            f"{differences}"
        )

    data = np.loadtxt(sets_path, delimiter=",", ndmin=2)
    expected_columns = 2 * spec.dimension + 1
    if data.ndim != 2 or data.shape[1] != expected_columns:
        raise ValueError(
            f"{spec.key} morse_sets must have {expected_columns} columns; "
            f"got shape {data.shape}"
        )
    if data.shape[0] != spec.expected_rows:
        raise ValueError(
            f"{spec.key} morse_sets row count does not match the expected "
            f"computation: expected={spec.expected_rows}, actual={data.shape[0]}"
        )
    labels = data[:, -1]
    if not np.all(np.isfinite(data)) or not np.all(labels == np.rint(labels)):
        raise ValueError(f"{spec.key} morse_sets contains non-finite or non-integer labels")
    actual_labels = frozenset(labels.astype(int).tolist())
    if actual_labels != spec.expected_set_labels:
        raise ValueError(
            f"{spec.key} Morse-set labels do not match: "
            f"expected={sorted(spec.expected_set_labels)}, "
            f"actual={sorted(actual_labels)}"
        )
    return graph, data


def _hide_axis_annotations(plot) -> None:
    """Remove coordinate names and ticks while preserving Morse-set labels."""

    plot.ax.set_xlabel("")
    plot.ax.set_ylabel("")
    plot.ax.set_xticks([])
    plot.ax.set_yticks([])


def _render_spec(
    spec: RenderSpec,
    output_dir: Path,
    *,
    show_axis_annotations: bool,
) -> dict[str, Any]:
    graph, data = _validate_spec(spec)
    palette = chafee_semantic_palette(
        len(graph.nodes),
        negative_label=spec.negative_label,
        positive_label=spec.positive_label,
        connecting_labels=spec.connecting_labels,
    )

    graph_outputs = render_morse_graph_from_dot(
        spec.source_dir / "morse_graph",
        output_dir,
        basename=spec.graph_basename,
        formats=("pdf", "png"),
        palette=palette,
    )
    plot = plot_morse_sets_from_csv(
        spec.source_dir / "morse_sets",
        bounds_lower=spec.bounds_lower,
        bounds_upper=spec.bounds_upper,
        palette=palette,
        paper_style=True,
        box_scale=1.0,
        min_box_side_frac=0.0,
    )
    if not show_axis_annotations:
        _hide_axis_annotations(plot)
    set_outputs = save_latent_figure(
        plot.fig,
        output_dir / spec.sets_basename,
        formats=("pdf", "png"),
        close=True,
    )

    outputs = [*graph_outputs, *set_outputs]
    return {
        "source": {
            "directory": _repo_path(spec.source_dir),
            "morse_graph_sha256": _sha256(spec.source_dir / "morse_graph"),
            "morse_sets_sha256": _sha256(spec.source_dir / "morse_sets"),
            "morse_sets_rows": int(data.shape[0]),
        },
        "node_semantics": {
            "negative": spec.negative_label,
            "positive": spec.positive_label,
            "unstable_or_connecting": list(spec.connecting_labels),
        },
        "palette_by_node": {
            str(node): palette[node]
            for node in sorted(graph.nodes)
        },
        "geometry": {
            "box_scale": 1.0,
            "minimum_display_side_fraction": 0.0,
            "coordinate_bounds_lower": list(spec.bounds_lower),
            "coordinate_bounds_upper": list(spec.bounds_upper),
        },
        "graph_labels": {
            str(node): graph.labels[node]
            for node in sorted(graph.nodes)
        },
        "outputs": {
            path.name: {
                "path": _repo_path(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        },
    }


def _build_specs(
    *,
    d1_dir: Path,
    d1_bounds_path: Path,
    d2_coarse_dir: Path,
    d2_manifest_path: Path,
    d2_fine_dir: Path | None,
) -> list[RenderSpec]:
    d1_lower, d1_upper = _load_bounds(d1_bounds_path)
    d2_lower, d2_upper = _load_bounds(
        d2_manifest_path,
        nested=("computation", "bounds"),
    )

    specs = [
        RenderSpec(
            key="latent_1d",
            source_dir=d1_dir,
            dimension=1,
            negative_label=0,
            positive_label=1,
            connecting_labels=(),
            expected_nodes=frozenset({0, 1, 2}),
            expected_edges=frozenset({(2, 0), (2, 1)}),
            expected_annotations={
                0: "(x-1, 0)",
                1: "(x-1, 0)",
                2: "(0, x-1)",
            },
            expected_rows=85,
            expected_set_labels=frozenset({0, 1, 2}),
            graph_basename="ci_latent_1d_morse_graph",
            sets_basename="ci_latent_1d_morse_sets",
            bounds_lower=d1_lower,
            bounds_upper=d1_upper,
        ),
        RenderSpec(
            key="latent_2d_coarse",
            source_dir=d2_coarse_dir,
            dimension=2,
            negative_label=1,
            positive_label=0,
            connecting_labels=(2,),
            expected_nodes=frozenset({0, 1, 2}),
            expected_edges=frozenset({(2, 0), (2, 1)}),
            expected_annotations={
                0: "M(0⁺)",
                1: "M(0⁻)",
                2: "M(1)",
            },
            expected_rows=4235,
            expected_set_labels=frozenset({0, 1, 2}),
            graph_basename="ci_coarse_morse_graph",
            sets_basename="ci_coarse_morse_sets",
            bounds_lower=d2_lower,
            bounds_upper=d2_upper,
        ),
    ]
    if d2_fine_dir is not None:
        specs.insert(
            1,
            RenderSpec(
                key="latent_2d_fine",
                source_dir=d2_fine_dir,
                dimension=2,
                negative_label=1,
                positive_label=0,
                connecting_labels=(),
                expected_nodes=frozenset(range(7)),
                expected_edges=MARCIO_D2_FINE_EDGES,
                expected_annotations=MARCIO_D2_FINE_ANNOTATIONS,
                expected_rows=1533,
                expected_set_labels=frozenset(range(7)),
                graph_basename="ci_morse_graph",
                sets_basename="ci_morse_sets",
                bounds_lower=d2_lower,
                bounds_upper=d2_upper,
            ),
        )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--d1-bounds", type=Path, default=DEFAULT_D1_BOUNDS)
    parser.add_argument("--d2-coarse-dir", type=Path, default=DEFAULT_D2_COARSE_DIR)
    parser.add_argument("--d2-manifest", type=Path, default=DEFAULT_D2_MANIFEST)
    parser.add_argument(
        "--d2-fine-dir",
        type=Path,
        help=(
            "exact persisted Marcio adaptive fine DOT/CSV directory; omitted by "
            "default because no exact snapshot is currently stored"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--show-axis-annotations",
        action="store_true",
        help="retain latent-coordinate labels and ticks (hidden by default)",
    )
    args = parser.parse_args()

    specs = _build_specs(
        d1_dir=args.d1_dir,
        d1_bounds_path=args.d1_bounds,
        d2_coarse_dir=args.d2_coarse_dir,
        d2_manifest_path=args.d2_manifest,
        d2_fine_dir=args.d2_fine_dir,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    rendered = {
        spec.key: _render_spec(
            spec,
            args.output,
            show_axis_annotations=args.show_axis_annotations,
        )
        for spec in specs
    }
    manifest = {
        "schema_version": 1,
        "purpose": "standardized Chafee--Infante color and axis staging",
        "render_only": True,
        "neural_network_evaluations": 0,
        "cmgdb_model_evaluations": 0,
        "axis_annotations_visible": bool(args.show_axis_annotations),
        "semantic_colors": {
            "negative_stable": chafee_semantic_palette(
                3,
                negative_label=0,
                positive_label=1,
            )[0],
            "positive_stable": chafee_semantic_palette(
                3,
                negative_label=0,
                positive_label=1,
            )[1],
            "explicit_unstable_or_connecting": chafee_semantic_palette(
                3,
                negative_label=0,
                positive_label=1,
                connecting_labels=(2,),
            )[2],
        },
        "computations": rendered,
        "latent_2d_fine": (
            {"status": "rendered", "source": _repo_path(args.d2_fine_dir)}
            if args.d2_fine_dir is not None
            else {
                "status": "omitted",
                "reason": (
                    "Marcio's exact adaptive fine computation is preserved only "
                    "as PDFs; no exact persisted DOT/CSV snapshot is currently "
                    "available. The saved package replay is scientifically "
                    "different and is rejected by the source checks."
                ),
            }
        ),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
