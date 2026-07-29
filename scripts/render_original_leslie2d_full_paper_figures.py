"""Render the full direct 2D Leslie graph and Morse sets in paper colors.

This is a render-only companion to ``compute_original_leslie.py``. It reads the
saved six-node CMGDB artifacts, verifies the expected Conley-index ordering,
and restores the trivial-index node without changing the semantic colors used
by the filtered figure in the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from latentdynamics.viz import render_morse_graph_from_dot, render_morse_sets_from_csv

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "leslie_2d_exact_restriction_s26_30_40_on_demand"
    / "MG"
)
DEFAULT_OUTPUT = DEFAULT_SOURCE.parent / "paper_full"

EXPECTED_INDICES = {
    0: "(x^6-1, 0, 0)",
    1: "(0, x^3+1, 0)",
    2: "(x-1, x-1, 0)",
    3: "(0, x^3-1, 0)",
    4: "(0, 0, 0)",
    5: "(0, 0, x-1)",
}

# Preserve the exact semantic correspondence used in the filtered paper view.
# The restored trivial-index node uses the otherwise-unused sixth paper color.
PAPER_NODE_COLORS = (
    "#DC267F",  # node 0: periodic attractor
    "#648FFF",  # node 1: index-matched latent node 2
    "#FFB000",  # node 2: invariant-circle attractor
    "#FE6100",  # node 3: index-matched latent node 3
    "#008080",  # node 4: restored trivial-index node
    "#785EF0",  # node 5: index-matched latent node 4
)

NODE_RE = re.compile(r'(?m)^\s*(\d+)\s+\[label="(\d+)\s*:\s*(\([^"]+\))"')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_graph(path: Path) -> None:
    found: dict[int, str] = {}
    for node_text, label_text, index in NODE_RE.findall(
        path.read_text(encoding="utf-8")
    ):
        node = int(node_text)
        label = int(label_text)
        if node != label:
            raise ValueError(f"node {node} has inconsistent displayed label {label}")
        found[node] = index
    if found != EXPECTED_INDICES:
        raise ValueError(
            "saved 2D Leslie graph does not have the expected indexed nodes: "
            f"expected {EXPECTED_INDICES}, found {found}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Directory containing the saved morse_graph and morse_sets files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for the full paper-colored PDF/PNG renders.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    graph_source = source_dir / "morse_graph"
    sets_source = source_dir / "morse_sets"
    for source in (graph_source, sets_source):
        if not source.is_file():
            raise FileNotFoundError(source)
    verify_graph(graph_source)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_outputs = render_morse_graph_from_dot(
        graph_source,
        output_dir,
        palette=PAPER_NODE_COLORS,
    )
    set_outputs = render_morse_sets_from_csv(
        sets_source,
        output_dir,
        bounds_lower=(0.0, 0.0),
        bounds_upper=(90.0, 70.0),
        palette=PAPER_NODE_COLORS,
        labels_2d=("$x_1$", "$x_2$"),
        min_box_side_frac=0.0025,
    )

    manifest = {
        "purpose": "full six-node paper-colored direct 2D Leslie view",
        "source": {
            "morse_graph": str(graph_source),
            "morse_graph_sha256": sha256(graph_source),
            "morse_sets": str(sets_source),
            "morse_sets_sha256": sha256(sets_source),
        },
        "expected_indices": {
            str(node): index for node, index in EXPECTED_INDICES.items()
        },
        "node_colors": {
            str(node): color for node, color in enumerate(PAPER_NODE_COLORS)
        },
        "rendering": {
            "bounds_lower": [0.0, 0.0],
            "bounds_upper": [90.0, 70.0],
            "min_box_side_frac": 0.0025,
        },
        "outputs": [str(path) for path in (*graph_outputs, *set_outputs)],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    for output in (*graph_outputs, *set_outputs):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
