"""Render the paper-style Morse graph for the original Leslie 3D ground truth.

The saved CMGDB screen supplies the graph topology. The Conley-index labels
below are the indices computed for the corresponding six Morse sets. The
zero-index node is omitted from the displayed graph, and the remaining colors
match the corresponding Conley indices in the latent Leslie 3D figures.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "ground_truth"
    / "absorbing_B_i29_m33_M36_L10000"
)
DEFAULT_SOURCE = RUN_ROOT / "screen" / "morse_graph"
DEFAULT_INDICES = RUN_ROOT / "saved_set_conley" / "summary.json"
DEFAULT_OUTPUT = RUN_ROOT / "paper_figure"

EXPECTED_NODES = tuple(range(6))
EXPECTED_EDGES = (
    (2, 1),
    (3, 0),
    (3, 2),
    (4, 2),
    (5, 3),
    (5, 4),
)
DISPLAYED_NODES = (0, 1, 2, 3, 4)
DISPLAYED_EDGES = tuple(
    (source, target)
    for source, target in EXPECTED_EDGES
    if source in DISPLAYED_NODES and target in DISPLAYED_NODES
)
COLORS = {
    0: "#FFB000FF",
    1: "#DC267FFF",
    2: "#FE6100FF",
    3: "#648FFFFF",
    4: "#785EF0FF",
}


def parse_topology(source: str) -> tuple[set[int], set[tuple[int, int]]]:
    """Extract numeric nodes and directed edges from a CMGDB DOT source."""
    nodes = {
        int(match.group(1))
        for match in re.finditer(r"(?m)^\s*(\d+)\s+\[", source)
    }
    edges = {
        (int(match.group(1)), int(match.group(2)))
        for match in re.finditer(r"(?m)^\s*(\d+)\s*->\s*(\d+)\s*;", source)
    }
    return nodes, edges


def load_indices(path: Path) -> dict[int, str]:
    """Load complete four-entry Conley indices from saved-set postprocessing."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"saved-set Conley computation is not complete: {path}")
    raw_indices = summary.get("conley_indices")
    if not isinstance(raw_indices, dict):
        raise ValueError(f"missing conley_indices object in {path}")

    indices: dict[int, str] = {}
    for raw_node, raw_index in raw_indices.items():
        node = int(raw_node)
        if (
            not isinstance(raw_index, list)
            or len(raw_index) != 4
            or not all(isinstance(entry, str) for entry in raw_index)
        ):
            raise ValueError(f"invalid Conley index for node {node}: {raw_index}")
        indices[node] = f"({', '.join(raw_index)})"
    if set(indices) != set(EXPECTED_NODES):
        raise ValueError(
            f"unexpected indexed nodes in {path}: "
            f"expected {list(EXPECTED_NODES)}, found {sorted(indices)}"
        )
    return indices


def paper_dot(conley_indices: dict[int, str]) -> str:
    """Return the paper-style Graphviz source."""
    if conley_indices[5] != "(0, 0, 0, 0)":
        raise ValueError(
            "node 5 can only be omitted when its Conley index is "
            f"(0, 0, 0, 0), found {conley_indices[5]}"
        )
    lines = ["digraph G {"]
    for node in DISPLAYED_NODES:
        lines.append(
            f'{node} [label="{node} : {conley_indices[node]}", '
            f"shape=ellipse, style=filled, fillcolor=\"{COLORS[node]}\", "
            'margin="0.11, 0.055"];'
        )
    lines.extend(
        [
            "subgraph {",
            "rank=same;",
            "0;",
            "1;",
            "}",
            "subgraph {",
            "rank=same;",
            "3;",
            "4;",
            "}",
        ]
    )
    lines.extend(f"{source} -> {target};" for source, target in DISPLAYED_EDGES)
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Saved CMGDB Morse-graph DOT source used to verify the topology.",
    )
    parser.add_argument(
        "--indices",
        type=Path,
        default=DEFAULT_INDICES,
        help="Saved-set Conley summary supplying the node annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for morse_graph, morse_graph.pdf, and morse_graph.png.",
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    nodes, edges = parse_topology(source_path.read_text(encoding="utf-8"))
    if nodes != set(EXPECTED_NODES):
        raise ValueError(
            f"unexpected Morse nodes in {source_path}: "
            f"expected {list(EXPECTED_NODES)}, found {sorted(nodes)}"
        )
    if edges != set(EXPECTED_EDGES):
        raise ValueError(
            f"unexpected Morse edges in {source_path}: "
            f"expected {list(EXPECTED_EDGES)}, found {sorted(edges)}"
        )
    indices_path = args.indices.resolve()
    if not indices_path.is_file():
        raise FileNotFoundError(indices_path)
    conley_indices = load_indices(indices_path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dot_path = output_dir / "morse_graph"
    dot_path.write_text(paper_dot(conley_indices), encoding="utf-8")

    for output_format in ("pdf", "png"):
        output_path = output_dir / f"morse_graph.{output_format}"
        command = ["dot", f"-T{output_format}"]
        if output_format == "png":
            command.append("-Gdpi=300")
        subprocess.run(
            [*command, str(dot_path), "-o", str(output_path)],
            check=True,
        )
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
