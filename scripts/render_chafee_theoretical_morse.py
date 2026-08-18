"""Render the theoretical Chafee--Infante Morse representations.

Generator of the paper's Chafee--Infante Hasse-diagram panels: the full
eleven-equilibrium Morse representation at lambda = 28 and its three-set
bistable coarsening.  The figures imitate ``CMGDB.PlotMorseGraph``: Graphviz
performs the top-to-bottom layout, nodes are filled ellipses, and edges are
directed.  The colors follow the paper-wide palette and are keyed to
dynamical meaning.  Pure rendering of known theory; no neural network or
CMGDB computation is involved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from graphviz import Digraph
from latentdynamics.viz.style import (
    CHAFEE_CONNECTING_COLOR,
    CHAFEE_NEGATIVE_COLOR,
    CHAFEE_POSITIVE_COLOR,
    PALETTE,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = CODE_ROOT / "output" / "chafee_infante_theoretical_morse"


LEVEL_COLORS = {
    "1": PALETTE[2],
    "2": PALETTE[3],
    "3": PALETTE[4],
    "4": PALETTE[5],
    "5": PALETTE[6],
}


def morse_label(level: str, sign: str | None = None) -> str:
    """Return an HTML-like Graphviz label with an upright, text-style M."""
    if sign is None:
        return f"<M({level})>"
    return f"<M({level}<SUP>{sign}</SUP>)>"


def base_graph(name: str, *, nodesep: float, ranksep: float) -> Digraph:
    graph = Digraph(name=name, engine="dot")
    graph.attr(
        rankdir="TB",
        bgcolor="transparent",
        margin="0",
        pad="0.04",
        nodesep=str(nodesep),
        ranksep=str(ranksep),
        ordering="out",
    )
    graph.attr(
        "node",
        shape="ellipse",
        style="filled",
        color="black",
        fontcolor="black",
        fontname="Times-Roman",
        fontsize="14",
        penwidth="1",
        margin="0.11,0.055",
    )
    graph.attr(
        "edge",
        color="black",
        penwidth="1",
        arrowsize="0.8",
    )
    return graph


def same_rank(graph: Digraph, *nodes: str) -> None:
    """Keep a level aligned and preserve its left-to-right sign order."""
    with graph.subgraph() as rank:
        rank.attr(rank="same")
        for node in nodes:
            rank.node(node)
        for left, right in zip(nodes, nodes[1:]):
            rank.edge(left, right, style="invis", weight="100")


def full_representation() -> Digraph:
    """The eleven-equilibrium Morse representation at lambda = 28."""
    graph = base_graph("chafee_infante_full", nodesep=0.42, ranksep=0.52)

    graph.node("m5", label=morse_label("5"), fillcolor=LEVEL_COLORS["5"])
    levels = [
        ("4", LEVEL_COLORS["4"]),
        ("3", LEVEL_COLORS["3"]),
        ("2", LEVEL_COLORS["2"]),
        ("1", LEVEL_COLORS["1"]),
    ]
    for level, color in levels:
        graph.node(f"m{level}m", label=morse_label(level, "&#8722;"), fillcolor=color)
        graph.node(f"m{level}p", label=morse_label(level, "+"), fillcolor=color)
        same_rank(graph, f"m{level}m", f"m{level}p")
    graph.node(
        "m0m",
        label=morse_label("0", "&#8722;"),
        fillcolor=CHAFEE_NEGATIVE_COLOR,
    )
    graph.node(
        "m0p",
        label=morse_label("0", "+"),
        fillcolor=CHAFEE_POSITIVE_COLOR,
    )
    same_rank(graph, "m0m", "m0p")

    graph.edge("m5", "m4m")
    graph.edge("m5", "m4p")
    for upper, lower in zip(("4", "3", "2", "1"), ("3", "2", "1", "0")):
        for source in ("m", "p"):
            for target in ("m", "p"):
                graph.edge(f"m{upper}{source}", f"m{lower}{target}")

    return graph


def coarse_representation() -> Digraph:
    """The three-set bistable quotient used in the latent-space experiment."""
    graph = base_graph("chafee_infante_coarse", nodesep=0.48, ranksep=0.58)
    graph.node("m1", label=morse_label("1"), fillcolor=CHAFEE_CONNECTING_COLOR)
    graph.node(
        "m0m",
        label=morse_label("0", "&#8722;"),
        fillcolor=CHAFEE_NEGATIVE_COLOR,
    )
    graph.node(
        "m0p",
        label=morse_label("0", "+"),
        fillcolor=CHAFEE_POSITIVE_COLOR,
    )
    same_rank(graph, "m0m", "m0p")
    graph.edge("m1", "m0m")
    graph.edge("m1", "m0p")
    return graph


def render(graph: Digraph, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        graph.render(
            filename=stem,
            directory=output_dir,
            format=file_format,
            cleanup=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
        choices=("pdf", "png", "svg"),
    )
    args = parser.parse_args()

    render(
        full_representation(),
        args.output_dir,
        "ci_morse_representation_full",
        args.formats,
    )
    render(
        coarse_representation(),
        args.output_dir,
        "ci_morse_representation_coarse",
        args.formats,
    )


if __name__ == "__main__":
    main()
