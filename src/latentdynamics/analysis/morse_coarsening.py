"""Order-preserving quotients of computed Morse graphs.

The quotient construction in this module is deliberately separate from a
Conley-index computation. Merging graph nodes gives a quotient poset and a
surjective, order-preserving map. It does not, by itself, construct the Morse
sets of a coarser Morse representation: a genuine coarse set generally also
contains connecting invariant dynamics between its fine components. Likewise,
the Conley index of a merged fiber cannot in general be recovered from the node
annotations of the fine graph.
"""

from __future__ import annotations

__all__ = [
    "MorseGraphQuotient",
    "coarsen_morse_graph",
    "write_morse_graph_dot",
    "write_quotient_morse_sets",
]

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .morse_graph_parser import MorseGraph


@dataclass
class MorseGraphQuotient:
    """A Morse-graph quotient together with its fine-to-coarse epimorphism."""

    graph: MorseGraph
    projection: dict[int, int]
    fibers: dict[int, frozenset[int]]


def _topological_order(nodes: list[int], edges: Mapping[int, Iterable[int]]) -> list[int]:
    node_set = set(nodes)
    indegree = dict.fromkeys(nodes, 0)
    for source in nodes:
        for target in edges.get(source, ()):
            if target in node_set:
                indegree[target] += 1

    ready = [node for node in nodes if indegree[node] == 0]
    order: list[int] = []
    while ready:
        node = ready.pop()
        order.append(node)
        for target in edges.get(node, ()):
            if target not in node_set:
                continue
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(order) != len(nodes):
        raise ValueError(
            "the requested Morse-node merge creates a cycle in the quotient; "
            "its fibers do not define a poset epimorphism"
        )
    return order


def _transitive_reduction(
    nodes: list[int], edges: Mapping[int, Iterable[int]]
) -> dict[int, list[int]]:
    order = _topological_order(nodes, edges)
    closure: dict[int, set[int]] = {}
    for node in reversed(order):
        reached: set[int] = set()
        for target in edges.get(node, ()):
            reached.add(target)
            reached.update(closure[target])
        closure[node] = reached

    reduced: dict[int, list[int]] = {}
    for source in nodes:
        targets = set(edges.get(source, ()))
        covers = {
            target
            for target in targets
            if not any(target in closure[other] for other in targets if other != target)
        }
        if covers:
            reduced[source] = sorted(covers)
    return reduced


def coarsen_morse_graph(
    morse_graph: MorseGraph,
    merge_groups: Iterable[Iterable[int]],
    *,
    labels: Mapping[frozenset[int], str] | None = None,
) -> MorseGraphQuotient:
    """Collapse selected graph nodes and return the induced poset quotient.

    ``merge_groups`` contains disjoint fibers. Nodes omitted from the groups
    remain singleton fibers. The quotient order is induced by reachability in
    the fine graph and then transitively reduced. A merge that identifies the
    ends of an order interval while leaving an intermediate node outside the
    fiber can create a directed cycle; such a merge is rejected.

    The returned ``projection`` is surjective and order-preserving. Labels for
    singleton fibers retain their fine-graph annotation. Merged fibers receive
    a set label unless explicitly supplied; no Conley index is synthesized.
    This graph operation alone does not compute an invariant enclosure of the
    connecting dynamics required for the corresponding coarse Morse set.
    """
    node_set = set(morse_graph.nodes)
    supplied: list[frozenset[int]] = []
    used: set[int] = set()
    for raw_group in merge_groups:
        group = frozenset(int(node) for node in raw_group)
        if not group:
            raise ValueError("Morse-node merge groups must be nonempty")
        unknown = group - node_set
        if unknown:
            raise ValueError(f"unknown Morse nodes in merge group: {sorted(unknown)}")
        overlap = group & used
        if overlap:
            raise ValueError(f"Morse nodes occur in more than one merge group: {sorted(overlap)}")
        supplied.append(group)
        used.update(group)

    fibers_in_order = supplied + [
        frozenset({node}) for node in morse_graph.nodes if node not in used
    ]
    fibers_in_order.sort(key=lambda group: (min(group), len(group), sorted(group)))
    fibers = dict(enumerate(fibers_in_order))
    projection = {fine: coarse for coarse, group in fibers.items() for fine in group}

    # Use fine-graph reachability, not only displayed Hasse edges, so the
    # quotient is independent of whether the input DOT is transitively reduced.
    quotient_order: dict[int, set[int]] = {coarse: set() for coarse in fibers}
    for fine_source in morse_graph.nodes:
        coarse_source = projection[fine_source]
        for fine_target in morse_graph.descendants[fine_source]:
            coarse_target = projection[fine_target]
            if coarse_target != coarse_source:
                quotient_order[coarse_source].add(coarse_target)

    quotient_nodes = sorted(fibers)
    quotient_edges = _transitive_reduction(quotient_nodes, quotient_order)
    custom_labels = labels or {}
    quotient_labels: dict[int, str] = {}
    quotient_colors: dict[int, str] = {}
    for coarse, group in fibers.items():
        if group in custom_labels:
            quotient_labels[coarse] = custom_labels[group]
        elif len(group) == 1:
            fine = next(iter(group))
            quotient_labels[coarse] = morse_graph.labels.get(fine, str(fine))
        else:
            members = ", ".join(str(node) for node in sorted(group))
            quotient_labels[coarse] = f"M_{{{members}}}"
        if len(group) == 1:
            fine = next(iter(group))
            if fine in morse_graph.colors:
                quotient_colors[coarse] = morse_graph.colors[fine]

    quotient_graph = MorseGraph(
        nodes=quotient_nodes,
        edges=quotient_edges,
        colors=quotient_colors,
        labels=quotient_labels,
    )
    return MorseGraphQuotient(
        graph=quotient_graph,
        projection=projection,
        fibers=fibers,
    )


def write_morse_graph_dot(morse_graph: MorseGraph, path: str | Path) -> Path:
    """Write a parsed or quotient Morse graph in CMGDB-compatible DOT form."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["digraph {"]
    for node in morse_graph.nodes:
        label = morse_graph.labels.get(node, str(node))
        color = morse_graph.colors.get(node, "#808080ff")
        lines.append(
            f'{node} [label={json.dumps(label)}, shape=ellipse, style=filled, fillcolor="{color}"];'
        )
    for source in morse_graph.nodes:
        for target in morse_graph.edges.get(source, ()):
            lines.append(f"{source} -> {target};")
    lines.append("}")
    output.write_text("\n".join(lines) + "\n")
    return output


def write_quotient_morse_sets(
    source: str | Path,
    destination: str | Path,
    projection: Mapping[int, int],
) -> Path:
    """Relabel a CMGDB ``morse_sets`` CSV by a Morse-poset projection.

    Every geometric box is retained. Thus a quotient fiber is visualized by
    the literal union of the fine Morse-set boxes assigned to it. The output
    does not add boxes enclosing connecting dynamics and should not be treated
    as a newly computed coarse Morse set.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open(newline="") as src, destination_path.open("w", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, lineterminator="\n")
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            fine_label = int(float(row[-1]))
            if fine_label not in projection:
                raise ValueError(
                    f"morse_sets row {row_number} uses label {fine_label}, "
                    "which is absent from the quotient projection"
                )
            row[-1] = str(projection[fine_label])
            writer.writerow(row)
    return destination_path
