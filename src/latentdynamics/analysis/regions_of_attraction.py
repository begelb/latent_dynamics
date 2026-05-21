"""Regions of attraction (ROA) from CMGDB Morse graph + Morse sets.

For each Morse node ``N``, compute the set of minimal Morse sets
forward-reachable from ``N`` in the Morse-graph DAG. Assign a single ROA label
per node:

- If exactly one minimal Morse set is reachable, the label is that node.
- If several minimal Morse sets are reachable, the label is the **least
  common ancestor** (LCA) of those nodes in the Morse-graph DAG — i.e.\\, the
  *lowest upper bound* in the Hasse partial order. By construction, ``N``
  itself is always such an ancestor, so the LCA defaults to ``N`` for
  unambiguous-but-non-minimal boundary nodes (saddles, sources).

Per-box labels in the CMGDB ``morse_sets`` CSV remain the recurrent Morse-node
ids. Transient regions of attraction are computed separately from the cell
graph; a recurrent Morse set is not assigned to the ROA of a lower Morse set.

This module operates purely on the saved CMGDB outputs, so it is O(M) for
``M`` nodes (typically <100) plus O(B) for ``B`` recorded boxes — fast enough
to run inline after every ``morse`` stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


_NODE_RE = re.compile(
    r'^\s*"?(\d+)"?\s*\[(?P<attrs>[^\]]*)\]\s*;?\s*$'
)
_EDGE_RE = re.compile(r'^\s*"?(\d+)"?\s*->\s*"?(\d+)"?\s*;?\s*$')
_COLOR_RE = re.compile(r'fillcolor\s*=\s*"(#[0-9A-Fa-f]+)"')
_LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')


@dataclass
class MorseGraph:
    """In-memory CMGDB Morse graph with derived ROA tables.

    All derived tables are precomputed at construction so that per-box
    lookups are constant time.
    """

    nodes: list[int]
    edges: dict[int, list[int]]
    colors: dict[int, str]
    labels: dict[int, str]
    minimal: set[int] = field(init=False)
    reachable_minimals: dict[int, frozenset[int]] = field(init=False)
    roa_label: dict[int, int] = field(init=False)
    descendants: dict[int, frozenset[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.minimal = {n for n in self.nodes if not self.edges.get(n)}
        self.descendants = self._precompute_descendants()
        self.reachable_minimals = {
            n: frozenset(d for d in self.descendants[n] if d in self.minimal)
            for n in self.nodes
        }
        self.roa_label = self._precompute_roa_labels()

    @classmethod
    def from_dot(cls, dot_path: str | Path) -> "MorseGraph":
        """Parse a CMGDB-emitted ``morse_graph`` DOT file."""
        text = Path(dot_path).read_text()
        nodes: list[int] = []
        edges: dict[int, list[int]] = {}
        colors: dict[int, str] = {}
        labels: dict[int, str] = {}
        for raw in text.splitlines():
            line = raw.strip()
            m = _NODE_RE.match(line)
            if m:
                node_id = int(m.group(1))
                nodes.append(node_id)
                attrs = m.group("attrs")
                cm = _COLOR_RE.search(attrs)
                if cm:
                    colors[node_id] = cm.group(1)
                lm = _LABEL_RE.search(attrs)
                if lm:
                    labels[node_id] = lm.group(1)
                continue
            m = _EDGE_RE.match(line)
            if m:
                src = int(m.group(1))
                dst = int(m.group(2))
                edges.setdefault(src, []).append(dst)
        # Sort node list for deterministic iteration.
        nodes = sorted(set(nodes))
        return cls(nodes=nodes, edges=edges, colors=colors, labels=labels)

    def _precompute_descendants(self) -> dict[int, frozenset[int]]:
        """``descendants[n]`` = set of all forward-reachable nodes (including ``n``)."""
        order = self._topological_order()
        out: dict[int, frozenset[int]] = {}
        # Process in reverse topo order so each node's descendants are known
        # by the time we hit it.
        for n in reversed(order):
            reached: set[int] = {n}
            for child in self.edges.get(n, []):
                reached |= out[child]
            out[n] = frozenset(reached)
        return out

    def _topological_order(self) -> list[int]:
        """Return nodes in topological order (sources first, sinks last).

        Tolerates the case where ``edges`` references unseen node ids: any
        edge target not in ``self.nodes`` is ignored.
        """
        indeg = {n: 0 for n in self.nodes}
        nodeset = set(self.nodes)
        for src, dsts in self.edges.items():
            if src not in nodeset:
                continue
            for d in dsts:
                if d in nodeset:
                    indeg[d] = indeg.get(d, 0) + 1
        ready = [n for n in self.nodes if indeg[n] == 0]
        order: list[int] = []
        while ready:
            n = ready.pop()
            order.append(n)
            for d in self.edges.get(n, []):
                if d not in nodeset:
                    continue
                indeg[d] -= 1
                if indeg[d] == 0:
                    ready.append(d)
        if len(order) != len(self.nodes):
            # Cycle detected — CMGDB Morse graphs are DAGs, so this would be
            # an upstream bug. Fall back to insertion order.
            return list(self.nodes)
        return order

    def _precompute_roa_labels(self) -> dict[int, int]:
        """Assign each node a single ROA label per the LCA rule."""
        out: dict[int, int] = {}
        for n in self.nodes:
            S = self.reachable_minimals[n]
            if len(S) == 0:
                # Defensive: node with no reachable minimals shouldn't exist
                # in a well-formed Morse graph (every node leads somewhere).
                out[n] = n
                continue
            if len(S) == 1:
                out[n] = next(iter(S))
                continue
            # |S| > 1 → ambiguous. Find the deepest descendant of n whose
            # reachable-minimal set equals S (i.e., still covers all minimal
            # Morse sets in S). N itself always qualifies; we look for a strictly deeper
            # candidate.
            best = n
            best_depth = -1  # depth from n in DAG; 0 = n itself
            stack: list[tuple[int, int]] = [(n, 0)]
            visited: set[int] = set()
            while stack:
                cur, depth = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                if self.reachable_minimals[cur] == S and depth > best_depth:
                    best = cur
                    best_depth = depth
                for d in self.edges.get(cur, []):
                    stack.append((d, depth + 1))
            out[n] = best
        return out

    def color_for(self, node_id: int) -> str | None:
        """Return the hex fillcolor of ``node_id`` (as written by CMGDB)."""
        return self.colors.get(node_id)

    def lca_of_minimals(self, S: frozenset[int]) -> int | None:
        """Smallest Morse node whose reachable minimals match (or contain) ``S``.

        Preference order:
        1. A node ``X`` with ``reachable_minimals(X) == S`` — if multiple, pick
           the one whose ``descendants`` are smallest (i.e., deepest in DAG).
        2. Otherwise the node with the smallest superset of ``S``.

        Returns ``None`` if ``S`` is empty (no minimal Morse set reachable).
        """
        if not S:
            return None
        exact = [n for n in self.nodes if self.reachable_minimals[n] == S]
        if exact:
            return min(exact, key=lambda n: len(self.descendants[n]))
        supers = [n for n in self.nodes if self.reachable_minimals[n] >= S]
        if not supers:
            return None
        return min(supers, key=lambda n: (len(self.reachable_minimals[n]), len(self.descendants[n])))


@dataclass
class BoxROATable:
    """Per-box ROA assignment, ready for visualization."""

    boxes: pd.DataFrame
    """Columns: ``lower_<i>``, ``upper_<i>`` for ``i`` in ``[0, dim)``,
    plus ``morse_node`` and ``roa_label``. One row per recurrent Morse-set box.
    For these recurrent boxes, ``roa_label`` preserves ``morse_node`` rather than
    assigning the box to a lower reachable Morse set."""

    morse_graph: MorseGraph
    dim: int

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)

    def boxes_for_label(self, label: int) -> pd.DataFrame:
        return self.boxes[self.boxes["roa_label"] == label]

    def label_summary(self) -> pd.DataFrame:
        """Box count and total area/volume per ROA label."""
        df = self.boxes
        widths = np.ones(len(df))
        for i in range(self.dim):
            widths *= (df[f"upper_{i}"].to_numpy() - df[f"lower_{i}"].to_numpy())
        out = (
            pd.DataFrame({"roa_label": df["roa_label"], "volume": widths})
            .groupby("roa_label", as_index=False)
            .agg(n_boxes=("volume", "size"), total_volume=("volume", "sum"))
        )
        out["is_minimal"] = out["roa_label"].apply(lambda n: n in self.morse_graph.minimal)
        return out.sort_values("roa_label").reset_index(drop=True)


def load_box_roa(
    morse_graph_dot: str | Path,
    morse_sets_csv: str | Path,
) -> BoxROATable:
    """Build the per-box ROA table from CMGDB outputs.

    The ``morse_sets`` CSV is the raw CMGDB layout: ``2*dim`` numeric columns
    followed by an integer Morse-node id. We infer ``dim`` from the column
    count.
    """
    mg = MorseGraph.from_dot(morse_graph_dot)
    raw = pd.read_csv(morse_sets_csv, header=None)
    ncols = raw.shape[1]
    if ncols < 3 or (ncols - 1) % 2 != 0:
        raise ValueError(
            f"unexpected morse_sets column count {ncols}; expected 2*dim + 1"
        )
    dim = (ncols - 1) // 2
    cols = []
    for i in range(dim):
        cols.append(f"lower_{i}")
    for i in range(dim):
        cols.append(f"upper_{i}")
    cols.append("morse_node")
    raw.columns = cols
    raw["morse_node"] = raw["morse_node"].astype(int)
    raw["roa_label"] = raw["morse_node"]
    return BoxROATable(boxes=raw, morse_graph=mg, dim=dim)
