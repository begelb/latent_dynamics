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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .morse_graph_parser import MorseGraph  # noqa: F401


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
