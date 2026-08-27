"""Which single Morse node a cell reaches, if exactly one.

CMGDB (>= 1.5.0) provides this natively as ``MorseSingletonReachability``.
It is not the same question as
:mod:`latentdynamics.analysis.cmgdb_roa` answers: that module walks the cell
graph *backwards* from minimal Morse sets and collapses to a least common
ancestor, whereas this walks *forwards* from a query cell and reports the one
Morse node its forward orbit can reach -- or a sentinel when the answer is none
or more than one.

The port below mirrors the native C++ so results agree cell for cell; where
the two differ the native routine remains authoritative, and
:func:`morse_singleton_reachability` prefers it whenever it is present.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MULTIPLE_MORSE_NODES",
    "NO_MORSE_NODE",
    "morse_singleton_reachability",
    "morse_singleton_reachability_python",
]

#: No Morse node is reachable from the cell.
NO_MORSE_NODE = -1
#: More than one distinct Morse node is reachable.
MULTIPLE_MORSE_NODES = -2

_UNSEEN, _ACTIVE, _DONE = 0, 1, 2


def _merge(left: int, right: int) -> int:
    """Combine two reachability summaries."""
    if left == NO_MORSE_NODE:
        return right
    if right == NO_MORSE_NODE:
        return left
    if left == right:
        return left
    return MULTIPLE_MORSE_NODES


def morse_singleton_reachability_python(
    map_graph: Any,
    morse_graph: Any,
    query_cell_ids: NDArray[np.int64],
    *,
    require_cache: bool = True,
) -> NDArray[np.int32]:
    """Pure-Python equivalent of CMGDB's native ``MorseSingletonReachability``.

    Returns one int32 per query cell: the id of the unique Morse node reachable
    from it, :data:`NO_MORSE_NODE`, or :data:`MULTIPLE_MORSE_NODES`.

    A recurrent node counts as singleton-reachable exactly when it has no
    outgoing edge to a *distinct* Morse node; otherwise its reachable set holds
    itself and at least one other, and ``MULTIPLE`` is already the answer.
    Reachability is then propagated by a post-order depth-first walk, so each
    cell is summarised once and reused.
    """
    query = np.asarray(query_cell_ids, dtype=np.int64)
    if query.ndim != 1:
        raise ValueError("query_cell_ids must be one-dimensional")
    # The walk is memoised -- each cell is summarised once -- so an uncached
    # MapGraph costs one box-map evaluation per reachable cell, not one per
    # query. That is affordable when the callback is a table lookup and ruinous
    # when it evaluates a network, so the caller has to say which it has.
    if require_cache and not bool(map_graph.has_cache()):
        raise RuntimeError(
            "morse_singleton_reachability requires a cached MapGraph; refusing "
            "to use on-demand map callbacks, which would re-evaluate the box "
            "map once per adjacency query. Pass require_cache=False when the "
            "map callback is a precomputed lookup."
        )

    n_vertices = int(map_graph.num_vertices())
    if query.size and (int(query.min()) < 0 or int(query.max()) >= n_vertices):
        raise IndexError(
            f"query cell outside [0, {n_vertices}): "
            f"[{int(query.min())}, {int(query.max())}]"
        )

    n_nodes = int(morse_graph.num_vertices())
    # A Morse node summarises to itself unless it reaches another one.
    morse_summary = list(range(n_nodes))
    for source, target in morse_graph.edges_unreduced():
        source, target = int(source), int(target)
        if not (0 <= source < n_nodes and 0 <= target < n_nodes):
            raise ValueError(f"invalid Morse-graph edge ({source}, {target})")
        if source != target:
            morse_summary[source] = MULTIPLE_MORSE_NODES

    state = bytearray(n_vertices)                      # _UNSEEN everywhere
    reach = [NO_MORSE_NODE] * n_vertices
    for node in range(n_nodes):
        for cell in morse_graph.morse_set(node):
            cell = int(cell)
            if cell >= n_vertices:
                raise ValueError(f"Morse cell {cell} outside the MapGraph")
            state[cell] = _DONE
            reach[cell] = _merge(reach[cell], morse_summary[node])

    adjacencies = map_graph.adjacencies
    result = np.full(query.shape, NO_MORSE_NODE, dtype=np.int32)
    for index, raw_query in enumerate(query):
        start = int(raw_query)
        if state[start] == _UNSEEN:
            state[start] = _ACTIVE
            # (vertex, its successors, how many of them are folded in already)
            stack: list[list[Any]] = [[start, None, 0]]
            while stack:
                frame = stack[-1]
                vertex = frame[0]
                if frame[1] is None:
                    frame[1] = [int(v) for v in adjacencies(vertex)]
                successors = frame[1]
                if frame[2] == len(successors):
                    state[vertex] = _DONE
                    stack.pop()
                    continue
                successor = successors[frame[2]]
                if successor >= n_vertices:
                    raise ValueError(
                        f"adjacency {successor} outside the MapGraph"
                    )
                if state[successor] == _UNSEEN:
                    state[successor] = _ACTIVE
                    stack.append([successor, None, 0])
                    continue
                if state[successor] == _ACTIVE:
                    raise RuntimeError(
                        "found a directed cycle not covered by the supplied "
                        "Morse sets; every recurrent cell must belong to one"
                    )
                reach[vertex] = _merge(reach[vertex], reach[successor])
                frame[2] += 1
        result[index] = reach[start]
    return result


def morse_singleton_reachability(
    map_graph: Any,
    morse_graph: Any,
    query_cell_ids: NDArray[np.int64],
    *,
    require_cache: bool = True,
) -> NDArray[np.int32]:
    """Native ``CMGDB.MorseSingletonReachability`` when available, else the port."""
    import CMGDB

    native = getattr(CMGDB, "MorseSingletonReachability", None)
    if callable(native):
        result = native(map_graph, morse_graph, np.asarray(query_cell_ids, dtype=np.int64))
        return np.ascontiguousarray(result, dtype=np.int32)
    return morse_singleton_reachability_python(
        map_graph, morse_graph, query_cell_ids, require_cache=require_cache
    )
