"""Order-preserving quotients of computed Morse graphs.

Merging graph nodes gives a quotient poset and a surjective, order-preserving
map. Given the underlying directed cell graph as well, this module can augment
each quotient fiber by the cells on connections between fine Morse components
in that fiber. The resulting combinatorial enclosure is the intersection of
the fiber's forward-reachable downset and reverse-reachable upset.

This construction is deliberately separate from a Conley-index computation.
The Conley index of a merged fiber cannot in general be recovered from the node
annotations of the fine graph.
"""

from __future__ import annotations

__all__ = [
    "ConnectionCompleteMorseSets",
    "MorseGraphQuotient",
    "coarsen_morse_graph",
    "compute_connection_complete_morse_sets",
    "compute_uniform_connection_complete_morse_sets",
    "uniform_cell_boxes",
    "uniform_grid_shape",
    "write_connection_complete_morse_sets",
    "write_morse_graph_dot",
    "write_quotient_morse_sets",
]

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .morse_graph_parser import MorseGraph


@dataclass
class MorseGraphQuotient:
    """A Morse-graph quotient together with its fine-to-coarse epimorphism."""

    graph: MorseGraph
    projection: dict[int, int]
    fibers: dict[int, frozenset[int]]


@dataclass(frozen=True)
class ConnectionCompleteMorseSets:
    """Cell-level enclosures for the Morse sets induced by a quotient.

    ``cells[q]`` contains the original fine Morse-set cells in the fiber over
    ``q`` together with every cell on a directed path between two such cells.
    ``connection_cells[q]`` contains only the cells added by that path
    completion. ``overlaps`` records cells enclosed by more than one coarse
    set; these can occur because a multivalued cell graph is an outer
    approximation even though genuine Morse sets are disjoint.
    """

    projection: dict[int, int]
    fibers: dict[int, frozenset[int]]
    cells: dict[int, np.ndarray]
    connection_cells: dict[int, np.ndarray]
    overlaps: dict[int, frozenset[int]]


def _forward_reachable(map_graph, targets: np.ndarray, n_vertices: int) -> np.ndarray:
    visited = np.zeros(n_vertices, dtype=bool)
    stack = [int(cell) for cell in targets]
    visited[targets] = True
    while stack:
        source = stack.pop()
        for raw_target in map_graph.adjacencies(source):
            target = int(raw_target)
            if target < 0 or target >= n_vertices:
                raise ValueError(
                    f"directed cell graph has edge {source} -> {target}, "
                    f"outside its vertex range [0, {n_vertices})"
                )
            if not visited[target]:
                visited[target] = True
                stack.append(target)
    return visited


def _reverse_csr(map_graph, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """Build compact reverse adjacency without one Python container per cell."""
    counts = np.zeros(n_vertices + 1, dtype=np.int64)
    for source in range(n_vertices):
        for raw_target in map_graph.adjacencies(source):
            target = int(raw_target)
            if target < 0 or target >= n_vertices:
                raise ValueError(
                    f"directed cell graph has edge {source} -> {target}, "
                    f"outside its vertex range [0, {n_vertices})"
                )
            counts[target + 1] += 1
    pointers = np.cumsum(counts)
    neighbors = np.empty(int(pointers[-1]), dtype=np.int64)
    fill = pointers[:-1].copy()
    for source in range(n_vertices):
        for raw_target in map_graph.adjacencies(source):
            target = int(raw_target)
            neighbors[fill[target]] = source
            fill[target] += 1
    return pointers, neighbors


def _backward_reachable(
    reverse_pointers: np.ndarray,
    reverse_neighbors: np.ndarray,
    targets: np.ndarray,
    n_vertices: int,
) -> np.ndarray:
    visited = np.zeros(n_vertices, dtype=bool)
    stack = [int(cell) for cell in targets]
    visited[targets] = True
    while stack:
        target = stack.pop()
        start = int(reverse_pointers[target])
        stop = int(reverse_pointers[target + 1])
        for index in range(start, stop):
            source = int(reverse_neighbors[index])
            if not visited[source]:
                visited[source] = True
                stack.append(source)
    return visited


def _native_directed_path_cells(
    map_graph,
    fine_morse_graph,
    source_nodes: Iterable[int],
    target_nodes: Iterable[int],
) -> np.ndarray | None:
    """Use CMGDB's cached-CSR traversal when the native helper is available."""
    has_cache = getattr(map_graph, "has_cache", None)
    if not callable(has_cache) or not bool(has_cache()):
        return None
    try:
        import CMGDB
    except ImportError:
        return None
    native = getattr(CMGDB, "MorseDirectedPathCells", None)
    if not callable(native):
        return None

    raw = native(
        map_graph,
        fine_morse_graph,
        sorted(int(node) for node in source_nodes),
        sorted(int(node) for node in target_nodes),
    )
    if (
        not isinstance(raw, np.ndarray)
        or raw.ndim != 1
        or raw.dtype != np.uint64
        or not raw.flags.c_contiguous
    ):
        raise TypeError(
            "CMGDB.MorseDirectedPathCells must return a C-contiguous uint64 "
            f"vector; got {type(raw).__name__}, "
            f"dtype={getattr(raw, 'dtype', None)}, "
            f"shape={getattr(raw, 'shape', None)}"
        )
    if raw.size > 1 and np.any(raw[1:] <= raw[:-1]):
        raise ValueError(
            "CMGDB.MorseDirectedPathCells returned cell ids that are not "
            "strictly increasing"
        )
    return raw.astype(np.int64, copy=False)


def _native_directed_path_cells_available(map_graph) -> bool:
    has_cache = getattr(map_graph, "has_cache", None)
    if not callable(has_cache) or not bool(has_cache()):
        return False
    try:
        import CMGDB
    except ImportError:
        return False
    return callable(getattr(CMGDB, "MorseDirectedPathCells", None))


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


def compute_connection_complete_morse_sets(
    map_graph,
    fine_morse_graph,
    projection: Mapping[int, int],
) -> ConnectionCompleteMorseSets:
    """Add within-fiber connections to the Morse sets of a quotient.

    ``map_graph`` is the directed graph on phase-space cells and must provide
    ``num_vertices()`` and ``adjacencies(cell)``. ``fine_morse_graph`` must
    provide ``morse_set(node)``, as a CMGDB Morse graph does. ``projection`` is
    the fine-to-coarse poset epimorphism.

    For a coarse node ``q`` with fiber ``S``, let ``U`` be the union of the
    fine Morse-set cells indexed by ``S``. The returned cell enclosure is

    ``forward_reachable(U) & backward_reachable(U)``.

    Equivalently, it contains exactly the cell-graph vertices on a directed
    path whose initial and terminal Morse components both map to ``q``. This
    adds connections internal to a collapsed fiber without adding outgoing
    connections from that fiber to a different coarse Morse node.

    A singleton fiber is returned verbatim rather than traversing the graph.
    If a completed fiber contains a fine recurrent component mapped somewhere
    else, the supplied projection is not order-convex for this cell graph and
    is rejected.
    """
    normalized_projection = {int(fine): int(coarse) for fine, coarse in projection.items()}
    if not normalized_projection:
        raise ValueError("the fine-to-coarse projection must be nonempty")
    if len(normalized_projection) != len(projection):
        raise ValueError("projection keys must remain unique after conversion to integer node ids")

    fibers_mutable: dict[int, set[int]] = {}
    for fine, coarse in normalized_projection.items():
        fibers_mutable.setdefault(coarse, set()).add(fine)
    fibers = {
        coarse: frozenset(fine_nodes)
        for coarse, fine_nodes in sorted(fibers_mutable.items())
    }

    n_vertices = int(map_graph.num_vertices())
    if n_vertices < 0:
        raise ValueError(f"directed cell graph reports negative vertex count {n_vertices}")

    fine_nodes = sorted(normalized_projection)
    fine_cells: dict[int, np.ndarray] = {}
    recurrent_owner = np.full(n_vertices, -1, dtype=np.int64)
    for owner_index, fine in enumerate(fine_nodes):
        cells = np.unique(
            np.fromiter(
                (int(cell) for cell in fine_morse_graph.morse_set(fine)),
                dtype=np.int64,
            )
        )
        if cells.size and (cells[0] < 0 or cells[-1] >= n_vertices):
            invalid = cells[(cells < 0) | (cells >= n_vertices)]
            raise ValueError(
                f"fine Morse node {fine} contains cells outside [0, {n_vertices}): "
                f"{invalid[:10].tolist()}"
            )
        if cells.size:
            occupied = recurrent_owner[cells] != -1
            if np.any(occupied):
                conflicts = cells[occupied]
                other_nodes = sorted(
                    {fine_nodes[int(index)] for index in recurrent_owner[conflicts]}
                )
                raise ValueError(
                    f"fine Morse node {fine} overlaps fine Morse nodes {other_nodes} "
                    f"on recurrent cells {conflicts[:10].tolist()}"
                )
            recurrent_owner[cells] = owner_index
        fine_cells[fine] = cells

    needs_completion = any(len(fiber) > 1 for fiber in fibers.values())
    fine_vertex_count = getattr(fine_morse_graph, "num_vertices", None)
    projection_is_total = (
        callable(fine_vertex_count)
        and set(normalized_projection)
        == set(range(int(fine_vertex_count())))
    )
    use_native = (
        needs_completion
        and projection_is_total
        and _native_directed_path_cells_available(map_graph)
    )
    reverse_pointers: np.ndarray | None = None
    reverse_neighbors: np.ndarray | None = None
    if needs_completion and not use_native:
        reverse_pointers, reverse_neighbors = _reverse_csr(map_graph, n_vertices)

    cells_by_coarse: dict[int, np.ndarray] = {}
    connections_by_coarse: dict[int, np.ndarray] = {}
    for coarse, fiber in fibers.items():
        base_parts = [fine_cells[fine] for fine in sorted(fiber) if fine_cells[fine].size]
        base = (
            np.unique(np.concatenate(base_parts))
            if base_parts
            else np.empty(0, dtype=np.int64)
        )
        if len(fiber) == 1 or base.size == 0:
            complete = base
        else:
            complete = (
                _native_directed_path_cells(
                    map_graph,
                    fine_morse_graph,
                    fiber,
                    fiber,
                )
                if use_native
                else None
            )
            if complete is None:
                forward = _forward_reachable(map_graph, base, n_vertices)
                assert reverse_pointers is not None and reverse_neighbors is not None
                backward = _backward_reachable(
                    reverse_pointers,
                    reverse_neighbors,
                    base,
                    n_vertices,
                )
                complete = np.flatnonzero(forward & backward).astype(
                    np.int64,
                    copy=False,
                )
            if complete.size and (complete[0] < 0 or complete[-1] >= n_vertices):
                raise ValueError(
                    "directed-path completion returned cells outside "
                    f"[0, {n_vertices})"
                )
            if not np.all(np.isin(base, complete, assume_unique=True)):
                raise ValueError(
                    f"directed-path completion for projection fiber "
                    f"{sorted(fiber)} omitted recurrent cells"
                )

            recurrent_indices = np.unique(
                recurrent_owner[complete][recurrent_owner[complete] != -1]
            )
            foreign_nodes = [
                fine_nodes[int(owner_index)]
                for owner_index in recurrent_indices
                if normalized_projection[fine_nodes[int(owner_index)]] != coarse
            ]
            if foreign_nodes:
                raise ValueError(
                    f"projection fiber {sorted(fiber)} is not order-convex: "
                    "its connection interval contains fine recurrent nodes "
                    f"{sorted(foreign_nodes)} mapped to another coarse node"
                )

        cells_by_coarse[coarse] = complete
        connections_by_coarse[coarse] = np.setdiff1d(
            complete,
            base,
            assume_unique=True,
        )

    coarse_nodes = sorted(cells_by_coarse)
    first_owner = np.full(n_vertices, -1, dtype=np.int64)
    overlap_members: dict[int, set[int]] = {}
    for coarse_index, coarse in enumerate(coarse_nodes):
        coarse_cells = cells_by_coarse[coarse]
        occupied = first_owner[coarse_cells] != -1
        for cell, previous_index in zip(
            coarse_cells[occupied],
            first_owner[coarse_cells[occupied]],
            strict=True,
        ):
            overlap_members.setdefault(
                int(cell),
                {coarse_nodes[int(previous_index)]},
            ).add(coarse)
        first_owner[coarse_cells[~occupied]] = coarse_index

    return ConnectionCompleteMorseSets(
        projection=normalized_projection,
        fibers=fibers,
        cells=cells_by_coarse,
        connection_cells=connections_by_coarse,
        overlaps={
            cell: frozenset(coarse_nodes_for_cell)
            for cell, coarse_nodes_for_cell in sorted(overlap_members.items())
        },
    )



def uniform_grid_shape(depth: int, dim: int) -> NDArray[np.int64]:
    """Per-axis bin counts of a uniform CMGDB tree of the given total depth.

    CMGDB bisects axis ``depth % dim``, so the axes receive the splits in turn.
    """
    shape = np.ones(dim, dtype=np.int64)
    for level in range(int(depth)):
        shape[level % dim] *= 2
    return shape


def boxes_to_uniform_cells(
    boxes: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    shape: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Uniform cell ids met by a set of boxes.

    A box claims every uniform cell it touches, so a box smaller than a cell
    still claims the cell containing it. The result is the smallest union of
    uniform cells containing the given boxes.
    """
    from .basin_statistics import cmgdb_morton_cell_indices

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    shape = np.asarray(shape, dtype=np.int64)
    dim = lower.size
    span = upper - lower
    ids: set[int] = set()
    for row in np.asarray(boxes, dtype=np.float64):
        lo = (row[:dim] - lower) / span * shape
        hi = (row[dim:] - lower) / span * shape
        lo_bin = np.clip(np.floor(lo).astype(np.int64), 0, shape - 1)
        hi_bin = np.clip(np.ceil(hi).astype(np.int64) - 1, 0, shape - 1)
        hi_bin = np.maximum(hi_bin, lo_bin)
        axes = [np.arange(a, b + 1, dtype=np.int64) for a, b in zip(lo_bin, hi_bin)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)
        ids.update(int(v) for v in cmgdb_morton_cell_indices(mesh, shape))
    return np.array(sorted(ids), dtype=np.int64)


def uniform_cell_boxes(
    cells: NDArray[np.int64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    shape: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Geometry of uniform cells, laid out as CMGDB writes Morse-set rows."""
    from .basin_statistics import cmgdb_morton_cell_indices

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    shape = np.asarray(shape, dtype=np.int64)
    dim = lower.size
    width = (upper - lower) / shape
    total = int(np.prod(shape))
    all_bins = np.stack(
        np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1
    ).reshape(-1, dim)
    order = np.empty(total, dtype=np.int64)
    order[cmgdb_morton_cell_indices(all_bins, shape)] = np.arange(total)
    origin = lower + all_bins[order[np.asarray(cells, dtype=np.int64)]] * width
    return np.concatenate([origin, origin + width], axis=1)


class _CellSetsAsMorseGraph:
    """Present explicit per-node cell ids the way a CMGDB Morse graph does."""

    def __init__(self, cells_by_node: Mapping[int, NDArray[np.int64]]) -> None:
        self._cells = {int(k): np.asarray(v, dtype=np.int64) for k, v in cells_by_node.items()}

    def morse_set(self, node: int) -> NDArray[np.int64]:
        return self._cells[int(node)]


def compute_uniform_connection_complete_morse_sets(
    uniform_map_graph,
    fine_morse_graph,
    projection: Mapping[int, int],
    *,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    depth: int,
) -> tuple[ConnectionCompleteMorseSets, dict[int, NDArray[np.int64]], NDArray[np.int64]]:
    """Connection-complete the quotient on a uniform grid of the given depth.

    Completing a merged fiber on the *adaptive* cell graph lets its deepest
    cells trace connecting orbits at a resolution the coarser structures around
    it do not share, so the enclosure reaches into regions those structures have
    already claimed. Rebuilding the cell graph uniformly -- at ``subdiv_min``,
    the depth every recurrent cell was carried to -- expresses the fine sets,
    their connections, and anything else computed on that grid in one common
    decomposition.

    The fine Morse sets come from the adaptive computation and are carried over
    by geometry; only the connecting orbits are recomputed here. Returns the
    completed sets, the per-fine-node uniform cells, and the grid shape.
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    shape = uniform_grid_shape(depth, lower.size)
    n_cells = int(uniform_map_graph.num_vertices())

    cells_by_node: dict[int, NDArray[np.int64]] = {}
    for node in sorted(int(n) for n in projection):
        boxes = np.asarray(fine_morse_graph.morse_set_boxes(node), dtype=np.float64)
        cells_by_node[node] = boxes_to_uniform_cells(boxes, lower, upper, shape)

    # Two fine sets can claim one cell once carried to this depth; give it to the
    # lowest-numbered claimant so the assignment stays a partition, which is what
    # the connection completion requires.
    owner = np.full(n_cells, -1, dtype=np.int64)
    for node in sorted(cells_by_node):
        cells = cells_by_node[node]
        free = cells[owner[cells] == -1]
        owner[free] = node
        cells_by_node[node] = free

    completed = compute_connection_complete_morse_sets(
        uniform_map_graph, _CellSetsAsMorseGraph(cells_by_node), projection
    )
    return completed, cells_by_node, shape

def write_morse_graph_dot(morse_graph: MorseGraph, path: str | Path) -> Path:
    """Write a parsed or quotient Morse graph in CMGDB-compatible DOT form."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["digraph {"]
    for node in morse_graph.nodes:
        label = morse_graph.labels.get(node, str(node))
        color = morse_graph.colors.get(node, "#808080ff")
        lines.append(
            f'{node} [label={json.dumps(label, ensure_ascii=False)}, '
            f'shape=ellipse, style=filled, fillcolor="{color}"];'
        )
    for source in morse_graph.nodes:
        for target in morse_graph.edges.get(source, ()):
            lines.append(f"{source} -> {target};")
    lines.append("}")
    output.write_text("\n".join(lines) + "\n")
    return output


def write_connection_complete_morse_sets(
    fine_morse_graph,
    morse_sets: ConnectionCompleteMorseSets,
    destination: str | Path,
    *,
    allow_overlaps: bool = False,
) -> Path:
    """Write connection-complete cell sets in CMGDB ``morse_sets`` CSV form.

    Cell geometry is obtained from ``fine_morse_graph.phase_space_box(cell)``.
    Overlapping combinatorial enclosures cannot be represented by a single
    label per cell, so they raise by default. With ``allow_overlaps=True`` the
    cell geometry is written once for every coarse set that contains it.
    """
    if morse_sets.overlaps and not allow_overlaps:
        examples = list(morse_sets.overlaps.items())[:10]
        raise ValueError(
            "connection-complete Morse-set enclosures overlap; pass "
            f"allow_overlaps=True to write duplicate rows. Examples: {examples}"
        )

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", newline="") as dst:
        writer = csv.writer(dst, lineterminator="\n")
        for coarse in sorted(morse_sets.cells):
            for cell in morse_sets.cells[coarse]:
                box = list(fine_morse_graph.phase_space_box(int(cell)))
                if not box or len(box) % 2:
                    raise ValueError(
                        f"phase_space_box({int(cell)}) returned {len(box)} coordinates; "
                        "expected lower and upper coordinates"
                    )
                writer.writerow([*box, coarse])
    return destination_path


def write_quotient_morse_sets(
    source: str | Path,
    destination: str | Path,
    projection: Mapping[int, int],
) -> Path:
    """Relabel a CMGDB ``morse_sets`` CSV by a Morse-poset projection.

    Every geometric box is retained. Thus a quotient fiber is visualized by
    the literal union of the fine Morse-set boxes assigned to it. The output
    does not add boxes enclosing connecting dynamics and should not be treated
    as a newly computed coarse Morse set. When the underlying cell graph is
    available, use :func:`compute_connection_complete_morse_sets` followed by
    :func:`write_connection_complete_morse_sets` instead.
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
