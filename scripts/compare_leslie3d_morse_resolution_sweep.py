#!/usr/bin/env python3
"""Compare Leslie3D Morse decompositions across subdivision choices.

The inputs are completed run roots; this script never calls CMGDB and never
modifies a run.  It writes a compact, provenance-bearing comparison bundle:

* ``runs.csv`` and ``comparison.json`` summarize every decomposition;
* ``pairwise.csv`` records exact and node-id-invariant graph comparisons;
* ``sweep_comparison.png`` and ``sweep_comparison.pdf`` visualize graph
  equivalence, recurrent-cover sizes, graph sizes, and runtimes.

Graph equality is deliberately reported at several strengths.  Raw DOT hashes
test byte equality.  ID-semantic equality compares the saved node ids, edges,
and Conley indices.  Indexed isomorphism ignores arbitrary node ids but
preserves each Conley index.  Role-indexed isomorphism additionally preserves
the invariant objects contained by each Morse set.

Example::

    python scripts/compare_leslie3d_morse_resolution_sweep.py \
      --run '(20,26)=output/notebooks/.../seed_20260809' \
      --run '(10,26)=output/notebooks/.../seed_20260809' \
      --baseline-label '(20,26)' \
      --encoded-points output/notebooks/.../analysis/encoded_invariant_points.csv \
      --output-dir ../output/leslie3d_subdivision_sweep/analysis
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = CODE_ROOT / "data" / "leslie3d_invariant_aware_v2" / "dataset_manifest.json"
EXPECTED_MAX = 36
EXPECTED_LIMIT = 10_000_000
CHUNK_ROWS = 250_000
CONTAINMENT_TOLERANCE = 1e-14

NODE_RE = re.compile(r'^\s*"?(\d+)"?\s*\[(?P<attrs>[^\]]*)\]\s*;?\s*$')
EDGE_RE = re.compile(r'^\s*"?(\d+)"?\s*->\s*"?(\d+)"?\s*;?\s*$')
LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')
INDEX_RE = re.compile(r":\s*\(([^)]*)\)")

ROLE_ORDER = ("P0", "P1", "S2", "S4", "p_star", "origin")


class ComparisonInputError(RuntimeError):
    """An input run is absent, malformed, or violates the sweep contract."""


@dataclass(frozen=True)
class GraphRecord:
    node_indices: dict[int, tuple[str, ...]]
    edges: tuple[tuple[int, int], ...]

    @property
    def nodes(self) -> tuple[int, ...]:
        return tuple(sorted(self.node_indices))

    @property
    def minimal(self) -> tuple[int, ...]:
        sources = {source for source, _ in self.edges}
        return tuple(node for node in self.nodes if node not in sources)


@dataclass(frozen=True)
class EncodedPhase:
    role: str
    phase: int
    point: tuple[float, float]


@dataclass
class RunRecord:
    label: str
    root: Path
    params: dict[str, Any]
    graph: GraphRecord
    graph_sha256: str
    sets_sha256: str
    checkpoint_sha256: str | None
    sets_size_bytes: int
    box_counts: dict[int, int]
    box_bounds: dict[int, tuple[float, float, float, float]]
    memberships: dict[str, dict[str, Any]]
    role_audit: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_file(path: Path, purpose: str) -> Path:
    if not path.is_file():
        raise ComparisonInputError(f"missing {purpose}: {path}")
    return path


def _parse_run_argument(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("run must have the form LABEL=RUN_ROOT")
    label, path_text = raw.split("=", 1)
    label = label.strip()
    if not label or not path_text.strip():
        raise argparse.ArgumentTypeError("run must have a nonempty label and path")
    return label, Path(path_text).expanduser().resolve()


def _parse_params(path: Path) -> dict[str, Any]:
    _require_file(path, "CMGDB parameter log")
    values: dict[str, Any] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if ":" not in line:
            raise ComparisonInputError(f"malformed parameter line {path}:{line_number}")
        key, raw = line.split(":", 1)
        key = key.strip()
        raw = raw.strip()
        if key in values:
            raise ComparisonInputError(f"duplicate parameter {key!r} in {path}")
        try:
            values[key] = json.loads(raw)
        except json.JSONDecodeError:
            if raw == "True":
                values[key] = True
            elif raw == "False":
                values[key] = False
            else:
                values[key] = raw
    for key in ("subdiv_init", "subdiv_min", "subdiv_max", "subdiv_limit"):
        if not isinstance(values.get(key), int):
            raise ComparisonInputError(f"{path} has no integer {key}")
    duration = values.get("duration_minutes")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ComparisonInputError(f"{path} has no positive duration_minutes")
    return values


def _parse_graph(path: Path) -> GraphRecord:
    _require_file(path, "Morse graph DOT")
    indices: dict[int, tuple[str, ...]] = {}
    edges: set[tuple[int, int]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        edge_match = EDGE_RE.match(line)
        if edge_match:
            edges.add((int(edge_match.group(1)), int(edge_match.group(2))))
            continue
        node_match = NODE_RE.match(line)
        if not node_match:
            continue
        node = int(node_match.group(1))
        label_match = LABEL_RE.search(node_match.group("attrs"))
        index_match = INDEX_RE.search(label_match.group(1)) if label_match else None
        if index_match is None:
            raise ComparisonInputError(f"node {node} has no Conley index at {path}:{line_number}")
        if node in indices:
            raise ComparisonInputError(f"duplicate node {node} at {path}:{line_number}")
        indices[node] = tuple(part.strip() for part in index_match.group(1).split(","))
    if not indices:
        raise ComparisonInputError(f"no Morse nodes parsed from {path}")
    unknown = {node for edge in edges for node in edge} - set(indices)
    if unknown:
        raise ComparisonInputError(f"edges in {path} reference unknown nodes {sorted(unknown)}")
    graph = GraphRecord(indices, tuple(sorted(edges)))
    nx_graph = _to_networkx(graph, {})
    if not nx.is_directed_acyclic_graph(nx_graph):
        raise ComparisonInputError(f"Morse graph is not a DAG: {path}")
    return graph


def _load_encoded_phases(path: Path) -> list[EncodedPhase]:
    _require_file(path, "encoded invariant-point CSV")
    phases: list[EncodedPhase] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"object", "phase", "z0", "z1"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ComparisonInputError(f"{path} lacks encoded point fields {sorted(required)}")
        for row in reader:
            phases.append(
                EncodedPhase(
                    role=str(row["object"]),
                    phase=int(row["phase"]),
                    point=(float(row["z0"]), float(row["z1"])),
                )
            )
    if not phases:
        raise ComparisonInputError(f"no encoded points found in {path}")
    duplicate_keys = [
        key for key, count in Counter((phase.role, phase.phase) for phase in phases).items() if count > 1
    ]
    if duplicate_keys:
        raise ComparisonInputError(f"duplicate encoded phases in {path}: {duplicate_keys}")
    return sorted(phases, key=lambda phase: (ROLE_ORDER.index(phase.role), phase.phase))


def _numeric_chunks(path: Path, rows_per_chunk: int = CHUNK_ROWS) -> Iterable[NDArray[np.float64]]:
    """Parse a five-column, headerless CMGDB Morse-set file in bounded memory."""

    with path.open("rb") as handle:
        while True:
            lines = list(islice(handle, rows_per_chunk))
            if not lines:
                return
            raw = b"".join(lines).decode("ascii").replace("\r", "").replace("\n", ",")
            values = np.fromstring(raw, sep=",", dtype=np.float64)
            expected = len(lines) * 5
            if values.size != expected:
                raise ComparisonInputError(
                    f"expected {expected} numeric values while parsing {path}, got {values.size}"
                )
            yield values.reshape((-1, 5))


def _scan_sets(
    path: Path,
    phases: Sequence[EncodedPhase],
) -> tuple[
    dict[int, int],
    dict[int, tuple[float, float, float, float]],
    dict[str, dict[str, Any]],
]:
    _require_file(path, "Morse-set CSV")
    points = np.asarray([phase.point for phase in phases], dtype=np.float64)
    containing: list[set[int]] = [set() for _ in phases]
    nearest_distance_sq = np.full(len(phases), np.inf, dtype=np.float64)
    nearest_label = np.full(len(phases), -1, dtype=np.int64)
    counts: Counter[int] = Counter()
    bounds: dict[int, list[float]] = {}

    for data in _numeric_chunks(path):
        labels_float = data[:, 4]
        labels = labels_float.astype(np.int64)
        if not np.array_equal(labels_float, labels.astype(np.float64)):
            raise ComparisonInputError(f"noninteger Morse labels in {path}")
        unique, chunk_counts = np.unique(labels, return_counts=True)
        counts.update({int(label): int(count) for label, count in zip(unique, chunk_counts, strict=True)})
        for label in unique:
            label_int = int(label)
            subset = data[labels == label, :4]
            candidate = [
                float(subset[:, 0].min()),
                float(subset[:, 1].min()),
                float(subset[:, 2].max()),
                float(subset[:, 3].max()),
            ]
            if label_int not in bounds:
                bounds[label_int] = candidate
            else:
                old = bounds[label_int]
                bounds[label_int] = [
                    min(old[0], candidate[0]),
                    min(old[1], candidate[1]),
                    max(old[2], candidate[2]),
                    max(old[3], candidate[3]),
                ]

        lower = data[:, :2]
        upper = data[:, 2:4]
        for point_index, point in enumerate(points):
            delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
            distance_sq = np.einsum("ij,ij->i", delta, delta)
            inside = labels[distance_sq <= CONTAINMENT_TOLERANCE**2]
            containing[point_index].update(int(label) for label in np.unique(inside))
            chunk_minimum = float(distance_sq.min())
            nearest_candidates = labels[distance_sq == chunk_minimum]
            candidate_label = int(nearest_candidates.min())
            if chunk_minimum < nearest_distance_sq[point_index] or (
                chunk_minimum == nearest_distance_sq[point_index]
                and candidate_label < nearest_label[point_index]
            ):
                nearest_distance_sq[point_index] = chunk_minimum
                nearest_label[point_index] = candidate_label

    if not counts:
        raise ComparisonInputError(f"no Morse boxes found in {path}")

    by_role: dict[str, list[int]] = defaultdict(list)
    for index, phase in enumerate(phases):
        by_role[phase.role].append(index)
    memberships: dict[str, dict[str, Any]] = {}
    for role in ROLE_ORDER:
        phase_indices = by_role.get(role, [])
        phase_nodes = [sorted(containing[index]) for index in phase_indices]
        singletons = {nodes[0] for nodes in phase_nodes if len(nodes) == 1}
        assigned = (
            next(iter(singletons))
            if phase_nodes and all(len(nodes) == 1 for nodes in phase_nodes) and len(singletons) == 1
            else None
        )
        memberships[role] = {
            "assigned_morse_node": assigned,
            "all_phases_in_one_unique_morse_set": assigned is not None,
            "phase_containing_nodes": phase_nodes,
            "phase_nearest_nodes": [int(nearest_label[index]) for index in phase_indices],
            "phase_nearest_cover_distances": [
                float(math.sqrt(nearest_distance_sq[index])) for index in phase_indices
            ],
        }
    return (
        dict(sorted(counts.items())),
        {node: tuple(value) for node, value in sorted(bounds.items())},
        memberships,
    )


def _transitive_descendants(graph: GraphRecord) -> dict[int, set[int]]:
    G = nx.DiGraph()
    G.add_nodes_from(graph.nodes)
    G.add_edges_from(graph.edges)
    return {node: {node, *nx.descendants(G, node)} for node in graph.nodes}


def _role_audit(
    graph: GraphRecord,
    memberships: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected_indices = manifest.get("expected_direct_indices", {})
    expected_edges = [tuple(edge) for edge in manifest.get("orbit_manifold_informed_reduced_edges", [])]
    expected_sources = {source for source, _ in expected_edges}
    expected_minimal_roles = set(ROLE_ORDER) - expected_sources
    descendants = _transitive_descendants(graph)

    object_checks: dict[str, dict[str, Any]] = {}
    assigned_nodes: list[int] = []
    for role in ROLE_ORDER:
        assigned_raw = memberships.get(role, {}).get("assigned_morse_node")
        assigned = int(assigned_raw) if assigned_raw is not None else None
        if assigned is not None:
            assigned_nodes.append(assigned)
        expected = tuple(str(value) for value in expected_indices.get(role, [])[:3])
        observed = graph.node_indices.get(assigned) if assigned is not None else None
        is_minimal = assigned in graph.minimal if assigned is not None else None
        object_checks[role] = {
            "assigned_node": assigned,
            "expected_index": list(expected),
            "observed_index": list(observed) if observed is not None else None,
            "index_matches": observed == expected,
            "is_minimal": is_minimal,
            "expected_minimal": role in expected_minimal_roles,
            "minimality_matches": is_minimal == (role in expected_minimal_roles),
        }

    expected_reachability = {role: {role} for role in ROLE_ORDER}
    for source, target in expected_edges:
        expected_reachability[source].add(target)
    changed = True
    while changed:
        changed = False
        for source in ROLE_ORDER:
            expanded = set(expected_reachability[source])
            for target in tuple(expanded):
                expanded.update(expected_reachability[target])
            if expanded != expected_reachability[source]:
                expected_reachability[source] = expanded
                changed = True

    reachability: dict[str, bool | None] = {}
    reachability_matches: list[bool] = []
    for source in ROLE_ORDER:
        for target in ROLE_ORDER:
            if source == target:
                continue
            source_node = object_checks[source]["assigned_node"]
            target_node = object_checks[target]["assigned_node"]
            observed = (
                target_node in descendants[source_node]
                if source_node is not None and target_node is not None and source_node != target_node
                else None
            )
            expected = target in expected_reachability[source]
            reachability[f"{source}->{target}"] = observed
            reachability_matches.append(observed == expected)

    all_assigned = len(assigned_nodes) == len(ROLE_ORDER)
    all_distinct = all_assigned and len(set(assigned_nodes)) == len(ROLE_ORDER)
    all_indices = all(check["index_matches"] for check in object_checks.values())
    all_minimality = all(check["minimality_matches"] for check in object_checks.values())
    all_reachability = all(reachability_matches)
    return {
        "object_checks": object_checks,
        "role_reachability": reachability,
        "all_objects_uniquely_assigned": all_assigned,
        "all_objects_in_distinct_nodes": all_distinct,
        "node_count_matches_six_roles": len(graph.nodes) == len(ROLE_ORDER),
        "all_object_indices_match": all_indices,
        "all_object_minimality_matches": all_minimality,
        "all_role_reachability_and_nonreachability_match": all_reachability,
        "exact_role_aligned_morse_graph_match": (
            all_assigned
            and all_distinct
            and len(graph.nodes) == len(ROLE_ORDER)
            and all_indices
            and all_minimality
            and all_reachability
        ),
    }


def _roles_by_node(record: RunRecord) -> dict[int, tuple[str, ...]]:
    roles: dict[int, list[str]] = defaultdict(list)
    for role, membership in record.memberships.items():
        assigned = membership.get("assigned_morse_node")
        if assigned is not None:
            roles[int(assigned)].append(role)
    return {node: tuple(sorted(values)) for node, values in roles.items()}


def _to_networkx(graph: GraphRecord, roles_by_node: Mapping[int, tuple[str, ...]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for node, index in graph.node_indices.items():
        G.add_node(node, conley_index=index, roles=roles_by_node.get(node, ()))
    G.add_edges_from(graph.edges)
    return G


def _id_semantic_payload(record: RunRecord) -> dict[str, Any]:
    return {
        "nodes": {str(node): list(index) for node, index in sorted(record.graph.node_indices.items())},
        "edges": [list(edge) for edge in record.graph.edges],
        "minimal": list(record.graph.minimal),
    }


def _graph_match(
    left: RunRecord,
    right: RunRecord,
    *,
    attributes: tuple[str, ...],
    compare_reachability: bool = False,
) -> tuple[bool, dict[int, int] | None]:
    left_graph = _to_networkx(left.graph, _roles_by_node(left))
    right_graph = _to_networkx(right.graph, _roles_by_node(right))
    if compare_reachability:
        left_graph = nx.transitive_closure_dag(left_graph)
        right_graph = nx.transitive_closure_dag(right_graph)

    def node_match(left_attrs: Mapping[str, Any], right_attrs: Mapping[str, Any]) -> bool:
        return all(left_attrs[name] == right_attrs[name] for name in attributes)

    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        left_graph,
        right_graph,
        node_match=node_match if attributes else None,
    )
    if not matcher.is_isomorphic():
        return False, None
    return True, {int(source): int(target) for source, target in matcher.mapping.items()}


def _pairwise(left: RunRecord, right: RunRecord) -> dict[str, Any]:
    unlabelled, unlabelled_mapping = _graph_match(left, right, attributes=())
    indexed, indexed_mapping = _graph_match(left, right, attributes=("conley_index",))
    role_indexed, role_indexed_mapping = _graph_match(
        left, right, attributes=("conley_index", "roles")
    )
    indexed_order, indexed_order_mapping = _graph_match(
        left,
        right,
        attributes=("conley_index",),
        compare_reachability=True,
    )
    role_indexed_order, role_indexed_order_mapping = _graph_match(
        left,
        right,
        attributes=("conley_index", "roles"),
        compare_reachability=True,
    )
    indexed_box_counts_equal = bool(
        indexed_order_mapping is not None
        and all(
            left.box_counts[source] == right.box_counts[target]
            for source, target in indexed_order_mapping.items()
        )
    )
    return {
        "left": left.label,
        "right": right.label,
        "id_semantic_equal": _id_semantic_payload(left) == _id_semantic_payload(right),
        "unlabelled_isomorphic": unlabelled,
        "unlabelled_mapping": unlabelled_mapping,
        "indexed_isomorphic": indexed,
        "indexed_mapping": indexed_mapping,
        "role_indexed_isomorphic": role_indexed,
        "role_indexed_mapping": role_indexed_mapping,
        "indexed_morse_order_isomorphic": indexed_order,
        "indexed_morse_order_mapping": indexed_order_mapping,
        "role_indexed_morse_order_isomorphic": role_indexed_order,
        "role_indexed_morse_order_mapping": role_indexed_order_mapping,
        "normalized_graph_equal": indexed_order,
        "normalized_role_graph_equal": role_indexed_order,
        "role_reachability_equal": (
            left.role_audit["role_reachability"] == right.role_audit["role_reachability"]
        ),
        "morse_sets_byte_equal": left.sets_sha256 == right.sets_sha256,
        "box_counts_by_id_equal": left.box_counts == right.box_counts,
        "box_counts_under_indexed_isomorphism_equal": indexed_box_counts_equal,
        "total_box_count_equal": sum(left.box_counts.values()) == sum(right.box_counts.values()),
    }


def _histogram(indices: Iterable[tuple[str, ...]]) -> dict[str, int]:
    values = Counter("(" + ", ".join(index) + ")" for index in indices)
    return dict(sorted(values.items()))


def _run_summary(record: RunRecord, baseline: RunRecord) -> dict[str, Any]:
    reference = _pairwise(record, baseline)
    assignments = {
        role: membership["assigned_morse_node"] for role, membership in record.memberships.items()
    }
    return {
        "label": record.label,
        "run_root": str(record.root),
        "parameters": {
            key: record.params[key]
            for key in (
                "subdiv_init",
                "subdiv_min",
                "subdiv_max",
                "subdiv_limit",
                "duration_minutes",
            )
        },
        "graph": {
            "nodes": len(record.graph.nodes),
            "edges": len(record.graph.edges),
            "minimal_nodes": len(record.graph.minimal),
            "minimal_node_ids": list(record.graph.minimal),
            "index_histogram": _histogram(record.graph.node_indices.values()),
            "minimal_index_histogram": _histogram(
                record.graph.node_indices[node] for node in record.graph.minimal
            ),
            "id_semantic_sha256": _canonical_sha256(_id_semantic_payload(record)),
            "dot_sha256": record.graph_sha256,
        },
        "morse_sets": {
            "boxes_total": sum(record.box_counts.values()),
            "boxes_by_node_id": {str(node): count for node, count in record.box_counts.items()},
            "cover_bounds_by_node_id": {
                str(node): list(bounds) for node, bounds in record.box_bounds.items()
            },
            "sha256": record.sets_sha256,
            "size_bytes": record.sets_size_bytes,
        },
        "checkpoint_sha256": record.checkpoint_sha256,
        "invariant_membership": record.memberships,
        "role_audit": record.role_audit,
        "role_assignments": assignments,
        "comparison_to_baseline": {
            key: value
            for key, value in reference.items()
            if key
            not in {
                "left",
                "right",
                "unlabelled_mapping",
                "indexed_mapping",
                "role_indexed_mapping",
                "indexed_morse_order_mapping",
                "role_indexed_morse_order_mapping",
            }
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _flat_run_row(record: RunRecord, baseline_label: str) -> dict[str, Any]:
    comparison = record.role_audit
    return {
        "label": record.label,
        "is_baseline": record.label == baseline_label,
        "subdiv_init": record.params["subdiv_init"],
        "subdiv_min": record.params["subdiv_min"],
        "subdiv_max": record.params["subdiv_max"],
        "subdiv_limit": record.params["subdiv_limit"],
        "duration_minutes": record.params["duration_minutes"],
        "n_nodes": len(record.graph.nodes),
        "n_edges": len(record.graph.edges),
        "n_minimal_nodes": len(record.graph.minimal),
        "index_histogram": _histogram(record.graph.node_indices.values()),
        "minimal_index_histogram": _histogram(
            record.graph.node_indices[node] for node in record.graph.minimal
        ),
        "morse_boxes_total": sum(record.box_counts.values()),
        "morse_boxes_by_node_id": record.box_counts,
        "morse_graph_sha256": record.graph_sha256,
        "morse_sets_sha256": record.sets_sha256,
        "checkpoint_sha256": record.checkpoint_sha256,
        "role_assignments": {
            role: value["assigned_morse_node"] for role, value in record.memberships.items()
        },
        "all_objects_uniquely_assigned": comparison["all_objects_uniquely_assigned"],
        "all_objects_in_distinct_nodes": comparison["all_objects_in_distinct_nodes"],
        "all_object_indices_match": comparison["all_object_indices_match"],
        "all_object_minimality_matches": comparison["all_object_minimality_matches"],
        "all_role_reachability_matches": comparison[
            "all_role_reachability_and_nonreachability_match"
        ],
        "exact_role_aligned_morse_graph_match": comparison[
            "exact_role_aligned_morse_graph_match"
        ],
        "run_root": str(record.root),
    }


def _bool_matrix(
    labels: Sequence[str],
    pairwise_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    key: str,
) -> NDArray[np.float64]:
    values = np.empty((len(labels), len(labels)), dtype=np.float64)
    for row, left in enumerate(labels):
        for column, right in enumerate(labels):
            values[row, column] = float(bool(pairwise_lookup[(left, right)][key]))
    return values


def _plot_matrix(
    ax: plt.Axes,
    matrix: NDArray[np.float64],
    labels: Sequence[str],
    title: str,
) -> None:
    ax.imshow(matrix, vmin=0, vmax=1, cmap=ListedColormap(["#C95D63", "#5B9A6F"]))
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(
                column,
                row,
                "same" if matrix[row, column] else "different",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white",
                fontweight="bold",
            )
    ax.set_title(title, fontsize=11, fontweight="bold")


def _plot_dashboard(
    records: Sequence[RunRecord],
    pairwise_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    labels = [record.label for record in records]
    lookup = {(str(row["left"]), str(row["right"])): row for row in pairwise_rows}
    indexed = _bool_matrix(labels, lookup, "indexed_morse_order_isomorphic")
    role_indexed = _bool_matrix(labels, lookup, "role_indexed_morse_order_isomorphic")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.4), constrained_layout=True)
    _plot_matrix(axes[0, 0], indexed, labels, "Conley-index-preserving Morse-order isomorphism")
    _plot_matrix(axes[0, 1], role_indexed, labels, "Role + index Morse-order isomorphism")

    box_totals = np.asarray([sum(record.box_counts.values()) for record in records], dtype=float)
    positions = np.arange(len(records))
    axes[1, 0].bar(positions, box_totals, color="#3978A8")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(positions, labels, rotation=45, ha="right")
    axes[1, 0].set_ylabel("saved Morse boxes (log scale)")
    axes[1, 0].set_title("Recurrent-cover size", fontsize=11, fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=0.25)
    for position, value in zip(positions, box_totals, strict=True):
        axes[1, 0].annotate(
            f"{int(value):,}",
            (position, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            rotation=30,
        )

    counts = {
        "nodes": [len(record.graph.nodes) for record in records],
        "edges": [len(record.graph.edges) for record in records],
        "minimal": [len(record.graph.minimal) for record in records],
    }
    for name, values in counts.items():
        axes[1, 1].plot(positions, values, marker="o", linewidth=2, label=name)
    axes[1, 1].set_xticks(positions, labels, rotation=45, ha="right")
    axes[1, 1].set_ylabel("graph count")
    axes[1, 1].grid(alpha=0.25)
    runtime_ax = axes[1, 1].twinx()
    runtime_ax.plot(
        positions,
        [float(record.params["duration_minutes"]) for record in records],
        color="#D17A22",
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label="runtime",
    )
    runtime_ax.set_ylabel("CMGDB runtime (minutes)", color="#A85E17")
    runtime_ax.tick_params(axis="y", labelcolor="#A85E17")
    lines, line_labels = axes[1, 1].get_legend_handles_labels()
    runtime_lines, runtime_labels = runtime_ax.get_legend_handles_labels()
    axes[1, 1].legend(lines + runtime_lines, line_labels + runtime_labels, loc="best", fontsize=8)
    axes[1, 1].set_title("Graph size and runtime", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Leslie3D subdivision sweep · max 36 · limit 10,000,000",
        fontsize=15,
        fontweight="bold",
    )
    written: list[Path] = []
    for suffix in (".png", ".pdf"):
        path = output_dir / f"sweep_comparison{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def compare(
    run_specs: Sequence[tuple[str, Path]],
    *,
    baseline_label: str,
    encoded_points_path: Path,
    manifest_path: Path,
    output_dir: Path,
    expected_max: int,
    expected_limit: int,
    allow_checkpoint_mismatch: bool,
) -> dict[str, Any]:
    if len(run_specs) < 2:
        raise ComparisonInputError("at least two --run inputs are required")
    labels = [label for label, _ in run_specs]
    if len(set(labels)) != len(labels):
        raise ComparisonInputError("run labels must be unique")
    if baseline_label not in labels:
        raise ComparisonInputError(f"baseline label {baseline_label!r} is not one of {labels}")

    phases = _load_encoded_phases(encoded_points_path)
    manifest = json.loads(_require_file(manifest_path, "dataset manifest").read_text())
    records: list[RunRecord] = []
    for label, root in run_specs:
        params = _parse_params(root / "mg_params_log.txt")
        if params["subdiv_max"] != expected_max:
            raise ComparisonInputError(
                f"{label}: subdiv_max={params['subdiv_max']}, expected {expected_max}"
            )
        if params["subdiv_limit"] != expected_limit:
            raise ComparisonInputError(
                f"{label}: subdiv_limit={params['subdiv_limit']}, expected {expected_limit}"
            )
        graph_path = root / "MG" / "morse_graph"
        sets_path = root / "MG" / "morse_sets"
        graph = _parse_graph(graph_path)
        box_counts, box_bounds, memberships = _scan_sets(sets_path, phases)
        if set(box_counts) != set(graph.nodes):
            raise ComparisonInputError(
                f"{label}: Morse-set labels {sorted(box_counts)} do not match graph nodes {graph.nodes}"
            )
        checkpoint_path = root / "models" / "autoencoder.pt"
        checkpoint_hash = _sha256(checkpoint_path) if checkpoint_path.is_file() else None
        record = RunRecord(
            label=label,
            root=root,
            params=params,
            graph=graph,
            graph_sha256=_sha256(graph_path),
            sets_sha256=_sha256(sets_path),
            checkpoint_sha256=checkpoint_hash,
            sets_size_bytes=sets_path.stat().st_size,
            box_counts=box_counts,
            box_bounds=box_bounds,
            memberships=memberships,
            role_audit={},
        )
        record.role_audit = _role_audit(graph, memberships, manifest)
        records.append(record)

    checkpoint_hashes = {record.checkpoint_sha256 for record in records}
    if not allow_checkpoint_mismatch and (None in checkpoint_hashes or len(checkpoint_hashes) != 1):
        raise ComparisonInputError(
            "runs do not all contain the same models/autoencoder.pt; "
            "use --allow-checkpoint-mismatch only for an intentional mixed-model comparison"
        )

    baseline = next(record for record in records if record.label == baseline_label)
    pairwise_rows = [_pairwise(left, right) for left in records for right in records]
    summaries = [_run_summary(record, baseline) for record in records]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "runs.csv", [_flat_run_row(record, baseline_label) for record in records])
    _write_csv(output_dir / "pairwise.csv", pairwise_rows)
    figure_paths = _plot_dashboard(records, pairwise_rows, output_dir)

    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scientific_status": (
            "finite-resolution numerical comparison; graph isomorphism and file equality are exact "
            "for the saved artifacts but are not a semiconjugacy or Conley-index correctness proof"
        ),
        "sweep_contract": {
            "subdiv_max": expected_max,
            "subdiv_limit": expected_limit,
            "same_checkpoint_required": not allow_checkpoint_mismatch,
            "baseline_label": baseline_label,
            "encoded_points": {
                "path": str(encoded_points_path.resolve()),
                "sha256": _sha256(encoded_points_path),
            },
            "dataset_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": _sha256(manifest_path),
            },
            "containment_tolerance": CONTAINMENT_TOLERANCE,
        },
        "definitions": {
            "id_semantic_equal": "same node ids, Conley indices, edges, and minimal-node ids",
            "unlabelled_isomorphic": "exact directed-graph isomorphism after ignoring node ids and attributes",
            "indexed_isomorphic": (
                "exact saved-edge directed-graph isomorphism after ignoring node ids while preserving "
                "Conley indices"
            ),
            "role_indexed_isomorphic": (
                "indexed isomorphism that also preserves the invariant-object roles assigned to each node"
            ),
            "normalized_graph_equal": (
                "exact Conley-index-preserving isomorphism of the reachability orders (transitive closures); "
                "this ignores node ids and harmless differences in redundant saved edges"
            ),
            "normalized_role_graph_equal": (
                "normalized graph equality that additionally preserves invariant-object roles"
            ),
            "morse_sets_byte_equal": "identical SHA-256 hashes of the raw saved Morse-set files",
        },
        "runs": summaries,
        "pairwise": pairwise_rows,
        "outputs": {
            "runs_csv": str((output_dir / "runs.csv").resolve()),
            "pairwise_csv": str((output_dir / "pairwise.csv").resolve()),
            "figures": [str(path.resolve()) for path in figure_paths],
        },
    }
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run_argument,
        metavar="LABEL=RUN_ROOT",
        help="completed run root; repeat for every subdivision variant",
    )
    parser.add_argument(
        "--baseline-label",
        required=True,
        help="one --run label to use for the per-run reference columns",
    )
    parser.add_argument(
        "--encoded-points",
        type=Path,
        required=True,
        help="one encoded_invariant_points.csv from the fixed checkpoint",
    )
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-max", type=int, default=EXPECTED_MAX)
    parser.add_argument("--expected-limit", type=int, default=EXPECTED_LIMIT)
    parser.add_argument(
        "--allow-checkpoint-mismatch",
        action="store_true",
        help="permit missing or different checkpoints (off by default)",
    )
    args = parser.parse_args()
    result = compare(
        args.run,
        baseline_label=args.baseline_label,
        encoded_points_path=args.encoded_points.expanduser().resolve(),
        manifest_path=args.dataset_manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        expected_max=args.expected_max,
        expected_limit=args.expected_limit,
        allow_checkpoint_mismatch=args.allow_checkpoint_mismatch,
    )
    print(json.dumps(result["outputs"], indent=2))


if __name__ == "__main__":
    main()
