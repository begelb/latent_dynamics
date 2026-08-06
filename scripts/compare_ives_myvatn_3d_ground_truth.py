#!/usr/bin/env python3
"""Compare direct-Ives 3D Morse computations across grid resolutions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Run:
    root: Path
    manifest: dict[str, Any]
    membership: dict[str, Any]
    source_level: int
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    graph: nx.DiGraph
    role_nodes: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_splits(level: int, dimension: int = 3) -> NDArray[np.int64]:
    return np.asarray(
        [(level - axis + dimension - 1) // dimension for axis in range(dimension)],
        dtype=np.int64,
    )


def _role_nodes(membership: dict[str, Any]) -> dict[str, str]:
    rows = membership.get("rows", [])
    cycle = [row for row in rows if int(row["vertex"]) == 0]
    fixed = [row for row in rows if int(row["vertex"]) == 1]
    if len(cycle) != 12 or len(fixed) != 1:
        raise ValueError("membership must contain 12 cycle phases and one fixed point")
    if any(len(row["morse_node_memberships"]) != 1 for row in rows):
        raise ValueError("every reference point must have one raw Morse membership")
    cycle_nodes = {str(row["morse_node_memberships"][0]) for row in cycle}
    if len(cycle_nodes) != 1:
        raise ValueError("cycle phases do not share one Morse component")
    fixed_node = str(fixed[0]["morse_node_memberships"][0])
    cycle_node = next(iter(cycle_nodes))
    if cycle_node == fixed_node:
        raise ValueError("cycle and fixed point share one Morse component")
    return {"period12_component": cycle_node, "fixed_component": fixed_node}


def _load_run(root: Path) -> Run:
    manifest_path = root / "manifest.json"
    membership_path = root / "reference_membership.json"
    source_path = root / "MG" / "morse_sets"
    for path in (manifest_path, membership_path, source_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    if _sha256(source_path) != manifest["artifacts"]["morse_sets"]["sha256"]:
        raise ValueError(f"Morse-set hash mismatch in {root}")
    lower = np.asarray(manifest["system"]["bounds"]["lower"], dtype=np.float64)
    upper = np.asarray(manifest["system"]["bounds"]["upper"], dtype=np.float64)
    source_level = int(manifest["morse_sets"]["source_level"])
    graph = nx.DiGraph()
    graph.add_nodes_from(str(node) for node in manifest["morse_graph"]["nodes"])
    graph.add_edges_from(
        (str(source), str(target))
        for source, target in manifest["morse_graph"]["edges"]
    )
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"Morse graph is not acyclic: {root}")
    roles = _role_nodes(membership)
    for node in graph:
        role = next((role for role, value in roles.items() if value == node), "unassigned")
        graph.nodes[node]["role"] = role
    return Run(root, manifest, membership, source_level, lower, upper, graph, roles)


def _configuration_fingerprint(run: Run) -> dict[str, Any]:
    return {
        "parameters": run.manifest["system"]["parameters"],
        "bounds": {
            "lower": run.lower.tolist(),
            "upper": run.upper.tolist(),
        },
        "rectangle_map": run.manifest["rectangle_map"],
        "cmgdb_revision": run.manifest["cmgdb"]["revision"],
        "cmgdb_version": run.manifest["cmgdb"]["version"],
        "reference_sha256": run.manifest["reference_invariant_points"]["sha256"],
    }


def _transitive_role_graph(run: Run) -> nx.DiGraph:
    reduced = nx.transitive_reduction(run.graph)
    for node in reduced:
        reduced.nodes[node]["role"] = run.graph.nodes[node]["role"]
    return reduced


def _graphs_role_isomorphic(coarse: Run, fine: Run) -> tuple[bool, dict[str, str]]:
    coarse_graph = _transitive_role_graph(coarse)
    fine_graph = _transitive_role_graph(fine)
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        coarse_graph,
        fine_graph,
        node_match=nx.algorithms.isomorphism.categorical_node_match("role", "unassigned"),
    )
    if not matcher.is_isomorphic():
        return False, {}
    return True, {str(key): str(value) for key, value in matcher.mapping.items()}


def _load_role_cell_ids(
    run: Run,
    *,
    target_level: int,
) -> dict[str, NDArray[np.int64]]:
    if target_level > run.source_level:
        raise ValueError("target grid cannot be finer than the source grid")
    data = np.loadtxt(run.root / "MG" / "morse_sets", delimiter=",", ndmin=2)
    labels = np.rint(data[:, 6]).astype(np.int64)
    if not np.array_equal(data[:, 6], labels):
        raise ValueError("nonintegral Morse label")
    target_splits = _axis_splits(target_level)
    target_counts = np.left_shift(1, target_splits)
    target_widths = (run.upper - run.lower) / target_counts
    indices_float = (data[:, :3] - run.lower) / target_widths
    indices = np.floor(indices_float + 1e-9).astype(np.int64)
    geometry = (
        (indices[:, 0] * target_counts[1] + indices[:, 1]) * target_counts[2]
        + indices[:, 2]
    )
    result: dict[str, NDArray[np.int64]] = {}
    for role, raw_node in run.role_nodes.items():
        node = int(raw_node)
        result[role] = np.unique(geometry[labels == node])
    return result


def _limit_warning_nodes(run: Run) -> list[str]:
    assessment = run.manifest["morse_sets"].get("subdivision_limit_assessment")
    if assessment is not None:
        return [str(node) for node in assessment["guaranteed_immediate_stop_nodes"]]
    return [
        str(node)
        for node in run.manifest["morse_sets"].get(
            "guaranteed_limit_stops_before_first_post_min_decomposition", []
        )
    ]


def _run_record(run: Run) -> dict[str, Any]:
    graph = run.graph
    sinks = {node for node in graph if graph.out_degree(node) == 0}
    splits = _axis_splits(run.source_level)
    widths = (run.upper - run.lower) / np.left_shift(1, splits)
    role_assignment = {
        role: {
            "node_id": node,
            "is_sink": node in sinks,
            "reference_coverage": 12 if role == "period12_component" else 1,
        }
        for role, node in run.role_nodes.items()
    }
    return {
        "path": str(run.root),
        "source_level": run.source_level,
        "cell_width": widths.tolist(),
        "compute_seconds": run.manifest["compute_seconds"],
        "morse_box_count": run.manifest["morse_sets"]["box_count"],
        "morse_nodes": graph.number_of_nodes(),
        "morse_edges": graph.number_of_edges(),
        "sink_count": len(sinks),
        "role_assignment": role_assignment,
        "boundary_touching_nodes": run.manifest["morse_sets"][
            "boundary_touching_nodes"
        ],
        "limit_warning_nodes": _limit_warning_nodes(run),
        "interval_audit_passed": bool(
            json.loads(
                (run.root / "interval_enclosure_audit.json").read_text(encoding="utf-8")
            )["passed"]
        ),
    }


def _compare_pair(coarse: Run, fine: Run) -> dict[str, Any]:
    if not np.array_equal(coarse.lower, fine.lower) or not np.array_equal(
        coarse.upper, fine.upper
    ):
        raise ValueError("run domains differ")
    isomorphic, node_mapping = _graphs_role_isomorphic(coarse, fine)
    coarse_ids = _load_role_cell_ids(coarse, target_level=coarse.source_level)
    fine_parent_ids = _load_role_cell_ids(fine, target_level=coarse.source_level)
    containment: dict[str, float] = {}
    for role in ("period12_component", "fixed_component"):
        inside = np.isin(fine_parent_ids[role], coarse_ids[role], assume_unique=True)
        containment[role] = float(np.mean(inside)) if inside.size else 0.0

    coarse_width = (coarse.upper - coarse.lower) / np.left_shift(
        1, _axis_splits(coarse.source_level)
    )
    fine_width = (fine.upper - fine.lower) / np.left_shift(
        1, _axis_splits(fine.source_level)
    )
    volume_ratio: dict[str, float] = {}
    for role in ("period12_component", "fixed_component"):
        coarse_count = coarse.manifest["morse_sets"]["boxes_per_node"][
            coarse.role_nodes[role]
        ]
        fine_count = fine.manifest["morse_sets"]["boxes_per_node"][
            fine.role_nodes[role]
        ]
        coarse_volume = float(coarse_count * np.prod(coarse_width))
        fine_volume = float(fine_count * np.prod(fine_width))
        volume_ratio[role] = fine_volume / coarse_volume

    return {
        "coarse_level": coarse.source_level,
        "fine_level": fine.source_level,
        "configuration_equal_except_resolution": (
            _configuration_fingerprint(coarse) == _configuration_fingerprint(fine)
        ),
        "colored_transitive_reductions_isomorphic": isomorphic,
        "coarse_to_fine_node_mapping": node_mapping,
        "fine_parent_containment_fraction": containment,
        "cell_width_ratio": (fine_width / coarse_width).tolist(),
        "outer_cell_volume_ratio": volume_ratio,
    }


def compare(run_paths: list[Path], output_dir: Path) -> dict[str, Any]:
    runs = sorted((_load_run(path.resolve()) for path in run_paths), key=lambda run: run.source_level)
    if len(runs) < 2:
        raise ValueError("at least two run directories are required")
    if len({run.source_level for run in runs}) != len(runs):
        raise ValueError("run source levels must be unique")
    records = [_run_record(run) for run in runs]
    comparisons = [
        _compare_pair(coarse, fine) for coarse, fine in pairwise(runs)
    ]
    last_two_records = records[-2:]
    last_comparison = comparisons[-1]
    enclosure_topology_stable = bool(
        last_comparison["configuration_equal_except_resolution"]
        and last_comparison["colored_transitive_reductions_isomorphic"]
        and all(
            value == 1.0
            for value in last_comparison["fine_parent_containment_fraction"].values()
        )
        and all(not record["boundary_touching_nodes"] for record in last_two_records)
        and all(not record["limit_warning_nodes"] for record in last_two_records)
        and all(record["interval_audit_passed"] for record in last_two_records)
    )
    resolved_by_run = [
        record["sink_count"] == 2
        and record["role_assignment"]["period12_component"]["is_sink"]
        and record["role_assignment"]["fixed_component"]["is_sink"]
        for record in last_two_records
    ]
    two_attractor_topology_resolved = bool(resolved_by_run[-1])
    fine_enough = bool(enclosure_topology_stable and all(resolved_by_run))
    payload = {
        "schema_version": 1,
        "purpose": "cross-resolution acceptance audit for direct 3D Ives Morse sets",
        "runs": records,
        "comparisons": comparisons,
        "acceptance": {
            "enclosure_topology_stable": enclosure_topology_stable,
            "two_attractor_topology_resolved": two_attractor_topology_resolved,
            "fine_enough": fine_enough,
            "criterion": (
                "The last two role-matched graphs must be isomorphic and nested, "
                "and both must place the period-12 orbit and fixed point in two "
                "distinct sinks without boundary or known limit-stop warnings."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "resolution_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "source_level",
                "compute_seconds",
                "morse_box_count",
                "morse_nodes",
                "morse_edges",
                "sink_count",
                "period12_is_sink",
                "fixed_is_sink",
                "boundary_touching_node_count",
                "limit_warning_node_count",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_level": record["source_level"],
                    "compute_seconds": record["compute_seconds"],
                    "morse_box_count": record["morse_box_count"],
                    "morse_nodes": record["morse_nodes"],
                    "morse_edges": record["morse_edges"],
                    "sink_count": record["sink_count"],
                    "period12_is_sink": record["role_assignment"][
                        "period12_component"
                    ]["is_sink"],
                    "fixed_is_sink": record["role_assignment"]["fixed_component"][
                        "is_sink"
                    ],
                    "boundary_touching_node_count": len(
                        record["boundary_touching_nodes"]
                    ),
                    "limit_warning_node_count": len(record["limit_warning_nodes"]),
                }
            )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = compare(args.runs, args.output_dir.resolve())
    print(json.dumps(payload["acceptance"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
