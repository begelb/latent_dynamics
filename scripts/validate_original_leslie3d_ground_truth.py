"""Validate a saved original-Leslie-3D Morse computation against known objects.

The postprocessor is deliberately independent of CMGDB. It streams the saved
``MG/morse_sets`` file, assigns the known recurrent points to Morse nodes by
box containment, parses the saved DOT graph, and checks the expected dynamical
relations. It accepts either a run root containing ``screen`` or ``conley`` or
one of those artifact directories directly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from bisect import bisect_left, bisect_right
from collections import deque
from pathlib import Path
from typing import Any

KNOWN_OBJECTS: dict[str, dict[str, Any]] = {
    "P0": {
        "description": "stable period-four orbit",
        "points": [
            [0.06476966192518428, 71.81567937129047, 3.2375662571576385],
            [1.2097281778002897, 0.04533876334762899, 50.27097555990333],
            [6.607278075831916, 0.8468097244602028, 0.031737134343340294],
            [102.59382767327212, 4.625094653082341, 0.5927668071221419],
        ],
    },
    "P1": {
        "description": "stable period-four orbit",
        "points": [
            [3.231447513714454, 30.156899956353254, 7.0621449189907235],
            [20.09019988641001, 2.2620132596001175, 21.109829969447276],
            [14.412540651001478, 14.063139920487005, 1.5834092817200822],
            [43.08128565193322, 10.088778455701034, 9.844197944340904],
        ],
    },
    "S2": {
        "description": "saddle period-two orbit",
        "points": [
            [4.995002957976051, 25.272089741366145, 2.4475514494082646],
            [36.10298534480878, 3.4965020705832353, 17.6904628189563],
        ],
    },
    "S4": {
        "description": "saddle period-four orbit",
        "points": [
            [0.6595601552884892, 46.017276427535535, 9.204435860581222],
            [5.960579974197539, 0.46169210870194244, 32.21209349927487],
            [18.78456298077801, 4.172405981938277, 0.3231844760913597],
            [65.73896632505077, 13.149194086544606, 2.9206841873567937],
        ],
    },
    "p_star": {
        "description": "positive saddle fixed point",
        "points": [[18.73654933147751, 13.115584532034255, 9.180909172423979]],
    },
    "origin": {
        "description": "origin",
        "points": [[0.0, 0.0, 0.0]],
    },
}

EXPECTED_INDICES = {
    "P0": "(x^4-1,0,0,0)",
    "P1": "(x^4-1,0,0,0)",
    "S2": "(0,x^2+1,0,0)",
    "S4": "(0,x^4-1,0,0)",
    "p_star": "(0,x+1,0,0)",
    "origin": "(0,0,0,0)",
}

NODE_RE = re.compile(r'^\s*"?(\d+)"?\s*\[(?P<attrs>[^\]]*)\]\s*;?\s*$')
EDGE_RE = re.compile(r'^\s*"?(\d+)"?\s*->\s*"?(\d+)"?\s*;?\s*$')
LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')


def resolve_artifact_dir(run: Path, mode: str) -> tuple[Path, str]:
    """Resolve a run root, artifact directory, or ``MG`` directory."""

    run = run.expanduser().resolve()
    if run.name == "MG" and (run / "morse_sets").is_file():
        run = run.parent

    def complete(candidate: Path) -> bool:
        return (
            (candidate / "morse_graph").is_file()
            and (candidate / "MG" / "morse_sets").is_file()
            and (candidate / "manifest.json").is_file()
        )

    if complete(run):
        return run, f"using artifact directory {run}"

    if mode != "auto":
        candidate = run / mode
        if not complete(candidate):
            raise FileNotFoundError(
                f"{candidate} does not contain morse_graph, MG/morse_sets, and manifest.json"
            )
        return candidate, f"selected requested {mode} artifacts"

    candidates = [candidate for name in ("conley", "screen") if complete(candidate := run / name)]
    if not candidates:
        raise FileNotFoundError(
            f"{run} contains no complete conley or screen artifact directory"
        )
    selected = candidates[0]
    if len(candidates) == 1:
        note = f"auto-selected {selected.name} artifacts"
    else:
        note = "both conley and screen artifacts exist, auto-selected conley"
    return selected, note


def parse_dot(path: Path) -> tuple[list[int], dict[int, list[int]], dict[int, str]]:
    nodes: set[int] = set()
    edges: dict[int, list[int]] = {}
    labels: dict[int, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        node_match = NODE_RE.match(raw)
        if node_match:
            node = int(node_match.group(1))
            nodes.add(node)
            label_match = LABEL_RE.search(node_match.group("attrs"))
            if label_match:
                labels[node] = label_match.group(1)
            continue
        edge_match = EDGE_RE.match(raw)
        if edge_match:
            source, target = map(int, edge_match.groups())
            nodes.update((source, target))
            edges.setdefault(source, []).append(target)
    if not nodes:
        raise ValueError(f"no graph nodes found in {path}")
    return sorted(nodes), {node: sorted(set(targets)) for node, targets in edges.items()}, labels


def flatten_known_points() -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    for object_name, data in KNOWN_OBJECTS.items():
        for phase, point in enumerate(data["points"]):
            phases.append(
                {
                    "key": f"{object_name}[{phase}]",
                    "object": object_name,
                    "phase": phase,
                    "point": point,
                }
            )
    return phases


def stream_morse_memberships(
    path: Path,
    tolerance: float,
) -> tuple[dict[str, set[int]], dict[str, dict[int, int]], dict[int, int], int]:
    """Stream boxes and retain only membership data for the known points."""

    phases = flatten_known_points()
    phases_by_x = sorted(phases, key=lambda phase: phase["point"][0])
    x_values = [phase["point"][0] for phase in phases_by_x]
    memberships = {phase["key"]: set() for phase in phases}
    hit_counts = {phase["key"]: {} for phase in phases}
    boxes_per_node: dict[int, int] = {}
    total_boxes = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            fields = stripped.split(",")
            if len(fields) != 7:
                raise ValueError(
                    f"{path}:{line_number}: expected 7 comma-separated fields, got {len(fields)}"
                )
            try:
                values = [float(field) for field in fields[:6]]
                raw_node = float(fields[6])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid numeric row") from error
            node = int(raw_node)
            if raw_node != node:
                raise ValueError(f"{path}:{line_number}: nonintegral Morse node {raw_node}")
            lower = values[:3]
            upper = values[3:]
            if any(
                not math.isfinite(value)
                for value in (*lower, *upper)
            ) or any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
                raise ValueError(f"{path}:{line_number}: invalid box {lower} -> {upper}")

            total_boxes += 1
            boxes_per_node[node] = boxes_per_node.get(node, 0) + 1
            first = bisect_left(x_values, lower[0] - tolerance)
            last = bisect_right(x_values, upper[0] + tolerance)
            for phase in phases_by_x[first:last]:
                point = phase["point"]
                if all(
                    lo - tolerance <= coordinate <= hi + tolerance
                    for coordinate, lo, hi in zip(point, lower, upper, strict=True)
                ):
                    key = phase["key"]
                    memberships[key].add(node)
                    counts = hit_counts[key]
                    counts[node] = counts.get(node, 0) + 1

    return memberships, hit_counts, boxes_per_node, total_boxes


def summarize_assignments(
    memberships: dict[str, set[int]],
    hit_counts: dict[str, dict[int, int]],
) -> dict[str, dict[str, Any]]:
    assignments: dict[str, dict[str, Any]] = {}
    for object_name, data in KNOWN_OBJECTS.items():
        phase_records = []
        phase_node_sets: list[set[int]] = []
        for phase, point in enumerate(data["points"]):
            key = f"{object_name}[{phase}]"
            nodes = memberships[key]
            phase_node_sets.append(nodes)
            phase_records.append(
                {
                    "phase": phase,
                    "point": point,
                    "nodes": sorted(nodes),
                    "containing_box_counts": {
                        str(node): count for node, count in sorted(hit_counts[key].items())
                    },
                }
            )
        union = set().union(*phase_node_sets)
        all_present = all(phase_node_sets)
        unique_node = next(iter(union)) if all_present and len(union) == 1 else None
        if not all_present:
            state = "unassigned"
        elif unique_node is None:
            state = "ambiguous_or_split"
        else:
            state = "unique"
        assignments[object_name] = {
            "description": data["description"],
            "state": state,
            "node": unique_node,
            "nodes_across_phases": sorted(union),
            "phases": phase_records,
        }
    return assignments


def compute_reachability(
    nodes: list[int],
    edges: dict[int, list[int]],
) -> dict[int, set[int]]:
    reachable: dict[int, set[int]] = {}
    for source in nodes:
        seen: set[int] = set()
        queue = deque(edges.get(source, []))
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(edges.get(node, []))
        reachable[source] = seen
    return reachable


def find_path(source: int, target: int, edges: dict[int, list[int]]) -> list[int] | None:
    if source == target:
        return [source]
    predecessor: dict[int, int | None] = {source: None}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for child in edges.get(node, []):
            if child in predecessor:
                continue
            predecessor[child] = node
            if child == target:
                path = [target]
                current = target
                while predecessor[current] is not None:
                    current = predecessor[current]  # type: ignore[assignment]
                    path.append(current)
                return list(reversed(path))
            queue.append(child)
    return None


def normalize_index(value: str) -> str:
    return "".join(value.split())


def conley_indices(
    manifest: dict[str, Any],
    graph_labels: dict[int, str],
) -> dict[int, str]:
    indices: dict[int, str] = {}
    manifest_indices = manifest.get("conley_indices")
    if isinstance(manifest_indices, dict):
        for raw_node, annotation in manifest_indices.items():
            if annotation is None:
                continue
            if isinstance(annotation, list):
                if len(annotation) == 1 and str(annotation[0]).lstrip().startswith("("):
                    rendered = str(annotation[0])
                else:
                    rendered = f"({', '.join(str(value) for value in annotation)})"
            else:
                rendered = str(annotation)
            indices[int(raw_node)] = normalize_index(rendered)

    for node, label in graph_labels.items():
        if node in indices or " : " not in label:
            continue
        rendered = label.split(" : ", 1)[1]
        if rendered.startswith("("):
            indices[node] = normalize_index(rendered)
    return indices


def criterion(
    description: str,
    passed: bool | None,
    expected: Any,
    observed: Any,
    group: str = "topology",
) -> dict[str, Any]:
    return {
        "description": description,
        "group": group,
        "status": "unavailable" if passed is None else ("pass" if passed else "fail"),
        "expected": expected,
        "observed": observed,
    }


def unique_node(assignments: dict[str, dict[str, Any]], name: str) -> int | None:
    value = assignments[name]["node"]
    return int(value) if value is not None else None


def build_criteria(
    assignments: dict[str, dict[str, Any]],
    nodes: list[int],
    edges: dict[int, list[int]],
    indices: dict[int, str],
) -> tuple[dict[str, dict[str, Any]], str, str]:
    object_nodes = {name: unique_node(assignments, name) for name in KNOWN_OBJECTS}
    all_unique = all(node is not None for node in object_nodes.values())
    core_unique = all(object_nodes[name] is not None for name in ("P0", "P1", "S2"))
    minimal = {node for node in nodes if not edges.get(node)}

    def distinct(left: str, right: str) -> bool | None:
        left_node, right_node = object_nodes[left], object_nodes[right]
        return None if left_node is None or right_node is None else left_node != right_node

    def is_minimal(name: str) -> bool | None:
        node = object_nodes[name]
        return None if node is None else node in minimal

    def reaches(source_name: str, target_name: str) -> tuple[bool | None, list[int] | None]:
        source = object_nodes[source_name]
        target = object_nodes[target_name]
        if source is None or target is None:
            return None, None
        path = find_path(source, target, edges)
        return source != target and path is not None, path

    def does_not_reach(source_name: str, target_name: str) -> tuple[bool | None, list[int] | None]:
        source = object_nodes[source_name]
        target = object_nodes[target_name]
        if source is None or target is None:
            return None, None
        path = find_path(source, target, edges)
        return source != target and path is None, path

    def has_expected_index(name: str) -> tuple[bool | None, str | None]:
        node = object_nodes[name]
        if node is None or node not in indices:
            return None, None if node is None else indices.get(node)
        observed = indices[node]
        return observed == EXPECTED_INDICES[name], observed

    p0 = object_nodes["P0"]
    p1 = object_nodes["P1"]
    expected_minimal = None if p0 is None or p1 is None else sorted({p0, p1})
    minimal_match = (
        None
        if expected_minimal is None
        else minimal == set(expected_minimal) and p0 != p1
    )
    s2_to_p1, s2_to_p1_path = reaches("S2", "P1")
    s2_not_p0, s2_to_p0_path = does_not_reach("S2", "P0")
    p1_not_s2, p1_to_s2_path = does_not_reach("P1", "S2")
    s4_to_p0, s4_to_p0_path = reaches("S4", "P0")
    s4_to_p1, s4_to_p1_path = reaches("S4", "P1")
    pstar_to_s2, pstar_to_s2_path = reaches("p_star", "S2")
    p0_index_ok, p0_index = has_expected_index("P0")
    p1_index_ok, p1_index = has_expected_index("P1")
    supplemental_indices = {
        name: has_expected_index(name) for name in ("S2", "S4", "p_star", "origin")
    }

    criteria = {
        "core_objects_have_unique_nodes": criterion(
            "Every phase of P0, P1, and S2 belongs to one Morse node",
            core_unique,
            "unique node for P0, P1, and S2",
            {name: object_nodes[name] for name in ("P0", "P1", "S2")},
        ),
        "all_known_objects_have_unique_nodes": criterion(
            "Every phase of every known object belongs to one Morse node",
            all_unique,
            "unique node for P0, P1, S2, S4, p_star, and origin",
            object_nodes,
            group="supplemental",
        ),
        "stable_period_four_orbits_are_distinct": criterion(
            "The two stable period-four orbits occupy different Morse nodes",
            distinct("P0", "P1"),
            "P0 node != P1 node",
            {"P0": p0, "P1": p1},
        ),
        "P1_and_S2_are_distinct": criterion(
            "The stable period-four orbit P1 is separated from the saddle period-two orbit S2",
            distinct("P1", "S2"),
            "P1 node != S2 node",
            {"P1": p1, "S2": object_nodes["S2"]},
        ),
        "P0_is_minimal": criterion(
            "P0 is represented by a minimal Morse node",
            is_minimal("P0"),
            True,
            {"node": p0, "minimal_nodes": sorted(minimal)},
        ),
        "P1_is_minimal": criterion(
            "P1 is represented by a minimal Morse node",
            is_minimal("P1"),
            True,
            {"node": p1, "minimal_nodes": sorted(minimal)},
        ),
        "minimal_nodes_are_exactly_P0_and_P1": criterion(
            "The two stable period-four nodes are exactly the minimal Morse nodes",
            minimal_match,
            expected_minimal,
            sorted(minimal),
        ),
        "S2_reaches_P1": criterion(
            "The saddle period-two node reaches the P1 attractor node",
            s2_to_p1,
            "nontrivial directed path S2 -> P1",
            s2_to_p1_path,
        ),
        "S2_does_not_reach_P0": criterion(
            "The saddle period-two node does not reach the P0 attractor node",
            s2_not_p0,
            "no directed path S2 -> P0",
            s2_to_p0_path,
        ),
        "P1_does_not_reach_S2": criterion(
            "There is no return path from the P1 attractor node to the saddle period-two node",
            p1_not_s2,
            "no directed path P1 -> S2",
            p1_to_s2_path,
        ),
        "S4_reaches_P0": criterion(
            "The saddle period-four node reaches the P0 attractor node",
            s4_to_p0,
            "nontrivial directed path S4 -> P0",
            s4_to_p0_path,
            group="supplemental",
        ),
        "S4_reaches_P1": criterion(
            "The saddle period-four node reaches the P1 attractor node",
            s4_to_p1,
            "nontrivial directed path S4 -> P1",
            s4_to_p1_path,
            group="supplemental",
        ),
        "P0_has_stable_period_four_index": criterion(
            "P0 has the stable period-four Conley index",
            p0_index_ok,
            EXPECTED_INDICES["P0"],
            p0_index,
            group="conley",
        ),
        "P1_has_stable_period_four_index": criterion(
            "P1 has the stable period-four Conley index",
            p1_index_ok,
            EXPECTED_INDICES["P1"],
            p1_index,
            group="conley",
        ),
        "p_star_reaches_S2": criterion(
            "The positive saddle fixed-point node reaches the saddle period-two node",
            pstar_to_s2,
            "nontrivial directed path p_star -> S2",
            pstar_to_s2_path,
            group="supplemental",
        ),
    }
    for name, (index_ok, observed_index) in supplemental_indices.items():
        criteria[f"{name}_has_expected_index"] = criterion(
            f"{name} has its numerically expected Conley index",
            index_ok,
            EXPECTED_INDICES[name],
            observed_index,
            group="supplemental",
        )

    topology_statuses = [
        value["status"] for value in criteria.values() if value["group"] == "topology"
    ]
    conley_statuses = [
        value["status"] for value in criteria.values() if value["group"] == "conley"
    ]
    topology_status = "pass" if all(status == "pass" for status in topology_statuses) else "fail"
    if topology_status == "fail" or "fail" in conley_statuses:
        full_status = "fail"
    elif "unavailable" in conley_statuses:
        full_status = "incomplete"
    else:
        full_status = "pass"
    return criteria, topology_status, full_status


def manifest_subset(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "system",
        "theta",
        "survival",
        "bounds",
        "subdivision",
        "box_map",
        "algorithm",
        "compute_seconds",
        "cmgdb",
        "subdivision_diagnostics",
    )
    return {key: manifest.get(key) for key in keys if key in manifest}


def render_text(report: dict[str, Any]) -> str:
    lines = [
        "Original Leslie 3D ground-truth validation",
        f"Artifacts: {report['source']['artifact_dir']}",
        f"Selection: {report['source']['selection']}",
    ]
    subdivision = report["configuration"].get("subdivision")
    if subdivision:
        lines.append(
            "Subdivisions: "
            f"{subdivision.get('init')}/{subdivision.get('min')}/{subdivision.get('max')}, "
            f"limit {subdivision.get('limit')}"
        )
    subdivision_diagnostics = report["configuration"].get("subdivision_diagnostics")
    if subdivision_diagnostics:
        stopped_nodes = subdivision_diagnostics.get(
            "nodes_guaranteed_to_exceed_limit_before_first_post_min_decomposition",
            [],
        )
        if stopped_nodes:
            lines.append(
                "Guaranteed limit stops before first post-min decomposition: "
                + ", ".join(map(str, stopped_nodes))
            )
    lines.extend(
        [
            (
                f"Boxes streamed: {report['morse_sets']['total_boxes']} "
                f"({report['morse_sets']['file_size_bytes']} bytes)"
            ),
            (
                f"Graph: {len(report['graph']['nodes'])} nodes, "
                f"{len(report['graph']['edges'])} edges, "
                f"minimal nodes {report['graph']['minimal_nodes']}"
            ),
            "",
            "Known-object assignments:",
        ]
    )
    display_names = {"p_star": "p*", "origin": "origin"}
    for name, assignment in report["assignments"].items():
        display = display_names.get(name, name)
        lines.append(
            f"  {display:7s} {assignment['state']:18s} "
            f"node={assignment['node']} phases={assignment['nodes_across_phases']}"
        )

    lines.extend(["", "Acceptance criteria:"])
    markers = {"pass": "PASS", "fail": "FAIL", "unavailable": "N/A "}
    for name, result in report["criteria"].items():
        lines.append(f"  {markers[result['status']]} [{result['group']}] {name}")
        if result["status"] != "pass":
            lines.append(f"       expected: {result['expected']}")
            lines.append(f"       observed: {result['observed']}")
    lines.extend(
        [
            "",
            f"Topology status: {report['overall']['topology'].upper()}",
            f"Full status: {report['overall']['full'].upper()}",
        ]
    )
    if report["overall"]["full"] == "incomplete":
        lines.append("Full validation requires a Conley run with saved node annotations.")
    return "\n".join(lines) + "\n"


def write_reports(
    prefix: Path,
    report: dict[str, Any],
    text: str,
    force: bool,
) -> tuple[Path, Path]:
    json_path = prefix.with_suffix(".json")
    text_path = prefix.with_suffix(".txt")
    existing = [path for path in (json_path, text_path) if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing report files: {joined}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    return json_path, text_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--mode",
        choices=("auto", "screen", "conley"),
        default="auto",
        help="Artifact directory to select when RUN is a run root.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance for point-in-box containment.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Report path without extension. Defaults inside the selected artifact directory.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the text report without writing JSON and text files.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing report files.")
    args = parser.parse_args()

    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative number")

    artifact_dir, selection_note = resolve_artifact_dir(args.run, args.mode)
    manifest_path = artifact_dir / "manifest.json"
    graph_path = artifact_dir / "morse_graph"
    morse_sets_path = artifact_dir / "MG" / "morse_sets"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes, edges, graph_labels = parse_dot(graph_path)
    memberships, hit_counts, boxes_per_node, total_boxes = stream_morse_memberships(
        morse_sets_path,
        args.tolerance,
    )
    assignments = summarize_assignments(memberships, hit_counts)
    reachable = compute_reachability(nodes, edges)
    indices = conley_indices(manifest, graph_labels)
    criteria, topology_status, full_status = build_criteria(
        assignments,
        nodes,
        edges,
        indices,
    )
    minimal = sorted(node for node in nodes if not edges.get(node))
    edge_list = [
        [source, target]
        for source in sorted(edges)
        for target in edges[source]
    ]
    manifest_box_counts = manifest.get("morse_boxes_per_node")
    normalized_manifest_counts = (
        {str(key): int(value) for key, value in manifest_box_counts.items()}
        if isinstance(manifest_box_counts, dict)
        else None
    )
    streamed_counts = {str(node): count for node, count in sorted(boxes_per_node.items())}
    manifest_nodes = manifest.get("morse_nodes")
    manifest_minimal = manifest.get("minimal_nodes")
    manifest_edges = manifest.get("edges")

    report = {
        "source": {
            "artifact_dir": str(artifact_dir),
            "selection": selection_note,
            "manifest": str(manifest_path),
            "morse_graph": str(graph_path),
            "morse_sets": str(morse_sets_path),
        },
        "configuration": manifest_subset(manifest),
        "containment_tolerance": args.tolerance,
        "morse_sets": {
            "file_size_bytes": morse_sets_path.stat().st_size,
            "total_boxes": total_boxes,
            "boxes_per_node": streamed_counts,
        },
        "graph": {
            "nodes": nodes,
            "edges": edge_list,
            "minimal_nodes": minimal,
            "labels": {str(node): label for node, label in sorted(graph_labels.items())},
            "conley_indices": {str(node): index for node, index in sorted(indices.items())},
            "strict_reachability": {
                str(node): sorted(reachable[node]) for node in nodes
            },
        },
        "saved_artifact_consistency": {
            "manifest_node_count_matches": (
                None if manifest_nodes is None else int(manifest_nodes) == len(nodes)
            ),
            "manifest_edges_match": (
                None
                if manifest_edges is None
                else sorted([list(map(int, edge)) for edge in manifest_edges]) == edge_list
            ),
            "manifest_minimal_nodes_match": (
                None
                if manifest_minimal is None
                else sorted(map(int, manifest_minimal)) == minimal
            ),
            "manifest_box_counts_match": (
                None
                if normalized_manifest_counts is None
                else normalized_manifest_counts == streamed_counts
            ),
            "csv_nodes_match_graph_nodes": sorted(boxes_per_node) == nodes,
        },
        "assignments": assignments,
        "criteria": criteria,
        "overall": {
            "topology": topology_status,
            "full": full_status,
            "supplemental_criteria_affect_overall": False,
        },
    }
    text = render_text(report)
    print(text, end="")

    if not args.no_write:
        prefix = args.output_prefix or (artifact_dir / "ground_truth_validation")
        json_path, text_path = write_reports(prefix, report, text, args.force)
        print(f"JSON report: {json_path}", file=sys.stderr)
        print(f"Text report: {text_path}", file=sys.stderr)
    return 0 if full_status in {"pass", "incomplete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
