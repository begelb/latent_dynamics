#!/usr/bin/env python3
"""Audit what saved CMGDB artifacts can prove about adaptive refinement depth.

CMGDB's ``morse_sets`` CSV stores the Morse components constructed at
``subdiv_min``.  It does not store the deeper Morse-decomposition hierarchy
used for spurious-set pruning.  Consequently, terminal refinement depths and
limit-hit flags are usually not recoverable exactly after the computation.

This script reports the strongest conclusions that *are* available without
rerunning CMGDB.  For a saved Morse node with N boxes at ``subdiv_min``, every
descendant work-node grid at absolute depth ``subdiv_min + r`` has at most
``N * 2**r`` boxes: each CMGDB subdivision doubles all valid leaves, while an
SCC component selected by decomposition cannot contain more boxes than its
parent grid.  Comparing that upper bound with ``subdiv_limit`` can prove that
the limit was impossible for some nodes, and can identify an immediate
post-minimum limit hit when ``2*N > subdiv_limit``.  Intermediate cases remain
honestly indeterminate from DOT/CSV/log artifacts alone.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

_DOT_NODE_RE = re.compile(r'^\s*"?(\d+)"?\s*\[(?P<attrs>.*)\]\s*;?\s*$')
_DOT_EDGE_RE = re.compile(r'^\s*"?(\d+)"?\s*->\s*"?(\d+)"?')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config must contain a YAML mapping: {path}")
    return data


def _parse_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    result: dict[str, Any] = {}
    for raw in path.read_text().splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        normalized = key.lower().replace(" ", "_")
        if normalized in {"lower_bounds", "upper_bounds"}:
            result[normalized] = ast.literal_eval(value)
        elif normalized in {
            "subdiv_init",
            "subdiv_min",
            "subdiv_max",
            "subdiv_limit",
        }:
            result[normalized] = int(value)
        elif normalized == "duration_minutes":
            result[normalized] = float(value)
        else:
            result[normalized] = value
    return result


def _cmgdb_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("cmgdb")
    if not isinstance(section, dict):
        raise ValueError("config is missing a 'cmgdb' mapping")
    return section


def _effective_parameters(
    cmgdb: dict[str, Any], log: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = ("subdiv_init", "subdiv_min", "subdiv_max", "subdiv_limit")
    effective: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    for key in required:
        config_value = cmgdb.get(key)
        log_value = log.get(key)
        if config_value is None and log_value is None:
            raise ValueError(f"neither config nor mg_params_log provides {key}")
        if config_value is not None and log_value is not None:
            if int(config_value) != int(log_value):
                mismatches.append(
                    {"key": key, "config": int(config_value), "log": int(log_value)}
                )
        effective[key] = int(log_value if log_value is not None else config_value)

    for key in ("lower_bounds", "upper_bounds"):
        config_value = cmgdb.get(key)
        log_value = log.get(key)
        if config_value is not None and log_value is not None:
            config_list = [float(v) for v in config_value]
            log_list = [float(v) for v in log_value]
            if config_list != log_list:
                mismatches.append({"key": key, "config": config_list, "log": log_list})
        value = log_value if log_value is not None else config_value
        if value is not None:
            effective[key] = [float(v) for v in value]

    init = effective["subdiv_init"]
    minimum = effective["subdiv_min"]
    maximum = effective["subdiv_max"]
    if not init <= minimum <= maximum:
        raise ValueError(
            "invalid effective subdivisions: require subdiv_init <= subdiv_min <= subdiv_max"
        )
    return effective, mismatches


def _parse_dot(
    path: Path,
) -> tuple[list[int], list[tuple[int, int]], dict[int, str]]:
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    attributes: dict[int, str] = {}
    for raw in path.read_text().splitlines():
        node_match = _DOT_NODE_RE.match(raw)
        if node_match:
            node = int(node_match.group(1))
            nodes.add(node)
            attributes[node] = node_match.group("attrs").strip()
            continue
        edge_match = _DOT_EDGE_RE.match(raw)
        if edge_match:
            source, target = map(int, edge_match.groups())
            nodes.update((source, target))
            edges.add((source, target))
    return sorted(nodes), sorted(edges), dict(sorted(attributes.items()))


def _minimal_nodes(nodes: list[int], edges: list[tuple[int, int]]) -> list[int]:
    return sorted(set(nodes) - {source for source, _ in edges})


def _dot_line_multiset(path: Path) -> Counter[str]:
    return Counter(line.strip() for line in path.read_text().splitlines() if line.strip())


def _comparison(
    *,
    current_run_root: Path,
    comparison_run_root: Path,
    current_morse_sets_sha256: str,
    current_morse_graph_sha256: str,
    current_nodes: list[int],
    current_edges: list[tuple[int, int]],
    current_attributes: dict[int, str],
) -> dict[str, Any]:
    comparison_run_root = comparison_run_root.resolve()
    comparison_sets = comparison_run_root / "MG" / "morse_sets"
    comparison_graph = comparison_run_root / "MG" / "morse_graph"
    comparison_log = comparison_run_root / "mg_params_log.txt"
    for path in (comparison_sets, comparison_graph):
        if not path.is_file():
            raise FileNotFoundError(path)

    comparison_sets_sha256 = _sha256(comparison_sets)
    comparison_graph_sha256 = _sha256(comparison_graph)
    comparison_nodes, comparison_edges, comparison_attributes = _parse_dot(comparison_graph)
    same_nodes = current_nodes == comparison_nodes
    same_edges = current_edges == comparison_edges
    same_attributes = current_attributes == comparison_attributes
    current_graph = current_run_root / "MG" / "morse_graph"
    return {
        "run_root": str(comparison_run_root),
        "parameters_from_log": _parse_log(comparison_log),
        "artifact_sha256": {
            "morse_sets": comparison_sets_sha256,
            "morse_graph": comparison_graph_sha256,
            "mg_params_log": _sha256(comparison_log) if comparison_log.is_file() else None,
        },
        "morse_sets_byte_identical": comparison_sets_sha256 == current_morse_sets_sha256,
        "morse_graph_byte_identical": comparison_graph_sha256 == current_morse_graph_sha256,
        "morse_graph_same_nodes": same_nodes,
        "morse_graph_same_edges": same_edges,
        "morse_graph_same_node_attributes": same_attributes,
        "morse_graph_same_minimal_nodes": _minimal_nodes(
            current_nodes, current_edges
        )
        == _minimal_nodes(comparison_nodes, comparison_edges),
        "morse_graph_semantically_identical": same_nodes and same_edges and same_attributes,
        "morse_graph_line_multiset_identical": _dot_line_multiset(current_graph)
        == _dot_line_multiset(comparison_graph),
    }


def _box_depth_profile(
    row: list[str],
    *,
    dimension: int,
    domain_widths: list[float] | None,
) -> tuple[int | None, tuple[int, ...] | None, float | None]:
    if domain_widths is None:
        return None, None, None
    lows = [float(value) for value in row[:dimension]]
    highs = [float(value) for value in row[dimension : 2 * dimension]]
    exponents: list[int] = []
    maximum_relative_error = 0.0
    for axis, (low, high, domain_width) in enumerate(
        zip(lows, highs, domain_widths, strict=True)
    ):
        box_width = high - low
        if not (domain_width > 0.0 and box_width > 0.0):
            raise ValueError(
                f"invalid width on axis {axis}: domain={domain_width}, box={box_width}"
            )
        raw_exponent = math.log2(domain_width / box_width)
        exponent = round(raw_exponent)
        reconstructed = domain_width / (2**exponent)
        relative_error = abs(box_width - reconstructed) / reconstructed
        maximum_relative_error = max(maximum_relative_error, relative_error)
        exponents.append(exponent)
    return sum(exponents), tuple(exponents), maximum_relative_error


def _scan_morse_sets(
    path: Path, *, lower_bounds: list[float] | None, upper_bounds: list[float] | None
) -> dict[str, Any]:
    row_counts: Counter[int] = Counter()
    depth_counts: dict[int, Counter[int]] = defaultdict(Counter)
    exponent_profiles: dict[int, Counter[tuple[int, ...]]] = defaultdict(Counter)
    maximum_relative_error = 0.0
    dimension: int | None = None
    total_rows = 0

    domain_widths: list[float] | None = None
    if lower_bounds is not None and upper_bounds is not None:
        if len(lower_bounds) != len(upper_bounds):
            raise ValueError("lower_bounds and upper_bounds have different dimensions")
        domain_widths = [
            high - low for low, high in zip(lower_bounds, upper_bounds, strict=True)
        ]

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if dimension is None:
                if len(row) < 3 or (len(row) - 1) % 2:
                    raise ValueError(
                        f"unexpected morse_sets column count {len(row)} on line {line_number}"
                    )
                dimension = (len(row) - 1) // 2
                if domain_widths is not None and len(domain_widths) != dimension:
                    raise ValueError(
                        "bounds dimension does not match morse_sets dimension: "
                        f"{len(domain_widths)} != {dimension}"
                    )
            elif len(row) != 2 * dimension + 1:
                raise ValueError(
                    f"inconsistent morse_sets columns on line {line_number}: {len(row)}"
                )
            try:
                node = int(float(row[-1]))
            except ValueError as exc:
                raise ValueError(f"invalid Morse label on line {line_number}: {row[-1]!r}") from exc
            row_counts[node] += 1
            total_rows += 1
            depth, profile, relative_error = _box_depth_profile(
                row, dimension=dimension, domain_widths=domain_widths
            )
            if depth is not None and profile is not None and relative_error is not None:
                depth_counts[node][depth] += 1
                exponent_profiles[node][profile] += 1
                maximum_relative_error = max(maximum_relative_error, relative_error)

    if dimension is None:
        raise ValueError(f"morse_sets CSV is empty: {path}")
    return {
        "dimension": dimension,
        "total_rows": total_rows,
        "row_counts": dict(sorted(row_counts.items())),
        "depth_counts": {
            node: dict(sorted(counts.items())) for node, counts in sorted(depth_counts.items())
        },
        "exponent_profiles": {
            node: {",".join(map(str, profile)): count for profile, count in sorted(counts.items())}
            for node, counts in sorted(exponent_profiles.items())
        },
        "maximum_width_reconstruction_relative_error": maximum_relative_error,
    }


def _first_depth_exceeding_limit(
    *, boxes_at_min: int, subdiv_min: int, subdiv_max: int, subdiv_limit: int
) -> int | None:
    for depth in range(subdiv_min + 1, subdiv_max + 1):
        if boxes_at_min * (2 ** (depth - subdiv_min)) > subdiv_limit:
            return depth
    return None


def _node_audit(
    *, node: int, boxes_at_min: int, subdiv_min: int, subdiv_max: int, subdiv_limit: int
) -> dict[str, Any]:
    span = subdiv_max - subdiv_min
    upper_bound_at_max = boxes_at_min * (2**span)
    first_possible = _first_depth_exceeding_limit(
        boxes_at_min=boxes_at_min,
        subdiv_min=subdiv_min,
        subdiv_max=subdiv_max,
        subdiv_limit=subdiv_limit,
    )
    if span == 0:
        no_limit_through = subdiv_min
        reached_at_least = subdiv_min
        status = "max_equals_min_no_post_minimum_refinement"
        max_depth_processed: bool | None = True
        direct_limit_hit: bool | None = None
    else:
        direct_size = boxes_at_min * 2
        direct_limit_hit = direct_size > subdiv_limit
        if direct_limit_hit:
            no_limit_through = subdiv_min
            reached_at_least = subdiv_min + 1
            status = "limit_hit_at_first_post_minimum_depth"
            max_depth_processed = False
        elif first_possible is None:
            no_limit_through = subdiv_max
            reached_at_least = subdiv_max
            status = "limit_impossible_max_depth_processed"
            max_depth_processed = True
        else:
            no_limit_through = first_possible - 1
            reached_at_least = min(subdiv_max, no_limit_through + 1)
            status = "later_limit_hit_or_max_depth_not_identifiable"
            max_depth_processed = None

    return {
        "node": node,
        "boxes_at_saved_min_depth": boxes_at_min,
        "first_post_minimum_work_size": boxes_at_min * 2 if span else None,
        "worst_case_descendant_size_at_max": upper_bound_at_max,
        "earliest_depth_at_which_limit_could_be_exceeded": first_possible,
        "guaranteed_no_limit_through_depth": no_limit_through,
        "guaranteed_descendant_grid_reached_depth_at_least": reached_at_least,
        "direct_post_minimum_limit_hit": direct_limit_hit,
        "max_depth_processed": max_depth_processed,
        "classification": status,
    }


def build_audit(
    config_path: Path,
    run_root: Path,
    comparison_run_root: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    run_root = run_root.resolve()
    morse_sets_path = run_root / "MG" / "morse_sets"
    morse_graph_path = run_root / "MG" / "morse_graph"
    params_log_path = run_root / "mg_params_log.txt"
    for path in (config_path, morse_sets_path, morse_graph_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = _load_yaml(config_path)
    cmgdb = _cmgdb_config(config)
    log = _parse_log(params_log_path)
    parameters, mismatches = _effective_parameters(cmgdb, log)
    lower_bounds = parameters.get("lower_bounds")
    upper_bounds = parameters.get("upper_bounds")
    scan = _scan_morse_sets(
        morse_sets_path,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    graph_nodes, edges, graph_attributes = _parse_dot(morse_graph_path)
    csv_nodes = sorted(scan["row_counts"])
    all_nodes = sorted(set(graph_nodes) | set(csv_nodes))
    if graph_nodes != csv_nodes:
        mismatches.append(
            {"key": "morse_node_ids", "dot": graph_nodes, "morse_sets": csv_nodes}
        )

    minimum = parameters["subdiv_min"]
    maximum = parameters["subdiv_max"]
    limit = parameters["subdiv_limit"]
    nodes = [
        _node_audit(
            node=node,
            boxes_at_min=int(scan["row_counts"].get(node, 0)),
            subdiv_min=minimum,
            subdiv_max=maximum,
            subdiv_limit=limit,
        )
        for node in all_nodes
    ]
    classifications = Counter(item["classification"] for item in nodes)
    observed_depths = sorted(
        {
            int(depth)
            for counts in scan["depth_counts"].values()
            for depth in counts
        }
    )
    saved_depth_matches_min = observed_depths == [minimum]

    morse_sets_sha256 = _sha256(morse_sets_path)
    morse_graph_sha256 = _sha256(morse_graph_path)
    result: dict[str, Any] = {
        "audit_version": 1,
        "inputs": {
            "config": str(config_path),
            "run_root": str(run_root),
            "morse_sets": str(morse_sets_path),
            "morse_graph": str(morse_graph_path),
            "mg_params_log": str(params_log_path) if params_log_path.is_file() else None,
        },
        "artifact_sha256": {
            "morse_sets": morse_sets_sha256,
            "morse_graph": morse_graph_sha256,
            "mg_params_log": _sha256(params_log_path) if params_log_path.is_file() else None,
        },
        "parameters": parameters,
        "config_log_mismatches": mismatches,
        "graph": {
            "nodes": graph_nodes,
            "edges": [list(edge) for edge in edges],
            "minimal_nodes": _minimal_nodes(graph_nodes, edges),
            "node_attributes": {str(node): attrs for node, attrs in graph_attributes.items()},
        },
        "saved_morse_sets": {
            "dimension": scan["dimension"],
            "total_boxes": scan["total_rows"],
            "observed_tree_depths_from_box_widths": observed_depths,
            "all_saved_boxes_match_subdiv_min": saved_depth_matches_min,
            "box_counts_by_node": {
                str(node): count for node, count in scan["row_counts"].items()
            },
            "depth_counts_by_node": {
                str(node): {str(depth): count for depth, count in counts.items()}
                for node, counts in scan["depth_counts"].items()
            },
            "axis_split_profiles_by_node": {
                str(node): profiles for node, profiles in scan["exponent_profiles"].items()
            },
            "maximum_width_reconstruction_relative_error": scan[
                "maximum_width_reconstruction_relative_error"
            ],
        },
        "refinement_audit": {
            "formula": "descendant_size(depth=min+r) <= boxes_at_min * 2**r",
            "nodes": nodes,
            "classification_counts": dict(sorted(classifications.items())),
        },
        "interpretation": {
            "exactly_inferable": [
                "saved minimum-grid box count and box tree depth for every Morse node",
                "whether the first post-minimum work node necessarily exceeded the limit",
                "whether the limit was impossible through subdiv_max by cardinality bound",
                "a guaranteed lower bound on the deepest descendant grid reached",
            ],
            "not_inferable_from_saved_dot_csv_log": [
                "the exact terminal depth of nodes classified as later-limit-or-max",
                "whether a later descendant exceeded subdiv_limit",
                "the sizes, SCC branching, or stop reasons of discarded hierarchy nodes",
            ],
            "cmgdb_semantics": [
                "the limit test is applied only at work-node depths strictly greater than subdiv_min",
                "each subdivision doubles all valid leaves",
                "MorseGraph vertices retain the decomposition grids built at subdiv_min",
                "DOT, morse_sets CSV, and mg_params_log do not serialize the deeper hierarchy",
            ],
            "implementation_evidence": [
                {
                    "source": "archive/CMGDB/src/CMGDB/_cmgdb/include/database/Compute_Morse_Graph.hpp",
                    "lines": "202-206",
                    "fact": "limit test uses depth > Min and work-node size > Limit",
                },
                {
                    "source": "archive/CMGDB/src/CMGDB/_cmgdb/include/database/Compute_Morse_Graph.hpp",
                    "lines": "218-223",
                    "fact": "each nonterminal decomposition component is spawned and subdivided",
                },
                {
                    "source": "archive/CMGDB/src/CMGDB/_cmgdb/include/database/Compute_Morse_Graph.hpp",
                    "lines": "281-302",
                    "fact": "reported MorseGraph vertices are assigned decomposition grids at Min",
                },
                {
                    "source": "archive/CMGDB/src/CMGDB/_cmgdb/include/database/GraphTheory.hpp",
                    "lines": "51-74",
                    "fact": "each SCC component grid is a subgrid of its parent work-node grid",
                },
                {
                    "source": "archive/CMGDB/src/CMGDB/_cmgdb/include/database/CompressedTree.h",
                    "lines": "22-25, 52-59",
                    "fact": "subdivision replaces every valid leaf with two valid children",
                },
                {
                    "source": "archive/CMGDB/src/CMGDB/SaveMorseData.py",
                    "lines": "5-12",
                    "fact": "morse_sets serializes only box bounds plus node label",
                },
            ],
        },
    }
    if comparison_run_root is not None:
        result["comparison"] = _comparison(
            current_run_root=run_root,
            comparison_run_root=comparison_run_root,
            current_morse_sets_sha256=morse_sets_sha256,
            current_morse_graph_sha256=morse_graph_sha256,
            current_nodes=graph_nodes,
            current_edges=edges,
            current_attributes=graph_attributes,
        )
    return result


def _fmt_int(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def render_markdown(audit: dict[str, Any]) -> str:
    params = audit["parameters"]
    saved = audit["saved_morse_sets"]
    refinement = audit["refinement_audit"]
    mismatch_text = (
        "none" if not audit["config_log_mismatches"] else "present; see JSON details"
    )
    lines = [
        "# CMGDB refinement-depth audit",
        "",
        f"- Run root: `{audit['inputs']['run_root']}`",
        f"- Subdivisions `(init, min, max)`: "
        f"`({params['subdiv_init']}, {params['subdiv_min']}, {params['subdiv_max']})`",
        f"- `subdiv_limit`: `{params['subdiv_limit']:,}`",
        f"- Saved Morse boxes: `{saved['total_boxes']:,}` across "
        f"`{len(audit['graph']['nodes'])}` nodes",
        f"- Tree depth(s) reconstructed from saved box widths: "
        f"`{saved['observed_tree_depths_from_box_widths']}`",
        f"- All saved boxes match `subdiv_min`: "
        f"`{saved['all_saved_boxes_match_subdiv_min']}`",
        f"- Config/log mismatches: `{mismatch_text}`",
        "",
        "## Node-level conclusions",
        "",
        "| Node | Boxes at min | Worst-case boxes at max | Earliest possible limit depth | "
        "Guaranteed reached depth | Conclusion |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    labels = {
        "max_equals_min_no_post_minimum_refinement": "max = min; no deeper pass",
        "limit_hit_at_first_post_minimum_depth": "limit hit at first post-min depth",
        "limit_impossible_max_depth_processed": "limit impossible; max processed",
        "later_limit_hit_or_max_depth_not_identifiable": "later limit or max; not identifiable",
    }
    for item in refinement["nodes"]:
        lines.append(
            f"| {item['node']} | {_fmt_int(item['boxes_at_saved_min_depth'])} | "
            f"{_fmt_int(item['worst_case_descendant_size_at_max'])} | "
            f"{_fmt_int(item['earliest_depth_at_which_limit_could_be_exceeded'])} | "
            f"{_fmt_int(item['guaranteed_descendant_grid_reached_depth_at_least'])} | "
            f"{labels[item['classification']]} |"
        )
    lines.extend(
        [
            "",
            "## What the artifacts do—and do not—record",
            "",
            "The saved `morse_sets` rows are the component grids constructed at "
            "`subdiv_min`, not snapshots of terminal descendant grids. Deeper refinement is "
            "used to decide spuriousness, but the hierarchy itself is not serialized. The DOT "
            "file stores graph structure and annotations; `mg_params_log.txt` stores only global "
            "settings and runtime.",
            "",
            "For a node with `N` saved boxes, the audit uses the exact upper bound "
            "`descendant_size(min + r) <= N * 2^r`. If that bound never exceeds the limit, "
            "the node necessarily has a surviving branch processed at `subdiv_max`. If `2N` "
            "already exceeds the limit, the first post-minimum work node definitely stopped at "
            "the limit. Between those cases, DOT/CSV/log files cannot distinguish a later limit "
            "hit from successful refinement to max.",
            "",
            "These statements follow the maintained CMGDB fork's limit test, SCC-subgrid "
            "construction, binary leaf subdivision, minimum-grid MorseGraph assignment, and "
            "CSV serializer. Exact source locations are recorded in the JSON audit under "
            "`interpretation.implementation_evidence`.",
            "",
            "## Reproducibility",
            "",
            f"- `morse_sets` SHA-256: `{audit['artifact_sha256']['morse_sets']}`",
            f"- `morse_graph` SHA-256: `{audit['artifact_sha256']['morse_graph']}`",
        ]
    )
    if audit["artifact_sha256"]["mg_params_log"]:
        lines.append(
            f"- `mg_params_log.txt` SHA-256: "
            f"`{audit['artifact_sha256']['mg_params_log']}`"
        )
    comparison = audit.get("comparison")
    if comparison:
        lines.extend(
            [
                "",
                "## Comparison run",
                "",
                f"- Run root: `{comparison['run_root']}`",
                f"- `morse_sets` byte-identical: "
                f"`{comparison['morse_sets_byte_identical']}`",
                f"- Morse graph semantically identical (nodes, edges, annotations): "
                f"`{comparison['morse_graph_semantically_identical']}`",
                f"- Morse graph byte-identical: `{comparison['morse_graph_byte_identical']}`",
                f"- DOT line multiset identical: "
                f"`{comparison['morse_graph_line_multiset_identical']}`",
            ]
        )
        if (
            comparison["morse_graph_semantically_identical"]
            and comparison["morse_graph_line_multiset_identical"]
            and not comparison["morse_graph_byte_identical"]
        ):
            lines.append(
                "- Interpretation: the DOT files differ only in line ordering; graph content is "
                "unchanged."
            )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="seed/run directory containing MG/ and mg_params_log.txt",
    )
    parser.add_argument("--json-out", type=Path, help="optional JSON output path")
    parser.add_argument("--markdown-out", type=Path, help="optional Markdown output path")
    parser.add_argument(
        "--compare-run-root",
        type=Path,
        help="optional second run root for byte and semantic artifact comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = build_audit(args.config, args.run_root, args.compare_run_root)
    json_text = json.dumps(audit, indent=2, sort_keys=False) + "\n"
    markdown_text = render_markdown(audit)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown_text)
    if not args.json_out and not args.markdown_out:
        print(markdown_text, end="")


if __name__ == "__main__":
    main()
