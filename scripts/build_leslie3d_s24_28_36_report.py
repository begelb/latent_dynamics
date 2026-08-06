#!/usr/bin/env python3
"""Build the Leslie3D (24, 28, 36), limit=10M Morse replay report.

This is intentionally a new builder.  It does not import, modify, or write to
the earlier max-30 report.  Graph outcomes, box counts, role assignments, and
subdivision settings are read from the completed replay artifacts and
cross-checked before any report output is written.

The script uses Matplotlib's multipage PDF backend so it runs in the project's
existing virtual environment without a separate document-generation runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import textwrap
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"

DEFAULT_CONFIG = (
    CODE_ROOT
    / "src"
    / "latentdynamics"
    / "configs"
    / "leslie3d_invariant_aware_v2_smooth_s24_28_36_limit10m.yaml"
)
DEFAULT_RUN_ROOT = (
    CODE_ROOT
    / "output"
    / "notebooks"
    / "leslie3d_invariant_aware_v2_smooth_s24_28_36_limit10m"
    / "seed_20260809"
)
DEFAULT_BASELINE_ROOT = (
    CODE_ROOT
    / "output"
    / "notebooks"
    / "leslie3d_invariant_aware_v2_smooth_max30"
    / "seed_20260809"
)
DEFAULT_REPORT_ROOT = (
    WORKSPACE_ROOT
    / "output"
    / "pdf"
    / "leslie3d_morse_report_s24_28_36_limit10m"
)
DEFAULT_DATA_MANIFEST = (
    CODE_ROOT / "data" / "leslie3d_invariant_aware_v2" / "dataset_manifest.json"
)
DEFAULT_TRAINING_SUMMARY = (
    CODE_ROOT
    / "output"
    / "leslie3d_invariant_aware_v2_smooth"
    / "seed_20260809"
    / "smooth_topology_summary.json"
)
DEFAULT_DIRECT_ROOT = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "ground_truth"
    / "absorbing_B_uniform_level33_recurrent_closure"
)
DEFAULT_RESIDUAL_JSON = (
    WORKSPACE_ROOT
    / "output"
    / "leslie3d_morse_report"
    / "analysis"
    / "sampled_residual_tolerance.json"
)
DEFAULT_RESIDUAL_MD = DEFAULT_RESIDUAL_JSON.with_suffix(".md")

EXPECTED_DATA_MANIFEST_SHA256 = (
    "658926337cc98e5e2d08ff9f442496c929e7400369deb7bc0382a6b73e87f5a1"
)
EXPECTED_TRAINING_SUMMARY_SHA256 = (
    "16f20b6fe689f34fb3cf4e85d826aa3e88b46cdc31d506cfeca0f078dde7252e"
)
EXPECTED_SUBDIVISION = (24, 28, 36)
EXPECTED_SUBDIV_LIMIT = 10_000_000

NAVY = "#16324F"
TEAL = "#007C83"
ORANGE = "#E77D22"
RED = "#A33A3A"
INK = "#18212B"
MID_GREY = "#667085"
LIGHT_GREY = "#F3F5F7"
LIGHT_BLUE = "#EAF1F7"
LIGHT_TEAL = "#E8F4F3"
LIGHT_ORANGE = "#FFF3E7"
LIGHT_RED = "#F9EAEA"
GRID_GREY = "#D5DBE1"


class ReportInputError(RuntimeError):
    """Raised when an input is missing or contradicts another source."""


@dataclass(frozen=True)
class DotGraph:
    node_indices: dict[int, tuple[str, ...]]
    edges: tuple[tuple[int, int], ...]
    minimal_nodes: tuple[int, ...]

    @property
    def nodes(self) -> tuple[int, ...]:
        return tuple(sorted(self.node_indices))


@dataclass(frozen=True)
class BaselineComparison:
    subdivision: tuple[int, int, int]
    subdiv_limit: int
    graph_semantically_identical: bool
    morse_sets_byte_identical: bool
    checkpoint_identical: bool


@dataclass
class ReportData:
    config_path: Path
    run_root: Path
    baseline_root: Path
    report_root: Path
    dataset_manifest_path: Path
    training_summary_path: Path
    direct_root: Path
    residual_json_path: Path
    residual_md_path: Path
    depth_audit_json_path: Path
    depth_audit_md_path: Path
    overlay_provenance_path: Path
    config: dict[str, Any]
    run_manifest: dict[str, Any]
    mg_params: dict[str, Any]
    analysis: dict[str, Any]
    dataset: dict[str, Any]
    training: dict[str, Any]
    direct: dict[str, Any]
    direct_display: dict[str, Any]
    residual: dict[str, Any]
    depth_audit: dict[str, Any]
    graph: DotGraph
    baseline_graph: DotGraph
    box_counts: dict[int, int]
    baseline: BaselineComparison
    overlay_provenance: dict[str, Any]
    overlay_png_source: Path
    overlay_output_sources: list[Path]
    source_hashes: dict[str, str]
    generated_at_utc: str


def _require_file(path: Path, purpose: str) -> Path:
    if not path.is_file():
        raise ReportInputError(f"missing {purpose}: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, purpose: str) -> dict[str, Any]:
    _require_file(path, purpose)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReportInputError(f"invalid JSON in {purpose} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportInputError(f"{purpose} must contain a JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    _require_file(path, "requested replay configuration")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ReportInputError(f"invalid YAML in replay configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportInputError(f"replay configuration must contain a mapping: {path}")
    return value


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ReportInputError(f"{label} mismatch: observed {actual!r}, expected {expected!r}")


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportInputError(f"{label} must be a mapping")
    return value


def _as_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReportInputError(f"{label} must be a list")
    return value


def _resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def _resolve_code_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (CODE_ROOT / path).resolve()


def _parse_mg_params(path: Path) -> dict[str, Any]:
    _require_file(path, "CMGDB parameter log")
    result: dict[str, Any] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if ":" not in line:
            raise ReportInputError(f"invalid parameter-log line {line_number}: {line!r}")
        key, raw = line.split(":", 1)
        key = key.strip()
        if not key or key in result:
            raise ReportInputError(f"invalid or duplicate parameter-log key at line {line_number}")
        try:
            result[key] = yaml.safe_load(raw.strip())
        except yaml.YAMLError as exc:
            raise ReportInputError(
                f"could not parse parameter-log value for {key!r}: {raw.strip()!r}"
            ) from exc
    return result


_NODE_RE = re.compile(r'^\s*(\d+)\s*\[label="\s*\d+\s*:\s*\((.*?)\)"')
_EDGE_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)\s*;")


def _parse_dot(path: Path) -> DotGraph:
    _require_file(path, "Morse graph DOT")
    node_indices: dict[int, tuple[str, ...]] = {}
    edges: set[tuple[int, int]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        node_match = _NODE_RE.match(line)
        if node_match:
            node = int(node_match.group(1))
            if node in node_indices:
                raise ReportInputError(f"duplicate node {node} in {path}:{line_number}")
            node_indices[node] = tuple(part.strip() for part in node_match.group(2).split(","))
            continue
        edge_match = _EDGE_RE.match(line)
        if edge_match:
            edges.add((int(edge_match.group(1)), int(edge_match.group(2))))
    if not node_indices:
        raise ReportInputError(f"no labeled Morse nodes parsed from {path}")
    unknown = sorted({node for edge in edges for node in edge} - set(node_indices))
    if unknown:
        raise ReportInputError(f"Morse graph edges reference unknown nodes {unknown}: {path}")
    minimal = tuple(sorted(set(node_indices) - {source for source, _ in edges}))
    graph = DotGraph(
        node_indices=node_indices,
        edges=tuple(sorted(edges)),
        minimal_nodes=minimal,
    )
    _validate_dag(graph, path)
    return graph


def _validate_dag(graph: DotGraph, path: Path) -> None:
    indegree = dict.fromkeys(graph.nodes, 0)
    children: dict[int, list[int]] = {node: [] for node in graph.nodes}
    for source, target in graph.edges:
        indegree[target] += 1
        children[source].append(target)
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for target in children[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(graph.nodes):
        raise ReportInputError(f"Morse graph is cyclic or malformed: {path}")


def _count_morse_boxes(path: Path) -> dict[int, int]:
    _require_file(path, "raw Morse-set CSV")
    counts: Counter[int] = Counter()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip(b"\r\n")
            if not line:
                raise ReportInputError(f"blank line in raw Morse-set CSV at row {line_number}")
            if line.count(b",") != 4:
                raise ReportInputError(
                    f"raw Morse-set row {line_number} must have five columns: {line[:120]!r}"
                )
            try:
                label = int(line.rsplit(b",", 1)[1])
            except ValueError as exc:
                raise ReportInputError(
                    f"invalid Morse-set label at row {line_number}: {line[-40:]!r}"
                ) from exc
            counts[label] += 1
    if not counts:
        raise ReportInputError(f"raw Morse-set CSV is empty: {path}")
    return dict(sorted(counts.items()))


def _check_config_sources(
    config: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    mg_params: Mapping[str, Any],
    run_root: Path,
) -> None:
    cmgdb = _as_mapping(config.get("cmgdb"), "configuration.cmgdb")
    requested = (
        int(cmgdb.get("subdiv_init")),
        int(cmgdb.get("subdiv_min")),
        int(cmgdb.get("subdiv_max")),
    )
    _expect_equal(requested, EXPECTED_SUBDIVISION, "requested subdivision tuple")
    _expect_equal(int(cmgdb.get("subdiv_limit")), EXPECTED_SUBDIV_LIMIT, "subdiv_limit")

    run_config = _as_mapping(run_manifest.get("config"), "run_manifest.config")
    run_cmgdb = _as_mapping(run_config.get("cmgdb"), "run_manifest.config.cmgdb")
    for key in (
        "subdiv_init",
        "subdiv_min",
        "subdiv_max",
        "subdiv_limit",
        "lower_bounds",
        "upper_bounds",
        "padding",
        "box_map_backend",
        "adaptive_precompute_subdiv",
        "precompute_batch_points",
        "compute_roa",
    ):
        _expect_equal(run_cmgdb.get(key), cmgdb.get(key), f"configuration vs run manifest: {key}")
    _expect_equal(run_config.get("experiment_name"), config.get("experiment_name"), "experiment name")
    _expect_equal(run_config.get("system"), config.get("system"), "system configuration")

    cell = _as_mapping(run_manifest.get("cell"), "run_manifest.cell")
    cell_output = _resolve_code_path(str(cell.get("output_dir")))
    _expect_equal(cell_output, run_root.resolve(), "run manifest output directory")
    _expect_equal(cell.get("device"), "cpu", "replay device")

    log_key = {"lower_bounds": "Lower bounds", "upper_bounds": "Upper bounds"}
    for key in ("subdiv_init", "subdiv_min", "subdiv_max", "subdiv_limit"):
        _expect_equal(mg_params.get(key), cmgdb.get(key), f"configuration vs parameter log: {key}")
    for key, log_name in log_key.items():
        _expect_equal(mg_params.get(log_name), cmgdb.get(key), f"configuration vs parameter log: {key}")
    for key in (
        "padding",
        "box_map_backend",
        "adaptive_precompute_subdiv",
        "precompute_batch_points",
        "compute_roa",
    ):
        _expect_equal(mg_params.get(key), cmgdb.get(key), f"configuration vs parameter log: {key}")
    duration = mg_params.get("duration_minutes")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ReportInputError("parameter log must record a positive duration_minutes")


def _check_graph_analysis(
    graph: DotGraph,
    box_counts: Mapping[int, int],
    analysis: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> None:
    comparison = _as_mapping(
        analysis.get("morse_graph_comparison"), "invariant analysis.morse_graph_comparison"
    )
    _expect_equal(tuple(sorted(int(v) for v in comparison.get("nodes", []))), graph.nodes, "DOT vs analysis nodes")
    analysis_edges = tuple(sorted((int(a), int(b)) for a, b in comparison.get("edges", [])))
    _expect_equal(analysis_edges, graph.edges, "DOT vs analysis edges")
    _expect_equal(
        tuple(sorted(int(v) for v in comparison.get("minimal_nodes", []))),
        graph.minimal_nodes,
        "DOT vs analysis minimal nodes",
    )
    analysis_indices = {
        int(node): tuple(str(value) for value in values)
        for node, values in _as_mapping(
            comparison.get("node_indices"), "analysis node_indices"
        ).items()
    }
    _expect_equal(analysis_indices, graph.node_indices, "DOT vs analysis Conley indices")
    _expect_equal(tuple(sorted(box_counts)), graph.nodes, "Morse-set labels vs DOT nodes")

    memberships = _as_mapping(analysis.get("morse_membership"), "analysis morse_membership")
    known_objects = _as_mapping(dataset.get("known_objects"), "dataset known_objects")
    _expect_equal(set(memberships), set(known_objects), "membership roles vs dataset roles")
    object_checks = _as_mapping(comparison.get("object_checks"), "analysis object_checks")
    _expect_equal(set(object_checks), set(known_objects), "object-check roles vs dataset roles")
    for role in known_objects:
        membership = _as_mapping(memberships[role], f"membership for {role}")
        check = _as_mapping(object_checks[role], f"object check for {role}")
        _expect_equal(
            membership.get("assigned_morse_node"), check.get("assigned_node"), f"assigned node for {role}"
        )
        node = int(check.get("assigned_node"))
        if node not in graph.node_indices:
            raise ReportInputError(f"{role} is assigned to absent Morse node {node}")
        _expect_equal(
            tuple(str(v) for v in check.get("observed_index", [])),
            graph.node_indices[node],
            f"observed index for {role}",
        )

    expected_edges = {
        (str(source), str(target))
        for source, target in _as_list(
            dataset.get("orbit_manifold_informed_reduced_edges"),
            "dataset orbit-manifold edge list",
        )
    }
    observed_checks = {
        (str(item["source_object"]), str(item["target_object"]))
        for item in _as_list(
            comparison.get("orbit_manifold_reachability_checks"),
            "analysis reachability checks",
        )
    }
    _expect_equal(observed_checks, expected_edges, "expected relation inventory")


def _validate_static_training_inputs(
    dataset_path: Path,
    dataset: Mapping[str, Any],
    training_path: Path,
    training: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    run_root: Path,
) -> None:
    dataset_hash = _sha256(dataset_path)
    training_hash = _sha256(training_path)
    _expect_equal(dataset_hash, EXPECTED_DATA_MANIFEST_SHA256, "pinned v2 dataset manifest hash")
    _expect_equal(training_hash, EXPECTED_TRAINING_SUMMARY_SHA256, "pinned training summary hash")
    _expect_equal(training.get("dataset_manifest_sha256"), dataset_hash, "training-to-dataset hash")

    train_csv = CODE_ROOT / "data" / "leslie3d_invariant_aware_v2" / "train.csv"
    val_csv = CODE_ROOT / "data" / "leslie3d_invariant_aware_v2" / "val.csv"
    _require_file(train_csv, "training CSV")
    _require_file(val_csv, "validation CSV")
    splits = _as_mapping(dataset.get("splits"), "dataset splits")
    _expect_equal(_sha256(train_csv), splits["train"]["csv_sha256"], "training CSV hash")
    _expect_equal(_sha256(val_csv), splits["validation"]["csv_sha256"], "validation CSV hash")
    _expect_equal(training.get("train_csv_sha256"), _sha256(train_csv), "training summary train hash")
    _expect_equal(training.get("validation_csv_sha256"), _sha256(val_csv), "training summary validation hash")

    checkpoint = run_root / "models" / "autoencoder.pt"
    sidecar = run_root / "models" / "autoencoder.json"
    _require_file(checkpoint, "accepted replay checkpoint")
    _require_file(sidecar, "accepted replay architecture sidecar")
    promoted_hashes = _as_mapping(
        training.get("promoted_checkpoint_sha256"), "training promoted_checkpoint_sha256"
    )
    _expect_equal(
        _sha256(checkpoint), promoted_hashes.get("autoencoder.pt"), "replay-to-training checkpoint hash"
    )
    _expect_equal(
        _sha256(sidecar),
        promoted_hashes.get("autoencoder.json"),
        "replay architecture sidecar hash",
    )

    artifacts = _as_mapping(run_manifest.get("artifacts"), "run manifest artifacts")
    _expect_equal(artifacts.get("train_csv_sha256"), _sha256(train_csv), "run manifest train hash")
    scaler = _resolve_code_path(str(artifacts.get("scaler")))
    _require_file(scaler, "replay scaler")
    _expect_equal(_sha256(scaler), artifacts.get("scaler_sha256"), "run manifest scaler hash")
    _expect_equal(_sha256(scaler), training.get("scaler_sha256"), "training summary scaler hash")


def _validate_direct_sources(
    dataset: Mapping[str, Any],
    direct: Mapping[str, Any],
    direct_path: Path,
    display: Mapping[str, Any],
) -> None:
    direct_source = _as_mapping(direct.get("source"), "direct ground-truth source")
    raw_direct = Path(str(direct_source.get("morse_sets"))).resolve()
    _require_file(raw_direct, "raw direct-system Morse sets")
    _expect_equal(_sha256(raw_direct), direct_source.get("morse_sets_sha256"), "direct raw-set hash")

    dataset_source = _as_mapping(
        dataset.get("direct_morse_sets_source"), "dataset direct_morse_sets_source"
    )
    _expect_equal(dataset_source.get("sha256"), _sha256(raw_direct), "dataset-to-direct raw-set hash")

    display_source = _as_mapping(display.get("source"), "direct display-cover source")
    uniform_manifest = _as_mapping(
        display_source.get("uniform_manifest"), "display-cover uniform_manifest"
    )
    _expect_equal(
        uniform_manifest.get("sha256"), _sha256(direct_path), "display-cover-to-direct-manifest hash"
    )
    display_csv = Path(str(display["cover"]["csv"]["path"])).resolve()
    _require_file(display_csv, "direct display-cover CSV")
    _expect_equal(_sha256(display_csv), display["cover"]["csv"]["sha256"], "display-cover CSV hash")

    roles = set(_as_mapping(dataset.get("known_objects"), "known objects"))
    direct_names = {
        "p_star" if name == "p*" else str(name)
        for name in _as_mapping(
            direct["saved_set_reachability_graph"].get("node_names"),
            "direct node names",
        ).values()
    }
    _expect_equal(direct_names, roles, "direct-system and dataset role names")


def _find_hash_entry(inputs: Mapping[str, Any], path: Path) -> str | None:
    resolved = path.resolve()
    for key, value in inputs.items():
        if _resolve_workspace_path(str(key)) == resolved:
            if isinstance(value, Mapping):
                hash_value = value.get("sha256")
            else:
                hash_value = value
            return None if hash_value is None else str(hash_value)
    return None


def _validate_overlay(
    provenance_path: Path,
    provenance: Mapping[str, Any],
    config: Mapping[str, Any],
    run_root: Path,
    direct_display: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> tuple[Path, list[Path]]:
    _expect_equal(
        provenance.get("configuration"), config.get("experiment_name"), "overlay configuration"
    )
    inputs = _as_mapping(provenance.get("input_sha256"), "overlay input_sha256")
    scaler = _resolve_code_path(str(run_manifest["artifacts"]["scaler"]))
    direct_csv = Path(str(direct_display["cover"]["csv"]["path"])).resolve()
    required = [
        run_root / "MG" / "morse_sets",
        run_root / "models" / "autoencoder.pt",
        run_root / "analysis" / "encoded_invariant_points.csv",
        direct_csv,
        scaler,
    ]
    for path in required:
        _require_file(path, f"overlay source {path.name}")
        recorded = _find_hash_entry(inputs, path)
        if recorded is None:
            raise ReportInputError(f"overlay provenance omits input hash for {path}")
        _expect_equal(recorded, _sha256(path), f"overlay input hash for {path.name}")

    outputs = _as_mapping(provenance.get("outputs"), "overlay outputs")
    output_paths: list[Path] = []
    png_candidates: list[Path] = []
    for name, raw in outputs.items():
        entry = _as_mapping(raw, f"overlay output {name}")
        output_path = Path(str(entry.get("path"))).resolve()
        _require_file(output_path, f"overlay output {name}")
        _expect_equal(_sha256(output_path), entry.get("sha256"), f"overlay output hash {name}")
        _expect_equal(output_path.stat().st_size, entry.get("size_bytes"), f"overlay output size {name}")
        output_paths.append(output_path)
        if output_path.suffix.lower() == ".png":
            png_candidates.append(output_path)
    if len(png_candidates) != 1:
        raise ReportInputError(
            f"overlay provenance must contain exactly one PNG output; found {len(png_candidates)}"
        )
    return png_candidates[0], output_paths


def _validate_residual_transfer(
    residual: Mapping[str, Any],
    residual_json_path: Path,
    current_graph: DotGraph,
    current_counts: Mapping[int, int],
    run_root: Path,
    baseline_root: Path,
    baseline_graph: DotGraph,
) -> None:
    _expect_equal(
        residual.get("status"),
        "finite_sample_diagnostic_not_a_certificate",
        "sampled residual/tolerance status",
    )
    _expect_equal(
        tuple(sorted(int(node) for node in residual.get("minimal_nodes", []))),
        current_graph.minimal_nodes,
        "residual minimal nodes vs current graph",
    )
    if current_graph.node_indices != baseline_graph.node_indices or current_graph.edges != baseline_graph.edges:
        raise ReportInputError(
            "cannot transfer sampled residual/tolerance: baseline and current Morse graphs differ semantically"
        )

    provenance = _as_mapping(residual.get("provenance"), "residual provenance")
    inputs = _as_mapping(provenance.get("inputs"), "residual provenance inputs")
    baseline_checkpoint = baseline_root / "models" / "autoencoder.pt"
    baseline_sets = baseline_root / "MG" / "morse_sets"
    baseline_dot = baseline_root / "MG" / "morse_graph"
    for path in (baseline_checkpoint, baseline_sets, baseline_dot):
        _require_file(path, f"residual source {path.name}")
        recorded = _find_hash_entry(inputs, path)
        if recorded is None:
            raise ReportInputError(f"residual provenance omits source hash for {path}")
        _expect_equal(recorded, _sha256(path), f"residual source hash for {path.name}")

    current_checkpoint = run_root / "models" / "autoencoder.pt"
    current_sets = run_root / "MG" / "morse_sets"
    _expect_equal(
        _sha256(current_checkpoint), _sha256(baseline_checkpoint), "residual transfer checkpoint hash"
    )
    _expect_equal(_sha256(current_sets), _sha256(baseline_sets), "residual transfer Morse-set hash")

    nodes = _as_mapping(residual.get("nodes"), "residual nodes")
    _expect_equal({int(node) for node in nodes}, set(current_graph.minimal_nodes), "residual node inventory")
    for node in current_graph.minimal_nodes:
        entry = _as_mapping(nodes[str(node)], f"residual node {node}")
        _expect_equal(entry.get("n_boxes"), current_counts[node], f"residual box count for node {node}")
        tolerance = _as_mapping(entry.get("tolerance"), f"residual tolerance for node {node}")
        residual_entry = _as_mapping(entry.get("residual"), f"sample residual for node {node}")
        comparison = _as_mapping(entry.get("comparison"), f"residual comparison for node {node}")
        for value, label in (
            (tolerance.get("sampled_minimum"), "sampled tolerance"),
            (residual_entry.get("sampled_maximum"), "sample residual"),
        ):
            if not isinstance(value, (int, float)) or value < 0:
                raise ReportInputError(f"{label} for node {node} must be finite and nonnegative")
        if comparison.get("conclusion") not in {"sampled_violation", "no_sampled_violation"}:
            raise ReportInputError(f"unrecognized residual comparison for node {node}")
    _require_file(residual_json_path, "sampled residual/tolerance JSON")


def _validate_depth_audit(
    audit: Mapping[str, Any],
    graph: DotGraph,
    box_counts: Mapping[int, int],
    run_root: Path,
    config_path: Path,
) -> None:
    parameters = _as_mapping(audit.get("parameters"), "refinement-depth parameters")
    _expect_equal(
        (
            int(parameters.get("subdiv_init")),
            int(parameters.get("subdiv_min")),
            int(parameters.get("subdiv_max")),
        ),
        EXPECTED_SUBDIVISION,
        "refinement-depth subdivision tuple",
    )
    _expect_equal(
        int(parameters.get("subdiv_limit")), EXPECTED_SUBDIV_LIMIT, "refinement-depth limit"
    )
    _expect_equal(audit.get("config_log_mismatches"), [], "refinement-depth config/log mismatches")

    inputs = _as_mapping(audit.get("inputs"), "refinement-depth inputs")
    _expect_equal(Path(str(inputs.get("config"))).resolve(), config_path.resolve(), "depth-audit config path")
    _expect_equal(Path(str(inputs.get("run_root"))).resolve(), run_root.resolve(), "depth-audit run root")
    artifacts = _as_mapping(audit.get("artifact_sha256"), "refinement-depth artifact hashes")
    for key, path in (
        ("morse_sets", run_root / "MG" / "morse_sets"),
        ("morse_graph", run_root / "MG" / "morse_graph"),
        ("mg_params_log", run_root / "mg_params_log.txt"),
    ):
        _expect_equal(artifacts.get(key), _sha256(path), f"refinement-depth {key} hash")

    audited_graph = _as_mapping(audit.get("graph"), "refinement-depth graph")
    _expect_equal(tuple(sorted(int(node) for node in audited_graph.get("nodes", []))), graph.nodes, "depth-audit nodes")
    _expect_equal(
        tuple(sorted((int(source), int(target)) for source, target in audited_graph.get("edges", []))),
        graph.edges,
        "depth-audit edges",
    )
    _expect_equal(
        tuple(sorted(int(node) for node in audited_graph.get("minimal_nodes", []))),
        graph.minimal_nodes,
        "depth-audit minimal nodes",
    )

    saved = _as_mapping(audit.get("saved_morse_sets"), "refinement-depth saved_morse_sets")
    _expect_equal(saved.get("all_saved_boxes_match_subdiv_min"), True, "saved box depth flag")
    _expect_equal(saved.get("observed_tree_depths_from_box_widths"), [28], "saved box tree depths")
    _expect_equal(saved.get("total_boxes"), sum(box_counts.values()), "depth-audit total boxes")
    audited_counts = {int(node): int(count) for node, count in saved.get("box_counts_by_node", {}).items()}
    _expect_equal(audited_counts, dict(box_counts), "depth-audit per-node box counts")

    refinement = _as_mapping(audit.get("refinement_audit"), "refinement-depth audit")
    node_rows = _as_list(refinement.get("nodes"), "refinement-depth node rows")
    _expect_equal({int(row["node"]) for row in node_rows}, set(graph.nodes), "depth-audit node inventory")
    for row in node_rows:
        node = int(row["node"])
        _expect_equal(
            int(row["boxes_at_saved_min_depth"]), box_counts[node], f"depth-audit boxes for node {node}"
        )
        guaranteed = int(row["guaranteed_descendant_grid_reached_depth_at_least"])
        if not EXPECTED_SUBDIVISION[1] <= guaranteed <= EXPECTED_SUBDIVISION[2]:
            raise ReportInputError(f"invalid guaranteed depth {guaranteed} for node {node}")
        processed = row.get("max_depth_processed")
        if processed is True:
            _expect_equal(guaranteed, EXPECTED_SUBDIVISION[2], f"max-depth guarantee for node {node}")
        elif processed is not None:
            raise ReportInputError(f"max_depth_processed for node {node} must be true or null")

    comparison = _as_mapping(audit.get("comparison"), "refinement-depth comparison")
    _expect_equal(comparison.get("morse_sets_byte_identical"), True, "depth-audit baseline set equality")
    _expect_equal(
        comparison.get("morse_graph_semantically_identical"), True, "depth-audit baseline graph equality"
    )


def _load_report_data(args: argparse.Namespace) -> ReportData:
    config_path = args.config.resolve()
    run_root = args.run_root.resolve()
    baseline_root = args.baseline_run_root.resolve()
    report_root = args.report_root.resolve()
    dataset_path = args.dataset_manifest.resolve()
    training_path = args.training_summary.resolve()
    direct_root = args.direct_root.resolve()
    residual_json_path = args.residual_json.resolve()
    residual_md_path = args.residual_markdown.resolve()
    depth_audit_json_path = run_root / "analysis" / "refinement_depth_audit.json"
    depth_audit_md_path = run_root / "analysis" / "refinement_depth_audit.md"
    overlay_provenance_path = (
        args.overlay_provenance.resolve()
        if args.overlay_provenance is not None
        else report_root / "assets" / "direct_ground_truth_overlay_provenance.json"
    )

    config = _load_yaml(config_path)
    run_manifest_path = run_root / "run_manifest.json"
    mg_params_path = run_root / "mg_params_log.txt"
    analysis_path = run_root / "analysis" / "invariant_aware_summary.json"
    dot_path = run_root / "MG" / "morse_graph"
    sets_path = run_root / "MG" / "morse_sets"
    run_manifest = _load_json(run_manifest_path, "new replay run manifest")
    mg_params = _parse_mg_params(mg_params_path)
    analysis = _load_json(analysis_path, "new replay invariant analysis")
    dataset = _load_json(dataset_path, "v2 dataset manifest")
    training = _load_json(training_path, "accepted training summary")
    direct_path = direct_root / "manifest.json"
    display_path = direct_root / "cubical_3d_level24_display_cover" / "manifest.json"
    direct = _load_json(direct_path, "direct-system ground-truth manifest")
    direct_display = _load_json(display_path, "direct-system display-cover manifest")
    residual = _load_json(residual_json_path, "sampled residual/tolerance audit")
    _require_file(residual_md_path, "sampled residual/tolerance Markdown")
    depth_audit = _load_json(depth_audit_json_path, "refinement-depth audit")
    _require_file(depth_audit_md_path, "refinement-depth Markdown")
    overlay_provenance = _load_json(overlay_provenance_path, "new replay overlay provenance")

    graph = _parse_dot(dot_path)
    box_counts = _count_morse_boxes(sets_path)
    baseline_graph = _parse_dot(baseline_root / "MG" / "morse_graph")
    baseline_params = _parse_mg_params(baseline_root / "mg_params_log.txt")

    _check_config_sources(config, run_manifest, mg_params, run_root)
    _validate_static_training_inputs(
        dataset_path, dataset, training_path, training, run_manifest, run_root
    )
    _check_graph_analysis(graph, box_counts, analysis, dataset)
    _validate_direct_sources(dataset, direct, direct_path, direct_display)

    _expect_equal(analysis.get("experiment"), config.get("experiment_name"), "analysis experiment")
    _expect_equal(
        Path(str(analysis.get("morse_directory"))).resolve(),
        (run_root / "MG").resolve(),
        "analysis Morse directory",
    )
    _expect_equal(
        Path(str(analysis["checkpoint"]["path"])).resolve(),
        (run_root / "models" / "autoencoder.pt").resolve(),
        "analysis checkpoint path",
    )
    _expect_equal(
        analysis.get("configured_cmgdb_bounds", {}).get("lower"),
        config["cmgdb"]["lower_bounds"],
        "analysis lower bounds",
    )
    _expect_equal(
        analysis.get("configured_cmgdb_bounds", {}).get("upper"),
        config["cmgdb"]["upper_bounds"],
        "analysis upper bounds",
    )

    current_checkpoint = run_root / "models" / "autoencoder.pt"
    baseline_checkpoint = baseline_root / "models" / "autoencoder.pt"
    current_sets = run_root / "MG" / "morse_sets"
    baseline_sets = baseline_root / "MG" / "morse_sets"
    baseline = BaselineComparison(
        subdivision=(
            int(baseline_params["subdiv_init"]),
            int(baseline_params["subdiv_min"]),
            int(baseline_params["subdiv_max"]),
        ),
        subdiv_limit=int(baseline_params["subdiv_limit"]),
        graph_semantically_identical=(
            graph.node_indices == baseline_graph.node_indices
            and graph.edges == baseline_graph.edges
            and graph.minimal_nodes == baseline_graph.minimal_nodes
        ),
        morse_sets_byte_identical=_sha256(current_sets) == _sha256(baseline_sets),
        checkpoint_identical=_sha256(current_checkpoint) == _sha256(baseline_checkpoint),
    )

    _validate_residual_transfer(
        residual,
        residual_json_path,
        graph,
        box_counts,
        run_root,
        baseline_root,
        baseline_graph,
    )
    _validate_depth_audit(depth_audit, graph, box_counts, run_root, config_path)
    overlay_png_source, overlay_output_sources = _validate_overlay(
        overlay_provenance_path,
        overlay_provenance,
        config,
        run_root,
        direct_display,
        run_manifest,
    )

    source_paths = [
        config_path,
        run_manifest_path,
        mg_params_path,
        analysis_path,
        dot_path,
        sets_path,
        run_root / "analysis" / "encoded_invariant_points.csv",
        current_checkpoint,
        run_root / "models" / "autoencoder.json",
        dataset_path,
        CODE_ROOT / "data" / "leslie3d_invariant_aware_v2" / "train.csv",
        CODE_ROOT / "data" / "leslie3d_invariant_aware_v2" / "val.csv",
        _resolve_code_path(str(run_manifest["artifacts"]["scaler"])),
        training_path,
        direct_path,
        Path(str(direct["source"]["morse_sets"])).resolve(),
        display_path,
        Path(str(direct_display["cover"]["csv"]["path"])).resolve(),
        direct_root / "paper_figure_pruned" / "morse_graph.png",
        direct_root
        / "cubical_3d_level24_display_cover"
        / "morse_sets_cubical_3d_labeled.png",
        baseline_root / "MG" / "morse_graph",
        baseline_sets,
        baseline_checkpoint,
        baseline_root / "mg_params_log.txt",
        residual_json_path,
        residual_md_path,
        depth_audit_json_path,
        depth_audit_md_path,
        overlay_provenance_path,
        *overlay_output_sources,
    ]
    source_hashes: dict[str, str] = {}
    for path in source_paths:
        _require_file(path, "report source")
        source_hashes[str(path.resolve())] = _sha256(path)

    return ReportData(
        config_path=config_path,
        run_root=run_root,
        baseline_root=baseline_root,
        report_root=report_root,
        dataset_manifest_path=dataset_path,
        training_summary_path=training_path,
        direct_root=direct_root,
        residual_json_path=residual_json_path,
        residual_md_path=residual_md_path,
        depth_audit_json_path=depth_audit_json_path,
        depth_audit_md_path=depth_audit_md_path,
        overlay_provenance_path=overlay_provenance_path,
        config=config,
        run_manifest=run_manifest,
        mg_params=mg_params,
        analysis=analysis,
        dataset=dataset,
        training=training,
        direct=direct,
        direct_display=direct_display,
        residual=residual,
        depth_audit=depth_audit,
        graph=graph,
        baseline_graph=baseline_graph,
        box_counts=box_counts,
        baseline=baseline,
        overlay_provenance=overlay_provenance,
        overlay_png_source=overlay_png_source,
        overlay_output_sources=overlay_output_sources,
        source_hashes=source_hashes,
        generated_at_utc=datetime.now(UTC).isoformat(),
    )


def _copy_if_needed(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def _prepare_bundle(data: ReportData) -> dict[str, Path]:
    assets = data.report_root / "assets"
    analysis_dir = data.report_root / "analysis"
    assets.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "learned_morse_sets.png": data.run_root / "MG" / "morse_sets.png",
        "learned_morse_graph.png": data.run_root / "MG" / "morse_graph.png",
        "encoded_invariants_on_morse_sets.png": data.run_root
        / "MG"
        / "encoded_invariants_on_morse_sets.png",
        "direct_ground_truth_morse_graph.png": data.direct_root
        / "paper_figure_pruned"
        / "morse_graph.png",
        "direct_ground_truth_3d_display_cover.png": data.direct_root
        / "cubical_3d_level24_display_cover"
        / "morse_sets_cubical_3d_labeled.png",
    }
    copied: dict[str, Path] = {}
    for name, source in sources.items():
        _require_file(source, f"report figure {name}")
        destination = assets / name
        _copy_if_needed(source, destination)
        copied[name] = destination

    for source in data.overlay_output_sources:
        destination = assets / source.name
        _copy_if_needed(source, destination)
        copied[source.name] = destination
    _copy_if_needed(
        data.overlay_provenance_path, assets / "direct_ground_truth_overlay_provenance.json"
    )
    copied["direct_ground_truth_overlay_provenance.json"] = (
        assets / "direct_ground_truth_overlay_provenance.json"
    )

    residual_json_destination = analysis_dir / "sampled_residual_tolerance.json"
    residual_md_destination = analysis_dir / "sampled_residual_tolerance.md"
    _copy_if_needed(data.residual_json_path, residual_json_destination)
    _copy_if_needed(data.residual_md_path, residual_md_destination)
    copied["sampled_residual_tolerance.json"] = residual_json_destination
    copied["sampled_residual_tolerance.md"] = residual_md_destination

    depth_json_destination = analysis_dir / "refinement_depth_audit.json"
    depth_md_destination = analysis_dir / "refinement_depth_audit.md"
    _copy_if_needed(data.depth_audit_json_path, depth_json_destination)
    _copy_if_needed(data.depth_audit_md_path, depth_md_destination)
    copied[depth_json_destination.name] = depth_json_destination
    copied[depth_md_destination.name] = depth_md_destination

    transfer = {
        "schema_version": 1,
        "status": "transferred_without_resampling",
        "reason": (
            "The requested replay has the identical checkpoint and byte-identical raw "
            "Morse-set CSV as the source audit, and its DOT graph is semantically identical."
        ),
        "source_audit": str(data.residual_json_path),
        "source_audit_sha256": _sha256(data.residual_json_path),
        "source_run": str(data.baseline_root),
        "target_run": str(data.run_root),
        "hash_equalities": {
            "checkpoint_sha256": _sha256(data.run_root / "models" / "autoencoder.pt"),
            "morse_sets_sha256": _sha256(data.run_root / "MG" / "morse_sets"),
        },
        "semantic_graph_equality": data.baseline.graph_semantically_identical,
        "minimal_nodes": list(data.graph.minimal_nodes),
        "generated_at_utc": data.generated_at_utc,
    }
    transfer_path = analysis_dir / "residual_tolerance_transfer_provenance.json"
    transfer_path.write_text(json.dumps(transfer, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    copied[transfer_path.name] = transfer_path
    return copied


def _fmt_index(values: Sequence[str]) -> str:
    return "(" + ", ".join(values) + ")"


def _fmt_num(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.6g}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _promoted_checkpoint_hash(data: ReportData) -> str:
    return str(data.training["promoted_checkpoint_sha256"]["autoencoder.pt"])


def _role_assignments(data: ReportData) -> dict[str, int]:
    return {
        role: int(entry["assigned_morse_node"])
        for role, entry in data.analysis["morse_membership"].items()
    }


def _outcome(data: ReportData) -> str:
    comparison = data.analysis["morse_graph_comparison"]
    checks = comparison["object_checks"]
    index_matches = sum(bool(check["index_matches"]) for check in checks.values())
    relation_checks = comparison["orbit_manifold_reachability_checks"]
    reachable = sum(bool(check["reachable"]) for check in relation_checks)
    return (
        f"The requested replay completed in {_fmt_num(float(data.mg_params['duration_minutes']))} minutes "
        f"and saved {sum(data.box_counts.values()):,} boxes in {len(data.graph.nodes)} Morse sets. "
        f"All {len(checks)} named invariant objects are uniquely assigned to distinct sets; "
        f"{index_matches}/{len(checks)} requested latent indices and "
        f"{reachable}/{len(relation_checks)} requested directed relations are recovered. "
        f"Exact role-aligned graph match: {'yes' if comparison['exact_role_aligned_morse_graph_match'] else 'no'}."
    )


def _draw_footer(fig: plt.Figure, page: int, generated_date: str) -> None:
    fig.lines.append(
        plt.Line2D([0.07, 0.93], [0.045, 0.045], transform=fig.transFigure, color=GRID_GREY, lw=0.7)
    )
    fig.text(0.07, 0.025, f"Generated {generated_date}", fontsize=6.8, color=MID_GREY)
    fig.text(0.93, 0.025, f"Page {page}", fontsize=6.8, color=MID_GREY, ha="right")


def _new_page(
    page: int,
    title: str,
    generated_date: str,
    *,
    subtitle: str | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.07, 0.972, "Leslie3D invariant-aware Morse replay", fontsize=7.2, color=MID_GREY)
    fig.text(0.93, 0.972, "Numerical experiment — not a Conley certificate", fontsize=7.2, color=MID_GREY, ha="right")
    fig.lines.append(
        plt.Line2D([0.07, 0.93], [0.96, 0.96], transform=fig.transFigure, color=GRID_GREY, lw=0.7)
    )
    fig.text(0.07, 0.925, title, fontsize=18, fontweight="bold", color=NAVY, va="top")
    if subtitle:
        fig.text(0.07, 0.89, subtitle, fontsize=9.2, color=MID_GREY, va="top")
    _draw_footer(fig, page, generated_date)
    return fig


def _paragraph(
    fig: plt.Figure,
    x: float,
    y: float,
    text: str,
    *,
    width: int = 94,
    fontsize: float = 8.5,
    color: str = INK,
    weight: str = "normal",
    line_spacing: float = 1.32,
) -> float:
    wrapped = textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)
    lines = wrapped.count("\n") + 1
    fig.text(
        x,
        y,
        wrapped,
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        va="top",
        linespacing=line_spacing,
    )
    line_height = (fontsize / 72.0) / 11.69 * line_spacing
    return y - lines * line_height


def _heading(fig: plt.Figure, y: float, text: str) -> float:
    fig.text(0.07, y, text, fontsize=11.5, fontweight="bold", color=TEAL, va="top")
    return y - 0.028


def _callout(
    fig: plt.Figure,
    y: float,
    text: str,
    *,
    background: str,
    border: str,
    height: float = 0.085,
) -> float:
    patch = FancyBboxPatch(
        (0.07, y - height),
        0.86,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.006",
        transform=fig.transFigure,
        facecolor=background,
        edgecolor=border,
        linewidth=0.8,
    )
    fig.patches.append(patch)
    _paragraph(fig, 0.085, y - 0.015, text, width=102, fontsize=8.1)
    return y - height - 0.012


def _table(
    fig: plt.Figure,
    y_top: float,
    rows: Sequence[Sequence[Any]],
    *,
    col_widths: Sequence[float] | None = None,
    height: float | None = None,
    font_size: float = 7.1,
) -> float:
    if not rows:
        return y_top
    if height is None:
        height = min(0.31, 0.035 + 0.032 * len(rows))
    ax = fig.add_axes([0.07, y_top - height, 0.86, height])
    ax.axis("off")
    materialized = [[str(value) for value in row] for row in rows]
    table = ax.table(
        cellText=materialized[1:],
        colLabels=materialized[0],
        colWidths=col_widths,
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_GREY)
        cell.set_linewidth(0.45)
        cell.PAD = 0.045
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else LIGHT_GREY)
            cell.get_text().set_color(INK)
    return y_top - height - 0.014


def _image(fig: plt.Figure, path: Path, box: tuple[float, float, float, float]) -> None:
    ax = fig.add_axes(list(box))
    ax.imshow(mpimg.imread(path))
    ax.axis("off")


def _component_rows(data: ReportData) -> list[list[str]]:
    splits = data.dataset["splits"]
    train = {entry["name"]: entry for entry in splits["train"]["components"]}
    validation = {entry["name"]: entry for entry in splits["validation"]["components"]}
    order = list(train) + [name for name in validation if name not in train]
    rows: list[list[str]] = [["Component", "Train", "Held out", "Recorded construction"]]
    for name in order:
        train_entry = train.get(name, {})
        val_entry = validation.get(name, {})
        detail_source = train_entry or val_entry
        if "route" in detail_source:
            detail = " → ".join(detail_source["route"])
        elif "trajectories" in detail_source and "steps" in detail_source:
            detail = f"{detail_source['trajectories']} trajectories × {detail_source['steps']} steps (train)"  # noqa: RUF001
        elif "initial_conditions" in detail_source:
            detail = f"{detail_source['initial_conditions']} starts × {detail_source['steps']} steps (train)"  # noqa: RUF001
        elif "unique_phases" in detail_source:
            detail = f"{detail_source['unique_phases']} phases × {detail_source['repeats_per_phase']} repeats"  # noqa: RUF001
        elif "scaled_radius_range" in detail_source:
            low, high = detail_source["scaled_radius_range"]
            detail = f"scaled radii {_fmt_num(float(low))} to {_fmt_num(float(high))}"
        elif "samples_per_nonorigin_node" in detail_source:
            detail = f"{detail_source['samples_per_nonorigin_node']} samples/non-origin node"
        else:
            detail = "exact analytic successors"
        rows.append(
            [
                name.replace("_", " "),
                f"{int(train_entry.get('rows', 0)):,}" if train_entry else "—",
                f"{int(val_entry.get('rows', 0)):,}" if val_entry else "—",
                detail,
            ]
        )
    rows.append(
        [
            "Total",
            f"{int(splits['train']['rows']):,}",
            f"{int(splits['validation']['rows']):,}",
            data.dataset["transition_policy"],
        ]
    )
    return rows


def _network_rows(data: ReportData) -> list[list[str]]:
    arch = data.run_manifest["config"]["arch"]
    frozen = set(data.training["frozen_components"])
    optimized = set(data.training["optimized_components"])
    specs = [
        ("encoder", "Encoder E", arch["high_dims"], arch["low_dims"]),
        ("latent_map", "Latent map g", arch["low_dims"], arch["low_dims"]),
        ("decoder", "Decoder D", arch["low_dims"], arch["high_dims"]),
    ]
    rows = [["Component", "Architecture", "Activation / output", "Accepted training"]]
    for key, label, input_dim, output_dim in specs:
        spec = arch[key]
        shape = "-".join(str(value) for value in [input_dim, *spec["hidden_shapes"], output_dim])
        status = "frozen" if key in frozen else "optimized" if key in optimized else "recorded"
        rows.append([label, shape, f"{spec['activation']} / {spec['out_activation']}", status])
    return rows


def _role_rows(data: ReportData) -> list[list[str]]:
    checks = data.analysis["morse_graph_comparison"]["object_checks"]
    known = data.dataset["known_objects"]
    rows = [["Object", "Period", "Node", "Observed index", "Expected index", "Match"]]
    for role in known:
        check = checks[role]
        rows.append(
            [
                role,
                str(len(known[role]["points"])),
                str(check["assigned_node"]),
                _fmt_index(check["observed_index"]),
                _fmt_index(check["expected_latent_index"]),
                "yes" if check["index_matches"] else "no",
            ]
        )
    return rows


def _box_rows(data: ReportData) -> list[list[str]]:
    assignments = _role_assignments(data)
    by_node: dict[int, list[str]] = {node: [] for node in data.graph.nodes}
    for role, node in assignments.items():
        by_node[node].append(role)
    rows = [["Node", "Saved boxes", "Index", "Named object", "Minimal"]]
    for node in data.graph.nodes:
        rows.append(
            [
                str(node),
                f"{data.box_counts[node]:,}",
                _fmt_index(data.graph.node_indices[node]),
                ", ".join(by_node[node]) or "extra",
                "yes" if node in data.graph.minimal_nodes else "no",
            ]
        )
    rows.append(["Total", f"{sum(data.box_counts.values()):,}", "", "", ""])
    return rows


def _depth_rows(data: ReportData) -> list[list[str]]:
    rows = [["Node", "Boxes at min", "Worst-case at max", "Earliest possible limit", "Guaranteed reached", "Conclusion"]]
    for entry in data.depth_audit["refinement_audit"]["nodes"]:
        earliest = entry["earliest_depth_at_which_limit_could_be_exceeded"]
        processed = entry["max_depth_processed"]
        rows.append(
            [
                str(entry["node"]),
                f"{int(entry['boxes_at_saved_min_depth']):,}",
                f"{int(entry['worst_case_descendant_size_at_max']):,}",
                "—" if earliest is None else str(earliest),
                str(entry["guaranteed_descendant_grid_reached_depth_at_least"]),
                "max 36 processed" if processed is True else "later stop vs max unknown",
            ]
        )
    return rows


def _residual_rows(data: ReportData) -> list[list[str]]:
    rows = [["Minimal node", "Role", "R-hat", "tau-hat", "R-hat / tau-hat", "Result"]]
    for node in data.graph.minimal_nodes:
        entry = data.residual["nodes"][str(node)]
        rows.append(
            [
                str(node),
                entry["role"],
                _fmt_num(float(entry["residual"]["sampled_maximum"])),
                _fmt_num(float(entry["tolerance"]["sampled_minimum"])),
                f"{float(entry['comparison']['sampled_residual_over_sampled_tolerance']):,.1f}",
                str(entry["comparison"]["conclusion"]).replace("_", " "),
            ]
        )
    return rows


def _relation_rows(data: ReportData) -> list[list[str]]:
    rows = [["Expected direct relation", "Latent nodes", "Reachable"]]
    for check in data.analysis["morse_graph_comparison"]["orbit_manifold_reachability_checks"]:
        rows.append(
            [
                f"{check['source_object']} → {check['target_object']}",
                f"{check['source_node']} → {check['target_node']}",
                "yes" if check["reachable"] else "no",
            ]
        )
    return rows


def _write_pdf(data: ReportData, assets: Mapping[str, Path], pdf_path: Path) -> None:
    generated_date = data.generated_at_utc[:10]
    comparison = data.analysis["morse_graph_comparison"]
    selected = data.training["selected"]
    replay = selected["replay"]
    overlay_png = assets[data.overlay_png_source.name]

    metadata = {
        "Title": "Leslie3D Morse replay: subdivision (24, 28, 36), limit 10,000,000",
        "Author": "Codex",
        "Subject": "Data, training, Morse sets, Morse graph, ground-truth overlay, and sampled residual/tolerance",
        "Keywords": "Leslie3D, Morse graph, Conley index, latent dynamics",
    }
    with PdfPages(pdf_path, metadata=metadata) as pdf:
        # Page 1: executive result.
        fig = _new_page(
            1,
            "High-resolution Morse replay",
            generated_date,
            subtitle="Subdivision (24, 28, 36) · subdiv_limit 10,000,000 · accepted checkpoint held fixed",
        )
        y = _callout(
            fig,
            0.845,
            _outcome(data),
            background=LIGHT_RED if not comparison["exact_role_aligned_morse_graph_match"] else LIGHT_TEAL,
            border=RED if not comparison["exact_role_aligned_morse_graph_match"] else TEAL,
            height=0.105,
        )
        _image(fig, assets["learned_morse_sets.png"], (0.07, 0.38, 0.58, 0.34))
        _image(fig, assets["learned_morse_graph.png"], (0.68, 0.38, 0.25, 0.34))
        fig.text(0.36, 0.36, "Learned latent Morse sets", fontsize=7.2, color=MID_GREY, ha="center")
        fig.text(0.805, 0.36, "Learned Morse graph", fontsize=7.2, color=MID_GREY, ha="center")
        y = 0.32
        baseline_text = (
            f"Comparison with subdivision {data.baseline.subdivision}, limit {data.baseline.subdiv_limit:,}: "
            f"graph semantics {'match' if data.baseline.graph_semantically_identical else 'do not match'}; "
            f"raw Morse sets are {'byte-identical' if data.baseline.morse_sets_byte_identical else 'different'}; "
            f"checkpoint hashes {'match' if data.baseline.checkpoint_identical else 'do not match'}."
        )
        y = _paragraph(fig, 0.07, y, baseline_text, width=104, fontsize=8.3)
        _paragraph(
            fig,
            0.07,
            y - 0.012,
            "Scope: the neural box map evaluates sampled corner images with padding. The result is a reproducible numerical outer-map heuristic, not an outward-rounded interval enclosure of every point in each box.",
            width=104,
            fontsize=8.0,
            color=MID_GREY,
        )
        pdf.savefig(fig, bbox_inches=None)
        plt.close(fig)

        # Page 2: construction and training.
        fig = _new_page(2, "1. Data construction and accepted training", generated_date)
        y = 0.875
        theta = data.dataset["parameters"]["theta"]
        survival = data.dataset["parameters"]["survival"]
        y = _paragraph(
            fig,
            0.07,
            y,
            "Physical map: f(x1,x2,x3) = ((theta1 x1 + theta2 x2 + theta3 x3) exp[-0.1(x1+x2+x3)], p1 x1, p2 x2), "
            f"with theta={theta}, p={survival}, on {data.dataset['domain']['lower']} to {data.dataset['domain']['upper']}.",
            width=105,
            fontsize=8.2,
        )
        y = _heading(fig, y - 0.012, "Exact analytic pair inventory")
        y = _table(
            fig,
            y,
            _component_rows(data),
            col_widths=[0.33, 0.11, 0.12, 0.44],
            height=0.315,
            font_size=6.25,
        )
        y = _heading(fig, y, "Network and optimization")
        y = _table(
            fig,
            y,
            _network_rows(data),
            col_widths=[0.20, 0.22, 0.34, 0.24],
            height=0.13,
            font_size=7.0,
        )
        training_rows = [
            ["Recorded diagnostic", "Accepted value"],
            ["Selected epoch / epochs run", f"{data.training['best_epoch']} / {data.training['epochs_run']}"],
            ["Validation reconstruction MSE", _fmt_num(float(replay["reconstruction"]))],
            ["Validation prediction MSE", _fmt_num(float(replay["prediction"]))],
            ["Validation semiconjugacy MSE", _fmt_num(float(replay["semiconjugacy"]))],
            ["Validation cycle MSE", _fmt_num(float(replay["cycle"]))],
            ["Maximum normalized anchor error", _fmt_num(float(selected["max_anchor_normalized_l2"]))],
            ["Maximum characteristic relative error", _fmt_num(float(selected["max_characteristic_relative_error"]))],
            ["Global trust RMSE", _fmt_num(float(selected["trust_global_rmse"]))],
            ["Promoted checkpoint SHA-256", _promoted_checkpoint_hash(data)],
        ]
        _table(
            fig,
            y,
            training_rows,
            col_widths=[0.38, 0.62],
            height=0.24,
            font_size=6.6,
        )
        pdf.savefig(fig, bbox_inches=None)
        plt.close(fig)

        # Page 3: requested settings and the honest attained-depth audit.
        fig = _new_page(3, "2. Morse parameters and attained-depth audit", generated_date)
        y = 0.875
        cmgdb_rows = [
            ["Item", "Recorded value"],
            ["Subdivision (init, min, max)", str(EXPECTED_SUBDIVISION)],
            ["Per-component subdiv_limit", f"{EXPECTED_SUBDIV_LIMIT:,}"],
            ["Latent lower bounds", str(data.config["cmgdb"]["lower_bounds"])],
            ["Latent upper bounds", str(data.config["cmgdb"]["upper_bounds"])],
            ["Box-map backend", str(data.mg_params["box_map_backend"])],
            ["Dense precompute level", str(data.mg_params["adaptive_precompute_subdiv"])],
            ["Padding", str(data.mg_params["padding"])],
            ["CMGDB version / device", f"{data.run_manifest['cmgdb_version']} / {data.run_manifest['cell']['device']}"],
            ["Recorded CMGDB duration", f"{_fmt_num(float(data.mg_params['duration_minutes']))} minutes"],
        ]
        y = _table(fig, y, cmgdb_rows, col_widths=[0.36, 0.64], height=0.245, font_size=6.8)
        processed_nodes = [
            int(entry["node"])
            for entry in data.depth_audit["refinement_audit"]["nodes"]
            if entry["max_depth_processed"] is True
        ]
        uncertain_depths = [
            f"node {entry['node']} ≥{entry['guaranteed_descendant_grid_reached_depth_at_least']}"
            for entry in data.depth_audit["refinement_audit"]["nodes"]
            if entry["max_depth_processed"] is None
        ]
        y = _callout(
            fig,
            y,
            f"Attained-depth result. Nodes {', '.join(map(str, processed_nodes))} are proved to have been processed through max 36. "
            f"For {', '.join(uncertain_depths)}, the saved DOT/CSV/log prove only the stated lower bound; a later limit hit versus successful processing through 36 is not identifiable because CMGDB does not serialize the deeper hierarchy.",
            background=LIGHT_ORANGE,
            border=ORANGE,
            height=0.105,
        )
        y = _heading(fig, y, "Node-level cardinality bounds")
        y = _table(
            fig,
            y,
            _depth_rows(data),
            col_widths=[0.07, 0.14, 0.18, 0.18, 0.16, 0.27],
            height=0.26,
            font_size=5.75,
        )
        _paragraph(
            fig,
            0.07,
            y,
            "Every saved Morse-set row is at tree depth 28. Those rows are the minimum-grid component covers, not terminal descendant snapshots. The audit uses descendant_size(min+r) ≤ boxes_at_min × 2^r together with the maintained CMGDB limit semantics; full implementation evidence is preserved in analysis/refinement_depth_audit.json.",  # noqa: RUF001
            width=105,
            fontsize=7.9,
            color=MID_GREY,
        )
        pdf.savefig(fig, bbox_inches=None)
        plt.close(fig)

        # Page 4: saved sets, graph, and role/index audit.
        fig = _new_page(4, "3. Saved Morse sets and graph audit", generated_date)
        y = 0.875
        y = _table(
            fig,
            y,
            _box_rows(data),
            col_widths=[0.08, 0.17, 0.29, 0.29, 0.17],
            height=0.265,
            font_size=6.7,
        )
        y = _heading(fig, y, "Reduced graph")
        edges_text = ", ".join(f"{source}→{target}" for source, target in data.graph.edges)
        y = _paragraph(
            fig,
            0.07,
            y,
            f"Reduced edges: {edges_text}. Minimal nodes: {', '.join(map(str, data.graph.minimal_nodes))}.",
            width=105,
            fontsize=8.1,
        )
        y = _heading(fig, y - 0.008, "Named-object index audit")
        y = _table(
            fig,
            y,
            _role_rows(data),
            col_widths=[0.11, 0.09, 0.09, 0.27, 0.29, 0.15],
            height=0.19,
            font_size=6.3,
        )
        y = _heading(fig, y, "Requested direct-system relations")
        _table(
            fig,
            y,
            _relation_rows(data),
            col_widths=[0.48, 0.25, 0.27],
            height=0.19,
            font_size=7.0,
        )
        pdf.savefig(fig, bbox_inches=None)
        plt.close(fig)

        # Page 5: requested direct-system overlay.
        fig = _new_page(5, "4. Direct ground truth over learned Morse sets", generated_date)
        y = _paragraph(
            fig,
            0.07,
            0.875,
            "The plot combines all saved learned latent Morse boxes, encoded centers of the direct-system render-only display cover, and the exact fixed/periodic phases. The display-cover centers are samples of E(M), not certified enclosures of encoded three-dimensional cells.",
            width=105,
            fontsize=8.2,
        )
        _image(fig, overlay_png, (0.08, 0.30, 0.84, 0.49))
        fig.text(
            0.08,
            0.285,
            "Direct-system display-cover centers and exact invariant phases over the requested learned Morse sets.",
            fontsize=7.1,
            color=MID_GREY,
        )
        membership_rows = [["Object", "Phases", "Assigned node", "One unique Morse set"]]
        for role, known in data.dataset["known_objects"].items():
            membership = data.analysis["morse_membership"][role]
            membership_rows.append(
                [
                    role,
                    str(len(known["points"])),
                    str(membership["assigned_morse_node"]),
                    "yes" if membership["all_phases_in_one_unique_morse_set"] else "no",
                ]
            )
        _table(
            fig,
            0.25,
            membership_rows,
            col_widths=[0.22, 0.16, 0.26, 0.36],
            height=0.18,
            font_size=6.9,
        )
        pdf.savefig(fig, bbox_inches=None)
        plt.close(fig)

        # Page 6: sampled residual/tolerance transfer and relation audit.
        fig = _new_page(6, "5. Minimal-node residual and tolerance audit", generated_date)
        y = 0.875
        y = _callout(
            fig,
            y,
            "Transferred without resampling. The source audit applies exactly because the accepted checkpoint SHA-256 and raw Morse-set SHA-256 are identical, while the current and source DOT graphs have the same nodes, indices, minimal nodes, and directed edges. Transfer provenance is saved in analysis/residual_tolerance_transfer_provenance.json.",
            background=LIGHT_BLUE,
            border=NAVY,
            height=0.105,
        )
        y = _heading(fig, y, "Finite-sample comparison")
        y = _table(
            fig,
            y,
            _residual_rows(data),
            col_widths=[0.12, 0.31, 0.14, 0.14, 0.16, 0.13],
            height=0.115,
            font_size=6.7,
        )
        y = _paragraph(
            fig,
            0.07,
            y,
            "Here R-hat is a sampled maximum of ||g(E(x))−E(f(x))||2 over accepted physical witnesses, while tau-hat is a sampled minimum clearance of g(z) from the complement of the candidate block interior. A sampled maximum lower-bounds the exact residual; a sampled minimum is an upper estimate of the exact tolerance. Therefore R-hat ≥ tau-hat supplies a numerical counter-witness to the strict sufficient lifting inequality for the evaluated candidate block.",  # noqa: RUF001
            width=105,
            fontsize=8.0,
        )
        y = _heading(fig, y - 0.012, "Interpretation")
        missing = [
            f"{check['source_object']}→{check['target_object']}"
            for check in comparison["orbit_manifold_reachability_checks"]
            if not check["reachable"]
        ]
        _callout(
            fig,
            y,
            "The six exact phases are all located, but phase membership is weaker than a global topological match. "
            + (f"Missing requested relation(s): {', '.join(missing)}." if missing else "All requested relations are reachable."),
            background=LIGHT_ORANGE if missing else LIGHT_TEAL,
            border=ORANGE if missing else TEAL,
            height=0.075,
        )
        pdf.savefig(fig, bbox_inches=None)
        plt.close(fig)

        # Page 7: interpretation and provenance.
        fig = _new_page(7, "6. Interpretation, limits, and provenance", generated_date)
        y = 0.875
        y = _heading(fig, y, "Conditional ideal-data statement")
        y = _paragraph(
            fig,
            0.07,
            y,
            "If continuous E, D, and g attain zero population reconstruction and semiconjugacy loss on a full-support measure over compact K, then D(E(x))=x and g(E(x))=E(f(x)) pointwise. Thus E is injective and is a homeomorphism from K to E(K), so the restricted dynamics are conjugate.",
            width=105,
            fontsize=8.4,
        )
        y = _paragraph(
            fig,
            0.07,
            y - 0.008,
            "Conley-index recovery additionally requires transport of a valid index pair, a latent isolating neighborhood containing exactly the encoded invariant set with no extra recurrence, and conjugate quotient index maps (or a certified shift equivalence). Semiconjugacy alone can collapse invariant sets and does not imply these conditions.",
            width=105,
            fontsize=8.4,
        )
        y = _heading(fig, y - 0.012, "What this replay establishes")
        conclusions = [
            _outcome(data),
            (
                "The requested max-36 replay did not change the saved sets or graph relative to the accepted max-30 source. The depth audit proves five nodes were processed through 36; three larger nodes have only rigorous lower bounds because exact terminal depths are not serialized."
                if data.baseline.graph_semantically_identical and data.baseline.morse_sets_byte_identical
                else "The high-resolution replay differs from the accepted max-30 source; see the manifest comparison fields."
            ),
            "The direct-system overlay confirms pointwise placement of the named phases, but it is sampled visual evidence rather than transport of an index pair.",
            "The transferred residual/tolerance witnesses contradict the strict sampled lifting inequality on both minimal-node candidate blocks.",
        ]
        for conclusion in conclusions:
            y = _paragraph(fig, 0.085, y, "• " + conclusion, width=101, fontsize=8.0)
            y -= 0.005
        y = _heading(fig, y - 0.008, "Primary provenance")
        provenance_rows = [
            ["Artifact", "SHA-256"],
            ["Accepted checkpoint", _sha256(data.run_root / "models" / "autoencoder.pt")],
            ["Requested raw Morse sets", _sha256(data.run_root / "MG" / "morse_sets")],
            ["Requested DOT graph", _sha256(data.run_root / "MG" / "morse_graph")],
            ["V2 dataset manifest", _sha256(data.dataset_manifest_path)],
            ["Accepted training summary", _sha256(data.training_summary_path)],
            ["Ground-truth overlay provenance", _sha256(data.overlay_provenance_path)],
            ["Transferred residual/tolerance JSON", _sha256(data.residual_json_path)],
        ]
        y = _table(
            fig,
            y,
            provenance_rows,
            col_widths=[0.36, 0.64],
            height=0.225,
            font_size=6.15,
        )
        _paragraph(
            fig,
            0.07,
            y,
            "Machine-readable hashes, graph facts, transfer checks, and output identifiers are recorded in report_manifest.json. The copied sampled audit remains labeled as a finite-sample diagnostic, not a certificate.",
            width=105,
            fontsize=7.8,
            color=MID_GREY,
        )
        pdf.savefig(fig, bbox_inches=None)
        plt.close(fig)


def _markdown_table(rows: Sequence[Sequence[Any]]) -> str:
    materialized = [[str(value).replace("|", "\\|") for value in row] for row in rows]
    header = "| " + " | ".join(materialized[0]) + " |"
    separator = "|" + "|".join("---" for _ in materialized[0]) + "|"
    body = ["| " + " | ".join(row) + " |" for row in materialized[1:]]
    return "\n".join([header, separator, *body])


def _write_markdown(data: ReportData, assets: Mapping[str, Path], markdown_path: Path) -> None:
    comparison = data.analysis["morse_graph_comparison"]
    selected = data.training["selected"]
    replay = selected["replay"]
    edges = ", ".join(f"{source}->{target}" for source, target in data.graph.edges)
    missing = [
        f"{check['source_object']}->{check['target_object']}"
        for check in comparison["orbit_manifold_reachability_checks"]
        if not check["reachable"]
    ]
    overlay_name = data.overlay_png_source.name
    text = f"""# Leslie3D Morse replay: subdivision (24, 28, 36), limit 10,000,000

Generated {data.generated_at_utc}. This is a numerical experiment, not a
Conley certificate.

## Outcome

{_outcome(data)}

The requested raw Morse-set file is
{'byte-identical' if data.baseline.morse_sets_byte_identical else 'not byte-identical'}
to the subdivision {data.baseline.subdivision}, limit
{data.baseline.subdiv_limit:,} source, and the graph is
{'semantically identical' if data.baseline.graph_semantically_identical else 'semantically different'}.

![Requested learned Morse sets](assets/learned_morse_sets.png)

![Requested learned Morse graph](assets/learned_morse_graph.png)

## Construction and accepted training

Every saved pair uses the analytic successor. The dataset manifest states:
`{data.dataset['transition_policy']}`

{_markdown_table(_component_rows(data))}

{_markdown_table(_network_rows(data))}

The accepted checkpoint selected epoch {data.training['best_epoch']} after
{data.training['epochs_run']} epochs. Its validation means are reconstruction
`{_fmt_num(float(replay['reconstruction']))}`, prediction
`{_fmt_num(float(replay['prediction']))}`, semiconjugacy
`{_fmt_num(float(replay['semiconjugacy']))}`, and cycle
`{_fmt_num(float(replay['cycle']))}`. The promoted checkpoint SHA-256 is
`{_promoted_checkpoint_hash(data)}`.

## Requested Morse computation

The configuration, run manifest, and `mg_params_log.txt` agree on subdivision
`{EXPECTED_SUBDIVISION}`, `subdiv_limit={EXPECTED_SUBDIV_LIMIT}`, latent bounds
`{data.config['cmgdb']['lower_bounds']}` to
`{data.config['cmgdb']['upper_bounds']}`, backend
`{data.mg_params['box_map_backend']}`, dense precompute level
`{data.mg_params['adaptive_precompute_subdiv']}`, padding
`{data.mg_params['padding']}`, and CPU execution. The recorded CMGDB duration is
`{_fmt_num(float(data.mg_params['duration_minutes']))}` minutes.

## Attained-depth audit

All saved Morse-set rows are minimum-grid covers at tree depth 28; they are not
terminal descendant snapshots. The node-level cardinality audit gives:

{_markdown_table(_depth_rows(data))}

Nodes marked `max 36 processed` are proved to have a surviving branch processed
through the requested maximum. For nodes marked `later stop vs max unknown`, the
displayed depth is only a rigorous lower bound: the saved DOT/CSV/log cannot
distinguish a later `subdiv_limit` hit from successful refinement to max 36
because the deeper hierarchy is not serialized.

{_markdown_table(_box_rows(data))}

Reduced edges: `{edges}`. Minimal nodes:
`{', '.join(map(str, data.graph.minimal_nodes))}`.

{_markdown_table(_role_rows(data))}

## Direct-system ground truth overlay

![Direct ground truth over the requested Morse sets](assets/{overlay_name})

The background is every saved learned Morse box. Colored direct-system clouds
are encoded centers of the render-only level-24 parent-cell display cover;
outlined symbols are the exact invariant phases. Center encoding is a sampled
visual comparison, not an enclosure of each direct three-dimensional box.

## Sampled residual and tolerance for minimal nodes

These values were **transferred without resampling**. The requested replay has
the same checkpoint SHA-256 and byte-identical raw Morse sets as the source
audit, and its graph is semantically identical. The transfer proof is recorded
in `analysis/residual_tolerance_transfer_provenance.json`.

{_markdown_table(_residual_rows(data))}

`R-hat` is a sampled maximum of the semiconjugacy residual and therefore a
lower bound on the exact supremum. `tau-hat` is a sampled minimum clearance and
therefore an upper estimate of the exact infimum. `R-hat >= tau-hat` supplies a
sampled counter-witness to the strict sufficient inequality for the evaluated
candidate block; it is not a positive certificate in the opposite direction.

## Direct relation audit

{_markdown_table(_relation_rows(data))}

Missing requested relation(s): `{', '.join(missing) if missing else 'none'}`.

## Conditional ideal-data statement

If continuous `E`, `D`, and `g` attain zero population reconstruction and
semiconjugacy loss on a full-support measure over compact `K`, then `E` is a
homeomorphism from `K` to `E(K)` and the restricted dynamics are conjugate.
Conley-index recovery still requires transport of a valid index pair, no extra
or lost invariant dynamics in the chosen isolating neighborhoods, and
conjugate quotient index maps (or certified shift equivalence). Semiconjugacy
alone is insufficient.

See `report_manifest.json`,
`assets/direct_ground_truth_overlay_provenance.json`, and the files under
`analysis/` for machine-readable provenance.
"""
    markdown_path.write_text(text, encoding="utf-8")


def _write_manifest(
    data: ReportData,
    assets: Mapping[str, Path],
    pdf_path: Path,
    markdown_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    outputs: dict[str, dict[str, Any]] = {}
    output_paths = [pdf_path, markdown_path, *assets.values()]
    seen: set[Path] = set()
    for path in output_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        relative = resolved.relative_to(data.report_root)
        outputs[str(relative)] = {
            "sha256": _sha256(resolved),
            "size_bytes": resolved.stat().st_size,
        }

    comparison = data.analysis["morse_graph_comparison"]
    manifest = {
        "schema_version": 1,
        "title": "Leslie3D Morse replay: subdivision (24, 28, 36), limit 10,000,000",
        "generated_at_utc": data.generated_at_utc,
        "status": "numerical_experiment_not_a_conley_certificate",
        "requested_parameters": {
            "subdivision": list(EXPECTED_SUBDIVISION),
            "subdiv_limit": EXPECTED_SUBDIV_LIMIT,
            "bounds": {
                "lower": data.config["cmgdb"]["lower_bounds"],
                "upper": data.config["cmgdb"]["upper_bounds"],
            },
            "backend": data.mg_params["box_map_backend"],
            "adaptive_precompute_subdiv": data.mg_params["adaptive_precompute_subdiv"],
            "padding": data.mg_params["padding"],
            "duration_minutes": data.mg_params["duration_minutes"],
        },
        "observed_graph": {
            "nodes": list(data.graph.nodes),
            "edges": [list(edge) for edge in data.graph.edges],
            "minimal_nodes": list(data.graph.minimal_nodes),
            "node_indices": {
                str(node): list(index) for node, index in data.graph.node_indices.items()
            },
            "morse_boxes_by_node": {
                str(node): count for node, count in data.box_counts.items()
            },
            "morse_boxes_total": sum(data.box_counts.values()),
        },
        "role_assignments": _role_assignments(data),
        "role_graph_checks": {
            key: comparison[key]
            for key in (
                "all_objects_uniquely_assigned",
                "all_objects_in_distinct_nodes",
                "node_count_matches_six_roles",
                "all_object_indices_match",
                "all_object_minimality_matches",
                "all_expected_relations_reachable",
                "all_role_reachability_and_nonreachability_match",
                "exact_role_aligned_morse_graph_match",
            )
        },
        "baseline_comparison": {
            "run_root": str(data.baseline_root),
            "subdivision": list(data.baseline.subdivision),
            "subdiv_limit": data.baseline.subdiv_limit,
            "graph_semantically_identical": data.baseline.graph_semantically_identical,
            "morse_sets_byte_identical": data.baseline.morse_sets_byte_identical,
            "checkpoint_identical": data.baseline.checkpoint_identical,
        },
        "refinement_depth_audit": {
            "source": "analysis/refinement_depth_audit.json",
            "all_saved_boxes_at_subdiv_min": data.depth_audit["saved_morse_sets"][
                "all_saved_boxes_match_subdiv_min"
            ],
            "observed_saved_tree_depths": data.depth_audit["saved_morse_sets"][
                "observed_tree_depths_from_box_widths"
            ],
            "nodes": {
                str(entry["node"]): {
                    "guaranteed_descendant_grid_reached_depth_at_least": entry[
                        "guaranteed_descendant_grid_reached_depth_at_least"
                    ],
                    "earliest_depth_at_which_limit_could_be_exceeded": entry[
                        "earliest_depth_at_which_limit_could_be_exceeded"
                    ],
                    "max_depth_processed": entry["max_depth_processed"],
                    "classification": entry["classification"],
                }
                for entry in data.depth_audit["refinement_audit"]["nodes"]
            },
            "caveat": (
                "For max_depth_processed=null, saved DOT/CSV/log artifacts cannot distinguish "
                "a later subdiv_limit hit from successful refinement to subdiv_max."
            ),
        },
        "residual_tolerance_transfer": {
            "status": "transferred_without_resampling",
            "source": str(data.residual_json_path),
            "source_sha256": _sha256(data.residual_json_path),
            "proof": "analysis/residual_tolerance_transfer_provenance.json",
            "nodes": {
                str(node): {
                    "sample_residual": data.residual["nodes"][str(node)]["residual"][
                        "sampled_maximum"
                    ],
                    "sample_tolerance": data.residual["nodes"][str(node)]["tolerance"][
                        "sampled_minimum"
                    ],
                    "conclusion": data.residual["nodes"][str(node)]["comparison"][
                        "conclusion"
                    ],
                }
                for node in data.graph.minimal_nodes
            },
        },
        "source_sha256": data.source_hashes,
        "outputs": outputs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate completed Leslie3D s24/28/36 artifacts and build a separate PDF report bundle."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--baseline-run-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--direct-root", type=Path, default=DEFAULT_DIRECT_ROOT)
    parser.add_argument("--residual-json", type=Path, default=DEFAULT_RESIDUAL_JSON)
    parser.add_argument("--residual-markdown", type=Path, default=DEFAULT_RESIDUAL_MD)
    parser.add_argument(
        "--overlay-provenance",
        type=Path,
        default=None,
        help=(
            "Default: <report-root>/assets/direct_ground_truth_overlay_provenance.json"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all scientific inputs and print a compact summary without writing the bundle.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        data = _load_report_data(args)
        summary = {
            "validated": True,
            "run_root": str(data.run_root),
            "subdivision": list(EXPECTED_SUBDIVISION),
            "subdiv_limit": EXPECTED_SUBDIV_LIMIT,
            "nodes": list(data.graph.nodes),
            "edges": [list(edge) for edge in data.graph.edges],
            "minimal_nodes": list(data.graph.minimal_nodes),
            "morse_boxes_total": sum(data.box_counts.values()),
            "baseline": data.baseline.__dict__,
        }
        if args.validate_only:
            print(json.dumps(summary, indent=2, allow_nan=False))
            return

        data.report_root.mkdir(parents=True, exist_ok=True)
        assets = _prepare_bundle(data)
        pdf_path = data.report_root / "leslie3d_morse_report_s24_28_36_limit10m.pdf"
        markdown_path = data.report_root / "report.md"
        manifest_path = data.report_root / "report_manifest.json"
        _write_pdf(data, assets, pdf_path)
        _write_markdown(data, assets, markdown_path)
        manifest = _write_manifest(
            data, assets, pdf_path, markdown_path, manifest_path
        )
        print(json.dumps({**summary, "report_root": str(data.report_root), "outputs": manifest["outputs"]}, indent=2, allow_nan=False))
    except (ReportInputError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Report input validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
