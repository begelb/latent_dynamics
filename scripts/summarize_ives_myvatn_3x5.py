"""Strictly summarize the Ives Lake Mývatn five-by-three replication.

The archived Bernardo run contains a four-node, three-edge Morse graph with
two sinks and edges ``1 -> 0``, ``3 -> 1``, and ``3 -> 2``.  Its JSON export
did not retain Conley indices, so this verifier classifies a new cell from the
saved graph incidence and the saved Morse boxes after encoding the archived
fixed point and period-12 phases with that cell's checkpoint and dataset
scaler.  Conley-derived periods are retained as a diagnostic, not a gate.

Strict mode writes nothing unless the exact 5 x 3 design is present and every
required artifact is non-empty and parseable.  ``--allow-incomplete`` writes
the same four reports as an explicitly provisional progress snapshot.
``--verify`` performs the strict audit without writing reports.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from latentdynamics.training.checkpoints import load_checkpoint  # noqa: E402

DATA_SEEDS = (2158, 4792, 3174, 688, 5727)
MODEL_SEEDS = (0, 1, 2)
EXPECTED_CELLS = tuple(
    (data_seed, model_seed) for data_seed in DATA_SEEDS for model_seed in MODEL_SEEDS
)
DEFAULT_SWEEP_ROOT = CODE_ROOT / "output" / "ives_myvatn_seedsweep_3x5_v1"
DEFAULT_DATA_ROOT = CODE_ROOT / "data" / "ives_myvatn_seedsweep_3x5_v1"
DEFAULT_REFERENCE_CSV = (
    CODE_ROOT / "src" / "latentdynamics" / "reference_data" / "ives_myvatn_invariant_points.csv"
)
EXPECTED_TRAIN_PAIRS = 20_000
EXPECTED_VALIDATION_PAIRS = 4_000
EXPECTED_VALIDATION_SEED = 9999
EXPECTED_AMBIENT_LOWER = [-3.0, -7.5, -3.0]
EXPECTED_AMBIENT_UPPER = [1.5, 1.5, 1.5]
EXPECTED_SCALING_EPSILON = 1.0e-6
EXPECTED_MODEL_PARAMS = {
    "r1": 3.873,
    "r2": 11.746,
    "c": 10**-6.435,
    "d": 0.5517,
    "p": 0.06659,
    "q": 0.9026,
    "coordinate_mode": "log",
}

# The exact incidence saved by archive/bernardo/.../morse_graph.json.  Node
# names here are roles only: isomorphism permits every new CMGDB node id.
REFERENCE_GRAPH_NODES = ("0", "1", "2", "3")
REFERENCE_GRAPH_EDGES = frozenset({("1", "0"), ("3", "1"), ("3", "2")})
REFERENCE_GRAPH_ROLES = {
    "0": "terminal_sink",
    "1": "middle",
    "2": "direct_sink",
    "3": "root",
}
REFERENCE_GRAPH_SIGNATURE_SHA256 = hashlib.sha256(
    json.dumps(
        sorted([list(edge) for edge in REFERENCE_GRAPH_EDGES]),
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

REQUIRED_FILES = {
    "checkpoint": Path("models/autoencoder.pt"),
    "checkpoint_sidecar": Path("models/autoencoder.json"),
    "history": Path("logs/history.json"),
    "training_summary": Path("training_summary.json"),
    "final_losses": Path("final_losses.txt"),
    "diagnose": Path("diagnose.json"),
    "morse_graph": Path("MG/morse_graph"),
    "morse_sets": Path("MG/morse_sets"),
    "mg_params_log": Path("mg_params_log.txt"),
    "morse_graph_pdf": Path("MG/morse_graph.pdf"),
    "morse_graph_png": Path("MG/morse_graph.png"),
    "morse_sets_pdf": Path("MG/morse_sets.pdf"),
    "morse_sets_png": Path("MG/morse_sets.png"),
    "metrics": Path("metrics.json"),
    "run_manifest": Path("run_manifest.json"),
}

SUCCESS_CRITERION = {
    "name": "archive_graph_and_invariant_membership",
    "archive_graph": {
        "n_nodes": 4,
        "n_edges": 3,
        "n_sinks": 2,
        "reference_edges": [["1", "0"], ["3", "1"], ["3", "2"]],
        "comparison": "directed graph isomorphism; observed node ids are unrestricted",
    },
    "fixed_point": "exactly one sink membership",
    "period_12_cycle": {
        "minimum_phases_uniquely_in_common_other_sink": 11,
        "n_phases": 12,
        "conflicting_sink_memberships_allowed": 0,
    },
    "conley_periods": {
        "diagnostic_target": [1, 12],
        "affects_machine_pass": False,
        "reason": "the archived reference JSON retained incidence and boxes but no indices",
    },
}

CSV_FIELDS = (
    "data_seed",
    "model_seed",
    "cell_status",
    "complete",
    "verification_passed",
    "machine_pass",
    "archive_graph_isomorphic",
    "n_morse_nodes",
    "n_morse_edges",
    "n_sinks",
    "sink_ids",
    "fixed_sink_id",
    "cycle_sink_id",
    "fixed_sink_memberships",
    "cycle_unique_target_count",
    "cycle_unassigned_count",
    "cycle_conflicting_phase_count",
    "cycle_phase_sink_memberships",
    "sink_conley_tuples",
    "sink_inferred_periods",
    "exact_conley_periods_1_12",
    "epochs_completed",
    "best_epoch",
    "selected_loss_total",
    "selected_losses",
    "train_duration_seconds",
    "checkpoint_sha256",
    "morse_graph_sha256",
    "morse_sets_sha256",
    "scaler_sha256",
    "artifact_hashes",
    "classification_evidence",
    "error_count",
    "errors",
    "cell_directory",
)

_LABEL_RE = re.compile(r'label\s*=\s*"((?:\\.|[^"\\])*)"')
_CONLEY_RE = re.compile(r"\([^()]*\)")
_PERIOD_RE = re.compile(r"^x(?:\^([1-9]\d*))?([+-])1$")
_NUMERIC_DIRECTORY_RE = re.compile(r"^(dataset|seed)_(-?\d+)$")


class SweepValidationError(RuntimeError):
    """The source sweep is not eligible for a strict final report."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(CODE_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _issue(
    scope: str,
    code: str,
    message: str,
    path: Path | None = None,
) -> dict[str, str]:
    result = {"scope": scope, "code": code, "message": message}
    if path is not None:
        result["path"] = _display(path)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    return {
        "path": _display(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def _parse_key_value(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read UTF-8 text: {exc}") from exc
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        if ":" not in raw:
            raise ValueError(f"line {line_number} lacks ':'")
        key, raw_value = raw.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"line {line_number} has an empty key")
        raw_value = raw_value.strip()
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        values[key] = value
    if not values:
        raise ValueError("file contains no key/value records")
    return values


def _unescape_dot_string(value: str) -> str:
    return (
        value.replace(r"\n", "\n")
        .replace(r'\"', '"')
        .replace(r"\\", "\\")
    )


def _dot_statements(text: str) -> list[str]:
    """Split ordinary CMGDB DOT without depending on pydot/Graphviz."""

    statements: list[str] = []
    buffer: list[str] = []
    quoted = False
    escaped = False
    bracket_depth = 0
    for char in text:
        if quoted:
            buffer.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
            buffer.append(char)
        elif char == "[":
            bracket_depth += 1
            buffer.append(char)
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            buffer.append(char)
        elif char == ";" or (char in "\r\n" and bracket_depth == 0):
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
        else:
            buffer.append(char)
    statement = "".join(buffer).strip()
    if statement:
        statements.append(statement)
    return statements


def _consume_dot_id(text: str, offset: int = 0) -> tuple[str, int] | None:
    index = offset
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return None
    if text[index] == '"':
        index += 1
        value: list[str] = []
        escaped = False
        while index < len(text):
            char = text[index]
            index += 1
            if escaped:
                value.append(char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return "".join(value), index
            else:
                value.append(char)
        return None
    match = re.match(
        r"(?:-?(?:\d+(?:\.\d*)?|\.\d+)|[A-Za-z_\200-\377][A-Za-z_\200-\3770-9]*)",
        text[index:],
    )
    if not match:
        return None
    return match.group(0), index + match.end()


def _normalize_conley_component(value: str) -> str:
    text = value.lower().strip().strip("$\"")
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("**", "^")
    text = re.sub(r"x\^\{(\d+)\}", r"x^\1", text)
    text = re.sub(r"\s+", "", text)
    return text


def _infer_period(components: list[str] | None) -> int | None:
    periods: list[int] = []
    for component in components or []:
        match = _PERIOD_RE.fullmatch(_normalize_conley_component(component))
        if not match:
            continue
        exponent = int(match.group(1)) if match.group(1) else 1
        periods.append(exponent if match.group(2) == "-" else 2 * exponent)
    return max(periods) if periods else None


def _parse_dot(path: Path) -> dict[str, Any]:
    """Parse CMGDB DOT while retaining ids, tuples, incidence, and degrees."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read DOT: {exc}") from exc
    nodes: dict[str, str] = {}
    edges: set[tuple[str, str]] = set()
    for original in _dot_statements(text):
        statement = original.strip().strip("{} ")
        if not statement or statement.startswith(("digraph", "strict digraph", "graph ")):
            continue
        consumed = _consume_dot_id(statement)
        if consumed is None:
            continue
        first, index = consumed
        rest = statement[index:].lstrip()
        if rest.startswith("->"):
            consumed_target = _consume_dot_id(rest, 2)
            if consumed_target is None:
                raise ValueError(f"invalid edge statement: {original!r}")
            target, _ = consumed_target
            edges.add((first, target))
            continue
        if not rest.startswith("["):
            continue
        if first.lower() in {"node", "edge", "graph"}:
            # DOT defaults such as ``node [shape=ellipse]`` are declarations,
            # not Morse nodes.
            continue
        label_match = _LABEL_RE.search(rest)
        nodes[first] = _unescape_dot_string(label_match.group(1)) if label_match else ""
    if not nodes:
        raise ValueError("DOT contains no declared Morse nodes")
    unknown = sorted({node for edge in edges for node in edge if node not in nodes})
    if unknown:
        raise ValueError(f"DOT edges reference undeclared nodes {unknown}")

    indegree = Counter(target for _, target in edges)
    outdegree = Counter(source for source, _ in edges)
    sink_ids = sorted(node_id for node_id in nodes if outdegree[node_id] == 0)
    parsed_nodes: list[dict[str, Any]] = []
    for node_id in sorted(nodes):
        label = nodes[node_id]
        matches = list(_CONLEY_RE.finditer(label))
        tuple_text = matches[-1].group(0) if matches else None
        raw_components = tuple_text[1:-1].split(",") if tuple_text is not None else None
        components = (
            [part.strip() for part in raw_components]
            if tuple_text is not None
            else None
        )
        parsed_nodes.append(
            {
                "id": node_id,
                "dot_label": label,
                "conley_tuple": tuple_text,
                "conley_components_exact": raw_components,
                "conley_components": components,
                "conley_components_normalized": (
                    [_normalize_conley_component(part) for part in components]
                    if components is not None
                    else None
                ),
                "inferred_period": _infer_period(components),
                "in_degree": indegree[node_id],
                "out_degree": outdegree[node_id],
                "is_sink": node_id in sink_ids,
                "is_minimal": node_id in sink_ids,
            }
        )
    edge_rows = [
        {"source": source, "target": target}
        for source, target in sorted(edges)
    ]
    canonical = {
        "nodes": sorted(nodes),
        "edges": [[edge["source"], edge["target"]] for edge in edge_rows],
    }
    return {
        "nodes": parsed_nodes,
        "edges": edge_rows,
        "sink_ids": sink_ids,
        "minimal_ids": list(sink_ids),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "topology_signature_sha256": hashlib.sha256(
            _json_text(canonical).encode("utf-8")
        ).hexdigest(),
    }


def _archive_graph_isomorphism(graph: dict[str, Any]) -> dict[str, Any]:
    observed_nodes = [node["id"] for node in graph.get("nodes", [])]
    observed_edges = {
        (edge["source"], edge["target"]) for edge in graph.get("edges", [])
    }
    if len(observed_nodes) != 4 or len(observed_edges) != 3:
        return {"isomorphic": False, "reference_to_observed": None}
    for permutation in itertools.permutations(observed_nodes):
        mapping = dict(zip(REFERENCE_GRAPH_NODES, permutation, strict=True))
        mapped_edges = {
            (mapping[source], mapping[target]) for source, target in REFERENCE_GRAPH_EDGES
        }
        if mapped_edges == observed_edges:
            observed_roles = {
                mapping[reference_id]: role
                for reference_id, role in REFERENCE_GRAPH_ROLES.items()
            }
            return {
                "isomorphic": True,
                "reference_to_observed": mapping,
                "observed_to_reference": {value: key for key, value in mapping.items()},
                "observed_roles": observed_roles,
                "normalized_archive_signature_sha256": REFERENCE_GRAPH_SIGNATURE_SHA256,
            }
    return {"isomorphic": False, "reference_to_observed": None}


def _canonical_box_label(value: str, line_number: int) -> str:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"line {line_number} has a non-numeric Morse label") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"line {line_number} has a non-integer Morse label")
    return str(int(number))


def _parse_morse_sets(path: Path) -> dict[str, Any]:
    lower: list[list[float]] = []
    upper: list[list[float]] = []
    labels: list[str] = []
    dimension: int | None = None
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"cannot open Morse sets: {exc}") from exc
    with handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 3 or len(row) % 2 == 0:
                raise ValueError(f"line {line_number} is not a 2*d+1 Morse box row")
            row_dimension = (len(row) - 1) // 2
            if dimension is None:
                dimension = row_dimension
            elif dimension != row_dimension:
                raise ValueError(f"line {line_number} changes Morse-box dimension")
            try:
                coordinates = [float(value) for value in row[:-1]]
            except ValueError as exc:
                raise ValueError(f"line {line_number} contains non-numeric coordinates") from exc
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError(f"line {line_number} contains non-finite coordinates")
            lo = coordinates[:row_dimension]
            hi = coordinates[row_dimension:]
            if any(left >= right for left, right in zip(lo, hi, strict=True)):
                raise ValueError(f"line {line_number} has a non-positive box width")
            lower.append(lo)
            upper.append(hi)
            labels.append(_canonical_box_label(row[-1].strip(), line_number))
    if dimension is None:
        raise ValueError("Morse-set file contains no boxes")
    label_counts = Counter(labels)
    return {
        "dimension": dimension,
        "total_boxes": len(labels),
        "boxes_by_node": {key: label_counts[key] for key in sorted(label_counts)},
        "_lower": np.asarray(lower, dtype=np.float64),
        "_upper": np.asarray(upper, dtype=np.float64),
        "_labels": np.asarray(labels, dtype=object),
    }


def _load_reference_csv(path: Path) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"cannot open reference CSV: {exc}") from exc
    records: list[dict[str, Any]] = []
    with handle:
        reader = csv.DictReader(handle)
        expected = {
            "vertex",
            "component_id",
            "barycenter_x",
            "barycenter_y",
            "barycenter_z",
        }
        if reader.fieldnames is None or not expected <= set(reader.fieldnames):
            raise ValueError(
                "reference CSV requires vertex, component_id, and barycenter_x/y/z columns"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                vertex = int(row["vertex"])
                component = int(row["component_id"])
                coordinates = [
                    float(row["barycenter_x"]),
                    float(row["barycenter_y"]),
                    float(row["barycenter_z"]),
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid reference row {line_number}") from exc
            if vertex not in (0, 1) or not all(math.isfinite(value) for value in coordinates):
                raise ValueError(f"invalid reference row {line_number}")
            records.append(
                {
                    "role": "period_12_cycle" if vertex == 0 else "fixed_point",
                    "vertex": vertex,
                    "phase": component,
                    "ambient_coordinates": coordinates,
                }
            )
    fixed = [record for record in records if record["role"] == "fixed_point"]
    cycle = sorted(
        (record for record in records if record["role"] == "period_12_cycle"),
        key=lambda record: record["phase"],
    )
    if len(fixed) != 1 or fixed[0]["phase"] != 0:
        raise ValueError("reference CSV must contain fixed vertex 1/component 0 exactly once")
    if len(cycle) != 12 or [record["phase"] for record in cycle] != list(range(12)):
        raise ValueError("reference CSV must contain cycle vertex 0/components 0..11 exactly once")
    ordered = [fixed[0], *cycle]
    points = np.asarray([record["ambient_coordinates"] for record in ordered], dtype=np.float64)
    return {
        "file": _file_ref(path),
        "coordinate_system": "log10 ambient coordinates",
        "fixed_point_count": 1,
        "period_12_phase_count": 12,
    }, points, ordered


def _validate_png(path: Path) -> None:
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"invalid PNG: {exc}") from exc


def _validate_pdf(path: Path) -> None:
    try:
        prefix = path.read_bytes()[:5]
    except OSError as exc:
        raise ValueError(f"cannot read PDF: {exc}") from exc
    if prefix != b"%PDF-":
        raise ValueError("PDF lacks a %PDF- header")


def _finite_series(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_as_float(item) is not None for item in value)


def _training_evidence(
    summary: dict[str, Any],
    history: dict[str, Any],
    final_losses: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    epochs = _as_int(summary.get("n_epochs_run"))
    if epochs is None or epochs < 1:
        errors.append("training_summary.n_epochs_run must be a positive integer")
    best_epoch = _as_int(summary.get("best_epoch"))
    if best_epoch is None or (epochs is not None and not (-1 <= best_epoch < epochs)):
        errors.append("training_summary.best_epoch is outside the completed epoch range")
    selected_raw = summary.get("selected_val")
    selected: dict[str, float] = {}
    if not isinstance(selected_raw, dict) or not selected_raw:
        errors.append("training_summary.selected_val must be a non-empty loss object")
    else:
        for key, value in selected_raw.items():
            number = _as_float(value)
            if number is None:
                errors.append(f"training_summary.selected_val.{key} is non-finite")
            else:
                selected[str(key)] = number
        if "loss_total" not in selected:
            errors.append("training_summary.selected_val.loss_total is required")
    for split in ("train", "val"):
        block = history.get(split)
        series = block.get("loss_total") if isinstance(block, dict) else None
        if not _finite_series(series):
            errors.append(f"history.{split}.loss_total must be a non-empty finite series")
        elif epochs is not None and len(series) != epochs:
            errors.append(f"history.{split}.loss_total length does not equal n_epochs_run")
    loss_values = {
        key: number
        for key, value in final_losses.items()
        if "loss" in key.lower() and (number := _as_float(value)) is not None
    }
    if not loss_values:
        errors.append("final_losses contains no finite loss value")
    return {
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "best_source": summary.get("best_source"),
        "selected_losses": selected,
        "selected_loss_total": selected.get("loss_total"),
        "train_duration_seconds": _as_float(summary.get("train_duration_seconds")),
        "final_losses": loss_values,
    }, errors


def _validate_mg_log(values: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    lower = values.get("Lower bounds", values.get("lower_bounds"))
    upper = values.get("Upper bounds", values.get("upper_bounds"))
    if not (
        isinstance(lower, (list, tuple))
        and isinstance(upper, (list, tuple))
        and len(lower) == len(upper) == 2
        and all(_as_float(item) is not None for item in (*lower, *upper))
    ):
        errors.append("mg_params_log requires two-dimensional lower and upper bounds")
    elif any(float(lo) >= float(hi) for lo, hi in zip(lower, upper, strict=True)):
        errors.append("mg_params_log bounds are not strictly ordered")
    expected_exact = {
        "subdiv_init": 18,
        "subdiv_min": 22,
        "subdiv_max": 30,
        "subdiv_limit": 100000,
        "bounds_epsilon_frac": 0.1,
        "padding": True,
        "box_map_backend": "adaptive_precomputed",
        "bounds_data_role": "system_grid",
        "bounds_grid_resolution": 64,
        "bounds_include_latent_image": True,
        "bounds_clip_lower": [-1.0, -1.0],
        "bounds_clip_upper": [1.0, 1.0],
        "adaptive_precompute_subdiv": "init",
        "precompute_batch_points": "auto",
        "compute_roa": False,
        "roa_max_vertices": 50000000,
        "collapse_roa_to_lca": True,
        "bounds_source": "encoded_system_grid_and_latent_image",
    }
    for key, expected in expected_exact.items():
        if values.get(key) != expected:
            errors.append(
                f"mg_params_log.{key}: expected {expected!r}, found {values.get(key)!r}"
            )
    duration = _as_float(values.get("duration_minutes"))
    if duration is None or duration < 0.0:
        errors.append("mg_params_log.duration_minutes must be a finite non-negative number")
    return errors


def _encode_reference(
    checkpoint_dir: Path,
    scaler_path: Path,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import joblib

    try:
        scaler = joblib.load(scaler_path)
    except Exception as exc:
        raise ValueError(f"cannot load scaler: {exc}") from exc
    if not hasattr(scaler, "transform"):
        raise ValueError("loaded scaler has no transform method")
    try:
        scaled = np.asarray(scaler.transform(reference_points), dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"scaler cannot transform reference points: {exc}") from exc
    if scaled.shape != reference_points.shape or not np.isfinite(scaled).all():
        raise ValueError(
            f"scaler returned shape {scaled.shape}, expected {reference_points.shape}, or non-finite values"
        )
    try:
        model, arch = load_checkpoint(checkpoint_dir, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"cannot load checkpoint and sidecar: {exc}") from exc
    if arch.high_dims != 3 or arch.low_dims != 2:
        raise ValueError(
            f"checkpoint architecture must be 3D -> 2D, got {arch.high_dims}D -> {arch.low_dims}D"
        )
    model.to("cpu").eval()
    with torch.no_grad():
        encoded = model.encoder(torch.as_tensor(scaled, dtype=torch.float32)).cpu().numpy()
    encoded = np.asarray(encoded, dtype=np.float64)
    if encoded.shape != (13, 2) or not np.isfinite(encoded).all():
        raise ValueError(f"encoder returned invalid reference shape {encoded.shape}")
    return scaled, encoded, arch.model_dump(mode="json")


def _membership_evidence(
    *,
    graph: dict[str, Any],
    morse_sets: dict[str, Any],
    reference_records: list[dict[str, Any]],
    scaled: np.ndarray,
    encoded: np.ndarray,
) -> dict[str, Any]:
    if morse_sets["dimension"] != 2:
        raise ValueError(f"reference membership requires 2D Morse boxes, got {morse_sets['dimension']}D")
    lower = morse_sets["_lower"]
    upper = morse_sets["_upper"]
    labels = morse_sets["_labels"]
    sink_ids = set(graph["sink_ids"])
    points: list[dict[str, Any]] = []
    for index, record in enumerate(reference_records):
        inside = np.all((lower <= encoded[index]) & (encoded[index] <= upper), axis=1)
        memberships = sorted({str(label) for label in labels[inside]})
        sink_memberships = sorted(set(memberships) & sink_ids)
        points.append(
            {
                **record,
                "scaled_coordinates": scaled[index].tolist(),
                "encoded_coordinates": encoded[index].tolist(),
                "morse_node_memberships": memberships,
                "sink_memberships": sink_memberships,
                "unique_sink_membership": (
                    sink_memberships[0] if len(sink_memberships) == 1 else None
                ),
                "sink_membership_status": (
                    "unique"
                    if len(sink_memberships) == 1
                    else "unassigned"
                    if not sink_memberships
                    else "ambiguous"
                ),
            }
        )
    return {"fixed_point": points[0], "period_12_phases": points[1:]}


def _classify(
    graph: dict[str, Any],
    memberships: dict[str, Any],
) -> dict[str, Any]:
    """Apply only the archive-grounded graph and membership criterion."""

    isomorphism = _archive_graph_isomorphism(graph)
    sinks = list(graph.get("sink_ids", []))
    fixed_memberships = list(memberships["fixed_point"].get("sink_memberships", []))
    fixed_unique = len(fixed_memberships) == 1
    fixed_sink = fixed_memberships[0] if fixed_unique else None
    other_sinks = [sink for sink in sinks if sink != fixed_sink] if fixed_sink else []
    cycle_sink = other_sinks[0] if len(sinks) == 2 and len(other_sinks) == 1 else None

    unique_target_phases: list[int] = []
    unassigned_phases: list[int] = []
    conflicting_phases: list[dict[str, Any]] = []
    phase_memberships: list[dict[str, Any]] = []
    for phase in memberships["period_12_phases"]:
        phase_id = int(phase["phase"])
        sink_memberships = list(phase.get("sink_memberships", []))
        phase_memberships.append({"phase": phase_id, "sink_memberships": sink_memberships})
        if cycle_sink is not None and sink_memberships == [cycle_sink]:
            unique_target_phases.append(phase_id)
        elif not sink_memberships:
            unassigned_phases.append(phase_id)
        else:
            conflicting_phases.append(
                {"phase": phase_id, "sink_memberships": sink_memberships}
            )
    membership_pass = (
        fixed_unique
        and cycle_sink is not None
        and len(unique_target_phases) >= 11
        and not conflicting_phases
    )
    by_id = {node["id"]: node for node in graph.get("nodes", [])}
    sink_periods = [by_id[sink]["inferred_period"] for sink in sinks if sink in by_id]
    exact_periods = (
        len(sink_periods) == 2
        and all(period is not None for period in sink_periods)
        and sorted(int(period) for period in sink_periods) == [1, 12]
    )
    cycle_assignment_pass = (
        cycle_sink is not None
        and len(unique_target_phases) >= 11
        and not conflicting_phases
    )
    distinct_sink_pass = (
        fixed_sink is not None
        and cycle_sink is not None
        and fixed_sink != cycle_sink
    )
    for node in graph.get("nodes", []):
        node["archive_role"] = isomorphism.get("observed_roles", {}).get(node["id"])
    return {
        "machine_pass": bool(isomorphism["isomorphic"] and membership_pass),
        "archive_graph_isomorphic": isomorphism["isomorphic"],
        "graph_shape_pass": isomorphism["isomorphic"],
        "graph_isomorphism": isomorphism,
        "fixed_point_unique_sink": fixed_unique,
        "fixed_assignment_pass": fixed_unique,
        "fixed_sink_id": fixed_sink,
        "cycle_sink_id": cycle_sink,
        "fixed_sink_memberships": fixed_memberships,
        "cycle_unique_target_phases": unique_target_phases,
        "cycle_unique_target_count": len(unique_target_phases),
        "cycle_coverage_count": len(unique_target_phases),
        "cycle_coverage_denominator": 12,
        "cycle_unassigned_phases": unassigned_phases,
        "cycle_unassigned_count": len(unassigned_phases),
        "cycle_conflicting_phases": conflicting_phases,
        "cycle_conflicting_phase_count": len(conflicting_phases),
        "cycle_uncovered_or_ambiguous_phases": sorted(
            [*unassigned_phases, *(item["phase"] for item in conflicting_phases)]
        ),
        "cycle_phase_sink_memberships": phase_memberships,
        "cycle_assignment_pass": cycle_assignment_pass,
        "distinct_sink_pass": distinct_sink_pass,
        "membership_pass": membership_pass,
        "sink_inferred_periods": sink_periods,
        "exact_conley_periods_1_12": exact_periods,
        "conley_periods_affect_machine_pass": False,
    }


def _scan_transition_csv(path: Path, *, expected_pairs: int) -> dict[str, Any]:
    expected_header = ["x0", "x1", "x2", "y0", "y1", "y2"]
    row_count = 0
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"cannot open transition CSV: {exc}") from exc
    with handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("transition CSV is empty") from exc
        if header != expected_header:
            raise ValueError(f"expected header {expected_header!r}, found {header!r}")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 6:
                raise ValueError(f"line {line_number} has {len(row)} columns, expected 6")
            try:
                values = [float(value) for value in row]
            except ValueError as exc:
                raise ValueError(f"line {line_number} contains non-numeric data") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"line {line_number} contains non-finite data")
            row_count += 1
    if row_count != expected_pairs:
        raise ValueError(f"expected {expected_pairs} transition pairs, found {row_count}")
    return {**_file_ref(path), "transition_pairs": row_count, "header": expected_header}


def _expected_dataset_metadata(*, role: str, data_seed: int) -> dict[str, Any]:
    is_train = role == "train"
    return {
        "system": "IvesModel",
        "dimension": 3,
        "n_samples": 1000 if is_train else 200,
        "n_iterations": 70,
        "skip_initial_steps": 50,
        "lower_bounds": EXPECTED_AMBIENT_LOWER,
        "upper_bounds": EXPECTED_AMBIENT_UPPER,
        "model_params": EXPECTED_MODEL_PARAMS,
        "dataset_name": "train" if is_train else "val",
        "sampling_method": "uniform",
        "sampling_seed": data_seed if is_train else EXPECTED_VALIDATION_SEED,
        "role": role,
    }


def _metadata_mismatches(payload: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    return [
        f"{key}: expected {expected_value!r}, found {payload.get(key)!r}"
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]


def _expected_manifest_config(
    *,
    data_seed: int,
    data_root: Path,
    sweep_root: Path,
) -> dict[str, Any]:
    from latentdynamics.config import load_config

    expected = copy.deepcopy(load_config("ives_myvatn").model_dump(mode="json"))
    expected["data"]["train_seed"] = data_seed
    expected["seeds"] = list(MODEL_SEEDS)
    expected["experiment_name"] = f"ives_myvatn_seedsweep_3x5_v1_dataset_{data_seed}"
    expected["paths"]["data_dir"] = str((data_root / f"dataset_{data_seed}").resolve())
    expected["paths"]["output_dir"] = str((sweep_root / f"dataset_{data_seed}").resolve())
    expected["paths"]["scaler_dir_override"] = None
    return expected


def _analyze_dataset(
    *,
    data_seed: int,
    data_root: Path,
    sweep_root: Path,
) -> dict[str, Any]:
    scope = f"dataset_{data_seed}"
    dataset_data_dir = data_root / f"dataset_{data_seed}"
    dataset_output_dir = sweep_root / f"dataset_{data_seed}"
    scaler_dir = dataset_output_dir / "scalers" / "train"
    paths = {
        "train_csv": dataset_data_dir / "train.csv",
        "train_metadata": dataset_data_dir / "train_metadata.json",
        "validation_csv": dataset_data_dir / "val.csv",
        "validation_metadata": dataset_data_dir / "val_metadata.json",
        "scaler": scaler_dir / "scaler.gz",
        "scaler_metadata": scaler_dir / "scaler_metadata.json",
    }
    errors: list[dict[str, str]] = []
    missing: list[str] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
            errors.append(
                _issue(scope, "missing_or_empty_dataset_artifact", f"missing or empty {name}", path)
            )
        else:
            try:
                artifacts[name] = _file_ref(path)
            except OSError as exc:
                errors.append(_issue(scope, "dataset_artifact_hash_error", f"{name}: {exc}", path))

    for role, artifact_name, expected_pairs in (
        ("train", "train_csv", EXPECTED_TRAIN_PAIRS),
        ("validation", "validation_csv", EXPECTED_VALIDATION_PAIRS),
    ):
        if artifact_name not in artifacts:
            continue
        try:
            artifacts[artifact_name] = _scan_transition_csv(
                paths[artifact_name], expected_pairs=expected_pairs
            )
        except ValueError as exc:
            errors.append(
                _issue(scope, "invalid_dataset_csv", f"{role}: {exc}", paths[artifact_name])
            )

    metadata_payloads: dict[str, dict[str, Any]] = {}
    for role, artifact_name in (
        ("train", "train_metadata"),
        ("validation", "validation_metadata"),
    ):
        if artifact_name not in artifacts:
            continue
        try:
            payload = _read_json_object(paths[artifact_name])
            metadata_payloads[role] = payload
            mismatches = _metadata_mismatches(
                payload,
                _expected_dataset_metadata(role="train" if role == "train" else "val", data_seed=data_seed),
            )
            for message in mismatches:
                errors.append(
                    _issue(scope, "dataset_metadata_mismatch", f"{role}.{message}", paths[artifact_name])
                )
        except ValueError as exc:
            errors.append(
                _issue(scope, "invalid_dataset_metadata", f"{role}: {exc}", paths[artifact_name])
            )

    scaler_metadata: dict[str, Any] | None = None
    if "scaler_metadata" in artifacts:
        try:
            scaler_metadata = _read_json_object(paths["scaler_metadata"])
            expected_scaler_metadata = {
                "train_file": "train",
                "train_csv_sha256": (artifacts.get("train_csv") or {}).get("sha256"),
                "scaling": "fixed_bounds",
                "high_dims": 3,
                "lower_bounds": EXPECTED_AMBIENT_LOWER,
                "upper_bounds": EXPECTED_AMBIENT_UPPER,
                "scaling_epsilon": EXPECTED_SCALING_EPSILON,
            }
            for message in _metadata_mismatches(scaler_metadata, expected_scaler_metadata):
                errors.append(
                    _issue(scope, "scaler_metadata_mismatch", message, paths["scaler_metadata"])
                )
            raw_train_csv = scaler_metadata.get("train_csv")
            if not isinstance(raw_train_csv, str) or Path(raw_train_csv).resolve() != paths["train_csv"].resolve():
                errors.append(
                    _issue(
                        scope,
                        "scaler_metadata_mismatch",
                        "train_csv path does not resolve to this dataset's train.csv",
                        paths["scaler_metadata"],
                    )
                )
        except ValueError as exc:
            errors.append(
                _issue(scope, "invalid_scaler_metadata", str(exc), paths["scaler_metadata"])
            )
    if "scaler" in artifacts:
        import joblib

        try:
            scaler = joblib.load(paths["scaler"])
            lower = np.asarray(scaler.lower_bounds, dtype=np.float64)
            upper = np.asarray(scaler.upper_bounds, dtype=np.float64)
            epsilon = _as_float(getattr(scaler, "epsilon", None))
            if (
                type(scaler).__name__ != "FixedBoundsScaler"
                or not np.array_equal(lower, np.asarray(EXPECTED_AMBIENT_LOWER))
                or not np.array_equal(upper, np.asarray(EXPECTED_AMBIENT_UPPER))
                or epsilon != EXPECTED_SCALING_EPSILON
            ):
                errors.append(
                    _issue(
                        scope,
                        "invalid_fixed_bounds_scaler",
                        "scaler class/bounds/epsilon do not match the frozen fixed-box map",
                        paths["scaler"],
                    )
                )
        except Exception as exc:
            errors.append(
                _issue(scope, "invalid_fixed_bounds_scaler", f"cannot validate scaler: {exc}", paths["scaler"])
            )

    expected_config = _expected_manifest_config(
        data_seed=data_seed,
        data_root=data_root,
        sweep_root=sweep_root,
    )
    return {
        "data_seed": data_seed,
        "data_directory": _display(dataset_data_dir),
        "output_directory": _display(dataset_output_dir),
        "artifacts": artifacts,
        "metadata": metadata_payloads,
        "scaler_metadata": scaler_metadata,
        "complete": not missing,
        "verification_passed": not missing and not errors,
        "errors": errors,
        "_data_dir": dataset_data_dir,
        "_output_dir": dataset_output_dir,
        "_scaler_path": paths["scaler"],
        "_expected_manifest_config": expected_config,
    }


def _path_matches(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).resolve() == expected.resolve()


def _validate_run_manifest(
    *,
    manifest: dict[str, Any],
    dataset: dict[str, Any],
    cell_dir: Path,
    model_seed: int,
) -> list[str]:
    from latentdynamics.cli.provenance import hash_config_dict

    errors: list[str] = []
    expected_config = dataset["_expected_manifest_config"]
    observed_config = manifest.get("config")
    if observed_config != expected_config:
        errors.append("config does not exactly match the resolved Ives Mývatn protocol")
    if manifest.get("config_hash") != hash_config_dict(expected_config):
        errors.append("config_hash does not match the frozen resolved config")
    cell = manifest.get("cell") if isinstance(manifest.get("cell"), dict) else {}
    if _as_int(cell.get("seed")) != model_seed:
        errors.append(f"cell.seed must be {model_seed}")
    if cell.get("train_file") != "train":
        errors.append("cell.train_file must be 'train'")
    if not _path_matches(cell.get("output_dir"), cell_dir):
        errors.append("cell.output_dir does not resolve to the expected cell")
    manifest_artifacts = (
        manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    )
    dataset_artifacts = dataset["artifacts"]
    expected_paths = {
        "train_csv": dataset["_data_dir"] / "train.csv",
        "scaler": dataset["_scaler_path"],
        "model_dir": cell_dir / "models",
        "morse_dir": cell_dir / "MG",
        "metrics": cell_dir / "metrics.json",
    }
    for key, expected_path in expected_paths.items():
        if not _path_matches(manifest_artifacts.get(key), expected_path):
            errors.append(f"artifacts.{key} does not resolve to the expected path")
    for key, artifact_name in (
        ("train_csv_sha256", "train_csv"),
        ("scaler_sha256", "scaler"),
    ):
        expected_hash = (dataset_artifacts.get(artifact_name) or {}).get("sha256")
        if manifest_artifacts.get(key) != expected_hash:
            errors.append(f"artifacts.{key} does not match the current source artifact")
    return errors


def _artifact_paths(cell_dir: Path) -> dict[str, Path]:
    return {name: cell_dir / relative for name, relative in REQUIRED_FILES.items()}


def _analyze_cell(
    *,
    sweep_root: Path,
    dataset: dict[str, Any],
    data_seed: int,
    model_seed: int,
    reference_points: np.ndarray | None,
    reference_records: list[dict[str, Any]] | None,
    reference_error: str | None,
) -> dict[str, Any]:
    scope = f"dataset_{data_seed}/seed_{model_seed}"
    dataset_dir = dataset["_output_dir"]
    cell_dir = dataset_dir / f"seed_{model_seed}"
    scaler_path = dataset["_scaler_path"]
    paths = _artifact_paths(cell_dir)
    paths["scaler"] = scaler_path
    errors: list[dict[str, str]] = []
    if not dataset["verification_passed"]:
        errors.append(
            _issue(
                scope,
                "invalid_dataset_contract",
                "shared dataset/scaler artifacts did not pass their dataset-level audit",
            )
        )
    missing: list[str] = []
    if not cell_dir.is_dir():
        errors.append(_issue(scope, "missing_cell_directory", "cell directory is absent", cell_dir))
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
            errors.append(_issue(scope, "missing_or_empty_artifact", f"missing or empty {name}", path))
    if reference_error is not None:
        errors.append(_issue(scope, "reference_unavailable", reference_error))

    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if path.is_file() and path.stat().st_size > 0:
            try:
                artifacts[name] = _file_ref(path)
            except OSError as exc:
                errors.append(_issue(scope, "artifact_hash_error", f"{name}: {exc}", path))

    payloads: dict[str, dict[str, Any]] = {}
    for name in ("checkpoint_sidecar", "history", "training_summary", "diagnose", "metrics", "run_manifest"):
        path = paths[name]
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            payloads[name] = _read_json_object(path)
        except ValueError as exc:
            errors.append(_issue(scope, "invalid_json_artifact", f"{name}: {exc}", path))

    final_losses: dict[str, Any] | None = None
    if "final_losses" in artifacts:
        try:
            final_losses = _parse_key_value(paths["final_losses"])
        except ValueError as exc:
            errors.append(_issue(scope, "invalid_final_losses", str(exc), paths["final_losses"]))
    mg_values: dict[str, Any] | None = None
    if "mg_params_log" in artifacts:
        try:
            mg_values = _parse_key_value(paths["mg_params_log"])
            for message in _validate_mg_log(mg_values):
                errors.append(_issue(scope, "invalid_mg_params_log", message, paths["mg_params_log"]))
        except ValueError as exc:
            errors.append(_issue(scope, "invalid_mg_params_log", str(exc), paths["mg_params_log"]))

    for name in ("morse_graph_png", "morse_sets_png"):
        if name in artifacts:
            try:
                _validate_png(paths[name])
            except ValueError as exc:
                errors.append(_issue(scope, "invalid_render", f"{name}: {exc}", paths[name]))
    for name in ("morse_graph_pdf", "morse_sets_pdf"):
        if name in artifacts:
            try:
                _validate_pdf(paths[name])
            except ValueError as exc:
                errors.append(_issue(scope, "invalid_render", f"{name}: {exc}", paths[name]))

    training: dict[str, Any] | None = None
    if (
        "training_summary" in payloads
        and "history" in payloads
        and final_losses is not None
    ):
        training, training_errors = _training_evidence(
            payloads["training_summary"], payloads["history"], final_losses
        )
        errors.extend(
            _issue(scope, "invalid_training_artifact", message, paths["training_summary"])
            for message in training_errors
        )

    manifest = payloads.get("run_manifest")
    if manifest is not None:
        errors.extend(
            _issue(scope, "run_manifest_mismatch", message, paths["run_manifest"])
            for message in _validate_run_manifest(
                manifest=manifest,
                dataset=dataset,
                cell_dir=cell_dir,
                model_seed=model_seed,
            )
        )

    graph: dict[str, Any] | None = None
    if "morse_graph" in artifacts:
        try:
            graph = _parse_dot(paths["morse_graph"])
        except ValueError as exc:
            errors.append(_issue(scope, "invalid_morse_graph", str(exc), paths["morse_graph"]))
    morse_sets_internal: dict[str, Any] | None = None
    morse_sets: dict[str, Any] | None = None
    if "morse_sets" in artifacts:
        try:
            morse_sets_internal = _parse_morse_sets(paths["morse_sets"])
            morse_sets = {
                key: value
                for key, value in morse_sets_internal.items()
                if not key.startswith("_")
            }
        except ValueError as exc:
            errors.append(_issue(scope, "invalid_morse_sets", str(exc), paths["morse_sets"]))
    if graph is not None and morse_sets_internal is not None:
        graph_ids = {node["id"] for node in graph["nodes"]}
        box_ids = set(morse_sets_internal["boxes_by_node"])
        if graph_ids != box_ids:
            errors.append(
                _issue(
                    scope,
                    "graph_morse_set_label_mismatch",
                    f"graph ids {sorted(graph_ids)} differ from Morse-box ids {sorted(box_ids)}",
                    paths["morse_sets"],
                )
            )
        for node in graph["nodes"]:
            node["n_boxes"] = morse_sets_internal["boxes_by_node"].get(node["id"], 0)

    memberships: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    checkpoint_arch: dict[str, Any] | None = None
    if (
        graph is not None
        and morse_sets_internal is not None
        and reference_points is not None
        and reference_records is not None
        and "checkpoint" in artifacts
        and "checkpoint_sidecar" in artifacts
        and "scaler" in artifacts
    ):
        try:
            scaled, encoded, checkpoint_arch = _encode_reference(
                paths["checkpoint"].parent,
                scaler_path,
                reference_points,
            )
            memberships = _membership_evidence(
                graph=graph,
                morse_sets=morse_sets_internal,
                reference_records=reference_records,
                scaled=scaled,
                encoded=encoded,
            )
            classification = _classify(graph, memberships)
            if checkpoint_arch != dataset["_expected_manifest_config"]["arch"]:
                errors.append(
                    _issue(
                        scope,
                        "checkpoint_architecture_mismatch",
                        "checkpoint sidecar architecture differs from the resolved Ives protocol",
                        paths["checkpoint_sidecar"],
                    )
                )
        except ValueError as exc:
            errors.append(_issue(scope, "invalid_reference_classification", str(exc)))

    complete = cell_dir.is_dir() and not missing and dataset["complete"]
    verification_passed = complete and not errors and classification is not None
    if not complete:
        status = "incomplete"
    elif not verification_passed:
        status = "invalid"
    elif classification["machine_pass"]:
        status = "verified_pass"
    else:
        status = "verified_fail"
    return {
        "data_seed": data_seed,
        "model_seed": model_seed,
        "cell_directory": _display(cell_dir),
        "cell_status": status,
        "complete": complete,
        "verification_passed": verification_passed,
        "machine_pass": classification["machine_pass"] if classification else None,
        "training": training,
        "diagnose": payloads.get("diagnose"),
        "metrics": payloads.get("metrics"),
        "run_manifest": manifest,
        "dataset_provenance": {
            key: value for key, value in dataset.items() if not key.startswith("_")
        },
        "checkpoint_arch": checkpoint_arch,
        "cmgdb_parameters": mg_values,
        "morse_graph": graph,
        "morse_sets": morse_sets,
        "reference_memberships": memberships,
        "classification": classification,
        "artifacts": artifacts,
        "errors": errors,
    }


def _scan_design(sweep_root: Path) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not sweep_root.is_dir():
        return [_issue("sweep", "missing_sweep_root", "sweep root is absent", sweep_root)]
    expected_dataset_names = {f"dataset_{data_seed}" for data_seed in DATA_SEEDS}
    for child in sweep_root.iterdir():
        if not child.is_dir():
            continue
        match = _NUMERIC_DIRECTORY_RE.fullmatch(child.name)
        if match and match.group(1) == "dataset" and child.name not in expected_dataset_names:
            issues.append(
                _issue("sweep", "unexpected_dataset", f"unexpected dataset directory {child.name}", child)
            )
    expected_seed_names = {f"seed_{model_seed}" for model_seed in MODEL_SEEDS}
    for data_seed in DATA_SEEDS:
        dataset_dir = sweep_root / f"dataset_{data_seed}"
        if not dataset_dir.is_dir():
            continue
        for child in dataset_dir.iterdir():
            if not child.is_dir():
                continue
            match = _NUMERIC_DIRECTORY_RE.fullmatch(child.name)
            if match and match.group(1) == "seed" and child.name not in expected_seed_names:
                issues.append(
                    _issue(
                        f"dataset_{data_seed}",
                        "unexpected_model_seed",
                        f"unexpected model-seed directory {child.name}",
                        child,
                    )
                )
    return issues


def _numeric_summary(values: list[Any]) -> dict[str, Any]:
    clean = [float(value) for value in values if _as_float(value) is not None]
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "population_std": statistics.pstdev(clean) if clean else None,
        "median": statistics.median(clean) if clean else None,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def _aggregate(
    *,
    cells: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    sweep_root: Path,
    data_root: Path,
    reference: dict[str, Any] | None,
    global_issues: list[dict[str, str]],
    provisional: bool,
) -> dict[str, Any]:
    selected_keys = sorted(
        {
            key
            for cell in cells
            for key in ((cell.get("training") or {}).get("selected_losses") or {})
        }
    )
    node_counts = Counter(
        cell["morse_graph"]["n_nodes"] for cell in cells if cell.get("morse_graph")
    )
    edge_counts = Counter(
        cell["morse_graph"]["n_edges"] for cell in cells if cell.get("morse_graph")
    )
    statuses = Counter(cell["cell_status"] for cell in cells)
    classification_evaluated = [
        cell for cell in cells if isinstance(cell.get("machine_pass"), bool)
    ]
    issues = [*global_issues, *(error for cell in cells for error in cell["errors"])]
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "provisional": provisional,
        "source_is_read_only": True,
        "sweep_root": _display(sweep_root),
        "data_root": _display(data_root),
        "expected_design": {
            "data_seeds": list(DATA_SEEDS),
            "model_seeds": list(MODEL_SEEDS),
            "n_cells": len(EXPECTED_CELLS),
        },
        "reference_invariant_points": reference,
        "dataset_contract": {
            "train_pairs_per_dataset": EXPECTED_TRAIN_PAIRS,
            "validation_pairs_per_dataset": EXPECTED_VALIDATION_PAIRS,
            "train_sampling_seeds": list(DATA_SEEDS),
            "shared_validation_seed": EXPECTED_VALIDATION_SEED,
            "validation_files_byte_identical": True,
            "training_csv_files_pairwise_distinct": True,
            "scaling": {
                "method": "fixed_bounds",
                "lower_bounds": EXPECTED_AMBIENT_LOWER,
                "upper_bounds": EXPECTED_AMBIENT_UPPER,
                "epsilon": EXPECTED_SCALING_EPSILON,
            },
        },
        "success_criterion": SUCCESS_CRITERION,
        "required_cell_artifacts": {
            **{key: str(value) for key, value in REQUIRED_FILES.items()},
            "scaler": "../scalers/train/scaler.gz (shared by a dataset's three model seeds)",
            "scaler_metadata": "../scalers/train/scaler_metadata.json",
            "train_csv": "DATA_ROOT/dataset_<data_seed>/train.csv",
            "train_metadata": "DATA_ROOT/dataset_<data_seed>/train_metadata.json",
            "validation_csv": "DATA_ROOT/dataset_<data_seed>/val.csv",
            "validation_metadata": "DATA_ROOT/dataset_<data_seed>/val_metadata.json",
        },
        "inventory": {
            "n_expected_cells": len(EXPECTED_CELLS),
            "n_cells": len(cells),
            "n_complete_cells": sum(cell["complete"] for cell in cells),
            "n_verified_cells": sum(cell["verification_passed"] for cell in cells),
            "n_incomplete_cells": sum(not cell["complete"] for cell in cells),
            "n_invalid_cells": sum(cell["cell_status"] == "invalid" for cell in cells),
            "status_counts": dict(sorted(statuses.items())),
            "n_issues": len(issues),
            "issue_counts_by_code": dict(
                sorted(Counter(issue["code"] for issue in issues).items())
            ),
            "issues": issues,
        },
        "classification": {
            "n_evaluated": len(classification_evaluated),
            "n_pass": sum(cell["machine_pass"] is True for cell in classification_evaluated),
            "n_fail": sum(cell["machine_pass"] is False for cell in classification_evaluated),
            "pass_rate_among_evaluated": (
                sum(cell["machine_pass"] is True for cell in classification_evaluated)
                / len(classification_evaluated)
                if classification_evaluated
                else None
            ),
            "n_archive_graph_isomorphic": sum(
                (cell.get("classification") or {}).get("archive_graph_isomorphic") is True
                for cell in cells
            ),
            "n_exact_conley_periods_1_12": sum(
                (cell.get("classification") or {}).get("exact_conley_periods_1_12") is True
                for cell in cells
            ),
            "conley_period_diagnostic_affects_pass": False,
        },
        "training": {
            "epochs_completed": _numeric_summary(
                [(cell.get("training") or {}).get("epochs_completed") for cell in cells]
            ),
            "best_epoch": _numeric_summary(
                [(cell.get("training") or {}).get("best_epoch") for cell in cells]
            ),
            "train_duration_seconds": _numeric_summary(
                [(cell.get("training") or {}).get("train_duration_seconds") for cell in cells]
            ),
            "selected_validation_losses": {
                key: _numeric_summary(
                    [
                        ((cell.get("training") or {}).get("selected_losses") or {}).get(key)
                        for cell in cells
                    ]
                )
                for key in selected_keys
            },
        },
        "topology": {
            "node_count_distribution": {str(key): value for key, value in sorted(node_counts.items())},
            "edge_count_distribution": {str(key): value for key, value in sorted(edge_counts.items())},
            "sink_count_distribution": dict(
                sorted(
                    Counter(
                        len(cell["morse_graph"]["sink_ids"])
                        for cell in cells
                        if cell.get("morse_graph")
                    ).items()
                )
            ),
        },
        "datasets": [
            {key: value for key, value in dataset.items() if not key.startswith("_")}
            for dataset in datasets
        ],
        "cell_outcomes": [
            {
                "data_seed": cell["data_seed"],
                "model_seed": cell["model_seed"],
                "status": cell["cell_status"],
                "complete": cell["complete"],
                "verification_passed": cell["verification_passed"],
                "machine_pass": cell["machine_pass"],
                "classification": cell["classification"],
                "error_codes": [error["code"] for error in cell["errors"]],
            }
            for cell in cells
        ],
        "derived_artifacts": {
            "cells_csv": "summary/cells.csv",
            "cells_json": "summary/cells.json",
            "aggregate_summary": "summary/aggregate_summary.json",
            "markdown": "summary/SUMMARY.md",
        },
    }


def audit_sweep(
    *,
    sweep_root: Path = DEFAULT_SWEEP_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    reference_csv: Path = DEFAULT_REFERENCE_CSV,
    provisional: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    sweep_root = sweep_root.resolve()
    data_root = data_root.resolve()
    reference_csv = reference_csv.resolve()
    global_issues = _scan_design(sweep_root)
    if not data_root.is_dir():
        global_issues.append(_issue("sweep", "missing_data_root", "data root is absent", data_root))
    else:
        expected_dataset_names = {f"dataset_{data_seed}" for data_seed in DATA_SEEDS}
        for child in data_root.iterdir():
            if (
                child.is_dir()
                and _NUMERIC_DIRECTORY_RE.fullmatch(child.name)
                and child.name not in expected_dataset_names
            ):
                global_issues.append(
                    _issue(
                        "sweep",
                        "unexpected_data_dataset",
                        f"unexpected data directory {child.name}",
                        child,
                    )
                )
    reference: dict[str, Any] | None = None
    reference_points: np.ndarray | None = None
    reference_records: list[dict[str, Any]] | None = None
    reference_error: str | None = None
    try:
        reference, reference_points, reference_records = _load_reference_csv(reference_csv)
    except ValueError as exc:
        reference_error = str(exc)
        global_issues.append(
            _issue("sweep", "invalid_reference_csv", reference_error, reference_csv)
        )
    datasets = [
        _analyze_dataset(
            data_seed=data_seed,
            data_root=data_root,
            sweep_root=sweep_root,
        )
        for data_seed in DATA_SEEDS
    ]
    global_issues.extend(error for dataset in datasets for error in dataset["errors"])
    train_hashes = [
        (dataset.get("artifacts", {}).get("train_csv") or {}).get("sha256")
        for dataset in datasets
    ]
    validation_hashes = [
        (dataset.get("artifacts", {}).get("validation_csv") or {}).get("sha256")
        for dataset in datasets
    ]
    validation_metadata_hashes = [
        (dataset.get("artifacts", {}).get("validation_metadata") or {}).get("sha256")
        for dataset in datasets
    ]
    if all(train_hashes) and len(set(train_hashes)) != len(DATA_SEEDS):
        global_issues.append(
            _issue(
                "sweep",
                "training_datasets_not_distinct",
                "the five train.csv files are not pairwise byte-distinct",
            )
        )
    if all(validation_hashes) and len(set(validation_hashes)) != 1:
        global_issues.append(
            _issue(
                "sweep",
                "validation_datasets_differ",
                "the five val.csv files are not byte-identical",
            )
        )
    if all(validation_metadata_hashes) and len(set(validation_metadata_hashes)) != 1:
        global_issues.append(
            _issue(
                "sweep",
                "validation_metadata_differ",
                "the five val_metadata.json files are not byte-identical",
            )
        )
    dataset_by_seed = {dataset["data_seed"]: dataset for dataset in datasets}
    cells = [
        _analyze_cell(
            sweep_root=sweep_root,
            dataset=dataset_by_seed[data_seed],
            data_seed=data_seed,
            model_seed=model_seed,
            reference_points=reference_points,
            reference_records=reference_records,
            reference_error=reference_error,
        )
        for data_seed, model_seed in EXPECTED_CELLS
    ]
    aggregate = _aggregate(
        cells=cells,
        datasets=datasets,
        sweep_root=sweep_root,
        data_root=data_root,
        reference=reference,
        global_issues=global_issues,
        provisional=provisional,
    )
    return cells, aggregate, reference


def _strict_error(aggregate: dict[str, Any]) -> SweepValidationError | None:
    inventory = aggregate["inventory"]
    if (
        inventory["n_cells"] == len(EXPECTED_CELLS)
        and inventory["n_complete_cells"] == len(EXPECTED_CELLS)
        and inventory["n_verified_cells"] == len(EXPECTED_CELLS)
        and inventory["n_issues"] == 0
    ):
        return None
    snippets = [
        f"{issue['scope']}/{issue['code']}: {issue['message']}"
        for issue in inventory["issues"][:8]
    ]
    suffix = "" if inventory["n_issues"] <= 8 else f"; plus {inventory['n_issues'] - 8} more"
    detail = "; ".join(snippets) + suffix if snippets else "unknown validation failure"
    return SweepValidationError(
        "strict Ives Mývatn summary refused: "
        f"verified {inventory['n_verified_cells']}/{len(EXPECTED_CELLS)} cells; {detail}"
    )


def _csv_row(cell: dict[str, Any]) -> dict[str, Any]:
    graph = cell.get("morse_graph") or {}
    classification = cell.get("classification") or {}
    training = cell.get("training") or {}
    artifacts = cell.get("artifacts") or {}
    by_id = {node["id"]: node for node in graph.get("nodes", [])}
    sink_ids = graph.get("sink_ids", [])
    return {
        "data_seed": cell["data_seed"],
        "model_seed": cell["model_seed"],
        "cell_status": cell["cell_status"],
        "complete": cell["complete"],
        "verification_passed": cell["verification_passed"],
        "machine_pass": cell["machine_pass"],
        "archive_graph_isomorphic": classification.get("archive_graph_isomorphic"),
        "n_morse_nodes": graph.get("n_nodes"),
        "n_morse_edges": graph.get("n_edges"),
        "n_sinks": len(sink_ids) if graph else None,
        "sink_ids": _json_text(sink_ids),
        "fixed_sink_id": classification.get("fixed_sink_id"),
        "cycle_sink_id": classification.get("cycle_sink_id"),
        "fixed_sink_memberships": _json_text(classification.get("fixed_sink_memberships")),
        "cycle_unique_target_count": classification.get("cycle_unique_target_count"),
        "cycle_unassigned_count": classification.get("cycle_unassigned_count"),
        "cycle_conflicting_phase_count": classification.get("cycle_conflicting_phase_count"),
        "cycle_phase_sink_memberships": _json_text(
            classification.get("cycle_phase_sink_memberships")
        ),
        "sink_conley_tuples": _json_text(
            [by_id[sink]["conley_tuple"] for sink in sink_ids if sink in by_id]
        ),
        "sink_inferred_periods": _json_text(classification.get("sink_inferred_periods")),
        "exact_conley_periods_1_12": classification.get("exact_conley_periods_1_12"),
        "epochs_completed": training.get("epochs_completed"),
        "best_epoch": training.get("best_epoch"),
        "selected_loss_total": training.get("selected_loss_total"),
        "selected_losses": _json_text(training.get("selected_losses")),
        "train_duration_seconds": training.get("train_duration_seconds"),
        "checkpoint_sha256": (artifacts.get("checkpoint") or {}).get("sha256"),
        "morse_graph_sha256": (artifacts.get("morse_graph") or {}).get("sha256"),
        "morse_sets_sha256": (artifacts.get("morse_sets") or {}).get("sha256"),
        "scaler_sha256": (artifacts.get("scaler") or {}).get("sha256"),
        "artifact_hashes": _json_text(
            {name: reference.get("sha256") for name, reference in artifacts.items()}
        ),
        "classification_evidence": _json_text(classification or None),
        "error_count": len(cell["errors"]),
        "errors": _json_text(cell["errors"]),
        "cell_directory": cell["cell_directory"],
    }


def _markdown(cells: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    inventory = aggregate["inventory"]
    classification = aggregate["classification"]
    status = (
        "PROVISIONAL / INCOMPLETE"
        if aggregate["provisional"]
        else "COMPLETE AND STRICTLY VERIFIED"
    )
    lines = [
        "# Ives Lake Mývatn 3x5 replication summary",
        "",
        f"**Report status:** {status}",
        "",
        f"Verified cells: {inventory['n_verified_cells']}/15; complete artifact sets: "
        f"{inventory['n_complete_cells']}/15; machine passes: {classification['n_pass']}/"
        f"{classification['n_evaluated']} evaluated.",
        "",
        "A machine pass requires a directed graph isomorphic to the archived four-node "
        "branch-then-chain graph, a fixed point in exactly one sink, and at least 11 of "
        "12 cycle phases uniquely in the other sink, with no phase in a conflicting sink. "
        "The sink-period set `{1, 12}` is reported only as a Conley-index diagnostic because "
        "the archived JSON did not retain indices.",
        "",
        "| data seed | model seed | status | nodes | edges | sinks | fixed sink | cycle sink | cycle phases | conflicts | periods | pass |",
        "|---:|---:|:---|---:|---:|---:|:---|:---|---:|---:|:---|:---:|",
    ]
    for cell in cells:
        graph = cell.get("morse_graph") or {}
        evidence = cell.get("classification") or {}
        lines.append(
            f"| {cell['data_seed']} | {cell['model_seed']} | {cell['cell_status']} | "
            f"{graph.get('n_nodes', '—')} | {graph.get('n_edges', '—')} | "
            f"{len(graph.get('sink_ids', [])) if graph else '—'} | "
            f"{evidence.get('fixed_sink_id', '—')} | {evidence.get('cycle_sink_id', '—')} | "
            f"{evidence.get('cycle_unique_target_count', '—')}/12 | "
            f"{evidence.get('cycle_conflicting_phase_count', '—')} | "
            f"`{_json_text(evidence.get('sink_inferred_periods'))}` | "
            f"{cell.get('machine_pass', '—')} |"
        )
    lines.extend(
        [
            "",
            "## Provenance and evidence",
            "",
            "`cells.json` retains every DOT node id, exact Conley tuple and components, "
            "directed edge, degree, sink/minimal flag, inferred period, all 13 encoded "
            "reference points, and every per-point Morse-node and sink membership. "
            "`cells.csv` includes the compact classification evidence, selected training "
            "losses, and SHA-256 hashes for every required artifact.",
        ]
    )
    if inventory["issues"]:
        lines.extend(["", "## Validation issues", ""])
        for issue in inventory["issues"]:
            location = f" (`{issue['path']}`)" if "path" in issue else ""
            lines.append(
                f"- `{issue['scope']}` / `{issue['code']}`: {issue['message']}{location}"
            )
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_reports(
    *,
    summary_dir: Path,
    cells: list[dict[str, Any]],
    aggregate: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Path]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cells_csv": summary_dir / "cells.csv",
        "cells_json": summary_dir / "cells.json",
        "aggregate_summary": summary_dir / "aggregate_summary.json",
        "summary_markdown": summary_dir / "SUMMARY.md",
    }
    csv_temporary = outputs["cells_csv"].with_name(
        f".{outputs['cells_csv'].name}.tmp.{os.getpid()}"
    )
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(_csv_row(cell) for cell in cells)
    csv_temporary.replace(outputs["cells_csv"])
    detailed = {
        "schema_version": 1,
        "generated_at_utc": aggregate["generated_at_utc"],
        "provisional": aggregate["provisional"],
        "expected_design": aggregate["expected_design"],
        "success_criterion": SUCCESS_CRITERION,
        "reference_invariant_points": reference,
        "cells": cells,
    }
    _atomic_text(
        outputs["cells_json"],
        json.dumps(detailed, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_text(
        outputs["aggregate_summary"],
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_text(outputs["summary_markdown"], _markdown(cells, aggregate))
    return outputs


def build_summary(
    *,
    sweep_root: Path = DEFAULT_SWEEP_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    reference_csv: Path = DEFAULT_REFERENCE_CSV,
    summary_dir: Path | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Path]:
    cells, aggregate, reference = audit_sweep(
        sweep_root=sweep_root,
        data_root=data_root,
        reference_csv=reference_csv,
        provisional=allow_incomplete,
    )
    strict_error = _strict_error(aggregate)
    if strict_error is not None and not allow_incomplete:
        raise strict_error
    destination = summary_dir.resolve() if summary_dir else sweep_root.resolve() / "summary"
    return _write_reports(
        summary_dir=destination,
        cells=cells,
        aggregate=aggregate,
        reference=reference,
    )


def _verify_existing_reports(
    *,
    sweep_root: Path,
    cells: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    """When reports exist, verify their grid, counts, and source hashes.

    ``--verify`` remains useful before the first report is written (it then
    audits the source tree only).  Once any final report exists, however, all
    four must exist and agree with a fresh read of the source artifacts.  This
    prevents a launcher's immutable-completion guard from accepting a stale or
    partially written summary directory.
    """

    summary_dir = sweep_root.resolve() / "summary"
    paths = {
        "cells_csv": summary_dir / "cells.csv",
        "cells_json": summary_dir / "cells.json",
        "aggregate_summary": summary_dir / "aggregate_summary.json",
        "summary_markdown": summary_dir / "SUMMARY.md",
    }
    present = {name: path.is_file() and path.stat().st_size > 0 for name, path in paths.items()}
    if not any(present.values()):
        return
    missing = [name for name, available in present.items() if not available]
    if missing:
        raise SweepValidationError(
            "existing summary is partial; missing or empty " + ", ".join(missing)
        )
    try:
        saved_aggregate = _read_json_object(paths["aggregate_summary"])
        saved_detailed = _read_json_object(paths["cells_json"])
    except ValueError as exc:
        raise SweepValidationError(f"existing JSON summary is invalid: {exc}") from exc
    saved_inventory = saved_aggregate.get("inventory")
    saved_classification = saved_aggregate.get("classification")
    if not isinstance(saved_inventory, dict) or not isinstance(saved_classification, dict):
        raise SweepValidationError("existing aggregate lacks inventory or classification")
    expected_counts = {
        "n_expected_cells": len(EXPECTED_CELLS),
        "n_complete_cells": len(EXPECTED_CELLS),
        "n_verified_cells": len(EXPECTED_CELLS),
        "n_incomplete_cells": 0,
        "n_invalid_cells": 0,
        "n_issues": 0,
    }
    observed_counts = {key: saved_inventory.get(key) for key in expected_counts}
    if saved_aggregate.get("provisional") is not False or observed_counts != expected_counts:
        raise SweepValidationError(
            f"existing aggregate is not a strict final report: {observed_counts!r}"
        )
    for key in ("n_evaluated", "n_pass", "n_fail"):
        if saved_classification.get(key) != aggregate["classification"].get(key):
            raise SweepValidationError(
                f"existing aggregate classification.{key} is stale: "
                f"{saved_classification.get(key)!r} != {aggregate['classification'].get(key)!r}"
            )

    saved_cells = saved_detailed.get("cells")
    if saved_detailed.get("provisional") is not False or not isinstance(saved_cells, list):
        raise SweepValidationError("existing cells.json is provisional or lacks a cells list")
    saved_by_key = {
        (_as_int(cell.get("data_seed")), _as_int(cell.get("model_seed"))): cell
        for cell in saved_cells
        if isinstance(cell, dict)
    }
    current_by_key = {(cell["data_seed"], cell["model_seed"]): cell for cell in cells}
    if len(saved_cells) != len(EXPECTED_CELLS) or set(saved_by_key) != set(current_by_key):
        raise SweepValidationError("existing cells.json does not contain the exact 15-cell grid")
    for key, current in current_by_key.items():
        saved = saved_by_key[key]
        if saved.get("machine_pass") is not current.get("machine_pass"):
            raise SweepValidationError(f"existing cells.json classification is stale for {key}")
        saved_hashes = {
            name: reference.get("sha256")
            for name, reference in (saved.get("artifacts") or {}).items()
            if isinstance(reference, dict)
        }
        current_hashes = {
            name: reference.get("sha256")
            for name, reference in current["artifacts"].items()
        }
        if saved_hashes != current_hashes:
            raise SweepValidationError(f"existing cells.json artifact hashes are stale for {key}")

    try:
        with paths["cells_csv"].open(encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise SweepValidationError(f"cannot parse existing cells.csv: {exc}") from exc
    csv_by_key = {
        (_as_int(row.get("data_seed")), _as_int(row.get("model_seed"))): row
        for row in csv_rows
    }
    if len(csv_rows) != len(EXPECTED_CELLS) or set(csv_by_key) != set(current_by_key):
        raise SweepValidationError("existing cells.csv does not contain the exact 15-cell grid")
    for key, current in current_by_key.items():
        row = csv_by_key[key]
        for column, artifact_name in (
            ("checkpoint_sha256", "checkpoint"),
            ("morse_graph_sha256", "morse_graph"),
            ("morse_sets_sha256", "morse_sets"),
            ("scaler_sha256", "scaler"),
        ):
            expected_hash = current["artifacts"][artifact_name]["sha256"]
            if row.get(column) != expected_hash:
                raise SweepValidationError(
                    f"existing cells.csv {column} is stale for {key}"
                )
    try:
        markdown = paths["summary_markdown"].read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SweepValidationError(f"cannot parse existing SUMMARY.md: {exc}") from exc
    if not markdown.startswith("# Ives Lake Mývatn 3x5 replication summary\n"):
        raise SweepValidationError("existing SUMMARY.md has an unexpected or invalid header")


def verify_sweep(
    *,
    sweep_root: Path = DEFAULT_SWEEP_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    reference_csv: Path = DEFAULT_REFERENCE_CSV,
) -> dict[str, Any]:
    cells, aggregate, _reference = audit_sweep(
        sweep_root=sweep_root,
        data_root=data_root,
        reference_csv=reference_csv,
        provisional=False,
    )
    strict_error = _strict_error(aggregate)
    if strict_error is not None:
        raise strict_error
    _verify_existing_reports(sweep_root=sweep_root, cells=cells, aggregate=aggregate)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--summary-dir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--allow-incomplete", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            aggregate = verify_sweep(
                sweep_root=args.sweep_root,
                data_root=args.data_root,
                reference_csv=args.reference_csv,
            )
            print(
                json.dumps(
                    {
                        "verified": True,
                        "n_verified_cells": aggregate["inventory"]["n_verified_cells"],
                        "n_machine_pass": aggregate["classification"]["n_pass"],
                    },
                    indent=2,
                )
            )
            return 0
        outputs = build_summary(
            sweep_root=args.sweep_root,
            data_root=args.data_root,
            reference_csv=args.reference_csv,
            summary_dir=args.summary_dir,
            allow_incomplete=args.allow_incomplete,
        )
    except (SweepValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({name: _display(path) for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
