"""Aggregate and verify a Leslie3D Example2, 5-dataset x 3-seed sweep.

Defaults point to the T=40 sweep and require T=40.  Alternate sweep/data roots,
trajectory length, and expected train/validation initial-condition counts can
be supplied through the Python API or CLI, so the same strict analyzer can be
used for follow-up designs such as T=25 with 40,000/10,000 initial conditions.

The training, CMGDB, and rendering artifacts are treated as immutable inputs.
When a legacy sweep omitted ``metrics.json``, ``--metrics-root`` may point at a
separate, derived metrics replay tree with the same ``dataset_N/seed_N``
layout.  This script only writes three derived files below
``SWEEP_ROOT/analysis`` (or ``--analysis-dir``):

* ``cells.csv`` -- one flattened row per expected dataset/model-seed cell;
* ``cells.json`` -- detailed parsed records, dataset metadata, and provenance;
* ``aggregate_summary.json`` -- sweep-level success, loss, duration, and
  topology summaries.

By default all 15 cells and all required analysis inputs must exist and pass
cross-file consistency checks before any report is written.  During a running
sweep, ``--allow-incomplete`` writes an explicitly provisional inventory.
Neither mode modifies source data, checkpoints, or CMGDB artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_ROOT = CODE_ROOT / "output" / "leslie3d_example2_seedsweep_t40"
DEFAULT_DATA_ROOT = CODE_ROOT / "data" / "leslie3d_example2_seedsweep_t40"
EXPECTED_DATASET_IDS = (1, 2, 3, 4, 5)
EXPECTED_MODEL_SEEDS = (0, 1, 2)
EXPECTED_T = 40
EXPECTED_BACKEND = "adaptive_precomputed"
TARGET_CONLEY_INDEX = ("x^4-1", "0", "0")
PERIODIC_H0_RE = re.compile(r"^x(?:\^([1-9]\d*))?-1$")

REQUIRED_CELL_FILES = {
    "morse_graph": Path("MG/morse_graph"),
    "morse_sets": Path("MG/morse_sets"),
    "metrics": Path("metrics.json"),
    "training_summary": Path("training_summary.json"),
    "diagnose": Path("diagnose.json"),
    "mg_params_log": Path("mg_params_log.txt"),
    "run_manifest": Path("run_manifest.json"),
}

NODE_RE = re.compile(r'^\s*"?(\d+)"?\s*\[(?P<attrs>[^\]]*)\]\s*;?\s*$')
EDGE_RE = re.compile(r'^\s*"?(\d+)"?\s*->\s*"?(\d+)"?\s*;?\s*$')
LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')
CONLEY_RE = re.compile(r"\(([^()]*)\)")

SUCCESS_CRITERION = {
    "name": "exact_two_period4_sink_indices",
    "definition": (
        "The parsed Morse graph has exactly two graph sinks, and the Conley-index "
        "tuple on each sink is exactly (x^4-1, 0, 0): x^4-1 in homological "
        "degree 0 and zero in the other saved degrees."
    ),
    "sink_count": 2,
    "required_sink_conley_index": list(TARGET_CONLEY_INDEX),
    "uses_graph_sinks_not_tolerance_classification": True,
}

MARCIO_STYLE_CRITERION = {
    "name": "exactly_two_nonzero_degree0_nodes",
    "definition": (
        "The parsed Morse graph has exactly two nodes whose Conley-index tuple "
        "has a nonzero homological degree-0 component. All Morse nodes are counted, "
        "not only graph sinks; higher-degree components do not disqualify a node, so "
        "both (x-1, x-1, 0) and (x^p-1, 0, 0) qualify as one stable-index node."
    ),
    "required_node_count": 2,
    "uses_all_morse_nodes": True,
    "uses_graph_sinks": False,
    "uses_tolerance_classification": False,
}

MINIMAL_NODE_CRITERION = {
    "name": "exactly_two_graph_minimal_nodes",
    "definition": (
        "The parsed Morse graph has exactly two minimal nodes (graph sinks, "
        "equivalently nodes with no outgoing edge). Conley-index type and sampled "
        "tolerance do not alter this topology-only classification."
    ),
    "required_node_count": 2,
    "uses_all_morse_nodes": False,
    "uses_graph_sinks": True,
    "uses_tolerance_classification": False,
}

PERIODIC_BISTABILITY_CRITERION = {
    "name": "exactly_two_periodic_index_minimal_nodes",
    "definition": (
        "The parsed Morse graph has exactly two minimal nodes, and each has "
        "Conley-index tuple (x^p-1, 0, 0) for an integer p >= 1. The two "
        "periods may differ. This is the sweep's requested bistability pass."
    ),
    "required_node_count": 2,
    "required_index_family": ["x^p-1", "0", "0"],
    "uses_graph_sinks": True,
    "uses_tolerance_classification": False,
}

CSV_FIELDS = (
    "dataset_id",
    "data_seed",
    "validation_seed",
    "model_seed",
    "trajectory_length_T",
    "discarded_steps_T0",
    "train_initial_conditions",
    "validation_initial_conditions",
    "train_transition_pairs_expected",
    "train_transition_pairs_observed",
    "validation_transition_pairs_expected",
    "validation_transition_pairs_observed",
    "n_morse_nodes",
    "n_morse_edges",
    "n_graph_sinks",
    "graph_sink_labels",
    "metrics_minimal_labels",
    "sink_conley_indices",
    "node_conley_indices",
    "n_attractor_type_nodes",
    "attractor_type_labels",
    "attractor_type_conley_indices",
    "marcio_style_success",
    "minimal_node_success",
    "periodic_bistability_success",
    "morse_boxes_total",
    "morse_boxes_by_label",
    "minimal_tolerance_details",
    "n_minimal_tolerance_failures",
    "all_minimal_tolerance_pass",
    "tolerance_status",
    "exact_conley_success",
    "diagnostic",
    "encoder_collapsed",
    "latent_map_overcontracted",
    "best_epoch",
    "n_epochs_run",
    "train_duration_seconds",
    "cmgdb_duration_seconds",
    "combined_train_cmgdb_duration_seconds",
    "train_reconstruction_final",
    "train_prediction_final",
    "train_semiconjugacy_final",
    "train_total_final",
    "validation_reconstruction_final",
    "validation_prediction_final",
    "validation_semiconjugacy_final",
    "validation_total_final",
    "train_total_best_epoch_value",
    "validation_total_best_epoch_value",
    "box_map_backend_manifest",
    "box_map_backend_log",
    "box_map_backend_is_explicit",
    "subdiv_init",
    "subdiv_min",
    "subdiv_max",
    "precompute_subdiv_role",
    "precompute_subdiv",
    "precompute_lattice_dimension",
    "precompute_axis_depth_M",
    "precompute_axis_depths",
    "precompute_cells_per_axis",
    "precompute_corners_per_axis",
    "precompute_lattice_shape",
    "precompute_table_points",
    "config_hash",
    "train_csv_sha256",
    "validation_csv_sha256",
    "model_sha256",
    "morse_graph_sha256",
    "morse_sets_sha256",
    "topology_signature_sha256",
    "complete",
    "verification_passed",
    "error_count",
    "warning_count",
    "errors",
    "warnings",
    "cell_directory",
)


class SweepValidationError(RuntimeError):
    """Raised when strict analysis encounters missing or inconsistent inputs."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(CODE_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_reference(path: Path, *, known_sha256: str | None = None) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": known_sha256 or _sha256(path),
    }


def _read_json_object(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse JSON object from {_display_path(path)}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{_display_path(path)} must contain a JSON object")
    return payload, _file_reference(path)


def _issue(code: str, message: str, path: Path | None = None) -> dict[str, str]:
    out = {"code": code, "message": message}
    if path is not None:
        out["path"] = _display_path(path)
    return out


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _deep_get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _scan_dataset_csv(path: Path) -> dict[str, Any]:
    """Hash and count a generated trajectory-pair CSV in one streaming pass."""
    digest = hashlib.sha256()
    header: str | None = None
    nonempty_lines = 0
    size = 0
    try:
        with path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                size += len(raw)
                if not raw.strip():
                    continue
                nonempty_lines += 1
                if header is None:
                    header = raw.decode("utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot scan {_display_path(path)}: {exc}") from exc
    if header is None:
        raise ValueError(f"{_display_path(path)} is empty")
    columns = [column.strip() for column in header.split(",")]
    return {
        "file": {
            "path": _display_path(path),
            "size_bytes": size,
            "sha256": digest.hexdigest(),
        },
        "header": columns,
        "transition_pairs_observed": nonempty_lines - 1,
    }


def _parse_conley_label(label: str | None) -> tuple[str, ...] | None:
    if not label:
        return None
    match = CONLEY_RE.search(label)
    if match is None:
        return None
    return tuple(part.strip().replace(" ", "") for part in match.group(1).split(","))


def _is_periodic_bistability_index(index: list[str] | None) -> bool:
    """Return whether an index belongs to ``(x^p-1, 0, 0)``, p >= 1."""
    return bool(
        index
        and len(index) == 3
        and PERIODIC_H0_RE.fullmatch(index[0])
        and index[1:] == ["0", "0"]
    )


def _parse_morse_graph(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    nodes: dict[int, str] = {}
    edges: set[tuple[int, int]] = set()
    for raw in text.splitlines():
        node_match = NODE_RE.match(raw)
        if node_match is not None:
            node_id = int(node_match.group(1))
            label_match = LABEL_RE.search(node_match.group("attrs"))
            nodes[node_id] = label_match.group(1) if label_match is not None else ""
            continue
        edge_match = EDGE_RE.match(raw)
        if edge_match is not None:
            edges.add((int(edge_match.group(1)), int(edge_match.group(2))))
    if not nodes:
        raise ValueError(f"{_display_path(path)} contains no integer Morse nodes")
    unknown = sorted({node for edge in edges for node in edge if node not in nodes})
    if unknown:
        raise ValueError(
            f"{_display_path(path)} has edges referencing unknown nodes {unknown}"
        )
    sources = {source for source, _ in edges}
    sinks = sorted(set(nodes) - sources)
    parsed_nodes = []
    for node_id in sorted(nodes):
        conley = _parse_conley_label(nodes[node_id])
        parsed_nodes.append(
            {
                "id": node_id,
                "dot_label": nodes[node_id],
                "conley_index": list(conley) if conley is not None else None,
                "is_sink": node_id in sinks,
            }
        )
    attractor_type_nodes = [
        node
        for node in parsed_nodes
        if node["conley_index"]
        and node["conley_index"][0] not in ("", "0")
    ]
    canonical_topology = {
        "nodes": [
            {"id": node["id"], "conley_index": node["conley_index"]}
            for node in parsed_nodes
        ],
        "edges": [list(edge) for edge in sorted(edges)],
        "sinks": sinks,
    }
    signature = hashlib.sha256(
        json.dumps(canonical_topology, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "nodes": parsed_nodes,
        "edges": [list(edge) for edge in sorted(edges)],
        "sinks": sinks,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "attractor_type_nodes": attractor_type_nodes,
        "n_attractor_type_nodes": len(attractor_type_nodes),
        "marcio_style_success": len(attractor_type_nodes) == 2,
        "topology_signature_sha256": signature,
        "file": _file_reference(path),
    }


def _parse_morse_sets(path: Path) -> dict[str, Any]:
    """Stream raw CMGDB boxes without loading a potentially huge file."""
    digest = hashlib.sha256()
    counts: Counter[int] = Counter()
    n_columns: int | None = None
    size = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            size += len(raw)
            if not raw.strip():
                continue
            try:
                parts = raw.decode("utf-8").strip().split(",")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{_display_path(path)}:{line_number} is not UTF-8"
                ) from exc
            if n_columns is None:
                n_columns = len(parts)
            if len(parts) != n_columns:
                raise ValueError(
                    f"{_display_path(path)}:{line_number} has {len(parts)} columns; "
                    f"expected {n_columns}"
                )
            if len(parts) < 3 or len(parts) % 2 == 0:
                raise ValueError(
                    f"{_display_path(path)}:{line_number} has invalid Morse-box shape"
                )
            try:
                coordinates = [float(value) for value in parts[:-1]]
                label_float = float(parts[-1])
            except ValueError as exc:
                raise ValueError(
                    f"{_display_path(path)}:{line_number} contains a non-numeric value"
                ) from exc
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError(
                    f"{_display_path(path)}:{line_number} contains non-finite coordinates"
                )
            if not math.isfinite(label_float) or not label_float.is_integer():
                raise ValueError(
                    f"{_display_path(path)}:{line_number} has non-integer label {parts[-1]!r}"
                )
            counts[int(label_float)] += 1
    if n_columns is None:
        raise ValueError(f"{_display_path(path)} contains no Morse boxes")
    return {
        "dimension": (n_columns - 1) // 2,
        "n_columns": n_columns,
        "total_boxes": sum(counts.values()),
        "boxes_by_label": {str(label): counts[label] for label in sorted(counts)},
        "file": {
            "path": _display_path(path),
            "size_bytes": size,
            "sha256": digest.hexdigest(),
        },
    }


def _parse_mg_params(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        if ":" not in raw:
            raise ValueError(f"{_display_path(path)}:{line_number} lacks ':'")
        key, raw_value = raw.split(":", 1)
        text = raw_value.strip()
        if text in {"True", "False"}:
            value: Any = text == "True"
        else:
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                value = text
        values[key.strip()] = value
    return {"parameters": values, "file": _file_reference(path)}


def _analyze_dataset(
    dataset_id: int,
    data_root: Path,
    expected_t: int,
    *,
    expected_train_initial_conditions: int | None = None,
    expected_validation_initial_conditions: int | None = None,
) -> dict[str, Any]:
    directory = data_root / f"dataset_{dataset_id}"
    errors: list[dict[str, str]] = []
    files: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    scans: dict[str, dict[str, Any]] = {}
    required = {
        "train_metadata": directory / "train_metadata.json",
        "validation_metadata": directory / "val_metadata.json",
        "train_csv": directory / "train.csv",
        "validation_csv": directory / "val.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    for name in missing:
        errors.append(_issue("missing_dataset_artifact", f"missing {name}", required[name]))

    for role, filename in (("train", "train_metadata.json"), ("validation", "val_metadata.json")):
        path = directory / filename
        if not path.is_file():
            continue
        try:
            payload, reference = _read_json_object(path)
            metadata[role] = payload
            files[f"{role}_metadata"] = reference
        except ValueError as exc:
            errors.append(_issue("invalid_dataset_metadata", str(exc), path))

    for role, filename in (("train", "train.csv"), ("validation", "val.csv")):
        path = directory / filename
        if not path.is_file():
            continue
        try:
            scans[role] = _scan_dataset_csv(path)
            files[f"{role}_csv"] = scans[role]["file"]
        except ValueError as exc:
            errors.append(_issue("invalid_dataset_csv", str(exc), path))

    for role in ("train", "validation"):
        meta = metadata.get(role, {})
        scan = scans.get(role, {})
        t = _as_int(meta.get("n_iterations"))
        skip = _as_int(meta.get("skip_initial_steps"))
        n_initial = _as_int(meta.get("n_samples"))
        if meta and (t is None or skip is None or n_initial is None):
            errors.append(
                _issue(
                    "invalid_dataset_counts",
                    f"{role} metadata requires integer n_samples, n_iterations, and skip_initial_steps",
                    directory / ("train_metadata.json" if role == "train" else "val_metadata.json"),
                )
            )
            continue
        if t is not None and t != expected_t:
            errors.append(
                _issue(
                    "unexpected_trajectory_length",
                    f"{role} n_iterations is {t}; expected T={expected_t}",
                )
            )
        expected_initial_conditions = (
            expected_train_initial_conditions
            if role == "train"
            else expected_validation_initial_conditions
        )
        if (
            expected_initial_conditions is not None
            and n_initial is not None
            and n_initial != expected_initial_conditions
        ):
            errors.append(
                _issue(
                    "unexpected_initial_condition_count",
                    f"{role} metadata n_samples is {n_initial}; "
                    f"expected {expected_initial_conditions}",
                    directory
                    / ("train_metadata.json" if role == "train" else "val_metadata.json"),
                )
            )
        if None not in (t, skip, n_initial):
            retained = t - skip
            if retained <= 0:
                errors.append(
                    _issue("invalid_retained_steps", f"{role} retains {retained} steps")
                )
            else:
                expected_pairs = n_initial * retained
                observed_pairs = scan.get("transition_pairs_observed")
                if observed_pairs is not None and observed_pairs != expected_pairs:
                    errors.append(
                        _issue(
                            "dataset_pair_count_mismatch",
                            f"{role} CSV has {observed_pairs} pairs; metadata implies {expected_pairs}",
                        )
                    )

    if metadata.get("train") and metadata.get("validation"):
        for key in ("n_iterations", "skip_initial_steps", "dimension", "system"):
            if metadata["train"].get(key) != metadata["validation"].get(key):
                errors.append(
                    _issue(
                        "train_validation_metadata_mismatch",
                        f"train and validation metadata differ for {key}",
                    )
                )

    def role_summary(role: str) -> dict[str, Any]:
        meta = metadata.get(role, {})
        scan = scans.get(role, {})
        n_initial = _as_int(meta.get("n_samples"))
        t = _as_int(meta.get("n_iterations"))
        skip = _as_int(meta.get("skip_initial_steps"))
        expected_pairs = None
        if None not in (n_initial, t, skip):
            expected_pairs = n_initial * (t - skip)
        return {
            "metadata": meta or None,
            "initial_conditions": n_initial,
            "sampling_seed": _as_int(meta.get("sampling_seed")),
            "trajectory_length_T": t,
            "discarded_steps_T0": skip,
            "transition_pairs_expected": expected_pairs,
            "transition_pairs_observed": scan.get("transition_pairs_observed"),
            "csv_header": scan.get("header"),
        }

    return {
        "dataset_id": dataset_id,
        "directory": _display_path(directory),
        "train": role_summary("train"),
        "validation": role_summary("validation"),
        "files": files,
        "complete": not missing and len(files) == len(required),
        "verification_passed": not errors and not missing,
        "errors": errors,
    }


def _dataset_design_summary(
    datasets: list[dict[str, Any]], expected_dataset_count: int
) -> dict[str, Any]:
    """Verify the independent-training/shared-holdout sweep design."""
    train_hashes_by_dataset = {
        str(dataset["dataset_id"]): _deep_get(dataset, "files", "train_csv", "sha256")
        for dataset in datasets
    }
    validation_hashes_by_dataset = {
        str(dataset["dataset_id"]): _deep_get(
            dataset, "files", "validation_csv", "sha256"
        )
        for dataset in datasets
    }
    validation_seeds_by_dataset = {
        str(dataset["dataset_id"]): dataset["validation"].get("sampling_seed")
        for dataset in datasets
    }
    train_hashes = sorted(
        value for value in train_hashes_by_dataset.values() if isinstance(value, str)
    )
    validation_hashes = sorted(
        value for value in validation_hashes_by_dataset.values() if isinstance(value, str)
    )
    validation_seeds = sorted(
        value for value in validation_seeds_by_dataset.values() if isinstance(value, int)
    )
    distinct_train_hashes = sorted(set(train_hashes))
    distinct_validation_hashes = sorted(set(validation_hashes))
    distinct_validation_seeds = sorted(set(validation_seeds))
    hash_seed_pairs = sorted(
        {
            (validation_hashes_by_dataset[str(dataset["dataset_id"])],
             validation_seeds_by_dataset[str(dataset["dataset_id"])])
            for dataset in datasets
            if isinstance(
                validation_hashes_by_dataset[str(dataset["dataset_id"])], str
            )
            and isinstance(
                validation_seeds_by_dataset[str(dataset["dataset_id"])], int
            )
        }
    )
    errors: list[dict[str, str]] = []
    if len(train_hashes) != expected_dataset_count:
        errors.append(
            _issue(
                "missing_training_dataset_hash",
                f"found {len(train_hashes)} training CSV hashes; expected {expected_dataset_count}",
            )
        )
    if len(distinct_train_hashes) != expected_dataset_count:
        errors.append(
            _issue(
                "training_datasets_not_distinct",
                f"found {len(distinct_train_hashes)} distinct training CSV hashes; "
                f"expected {expected_dataset_count}",
            )
        )
    if len(validation_hashes) != expected_dataset_count:
        errors.append(
            _issue(
                "missing_validation_dataset_hash",
                f"found {len(validation_hashes)} validation CSV hashes; "
                f"expected {expected_dataset_count}",
            )
        )
    if len(distinct_validation_hashes) != 1:
        errors.append(
            _issue(
                "validation_holdout_hash_not_shared",
                f"found {len(distinct_validation_hashes)} distinct validation CSV hashes; expected 1",
            )
        )
    if len(validation_seeds) != expected_dataset_count:
        errors.append(
            _issue(
                "missing_validation_seed",
                f"found {len(validation_seeds)} validation seeds; expected {expected_dataset_count}",
            )
        )
    if len(distinct_validation_seeds) != 1:
        errors.append(
            _issue(
                "validation_holdout_seed_not_shared",
                f"found {len(distinct_validation_seeds)} distinct validation seeds; expected 1",
            )
        )
    if len(hash_seed_pairs) != 1:
        errors.append(
            _issue(
                "validation_holdout_hash_seed_pair_not_shared",
                f"found {len(hash_seed_pairs)} validation (CSV hash, seed) pairs; expected 1",
            )
        )
    return {
        "expected_dataset_count": expected_dataset_count,
        "training": {
            "csv_sha256_by_dataset": train_hashes_by_dataset,
            "distinct_csv_sha256": distinct_train_hashes,
            "n_hashes_present": len(train_hashes),
            "n_distinct_csv_sha256": len(distinct_train_hashes),
            "all_training_csvs_distinct": (
                len(train_hashes) == expected_dataset_count
                and len(distinct_train_hashes) == expected_dataset_count
            ),
        },
        "validation": {
            "csv_sha256_by_dataset": validation_hashes_by_dataset,
            "seed_by_dataset": validation_seeds_by_dataset,
            "distinct_csv_sha256": distinct_validation_hashes,
            "distinct_seeds": distinct_validation_seeds,
            "n_hashes_present": len(validation_hashes),
            "n_distinct_csv_sha256": len(distinct_validation_hashes),
            "n_seeds_present": len(validation_seeds),
            "n_distinct_seeds": len(distinct_validation_seeds),
            "distinct_hash_seed_pairs": [
                {"csv_sha256": sha256, "seed": seed} for sha256, seed in hash_seed_pairs
            ],
            "shared_csv_sha256": (
                distinct_validation_hashes[0]
                if len(validation_hashes) == expected_dataset_count
                and len(distinct_validation_hashes) == 1
                else None
            ),
            "shared_seed": (
                distinct_validation_seeds[0]
                if len(validation_seeds) == expected_dataset_count
                and len(distinct_validation_seeds) == 1
                else None
            ),
            "shared_hash_and_seed": (
                len(validation_hashes) == expected_dataset_count
                and len(validation_seeds) == expected_dataset_count
                and len(hash_seed_pairs) == 1
            ),
        },
        "verification_passed": not errors,
        "errors": errors,
    }


def _adaptive_precompute_lattice(
    *,
    backend_manifest: Any,
    backend_log: Any,
    mg_params: dict[str, Any] | None,
    morse_sets: dict[str, Any] | None,
    max_table_points: Any,
) -> dict[str, Any] | None:
    """Derive the exact alternating-axis lookup lattice used by CMGDB.

    A subdivision depth is global: each refinement splits one coordinate axis
    in round-robin order.  Consequently an odd depth in two dimensions does
    *not* produce a square ``(2^ceil(depth/2) + 1)^2`` table.  For example,
    depth 25 produces ``(8193, 4097)``, matching the runtime precompute table.
    """
    if (
        backend_manifest != "adaptive_precomputed"
        or backend_log != "adaptive_precomputed"
        or mg_params is None
        or morse_sets is None
    ):
        return None
    dimension = _as_int(morse_sets.get("dimension"))
    parameters = mg_params["parameters"]
    subdiv_max = _as_int(parameters.get("subdiv_max"))
    raw_role = parameters.get("adaptive_precompute_subdiv", "max")
    if isinstance(raw_role, str) and raw_role in {"init", "min", "max"}:
        precompute_role = raw_role
        precompute_subdiv = _as_int(parameters.get(f"subdiv_{raw_role}"))
    else:
        precompute_role = "explicit"
        precompute_subdiv = _as_int(raw_role)
    if (
        dimension is None
        or dimension <= 0
        or subdiv_max is None
        or subdiv_max < 0
        or precompute_subdiv is None
        or precompute_subdiv < 0
    ):
        return None
    axis_depths = [
        (precompute_subdiv + dimension - 1 - axis) // dimension
        for axis in range(dimension)
    ]
    cells_per_axis = [2**depth for depth in axis_depths]
    corners_per_axis = [cells + 1 for cells in cells_per_axis]
    table_points = math.prod(corners_per_axis)
    configured_limit = _as_int(max_table_points)
    return {
        "backend": "adaptive_precomputed",
        "dimension": dimension,
        "subdiv_max": subdiv_max,
        "precompute_subdiv_role": precompute_role,
        "precompute_subdiv": precompute_subdiv,
        "axis_depth_M": max(axis_depths),
        "axis_depths": axis_depths,
        "axis_depth_formula": (
            "ceil((precompute_subdiv - axis_index) / dimension)"
        ),
        "cells_per_axis": cells_per_axis,
        "corners_per_axis": corners_per_axis,
        "lattice_shape": corners_per_axis,
        "table_points": table_points,
        "table_points_formula": "product(2^axis_depth + 1)",
        "configured_max_table_points": configured_limit,
        "within_configured_max_table_points": (
            None if configured_limit is None else table_points <= configured_limit
        ),
    }


def _loss_summary(training: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "best_epoch": _as_int(training.get("best_epoch")),
        "n_epochs_run": _as_int(training.get("n_epochs_run")),
        "duration_seconds": _as_float(training.get("train_duration_seconds")),
        "train": {},
        "validation": {},
    }
    for source_key, output_key in (("train", "train"), ("val", "validation")):
        split = training.get(source_key)
        if not isinstance(split, dict):
            continue
        parsed: dict[str, Any] = {}
        for loss_name in (
            "loss_reconstruction",
            "loss_prediction",
            "loss_semiconjugacy",
            "loss_total",
        ):
            stats = split.get(loss_name)
            if isinstance(stats, dict):
                parsed[loss_name] = {
                    key: _as_float(stats.get(key))
                    for key in ("final", "best_epoch_value", "min", "mean", "max")
                }
        result[output_key] = parsed
    return result


def _tolerance_summary(
    metrics: dict[str, Any], sinks: list[int] | None
) -> dict[str, Any]:
    minimal_raw = metrics.get("minimal_morse_labels")
    metric_labels = (
        [_as_int(value) for value in minimal_raw] if isinstance(minimal_raw, list) else None
    )
    if metric_labels is not None and any(value is None for value in metric_labels):
        metric_labels = None
    minimal_sets = metrics.get("minimal_morse_sets")
    details: dict[str, Any] = {}
    if isinstance(minimal_sets, dict):
        labels_for_details = sinks if sinks is not None else []
        for label in labels_for_details:
            raw = minimal_sets.get(str(label))
            if not isinstance(raw, dict):
                details[str(label)] = {"available": False, "tolerance_pass": None}
                continue
            failed = raw.get("is_spurious_attractor")
            failed_bool = failed if isinstance(failed, bool) else None
            tau = _as_float(raw.get("tau_bar"))
            error = _as_float(raw.get("max_semiconjugacy_error"))
            n_samples = _as_int(raw.get("n_semiconjugacy_samples"))
            # A sampled test with no samples contains no evidence either way,
            # even if an older metrics writer happened to serialize ``false``.
            if n_samples == 0:
                failed_bool = None
            ratio = None
            if tau is not None and error is not None and tau > 0:
                ratio = error / tau
            details[str(label)] = {
                "available": True,
                "n_boxes": _as_int(raw.get("n_boxes")),
                "tau_bar": tau,
                "n_semiconjugacy_samples": n_samples,
                "max_semiconjugacy_error": error,
                "max_error_to_tau_ratio": ratio,
                "is_spurious_attractor": failed_bool,
                "tolerance_pass": None if failed_bool is None else not failed_bool,
            }
    passes = [value.get("tolerance_pass") for value in details.values()]
    all_pass = None
    n_failures = None
    if passes and all(isinstance(value, bool) for value in passes):
        all_pass = all(passes)
        n_failures = sum(not value for value in passes)
    if not passes or any(value is None for value in passes):
        status = "inconclusive"
    elif all_pass:
        status = "pass"
    else:
        status = "fail"
    return {
        "metrics_minimal_labels": metric_labels,
        "minimal_sets": details,
        "n_failures": n_failures,
        "all_pass": all_pass,
        "status": status,
    }


def _optional_provenance(cell_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    paths = {
        "model_checkpoint": cell_dir / "models" / "autoencoder.pt",
        "model_architecture": cell_dir / "models" / "autoencoder.json",
    }
    found: dict[str, Any] = {}
    warnings: list[dict[str, str]] = []
    for name, path in paths.items():
        if path.is_file():
            found[name] = _file_reference(path)
        else:
            warnings.append(_issue("missing_optional_provenance", f"missing {name}", path))
    return found, warnings


def _analyze_cell(
    dataset_id: int,
    model_seed: int,
    sweep_root: Path,
    dataset: dict[str, Any],
    *,
    expected_t: int,
    expected_backend: str | None,
    expected_train_initial_conditions: int | None,
    expected_validation_initial_conditions: int | None,
    metrics_root: Path | None,
) -> dict[str, Any]:
    cell_dir = sweep_root / f"dataset_{dataset_id}" / f"seed_{model_seed}"
    paths = {name: cell_dir / relative for name, relative in REQUIRED_CELL_FILES.items()}
    if metrics_root is not None:
        flat_metrics = (
            metrics_root / f"dataset_{dataset_id}" / f"seed_{model_seed}" / "metrics.json"
        )
        replay_metrics = (
            metrics_root
            / f"{sweep_root.name}_dataset_{dataset_id}"
            / f"seed_{model_seed}"
            / "metrics.json"
        )
        paths["metrics"] = replay_metrics if replay_metrics.is_file() else flat_metrics
    missing = [name for name, path in paths.items() if not path.is_file()]
    errors = list(dataset["errors"])
    warnings: list[dict[str, str]] = []
    for name in missing:
        errors.append(_issue("missing_cell_artifact", f"missing {name}", paths[name]))

    payloads: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    for name in ("metrics", "training_summary", "diagnose", "run_manifest"):
        path = paths[name]
        if not path.is_file():
            continue
        try:
            payloads[name], artifacts[name] = _read_json_object(path)
        except ValueError as exc:
            errors.append(_issue("invalid_json_artifact", str(exc), path))

    graph: dict[str, Any] | None = None
    if paths["morse_graph"].is_file():
        try:
            graph = _parse_morse_graph(paths["morse_graph"])
            artifacts["morse_graph"] = graph["file"]
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            errors.append(_issue("invalid_morse_graph", str(exc), paths["morse_graph"]))

    morse_sets: dict[str, Any] | None = None
    if paths["morse_sets"].is_file():
        try:
            morse_sets = _parse_morse_sets(paths["morse_sets"])
            artifacts["morse_sets"] = morse_sets["file"]
        except (OSError, ValueError) as exc:
            errors.append(_issue("invalid_morse_sets", str(exc), paths["morse_sets"]))

    mg_params: dict[str, Any] | None = None
    if paths["mg_params_log"].is_file():
        try:
            mg_params = _parse_mg_params(paths["mg_params_log"])
            artifacts["mg_params_log"] = mg_params["file"]
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            errors.append(_issue("invalid_mg_params_log", str(exc), paths["mg_params_log"]))

    optional_artifacts, optional_warnings = _optional_provenance(cell_dir)
    artifacts.update(optional_artifacts)
    warnings.extend(optional_warnings)

    manifest = payloads.get("run_manifest", {})
    metrics = payloads.get("metrics", {})
    training_raw = payloads.get("training_summary", {})
    diagnose = payloads.get("diagnose", {})
    config_data = _deep_get(manifest, "config", "data")
    config_cmgdb = _deep_get(manifest, "config", "cmgdb")
    manifest_cell = manifest.get("cell") if isinstance(manifest.get("cell"), dict) else {}
    if not isinstance(config_data, dict):
        config_data = {}
    if not isinstance(config_cmgdb, dict):
        config_cmgdb = {}

    data_seed = dataset["train"].get("sampling_seed")
    validation_seed = dataset["validation"].get("sampling_seed")
    train_initial_conditions = (
        expected_train_initial_conditions
        if expected_train_initial_conditions is not None
        else dataset["train"].get("initial_conditions")
    )
    validation_initial_conditions = (
        expected_validation_initial_conditions
        if expected_validation_initial_conditions is not None
        else dataset["validation"].get("initial_conditions")
    )
    manifest_checks = (
        ("model seed", _as_int(manifest_cell.get("seed")), model_seed),
        ("training-data seed", _as_int(config_data.get("train_seed")), data_seed),
        ("validation seed", _as_int(config_data.get("val_seed")), validation_seed),
        ("trajectory length", _as_int(config_data.get("n_iterations")), expected_t),
        (
            "training initial-condition count",
            _as_int(config_data.get("n_samples_train")),
            train_initial_conditions,
        ),
        (
            "validation initial-condition count",
            _as_int(config_data.get("n_samples_val")),
            validation_initial_conditions,
        ),
    )
    if manifest:
        for name, actual, expected in manifest_checks:
            if actual != expected:
                errors.append(
                    _issue(
                        "manifest_mismatch",
                        f"run manifest {name} is {actual!r}; expected {expected!r}",
                        paths["run_manifest"],
                    )
                )
        recorded_train_hash = _deep_get(manifest, "artifacts", "train_csv_sha256")
        actual_train_hash = _deep_get(dataset, "files", "train_csv", "sha256")
        if recorded_train_hash != actual_train_hash:
            errors.append(
                _issue(
                    "train_csv_hash_mismatch",
                    "run manifest train CSV hash does not match the current dataset",
                    paths["run_manifest"],
                )
            )

    backend_manifest = config_cmgdb.get("box_map_backend")
    backend_log = None if mg_params is None else mg_params["parameters"].get("box_map_backend")
    backend_is_explicit = (
        isinstance(backend_manifest, str)
        and backend_manifest != "auto"
        and backend_manifest == backend_log
    )
    if manifest and mg_params is not None and backend_manifest != backend_log:
        errors.append(
            _issue(
                "backend_mismatch",
                f"manifest backend {backend_manifest!r} != mg_params backend {backend_log!r}",
            )
        )
    if expected_backend is not None and (
        backend_manifest != expected_backend or backend_log != expected_backend
    ):
        errors.append(
            _issue(
                "unexpected_box_map_backend",
                f"expected explicit {expected_backend!r}; manifest={backend_manifest!r}, "
                f"mg_params={backend_log!r}",
            )
        )

    precompute_lattice = _adaptive_precompute_lattice(
        backend_manifest=backend_manifest,
        backend_log=backend_log,
        mg_params=mg_params,
        morse_sets=morse_sets,
        max_table_points=config_cmgdb.get("max_table_points"),
    )

    sinks = None if graph is None else graph["sinks"]
    tolerance = _tolerance_summary(metrics, sinks)
    metric_labels = tolerance["metrics_minimal_labels"]
    if sinks is not None and metric_labels is not None and sorted(metric_labels) != sinks:
        errors.append(
            _issue(
                "minimal_label_mismatch",
                f"metrics minimal labels {sorted(metric_labels)} != graph sinks {sinks}",
            )
        )
    consistency = metrics.get("morse_graph_consistency")
    if graph is not None and isinstance(consistency, dict):
        sink_set = set(sinks or [])
        n_attractor_type_sinks = sum(
            1
            for node in graph["nodes"]
            if node["id"] in sink_set
            and isinstance(node.get("conley_index"), list)
            and node["conley_index"]
            and node["conley_index"][0] not in ("0", "")
        )
        expected_pairs = (
            ("n_morse_sets", graph["n_nodes"]),
            ("n_minimal_attractors", n_attractor_type_sinks),
        )
        for key, expected in expected_pairs:
            if _as_int(consistency.get(key)) != expected:
                errors.append(
                    _issue(
                        "metrics_graph_count_mismatch",
                        f"metrics {key}={consistency.get(key)!r}; graph implies {expected}",
                    )
                )

    if graph is not None and morse_sets is not None:
        graph_labels = {node["id"] for node in graph["nodes"]}
        box_labels = {int(label) for label in morse_sets["boxes_by_label"]}
        if not box_labels <= graph_labels:
            errors.append(
                _issue(
                    "morse_set_label_mismatch",
                    f"Morse boxes reference labels {sorted(box_labels - graph_labels)} absent from graph",
                )
            )
        for node in graph["nodes"]:
            node["n_boxes"] = morse_sets["boxes_by_label"].get(str(node["id"]), 0)
        for label, detail in tolerance["minimal_sets"].items():
            reported = detail.get("n_boxes")
            observed = morse_sets["boxes_by_label"].get(label, 0)
            if reported is not None and reported != observed:
                errors.append(
                    _issue(
                        "minimal_box_count_mismatch",
                        f"metrics label {label} n_boxes={reported}; raw morse_sets has {observed}",
                    )
                )

    sink_indices: list[list[str] | None] | None = None
    exact_success: bool | None = None
    marcio_style_success: bool | None = None
    minimal_node_success: bool | None = None
    periodic_bistability_success: bool | None = None
    if graph is not None:
        by_id = {node["id"]: node for node in graph["nodes"]}
        sink_indices = [by_id[label]["conley_index"] for label in graph["sinks"]]
        exact_success = len(graph["sinks"]) == 2 and all(
            index == list(TARGET_CONLEY_INDEX) for index in sink_indices
        )
        marcio_style_success = graph["marcio_style_success"]
        minimal_node_success = len(graph["sinks"]) == 2
        periodic_bistability_success = minimal_node_success and all(
            _is_periodic_bistability_index(index) for index in sink_indices
        )

    training = _loss_summary(training_raw)
    cmgdb_minutes = (
        None
        if mg_params is None
        else _as_float(mg_params["parameters"].get("duration_minutes"))
    )
    cmgdb_seconds = None if cmgdb_minutes is None else 60.0 * cmgdb_minutes
    train_seconds = training.get("duration_seconds")
    combined_seconds = (
        train_seconds + cmgdb_seconds
        if train_seconds is not None and cmgdb_seconds is not None
        else None
    )

    diagnostic = {
        "status": diagnose.get("diagnostic"),
        "hard_flags": diagnose.get("hard_flags"),
        "encoder": diagnose.get("encoder"),
        "latent_map": diagnose.get("latent_map"),
        "bounds": diagnose.get("bounds"),
    }
    complete = dataset["complete"] and not missing and len(artifacts) >= len(
        REQUIRED_CELL_FILES
    )
    verification_passed = complete and not errors
    provenance = {
        "config_hash": manifest.get("config_hash"),
        "created_at_utc": manifest.get("created_at_utc"),
        "latentdynamics_version": manifest.get("latentdynamics_version"),
        "python": manifest.get("python"),
        "platform": manifest.get("platform"),
        "torch": manifest.get("torch"),
        "cmgdb_version": manifest.get("cmgdb_version"),
        "requested_stages": manifest.get("requested_stages"),
        "artifacts": artifacts,
        "dataset_files": dataset["files"],
    }
    return {
        "dataset_id": dataset_id,
        "data_seed": data_seed,
        "validation_seed": validation_seed,
        "model_seed": model_seed,
        "cell_directory": _display_path(cell_dir),
        "dataset": {
            "train": dataset["train"],
            "validation": dataset["validation"],
        },
        "morse_graph": graph,
        "morse_sets": morse_sets,
        "metrics": {
            "minimal_morse_labels": metric_labels,
            "minimal_tolerance": tolerance["minimal_sets"],
            "n_minimal_tolerance_failures": tolerance["n_failures"],
            "all_minimal_tolerance_pass": tolerance["all_pass"],
            "tolerance_status": tolerance["status"],
            "morse_graph_consistency": consistency,
        },
        "sink_conley_indices": sink_indices,
        "marcio_style_success": marcio_style_success,
        "minimal_node_success": minimal_node_success,
        "periodic_bistability_success": periodic_bistability_success,
        "exact_conley_success": exact_success,
        "training": training,
        "cmgdb": {
            "parameters": None if mg_params is None else mg_params["parameters"],
            "duration_seconds": cmgdb_seconds,
            "box_map_backend_manifest": backend_manifest,
            "box_map_backend_log": backend_log,
            "box_map_backend_is_explicit": backend_is_explicit,
            "adaptive_precompute_lattice": precompute_lattice,
        },
        "diagnose": diagnostic,
        "provenance": provenance,
        "complete": complete,
        "verification_passed": verification_passed,
        "errors": errors,
        "warnings": warnings,
        "durations": {
            "training_seconds": train_seconds,
            "cmgdb_seconds": cmgdb_seconds,
            "combined_training_cmgdb_seconds": combined_seconds,
        },
    }


def _loss_value(cell: dict[str, Any], split: str, loss: str, statistic: str) -> Any:
    return _deep_get(cell, "training", split, loss, statistic)


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _cell_csv_record(cell: dict[str, Any]) -> dict[str, Any]:
    graph = cell.get("morse_graph") or {}
    sets = cell.get("morse_sets") or {}
    metrics = cell["metrics"]
    train = cell["dataset"]["train"]
    validation = cell["dataset"]["validation"]
    params = cell["cmgdb"].get("parameters") or {}
    provenance = cell["provenance"]
    artifacts = provenance["artifacts"]
    dataset_files = provenance["dataset_files"]
    lattice = cell["cmgdb"].get("adaptive_precompute_lattice") or {}
    hard_flags = cell["diagnose"].get("hard_flags")
    if not isinstance(hard_flags, dict):
        hard_flags = {}
    node_indices = {
        str(node["id"]): node["conley_index"] for node in graph.get("nodes", [])
    }
    attractor_type_nodes = graph.get("attractor_type_nodes")
    attractor_type_labels = (
        [node["id"] for node in attractor_type_nodes]
        if isinstance(attractor_type_nodes, list)
        else None
    )
    attractor_type_indices = (
        [node["conley_index"] for node in attractor_type_nodes]
        if isinstance(attractor_type_nodes, list)
        else None
    )
    return {
        "dataset_id": cell["dataset_id"],
        "data_seed": cell["data_seed"],
        "validation_seed": cell["validation_seed"],
        "model_seed": cell["model_seed"],
        "trajectory_length_T": train.get("trajectory_length_T"),
        "discarded_steps_T0": train.get("discarded_steps_T0"),
        "train_initial_conditions": train.get("initial_conditions"),
        "validation_initial_conditions": validation.get("initial_conditions"),
        "train_transition_pairs_expected": train.get("transition_pairs_expected"),
        "train_transition_pairs_observed": train.get("transition_pairs_observed"),
        "validation_transition_pairs_expected": validation.get("transition_pairs_expected"),
        "validation_transition_pairs_observed": validation.get("transition_pairs_observed"),
        "n_morse_nodes": graph.get("n_nodes"),
        "n_morse_edges": graph.get("n_edges"),
        "n_graph_sinks": len(graph["sinks"]) if graph.get("sinks") is not None else None,
        "graph_sink_labels": _json_text(graph.get("sinks")),
        "metrics_minimal_labels": _json_text(metrics["minimal_morse_labels"]),
        "sink_conley_indices": _json_text(cell["sink_conley_indices"]),
        "node_conley_indices": _json_text(node_indices),
        "n_attractor_type_nodes": graph.get("n_attractor_type_nodes"),
        "attractor_type_labels": _json_text(attractor_type_labels),
        "attractor_type_conley_indices": _json_text(attractor_type_indices),
        "marcio_style_success": cell["marcio_style_success"],
        "minimal_node_success": cell["minimal_node_success"],
        "periodic_bistability_success": cell["periodic_bistability_success"],
        "morse_boxes_total": sets.get("total_boxes"),
        "morse_boxes_by_label": _json_text(sets.get("boxes_by_label")),
        "minimal_tolerance_details": _json_text(metrics["minimal_tolerance"]),
        "n_minimal_tolerance_failures": metrics["n_minimal_tolerance_failures"],
        "all_minimal_tolerance_pass": metrics["all_minimal_tolerance_pass"],
        "tolerance_status": metrics["tolerance_status"],
        "exact_conley_success": cell["exact_conley_success"],
        "diagnostic": cell["diagnose"].get("status"),
        "encoder_collapsed": hard_flags.get("encoder_collapsed"),
        "latent_map_overcontracted": hard_flags.get("latent_map_overcontracted"),
        "best_epoch": cell["training"].get("best_epoch"),
        "n_epochs_run": cell["training"].get("n_epochs_run"),
        "train_duration_seconds": cell["durations"]["training_seconds"],
        "cmgdb_duration_seconds": cell["durations"]["cmgdb_seconds"],
        "combined_train_cmgdb_duration_seconds": cell["durations"][
            "combined_training_cmgdb_seconds"
        ],
        "train_reconstruction_final": _loss_value(
            cell, "train", "loss_reconstruction", "final"
        ),
        "train_prediction_final": _loss_value(cell, "train", "loss_prediction", "final"),
        "train_semiconjugacy_final": _loss_value(
            cell, "train", "loss_semiconjugacy", "final"
        ),
        "train_total_final": _loss_value(cell, "train", "loss_total", "final"),
        "validation_reconstruction_final": _loss_value(
            cell, "validation", "loss_reconstruction", "final"
        ),
        "validation_prediction_final": _loss_value(
            cell, "validation", "loss_prediction", "final"
        ),
        "validation_semiconjugacy_final": _loss_value(
            cell, "validation", "loss_semiconjugacy", "final"
        ),
        "validation_total_final": _loss_value(cell, "validation", "loss_total", "final"),
        "train_total_best_epoch_value": _loss_value(
            cell, "train", "loss_total", "best_epoch_value"
        ),
        "validation_total_best_epoch_value": _loss_value(
            cell, "validation", "loss_total", "best_epoch_value"
        ),
        "box_map_backend_manifest": cell["cmgdb"]["box_map_backend_manifest"],
        "box_map_backend_log": cell["cmgdb"]["box_map_backend_log"],
        "box_map_backend_is_explicit": cell["cmgdb"]["box_map_backend_is_explicit"],
        "subdiv_init": params.get("subdiv_init"),
        "subdiv_min": params.get("subdiv_min"),
        "subdiv_max": params.get("subdiv_max"),
        "precompute_subdiv_role": lattice.get("precompute_subdiv_role"),
        "precompute_subdiv": lattice.get("precompute_subdiv"),
        "precompute_lattice_dimension": lattice.get("dimension"),
        "precompute_axis_depth_M": lattice.get("axis_depth_M"),
        "precompute_axis_depths": _json_text(lattice.get("axis_depths")),
        "precompute_cells_per_axis": _json_text(lattice.get("cells_per_axis")),
        "precompute_corners_per_axis": _json_text(lattice.get("corners_per_axis")),
        "precompute_lattice_shape": _json_text(lattice.get("lattice_shape")),
        "precompute_table_points": lattice.get("table_points"),
        "config_hash": provenance.get("config_hash"),
        "train_csv_sha256": _deep_get(dataset_files, "train_csv", "sha256"),
        "validation_csv_sha256": _deep_get(dataset_files, "validation_csv", "sha256"),
        "model_sha256": _deep_get(artifacts, "model_checkpoint", "sha256"),
        "morse_graph_sha256": _deep_get(artifacts, "morse_graph", "sha256"),
        "morse_sets_sha256": _deep_get(artifacts, "morse_sets", "sha256"),
        "topology_signature_sha256": graph.get("topology_signature_sha256"),
        "complete": cell["complete"],
        "verification_passed": cell["verification_passed"],
        "error_count": len(cell["errors"]),
        "warning_count": len(cell["warnings"]),
        "errors": _json_text(cell["errors"]),
        "warnings": _json_text(cell["warnings"]),
        "cell_directory": cell["cell_directory"],
    }


def _numeric_summary(values: list[Any]) -> dict[str, Any]:
    clean = [float(value) for value in values if _as_float(value) is not None]
    if not clean:
        return {"count": 0, "mean": None, "population_std": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "population_std": statistics.pstdev(clean),
        "min": min(clean),
        "max": max(clean),
    }


def _success_group(
    cells: list[dict[str, Any]],
    field: str = "exact_conley_success",
) -> dict[str, Any]:
    evaluated = [cell for cell in cells if isinstance(cell.get(field), bool)]
    successes = sum(cell.get(field) is True for cell in evaluated)
    return {
        "n_cells": len(cells),
        "n_evaluated": len(evaluated),
        "n_successes": successes,
        "success_rate_among_evaluated": successes / len(evaluated) if evaluated else None,
    }


def _aggregate(
    cells: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    dataset_design: dict[str, Any],
    *,
    sweep_root: Path,
    data_root: Path,
    dataset_ids: tuple[int, ...],
    model_seeds: tuple[int, ...],
    expected_t: int,
    expected_backend: str | None,
    expected_train_initial_conditions: int | None,
    expected_validation_initial_conditions: int | None,
    metrics_root: Path | None,
    provisional: bool,
) -> dict[str, Any]:
    signature_counts = Counter(
        cell["morse_graph"]["topology_signature_sha256"]
        for cell in cells
        if cell.get("morse_graph") is not None
    )
    sink_count_distribution = Counter(
        len(cell["morse_graph"]["sinks"])
        for cell in cells
        if cell.get("morse_graph") is not None
    )
    sink_index_distribution = Counter(
        _json_text(cell["sink_conley_indices"])
        for cell in cells
        if cell.get("sink_conley_indices") is not None
    )
    errors = Counter(error["code"] for cell in cells for error in cell["errors"])
    warnings = Counter(warning["code"] for cell in cells for warning in cell["warnings"])
    lattice_counts = Counter(
        _json_text(cell["cmgdb"]["adaptive_precompute_lattice"])
        for cell in cells
        if cell["cmgdb"].get("adaptive_precompute_lattice") is not None
    )
    loss_paths = {
        "train_reconstruction_final": ("train", "loss_reconstruction", "final"),
        "train_prediction_final": ("train", "loss_prediction", "final"),
        "train_semiconjugacy_final": ("train", "loss_semiconjugacy", "final"),
        "train_total_final": ("train", "loss_total", "final"),
        "validation_reconstruction_final": (
            "validation",
            "loss_reconstruction",
            "final",
        ),
        "validation_prediction_final": ("validation", "loss_prediction", "final"),
        "validation_semiconjugacy_final": (
            "validation",
            "loss_semiconjugacy",
            "final",
        ),
        "validation_total_final": ("validation", "loss_total", "final"),
    }
    loss_summaries = {
        name: _numeric_summary([_loss_value(cell, *path) for cell in cells])
        for name, path in loss_paths.items()
    }
    by_dataset = {
        str(dataset_id): _success_group(
            [cell for cell in cells if cell["dataset_id"] == dataset_id]
        )
        for dataset_id in dataset_ids
    }
    by_model_seed = {
        str(seed): _success_group([cell for cell in cells if cell["model_seed"] == seed])
        for seed in model_seeds
    }
    marcio_by_dataset = {
        str(dataset_id): _success_group(
            [cell for cell in cells if cell["dataset_id"] == dataset_id],
            "marcio_style_success",
        )
        for dataset_id in dataset_ids
    }
    marcio_by_model_seed = {
        str(seed): _success_group(
            [cell for cell in cells if cell["model_seed"] == seed],
            "marcio_style_success",
        )
        for seed in model_seeds
    }
    minimal_by_dataset = {
        str(dataset_id): _success_group(
            [cell for cell in cells if cell["dataset_id"] == dataset_id],
            "minimal_node_success",
        )
        for dataset_id in dataset_ids
    }
    minimal_by_model_seed = {
        str(seed): _success_group(
            [cell for cell in cells if cell["model_seed"] == seed],
            "minimal_node_success",
        )
        for seed in model_seeds
    }
    periodic_by_dataset = {
        str(dataset_id): _success_group(
            [cell for cell in cells if cell["dataset_id"] == dataset_id],
            "periodic_bistability_success",
        )
        for dataset_id in dataset_ids
    }
    periodic_by_model_seed = {
        str(seed): _success_group(
            [cell for cell in cells if cell["model_seed"] == seed],
            "periodic_bistability_success",
        )
        for seed in model_seeds
    }
    tolerance_status_counts = Counter(
        cell["metrics"].get("tolerance_status") for cell in cells
    )
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "provisional": provisional,
        "source_is_read_only": True,
        "metrics_are_derived_replay": metrics_root is not None,
        "metrics_root": None if metrics_root is None else _display_path(metrics_root),
        "sweep_root": _display_path(sweep_root),
        "data_root": _display_path(data_root),
        "analysis_script": _file_reference(Path(__file__)),
        "expected_design": {
            "dataset_ids": list(dataset_ids),
            "model_seeds": list(model_seeds),
            "n_cells": len(dataset_ids) * len(model_seeds),
            "trajectory_length_T": expected_t,
            "train_initial_conditions": expected_train_initial_conditions,
            "validation_initial_conditions": expected_validation_initial_conditions,
            "box_map_backend": expected_backend,
        },
        "success_criterion": SUCCESS_CRITERION,
        "primary_success_criterion": MARCIO_STYLE_CRITERION,
        "secondary_success_criterion": MINIMAL_NODE_CRITERION,
        "topology_criteria": [MARCIO_STYLE_CRITERION, MINIMAL_NODE_CRITERION],
        "requested_bistability_criterion": PERIODIC_BISTABILITY_CRITERION,
        "inventory": {
            "n_expected_cells": len(dataset_ids) * len(model_seeds),
            "n_records": len(cells),
            "n_complete_cells": sum(cell["complete"] for cell in cells),
            "n_verified_cells": sum(cell["verification_passed"] for cell in cells),
            "n_incomplete_cells": sum(not cell["complete"] for cell in cells),
            "n_cells_with_errors": sum(bool(cell["errors"]) for cell in cells),
            "n_complete_datasets": sum(dataset["complete"] for dataset in datasets),
            "n_verified_datasets": sum(dataset["verification_passed"] for dataset in datasets),
            "n_sweep_design_errors": len(dataset_design["errors"]),
            "error_counts_by_code": dict(sorted(errors.items())),
            "warning_counts_by_code": dict(sorted(warnings.items())),
        },
        "dataset_design": dataset_design,
        "exact_conley_success": {
            **_success_group(cells),
            "rate_over_expected_15_cells": (
                sum(cell["exact_conley_success"] is True for cell in cells)
                / (len(dataset_ids) * len(model_seeds))
            ),
            "by_dataset": by_dataset,
            "by_model_seed": by_model_seed,
        },
        "marcio_style_success": {
            **_success_group(cells, "marcio_style_success"),
            "rate_over_expected_15_cells": (
                sum(cell["marcio_style_success"] is True for cell in cells)
                / (len(dataset_ids) * len(model_seeds))
            ),
            "by_dataset": marcio_by_dataset,
            "by_model_seed": marcio_by_model_seed,
        },
        "minimal_node_success": {
            **_success_group(cells, "minimal_node_success"),
            "rate_over_expected_15_cells": (
                sum(cell["minimal_node_success"] is True for cell in cells)
                / (len(dataset_ids) * len(model_seeds))
            ),
            "by_dataset": minimal_by_dataset,
            "by_model_seed": minimal_by_model_seed,
        },
        "periodic_bistability_success": {
            **_success_group(cells, "periodic_bistability_success"),
            "rate_over_expected_15_cells": (
                sum(cell["periodic_bistability_success"] is True for cell in cells)
                / (len(dataset_ids) * len(model_seeds))
            ),
            "by_dataset": periodic_by_dataset,
            "by_model_seed": periodic_by_model_seed,
        },
        "tolerance": {
            "n_cells_evaluated": sum(
                isinstance(cell["metrics"]["all_minimal_tolerance_pass"], bool)
                for cell in cells
            ),
            "n_cells_all_minimal_pass": sum(
                cell["metrics"]["all_minimal_tolerance_pass"] is True for cell in cells
            ),
            "total_failed_minimal_sets": sum(
                cell["metrics"]["n_minimal_tolerance_failures"] or 0 for cell in cells
            ),
            "status_counts": {
                str(key): value for key, value in sorted(tolerance_status_counts.items())
            },
        },
        "topology": {
            "n_distinct_signatures": len(signature_counts),
            "signature_counts": dict(sorted(signature_counts.items())),
            "sink_count_distribution": {
                str(key): value for key, value in sorted(sink_count_distribution.items())
            },
            "sink_conley_index_multiset_counts": dict(sorted(sink_index_distribution.items())),
        },
        "adaptive_precompute_lattice": {
            "n_cells_recorded": sum(lattice_counts.values()),
            "n_distinct_lattices": len(lattice_counts),
            "all_recorded_cells_same_lattice": len(lattice_counts) == 1,
            "lattices": [
                {**json.loads(serialized), "n_cells": count}
                for serialized, count in sorted(lattice_counts.items())
            ],
        },
        "losses": loss_summaries,
        "durations_seconds": {
            "training": _numeric_summary(
                [cell["durations"]["training_seconds"] for cell in cells]
            ),
            "cmgdb": _numeric_summary([cell["durations"]["cmgdb_seconds"] for cell in cells]),
            "combined_training_cmgdb": _numeric_summary(
                [cell["durations"]["combined_training_cmgdb_seconds"] for cell in cells]
            ),
            "sum_training": sum(
                cell["durations"]["training_seconds"] or 0.0 for cell in cells
            ),
            "sum_cmgdb": sum(cell["durations"]["cmgdb_seconds"] or 0.0 for cell in cells),
        },
        "dataset_provenance": datasets,
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def analyze_sweep(
    *,
    sweep_root: Path = DEFAULT_SWEEP_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    analysis_dir: Path | None = None,
    metrics_root: Path | None = None,
    dataset_ids: tuple[int, ...] = EXPECTED_DATASET_IDS,
    model_seeds: tuple[int, ...] = EXPECTED_MODEL_SEEDS,
    expected_t: int = EXPECTED_T,
    expected_backend: str | None = EXPECTED_BACKEND,
    expected_train_initial_conditions: int | None = None,
    expected_validation_initial_conditions: int | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Path]:
    """Parse, cross-check, aggregate, and atomically write sweep reports."""
    for name, value in (
        ("expected_train_initial_conditions", expected_train_initial_conditions),
        ("expected_validation_initial_conditions", expected_validation_initial_conditions),
    ):
        if value is not None and (isinstance(value, bool) or value <= 0):
            raise ValueError(f"{name} must be a positive integer or None")
    sweep_root = sweep_root.resolve()
    data_root = data_root.resolve()
    analysis_dir = (analysis_dir or sweep_root / "analysis").resolve()
    metrics_root = None if metrics_root is None else metrics_root.resolve()
    datasets = [
        _analyze_dataset(
            dataset_id,
            data_root,
            expected_t,
            expected_train_initial_conditions=expected_train_initial_conditions,
            expected_validation_initial_conditions=expected_validation_initial_conditions,
        )
        for dataset_id in dataset_ids
    ]
    dataset_design = _dataset_design_summary(datasets, len(dataset_ids))
    by_id = {dataset["dataset_id"]: dataset for dataset in datasets}
    cells = [
        _analyze_cell(
            dataset_id,
            model_seed,
            sweep_root,
            by_id[dataset_id],
            expected_t=expected_t,
            expected_backend=expected_backend,
            expected_train_initial_conditions=expected_train_initial_conditions,
            expected_validation_initial_conditions=expected_validation_initial_conditions,
            metrics_root=metrics_root,
        )
        for dataset_id in dataset_ids
        for model_seed in model_seeds
    ]
    invalid = [cell for cell in cells if not cell["verification_passed"]]
    design_errors = dataset_design["errors"]
    if (invalid or design_errors) and not allow_incomplete:
        examples = "; ".join(
            f"dataset_{cell['dataset_id']}/seed_{cell['model_seed']}: "
            + ", ".join(error["code"] for error in cell["errors"][:3])
            for cell in invalid[:5]
        )
        design_example = ", ".join(error["code"] for error in design_errors)
        details = "; ".join(value for value in (examples, design_example) if value)
        raise SweepValidationError(
            f"strict analysis requires {len(dataset_ids) * len(model_seeds)} verified cells; "
            f"{len(invalid)} cells failed verification and {len(design_errors)} sweep-design "
            f"checks failed ({details}). "
            "Use --allow-incomplete only for a provisional inventory."
        )

    provisional = bool(invalid or design_errors)
    aggregate = _aggregate(
        cells,
        datasets,
        dataset_design,
        sweep_root=sweep_root,
        data_root=data_root,
        dataset_ids=dataset_ids,
        model_seeds=model_seeds,
        expected_t=expected_t,
        expected_backend=expected_backend,
        expected_train_initial_conditions=expected_train_initial_conditions,
        expected_validation_initial_conditions=expected_validation_initial_conditions,
        metrics_root=metrics_root,
        provisional=provisional,
    )
    detailed = {
        "schema_version": 1,
        "generated_at_utc": aggregate["generated_at_utc"],
        "provisional": provisional,
        "source_is_read_only": True,
        "metrics_are_derived_replay": metrics_root is not None,
        "metrics_root": None if metrics_root is None else _display_path(metrics_root),
        "sweep_root": _display_path(sweep_root),
        "data_root": _display_path(data_root),
        "expected_design": aggregate["expected_design"],
        "success_criterion": SUCCESS_CRITERION,
        "primary_success_criterion": MARCIO_STYLE_CRITERION,
        "secondary_success_criterion": MINIMAL_NODE_CRITERION,
        "topology_criteria": [MARCIO_STYLE_CRITERION, MINIMAL_NODE_CRITERION],
        "requested_bistability_criterion": PERIODIC_BISTABILITY_CRITERION,
        "dataset_design": dataset_design,
        "datasets": datasets,
        "cells": cells,
    }
    csv_rows = [_cell_csv_record(cell) for cell in cells]
    outputs = {
        "cells_csv": analysis_dir / "cells.csv",
        "cells_json": analysis_dir / "cells.json",
        "aggregate_summary": analysis_dir / "aggregate_summary.json",
    }
    _atomic_write_csv(outputs["cells_csv"], csv_rows)
    _atomic_write_json(outputs["cells_json"], detailed)
    _atomic_write_json(outputs["aggregate_summary"], aggregate)
    return outputs


def _parse_int_tuple(raw: str, *, option: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must be comma-separated integers") from exc
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{option} must contain distinct integers")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="default: SWEEP_ROOT/analysis",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=None,
        help=(
            "optional derived metrics replay root with dataset_N/seed_N/metrics.json; "
            "source run directories remain untouched"
        ),
    )
    parser.add_argument(
        "--dataset-ids",
        default=",".join(map(str, EXPECTED_DATASET_IDS)),
        help="expected dataset ids (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--model-seeds",
        default=",".join(map(str, EXPECTED_MODEL_SEEDS)),
        help="expected model seeds (default: 0,1,2)",
    )
    parser.add_argument("--expected-t", type=int, default=EXPECTED_T)
    parser.add_argument(
        "--expected-train-initial-conditions",
        type=int,
        default=None,
        help="optional strict n_samples requirement for every training dataset/manifest",
    )
    parser.add_argument(
        "--expected-validation-initial-conditions",
        type=int,
        default=None,
        help="optional strict n_samples requirement for every validation dataset/manifest",
    )
    parser.add_argument(
        "--expected-backend",
        default=EXPECTED_BACKEND,
        help="explicit backend required in both run_manifest and mg_params_log",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a provisional report containing missing/invalid cells",
    )
    args = parser.parse_args(argv)
    try:
        dataset_ids = _parse_int_tuple(args.dataset_ids, option="--dataset-ids")
        model_seeds = _parse_int_tuple(args.model_seeds, option="--model-seeds")
        outputs = analyze_sweep(
            sweep_root=args.sweep_root,
            data_root=args.data_root,
            analysis_dir=args.analysis_dir,
            metrics_root=args.metrics_root,
            dataset_ids=dataset_ids,
            model_seeds=model_seeds,
            expected_t=args.expected_t,
            expected_backend=args.expected_backend,
            expected_train_initial_conditions=args.expected_train_initial_conditions,
            expected_validation_initial_conditions=(
                args.expected_validation_initial_conditions
            ),
            allow_incomplete=args.allow_incomplete,
        )
    except (argparse.ArgumentTypeError, SweepValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({name: _display_path(path) for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
