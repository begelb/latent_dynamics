"""Build the strict final report for the Leslie3D Marcio-style 5x3 run.

The source sweep is read-only.  The five derived artifacts are written below
``SWEEP_ROOT/summary``.  Strict mode writes nothing unless all 15 expected
cells exist and their canonical training, CMGDB, topology, metric, checkpoint,
and rendering artifacts can be parsed.  ``--allow-incomplete`` instead emits
an explicitly provisional report which keeps missing and invalid cells visible.
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
from tempfile import TemporaryDirectory
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_ROOT = CODE_ROOT / "output" / "leslie3d_example2_marcio_5x3_v1"
DEFAULT_DATA_ROOT = CODE_ROOT / "data" / "leslie3d_example2_marcio_5x3_v1"
DATASET_IDS = (1, 2, 3, 4, 5)
MODEL_SEEDS = (0, 1, 2)
MARCIO_OBJECTIVE = "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)"
EXPECTED_CMGDB_BOUNDS_SOURCE = "encoded_train_pairs"
SUMMARY_MIN_BOX_SIDE_FRAC = 0.0025
SUMMARY_SCHEMA_VERSION = 3
EXPECTED_CONLEY_COMPONENTS = 3

REQUIRED_FILES = {
    "checkpoint": Path("models/autoencoder.pt"),
    "checkpoint_metadata": Path("models/autoencoder.json"),
    "history": Path("logs/history.json"),
    "training_summary": Path("training_summary.json"),
    "final_losses": Path("final_losses.txt"),
    "diagnose": Path("diagnose.json"),
    "morse_graph": Path("MG/morse_graph"),
    "morse_sets": Path("MG/morse_sets"),
    "morse_graph_png": Path("MG/morse_graph.png"),
    "morse_graph_pdf": Path("MG/morse_graph.pdf"),
    "morse_sets_png": Path("MG/morse_sets.png"),
    "morse_sets_pdf": Path("MG/morse_sets.pdf"),
    "mg_params_log": Path("mg_params_log.txt"),
    "metrics": Path("metrics.json"),
    "run_manifest": Path("run_manifest.json"),
}

SUCCESS_CRITERION = {
    "name": "bistability_exactly_two_stable_conley_index_nodes",
    "definition": (
        "A cell passes the bistability criterion when exactly two Morse nodes have "
        "a stable Conley-index signature, identified by a nonzero degree-0 "
        "component. Higher-degree components may be nonzero; for example, both "
        "(x^4-1, 0, 0) and (x-1, x-1, 0) qualify. Graph edges/minimality and "
        "sampled tolerance metrics are diagnostics and do not affect this classification."
    ),
    "required_stable_conley_index_node_count": 2,
    "stable_conley_index_pattern": ["nonzero", "any", "any"],
    "expected_conley_index_components": EXPECTED_CONLEY_COMPONENTS,
    "conley_index_affects_classification": True,
    "graph_edges_affect_classification": False,
    "graph_minimality_affects_classification": False,
    "tolerance_affects_classification": False,
}

CSV_FIELDS = (
    "dataset_id", "model_seed", "cell_status", "complete", "verification_passed",
    "bistability_pass", "n_stable_conley_index_nodes", "stable_index_labels",
    "stable_conley_indices", "training_method", "training_seed",
    "data_train_seed", "data_validation_seed", "train_csv_sha256", "validation_csv_sha256",
    "epochs_completed", "n_training_pairs", "loss_reconstruction_final",
    "loss_prediction_final", "loss_total_final", "final_losses",
    "diagnosis", "encoder_collapsed", "latent_map_overcontracted",
    "cmgdb_duration_seconds", "cmgdb_precompute_seconds", "cmgdb_total_seconds",
    "subdiv_init", "subdiv_min", "subdiv_max", "box_map_backend",
    "lower_bounds", "upper_bounds", "bounds_source", "n_morse_nodes",
    "n_morse_edges", "n_graph_sinks", "sink_labels", "sink_conley_indices",
    "sink_degree0_normalized", "node_conley_indices", "morse_boxes_total",
    "morse_boxes_by_label", "minimal_tolerance", "all_minimal_tolerance_pass",
    "n_minimal_tolerance_failures", "checkpoint_sha256", "morse_graph_sha256",
    "morse_sets_sha256", "config_hash", "error_count", "warning_count",
    "errors", "warnings", "cell_directory",
)

NODE_RE = re.compile(r'^\s*"?(\d+)"?\s*\[(?P<attrs>[^\]]*)\]\s*;?\s*$')
EDGE_RE = re.compile(r'^\s*"?(\d+)"?\s*->\s*"?(\d+)"?(?:\s*\[[^\]]*\])?\s*;?\s*$')
LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')
CONLEY_RE = re.compile(r"\(([^()]*)\)")


class SweepValidationError(RuntimeError):
    """Strict reporting encountered missing or invalid source artifacts."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _display(path: Path) -> str:
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


def _file_ref(path: Path, *, sha256: str | None = None) -> dict[str, Any]:
    return {"path": _display(path), "size_bytes": path.stat().st_size,
            "sha256": sha256 or _sha256(path)}


def _issue(code: str, message: str, path: Path | None = None) -> dict[str, str]:
    result = {"code": code, "message": message}
    if path is not None:
        result["path"] = _display(path)
    return result


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _deep_get(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload, _file_ref(path)


def _resolve_directory(parent: Path, names: tuple[str, ...]) -> tuple[Path, list[dict[str, str]]]:
    existing = [parent / name for name in names if (parent / name).is_dir()]
    if len(existing) > 1:
        return existing[0], [_issue(
            "ambiguous_directory_aliases",
            "multiple aliases exist: " + ", ".join(_display(path) for path in existing),
        )]
    return (existing[0] if existing else parent / names[0]), []


def _dataset_dir(root: Path, dataset_id: int) -> tuple[Path, list[dict[str, str]]]:
    return _resolve_directory(root, (f"dataset_{dataset_id:02d}", f"dataset_{dataset_id}"))


def _cell_dir(dataset_dir: Path, seed: int) -> tuple[Path, list[dict[str, str]]]:
    # The standard pipeline uses seed_0; older sweep material sometimes uses seed_00.
    return _resolve_directory(dataset_dir, (f"seed_{seed}", f"seed_{seed:02d}"))


def _normalize_index(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("**", "^").replace("\u2074", "^4")
    text = re.sub(r"x\^\{(\d+)\}", r"x^\1", text)
    text = re.sub(r"(?<!\^)x4(?=-)", "x^4", text)
    text = re.sub(r"\s+", "", text).strip("$\"")
    return text


def _is_stable_conley_index(index: Any) -> bool:
    """Return whether a 2-D Conley tuple has nontrivial H0.

    The polynomial in degree zero may encode any stable recurrent behavior;
    its period is deliberately not prescribed, and higher homology may also be
    nontrivial. Requiring all three expected components prevents a truncated or
    otherwise incomplete tuple from being classified as stable.
    """
    return (
        isinstance(index, list)
        and len(index) == EXPECTED_CONLEY_COMPONENTS
        and isinstance(index[0], str)
        and index[0] not in ("", "0")
    )


def _parse_dot(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    nodes: dict[int, str] = {}
    edges: set[tuple[int, int]] = set()
    for line in text.splitlines():
        node_match = NODE_RE.match(line)
        if node_match:
            label_match = LABEL_RE.search(node_match.group("attrs"))
            nodes[int(node_match.group(1))] = label_match.group(1) if label_match else ""
            continue
        edge_match = EDGE_RE.match(line)
        if edge_match:
            edges.add((int(edge_match.group(1)), int(edge_match.group(2))))
    if not nodes:
        raise ValueError("DOT contains no integer Morse nodes")
    unknown = sorted({n for edge in edges for n in edge if n not in nodes})
    if unknown:
        raise ValueError(f"DOT edges reference unknown nodes {unknown}")
    sinks = sorted(set(nodes) - {source for source, _ in edges})
    parsed_nodes = []
    for node_id in sorted(nodes):
        match = CONLEY_RE.search(nodes[node_id])
        conley = [part.strip() for part in match.group(1).split(",")] if match else None
        parsed_nodes.append({
            "id": node_id,
            "dot_label": nodes[node_id],
            "conley_index": conley,
            "conley_index_normalized": (
                [_normalize_index(part) for part in conley] if conley is not None else None
            ),
            "is_sink": node_id in sinks,
        })
    canonical = {
        "nodes": [{"id": n["id"], "conley": n["conley_index_normalized"]}
                  for n in parsed_nodes],
        "edges": [list(edge) for edge in sorted(edges)], "sinks": sinks,
    }
    signature = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "nodes": parsed_nodes, "edges": canonical["edges"], "sinks": sinks,
        "n_nodes": len(nodes), "n_edges": len(edges),
        "topology_signature_sha256": signature,
        "file": _file_ref(path, sha256=hashlib.sha256(raw).hexdigest()),
    }


def _parse_morse_sets(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    counts: Counter[int] = Counter()
    n_columns: int | None = None
    size = 0
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, start=1):
            digest.update(raw)
            size += len(raw)
            if not raw.strip():
                continue
            try:
                parts = raw.decode("utf-8").strip().split(",")
            except UnicodeDecodeError as exc:
                raise ValueError(f"line {line_no} is not UTF-8") from exc
            if n_columns is None:
                n_columns = len(parts)
            if len(parts) != n_columns or len(parts) < 3 or len(parts) % 2 == 0:
                raise ValueError(f"line {line_no} has invalid Morse-box shape")
            try:
                coords = [float(value) for value in parts[:-1]]
                label = float(parts[-1])
            except ValueError as exc:
                raise ValueError(f"line {line_no} contains non-numeric data") from exc
            if not all(math.isfinite(value) for value in coords):
                raise ValueError(f"line {line_no} contains non-finite coordinates")
            if not math.isfinite(label) or not label.is_integer():
                raise ValueError(f"line {line_no} has a non-integer label")
            counts[int(label)] += 1
    if n_columns is None:
        raise ValueError("Morse-set file contains no boxes")
    return {
        "dimension": (n_columns - 1) // 2,
        "total_boxes": sum(counts.values()),
        "boxes_by_label": {str(key): counts[key] for key in sorted(counts)},
        "file": {"path": _display(path), "size_bytes": size, "sha256": digest.hexdigest()},
    }


def _parse_key_value(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    values: dict[str, Any] = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        if ":" not in raw:
            raise ValueError(f"line {line_no} lacks ':'")
        key, value_text = raw.split(":", 1)
        value_text = value_text.strip()
        try:
            value = ast.literal_eval(value_text)
        except (ValueError, SyntaxError):
            try:
                value = float(value_text)
            except ValueError:
                value = value_text
        values[key.strip()] = value
    if not values:
        raise ValueError("file has no key/value records")
    return values, _file_ref(path), text


def _mg_summary(parameters: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    lower = parameters.get("Lower bounds", parameters.get("lower_bounds"))
    upper = parameters.get("Upper bounds", parameters.get("upper_bounds"))
    if not (isinstance(lower, (list, tuple)) and isinstance(upper, (list, tuple))
            and len(lower) == len(upper) and len(lower) > 0
            and all(_as_float(v) is not None for v in (*lower, *upper))):
        errors.append(_issue("invalid_cmgdb_bounds", "lower and upper CMGDB bounds are required"))
        lower = upper = None
    elif any(float(lo) >= float(hi) for lo, hi in zip(lower, upper, strict=True)):
        errors.append(_issue("invalid_cmgdb_bounds", "each lower bound must be below its upper bound"))
    settings = {
        key: parameters.get(key) for key in (
            "subdiv_init", "subdiv_min", "subdiv_max", "subdiv_limit",
            "padding", "box_map_backend", "max_table_points", "bounds_epsilon_frac",
        ) if key in parameters
    }
    for key in ("subdiv_init", "subdiv_min", "subdiv_max"):
        if _as_int(parameters.get(key)) is None:
            errors.append(_issue("missing_cmgdb_setting", f"{key} is required"))
    if not isinstance(parameters.get("box_map_backend"), str):
        errors.append(_issue("missing_cmgdb_setting", "box_map_backend is required"))
    bounds_source = parameters.get("bounds_source")
    if bounds_source != EXPECTED_CMGDB_BOUNDS_SOURCE:
        errors.append(
            _issue(
                "invalid_cmgdb_bounds_source",
                f"bounds_source must be {EXPECTED_CMGDB_BOUNDS_SOURCE!r}",
            )
        )
    duration = _as_float(parameters.get("cmgdb_seconds"))
    if duration is None:
        duration = _as_float(parameters.get("duration_seconds"))
    if duration is None:
        minutes = _as_float(parameters.get("duration_minutes"))
        duration = None if minutes is None else 60.0 * minutes
    if duration is None:
        errors.append(_issue("missing_cmgdb_timing", "no CMGDB duration was recorded"))
    timing = {
        "duration_seconds": duration,
        "precompute_seconds": _as_float(parameters.get("precompute_seconds")),
        "cmgdb_seconds": _as_float(parameters.get("cmgdb_seconds")),
        "total_seconds": _as_float(parameters.get("total_seconds")),
    }
    return {
        "parameters": parameters, "settings": settings,
        "bounds": {"lower": list(lower) if lower is not None else None,
                   "upper": list(upper) if upper is not None else None,
                   "source": bounds_source},
        "timing": timing,
    }, errors


def _scan_csv(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    nonempty = 0
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            size += len(raw)
            nonempty += bool(raw.strip())
    return {"path": _display(path), "size_bytes": size, "sha256": digest.hexdigest(),
            "transition_pairs": max(0, nonempty - 1)}


def _analyze_dataset(dataset_id: int, sweep_root: Path, data_root: Path) -> dict[str, Any]:
    output_dir, output_errors = _dataset_dir(sweep_root, dataset_id)
    data_dir, data_errors = _dataset_dir(data_root, dataset_id)
    errors = [*output_errors, *data_errors]
    warnings: list[dict[str, str]] = []
    metadata: dict[str, Any] = {}
    files: dict[str, Any] = {}
    for role, filename in (("train", "train_metadata.json"), ("validation", "val_metadata.json")):
        path = data_dir / filename
        if not path.is_file():
            warnings.append(_issue("missing_optional_data_provenance", f"missing {filename}", path))
            continue
        try:
            payload, reference = _read_json(path)
            metadata[role] = payload
            files[f"{role}_metadata"] = reference
        except ValueError as exc:
            errors.append(_issue("invalid_data_metadata", str(exc), path))
    for role, filename in (("train", "train.csv"), ("validation", "val.csv")):
        path = data_dir / filename
        if not path.is_file():
            warnings.append(_issue("missing_optional_data_provenance", f"missing {filename}", path))
            continue
        try:
            files[f"{role}_csv"] = _scan_csv(path)
        except OSError as exc:
            errors.append(_issue("invalid_data_csv", str(exc), path))
    scaler_dir = output_dir / "scalers" / "train"
    for key, filename in (("scaler", "scaler.gz"), ("scaler_metadata", "scaler_metadata.json")):
        path = scaler_dir / filename
        if path.is_file():
            files[key] = _file_ref(path)
        else:
            warnings.append(_issue("missing_optional_scaler_provenance", f"missing {filename}", path))
    return {
        "dataset_id": dataset_id, "output_directory": _display(output_dir),
        "data_directory": _display(data_dir), "resolved_output_path": output_dir,
        "metadata": metadata,
        "seeds": {
            "train": _as_int(_deep_get(metadata, "train", "sampling_seed")),
            "validation": _as_int(_deep_get(metadata, "validation", "sampling_seed")),
        },
        "files": files, "errors": errors, "warnings": warnings,
    }


def _training_validation(
    summary: dict[str, Any], history: dict[str, Any], checkpoint_metadata: dict[str, Any],
    model_seed: int, summary_path: Path,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    required_equal = {
        "training_method": "marcio_full_batch", "objective": MARCIO_OBJECTIVE,
        "checkpoint_selection": "final_epoch", "validation_used": False,
        "early_stopping_used": False, "best_weight_restoration_used": False,
    }
    for key, expected in required_equal.items():
        if summary.get(key) != expected:
            errors.append(_issue("not_exact_marcio_training", f"{key}={summary.get(key)!r}; expected {expected!r}", summary_path))
    if _deep_get(summary, "data", "full_batch") is not True:
        errors.append(_issue("not_exact_marcio_training", "data.full_batch must be true", summary_path))
    if _as_int(summary.get("seed")) != model_seed:
        errors.append(_issue("training_seed_mismatch", f"summary seed {summary.get('seed')!r} != model seed {model_seed}", summary_path))
    epochs = _as_int(summary.get("epochs_completed"))
    if epochs is None or epochs < 1 or any(_as_int(summary.get(key)) != epochs for key in ("epochs_requested", "checkpoint_epoch")):
        errors.append(_issue("invalid_marcio_epoch_endpoint", "epochs_requested/completed/checkpoint_epoch must be one equal positive integer", summary_path))
    losses = summary.get("final_epoch_train")
    if not isinstance(losses, dict) or any(_as_float(losses.get(key)) is None for key in ("loss_reconstruction", "loss_prediction", "loss_total")):
        errors.append(_issue("invalid_marcio_final_losses", "final_epoch_train requires finite reconstruction, prediction, and total losses", summary_path))
    elif not math.isclose(float(losses["loss_reconstruction"]) + float(losses["loss_prediction"]), float(losses["loss_total"]), rel_tol=2e-5, abs_tol=1e-8):
        errors.append(_issue("invalid_marcio_final_losses", "loss_total is inconsistent with the two-term objective", summary_path))
    recorded_arch = checkpoint_metadata.get("arch", checkpoint_metadata)
    if summary.get("arch") != recorded_arch:
        errors.append(_issue("checkpoint_architecture_mismatch", "training summary arch differs from autoencoder.json", summary_path))
    if history.get("training_method") != "marcio_full_batch":
        errors.append(_issue("invalid_marcio_history", "history training_method is not marcio_full_batch"))
    train_history = history.get("train")
    if not isinstance(train_history, dict):
        errors.append(_issue("invalid_marcio_history", "history.train is missing"))
    elif epochs is not None:
        for key in ("loss_reconstruction", "loss_prediction", "loss_total", "learning_rate"):
            values = train_history.get(key)
            if not isinstance(values, list) or len(values) != epochs:
                errors.append(_issue("invalid_marcio_history", f"history {key} length does not equal epochs_completed"))
        if isinstance(losses, dict):
            for key in ("loss_reconstruction", "loss_prediction", "loss_total"):
                values = train_history.get(key)
                if isinstance(values, list) and values and _as_float(values[-1]) is not None and _as_float(losses.get(key)) is not None and not math.isclose(float(values[-1]), float(losses[key]), rel_tol=1e-7, abs_tol=1e-10):
                    errors.append(_issue("training_endpoint_mismatch", f"history and summary differ for {key}"))
    return errors


def _metrics_summary(raw: dict[str, Any], sinks: list[int] | None, boxes: dict[str, int] | None) -> tuple[dict[str, Any], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    labels_raw = raw.get("minimal_morse_labels")
    labels = [_as_int(value) for value in labels_raw] if isinstance(labels_raw, list) else None
    if labels is None or any(value is None for value in labels):
        errors.append(_issue("invalid_metrics", "minimal_morse_labels must be an integer list"))
        labels = []
    labels_int = [int(value) for value in labels]
    if sinks is not None and sorted(labels_int) != sorted(sinks):
        errors.append(_issue("metrics_graph_sink_mismatch", f"metrics labels {labels_int} != graph sinks {sinks}"))
    raw_sets = raw.get("minimal_morse_sets")
    if not isinstance(raw_sets, dict):
        errors.append(_issue("invalid_metrics", "minimal_morse_sets must be an object"))
        raw_sets = {}
    details: dict[str, Any] = {}
    for label in (sinks if sinks is not None else labels_int):
        item = raw_sets.get(str(label))
        if not isinstance(item, dict):
            errors.append(_issue("missing_minimal_set_metric", f"missing metrics for sink {label}"))
            details[str(label)] = {"available": False, "tolerance_pass": None}
            continue
        tau = _as_float(item.get("tau_bar"))
        residual = _as_float(item.get("max_semiconjugacy_error"))
        passed = item.get("tolerance_pass") if isinstance(item.get("tolerance_pass"), bool) else None
        if passed is None and isinstance(item.get("is_spurious_attractor"), bool):
            passed = not item["is_spurious_attractor"]
        if passed is None and tau is not None and residual is not None:
            passed = residual <= tau
        reported_boxes = _as_int(item.get("n_boxes"))
        observed_boxes = None if boxes is None else boxes.get(str(label), 0)
        if reported_boxes is not None and observed_boxes is not None and reported_boxes != observed_boxes:
            errors.append(_issue("minimal_box_count_mismatch", f"metrics sink {label} has {reported_boxes} boxes; file has {observed_boxes}"))
        details[str(label)] = {
            "available": True, "n_boxes": reported_boxes, "tau_bar": tau,
            "n_semiconjugacy_samples": _as_int(item.get("n_semiconjugacy_samples")),
            "max_semiconjugacy_error": residual,
            "max_error_to_tau_ratio": (residual / tau if tau and residual is not None else None),
            "is_spurious_attractor": item.get("is_spurious_attractor") if isinstance(item.get("is_spurious_attractor"), bool) else None,
            "tolerance_pass": passed, "raw": item,
        }
    passes = [item["tolerance_pass"] for item in details.values()]
    # A known failure dominates unknown/unsampled sets.  Report ``None`` only
    # when no set failed and at least one set still lacks a boolean result.
    if any(value is False for value in passes):
        all_pass = False
    elif passes and all(value is True for value in passes):
        all_pass = True
    else:
        all_pass = None
    return {
        "minimal_morse_labels": labels_int, "minimal_sets": details,
        "all_minimal_tolerance_pass": all_pass,
        "n_minimal_tolerance_failures": sum(value is False for value in passes),
        "raw": raw,
    }, errors


def _validate_image(path: Path) -> None:
    from PIL import Image
    with Image.open(path) as image:
        image.verify()


def _failure_evidence(cell_dir: Path) -> dict[str, Any] | None:
    for filename in ("failure.json", "failed.json", "error.json", "status.json"):
        path = cell_dir / filename
        if path.is_file():
            try:
                payload, reference = _read_json(path)
                return {"file": reference, "payload": payload}
            except ValueError as exc:
                return {"file": _display(path), "parse_error": str(exc)}
    return None


def _analyze_cell(dataset: dict[str, Any], model_seed: int) -> dict[str, Any]:
    dataset_id = dataset["dataset_id"]
    dataset_dir = dataset["resolved_output_path"]
    cell_dir, alias_errors = _cell_dir(dataset_dir, model_seed)
    errors = [*dataset["errors"], *alias_errors]
    warnings = list(dataset["warnings"])
    paths = {name: cell_dir / relative for name, relative in REQUIRED_FILES.items()}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if not cell_dir.is_dir():
        errors.append(_issue("missing_cell_directory", "expected cell directory is absent", cell_dir))
    for name in missing:
        errors.append(_issue("missing_cell_artifact", f"missing {name}", paths[name]))
    artifacts: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name in ("checkpoint_metadata", "history", "training_summary", "diagnose", "metrics", "run_manifest"):
        if not paths[name].is_file():
            continue
        try:
            payloads[name], artifacts[name] = _read_json(paths[name])
        except ValueError as exc:
            errors.append(_issue("invalid_json_artifact", f"{name}: {exc}", paths[name]))
    for name in ("checkpoint", "morse_graph_png", "morse_graph_pdf", "morse_sets_png", "morse_sets_pdf"):
        path = paths[name]
        if not path.is_file():
            continue
        artifacts[name] = _file_ref(path)
        if path.stat().st_size == 0:
            errors.append(_issue("empty_artifact", f"{name} is empty", path))
        elif name.endswith("_png"):
            try:
                _validate_image(path)
            except Exception as exc:
                errors.append(_issue("invalid_rendered_image", f"{name}: {exc}", path))
        elif name.endswith("_pdf") and not path.read_bytes()[:5] == b"%PDF-":
            errors.append(_issue("invalid_rendered_pdf", f"{name} lacks a PDF header", path))
    graph = None
    if paths["morse_graph"].is_file():
        try:
            graph = _parse_dot(paths["morse_graph"])
            artifacts["morse_graph"] = graph["file"]
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            errors.append(_issue("invalid_morse_graph", str(exc), paths["morse_graph"]))
    sets = None
    if paths["morse_sets"].is_file():
        try:
            sets = _parse_morse_sets(paths["morse_sets"])
            artifacts["morse_sets"] = sets["file"]
        except (OSError, ValueError) as exc:
            errors.append(_issue("invalid_morse_sets", str(exc), paths["morse_sets"]))
    mg = None
    if paths["mg_params_log"].is_file():
        try:
            params, reference, _ = _parse_key_value(paths["mg_params_log"])
            artifacts["mg_params_log"] = reference
            mg, mg_errors = _mg_summary(params)
            errors.extend(mg_errors)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            errors.append(_issue("invalid_mg_params_log", str(exc), paths["mg_params_log"]))
    final_losses = None
    if paths["final_losses"].is_file():
        try:
            values, reference, raw_text = _parse_key_value(paths["final_losses"])
            artifacts["final_losses"] = reference
            final_losses = {"values": values, "raw_text": raw_text, "file": reference}
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            errors.append(_issue("invalid_final_losses", str(exc), paths["final_losses"]))
    training = payloads.get("training_summary")
    history = payloads.get("history")
    checkpoint_metadata = payloads.get("checkpoint_metadata")
    if training is not None and history is not None and checkpoint_metadata is not None:
        errors.extend(_training_validation(training, history, checkpoint_metadata, model_seed, paths["training_summary"]))
    manifest = payloads.get("run_manifest", {})
    manifest_seed = _deep_get(manifest, "cell", "seed")
    if manifest and _as_int(manifest_seed) != model_seed:
        errors.append(_issue("manifest_seed_mismatch", f"run manifest seed {manifest_seed!r} != {model_seed}", paths["run_manifest"]))
    manifest_data = _deep_get(manifest, "config", "data")
    if not isinstance(manifest_data, dict):
        manifest_data = {}
    train_seed = dataset["seeds"]["train"]
    validation_seed = dataset["seeds"]["validation"]
    if train_seed is None:
        train_seed = _as_int(manifest_data.get("train_seed"))
    elif manifest_data and _as_int(manifest_data.get("train_seed")) != train_seed:
        errors.append(_issue("manifest_data_seed_mismatch", "manifest train_seed differs from train metadata"))
    if validation_seed is None:
        validation_seed = _as_int(manifest_data.get("val_seed"))
    elif manifest_data and _as_int(manifest_data.get("val_seed")) != validation_seed:
        errors.append(_issue("manifest_data_seed_mismatch", "manifest val_seed differs from validation metadata"))
    actual_train_hash = _deep_get(dataset, "files", "train_csv", "sha256")
    recorded_train_hash = _deep_get(manifest, "artifacts", "train_csv_sha256")
    if actual_train_hash and recorded_train_hash and actual_train_hash != recorded_train_hash:
        errors.append(_issue("train_csv_hash_mismatch", "run manifest train hash differs from current data"))
    if graph is not None and sets is not None:
        graph_labels = {node["id"] for node in graph["nodes"]}
        box_labels = {int(label) for label in sets["boxes_by_label"]}
        if not box_labels <= graph_labels:
            errors.append(_issue("morse_set_label_mismatch", f"box labels absent from graph: {sorted(box_labels - graph_labels)}"))
        for node in graph["nodes"]:
            node["n_boxes"] = sets["boxes_by_label"].get(str(node["id"]), 0)
    metrics = None
    if "metrics" in payloads:
        metrics, metric_errors = _metrics_summary(
            payloads["metrics"], None if graph is None else graph["sinks"],
            None if sets is None else sets["boxes_by_label"],
        )
        errors.extend(metric_errors)
    sink_indices = None
    sink_h0 = None
    stable_index_nodes = None
    bistability_pass = None
    if graph is not None:
        by_id = {node["id"]: node for node in graph["nodes"]}
        sink_indices = [by_id[label]["conley_index"] for label in graph["sinks"]]
        sink_h0 = [
            (normalized[0] if normalized else None)
            for normalized in (by_id[label]["conley_index_normalized"] for label in graph["sinks"])
        ]
        stable_index_nodes = [
            {
                "id": node["id"],
                "conley_index": node["conley_index"],
                "conley_index_normalized": node["conley_index_normalized"],
                "is_graph_sink": node["is_sink"],
            }
            for node in graph["nodes"]
            if _is_stable_conley_index(node["conley_index_normalized"])
        ]
        bistability_pass = (
            len(stable_index_nodes)
            == SUCCESS_CRITERION["required_stable_conley_index_node_count"]
        )
    complete = cell_dir.is_dir() and not missing
    verified = complete and not errors
    status = "incomplete"
    if complete and errors:
        status = "invalid"
    elif verified:
        status = "verified_success" if bistability_pass else "verified_criterion_failure"
    train_seconds = _as_float(training.get("train_duration_seconds")) if training else None
    cmgdb_seconds = _deep_get(mg, "timing", "duration_seconds")
    return {
        "dataset_id": dataset_id, "model_seed": model_seed,
        "cell_directory": _display(cell_dir), "resolved_cell_path": cell_dir,
        "cell_status": status, "complete": complete, "verification_passed": verified,
        "bistability_pass": bistability_pass, "success_criterion": SUCCESS_CRITERION,
        "seeds": {"training": _as_int(training.get("seed")) if training else None,
                  "data_train": train_seed, "data_validation": validation_seed},
        "dataset_provenance": {"files": dataset["files"], "metadata": dataset["metadata"]},
        "training_summary": training, "training_history_endpoint": (
            {key: values[-1] for key, values in history.get("train", {}).items()
             if isinstance(values, list) and values} if history else None
        ),
        "final_losses": final_losses, "diagnose": payloads.get("diagnose"),
        "cmgdb": mg, "morse_graph": graph, "morse_sets": sets, "metrics": metrics,
        "sink_conley_indices": sink_indices, "sink_degree0_normalized": sink_h0,
        "stable_conley_index_nodes": stable_index_nodes,
        "stable_index_labels": (
            [node["id"] for node in stable_index_nodes]
            if stable_index_nodes is not None else None
        ),
        "n_stable_conley_index_nodes": (
            len(stable_index_nodes) if stable_index_nodes is not None else None
        ),
        "run_manifest": manifest or None, "checkpoint_metadata": checkpoint_metadata,
        "failure_evidence": _failure_evidence(cell_dir), "artifacts": artifacts,
        "durations": {"training_seconds": train_seconds, "cmgdb_seconds": cmgdb_seconds,
                      "combined_seconds": (train_seconds + cmgdb_seconds if train_seconds is not None and cmgdb_seconds is not None else None)},
        "errors": errors, "warnings": warnings,
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _csv_row(cell: dict[str, Any]) -> dict[str, Any]:
    graph = cell["morse_graph"] or {}
    sets = cell["morse_sets"] or {}
    metrics = cell["metrics"] or {}
    training = cell["training_summary"] or {}
    losses = training.get("final_epoch_train") if isinstance(training.get("final_epoch_train"), dict) else {}
    diagnosis = cell["diagnose"] or {}
    flags = diagnosis.get("hard_flags") if isinstance(diagnosis.get("hard_flags"), dict) else {}
    mg = cell["cmgdb"] or {}
    settings, bounds, timing = mg.get("settings", {}), mg.get("bounds", {}), mg.get("timing", {})
    refs = cell["artifacts"]
    node_indices = {str(node["id"]): node["conley_index"] for node in graph.get("nodes", [])}
    return {
        "dataset_id": cell["dataset_id"], "model_seed": cell["model_seed"],
        "cell_status": cell["cell_status"], "complete": cell["complete"],
        "verification_passed": cell["verification_passed"],
        "bistability_pass": cell["bistability_pass"],
        "n_stable_conley_index_nodes": cell["n_stable_conley_index_nodes"],
        "stable_index_labels": _json_text(cell["stable_index_labels"]),
        "stable_conley_indices": _json_text([
            node["conley_index"]
            for node in (cell["stable_conley_index_nodes"] or [])
        ]),
        "training_method": training.get("training_method"), "training_seed": cell["seeds"]["training"],
        "data_train_seed": cell["seeds"]["data_train"], "data_validation_seed": cell["seeds"]["data_validation"],
        "train_csv_sha256": _deep_get(cell, "dataset_provenance", "files", "train_csv", "sha256"),
        "validation_csv_sha256": _deep_get(cell, "dataset_provenance", "files", "validation_csv", "sha256"),
        "epochs_completed": training.get("epochs_completed"), "n_training_pairs": _deep_get(training, "data", "n_pairs"),
        "loss_reconstruction_final": losses.get("loss_reconstruction"), "loss_prediction_final": losses.get("loss_prediction"),
        "loss_total_final": losses.get("loss_total"), "final_losses": _json_text(_deep_get(cell, "final_losses", "values")),
        "diagnosis": diagnosis.get("diagnostic"), "encoder_collapsed": flags.get("encoder_collapsed"),
        "latent_map_overcontracted": flags.get("latent_map_overcontracted"),
        "cmgdb_duration_seconds": timing.get("duration_seconds"), "cmgdb_precompute_seconds": timing.get("precompute_seconds"),
        "cmgdb_total_seconds": timing.get("total_seconds"), "subdiv_init": settings.get("subdiv_init"),
        "subdiv_min": settings.get("subdiv_min"), "subdiv_max": settings.get("subdiv_max"),
        "box_map_backend": settings.get("box_map_backend"), "lower_bounds": _json_text(bounds.get("lower")),
        "upper_bounds": _json_text(bounds.get("upper")), "bounds_source": bounds.get("source"),
        "n_morse_nodes": graph.get("n_nodes"), "n_morse_edges": graph.get("n_edges"),
        "n_graph_sinks": len(graph["sinks"]) if "sinks" in graph else None,
        "sink_labels": _json_text(graph.get("sinks")), "sink_conley_indices": _json_text(cell["sink_conley_indices"]),
        "sink_degree0_normalized": _json_text(cell["sink_degree0_normalized"]), "node_conley_indices": _json_text(node_indices),
        "morse_boxes_total": sets.get("total_boxes"), "morse_boxes_by_label": _json_text(sets.get("boxes_by_label")),
        "minimal_tolerance": _json_text(metrics.get("minimal_sets")),
        "all_minimal_tolerance_pass": metrics.get("all_minimal_tolerance_pass"),
        "n_minimal_tolerance_failures": metrics.get("n_minimal_tolerance_failures"),
        "checkpoint_sha256": _deep_get(refs, "checkpoint", "sha256"),
        "morse_graph_sha256": _deep_get(refs, "morse_graph", "sha256"),
        "morse_sets_sha256": _deep_get(refs, "morse_sets", "sha256"),
        "config_hash": _deep_get(cell, "run_manifest", "config_hash"),
        "error_count": len(cell["errors"]), "warning_count": len(cell["warnings"]),
        "errors": _json_text(cell["errors"]), "warnings": _json_text(cell["warnings"]),
        "cell_directory": cell["cell_directory"],
    }


def _numeric_summary(values: list[Any]) -> dict[str, Any]:
    clean = [float(value) for value in values if _as_float(value) is not None]
    return {
        "count": len(clean), "mean": statistics.fmean(clean) if clean else None,
        "population_std": statistics.pstdev(clean) if clean else None,
        "min": min(clean) if clean else None, "max": max(clean) if clean else None,
    }


def _load_sweep_summary(sweep_root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    path = sweep_root / "sweep_summary.json"
    if not path.is_file():
        return {"available": False, "reason": "sweep_summary.json absent; cell artifacts used directly"}, []
    try:
        payload, reference = _read_json(path)
        return {"available": True, "file": reference, "payload": payload}, []
    except ValueError as exc:
        return {"available": True, "file": _display(path), "parse_error": str(exc)}, [_issue("invalid_sweep_summary", str(exc), path)]


def _aggregate(cells: list[dict[str, Any]], datasets: list[dict[str, Any]], sweep_root: Path,
               data_root: Path, provisional: bool, sweep_summary: dict[str, Any],
               global_errors: list[dict[str, str]]) -> dict[str, Any]:
    topology = Counter(cell["morse_graph"]["topology_signature_sha256"] for cell in cells if cell["morse_graph"])
    sink_counts = Counter(len(cell["morse_graph"]["sinks"]) for cell in cells if cell["morse_graph"])
    stable_index_counts = Counter(
        cell["n_stable_conley_index_nodes"]
        for cell in cells
        if cell["n_stable_conley_index_nodes"] is not None
    )
    errors = Counter(error["code"] for cell in cells for error in cell["errors"])
    errors.update(error["code"] for error in global_errors)
    warnings = Counter(warning["code"] for cell in cells for warning in cell["warnings"])
    successes = sum(cell["bistability_pass"] is True for cell in cells)
    evaluated = sum(isinstance(cell["bistability_pass"], bool) for cell in cells)
    def final_loss(key: str) -> list[Any]:
        return [
            _deep_get(cell, "training_summary", "final_epoch_train", key)
            for cell in cells
        ]
    setting_profiles = Counter(_json_text(_deep_get(cell, "cmgdb", "settings")) for cell in cells if cell["cmgdb"])
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(), "provisional": provisional,
        "source_is_read_only": True, "sweep_root": _display(sweep_root), "data_root": _display(data_root),
        "expected_design": {"dataset_ids": list(DATASET_IDS), "model_seeds": list(MODEL_SEEDS), "n_cells": 15},
        "success_criterion": SUCCESS_CRITERION,
        "inventory": {
            "n_expected_cells": 15, "n_records": len(cells),
            "n_complete_cells": sum(cell["complete"] for cell in cells),
            "n_verified_cells": sum(cell["verification_passed"] for cell in cells),
            "n_incomplete_cells": sum(not cell["complete"] for cell in cells),
            "n_invalid_cells": sum(cell["cell_status"] == "invalid" for cell in cells),
            "error_counts_by_code": dict(sorted(errors.items())),
            "warning_counts_by_code": dict(sorted(warnings.items())),
            "global_errors": global_errors,
        },
        "bistability": {
            "n_evaluated": evaluated, "n_successes": successes,
            "n_failures": evaluated - successes,
            "rate_among_evaluated": successes / evaluated if evaluated else None,
            "rate_over_expected_15_cells": successes / 15,
            "stable_conley_index_node_count_distribution": {
                str(key): value for key, value in sorted(stable_index_counts.items())
            },
            "by_dataset": {str(dataset_id): {
                "n_successes": sum(
                    cell["bistability_pass"] is True
                    for cell in cells if cell["dataset_id"] == dataset_id
                ),
                "n_evaluated": sum(
                    isinstance(cell["bistability_pass"], bool)
                    for cell in cells if cell["dataset_id"] == dataset_id
                ),
            } for dataset_id in DATASET_IDS},
        },
        "training": {
            "methods": dict(Counter(_deep_get(cell, "training_summary", "training_method") for cell in cells if cell["training_summary"])),
            "final_epoch_train": {
                "loss_reconstruction": _numeric_summary(final_loss("loss_reconstruction")),
                "loss_prediction": _numeric_summary(final_loss("loss_prediction")),
                "loss_total": _numeric_summary(final_loss("loss_total")),
            },
            "epochs_completed": _numeric_summary([_deep_get(cell, "training_summary", "epochs_completed") for cell in cells]),
        },
        "diagnosis": dict(Counter(_deep_get(cell, "diagnose", "diagnostic") for cell in cells if cell["diagnose"])),
        "topology": {"n_distinct_signatures": len(topology), "signature_counts": dict(topology),
                     "sink_count_distribution": {str(key): value for key, value in sorted(sink_counts.items())}},
        "tolerance": {
            "n_cells_evaluated": sum(isinstance(_deep_get(cell, "metrics", "all_minimal_tolerance_pass"), bool) for cell in cells),
            "n_cells_all_minimal_pass": sum(_deep_get(cell, "metrics", "all_minimal_tolerance_pass") is True for cell in cells),
            "total_failed_minimal_sets": sum(_deep_get(cell, "metrics", "n_minimal_tolerance_failures") or 0 for cell in cells),
        },
        "durations_seconds": {
            "training": _numeric_summary([_deep_get(cell, "durations", "training_seconds") for cell in cells]),
            "cmgdb": _numeric_summary([_deep_get(cell, "durations", "cmgdb_seconds") for cell in cells]),
            "combined": _numeric_summary([_deep_get(cell, "durations", "combined_seconds") for cell in cells]),
        },
        "cmgdb_setting_profiles": [{"settings": json.loads(profile), "n_cells": count} for profile, count in sorted(setting_profiles.items())],
        "cell_outcomes": [{"dataset_id": cell["dataset_id"], "model_seed": cell["model_seed"],
                           "status": cell["cell_status"],
                           "bistability_pass": cell["bistability_pass"],
                           "n_stable_conley_index_nodes": cell["n_stable_conley_index_nodes"],
                           "stable_index_labels": cell["stable_index_labels"],
                           "graph_sink_labels": (
                               cell["morse_graph"]["sinks"] if cell["morse_graph"] else None
                           ),
                           "error_codes": [error["code"] for error in cell["errors"]]} for cell in cells],
        "dataset_provenance": [{key: value for key, value in dataset.items() if key != "resolved_output_path"} for dataset in datasets],
        "runner_sweep_summary": sweep_summary,
    }


def _format_number(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    return "-" if number is None else f"{number:.{digits}g}"


def _markdown(cells: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    inv = aggregate["inventory"]
    success = aggregate["bistability"]
    state = "PROVISIONAL / INCOMPLETE" if aggregate["provisional"] else "COMPLETE AND VERIFIED"
    lines = [
        "# Leslie3D Example 2 - Marcio-style 5x3 summary", "", f"**Report status:** {state}", "",
        f"All 15 expected cells are represented. {inv['n_complete_cells']}/15 are complete, "
        f"{inv['n_verified_cells']}/15 passed artifact verification, and {success['n_successes']}/15 "
        "meet the bistability criterion.", "", "## Bistability criterion", "",
        SUCCESS_CRITERION["definition"], "",
        "Graph sink/minimal status and sampled tolerance are shown separately; neither changes the index-based pass.", "",
        f"**Visualization note:** Morse-set panels apply a display-only minimum box side of "
        f"{100 * aggregate['visualization']['summary_morse_sets']['min_box_side_frac']:g}% "
        "of each plotted axis span. The saved CMGDB boxes and all reported topology are unchanged.", "",
        "## Cells", "",
        "| Dataset | Model seed | Status | Final train total | Nodes/edges/sinks | Stable-index nodes (labels: H0) | Tolerance | Diagnosis | CMGDB min |",
        "|---:|---:|---|---:|---|---|---|---|---:|",
    ]
    for cell in cells:
        graph = cell["morse_graph"] or {}
        loss = _deep_get(cell, "training_summary", "final_epoch_train", "loss_total")
        tolerance = _deep_get(cell, "metrics", "all_minimal_tolerance_pass")
        tolerance_text = "pass" if tolerance is True else "fail" if tolerance is False else "unknown"
        diagnosis = _deep_get(cell, "diagnose", "diagnostic") or "-"
        seconds = _deep_get(cell, "durations", "cmgdb_seconds")
        topology = f"{graph.get('n_nodes', '-')}/{graph.get('n_edges', '-')}/{len(graph['sinks']) if 'sinks' in graph else '-'}"
        stable_labels = ",".join(map(str, cell["stable_index_labels"] or [])) or "-"
        stable_h0 = ",".join(
            str(node["conley_index_normalized"][0])
            for node in (cell["stable_conley_index_nodes"] or [])
        ) or "-"
        stable = f"{cell['n_stable_conley_index_nodes']} [{stable_labels}]: {stable_h0}"
        lines.append(f"| {cell['dataset_id']:02d} | {cell['model_seed']} | {cell['cell_status']} | {_format_number(loss)} | {topology} | {stable} | {tolerance_text} | {diagnosis} | {_format_number(seconds / 60 if seconds is not None else None)} |")
    operational = [cell for cell in cells if not cell["verification_passed"]]
    criterion_failures = [
        cell for cell in cells
        if cell["verification_passed"] and cell["bistability_pass"] is False
    ]
    lines += ["", "## Operationally incomplete or invalid cells", ""]
    if not operational:
        lines.append("None.")
    for cell in operational:
        codes = ", ".join(dict.fromkeys(error["code"] for error in cell["errors"])) or "unknown"
        lines.append(f"- `dataset_{cell['dataset_id']:02d}/seed_{cell['model_seed']}`: {cell['cell_status']} ({codes})")
    lines += ["", "## Verified cells that fail the bistability criterion", ""]
    if not criterion_failures:
        lines.append("None.")
    for cell in criterion_failures:
        lines.append(
            f"- `dataset_{cell['dataset_id']:02d}/seed_{cell['model_seed']}`: "
            f"stable-index nodes={cell['n_stable_conley_index_nodes']} "
            f"{cell['stable_index_labels']}; graph sinks={cell['morse_graph']['sinks']}"
        )
    lines += ["", "## Derived artifacts", "", "- `cells.csv` - flat cell inventory", "- `cells.json` - exact parsed per-cell records", "- `aggregate_summary.json` - aggregate counts and distributions", "- `summary.pdf` - six-page visual report", ""]
    return "\n".join(lines)


def _render_summary_morse_set_images(
    cells: list[dict[str, Any]],
    out_dir: Path,
    *,
    min_box_side_frac: float,
) -> dict[tuple[int, int], Path]:
    """Render display-enhanced Morse-set PNGs without changing source artifacts."""
    if not math.isfinite(min_box_side_frac) or min_box_side_frac < 0.0:
        raise ValueError("summary_min_box_side_frac must be finite and nonnegative")
    if min_box_side_frac == 0.0:
        return {}

    import matplotlib
    matplotlib.use("Agg")
    from latentdynamics.viz.morse_plots import render_morse_sets_from_csv

    rendered: dict[tuple[int, int], Path] = {}
    for cell in cells:
        cell_dir = cell["resolved_cell_path"]
        csv_path = cell_dir / REQUIRED_FILES["morse_sets"]
        if not csv_path.is_file() or csv_path.stat().st_size == 0:
            continue
        lower = _deep_get(cell, "cmgdb", "bounds", "lower")
        upper = _deep_get(cell, "cmgdb", "bounds", "upper")
        if not isinstance(lower, list) or not isinstance(upper, list):
            lower = upper = None
        cell_out = out_dir / f"dataset_{cell['dataset_id']:02d}" / f"seed_{cell['model_seed']}"
        # Paper styling adjusts global matplotlib rcParams. Keep those changes
        # local so they cannot affect the six-page PdfPages canvas below.
        with matplotlib.rc_context():
            paths = render_morse_sets_from_csv(
                csv_path,
                cell_out,
                bounds_lower=lower,
                bounds_upper=upper,
                basename="morse_sets_summary",
                formats=("png",),
                box_scale="auto",
                min_box_side_frac=min_box_side_frac,
            )
        rendered[(cell["dataset_id"], cell["model_seed"])] = paths[0]
    return rendered


def _render_pdf(
    path: Path,
    cells: list[dict[str, Any]],
    aggregate: dict[str, Any],
    *,
    summary_morse_set_images: dict[tuple[int, int], Path] | None = None,
    summary_min_box_side_frac: float = 0.0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    green, red, gray = "#1a9850", "#d73027", "#777777"
    if summary_min_box_side_frac > 0.0:
        morse_set_note = (
            f"Morse-set panels use a {100 * summary_min_box_side_frac:g}% display-only "
            "minimum box side; raw CMGDB boxes are unchanged."
        )
    else:
        morse_set_note = "Morse-set panels preserve the raw CMGDB box dimensions."
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with PdfPages(temporary) as pdf:
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.955, "Leslie3D Example 2 - Marcio-style 5x3 run", ha="center", va="top", fontsize=16, weight="bold")
            inv = aggregate["inventory"]
            fig.text(0.5, 0.91, f"{inv['n_verified_cells']}/15 verified | {aggregate['bistability']['n_successes']}/15 bistable | report {'PROVISIONAL' if aggregate['provisional'] else 'COMPLETE'}", ha="center", fontsize=10, color=red if aggregate["provisional"] else green)
            fig.text(
                0.055,
                0.865,
                "Bistability: exactly 2 Morse nodes with nonzero H0 Conley component; "
                "higher components may be nonzero.\n"
                "Graph edges/minimality and sampled tolerance are diagnostics only.\n"
                "Training: exact Marcio full-batch two-term objective.\n"
                + morse_set_note,
                va="top",
                fontsize=8.5,
            )
            headers = ["data", "seed", "state", "train total", "nodes", "edges", "sinks", "stable idx", "tol", "CMGDB min"]
            table_rows, colors = [], []
            for cell in cells:
                graph = cell["morse_graph"] or {}
                tol = _deep_get(cell, "metrics", "all_minimal_tolerance_pass")
                seconds = _deep_get(cell, "durations", "cmgdb_seconds")
                state = "PASS" if cell["verification_passed"] and cell["bistability_pass"] else "FAIL" if cell["verification_passed"] else "INVALID"
                table_rows.append([f"{cell['dataset_id']:02d}", str(cell["model_seed"]), state,
                                   _format_number(_deep_get(cell, "training_summary", "final_epoch_train", "loss_total")),
                                   str(graph.get("n_nodes", "-")), str(graph.get("n_edges", "-")),
                                   str(len(graph["sinks"])) if "sinks" in graph else "-",
                                   (
                                       f"{cell['n_stable_conley_index_nodes']} | "
                                       + ",".join(
                                           str(node["conley_index_normalized"][0])
                                           for node in (cell["stable_conley_index_nodes"] or [])
                                       )
                                   ),
                                   "pass" if tol is True else "fail" if tol is False else "?",
                                   _format_number(seconds / 60 if seconds is not None else None)])
                color = green if state == "PASS" else red if state == "FAIL" else gray
                colors.append(["white", "white", color, "white", "white", "white", "white", "white", "white", "white"])
            ax = fig.add_axes([0.035, 0.08, 0.93, 0.69])
            ax.axis("off")
            table = ax.table(cellText=table_rows, colLabels=headers, cellColours=colors, loc="center", cellLoc="center",
                             colWidths=[0.055, 0.055, 0.08, 0.1, 0.06, 0.06, 0.06, 0.25, 0.06, 0.09])
            table.auto_set_font_size(False)
            table.set_fontsize(7.2)
            table.scale(1, 1.36)
            for (row, _), item in table.get_celld().items():
                if row == 0:
                    item.set_facecolor("#e8e8e8")
                    item.set_text_props(weight="bold")
                elif item.get_facecolor()[:3] not in ((1.0, 1.0, 1.0),):
                    item.set_text_props(color="white", weight="bold")
            fig.text(0.5, 0.025, "Page 1 of 6", ha="center", fontsize=7, color="#555555")
            pdf.savefig(fig)
            plt.close(fig)

            for page_no, dataset_id in enumerate(DATASET_IDS, start=2):
                ds_cells = [cell for cell in cells if cell["dataset_id"] == dataset_id]
                fig = plt.figure(figsize=(11, 8.5))
                fig.text(0.5, 0.955, f"dataset_{dataset_id:02d}", ha="center", va="top", fontsize=15, weight="bold")
                dataset_cell = ds_cells[0] if ds_cells else None
                seeds = dataset_cell["seeds"] if dataset_cell else {}
                train_hash = _deep_get(dataset_cell, "dataset_provenance", "files", "train_csv", "sha256") if dataset_cell else None
                fig.text(0.5, 0.915, f"data seed={seeds.get('data_train')} | validation seed={seeds.get('data_validation')} | train SHA256={(train_hash or 'unavailable')[:12]}", ha="center", fontsize=8)
                for col, seed in enumerate(MODEL_SEEDS):
                    cell = next((item for item in ds_cells if item["model_seed"] == seed), None)
                    left, width = 0.045 + col * 0.32, 0.29
                    verified = bool(cell and cell["verification_passed"])
                    successful = bool(cell and cell["bistability_pass"])
                    border = green if verified and successful else red if verified else gray
                    graph = cell["morse_graph"] if cell else None
                    tol = _deep_get(cell, "metrics", "all_minimal_tolerance_pass") if cell else None
                    state = "BISTABLE PASS" if verified and successful else "BISTABILITY FAIL" if verified else "INCOMPLETE / INVALID"
                    detail = f"{graph['n_nodes']} nodes / {graph['n_edges']} edges / {len(graph['sinks'])} sinks" if graph else "graph unavailable"
                    stable_detail = (
                        f"stable H0={cell['n_stable_conley_index_nodes']} "
                        "["
                        + ",".join(
                            str(node["conley_index_normalized"][0])
                            for node in (cell["stable_conley_index_nodes"] or [])
                        )
                        + "]"
                        if cell else "stable H0 unavailable"
                    )
                    fig.text(
                        left + width / 2,
                        0.88,
                        f"seed {seed} - {state} | {stable_detail}\n{detail} | "
                        f"tolerance {'pass' if tol is True else 'fail' if tol is False else 'unknown'}",
                        ha="center",
                        va="top",
                        fontsize=8.1,
                        color=border,
                        weight="bold",
                    )
                    for row, artifact_key in enumerate(("morse_graph_png", "morse_sets_png")):
                        bottom = 0.49 if row == 0 else 0.09
                        ax = fig.add_axes([left, bottom, width, 0.32])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        image_path = None
                        uses_display_floor = False
                        if cell is not None:
                            if artifact_key == "morse_sets_png" and summary_morse_set_images:
                                image_path = summary_morse_set_images.get(
                                    (cell["dataset_id"], cell["model_seed"])
                                )
                                uses_display_floor = image_path is not None
                            image_path = image_path or cell["resolved_cell_path"] / REQUIRED_FILES[artifact_key]
                        if image_path is not None and image_path.is_file():
                            try:
                                ax.imshow(mpimg.imread(str(image_path)))
                            except Exception as exc:
                                ax.text(0.5, 0.5, f"unreadable image\n{type(exc).__name__}", ha="center", va="center", transform=ax.transAxes)
                        else:
                            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes, color=gray)
                        set_title = (
                            f"Morse sets ({100 * summary_min_box_side_frac:g}% display floor)"
                            if uses_display_floor
                            else "Morse sets"
                        )
                        ax.set_title("Morse graph" if row == 0 else set_title, fontsize=8)
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                    fig.add_artist(Rectangle((left - 0.008, 0.075), width + 0.016, 0.75, fill=False, edgecolor=border, lw=2.2, transform=fig.transFigure))
                fig.text(0.5, 0.025, f"Page {page_no} of 6", ha="center", fontsize=7, color="#555555")
                pdf.savefig(fig)
                plt.close(fig)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_summary(*, sweep_root: Path = DEFAULT_SWEEP_ROOT, data_root: Path = DEFAULT_DATA_ROOT,
                  summary_dir: Path | None = None, allow_incomplete: bool = False,
                  summary_min_box_side_frac: float = SUMMARY_MIN_BOX_SIDE_FRAC) -> dict[str, Path]:
    if not math.isfinite(summary_min_box_side_frac) or summary_min_box_side_frac < 0.0:
        raise ValueError("summary_min_box_side_frac must be finite and nonnegative")
    sweep_root = sweep_root.resolve()
    data_root = data_root.resolve()
    summary_dir = (summary_dir or sweep_root / "summary").resolve()
    datasets = [_analyze_dataset(dataset_id, sweep_root, data_root) for dataset_id in DATASET_IDS]
    cells = [_analyze_cell(dataset, seed) for dataset in datasets for seed in MODEL_SEEDS]
    sweep_summary, global_errors = _load_sweep_summary(sweep_root)
    invalid = [cell for cell in cells if not cell["verification_passed"]]
    if (invalid or global_errors) and not allow_incomplete:
        examples = "; ".join(f"dataset_{cell['dataset_id']:02d}/seed_{cell['model_seed']}: " + ",".join(dict.fromkeys(error["code"] for error in cell["errors"])) for cell in invalid[:4])
        raise SweepValidationError(
            f"strict summary requires all 15 cells to be complete and verified; {len(invalid)} failed verification"
            + (f" ({examples})" if examples else "")
            + ". Use --allow-incomplete for a provisional report."
        )
    provisional = bool(invalid or global_errors)
    aggregate = _aggregate(cells, datasets, sweep_root, data_root, provisional, sweep_summary, global_errors)
    aggregate["visualization"] = {
        "summary_morse_sets": {
            "source": "raw MG/morse_sets CSV",
            "box_scale": "auto",
            "min_box_side_frac": summary_min_box_side_frac,
            "display_only": True,
            "changes_scientific_artifacts": False,
        }
    }
    detailed = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": aggregate["generated_at_utc"],
        "provisional": provisional, "sweep_root": _display(sweep_root), "data_root": _display(data_root),
        "success_criterion": SUCCESS_CRITERION,
        "visualization": aggregate["visualization"],
        "datasets": [{key: value for key, value in dataset.items() if key != "resolved_output_path"} for dataset in datasets],
        "cells": [{key: value for key, value in cell.items() if key not in ("resolved_cell_path",)} for cell in cells],
    }
    summary_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cells_csv": summary_dir / "cells.csv", "cells_json": summary_dir / "cells.json",
        "aggregate_summary": summary_dir / "aggregate_summary.json", "summary_markdown": summary_dir / "SUMMARY.md",
        "summary_pdf": summary_dir / "summary.pdf",
    }
    _atomic_csv(outputs["cells_csv"], [_csv_row(cell) for cell in cells])
    _atomic_json(outputs["cells_json"], detailed)
    _atomic_json(outputs["aggregate_summary"], aggregate)
    _atomic_text(outputs["summary_markdown"], _markdown(cells, aggregate))
    with TemporaryDirectory(prefix=".morse_sets_summary_", dir=summary_dir) as temporary:
        summary_images = _render_summary_morse_set_images(
            cells,
            Path(temporary),
            min_box_side_frac=summary_min_box_side_frac,
        )
        _render_pdf(
            outputs["summary_pdf"],
            cells,
            aggregate,
            summary_morse_set_images=summary_images,
            summary_min_box_side_frac=summary_min_box_side_frac,
        )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--summary-min-box-side-frac",
        type=float,
        default=SUMMARY_MIN_BOX_SIDE_FRAC,
        help="display-only minimum Morse-box side as a fraction of each plotted axis span",
    )
    args = parser.parse_args(argv)
    try:
        outputs = build_summary(sweep_root=args.sweep_root, data_root=args.data_root,
                                summary_dir=args.summary_dir, allow_incomplete=args.allow_incomplete,
                                summary_min_box_side_frac=args.summary_min_box_side_frac)
    except (SweepValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({name: _display(path) for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
