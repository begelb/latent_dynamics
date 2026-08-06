"""Summarize the fixed Leslie3D ground-box curriculum 3x5 sweep.

The source sweep is read-only.  Reports are written only below
``SWEEP_ROOT/summary``.  Strict mode writes nothing unless the expected five
data seeds by three model seeds are present, every training summary satisfies
the fixed full-batch curriculum contract, and the saved Morse graph for every
cell is parseable.  ``--allow-incomplete`` writes the same report as an
explicitly provisional progress audit.
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
DEFAULT_SWEEP_ROOT = CODE_ROOT / "output" / "leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1"
DATA_SEEDS = (2158, 4792, 3174, 688, 5727)
MODEL_SEEDS = (0, 1, 2)
EXPECTED_CELLS = tuple(
    (data_seed, model_seed) for data_seed in DATA_SEEDS for model_seed in MODEL_SEEDS
)
EXPECTED_STAGE_WEIGHTS = ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0))
EXPECTED_STAGE_NAMES = ("autoencoder", "decoded_prediction", "semiconjugacy")
EXPECTED_STAGE_TRAINABLE = (
    ("encoder", "decoder"),
    ("encoder", "latent_map", "decoder"),
    ("encoder", "latent_map", "decoder"),
)
EXPECTED_LEARNING_RATE = 0.003
EXPECTED_STAGE_EPOCHS = 4000
EXPECTED_TOTAL_EPOCHS = 12000
EXPECTED_TRAINING_PAIRS = 20_000
EXPECTED_HOLDOUT_PAIRS = 4_000
EXPECTED_ADAMW_BETAS = (0.9, 0.999)
EXPECTED_ADAMW_EPS = 1.0e-8
EXPECTED_ADAMW_WEIGHT_DECAY = 0.0
EXPECTED_LBFGS_OUTER_STEPS = 12
EXPECTED_LBFGS_LEARNING_RATE = 0.25
EXPECTED_LBFGS_MAX_ITER = 10
EXPECTED_LBFGS_MAX_EVAL = 25
EXPECTED_LBFGS_HISTORY_SIZE = 50
EXPECTED_LBFGS_TOLERANCE_GRAD = 1.0e-9
EXPECTED_LBFGS_TOLERANCE_CHANGE = 1.0e-12
LOSS_KEYS = (
    "loss_reconstruction",
    "loss_prediction",
    "loss_semiconjugacy",
    "loss_total",
)
LOSS_SHORT = {
    "loss_reconstruction": "l1",
    "loss_prediction": "l2",
    "loss_semiconjugacy": "l3",
    "loss_total": "total",
}
PERIODIC_H0 = re.compile(r"^x(?:\^([1-9]\d*))?-1$")
NODE_RE = re.compile(r'^\s*"?(-?\d+)"?\s*\[')
EDGE_RE = re.compile(r'^\s*"?(-?\d+)"?\s*->\s*"?(-?\d+)"?')
LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')
INDEX_RE = re.compile(r"\(([^()]*)\)")


BASE_CSV_FIELDS = (
    "data_seed",
    "model_seed",
    "cell_status",
    "training_contract_valid",
    "training_method",
    "training_seed",
    "epochs_completed",
    "first_order_epochs_completed",
    "n_training_pairs",
    "n_holdout_pairs",
    "full_batch",
    "train_duration_seconds",
)
OPTIMIZER_CSV_FIELDS = (
    "optimizer_sequence",
    "adamw_betas",
    "adamw_eps",
    "adamw_weight_decay",
    "adamw_amsgrad",
    "adamw_foreach",
    "adamw_fused",
    "adamw_updates_completed",
    "adamw_device",
    "adamw_dtype",
    "lbfgs_device",
    "lbfgs_dtype",
    "lbfgs_outer_steps_requested",
    "lbfgs_outer_steps_completed",
    "lbfgs_learning_rate",
    "lbfgs_max_iter",
    "lbfgs_max_eval",
    "lbfgs_history_size",
    "lbfgs_tolerance_grad",
    "lbfgs_tolerance_change",
    "lbfgs_line_search_fn",
    "lbfgs_internal_iterations",
    "lbfgs_closure_evaluations",
    "checkpoint_selection",
    "checkpoint_source",
)
STAGE_CSV_FIELDS = tuple(
    field
    for stage in range(1, 4)
    for field in (
        f"stage{stage}_name",
        f"stage{stage}_learning_rate",
        f"stage{stage}_loss_weights",
        f"stage{stage}_trainable_components",
        *(f"stage{stage}_train_{LOSS_SHORT[key]}" for key in LOSS_KEYS),
        *(f"stage{stage}_holdout_{LOSS_SHORT[key]}" for key in LOSS_KEYS),
    )
)
ADAMW_ENDPOINT_CSV_FIELDS = tuple(
    f"adamw_{endpoint}_{LOSS_SHORT[key]}" for endpoint in ("train", "holdout") for key in LOSS_KEYS
)
FINAL_CSV_FIELDS = tuple(
    f"final_{endpoint}_{LOSS_SHORT[key]}" for endpoint in ("train", "holdout") for key in LOSS_KEYS
)
POLISH_DELTA_CSV_FIELDS = tuple(
    f"lbfgs_delta_{endpoint}_{LOSS_SHORT[key]}"
    for endpoint in ("train", "holdout")
    for key in LOSS_KEYS
)
TOPOLOGY_CSV_FIELDS = (
    "n_morse_nodes",
    "n_morse_edges",
    "n_minimal_nodes",
    "n_attractor_type_nodes",
    "n_periodic_attractor_nodes",
    "sink_nodes",
    "sink_conley_indices",
    "bistability_pass",
    "exact_period4_bistability_pass",
    "sweep_reported_n_morse_nodes",
    "sweep_reported_n_morse_edges",
    "sweep_reported_n_sinks",
    "sweep_reported_bistability_pass",
    "latent_lower_bounds",
    "latent_upper_bounds",
    "latent_bounds_source",
    "cmgdb_subdiv_init",
    "cmgdb_subdiv_min",
    "cmgdb_subdiv_max",
    "cmgdb_subdiv_limit",
    "box_map_backend",
    "bounds_data_role",
    "adaptive_precompute_subdiv",
    "diagnostic",
    "encoder_collapsed",
    "latent_map_overcontracted",
    "n_metric_minimal_components",
    "n_sampled_tolerance_evaluable",
    "n_sampled_tolerance_pass",
    "minimal_component_metrics",
)
ARTIFACT_CSV_FIELDS = (
    "cell_directory",
    "training_summary_path",
    "checkpoint_path",
    "checkpoint_metadata_path",
    "adamw_checkpoint_path",
    "adamw_checkpoint_metadata_path",
    "history_path",
    "morse_graph_path",
    "morse_sets_path",
    "mg_params_log_path",
    "metrics_path",
    "diagnose_path",
    "run_manifest_path",
    "artifact_paths",
    "training_summary_sha256",
    "checkpoint_sha256",
    "adamw_checkpoint_sha256",
    "morse_graph_sha256",
    "artifact_sha256",
    "artifact_exists",
    "validation_errors",
)
CSV_FIELDS = (
    BASE_CSV_FIELDS
    + OPTIMIZER_CSV_FIELDS
    + STAGE_CSV_FIELDS
    + ADAMW_ENDPOINT_CSV_FIELDS
    + FINAL_CSV_FIELDS
    + POLISH_DELTA_CSV_FIELDS
    + TOPOLOGY_CSV_FIELDS
    + ARTIFACT_CSV_FIELDS
)


class SweepValidationError(RuntimeError):
    """The source sweep does not satisfy the fixed reporting contract."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(CODE_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


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


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _issue(scope: str, code: str, message: str, path: Path | None = None) -> dict[str, str]:
    result = {"scope": scope, "code": code, "message": message}
    if path is not None:
        result["path"] = _display(path)
    return result


def _resolve_path(raw: Any, *, bases: tuple[Path, ...], fallback: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        return fallback.resolve()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = tuple((base / path).resolve() for base in bases)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _output_directory(
    record: dict[str, Any], sweep_root: Path, data_seed: int, model_seed: int
) -> Path:
    fallback = sweep_root / f"dataset_{data_seed}" / f"seed_{model_seed}"
    return _resolve_path(
        record.get("output_dir"),
        bases=(CODE_ROOT, sweep_root),
        fallback=fallback,
    )


def _training_artifacts(output_dir: Path, summary: dict[str, Any] | None) -> dict[str, Path]:
    declared = summary.get("artifacts", {}) if isinstance(summary, dict) else {}
    if not isinstance(declared, dict):
        declared = {}

    def declared_path(name: str, fallback: str) -> Path:
        return _resolve_path(
            declared.get(name),
            bases=(output_dir,),
            fallback=output_dir / fallback,
        )

    artifacts = {
        "training_summary": output_dir / "training_summary.json",
        "checkpoint": declared_path("checkpoint", "models/autoencoder.pt"),
        "checkpoint_metadata": declared_path("checkpoint_metadata", "models/autoencoder.json"),
        "adamw_checkpoint": declared_path(
            "adamw_checkpoint", "adamw_endpoint/models/autoencoder.pt"
        ),
        "adamw_checkpoint_metadata": declared_path(
            "adamw_checkpoint_metadata", "adamw_endpoint/models/autoencoder.json"
        ),
        "history": declared_path("history", "logs/history.json"),
        "morse_graph": output_dir / "MG" / "morse_graph",
        "morse_sets": output_dir / "MG" / "morse_sets",
        "mg_params_log": output_dir / "mg_params_log.txt",
        "metrics": output_dir / "metrics.json",
        "diagnose": output_dir / "diagnose.json",
        "run_manifest": output_dir / "run_manifest.json",
    }
    stages = summary.get("curriculum") if isinstance(summary, dict) else None
    if isinstance(stages, list):
        for offset, expected_name in enumerate(EXPECTED_STAGE_NAMES, start=1):
            stage = stages[offset - 1] if offset <= len(stages) else None
            stage = stage if isinstance(stage, dict) else {}
            stage_root = (
                output_dir / "stage_checkpoints" / f"{offset:02d}_{expected_name}" / "models"
            )
            artifacts[f"stage{offset}_checkpoint"] = _resolve_path(
                stage.get("checkpoint"),
                bases=(output_dir,),
                fallback=stage_root / "autoencoder.pt",
            )
            artifacts[f"stage{offset}_checkpoint_metadata"] = _resolve_path(
                stage.get("checkpoint_metadata"),
                bases=(output_dir,),
                fallback=stage_root / "autoencoder.json",
            )
    return artifacts


def _expect(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, found {actual!r}")


def _loss_block(errors: list[str], label: str, value: Any) -> dict[str, float | None]:
    block = value if isinstance(value, dict) else {}
    if not isinstance(value, dict):
        errors.append(f"{label}: expected a loss object")
    out: dict[str, float | None] = {}
    for key in LOSS_KEYS:
        number = _as_float(block.get(key))
        if number is None:
            errors.append(f"{label}.{key}: missing or non-finite")
        out[key] = number
    return out


def _validate_weighted_total(
    errors: list[str],
    label: str,
    losses: dict[str, float | None],
    weights: tuple[float, float, float],
) -> None:
    terms = tuple(losses[key] for key in LOSS_KEYS[:3])
    total = losses["loss_total"]
    if total is None or any(term is None for term in terms):
        return
    expected = sum(weight * float(term) for weight, term in zip(weights, terms, strict=True))
    if not math.isclose(total, expected, rel_tol=1e-6, abs_tol=1e-9):
        errors.append(f"{label}.loss_total: expected weighted sum {expected!r}, found {total!r}")


def _validate_training(
    payload: dict[str, Any], *, model_seed: int
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict):
        optimizer = {}
    first_order = optimizer.get("first_order")
    if not isinstance(first_order, dict):
        first_order = {}
    polish = optimizer.get("polish")
    if not isinstance(polish, dict):
        polish = {}
    _expect(errors, "training_method", payload.get("training_method"), "curriculum_full_batch")
    _expect(errors, "seed", _as_int(payload.get("seed")), model_seed)
    for field in ("n_epochs_run", "epochs_requested", "epochs_completed"):
        _expect(errors, field, _as_int(payload.get(field)), EXPECTED_TOTAL_EPOCHS)
    _expect(
        errors,
        "first_order_epochs_completed",
        _as_int(payload.get("first_order_epochs_completed")),
        EXPECTED_TOTAL_EPOCHS,
    )
    # The saved endpoint follows L-BFGS and therefore is not an epoch.  An old
    # ``checkpoint_epoch=12000`` field would misstate its provenance.
    _expect(errors, "checkpoint_epoch", payload.get("checkpoint_epoch"), None)
    _expect(errors, "scheduler", payload.get("scheduler"), None)
    _expect(errors, "scheduler_used", payload.get("scheduler_used"), False)
    _expect(errors, "early_stopping_used", payload.get("early_stopping_used"), False)
    _expect(errors, "patience_used", payload.get("patience_used"), False)
    _expect(errors, "gradient_clipping_used", payload.get("gradient_clipping_used"), False)
    _expect(
        errors,
        "checkpoint_selection",
        payload.get("checkpoint_selection"),
        "final_lbfgs_float32_endpoint",
    )
    _expect(
        errors,
        "checkpoint_source",
        payload.get("checkpoint_source"),
        "lbfgs_float32_endpoint",
    )
    _expect(errors, "best_epoch", payload.get("best_epoch"), None)
    _expect(errors, "validation_evaluated", payload.get("validation_evaluated"), True)
    _expect(
        errors,
        "validation_used_for_optimization",
        payload.get("validation_used_for_optimization"),
        False,
    )
    _expect(
        errors,
        "validation_used_for_checkpoint_selection",
        payload.get("validation_used_for_checkpoint_selection"),
        False,
    )
    _expect(
        errors, "best_weight_restoration_used", payload.get("best_weight_restoration_used"), False
    )
    sequence_raw = optimizer.get("sequence")
    sequence = tuple(sequence_raw) if isinstance(sequence_raw, list) else None
    _expect(errors, "optimizer.sequence", sequence, ("AdamW", "LBFGS"))
    _expect(errors, "optimizer.first_order.name", first_order.get("name"), "AdamW")
    betas_raw = first_order.get("betas")
    betas = (
        tuple(_as_float(value) for value in betas_raw)
        if isinstance(betas_raw, list) and len(betas_raw) == 2
        else None
    )
    _expect(errors, "optimizer.first_order.betas", betas, EXPECTED_ADAMW_BETAS)
    _expect(
        errors,
        "optimizer.first_order.eps",
        _as_float(first_order.get("eps")),
        EXPECTED_ADAMW_EPS,
    )
    _expect(
        errors,
        "optimizer.first_order.weight_decay",
        _as_float(first_order.get("weight_decay")),
        EXPECTED_ADAMW_WEIGHT_DECAY,
    )
    for flag in ("amsgrad", "foreach", "fused"):
        _expect(errors, f"optimizer.first_order.{flag}", first_order.get(flag), False)
    _expect(
        errors,
        "optimizer.first_order.state_continues_across_stages",
        first_order.get("state_continues_across_stages"),
        True,
    )
    optimizer_lrs = first_order.get("stage_learning_rates")
    normalized_lrs = (
        tuple(_as_float(value) for value in optimizer_lrs)
        if isinstance(optimizer_lrs, list) and len(optimizer_lrs) == 3
        else None
    )
    _expect(
        errors,
        "optimizer.first_order.stage_learning_rates",
        normalized_lrs,
        (EXPECTED_LEARNING_RATE,) * 3,
    )
    _expect(
        errors,
        "optimizer.first_order.updates_completed",
        _as_int(first_order.get("updates_completed")),
        EXPECTED_TOTAL_EPOCHS,
    )
    first_order_device = first_order.get("device")
    if not isinstance(first_order_device, str) or not first_order_device.strip():
        errors.append("optimizer.first_order.device: expected a non-empty device string")
    _expect(errors, "optimizer.first_order.dtype", first_order.get("dtype"), "float32")

    _expect(errors, "optimizer.polish.name", polish.get("name"), "LBFGS")
    _expect(
        errors,
        "optimizer.polish.starts_with_fresh_optimizer_state",
        polish.get("starts_with_fresh_optimizer_state"),
        True,
    )
    _expect(errors, "optimizer.polish.device", polish.get("device"), "cpu")
    _expect(errors, "optimizer.polish.dtype", polish.get("dtype"), "float64")
    for field, expected in (
        ("outer_steps_requested", EXPECTED_LBFGS_OUTER_STEPS),
        ("outer_steps_completed", EXPECTED_LBFGS_OUTER_STEPS),
        ("max_iter", EXPECTED_LBFGS_MAX_ITER),
        ("max_eval", EXPECTED_LBFGS_MAX_EVAL),
        ("history_size", EXPECTED_LBFGS_HISTORY_SIZE),
    ):
        _expect(errors, f"optimizer.polish.{field}", _as_int(polish.get(field)), expected)
    for field, expected in (
        ("learning_rate", EXPECTED_LBFGS_LEARNING_RATE),
        ("tolerance_grad", EXPECTED_LBFGS_TOLERANCE_GRAD),
        ("tolerance_change", EXPECTED_LBFGS_TOLERANCE_CHANGE),
    ):
        _expect(errors, f"optimizer.polish.{field}", _as_float(polish.get(field)), expected)
    _expect(
        errors,
        "optimizer.polish.line_search_fn",
        polish.get("line_search_fn"),
        "strong_wolfe",
    )
    polish_weights_raw = polish.get("loss_weights")
    polish_weights = (
        tuple(_as_float(value) for value in polish_weights_raw)
        if isinstance(polish_weights_raw, list) and len(polish_weights_raw) == 3
        else None
    )
    _expect(
        errors,
        "optimizer.polish.loss_weights",
        polish_weights,
        EXPECTED_STAGE_WEIGHTS[-1],
    )
    polish_trainable_raw = polish.get("trainable_components")
    polish_trainable = (
        tuple(polish_trainable_raw) if isinstance(polish_trainable_raw, list) else None
    )
    _expect(
        errors,
        "optimizer.polish.trainable_components",
        polish_trainable,
        EXPECTED_STAGE_TRAINABLE[-1],
    )
    closure_evaluations = _as_int(polish.get("closure_evaluations"))
    internal_iterations = _as_int(polish.get("internal_iterations"))
    if closure_evaluations is None or not (
        EXPECTED_LBFGS_OUTER_STEPS
        <= closure_evaluations
        <= EXPECTED_LBFGS_OUTER_STEPS * EXPECTED_LBFGS_MAX_EVAL
    ):
        errors.append(
            "optimizer.polish.closure_evaluations: expected an integer in "
            f"[{EXPECTED_LBFGS_OUTER_STEPS}, "
            f"{EXPECTED_LBFGS_OUTER_STEPS * EXPECTED_LBFGS_MAX_EVAL}], "
            f"found {polish.get('closure_evaluations')!r}"
        )
    if internal_iterations is None or not (
        0 <= internal_iterations <= EXPECTED_LBFGS_OUTER_STEPS * EXPECTED_LBFGS_MAX_ITER
    ):
        errors.append(
            "optimizer.polish.internal_iterations: expected an integer in "
            f"[0, {EXPECTED_LBFGS_OUTER_STEPS * EXPECTED_LBFGS_MAX_ITER}], "
            f"found {polish.get('internal_iterations')!r}"
        )
    _expect(
        errors,
        "final_learning_rate",
        _as_float(payload.get("final_learning_rate")),
        EXPECTED_LEARNING_RATE,
    )
    _expect(errors, "model_initialized_by_helper", payload.get("model_initialized_by_helper"), True)

    arch = payload.get("arch") if isinstance(payload.get("arch"), dict) else {}
    _expect(errors, "arch.high_dims", _as_int(arch.get("high_dims")), 3)
    _expect(errors, "arch.low_dims", _as_int(arch.get("low_dims")), 2)
    for component, hidden_shapes in (
        ("encoder", [128, 64]),
        ("latent_map", [64, 64]),
        ("decoder", [64, 128]),
    ):
        block = arch.get(component) if isinstance(arch.get(component), dict) else {}
        _expect(
            errors, f"arch.{component}.hidden_shapes", block.get("hidden_shapes"), hidden_shapes
        )
        _expect(errors, f"arch.{component}.activation", block.get("activation"), "tanh")
        _expect(errors, f"arch.{component}.out_activation", block.get("out_activation"), "none")

    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    _expect(errors, "data.full_batch", data.get("full_batch"), True)
    _expect(
        errors,
        "data.n_training_pairs",
        _as_int(data.get("n_training_pairs")),
        EXPECTED_TRAINING_PAIRS,
    )
    _expect(
        errors,
        "data.n_validation_pairs",
        _as_int(data.get("n_validation_pairs")),
        EXPECTED_HOLDOUT_PAIRS,
    )
    _expect(errors, "data.high_dims", _as_int(data.get("high_dims")), 3)
    _expect(errors, "data.dtype", data.get("dtype"), "float32")

    stages_raw = payload.get("curriculum")
    stages = stages_raw if isinstance(stages_raw, list) else []
    _expect(errors, "curriculum stage count", len(stages), 3)
    normalized_stages: list[dict[str, Any]] = []
    for offset, expected_weights in enumerate(EXPECTED_STAGE_WEIGHTS):
        stage_number = offset + 1
        stage = stages[offset] if offset < len(stages) and isinstance(stages[offset], dict) else {}
        if not stage:
            errors.append(f"curriculum[{offset}]: missing stage object")
        _expect(errors, f"curriculum[{offset}].index", _as_int(stage.get("index")), stage_number)
        _expect(
            errors, f"curriculum[{offset}].name", stage.get("name"), EXPECTED_STAGE_NAMES[offset]
        )
        _expect(
            errors,
            f"curriculum[{offset}].epochs",
            _as_int(stage.get("epochs")),
            EXPECTED_STAGE_EPOCHS,
        )
        _expect(
            errors,
            f"curriculum[{offset}].start_epoch_one_based",
            _as_int(stage.get("start_epoch_one_based")),
            offset * EXPECTED_STAGE_EPOCHS + 1,
        )
        _expect(
            errors,
            f"curriculum[{offset}].end_epoch_one_based",
            _as_int(stage.get("end_epoch_one_based")),
            stage_number * EXPECTED_STAGE_EPOCHS,
        )
        raw_weights = stage.get("loss_weights")
        weights = (
            tuple(_as_float(value) for value in raw_weights)
            if isinstance(raw_weights, list) and len(raw_weights) == 3
            else None
        )
        _expect(errors, f"curriculum[{offset}].loss_weights", weights, expected_weights)
        _expect(
            errors,
            f"curriculum[{offset}].learning_rate",
            _as_float(stage.get("learning_rate")),
            EXPECTED_LEARNING_RATE,
        )
        raw_trainable = stage.get("trainable_components")
        trainable = tuple(raw_trainable) if isinstance(raw_trainable, list) else None
        _expect(
            errors,
            f"curriculum[{offset}].trainable_components",
            trainable,
            EXPECTED_STAGE_TRAINABLE[offset],
        )
        _expect(
            errors,
            f"curriculum[{offset}].optimizer_state_continued_from_previous_stage",
            stage.get("optimizer_state_continued_from_previous_stage"),
            offset > 0,
        )
        train_endpoint = _loss_block(
            errors,
            f"curriculum[{offset}].train_endpoint_post_update",
            stage.get("train_endpoint_post_update"),
        )
        holdout_endpoint = _loss_block(
            errors,
            f"curriculum[{offset}].holdout_endpoint_post_update",
            stage.get("holdout_endpoint_post_update"),
        )
        _validate_weighted_total(
            errors,
            f"curriculum[{offset}].train_endpoint_post_update",
            train_endpoint,
            expected_weights,
        )
        _validate_weighted_total(
            errors,
            f"curriculum[{offset}].holdout_endpoint_post_update",
            holdout_endpoint,
            expected_weights,
        )
        normalized_stages.append(
            {
                "index": stage_number,
                "name": stage.get("name"),
                "learning_rate": _as_float(stage.get("learning_rate")),
                "loss_weights": list(weights) if weights is not None else None,
                "trainable_components": stage.get("trainable_components"),
                "train": train_endpoint,
                "holdout": holdout_endpoint,
            }
        )

    final_weights_raw = payload.get("loss_weights")
    final_weights = (
        tuple(_as_float(value) for value in final_weights_raw)
        if isinstance(final_weights_raw, list) and len(final_weights_raw) == 3
        else None
    )
    _expect(errors, "loss_weights", final_weights, EXPECTED_STAGE_WEIGHTS[-1])
    adamw_train = _loss_block(errors, "adamw_endpoint_train", payload.get("adamw_endpoint_train"))
    adamw_holdout = _loss_block(
        errors, "adamw_endpoint_holdout", payload.get("adamw_endpoint_holdout")
    )
    final_train = _loss_block(
        errors, "final_checkpoint_train", payload.get("final_checkpoint_train")
    )
    final_holdout = _loss_block(errors, "final_holdout", payload.get("final_holdout"))
    _validate_weighted_total(
        errors, "adamw_endpoint_train", adamw_train, EXPECTED_STAGE_WEIGHTS[-1]
    )
    _validate_weighted_total(
        errors, "adamw_endpoint_holdout", adamw_holdout, EXPECTED_STAGE_WEIGHTS[-1]
    )
    _validate_weighted_total(
        errors, "final_checkpoint_train", final_train, EXPECTED_STAGE_WEIGHTS[-1]
    )
    _validate_weighted_total(errors, "final_holdout", final_holdout, EXPECTED_STAGE_WEIGHTS[-1])

    for endpoint, before, after in (
        ("train", adamw_train, final_train),
        ("holdout", adamw_holdout, final_holdout),
    ):
        reported_delta = _loss_block(
            errors,
            f"polish_delta_{endpoint}",
            payload.get(f"polish_delta_{endpoint}"),
        )
        for key in LOSS_KEYS:
            before_value = before[key]
            after_value = after[key]
            reported_value = reported_delta[key]
            if before_value is not None and after_value is not None and reported_value is not None:
                expected_delta = after_value - before_value
                if not math.isclose(reported_value, expected_delta, rel_tol=1e-7, abs_tol=1e-10):
                    errors.append(
                        f"polish_delta_{endpoint}.{key}: expected final-minus-AdamW "
                        f"delta {expected_delta!r}, found {reported_value!r}"
                    )

    # Stage 3 is the saved AdamW endpoint immediately before the fresh L-BFGS
    # optimizer is created.  Require both records to identify the same weights.
    if len(normalized_stages) == 3:
        for endpoint, adamw_block in (("train", adamw_train), ("holdout", adamw_holdout)):
            stage_block = normalized_stages[2][endpoint]
            for key in LOSS_KEYS:
                left = stage_block[key]
                right = adamw_block[key]
                if (
                    left is not None
                    and right is not None
                    and not math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-10)
                ):
                    errors.append(
                        f"adamw_endpoint_{endpoint}.{key}: expected stage-3 endpoint "
                        f"{left!r}, found {right!r}"
                    )

    adamw_total = adamw_train["loss_total"]
    final_total = final_train["loss_total"]
    if (
        adamw_total is not None
        and final_total is not None
        and final_total > adamw_total
        and not math.isclose(final_total, adamw_total, rel_tol=1e-6, abs_tol=1e-9)
    ):
        errors.append(
            "final_checkpoint_train.loss_total: L-BFGS float32 endpoint increased the "
            f"training objective from {adamw_total!r} to {final_total!r}"
        )
    normalized = {
        "training_method": payload.get("training_method"),
        "seed": _as_int(payload.get("seed")),
        "epochs_completed": _as_int(payload.get("epochs_completed")),
        "first_order_epochs_completed": _as_int(payload.get("first_order_epochs_completed")),
        "n_training_pairs": _as_int(data.get("n_training_pairs")),
        "n_holdout_pairs": _as_int(data.get("n_validation_pairs")),
        "full_batch": data.get("full_batch"),
        "train_duration_seconds": _as_float(payload.get("train_duration_seconds")),
        "optimizer": {
            "sequence": list(sequence) if sequence is not None else None,
            "first_order": {
                "betas": list(betas) if betas is not None else None,
                "eps": _as_float(first_order.get("eps")),
                "weight_decay": _as_float(first_order.get("weight_decay")),
                "amsgrad": first_order.get("amsgrad"),
                "foreach": first_order.get("foreach"),
                "fused": first_order.get("fused"),
                "updates_completed": _as_int(first_order.get("updates_completed")),
                "device": first_order.get("device"),
                "dtype": first_order.get("dtype"),
            },
            "polish": {
                "device": polish.get("device"),
                "dtype": polish.get("dtype"),
                "outer_steps_requested": _as_int(polish.get("outer_steps_requested")),
                "outer_steps_completed": _as_int(polish.get("outer_steps_completed")),
                "learning_rate": _as_float(polish.get("learning_rate")),
                "max_iter": _as_int(polish.get("max_iter")),
                "max_eval": _as_int(polish.get("max_eval")),
                "history_size": _as_int(polish.get("history_size")),
                "tolerance_grad": _as_float(polish.get("tolerance_grad")),
                "tolerance_change": _as_float(polish.get("tolerance_change")),
                "line_search_fn": polish.get("line_search_fn"),
                "internal_iterations": internal_iterations,
                "closure_evaluations": closure_evaluations,
            },
        },
        "checkpoint_selection": payload.get("checkpoint_selection"),
        "checkpoint_source": payload.get("checkpoint_source"),
        "stages": normalized_stages,
        "adamw_train": adamw_train,
        "adamw_holdout": adamw_holdout,
        "final_train": final_train,
        "final_holdout": final_holdout,
    }
    return normalized, errors


def _parse_morse_graph(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    nodes: dict[int, tuple[str, ...]] = {}
    edges: list[tuple[int, int]] = []
    for line in text.splitlines():
        edge_match = EDGE_RE.match(line)
        if edge_match:
            edges.append((int(edge_match.group(1)), int(edge_match.group(2))))
            continue
        node_match = NODE_RE.match(line)
        if not node_match:
            continue
        label_match = LABEL_RE.search(line)
        index_match = INDEX_RE.search(label_match.group(1)) if label_match else None
        if index_match is None:
            raise ValueError(f"node {node_match.group(1)} has no parseable Conley index")
        nodes[int(node_match.group(1))] = tuple(
            component.strip().replace(" ", "") for component in index_match.group(1).split(",")
        )
    if not nodes:
        raise ValueError("Morse graph has no parseable nodes")
    unknown = {node for edge in edges for node in edge} - set(nodes)
    if unknown:
        raise ValueError(f"edges reference unknown nodes {sorted(unknown)}")
    outgoing = {source for source, _ in edges}
    sink_ids = sorted(set(nodes) - outgoing)

    def period(index: tuple[str, ...]) -> int | None:
        if not index or any(component != "0" for component in index[1:]):
            return None
        match = PERIODIC_H0.fullmatch(index[0])
        return int(match.group(1) or 1) if match else None

    sink_nodes = [
        {
            "node": node,
            "index": f"({', '.join(nodes[node])})",
            "period": period(nodes[node]),
        }
        for node in sink_ids
    ]
    return {
        "n_morse_nodes": len(nodes),
        "n_morse_edges": len(edges),
        "n_minimal_nodes": len(sink_nodes),
        "n_attractor_type_nodes": sum(
            bool(index) and index[0] not in ("", "0") for index in nodes.values()
        ),
        "n_periodic_attractor_nodes": sum(period(index) is not None for index in nodes.values()),
        "sink_nodes": sink_nodes,
        "sink_conley_indices": [sink["index"] for sink in sink_nodes],
        "bistability_pass": len(sink_nodes) == 2
        and all(sink["period"] is not None for sink in sink_nodes),
        "exact_period4_bistability_pass": len(sink_nodes) == 2
        and all(sink["period"] == 4 for sink in sink_nodes),
    }


def _parse_mg_params_log(path: Path) -> tuple[dict[str, Any], list[str]]:
    raw: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        raw[key.strip().lower().replace(" ", "_")] = value.strip()

    errors: list[str] = []

    def integer(name: str) -> int | None:
        try:
            return int(raw[name])
        except (KeyError, TypeError, ValueError):
            errors.append(f"mg_params_log.{name}: missing or not an integer")
            return None

    def bounds(name: str) -> list[float] | None:
        try:
            value = ast.literal_eval(raw[name])
            parsed = [float(item) for item in value]
        except (KeyError, TypeError, ValueError, SyntaxError):
            errors.append(f"mg_params_log.{name}: missing or not a numeric list")
            return None
        if len(parsed) != 2 or not all(math.isfinite(item) for item in parsed):
            errors.append(f"mg_params_log.{name}: expected two finite coordinates")
            return None
        return parsed

    lower = bounds("lower_bounds")
    upper = bounds("upper_bounds")
    if (
        lower is not None
        and upper is not None
        and any(lo >= hi for lo, hi in zip(lower, upper, strict=True))
    ):
        errors.append("mg_params_log: each lower latent bound must be below its upper bound")

    expected = {
        "subdiv_init": 25,
        "subdiv_min": 28,
        "subdiv_max": 29,
        "subdiv_limit": 10_000,
    }
    parsed_ints = {name: integer(name) for name in expected}
    for name, value in expected.items():
        if parsed_ints[name] is not None and parsed_ints[name] != value:
            errors.append(f"mg_params_log.{name}: expected {value}, found {parsed_ints[name]}")
    for name, value in (
        ("padding", "True"),
        ("box_map_backend", "adaptive_precomputed"),
        ("bounds_source", "encoded_train_pairs"),
        ("bounds_data_role", "train_pairs"),
        ("adaptive_precompute_subdiv", "init"),
    ):
        if raw.get(name) != value:
            errors.append(f"mg_params_log.{name}: expected {value!r}, found {raw.get(name)!r}")
    try:
        epsilon = float(raw["bounds_epsilon_frac"])
    except (KeyError, TypeError, ValueError):
        epsilon = None
        errors.append("mg_params_log.bounds_epsilon_frac: missing or not numeric")
    if epsilon is not None and not math.isclose(epsilon, 0.01, rel_tol=0.0, abs_tol=1e-12):
        errors.append(f"mg_params_log.bounds_epsilon_frac: expected 0.01, found {epsilon}")

    return (
        {
            "lower_bounds": lower,
            "upper_bounds": upper,
            "bounds_source": raw.get("bounds_source"),
            "subdiv_init": parsed_ints["subdiv_init"],
            "subdiv_min": parsed_ints["subdiv_min"],
            "subdiv_max": parsed_ints["subdiv_max"],
            "subdiv_limit": parsed_ints["subdiv_limit"],
            "box_map_backend": raw.get("box_map_backend"),
            "bounds_data_role": raw.get("bounds_data_role"),
            "adaptive_precompute_subdiv": raw.get("adaptive_precompute_subdiv"),
        },
        errors,
    )


def _compare_topology(record: dict[str, Any], graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    comparisons = (
        ("n_morse_nodes", "n_morse_nodes"),
        ("n_morse_edges", "n_morse_edges"),
        ("n_sinks", "n_minimal_nodes"),
        ("n_attractor_type_nodes", "n_attractor_type_nodes"),
        ("n_periodic_attractor_nodes", "n_periodic_attractor_nodes"),
        ("bistability_pass", "bistability_pass"),
    )
    for reported_key, parsed_key in comparisons:
        reported = record.get(reported_key)
        if reported is not None and reported != graph[parsed_key]:
            errors.append(
                f"sweep_summary.{reported_key}={reported!r} conflicts with "
                f"saved Morse graph value {graph[parsed_key]!r}"
            )
    reported_sinks = record.get("sink_nodes")
    if isinstance(reported_sinks, list) and reported_sinks != graph["sink_nodes"]:
        errors.append("sweep_summary.sink_nodes conflicts with the saved Morse graph")
    return errors


def _blank_row(data_seed: int, model_seed: int) -> dict[str, Any]:
    return dict.fromkeys(CSV_FIELDS) | {
        "data_seed": data_seed,
        "model_seed": model_seed,
        "cell_status": "missing",
        "training_contract_valid": False,
    }


def _row_for_cell(
    *,
    record: dict[str, Any] | None,
    sweep_root: Path,
    data_seed: int,
    model_seed: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    scope = f"data_seed={data_seed},model_seed={model_seed}"
    row = _blank_row(data_seed, model_seed)
    issues: list[dict[str, str]] = []
    if record is None:
        issues.append(
            _issue(scope, "missing_cell", "expected cell is absent from sweep_summary.json")
        )
        row["validation_errors"] = _json_text([issue["message"] for issue in issues])
        return row, issues

    output_dir = _output_directory(record, sweep_root, data_seed, model_seed)
    row["cell_directory"] = _display(output_dir)
    training_path = output_dir / "training_summary.json"
    training_payload: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    training_errors: list[str] = []
    if not training_path.is_file():
        training_errors.append("training_summary.json is missing")
    else:
        try:
            training_payload = _read_json(training_path)
            training, training_errors = _validate_training(training_payload, model_seed=model_seed)
        except ValueError as exc:
            training_errors.append(str(exc))
    for message in training_errors:
        issues.append(_issue(scope, "invalid_training_contract", message, training_path))

    artifacts = _training_artifacts(output_dir, training_payload)
    required_artifacts = [
        "checkpoint",
        "checkpoint_metadata",
        "adamw_checkpoint",
        "adamw_checkpoint_metadata",
        "history",
        "morse_graph",
        "morse_sets",
        "mg_params_log",
        "metrics",
        "diagnose",
        "run_manifest",
    ]
    required_artifacts.extend(
        name
        for name in artifacts
        if name.startswith("stage") and (name.endswith("checkpoint") or name.endswith("metadata"))
    )
    for name in required_artifacts:
        if not artifacts[name].is_file() or artifacts[name].stat().st_size == 0:
            issues.append(
                _issue(
                    scope,
                    "missing_artifact",
                    f"required artifact {name!r} is missing or empty",
                    artifacts[name],
                )
            )

    graph: dict[str, Any] | None = None
    if artifacts["morse_graph"].is_file():
        try:
            graph = _parse_morse_graph(artifacts["morse_graph"])
            for message in _compare_topology(record, graph):
                issues.append(_issue(scope, "topology_conflict", message, artifacts["morse_graph"]))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            issues.append(_issue(scope, "invalid_morse_graph", str(exc), artifacts["morse_graph"]))

    mg_params: dict[str, Any] | None = None
    if artifacts["mg_params_log"].is_file():
        try:
            mg_params, mg_errors = _parse_mg_params_log(artifacts["mg_params_log"])
            for message in mg_errors:
                issues.append(
                    _issue(scope, "invalid_cmgdb_contract", message, artifacts["mg_params_log"])
                )
        except (OSError, UnicodeDecodeError) as exc:
            issues.append(
                _issue(scope, "invalid_cmgdb_contract", str(exc), artifacts["mg_params_log"])
            )

    diagnosis: dict[str, Any] | None = None
    if artifacts["diagnose"].is_file():
        try:
            diagnosis_payload = _read_json(artifacts["diagnose"])
            hard_flags = diagnosis_payload.get("hard_flags")
            hard_flags = hard_flags if isinstance(hard_flags, dict) else {}
            diagnosis = {
                "diagnostic": diagnosis_payload.get("diagnostic"),
                "encoder_collapsed": hard_flags.get("encoder_collapsed"),
                "latent_map_overcontracted": hard_flags.get("latent_map_overcontracted"),
            }
            if diagnosis["diagnostic"] not in {
                "ok",
                "encoder_collapsed",
                "latent_map_overcontracted",
                "encoder_collapsed_and_latent_overcontracted",
            }:
                issues.append(
                    _issue(
                        scope,
                        "invalid_diagnosis",
                        f"unexpected diagnostic status {diagnosis['diagnostic']!r}",
                        artifacts["diagnose"],
                    )
                )
            for name in ("encoder_collapsed", "latent_map_overcontracted"):
                if not isinstance(diagnosis[name], bool):
                    issues.append(
                        _issue(
                            scope,
                            "invalid_diagnosis",
                            f"hard flag {name!r} is missing or not boolean",
                            artifacts["diagnose"],
                        )
                    )
        except ValueError as exc:
            issues.append(_issue(scope, "invalid_diagnosis", str(exc), artifacts["diagnose"]))

    component_metrics: dict[str, Any] | None = None
    if artifacts["metrics"].is_file():
        try:
            metrics_payload = _read_json(artifacts["metrics"])
            raw_components = metrics_payload.get("minimal_morse_sets")
            if not isinstance(raw_components, dict):
                raise ValueError("metrics.minimal_morse_sets must be an object")
            component_metrics = {}
            for label, raw_block in raw_components.items():
                block = raw_block if isinstance(raw_block, dict) else {}
                tau = _as_float(block.get("tau_bar"))
                residual = _as_float(block.get("max_semiconjugacy_error"))
                component_metrics[str(label)] = {
                    "n_boxes": _as_int(block.get("n_boxes")),
                    "tau_bar": tau,
                    "n_semiconjugacy_samples": _as_int(block.get("n_semiconjugacy_samples")),
                    "max_semiconjugacy_error": residual,
                    # A sampled inequality diagnostic only. It is deliberately
                    # not treated as a classifier of component correspondence.
                    "sampled_inequality_pass": (
                        residual <= tau if residual is not None and tau is not None else None
                    ),
                }
        except ValueError as exc:
            issues.append(_issue(scope, "invalid_metrics", str(exc), artifacts["metrics"]))

    if training is not None:
        row.update(
            {
                "training_method": training["training_method"],
                "training_seed": training["seed"],
                "epochs_completed": training["epochs_completed"],
                "first_order_epochs_completed": training["first_order_epochs_completed"],
                "n_training_pairs": training["n_training_pairs"],
                "n_holdout_pairs": training["n_holdout_pairs"],
                "full_batch": training["full_batch"],
                "train_duration_seconds": training["train_duration_seconds"],
            }
        )
        first_order = training["optimizer"]["first_order"]
        polish = training["optimizer"]["polish"]
        row.update(
            {
                "optimizer_sequence": _json_text(training["optimizer"]["sequence"]),
                "adamw_betas": _json_text(first_order["betas"]),
                "adamw_eps": first_order["eps"],
                "adamw_weight_decay": first_order["weight_decay"],
                "adamw_amsgrad": first_order["amsgrad"],
                "adamw_foreach": first_order["foreach"],
                "adamw_fused": first_order["fused"],
                "adamw_updates_completed": first_order["updates_completed"],
                "adamw_device": first_order["device"],
                "adamw_dtype": first_order["dtype"],
                "lbfgs_device": polish["device"],
                "lbfgs_dtype": polish["dtype"],
                "lbfgs_outer_steps_requested": polish["outer_steps_requested"],
                "lbfgs_outer_steps_completed": polish["outer_steps_completed"],
                "lbfgs_learning_rate": polish["learning_rate"],
                "lbfgs_max_iter": polish["max_iter"],
                "lbfgs_max_eval": polish["max_eval"],
                "lbfgs_history_size": polish["history_size"],
                "lbfgs_tolerance_grad": polish["tolerance_grad"],
                "lbfgs_tolerance_change": polish["tolerance_change"],
                "lbfgs_line_search_fn": polish["line_search_fn"],
                "lbfgs_internal_iterations": polish["internal_iterations"],
                "lbfgs_closure_evaluations": polish["closure_evaluations"],
                "checkpoint_selection": training["checkpoint_selection"],
                "checkpoint_source": training["checkpoint_source"],
            }
        )
        for stage in training["stages"]:
            number = stage["index"]
            row[f"stage{number}_name"] = stage["name"]
            row[f"stage{number}_learning_rate"] = stage["learning_rate"]
            row[f"stage{number}_loss_weights"] = _json_text(stage["loss_weights"])
            row[f"stage{number}_trainable_components"] = _json_text(stage["trainable_components"])
            for endpoint in ("train", "holdout"):
                for key in LOSS_KEYS:
                    row[f"stage{number}_{endpoint}_{LOSS_SHORT[key]}"] = stage[endpoint][key]
        for endpoint, values in (
            ("train", training["adamw_train"]),
            ("holdout", training["adamw_holdout"]),
        ):
            for key in LOSS_KEYS:
                row[f"adamw_{endpoint}_{LOSS_SHORT[key]}"] = values[key]
        for endpoint, values in (
            ("train", training["final_train"]),
            ("holdout", training["final_holdout"]),
        ):
            for key in LOSS_KEYS:
                row[f"final_{endpoint}_{LOSS_SHORT[key]}"] = values[key]
                adamw_value = training[f"adamw_{endpoint}"][key]
                final_value = values[key]
                row[f"lbfgs_delta_{endpoint}_{LOSS_SHORT[key]}"] = (
                    final_value - adamw_value
                    if final_value is not None and adamw_value is not None
                    else None
                )

    if graph is not None:
        for field in (
            "n_morse_nodes",
            "n_morse_edges",
            "n_minimal_nodes",
            "n_attractor_type_nodes",
            "n_periodic_attractor_nodes",
            "bistability_pass",
            "exact_period4_bistability_pass",
        ):
            row[field] = graph[field]
        row["sink_nodes"] = _json_text(graph["sink_nodes"])
        row["sink_conley_indices"] = _json_text(graph["sink_conley_indices"])
    if mg_params is not None:
        row.update(
            {
                "latent_lower_bounds": _json_text(mg_params["lower_bounds"]),
                "latent_upper_bounds": _json_text(mg_params["upper_bounds"]),
                "latent_bounds_source": mg_params["bounds_source"],
                "cmgdb_subdiv_init": mg_params["subdiv_init"],
                "cmgdb_subdiv_min": mg_params["subdiv_min"],
                "cmgdb_subdiv_max": mg_params["subdiv_max"],
                "cmgdb_subdiv_limit": mg_params["subdiv_limit"],
                "box_map_backend": mg_params["box_map_backend"],
                "bounds_data_role": mg_params["bounds_data_role"],
                "adaptive_precompute_subdiv": mg_params["adaptive_precompute_subdiv"],
            }
        )
    if diagnosis is not None:
        row.update(diagnosis)
    if component_metrics is not None:
        sampled = [
            block["sampled_inequality_pass"]
            for block in component_metrics.values()
            if isinstance(block["sampled_inequality_pass"], bool)
        ]
        row.update(
            {
                "n_metric_minimal_components": len(component_metrics),
                "n_sampled_tolerance_evaluable": len(sampled),
                "n_sampled_tolerance_pass": sum(value is True for value in sampled),
                "minimal_component_metrics": _json_text(component_metrics),
            }
        )
    row.update(
        {
            "sweep_reported_n_morse_nodes": record.get("n_morse_nodes"),
            "sweep_reported_n_morse_edges": record.get("n_morse_edges"),
            "sweep_reported_n_sinks": record.get("n_sinks"),
            "sweep_reported_bistability_pass": record.get("bistability_pass"),
            "training_summary_path": _display(artifacts["training_summary"]),
            "checkpoint_path": _display(artifacts["checkpoint"]),
            "checkpoint_metadata_path": _display(artifacts["checkpoint_metadata"]),
            "adamw_checkpoint_path": _display(artifacts["adamw_checkpoint"]),
            "adamw_checkpoint_metadata_path": _display(artifacts["adamw_checkpoint_metadata"]),
            "history_path": _display(artifacts["history"]),
            "morse_graph_path": _display(artifacts["morse_graph"]),
            "morse_sets_path": _display(artifacts["morse_sets"]),
            "mg_params_log_path": _display(artifacts["mg_params_log"]),
            "metrics_path": _display(artifacts["metrics"]),
            "diagnose_path": _display(artifacts["diagnose"]),
            "run_manifest_path": _display(artifacts["run_manifest"]),
            "artifact_paths": _json_text(
                {name: _display(path) for name, path in artifacts.items()}
            ),
            "training_summary_sha256": _sha256(artifacts["training_summary"]),
            "checkpoint_sha256": _sha256(artifacts["checkpoint"]),
            "adamw_checkpoint_sha256": _sha256(artifacts["adamw_checkpoint"]),
            "morse_graph_sha256": _sha256(artifacts["morse_graph"]),
            "artifact_sha256": _json_text(
                {name: _sha256(path) for name, path in artifacts.items()}
            ),
            "artifact_exists": _json_text(
                {
                    name: path.is_file() and path.stat().st_size > 0
                    for name, path in artifacts.items()
                }
            ),
        }
    )
    row["training_contract_valid"] = not training_errors
    row["cell_status"] = "complete" if not issues else "invalid"
    row["validation_errors"] = _json_text([issue["message"] for issue in issues])
    return row, issues


def _numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if _as_float(row.get(field)) is not None]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "population_std": statistics.pstdev(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _aggregate(
    *,
    rows: list[dict[str, Any]],
    sweep_root: Path,
    sweep_path: Path,
    sweep_payload: dict[str, Any],
    issues: list[dict[str, str]],
    provisional: bool,
) -> dict[str, Any]:
    stage_endpoints: dict[str, Any] = {}
    for stage in range(1, 4):
        stage_endpoints[str(stage)] = {
            endpoint: {
                LOSS_SHORT[key]: _numeric_summary(
                    rows, f"stage{stage}_{endpoint}_{LOSS_SHORT[key]}"
                )
                for key in LOSS_KEYS
            }
            for endpoint in ("train", "holdout")
        }
    adamw_losses = {
        endpoint: {
            LOSS_SHORT[key]: _numeric_summary(rows, f"adamw_{endpoint}_{LOSS_SHORT[key]}")
            for key in LOSS_KEYS
        }
        for endpoint in ("train", "holdout")
    }
    final_losses = {
        endpoint: {
            LOSS_SHORT[key]: _numeric_summary(rows, f"final_{endpoint}_{LOSS_SHORT[key]}")
            for key in LOSS_KEYS
        }
        for endpoint in ("train", "holdout")
    }
    polish_deltas = {
        endpoint: {
            LOSS_SHORT[key]: _numeric_summary(rows, f"lbfgs_delta_{endpoint}_{LOSS_SHORT[key]}")
            for key in LOSS_KEYS
        }
        for endpoint in ("train", "holdout")
    }
    valid_topology = [row for row in rows if _as_int(row.get("n_morse_nodes")) is not None]
    sink_index_profiles = Counter(row["sink_conley_indices"] for row in valid_topology)
    pass_rows = [row for row in valid_topology if isinstance(row.get("bistability_pass"), bool)]
    diagnosed = [row for row in rows if isinstance(row.get("diagnostic"), str)]
    return {
        "schema_version": 2,
        "generated_at_utc": _utc_now(),
        "provisional": provisional,
        "source_is_read_only": True,
        "sweep_root": _display(sweep_root),
        "source_sweep_summary": {
            "path": _display(sweep_path),
            "sha256": _sha256(sweep_path),
            "example": sweep_payload.get("example"),
            "tag": sweep_payload.get("tag"),
            "data_size": sweep_payload.get("data_size"),
            "cmgdb_subdiv": sweep_payload.get("cmgdb_subdiv"),
            "box_map_backend": sweep_payload.get("box_map_backend"),
            "pass_criterion": sweep_payload.get("pass_criterion"),
        },
        "expected_design": {
            "data_seeds": list(DATA_SEEDS),
            "model_seeds": list(MODEL_SEEDS),
            "n_cells": len(EXPECTED_CELLS),
        },
        "training_contract": {
            "training_method": "curriculum_full_batch",
            "first_order_epochs_total": EXPECTED_TOTAL_EPOCHS,
            "stages": [
                {"epochs": EXPECTED_STAGE_EPOCHS, "loss_weights": list(weights)}
                for weights in EXPECTED_STAGE_WEIGHTS
            ],
            "full_batch": True,
            "n_training_pairs": EXPECTED_TRAINING_PAIRS,
            "n_holdout_pairs": EXPECTED_HOLDOUT_PAIRS,
            "scheduler": None,
            "scheduler_used": False,
            "early_stopping_used": False,
            "patience_used": False,
            "optimizer": {
                "sequence": ["AdamW", "LBFGS"],
                "first_order": {
                    "stage_learning_rates": [EXPECTED_LEARNING_RATE] * 3,
                    "betas": list(EXPECTED_ADAMW_BETAS),
                    "eps": EXPECTED_ADAMW_EPS,
                    "weight_decay": EXPECTED_ADAMW_WEIGHT_DECAY,
                    "state_continues_across_stages": True,
                },
                "polish": {
                    "device": "cpu",
                    "dtype": "float64",
                    "outer_steps": EXPECTED_LBFGS_OUTER_STEPS,
                    "learning_rate": EXPECTED_LBFGS_LEARNING_RATE,
                    "max_iter_per_outer_step": EXPECTED_LBFGS_MAX_ITER,
                    "max_eval_per_outer_step": EXPECTED_LBFGS_MAX_EVAL,
                    "history_size": EXPECTED_LBFGS_HISTORY_SIZE,
                    "tolerance_grad": EXPECTED_LBFGS_TOLERANCE_GRAD,
                    "tolerance_change": EXPECTED_LBFGS_TOLERANCE_CHANGE,
                    "line_search_fn": "strong_wolfe",
                    "fresh_optimizer_state": True,
                },
            },
            "checkpoint_selection": "final_lbfgs_float32_endpoint",
            "checkpoint_source": "lbfgs_float32_endpoint",
        },
        "inventory": {
            "n_expected_cells": len(EXPECTED_CELLS),
            "n_complete_cells": sum(row["cell_status"] == "complete" for row in rows),
            "n_invalid_cells": sum(row["cell_status"] == "invalid" for row in rows),
            "n_missing_cells": sum(row["cell_status"] == "missing" for row in rows),
            "status_counts": dict(sorted(Counter(row["cell_status"] for row in rows).items())),
            "n_issues": len(issues),
            "issue_counts_by_code": dict(
                sorted(Counter(issue["code"] for issue in issues).items())
            ),
            "issues": issues,
        },
        "losses": {
            "stage_endpoints_post_update": stage_endpoints,
            "adamw_endpoint": adamw_losses,
            "final_checkpoint": final_losses,
            "lbfgs_delta_final_minus_adamw": polish_deltas,
        },
        "optimizer_accounting": {
            "lbfgs_internal_iterations": _numeric_summary(rows, "lbfgs_internal_iterations"),
            "lbfgs_closure_evaluations": _numeric_summary(rows, "lbfgs_closure_evaluations"),
            "lbfgs_train_total_nonincrease": {
                "n_evaluated": sum(
                    _as_float(row.get("adamw_train_total")) is not None
                    and _as_float(row.get("final_train_total")) is not None
                    for row in rows
                ),
                "n_pass": sum(
                    _as_float(row.get("adamw_train_total")) is not None
                    and _as_float(row.get("final_train_total")) is not None
                    and (
                        float(row["final_train_total"]) <= float(row["adamw_train_total"])
                        or math.isclose(
                            float(row["final_train_total"]),
                            float(row["adamw_train_total"]),
                            rel_tol=1e-6,
                            abs_tol=1e-9,
                        )
                    )
                    for row in rows
                ),
            },
        },
        "topology": {
            "n_cells_evaluated": len(valid_topology),
            "morse_node_count_distribution": dict(
                sorted(Counter(row["n_morse_nodes"] for row in valid_topology).items())
            ),
            "morse_edge_count_distribution": dict(
                sorted(Counter(row["n_morse_edges"] for row in valid_topology).items())
            ),
            "minimal_node_count_distribution": dict(
                sorted(Counter(row["n_minimal_nodes"] for row in valid_topology).items())
            ),
            "sink_conley_index_profiles": dict(sorted(sink_index_profiles.items())),
            "bistability_pass": {
                "n_evaluated": len(pass_rows),
                "n_pass": sum(row["bistability_pass"] is True for row in pass_rows),
                "n_fail": sum(row["bistability_pass"] is False for row in pass_rows),
                "rate_among_evaluated": (
                    sum(row["bistability_pass"] is True for row in pass_rows) / len(pass_rows)
                    if pass_rows
                    else None
                ),
            },
            "exact_period4_bistability_pass": {
                "n_evaluated": len(valid_topology),
                "n_pass": sum(
                    row["exact_period4_bistability_pass"] is True for row in valid_topology
                ),
                "n_fail": sum(
                    row["exact_period4_bistability_pass"] is False for row in valid_topology
                ),
            },
        },
        "diagnostics": {
            "n_cells_evaluated": len(diagnosed),
            "status_distribution": dict(
                sorted(Counter(row["diagnostic"] for row in diagnosed).items())
            ),
            "n_encoder_collapsed": sum(row["encoder_collapsed"] is True for row in diagnosed),
            "n_latent_map_overcontracted": sum(
                row["latent_map_overcontracted"] is True for row in diagnosed
            ),
        },
        "sampled_tolerance_diagnostic": {
            "interpretation": (
                "Counts only the saved sampled residual-versus-tau inequality; "
                "it is not a classifier of spuriousness or invariant-set correspondence."
            ),
            "n_minimal_components": sum(
                _as_int(row.get("n_metric_minimal_components")) or 0 for row in rows
            ),
            "n_evaluable": sum(
                _as_int(row.get("n_sampled_tolerance_evaluable")) or 0 for row in rows
            ),
            "n_pass": sum(_as_int(row.get("n_sampled_tolerance_pass")) or 0 for row in rows),
        },
        "derived_artifacts": {
            "cells_csv": "summary/cells.csv",
            "markdown": "summary/SUMMARY.md",
        },
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    number = _as_float(value)
    if number is not None:
        return f"{number:.5e}"
    return "—" if value is None else str(value)


def _markdown(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    inventory = aggregate["inventory"]
    status = (
        "PROVISIONAL — incomplete or invalid source cells"
        if aggregate["provisional"]
        else "COMPLETE — all 15 cells satisfy the reporting contract"
    )
    lines = [
        "# Leslie3D ground-box curriculum 3x5 summary",
        "",
        f"**Status:** {status}",
        "",
        f"Source sweep: `{aggregate['source_sweep_summary']['path']}` (read-only). Derived files are confined to `summary/`.",
        "",
        "The fixed design is five data seeds `2158, 4792, 3174, 688, 5727` by model seeds `0, 1, 2`. "
        "Each run must use one continuous full-batch AdamW optimizer for three 4,000-update stages with weights "
        "`[1,0,0]`, `[1,1,0]`, and `[1,1,1]`, followed by a fresh 12-step CPU float64 L-BFGS polish of the final joint objective. "
        "The saved checkpoint is the float32-cast L-BFGS endpoint; it is not selected by validation and is not described as an epoch. "
        "There is no scheduler, patience, or early stopping. Here L1 is reconstruction, L2 one-step prediction, and L3 semiconjugacy. "
        "Stage endpoint losses are post-update raw terms; `total` is the stage-weighted sum.",
        "",
        "## Inventory and topology",
        "",
        f"Complete: {inventory['n_complete_cells']}/15; invalid: {inventory['n_invalid_cells']}; missing: {inventory['n_missing_cells']}; validation issues: {inventory['n_issues']}.",
        "",
        "| data seed | model seed | status | diagnosis | nodes | edges | minimal | sink indices | periodic pass | period-4 pass | output directory |",
        "|---:|---:|:---|:---|---:|---:|---:|:---|:---:|:---:|:---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['data_seed']} | {row['model_seed']} | {row['cell_status']} | "
            f"{_fmt(row['diagnostic'])} | "
            f"{_fmt(row['n_morse_nodes'])} | {_fmt(row['n_morse_edges'])} | {_fmt(row['n_minimal_nodes'])} | "
            f"`{row['sink_conley_indices'] or '—'}` | {_fmt(row['bistability_pass'])} | "
            f"{_fmt(row['exact_period4_bistability_pass'])} | `{row['cell_directory'] or '—'}` |"
        )

    for stage in range(1, 4):
        lines.extend(
            [
                "",
                f"## Stage {stage} endpoint losses",
                "",
                "| cell | train L1 | train L2 | train L3 | train total | holdout L1 | holdout L2 | holdout L3 | holdout total |",
                "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            values = [
                row[f"stage{stage}_{endpoint}_{name}"]
                for endpoint in ("train", "holdout")
                for name in ("l1", "l2", "l3", "total")
            ]
            lines.append(
                f"| {row['data_seed']}/{row['model_seed']} | "
                + " | ".join(_fmt(value) for value in values)
                + " |"
            )

    lines.extend(
        [
            "",
            "## AdamW-to-L-BFGS polish",
            "",
            "Deltas are `final float32 checkpoint - AdamW endpoint`; a negative training delta is an improvement. Holdout deltas are reported but were not used for optimization or checkpoint selection.",
            "",
            "| cell | AdamW train total | final train total | train delta | AdamW holdout total | final holdout total | holdout delta | L-BFGS iterations | closure evaluations |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        values = (
            row["adamw_train_total"],
            row["final_train_total"],
            row["lbfgs_delta_train_total"],
            row["adamw_holdout_total"],
            row["final_holdout_total"],
            row["lbfgs_delta_holdout_total"],
            row["lbfgs_internal_iterations"],
            row["lbfgs_closure_evaluations"],
        )
        lines.append(
            f"| {row['data_seed']}/{row['model_seed']} | "
            + " | ".join(_fmt(value) for value in values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Final float32-checkpoint losses",
            "",
            "| cell | train L1 | train L2 | train L3 | train total | holdout L1 | holdout L2 | holdout L3 | holdout total |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        values = [
            row[f"final_{endpoint}_{name}"]
            for endpoint in ("train", "holdout")
            for name in ("l1", "l2", "l3", "total")
        ]
        lines.append(
            f"| {row['data_seed']}/{row['model_seed']} | "
            + " | ".join(_fmt(value) for value in values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Artifact provenance",
            "",
            "Exact per-cell final and AdamW-endpoint checkpoint paths, training-summary, history, Morse-graph, Morse-set, metric, diagnosis, and manifest paths plus SHA-256 hashes are in `cells.csv`. The CSV also records the frozen optimizer settings, actual L-BFGS iterations and closure evaluations, latent CMGDB bounds, and the raw per-component sampled residual/tolerance values. The sampled inequality is a diagnostic, not a classifier of spuriousness or invariant-set correspondence. The saved Morse graph, rather than a manifest default, is the authoritative source for node, edge, sink, index, and pass fields.",
        ]
    )
    if aggregate["inventory"]["issues"]:
        lines.extend(["", "## Validation issues", ""])
        for issue in aggregate["inventory"]["issues"]:
            where = f" (`{issue['path']}`)" if "path" in issue else ""
            lines.append(f"- `{issue['scope']}` / `{issue['code']}`: {issue['message']}{where}")
    return "\n".join(lines) + "\n"


def _write_reports(
    summary_dir: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any]
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    (summary_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (summary_dir / "SUMMARY.md").write_text(_markdown(rows, aggregate), encoding="utf-8")


def _records_by_cell(
    payload: dict[str, Any], *, sweep_path: Path
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    for key, expected in (("ic_seeds", list(DATA_SEEDS)), ("model_seeds", list(MODEL_SEEDS))):
        if key in payload and payload[key] != expected:
            issues.append(
                _issue(
                    "sweep",
                    "design_mismatch",
                    f"{key}: expected {expected!r}, found {payload[key]!r}",
                    sweep_path,
                )
            )
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list):
        issues.append(
            _issue("sweep", "invalid_cells", "sweep_summary.cells must be a list", sweep_path)
        )
        return {}, issues
    records: dict[tuple[int, int], dict[str, Any]] = {}
    expected = set(EXPECTED_CELLS)
    for offset, record in enumerate(raw_cells):
        if not isinstance(record, dict):
            issues.append(
                _issue("sweep", "invalid_cell", f"cells[{offset}] is not an object", sweep_path)
            )
            continue
        data_seed = _as_int(record.get("ic_seed", record.get("data_seed")))
        model_seed = _as_int(record.get("model_seed", record.get("seed")))
        key = (data_seed, model_seed)
        if data_seed is None or model_seed is None:
            issues.append(
                _issue(
                    "sweep",
                    "invalid_cell_key",
                    f"cells[{offset}] has no integer data/model seed",
                    sweep_path,
                )
            )
        elif key not in expected:
            issues.append(
                _issue(
                    "sweep",
                    "unexpected_cell",
                    f"cells[{offset}] has unexpected key {key}",
                    sweep_path,
                )
            )
        elif key in records:
            issues.append(
                _issue("sweep", "duplicate_cell", f"duplicate cell key {key}", sweep_path)
            )
        else:
            records[key] = record
    return records, issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=DEFAULT_SWEEP_ROOT,
        help=f"source sweep root (default: {DEFAULT_SWEEP_ROOT})",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write an explicitly provisional report despite missing or invalid cells",
    )
    args = parser.parse_args(argv)

    sweep_root = args.sweep_root.expanduser().resolve()
    sweep_path = sweep_root / "sweep_summary.json"
    global_issues: list[dict[str, str]] = []
    if sweep_path.is_file():
        try:
            sweep_payload = _read_json(sweep_path)
        except ValueError as exc:
            sweep_payload = {}
            global_issues.append(_issue("sweep", "invalid_sweep_summary", str(exc), sweep_path))
    else:
        sweep_payload = {}
        global_issues.append(
            _issue("sweep", "missing_sweep_summary", "sweep_summary.json is absent", sweep_path)
        )

    records, record_issues = _records_by_cell(sweep_payload, sweep_path=sweep_path)
    issues = global_issues + record_issues
    rows: list[dict[str, Any]] = []
    for data_seed, model_seed in EXPECTED_CELLS:
        row, cell_issues = _row_for_cell(
            record=records.get((data_seed, model_seed)),
            sweep_root=sweep_root,
            data_seed=data_seed,
            model_seed=model_seed,
        )
        rows.append(row)
        issues.extend(cell_issues)

    provisional = bool(issues)
    aggregate = _aggregate(
        rows=rows,
        sweep_root=sweep_root,
        sweep_path=sweep_path,
        sweep_payload=sweep_payload,
        issues=issues,
        provisional=provisional,
    )
    if provisional and not args.allow_incomplete:
        counts = Counter(issue["code"] for issue in issues)
        details = ", ".join(f"{code}={count}" for code, count in sorted(counts.items()))
        raise SweepValidationError(
            f"strict summary refused: {len(issues)} validation issue(s) ({details}). "
            "No report was written; use --allow-incomplete for a provisional audit."
        )

    summary_dir = sweep_root / "summary"
    _write_reports(summary_dir, rows, aggregate)
    print(
        f"wrote {'provisional' if provisional else 'complete'} summary to {summary_dir} "
        f"({aggregate['inventory']['n_complete_cells']}/15 complete cells)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SweepValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
