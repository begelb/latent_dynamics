"""Run an isolated, resumable sweep of reference-faithful Chafee d=1 fits.

This is an exploratory training driver only.  It deliberately does not run
CMGDB or inspect any region-of-attraction statistic.  Every candidate uses the
exact archived 30,000 unscaled pairs, the canonical d=1 architecture, and
``train_reference_full_batch``.  Consequently, one epoch is one direct
full-batch Adam update with the coauthor reference decoded two-term objective.

The default first-launch matrix is a modest eight-seed replication at the
canonical 4,000 epochs and learning rate 0.003.  A different predeclared
matrix can be supplied as JSON before the first launch::

    {
      "schema_version": 1,
      "runs": [
        {"run_id": "seed_08_lr3e3_e4000", "seed": 8},
        {
          "run_id": "seed_09_lr1e3_e10000",
          "seed": 9,
          "epochs": 10000,
          "learning_rate": 0.001
        }
      ]
    }

The first invocation freezes the complete matrix, device, architecture, input
hashes, and implementation hashes in ``sweep_plan.json``.  Later invocations
resume that frozen plan: verified completed runs are skipped and failed or
interrupted runs receive a new immutable attempt directory.  No checkpoint or
history is overwritten.  Resumption is therefore at run boundaries rather
than within an individual 4,000-epoch fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from latentdynamics.config import ArchConfig
from latentdynamics.training import load_checkpoint, train_reference_full_batch

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = (
    CODE_ROOT / "replay_sources" / "chafee_infante" / "reference_inputs"
)
TRAIN_DATA = DEFAULT_REFERENCE_ROOT / "train_data.csv"
LATENT_1D_OUTPUT_ROOT = (
    CODE_ROOT / "output" / "chafee_latent_dimension_study" / "latent_1d"
)


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


CANONICAL_RUN = _first_existing(
    CODE_ROOT
    / "replay_sources"
    / "chafee_infante"
    / "latent_dimension_study"
    / "latent_1d"
    / "seed_0",
    LATENT_1D_OUTPUT_ROOT / "seed_0",
)
DEFAULT_OUTPUT = LATENT_1D_OUTPUT_ROOT / "exploratory_full_batch_seed_sweep_v1"
REFERENCE_IMPLEMENTATION = (
    CODE_ROOT / "src" / "latentdynamics" / "training" / "reference_recipe.py"
)
RUNNER_IMPLEMENTATION = Path(__file__).resolve()

TRAIN_DATA_SHA256 = "890fea29a2d26b31b44bbc4fbd773af0ebf742b08893c27ddaf1bdbf87e30329"
CANONICAL_CHECKPOINT_SHA256 = (
    "f2d1ad7dcc094e4565f25446e613d4b528261012810bb493ef70d1a3977c0f91"
)
CANONICAL_SIDECAR_SHA256 = (
    "da59bea7227de80b8ee94a09c695b5447e472e69c72038693fa63e6720e97798"
)

HIGH_DIMENSION = 64
TRAINING_ROWS = 30_000
PLAN_SCHEMA_VERSION = 1
MATRIX_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
ATTEMPT_DIRECTORY = re.compile(r"^attempt_(\d{3,})$")
HISTORY_KEYS = (
    "loss_reconstruction",
    "loss_prediction",
    "loss_total",
    "learning_rate",
)


@dataclass(frozen=True)
class FullBatchRunSpec:
    """One predeclared reference-faithful full-batch candidate."""

    run_id: str
    seed: int
    epochs: int = 4_000
    learning_rate: float = 0.003
    scheduler_factor: float = 0.5
    scheduler_patience: int = 100
    scheduler_threshold: float = 1e-4
    scheduler_min_lr: float = 1e-6

    def __post_init__(self) -> None:
        if (
            not SAFE_RUN_ID.fullmatch(self.run_id)
            or self.run_id in {".", ".."}
            or Path(self.run_id).name != self.run_id
        ):
            raise ValueError(
                "run_id must be a safe lowercase file-name token of at most "
                "96 characters"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs < 1
        ):
            raise ValueError("epochs must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must lie strictly between 0 and 1")
        if (
            isinstance(self.scheduler_patience, bool)
            or not isinstance(self.scheduler_patience, int)
            or self.scheduler_patience < 0
        ):
            raise ValueError(
                "scheduler_patience must be a non-negative integer"
            )
        if (
            not math.isfinite(self.scheduler_threshold)
            or self.scheduler_threshold < 0
        ):
            raise ValueError(
                "scheduler_threshold must be non-negative and finite"
            )
        if (
            not math.isfinite(self.scheduler_min_lr)
            or self.scheduler_min_lr < 0
            or self.scheduler_min_lr > self.learning_rate
        ):
            raise ValueError(
                "scheduler_min_lr must be finite and lie in "
                "[0, learning_rate]"
            )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FullBatchRunSpec:
        """Parse a matrix row while allowing documented defaults."""

        if not isinstance(payload, dict):
            raise ValueError("each matrix run must be a JSON object")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown run-spec keys: {sorted(unknown)}")
        missing = {"run_id", "seed"} - set(payload)
        if missing:
            raise ValueError(f"missing run-spec keys: {sorted(missing)}")
        return cls(**payload)


# This tuple is used only when creating a new sweep without --matrix-json.
# Existing sweeps always use their persisted frozen plan, so this can be edited
# safely for a future sweep rooted at a new --output-dir.
DEFAULT_RUNS: tuple[FullBatchRunSpec, ...] = tuple(
    FullBatchRunSpec(
        run_id=f"seed_{seed:02d}_lr3e3_e4000",
        seed=seed,
    )
    for seed in range(8)
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _checked_source(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = _sha256(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for {resolved}: expected {expected_sha256}, "
            f"observed {digest}"
        )
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": resolved.stat().st_size,
    }


def _current_source_provenance(
    train_data: Path = TRAIN_DATA,
) -> dict[str, dict[str, Any]]:
    resolved_train_data = train_data.resolve()
    expected_train_hash = (
        TRAIN_DATA_SHA256
        if resolved_train_data == TRAIN_DATA.resolve()
        else None
    )
    model_dir = CANONICAL_RUN / "models"
    return {
        "train_data": _checked_source(
            resolved_train_data,
            expected_sha256=expected_train_hash,
        ),
        "canonical_checkpoint": _checked_source(
            model_dir / "autoencoder.pt",
            expected_sha256=CANONICAL_CHECKPOINT_SHA256,
        ),
        "canonical_architecture_sidecar": _checked_source(
            model_dir / "autoencoder.json",
            expected_sha256=CANONICAL_SIDECAR_SHA256,
        ),
        "reference_training_implementation": _checked_source(
            REFERENCE_IMPLEMENTATION,
        ),
        "sweep_runner_implementation": _checked_source(
            RUNNER_IMPLEMENTATION,
        ),
    }


def _validate_plan_sources(plan: dict[str, Any]) -> None:
    sources = plan.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("frozen sweep plan has no source provenance")
    for name, source in sources.items():
        if not isinstance(source, dict):
            raise ValueError(f"malformed frozen source {name!r}")
        try:
            path = Path(source["path"])
            expected_hash = str(source["sha256"])
            expected_size = int(source["size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed frozen source {name!r}") from exc
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or _sha256(path) != expected_hash
        ):
            raise ValueError(
                f"frozen source {name!r} changed or is missing: {path}"
            )


def _load_canonical_arch() -> ArchConfig:
    _, arch = load_checkpoint(CANONICAL_RUN / "models", map_location="cpu")
    if arch.high_dims != HIGH_DIMENSION or arch.low_dims != 1:
        raise ValueError(
            "canonical checkpoint architecture is not the required 64->1 model"
        )
    return arch


def _load_training_pairs(
    train_data: Path = TRAIN_DATA,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    source = train_data.resolve()
    pairs = np.loadtxt(source, delimiter=",", dtype=np.float64)
    expected = (TRAINING_ROWS, 2 * HIGH_DIMENSION)
    if pairs.shape != expected:
        raise ValueError(f"{source} has shape {pairs.shape}; expected {expected}")
    if not np.all(np.isfinite(pairs)):
        raise ValueError(f"{source} contains non-finite values")
    return (
        np.ascontiguousarray(pairs[:, :HIGH_DIMENSION]),
        np.ascontiguousarray(pairs[:, HIGH_DIMENSION:]),
    )


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_unique_specs(
    specs: Sequence[FullBatchRunSpec],
) -> tuple[FullBatchRunSpec, ...]:
    frozen = tuple(specs)
    if not frozen:
        raise ValueError("sweep matrix must contain at least one run")
    run_ids = [spec.run_id for spec in frozen]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("sweep run_id values must be unique")
    return frozen


def _load_matrix(path: Path) -> tuple[FullBatchRunSpec, ...]:
    payload = _read_json(path)
    if set(payload) != {"schema_version", "runs"}:
        raise ValueError(
            "matrix JSON must contain exactly 'schema_version' and 'runs'"
        )
    if payload["schema_version"] != MATRIX_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported matrix schema {payload['schema_version']!r}"
        )
    rows = payload["runs"]
    if not isinstance(rows, list):
        raise ValueError("matrix 'runs' must be a JSON list")
    return _validate_unique_specs(
        tuple(FullBatchRunSpec.from_payload(row) for row in rows)
    )


def _build_plan(
    *,
    specs: Sequence[FullBatchRunSpec],
    device: torch.device,
    arch: ArchConfig,
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    specs = _validate_unique_specs(specs)
    return {
        "purpose": (
            "exploratory direct-full-batch Chafee d=1 training sweep; "
            "RoA analysis is intentionally separate"
        ),
        "training_entrypoint": (
            "latentdynamics.training.train_reference_full_batch"
        ),
        "training_semantics": {
            "data_rows": TRAINING_ROWS,
            "high_dimension": HIGH_DIMENSION,
            "latent_dimension": 1,
            "dtype": "float32",
            "full_batch": True,
            "optimizer": {
                "name": "Adam",
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
            "objective": "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)",
            "validation_used": False,
            "early_stopping_used": False,
            "checkpoint_selection": "fixed final epoch",
        },
        "resolved_device": str(device),
        "architecture": arch.model_dump(mode="json"),
        "sources": sources,
        "runs": [asdict(spec) for spec in specs],
    }


def _plan_envelope(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_sha256": _payload_sha256(plan),
        "plan": plan,
    }


def _validate_plan_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"schema_version", "plan_sha256", "plan"}:
        raise ValueError("malformed sweep plan envelope")
    if payload["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported sweep plan schema {payload['schema_version']!r}"
        )
    plan = payload["plan"]
    if not isinstance(plan, dict):
        raise ValueError("sweep plan body must be a JSON object")
    observed = _payload_sha256(plan)
    if payload["plan_sha256"] != observed:
        raise ValueError(
            "sweep plan hash mismatch; refusing to run a modified plan"
        )
    return plan


def _plan_specs(plan: dict[str, Any]) -> tuple[FullBatchRunSpec, ...]:
    rows = plan.get("runs")
    if not isinstance(rows, list):
        raise ValueError("frozen sweep plan has no run list")
    return _validate_unique_specs(
        tuple(FullBatchRunSpec.from_payload(row) for row in rows)
    )


def _create_or_load_plan(
    *,
    output_dir: Path,
    device_name: str | None,
    requested_specs: Sequence[FullBatchRunSpec] | None,
    train_data: Path | None,
) -> tuple[dict[str, Any], str, tuple[FullBatchRunSpec, ...], ArchConfig]:
    plan_path = output_dir / "sweep_plan.json"
    if plan_path.exists():
        plan = _validate_plan_envelope(_read_json(plan_path))
        _validate_plan_sources(plan)
        frozen_specs = _plan_specs(plan)
        if requested_specs is not None:
            requested = _validate_unique_specs(requested_specs)
            if [asdict(item) for item in requested] != plan["runs"]:
                raise ValueError(
                    "requested matrix differs from the frozen sweep plan"
                )
        if device_name is not None:
            requested_device = str(_resolve_device(device_name))
            if requested_device != plan.get("resolved_device"):
                raise ValueError(
                    f"requested device {requested_device!r} differs from frozen "
                    f"device {plan.get('resolved_device')!r}"
                )
        if train_data is not None:
            frozen_train_data = Path(plan["sources"]["train_data"]["path"]).resolve()
            if train_data.resolve() != frozen_train_data:
                raise ValueError(
                    f"requested train_data {train_data.resolve()} differs from "
                    f"frozen source {frozen_train_data}"
                )
        arch = ArchConfig.model_validate(plan.get("architecture"))
        current_arch = _load_canonical_arch()
        if current_arch != arch:
            raise ValueError(
                "canonical d=1 architecture differs from the frozen sweep plan"
            )
        return (
            plan,
            str(_read_json(plan_path)["plan_sha256"]),
            frozen_specs,
            arch,
        )

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} exists without sweep_plan.json; refusing to reuse or "
            "overwrite it"
        )
    specs = _validate_unique_specs(
        DEFAULT_RUNS if requested_specs is None else requested_specs
    )
    resolved_device = _resolve_device(device_name or "auto")
    sources = _current_source_provenance(
        TRAIN_DATA if train_data is None else train_data
    )
    arch = _load_canonical_arch()
    plan = _build_plan(
        specs=specs,
        device=resolved_device,
        arch=arch,
        sources=sources,
    )
    envelope = _plan_envelope(plan)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(plan_path, envelope)
    return plan, str(envelope["plan_sha256"]), specs, arch


def _run_spec_payload(
    spec: FullBatchRunSpec,
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "plan_sha256": plan_sha256,
        "run": asdict(spec),
    }


def _prepare_run_root(
    run_root: Path,
    *,
    spec: FullBatchRunSpec,
    plan_sha256: str,
) -> None:
    expected = _run_spec_payload(spec, plan_sha256=plan_sha256)
    spec_path = run_root / "run_spec.json"
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        _write_json_exclusive(spec_path, expected)
        return
    if not spec_path.is_file():
        raise ValueError(
            f"{run_root} exists without run_spec.json; refusing to write into it"
        )
    if _read_json(spec_path) != expected:
        raise ValueError(f"run specification changed for {run_root}")


def _next_attempt_directory(run_root: Path) -> tuple[int, Path]:
    attempts_root = run_root / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    indices = [
        int(match.group(1))
        for child in attempts_root.iterdir()
        if child.is_dir()
        and (match := ATTEMPT_DIRECTORY.fullmatch(child.name)) is not None
    ]
    index = max(indices, default=0) + 1
    attempt = attempts_root / f"attempt_{index:03d}"
    attempt.mkdir(exist_ok=False)
    return index, attempt


def _artifact_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 1:
        raise ValueError(f"required artifact is missing or empty: {path}")
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_training_artifacts(
    *,
    attempt: Path,
    spec: FullBatchRunSpec,
    arch: ArchConfig,
) -> dict[str, Path]:
    paths = {
        "checkpoint": attempt / "models" / "autoencoder.pt",
        "checkpoint_metadata": attempt / "models" / "autoencoder.json",
        "history": attempt / "logs" / "history.json",
        "training_summary": attempt / "training_summary.json",
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size < 1:
            raise ValueError(f"training did not produce required artifact {path}")

    summary = _read_json(paths["training_summary"])
    # Summaries written before the training module was renamed carry the
    # older "marcio_full_batch" value; both label the same recipe.
    if summary.get("training_method") not in (
        "reference_full_batch",
        "marcio_full_batch",
    ):
        raise ValueError(
            f"training summary field 'training_method' is "
            f"{summary.get('training_method')!r}; expected the reference "
            "full-batch recipe"
        )
    expected_summary = {
        "seed": spec.seed,
        "epochs_requested": spec.epochs,
        "epochs_completed": spec.epochs,
        "checkpoint_epoch": spec.epochs,
        "checkpoint_selection": "final_epoch",
        "validation_used": False,
        "early_stopping_used": False,
        "best_weight_restoration_used": False,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"training summary field {key!r} is {summary.get(key)!r}; "
                f"expected {expected!r}"
            )
    if summary.get("arch") != arch.model_dump(mode="json"):
        raise ValueError("training summary architecture differs from frozen plan")
    if summary.get("optimizer") != {
        "name": "Adam",
        "learning_rate": spec.learning_rate,
    }:
        raise ValueError("training summary optimizer differs from run spec")
    expected_scheduler = {
        "name": "ReduceLROnPlateau",
        "monitor": "train.loss_total",
        "mode": "min",
        "factor": spec.scheduler_factor,
        "patience": spec.scheduler_patience,
        "threshold": spec.scheduler_threshold,
        "threshold_mode": "rel",
        "min_lr": spec.scheduler_min_lr,
    }
    if summary.get("scheduler") != expected_scheduler:
        raise ValueError("training summary scheduler differs from run spec")

    history_payload = _read_json(paths["history"])
    history = history_payload.get("train")
    if (
        # Histories written before the training module was renamed carry the
        # older "marcio_full_batch" value; both label the same recipe.
        history_payload.get("training_method")
        not in ("reference_full_batch", "marcio_full_batch")
        or not isinstance(history, dict)
        or set(history) != set(HISTORY_KEYS)
    ):
        raise ValueError("malformed reference training history")
    for key in HISTORY_KEYS:
        values = np.asarray(history[key], dtype=np.float64)
        if values.shape != (spec.epochs,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"history {key!r} does not contain {spec.epochs} finite values"
            )

    sidecar = _read_json(paths["checkpoint_metadata"])
    if (
        sidecar.get("version") != 1
        or sidecar.get("arch") != arch.model_dump(mode="json")
    ):
        raise ValueError("checkpoint sidecar differs from frozen architecture")
    return paths


def _build_artifact_manifest(
    *,
    attempt: Path,
    run_root: Path,
    spec: FullBatchRunSpec,
    arch: ArchConfig,
    plan: dict[str, Any],
    plan_sha256: str,
    attempt_index: int,
) -> dict[str, Any]:
    paths = _validate_training_artifacts(
        attempt=attempt,
        spec=spec,
        arch=arch,
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "plan_sha256": plan_sha256,
        "run": asdict(spec),
        "attempt": attempt_index,
        "training_entrypoint": plan["training_entrypoint"],
        "training_semantics": plan["training_semantics"],
        "resolved_device": plan["resolved_device"],
        "architecture": plan["architecture"],
        "sources": plan["sources"],
        "artifacts": {
            name: _artifact_record(path, relative_to=run_root)
            for name, path in paths.items()
        },
        "roa_analysis": {
            "performed": False,
            "note": (
                "Use the checkpoint and frozen source provenance in this "
                "manifest as inputs to a separate blinded batch analysis."
            ),
        },
    }


def _completion_payload(
    *,
    manifest_path: Path,
    run_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = manifest["artifacts"]["checkpoint"]
    return {
        "schema_version": 1,
        "status": "completed",
        "plan_sha256": manifest["plan_sha256"],
        "run": manifest["run"],
        "attempt": manifest["attempt"],
        "artifact_manifest": {
            "path": str(manifest_path.relative_to(run_root)),
            "sha256": _sha256(manifest_path),
        },
        "checkpoint": checkpoint,
    }


def _validate_completed_run(
    *,
    run_root: Path,
    spec: FullBatchRunSpec,
    plan_sha256: str,
) -> dict[str, Any] | None:
    completion_path = run_root / "completed.json"
    if not completion_path.exists():
        return None
    completion = _read_json(completion_path)
    if (
        completion.get("schema_version") != 1
        or completion.get("status") != "completed"
        or completion.get("plan_sha256") != plan_sha256
        or completion.get("run") != asdict(spec)
    ):
        raise ValueError(f"invalid completion marker for {run_root}")
    manifest_ref = completion.get("artifact_manifest")
    if not isinstance(manifest_ref, dict):
        raise ValueError(f"missing manifest reference in {completion_path}")
    manifest_path = run_root / str(manifest_ref.get("path"))
    if (
        not manifest_path.is_file()
        or _sha256(manifest_path) != manifest_ref.get("sha256")
    ):
        raise ValueError(f"artifact manifest hash mismatch for {run_root}")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("plan_sha256") != plan_sha256
        or manifest.get("run") != asdict(spec)
    ):
        raise ValueError(f"invalid artifact manifest for {run_root}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "checkpoint",
        "checkpoint_metadata",
        "history",
        "training_summary",
    }:
        raise ValueError(f"incomplete artifact inventory for {run_root}")
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValueError(f"malformed artifact {name!r} for {run_root}")
        path = run_root / str(record.get("path"))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise ValueError(f"artifact {name!r} failed hash validation")
    return completion


def _next_summary_path(output_dir: Path) -> tuple[int, Path]:
    root = output_dir / "summaries"
    root.mkdir(parents=True, exist_ok=True)
    indices: list[int] = []
    for child in root.iterdir():
        match = re.fullmatch(r"invocation_(\d{4,})\.json", child.name)
        if match is not None:
            indices.append(int(match.group(1)))
    index = max(indices, default=0) + 1
    return index, root / f"invocation_{index:04d}.json"


def _runtime_provenance() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def run_sweep(
    *,
    output_dir: Path,
    device_name: str | None = None,
    run_specs: Sequence[FullBatchRunSpec] | None = None,
    train_data: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Create or resume a frozen sweep and return its invocation summary."""

    output_dir = output_dir.resolve()
    canonical = CANONICAL_RUN.resolve()
    if (
        output_dir == canonical
        or output_dir in canonical.parents
        or canonical in output_dir.parents
    ):
        raise ValueError(
            "sweep output must not be the canonical run, one of its parents, "
            "or one of its descendants"
        )

    plan, plan_sha256, specs, arch = _create_or_load_plan(
        output_dir=output_dir,
        device_name=device_name,
        requested_specs=run_specs,
        train_data=train_data,
    )
    device = torch.device(plan["resolved_device"])
    frozen_train_data = Path(plan["sources"]["train_data"]["path"]).resolve()
    training_pairs: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ] | None = None

    invocation_index, summary_path = _next_summary_path(output_dir)
    started_at = _utc_now()
    rows: list[dict[str, Any]] = []

    for spec in specs:
        run_root = output_dir / "runs" / spec.run_id
        row: dict[str, Any] = {
            "run_id": spec.run_id,
            "seed": spec.seed,
            "epochs": spec.epochs,
            "learning_rate": spec.learning_rate,
        }
        try:
            _prepare_run_root(
                run_root,
                spec=spec,
                plan_sha256=plan_sha256,
            )
            completion = _validate_completed_run(
                run_root=run_root,
                spec=spec,
                plan_sha256=plan_sha256,
            )
            if completion is not None:
                row.update(
                    {
                        "status": "already_completed",
                        "attempt": completion["attempt"],
                        "checkpoint": completion["checkpoint"],
                    }
                )
                rows.append(row)
                if verbose:
                    print(f"{spec.run_id}: already completed; verified and skipped")
                continue
        except Exception as exc:
            row.update(
                {
                    "status": "invalid_existing_run",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            rows.append(row)
            if verbose:
                print(f"{spec.run_id}: invalid existing run: {exc}")
            continue

        attempt_index, attempt = _next_attempt_directory(run_root)
        started_payload = {
            "schema_version": 1,
            "status": "started",
            "started_at_utc": _utc_now(),
            "plan_sha256": plan_sha256,
            "run": asdict(spec),
            "attempt": attempt_index,
        }
        _write_json_exclusive(attempt / "attempt_started.json", started_payload)
        if verbose:
            print(
                f"{spec.run_id}: starting attempt {attempt_index:03d} "
                f"({spec.epochs} epochs, seed={spec.seed}, "
                f"lr={spec.learning_rate:g})",
                flush=True,
            )

        try:
            if training_pairs is None:
                training_pairs = _load_training_pairs(frozen_train_data)
            x, y = training_pairs
            train_reference_full_batch(
                arch=arch,
                x=x,
                y=y,
                epochs=spec.epochs,
                learning_rate=spec.learning_rate,
                seed=spec.seed,
                device=device,
                output_dir=attempt,
                scheduler_factor=spec.scheduler_factor,
                scheduler_patience=spec.scheduler_patience,
                scheduler_threshold=spec.scheduler_threshold,
                scheduler_min_lr=spec.scheduler_min_lr,
                verbose=verbose,
            )
            manifest = _build_artifact_manifest(
                attempt=attempt,
                run_root=run_root,
                spec=spec,
                arch=arch,
                plan=plan,
                plan_sha256=plan_sha256,
                attempt_index=attempt_index,
            )
            manifest_path = _write_json_exclusive(
                attempt / "artifact_manifest.json",
                manifest,
            )
            completed = _completion_payload(
                manifest_path=manifest_path,
                run_root=run_root,
                manifest=manifest,
            )
            _write_json_exclusive(run_root / "completed.json", completed)
            row.update(
                {
                    "status": "completed",
                    "attempt": attempt_index,
                    "checkpoint": completed["checkpoint"],
                    "artifact_manifest": completed["artifact_manifest"],
                }
            )
            if verbose:
                print(f"{spec.run_id}: completed", flush=True)
        except Exception as exc:
            failure = {
                "schema_version": 1,
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "plan_sha256": plan_sha256,
                "run": asdict(spec),
                "attempt": attempt_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_json_exclusive(attempt / "attempt_failed.json", failure)
            row.update(
                {
                    "status": "failed",
                    "attempt": attempt_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if verbose:
                print(
                    f"{spec.run_id}: failed attempt {attempt_index:03d}: {exc}",
                    flush=True,
                )
        rows.append(row)

    completed_statuses = {"completed", "already_completed"}
    counts = {
        "completed": sum(row["status"] in completed_statuses for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "invalid_existing_run": sum(
            row["status"] == "invalid_existing_run" for row in rows
        ),
        "total": len(rows),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "invocation": invocation_index,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "output_dir": str(output_dir),
        "plan_sha256": plan_sha256,
        "resolved_device": str(device),
        "counts": counts,
        "all_runs_completed": counts["completed"] == counts["total"],
        "runs": rows,
        "roa_analysis_performed": False,
        "runtime": _runtime_provenance(),
    }
    _write_json_exclusive(summary_path, summary)
    _write_json_atomic(
        output_dir / "latest_summary.json",
        {
            "schema_version": 1,
            "summary": {
                "path": str(summary_path.relative_to(output_dir)),
                "sha256": _sha256(summary_path),
            },
            "all_runs_completed": summary["all_runs_completed"],
            "counts": counts,
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "device for a new sweep (default: auto); on resume the frozen "
            "device is reused unless this explicit value matches it"
        ),
    )
    parser.add_argument(
        "--matrix-json",
        type=Path,
        help=(
            "predeclared run matrix for a new sweep; if provided on resume it "
            "must match the frozen matrix exactly"
        ),
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        help=(
            "training-pair CSV for a new sweep; its exact path/hash is frozen "
            "in the plan, and an explicit value on resume must match"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_specs = (
        _load_matrix(args.matrix_json)
        if args.matrix_json is not None
        else None
    )
    summary = run_sweep(
        output_dir=args.output_dir,
        device_name=args.device,
        run_specs=run_specs,
        train_data=args.train_data,
        verbose=not args.quiet,
    )
    print(
        json.dumps(
            {
                "all_runs_completed": summary["all_runs_completed"],
                "counts": summary["counts"],
                "output_dir": summary["output_dir"],
                "plan_sha256": summary["plan_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["all_runs_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
