"""Analyze every completed run in a frozen direct-full-batch Chafee d=1 sweep.

The source sweep is read-only.  Before any basin computation begins, this
driver verifies the frozen sweep plan and every completed run's completion
marker, artifact manifest, checkpoint, architecture sidecar, history, and
training summary.  All valid completed checkpoints are then copied into one
fresh sibling diagnostics directory and bound into an immutable analysis plan.
Runs that are incomplete or whose source artifacts fail verification are
reported but never analyzed.

Every accepted checkpoint receives the same established 1-D stages, in frozen
sweep-plan order: bounds, coarse lookup precomputation, uniform graph, fine
lookup precomputation, adaptive graph, and basin statistics.  A failed
candidate does not stop later candidates.  Basin results never determine which
checkpoints are included or the execution order.  The final JSON may include a
clearly labeled exploratory post-hoc ranking, but that ranking is produced only
after the uniform all-candidate computation is finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

import chafee_latent_dimension_study as study
import sweep_chafee_d1_full_batch as sweep
from latentdynamics.training import load_checkpoint

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = sweep.DEFAULT_OUTPUT
DEFAULT_ANALYSIS = DEFAULT_SOURCE.with_name(f"{DEFAULT_SOURCE.name}_roa_diagnostics")
DEFAULT_REFERENCE_ROOT = study.DEFAULT_REFERENCE_ROOT
# Rebound in main() when --reference-root is supplied.
REFERENCE_ROOT = DEFAULT_REFERENCE_ROOT
REFERENCE_README = REFERENCE_ROOT / "Readme.txt"
LATENT_1D_ROOT = sweep.LATENT_1D_OUTPUT_ROOT
CANONICAL_4K = sweep.CANONICAL_RUN
CANONICAL_10K = LATENT_1D_ROOT / "seed_0_epoch_10000"
MINIBATCH_SOURCE = LATENT_1D_ROOT / "seed_0_minibatch_b1024_lr1e3"
MINIBATCH_ANALYSIS = LATENT_1D_ROOT / "seed_0_minibatch_b1024_lr1e3_roa"

REFERENCE_README_SHA256 = (
    "eac15b7db0ef7d9da2d3b29fc5aafdf19d1ca1b0fdcc3b971eda564a087b8beb"
)
CANONICAL_10K_STATS_SHA256 = (
    "6c1dc4d8b8686400e31c58948240326e9360410cbfc255da29abcc7e57be816b"
)
FULL_BATCH_10K_COMBINED_PERCENT = 50.01271940981938
REFERENCE_ARCHIVED_COMBINED_PERCENT = 78.38972271686593
CONDITIONED_TRAJECTORIES = 7_862
STAGES = (
    "bounds",
    "precompute-coarse",
    "uniform",
    "precompute-fine",
    "adaptive",
    "stats",
)

COUNT_KEYS = {
    "negative": "correctly_classified_in_negative_basin",
    "positive": "correctly_classified_in_positive_basin",
    "misclassified_negative": "misclassified_in_negative_basin",
    "misclassified_positive": "misclassified_in_positive_basin",
    "outside": "outside_both_basins",
}
REFERENCE_NAMES = ("full_batch_10000", "reference_archived")
CSV_FIELDS = (
    "plan_index",
    "run_id",
    "seed",
    "epochs",
    "learning_rate",
    "source_status",
    "status",
    "attempt",
    "checkpoint_sha256",
    "history_sha256",
    "loss_reconstruction",
    "loss_prediction",
    "loss_total",
    "history_final_learning_rate",
    "optimizer_final_learning_rate",
    "conditioned_trajectories",
    "negative_correct_count",
    "negative_correct_percent",
    "positive_correct_count",
    "positive_correct_percent",
    "correct_combined_count",
    "correct_combined_percent",
    "outside_count",
    "outside_percent",
    "misc_count",
    "misc_percent",
    "uniform_is_bistable",
    "adaptive_is_bistable",
    "roots_define_two_distinct_attractor_basins",
    "eligible_for_bistable_dimension_table",
    "delta_vs_full_batch_10000_percentage_points",
    "beats_full_batch_10000",
    "delta_vs_reference_archived_percentage_points",
    "beats_reference_archived",
    "analysis_directory",
    "error_stage",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class FrozenCandidate:
    """One source-completed run whose artifact chain passed verification."""

    plan_index: int
    spec: sweep.FullBatchRunSpec
    run_root: Path
    attempt: int
    run_spec: Path
    run_spec_sha256: str
    completion: Path
    completion_sha256: str
    artifact_manifest: Path
    artifact_manifest_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    sidecar: Path
    sidecar_sha256: str
    history: Path
    history_sha256: str
    training_summary: Path
    training_summary_sha256: str
    final_train: dict[str, float]
    history_final_learning_rate: float
    optimizer_final_learning_rate: float

    @property
    def snapshot_name(self) -> str:
        return f"run_{self.plan_index:03d}_{self.spec.run_id}"

    def plan_record(
        self,
        *,
        source_root: Path,
        snapshot_root: Path,
    ) -> dict[str, Any]:
        snapshot = snapshot_root / self.snapshot_name
        return {
            "plan_index": self.plan_index,
            "run": asdict(self.spec),
            "source_status": "completed_valid",
            "attempt": self.attempt,
            "source_files": {
                "run_spec": _file_reference(self.run_spec, source_root),
                "completion": _file_reference(self.completion, source_root),
                "artifact_manifest": _file_reference(
                    self.artifact_manifest,
                    source_root,
                ),
                "checkpoint": _file_reference(self.checkpoint, source_root),
                "checkpoint_metadata": _file_reference(
                    self.sidecar,
                    source_root,
                ),
                "history": _file_reference(self.history, source_root),
                "training_summary": _file_reference(
                    self.training_summary,
                    source_root,
                ),
            },
            "snapshot_directory": str(snapshot.relative_to(snapshot_root.parent)),
            "training_endpoint": {
                **self.final_train,
                "history_final_learning_rate": (
                    self.history_final_learning_rate
                ),
                "optimizer_final_learning_rate": (
                    self.optimizer_final_learning_rate
                ),
            },
        }


@dataclass(frozen=True)
class SourceInventoryItem:
    """One frozen-plan row, whether or not it can be analyzed."""

    plan_index: int
    spec: sweep.FullBatchRunSpec
    source_status: str
    candidate: FrozenCandidate | None = None
    error_type: str = ""
    error_message: str = ""

    def plan_record(
        self,
        *,
        source_root: Path,
        snapshot_root: Path,
    ) -> dict[str, Any]:
        if self.candidate is not None:
            return self.candidate.plan_record(
                source_root=source_root,
                snapshot_root=snapshot_root,
            )
        return {
            "plan_index": self.plan_index,
            "run": asdict(self.spec),
            "source_status": self.source_status,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class FrozenSweep:
    root: Path
    plan_path: Path
    plan_sha256: str
    plan: dict[str, Any]
    inventory: tuple[SourceInventoryItem, ...]

    @property
    def candidates(self) -> tuple[FrozenCandidate, ...]:
        return tuple(
            item.candidate
            for item in self.inventory
            if item.candidate is not None
        )


class ExactRunPaths(study.DimensionPaths):
    """Route existing 1-D stage helpers into one exact analysis directory."""

    @property
    def run(self) -> Path:
        return self.output_root


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assert_safe_target(source_sweep: Path, analysis_root: Path) -> None:
    source = source_sweep.resolve()
    target = analysis_root.resolve()
    if target.parent != source.parent:
        raise ValueError(
            "analysis target must be an isolated sibling of the source sweep"
        )
    protected = (
        source,
        CANONICAL_4K.resolve(),
        CANONICAL_10K.resolve(),
        MINIBATCH_SOURCE.resolve(),
        MINIBATCH_ANALYSIS.resolve(),
    )
    for root in protected:
        if (
            target == root
            or _is_within(target, root)
            or _is_within(root, target)
        ):
            raise ValueError(
                f"analysis target {target} overlaps protected directory {root}"
            )
    if analysis_root.is_symlink():
        raise ValueError(f"analysis target must not be a symlink: {analysis_root}")
    if analysis_root.exists():
        raise FileExistsError(
            f"analysis target already exists; refusing to overwrite: {analysis_root}"
        )


def _declared_under(root: Path, value: Any, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing declared {description} path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"declared {description} path must be relative")
    resolved = (root / relative).resolve()
    if not _is_within(resolved, root.resolve()):
        raise ValueError(f"declared {description} path escapes its run root")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _file_reference(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": study.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_statistics(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    statistics = payload.get("statistics", {})
    counts = statistics.get("counts", {})
    percentages = statistics.get("percentages", {})
    if (
        statistics.get("total_trajectories") != 10_000
        or statistics.get("excluded_zero_trajectories") != 2_138
        or statistics.get("conditioned_trajectories") != CONDITIONED_TRAJECTORIES
    ):
        raise ValueError("basin-statistics denominators do not match the fixed benchmark")
    if sum(int(value) for value in counts.values()) != CONDITIONED_TRAJECTORIES:
        raise ValueError("basin counts do not conserve the conditioned trajectories")
    if not math.isclose(
        sum(float(value) for value in percentages.values()),
        100.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("basin percentages do not sum to 100")
    if (
        payload.get("uniform_is_bistable") is not True
        or payload.get("roots_define_two_distinct_attractor_basins") is not True
        or payload.get("eligible_for_bistable_dimension_table") is not True
    ):
        raise ValueError("selected checkpoint did not produce a comparable bistable graph")
    return payload


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "analysis_manifest.json":
            continue
        manifest[str(path.relative_to(root))] = {
            "size_bytes": path.stat().st_size,
            "sha256": study.sha256_file(path),
        }
    return manifest


def _verify_model(checkpoint: Path, expected_arch: dict[str, Any]) -> None:
    model, arch = load_checkpoint(checkpoint.parent, map_location="cpu")
    if arch.model_dump(mode="json") != expected_arch:
        raise ValueError(f"{checkpoint} architecture differs from the sweep plan")
    if arch != study.reference_architecture(1):
        raise ValueError(f"{checkpoint} is not the exact reference 64-to-1 architecture")
    if not all(
        torch.isfinite(value).all().item()
        for value in model.state_dict().values()
    ):
        raise ValueError(f"{checkpoint} contains non-finite parameters")
    model.eval()
    with torch.inference_mode():
        probe = torch.zeros((2, 64), dtype=torch.float32)
        encoded = model.encoder(probe)
        mapped = model.latent_map(encoded)
        decoded = model.decoder(mapped)
    if not all(
        torch.isfinite(value).all().item()
        for value in (encoded, mapped, decoded)
    ):
        raise ValueError(f"{checkpoint} produces non-finite probe outputs")


def _verify_training_semantics(plan: dict[str, Any]) -> None:
    # Plans frozen before the training module was renamed record the older
    # "train_marcio_full_batch" entrypoint; both label the same recipe.
    if plan.get("training_entrypoint") not in (
        "latentdynamics.training.train_reference_full_batch",
        "latentdynamics.training.train_marcio_full_batch",
    ):
        raise ValueError("sweep did not use the exact reference training entrypoint")
    semantics = plan.get("training_semantics")
    expected = {
        "data_rows": 30_000,
        "high_dimension": 64,
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
    }
    if semantics != expected:
        raise ValueError("sweep training semantics differ from the fixed protocol")
    expected_arch = study.reference_architecture(1).model_dump(mode="json")
    if plan.get("architecture") != expected_arch:
        raise ValueError("sweep architecture is not the exact reference d=1 model")


def _verify_candidate(
    *,
    source_root: Path,
    plan_sha256: str,
    plan: dict[str, Any],
    plan_index: int,
    spec: sweep.FullBatchRunSpec,
) -> FrozenCandidate:
    run_root = source_root / "runs" / spec.run_id
    run_spec = run_root / "run_spec.json"
    expected_run_spec = sweep._run_spec_payload(
        spec,
        plan_sha256=plan_sha256,
    )
    if _read_json(run_spec) != expected_run_spec:
        raise ValueError(f"{spec.run_id} run_spec.json differs from the frozen plan")

    completion = sweep._validate_completed_run(
        run_root=run_root,
        spec=spec,
        plan_sha256=plan_sha256,
    )
    if completion is None:
        raise ValueError(f"{spec.run_id} has no completion marker")
    manifest_ref = completion.get("artifact_manifest")
    if not isinstance(manifest_ref, dict):
        raise ValueError(f"{spec.run_id} has no artifact manifest reference")
    manifest_path = _declared_under(
        run_root,
        manifest_ref.get("path"),
        description="artifact manifest",
    )
    manifest = _read_json(manifest_path)
    if study.sha256_file(manifest_path) != manifest_ref.get("sha256"):
        raise ValueError(f"{spec.run_id} artifact manifest hash mismatch")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{spec.run_id} artifact inventory is malformed")

    def artifact(name: str) -> tuple[Path, str]:
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"{spec.run_id} is missing artifact {name!r}")
        path = _declared_under(
            run_root,
            record.get("path"),
            description=f"artifact {name}",
        )
        digest = study.sha256_file(path)
        if (
            digest != record.get("sha256")
            or path.stat().st_size != record.get("size_bytes")
        ):
            raise ValueError(f"{spec.run_id} artifact {name!r} failed verification")
        return path, digest

    checkpoint, checkpoint_hash = artifact("checkpoint")
    sidecar, sidecar_hash = artifact("checkpoint_metadata")
    history, history_hash = artifact("history")
    training_summary, summary_hash = artifact("training_summary")
    attempt_root = checkpoint.parent.parent
    expected_paths = sweep._validate_training_artifacts(
        attempt=attempt_root,
        spec=spec,
        arch=study.reference_architecture(1),
    )
    observed_paths = {
        "checkpoint": checkpoint,
        "checkpoint_metadata": sidecar,
        "history": history,
        "training_summary": training_summary,
    }
    if {
        key: value.resolve()
        for key, value in expected_paths.items()
    } != {
        key: value.resolve()
        for key, value in observed_paths.items()
    }:
        raise ValueError(f"{spec.run_id} manifest mixes multiple attempt directories")

    _verify_model(checkpoint, plan["architecture"])
    history_payload = _read_json(history)
    train = history_payload["train"]
    final_train = {
        key: float(train[key][-1])
        for key in (
            "loss_reconstruction",
            "loss_prediction",
            "loss_total",
        )
    }
    history_final_lr = float(train["learning_rate"][-1])
    training_payload = _read_json(training_summary)
    optimizer_final_lr = float(training_payload["final_learning_rate"])
    numeric_values = (*final_train.values(), history_final_lr, optimizer_final_lr)
    if not all(math.isfinite(value) and value >= 0.0 for value in numeric_values):
        raise ValueError(f"{spec.run_id} has invalid final training metrics")
    attempt = int(completion.get("attempt", -1))
    if attempt < 1 or attempt_root.name != f"attempt_{attempt:03d}":
        raise ValueError(f"{spec.run_id} completion attempt is inconsistent")

    return FrozenCandidate(
        plan_index=plan_index,
        spec=spec,
        run_root=run_root.resolve(),
        attempt=attempt,
        run_spec=run_spec.resolve(),
        run_spec_sha256=study.sha256_file(run_spec),
        completion=(run_root / "completed.json").resolve(),
        completion_sha256=study.sha256_file(run_root / "completed.json"),
        artifact_manifest=manifest_path,
        artifact_manifest_sha256=study.sha256_file(manifest_path),
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        sidecar=sidecar,
        sidecar_sha256=sidecar_hash,
        history=history,
        history_sha256=history_hash,
        training_summary=training_summary,
        training_summary_sha256=summary_hash,
        final_train=final_train,
        history_final_learning_rate=history_final_lr,
        optimizer_final_learning_rate=optimizer_final_lr,
    )


def _verify_source_sweep(source_sweep: Path) -> FrozenSweep:
    root = source_sweep.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    plan_path = root / "sweep_plan.json"
    envelope = _read_json(plan_path)
    plan = sweep._validate_plan_envelope(envelope)
    plan_sha256 = str(envelope["plan_sha256"])
    sweep._validate_plan_sources(plan)
    _verify_training_semantics(plan)
    specs = sweep._plan_specs(plan)

    inventory: list[SourceInventoryItem] = []
    for plan_index, spec in enumerate(specs, start=1):
        run_root = root / "runs" / spec.run_id
        completion = run_root / "completed.json"
        if not completion.is_file():
            if run_root.exists():
                try:
                    expected = sweep._run_spec_payload(
                        spec,
                        plan_sha256=plan_sha256,
                    )
                    if _read_json(run_root / "run_spec.json") != expected:
                        raise ValueError("run_spec.json differs from frozen plan")
                except Exception as error:
                    inventory.append(
                        SourceInventoryItem(
                            plan_index=plan_index,
                            spec=spec,
                            source_status="source_invalid",
                            error_type=type(error).__name__,
                            error_message=str(error),
                        )
                    )
                    continue
            inventory.append(
                SourceInventoryItem(
                    plan_index=plan_index,
                    spec=spec,
                    source_status="source_not_completed",
                    error_message="source run has no completed.json",
                )
            )
            continue
        try:
            candidate = _verify_candidate(
                source_root=root,
                plan_sha256=plan_sha256,
                plan=plan,
                plan_index=plan_index,
                spec=spec,
            )
        except Exception as error:
            inventory.append(
                SourceInventoryItem(
                    plan_index=plan_index,
                    spec=spec,
                    source_status="source_invalid",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
        else:
            inventory.append(
                SourceInventoryItem(
                    plan_index=plan_index,
                    spec=spec,
                    source_status="completed_valid",
                    candidate=candidate,
                )
            )
    return FrozenSweep(
        root=root,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        plan=plan,
        inventory=tuple(inventory),
    )


def _exact_inputs_for_training_data(
    train_data: Path | None,
) -> study.ExactInputs:
    """Bind one training CSV to the fixed trajectory truth and stable roots."""

    canonical = study.verify_exact_inputs(REFERENCE_ROOT)
    if train_data is None:
        return canonical
    source = train_data.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    # Validate the complete numeric contract before any analysis output exists.
    study._load_training_pairs(source)
    hashes = dict(canonical.hashes)
    sizes = dict(canonical.sizes_bytes)
    hashes["train_data.csv"] = study.sha256_file(source)
    sizes["train_data.csv"] = int(source.stat().st_size)
    return study.ExactInputs(
        archive_dir=canonical.archive_dir,
        train_data=source,
        trajectory_labels=canonical.trajectory_labels,
        stable_roots=canonical.stable_roots,
        hashes=hashes,
        sizes_bytes=sizes,
    )


def _verify_analysis_training_input(
    source: FrozenSweep,
    inputs: study.ExactInputs,
) -> None:
    record = source.plan.get("sources", {}).get("train_data")
    if not isinstance(record, dict):
        raise ValueError("source sweep has no frozen train_data provenance")
    expected = {
        "path": str(inputs.train_data.resolve()),
        "sha256": inputs.hashes["train_data.csv"],
        "size_bytes": inputs.sizes_bytes["train_data.csv"],
    }
    observed = {
        "path": str(Path(record.get("path", "")).resolve()),
        "sha256": record.get("sha256"),
        "size_bytes": record.get("size_bytes"),
    }
    if observed != expected:
        raise ValueError(
            "analysis train_data does not match the source sweep's frozen "
            f"training input: expected {observed}, received {expected}"
        )


def _stats_fields(payload: dict[str, Any]) -> dict[str, Any]:
    statistics = payload["statistics"]
    counts = statistics["counts"]
    percentages = statistics["percentages"]
    negative_count = int(counts[COUNT_KEYS["negative"]])
    positive_count = int(counts[COUNT_KEYS["positive"]])
    outside_count = int(counts[COUNT_KEYS["outside"]])
    misc_count = int(counts[COUNT_KEYS["misclassified_negative"]]) + int(
        counts[COUNT_KEYS["misclassified_positive"]]
    )
    negative_percent = float(percentages[COUNT_KEYS["negative"]])
    positive_percent = float(percentages[COUNT_KEYS["positive"]])
    outside_percent = float(percentages[COUNT_KEYS["outside"]])
    misc_percent = float(
        percentages[COUNT_KEYS["misclassified_negative"]]
    ) + float(percentages[COUNT_KEYS["misclassified_positive"]])
    return {
        "conditioned_trajectories": int(statistics["conditioned_trajectories"]),
        "negative_correct_count": negative_count,
        "negative_correct_percent": negative_percent,
        "positive_correct_count": positive_count,
        "positive_correct_percent": positive_percent,
        "correct_combined_count": negative_count + positive_count,
        "correct_combined_percent": negative_percent + positive_percent,
        "outside_count": outside_count,
        "outside_percent": outside_percent,
        "misc_count": misc_count,
        "misc_percent": misc_percent,
        "uniform_is_bistable": payload.get("uniform_is_bistable"),
        "adaptive_is_bistable": payload.get("adaptive_graph", {}).get(
            "is_bistable"
        ),
        "roots_define_two_distinct_attractor_basins": payload.get(
            "roots_define_two_distinct_attractor_basins"
        ),
        "eligible_for_bistable_dimension_table": payload.get(
            "eligible_for_bistable_dimension_table"
        ),
    }


def _reference_archived_statistic() -> dict[str, Any]:
    if study.sha256_file(REFERENCE_README) != REFERENCE_README_SHA256:
        raise ValueError("reference Readme.txt hash does not match the archive")
    text = REFERENCE_README.read_text(encoding="utf-8")

    def count(label: str) -> int:
        match = re.search(rf"^{re.escape(label)}:\s*(\d+)\s*$", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"missing archived reference count {label!r}")
        return int(match.group(1))

    total = count("Total number of attractor trajectories")
    excluded = count("Number of non-converging trajectories")
    outside = count(
        "Number of points not classified (not in an attractor basin)"
    )
    misc_negative = count(
        "Number of points misclassified in basin of negative root"
    )
    misc_positive = count(
        "Number of points misclassified in basin of positive root"
    )
    negative = count(
        "Number of points correctly classified in basin of negative root"
    )
    positive = count(
        "Number of points correctly classified in basin of positive root"
    )
    conditioned = total - excluded
    if (
        conditioned != CONDITIONED_TRAJECTORIES
        or negative + positive + outside + misc_negative + misc_positive
        != conditioned
    ):
        raise ValueError("archived reference counts do not conserve trajectories")

    def percent(value: int) -> float:
        return 100.0 * value / conditioned

    combined = percent(negative + positive)
    if not math.isclose(
        combined,
        REFERENCE_ARCHIVED_COMBINED_PERCENT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("archived combined reference percentage changed")
    return {
        "label": "coauthor archived d=1 statistic",
        "source": {
            "path": str(REFERENCE_README.resolve()),
            "sha256": REFERENCE_README_SHA256,
        },
        "conditioned_trajectories": conditioned,
        "negative_correct_count": negative,
        "negative_correct_percent": percent(negative),
        "positive_correct_count": positive,
        "positive_correct_percent": percent(positive),
        "correct_combined_count": negative + positive,
        "correct_combined_percent": combined,
        "outside_count": outside,
        "outside_percent": percent(outside),
        "misc_count": misc_negative + misc_positive,
        "misc_percent": percent(misc_negative + misc_positive),
    }


def _reference_inventory() -> dict[str, dict[str, Any]]:
    canonical_path = CANONICAL_10K / "basin_statistics.json"
    if study.sha256_file(canonical_path) != CANONICAL_10K_STATS_SHA256:
        raise ValueError("canonical 10,000-epoch basin statistics hash changed")
    canonical_payload = _validate_statistics(canonical_path)
    canonical = {
        "label": "fresh seed-0 full-batch 10,000-epoch run",
        "source": {
            "path": str(canonical_path.resolve()),
            "sha256": CANONICAL_10K_STATS_SHA256,
        },
        **_stats_fields(canonical_payload),
    }
    if not math.isclose(
        float(canonical["correct_combined_percent"]),
        FULL_BATCH_10K_COMBINED_PERCENT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("canonical 10,000-epoch reference percentage changed")
    return {
        "full_batch_10000": canonical,
        "reference_archived": _reference_archived_statistic(),
    }


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> Path:
    if study.sha256_file(source) != expected_sha256:
        raise ValueError(f"source changed before diagnostics snapshot: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if study.sha256_file(target) != expected_sha256:
        raise ValueError(f"diagnostics copy hash changed: {target}")
    return target


def _snapshot_candidate(
    snapshot_root: Path,
    candidate: FrozenCandidate,
) -> Path:
    destination = snapshot_root / candidate.snapshot_name
    copies = (
        (candidate.run_spec, destination / "run_spec.json", candidate.run_spec_sha256),
        (
            candidate.completion,
            destination / "completed.json",
            candidate.completion_sha256,
        ),
        (
            candidate.artifact_manifest,
            destination / "artifact_manifest.json",
            candidate.artifact_manifest_sha256,
        ),
        (
            candidate.checkpoint,
            destination / "models" / "autoencoder.pt",
            candidate.checkpoint_sha256,
        ),
        (
            candidate.sidecar,
            destination / "models" / "autoencoder.json",
            candidate.sidecar_sha256,
        ),
        (
            candidate.history,
            destination / "logs" / "history.json",
            candidate.history_sha256,
        ),
        (
            candidate.training_summary,
            destination / "training_summary.json",
            candidate.training_summary_sha256,
        ),
    )
    for source, target, digest in copies:
        _copy_verified(source, target, digest)
    return destination


def _snapshot_source(
    *,
    target: Path,
    source: FrozenSweep,
    references: dict[str, dict[str, Any]],
) -> Path:
    snapshot_root = target / "source_snapshot"
    _copy_verified(
        source.plan_path,
        snapshot_root / "sweep_plan.json",
        study.sha256_file(source.plan_path),
    )
    for candidate in source.candidates:
        _snapshot_candidate(snapshot_root, candidate)
    for name, reference in references.items():
        source_record = reference["source"]
        _copy_verified(
            Path(source_record["path"]),
            snapshot_root / "references" / f"{name}{Path(source_record['path']).suffix}",
            str(source_record["sha256"]),
        )
    return snapshot_root


def _analysis_plan(
    *,
    target: Path,
    source: FrozenSweep,
    snapshot_root: Path,
    references: dict[str, dict[str, Any]],
    inputs: study.ExactInputs,
    device: torch.device,
    batch_points: int | str,
    uniform_only: bool,
) -> dict[str, Any]:
    scripts = {
        "batch_analyzer": Path(__file__).resolve(),
        "sweep_runner": Path(sweep.__file__).resolve(),
        "study_driver": Path(study.__file__).resolve(),
    }
    return {
        "schema_version": 1,
        "status": "frozen_before_analysis",
        "created_at_utc": _utc_now(),
        "source_sweep": str(source.root),
        "source_sweep_plan": {
            "path": str(source.plan_path),
            "payload_sha256": source.plan_sha256,
            "file_sha256": study.sha256_file(source.plan_path),
        },
        "source_snapshot": str(snapshot_root.relative_to(target)),
        "source_inventory": [
            item.plan_record(
                source_root=source.root,
                snapshot_root=snapshot_root,
            )
            for item in source.inventory
        ],
        "analysis_protocol": {
            "frozen_plan_order_all_runs": [
                item.spec.run_id for item in source.inventory
            ],
            "analysis_order_source_completed_valid_runs": [
                candidate.spec.run_id for candidate in source.candidates
            ],
            "checkpoint_selection": (
                "none; analyze every source-completed valid checkpoint in "
                "frozen sweep-plan order"
            ),
            "basin_statistics_used_for_candidate_inclusion_or_order": False,
            "post_hoc_ranking_used_for_computation_or_selection": False,
            "dimension": 1,
            "device": str(device),
            "batch_points": batch_points,
            "stages": (
                ["bounds", "precompute-coarse", "uniform"]
                if uniform_only
                else list(STAGES)
            ),
            "cmgdb_uniform_subdivision": 8,
            "cmgdb_adaptive_subdivision": 11,
            "headline_validity_protocol": (
                "uniform graph has exactly two minimal attractors and the two "
                "encoded stable roots have distinct unique singleton "
                "associations"
            ),
            "adaptive_graph_required_for_headline_validity": False,
            "continue_after_candidate_failure": True,
        },
        "analysis_scripts": {
            name: {
                "path": str(path),
                "sha256": study.sha256_file(path),
            }
            for name, path in scripts.items()
        },
        "archived_inputs": inputs.provenance(),
        "references": references,
    }


def _base_candidate_row(
    candidate: FrozenCandidate,
    *,
    analysis_directory: Path,
) -> dict[str, Any]:
    return {
        "plan_index": candidate.plan_index,
        "run_id": candidate.spec.run_id,
        "seed": candidate.spec.seed,
        "epochs": candidate.spec.epochs,
        "learning_rate": candidate.spec.learning_rate,
        "source_status": "completed_valid",
        "status": "running",
        "attempt": candidate.attempt,
        "checkpoint_sha256": candidate.checkpoint_sha256,
        "history_sha256": candidate.history_sha256,
        **candidate.final_train,
        "history_final_learning_rate": candidate.history_final_learning_rate,
        "optimizer_final_learning_rate": candidate.optimizer_final_learning_rate,
        "analysis_directory": str(analysis_directory),
        "error_stage": "",
        "error_type": "",
        "error_message": "",
    }


def _source_issue_row(item: SourceInventoryItem) -> dict[str, Any]:
    return {
        "plan_index": item.plan_index,
        "run_id": item.spec.run_id,
        "seed": item.spec.seed,
        "epochs": item.spec.epochs,
        "learning_rate": item.spec.learning_rate,
        "source_status": item.source_status,
        "status": item.source_status,
        "analysis_directory": "",
        "error_stage": "source-verification",
        "error_type": item.error_type,
        "error_message": item.error_message,
    }


def _add_reference_deltas(
    row: dict[str, Any],
    references: dict[str, dict[str, Any]],
) -> None:
    score = row.get("correct_combined_percent")
    for name in REFERENCE_NAMES:
        reference = references[name]
        delta_key = f"delta_vs_{name}_percentage_points"
        beats_key = f"beats_{name}"
        if score is None:
            row[delta_key] = None
            row[beats_key] = None
        else:
            delta = float(score) - float(
                reference["correct_combined_percent"]
            )
            row[delta_key] = delta
            row[beats_key] = delta > 0.0


def _add_partial_bistability(row: dict[str, Any], run: Path) -> None:
    records = (
        ("uniform_is_bistable", run / "stage_status" / "uniform.json"),
        ("adaptive_is_bistable", run / "stage_status" / "adaptive.json"),
    )
    for field, path in records:
        if field in row or not path.is_file():
            continue
        try:
            row[field] = _read_json(path).get("is_bistable")
        except (OSError, ValueError):
            continue


def _finalize_uniform_only_statistics(
    paths: ExactRunPaths,
) -> dict[str, Any]:
    payload = _read_json(paths.stats)
    attractors = {
        int(node) for node in payload.get("cmgdb", {}).get("attractor_nodes", [])
    }
    roots = payload.get("stable_roots", {})
    root_labels = {
        int(roots["negative_basin_label"]),
        int(roots["positive_basin_label"]),
    }
    uniform_is_bistable = len(attractors) == 2
    roots_distinct = len(root_labels) == 2 and root_labels.issubset(attractors)
    payload.update(
        {
            "uniform_is_bistable": uniform_is_bistable,
            "adaptive_graph": {
                "status": "not_run_for_matched_uniform_protocol",
                "is_bistable": None,
            },
            "roots_define_two_distinct_attractor_basins": roots_distinct,
            "eligible_for_bistable_dimension_table": False,
            "uniform_headline_valid": bool(
                uniform_is_bistable and roots_distinct
            ),
        }
    )
    study._write_json(paths.stats, payload)
    statistics = payload.get("statistics", {})
    counts = statistics.get("counts", {})
    percentages = statistics.get("percentages", {})
    if (
        statistics.get("total_trajectories") != 10_000
        or statistics.get("excluded_zero_trajectories") != 2_138
        or statistics.get("conditioned_trajectories") != CONDITIONED_TRAJECTORIES
        or sum(int(value) for value in counts.values())
        != CONDITIONED_TRAJECTORIES
        or not math.isclose(
            sum(float(value) for value in percentages.values()),
            100.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("uniform basin statistics failed conservation checks")
    if payload["uniform_headline_valid"] is not True:
        raise ValueError(
            "uniform graph/root associations do not satisfy the matched "
            "headline-validity protocol"
        )
    return payload


def _analyze_candidate(
    *,
    target: Path,
    snapshot_root: Path,
    candidate: FrozenCandidate,
    analysis_plan_sha256: str,
    inputs: study.ExactInputs,
    device: torch.device,
    batch_points: int | str,
    references: dict[str, dict[str, Any]],
    uniform_only: bool = False,
) -> dict[str, Any]:
    run = target / "by_run" / candidate.snapshot_name
    run.mkdir(parents=True, exist_ok=False)
    row = _base_candidate_row(candidate, analysis_directory=run)
    manifest_path = run / "analysis_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": _utc_now(),
        "plan_index": candidate.plan_index,
        "run": asdict(candidate.spec),
        "analysis_plan_sha256": analysis_plan_sha256,
        "source_checkpoint_sha256": candidate.checkpoint_sha256,
        "completed_stages": [],
    }
    _write_json(manifest_path, manifest)
    active_stage = "checkpoint_copy"
    started = time.perf_counter()
    try:
        snapshot = snapshot_root / candidate.snapshot_name
        _copy_verified(
            snapshot / "models" / "autoencoder.pt",
            run / "models" / "autoencoder.pt",
            candidate.checkpoint_sha256,
        )
        _copy_verified(
            snapshot / "models" / "autoencoder.json",
            run / "models" / "autoencoder.json",
            candidate.sidecar_sha256,
        )
        source_manifests = run / "source_manifests"
        _copy_verified(
            snapshot / "run_spec.json",
            source_manifests / "run_spec.json",
            candidate.run_spec_sha256,
        )
        _copy_verified(
            snapshot / "completed.json",
            source_manifests / "completed.json",
            candidate.completion_sha256,
        )
        _copy_verified(
            snapshot / "artifact_manifest.json",
            source_manifests / "artifact_manifest.json",
            candidate.artifact_manifest_sha256,
        )
        _copy_verified(
            snapshot / "logs" / "history.json",
            source_manifests / "history.json",
            candidate.history_sha256,
        )
        _copy_verified(
            snapshot / "training_summary.json",
            source_manifests / "training_summary.json",
            candidate.training_summary_sha256,
        )

        paths = ExactRunPaths(output_root=run, dimension=1)
        uniform_runners = (
            ("bounds", lambda: study._run_bounds(paths, inputs, device=device)),
            (
                "precompute-coarse",
                lambda: study._run_precompute_coarse(
                    paths,
                    device=device,
                    batch_points=batch_points,
                ),
            ),
            (
                "uniform",
                lambda: study._run_uniform(paths, inputs, device=device),
            ),
        )
        adaptive_runners = (
            (
                "precompute-fine",
                lambda: study._run_precompute_fine(
                    paths,
                    device=device,
                    batch_points=batch_points,
                ),
            ),
            ("adaptive", lambda: study._run_adaptive(paths, topology_only=False)),
            ("stats", lambda: study._run_statistics(paths)),
        )
        runners = (
            uniform_runners
            if uniform_only
            else (*uniform_runners, *adaptive_runners)
        )
        for active_stage, runner in runners:
            print(
                f"run={candidate.spec.run_id} start_stage={active_stage}",
                flush=True,
            )
            runner()
            manifest["completed_stages"].append(active_stage)
            _write_json(manifest_path, manifest)

        active_stage = "validate-statistics"
        statistics = (
            _finalize_uniform_only_statistics(paths)
            if uniform_only
            else _validate_statistics(paths.stats)
        )
        row.update(_stats_fields(statistics))
        row["status"] = "complete"
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "basin_statistics": statistics["statistics"],
                "bistability": {
                    "uniform": statistics["uniform_is_bistable"],
                    "adaptive": statistics["adaptive_graph"]["is_bistable"],
                    "roots_distinct": statistics[
                        "roots_define_two_distinct_attractor_basins"
                    ],
                    "eligible": statistics[
                        "eligible_for_bistable_dimension_table"
                    ],
                    "uniform_headline_valid": statistics.get(
                        "uniform_headline_valid",
                        bool(
                            statistics["uniform_is_bistable"]
                            and statistics[
                                "roots_define_two_distinct_attractor_basins"
                            ]
                        ),
                    ),
                },
            }
        )
    except Exception as error:
        statistics_path = run / "basin_statistics.json"
        if statistics_path.is_file():
            try:
                row.update(_stats_fields(_read_json(statistics_path)))
            except (KeyError, TypeError, ValueError):
                pass
        _add_partial_bistability(row, run)
        row.update(
            {
                "status": "failed",
                "error_stage": active_stage,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "error_stage": active_stage,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
    _add_reference_deltas(row, references)
    manifest["output_files"] = _file_manifest(run)
    _write_json(manifest_path, manifest)
    print(
        f"run={candidate.spec.run_id} analysis_status={row['status']}",
        flush=True,
    )
    return row


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    if row.get(field) is not None
                    else ""
                    for field in CSV_FIELDS
                }
            )
    temporary.replace(path)
    return path


def _post_hoc_ranking(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get("status") == "complete"
        and isinstance(row.get("correct_combined_percent"), (int, float))
    ]
    ordered = sorted(
        eligible,
        key=lambda row: (
            -float(row["correct_combined_percent"]),
            int(row["plan_index"]),
        ),
    )
    return [
        {
            "exploratory_rank": rank,
            "run_id": row["run_id"],
            "plan_index": row["plan_index"],
            "seed": row["seed"],
            "correct_combined_percent": row["correct_combined_percent"],
            "delta_vs_full_batch_10000_percentage_points": row[
                "delta_vs_full_batch_10000_percentage_points"
            ],
            "delta_vs_reference_archived_percentage_points": row[
                "delta_vs_reference_archived_percentage_points"
            ],
        }
        for rank, row in enumerate(ordered, start=1)
    ]


def _write_results(
    target: Path,
    *,
    analysis_plan_sha256: str,
    references: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    expected_runs: int,
    final: bool,
) -> dict[str, Any]:
    indices = [int(row["plan_index"]) for row in rows]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("result rows must be unique and in frozen plan order")
    if final and indices != list(range(1, expected_runs + 1)):
        raise ValueError("final result must contain every frozen sweep-plan row")

    analysis_failed = sum(row.get("status") == "failed" for row in rows)
    source_invalid = sum(
        row.get("status") == "source_invalid" for row in rows
    )
    source_not_completed = sum(
        row.get("status") == "source_not_completed" for row in rows
    )
    status = "running"
    if final:
        status = (
            "complete_with_failures"
            if analysis_failed or source_invalid or source_not_completed
            else "complete"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "analysis_plan_sha256": analysis_plan_sha256,
        "ordering": (
            "frozen sweep-plan order; results are not performance sorted"
        ),
        "checkpoint_selection": (
            "none; every source-completed valid run is analyzed"
        ),
        "basin_statistics_used_for_candidate_inclusion_or_order": False,
        "runs_expected": expected_runs,
        "rows_reported": len(rows),
        "source_completed_valid": sum(
            row.get("source_status") == "completed_valid" for row in rows
        ),
        "analysis_complete": sum(
            row.get("status") == "complete" for row in rows
        ),
        "analysis_failed": analysis_failed,
        "source_invalid": source_invalid,
        "source_not_completed": source_not_completed,
        "references": references,
        "results_in_frozen_plan_order": rows,
        "post_hoc_exploratory_ranking": {
            "generated_only_after_uniform_candidate_computation": final,
            "used_for_computation_candidate_inclusion_or_checkpoint_selection": False,
            "sort": (
                "descending correct_combined_percent; frozen plan index "
                "tie-break"
            ),
            "rows": _post_hoc_ranking(rows) if final else [],
        },
    }
    _write_json(target / "results_by_run.json", payload)
    _write_csv(target / "results_by_run.csv", rows)
    return payload


def run_batch_analysis(
    *,
    source_sweep: Path,
    analysis_root: Path,
    device_name: str,
    batch_points: int | str,
    train_data: Path | None = None,
    uniform_only: bool = False,
) -> dict[str, Any]:
    """Run the frozen by-candidate analysis without source mutation."""

    _assert_safe_target(source_sweep, analysis_root)
    source = _verify_source_sweep(source_sweep)
    inputs = _exact_inputs_for_training_data(train_data)
    _verify_analysis_training_input(source, inputs)
    references = _reference_inventory()
    device = study._resolve_device(device_name)

    target = analysis_root.resolve()
    target.mkdir(parents=False, exist_ok=False)
    snapshot_root = _snapshot_source(
        target=target,
        source=source,
        references=references,
    )
    plan = _analysis_plan(
        target=target,
        source=source,
        snapshot_root=snapshot_root,
        references=references,
        inputs=inputs,
        device=device,
        batch_points=batch_points,
        uniform_only=uniform_only,
    )
    plan_path = _write_json(target / "analysis_plan.json", plan)
    plan_hash = study.sha256_file(plan_path)
    batch_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": _utc_now(),
        "analysis_plan_sha256": plan_hash,
        "runs_expected": len(source.inventory),
        "source_completed_valid": len(source.candidates),
        "rows_processed": 0,
    }
    _write_json(target / "batch_manifest.json", batch_manifest)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for item in source.inventory:
        if item.candidate is None:
            row = _source_issue_row(item)
            _add_reference_deltas(row, references)
        else:
            row = _analyze_candidate(
                target=target,
                snapshot_root=snapshot_root,
                candidate=item.candidate,
                analysis_plan_sha256=plan_hash,
                inputs=inputs,
                device=device,
                batch_points=batch_points,
                references=references,
                uniform_only=uniform_only,
            )
        rows.append(row)
        _write_results(
            target,
            analysis_plan_sha256=plan_hash,
            references=references,
            rows=rows,
            expected_runs=len(source.inventory),
            final=False,
        )
        batch_manifest["rows_processed"] = len(rows)
        batch_manifest["analysis_failed"] = sum(
            current["status"] == "failed" for current in rows
        )
        _write_json(target / "batch_manifest.json", batch_manifest)

    results = _write_results(
        target,
        analysis_plan_sha256=plan_hash,
        references=references,
        rows=rows,
        expected_runs=len(source.inventory),
        final=True,
    )
    batch_manifest.update(
        {
            "status": results["status"],
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "rows_processed": len(rows),
            "analysis_complete": results["analysis_complete"],
            "analysis_failed": results["analysis_failed"],
            "source_invalid": results["source_invalid"],
            "source_not_completed": results["source_not_completed"],
            "results_json": "results_by_run.json",
            "results_csv": "results_by_run.csv",
        }
    )
    _write_json(target / "batch_manifest.json", batch_manifest)
    return results


def _batch_points(value: str) -> int | str:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "batch points must be positive or 'auto'"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sweep", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-points", type=_batch_points, default="auto")
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help=(
            "root of the archived reference inputs "
            "(train_data.csv, traj_attractors.pkl, stable_solutions.csv, "
            "Readme.txt)"
        ),
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        help=(
            "training CSV frozen by the source sweep; the fixed trajectory "
            "truth and stable roots still come from the canonical archive"
        ),
    )
    parser.add_argument(
        "--uniform-only",
        action="store_true",
        help=(
            "stop after the validated level-8 uniform graph/statistics; this "
            "is the exact archived comparison protocol"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    global REFERENCE_ROOT, REFERENCE_README
    REFERENCE_ROOT = args.reference_root.resolve()
    REFERENCE_README = REFERENCE_ROOT / "Readme.txt"
    results = run_batch_analysis(
        source_sweep=args.source_sweep,
        analysis_root=args.analysis_root,
        device_name=args.device,
        batch_points=args.batch_points,
        train_data=args.train_data,
        uniform_only=args.uniform_only,
    )
    print(
        f"batch_analysis_status={results['status']} "
        f"complete={results['analysis_complete']} "
        f"failed={results['analysis_failed']} "
        f"source_invalid={results['source_invalid']} "
        f"source_not_completed={results['source_not_completed']} "
        f"output={args.analysis_root.resolve()}",
        flush=True,
    )
    return 0 if results["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
