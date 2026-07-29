"""Analyze every frozen checkpoint from the long 1-D accumulation run.

The source training run is read-only.  Before any basin computation starts,
this script verifies the completed training contract, inventories all 20
predeclared milestones, and copies their exact bytes into a fresh sibling
diagnostics directory.  The resulting ``analysis_plan.json`` is immutable and
binds the chronological analysis to those hashes.

Each milestone is then evaluated independently with the exact archived
Chafee--Infante inputs and the same 1-D bounds, lookup-only CMGDB, adaptive
graph, and basin-statistics stages used by ``chafee_latent_dimension_study``.
Failures are recorded and do not prevent later milestones from being tested.
No checkpoint is selected using basin information.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

import analyze_chafee_d1_checkpoint as single
import chafee_latent_dimension_study as study
from latentdynamics.training import load_checkpoint

CODE_ROOT = Path(__file__).resolve().parents[1]
LATENT_1D_ROOT = (
    CODE_ROOT / "output" / "chafee_latent_dimension_study" / "latent_1d"
)
DEFAULT_SOURCE = LATENT_1D_ROOT / "seed_0_gradaccum_b1024_epoch_20000"
DEFAULT_ANALYSIS = (
    LATENT_1D_ROOT / "seed_0_gradaccum_b1024_epoch_20000_roa_milestones"
)

MILESTONE_EPOCHS = tuple(range(1_000, 20_001, 1_000))
STAGES = single.STAGES
EXPECTED_SETTINGS: dict[str, Any] = {
    "seed": 0,
    "microbatch_size": 1_024,
    "effective_batch_size": 30_000,
    "learning_rate": 0.003,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.0,
    "epochs": 20_000,
    "scheduler_factor": 0.5,
    "scheduler_patience": 100,
    "scheduler_threshold": 1e-4,
    "scheduler_min_lr": 1e-6,
    "resume_interval": 250,
    "milestone_interval": 1_000,
}
REFERENCE_PATHS = {
    "full_batch_4000": single.CANONICAL_4K / "basin_statistics.json",
    "full_batch_10000": single.CANONICAL_10K / "basin_statistics.json",
    "minibatch_adam_b1024_lr1e3": single.DEFAULT_ANALYSIS
    / "basin_statistics.json",
}
COUNT_KEYS = {
    "negative": "correctly_classified_in_negative_basin",
    "positive": "correctly_classified_in_positive_basin",
    "misclassified_negative": "misclassified_in_negative_basin",
    "misclassified_positive": "misclassified_in_positive_basin",
    "outside": "outside_both_basins",
}
REFERENCE_NAMES = tuple(REFERENCE_PATHS)
CSV_FIELDS = (
    "epoch",
    "status",
    "checkpoint_sha256",
    "loss_reconstruction",
    "loss_prediction",
    "loss_total",
    "learning_rate",
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
    *tuple(
        field
        for name in REFERENCE_NAMES
        for field in (
            f"delta_vs_{name}_percentage_points",
            f"beats_{name}",
        )
    ),
    "analysis_directory",
    "error_stage",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class FrozenMilestone:
    epoch: int
    manifest: Path
    manifest_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    sidecar: Path
    sidecar_sha256: str
    train: dict[str, float]
    learning_rate: float

    def plan_record(self, *, source_run: Path, snapshot_root: Path) -> dict[str, Any]:
        snapshot = snapshot_root / f"epoch_{self.epoch:05d}"
        return {
            "epoch": self.epoch,
            "source_manifest": str(self.manifest.relative_to(source_run)),
            "source_manifest_sha256": self.manifest_sha256,
            "source_checkpoint": str(self.checkpoint.relative_to(source_run)),
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_sidecar": str(self.sidecar.relative_to(source_run)),
            "sidecar_sha256": self.sidecar_sha256,
            "snapshot_manifest": str(
                (snapshot / "manifest.json").relative_to(snapshot_root.parent)
            ),
            "snapshot_checkpoint": str(
                (snapshot / "models" / "autoencoder.pt").relative_to(
                    snapshot_root.parent
                )
            ),
            "snapshot_sidecar": str(
                (snapshot / "models" / "autoencoder.json").relative_to(
                    snapshot_root.parent
                )
            ),
            "train": self.train,
            "learning_rate": self.learning_rate,
        }


@dataclass(frozen=True)
class FrozenSource:
    run: Path
    run_plan: Path
    run_plan_sha256: str
    training_summary: Path
    training_summary_sha256: str
    completion: Path
    completion_sha256: str
    milestones: tuple[FrozenMilestone, ...]


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assert_safe_target(source_run: Path, analysis_root: Path) -> None:
    source = source_run.resolve()
    target = analysis_root.resolve()
    if target.parent != source.parent:
        raise ValueError(
            "analysis target must be an isolated sibling of the source run"
        )
    protected = (
        source,
        single.CANONICAL_4K.resolve(),
        single.CANONICAL_10K.resolve(),
        single.DEFAULT_SOURCE.resolve(),
        single.DEFAULT_ANALYSIS.resolve(),
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


def _declared_file(run: Path, value: Any, *, expected: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing declared path for {expected.relative_to(run)}")
    declared = Path(value)
    if declared.is_absolute():
        raise ValueError(f"declared source path must be relative: {declared}")
    resolved = (run / declared).resolve()
    if not _is_within(resolved, run):
        raise ValueError(f"declared source path escapes the run: {declared}")
    if resolved != expected.resolve():
        raise ValueError(f"unexpected declared source path: {declared}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _verify_run_plan_payload(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported gradient-accumulation run-plan schema")
    if plan.get("status") != "frozen_before_training":
        raise ValueError("run plan was not frozen before training")
    if plan.get("purpose") != (
        "long fixed-horizon 1-D gradient-accumulation experiment"
    ):
        raise ValueError("source run has an unexpected purpose")
    if plan.get("settings") != EXPECTED_SETTINGS:
        raise ValueError("source run settings differ from the fixed protocol")
    if plan.get("milestone_epochs") != list(MILESTONE_EPOCHS):
        raise ValueError("source did not predeclare the exact 20 milestones")
    architecture = plan.get("architecture", {})
    if (
        architecture.get("high_dims") != 64
        or architecture.get("low_dims") != 1
    ):
        raise ValueError("source architecture is not the 64-to-1 benchmark")
    data = plan.get("data", {})
    if (
        data.get("sha256") != single.EXPECTED_TRAINING_SHA256
        or data.get("shape") != [30_000, 128]
        or data.get("scaling") != "none"
        or data.get("shuffle") is not False
        or data.get("drop_last") is not False
    ):
        raise ValueError("source training data contract is not exact")
    if plan.get("objective") != {
        "formula": "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)",
        "weights": [1.0, 1.0, 0.0],
    }:
        raise ValueError("source objective differs from the fixed protocol")
    if (
        plan.get("early_stopping") is not False
        or plan.get("validation_used") is not False
        or plan.get("basin_or_cmgdb_inputs_allowed") is not False
    ):
        raise ValueError("source run violated the no-selection protocol")
    accumulation = plan.get("accumulation", {})
    if (
        accumulation.get("microbatches_per_update") != 30
        or accumulation.get("last_microbatch_rows") != 304
        or accumulation.get("optimizer_steps_per_epoch") != 1
    ):
        raise ValueError("source accumulation contract is not exact")


def _verify_model(checkpoint: Path) -> None:
    model, arch = load_checkpoint(
        checkpoint.parent,
        basename="autoencoder",
        map_location="cpu",
    )
    if arch != study.marcio_architecture(1):
        raise ValueError(f"{checkpoint} has the wrong architecture")
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


def _verify_milestone(
    run: Path,
    *,
    epoch: int,
    run_plan_sha256: str,
    verify_model: bool = True,
) -> FrozenMilestone:
    milestone = run / "milestones" / f"epoch_{epoch:05d}"
    manifest_path = milestone / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("epoch") != epoch
        or manifest.get("optimizer_updates") != epoch
        or manifest.get("run_plan_sha256") != run_plan_sha256
        or manifest.get("basin_artifacts_accessed") is not False
    ):
        raise ValueError(f"milestone {epoch} does not match the frozen protocol")
    checkpoint_record = manifest.get("checkpoint", {})
    checkpoint = _declared_file(
        run,
        checkpoint_record.get("path"),
        expected=milestone / "models" / "autoencoder.pt",
    )
    sidecar = _declared_file(
        run,
        checkpoint_record.get("sidecar_path"),
        expected=milestone / "models" / "autoencoder.json",
    )
    checkpoint_hash = study.sha256_file(checkpoint)
    sidecar_hash = study.sha256_file(sidecar)
    if checkpoint_record.get("sha256") != checkpoint_hash:
        raise ValueError(f"milestone {epoch} checkpoint hash mismatch")
    if checkpoint_record.get("sidecar_sha256") != sidecar_hash:
        raise ValueError(f"milestone {epoch} sidecar hash mismatch")
    train_record = manifest.get("train", {})
    required_metrics = (
        "loss_reconstruction",
        "loss_prediction",
        "loss_total",
    )
    try:
        train = {key: float(train_record[key]) for key in required_metrics}
        learning_rate = float(manifest["learning_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"milestone {epoch} has invalid training metrics") from error
    if (
        not all(math.isfinite(value) and value >= 0.0 for value in train.values())
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
    ):
        raise ValueError(f"milestone {epoch} has non-finite training metrics")
    if verify_model:
        _verify_model(checkpoint)
    return FrozenMilestone(
        epoch=epoch,
        manifest=manifest_path.resolve(),
        manifest_sha256=study.sha256_file(manifest_path),
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        sidecar=sidecar,
        sidecar_sha256=sidecar_hash,
        train=train,
        learning_rate=learning_rate,
    )


def _verify_completed_source(source_run: Path) -> FrozenSource:
    run = source_run.resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    run_plan = run / "run_plan.json"
    training_summary = run / "training_summary.json"
    completion = run / "completion.json"
    plan = _read_json(run_plan)
    summary = _read_json(training_summary)
    completed = _read_json(completion)
    _verify_run_plan_payload(plan)
    plan_hash = study.sha256_file(run_plan)

    training_script = plan.get("training_script", {})
    script_path = Path(str(training_script.get("path", ""))).resolve()
    if (
        not script_path.is_file()
        or study.sha256_file(script_path) != training_script.get("sha256")
    ):
        raise ValueError("recorded training script no longer matches its frozen hash")

    summary_hash = study.sha256_file(training_summary)
    if (
        summary.get("schema_version") != 1
        or summary.get("training_method") != "marcio_gradient_accumulation"
        or summary.get("run_plan_sha256") != plan_hash
        or summary.get("settings") != EXPECTED_SETTINGS
        or summary.get("epochs_requested") != 20_000
        or summary.get("epochs_completed") != 20_000
        or summary.get("optimizer_updates") != 20_000
        or summary.get("early_stopping_used") is not False
        or summary.get("validation_used") is not False
        or summary.get("basin_artifacts_accessed") is not False
    ):
        raise ValueError("training summary does not certify the completed protocol")
    if (
        completed.get("schema_version") != 1
        or completed.get("status") != "complete"
        or completed.get("run_plan_sha256") != plan_hash
        or completed.get("training_summary_sha256") != summary_hash
        or completed.get("epochs_completed") != 20_000
    ):
        raise ValueError("completion record does not certify all 20,000 updates")

    artifacts = summary.get("artifacts", {})
    final_checkpoint = _declared_file(
        run,
        artifacts.get("checkpoint"),
        expected=run / "models" / "autoencoder.pt",
    )
    final_sidecar = _declared_file(
        run,
        artifacts.get("sidecar"),
        expected=run / "models" / "autoencoder.json",
    )
    if (
        study.sha256_file(final_checkpoint) != artifacts.get("checkpoint_sha256")
        or study.sha256_file(final_sidecar) != artifacts.get("sidecar_sha256")
        or study.sha256_file(final_checkpoint)
        != completed.get("final_checkpoint_sha256")
    ):
        raise ValueError("final checkpoint does not match completion manifests")

    milestones = tuple(
        _verify_milestone(
            run,
            epoch=epoch,
            run_plan_sha256=plan_hash,
        )
        for epoch in MILESTONE_EPOCHS
    )
    if tuple(item.epoch for item in milestones) != MILESTONE_EPOCHS:
        raise ValueError("milestone inventory is not chronological and complete")
    return FrozenSource(
        run=run,
        run_plan=run_plan,
        run_plan_sha256=plan_hash,
        training_summary=training_summary,
        training_summary_sha256=summary_hash,
        completion=completion,
        completion_sha256=study.sha256_file(completion),
        milestones=milestones,
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
        "conditioned_trajectories": int(
            statistics["conditioned_trajectories"]
        ),
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


def _reference_inventory() -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for name, path in REFERENCE_PATHS.items():
        payload = single._validate_statistics(path)
        references[name] = {
            "path": str(path.resolve()),
            "sha256": study.sha256_file(path),
            **_stats_fields(payload),
        }
    return references


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> Path:
    if study.sha256_file(source) != expected_sha256:
        raise ValueError(f"frozen source changed before snapshot: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if study.sha256_file(target) != expected_sha256:
        raise ValueError(f"snapshot copy hash changed: {target}")
    return target


def _snapshot_source(
    target: Path,
    source: FrozenSource,
    references: dict[str, dict[str, Any]],
) -> Path:
    snapshot_root = target / "source_snapshot"
    global_files = (
        (source.run_plan, source.run_plan_sha256),
        (source.training_summary, source.training_summary_sha256),
        (source.completion, source.completion_sha256),
    )
    for path, digest in global_files:
        _copy_verified(path, snapshot_root / path.name, digest)
    for name, record in references.items():
        reference_path = Path(record["path"])
        _copy_verified(
            reference_path,
            snapshot_root / "references" / f"{name}.json",
            str(record["sha256"]),
        )
    for milestone in source.milestones:
        destination = snapshot_root / f"epoch_{milestone.epoch:05d}"
        _copy_verified(
            milestone.manifest,
            destination / "manifest.json",
            milestone.manifest_sha256,
        )
        _copy_verified(
            milestone.checkpoint,
            destination / "models" / "autoencoder.pt",
            milestone.checkpoint_sha256,
        )
        _copy_verified(
            milestone.sidecar,
            destination / "models" / "autoencoder.json",
            milestone.sidecar_sha256,
        )
    return snapshot_root


def _analysis_plan(
    *,
    target: Path,
    source: FrozenSource,
    snapshot_root: Path,
    inputs: study.ExactInputs,
    references: dict[str, dict[str, Any]],
    device: torch.device,
    batch_points: int | str,
) -> dict[str, Any]:
    scripts = {
        "batch_analyzer": Path(__file__).resolve(),
        "single_checkpoint_analyzer": Path(single.__file__).resolve(),
        "study_driver": Path(study.__file__).resolve(),
    }
    return {
        "schema_version": 1,
        "status": "frozen_before_analysis",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_run": str(source.run),
        "source_manifests": {
            "run_plan": {
                "path": str(source.run_plan),
                "sha256": source.run_plan_sha256,
            },
            "training_summary": {
                "path": str(source.training_summary),
                "sha256": source.training_summary_sha256,
            },
            "completion": {
                "path": str(source.completion),
                "sha256": source.completion_sha256,
            },
        },
        "source_snapshot": str(snapshot_root.relative_to(target)),
        "checkpoint_inventory": [
            milestone.plan_record(
                source_run=source.run,
                snapshot_root=snapshot_root,
            )
            for milestone in source.milestones
        ],
        "analysis_protocol": {
            "checkpoint_order": list(MILESTONE_EPOCHS),
            "checkpoint_selection": "none; analyze every predeclared milestone",
            "basin_statistics_used_for_selection": False,
            "dimension": 1,
            "seed": 0,
            "device": str(device),
            "batch_points": batch_points,
            "stages": list(STAGES),
            "cmgdb_uniform_subdivision": 8,
            "cmgdb_adaptive_subdivision": 11,
            "continue_after_checkpoint_failure": True,
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


def _base_row(
    milestone: FrozenMilestone,
    *,
    analysis_directory: Path,
) -> dict[str, Any]:
    return {
        "epoch": milestone.epoch,
        "status": "running",
        "checkpoint_sha256": milestone.checkpoint_sha256,
        **milestone.train,
        "learning_rate": milestone.learning_rate,
        "analysis_directory": str(analysis_directory),
        "error_stage": "",
        "error_type": "",
        "error_message": "",
    }


def _add_reference_deltas(
    row: dict[str, Any],
    references: dict[str, dict[str, Any]],
) -> None:
    score = row.get("correct_combined_percent")
    for name, reference in references.items():
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


def _analyze_milestone(
    *,
    target: Path,
    snapshot_root: Path,
    milestone: FrozenMilestone,
    analysis_plan_sha256: str,
    inputs: study.ExactInputs,
    device: torch.device,
    batch_points: int | str,
    references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run = target / f"epoch_{milestone.epoch:05d}"
    run.mkdir(parents=False, exist_ok=False)
    row = _base_row(milestone, analysis_directory=run)
    manifest_path = run / "analysis_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "epoch": milestone.epoch,
        "analysis_plan_sha256": analysis_plan_sha256,
        "source_checkpoint_sha256": milestone.checkpoint_sha256,
        "completed_stages": [],
    }
    _write_json(manifest_path, manifest)
    active_stage = "checkpoint_copy"
    started = time.perf_counter()
    try:
        snapshot = snapshot_root / f"epoch_{milestone.epoch:05d}"
        _copy_verified(
            snapshot / "models" / "autoencoder.pt",
            run / "models" / "autoencoder.pt",
            milestone.checkpoint_sha256,
        )
        _copy_verified(
            snapshot / "models" / "autoencoder.json",
            run / "models" / "autoencoder.json",
            milestone.sidecar_sha256,
        )
        _copy_verified(
            snapshot / "manifest.json",
            run / "source_milestone_manifest.json",
            milestone.manifest_sha256,
        )

        paths = single.ExactRunPaths(output_root=run, dimension=1)
        runners = (
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
                lambda: study._run_uniform(
                    paths,
                    inputs,
                    device=device,
                ),
            ),
            (
                "precompute-fine",
                lambda: study._run_precompute_fine(
                    paths,
                    device=device,
                    batch_points=batch_points,
                ),
            ),
            (
                "adaptive",
                lambda: study._run_adaptive(paths, topology_only=False),
            ),
            ("stats", lambda: study._run_statistics(paths)),
        )
        for active_stage, runner in runners:
            print(
                f"epoch={milestone.epoch:05d} start_stage={active_stage}",
                flush=True,
            )
            runner()
            manifest["completed_stages"].append(active_stage)
            _write_json(manifest_path, manifest)

        active_stage = "validate-statistics"
        statistics = single._validate_statistics(paths.stats)
        row.update(_stats_fields(statistics))
        row["status"] = "complete"
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(UTC).isoformat(),
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
                },
            }
        )
    except Exception as error:
        stats_path = run / "basin_statistics.json"
        if stats_path.is_file():
            try:
                row.update(_stats_fields(_read_json(stats_path)))
            except (KeyError, TypeError, ValueError):
                pass
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
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "error_stage": active_stage,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
    _add_reference_deltas(row, references)
    manifest["output_files"] = single._file_manifest(run)
    _write_json(manifest_path, manifest)
    print(
        f"epoch={milestone.epoch:05d} analysis_status={row['status']}",
        flush=True,
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
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


def _write_comparison(
    target: Path,
    *,
    analysis_plan_sha256: str,
    references: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    final: bool,
) -> dict[str, Any]:
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != sorted(epochs) or len(epochs) != len(set(epochs)):
        raise ValueError("comparison rows must be unique and chronological")
    failed = sum(row.get("status") == "failed" for row in rows)
    status = "running"
    if final:
        if tuple(epochs) != MILESTONE_EPOCHS:
            raise ValueError("final comparison must contain all 20 milestones")
        status = "complete_with_failures" if failed else "complete"
    payload = {
        "schema_version": 1,
        "status": status,
        "analysis_plan_sha256": analysis_plan_sha256,
        "checkpoint_selection": "none; chronological all-milestone report",
        "basin_statistics_used_for_selection": False,
        "milestones_expected": len(MILESTONE_EPOCHS),
        "milestones_processed": len(rows),
        "milestones_complete": sum(
            row.get("status") == "complete" for row in rows
        ),
        "milestones_failed": failed,
        "references": references,
        "results": rows,
    }
    _write_json(target / "comparison.json", payload)
    _write_csv(target / "comparison.csv", rows)
    return payload


def run_batch_analysis(
    *,
    source_run: Path,
    analysis_root: Path,
    device_name: str,
    batch_points: int | str,
) -> dict[str, Any]:
    _assert_safe_target(source_run, analysis_root)
    source = _verify_completed_source(source_run)
    inputs = study.verify_exact_inputs(study.DEFAULT_ARCHIVE_DIR)
    references = _reference_inventory()
    device = study._resolve_device(device_name)

    target = analysis_root.resolve()
    target.mkdir(parents=False, exist_ok=False)
    snapshot_root = _snapshot_source(target, source, references)
    plan = _analysis_plan(
        target=target,
        source=source,
        snapshot_root=snapshot_root,
        inputs=inputs,
        references=references,
        device=device,
        batch_points=batch_points,
    )
    plan_path = _write_json(target / "analysis_plan.json", plan)
    plan_hash = study.sha256_file(plan_path)
    batch_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "analysis_plan_sha256": plan_hash,
        "milestones_expected": len(MILESTONE_EPOCHS),
        "milestones_processed": 0,
    }
    _write_json(target / "batch_manifest.json", batch_manifest)

    previous_max_vertices = os.environ.get("CMGDB_MAPGRAPH_MAX_VERTICES")
    os.environ["CMGDB_MAPGRAPH_MAX_VERTICES"] = str(2**24)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for milestone in source.milestones:
            row = _analyze_milestone(
                target=target,
                snapshot_root=snapshot_root,
                milestone=milestone,
                analysis_plan_sha256=plan_hash,
                inputs=inputs,
                device=device,
                batch_points=batch_points,
                references=references,
            )
            rows.append(row)
            _write_comparison(
                target,
                analysis_plan_sha256=plan_hash,
                references=references,
                rows=rows,
                final=False,
            )
            batch_manifest["milestones_processed"] = len(rows)
            batch_manifest["milestones_failed"] = sum(
                item["status"] == "failed" for item in rows
            )
            _write_json(target / "batch_manifest.json", batch_manifest)
    finally:
        if previous_max_vertices is None:
            os.environ.pop("CMGDB_MAPGRAPH_MAX_VERTICES", None)
        else:
            os.environ["CMGDB_MAPGRAPH_MAX_VERTICES"] = previous_max_vertices

    comparison = _write_comparison(
        target,
        analysis_plan_sha256=plan_hash,
        references=references,
        rows=rows,
        final=True,
    )
    batch_manifest.update(
        {
            "status": comparison["status"],
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "milestones_processed": len(rows),
            "milestones_complete": comparison["milestones_complete"],
            "milestones_failed": comparison["milestones_failed"],
            "comparison_json": "comparison.json",
            "comparison_csv": "comparison.csv",
        }
    )
    _write_json(target / "batch_manifest.json", batch_manifest)
    return comparison


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
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-points", type=_batch_points, default="auto")
    return parser


def main() -> int:
    args = _parser().parse_args()
    comparison = run_batch_analysis(
        source_run=args.source_run,
        analysis_root=args.analysis_root,
        device_name=args.device,
        batch_points=args.batch_points,
    )
    print(
        f"batch_analysis_status={comparison['status']} "
        f"complete={comparison['milestones_complete']} "
        f"failed={comparison['milestones_failed']} "
        f"output={args.analysis_root.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
