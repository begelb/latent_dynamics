"""Run the exact 1-D Chafee--Infante RoA analysis on a frozen checkpoint.

The selected checkpoint is verified against its pre-basin selection record,
copied byte-for-byte into a fresh analysis directory, and only then exposed to
the archived trajectory labels and CMGDB stages.  Canonical full-batch runs and
the source training run are treated as immutable protected directories.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import chafee_latent_dimension_study as study
from latentdynamics.training import load_checkpoint

CODE_ROOT = Path(__file__).resolve().parents[1]
LATENT_1D_ROOT = (
    CODE_ROOT / "output" / "chafee_latent_dimension_study" / "latent_1d"
)
CANONICAL_4K = LATENT_1D_ROOT / "seed_0"
CANONICAL_10K = LATENT_1D_ROOT / "seed_0_epoch_10000"
DEFAULT_SOURCE = LATENT_1D_ROOT / "seed_0_minibatch_b1024_lr1e3"
DEFAULT_ANALYSIS = LATENT_1D_ROOT / "seed_0_minibatch_b1024_lr1e3_roa"

EXPECTED_TRAINING_SHA256 = (
    "890fea29a2d26b31b44bbc4fbd773af0ebf742b08893c27ddaf1bdbf87e30329"
)
EXPECTED_VALIDATION_SHA256 = (
    "957b7fe13d03550d88f7fd4845c0870af890b248ef91fab26950a61c5b9b10a3"
)
STAGES = (
    "bounds",
    "precompute-coarse",
    "uniform",
    "precompute-fine",
    "adaptive",
    "stats",
)


@dataclass(frozen=True)
class FrozenSource:
    run: Path
    checkpoint: Path
    sidecar: Path
    selected_epoch: int
    checkpoint_sha256: str
    sidecar_sha256: str
    run_plan: Path
    run_plan_sha256: str
    selection_record: Path
    selection_record_sha256: str
    training_summary: Path
    training_summary_sha256: str


class ExactRunPaths(study.DimensionPaths):
    """Route existing 1-D stage helpers into one exact analysis directory."""

    @property
    def run(self) -> Path:
        return self.output_root


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


def _assert_safe_target(source: Path, target: Path) -> None:
    source_resolved = source.resolve()
    target_resolved = target.resolve()
    protected = (
        source_resolved,
        CANONICAL_4K.resolve(),
        CANONICAL_10K.resolve(),
    )
    for root in protected:
        if (
            target_resolved == root
            or _is_within(target_resolved, root)
            or _is_within(root, target_resolved)
        ):
            raise ValueError(
                f"analysis target {target_resolved} overlaps protected directory {root}"
            )
    if target.is_symlink():
        raise ValueError(f"analysis target must not be a symlink: {target}")
    if target.exists():
        raise FileExistsError(
            f"analysis target already exists; refusing to overwrite: {target}"
        )


def _verified_frozen_source(source_run: Path) -> FrozenSource:
    run = source_run.resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    run_plan_path = run / "run_plan.json"
    selection_path = run / "selection_record.json"
    summary_path = run / "training_summary.json"
    run_plan = _read_json(run_plan_path)
    selection = _read_json(selection_path)
    summary = _read_json(summary_path)

    plan_hash = study.sha256_file(run_plan_path)
    if summary.get("run_plan_sha256") != plan_hash:
        raise ValueError("training summary does not match frozen run_plan.json")
    if selection.get("run_plan_sha256") != plan_hash:
        raise ValueError("selection record does not match frozen run_plan.json")
    if run_plan.get("status") != "frozen_before_training":
        raise ValueError("run plan was not frozen before training")
    if (
        selection.get("basin_artifacts_accessed_before_selection_freeze")
        is not False
    ):
        raise ValueError("selection record does not exclude pre-selection basin access")
    if selection.get("selected_basename") != "selected":
        raise ValueError("selection record must freeze basename 'selected'")
    if summary.get("training_method") != "marcio_seeded_minibatch":
        raise ValueError("source is not the expected mini-batch training method")
    settings = summary.get("settings", {})
    expected_settings = {
        "seed": 0,
        "batch_size": 1024,
        "learning_rate": 1e-3,
    }
    for key, expected in expected_settings.items():
        if settings.get(key) != expected:
            raise ValueError(
                f"unexpected mini-batch setting {key}: "
                f"{settings.get(key)!r} != {expected!r}"
            )
    data = summary.get("data", {})
    if (
        data.get("scaling") != "none"
        or data.get("shuffle") is not True
        or data.get("drop_last") is not False
        or data.get("training_pairs") != 30_000
    ):
        raise ValueError("source data/batching contract is not the frozen protocol")
    sources = summary.get("sources", {})
    if (
        sources.get("training_data", {}).get("sha256")
        != EXPECTED_TRAINING_SHA256
        or sources.get("validation_data", {}).get("sha256")
        != EXPECTED_VALIDATION_SHA256
    ):
        raise ValueError("source train/validation hashes do not match the run contract")

    selected = selection.get("selected_checkpoint", {})
    checkpoint = run / str(selected.get("path", ""))
    sidecar = run / str(selected.get("sidecar_path", ""))
    checkpoint_hash = study.sha256_file(checkpoint)
    sidecar_hash = study.sha256_file(sidecar)
    if checkpoint_hash != selected.get("sha256"):
        raise ValueError("selected checkpoint hash does not match selection record")
    if sidecar_hash != selected.get("sidecar_sha256"):
        raise ValueError("selected sidecar hash does not match selection record")
    if summary.get("artifacts", {}).get("primary_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("training summary and selection record disagree on checkpoint")

    model, arch = load_checkpoint(
        checkpoint.parent,
        basename="selected",
        map_location="cpu",
    )
    if arch.high_dims != 64 or arch.low_dims != 1:
        raise ValueError(f"selected architecture is {arch.high_dims}->{arch.low_dims}")
    if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise ValueError("selected checkpoint contains non-finite parameters")
    model.eval()
    with torch.inference_mode():
        probe = torch.zeros((2, arch.high_dims), dtype=torch.float32)
        encoded = model.encoder(probe)
        mapped = model.latent_map(encoded)
        decoded = model.decoder(mapped)
    if not all(torch.isfinite(value).all().item() for value in (encoded, mapped, decoded)):
        raise ValueError("selected checkpoint produces non-finite probe outputs")

    selected_epoch = int(selected.get("epoch", -1))
    if selected_epoch != int(summary.get("checkpoint_epoch", -2)):
        raise ValueError("selected epoch disagrees between source manifests")
    return FrozenSource(
        run=run,
        checkpoint=checkpoint,
        sidecar=sidecar,
        selected_epoch=selected_epoch,
        checkpoint_sha256=checkpoint_hash,
        sidecar_sha256=sidecar_hash,
        run_plan=run_plan_path,
        run_plan_sha256=plan_hash,
        selection_record=selection_path,
        selection_record_sha256=study.sha256_file(selection_path),
        training_summary=summary_path,
        training_summary_sha256=study.sha256_file(summary_path),
    )


def _validate_statistics(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    statistics = payload.get("statistics", {})
    counts = statistics.get("counts", {})
    percentages = statistics.get("percentages", {})
    if (
        statistics.get("total_trajectories") != 10_000
        or statistics.get("excluded_zero_trajectories") != 2_138
        or statistics.get("conditioned_trajectories") != 7_862
    ):
        raise ValueError("basin-statistics denominators do not match the fixed benchmark")
    if sum(int(value) for value in counts.values()) != 7_862:
        raise ValueError("basin counts do not conserve the conditioned trajectories")
    if not np.isclose(
        sum(float(value) for value in percentages.values()),
        100.0,
        rtol=0.0,
        atol=1e-9,
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


def run_analysis(
    *,
    source_run: Path,
    analysis_run: Path,
    device_name: str,
    batch_points: int | str,
) -> dict[str, Any]:
    _assert_safe_target(source_run, analysis_run)
    source = _verified_frozen_source(source_run)
    inputs = study.verify_exact_inputs(study.DEFAULT_ARCHIVE_DIR)
    device = study._resolve_device(device_name)
    target = analysis_run.resolve()
    target.mkdir(parents=True, exist_ok=False)
    manifest_path = target / "analysis_manifest.json"
    started = time_started = datetime.now(UTC)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": started.isoformat(),
        "source_run": str(source.run),
        "selected_epoch": source.selected_epoch,
        "selected_checkpoint": {
            "path": str(source.checkpoint),
            "sha256": source.checkpoint_sha256,
            "sidecar_path": str(source.sidecar),
            "sidecar_sha256": source.sidecar_sha256,
        },
        "source_manifests": {
            "run_plan_sha256": source.run_plan_sha256,
            "selection_record_sha256": source.selection_record_sha256,
            "training_summary_sha256": source.training_summary_sha256,
        },
        "analysis": {
            "device": str(device),
            "batch_points": batch_points,
            "stages": list(STAGES),
            "dimension": 1,
            "seed": 0,
            "cmgdb_uniform_subdivision": 8,
            "cmgdb_adaptive_subdivision": 11,
        },
        "archived_inputs": inputs.provenance(),
    }
    _write_json(manifest_path, manifest)

    try:
        model_dir = target / "models"
        model_dir.mkdir(parents=True)
        copied_checkpoint = model_dir / "autoencoder.pt"
        copied_sidecar = model_dir / "autoencoder.json"
        shutil.copy2(source.checkpoint, copied_checkpoint)
        shutil.copy2(source.sidecar, copied_sidecar)
        if study.sha256_file(copied_checkpoint) != source.checkpoint_sha256:
            raise ValueError("copied checkpoint hash changed")
        if study.sha256_file(copied_sidecar) != source.sidecar_sha256:
            raise ValueError("copied sidecar hash changed")

        source_manifest_dir = target / "source_manifests"
        source_manifest_dir.mkdir()
        for path in (
            source.run_plan,
            source.selection_record,
            source.training_summary,
        ):
            shutil.copy2(path, source_manifest_dir / path.name)

        paths = ExactRunPaths(output_root=target, dimension=1)
        os.environ["CMGDB_MAPGRAPH_MAX_VERTICES"] = str(2**24)
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
                lambda: study._run_uniform(paths, inputs, device=device),
            ),
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
        completed_stages: list[str] = []
        for name, runner in runners:
            print(f"start_stage={name}", flush=True)
            runner()
            completed_stages.append(name)
            manifest["completed_stages"] = completed_stages
            _write_json(manifest_path, manifest)
            print(f"completed_stage={name}", flush=True)

        statistics = _validate_statistics(paths.stats)
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": (
                    datetime.now(UTC) - time_started
                ).total_seconds(),
                "basin_statistics": statistics["statistics"],
                "bistability": {
                    "uniform": statistics["uniform_is_bistable"],
                    "adaptive_graph": statistics["adaptive_graph"],
                    "roots_distinct": statistics[
                        "roots_define_two_distinct_attractor_basins"
                    ],
                    "eligible": statistics[
                        "eligible_for_bistable_dimension_table"
                    ],
                },
                "output_files": _file_manifest(target),
            }
        )
        _write_json(manifest_path, manifest)
        return manifest
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        _write_json(manifest_path, manifest)
        raise


def _batch_points(value: str) -> int | str:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("batch points must be positive or 'auto'")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--analysis-run", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-points", type=_batch_points, default="auto")
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = run_analysis(
        source_run=args.source_run,
        analysis_run=args.analysis_run,
        device_name=args.device,
        batch_points=args.batch_points,
    )
    print(
        f"analysis_status={manifest['status']} output={args.analysis_run.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
