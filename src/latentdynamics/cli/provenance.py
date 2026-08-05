"""Run provenance manifests for pipeline cells."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from latentdynamics import __version__

from ..config import ExperimentConfig


def _canonical_config(cfg: ExperimentConfig) -> dict[str, Any]:
    return cfg.model_dump(mode="json")


def hash_config_dict(config: dict[str, Any]) -> str:
    """Hash an already-serialized config dict.

    Stamping uses :func:`config_hash`; this exists so an archived manifest can
    be checked against *its own* recorded config. That check is stable forever,
    because it never involves the current schema.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def config_hash(cfg: ExperimentConfig) -> str:
    return hash_config_dict(_canonical_config(cfg))


#: Dotted config paths deliberately removed from the schema.
#:
#: Archived manifests recorded these, so the current config has no value to
#: compare against. Declaring a path here says the removal was intentional,
#: which keeps :func:`config_conflicts_with_manifest` strict about every key it
#: has not been told about. Entries are never removed: a manifest stamped in
#: 2026 outlives the field it names.
#:
#: - ``cmgdb.max_table_points``: a hard ceiling on precomputed lattice size.
#:   Removed because it refused runs up front on a guess about available
#:   memory rather than letting them attempt the allocation, and every
#:   experiment config had to carry an override line to defeat it.
RETIRED_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "cmgdb.max_table_points",
    }
)


def config_conflicts_with_manifest(
    cfg: ExperimentConfig, manifest_config: dict[str, Any]
) -> list[str]:
    """Fields the manifest recorded whose value the current config disagrees on.

    Returns dotted paths, empty when compatible.

    Only keys present in ``manifest_config`` are compared. Adding a field to the
    schema therefore does not invalidate an archived manifest: the recorded run
    simply predates the field, and the field's default describes it. The
    alternative -- comparing full serializations -- makes every additive schema
    change break every stored manifest at once, which is what this replaces.

    A key the manifest recorded that the current schema lacks is still drift,
    because it usually means a rename or a typo rather than a decision. A field
    removed on purpose must be declared in :data:`RETIRED_CONFIG_FIELDS`; that
    keeps the check sharp for genuine unknowns while letting archived manifests
    stay truthful about knobs that no longer exist.

    Note that a *stored* hash cannot be reproduced from the current schema for
    the same reason. Check record integrity with :func:`hash_config_dict`
    against the manifest's own config instead.
    """

    def walk(cur: Any, old: Any, path: str) -> list[str]:
        if isinstance(old, dict):
            if not isinstance(cur, dict):
                return [path or "<root>"]
            bad: list[str] = []
            for key, old_val in old.items():
                sub = f"{path}.{key}" if path else key
                if key not in cur:
                    if sub not in RETIRED_CONFIG_FIELDS:
                        bad.append(sub)
                else:
                    bad.extend(walk(cur[key], old_val, sub))
            return bad
        return [] if cur == old else [path or "<root>"]

    return walk(_canonical_config(cfg), manifest_config, "")


def _cmgdb_version() -> str | None:
    try:
        return importlib.metadata.version("cmgdb")
    except importlib.metadata.PackageNotFoundError:
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_manifest(
    seed_cfg: ExperimentConfig,
    root_cfg: ExperimentConfig,
    *,
    cell_summary: dict[str, Any],
    stages: list[str],
    train_file: str,
    out_dir: Path | None = None,
) -> Path:
    """Write ``run_manifest.json``.

    The manifest is written to ``out_dir`` when provided (replay-routing) or
    to ``seed_cfg.paths.output_dir`` otherwise. The manifest body still names
    the *source* cell directory under ``cell.output_dir`` so provenance is
    preserved; ``cell.replay_dir`` records the replay write-root when it
    differs from source.
    """
    source_dir = seed_cfg.paths.output_dir
    write_dir = Path(out_dir) if out_dir is not None else source_dir
    write_dir.mkdir(parents=True, exist_ok=True)

    train_csv = root_cfg.paths.data_dir / f"{train_file}.csv"
    scaler_path = root_cfg.paths.scaler_path(train_file)
    cell_block: dict[str, Any] = {
        "cell_index": cell_summary.get("cell_index"),
        "train_file": train_file,
        "seed": cell_summary.get("seed"),
        "output_dir": str(source_dir),
        "device": cell_summary.get("device"),
        "skipped_stages": cell_summary.get("skipped_stages", []),
    }
    if write_dir != source_dir:
        cell_block["replay_dir"] = str(write_dir)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "latentdynamics_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "cmgdb_version": _cmgdb_version(),
        "requested_stages": stages,
        "cell": cell_block,
        "config_hash": config_hash(root_cfg),
        "config": _canonical_config(root_cfg),
        "artifacts": {
            "train_csv": str(train_csv),
            "train_csv_sha256": _file_sha256(train_csv),
            "scaler": str(scaler_path),
            "scaler_sha256": _file_sha256(scaler_path),
            "model_dir": str(seed_cfg.paths.model_dir),
            "morse_dir": str(seed_cfg.paths.morse_dir),
            "metrics": str(write_dir / "metrics.json"),
        },
    }

    path = write_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
