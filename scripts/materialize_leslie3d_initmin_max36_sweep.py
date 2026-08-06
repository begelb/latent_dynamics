#!/usr/bin/env python3
"""Materialize immutable configs for the accepted Leslie3D init/min sweep."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    CODE_ROOT
    / "src"
    / "latentdynamics"
    / "configs"
    / "leslie3d_invariant_aware_v2_smooth_s24_28_36_limit10m.yaml"
)
SOURCE_MODELS = (
    CODE_ROOT
    / "output"
    / "notebooks"
    / "leslie3d_invariant_aware_v2_smooth_s24_28_36_limit10m"
    / "seed_20260809"
    / "models"
)
SWEEP_ROOT = (
    CODE_ROOT
    / "output"
    / "notebooks"
    / "leslie3d_invariant_aware_v2_smooth_initmin_sweep_max36_limit10m"
)
PAIRS = ((20, 26), (10, 26), (20, 22), (16, 24), (20, 28), (24, 30))
EXPECTED_CHECKPOINT_SHA256 = (
    "9fbee2cde690d58d2413c0d3521763838abaeb493736120f7173035612da3f3d"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    source_pt = SOURCE_MODELS / "autoencoder.pt"
    source_json = SOURCE_MODELS / "autoencoder.json"
    if _sha256(source_pt) != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("accepted checkpoint hash mismatch")

    cells = []
    for index, (initial, minimum) in enumerate(PAIRS):
        cell_id = f"i{initial}_m{minimum}_max36_limit10m"
        output_dir = Path("output/notebooks") / f"leslie3d_invariant_aware_v2_smooth_{cell_id}"
        config = json.loads(json.dumps(base))
        config["cmgdb"].update(
            {
                "subdiv_init": initial,
                "subdiv_min": minimum,
                "subdiv_max": 36,
                "subdiv_limit": 10_000_000,
                "adaptive_precompute_subdiv": "init",
                "compute_roa": False,
            }
        )
        config["paths"]["output_dir"] = output_dir.as_posix()
        config["experiment_name"] = f"leslie3d_invariant_aware_v2_smooth_{cell_id}"
        config_path = SWEEP_ROOT / "configs" / f"{cell_id}.yaml"
        rendered = yaml.safe_dump(config, sort_keys=False)
        if config_path.exists() and config_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"refusing to replace a different config: {config_path}")
        _write_text_atomic(config_path, rendered)

        run_root = CODE_ROOT / output_dir / "seed_20260809"
        model_root = run_root / "models"
        model_root.mkdir(parents=True, exist_ok=True)
        for source in (source_pt, source_json):
            target = model_root / source.name
            if not target.exists():
                target.write_bytes(source.read_bytes())
            if _sha256(target) != _sha256(source):
                raise SystemExit(f"checkpoint copy mismatch: {target}")

        cells.append(
            {
                "index": index,
                "cell_id": cell_id,
                "subdiv_init": initial,
                "subdiv_min": minimum,
                "subdiv_max": 36,
                "subdiv_limit": 10_000_000,
                "config": str(config_path.relative_to(CODE_ROOT)),
                "run_root": str(run_root.relative_to(CODE_ROOT)),
            }
        )

    plan = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_config": str(BASE_CONFIG.relative_to(CODE_ROOT)),
        "base_config_sha256": _sha256(BASE_CONFIG),
        "accepted_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "accepted_sidecar_sha256": _sha256(source_json),
        "execution_policy": "serial independent CPU processes; Morse before rendering",
        "fixed": {
            "subdiv_max": 36,
            "subdiv_limit": 10_000_000,
            "adaptive_precompute_subdiv": "init",
            "bounds": {
                "lower": base["cmgdb"]["lower_bounds"],
                "upper": base["cmgdb"]["upper_bounds"],
            },
            "padding": base["cmgdb"]["padding"],
            "box_map_backend": base["cmgdb"]["box_map_backend"],
        },
        "cells": cells,
    }
    _write_text_atomic(
        SWEEP_ROOT / "sweep_plan.json",
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
