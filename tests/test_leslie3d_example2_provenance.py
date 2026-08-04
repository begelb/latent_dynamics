"""Regression checks for Patrick's corrected Leslie3D replay provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from latentdynamics.cli.provenance import config_conflicts_with_manifest, hash_config_dict
from latentdynamics.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_ROOT = REPO_ROOT / "replay_sources" / "leslie3d_example2"
AUTHORITATIVE_MANIFEST = REPLAY_ROOT / "run_manifest.json"
HISTORICAL_MANIFEST = REPLAY_ROOT / "run_manifest.render-2026-05-27.json"
CORRECTED_THETA = {"th1": 28.9, "th2": 29.8, "th3": 22.0}
OBSOLETE_THETA = {"th1": 19.6, "th2": 23.68, "th3": 23.68}
HISTORICAL_MANIFEST_SHA256 = "b94ac98f8c21d775372eabe9142abe211b20f95e3fd2b53535ea4d1b85775654"


def _theta(manifest: dict) -> dict[str, float]:
    params = manifest["config"]["system"]["params"]
    return {name: params[name] for name in ("th1", "th2", "th3")}


def test_authoritative_manifest_matches_replay_config() -> None:
    cfg = load_config("leslie3d_example2_replay")
    manifest = json.loads(AUTHORITATIVE_MANIFEST.read_text())

    assert cfg.experiment_name == "leslie3d_example2_patrick"
    assert _theta(manifest) == CORRECTED_THETA
    # Every field the manifest recorded must still hold. Fields added to the
    # schema after this run was stamped are absent here and are not compared:
    # the recorded run predates them and their defaults describe it. Comparing
    # full serializations instead would make any additive config change break
    # every archived manifest at once.
    assert config_conflicts_with_manifest(cfg, manifest["config"]) == []
    # The stored hash is checked against the manifest's own config, which is
    # what it was computed from. It cannot be reproduced from the current
    # schema once a field has been added, and should not be.
    assert manifest["config_hash"] == hash_config_dict(manifest["config"])
    assert manifest["manifest_role"] == "corrected_authoritative_reproduction_metadata"
    assert manifest["historical_render_manifest"] == str(
        HISTORICAL_MANIFEST.relative_to(REPO_ROOT)
    )


def test_historical_manifest_is_retained_and_correction_is_auditable() -> None:
    authoritative = json.loads(AUTHORITATIVE_MANIFEST.read_text())
    historical = json.loads(HISTORICAL_MANIFEST.read_text())
    parameter_correction, data_correction = authoritative["provenance_corrections"]

    assert hashlib.sha256(HISTORICAL_MANIFEST.read_bytes()).hexdigest() == (
        HISTORICAL_MANIFEST_SHA256
    )
    assert _theta(historical) == OBSOLETE_THETA
    assert historical["config_hash"] == parameter_correction["original_config_hash"]
    assert {
        name: parameter_correction["recorded_value"][name] for name in OBSOLETE_THETA
    } == OBSOLETE_THETA
    assert {
        name: parameter_correction["corrected_value"][name] for name in CORRECTED_THETA
    } == CORRECTED_THETA

    train_csv = REPO_ROOT / authoritative["artifacts"]["train_csv"]
    current_train_hash = hashlib.sha256(train_csv.read_bytes()).hexdigest()
    assert authoritative["artifacts"]["train_csv_sha256"] == current_train_hash
    assert data_correction["recorded_value"] == historical["artifacts"]["train_csv_sha256"]
    assert data_correction["corrected_value"] == current_train_hash
