"""Provenance comparison must tolerate schema growth without going blind.

`config_conflicts_with_manifest` replaced an exact `manifest["config"] ==
cfg.model_dump()` equality check, because adding one defaulted field to
`CMGDBConfig` invalidated every archived manifest at once (725 files in the tree
carry a `config_hash`). The risk of that relaxation is that it stops detecting
real drift, so these tests pin both halves: new keys are tolerated, changed
values are not.
"""

from __future__ import annotations

import json

import pytest

from latentdynamics.cli.provenance import (
    config_conflicts_with_manifest,
    config_hash,
    hash_config_dict,
)
from latentdynamics.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config("leslie3d_example2_replay")


def test_identical_config_has_no_conflicts(cfg):
    assert config_conflicts_with_manifest(cfg, cfg.model_dump(mode="json")) == []


def test_missing_key_in_manifest_is_tolerated(cfg):
    """A manifest stamped before a field existed stays compatible."""
    recorded = cfg.model_dump(mode="json")
    del recorded["cmgdb"]["adaptive_precompute_subdiv"]
    del recorded["cmgdb"]["subdiv_max"]
    assert config_conflicts_with_manifest(cfg, recorded) == []


def test_changed_scalar_is_detected(cfg):
    recorded = cfg.model_dump(mode="json")
    recorded["cmgdb"]["subdiv_max"] = recorded["cmgdb"]["subdiv_max"] + 1
    assert config_conflicts_with_manifest(cfg, recorded) == ["cmgdb.subdiv_max"]


def test_changed_nested_value_is_detected(cfg):
    recorded = cfg.model_dump(mode="json")
    recorded["system"]["params"]["th1"] = 0.0
    assert config_conflicts_with_manifest(cfg, recorded) == ["system.params.th1"]


def test_key_dropped_from_current_schema_is_detected(cfg):
    """A field the manifest recorded but the schema no longer has is drift."""
    recorded = cfg.model_dump(mode="json")
    recorded["cmgdb"]["a_field_that_no_longer_exists"] = 7
    assert config_conflicts_with_manifest(cfg, recorded) == [
        "cmgdb.a_field_that_no_longer_exists"
    ]


def test_multiple_conflicts_are_all_reported(cfg):
    recorded = cfg.model_dump(mode="json")
    recorded["cmgdb"]["padding"] = not recorded["cmgdb"]["padding"]
    recorded["arch"]["low_dims"] = recorded["arch"]["low_dims"] + 5
    conflicts = config_conflicts_with_manifest(cfg, recorded)
    assert set(conflicts) == {"cmgdb.padding", "arch.low_dims"}


def test_type_change_dict_to_scalar_is_detected(cfg):
    recorded = cfg.model_dump(mode="json")
    recorded["cmgdb"] = 5
    assert config_conflicts_with_manifest(cfg, recorded) == ["cmgdb"]


def test_hash_of_dict_is_self_consistent(cfg):
    """The stamping hash and the record-integrity hash agree on one config."""
    dumped = cfg.model_dump(mode="json")
    assert hash_config_dict(dumped) == config_hash(cfg)


def test_hash_of_dict_detects_tampering(cfg):
    dumped = cfg.model_dump(mode="json")
    original = hash_config_dict(dumped)
    dumped["cmgdb"]["subdiv_max"] += 1
    assert hash_config_dict(dumped) != original


def test_hash_is_insensitive_to_key_order(cfg):
    dumped = cfg.model_dump(mode="json")
    reordered = json.loads(json.dumps(dumped))
    reordered["cmgdb"] = dict(reversed(list(reordered["cmgdb"].items())))
    assert hash_config_dict(reordered) == hash_config_dict(dumped)
