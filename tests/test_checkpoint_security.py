"""Tests for checkpoint-loading security (training/checkpoints.py).

The state_dict + sidecar format must load without any opt-in; the legacy
pickled-module format must be blocked unless
``LATENTDYNAMICS_ALLOW_LEGACY_CHECKPOINTS=1`` is set.
"""

from __future__ import annotations

import pytest
import torch

from latentdynamics.config.schema import ArchConfig
from latentdynamics.models.autoencoder import build_autoencoder
from latentdynamics.training.checkpoints import (
    LEGACY_CHECKPOINTS_ENV,
    LEGACY_FILES,
    load_any_checkpoint,
    load_legacy_checkpoint,
    save_checkpoint,
)


def _tiny_arch() -> ArchConfig:
    return ArchConfig(num_layers=1, hidden_shape=4, high_dims=3, low_dims=2)


def _fake_legacy_dir(tmp_path):
    """Non-empty legacy-format files that are not real checkpoints."""
    legacy_dir = tmp_path / "models"
    legacy_dir.mkdir()
    for name in LEGACY_FILES:
        (legacy_dir / name).write_bytes(b"not a real checkpoint")
    return legacy_dir


class TestLegacyBlockedWithoutEnv:
    def test_load_any_checkpoint_blocks_legacy(self, tmp_path, monkeypatch):
        monkeypatch.delenv(LEGACY_CHECKPOINTS_ENV, raising=False)
        legacy_dir = _fake_legacy_dir(tmp_path)

        with pytest.raises(RuntimeError, match=LEGACY_CHECKPOINTS_ENV) as exc_info:
            load_any_checkpoint(legacy_dir, arch=_tiny_arch(), legacy_root=tmp_path)
        message = str(exc_info.value)
        assert "state_dict" in message
        assert "weights_only" in message

    def test_load_legacy_checkpoint_blocks_directly(self, tmp_path, monkeypatch):
        monkeypatch.delenv(LEGACY_CHECKPOINTS_ENV, raising=False)
        legacy_dir = _fake_legacy_dir(tmp_path)

        with pytest.raises(RuntimeError, match=LEGACY_CHECKPOINTS_ENV):
            load_legacy_checkpoint(legacy_dir, _tiny_arch(), legacy_root=tmp_path)

    def test_env_must_be_exactly_one(self, tmp_path, monkeypatch):
        monkeypatch.setenv(LEGACY_CHECKPOINTS_ENV, "yes")
        legacy_dir = _fake_legacy_dir(tmp_path)

        with pytest.raises(RuntimeError, match=LEGACY_CHECKPOINTS_ENV):
            load_any_checkpoint(legacy_dir, arch=_tiny_arch(), legacy_root=tmp_path)

    def test_env_opens_the_gate(self, tmp_path, monkeypatch):
        """With the opt-in set, the failure moves past the gate into unpickling."""
        monkeypatch.setenv(LEGACY_CHECKPOINTS_ENV, "1")
        legacy_dir = _fake_legacy_dir(tmp_path)

        with pytest.raises(Exception) as exc_info:
            load_any_checkpoint(legacy_dir, arch=_tiny_arch(), legacy_root=tmp_path)
        # The gate no longer fires; torch rejects the garbage bytes instead.
        assert LEGACY_CHECKPOINTS_ENV not in str(exc_info.value)


class TestStateDictPath:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.delenv(LEGACY_CHECKPOINTS_ENV, raising=False)
        arch = _tiny_arch()
        model = build_autoencoder(arch)
        save_checkpoint(model, arch, tmp_path)

        loaded, loaded_arch = load_any_checkpoint(tmp_path)

        assert loaded_arch == arch
        for key, tensor in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], tensor)

    def test_new_format_wins_over_legacy_garbage(self, tmp_path, monkeypatch):
        """A dir holding both formats loads the state_dict pair without opt-in."""
        monkeypatch.delenv(LEGACY_CHECKPOINTS_ENV, raising=False)
        arch = _tiny_arch()
        model = build_autoencoder(arch)
        save_checkpoint(model, arch, tmp_path)
        for name in LEGACY_FILES:
            (tmp_path / name).write_bytes(b"not a real checkpoint")

        loaded, loaded_arch = load_any_checkpoint(tmp_path)
        assert loaded_arch == arch
