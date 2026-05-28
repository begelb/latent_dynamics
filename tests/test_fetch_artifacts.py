"""Tests for the fetch_artifacts API."""

from __future__ import annotations

import io
import tarfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from latentdynamics.replay.fetch import (
    _KNOWN_EXPERIMENTS,
    _RELEASE_URL,
    _normalize_experiment_name,
    fetch_artifacts,
)


class TestFetchArtifactsMapping:
    """Test the URL mapping table."""

    def test_all_known_experiments_have_tarballs(self):
        """Each known experiment has a .tar.gz mapping."""
        for name, tarball in _KNOWN_EXPERIMENTS.items():
            assert tarball.endswith(".tar.gz"), f"{name} -> {tarball}"

    def test_known_experiments_are_not_empty(self):
        """At least the five main experiments are known."""
        assert len(_KNOWN_EXPERIMENTS) >= 5
        expected = {
            "leslie_2gen_contraction",
            "leslie3d_example1",
            "leslie3d_example2",
            "chafee_infante",
            "coral",
        }
        assert expected.issubset(set(_KNOWN_EXPERIMENTS.keys()))


class TestFetchArtifactsValidation:
    """Test error handling for invalid inputs."""

    def test_unknown_experiment_raises(self):
        """Unknown names raise ValueError with a helpful message."""
        with pytest.raises(ValueError, match="unknown experiment.*known:"):
            fetch_artifacts("nonexistent_system")

    def test_error_message_lists_known_experiments(self):
        """Error message includes the list of known experiments."""
        with pytest.raises(ValueError) as exc_info:
            fetch_artifacts("invalid")
        error_msg = str(exc_info.value)
        assert "leslie_2gen_contraction" in error_msg


class TestFetchArtifactsCachePath:
    """Test cache-hit and cache-miss paths."""

    def test_cache_hit_returns_existing_dir(self, tmp_path):
        """If extracted path already exists, return it without downloading."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        extracted = repo_root / "replay_sources" / "leslie_2gen_contraction"
        extracted.mkdir(parents=True)
        (extracted / "marker_file").touch()

        with patch(
            "latentdynamics.replay.fetch.get_repo_root",
            return_value=repo_root,
        ):
            result = fetch_artifacts("leslie_2gen_contraction")
            assert result == extracted
            assert (result / "marker_file").exists()

    def test_download_creates_replay_sources_if_missing(self, tmp_path):
        """If replay_sources dir doesn't exist, it's created."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        extracted = repo_root / "replay_sources" / "leslie_2gen_contraction"

        def mock_extract(path, **kwargs):
            extracted.mkdir(parents=True, exist_ok=True)

        mock_tar_instance = MagicMock()
        mock_tar_instance.__enter__.return_value.extractall.side_effect = mock_extract
        mock_tar_instance.__exit__.return_value = None

        with patch(
            "latentdynamics.replay.fetch.get_repo_root",
            return_value=repo_root,
        ), patch(
            "latentdynamics.replay.fetch.urllib.request.urlretrieve"
        ), patch(
            "latentdynamics.replay.fetch.tarfile.open",
            return_value=mock_tar_instance,
        ):
            result = fetch_artifacts("leslie_2gen_contraction")
            # replay_sources dir was created and extraction succeeded
            assert (repo_root / "replay_sources").exists()
            assert result == extracted


class TestFetchArtifactsErrorHandling:
    """Test error paths (network, corrupt archive)."""

    def test_network_error_adds_context(self, tmp_path):
        """Download failures are wrapped with helpful context."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        original_error = urllib.error.URLError("404 Not Found")

        with patch(
            "latentdynamics.replay.fetch.get_repo_root",
            return_value=repo_root,
        ), patch(
            "latentdynamics.replay.fetch.urllib.request.urlretrieve",
            side_effect=original_error,
        ):
            with pytest.raises(urllib.error.URLError, match="failed to download"):
                fetch_artifacts("leslie_2gen_contraction")

    def test_corrupt_tarball_raises_valueerror(self, tmp_path):
        """Corrupt tarballs are reported clearly."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        tar_error = tarfile.TarError("corrupted tar header")

        def mock_urlretrieve(url, path):
            Path(path).write_bytes(b"fake tar content")

        with patch(
            "latentdynamics.replay.fetch.get_repo_root",
            return_value=repo_root,
        ), patch(
            "latentdynamics.replay.fetch.urllib.request.urlretrieve",
            side_effect=mock_urlretrieve,
        ), patch(
            "latentdynamics.replay.fetch.tarfile.open",
            side_effect=tar_error,
        ):
            with pytest.raises(ValueError, match="failed to extract"):
                fetch_artifacts("leslie_2gen_contraction")

    def test_missing_extracted_dir_raises_runtime_error(self, tmp_path):
        """If extraction doesn't create the expected dir, raise RuntimeError."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        mock_tar_instance = MagicMock()
        # extractall does nothing (doesn't create the dir)
        mock_tar_instance.__enter__.return_value.extractall.side_effect = lambda path, **kwargs: None
        mock_tar_instance.__exit__.return_value = None

        with patch(
            "latentdynamics.replay.fetch.get_repo_root",
            return_value=repo_root,
        ), patch(
            "latentdynamics.replay.fetch.urllib.request.urlretrieve"
        ), patch(
            "latentdynamics.replay.fetch.tarfile.open",
            return_value=mock_tar_instance,
        ):
            with pytest.raises(RuntimeError, match="extraction did not create"):
                fetch_artifacts("leslie_2gen_contraction")


class TestNormalizeExperimentName:
    """Test the name normalization helper."""

    def test_strip_replay_suffix(self):
        """Strip trailing _replay suffix."""
        assert _normalize_experiment_name("chafee_infante_replay") == "chafee_infante"
        assert _normalize_experiment_name("leslie_2gen_contraction_replay") == "leslie_2gen_contraction"
        assert _normalize_experiment_name("leslie3d_example1_replay") == "leslie3d_example1"
        assert _normalize_experiment_name("leslie3d_example2_replay") == "leslie3d_example2"

    def test_coral_variants_map_to_coral(self):
        """All coral configs share the single coral artifact bundle."""
        assert _normalize_experiment_name("coral_data_scaling") == "coral"
        assert _normalize_experiment_name("coral_adaptive") == "coral"
        assert _normalize_experiment_name("coral_basic") == "coral"
        assert _normalize_experiment_name("coral") == "coral"

    def test_base_name_idempotent(self):
        """Base names pass through unchanged."""
        assert _normalize_experiment_name("chafee_infante") == "chafee_infante"
        assert _normalize_experiment_name("leslie_2gen_contraction") == "leslie_2gen_contraction"
        assert _normalize_experiment_name("coral") == "coral"

    def test_unknown_name_still_raises_from_fetch(self):
        """Unknown names raise ValueError when passed to fetch_artifacts."""
        with pytest.raises(ValueError, match="unknown experiment"):
            fetch_artifacts("totally_fake_name")


class TestFetchArtifactsIntegration:
    """Integration tests using mocked network."""

    def test_release_url_is_well_formed(self):
        """Release URL contains the GitHub path and tag."""
        assert "github.com" in _RELEASE_URL
        assert "v0.1.0-data" in _RELEASE_URL
        assert "releases/download" in _RELEASE_URL

    def test_fetch_with_replay_suffix_normalizes_name(self, tmp_path):
        """Fetch with _replay suffix normalizes and extracts to base-key dir."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        extracted = repo_root / "replay_sources" / "leslie_2gen_contraction"

        def mock_extract(path, **kwargs):
            extracted.mkdir(parents=True, exist_ok=True)
            (extracted / "marker").touch()

        mock_tar_instance = MagicMock()
        mock_tar_instance.__enter__.return_value.extractall.side_effect = mock_extract
        mock_tar_instance.__exit__.return_value = None

        with patch(
            "latentdynamics.replay.fetch.get_repo_root",
            return_value=repo_root,
        ), patch(
            "latentdynamics.replay.fetch.urllib.request.urlretrieve"
        ), patch(
            "latentdynamics.replay.fetch.tarfile.open",
            return_value=mock_tar_instance,
        ):
            result = fetch_artifacts("leslie_2gen_contraction_replay")
            assert result == extracted
            assert (result / "marker").exists()

    def test_fetch_coral_data_scaling_maps_to_coral(self, tmp_path):
        """Fetch coral_data_scaling maps to coral base key."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        extracted = repo_root / "replay_sources" / "coral"

        def mock_extract(path, **kwargs):
            extracted.mkdir(parents=True, exist_ok=True)
            (extracted / "marker").touch()

        mock_tar_instance = MagicMock()
        mock_tar_instance.__enter__.return_value.extractall.side_effect = mock_extract
        mock_tar_instance.__exit__.return_value = None

        with patch(
            "latentdynamics.replay.fetch.get_repo_root",
            return_value=repo_root,
        ), patch(
            "latentdynamics.replay.fetch.urllib.request.urlretrieve"
        ), patch(
            "latentdynamics.replay.fetch.tarfile.open",
            return_value=mock_tar_instance,
        ):
            result = fetch_artifacts("coral_data_scaling")
            assert result == extracted
            assert (result / "marker").exists()

    def test_fresh_install_end_to_end(self, tmp_path):
        """End-to-end: fresh repo root, tarball extract, path resolution."""
        repo_root = tmp_path / "fresh_repo"
        repo_root.mkdir()

        # Create a minimal fixture tarball rooted at "chafee_infante/"
        fixture_tar = tmp_path / "fixture.tar.gz"
        with tarfile.open(fixture_tar, "w:gz") as tar:
            # Add a small file at chafee_infante/replay/models/autoencoder.json
            info = tarfile.TarInfo(name="chafee_infante/replay/models/autoencoder.json")
            content = b"{}"
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

        def mock_urlretrieve(url, path):
            # Simulate download by copying fixture to target
            Path(path).write_bytes(fixture_tar.read_bytes())

        with patch(
            "latentdynamics.replay.fetch.get_repo_root",
            return_value=repo_root,
        ), patch(
            "latentdynamics.replay.fetch.urllib.request.urlretrieve",
            side_effect=mock_urlretrieve,
        ):
            result = fetch_artifacts("chafee_infante_replay")
            assert result == repo_root / "replay_sources" / "chafee_infante"
            assert (result / "replay" / "models" / "autoencoder.json").exists()
            assert (result / "replay" / "models" / "autoencoder.json").read_text() == "{}"
