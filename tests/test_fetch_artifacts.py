"""Tests for the fetch_artifacts API."""

from __future__ import annotations

import tarfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from latentdynamics.replay.fetch import (
    _KNOWN_EXPERIMENTS,
    _RELEASE_URL,
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
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        extracted = cache_dir / "leslie_2gen_contraction"
        extracted.mkdir()
        (extracted / "marker_file").touch()

        with patch(
            "latentdynamics.replay.fetch.get_cache_dir",
            return_value=cache_dir,
        ):
            result = fetch_artifacts("leslie_2gen_contraction")
            assert result == extracted
            assert (result / "marker_file").exists()

    def test_download_creates_cache_if_missing(self, tmp_path):
        """If cache dir doesn't exist, it's created."""
        cache_dir = tmp_path / "new_cache"
        extracted = cache_dir / "leslie_2gen_contraction"

        def mock_extract(path, **kwargs):
            extracted.mkdir(parents=True, exist_ok=True)

        mock_tar_instance = MagicMock()
        mock_tar_instance.__enter__.return_value.extractall.side_effect = mock_extract
        mock_tar_instance.__exit__.return_value = None

        with patch(
            "latentdynamics.replay.fetch.get_cache_dir",
            return_value=cache_dir,
        ), patch(
            "latentdynamics.replay.fetch.urllib.request.urlretrieve"
        ), patch(
            "latentdynamics.replay.fetch.tarfile.open",
            return_value=mock_tar_instance,
        ):
            result = fetch_artifacts("leslie_2gen_contraction")
            # Cache dir was created and extraction succeeded
            assert cache_dir.exists()
            assert result == extracted


class TestFetchArtifactsErrorHandling:
    """Test error paths (network, corrupt archive)."""

    def test_network_error_adds_context(self, tmp_path):
        """Download failures are wrapped with helpful context."""
        cache_dir = tmp_path / "cache"

        original_error = urllib.error.URLError("404 Not Found")

        with patch(
            "latentdynamics.replay.fetch.get_cache_dir",
            return_value=cache_dir,
        ), patch(
            "latentdynamics.replay.fetch.urllib.request.urlretrieve",
            side_effect=original_error,
        ):
            with pytest.raises(urllib.error.URLError, match="failed to download"):
                fetch_artifacts("leslie_2gen_contraction")

    def test_corrupt_tarball_raises_valueerror(self, tmp_path):
        """Corrupt tarballs are reported clearly."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        tar_error = tarfile.TarError("corrupted tar header")

        def mock_urlretrieve(url, path):
            Path(path).write_bytes(b"fake tar content")

        with patch(
            "latentdynamics.replay.fetch.get_cache_dir",
            return_value=cache_dir,
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
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        mock_tar_instance = MagicMock()
        # extractall does nothing (doesn't create the dir)
        mock_tar_instance.__enter__.return_value.extractall.side_effect = lambda path, **kwargs: None
        mock_tar_instance.__exit__.return_value = None

        with patch(
            "latentdynamics.replay.fetch.get_cache_dir",
            return_value=cache_dir,
        ), patch(
            "latentdynamics.replay.fetch.urllib.request.urlretrieve"
        ), patch(
            "latentdynamics.replay.fetch.tarfile.open",
            return_value=mock_tar_instance,
        ):
            with pytest.raises(RuntimeError, match="extraction did not create"):
                fetch_artifacts("leslie_2gen_contraction")


class TestFetchArtifactsIntegration:
    """Integration tests using mocked network."""

    def test_release_url_is_well_formed(self):
        """Release URL contains the GitHub path and tag."""
        assert "github.com" in _RELEASE_URL
        assert "v0.1.0-data" in _RELEASE_URL
        assert "releases/download" in _RELEASE_URL
