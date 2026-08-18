"""Tests for manifest-driven artifact fetching (replay/fetch.py).

All tests run against locally built tarballs and a temporary repo root; no
network access and no real bundles are needed.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

import latentdynamics.replay.fetch as fetch_mod
from latentdynamics.replay.fetch import (
    _KNOWN_EXPERIMENTS,
    ArtifactIntegrityError,
    ArtifactsNotPublishedError,
    FetchError,
    RECEIPT_NAME,
    _normalize_experiment_name,
    fetch_artifacts,
    fetch_bundle,
)

KEY = "leslie3d_example1"
URL_BASE = "https://releases.invalid/latent-dynamics/v1"


# -- tarball / manifest builders ------------------------------------------- #
def _add_file(tar: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))


def _add_special(tar: tarfile.TarFile, name: str, tar_type: bytes, linkname: str = "") -> None:
    info = tarfile.TarInfo(name=name)
    info.type = tar_type
    info.linkname = linkname
    tar.addfile(info)


def _make_bundle(tmp_path: Path, key: str = KEY, extra=None) -> Path:
    """Build a well-formed bundle tarball rooted at ``<key>/``."""
    tar_path = tmp_path / f"replay_{key}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        _add_file(tar, f"{key}/models/autoencoder.json", b"{}")
        _add_file(tar, f"{key}/models/autoencoder.pt", b"fake weights")
        if extra is not None:
            extra(tar)
    return tar_path


def _write_manifest(
    repo_root: Path,
    tar_path: Path,
    *,
    key: str = KEY,
    url_base: str = URL_BASE,
    sha256: str | None = None,
    size_bytes: int | None = None,
    members: dict | None = None,
) -> None:
    entry = {
        "filename": tar_path.name,
        "sha256": sha256 if sha256 is not None else hashlib.sha256(tar_path.read_bytes()).hexdigest(),
        "size_bytes": size_bytes if size_bytes is not None else tar_path.stat().st_size,
        "extract_to": "replay_sources/",
    }
    if members is not None:
        entry["members"] = members
    manifest = {"version": 1, "release_url_base": url_base, "artifacts": {key: entry}}
    manifest_dir = repo_root / "artifacts"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


@pytest.fixture
def repo_root(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(fetch_mod, "get_repo_root", lambda: root)
    return root


@pytest.fixture
def serve_tar(monkeypatch):
    """Patch _download to 'serve' a local fixture tarball; records call count."""

    def _install(tar_path: Path):
        calls = []

        def fake_download(url, dest):
            calls.append(url)
            Path(dest).write_bytes(tar_path.read_bytes())

        monkeypatch.setattr(fetch_mod, "_download", fake_download)
        return calls

    return _install


def _assert_nothing_extracted(repo_root: Path, key: str = KEY) -> None:
    extracted = repo_root / "replay_sources" / key
    assert not extracted.exists()
    sources = repo_root / "replay_sources"
    if sources.is_dir():
        leftovers = [p for p in sources.iterdir()]
        assert leftovers == [], f"temp leftovers not cleaned up: {leftovers}"


# -- known experiments / normalization ------------------------------------- #
class TestKnownExperiments:
    def test_exactly_the_paper_bundles(self):
        assert set(_KNOWN_EXPERIMENTS) == {
            "leslie_2gen_contraction",
            "leslie3d_example1",
            "chafee_infante",
            "coral",
        }

    def test_unknown_experiment_raises(self, repo_root):
        with pytest.raises(ValueError, match="unknown experiment.*known:"):
            fetch_artifacts("nonexistent_system")

    def test_dropped_bundle_is_unknown(self, repo_root):
        with pytest.raises(ValueError, match="unknown experiment"):
            fetch_artifacts("leslie3d_example2")


class TestNormalizeExperimentName:
    def test_strip_replay_suffix(self):
        assert _normalize_experiment_name("chafee_infante_replay") == "chafee_infante"
        assert (
            _normalize_experiment_name("leslie_2gen_contraction_replay")
            == "leslie_2gen_contraction"
        )
        assert _normalize_experiment_name("leslie3d_example1_replay") == "leslie3d_example1"

    def test_coral_variants_map_to_coral(self):
        assert _normalize_experiment_name("coral_data_scaling") == "coral"
        assert _normalize_experiment_name("coral_adaptive") == "coral"
        assert _normalize_experiment_name("coral_basic") == "coral"
        assert _normalize_experiment_name("coral") == "coral"

    def test_base_name_idempotent(self):
        assert _normalize_experiment_name("chafee_infante") == "chafee_infante"
        assert _normalize_experiment_name("leslie_2gen_contraction") == "leslie_2gen_contraction"


# -- happy path ------------------------------------------------------------- #
class TestHappyPath:
    def test_fetch_verifies_extracts_and_protects(self, repo_root, serve_tar, tmp_path):
        tar_path = _make_bundle(tmp_path)
        _write_manifest(repo_root, tar_path)
        calls = serve_tar(tar_path)

        result = fetch_artifacts(KEY)

        assert result == repo_root / "replay_sources" / KEY
        assert calls == [f"{URL_BASE}/{tar_path.name}"]
        sidecar = result / "models" / "autoencoder.json"
        assert sidecar.read_text() == "{}"
        # Extracted files are read-only.
        assert sidecar.stat().st_mode & 0o777 == 0o444
        # No tarball or temp dir left behind.
        leftovers = [p.name for p in (repo_root / "replay_sources").iterdir() if p.name != KEY]
        assert leftovers == []

    def test_receipt_recorded(self, repo_root, serve_tar, tmp_path):
        tar_path = _make_bundle(tmp_path)
        _write_manifest(repo_root, tar_path)
        serve_tar(tar_path)

        result = fetch_artifacts(KEY)

        receipt = json.loads((result / RECEIPT_NAME).read_text())
        assert receipt["bundle"] == KEY
        assert receipt["sha256"] == hashlib.sha256(tar_path.read_bytes()).hexdigest()
        assert receipt["source_url"] == f"{URL_BASE}/{tar_path.name}"
        assert "fetched_at" in receipt

    def test_replay_suffix_and_coral_mapping(self, repo_root, serve_tar, tmp_path):
        tar_path = _make_bundle(tmp_path, key="coral")
        _write_manifest(repo_root, tar_path, key="coral")
        serve_tar(tar_path)

        result = fetch_artifacts("coral_data_scaling")
        assert result == repo_root / "replay_sources" / "coral"

    def test_internal_symlink_accepted(self, repo_root, serve_tar, tmp_path):
        def extra(tar):
            _add_special(tar, f"{KEY}/models/latest.json", tarfile.SYMTYPE, "autoencoder.json")

        tar_path = _make_bundle(tmp_path, extra=extra)
        _write_manifest(repo_root, tar_path)
        serve_tar(tar_path)

        result = fetch_artifacts(KEY)
        assert (result / "models" / "latest.json").read_text() == "{}"


# -- non-experiment bundles -------------------------------------------------- #
class TestFetchBundle:
    def test_any_manifest_key_fetches_without_normalization(
        self, repo_root, serve_tar, tmp_path
    ):
        key = "chafee_training_datasets"
        tar_path = _make_bundle(tmp_path, key=key)
        _write_manifest(repo_root, tar_path, key=key)
        calls = serve_tar(tar_path)

        result = fetch_bundle(key)

        assert result == repo_root / "replay_sources" / key
        assert calls == [f"{URL_BASE}/{tar_path.name}"]
        assert (result / "models" / "autoencoder.json").read_text() == "{}"
        receipt = json.loads((result / RECEIPT_NAME).read_text())
        assert receipt["bundle"] == key

    def test_unknown_key_raises(self, repo_root, tmp_path):
        tar_path = _make_bundle(tmp_path)
        _write_manifest(repo_root, tar_path)

        with pytest.raises(FetchError, match="no manifest entry"):
            fetch_bundle("not_in_the_manifest")
        _assert_nothing_extracted(repo_root, key="not_in_the_manifest")


# -- verification before extraction ----------------------------------------- #
class TestVerificationBeforeExtract:
    def test_checksum_mismatch_rejected_before_extract(
        self, repo_root, serve_tar, tmp_path, monkeypatch
    ):
        tar_path = _make_bundle(tmp_path)
        actual_sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        _write_manifest(repo_root, tar_path, sha256="0" * 64)
        serve_tar(tar_path)

        def no_extract(*args, **kwargs):
            raise AssertionError("tarfile.open called despite checksum mismatch")

        monkeypatch.setattr(fetch_mod.tarfile, "open", no_extract)

        with pytest.raises(ArtifactIntegrityError) as exc_info:
            fetch_artifacts(KEY)
        message = str(exc_info.value)
        assert "0" * 64 in message
        assert actual_sha in message
        _assert_nothing_extracted(repo_root)

    def test_size_mismatch_rejected(self, repo_root, serve_tar, tmp_path, monkeypatch):
        tar_path = _make_bundle(tmp_path)
        real_size = tar_path.stat().st_size
        _write_manifest(repo_root, tar_path, size_bytes=real_size + 1)
        serve_tar(tar_path)

        def no_extract(*args, **kwargs):
            raise AssertionError("tarfile.open called despite size mismatch")

        monkeypatch.setattr(fetch_mod.tarfile, "open", no_extract)

        with pytest.raises(ArtifactIntegrityError, match="size mismatch") as exc_info:
            fetch_artifacts(KEY)
        message = str(exc_info.value)
        assert str(real_size) in message
        assert str(real_size + 1) in message
        _assert_nothing_extracted(repo_root)


# -- member filtering -------------------------------------------------------- #
class TestMemberFiltering:
    def _fetch_bad_bundle(self, repo_root, serve_tar, tmp_path, extra, match):
        tar_path = _make_bundle(tmp_path, extra=extra)
        _write_manifest(repo_root, tar_path)
        serve_tar(tar_path)
        with pytest.raises(ArtifactIntegrityError, match=match):
            fetch_artifacts(KEY)
        _assert_nothing_extracted(repo_root)

    def test_traversal_member_rejected(self, repo_root, serve_tar, tmp_path):
        self._fetch_bad_bundle(
            repo_root,
            serve_tar,
            tmp_path,
            lambda tar: _add_file(tar, "../evil", b"x"),
            match="'[.][.]'",
        )

    def test_nested_traversal_member_rejected(self, repo_root, serve_tar, tmp_path):
        self._fetch_bad_bundle(
            repo_root,
            serve_tar,
            tmp_path,
            lambda tar: _add_file(tar, f"{KEY}/../../evil", b"x"),
            match="'[.][.]'",
        )

    def test_absolute_path_member_rejected(self, repo_root, serve_tar, tmp_path):
        self._fetch_bad_bundle(
            repo_root,
            serve_tar,
            tmp_path,
            lambda tar: _add_file(tar, "/etc/evil", b"x"),
            match="absolute path",
        )

    def test_symlink_escape_rejected(self, repo_root, serve_tar, tmp_path):
        self._fetch_bad_bundle(
            repo_root,
            serve_tar,
            tmp_path,
            lambda tar: _add_special(tar, f"{KEY}/link", tarfile.SYMTYPE, "../../outside"),
            match="escapes the extraction root",
        )

    def test_absolute_symlink_rejected(self, repo_root, serve_tar, tmp_path):
        self._fetch_bad_bundle(
            repo_root,
            serve_tar,
            tmp_path,
            lambda tar: _add_special(tar, f"{KEY}/link", tarfile.SYMTYPE, "/etc/passwd"),
            match="escapes the extraction root",
        )

    def test_hardlink_escape_rejected(self, repo_root, serve_tar, tmp_path):
        self._fetch_bad_bundle(
            repo_root,
            serve_tar,
            tmp_path,
            lambda tar: _add_special(tar, f"{KEY}/hardlink", tarfile.LNKTYPE, "../outside"),
            match="escapes the extraction root",
        )

    def test_fifo_member_rejected(self, repo_root, serve_tar, tmp_path):
        self._fetch_bad_bundle(
            repo_root,
            serve_tar,
            tmp_path,
            lambda tar: _add_special(tar, f"{KEY}/pipe", tarfile.FIFOTYPE),
            match="device or FIFO",
        )

    def test_zero_byte_member_rejected(self, repo_root, serve_tar, tmp_path):
        self._fetch_bad_bundle(
            repo_root,
            serve_tar,
            tmp_path,
            lambda tar: _add_file(tar, f"{KEY}/logs/empty.log", b""),
            match="zero-byte member",
        )

    def test_allowlisted_zero_byte_accepted(self, repo_root, serve_tar, tmp_path):
        def extra(tar):
            _add_file(tar, f"{KEY}/logs/empty.log", b"")

        tar_path = _make_bundle(tmp_path, extra=extra)
        _write_manifest(
            repo_root, tar_path, members={"zero_byte_ok": [f"{KEY}/logs/*.log"]}
        )
        serve_tar(tar_path)

        result = fetch_artifacts(KEY)
        assert (result / "logs" / "empty.log").stat().st_size == 0


# -- atomicity --------------------------------------------------------------- #
class TestAtomicity:
    def test_partial_extraction_not_promoted(self, repo_root, serve_tar, tmp_path, monkeypatch):
        tar_path = _make_bundle(tmp_path)
        _write_manifest(repo_root, tar_path)
        serve_tar(tar_path)

        def partial_extractall(self, path, members=None, **kwargs):
            # Write one file, then die mid-extraction.
            partial = Path(path) / KEY / "models"
            partial.mkdir(parents=True)
            (partial / "autoencoder.json").write_text("{}")
            raise tarfile.TarError("disk full halfway through")

        monkeypatch.setattr(fetch_mod.tarfile.TarFile, "extractall", partial_extractall)

        with pytest.raises(ArtifactIntegrityError, match="failed to extract"):
            fetch_artifacts(KEY)
        _assert_nothing_extracted(repo_root)

    def test_wrong_archive_root_not_promoted(self, repo_root, serve_tar, tmp_path):
        tar_path = _make_bundle(tmp_path, key="wrong_key")
        _write_manifest(repo_root, tar_path)
        serve_tar(tar_path)

        with pytest.raises(FetchError, match="not rooted at"):
            fetch_artifacts(KEY)
        _assert_nothing_extracted(repo_root)


# -- cache and manifest state ------------------------------------------------ #
class TestCacheAndManifest:
    def test_cache_hit_skips_download(self, repo_root, monkeypatch):
        extracted = repo_root / "replay_sources" / KEY
        extracted.mkdir(parents=True)
        (extracted / "marker").touch()

        def no_download(url, dest):
            raise AssertionError("download attempted despite cache hit")

        monkeypatch.setattr(fetch_mod, "_download", no_download)

        # No manifest is written: a cache hit must not need one either.
        result = fetch_artifacts(KEY)
        assert result == extracted
        assert (result / "marker").exists()

    def test_pending_release_url_raises(self, repo_root, tmp_path):
        tar_path = _make_bundle(tmp_path)
        _write_manifest(repo_root, tar_path, url_base="PENDING", sha256="TBD", size_bytes=0)

        with pytest.raises(ArtifactsNotPublishedError, match="not yet published") as exc_info:
            fetch_artifacts(KEY)
        # The error names the manual placement path.
        assert str(repo_root / "replay_sources" / KEY) in str(exc_info.value)

    def test_missing_manifest_raises(self, repo_root):
        with pytest.raises(FetchError, match="manifest not found"):
            fetch_artifacts(KEY)

    def test_missing_manifest_entry_raises(self, repo_root, tmp_path):
        tar_path = _make_bundle(tmp_path, key="coral")
        _write_manifest(repo_root, tar_path, key="coral")

        with pytest.raises(FetchError, match="no manifest entry"):
            fetch_artifacts(KEY)

    def test_placeholder_sha_raises_before_download(self, repo_root, tmp_path, monkeypatch):
        tar_path = _make_bundle(tmp_path)
        _write_manifest(repo_root, tar_path, sha256="TBD")

        def no_download(url, dest):
            raise AssertionError("download attempted with unverifiable manifest entry")

        monkeypatch.setattr(fetch_mod, "_download", no_download)

        with pytest.raises(FetchError, match="no finalized sha256"):
            fetch_artifacts(KEY)
