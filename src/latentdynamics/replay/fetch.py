"""Manifest-driven download and verification of replay artifact bundles.

Bundles are described by ``artifacts/manifest.json`` at the repo root:

.. code-block:: json

    {
      "version": 1,
      "release_url_base": "https://... or PENDING",
      "artifacts": {
        "<bundle_key>": {
          "filename": "replay_<bundle_key>.tar.gz",
          "sha256": "<64-char hex digest of the tarball>",
          "size_bytes": 12345,
          "extract_to": "replay_sources/",
          "members": {"zero_byte_ok": ["<fnmatch pattern>", "..."]}
        }
      }
    }

``members`` is optional; ``zero_byte_ok`` lists archive member paths (fnmatch
patterns) that are allowed to be empty files. Every other zero-byte member is
rejected, as are absolute paths, ``..`` components, links pointing outside the
archive, and device/FIFO members.

A bundle is downloaded to a temporary file, verified against the manifest's
size and sha256 *before* any extraction, extracted into a temporary directory,
and only then renamed into ``<repo-root>/<extract_to>/<bundle_key>`` -- so a
partial download or extraction never becomes a valid cache. Extracted files
are made read-only and a ``.fetch_receipt.json`` records what was fetched.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

from .._paths import get_repo_root

MANIFEST_VERSION = 1
MANIFEST_RELPATH = Path("artifacts") / "manifest.json"
RECEIPT_NAME = ".fetch_receipt.json"

_KNOWN_EXPERIMENTS = (
    "leslie_2gen_contraction",
    "leslie3d_example1",
    "chafee_infante",
    "coral",
)


class FetchError(RuntimeError):
    """An artifact bundle could not be fetched (manifest problem, bad layout)."""


class ArtifactsNotPublishedError(FetchError):
    """The manifest's ``release_url_base`` is still ``PENDING``."""


class ArtifactIntegrityError(FetchError):
    """A downloaded bundle failed size/sha256 verification or holds unsafe members."""


def _normalize_experiment_name(name: str) -> str:
    """Map incoming experiment name to its base artifact key.

    Handles:
    - Strips a single trailing "_replay" suffix if present
    - Maps any coral config (coral, coral_basic, coral_data_scaling,
      coral_adaptive) to the single "coral" artifact bundle

    Args:
        name: User-provided experiment name (e.g. from load_experiment).

    Returns:
        Base artifact key (e.g. "leslie_2gen_contraction").
    """
    normalized = name

    # Strip trailing "_replay" suffix
    if normalized.endswith("_replay"):
        normalized = normalized[: -len("_replay")]

    # All coral configs share one artifact bundle.
    if normalized.startswith("coral"):
        normalized = "coral"

    return normalized


def _load_manifest(root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = root / MANIFEST_RELPATH
    if not manifest_path.is_file():
        raise FetchError(
            f"artifact manifest not found at {manifest_path}; it ships with the "
            f"repository as {MANIFEST_RELPATH}. If this package was pip-installed, "
            f"set LATENTDYNAMICS_REPO_ROOT to a repository checkout."
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise FetchError(f"artifact manifest {manifest_path} is not valid JSON: {exc}") from exc
    if manifest.get("version") != MANIFEST_VERSION:
        raise FetchError(
            f"unsupported manifest version {manifest.get('version')!r} in "
            f"{manifest_path} (expected {MANIFEST_VERSION})"
        )
    if not isinstance(manifest.get("artifacts"), dict):
        raise FetchError(f"artifact manifest {manifest_path} has no 'artifacts' table")
    return manifest, manifest_path


def _validate_member(member: tarfile.TarInfo, zero_byte_ok: tuple[str, ...]) -> None:
    """Reject archive members that could write outside the extraction root."""
    name = member.name
    path = PurePosixPath(name)
    if path.is_absolute() or name.startswith("/"):
        raise ArtifactIntegrityError(f"archive member has an absolute path: {name!r}")
    if ".." in path.parts:
        raise ArtifactIntegrityError(f"archive member path contains '..': {name!r}")
    if member.isdev():
        raise ArtifactIntegrityError(f"archive contains a device or FIFO member: {name!r}")
    if member.issym() or member.islnk():
        target = PurePosixPath(member.linkname)
        if member.issym():
            # Symlink targets resolve relative to the member's own directory.
            resolved = posixpath.normpath(str(path.parent / target))
        else:
            # Hardlink targets are archive-root-relative.
            resolved = posixpath.normpath(member.linkname)
        if target.is_absolute() or resolved == ".." or resolved.startswith("../"):
            raise ArtifactIntegrityError(
                f"archive link escapes the extraction root: {name!r} -> {member.linkname!r}"
            )
        return
    if not (member.isfile() or member.isdir()):
        raise ArtifactIntegrityError(
            f"archive member {name!r} has unsupported type {member.type!r}"
        )
    if member.isfile() and member.size == 0:
        if not any(fnmatch(name, pattern) for pattern in zero_byte_ok):
            raise ArtifactIntegrityError(
                f"archive contains zero-byte member {name!r} not covered by the "
                f"manifest entry's members.zero_byte_ok allowlist"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_ALLOWED_URL_SCHEMES = ("https", "file")


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` into ``dest``.

    Only ``https`` (published releases) and ``file`` (locally staged
    bundles) URLs are accepted; integrity is enforced separately by the
    SHA-256 check before extraction.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(
            f"refusing to download {url!r}: URL scheme {scheme!r} is not one of "
            f"{_ALLOWED_URL_SCHEMES}"
        )
    try:
        with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.URLError as exc:
        raise urllib.error.URLError(
            f"failed to download {url}: {exc}. Check your internet connection or "
            f"verify that the release assets exist."
        ) from exc


def fetch_artifacts(name: str) -> Path:
    """Resolve the local artifact root for experiment ``name``.

    Normalizes the experiment name (strips trailing "_replay", maps
    "coral_data_scaling" -> "coral") and returns the cached bundle directory
    if it already exists. Otherwise the bundle listed in
    ``artifacts/manifest.json`` is downloaded, verified against the manifest's
    ``size_bytes`` and ``sha256`` before extraction, extracted through a
    strict member filter into a temporary directory, and atomically renamed
    into ``<repo-root>/<extract_to>/<bundle_key>``. On a pip install,
    repo-root falls back to the cache directory.

    Args:
        name: Experiment name (e.g. "leslie_2gen_contraction_replay",
            "chafee_infante", or "coral_data_scaling"). One of the known
            base names or their "_replay" variants.

    Returns:
        Path to the verified artifact directory.

    Raises:
        ValueError: If the normalized experiment name is not recognized.
        ArtifactsNotPublishedError: If the manifest's ``release_url_base``
            is still ``PENDING``.
        ArtifactIntegrityError: If the download fails size/sha256
            verification or the archive holds unsafe members.
        FetchError: For other manifest or bundle-layout problems.
        urllib.error.URLError: If the download itself fails.
    """
    base_key = _normalize_experiment_name(name)

    if base_key not in _KNOWN_EXPERIMENTS:
        raise ValueError(
            f"unknown experiment {name!r}; known: {', '.join(sorted(_KNOWN_EXPERIMENTS))}"
        )

    return _resolve_bundle(base_key)


def fetch_bundle(key: str) -> Path:
    """Resolve the local directory for any bundle listed in the manifest.

    Unlike :func:`fetch_artifacts`, ``key`` is used verbatim -- no
    experiment-name normalization and no known-experiment check -- so every
    key present in ``artifacts/manifest.json`` (reference datasets and other
    non-experiment bundles included) can be fetched through the same
    verification and extraction path.

    Args:
        key: Bundle key exactly as it appears in the manifest's
            ``artifacts`` table.

    Returns:
        Path to the verified bundle directory.

    Raises:
        ArtifactsNotPublishedError: If the manifest's ``release_url_base``
            is still ``PENDING``.
        ArtifactIntegrityError: If the download fails size/sha256
            verification or the archive holds unsafe members.
        FetchError: If ``key`` has no manifest entry, or for other manifest
            or bundle-layout problems.
        urllib.error.URLError: If the download itself fails.
    """
    return _resolve_bundle(key)


def _resolve_bundle(base_key: str) -> Path:
    """Shared cache-or-download path for a manifest bundle key."""
    root = get_repo_root()

    # Cache hit under the conventional layout needs no manifest at all.
    conventional = root / "replay_sources" / base_key
    if conventional.is_dir():
        return conventional

    manifest, manifest_path = _load_manifest(root)
    entry = manifest["artifacts"].get(base_key)
    if entry is None:
        raise FetchError(
            f"no manifest entry for bundle {base_key!r} in {manifest_path}; "
            f"entries: {', '.join(sorted(manifest['artifacts']))}"
        )

    extract_to = PurePosixPath(str(entry.get("extract_to") or "replay_sources/"))
    if extract_to.is_absolute() or ".." in extract_to.parts:
        raise FetchError(
            f"manifest entry for {base_key!r} has unsafe extract_to {str(extract_to)!r}"
        )
    dest_parent = root / extract_to
    extracted_path = dest_parent / base_key
    if extracted_path.is_dir():
        return extracted_path

    url_base = manifest.get("release_url_base")
    if not url_base or url_base == "PENDING":
        raise ArtifactsNotPublishedError(
            f"replay artifacts are not yet published (release_url_base is PENDING "
            f"in {manifest_path}). Place the extracted bundle manually at "
            f"{extracted_path}, or update the manifest once the release exists."
        )

    filename = entry.get("filename")
    if not filename:
        raise FetchError(f"manifest entry for {base_key!r} has no 'filename'")
    sha_expected = str(entry.get("sha256", "")).lower()
    if len(sha_expected) != 64 or any(c not in "0123456789abcdef" for c in sha_expected):
        raise FetchError(
            f"manifest entry for {base_key!r} has no finalized sha256 "
            f"({entry.get('sha256')!r}); the bundle cannot be verified"
        )
    size_expected = entry.get("size_bytes")
    if not isinstance(size_expected, int) or size_expected <= 0:
        raise FetchError(
            f"manifest entry for {base_key!r} has no finalized size_bytes "
            f"({size_expected!r}); the bundle cannot be verified"
        )
    zero_byte_ok = tuple((entry.get("members") or {}).get("zero_byte_ok") or ())

    dest_parent.mkdir(parents=True, exist_ok=True)
    url = f"{str(url_base).rstrip('/')}/{filename}"
    # Same filesystem as the destination so the final os.replace is atomic.
    work_dir = Path(tempfile.mkdtemp(prefix=".fetch_", dir=dest_parent))
    try:
        tar_path = work_dir / filename
        _download(url, tar_path)

        actual_size = tar_path.stat().st_size
        if actual_size != size_expected:
            raise ArtifactIntegrityError(
                f"{filename}: size mismatch (manifest says {size_expected} bytes, "
                f"downloaded {actual_size}); refusing to extract"
            )
        actual_sha = _sha256_file(tar_path)
        if actual_sha != sha_expected:
            raise ArtifactIntegrityError(
                f"{filename}: sha256 mismatch; refusing to extract\n"
                f"  manifest:   {sha_expected}\n"
                f"  downloaded: {actual_sha}"
            )

        extract_dir = work_dir / "extract"
        extract_dir.mkdir()
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                members = tar.getmembers()
                for member in members:
                    _validate_member(member, zero_byte_ok)
                tar.extractall(path=extract_dir, members=members, filter="data")
        except tarfile.TarError as exc:
            raise ArtifactIntegrityError(
                f"failed to extract {filename}: {exc}; the archive may be corrupted"
            ) from exc

        staged = extract_dir / base_key
        if not staged.is_dir():
            raise FetchError(
                f"{filename}: archive is not rooted at {base_key!r}/ (extraction "
                f"produced: {', '.join(sorted(p.name for p in extract_dir.iterdir())) or 'nothing'})"
            )

        receipt = {
            "bundle": base_key,
            "sha256": actual_sha,
            "size_bytes": actual_size,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_url": url,
        }
        (staged / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n")

        # Fetched bundles are verified inputs; keep them read-only.
        for file in staged.rglob("*"):
            if file.is_file() and not file.is_symlink():
                file.chmod(0o444)

        try:
            os.replace(staged, extracted_path)
        except OSError:
            if extracted_path.is_dir():
                # Another process promoted its own copy first; use it.
                return extracted_path
            raise
        return extracted_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
