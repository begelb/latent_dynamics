"""Lazy download of replay artifacts from GitHub Release."""

from __future__ import annotations

import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from .._paths import get_repo_root

# TODO: SHA256 verification against manifest (deferred to C.2 release publishing)

_RELEASE_TAG = "v0.1.0-data"
_RELEASE_URL = f"https://github.com/begelb/latent_dynamics/releases/download/{_RELEASE_TAG}"

_KNOWN_EXPERIMENTS = {
    "leslie_2gen_contraction": "replay_leslie_2gen_contraction.tar.gz",
    "leslie3d_example1": "replay_leslie3d_example1.tar.gz",
    "leslie3d_example2": "replay_leslie3d_example2.tar.gz",
    "chafee_infante": "replay_chafee_infante.tar.gz",
    "coral": "replay_coral.tar.gz",
}


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

    Raises:
        ValueError: If the normalized name is not in _KNOWN_EXPERIMENTS.
    """
    normalized = name

    # Strip trailing "_replay" suffix
    if normalized.endswith("_replay"):
        normalized = normalized[:-7]  # len("_replay") == 7

    # All coral configs share one artifact bundle.
    if normalized.startswith("coral"):
        normalized = "coral"

    return normalized


def fetch_artifacts(name: str) -> Path:
    """Resolve the local artifact root for experiment `name`.

    Normalizes the experiment name (strips trailing "_replay", maps
    "coral_data_scaling" -> "coral"), then downloads the corresponding
    tarball from GitHub Release v0.1.0-data if needed and extracts it into
    <repo-root>/replay_sources/<base_key>. On a pip install, repo-root
    falls back to the cache directory, so artifacts are cached correctly
    in either environment.

    The download is skipped if the extracted path already exists.

    Args:
        name: Experiment name (e.g. "leslie_2gen_contraction_replay",
            "chafee_infante", or "coral_data_scaling"). One of the known
            base names or their "_replay" variants.

    Returns:
        Path to the extracted artifact directory (under <repo-root>/replay_sources/<base_key>).

    Raises:
        ValueError: If the normalized experiment name is not recognized.
        urllib.error.URLError: If the download fails (network error, 404, etc.).
    """
    base_key = _normalize_experiment_name(name)

    if base_key not in _KNOWN_EXPERIMENTS:
        raise ValueError(
            f"unknown experiment {name!r}; known: {', '.join(sorted(_KNOWN_EXPERIMENTS.keys()))}"
        )

    artifact_name = _KNOWN_EXPERIMENTS[base_key]
    root = get_repo_root()
    dest_parent = root / "replay_sources"
    dest_parent.mkdir(parents=True, exist_ok=True)
    extracted_path = dest_parent / base_key

    # Skip download if already extracted
    if extracted_path.is_dir():
        return extracted_path

    # Download and extract the tarball
    url = f"{_RELEASE_URL}/{artifact_name}"
    tar_path = dest_parent / artifact_name

    try:
        urllib.request.urlretrieve(url, tar_path)
    except urllib.error.URLError as e:
        raise urllib.error.URLError(
            f"failed to download {url}: {e}. Check your internet connection or "
            f"verify the release exists at https://github.com/begelb/latent_dynamics/releases"
        ) from e

    # Extract
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_parent, filter='data')
    except tarfile.TarError as e:
        raise ValueError(
            f"failed to extract {tar_path}: {e}. The archive may be corrupted."
        ) from e
    finally:
        # Clean up the tarball after extraction
        tar_path.unlink(missing_ok=True)

    if not extracted_path.is_dir():
        raise RuntimeError(
            f"extraction did not create expected directory {extracted_path}"
        )

    return extracted_path
