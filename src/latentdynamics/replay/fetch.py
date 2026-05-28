"""Lazy download of replay artifacts from GitHub Release."""

from __future__ import annotations

import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from .._paths import get_cache_dir

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


def fetch_artifacts(name: str) -> Path:
    """Resolve the local artifact root for experiment `name`.

    If artifacts are already present in the local replay_sources (for local
    development), returns that path. Otherwise downloads the tarball from the
    GitHub Release v0.1.0-data on begelb/latent_dynamics into the user cache dir
    and returns the extracted path.

    The download is skipped if the extracted path already exists; set
    ``LATENTDYNAMICS_CACHE_DIR`` to control the cache location (default:
    ``$XDG_CACHE_HOME/latentdynamics/`` or ``~/.cache/latentdynamics/``).

    Args:
        name: Experiment name, one of ``leslie_2gen_contraction``,
            ``leslie3d_example1``, ``leslie3d_example2``, ``chafee_infante``,
            ``coral``.

    Returns:
        Path to the extracted artifact directory (under cache or repo).

    Raises:
        ValueError: If the experiment name is not recognized.
        urllib.error.URLError: If the download fails (network error, 404, etc.).
    """
    if name not in _KNOWN_EXPERIMENTS:
        raise ValueError(
            f"unknown experiment {name!r}; known: {', '.join(sorted(_KNOWN_EXPERIMENTS.keys()))}"
        )

    artifact_name = _KNOWN_EXPERIMENTS[name]
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted_path = cache_dir / name

    # Skip download if already extracted
    if extracted_path.is_dir():
        return extracted_path

    # Download and extract the tarball
    url = f"{_RELEASE_URL}/{artifact_name}"
    tar_path = cache_dir / artifact_name

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
            tar.extractall(path=cache_dir, filter='data')
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
