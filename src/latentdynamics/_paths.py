"""Path resolution for the latentdynamics package.

This module handles finding the package repo root and cache directories,
supporting both local development and pip-installed (Colab) environments.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    """Resolve the package repo root directory.

    Tries in order:
    1. ``LATENTDYNAMICS_REPO_ROOT`` env var if set
    2. Walking up from ``__file__`` looking for a dir containing both
       ``pyproject.toml`` and ``src/latentdynamics/`` (the validity test)
    3. Fallback to a user cache dir (``~/.cache/latentdynamics/`` or
       ``$XDG_CACHE_HOME/latentdynamics/``)

    The last option exists so that on a fresh pip install with no local repo,
    the package can still resolve a cache directory where ``fetch_artifacts``
    will populate things.
    """
    # Try env var first
    env_root = os.environ.get("LATENTDYNAMICS_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # Try walking up from this module's location
    current = Path(__file__).resolve().parent.parent.parent  # src/latentdynamics/ -> src/ -> code/
    for _ in range(10):  # reasonable depth limit
        if (current / "pyproject.toml").exists() and (current / "src" / "latentdynamics").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Fall back to cache dir
    return get_cache_dir()


def get_cache_dir() -> Path:
    """Resolve the user cache directory for latentdynamics artifacts.

    Uses ``$XDG_CACHE_HOME/latentdynamics/`` if set, else ``~/.cache/latentdynamics/``.
    On Colab (detected by the presence of ``/content`` and absence of a home cache),
    uses ``/content/latentdynamics_cache/``.
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "latentdynamics"

    # Check for Colab environment
    if Path("/content").exists() and not Path.home().is_dir():
        return Path("/content/latentdynamics_cache")

    return Path.home() / ".cache" / "latentdynamics"
