"""Identify which CMGDB build a run is about to use.

The project depends on features that exist only in the maintained fork, so a
long run started against upstream CMGDB is wasted work. The fork used to be
recognisable by its location -- an editable checkout under ``archive/CMGDB`` --
but it now also ships prebuilt wheels, so location no longer implies identity.
These helpers check for the fork's API instead, and record whichever provenance
the installation can actually supply.
"""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import CMGDB

__all__ = ["FORK_ONLY_ATTRIBUTES", "cmgdb_provenance", "require_fork_cmgdb"]

#: Entry points added by the fork. ``PrecomputedBoxMap`` carries the batched
#: box-map evaluation the pipeline is built on; the other two are the native
#: reachability queries.
FORK_ONLY_ATTRIBUTES = (
    "PrecomputedBoxMap",
    "MorseDirectedPathCells",
    "ComputeMorseSetReachability",
)


def require_fork_cmgdb() -> Path:
    """Fail before a long run unless CMGDB is the maintained fork.

    Returns the imported module's path.
    """
    missing = [name for name in FORK_ONLY_ATTRIBUTES if not hasattr(CMGDB, name)]
    if missing:
        raise RuntimeError(
            f"CMGDB {_version()} at {Path(CMGDB.__file__).resolve()} is missing "
            f"{', '.join(missing)}, so it is not the maintained fork. Install the "
            "fork with `uv sync` from the repository root, or from its prebuilt "
            "wheels: https://github.com/bernardorivas/CMGDB/releases"
        )
    return Path(CMGDB.__file__).resolve()


def cmgdb_provenance(checkout: Path | None = None) -> dict[str, Any]:
    """Describe the CMGDB build for a run manifest.

    ``checkout`` is the source tree to read git state from, used only when the
    imported module actually comes from it. A wheel install records its version
    instead, which is what pins it.
    """
    module_path = require_fork_cmgdb()
    state: dict[str, Any] = {
        "version": _version(),
        "module_path": str(module_path),
    }
    if checkout is not None and checkout.resolve() in module_path.parents:
        state["repository"] = str(checkout.resolve())
        state.update(_git_state(checkout.resolve()))
    else:
        state["source"] = "installed distribution"
    return state


def _version() -> str:
    try:
        return version("CMGDB")
    except PackageNotFoundError:  # pragma: no cover - import without metadata
        return "unknown"


def _git_state(repository: Path) -> dict[str, Any]:
    """Git revision and dirtiness, or a marker for a non-git source tree.

    A tarball or wheel-adjacent tree has no ``.git``; that is a valid
    installation, not an error, so it is recorded rather than raised.
    """
    if not (repository / ".git").exists():
        return {"git": "not a git checkout"}
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git": "not a git checkout"}
    return {"revision": revision, "dirty": bool(status.strip())}
