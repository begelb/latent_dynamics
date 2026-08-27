"""Which CMGDB build is installed, and what it can do.

The pipeline relies on CMGDB entry points that arrived in release 1.5.0 (the
native reachability queries, ``ComputeConleyIndexForCells``, the
``cache_map_graph`` kwarg). Older installations lack some of them. This module
reports the difference as a capability map rather than as a version check, so a
caller can require exactly the features it uses and a run manifest can record
what was actually available.

The distinction matters because "which version is this?" and "can this build do
what I am about to ask?" are different questions. Answering only the first
rejects a build that would have served perfectly, and gives an unhelpful
message when a genuinely missing routine is the problem.
"""

from __future__ import annotations

import inspect
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import CMGDB

__all__ = [
    "CAPABILITIES",
    "cmgdb_capabilities",
    "cmgdb_provenance",
    "missing_cmgdb_features",
    "require_cmgdb_features",
]

#: Feature name -> the CMGDB attributes that implement it. A feature is present
#: only when every attribute backing it is. Features whose presence cannot be
#: decided by attribute name alone use a predicate instead -- see PREDICATES.
CAPABILITIES: dict[str, tuple[str, ...]] = {
    # The corner-lattice box-map builder class. On some builds the attribute
    # resolves to a submodule rather than the class, so it needs a predicate
    # rather than an attribute check.
    "precomputed_box_map_class": (),
    # Conley index of an arbitrary cell subset, deriving the index pair
    # internally. ComputeConleyIndex takes the pair explicitly and needs one
    # uniform grid, so this is the only route for a pair that spans
    # subdivision depths.
    "conley_index_for_cells": ("ComputeConleyIndexForCells",),
    # Native reachability queries on the cached cell graph.
    "morse_directed_path_cells": ("MorseDirectedPathCells",),
    "morse_singleton_reachability": ("MorseSingletonReachability",),
    "morse_reachability_masks": ("MorseReachabilityMasks",),
    # The compute calls accept cache_map_graph= and return an eagerly cached
    # map_graph when asked; a lazy one can be upgraded via build_cache().
    "cache_map_graph_kwarg": (),
    "map_graph_build_cache": (),
}


def _has_lattice_box_map_class() -> bool:
    """Is ``CMGDB.PrecomputedBoxMap`` the corner-lattice builder class?

    The attribute may resolve to the submodule of the same name instead of the
    class, depending on how the installed build re-exports it, and checking the
    name alone would then call a module. The lattice builder is recognized by
    its ``(f, lower_bounds, upper_bounds, subdiv_max, ...)`` signature.
    """
    candidate = getattr(CMGDB, "PrecomputedBoxMap", None)
    if not inspect.isclass(candidate):
        return False
    try:
        parameters = inspect.signature(candidate.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    return "subdiv_max" in parameters


def _has_cache_map_graph_kwarg() -> bool:
    """Do the compute calls accept ``cache_map_graph=``?

    The compute functions are compiled extensions, so the kwarg is probed
    through the generated signature line in the docstring rather than
    ``inspect.signature``.
    """
    doc = getattr(CMGDB.ComputeMorseGraph, "__doc__", "") or ""
    return "cache_map_graph" in doc


def _has_map_graph_build_cache() -> bool:
    """Can a lazily returned map_graph be upgraded with ``build_cache()``?"""
    return hasattr(getattr(CMGDB, "MapGraph", None), "build_cache")


#: Features whose presence cannot be decided by attribute name alone.
PREDICATES = {
    "precomputed_box_map_class": _has_lattice_box_map_class,
    "cache_map_graph_kwarg": _has_cache_map_graph_kwarg,
    "map_graph_build_cache": _has_map_graph_build_cache,
}


def cmgdb_capabilities() -> dict[str, bool]:
    """Which named features the installed CMGDB provides."""
    return {
        feature: (
            PREDICATES[feature]()
            if feature in PREDICATES
            else all(hasattr(CMGDB, attr) for attr in attrs)
        )
        for feature, attrs in CAPABILITIES.items()
    }


def missing_cmgdb_features(*features: str) -> list[str]:
    """The requested features the installed CMGDB does not provide."""
    unknown = [f for f in features if f not in CAPABILITIES]
    if unknown:
        raise KeyError(
            f"unknown CMGDB feature(s) {unknown}; known: {sorted(CAPABILITIES)}"
        )
    available = cmgdb_capabilities()
    return [f for f in features if not available[f]]


def require_cmgdb_features(*features: str) -> Path:
    """Fail before a long run unless CMGDB provides every named feature.

    Returns the imported module's path. The error names the missing feature and
    the attributes behind it, so the message points at what to do rather than
    at which build is installed.
    """
    missing = missing_cmgdb_features(*features)
    if missing:
        detail = "; ".join(
            f"{f} (needs {', '.join(CAPABILITIES[f])})" if CAPABILITIES[f] else f
            for f in missing
        )
        raise RuntimeError(
            f"CMGDB {_version()} at {Path(CMGDB.__file__).resolve()} does not "
            f"provide: {detail}. Upgrade the package: "
            f"pip install --upgrade 'cmgdb>=1.5.0'."
        )
    return Path(CMGDB.__file__).resolve()


def cmgdb_provenance(checkout: Path | None = None) -> dict[str, Any]:
    """Describe the CMGDB build for a run manifest.

    Records rather than gates: a manifest must be writable for whatever build
    produced the run, including one missing newer routines. ``checkout`` is
    the source tree to read git state from, used only when the imported module
    actually comes from it; a wheel install records its version instead, which
    is what pins it.
    """
    module_path = Path(CMGDB.__file__).resolve()
    state: dict[str, Any] = {
        "version": _version(),
        "module_path": str(module_path),
        "capabilities": cmgdb_capabilities(),
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
