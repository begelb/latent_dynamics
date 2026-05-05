"""YAML loader with deep-merge of a shared defaults file."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .schema import ExperimentConfig

DEFAULTS_BASENAME = "defaults.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping; got {type(data).__name__} in {path}")
    return data


def _find_defaults(start: Path) -> Path | None:
    """Walk up from ``start`` looking for ``_shared/defaults.yaml``.

    Returns the first match, or None. The walk is bounded by the filesystem
    root, so deeply nested configs (e.g. ``configs/scratch/X.yaml``) still
    inherit ``configs/_shared/defaults.yaml``.
    """
    current = start.resolve()
    while True:
        candidate = current / "_shared" / DEFAULTS_BASENAME
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def load_config(path: str | Path, *, apply_defaults: bool = True) -> ExperimentConfig:
    """Load and validate an experiment config.

    Looks for ``_shared/defaults.yaml`` in any ancestor directory of ``path``
    and deep-merges it under the file's own keys, so config files nested in
    subdirectories (e.g. ``configs/scratch/<expt>.yaml``) still inherit the
    shared defaults.
    """
    cfg_path = Path(path)
    raw = _load_yaml(cfg_path)

    if apply_defaults:
        defaults_path = _find_defaults(cfg_path.parent)
        if defaults_path is not None:
            raw = _deep_merge(_load_yaml(defaults_path), raw)

    if not raw.get("experiment_name"):
        raw["experiment_name"] = cfg_path.stem

    return ExperimentConfig.model_validate(raw)
