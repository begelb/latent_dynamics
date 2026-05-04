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


def load_config(path: str | Path, *, apply_defaults: bool = True) -> ExperimentConfig:
    """Load and validate an experiment config.

    Looks for ``_shared/defaults.yaml`` in the parent directory of ``path``
    and deep-merges it under the file's own keys.
    """
    cfg_path = Path(path)
    raw = _load_yaml(cfg_path)

    if apply_defaults:
        defaults_path = cfg_path.parent / "_shared" / DEFAULTS_BASENAME
        if defaults_path.exists():
            raw = _deep_merge(_load_yaml(defaults_path), raw)

    return ExperimentConfig.model_validate(raw)
