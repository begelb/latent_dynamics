"""YAML loader for experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import ExperimentConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping; got {type(data).__name__} in {path}")
    return data


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config.

    Every config is expected to be fully self-contained: schema-level pydantic
    defaults fill in fields the YAML omits, but in practice configs spell out
    every field so users see every knob that can be tuned.
    """
    cfg_path = Path(path)
    raw = _load_yaml(cfg_path)
    if not raw.get("experiment_name"):
        raw["experiment_name"] = cfg_path.stem
    return ExperimentConfig.model_validate(raw)
