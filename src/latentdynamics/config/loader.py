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


def _get_packaged_configs_dir() -> Path:
    """Return the path to the bundled configs directory in the package."""
    return Path(__file__).resolve().parent.parent / "configs"


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config.

    Args:
        path: Either a bare name (e.g. ``"chafee_infante"``, resolved from
            the packaged configs), a relative path (resolved from the project
            repo root if available, else the cache dir), or an absolute path.

    Every config is expected to be fully self-contained: schema-level pydantic
    defaults fill in fields the YAML omits, but in practice configs spell out
    every field so users see every knob that can be tuned.
    """
    cfg_path = Path(path)

    # If it's already a valid absolute path to a file, use it as-is.
    if cfg_path.is_absolute() and cfg_path.is_file():
        raw = _load_yaml(cfg_path)
        if not raw.get("experiment_name"):
            raw["experiment_name"] = cfg_path.stem
        return ExperimentConfig.model_validate(raw)

    # If it has a .yaml suffix and exists as a relative path, use it.
    if cfg_path.suffix == ".yaml" and cfg_path.is_file():
        raw = _load_yaml(cfg_path)
        if not raw.get("experiment_name"):
            raw["experiment_name"] = cfg_path.stem
        return ExperimentConfig.model_validate(raw)

    # Otherwise, treat it as a bare name and look it up in the packaged configs.
    # (Ignore any .yaml suffix if present; add it ourselves.)
    stem = cfg_path.stem if cfg_path.suffix else cfg_path.name
    packaged_path = _get_packaged_configs_dir() / f"{stem}.yaml"
    if packaged_path.is_file():
        raw = _load_yaml(packaged_path)
        if not raw.get("experiment_name"):
            raw["experiment_name"] = stem
        return ExperimentConfig.model_validate(raw)

    # If nothing matched, raise a helpful error.
    raise FileNotFoundError(
        f"no config for {path!r}; expected a packaged config name "
        f"(from {_get_packaged_configs_dir()}), an existing file path, or an absolute path"
    )
