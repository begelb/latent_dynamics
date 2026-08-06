"""Typed configuration system."""

from .loader import load_config
from .schema import (
    ArchConfig,
    CMGDBConfig,
    ComponentArchConfig,
    CurriculumLBFGSPolishConfig,
    CurriculumOptimizerConfig,
    CurriculumStageConfig,
    DataConfig,
    ExperimentConfig,
    PathsConfig,
    ResolvedComponentConfig,
    SystemConfig,
    TrainingConfig,
)

__all__ = [
    "ArchConfig",
    "CMGDBConfig",
    "ComponentArchConfig",
    "CurriculumLBFGSPolishConfig",
    "CurriculumOptimizerConfig",
    "CurriculumStageConfig",
    "DataConfig",
    "ExperimentConfig",
    "PathsConfig",
    "ResolvedComponentConfig",
    "SystemConfig",
    "TrainingConfig",
    "load_config",
]
