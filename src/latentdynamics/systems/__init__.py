"""Dynamical systems studied in the paper.

The package exposes a small registry mapping the YAML ``system.name`` field
to a concrete :class:`DynamicalSystem` subclass.
"""

from __future__ import annotations

from typing import Any

from .base import ContinuousFlow, DiscreteMap, DynamicalSystem
from .chafee_infante import ChafeeInfante
from .coral import RedCoralModel
from .ives import IvesModel
from .leslie import LeslieContraction, LeslieModel3D, LeslieModel4D

SYSTEM_REGISTRY: dict[str, type[DynamicalSystem]] = {
    "leslie_contraction": LeslieContraction,
    "leslie3d": LeslieModel3D,
    "leslie4d": LeslieModel4D,
    "coral": RedCoralModel,
    "chafee_infante": ChafeeInfante,
    "ives": IvesModel,
}


def build_system(name: str, params: dict[str, Any] | None = None) -> DynamicalSystem:
    """Instantiate a system by registry name with optional keyword parameters."""
    if name not in SYSTEM_REGISTRY:
        valid = sorted(SYSTEM_REGISTRY.keys())
        raise KeyError(f"unknown system {name!r}; valid choices: {valid}")
    return SYSTEM_REGISTRY[name](**(params or {}))


__all__ = [
    "SYSTEM_REGISTRY",
    "ChafeeInfante",
    "ContinuousFlow",
    "DiscreteMap",
    "DynamicalSystem",
    "IvesModel",
    "LeslieContraction",
    "LeslieModel3D",
    "LeslieModel4D",
    "RedCoralModel",
    "build_system",
]
