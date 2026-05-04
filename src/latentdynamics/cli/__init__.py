"""Pipeline entry-points used by the thin scripts in /scripts."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = [
    "make_data",
    "metrics",
    "morse_graph",
    "pipeline",
    "render",
    "scale_data",
    "train",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
