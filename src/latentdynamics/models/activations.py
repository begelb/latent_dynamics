"""Activation function registry; honours the value of ``cfg.arch.activation``."""

from __future__ import annotations

from torch import nn

_HIDDEN: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}

_TERMINAL: dict[str, type[nn.Module] | None] = {
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "none": None,
}


def hidden_activation(name: str) -> nn.Module:
    """Instantiate a hidden-layer activation by name."""
    key = name.lower()
    if key not in _HIDDEN:
        raise KeyError(f"unknown hidden activation {name!r}; valid: {sorted(_HIDDEN)}")
    return _HIDDEN[key](inplace=True) if key == "relu" else _HIDDEN[key]()


def terminal_activation(name: str) -> nn.Module | None:
    """Instantiate a terminal (output-layer) activation by name; ``'none'`` returns None."""
    key = name.lower()
    if key not in _TERMINAL:
        raise KeyError(f"unknown terminal activation {name!r}; valid: {sorted(_TERMINAL)}")
    cls = _TERMINAL[key]
    return cls() if cls is not None else None
