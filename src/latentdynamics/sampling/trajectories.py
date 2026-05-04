"""Trajectory dataset construction by iterating a dynamical system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..systems.base import DynamicalSystem
from .strategies import SamplingStrategy


@dataclass(frozen=True)
class TrajectoryDataset:
    """A flat collection of (x_t, x_{t+1}) pairs from many trajectories."""

    X: NDArray[np.float64]
    Y: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def header(self) -> str:
        d = self.dim
        return ",".join([f"x{i}" for i in range(d)] + [f"y{i}" for i in range(d)])

    def to_csv(self, path: str | Path, *, fmt: str = "%.8f") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.hstack([self.X, self.Y])
        np.savetxt(path, data, delimiter=",", header=self.header, comments="", fmt=fmt)

    def save_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.metadata, f, indent=4, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


def sample_trajectories(
    system: DynamicalSystem,
    strategy: SamplingStrategy,
    n_samples: int,
    n_iterations: int,
    skip: int = 0,
    *,
    metadata_extra: dict[str, Any] | None = None,
) -> TrajectoryDataset:
    """Iterate the system from ``n_samples`` initial conditions for ``n_iterations`` steps.

    Pairs (x_t, x_{t+1}) are recorded for ``t >= skip`` only.
    """
    if n_iterations <= skip:
        raise ValueError(f"n_iterations ({n_iterations}) must exceed skip ({skip})")

    initial_conditions = strategy.sample(system.lower_bounds, system.upper_bounds, n_samples)

    points = initial_conditions
    X_chunks: list[NDArray[np.float64]] = []
    Y_chunks: list[NDArray[np.float64]] = []
    for iteration in range(n_iterations):
        next_points = system.step(points)
        if iteration >= skip:
            X_chunks.append(points)
            Y_chunks.append(next_points)
        points = next_points

    X = np.concatenate(X_chunks, axis=0)
    Y = np.concatenate(Y_chunks, axis=0)

    metadata: dict[str, Any] = {
        "system": type(system).__name__,
        "dimension": system.dim,
        "n_samples": int(n_samples),
        "n_iterations": int(n_iterations),
        "skip_initial_steps": int(skip),
        "lower_bounds": system.lower_bounds.tolist(),
        "upper_bounds": system.upper_bounds.tolist(),
        "model_params": system.params,
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    return TrajectoryDataset(X=X, Y=Y, metadata=metadata)
