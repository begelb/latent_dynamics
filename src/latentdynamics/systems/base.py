"""Abstract base classes for dynamical systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp


class DynamicalSystem(ABC):
    """Common interface for the ground-truth systems studied in the paper."""

    dim: int
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]

    def _set_bounds(self, lower: ArrayLike, upper: ArrayLike) -> None:
        self.lower_bounds = np.asarray(lower, dtype=np.float64)
        self.upper_bounds = np.asarray(upper, dtype=np.float64)
        if self.lower_bounds.shape != self.upper_bounds.shape:
            raise ValueError("lower and upper bounds must have the same shape")
        self.dim = int(self.lower_bounds.shape[0])

    @abstractmethod
    def step(self, x: ArrayLike) -> NDArray[np.float64]:
        """Advance the state by one time unit. Vectorised over leading axis."""

    def f(self, x: ArrayLike) -> list[float]:
        """Backward-compatible scalar wrapper for legacy callers."""
        return self.step(np.asarray(x, dtype=np.float64)).tolist()

    @property
    def params(self) -> dict[str, Any]:
        """Return system parameters as a JSON-serialisable dict (for metadata)."""
        return {}


class DiscreteMap(DynamicalSystem):
    """A system whose dynamics are defined by a discrete map x_{t+1} = f(x_t)."""


class ContinuousFlow(DynamicalSystem):
    """A system whose dynamics are an ODE; ``step`` is the time-tau map."""

    tau: float

    @abstractmethod
    def vector_field(self, t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Right-hand side of the ODE."""

    def time_tau_map(
        self,
        x0: NDArray[np.float64],
        tau: float | None = None,
        method: str = "RK45",
    ) -> NDArray[np.float64]:
        """Integrate the ODE from x0 over a time interval of length tau."""
        tau_eff = self.tau if tau is None else tau
        sol = solve_ivp(self.vector_field, (0.0, tau_eff), x0, method=method)
        return sol.y[:, -1]

    def step(self, x: ArrayLike) -> NDArray[np.float64]:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            return self.time_tau_map(x_arr)
        return np.stack([self.time_tau_map(row) for row in x_arr])
