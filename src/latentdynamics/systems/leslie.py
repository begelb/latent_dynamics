"""Leslie population maps used in the paper (2D contraction, 3D, 4D)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import DiscreteMap


class LeslieContraction(DiscreteMap):
    """Two-class Leslie map padded with contracting tail dimensions.

    The first two coordinates evolve under the classical Leslie recurrence
    with Ricker-type density dependence; remaining coordinates contract by
    a factor of 0.25 each step. Used to demonstrate the 2D Morse graph in
    the paper (Fig. 1.83).
    """

    def __init__(
        self,
        th1: float = 23.5,
        th2: float = 23.5,
        survival_p1: float = 0.7,
        contraction: float = 0.25,
        lower_bounds: ArrayLike = (0,) * 10,
        upper_bounds: ArrayLike = (90, 70, 100, 100, 100, 100, 100, 100, 100, 100),
    ) -> None:
        self.th1 = float(th1)
        self.th2 = float(th2)
        self.survival_p1 = float(survival_p1)
        self.contraction = float(contraction)
        self._set_bounds(lower_bounds, upper_bounds)

    def step(self, x: ArrayLike) -> NDArray[np.float64]:
        x_arr = np.asarray(x, dtype=np.float64)
        s = x_arr[..., 0] + x_arr[..., 1]
        decay = np.exp(-0.1 * s)
        head0 = (self.th1 * x_arr[..., 0] + self.th2 * x_arr[..., 1]) * decay
        head1 = self.survival_p1 * x_arr[..., 0]
        tail = self.contraction * x_arr[..., 2:]
        return np.concatenate([head0[..., None], head1[..., None], tail], axis=-1)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "th1": self.th1,
            "th2": self.th2,
            "survival_p1": self.survival_p1,
            "contraction": self.contraction,
        }


class LeslieModel3D(DiscreteMap):
    """Three-class Leslie map with Ricker density dependence (paper Sec. 1.100)."""

    def __init__(
        self,
        th1: float = 19.6,
        th2: float = 23.68,
        th3: float = 23.68,
        survival_p1: float = 0.7,
        survival_p2: float = 0.7,
        lower_bounds: ArrayLike = (0, 0, 0),
        upper_bounds: ArrayLike = (220, 154, 108),
    ) -> None:
        self.th1 = float(th1)
        self.th2 = float(th2)
        self.th3 = float(th3)
        self.survival_p1 = float(survival_p1)
        self.survival_p2 = float(survival_p2)
        self._set_bounds(lower_bounds, upper_bounds)

    def step(self, x: ArrayLike) -> NDArray[np.float64]:
        x_arr = np.asarray(x, dtype=np.float64)
        s = x_arr[..., 0] + x_arr[..., 1] + x_arr[..., 2]
        decay = np.exp(-0.1 * s)
        head = (
            self.th1 * x_arr[..., 0]
            + self.th2 * x_arr[..., 1]
            + self.th3 * x_arr[..., 2]
        ) * decay
        c1 = self.survival_p1 * x_arr[..., 0]
        c2 = self.survival_p2 * x_arr[..., 1]
        return np.stack([head, c1, c2], axis=-1)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "th1": self.th1,
            "th2": self.th2,
            "th3": self.th3,
            "survival_p1": self.survival_p1,
            "survival_p2": self.survival_p2,
        }


class LeslieModel4D(DiscreteMap):
    """Four-class Leslie map; reference variant for higher-dimensional sweeps."""

    def __init__(
        self,
        th1: float = 80,
        th2: float = 80,
        th3: float = 80,
        th4: float = 80,
        p1: float = 0.5,
        p2: float = 0.7,
        p3: float = 0.7,
        lower_bounds: ArrayLike = (0, 0, 0, 0),
        upper_bounds: ArrayLike = (295, 148, 104, 73),
    ) -> None:
        self.th1 = float(th1)
        self.th2 = float(th2)
        self.th3 = float(th3)
        self.th4 = float(th4)
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.p3 = float(p3)
        self._set_bounds(lower_bounds, upper_bounds)

    def step(self, x: ArrayLike) -> NDArray[np.float64]:
        x_arr = np.asarray(x, dtype=np.float64)
        s = x_arr[..., 0] + x_arr[..., 1] + x_arr[..., 2] + x_arr[..., 3]
        decay = np.exp(-0.1 * s)
        head = (
            self.th1 * x_arr[..., 0]
            + self.th2 * x_arr[..., 1]
            + self.th3 * x_arr[..., 2]
            + self.th4 * x_arr[..., 3]
        ) * decay
        c1 = self.p1 * x_arr[..., 0]
        c2 = self.p2 * x_arr[..., 1]
        c3 = self.p3 * x_arr[..., 2]
        return np.stack([head, c1, c2, c3], axis=-1)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "th1": self.th1,
            "th2": self.th2,
            "th3": self.th3,
            "th4": self.th4,
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
        }
