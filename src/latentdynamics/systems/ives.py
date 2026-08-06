"""Ives midge--algae--detritus map for the Lake Myvatn example."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import DiscreteMap


class IvesModel(DiscreteMap):
    """The Ives et al. ecological map expressed in base-10 log coordinates.

    The state coordinates are log10 midge, algae, and detritus abundances.  The
    defaults reproduce the canonical Lake Myvatn example in the archived
    implementation, including its sampling box and the exact paper value of
    the consumption exponent ``q``.
    """

    def __init__(
        self,
        r1: float = 3.873,
        r2: float = 11.746,
        c: float = 10**-6.435,
        d: float = 0.5517,
        p: float = 0.06659,
        q: float = 0.9026,
        coordinate_mode: str = "log",
        lower_bounds: ArrayLike = (-3.0, -7.5, -3.0),
        upper_bounds: ArrayLike = (1.5, 1.5, 1.5),
    ) -> None:
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.c = float(c)
        self.d = float(d)
        self.p = float(p)
        self.q = float(q)
        self.coordinate_mode = str(coordinate_mode)
        if self.r1 <= 0.0 or self.r2 <= 0.0:
            raise ValueError("r1 and r2 must be positive")
        if self.c <= 0.0:
            raise ValueError("c must be positive")
        if not 0.0 <= self.d <= 1.0:
            raise ValueError("d must lie in [0, 1]")
        if self.p < 0.0 or self.q < 0.0:
            raise ValueError("p and q must be non-negative")
        if self.coordinate_mode != "log":
            raise ValueError("IvesModel supports only coordinate_mode='log'")
        self._set_bounds(lower_bounds, upper_bounds)
        if self.dim != 3:
            raise ValueError("IvesModel bounds must have exactly three coordinates")
        if np.any(self.upper_bounds <= self.lower_bounds):
            raise ValueError("each upper bound must be greater than its lower bound")

    def step(self, x: ArrayLike) -> NDArray[np.float64]:
        """Advance one generation, preserving scalar or leading batch axes."""

        log_x = np.asarray(x, dtype=np.float64)
        if log_x.ndim == 0 or log_x.shape[-1] != self.dim:
            raise ValueError(f"expected states with final dimension {self.dim}")

        linear_x = np.power(10.0, log_x)
        midge = linear_x[..., 0]
        algae = linear_x[..., 1]
        detritus = linear_x[..., 2]
        resource = algae + self.p * detritus

        midge_next = self.r1 * midge * np.power(1.0 + midge / resource, -self.q)
        algae_consumed = (algae / resource) * midge_next
        detritus_consumed = (self.p * detritus / resource) * midge_next

        algae_next = np.maximum(
            self.c,
            self.r2 * algae / (1.0 + algae) - algae_consumed + self.c,
        )
        detritus_next = np.maximum(
            self.c,
            self.d * detritus + algae - detritus_consumed + self.c,
        )
        return np.log10(np.stack([midge_next, algae_next, detritus_next], axis=-1))

    @property
    def params(self) -> dict[str, Any]:
        return {
            "r1": self.r1,
            "r2": self.r2,
            "c": self.c,
            "d": self.d,
            "p": self.p,
            "q": self.q,
            "coordinate_mode": self.coordinate_mode,
        }


__all__ = ["IvesModel"]
