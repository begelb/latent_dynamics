"""Chafee-Infante PDE in spectral form (paper Sec. 1.256)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import ContinuousFlow


class ChafeeInfante(ContinuousFlow):
    """Spectral truncation of u_t = u_xx + alpha (u - u^3) on (0, pi)."""

    def __init__(
        self,
        N: int = 64,
        alpha: float = 28.0,
        tau: float = 0.1,
        amplitude: float = 2.0,
        decay: float = 0.5,
    ) -> None:
        if N < 1:
            raise ValueError("N must be a positive integer")
        self.N = int(N)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.amplitude = float(amplitude)
        self.decay = float(decay)
        self._L_eig = -(np.arange(1, self.N + 1, dtype=np.float64) ** 2)
        self._set_bounds(
            lower=-self.amplitude * np.exp(-self.decay * np.arange(self.N)),
            upper=self.amplitude * np.exp(-self.decay * np.arange(self.N)),
        )

    def _nonlinear(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        a_ext = np.concatenate((-a[::-1], np.zeros(1), a))
        conv2 = np.convolve(a_ext, a_ext)
        conv3 = np.convolve(conv2, a_ext)
        center = 3 * self.N
        return conv3[center + 1 : center + 1 + self.N]

    def vector_field(self, t: float, a: NDArray[np.float64]) -> NDArray[np.float64]:
        return (self.alpha + self._L_eig) * a + (self.alpha / 4.0) * self._nonlinear(a)

    def sample_initial_condition(self, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.uniform(-self.amplitude, self.amplitude, self.N) * np.exp(
            -self.decay * np.arange(self.N)
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "N": self.N,
            "alpha": self.alpha,
            "tau": self.tau,
            "amplitude": self.amplitude,
            "decay": self.decay,
        }
