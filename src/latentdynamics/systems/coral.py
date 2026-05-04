"""Mediterranean red-coral 13-class Leslie-type matrix model (paper Sec. 1.297)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import DiscreteMap


class RedCoralModel(DiscreteMap):
    """Density-dependent 13-class population model with two stable fixed points."""

    DEFAULT_BIRTH_RATES: tuple[float, ...] = (
        0.0, 0.0, 2.89, 10.03, 21.59, 39.02, 56.41, 77.72, 103.23, 131.87,
        164.57, 201.46, 242.65,
    )
    DEFAULT_SURVIVAL_RATES: tuple[float, ...] = (
        0.889, 0.633, 0.697, 0.517, 0.437, 0.287, 0.571, 0.333, 0.75, 1.0,
        0.333, 1.0,
    )
    DEFAULT_UPPER_BOUNDS: tuple[float, ...] = (
        1300, 1150, 750, 520, 270, 120, 35, 20, 7, 5, 5, 2, 2,
    )

    FIXED_POINTS: dict[str, NDArray[np.float64]] = {
        "a0": np.zeros(13),
        "a1": np.array([
            868.12066371, 771.75927004, 488.52361793, 340.5009617, 176.0389972,
            76.92904178, 22.07863499, 12.60690058, 4.19809789, 3.14857342,
            3.14857342, 1.04847495, 1.04847495,
        ]),
        "r": np.array([
            321.84389612752153, 286.11922365736666, 181.1134685751131,
            126.23608759685382, 65.26405728757342, 28.52039303466959,
            8.185352800950172, 4.673836449342548, 1.5563875376310683,
            1.1672906532233014, 1.1672906532233014, 0.3887077875233593,
            0.3887077875233593,
        ]),
    }

    def __init__(
        self,
        b: ArrayLike | None = None,
        survival_rates: ArrayLike | None = None,
        surface_area: float = 36.0,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
    ) -> None:
        self.b = np.asarray(
            b if b is not None else self.DEFAULT_BIRTH_RATES, dtype=np.float64
        )
        self.survival_rates = np.asarray(
            survival_rates if survival_rates is not None else self.DEFAULT_SURVIVAL_RATES,
            dtype=np.float64,
        )
        self.surface_area = float(surface_area)
        self._set_bounds(
            lower_bounds if lower_bounds is not None else np.zeros(13),
            upper_bounds if upper_bounds is not None else self.DEFAULT_UPPER_BOUNDS,
        )

    def step(self, x: ArrayLike) -> NDArray[np.float64]:
        x_arr = np.asarray(x, dtype=np.float64)
        total = x_arr.sum(axis=-1)
        adult_density = (total - x_arr[..., 0]) / self.surface_area
        # Density-dependent larval survival L(rho); see paper Eq. (coral-recruitment).
        larval_survival = 2.94 / (adult_density + 520.0 * np.exp(-0.14 * adult_density))
        recruits = larval_survival * (x_arr * self.b).sum(axis=-1)
        survivors = x_arr[..., :-1] * self.survival_rates
        return np.concatenate([recruits[..., None], survivors], axis=-1)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "b": self.b.tolist(),
            "survival_rates": self.survival_rates.tolist(),
            "surface_area": self.surface_area,
        }
