"""Initial-condition sampling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc


class SamplingStrategy(ABC):
    """Draw ``n`` initial conditions inside a hyper-rectangle."""

    @abstractmethod
    def sample(
        self,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
        n: int,
    ) -> NDArray[np.float64]:
        ...


class UniformStrategy(SamplingStrategy):
    """Independent uniform draws inside the bounding box."""

    def __init__(self, rng: np.random.Generator | int | None = None) -> None:
        self._rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def sample(
        self,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
        n: int,
    ) -> NDArray[np.float64]:
        return self._rng.uniform(lower_bounds, upper_bounds, size=(n, lower_bounds.shape[0]))


class SobolStrategy(SamplingStrategy):
    """Scrambled Sobol low-discrepancy sequence inside the bounding box."""

    def __init__(self, seed: int = 42, scramble: bool = True) -> None:
        self.seed = int(seed)
        self.scramble = bool(scramble)

    def sample(
        self,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
        n: int,
    ) -> NDArray[np.float64]:
        sampler = qmc.Sobol(
            d=lower_bounds.shape[0],
            scramble=self.scramble,
            seed=self.seed,
        )
        unit = sampler.random(n=n)
        return qmc.scale(unit, lower_bounds, upper_bounds)


def build_strategy(method: str, *, role: str = "train", config: object | None = None) -> SamplingStrategy:
    """Build a strategy from a sampling method name and an experiment config.

    For both ``uniform`` and ``sobol`` the seed is taken from
    ``config.sobol_<role>_seed`` (default 42 for ``train``, 9999 for ``test``)
    so data generation is reproducible without relying on global RNG state.
    """
    method = method.lower()
    default_seed = 42 if role == "train" else 9999
    seed = default_seed
    if config is not None:
        seed = getattr(config, f"sobol_{role}_seed", default_seed)

    if method == "uniform":
        return UniformStrategy(rng=int(seed))
    if method == "sobol":
        return SobolStrategy(seed=int(seed))
    if method == "adaptive":
        raise NotImplementedError(
            "adaptive sampling requires a trained model and is built in latentdynamics.experiments"
        )
    raise ValueError(f"unknown sampling method: {method!r}")
