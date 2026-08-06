"""Ambient-coordinate scaler fit / save / load helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import MinMaxScaler


class IdentityScaler:
    """Scikit-learn-like no-op scaler for experiments trained in raw coordinates."""

    def __init__(self, n_features: int) -> None:
        self.n_features_in_ = int(n_features)

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {arr.shape[-1]}")
        return arr.copy()

    def inverse_transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.transform(x)


class FixedBoundsScaler:
    """Scale with a prescribed ambient box instead of observed data extrema.

    The forward map intentionally matches the archived examples exactly:
    ``(x - lower) / (upper - lower + epsilon)``.  Unlike fitting a
    :class:`~sklearn.preprocessing.MinMaxScaler`, this gives every replicate
    the same coordinates even when its finite training sample misses a box
    boundary.
    """

    def __init__(
        self,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        *,
        epsilon: float = 1e-6,
    ) -> None:
        lower = np.asarray(lower_bounds, dtype=np.float64)
        upper = np.asarray(upper_bounds, dtype=np.float64)
        epsilon = float(epsilon)
        if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
            raise ValueError("lower and upper bounds must be matching one-dimensional arrays")
        if lower.size == 0:
            raise ValueError("fixed bounds must contain at least one feature")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("fixed bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("each upper bound must be greater than its lower bound")
        if not np.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("epsilon must be finite and non-negative")

        self.lower_bounds = lower.copy()
        self.upper_bounds = upper.copy()
        self.epsilon = epsilon
        self.n_features_in_ = int(lower.size)
        self.data_min_ = self.lower_bounds.copy()
        self.data_max_ = self.upper_bounds.copy()
        self.data_range_ = self.upper_bounds - self.lower_bounds
        self.scale_ = 1.0 / (self.data_range_ + self.epsilon)
        self.min_ = -self.lower_bounds * self.scale_
        self.feature_range = (0.0, 1.0)

    def _array(self, x: ArrayLike) -> NDArray[np.float64]:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 0 or arr.shape[-1] != self.n_features_in_:
            actual = 0 if arr.ndim == 0 else arr.shape[-1]
            raise ValueError(f"expected {self.n_features_in_} features, got {actual}")
        return arr

    def transform(self, x: ArrayLike) -> NDArray[np.float64]:
        arr = self._array(x)
        return (arr - self.lower_bounds) / (self.data_range_ + self.epsilon)

    def inverse_transform(self, x: ArrayLike) -> NDArray[np.float64]:
        arr = self._array(x)
        return self.lower_bounds + arr * (self.data_range_ + self.epsilon)


Scaler = MinMaxScaler | IdentityScaler | FixedBoundsScaler


def fit_minmax_scaler(x: NDArray[np.float64], y: NDArray[np.float64]) -> MinMaxScaler:
    """Fit a MinMaxScaler on the union of x_t and x_{t+1} samples."""
    combined = np.vstack([x, y])
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(combined)
    return scaler


def fit_identity_scaler(n_features: int) -> IdentityScaler:
    """Return a no-op scaler with the same transform API as MinMaxScaler."""
    return IdentityScaler(n_features=n_features)


def fit_fixed_bounds_scaler(
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    *,
    epsilon: float = 1e-6,
) -> FixedBoundsScaler:
    """Return a scaler pinned to a known ambient-coordinate box."""
    return FixedBoundsScaler(lower_bounds, upper_bounds, epsilon=epsilon)


def save_scaler(scaler: Scaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str | Path) -> Scaler:
    return joblib.load(path)
