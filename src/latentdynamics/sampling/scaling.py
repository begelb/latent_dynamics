"""MinMax scaler fit / save / load helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler


class IdentityScaler:
    """Scikit-learn-like no-op scaler for experiments trained in raw coordinates."""

    def __init__(self, n_features: int) -> None:
        self.n_features_in_ = int(n_features)

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {arr.shape[-1]}"
            )
        return arr.copy()

    def inverse_transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.transform(x)


def fit_minmax_scaler(x: NDArray[np.float64], y: NDArray[np.float64]) -> MinMaxScaler:
    """Fit a MinMaxScaler on the union of x_t and x_{t+1} samples."""
    combined = np.vstack([x, y])
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(combined)
    return scaler


def fit_identity_scaler(n_features: int) -> IdentityScaler:
    """Return a no-op scaler with the same transform API as MinMaxScaler."""
    return IdentityScaler(n_features=n_features)


def save_scaler(scaler: MinMaxScaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str | Path) -> MinMaxScaler:
    return joblib.load(path)
