"""Initial-condition sampling and trajectory dataset construction."""

from .scaling import (
    IdentityScaler,
    fit_identity_scaler,
    fit_minmax_scaler,
    load_scaler,
    save_scaler,
)
from .strategies import SamplingStrategy, SobolStrategy, UniformStrategy, build_strategy
from .trajectories import TrajectoryDataset, sample_trajectories

__all__ = [
    "IdentityScaler",
    "SamplingStrategy",
    "SobolStrategy",
    "TrajectoryDataset",
    "UniformStrategy",
    "build_strategy",
    "fit_identity_scaler",
    "fit_minmax_scaler",
    "load_scaler",
    "sample_trajectories",
    "save_scaler",
]
