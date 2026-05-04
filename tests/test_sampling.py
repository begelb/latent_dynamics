"""Tests for sampling strategies and trajectory generation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from latentdynamics.sampling import (
    IdentityScaler,
    SobolStrategy,
    UniformStrategy,
    build_strategy,
    fit_identity_scaler,
    fit_minmax_scaler,
    load_scaler,
    sample_trajectories,
    save_scaler,
)
from latentdynamics.systems import LeslieModel3D, RedCoralModel


class TestStrategies:
    def test_uniform_samples_inside_bounds(self):
        rng = np.random.default_rng(0)
        strategy = UniformStrategy(rng=rng)
        lo = np.array([-1.0, 0.0, 5.0])
        hi = np.array([1.0, 2.0, 10.0])
        samples = strategy.sample(lo, hi, 256)
        assert samples.shape == (256, 3)
        assert np.all(samples >= lo)
        assert np.all(samples <= hi)

    def test_sobol_is_seed_reproducible(self):
        lo = np.zeros(2)
        hi = np.ones(2)
        a = SobolStrategy(seed=7).sample(lo, hi, 32)
        b = SobolStrategy(seed=7).sample(lo, hi, 32)
        np.testing.assert_array_equal(a, b)

    def test_sobol_different_seeds_differ(self):
        lo = np.zeros(2)
        hi = np.ones(2)
        a = SobolStrategy(seed=1).sample(lo, hi, 32)
        b = SobolStrategy(seed=2).sample(lo, hi, 32)
        assert not np.allclose(a, b)

    def test_build_strategy_uniform_and_sobol(self):
        assert isinstance(build_strategy("uniform"), UniformStrategy)
        assert isinstance(build_strategy("sobol"), SobolStrategy)

    def test_build_strategy_adaptive_not_implemented(self):
        with pytest.raises(NotImplementedError):
            build_strategy("adaptive")

    def test_build_strategy_unknown(self):
        with pytest.raises(ValueError):
            build_strategy("nonsense")


class TestTrajectories:
    def test_pair_count_and_shape(self):
        system = LeslieModel3D()
        ds = sample_trajectories(
            system=system,
            strategy=UniformStrategy(rng=0),
            n_samples=12,
            n_iterations=4,
            skip=1,
        )
        # Records 4-1=3 pairs per IC; 12 ICs => 36 rows.
        assert ds.X.shape == (36, 3)
        assert ds.Y.shape == (36, 3)
        assert ds.dim == 3
        assert ds.header == "x0,x1,x2,y0,y1,y2"

    def test_skip_must_be_less_than_iterations(self):
        with pytest.raises(ValueError):
            sample_trajectories(LeslieModel3D(), UniformStrategy(rng=0), 4, 3, skip=3)

    def test_pair_consistency_with_step(self):
        system = LeslieModel3D()
        ds = sample_trajectories(system, UniformStrategy(rng=42), 5, 2, skip=0)
        np.testing.assert_allclose(system.step(ds.X), ds.Y, rtol=1e-12)

    def test_csv_and_metadata_roundtrip(self, tmp_path):
        system = RedCoralModel()
        ds = sample_trajectories(system, UniformStrategy(rng=0), 3, 2)
        csv_path = tmp_path / "data.csv"
        meta_path = tmp_path / "data_metadata.json"
        ds.to_csv(csv_path)
        ds.save_metadata(meta_path)

        loaded = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        assert loaded.shape == (6, 26)
        np.testing.assert_allclose(loaded[:, :13], ds.X, rtol=1e-6)
        np.testing.assert_allclose(loaded[:, 13:], ds.Y, rtol=1e-6)

        with meta_path.open() as f:
            meta = json.load(f)
        assert meta["system"] == "RedCoralModel"
        assert meta["dimension"] == 13
        assert meta["n_samples"] == 3
        assert meta["n_iterations"] == 2


class TestScaling:
    def test_fit_save_load_roundtrip(self, tmp_path):
        rng = np.random.default_rng(0)
        x = rng.uniform(-3, 7, size=(50, 4))
        y = rng.uniform(-3, 7, size=(50, 4))
        scaler = fit_minmax_scaler(x, y)

        path = tmp_path / "scaler.gz"
        save_scaler(scaler, path)
        loaded = load_scaler(path)

        np.testing.assert_allclose(scaler.transform(x), loaded.transform(x), rtol=1e-14)
        # Combined min/max should map to 0/1 within the joint domain.
        combined = np.vstack([x, y])
        scaled = scaler.transform(combined)
        assert scaled.min() == pytest.approx(0.0, abs=1e-12)
        assert scaled.max() == pytest.approx(1.0, abs=1e-12)

    def test_identity_scaler_roundtrip(self):
        x = np.array([[1.0, -2.0], [3.0, 4.0]])
        scaler = fit_identity_scaler(2)
        assert isinstance(scaler, IdentityScaler)
        np.testing.assert_array_equal(scaler.transform(x), x)
        np.testing.assert_array_equal(scaler.inverse_transform(x), x)
