"""Numerical regression tests for ground-truth systems."""

from __future__ import annotations

import math

import numpy as np
import pytest

from latentdynamics.config import load_config
from latentdynamics.systems import (
    SYSTEM_REGISTRY,
    ChafeeInfante,
    DiscreteMap,
    LeslieContraction,
    LeslieModel3D,
    LeslieModel4D,
    RedCoralModel,
    build_system,
)


class TestLeslie3D:
    def test_step_against_hand_computation(self):
        m = LeslieModel3D()
        x = np.array([1.0, 2.0, 3.0])
        result = m.step(x)
        expected = np.array(
            [
                (28.9 * 1 + 29.8 * 2 + 22.0 * 3) * math.exp(-0.6),
                0.7 * 1,
                0.7 * 2,
            ]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_vectorised_matches_scalar(self):
        m = LeslieModel3D()
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 100, size=(5, 3))
        batch_out = m.step(X)
        scalar_out = np.stack([m.step(row) for row in X])
        np.testing.assert_allclose(batch_out, scalar_out, rtol=1e-14)

    def test_legacy_f_matches_step(self):
        m = LeslieModel3D()
        x = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(m.f(x), m.step(np.asarray(x)), rtol=1e-14)


class TestLeslieContraction:
    def test_step_against_hand_computation(self):
        m = LeslieContraction(
            th1=23.5,
            th2=23.5,
            survival_p1=0.7,
            contraction=0.25,
            lower_bounds=[0] * 6,
            upper_bounds=[100] * 6,
        )
        x = np.array([10.0, 5.0, 2.0, 4.0, 6.0, 8.0])
        result = m.step(x)
        expected_head0 = (23.5 * 10 + 23.5 * 5) * math.exp(-0.1 * 15)
        np.testing.assert_allclose(
            result,
            np.array([expected_head0, 7.0, 0.5, 1.0, 1.5, 2.0]),
            rtol=1e-12,
        )

    def test_dim_set_from_bounds(self):
        m = LeslieContraction()
        assert m.dim == 10
        assert m.lower_bounds.shape == (10,)

    def test_paper_config_matches_exact_2d_restriction(self):
        config = load_config("leslie_2gen_contraction")
        m = build_system(config.system.name, config.system.params)

        assert m.params == {
            "th1": 23.5,
            "th2": 23.5,
            "survival_p1": 0.7,
            "contraction": 0.25,
        }
        np.testing.assert_array_equal(m.lower_bounds, np.zeros(10))
        np.testing.assert_array_equal(
            m.upper_bounds,
            np.array([90.0, 70.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
        )

        points_2d = np.array([[0.0, 0.0], [10.0, 5.0], [90.0, 70.0]])
        embedded = np.zeros((len(points_2d), 10))
        embedded[:, :2] = points_2d
        projected = m.step(embedded)[:, :2]
        x0, x1 = points_2d.T
        expected = np.column_stack(((23.5 * x0 + 23.5 * x1) * np.exp(-0.1 * (x0 + x1)), 0.7 * x0))
        np.testing.assert_allclose(projected, expected, rtol=1e-14, atol=0.0)


class TestLeslieModel4D:
    def test_step_shape(self):
        m = LeslieModel4D()
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 50, size=(7, 4))
        out = m.step(X)
        assert out.shape == (7, 4)


class TestRedCoral:
    def test_step_against_legacy_implementation(self):
        m = RedCoralModel()
        x = np.array([100.0, 80.0, 50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 1.0, 0.5, 0.5, 0.2, 0.1])

        adult = (x.sum() - x[0]) / m.surface_area
        ls = 2.94 / (adult + 520.0 * math.exp(-0.14 * adult))
        recruits = ls * float(np.dot(x, m.b))
        survivors = (x[:-1] * m.survival_rates).tolist()
        expected = np.array([recruits, *survivors])

        np.testing.assert_allclose(m.step(x), expected, rtol=1e-12)

    def test_vectorised_matches_scalar(self):
        m = RedCoralModel()
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 100, size=(4, 13))
        batch_out = m.step(X)
        scalar_out = np.stack([m.step(row) for row in X])
        np.testing.assert_allclose(batch_out, scalar_out, rtol=1e-14)

    def test_fixed_points_have_correct_dimension(self):
        for name, point in RedCoralModel.FIXED_POINTS.items():
            assert point.shape == (13,), f"fixed point {name} has wrong shape"

    def test_a0_is_a_fixed_point(self):
        m = RedCoralModel()
        a0 = RedCoralModel.FIXED_POINTS["a0"]
        np.testing.assert_allclose(m.step(a0), a0, atol=1e-12)


class TestChafeeInfante:
    def test_zero_is_a_fixed_point_of_vector_field(self):
        s = ChafeeInfante(N=8)
        np.testing.assert_allclose(s.vector_field(0.0, np.zeros(8)), np.zeros(8), atol=1e-14)

    def test_step_returns_correct_shape(self):
        s = ChafeeInfante(N=8, tau=0.05)
        x = np.zeros(8)
        x[0] = 0.1
        out = s.step(x)
        assert out.shape == (8,)
        assert np.isfinite(out).all()

    def test_bounds_match_amplitude_envelope(self):
        s = ChafeeInfante(N=4, amplitude=2.0, decay=0.5)
        expected = 2.0 * np.exp(-0.5 * np.arange(4))
        np.testing.assert_allclose(s.upper_bounds, expected, rtol=1e-14)
        np.testing.assert_allclose(s.lower_bounds, -expected, rtol=1e-14)


class TestRegistry:
    @pytest.mark.parametrize("name", sorted(SYSTEM_REGISTRY.keys()))
    def test_build_system_with_defaults(self, name: str):
        params: dict = {}
        if name == "chafee_infante":
            params = {"N": 8}
        sys_obj = build_system(name, params)
        assert isinstance(sys_obj, DiscreteMap) or hasattr(sys_obj, "vector_field")
        assert sys_obj.dim >= 1

    def test_unknown_system_raises(self):
        with pytest.raises(KeyError):
            build_system("not_a_system")
