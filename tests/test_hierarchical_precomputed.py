"""Tests for the lookup-only hierarchical CMGDB box map."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from latentdynamics.analysis.hierarchical_precomputed import (
    HierarchicalPrecomputedBoxMap,
)


class CountingLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.register_buffer(
            "matrix",
            torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32),
        )

    def forward(self, x):
        self.calls += 1
        return x @ self.matrix.T


class CountingScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return 1.5 * x - 0.25


def _direct_corner_bounds(module, rect, *, padding):
    rect = np.asarray(rect, dtype=np.float64)
    dim = rect.size // 2
    points = np.asarray(
        [
            [rect[axis + (dim if bit else 0)] for axis, bit in enumerate(combo)]
            for combo in np.ndindex(*(2,) * dim)
        ],
        dtype=np.float32,
    )
    with torch.no_grad():
        values = module(torch.as_tensor(points)).numpy()
    lo, hi = values.min(axis=0), values.max(axis=0)
    if padding:
        width = rect[dim:] - rect[:dim]
        lo, hi = lo - width, hi + width
    return np.concatenate((lo, hi))


def test_coarse_and_fine_callbacks_are_lookup_only_and_match_corner_evaluation():
    module = CountingLinear()
    box_map = HierarchicalPrecomputedBoxMap.precompute_coarse(
        module,
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        coarse_subdiv=4,
        fine_subdiv=8,
        padding=True,
        batch_points=5,
        device="cpu",
    )
    calls_after_coarse = module.calls

    active_box = np.asarray([[0.25, 0.5, 0.5, 0.75]])
    box_map.precompute_fine_blocks(
        module,
        active_box,
        batch_points=7,
        device="cpu",
    )
    assert module.calls > calls_after_coarse

    coarse_rect = [0.0, 0.0, 0.5, 0.5]
    fine_rect = [0.25, 0.5, 0.3125, 0.5625]
    expected_coarse = _direct_corner_bounds(module, coarse_rect, padding=True)
    expected_fine = _direct_corner_bounds(module, fine_rect, padding=True)
    calls_before_lookup = module.calls

    actual = np.asarray(box_map.batch([coarse_rect, fine_rect]))
    np.testing.assert_allclose(actual[0], expected_coarse, rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual[1], expected_fine, rtol=0, atol=1e-7)
    assert module.calls == calls_before_lookup


def test_round_trip_preserves_dense_and_sparse_lookup(tmp_path):
    module = CountingLinear()
    box_map = HierarchicalPrecomputedBoxMap.precompute_coarse(
        module,
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
        coarse_subdiv=2,
        fine_subdiv=4,
        padding=False,
        batch_points=3,
        device="cpu",
    )
    box_map.precompute_fine_blocks(
        module,
        [[-1.0, -1.0, 0.0, 0.0]],
        batch_points=3,
        device="cpu",
    )
    metadata = box_map.save(tmp_path)
    assert metadata.exists()

    loaded = HierarchicalPrecomputedBoxMap.load(tmp_path)
    rects = [[-1.0, -1.0, 0.0, 0.0], [-1.0, -1.0, -0.5, -0.5]]
    np.testing.assert_allclose(loaded.batch(rects), box_map.batch(rects), rtol=0, atol=0)


def test_rejects_fine_lookup_outside_prepared_coarse_cells():
    module = CountingLinear()
    box_map = HierarchicalPrecomputedBoxMap.precompute_coarse(
        module,
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        coarse_subdiv=2,
        fine_subdiv=4,
        padding=False,
        device="cpu",
    )
    box_map.precompute_fine_blocks(module, [[0.0, 0.0, 0.5, 0.5]], device="cpu")

    with pytest.raises(KeyError, match="unprepared coarse cell"):
        box_map([0.5, 0.5, 0.75, 0.75])


def test_fine_rectangle_can_take_corners_from_different_coarse_blocks():
    module = CountingScale()
    box_map = HierarchicalPrecomputedBoxMap.precompute_coarse(
        module,
        lower=[0.0],
        upper=[1.0],
        coarse_subdiv=2,
        fine_subdiv=4,
        padding=True,
        device="cpu",
    )
    box_map.precompute_fine_blocks(
        module,
        [[0.0, 0.25], [0.25, 0.5]],
        device="cpu",
    )
    # Fine-grid corners 3/16 and 5/16 lie strictly in different level-2
    # coarse cells, while the rectangle remains narrower than one coarse cell.
    rect = [3.0 / 16.0, 5.0 / 16.0]
    expected = _direct_corner_bounds(module, rect, padding=True)
    calls_before_lookup = module.calls

    actual = np.asarray(box_map(rect))

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    assert module.calls == calls_before_lookup


def test_boundary_corner_falls_back_to_any_prepared_adjacent_sparse_block():
    module = CountingScale()
    box_map = HierarchicalPrecomputedBoxMap.precompute_coarse(
        module,
        lower=[0.0, 0.0, 0.0],
        upper=[1.0, 1.0, 1.0],
        coarse_subdiv=3,
        fine_subdiv=6,
        padding=False,
        device="cpu",
    )
    # Prepare only the negative-side block. At the upper corner [0.5]^3,
    # the primary positive-side block [1, 1, 1] and every other positive-side
    # neighbor are inactive; the corner must fall back to block [0, 0, 0].
    box_map.precompute_fine_blocks(
        module,
        [[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]],
        device="cpu",
    )
    rect = [0.25, 0.25, 0.25, 0.5, 0.5, 0.5]
    expected = _direct_corner_bounds(module, rect, padding=False)
    calls_before_lookup = module.calls

    actual = np.asarray(box_map(rect))

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    assert module.calls == calls_before_lookup
