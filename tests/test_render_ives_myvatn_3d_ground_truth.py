"""Tests for the direct-Ives 3D display-cover renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "render_ives_myvatn_3d_ground_truth.py"
)
SPEC = importlib.util.spec_from_file_location("render_ives_ground_truth", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RENDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDER)


def test_display_cover_increases_level_until_labels_do_not_collide() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    source_indices = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    lower = np.zeros(3, dtype=np.float64)
    upper = np.ones(3, dtype=np.float64)

    coarse, diagnostics = RENDER._display_cover(
        labels,
        source_indices,
        lower,
        upper,
        source_level=6,
        display_level=3,
    )
    selected, selected_diagnostics = RENDER._choose_display_cover(
        labels,
        source_indices,
        lower,
        upper,
        source_level=6,
        minimum_display_level=3,
    )

    assert coarse.shape == (2, 7)
    assert diagnostics["cross_label_display_cell_overlaps"] == 1
    assert selected.shape == (2, 7)
    assert selected_diagnostics["display_level"] == 4
    assert selected_diagnostics["cross_label_display_cell_overlaps"] == 0
    assert selected_diagnostics["candidate_levels_tested"] == [3, 4]
