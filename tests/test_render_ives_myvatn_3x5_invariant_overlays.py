from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_ives_myvatn_3x5_invariant_overlays.py"
SPEC = importlib.util.spec_from_file_location("render_ives_overlays", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RENDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDER)


def test_rasterize_uniform_cells_preserves_every_labelled_cell() -> None:
    data = np.asarray(
        [
            [0.00, 0.00, 0.25, 0.25, 0.0],
            [0.25, 0.00, 0.50, 0.25, 1.0],
            [0.75, 0.75, 1.00, 1.00, 2.0],
        ]
    )
    image, shape = RENDER._rasterize_uniform_cells(
        data,
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 1.0]),
    )

    assert shape == (4, 4)
    assert image.shape == (4, 4, 4)
    assert np.count_nonzero(image[:, :, 3]) == 3
    assert image[0, 0].tolist() != image[0, 1].tolist()
    assert image[3, 3, 3] == 255


def test_rasterize_rejects_duplicate_or_misaligned_cells() -> None:
    duplicate = np.asarray(
        [
            [0.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        RENDER._rasterize_uniform_cells(
            duplicate,
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
        )

    misaligned = np.asarray([[0.1, 0.0, 0.6, 0.5, 0.0]])
    with pytest.raises(ValueError, match="aligned"):
        RENDER._rasterize_uniform_cells(
            misaligned,
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
        )


def test_memberships_include_shared_box_boundary() -> None:
    data = np.asarray(
        [
            [0.0, 0.0, 0.5, 1.0, 0.0],
            [0.5, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    points = np.asarray([[0.25, 0.5], [0.5, 0.5], [0.75, 0.5]])
    assert RENDER._memberships(data, points) == [["0"], ["0", "1"], ["1"]]


def test_view_limits_include_boxes_and_overlay_points() -> None:
    data = np.asarray([[0.2, 0.2, 0.3, 0.3, 0.0]])
    points = np.asarray([[0.85, 0.9], [0.1, 0.15]])
    lower, upper = RENDER._view_limits(
        data,
        points,
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 1.0]),
    )
    assert np.all(lower <= np.asarray([0.1, 0.15]))
    assert np.all(upper >= np.asarray([0.85, 0.9]))
    assert np.all(lower >= 0.0)
    assert np.all(upper <= 1.0)
