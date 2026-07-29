"""Exactness checks for the adaptive three-dimensional coarse-set renderer."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


def _load_renderer_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "render_chafee_infante_3d_coarse.py"
    )
    spec = importlib.util.spec_from_file_location(
        "render_chafee_infante_3d_coarse",
        script,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RENDERER = _load_renderer_module()


def test_primary_cubical_axes_default_to_no_ticks_or_coordinate_labels() -> None:
    signature = inspect.signature(RENDERER._render_cubical)
    assert signature.parameters["show_ticks"].default is False
    assert signature.parameters["show_axis_labels"].default is False

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    RENDERER._configure_cubical_axes(ax)

    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_zlabel() == ""
    assert ax.get_xticks().size == 0
    assert ax.get_yticks().size == 0
    assert ax.get_zticks().size == 0
    assert not any(text.get_text() == "$z_3$" for text in ax.texts)
    plt.close(fig)


def test_cubical_axes_can_restore_ticks_and_coordinate_labels() -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    RENDERER._configure_cubical_axes(
        ax,
        show_ticks=True,
        show_axis_labels=True,
    )

    assert ax.get_xlabel() == "$z_1$"
    assert ax.get_ylabel() == "$z_2$"
    assert ax.get_zlabel() == ""
    assert ax.get_xticks().size > 0
    assert ax.get_yticks().size > 0
    assert ax.get_zticks().size > 0
    assert any(text.get_text() == "$z_3$" for text in ax.texts)
    plt.close(fig)


def _write_adaptive_example(path: Path) -> None:
    np.savetxt(
        path,
        np.asarray(
            [
                [0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0, 3.0, 1.0, 1.0, 1.0],
                [3.0, 0.0, 0.0, 4.0, 1.0, 1.0, 2.0],
            ]
        ),
        delimiter=",",
    )


def test_adaptive_surface_cache_preserves_exact_terminal_union(
    tmp_path: Path,
) -> None:
    """A scale-two box must equal its two terminal voxels and ten faces."""

    source = tmp_path / "morse_sets"
    _write_adaptive_example(source)
    cache = tmp_path / "cache"
    diagnostics = RENDERER._build_surface_cache(
        source,
        cache,
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([4.0, 2.0, 1.0]),
    )

    assert diagnostics["terminal_grid_shape"] == [4, 2, 1]
    assert diagnostics["terminal_voxel_count"] == 4
    assert diagnostics["terminal_voxel_counts_by_label"] == {
        "0": 2,
        "1": 1,
        "2": 1,
    }
    assert diagnostics["same_label_exposed_face_counts"] == {
        "0": 10,
        "1": 6,
        "2": 6,
    }
    assert diagnostics["cross_label_face_adjacency_counts"] == {
        "0,1": 1,
        "0,2": 0,
        "1,2": 1,
    }
    assert diagnostics["rendered_face_counts_by_label"] == {
        "0": 10,
        "1": 5,
        "2": 5,
    }
    assert diagnostics["rendered_face_count"] == 20

    codes = np.load(cache / "surface_face_codes.npy")
    sides = np.load(cache / "surface_face_sides.npy")
    faces, axes = RENDERER._decode_faces(
        codes,
        sides,
        encoding_base=diagnostics["face_encoding_base"],
        origin=np.asarray(diagnostics["bounds_lower"]),
        widths=np.asarray(diagnostics["terminal_widths"]),
    )
    assert faces.shape == (20, 4, 3)
    assert set(np.unique(axes)) == {0, 1, 2}
    assert np.all(faces >= np.asarray([0.0, 0.0, 0.0]))
    assert np.all(faces <= np.asarray([4.0, 1.0, 1.0]))


def test_surface_cache_rebuilds_for_bounds_or_file_integrity_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "morse_sets"
    _write_adaptive_example(source)
    cache = tmp_path / "cache"
    lower = np.asarray([0.0, 0.0, 0.0])
    original_upper = np.asarray([4.0, 2.0, 1.0])
    changed_upper = np.asarray([5.0, 2.0, 1.0])

    original = RENDERER._load_or_build_surface_cache(
        source,
        cache,
        lower,
        original_upper,
    )
    assert original["terminal_grid_shape"] == [4, 2, 1]

    changed = RENDERER._load_or_build_surface_cache(
        source,
        cache,
        lower,
        changed_upper,
    )
    assert changed["bounds_upper"] == changed_upper.tolist()
    assert changed["terminal_grid_shape"] == [5, 2, 1]

    labels_path = cache / "surface_face_labels.npy"
    labels_path.write_bytes(b"corrupt")
    repaired = RENDERER._load_or_build_surface_cache(
        source,
        cache,
        lower,
        changed_upper,
    )
    assert repaired["schema_version"] == RENDERER.SURFACE_CACHE_SCHEMA_VERSION
    assert np.load(labels_path).shape == (20,)


@pytest.mark.parametrize("overlap_label", [0, 1])
def test_terminal_expansion_rejects_adaptive_box_overlap(
    overlap_label: int,
) -> None:
    """Nested cells must be rejected for both same- and cross-label overlap."""

    lower_indices = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
        ],
        dtype=np.int64,
    )
    scales = np.asarray(
        [
            [2, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    labels = np.asarray([0, overlap_label, 1, 2], dtype=np.int64)

    expected = (
        "same-label adaptive boxes overlap"
        if overlap_label == 0
        else "coarse labels overlap"
    )
    with pytest.raises(ValueError, match=expected):
        RENDERER._expand_terminal_voxel_keys(
            lower_indices,
            scales,
            labels,
            np.asarray([4, 1, 1], dtype=np.int64),
        )
