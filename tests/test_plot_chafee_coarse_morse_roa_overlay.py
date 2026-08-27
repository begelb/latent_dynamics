"""Focused rendering tests for the Chafee--Infante attraction basins."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import plot_chafee_coarse_morse_roa_overlay as roa_plot  # noqa: E402


class _FakeMorseGraph:
    """Two attractors and four uniform cells on a 2-by-2 grid."""

    _boxes: ClassVar[dict[int, tuple[float, float, float, float]]] = {
        0: (0.0, 0.0, 1.0, 0.5),
        1: (1.0, 0.0, 2.0, 0.5),
        2: (0.0, 0.5, 1.0, 1.0),
        3: (1.0, 0.5, 2.0, 1.0),
    }
    _morse_sets: ClassVar[dict[int, tuple[int, ...]]] = {7: (0,), 9: (1,)}

    def phase_space_box(self, cell: int):
        return self._boxes[cell]

    def morse_set(self, node: int):
        return self._morse_sets[node]

    def num_vertices(self) -> int:
        return 2


class _FakeMapGraph:
    def num_vertices(self) -> int:
        return 4


def _fixture_data():
    graph = _FakeMorseGraph()
    bounds = SimpleNamespace(
        lower=np.array([0.0, 0.0]),
        upper=np.array([2.0, 1.0]),
    )
    basins = {7: [0, 2], 9: [1, 3]}
    return graph, bounds, basins


def test_basin_image_uses_physical_colors_and_opaque_attractors():
    graph, bounds, basins = _fixture_data()

    image = roa_plot._basin_image(
        graph,
        basins,
        [7, 9],
        bounds,
        resolution=2,
    )

    negative = np.asarray(to_rgba(roa_plot.CHAFEE_NEGATIVE_COLOR))
    positive = np.asarray(to_rgba(roa_plot.CHAFEE_POSITIVE_COLOR))
    np.testing.assert_allclose(image[0, 0], negative)
    np.testing.assert_allclose(image[1, 0], (*negative[:3], roa_plot.BASIN_ALPHA))
    np.testing.assert_allclose(image[0, 1], positive)
    np.testing.assert_allclose(image[1, 1], (*positive[:3], roa_plot.BASIN_ALPHA))


def test_paper_axes_show_labeled_ticks_by_default_and_can_hide_them():
    fig, ax = plt.subplots()
    ax.grid(True)

    roa_plot._style_axes(ax)
    fig.canvas.draw()

    assert ax.get_xlabel() == "$z_1$"
    assert ax.get_ylabel() == "$z_2$"
    assert ax.get_xticks().size > 0
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    assert not any(line.get_visible() for line in ax.get_ygridlines())
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.grid(True)

    roa_plot._style_axes(ax, show_ticks=False, show_axis_labels=False)
    fig.canvas.draw()

    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_xticks().size == 0
    assert ax.get_yticks().size == 0
    plt.close(fig)


def test_main_writes_basin_and_overlay_from_rgba_without_running_cmgdb(
    tmp_path,
    monkeypatch,
):
    graph, bounds, basins = _fixture_data()
    coarse_sets = np.array(
        [
            [1.0, 0.0, 1.1, 0.1, 0],
            [0.0, 0.0, 0.1, 0.1, 1],
            [0.8, 0.4, 1.2, 0.6, 2],
        ],
        dtype=np.float64,
    )
    coarse_path = tmp_path / "morse_sets"
    np.savetxt(coarse_path, coarse_sets, delimiter=",")
    basin_output = tmp_path / "attractor_basins"
    overlay_output = tmp_path / "morse_roa_overlay"

    monkeypatch.setattr(
        roa_plot,
        "_compute_uniform_basins",
        lambda _device: (
            graph,
            _FakeMapGraph(),
            bounds,
            basins,
            [9, 7],
            {"negative": 9, "positive": 7},
            2,
        ),
    )
    captured: dict[str, dict[str, object]] = {}

    def fake_save_figure(fig, output, **_kwargs):
        ax = fig.axes[0]
        fig.canvas.draw()
        captured[Path(output).name] = {
            "images": len(ax.images),
            "collections": len(ax.collections),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "xticks": ax.get_xticks().copy(),
            "yticks": ax.get_yticks().copy(),
            "xgrid": [line.get_visible() for line in ax.get_xgridlines()],
            "ygrid": [line.get_visible() for line in ax.get_ygridlines()],
            "collection_colors": [
                collection.get_facecolors()[0].copy()
                for collection in ax.collections
            ],
        }
        plt.close(fig)
        output = Path(output)
        return [output.with_suffix(".pdf"), output.with_suffix(".png")]

    monkeypatch.setattr(roa_plot, "save_figure", fake_save_figure)

    result = roa_plot.main(
        [
            "--coarse-sets",
            str(coarse_path),
            "--basin-output",
            str(basin_output),
            "--output",
            str(overlay_output),
        ]
    )

    assert result == 0
    basin_render = captured["attractor_basins"]
    overlay_render = captured["morse_roa_overlay"]
    # The basin layer is vector rectangles now: no raster image, and one
    # PatchCollection per distinct basin colour (each physical state at the
    # attractor alpha and at BASIN_ALPHA).
    assert basin_render["images"] == overlay_render["images"] == 0
    assert basin_render["collections"] == 4
    assert overlay_render["collections"] == 7
    for render in (basin_render, overlay_render):
        assert render["xlabel"] == "$z_1$"
        assert render["ylabel"] == "$z_2$"
        assert np.asarray(render["xticks"]).size > 0
        assert np.asarray(render["yticks"]).size > 0
        assert not any(render["xgrid"])
        assert not any(render["ygrid"])

    def composited(color, alpha):
        # Mirror _draw_basin_vector: 8-bit quantisation, then the alpha is
        # composited against white into an opaque colour.
        rgba = np.round(np.asarray(to_rgba((*to_rgba(color)[:3], alpha))) * 255.0) / 255.0
        return tuple(np.round(rgba[3] * rgba[:3] + (1.0 - rgba[3]), 6)) + (1.0,)

    expected_basin = {
        composited(color, alpha)
        for color in (roa_plot.CHAFEE_NEGATIVE_COLOR, roa_plot.CHAFEE_POSITIVE_COLOR)
        for alpha in (1.0, roa_plot.BASIN_ALPHA)
    }
    for render in (basin_render, overlay_render):
        actual = {
            tuple(np.round(np.asarray(c, dtype=float), 6))
            for c in render["collection_colors"][:4]
        }
        assert actual == expected_basin

    connecting, positive, negative = overlay_render["collection_colors"][-3:]
    np.testing.assert_allclose(connecting, to_rgba(roa_plot.CHAFEE_CONNECTING_COLOR))
    np.testing.assert_allclose(positive, to_rgba(roa_plot.CHAFEE_POSITIVE_COLOR))
    np.testing.assert_allclose(negative, to_rgba(roa_plot.CHAFEE_NEGATIVE_COLOR))

    metadata = json.loads(overlay_output.with_suffix(".json").read_text())
    assert metadata["uniform_attractors_left_to_right"] == [7, 9]
    assert metadata["uniform_attractors_by_physical_state"] == {
        "negative": 9,
        "positive": 7,
    }
    assert metadata["rendering"] == {
        "basin_layer": "per-cell vector rectangles, edge in face colour",
        "per_cell_scatter": False,
        "basin_alpha": 0.35,
        "attractor_set_alpha": 1.0,
    }
    assert metadata["axis_visibility"] == {
        "ticks": True,
        "latent_coordinate_labels": True,
        "grid": False,
    }
    assert metadata["physical_colors"] == {
        "negative": roa_plot.CHAFEE_NEGATIVE_COLOR,
        "positive": roa_plot.CHAFEE_POSITIVE_COLOR,
        "coarse_unstable_connecting": roa_plot.CHAFEE_CONNECTING_COLOR,
    }
    assert metadata["basin_only_outputs"] == [
        str(basin_output.with_suffix(".pdf")),
        str(basin_output.with_suffix(".png")),
    ]
    assert metadata["overlay_outputs"] == [
        str(overlay_output.with_suffix(".pdf")),
        str(overlay_output.with_suffix(".png")),
    ]
