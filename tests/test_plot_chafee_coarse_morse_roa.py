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
import plot_chafee_coarse_morse_roa as roa_plot  # noqa: E402


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


def test_basins_come_from_the_native_singleton_query(monkeypatch):
    # CMGDB reports the one Morse node reachable from each cell and -2 when
    # several are: only the cells with a single one belong to a basin.
    queried = {}

    def fake_query(map_graph, morse_graph, cells):
        queried["cells"] = np.asarray(cells)
        return np.array([7, -2, 9, 7], dtype=np.int32)

    monkeypatch.setattr(roa_plot.CMGDB, "MorseSingletonReachability", fake_query)
    basins = roa_plot.attractor_basins(_FakeMapGraph(), _FakeMorseGraph(), [7, 9])

    assert basins == {7: [0, 3], 9: [2]}
    np.testing.assert_array_equal(queried["cells"], np.arange(4))


def test_layer_colors_composite_the_basin_alpha_and_keep_the_sets_opaque():
    def composited(color):
        alpha = roa_plot.BASIN_ALPHA
        return tuple(alpha * channel + (1.0 - alpha) for channel in to_rgba(color)[:3])

    colors = roa_plot.LAYER_COLORS
    assert colors[roa_plot.LAYER_MERGED] == roa_plot.CHAFEE_CONNECTING_COLOR
    for layer, color in (
        (roa_plot.LAYER_BASIN_NEGATIVE, roa_plot.CHAFEE_NEGATIVE_COLOR),
        (roa_plot.LAYER_BASIN_POSITIVE, roa_plot.CHAFEE_POSITIVE_COLOR),
    ):
        np.testing.assert_allclose(
            to_rgba(colors[layer])[:3], composited(color), atol=1 / 255
        )
    for layer, color in (
        (roa_plot.LAYER_UNIFORM_NEGATIVE, roa_plot.CHAFEE_NEGATIVE_COLOR),
        (roa_plot.LAYER_COARSE_NEGATIVE, roa_plot.CHAFEE_NEGATIVE_COLOR),
        (roa_plot.LAYER_UNIFORM_POSITIVE, roa_plot.CHAFEE_POSITIVE_COLOR),
        (roa_plot.LAYER_COARSE_POSITIVE, roa_plot.CHAFEE_POSITIVE_COLOR),
    ):
        assert colors[layer] == color


def test_basin_rows_carry_cmgdb_boxes_labelled_by_layer():
    graph, _bounds, basins = _fixture_data()

    rows = roa_plot._basin_rows(graph, basins, {"negative": 9, "positive": 7})

    by_layer: dict[int, set[tuple[float, ...]]] = {}
    for *box, layer in rows:
        by_layer.setdefault(int(layer), set()).add(tuple(box))
    # Attractor 9 is the negative state, so its basin cells 1 and 3, and its own
    # cell 1, land on the negative layers -- each box the one CMGDB reports.
    assert by_layer[roa_plot.LAYER_BASIN_NEGATIVE] == {
        graph.phase_space_box(1),
        graph.phase_space_box(3),
    }
    assert by_layer[roa_plot.LAYER_UNIFORM_NEGATIVE] == {graph.phase_space_box(1)}
    assert by_layer[roa_plot.LAYER_BASIN_POSITIVE] == {
        graph.phase_space_box(0),
        graph.phase_space_box(2),
    }
    assert by_layer[roa_plot.LAYER_UNIFORM_POSITIVE] == {graph.phase_space_box(0)}


def test_paper_axes_show_labeled_ticks_by_default_and_can_hide_them():
    graph, bounds, basins = _fixture_data()
    rows = roa_plot._basin_rows(graph, basins, {"negative": 9, "positive": 7})

    fig = roa_plot._plot_layers(rows, bounds)
    ax = fig.axes[0]
    fig.canvas.draw()
    assert ax.get_xlabel() == "$z_1$"
    assert ax.get_ylabel() == "$z_2$"
    assert ax.get_xticks().size > 0
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    assert not any(line.get_visible() for line in ax.get_ygridlines())
    plt.close(fig)

    fig = roa_plot._plot_layers(rows, bounds, show_ticks=False, show_axis_labels=False)
    ax = fig.axes[0]
    fig.canvas.draw()
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_xticks().size == 0
    assert ax.get_yticks().size == 0
    plt.close(fig)


def test_main_writes_basin_and_overlay_without_running_cmgdb(tmp_path, monkeypatch):
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
            "patches": len(ax.patches),
            "collections": len(ax.collections),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "xticks": ax.get_xticks().copy(),
            "yticks": ax.get_yticks().copy(),
            "xgrid": [line.get_visible() for line in ax.get_xgridlines()],
            "ygrid": [line.get_visible() for line in ax.get_ygridlines()],
            "face_colors": [patch.get_facecolor() for patch in ax.patches],
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
    # CMGDB merges each layer into one outline: the two basins and the two
    # attracting sets, plus the three coarse sets in the overlay. No raster.
    assert basin_render["images"] == overlay_render["images"] == 0
    assert basin_render["patches"] == 4 and basin_render["collections"] == 0
    assert overlay_render["patches"] == 7 and overlay_render["collections"] == 0
    for render in (basin_render, overlay_render):
        assert render["xlabel"] == "$z_1$"
        assert render["ylabel"] == "$z_2$"
        assert np.asarray(render["xticks"]).size > 0
        assert np.asarray(render["yticks"]).size > 0
        assert not any(render["xgrid"])
        assert not any(render["ygrid"])

    expected = [to_rgba(color) for color in roa_plot.LAYER_COLORS]
    # Painted in PAINT_ORDER; the basin figure has no coarse layers to paint.
    for render, layers in (
        (basin_render, roa_plot.PAINT_ORDER[1:5]),
        (overlay_render, roa_plot.PAINT_ORDER),
    ):
        for drawn, layer in zip(render["face_colors"], layers, strict=True):
            np.testing.assert_allclose(
                np.asarray(drawn).reshape(-1)[:4], expected[layer]
            )

    metadata = json.loads(overlay_output.with_suffix(".json").read_text())
    assert metadata["uniform_attractors_left_to_right"] == [7, 9]
    assert metadata["uniform_attractors_by_physical_state"] == {
        "negative": 9,
        "positive": 7,
    }
    assert metadata["basin_method"] == (
        "CMGDB.MorseSingletonReachability on the cached cell graph"
    )
    assert metadata["rendering"] == {
        "basin_layer": "CMGDB.PlotMorseSets, one merged outline per layer",
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
