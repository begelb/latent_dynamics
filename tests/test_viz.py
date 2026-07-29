"""Tests for the viz package: palette, style, and basic rendering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from latentdynamics.analysis.cell_graph import CellGraphROA, UniformGrid
from latentdynamics.analysis.regions_of_attraction import BoxROATable, MorseGraph
from latentdynamics.viz import (
    CHAFEE_CONNECTING_COLOR,
    CHAFEE_NEGATIVE_COLOR,
    CHAFEE_POSITIVE_COLOR,
    PALETTE,
    PAPER_RCPARAMS,
    apply_paper_style,
    chafee_semantic_palette,
    color_for,
    plot_final_population_histogram,
    plot_morse_set_projections_from_csv,
    plot_morse_sets_3d_cubical_from_csv,
    plot_morse_sets_from_csv,
    render_morse_set_projections_from_csv,
    render_morse_sets_from_csv,
)
from latentdynamics.viz.morse_plots import _exposed_cubical_faces, _resolve_box_scales
from latentdynamics.viz.regions_of_attraction import plot_roa_overlay_cell_graph


class TestPalette:
    def test_seven_distinct_hex_colors(self):
        assert len(PALETTE) == 7
        assert len(set(PALETTE)) == 7
        for c in PALETTE:
            assert c.startswith("#") and len(c) == 7

    def test_color_for_wraps_modulo(self):
        n = len(PALETTE)
        assert color_for(0) == PALETTE[0]
        assert color_for(n) == PALETTE[0]
        assert color_for(n + 3) == PALETTE[3]

    def test_chafee_palette_is_semantic_not_node_order(self):
        assert chafee_semantic_palette(
            7,
            negative_label=1,
            positive_label=0,
        ) == (
            CHAFEE_POSITIVE_COLOR,
            CHAFEE_NEGATIVE_COLOR,
            *PALETTE[2:],
        )
        assert chafee_semantic_palette(
            3,
            negative_label=0,
            positive_label=1,
            connecting_labels=(2,),
        ) == (
            CHAFEE_NEGATIVE_COLOR,
            CHAFEE_POSITIVE_COLOR,
            CHAFEE_CONNECTING_COLOR,
        )

    def test_chafee_palette_rejects_overlapping_semantics(self):
        with pytest.raises(ValueError, match="distinct"):
            chafee_semantic_palette(3, negative_label=0, positive_label=0)
        with pytest.raises(ValueError, match="cannot also be connecting"):
            chafee_semantic_palette(
                3,
                negative_label=0,
                positive_label=1,
                connecting_labels=(1,),
            )


class TestStyle:
    def test_apply_paper_style_sets_serif(self):
        apply_paper_style()
        assert plt.rcParams["font.family"] == ["serif"]
        assert plt.rcParams["mathtext.fontset"] == "stix"

    def test_paper_rcparams_has_required_keys(self):
        for key in ("font.family", "font.serif", "mathtext.fontset", "savefig.dpi"):
            assert key in PAPER_RCPARAMS


class TestPopulationHistogram:
    def test_renders_histogram_to_disk(self, tmp_path):
        # Synthetic CSV: 3 trajectories of length 4 in 2-D coral-like format.
        rows = []
        for traj in range(3):
            for step in range(4):
                rows.append([traj, step, traj + step, 2 * (traj + step)])
        header = "x0,x1,y0,y1"
        csv_path = tmp_path / "synthetic.csv"
        csv_path.write_text(header + "\n" + "\n".join(",".join(map(str, row)) for row in rows))
        out_path = tmp_path / "hist.pdf"
        result = plot_final_population_histogram(
            csv_path,
            out_path,
            steps_per_trajectory=4,
            ymax=None,
            style=False,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_missing_csv_raises(self, tmp_path):
        try:
            plot_final_population_histogram(
                tmp_path / "missing.csv",
                tmp_path / "out.pdf",
                steps_per_trajectory=4,
                style=False,
            )
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


class TestBoxScaleResolution:
    """One big set (label 0, extent 4.0) + one tiny set (label 1, extent 0.01)
    in a view of span ~5; the auto floor decides which sets get inflated."""

    def _arrays(self):
        lx = np.array([0.0, 5.0])
        ly = np.array([0.0, 5.0])
        ux = np.array([4.0, 5.01])
        uy = np.array([4.0, 5.01])
        lbls = np.array([0, 1])
        return lx, ly, ux, uy, lbls

    def test_auto_inflates_only_sets_below_floor(self):
        scale_for = _resolve_box_scales("auto", *self._arrays())
        assert scale_for(0) == 1.0
        assert scale_for(1) > 1.0

    def test_auto_max_scale_caps_inflation(self):
        # default floor 0.025 * 5.01 / 0.01 ≈ 12.5 wants more than the cap
        capped = _resolve_box_scales("auto", *self._arrays(), max_scale=10.0)
        uncapped = _resolve_box_scales("auto", *self._arrays(), max_scale=25.0)
        assert capped(1) == 10.0
        assert uncapped(1) > 10.0

    def test_auto_min_frac_raises_floor(self):
        # with a floor of 90% of the span, even the big set falls below it
        scale_for = _resolve_box_scales("auto", *self._arrays(), min_frac=0.9)
        assert scale_for(0) > 1.0

    def test_dict_mode_ignores_auto_knobs(self):
        scale_for = _resolve_box_scales(
            {1: 3.0}, *self._arrays(), min_frac=0.9, max_scale=2.0
        )
        assert scale_for(1) == 3.0
        assert scale_for(0) == 1.0

    def test_float_mode_is_global(self):
        scale_for = _resolve_box_scales(2.5, *self._arrays())
        assert scale_for(0) == scale_for(1) == 2.5


class TestMorseSetPlotting:
    @staticmethod
    def _write_3d_morse_sets(path):
        data = np.array(
            [
                [10.0, 20.0, 30.0, 11.0, 22.0, 33.0, 1],
                [11.0, 21.0, 31.0, 12.0, 23.0, 34.0, 0],
            ],
            dtype=np.float64,
        )
        np.savetxt(path, data, delimiter=",")
        return data

    def test_plot_morse_set_projections_uses_all_3d_coordinate_pairs(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        data = self._write_3d_morse_sets(csv_path)
        lower = np.array([9.0, 18.0, 27.0])
        upper = np.array([13.0, 25.0, 37.0])

        plots = plot_morse_set_projections_from_csv(
            csv_path,
            bounds_lower=lower,
            bounds_upper=upper,
            paper_style=False,
        )

        assert list(plots) == [(0, 1), (0, 2), (1, 2)]
        for pair, plot in plots.items():
            i, j = pair
            expected = data[:, [i, j, 3 + i, 3 + j, 6]]
            np.testing.assert_allclose(plot.data, expected)
            np.testing.assert_allclose(plot.ax.get_xlim(), (lower[i], upper[i]))
            np.testing.assert_allclose(plot.ax.get_ylim(), (lower[j], upper[j]))
            assert plot.ax.get_xlabel() == f"$z_{{{i + 1}}}$"
            assert plot.ax.get_ylabel() == f"$z_{{{j + 1}}}$"
            colors = plot.ax.collections[0].get_facecolors()
            np.testing.assert_allclose(colors[0], to_rgba(PALETTE[1]))
            np.testing.assert_allclose(colors[1], to_rgba(PALETTE[0]))
            plt.close(plot.fig)

    def test_2d_label_draw_order_groups_boxes_back_to_front(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        np.savetxt(
            csv_path,
            np.array(
                [
                    [0.0, 0.0, 1.0, 1.0, 0],
                    [0.0, 0.0, 1.0, 1.0, 2],
                    [0.0, 0.0, 1.0, 1.0, 1],
                    [1.0, 0.0, 2.0, 1.0, 2],
                ],
                dtype=np.float64,
            ),
            delimiter=",",
        )

        plot = plot_morse_sets_from_csv(
            csv_path,
            paper_style=False,
            label_draw_order=[2, 0, 1],
        )

        colors = plot.ax.collections[0].get_facecolors()
        expected = [PALETTE[2], PALETTE[2], PALETTE[0], PALETTE[1]]
        np.testing.assert_allclose(colors, [to_rgba(color) for color in expected])
        plt.close(plot.fig)

    def test_projection_label_draw_order_applies_to_every_panel(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        self._write_3d_morse_sets(csv_path)

        plots = plot_morse_set_projections_from_csv(
            csv_path,
            paper_style=False,
            label_draw_order=[0, 1],
        )

        expected = [to_rgba(PALETTE[0]), to_rgba(PALETTE[1])]
        for plot in plots.values():
            np.testing.assert_allclose(
                plot.ax.collections[0].get_facecolors(),
                expected,
            )
            plt.close(plot.fig)

    def test_cubical_3d_plot_removes_same_label_internal_faces(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        data = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2],
                [1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 2],
            ],
            dtype=np.float64,
        )
        np.savetxt(csv_path, data, delimiter=",")

        faces, labels = _exposed_cubical_faces(data)
        assert faces.shape == (10, 4, 3)
        np.testing.assert_array_equal(labels, np.full(10, 2))

        plot = plot_morse_sets_3d_cubical_from_csv(
            csv_path,
            paper_style=False,
            show_legend=False,
        )
        assert plot.dim == 3
        assert plot.ax.get_xlabel() == "$z_1$"
        assert plot.ax.get_ylabel() == "$z_2$"
        assert plot.ax.get_zlabel() == "$z_3$"
        plt.close(plot.fig)

    def test_cubical_3d_plot_accepts_custom_legend_labels(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        np.savetxt(
            csv_path,
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0],
                    [2.0, 0.0, 0.0, 3.0, 1.0, 1.0, 1],
                    [4.0, 0.0, 0.0, 5.0, 1.0, 1.0, 2],
                ],
                dtype=np.float64,
            ),
            delimiter=",",
        )
        legend_labels = {
            0: "$M(0^+)$",
            1: "$M(0^-)$",
            2: "$M(1)$",
        }

        plot = plot_morse_sets_3d_cubical_from_csv(
            csv_path,
            paper_style=False,
            legend_labels=legend_labels,
        )

        legend = plot.ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == list(
            legend_labels.values()
        )
        plt.close(plot.fig)

    def test_cubical_3d_plot_can_hide_ticks_and_coordinate_labels(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        data = np.array(
            [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0]],
            dtype=np.float64,
        )
        np.savetxt(csv_path, data, delimiter=",")

        plot = plot_morse_sets_3d_cubical_from_csv(
            csv_path,
            paper_style=False,
            show_ticks=False,
            show_axis_labels=False,
            show_legend=False,
        )

        assert plot.ax.get_xlabel() == ""
        assert plot.ax.get_ylabel() == ""
        assert plot.ax.get_zlabel() == ""
        assert plot.ax.get_xticks().size == 0
        assert plot.ax.get_yticks().size == 0
        assert plot.ax.get_zticks().size == 0
        assert plot.ax.get_legend() is None
        plt.close(plot.fig)

    def test_plot_morse_set_projections_supports_ordered_nd_pairs(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        # 4-D: lower[0:4], upper[4:8], label[8].
        data = np.array([[1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0, 2]])
        np.savetxt(csv_path, data, delimiter=",")

        plots = plot_morse_set_projections_from_csv(
            csv_path,
            pairs=[(3, 1)],
            paper_style=False,
        )

        plot = plots[(3, 1)]
        np.testing.assert_allclose(plot.data, [[4.0, 2.0, 14.0, 12.0, 2.0]])
        assert plot.ax.get_xlabel() == "$z_{4}$"
        assert plot.ax.get_ylabel() == "$z_{2}$"
        plt.close(plot.fig)

    def test_render_morse_set_projections_names_each_requested_pair(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        self._write_3d_morse_sets(csv_path)

        rendered = render_morse_set_projections_from_csv(
            csv_path,
            tmp_path / "rendered",
            pairs=[(0, 2), (1, 2)],
            basename="three_d_sets",
            formats=("png",),
            paper_style=False,
        )

        assert rendered == {
            (0, 2): [tmp_path / "rendered" / "three_d_sets_z1_z3.png"],
            (1, 2): [tmp_path / "rendered" / "three_d_sets_z2_z3.png"],
        }
        assert all(path.exists() and path.stat().st_size > 0 for paths in rendered.values() for path in paths)

    def test_projected_visibility_floor_uses_pair_specific_view_span(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        np.savetxt(
            csv_path,
            np.array([[0.0, 10.0, 20.0, 0.001, 10.002, 20.003, 0]]),
            delimiter=",",
        )

        plot = plot_morse_set_projections_from_csv(
            csv_path,
            pairs=[(2, 0)],
            bounds_lower=[-1.0, 9.0, 18.0],
            bounds_upper=[1.0, 11.0, 22.0],
            paper_style=False,
            min_box_side_frac=0.1,
        )[(2, 0)]
        extent = plot.ax.collections[0].get_paths()[0].get_extents()
        x_span = plot.ax.get_xlim()[1] - plot.ax.get_xlim()[0]
        y_span = plot.ax.get_ylim()[1] - plot.ax.get_ylim()[0]

        assert extent.width >= 0.1 * x_span
        assert extent.height >= 0.1 * y_span
        plt.close(plot.fig)

    def test_plot_morse_sets_from_csv_allows_1d_overlays(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        np.savetxt(
            csv_path,
            np.array([[-1.0, -0.25, 0], [0.25, 1.0, 1]], dtype=np.float64),
            delimiter=",",
        )

        plot = plot_morse_sets_from_csv(csv_path, paper_style=False)
        assert plot.dim == 1
        assert plot.label_to_y == {0: 0.0, 1: 0.0}

        plot.ax.scatter([0.0], [plot.label_to_y[0]], color="black", zorder=10)
        out_path = tmp_path / "overlay.png"
        plot.fig.savefig(out_path)
        plt.close(plot.fig)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_morse_sets_from_csv_accepts_existing_2d_axis(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        np.savetxt(
            csv_path,
            np.array([[0.0, 0.0, 1.0, 1.0, 0], [1.0, 0.0, 2.0, 1.0, 1]]),
            delimiter=",",
        )

        fig, ax = plt.subplots()
        plot = plot_morse_sets_from_csv(csv_path, ax=ax, paper_style=False)
        assert plot.dim == 2
        assert plot.fig is fig
        assert plot.ax is ax

        plot.ax.plot([0.0, 2.0], [0.5, 0.5], color="black")
        out_path = tmp_path / "overlay_2d.png"
        plot.fig.savefig(out_path)
        plt.close(plot.fig)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_2d_morse_set_plot_uses_adaptive_limits_clipped_to_cmgdb_bounds(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        np.savetxt(
            csv_path,
            np.array(
                [
                    [4.0, 5.0, 4.1, 5.1, 0],
                    [6.0, 7.0, 6.1, 7.1, 1],
                ],
                dtype=np.float64,
            ),
            delimiter=",",
        )

        plot = plot_morse_sets_from_csv(
            csv_path,
            bounds_lower=[0.0, 0.0],
            bounds_upper=[10.0, 20.0],
            paper_style=False,
        )

        np.testing.assert_allclose(plot.ax.get_xlim(), (3.8, 6.3))
        np.testing.assert_allclose(plot.ax.get_ylim(), (4.8, 7.3))
        plt.close(plot.fig)

    def test_render_morse_sets_from_csv_keeps_file_output_wrapper(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        np.savetxt(csv_path, np.array([[0.0, 1.0, 0]], dtype=np.float64), delimiter=",")

        rendered = render_morse_sets_from_csv(
            csv_path,
            tmp_path,
            basename="base",
            formats=("png",),
            paper_style=False,
        )

        assert rendered == [tmp_path / "base.png"]
        assert rendered[0].exists()
        assert rendered[0].stat().st_size > 0

    def test_render_morse_sets_threads_box_scale_knobs(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        np.savetxt(
            csv_path,
            np.array([[0.0, 0.0, 4.0, 4.0, 0], [5.0, 5.0, 5.01, 5.01, 1]], dtype=np.float64),
            delimiter=",",
        )
        rendered = render_morse_sets_from_csv(
            csv_path,
            tmp_path,
            basename="scaled",
            formats=("png",),
            paper_style=False,
            box_scale="auto",
            box_scale_min_frac=0.5,
            box_scale_max=25.0,
        )
        assert rendered[0].exists() and rendered[0].stat().st_size > 0

    def test_min_box_side_frac_applies_display_floor(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        np.savetxt(
            csv_path,
            np.array([[0.0, 0.0, 0.001, 0.002, 0], [1.0, 2.0, 1.001, 2.002, 1]]),
            delimiter=",",
        )

        plot = plot_morse_sets_from_csv(
            csv_path,
            bounds_lower=[0.0, 0.0],
            bounds_upper=[2.0, 3.0],
            paper_style=False,
            min_box_side_frac=0.01,
        )
        x_span = plot.ax.get_xlim()[1] - plot.ax.get_xlim()[0]
        y_span = plot.ax.get_ylim()[1] - plot.ax.get_ylim()[0]
        extent = plot.ax.collections[0].get_paths()[0].get_extents()

        assert extent.width >= 0.01 * x_span
        assert extent.height >= 0.01 * y_span
        plt.close(plot.fig)

    def test_min_box_side_frac_rejects_negative_values(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        np.savetxt(
            csv_path,
            np.array([[0.0, 0.0, 1.0, 1.0, 0]], dtype=np.float64),
            delimiter=",",
        )

        try:
            plot_morse_sets_from_csv(
                csv_path,
                paper_style=False,
                min_box_side_frac=-0.01,
            )
        except ValueError as exc:
            assert "nonnegative" in str(exc)
            return
        raise AssertionError("expected ValueError")

    def test_roa_overlay_colors_recurrent_morse_sets_by_morse_node_not_lower_basin(self):
        mg = MorseGraph(
            nodes=[0, 1],
            edges={1: [0]},
            colors={0: "#111111", 1: "#eeeeee"},
            labels={0: "0", 1: "1"},
        )
        grid = UniformGrid(
            bounds_lo=np.array([0.0, 0.0]),
            bounds_hi=np.array([1.0, 1.0]),
            resolution=1,
        )
        cg = CellGraphROA(
            grid=grid,
            morse_graph=mg,
            box_roa=np.array([0], dtype=np.int32),
            minimal_grid_boxes={0: np.array([0], dtype=np.int64)},
        )
        table = BoxROATable(
            boxes=pd.DataFrame(
                [
                    {
                        "lower_0": 0.8,
                        "lower_1": 0.8,
                        "upper_0": 0.9,
                        "upper_1": 0.9,
                        "morse_node": 1,
                        "roa_label": 1,
                    }
                ]
            ),
            morse_graph=mg,
            dim=2,
        )

        fig = plot_roa_overlay_cell_graph(cg, table)
        fig.canvas.draw()
        overlay = fig.axes[0].collections[-1]

        np.testing.assert_allclose(overlay.get_facecolors()[0], to_rgba("#eeeeee", 0.95))
        plt.close(fig)


# silence unused-import lint
_ = np
