"""Tests for the viz package: palette, style, and basic rendering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from latentdynamics.viz import (
    PALETTE,
    PAPER_RCPARAMS,
    apply_paper_style,
    color_for,
    plot_final_population_histogram,
    plot_morse_sets_from_csv,
    render_morse_sets_from_csv,
)


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


class TestMorseSetPlotting:
    def test_plot_morse_sets_from_csv_allows_1d_overlays(self, tmp_path):
        csv_path = tmp_path / "morse_sets"
        np.savetxt(
            csv_path,
            np.array([[-1.0, -0.25, 0], [0.25, 1.0, 1]], dtype=np.float64),
            delimiter=",",
        )

        plot = plot_morse_sets_from_csv(csv_path, paper_style=False)
        assert plot.dim == 1
        assert plot.label_to_y == {0: 0.0, 1: 1.0}

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


# silence unused-import lint
_ = np
