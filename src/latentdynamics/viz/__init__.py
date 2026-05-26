"""Plotting helpers; the single source of truth for paper figure styling.

Imports the non-interactive ``Agg`` matplotlib backend on first use so that
nothing in the pipeline pops up a GUI window. Override by setting the
``MPLBACKEND`` environment variable before any ``latentdynamics`` import.
"""

import os as _os

import matplotlib as _mpl

if "MPLBACKEND" not in _os.environ:
    _mpl.use("Agg", force=True)

from .data_trajectories import plot_data_trajectories, save_default_trajectory_plot
from .histograms import emit_population_histogram, plot_final_population_histogram
from .morse_plots import (
    MorseSetsPlot,
    RenderedMorseFigures,
    plot_morse_sets_from_csv,
    render_morse_from_files,
    render_morse_graph,
    render_morse_graph_from_dot,
    render_morse_outputs,
    render_morse_sets,
    render_morse_sets_from_csv,
    save_morse_graph_artifacts,
)
from .style import PALETTE, PAPER_RCPARAMS, apply_paper_style, color_for, save_figure
from .trajectory_plots import plot_latent_trajectory

__all__ = [
    "PALETTE",
    "PAPER_RCPARAMS",
    "MorseSetsPlot",
    "RenderedMorseFigures",
    "apply_paper_style",
    "color_for",
    "save_figure",
    "emit_population_histogram",
    "plot_data_trajectories",
    "plot_final_population_histogram",
    "plot_latent_trajectory",
    "plot_morse_sets_from_csv",
    "render_morse_from_files",
    "render_morse_graph",
    "render_morse_graph_from_dot",
    "render_morse_outputs",
    "render_morse_sets",
    "render_morse_sets_from_csv",
    "save_default_trajectory_plot",
    "save_morse_graph_artifacts",
]
