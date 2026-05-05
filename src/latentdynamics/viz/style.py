"""Single source of truth for paper colors and matplotlib rcParams."""

from __future__ import annotations

import matplotlib.pyplot as plt

PALETTE: list[str] = [
    "#FFB000",
    "#DC267F",
    "#648FFF",
    "#FE6100",
    "#785EF0",
    "#008080",
    "#FCC2E8",
]
"""Color-blind-safe seven-step palette used throughout the paper figures."""


PAPER_RCPARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_paper_style() -> None:
    """Apply the canonical paper figure rcParams to the global matplotlib state."""
    plt.rcParams.update(PAPER_RCPARAMS)


def color_for(label: int) -> str:
    """Return the palette color assigned to a Morse-set label."""
    return PALETTE[int(label) % len(PALETTE)]
