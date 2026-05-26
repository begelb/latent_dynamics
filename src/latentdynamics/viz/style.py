"""Single source of truth for paper colors and matplotlib rcParams."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    close: bool = False,
    **savefig_kwargs: object,
) -> list[Path]:
    """Save ``fig`` once per format, reusing ``path``'s stem.

    Paper figures are vector-first: a ``.pdf`` is always written alongside the
    raster ``.png``. ``path`` may carry any suffix (or none); the stem is reused
    for each format. Returns the written paths in ``formats`` order.
    """
    path = Path(path)
    written: list[Path] = []
    for fmt in formats:
        out_path = path.with_suffix(f".{fmt}")
        fig.savefig(out_path, **savefig_kwargs)
        written.append(out_path)
    if close:
        plt.close(fig)
    return written
