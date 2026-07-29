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

CHAFEE_NEGATIVE_COLOR = PALETTE[0]
CHAFEE_POSITIVE_COLOR = PALETTE[1]
CHAFEE_CONNECTING_COLOR = "#7F7F7F"
"""Canonical Chafee--Infante colors keyed by dynamical meaning."""


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


# --------------------------------------------------------------------------- #
# Latent-figure paper geometry                                                 #
# --------------------------------------------------------------------------- #
# The latent figures (Morse sets, latent trajectory, RoA) sit at 0.6*textwidth
# in the manuscript (amsart, 12pt, a4paper -> textwidth ~6.14in). Sizing each
# figure to that physical width and saving WITHOUT a tight bbox (using
# constrained_layout to keep labels inside) makes \includegraphics[width=0.6
# \textwidth] a 1:1 placement, so the point sizes below ARE the on-page sizes
# and stay consistent across every latent figure. (The coral success-rate plot
# is NOT a latent figure: it keeps the global savefig.bbox="tight" so its
# outside-right legend is not clipped.)
TEXTWIDTH_IN: float = 6.14
LATENT_FIG_WIDTH_IN: float = round(0.6 * TEXTWIDTH_IN, 3)  # 3.685
LATENT_LABEL_PT: int = 10
LATENT_TICK_PT: int = 8
LATENT_MAX_TICKS: int = 3
LATENT_ARROW_MUTATION_SCALE: int = 11


def apply_paper_style() -> None:
    """Apply the canonical paper figure rcParams to the global matplotlib state."""
    plt.rcParams.update(PAPER_RCPARAMS)


def style_latent_axes(ax: plt.Axes, *, two_d: bool) -> None:
    """Apply the shared latent-figure axis styling: sparse ticks + paper fonts.

    Caps each visible axis at :data:`LATENT_MAX_TICKS` major ticks and sets tick
    label / axis label sizes to the on-page point sizes. ``two_d`` controls
    whether the y-axis is also thinned (1-D Morse-set plots hide their y-axis).
    """
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(LATENT_MAX_TICKS))
    if two_d:
        ax.yaxis.set_major_locator(MaxNLocator(LATENT_MAX_TICKS))
    ax.tick_params(labelsize=LATENT_TICK_PT)
    ax.xaxis.label.set_size(LATENT_LABEL_PT)
    ax.yaxis.label.set_size(LATENT_LABEL_PT)


def color_for(label: int) -> str:
    """Return the palette color assigned to a Morse-set label."""
    return PALETTE[int(label) % len(PALETTE)]


def chafee_semantic_palette(
    size: int,
    *,
    negative_label: int,
    positive_label: int,
    connecting_labels: tuple[int, ...] = (),
) -> tuple[str, ...]:
    """Return a node-indexed palette with stable Chafee semantic colors.

    The two stable physical solutions retain their colors even when CMGDB node
    ids differ between latent dimensions. Any explicitly coarsened
    unstable/connecting class is gray; remaining fine Morse nodes keep the
    repository-wide cyclic palette.
    """

    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    negative = int(negative_label)
    positive = int(positive_label)
    connecting = tuple(int(label) for label in connecting_labels)
    semantic_labels = (negative, positive, *connecting)
    if any(label < 0 or label >= size for label in semantic_labels):
        raise ValueError(
            f"semantic labels {semantic_labels} must lie in [0, {size})"
        )
    if negative == positive:
        raise ValueError("negative_label and positive_label must be distinct")
    if negative in connecting or positive in connecting:
        raise ValueError("stable labels cannot also be connecting labels")

    colors = [color_for(label) for label in range(size)]
    colors[negative] = CHAFEE_NEGATIVE_COLOR
    colors[positive] = CHAFEE_POSITIVE_COLOR
    for label in connecting:
        colors[label] = CHAFEE_CONNECTING_COLOR
    return tuple(colors)


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


def save_latent_figure(
    fig: Figure,
    path: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    close: bool = False,
    **savefig_kwargs: object,
) -> list[Path]:
    """Save a latent figure at its TRUE physical size (no tight-bbox crop).

    The global ``savefig.bbox="tight"`` (needed by the coral success-rate plot's
    outside-right legend) would crop the canvas and break the 1:1 page scale; a
    bare ``bbox_inches=None`` does not help because matplotlib then falls back to
    that rcParam. We neutralise it locally so the saved PDF width equals the
    figure's :data:`LATENT_FIG_WIDTH_IN`, making ``\\includegraphics[width=0.6
    \\textwidth]`` a 1:1 placement.
    """
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        return save_figure(fig, path, formats=formats, close=close, **savefig_kwargs)
