"""Render Morse-set figures across box_scale settings for side-by-side comparison.

For each (2-D) experiment this renders the saved ``morse_sets`` CSV once per
box-scale setting and assembles a contact sheet, so a human can pick the value
that makes small Morse sets legible at paper figure size without distorting
the rest. Nothing is written outside ``--out``; promoting a choice into
``paper_figures/`` stays a manual step.

Usage:
    python scripts/sweep_box_scale.py
    python scripts/sweep_box_scale.py --experiments leslie3d_example1_replay \
        --box-scales auto 1 4 --min-fracs 0.025 0.08 --max-scales 10 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from latentdynamics.replay import REPO_ROOT, load_experiment
from latentdynamics.viz import render_morse_sets_from_csv

DEFAULT_EXPERIMENTS = (
    "leslie_2gen_contraction_replay",
    "leslie3d_example1_replay",
    "leslie3d_example2_replay",
    "chafee_infante_replay",
)


def _combos(
    box_scales: list[str], min_fracs: list[float], max_scales: list[float]
) -> list[tuple[str, float | str, float, float]]:
    """Expand the requested settings into (tag, box_scale, min_frac, max_scale).

    ``min_fracs`` / ``max_scales`` only affect ``"auto"`` mode, so numeric
    scales are rendered once each instead of once per (min_frac, max_scale).
    """
    out: list[tuple[str, float | str, float, float]] = []
    for raw in box_scales:
        if raw == "auto":
            for mf in min_fracs:
                for ms in max_scales:
                    out.append((f"auto_mf-{mf:g}_ms-{ms:g}", "auto", mf, ms))
        else:
            value = float(raw)
            out.append((f"bs-{value:g}", value, 0.025, 10.0))
    return out


def _contact_sheet(panels: list[tuple[str, Path]], out_path: Path) -> None:
    ncols = min(4, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
    axes_flat = [axes] if (nrows == 1 and ncols == 1) else list(axes.flat)
    for ax in axes_flat:
        ax.axis("off")
    for ax, (tag, png) in zip(axes_flat, panels, strict=False):
        ax.imshow(plt.imread(png))
        ax.set_title(tag, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def sweep_experiment(
    name: str,
    out_root: Path,
    box_scales: list[str],
    min_fracs: list[float],
    max_scales: list[float],
) -> None:
    exp = load_experiment(name)
    if exp.arch.low_dims != 2:
        print(f"{name}: skipped (latent dim {exp.arch.low_dims}; box_scale is a 2-D knob)")
        return
    csv_path = exp.morse_dir / "morse_sets"
    lower, upper = exp.morse_bounds()
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, Path]] = []
    index_lines = [f"# box_scale sweep: {name}", ""]
    for tag, box_scale, min_frac, max_scale in _combos(box_scales, min_fracs, max_scales):
        basename = f"morse_sets__{tag}"
        paths = render_morse_sets_from_csv(
            csv_path,
            out_dir,
            bounds_lower=lower,
            bounds_upper=upper,
            basename=basename,
            box_scale=box_scale,
            box_scale_min_frac=min_frac,
            box_scale_max=max_scale,
        )
        png = next(p for p in paths if p.suffix == ".png")
        panels.append((tag, png))
        index_lines.append(
            f"- `{tag}`: box_scale={box_scale!r}, min_frac={min_frac}, "
            f"max_scale={max_scale} -> {png.name}"
        )
        print(f"{name}: rendered {tag}")

    sheet = out_dir / "comparison.png"
    _contact_sheet(panels, sheet)
    index_lines += ["", f"Contact sheet: {sheet.name}", ""]
    (out_dir / "index.md").write_text("\n".join(index_lines))
    print(f"{name}: contact sheet -> {sheet}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument(
        "--box-scales",
        nargs="+",
        default=["auto", "1", "2", "4", "8"],
        help="box_scale values; floats or 'auto'",
    )
    parser.add_argument("--min-fracs", nargs="+", type=float, default=[0.025, 0.05, 0.08])
    parser.add_argument("--max-scales", nargs="+", type=float, default=[10.0, 25.0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "box_scale_sweep")
    args = parser.parse_args()

    for name in args.experiments:
        sweep_experiment(name, args.out, args.box_scales, args.min_fracs, args.max_scales)


if __name__ == "__main__":
    main()
