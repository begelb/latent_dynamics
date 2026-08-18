"""Re-render the extended-Leslie latent Morse sets without numeric ticks.

Generator of the paper's tick-free latent Morse-set panel for the extended
(10-D embedded, 2-D latent) Leslie contraction example.  Display-only: the
boxes come from the saved CMGDB ``morse_sets`` CSV shipped in
``replay_sources/leslie_2gen_contraction``; no Morse graph or dynamics is
recomputed.  The latent bounds below are the encoded-data bounds recorded in
that run's ``mg_params_log.txt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from latentdynamics.viz.morse_plots import plot_morse_sets_from_csv
from latentdynamics.viz.style import save_latent_figure

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = CODE_ROOT / "replay_sources" / "leslie_2gen_contraction" / "MG" / "morse_sets"
DEFAULT_OUTPUT = CODE_ROOT / "output" / "leslie_2gen_contraction_figures"

BOUNDS_LOWER = [-0.7128865718841553, -0.28969284892082214]
BOUNDS_UPPER = [0.9668664336204529, 0.9260909557342529]


def render(csv_path: Path, output_dir: Path) -> list[Path]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"missing saved Morse sets {csv_path}; fetch the "
            "leslie_2gen_contraction artifacts first"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot = plot_morse_sets_from_csv(
        csv_path,
        bounds_lower=BOUNDS_LOWER,
        bounds_upper=BOUNDS_UPPER,
        box_scale="auto",
        min_box_side_frac=0.0025,
    )
    plot.ax.set_xticks([])
    plot.ax.set_yticks([])
    plot.ax.tick_params(axis="both", which="both", length=0)
    written = save_latent_figure(
        plot.fig,
        output_dir / "morse_sets",
        formats=("pdf", "png"),
        close=True,
    )
    for path in written:
        print("wrote", path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.csv.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
