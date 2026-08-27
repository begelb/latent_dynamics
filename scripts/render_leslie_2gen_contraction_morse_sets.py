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

from latentdynamics.viz.morse_plots import plot_morse_sets_2d_cmgdb
from latentdynamics.viz.style import save_latent_figure

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = CODE_ROOT / "replay_sources" / "leslie_2gen_contraction" / "MG" / "morse_sets"
DEFAULT_OUTPUT = CODE_ROOT / "output" / "leslie_2gen_contraction_figures"

BOUNDS_LOWER = [-0.7128865718841553, -0.28969284892082214]
BOUNDS_UPPER = [0.9668664336204529, 0.9260909557342529]

# Per-Morse-set display inflation, indexed by label. The five latent Morse sets
# differ by orders of magnitude in extent -- the period-6 orbit is a few boxes
# beside a large invariant circle -- so a faithful drawing hides the small ones.
# Each factor enlarges its set about its own centre: positions and relative
# geometry are untouched, only the drawn size. Explicit factors replace the
# former "auto" rule, which inflated only sets below a size threshold and
# capped the result, so the chosen emphasis is now exactly what is drawn.
DEFAULT_BOX_SCALE = [1.0, 20.0, 20.0, 50.0, 1.0]



def _box_scale_map(factors: list[float]) -> dict[int, float] | float:
    """Per-label scale map, or a single factor when one value is given.

    A dict reaches the renderer unclamped, so these factors apply as written --
    unlike ``"auto"``, whose cap would flatten the larger ones.
    """
    if not factors:
        return 1.0
    if len(factors) == 1:
        return float(factors[0])
    return {index: float(value) for index, value in enumerate(factors)}


def render(
    csv_path: Path,
    output_dir: Path,
    box_scale: list[float] | None = None,
) -> list[Path]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"missing saved Morse sets {csv_path}; fetch the "
            "leslie_2gen_contraction artifacts first"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    factors = DEFAULT_BOX_SCALE if box_scale is None else box_scale
    written = [
        plot_morse_sets_2d_cmgdb(
            csv_path,
            output_dir / "morse_sets.pdf",
            scale_factor=factors or None,
            xlabel="$z_1$",
            ylabel="$z_2$",
        )
    ]
    for path in written:
        print("wrote", path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--box-scale", type=float, nargs="*", default=None,
                        metavar="FACTOR",
                        help="per-Morse-set display inflation, indexed by label "
                             f"(default {DEFAULT_BOX_SCALE})")
    args = parser.parse_args()
    render(args.csv.resolve(), args.output.resolve(), box_scale=args.box_scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
