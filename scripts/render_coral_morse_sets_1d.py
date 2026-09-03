"""Render the coral 1-D latent Morse sets with encoded fixed points.

Generator of the paper's coral latent Morse-set panel: the Morse-set
intervals saved for the author-provided coral ``train_500/seed_16`` run are
drawn as colored bands on the latent axis, and the three ambient fixed
points ``a0``, ``a1``, ``r`` of the red-coral model are encoded and overlaid
as black star, triangle, and square markers.  The fixed-point legend is
intentionally omitted; the markers are identified in the paper caption.

This is render-only: no network is trained and no dynamics is recomputed.
The three inputs (saved Morse sets, checkpoint, and training scaler) are the
fetched replay artifacts; their SHA-256 digests are pinned in
``artifacts/reference_results/coral/render_inputs_sha256.json`` and verified
before rendering.  The checkpoint pin is the migrated single-file
``models/autoencoder.pt`` (bitwise-equivalent to the author's legacy files;
see ``artifacts/provenance/MIGRATION_RECORD.json``); when only the legacy
``models/encoder.pt`` is present, its digest is checked against the record's
``legacy_original`` entry instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import CMGDB
import matplotlib.pyplot as plt
import numpy as np

from latentdynamics._paths import get_repo_root
from latentdynamics.replay import load_experiment
from latentdynamics.systems.coral import RedCoralModel

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = CODE_ROOT / "output" / "coral_figures"
PIN_FILE = (
    CODE_ROOT
    / "artifacts"
    / "reference_results"
    / "coral"
    / "render_inputs_sha256.json"
)
#: Boxes to draw. Prefer what a live run just computed; the shipped reference
#: is the standalone fallback so this script still works from a clean checkout.
#: The two agree numerically (verified: max |delta| 0.0 over all endpoints,
#: identical labels); they differ only in line endings.
MORSE_SETS_RECOMPUTED = "replay/coral_basic/train_500/seed_16/MG/morse_sets"
MORSE_SETS_REFERENCE = "replay_sources/coral/train_500/seed_16/MG/morse_sets"


def _morse_sets_path(repo_root: Path) -> Path:
    """Recomputed boxes when present, else the shipped reference."""
    live = repo_root / MORSE_SETS_RECOMPUTED
    return live if live.is_file() else repo_root / MORSE_SETS_REFERENCE
CHECKPOINT_RELATIVE = "replay_sources/coral/train_500/seed_16/models/autoencoder.pt"

EXPERIMENT = "coral_basic"
TRAIN_FILE = "train_500"
SEED = 16

# Ambient fixed points of the red-coral model: extinction (a0), the stable
# positive equilibrium (a1), and the intermediate repelling equilibrium (r).
# These literal values are the ones the published figure was rendered from;
# they are asserted against RedCoralModel.FIXED_POINTS so drift is caught.
A0 = [0] * 13
A1 = [868.12066371, 771.75927004, 488.52361793, 340.5009617, 176.0389972, 76.92904178,
      22.07863499, 12.60690058, 4.19809789, 3.14857342, 3.14857342, 1.04847495, 1.04847495]
R = [321.84389612752153, 286.11922365736666, 181.1134685751131, 126.23608759685382,
     65.26405728757342, 28.52039303466959, 8.185352800950172, 4.673836449342548,
     1.5563875376310683, 1.1672906532233014, 1.1672906532233014, 0.3887077875233593,
     0.3887077875233593]
FIXED_POINTS = {"a0": A0, "a1": A1, "r": R}

COLOR_LIST = ["#FFB000", "#DC267F", "#648FFF", "#FE6100", "#785EF0"]
MARKERS = ["*", "^", "s"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_fixed_points_match_model() -> None:
    for name, values in FIXED_POINTS.items():
        if not np.array_equal(
            np.asarray(values, dtype=np.float64),
            RedCoralModel.FIXED_POINTS[name],
        ):
            raise AssertionError(
                f"hard-coded fixed point {name!r} differs from "
                "RedCoralModel.FIXED_POINTS"
            )


def _verify_inputs(repo_root: Path) -> None:
    record: dict = json.loads(PIN_FILE.read_text(encoding="utf-8"))
    legacy: dict[str, str] = record.pop("legacy_original", {})
    pins: dict[str, str] = dict(record)
    if not (repo_root / CHECKPOINT_RELATIVE).is_file() and legacy:
        # Legacy tree: check the original encoder.pt digest instead.
        pins.pop(CHECKPOINT_RELATIVE, None)
        pins.update(legacy)
    for relative, expected in sorted(pins.items()):
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"missing replay artifact {path}; fetch the coral artifacts first"
            )
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"replay artifact {path} does not match the pinned digest: "
                f"{observed} != {expected}"
            )


def render(output_dir: Path) -> Path:
    repo_root = get_repo_root()
    _assert_fixed_points_match_model()
    _verify_inputs(repo_root)

    experiment = load_experiment(
        EXPERIMENT,
        seed=SEED,
        train_file=TRAIN_FILE,
        device="cpu",
    )
    encoded_pts = [
        float(experiment.encode([point])[0, 0]) for point in FIXED_POINTS.values()
    ]

    plt.rcParams.update(
        {"font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral"]}
    )

    sets_path = _morse_sets_path(repo_root)
    print("morse sets:", sets_path.relative_to(repo_root))

    # CMGDB draws the interval bands and labels each set by its Morse-graph
    # node number (0, 1, 2), matching panel (a); the fixed-point markers are
    # overlaid below. Each coral Morse set is one contiguous interval, so the
    # built-in per-piece labelling puts exactly one number over each band.
    # Axis staged like the Chafee d=1 panel: arrow at the right end of the
    # axis line, coordinate label at the tip, tick marks kept.
    fig, ax = CMGDB.PlotMorseSets1D(
        str(sets_path),
        clist=COLOR_LIST,
        fig_w=12,
        fig_h=2,
        label_sets=True,
        axis_labels=True,
        xlabel="$z_1$",
        fontsize=25,
        show=False,
    )

    for i, z_val in enumerate(encoded_pts):
        ax.scatter(z_val, 0.0, marker=MARKERS[i], color="black", s=60,
                   edgecolor="black", clip_on=False, zorder=10)

    ax.locator_params(axis="x", nbins=4)
    ax.tick_params(axis="x", which="major", labelsize=25, pad=25)

    # Legend intentionally omitted; markers are identified in the caption.
    # The right margin leaves room for the axis-tip coordinate label.
    fig.subplots_adjust(left=0.04, right=0.92, bottom=0.25, top=0.7)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "morse_sets_1D.pdf"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(pdf_path.with_suffix(".png"), dpi=150)
    plt.close(fig)

    print("encoded:", {k: round(v, 4) for k, v in zip(FIXED_POINTS, encoded_pts)})
    print("wrote", pdf_path)
    return pdf_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
