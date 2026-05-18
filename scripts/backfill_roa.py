"""Backfill ``regions_of_attraction.png`` under every completed seed dir.

Walks ``code/output/`` for directories matching
``<example>/[train_*/]seed_*/MG/`` that contain both ``morse_graph`` and
``morse_sets`` and have a sibling ``../models/`` checkpoint, then writes
``regions_of_attraction.png`` next to the existing Morse plots.

2D-only (skips when the checkpoint's latent dim is not 2). Idempotent:
existing RoA files are kept unless ``--force`` is passed.

Usage:
    python scripts/backfill_roa.py [--output-root code/output] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from latentdynamics.analysis.regions_of_attraction import MorseGraph
from latentdynamics.training import (
    has_legacy_checkpoint,
    has_new_checkpoint,
    load_any_checkpoint,
)
from latentdynamics.viz.regions_of_attraction import render_cell_graph_roa


def find_runs(output_root: Path) -> list[Path]:
    """Return every ``<example>/[train_*/]seed_*/MG`` directory with both
    ``morse_graph`` and ``morse_sets`` present."""
    out: list[Path] = []
    for mg in output_root.rglob("MG/morse_graph"):
        if (mg.parent / "morse_sets").exists():
            out.append(mg.parent)
    return sorted(out)


def backfill_one(mg_dir: Path, *, force: bool) -> tuple[str, str]:
    out_path = mg_dir / "regions_of_attraction.png"
    if out_path.exists() and not force:
        return "skip", "already exists"

    seed_dir = mg_dir.parent
    model_dir = seed_dir / "models"
    if not (has_legacy_checkpoint(model_dir) or has_new_checkpoint(model_dir)):
        return "skip", "no checkpoint"

    model, arch = load_any_checkpoint(model_dir)
    if arch.low_dims != 2:
        return "skip", f"latent dim {arch.low_dims} != 2"

    device = torch.device("cpu")
    model.to(device).eval()

    n_min = len(MorseGraph.from_dot(mg_dir / "morse_graph").minimal)
    system_name = seed_dir.parent.name
    if seed_dir.parent.name.startswith("train_"):
        system_name = seed_dir.parent.parent.name
    title = (
        f"{system_name} — regions of attraction "
        f"({n_min} minimal Morse set{'s' if n_min != 1 else ''})"
    )
    render_cell_graph_roa(
        mg_dir / "morse_graph",
        mg_dir / "morse_sets",
        model.latent_map,
        out_path,
        device=str(device),
        title=title,
    )
    return "ok", f"{n_min} minimal"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runs = find_runs(args.output_root)
    if not runs:
        print(f"no MG directories under {args.output_root}")
        return 0
    print(f"found {len(runs)} MG director{'ies' if len(runs) != 1 else 'y'}")
    n_ok = 0
    for mg_dir in runs:
        status, msg = backfill_one(mg_dir, force=args.force)
        rel = mg_dir.relative_to(args.output_root)
        print(f"  {status:<4s}  {rel}  ({msg})")
        if status == "ok":
            n_ok += 1
    print(f"wrote {n_ok} RoA image(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
