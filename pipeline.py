"""Single config-driven entry point for the entire experiment pipeline.

Usage::

    # Full re-run (all stages, all seeds in cfg.seeds):
    python pipeline.py --config coral_basic

    # Re-render figures + recompute metrics from saved artifacts (no CMGDB):
    python pipeline.py --config coral_basic --stages render,metrics

    # Cap the seed sweep for laptop smoke checks:
    python pipeline.py --config coral_data_scaling --max-seeds 3

Bare names (no path separator) resolve from the packaged configs directory.
An explicit path (relative or absolute) is also accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from latentdynamics.cli import pipeline
from latentdynamics.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="path to a YAML experiment config"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=",".join(pipeline.ALL_STAGES),
        help=f"comma-separated subset of {list(pipeline.ALL_STAGES)} or 'all' (default: all)",
    )
    parser.add_argument("--max-seeds", type=int, default=None, help="cap the number of seeds")
    parser.add_argument(
        "--device", type=str, default=None, help="torch device override (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--cell-index",
        type=int,
        default=None,
        help="run only one zero-based train-file/seed cell",
    )
    parser.add_argument(
        "--expected-cells",
        type=int,
        default=None,
        help="assert the expanded config has this many cells (optional sanity check)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="resume safely by skipping stages whose expected artifacts already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the train-file/seed cell plan and exit without running stages",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="bypass paths.read_only and legacy-checkpoint guards; only use to "
        "intentionally clobber preserved paper artifacts",
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=None,
        help=(
            "destination for derived render/metrics/diagnose/manifest writes "
            "when running against a paths.read_only=true config. "
            f"Defaults to {pipeline.DEFAULT_REPLAY_ROOT} when omitted."
        ),
    )
    parser.add_argument(
        "--figures",
        type=str,
        default=None,
        help="comma-separated subset of {morse,roa,overlay,extras} for the render stage "
        "(default: all); e.g. '--figures overlay' regenerates only the orbit overlay and "
        "skips the regions-of-attraction recompute (the one expensive figure)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    stages = (
        list(pipeline.ALL_STAGES)
        if args.stages == "all"
        else [s for s in args.stages.split(",") if s]
    )
    cell_index = args.cell_index

    if args.dry_run:
        plan = pipeline.plan_cells(cfg, max_seeds=args.max_seeds)
        print(json.dumps({"config": str(args.config), "stages": stages, "cells": plan}, indent=2))
        return 0

    figures = {f for f in args.figures.split(",") if f} if args.figures else None

    results = pipeline.run(
        cfg,
        stages=stages,
        max_seeds=args.max_seeds,
        device=args.device,
        cell_index=cell_index,
        expected_cells=args.expected_cells,
        skip_completed=args.skip_completed,
        force_overwrite=args.force_overwrite,
        replay_root=args.replay_root,
        figures=figures,
        verbose=not args.quiet,
    )

    summary_name = (
        f"pipeline_summary_cell_{cell_index}.json"
        if cell_index is not None
        else "pipeline_summary.json"
    )
    resolved_replay_root = pipeline._resolve_replay_root(
        cfg,
        args.replay_root,
        force_overwrite=args.force_overwrite,
    )
    summary_root = (
        cfg.paths.output_dir
        if resolved_replay_root is None
        else Path(resolved_replay_root) / (cfg.experiment_name or "unnamed")
    )
    summary_path = summary_root / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    if not args.quiet:
        print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
