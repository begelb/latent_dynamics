"""Replay paper figures through the unified config pipeline.

Each entry in :data:`EXPERIMENTS` maps a paper-figure label to the YAML config
that drives :func:`latentdynamics.cli.pipeline.run`. Default behavior is to
re-render figures and recompute paper metrics from the **saved** Morse and
checkpoint artifacts on disk - no CMGDB or training is invoked. Some archived
figures remain partial because their source artifacts are missing or zero-byte.
To retrain, pass ``--stages all``.

Usage:
    python reproduce_paper.py --only fig_leslie3d_spurious # one replay-ready figure
    python reproduce_paper.py --only fig_leslie_contraction
    python reproduce_paper.py --stages all --max-seeds 1   # full re-run, capped seeds
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from latentdynamics.cli import pipeline
from latentdynamics.config import load_config

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

EXPERIMENTS: dict[str, str] = {
    "fig_leslie_contraction": "leslie_contraction_patrick.yaml",
    "fig_leslie3d_spurious": "leslie3d_spurious_brittany.yaml",
    "fig_leslie3d_success": "leslie3d_success_patrick.yaml",
    "fig_chafee_infante": "chafee_infante_marcio.yaml",
    "fig_coral_basic": "coral_basic.yaml",
    "fig_coral_data_scaling": "coral_data_scaling.yaml",
    "fig_coral_adaptive": "coral_adaptive.yaml",
}


def _summarise_results(results: list[dict]) -> str:
    rendered = sum(
        1 for r in results if isinstance(r.get("render"), dict) and "skipped" not in r["render"]
    )
    skipped = sum(
        1 for r in results if isinstance(r.get("render"), dict) and "skipped" in r["render"]
    )
    metric_errors = sum(
        1 for r in results if isinstance(r.get("metrics"), dict) and "error" in r["metrics"]
    )
    parts = [f"{len(results)} cell(s)"]
    if rendered:
        parts.append(f"{rendered} rendered")
    if skipped:
        parts.append(f"{skipped} skipped (no artifacts)")
    if metric_errors:
        parts.append(f"{metric_errors} metric error(s)")
    return ", ".join(parts)


def _run_one(
    name: str,
    config_name: str,
    *,
    stages: list[str],
    max_seeds: int | None,
    verbose: bool,
    force_overwrite: bool = False,
    replay_root: Path | None = None,
) -> str:
    cfg_path = CONFIGS_DIR / config_name
    if not cfg_path.exists():
        return f"missing config: {cfg_path}"

    cfg = load_config(cfg_path)
    if not cfg.paths.output_dir.exists() and "data" not in stages:
        return f"no on-disk artifacts at {cfg.paths.output_dir}; rerun with --stages all to retrain"

    results = pipeline.run(
        cfg,
        stages=stages,
        max_seeds=max_seeds,
        verbose=verbose,
        force_overwrite=force_overwrite,
        replay_root=replay_root,
    )
    return _summarise_results(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", choices=list(EXPERIMENTS), help="Run a single experiment by name")
    parser.add_argument(
        "--stages",
        type=str,
        default="render,metrics",
        help="comma-separated stages or 'all' (default: render,metrics)",
    )
    parser.add_argument("--max-seeds", type=int, default=None, help="Cap the seeds per sweep")
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="bypass paths.read_only and legacy-checkpoint guards",
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
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    stages = (
        list(pipeline.ALL_STAGES)
        if args.stages == "all"
        else [s for s in args.stages.split(",") if s]
    )
    targets = [args.only] if args.only else list(EXPERIMENTS)

    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    for name in targets:
        print(f"\n========== {name} ==========")
        t0 = time.perf_counter()
        try:
            status = _run_one(
                name,
                EXPERIMENTS[name],
                stages=stages,
                max_seeds=args.max_seeds,
                verbose=not args.quiet,
                force_overwrite=args.force_overwrite,
                replay_root=args.replay_root,
            )
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
            failures.append(name)
            continue
        elapsed = time.perf_counter() - t0
        if status.startswith("no on-disk artifacts") or status.startswith("missing config"):
            print(f"[SKIP] {name}: {status}")
            skipped.append((name, status))
        else:
            print(f"[OK]   {name} in {elapsed:.1f}s ({status})")

    print()
    if skipped:
        print(f"{len(skipped)} skipped:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    if failures:
        print(f"{len(failures)} failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
