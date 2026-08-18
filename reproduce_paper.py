"""Reproduce the paper's four computational example families.

Each entry in :data:`EXPERIMENTS` describes one family as an ordered list of
steps. A step is either a packaged pipeline config (resolved by
:func:`latentdynamics.config.load_config` and run through
:func:`latentdynamics.cli.pipeline.run`) or a script invocation under
``scripts/``. Every step carries a tier:

* ``replay``    -- re-render figures / recompute metrics from saved artifacts;
                   seconds to minutes, runs by default.
* ``recompute`` -- rerun a CMGDB or analysis computation from saved inputs
                   (checkpoints, datasets); no training. Minutes to hours.
* ``retrain``   -- retrain models before analysis. Longest tier.

Default behavior runs only the ``replay`` tier. Pass ``--tiers
replay,recompute`` (or ``all``) to escalate. Runtime notes are honest
estimates from audited runs on an M4 Pro laptop; they are not promises.

Usage:
    python reproduce_paper.py --list
    python reproduce_paper.py --only coral
    python reproduce_paper.py --only leslie3d_example1 --tiers replay,recompute
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

from latentdynamics.cli import pipeline
from latentdynamics.config import load_config

REPO_ROOT = Path(__file__).resolve().parent

TIERS = ("replay", "recompute", "retrain")

# Each step: name, tier, runtime note, and either a packaged config stem
# ("config") or a scripts/ argv ("command", relative to the repo root).
EXPERIMENTS: dict[str, dict] = {
    "leslie_2gen_contraction": {
        "description": (
            "Extended two-generation Leslie map (2D dynamics embedded in 10D): "
            "latent replay plus the 2D direct reference computation."
        ),
        "steps": [
            {
                "name": "latent_replay",
                "tier": "replay",
                "config": "leslie_2gen_contraction_replay",
                "runtime": "seconds-minutes (render+metrics from saved Morse artifacts)",
            },
            {
                "name": "latent_morse_sets_render",
                "tier": "replay",
                "command": ["scripts/render_leslie_2gen_contraction_morse_sets.py"],
                "runtime": "seconds (saved morse_sets CSV render)",
            },
            {
                "name": "direct_2d_reference",
                "tier": "recompute",
                "command": [
                    "scripts/compute_original_leslie.py",
                    "--system", "2d",
                    "--subdiv", "26", "30", "40",
                    "--box-map-backend", "on_demand",
                ],
                "runtime": "~54 min (audited on-demand CMGDB run at subdiv 26/30/40)",
            },
            {
                "name": "direct_2d_reference_figures",
                "tier": "replay",
                "command": ["scripts/render_original_leslie2d_full_paper_figures.py"],
                "runtime": "seconds-minutes (render-only; needs direct_2d_reference outputs)",
            },
        ],
    },
    "leslie3d_example1": {
        "description": (
            "3D Leslie map with an author-provided 2D latent model: direct 3D "
            "reference, adaptive latent fine run, coarsened Morse graph, and "
            "the uniform depth-22 comparison."
        ),
        "steps": [
            {
                "name": "latent_fine_replay",
                "tier": "replay",
                "config": "leslie3d_example1_replay",
                "runtime": "seconds-minutes (render+metrics from saved artifacts)",
            },
            {
                "name": "paper_figures_render",
                "tier": "replay",
                "command": ["scripts/render_leslie3d_example1_figures.py"],
                "runtime": "seconds-minutes (display-only rerender of saved Morse data)",
            },
            {
                "name": "coarsen_morse_graph",
                "tier": "recompute",
                "command": ["scripts/leslie3d_example1_coarsen_morse_graph.py"],
                "runtime": "~6 min (rebuilds the 23/23/27 adaptive cell graph, then merges nodes 4,5)",
            },
            {
                "name": "uniform_grid_22",
                "tier": "recompute",
                "command": ["scripts/leslie3d_example1_uniform_grid.py", "--depth", "22"],
                "runtime": "minutes (fixed-depth 22 CMGDB recompute; wall clock unrecorded in the audit)",
            },
            {
                "name": "uniform_sampled_metrics",
                "tier": "recompute",
                "command": ["scripts/leslie3d_example1_uniform_sampled_metrics.py", "--depth", "22"],
                "runtime": "minutes-hours (dense residual/tolerance sampling)",
            },
            {
                "name": "verify_closures",
                "tier": "recompute",
                "command": ["scripts/leslie3d_example1_verify_closures.py"],
                "runtime": "minutes",
            },
            {
                "name": "direct_3d_reference_screen",
                "tier": "recompute",
                "command": [
                    "scripts/screen_original_leslie3d_initial.py",
                    "29",
                    "--subdiv-min", "33",
                    "--subdiv-max", "36",
                ],
                "runtime": "~97 min (audited level-33 CMGDB screen)",
            },
            {
                "name": "direct_3d_reference_conley",
                "tier": "recompute",
                "command": [
                    "scripts/compute_original_leslie3d_conley_from_saved_sets.py",
                    "--node", "0", "--node", "1", "--node", "2",
                    "--node", "3", "--node", "4", "--node", "5",
                    "--output-dir", "output/original_leslie3d/conley_from_saved_sets",
                ],
                "runtime": "minutes-hours (Conley indices on the saved level-33 sets)",
            },
            {
                "name": "direct_3d_reference_graph_figure",
                "tier": "replay",
                "command": ["scripts/plot_original_leslie3d_ground_truth_morse_graph.py"],
                "runtime": "seconds (DOT render; needs the reference outputs)",
            },
            {
                "name": "direct_3d_reference_cubical_figure",
                "tier": "replay",
                "command": ["scripts/render_original_leslie3d_morse_sets_cubical.py"],
                "runtime": "minutes (154 MB cell CSV parse, render-only)",
            },
        ],
    },
    "chafee_infante": {
        "description": (
            "Chafee--Infante PDE discretization: d=1/2/3 latent-dimension "
            "study, coarsened d=2 representation, RoA overlay, and the "
            "45-computation basin-classification statistics."
        ),
        "steps": [
            {
                "name": "latent_dimension_study",
                "tier": "recompute",
                "command": ["scripts/chafee_latent_dimension_study.py"],
                "runtime": (
                    "minutes (d=1,2) to hours (d=3 adaptive level-33); adopts the "
                    "saved d=1/d=3 checkpoints when present, otherwise retrains them"
                ),
            },
            {
                "name": "coarsen_d2",
                "tier": "recompute",
                "command": ["scripts/coarsen_chafee_infante.py"],
                "runtime": "minutes (2D CMGDB at subdiv 14/16/22 from the archived inputs)",
            },
            {
                "name": "roa_overlay",
                "tier": "recompute",
                "command": ["scripts/plot_chafee_coarse_morse_roa_overlay.py"],
                "runtime": "minutes (uniform 16/16/16 basin graph plus render)",
            },
            {
                "name": "standardized_render",
                "tier": "replay",
                "command": ["scripts/render_chafee_infante_standardized.py"],
                "runtime": "seconds-minutes (render-only; needs study outputs)",
            },
            {
                "name": "d3_graph_palette_render",
                "tier": "replay",
                "command": ["scripts/render_chafee_infante_3d_graph_palette.py"],
                "runtime": "seconds (render-only)",
            },
            {
                "name": "basin_stats_d2_archive",
                "tier": "recompute",
                "command": ["scripts/analyze_chafee_d2_archive.py"],
                "runtime": "minutes (15 archived checkpoints, uniform level-16 basins each)",
            },
            {
                "name": "basin_stats_d1_matched",
                "tier": "retrain",
                "command": ["scripts/run_chafee_d1_matched_5x3.py"],
                "runtime": "hours (15 full-batch d=1 trainings plus analysis)",
            },
            {
                "name": "basin_stats_d3_training",
                "tier": "retrain",
                "command": ["scripts/run_chafee_d3_matched_5x3_training.py"],
                "runtime": "hours (15 full-batch d=3 trainings)",
            },
            {
                "name": "basin_stats_d3_analysis",
                "tier": "recompute",
                "command": ["scripts/run_chafee_d3_ondemand_5x3_controller.py"],
                "runtime": "hours (15 on-demand 3D CMGDB basin computations)",
            },
            {
                "name": "basin_table",
                "tier": "replay",
                "command": ["scripts/chafee_basin_table.py"],
                "runtime": "seconds (derives the printed table from the shipped per-IC CSV)",
            },
            {
                "name": "residual_audit",
                "tier": "recompute",
                "command": ["scripts/audit_chafee_dimension_residuals.py"],
                "runtime": "minutes (finite-data residuals on the three saved checkpoints)",
            },
        ],
    },
    "coral": {
        "description": (
            "13D red-coral model with a 1D latent space: replay of the "
            "author-provided train_500/seed_16 run."
        ),
        "steps": [
            {
                "name": "latent_replay",
                "tier": "replay",
                "config": "coral_basic",
                "runtime": "seconds-minutes (render+metrics from saved artifacts)",
            },
            {
                "name": "morse_sets_render",
                "tier": "replay",
                "command": ["scripts/render_coral_morse_sets_1d.py"],
                "runtime": "seconds (1D bands with encoded fixed points)",
            },
        ],
    },
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


def _run_pipeline_step(
    step: dict,
    *,
    stages: list[str],
    max_seeds: int | None,
    verbose: bool,
    replay_root: Path | None,
) -> str:
    try:
        cfg = load_config(step["config"])
    except FileNotFoundError:
        return f"missing config: {step['config']}"
    if not cfg.paths.output_dir.exists() and "data" not in stages:
        return (
            f"no on-disk artifacts at {cfg.paths.output_dir}; fetch the replay "
            "bundle first (latentdynamics.replay.fetch_artifacts)"
        )
    results = pipeline.run(
        cfg,
        stages=stages,
        max_seeds=max_seeds,
        verbose=verbose,
        replay_root=replay_root,
    )
    return _summarise_results(results)


def _run_script_step(step: dict) -> str:
    argv = [sys.executable, *step["command"]]
    argv[1] = str(REPO_ROOT / argv[1])
    completed = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{Path(step['command'][0]).name} exited with {completed.returncode}"
        )
    return " ".join(step["command"])


def _print_plan() -> None:
    for name, experiment in EXPERIMENTS.items():
        print(f"\n{name}: {experiment['description']}")
        for step in experiment["steps"]:
            target = step.get("config") or " ".join(step["command"])
            print(f"  [{step['tier']:9s}] {step['name']}: {target}")
            print(f"              {step['runtime']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", choices=list(EXPERIMENTS), help="run a single family")
    parser.add_argument(
        "--tiers",
        default="replay",
        help="comma-separated subset of replay,recompute,retrain, or 'all' (default: replay)",
    )
    parser.add_argument(
        "--list", action="store_true", help="print every step with tier and runtime, then exit"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="render,metrics",
        help="pipeline stages for config-backed steps (default: render,metrics)",
    )
    parser.add_argument("--max-seeds", type=int, default=None, help="cap seeds per config sweep")
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=None,
        help=(
            "destination for derived writes when replaying a read_only config; "
            f"defaults to {pipeline.DEFAULT_REPLAY_ROOT}"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        _print_plan()
        return 0

    tiers = set(TIERS) if args.tiers == "all" else {t for t in args.tiers.split(",") if t}
    unknown = tiers - set(TIERS)
    if unknown:
        parser.error(f"unknown tiers: {sorted(unknown)}; valid: {TIERS}")
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
        for step in EXPERIMENTS[name]["steps"]:
            label = f"{name}/{step['name']}"
            if step["tier"] not in tiers:
                print(f"[SKIP] {label}: tier {step['tier']!r} not selected ({step['runtime']})")
                continue
            t0 = time.perf_counter()
            try:
                if "config" in step:
                    status = _run_pipeline_step(
                        step,
                        stages=stages,
                        max_seeds=args.max_seeds,
                        verbose=not args.quiet,
                        replay_root=args.replay_root,
                    )
                else:
                    status = _run_script_step(step)
            except Exception as exc:
                print(f"[FAIL] {label}: {exc}")
                traceback.print_exc()
                failures.append(label)
                continue
            elapsed = time.perf_counter() - t0
            if status.startswith(("no on-disk artifacts", "missing config")):
                print(f"[SKIP] {label}: {status}")
                skipped.append((label, status))
            else:
                print(f"[OK]   {label} in {elapsed:.1f}s ({status})")

    print()
    if skipped:
        print(f"{len(skipped)} skipped:")
        for label, reason in skipped:
            print(f"  - {label}: {reason}")
    if failures:
        print(f"{len(failures)} failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
