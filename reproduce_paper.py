"""Reproduce the paper's four computational example families.

Each entry in :data:`EXPERIMENTS` describes one family as an ordered list of
steps. A step is either a packaged pipeline config (resolved by
:func:`latentdynamics.config.load_config` and run through
:func:`latentdynamics.cli.pipeline.run`) or a script invocation under
``scripts/``. Every step carries a tier:

* ``replay``    -- rerun CMGDB and everything downstream of it on the saved
                   trained models, then render and re-derive metrics. The
                   networks, scalers and datasets are reused; the box map,
                   Morse graph, Conley indices, coarsenings, regions of
                   attraction and residual/tolerance estimates are recomputed.
                   Runs by default.
* ``recompute`` -- rerun a CMGDB or analysis computation from saved inputs
                   (checkpoints, datasets); no training. Minutes to hours.
* ``retrain``   -- retrain models before analysis. Longest tier.
* ``blocked``   -- inputs were not preserved, so the step cannot run at all.
                   Never selected by ``--tiers all``; listed so the gap is
                   visible rather than surfacing as a late failure.

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

TIERS = ("replay", "recompute", "retrain", "blocked")

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
                "name": "latent_cmgdb",
                "tier": "replay",
                "config": "leslie_2gen_contraction_replay",
                "stages": "morse,render,metrics",
                # float32 box-map evaluation differs by device: on a machine
                # where MPS is available the adaptive grid resolves a handful
                # of boundary cells differently. The published runs were CPU,
                # and pinning it reproduces their Morse sets exactly.
                "device": "cpu",
                "runtime": "~1 h (CMGDB rerun on the saved model, then render+metrics)",
            },
            {
                "name": "residual_tolerance",
                "tier": "replay",
                "command": ["scripts/compute_sampled_residual_tolerance.py",
                            "leslie_2gen_contraction"],
                "runtime": "~40 min (sampled residual/tolerance rows)",
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
                "tier": "recompute",
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
                "name": "latent_cmgdb",
                "tier": "replay",
                "config": "leslie3d_example1_replay",
                "stages": "morse,render,metrics",
                # float32 box-map evaluation differs by device: on a machine
                # where MPS is available the adaptive grid resolves a handful
                # of boundary cells differently. The published runs were CPU,
                # and pinning it reproduces their Morse sets exactly.
                "device": "cpu",
                "runtime": "minutes (adaptive 23/23/27 CMGDB rerun on the saved model)",
            },
            {
                "name": "coarsen_morse_graph",
                "tier": "replay",
                "command": ["scripts/leslie3d_example1_coarsen_morse_graph.py"],
                "runtime": "~6 min (rebuilds the 23/23/27 adaptive cell graph, then merges nodes 4,5)",
            },
            {
                "name": "uniform_grid_22",
                "tier": "replay",
                "command": ["scripts/leslie3d_example1_uniform_grid.py", "--depth", "22"],
                "runtime": "minutes (fixed-depth 22 CMGDB recompute; wall clock unrecorded in the audit)",
            },
            {
                "name": "uniform_sampled_metrics",
                "tier": "replay",
                "command": ["scripts/leslie3d_example1_uniform_sampled_metrics.py", "--depth", "22"],
                "runtime": "minutes-hours (dense residual/tolerance sampling)",
            },
            {
                "name": "verify_closures",
                "tier": "replay",
                "command": ["scripts/leslie3d_example1_verify_closures.py"],
                "runtime": "minutes",
            },
            {
                # Paper panels straight from CMGDB: coarsened sets/graph, the
                # fine sets with the 4/5 inset, and the coarse (22,22,24) pair.
                # Needs no Conley index for cells, so it runs on any build.
                "name": "paper_panels",
                "tier": "replay",
                "command": ["scripts/render_paper_figures.py",
                            "--only", "leslie3d_example1"],
                "runtime": "minutes (panels a-d and coarse a,b)",
            },
            {
                "name": "residual_tolerance",
                "tier": "replay",
                "command": ["scripts/compute_sampled_residual_tolerance.py",
                            "leslie3d_example1"],
                "runtime": "~1 h (sampled residual/tolerance, fine rows)",
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
                "tier": "recompute",
                "command": ["scripts/plot_original_leslie3d_ground_truth_morse_graph.py"],
                "runtime": "seconds (DOT render; needs the reference outputs)",
            },
            {
                "name": "direct_3d_reference_cubical_figure",
                "tier": "recompute",
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
                "name": "latent_cmgdb_d2",
                "tier": "replay",
                "config": "chafee_infante_replay",
                "stages": "morse,render,metrics",
                # float32 box-map evaluation differs by device: on a machine
                # where MPS is available the adaptive grid resolves a handful
                # of boundary cells differently. The published runs were CPU,
                # and pinning it reproduces their Morse sets exactly.
                "device": "cpu",
                "runtime": "minutes-hours (d=2 CMGDB rerun on the saved model)",
            },
            {
                # Single-seed. The published row is a nine-seed ensemble --
                # fresh seeds 20260727-31 plus decoder seeds 20260732-35, folded
                # together by --stage merge (see fresh_trajectory_ensemble_seeds
                # in the reference dense_sampling.json). This runs the base seed
                # only, which reproduces its counterpart to ~6e-6 but samples
                # 1 of 9 point sets, so the residual maximum is a weaker lower
                # bound than the published one. R < tau still holds.
                "name": "residual_tolerance_d2",
                "tier": "replay",
                "command": ["scripts/compute_sampled_residual_tolerance.py",
                            "chafee_infante_current"],
                "runtime": "~30 min (base seed only; the published row merges 9 seeds)",
            },
            {
                "name": "latent_dimension_study",
                "tier": "recompute",
                "command": ["scripts/chafee_latent_dimension_study.py"],
                "runtime": (
                    "hours; the saved d=1/d=3 checkpoints no longer exist, so this "
                    "retrains them and results will not match the published models"
                ),
            },
            {
                "name": "coarsen_d2",
                "tier": "blocked",
                "command": ["scripts/coarsen_chafee_infante.py"],
                "runtime": "BLOCKED: needs replay_sources/chafee_infante/reference_inputs/{ci_model_weights.pth,train_data.csv}, which came from the coauthor archive and was never committed",
            },
            {
                "name": "roa_overlay",
                "tier": "blocked",
                "command": ["scripts/plot_chafee_coarse_morse_roa_overlay.py"],
                "runtime": "BLOCKED: consumes the coarsen_d2 outputs above",
            },
            {
                "name": "standardized_render",
                "tier": "blocked",
                "command": ["scripts/render_chafee_infante_standardized.py"],
                "runtime": "BLOCKED: d=1 panels; the chafee_latent_dimension_study checkpoints were stripped by the output/**/*.pt ignore rule",
            },
            {
                "name": "d3_graph_palette_render",
                "tier": "blocked",
                "command": ["scripts/render_chafee_infante_3d_graph_palette.py"],
                "runtime": "BLOCKED: d=3 panels; same missing checkpoints as the d=1 render",
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
                "tier": "blocked",
                "command": ["scripts/chafee_basin_table.py"],
                "runtime": "BLOCKED: needs replay_sources/chafee_infante/statistics/ci_completed_10k_raw_classifications_45_runs.csv, never committed",
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
                "name": "latent_cmgdb",
                "tier": "replay",
                "config": "coral_basic",
                "stages": "morse,render,metrics",
                # float32 box-map evaluation differs by device: on a machine
                # where MPS is available the adaptive grid resolves a handful
                # of boundary cells differently. The published runs were CPU,
                # and pinning it reproduces their Morse sets exactly.
                "device": "cpu",
                "cell_index": 16,
                "runtime": "seconds (CMGDB rerun on the saved seed-16 model)",
            },
            {
                "name": "residual_tolerance",
                "tier": "replay",
                "command": ["scripts/compute_sampled_residual_tolerance.py",
                            "coral_candidate_train500_seed16"],
                "runtime": "minutes (sampled residual/tolerance rows)",
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
    # A step may pin its own stages (recomputing CMGDB rather than re-rendering
    # it) and its own cell (the one seed a paper figure was drawn from).
    step_stages = step.get("stages")
    if step_stages is not None:
        stages = [s.strip() for s in step_stages.split(",") if s.strip()]
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
        cell_index=step.get("cell_index"),
        device=step.get("device"),
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

    # 'all' means everything runnable; blocked steps must be asked for by name.
    tiers = set(TIERS) - {"blocked"} if args.tiers == "all" else {
        t for t in args.tiers.split(",") if t
    }
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
