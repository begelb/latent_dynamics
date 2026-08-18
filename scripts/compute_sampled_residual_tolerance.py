#!/usr/bin/env python3
"""Compute the sampled residual/tolerance estimates for the paper table.

Dispatches per experiment key to :mod:`latentdynamics.analysis.sampled_metrics`:

* ``leslie3d_example1``, ``leslie_2gen_contraction``,
  ``coral_candidate_train500_seed16``, ``chafee_infante_current`` run the
  dense tolerance search and the dense residual search on the fetched replay
  artifacts.  The Chafee--Infante d=2 residual search is a seed ensemble:
  repeat the residual stage with ``--seed``/``--output-suffix`` and combine
  the runs with ``--merge-suffixes``.
* ``chafee_latent_dimensions`` runs the staged d=1/d=3 appendix pipeline
  (``validate``, ``tolerance --dimension {1,3}``, ``stored``,
  ``fresh --seed S``, ``decoder --seed S``, ``merge``).

Frozen published copies of every output are shipped under
``artifacts/reference_results/sampled_residual_tolerance/``.  New runs write
under ``output/sampled_residual_tolerance/`` unless ``--output-root`` is given.

Examples::

    python scripts/compute_sampled_residual_tolerance.py leslie3d_example1
    python scripts/compute_sampled_residual_tolerance.py chafee_infante_current \
        --stage residual --seed 20260728 --output-suffix seed20260728
    python scripts/compute_sampled_residual_tolerance.py chafee_latent_dimensions \
        --stage tolerance --dimension 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from latentdynamics.analysis.sampled_metrics import (
    EXAMPLES,
    chafee_appendix,
    merge_chafee_dense_runs,
    run_dense_sampling,
    run_tolerance_evaluation,
)
from latentdynamics.analysis.sampled_metrics.residual_protocol import (
    BASE_SEED,
    CHAFEE_INITIALS,
)
from latentdynamics.analysis.sampled_metrics.tolerance_protocol import (
    DEFAULT_LOCAL_BOXES,
    DEFAULT_SAMPLE_TARGET,
    DEFAULT_SOBOL_SCRAMBLES,
)

APPENDIX_STAGES = ("validate", "tolerance", "stored", "fresh", "decoder", "merge")


def add_common_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        type=Path,
        help="directory for new results (default: <repo>/output/sampled_residual_tolerance)",
    )
    parser.add_argument(
        "--blocks-root",
        type=Path,
        help=(
            "directory with per-example attracting-block artifacts "
            "(default: <repo>/artifacts/reference_results/sampled_residual_tolerance)"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    for name in sorted(EXAMPLES):
        sub = subparsers.add_parser(name)
        sub.add_argument(
            "--stage", choices=("tolerance", "residual", "merge", "all"), default="all"
        )
        sub.add_argument("--label", type=int, action="append")
        sub.add_argument("--local-boxes", type=int, default=DEFAULT_LOCAL_BOXES)
        sub.add_argument("--sample-target", type=int, default=DEFAULT_SAMPLE_TARGET)
        sub.add_argument("--sobol-scrambles", type=int, default=DEFAULT_SOBOL_SCRAMBLES)
        sub.add_argument("--seed", type=int, default=BASE_SEED)
        sub.add_argument("--output-suffix")
        sub.add_argument("--skip-fresh-trajectories", action="store_true")
        sub.add_argument("--chafee-initials", type=int, default=CHAFEE_INITIALS)
        sub.add_argument(
            "--merge-suffixes",
            nargs="+",
            metavar="SUFFIX",
            help="dense_sampling_<suffix>.json runs to fold into dense_sampling.json",
        )
        add_common_roots(sub)

    sub = subparsers.add_parser("chafee_latent_dimensions")
    sub.add_argument("--stage", choices=APPENDIX_STAGES, required=True)
    sub.add_argument("--dimension", type=int, choices=chafee_appendix.DIMENSIONS)
    sub.add_argument("--seed", type=int)
    sub.add_argument("--initials", type=int)
    sub.add_argument(
        "--output-root",
        type=Path,
        help=(
            "directory for appendix results "
            "(default: <repo>/output/sampled_residual_tolerance/chafee_latent_dimensions)"
        ),
    )
    return parser


def run_standard(args: argparse.Namespace) -> None:
    if args.stage == "merge":
        if args.experiment != "chafee_infante_current":
            raise SystemExit("--stage merge applies only to chafee_infante_current")
        if not args.merge_suffixes:
            raise SystemExit("--stage merge requires --merge-suffixes")
        results_dir = None
        if args.output_root is not None:
            results_dir = args.output_root / args.experiment
        print(merge_chafee_dense_runs(args.merge_suffixes, results_dir=results_dir))
        return
    if args.stage in ("tolerance", "all"):
        path = run_tolerance_evaluation(
            args.experiment,
            labels=args.label,
            local_boxes=args.local_boxes,
            sample_target=args.sample_target,
            sobol_scrambles=args.sobol_scrambles,
            output_root=args.output_root,
            blocks_root=args.blocks_root,
        )
        print(path)
    if args.stage in ("residual", "all"):
        path = run_dense_sampling(
            args.experiment,
            seed=args.seed,
            chafee_initials=args.chafee_initials,
            output_suffix=args.output_suffix,
            skip_fresh_trajectories=args.skip_fresh_trajectories,
            output_root=args.output_root,
            blocks_root=args.blocks_root,
        )
        print(path)


def run_appendix(args: argparse.Namespace) -> None:
    result_root = args.output_root
    if args.stage == "validate":
        chafee_appendix.validate_inputs()
    elif args.stage == "tolerance":
        if args.dimension is None:
            raise SystemExit("--stage tolerance requires --dimension")
        print(chafee_appendix.run_tolerance(args.dimension, result_root=result_root))
    elif args.stage == "stored":
        print(chafee_appendix.run_stored(result_root=result_root))
    elif args.stage == "fresh":
        if args.seed is None:
            raise SystemExit("--stage fresh requires --seed")
        expected = dict(chafee_appendix.FRESH_RUNS).get(args.seed)
        if expected is None:
            raise SystemExit(
                f"fresh seed must be one of "
                f"{[seed for seed, _ in chafee_appendix.FRESH_RUNS]}"
            )
        initials = args.initials if args.initials is not None else expected
        if initials != expected:
            raise SystemExit(
                f"fresh seed {args.seed} requires {expected} initials, not {initials}"
            )
        print(chafee_appendix.run_fresh(args.seed, initials, result_root=result_root))
    elif args.stage == "decoder":
        if args.seed not in chafee_appendix.DECODER_SEEDS:
            raise SystemExit(
                f"decoder seed must be one of {chafee_appendix.DECODER_SEEDS}"
            )
        print(chafee_appendix.run_decoder(args.seed, result_root=result_root))
    elif args.stage == "merge":
        print(chafee_appendix.merge_partials(result_root=result_root))
    else:
        raise AssertionError(args.stage)


def main() -> None:
    args = build_parser().parse_args()
    if args.experiment == "chafee_latent_dimensions":
        run_appendix(args)
    else:
        run_standard(args)


if __name__ == "__main__":
    main()
