"""Retrain each paper example over a grid of initial-condition seeds x model seeds.

This reproduces, in the modern config-driven pipeline, the structure of Marcio's
``run_dataset_1 .. run_dataset_5`` Chafee-Infante study: each *dataset* is one
random seed for the trajectory initial conditions, and within each dataset the
autoencoder is trained several times from different weight initializations.

Two independent seed axes:

* ``--ic-seeds``     -> ``data.train_seed``: seeds the sampled trajectory initial
                       conditions, so each value yields a distinct training set
                       (Marcio's ``run_dataset_N``).
* ``--model-seeds``  -> ``seeds``: seeds weight init + batch shuffling, so each
                       value is an independent training of the same dataset
                       (Marcio's ``ci_model_weights_1/2/3``).

The default grid is 5 IC seeds x 3 model seeds = 15 runs per example.

Each (example, ic_seed) pair is run as one ``pipeline.run`` over an isolated tree::

    data/<example>_seedsweep/dataset_<ic>/{train.csv, val.csv, ...}
    output/<example>_seedsweep/dataset_<ic>/seed_<model>/{models, MG, metrics.json, ...}

Nothing under the original ``data/<example>`` or ``output/<example>`` paper trees
is touched. Every per-cell parameter other than the two seeds (architecture,
training, and CMGDB settings, including the bounding-box recipe) is inherited
verbatim from the packaged config so the sweep measures seed robustness of the
exact paper recipe.

Examples::

    # full 5x3 sweep for one example
    python scripts/retrain_seed_sweep.py --example chafee_infante

    # all three examples
    python scripts/retrain_seed_sweep.py --example all

    # smoke check: just dataset_1 / seed_0, all stages
    python scripts/retrain_seed_sweep.py --example chafee_infante \
        --max-datasets 1 --max-seeds 1

    # print the cell plan without running anything
    python scripts/retrain_seed_sweep.py --example all --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from latentdynamics.cli import pipeline
from latentdynamics.config import ExperimentConfig, load_config

# code/ : parent of this script's scripts/ directory. All sweep data/output is
# rooted here so the runner is insensitive to the current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# example key (and accepted aliases) -> packaged config name
EXAMPLES: dict[str, str] = {
    "leslie_2gen_contraction": "leslie_2gen_contraction",
    "leslie3d_example2": "leslie3d_example2",
    "chafee_infante": "chafee_infante",
}
ALIASES: dict[str, str] = {
    "leslie_contraction": "leslie_2gen_contraction",
    "contraction": "leslie_2gen_contraction",
    "leslie3d": "leslie3d_example2",
    "example2": "leslie3d_example2",
    "chafee": "chafee_infante",
    "chafee_infante_relu": "chafee_infante",
}

DEFAULT_IC_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_MODEL_SEEDS = [0, 1, 2]


def _resolve_example(name: str) -> str:
    key = name.strip()
    if key in EXAMPLES:
        return EXAMPLES[key]
    if key in ALIASES:
        return ALIASES[key]
    raise SystemExit(
        f"unknown example {name!r}; choose from {sorted(EXAMPLES)} "
        f"(aliases: {sorted(ALIASES)}) or 'all'"
    )


def _parse_int_list(raw: str) -> list[int]:
    return [int(tok) for tok in raw.split(",") if tok.strip() != ""]


def _sweep_label(config_name: str, tag: str | None) -> str:
    return f"{config_name}_seedsweep" + (f"_{tag}" if tag else "")


def _dataset_config(
    config_name: str,
    *,
    ic_seed: int,
    model_seeds: list[int],
    tag: str | None = None,
    cmgdb_subdiv: tuple[int, int, int] | None = None,
    full_batch: bool = False,
    hidden_shapes: list[int] | None = None,
) -> ExperimentConfig:
    """Clone the packaged config for one initial-condition seed.

    Overrides the IC seed (``data.train_seed``), the model-seed list (``seeds``),
    and the two path roots. Optional overrides: ``cmgdb_subdiv`` sets the CMGDB
    (init, min, max) subdivision ladder; ``full_batch`` sets ``batch_size`` to the
    full training-pair count; ``tag`` namespaces the sweep tree (so e.g. a
    full-batch sweep does not collide with the standard one).
    """
    cfg = load_config(config_name).model_copy(deep=True)
    label = _sweep_label(config_name, tag)
    dataset_dir = f"dataset_{ic_seed}"

    cfg.data.train_seed = ic_seed
    cfg.seeds = list(model_seeds)
    if cmgdb_subdiv is not None:
        cfg.cmgdb.subdiv_init, cfg.cmgdb.subdiv_min, cfg.cmgdb.subdiv_max = cmgdb_subdiv
    if full_batch:
        n_train = cfg.data.n_samples_train
        n_train = n_train if isinstance(n_train, int) else max(n_train)
        cfg.training.batch_size = int(n_train) * int(cfg.data.n_iterations)
    cfg.experiment_name = f"{label}_{dataset_dir}"
    cfg.paths.data_dir = REPO_ROOT / "data" / label / dataset_dir
    cfg.paths.output_dir = REPO_ROOT / "output" / label / dataset_dir
    # scaler defaults to output_dir/scalers; let the pipeline pin it per dataset.
    cfg.paths.scaler_dir_override = None
    return cfg


def _read_n_attractors(metrics_path: Path) -> int | None:
    if not metrics_path.is_file():
        return None
    try:
        data = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return None
    labels = data.get("minimal_morse_labels")
    return len(labels) if isinstance(labels, list) else None


def _morse_node_summary(output_dir: Path) -> dict:
    """Count attractor-type Conley nodes vs graph sinks from the saved DOT.

    Attractor-type = Conley index nontrivial in homological degree 0 (the first
    Poincare-polynomial slot, e.g. ``x^6-1``, ``x-1``). Sinks = nodes with no
    outgoing edge. They differ when the adaptive method leaves a spurious
    ``minimal -> X`` edge, which the index-based count sees through.
    """
    import re

    dot = output_dir / "MG" / "morse_graph"
    out = {"n_attractor_type": None, "n_sinks": None, "attractor_indices": None}
    if not dot.is_file() or dot.stat().st_size == 0:
        return out
    try:
        import pydot

        g = pydot.graph_from_dot_file(str(dot))[0]
    except Exception:
        return out
    nodes: dict[str, list[str]] = {}
    for n in g.get_nodes():
        nm = n.get_name().strip('"')
        if not nm.lstrip("-").isdigit():
            continue
        lbl = (n.get_label() or "").strip('"')
        m = re.search(r"\(([^)]*)\)", lbl)
        nodes[nm] = [s.strip() for s in m.group(1).split(",")] if m else []
    edges = [(e.get_source().strip('"'), e.get_destination().strip('"')) for e in g.get_edges()]
    srcs = {s for s, _ in edges}
    attr = [c[0] for n, c in nodes.items() if c and c[0] not in ("0", "")]
    out["n_attractor_type"] = len(attr)
    out["attractor_indices"] = sorted(attr)
    out["n_sinks"] = len([n for n in nodes if n not in srcs])
    return out


def _plan(
    config_names: list[str],
    ic_seeds: list[int],
    model_seeds: list[int],
    tag: str | None = None,
) -> list[dict]:
    cells: list[dict] = []
    for config_name in config_names:
        sweep_label = _sweep_label(config_name, tag)
        for ic in ic_seeds:
            for model in model_seeds:
                cells.append(
                    {
                        "example": config_name,
                        "ic_seed": ic,
                        "model_seed": model,
                        "output_dir": str(
                            REPO_ROOT
                            / "output"
                            / sweep_label
                            / f"dataset_{ic}"
                            / f"seed_{model}"
                        ),
                    }
                )
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--example",
        required=True,
        help=f"one of {sorted(EXAMPLES)} (aliases allowed) or 'all'",
    )
    parser.add_argument(
        "--ic-seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_IC_SEEDS),
        help="comma-separated data.train_seed values (initial-condition seeds)",
    )
    parser.add_argument(
        "--model-seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_MODEL_SEEDS),
        help="comma-separated weight-init seeds (trainings per dataset)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help=f"comma-separated subset of {list(pipeline.ALL_STAGES)} or 'all'",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or mps")
    parser.add_argument(
        "--cmgdb-subdiv",
        type=str,
        default=None,
        help="override CMGDB subdivision as 'init,min,max' (e.g. '24,25,29')",
    )
    parser.add_argument(
        "--full-batch",
        action="store_true",
        help="set batch_size to the full training-pair count (full-batch training)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="namespace the sweep tree: data|output/<example>_seedsweep_<tag>/",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="run only the first N initial-condition seeds (smoke checks)",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="cap the number of model seeds per dataset (smoke checks)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="skip stages whose expected artifacts already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the (example, ic_seed, model_seed) cell plan and exit",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.example.strip().lower() == "all":
        config_names = list(EXAMPLES.values())
    else:
        config_names = [_resolve_example(args.example)]

    ic_seeds = _parse_int_list(args.ic_seeds)
    model_seeds = _parse_int_list(args.model_seeds)
    if args.max_datasets is not None:
        ic_seeds = ic_seeds[: args.max_datasets]
    effective_model_seeds = (
        model_seeds[: args.max_seeds] if args.max_seeds is not None else model_seeds
    )

    stages = (
        list(pipeline.ALL_STAGES)
        if args.stages == "all"
        else [s for s in args.stages.split(",") if s]
    )
    cmgdb_subdiv = None
    if args.cmgdb_subdiv:
        parts = _parse_int_list(args.cmgdb_subdiv)
        if len(parts) != 3:
            raise SystemExit("--cmgdb-subdiv must be 'init,min,max' (3 ints)")
        cmgdb_subdiv = (parts[0], parts[1], parts[2])
    tag = args.tag

    if args.dry_run:
        plan = _plan(config_names, ic_seeds, effective_model_seeds, tag=tag)
        print(
            json.dumps(
                {
                    "examples": config_names,
                    "ic_seeds": ic_seeds,
                    "model_seeds": effective_model_seeds,
                    "stages": stages,
                    "tag": tag,
                    "cmgdb_subdiv": cmgdb_subdiv,
                    "full_batch": args.full_batch,
                    "n_cells": len(plan),
                    "cells": plan,
                },
                indent=2,
            )
        )
        return 0

    verbose = not args.quiet
    all_results: list[dict] = []
    for config_name in config_names:
        label = _sweep_label(config_name, tag)
        for ic in ic_seeds:
            cfg = _dataset_config(
                config_name,
                ic_seed=ic,
                model_seeds=model_seeds,
                tag=tag,
                cmgdb_subdiv=cmgdb_subdiv,
                full_batch=args.full_batch,
            )
            if verbose:
                print(
                    f"\n=== {label}  dataset_{ic} (train_seed={ic}, "
                    f"seeds={effective_model_seeds}, subdiv="
                    f"{(cfg.cmgdb.subdiv_init, cfg.cmgdb.subdiv_min, cfg.cmgdb.subdiv_max)}, "
                    f"batch={cfg.training.batch_size}) ===",
                    flush=True,
                )
            cell_results = pipeline.run(
                cfg,
                stages=stages,
                max_seeds=args.max_seeds,
                device=args.device,
                skip_completed=args.skip_completed,
                verbose=verbose,
            )
            for cell in cell_results:
                out_dir = Path(cell["output_dir"])
                node_summary = _morse_node_summary(out_dir)
                cell_record = {
                    "example": config_name,
                    "ic_seed": ic,
                    "model_seed": cell.get("seed"),
                    "output_dir": cell["output_dir"],
                    "n_attractors_sinks": _read_n_attractors(out_dir / "metrics.json"),
                    "n_attractor_type_nodes": node_summary["n_attractor_type"],
                    "n_sinks": node_summary["n_sinks"],
                    "attractor_indices": node_summary["attractor_indices"],
                }
                all_results.append(cell_record)

        # persist a per-example sweep summary alongside the dataset outputs
        summary_dir = REPO_ROOT / "output" / label
        summary_dir.mkdir(parents=True, exist_ok=True)
        example_records = [r for r in all_results if r["example"] == config_name]
        (summary_dir / "sweep_summary.json").write_text(
            json.dumps(
                {
                    "example": config_name,
                    "tag": tag,
                    "cmgdb_subdiv": cmgdb_subdiv,
                    "full_batch": args.full_batch,
                    "ic_seeds": ic_seeds,
                    "model_seeds": effective_model_seeds,
                    "stages": stages,
                    "cells": example_records,
                },
                indent=2,
            )
        )

    if verbose and all_results:
        print("\n=== per-cell: attractor-type Conley nodes (sinks) ===")
        for r in all_results:
            print(
                f"  {r['example']:<26} ic={r['ic_seed']} model={r['model_seed']}  "
                f"attractor_type={r['n_attractor_type_nodes']} sinks={r['n_sinks']} "
                f"indices={r['attractor_indices']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
