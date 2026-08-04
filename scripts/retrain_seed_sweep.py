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

The validation seed is inherited unchanged from the packaged config. Thus the
five training datasets share one holdout set, making validation losses directly
comparable across both seed axes.

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

    # Leslie3D Example2 trajectory-length sweep (T=40), isolated from T=20
    python scripts/retrain_seed_sweep.py --example leslie3d_example2 \
        --trajectory-length 40 --box-map-backend adaptive_precomputed --tag t40

    # Leslie3D Example2: T=25 and N=50,000 total ICs (40,000/10,000 split)
    python scripts/retrain_seed_sweep.py --example leslie3d_example2 \
        --trajectory-length 25 --total-initial-conditions 50000 \
        --box-map-backend adaptive_precomputed --tag t25_n50000

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
BOX_MAP_BACKENDS = ("auto", "uniform_precomputed", "adaptive_precomputed")
BOUNDS_DATA_ROLES = ("train_and_validation_pairs", "train_pairs")
ADAPTIVE_PRECOMPUTE_SUBDIVS = ("init", "min", "max")
RENDER_FIGURE_GROUPS = frozenset({"morse", "roa", "overlay", "extras"})


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


def _parse_figure_set(raw: str | None) -> set[str] | None:
    """Parse an optional comma-separated render-group selection.

    ``None`` deliberately remains distinct from an empty set: the pipeline's
    longstanding ``None`` behavior renders every group, while an explicit
    selection lets long sweeps omit the expensive regions-of-attraction pass.
    """
    if raw is None:
        return None
    figures = {token.strip() for token in raw.split(",") if token.strip()}
    if not figures:
        raise ValueError("--figures must select at least one render group")
    unknown = figures - RENDER_FIGURE_GROUPS
    if unknown:
        raise ValueError(
            "unknown --figures group(s) "
            f"{sorted(unknown)}; choose from {sorted(RENDER_FIGURE_GROUPS)}"
        )
    return figures


def _sweep_label(config_name: str, tag: str | None) -> str:
    return f"{config_name}_seedsweep" + (f"_{tag}" if tag else "")


def _split_total_initial_conditions(
    cfg: ExperimentConfig, total_initial_conditions: int
) -> tuple[int, int]:
    """Allocate a requested total using the packaged train/validation ratio."""
    if total_initial_conditions < 2:
        raise ValueError("total_initial_conditions must be at least 2")
    if not isinstance(cfg.data.n_samples_train, int):
        raise ValueError("total_initial_conditions requires a scalar packaged data.n_samples_train")

    packaged_train = int(cfg.data.n_samples_train)
    packaged_validation = int(cfg.data.n_samples_val)
    packaged_total = packaged_train + packaged_validation
    train = round(total_initial_conditions * packaged_train / packaged_total)
    train = min(max(train, 1), total_initial_conditions - 1)
    return train, total_initial_conditions - train


def _data_size_summary(
    cfg: ExperimentConfig,
    *,
    requested_total_initial_conditions: int | None,
    dataset_count: int,
) -> dict:
    """Describe IC and retained one-step-pair counts without reading data files."""
    if not isinstance(cfg.data.n_samples_train, int):
        raise ValueError("seed-sweep data-size summaries require scalar training counts")

    train_initial = int(cfg.data.n_samples_train)
    validation_initial = int(cfg.data.n_samples_val)
    retained_steps = int(cfg.data.n_iterations) - int(cfg.data.skip)
    train_pairs = train_initial * retained_steps
    validation_pairs = validation_initial * retained_steps
    return {
        "requested_total_initial_conditions_per_dataset": requested_total_initial_conditions,
        "effective_initial_conditions_per_dataset": {
            "train": train_initial,
            "validation": validation_initial,
            "total": train_initial + validation_initial,
        },
        "trajectory": {
            "generated_steps": int(cfg.data.n_iterations),
            "discarded_steps": int(cfg.data.skip),
            "retained_steps": retained_steps,
        },
        "transition_pairs_per_dataset": {
            "train": train_pairs,
            "validation": validation_pairs,
            "total": train_pairs + validation_pairs,
        },
        "planned_dataset_count": dataset_count,
        "transition_pairs_across_dataset_trees": {
            "train": train_pairs * dataset_count,
            "validation": validation_pairs * dataset_count,
            "total": (train_pairs + validation_pairs) * dataset_count,
        },
    }


def _dataset_config(
    config_name: str,
    *,
    ic_seed: int,
    model_seeds: list[int],
    tag: str | None = None,
    cmgdb_subdiv: tuple[int, int, int] | None = None,
    box_map_backend: str | None = None,
    bounds_data_role: str | None = None,
    adaptive_precompute_subdiv: str | None = None,
    trajectory_length: int | None = None,
    total_initial_conditions: int | None = None,
    full_batch: bool = False,
    hidden_shapes: list[int] | None = None,
) -> ExperimentConfig:
    """Clone the packaged config for one initial-condition seed.

    Overrides the IC seed (``data.train_seed``), the model-seed list (``seeds``),
    and the two path roots. Optional overrides: ``cmgdb_subdiv`` sets the CMGDB
    (init, min, max) subdivision ladder; ``box_map_backend`` makes the CMGDB map
    evaluation strategy explicit; ``bounds_data_role`` controls whether the
    validation holdout participates in inferred CMGDB bounds;
    ``adaptive_precompute_subdiv`` selects the dense lookup depth before
    batched on-demand evaluation; ``trajectory_length`` sets the number of
    generated map steps; ``total_initial_conditions`` changes the combined
    training/validation IC count while preserving the packaged split ratio;
    ``full_batch`` sets ``batch_size`` to the full retained training-pair count;
    ``tag`` namespaces the sweep tree (so e.g. a full-batch, data-size, or
    trajectory-length sweep does not collide with the standard one).
    """
    cfg = load_config(config_name).model_copy(deep=True)
    label = _sweep_label(config_name, tag)
    dataset_dir = f"dataset_{ic_seed}"

    cfg.data.train_seed = ic_seed
    cfg.seeds = list(model_seeds)
    if cmgdb_subdiv is not None:
        cfg.cmgdb.subdiv_init, cfg.cmgdb.subdiv_min, cfg.cmgdb.subdiv_max = cmgdb_subdiv
    if box_map_backend is not None:
        if box_map_backend not in BOX_MAP_BACKENDS:
            raise ValueError(
                f"unknown box_map_backend {box_map_backend!r}; choose from {BOX_MAP_BACKENDS}"
            )
        cfg.cmgdb.box_map_backend = box_map_backend
    if bounds_data_role is not None:
        if bounds_data_role not in BOUNDS_DATA_ROLES:
            raise ValueError(
                f"unknown bounds_data_role {bounds_data_role!r}; choose from {BOUNDS_DATA_ROLES}"
            )
        cfg.cmgdb.bounds_data_role = bounds_data_role
    if adaptive_precompute_subdiv is not None:
        if adaptive_precompute_subdiv not in ADAPTIVE_PRECOMPUTE_SUBDIVS:
            raise ValueError(
                "unknown adaptive_precompute_subdiv "
                f"{adaptive_precompute_subdiv!r}; "
                f"choose from {ADAPTIVE_PRECOMPUTE_SUBDIVS}"
            )
        cfg.cmgdb.adaptive_precompute_subdiv = adaptive_precompute_subdiv
    if trajectory_length is not None:
        if trajectory_length <= cfg.data.skip:
            raise ValueError(
                "trajectory_length must be greater than data.skip "
                f"({cfg.data.skip}); got {trajectory_length}"
            )
        cfg.data.n_iterations = trajectory_length
    if total_initial_conditions is not None:
        n_train, n_validation = _split_total_initial_conditions(cfg, total_initial_conditions)
        cfg.data.n_samples_train = n_train
        cfg.data.n_samples_val = n_validation
    if full_batch:
        n_train = cfg.data.n_samples_train
        n_train = n_train if isinstance(n_train, int) else max(n_train)
        retained_steps = int(cfg.data.n_iterations) - int(cfg.data.skip)
        cfg.training.batch_size = int(n_train) * retained_steps
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


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _training_summary(output_dir: Path) -> dict:
    """Return the compact training fields needed by the sweep-level report."""

    path = output_dir / "training_summary.json"
    out = {
        "complete": False,
        "epochs_run": None,
        "best_epoch": None,
        "duration_minutes": None,
        "best_validation_total": None,
    }
    if not _nonempty(path):
        return out
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return out
    val_total = payload.get("val", {}).get("loss_total", {})
    out.update(
        {
            "complete": True,
            "epochs_run": payload.get("n_epochs_run"),
            "best_epoch": payload.get("best_epoch"),
            "duration_minutes": payload.get("train_duration_minutes"),
            "best_validation_total": val_total.get("best_epoch_value"),
        }
    )
    return out


def _artifact_summary(output_dir: Path) -> dict[str, bool]:
    """Presence checks for the four products requested by the replication."""

    model_dir = output_dir / "models"
    checkpoint = _nonempty(model_dir / "autoencoder.pt") or all(
        _nonempty(model_dir / name) for name in ("encoder.pt", "dynamics.pt", "decoder.pt")
    )
    return {
        "checkpoint": checkpoint,
        "training_summary": _nonempty(output_dir / "training_summary.json"),
        "morse_graph": _nonempty(output_dir / "MG" / "morse_graph"),
        "morse_sets": _nonempty(output_dir / "MG" / "morse_sets"),
    }


def _periodic_attractor_period(components: list[str]) -> int | None:
    """Return p exactly for ``(x^p-1, 0, 0)``, including ``x-1`` as p=1."""

    import re

    if len(components) != 3 or components[1:] != ["0", "0"]:
        return None
    match = re.fullmatch(r"x(?:\^([1-9]\d*))?-1", components[0].replace(" ", ""))
    if match is None:
        return None
    return int(match.group(1) or 1)


def _morse_node_summary(output_dir: Path) -> dict:
    """Count attractor-type Conley nodes vs graph sinks from the saved DOT.

    Attractor-type = Conley index nontrivial in homological degree 0 (the first
    Poincare-polynomial slot, e.g. ``x^6-1``, ``x-1``). Sinks = nodes with no
    outgoing edge. They differ when the adaptive method leaves a spurious
    ``minimal -> X`` edge, which the index-based count sees through.
    """
    import re

    dot = output_dir / "MG" / "morse_graph"
    out = {
        "n_nodes": None,
        "n_edges": None,
        "n_attractor_type": None,
        "n_periodic_attractor_nodes": None,
        "n_sinks": None,
        "attractor_indices": None,
        "sink_nodes": None,
        "bistability_pass": None,
    }
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
    attr = [c[0] for c in nodes.values() if c and c[0] not in ("0", "")]
    periodic = {
        node: period
        for node, components in nodes.items()
        if (period := _periodic_attractor_period(components)) is not None
    }
    sink_names = sorted((node for node in nodes if node not in srcs), key=lambda value: int(value))
    sink_nodes = [
        {
            "node": int(node),
            "index": f"({', '.join(nodes[node])})" if nodes[node] else None,
            "period": periodic.get(node),
        }
        for node in sink_names
    ]
    out["n_nodes"] = len(nodes)
    out["n_edges"] = len(edges)
    out["n_attractor_type"] = len(attr)
    out["n_periodic_attractor_nodes"] = len(periodic)
    out["attractor_indices"] = sorted(attr)
    out["n_sinks"] = len(sink_nodes)
    out["sink_nodes"] = sink_nodes
    out["bistability_pass"] = len(sink_nodes) == 2 and all(
        sink["period"] is not None for sink in sink_nodes
    )
    return out


def _outcome_summary(records: list[dict]) -> dict:
    classified = [r for r in records if r["bistability_pass"] is not None]
    passed = [r for r in classified if r["bistability_pass"]]
    return {
        "planned_cells": len(records),
        "training_complete": sum(r["training"]["complete"] for r in records),
        "morse_graphs_complete": sum(r["artifacts"]["morse_graph"] for r in records),
        "morse_sets_complete": sum(r["artifacts"]["morse_sets"] for r in records),
        "classified_cells": len(classified),
        "passed_cells": len(passed),
        "pass_rate_among_classified": (len(passed) / len(classified) if classified else None),
    }


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
                            REPO_ROOT / "output" / sweep_label / f"dataset_{ic}" / f"seed_{model}"
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
    parser.add_argument(
        "--figures",
        type=str,
        default=None,
        help=(
            "optional comma-separated render groups from "
            "morse,roa,overlay,extras; omitted preserves the pipeline default "
            "of rendering all groups"
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or mps")
    parser.add_argument(
        "--cmgdb-subdiv",
        type=str,
        default=None,
        help="override CMGDB subdivision as 'init,min,max' (e.g. '24,25,29')",
    )
    parser.add_argument(
        "--box-map-backend",
        choices=BOX_MAP_BACKENDS,
        default=None,
        help="override CMGDB box-map backend (use adaptive_precomputed to pin precomputation)",
    )
    parser.add_argument(
        "--bounds-data-role",
        choices=BOUNDS_DATA_ROLES,
        default=None,
        help=(
            "override which pairs define inferred latent CMGDB bounds; "
            "train_pairs keeps the validation holdout excluded"
        ),
    )
    parser.add_argument(
        "--adaptive-precompute-subdiv",
        choices=ADAPTIVE_PRECOMPUTE_SUBDIVS,
        default=None,
        help=(
            "dense lookup depth for adaptive_precomputed before batched on-demand corner evaluation"
        ),
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=None,
        help="override data.n_iterations (T); use --tag to isolate the output tree",
    )
    parser.add_argument(
        "--total-initial-conditions",
        type=int,
        default=None,
        help=(
            "override total training + validation initial conditions per dataset, "
            "preserving the packaged split ratio; use --tag to isolate outputs"
        ),
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
    try:
        figures = _parse_figure_set(args.figures)
    except ValueError as exc:
        parser.error(str(exc))
    cmgdb_subdiv = None
    if args.cmgdb_subdiv:
        parts = _parse_int_list(args.cmgdb_subdiv)
        if len(parts) != 3:
            raise SystemExit("--cmgdb-subdiv must be 'init,min,max' (3 ints)")
        cmgdb_subdiv = (parts[0], parts[1], parts[2])
    tag = args.tag

    if args.dry_run:
        plan = _plan(config_names, ic_seeds, effective_model_seeds, tag=tag)
        dry_configs = {
            name: _dataset_config(
                name,
                ic_seed=ic_seeds[0] if ic_seeds else load_config(name).data.train_seed,
                model_seeds=effective_model_seeds,
                tag=tag,
                cmgdb_subdiv=cmgdb_subdiv,
                box_map_backend=args.box_map_backend,
                bounds_data_role=args.bounds_data_role,
                adaptive_precompute_subdiv=args.adaptive_precompute_subdiv,
                trajectory_length=args.trajectory_length,
                total_initial_conditions=args.total_initial_conditions,
                full_batch=args.full_batch,
            )
            for name in config_names
        }
        print(
            json.dumps(
                {
                    "examples": config_names,
                    "ic_seeds": ic_seeds,
                    "model_seeds": effective_model_seeds,
                    "stages": stages,
                    "figures": sorted(figures) if figures is not None else None,
                    "tag": tag,
                    "cmgdb_subdiv": cmgdb_subdiv,
                    "box_map_backend": args.box_map_backend,
                    "bounds_data_role": args.bounds_data_role,
                    "adaptive_precompute_subdiv": args.adaptive_precompute_subdiv,
                    "trajectory_length": args.trajectory_length,
                    "total_initial_conditions": args.total_initial_conditions,
                    "full_batch": args.full_batch,
                    "shared_val_seeds": {
                        name: cfg.data.val_seed for name, cfg in dry_configs.items()
                    },
                    "data_sizes": {
                        name: _data_size_summary(
                            cfg,
                            requested_total_initial_conditions=(args.total_initial_conditions),
                            dataset_count=len(ic_seeds),
                        )
                        for name, cfg in dry_configs.items()
                    },
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
                box_map_backend=args.box_map_backend,
                bounds_data_role=args.bounds_data_role,
                adaptive_precompute_subdiv=args.adaptive_precompute_subdiv,
                trajectory_length=args.trajectory_length,
                total_initial_conditions=args.total_initial_conditions,
                full_batch=args.full_batch,
            )
            if verbose:
                print(
                    f"\n=== {label}  dataset_{ic} (train_seed={ic}, "
                    f"seeds={effective_model_seeds}, subdiv="
                    f"{(cfg.cmgdb.subdiv_init, cfg.cmgdb.subdiv_min, cfg.cmgdb.subdiv_max)}, "
                    f"backend={cfg.cmgdb.box_map_backend}, T={cfg.data.n_iterations}, "
                    f"bounds={cfg.cmgdb.bounds_data_role}, "
                    f"precompute={cfg.cmgdb.adaptive_precompute_subdiv}, "
                    f"N={int(cfg.data.n_samples_train) + cfg.data.n_samples_val} "
                    f"({cfg.data.n_samples_train}/{cfg.data.n_samples_val}), "
                    f"batch={cfg.training.batch_size}, val_seed={cfg.data.val_seed}) ===",
                    flush=True,
                )
            cell_results = pipeline.run(
                cfg,
                stages=stages,
                max_seeds=args.max_seeds,
                device=args.device,
                skip_completed=args.skip_completed,
                verbose=verbose,
                figures=figures,
            )
            for cell in cell_results:
                out_dir = Path(cell["output_dir"])
                node_summary = _morse_node_summary(out_dir)
                cell_record = {
                    "example": config_name,
                    "ic_seed": ic,
                    "val_seed": cfg.data.val_seed,
                    "model_seed": cell.get("seed"),
                    "output_dir": cell["output_dir"],
                    "n_attractors_sinks": _read_n_attractors(out_dir / "metrics.json"),
                    "n_attractor_type_nodes": node_summary["n_attractor_type"],
                    "n_periodic_attractor_nodes": node_summary["n_periodic_attractor_nodes"],
                    "n_morse_nodes": node_summary["n_nodes"],
                    "n_morse_edges": node_summary["n_edges"],
                    "n_sinks": node_summary["n_sinks"],
                    "attractor_indices": node_summary["attractor_indices"],
                    "sink_nodes": node_summary["sink_nodes"],
                    "bistability_pass": node_summary["bistability_pass"],
                    "training": _training_summary(out_dir),
                    "artifacts": _artifact_summary(out_dir),
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
                    "box_map_backend": args.box_map_backend,
                    "bounds_data_role": args.bounds_data_role,
                    "adaptive_precompute_subdiv": args.adaptive_precompute_subdiv,
                    "trajectory_length": args.trajectory_length,
                    "total_initial_conditions": args.total_initial_conditions,
                    "full_batch": args.full_batch,
                    "shared_val_seed": cfg.data.val_seed,
                    "ic_seeds": ic_seeds,
                    "model_seeds": effective_model_seeds,
                    "stages": stages,
                    "figures": sorted(figures) if figures is not None else None,
                    "pass_criterion": {
                        "name": "two_periodic_attractor_sinks",
                        "definition": (
                            "exactly two graph sinks and each full Conley index "
                            "matches (x^p-1, 0, 0) for an integer p >= 1"
                        ),
                        "periods_may_differ": True,
                    },
                    "data_size": _data_size_summary(
                        cfg,
                        requested_total_initial_conditions=(args.total_initial_conditions),
                        dataset_count=len(ic_seeds),
                    ),
                    "outcome": _outcome_summary(example_records),
                    "cells": example_records,
                },
                indent=2,
            )
        )

    if verbose and all_results:
        print("\n=== per-cell: periodic-attractor bistability ===")
        for r in all_results:
            print(
                f"  {r['example']:<26} ic={r['ic_seed']} model={r['model_seed']}  "
                f"attractor_type={r['n_attractor_type_nodes']} sinks={r['n_sinks']} "
                f"pass={r['bistability_pass']} sink_nodes={r['sink_nodes']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
