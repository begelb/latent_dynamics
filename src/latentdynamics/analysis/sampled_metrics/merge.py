"""Merge independent Chafee--Infante d=2 fresh-trajectory searches.

Independent :func:`~.residual_protocol.run_dense_sampling` runs for
``chafee_infante_current`` (each with its own seed and ``output_suffix``)
are folded into the base ``dense_sampling.json`` without double-counting the
stored transitions, which every run evaluates.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Sequence

from .tolerance_protocol import default_output_root

FRESH_SOURCE = "fresh_sobol_trajectories_scale_1"
STORED_SOURCE = "replay_sources/chafee_infante/data/train.csv"


def seed_from_run(run: dict[str, object]) -> int:
    return int(
        run["sampling_protocol"]["fresh_trajectories"]["seed_base"]
    )


def relabel_source(node: dict[str, object], old: str, new: str) -> None:
    summaries = node["residual"]["source_summaries"]
    if old not in summaries:
        return
    summaries[new] = summaries.pop(old)
    witness = node["residual"]["witness"]
    if witness is not None and witness["source"] == old:
        witness["source"] = new


def merge_chafee_dense_runs(
    suffixes: Sequence[str],
    *,
    results_dir: Path | None = None,
) -> Path:
    """Merge ``dense_sampling_<suffix>.json`` runs into ``dense_sampling.json``.

    ``results_dir`` defaults to the ``chafee_infante_current`` directory under
    the default output root.  The merged file is rewritten in place and its
    path returned.
    """
    if results_dir is not None:
        results_dir = Path(results_dir)
    else:
        results_dir = default_output_root() / "chafee_infante_current"

    output_path = results_dir / "dense_sampling.json"
    merged = json.loads(output_path.read_text())
    base_seed = seed_from_run(merged)
    base_source = f"{FRESH_SOURCE}_seed_{base_seed}"
    for node in merged["nodes"].values():
        relabel_source(node, FRESH_SOURCE, base_source)

    ensemble = [
        copy.deepcopy(merged["sampling_protocol"]["fresh_trajectories"])
    ]
    supplemental_runs = []
    for suffix in suffixes:
        path = results_dir / f"dense_sampling_{suffix}.json"
        run = json.loads(path.read_text())
        seed = seed_from_run(run)
        fresh_protocol = copy.deepcopy(
            run["sampling_protocol"]["fresh_trajectories"]
        )
        if int(fresh_protocol["candidate_transitions"]) > 0:
            ensemble.append(fresh_protocol)
        supplemental_runs.append(
            {
                "suffix": suffix,
                "seed": seed,
                "fresh_trajectories": fresh_protocol,
                "decoder_guided_preimages": copy.deepcopy(
                    run["sampling_protocol"]["decoder_guided_preimages"]
                ),
            }
        )
        for label, node in merged["nodes"].items():
            other = run["nodes"][label]
            residual = node["residual"]
            supplemental = {
                source: copy.deepcopy(summary)
                for source, summary in other["residual"]["source_summaries"].items()
                if source != STORED_SOURCE
            }
            for source, summary in supplemental.items():
                run_source = f"{source}_seed_{seed}"
                residual["source_summaries"][run_source] = summary
                residual["evaluated_samples"] += int(
                    summary["evaluated_samples"]
                )
                residual["accepted_samples"] += int(
                    summary["accepted_samples"]
                )
            sources_with_values = [
                (float(summary["max_euclidean_residual"]), source)
                for source, summary in supplemental.items()
                if summary["max_euclidean_residual"] is not None
            ]
            if sources_with_values:
                other_max, other_source = max(sources_with_values)
            else:
                other_max, other_source = None, None
            if (
                other_max is not None
                and other_max > residual["sampled_maximum"]
            ):
                residual["sampled_maximum"] = other_max
                residual["squared_value_diagnostic"] = other_max**2
                witness = copy.deepcopy(other["residual"]["witness"])
                if witness["source"] == other_source:
                    witness["source"] = f"{other_source}_seed_{seed}"
                residual["witness"] = witness

            tau = node["tolerance"]["sampled_minimum"]
            ratio = residual["sampled_maximum"] / tau
            node["comparison"] = {
                "sampled_residual_over_sampled_tolerance": ratio,
                "sampled_conclusion": (
                    "sampled_violation"
                    if ratio >= 1.0
                    else "no_sampled_violation_found"
                ),
            }

    merged["sampling_protocol"]["fresh_trajectory_ensemble"] = ensemble
    merged["sampling_protocol"]["supplemental_runs"] = supplemental_runs
    merged["sampling_protocol"]["fresh_trajectory_ensemble_seeds"] = [
        item["seed_base"] for item in ensemble
    ]
    merged["sampling_protocol"]["fresh_trajectory_ensemble_candidate_transitions"] = sum(
        int(item["candidate_transitions"]) for item in ensemble
    )
    output_path.write_text(json.dumps(merged, indent=2))
    return output_path
