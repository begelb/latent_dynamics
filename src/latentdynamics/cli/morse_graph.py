"""Compute the Conley-Morse graph of a trained latent dynamics map and persist it.

Only computation + serialization lives here; rendering is the ``render`` stage.
The two artifacts produced are:

- ``MG/morse_graph`` : graphviz DOT (via ``CMGDB.PlotMorseGraph(...).save``)
- ``MG/morse_sets``  : box CSV (via ``CMGDB.SaveMorseSets``)

plus ``mg_params_log.txt`` recording the latent bounds used by CMGDB.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import torch

from ..analysis.cmgdb_roa import compute_and_save_exact_roa
from ..analysis.morse import LatentBounds, compute_morse_graph, infer_latent_bounds
from ..config import ExperimentConfig
from ..sampling import load_scaler
from ..training import load_any_checkpoint
from ..viz import save_morse_graph_artifacts

BoundsDataRole = Literal["train_and_validation_pairs", "train_pairs"]


def write_mg_params_log(
    output_root,
    *,
    bounds: LatentBounds,
    cmgdb_cfg,
    bounds_source: str,
    bounds_data_role: BoundsDataRole | None = None,
    duration_s: float,
):
    """Write ``mg_params_log.txt`` recording the CMGDB parameters of a run.

    The format is parsed back by ``cli.pipeline._parse_mg_params_log`` (replay
    bounds, stage-completeness checks), so every producer of Morse artifacts
    must write it through this helper.
    """
    log_path = output_root / "mg_params_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"Lower bounds: {bounds.lower.tolist()}",
                f"Upper bounds: {bounds.upper.tolist()}",
                f"subdiv_init: {cmgdb_cfg.subdiv_init}",
                f"subdiv_min: {cmgdb_cfg.subdiv_min}",
                f"subdiv_max: {cmgdb_cfg.subdiv_max}",
                f"subdiv_limit: {cmgdb_cfg.subdiv_limit}",
                f"bounds_epsilon_frac: {cmgdb_cfg.bounds_epsilon_frac}",
                f"padding: {cmgdb_cfg.padding}",
                f"box_map_backend: {cmgdb_cfg.box_map_backend}",
                f"bounds_data_role: {bounds_data_role or cmgdb_cfg.bounds_data_role}",
                f"adaptive_precompute_subdiv: {cmgdb_cfg.adaptive_precompute_subdiv}",
                f"max_table_points: {cmgdb_cfg.max_table_points}",
                f"precompute_batch_points: {cmgdb_cfg.precompute_batch_points}",
                f"compute_roa: {cmgdb_cfg.compute_roa}",
                f"roa_max_vertices: {cmgdb_cfg.roa_max_vertices}",
                f"collapse_roa_to_lca: {cmgdb_cfg.collapse_roa_to_lca}",
                f"bounds_source: {bounds_source}",
                f"duration_minutes: {duration_s / 60.0:.4f}",
            ]
        )
        + "\n"
    )
    return log_path


def _load_data_and_scale(
    cfg: ExperimentConfig,
    train_file: str,
    *,
    bounds_data_role: BoundsDataRole = "train_and_validation_pairs",
) -> np.ndarray:
    """Load the scaled ambient pairs used to infer a latent CMGDB rectangle.

    The default preserves the historical pipeline behavior: current and next
    states from both the training and validation CSVs are encoded.  Callers
    that require a strict evaluation holdout may opt into ``"train_pairs"``
    to use only current and next states from the training CSV.
    """

    if bounds_data_role not in {"train_and_validation_pairs", "train_pairs"}:
        raise ValueError(f"unknown CMGDB bounds data role {bounds_data_role!r}")
    train = np.loadtxt(cfg.paths.data_dir / f"{train_file}.csv", delimiter=",", skiprows=1, ndmin=2)
    scaler = load_scaler(cfg.paths.scaler_path(train_file))
    high = cfg.arch.high_dims
    train_current = scaler.transform(train[:, :high])
    train_next = scaler.transform(train[:, high:])
    if bounds_data_role == "train_and_validation_pairs":
        val = np.loadtxt(cfg.paths.val_csv(), delimiter=",", skiprows=1, ndmin=2)
        return np.vstack(
            [
                train_current,
                scaler.transform(val[:, :high]),
                train_next,
                scaler.transform(val[:, high:]),
            ]
        )
    return np.vstack([train_current, train_next])


def _morse_artifacts_present(morse_dir) -> bool:
    """Return True if a non-empty Morse DOT or CSV already lives in ``morse_dir``."""
    dot = morse_dir / "morse_graph"
    csv = morse_dir / "morse_sets"
    if dot.is_file() and dot.stat().st_size > 0:
        return True
    if csv.is_file() and csv.stat().st_size > 0:
        return True
    return False


def run(
    cfg: ExperimentConfig,
    *,
    train_file: str = "train",
    bounds_data_role: BoundsDataRole | None = None,
    output_subdir: str | None = None,
    device: torch.device | str | None = None,
    verbose: bool = True,
    force_overwrite: bool = False,
) -> None:
    """Run the full Morse-graph pipeline for one trained model."""
    output_root = cfg.paths.output_dir
    if output_subdir is not None:
        output_root = output_root / output_subdir

    morse_dir = output_root / "MG"
    if _morse_artifacts_present(morse_dir) and not force_overwrite:
        raise RuntimeError(
            f"prior Morse artifacts present at {morse_dir} "
            f"(morse_graph DOT or morse_sets CSV is non-empty). Refusing to "
            f"overwrite a potentially expensive CMGDB run. "
            f"Pass --force-overwrite to proceed."
        )

    model, _arch = load_any_checkpoint(output_root / "models", arch=cfg.arch)
    if device is None:
        device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    model.to(device)

    if bounds_data_role is None:
        bounds_data_role = cfg.cmgdb.bounds_data_role

    all_scaled = _load_data_and_scale(
        cfg,
        train_file,
        bounds_data_role=bounds_data_role,
    )
    if cfg.cmgdb.lower_bounds is not None and cfg.cmgdb.upper_bounds is not None:
        bounds = LatentBounds(
            lower=np.asarray(cfg.cmgdb.lower_bounds, dtype=np.float64),
            upper=np.asarray(cfg.cmgdb.upper_bounds, dtype=np.float64),
        )
        bounds_source = "config"
    else:
        bounds = infer_latent_bounds(
            model.encoder, all_scaled, epsilon_frac=cfg.cmgdb.bounds_epsilon_frac, device=device
        )
        bounds_source = (
            "encoded_train_pairs" if bounds_data_role == "train_pairs" else "encoded_data"
        )
    if not (np.all(np.isfinite(bounds.lower)) and np.all(np.isfinite(bounds.upper))):
        raise ValueError(
            f"latent bounds contain NaN/Inf (lower={bounds.lower.tolist()}, "
            f"upper={bounds.upper.tolist()}); training likely diverged - "
            f"check {output_root / 'final_losses.txt'}"
        )
    if verbose:
        print(
            f"latent bounds ({bounds_source}): "
            f"lower={bounds.lower.tolist()} upper={bounds.upper.tolist()}"
        )
        print(
            f"CMGDB: subdiv_init={cfg.cmgdb.subdiv_init} "
            f"min={cfg.cmgdb.subdiv_min} max={cfg.cmgdb.subdiv_max} "
            f"padding={cfg.cmgdb.padding} backend={cfg.cmgdb.box_map_backend}"
        )

    # The returned MapGraph costs a full extra box-map pass over the phase
    # space. Exact RoA is the only consumer, and it is 2-D only (see below), so
    # ask for it under exactly the condition that it will be used.
    need_map_graph = bool(cfg.cmgdb.compute_roa and cfg.arch.low_dims == 2)

    t0 = time.perf_counter()
    morse_graph, map_graph = compute_morse_graph(
        model, bounds, cfg.cmgdb, device=device, need_map_graph=need_map_graph
    )
    duration_s = time.perf_counter() - t0

    dot_path, csv_path = save_morse_graph_artifacts(morse_graph, morse_dir)
    exact_roa_path = None
    if cfg.cmgdb.compute_roa:
        if cfg.arch.low_dims != 2:
            if verbose:
                print(f"exact RoA: skipped (latent dim {cfg.arch.low_dims} != 2)")
        else:
            exact_roa_path = compute_and_save_exact_roa(
                map_graph=map_graph,
                cmgdb_morse_graph=morse_graph,
                morse_graph_dot=dot_path,
                out_dir=morse_dir,
                lower_bounds=bounds.lower,
                upper_bounds=bounds.upper,
                max_vertices=cfg.cmgdb.roa_max_vertices,
                collapse_to_lca=cfg.cmgdb.collapse_roa_to_lca,
            )

    if verbose:
        print(f"morse graph DOT  -> {dot_path}")
        print(f"morse sets CSV   -> {csv_path}")
        if exact_roa_path is not None:
            print(f"exact RoA        -> {exact_roa_path}")
        print(f"computation took {duration_s / 60.0:.2f} min")

    write_mg_params_log(
        output_root,
        bounds=bounds,
        cmgdb_cfg=cfg.cmgdb,
        bounds_source=bounds_source,
        bounds_data_role=bounds_data_role,
        duration_s=duration_s,
    )
