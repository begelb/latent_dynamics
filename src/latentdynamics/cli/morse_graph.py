"""Compute the Conley-Morse graph of a trained latent dynamics map and persist it.

Only computation + serialization lives here; rendering is the ``render`` stage.
The two artifacts produced are:

- ``MG/morse_graph`` : graphviz DOT (via ``CMGDB.PlotMorseGraph(...).save``)
- ``MG/morse_sets``  : box CSV (via ``CMGDB.SaveMorseSets``)

plus ``mg_params_log.txt`` recording the latent bounds used by CMGDB.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..analysis.morse import LatentBounds, compute_morse_graph, infer_latent_bounds
from ..config import ExperimentConfig
from ..sampling import load_scaler
from ..training import load_checkpoint
from ..viz import save_morse_graph_artifacts


def _load_data_and_scale(cfg: ExperimentConfig, train_file: str) -> np.ndarray:
    train = np.loadtxt(cfg.paths.data_dir / f"{train_file}.csv", delimiter=",", skiprows=1)
    test = np.loadtxt(cfg.paths.data_dir / "test.csv", delimiter=",", skiprows=1)
    scaler = load_scaler(cfg.paths.scaler_path(train_file))
    high = cfg.arch.high_dims
    pieces = [
        scaler.transform(train[:, :high]),
        scaler.transform(test[:, :high]),
        scaler.transform(train[:, high:]),
        scaler.transform(test[:, high:]),
    ]
    return np.vstack(pieces)


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

    model, _arch = load_checkpoint(output_root / "models")
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

    all_scaled = _load_data_and_scale(cfg, train_file)
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
        bounds_source = "encoded_data"
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

    t0 = time.perf_counter()
    morse_graph, _map_graph = compute_morse_graph(model, bounds, cfg.cmgdb, device=device)
    duration_s = time.perf_counter() - t0

    dot_path, csv_path = save_morse_graph_artifacts(morse_graph, morse_dir)

    if verbose:
        print(f"morse graph DOT  -> {dot_path}")
        print(f"morse sets CSV   -> {csv_path}")
        print(f"computation took {duration_s / 60.0:.2f} min")

    log_path = output_root / "mg_params_log.txt"
    log_path.write_text(
        "\n".join(
            [
                f"Lower bounds: {bounds.lower.tolist()}",
                f"Upper bounds: {bounds.upper.tolist()}",
                f"subdiv_init: {cfg.cmgdb.subdiv_init}",
                f"subdiv_min: {cfg.cmgdb.subdiv_min}",
                f"subdiv_max: {cfg.cmgdb.subdiv_max}",
                f"subdiv_limit: {cfg.cmgdb.subdiv_limit}",
                f"padding: {cfg.cmgdb.padding}",
                f"box_map_backend: {cfg.cmgdb.box_map_backend}",
                f"bounds_source: {bounds_source}",
                f"duration_minutes: {duration_s / 60.0:.4f}",
            ]
        )
        + "\n"
    )
