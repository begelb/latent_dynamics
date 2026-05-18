"""Render figures from on-disk artifacts; never invokes CMGDB.

The new pipeline writes ``MG/morse_graph`` (graphviz DOT) and ``MG/morse_sets``
(box CSV) during the ``morse`` stage. The render stage reads those files and
re-emits PDF/PNG plots via the unified palette in :mod:`latentdynamics.viz`.

Experiment-specific extras (e.g. the latent-trajectory plot for the Leslie 3D
spurious case) are dispatched on ``cfg.system.name`` via :func:`render_extras`.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from ..analysis.regions_of_attraction import MorseGraph
from ..config import ExperimentConfig
from ..training import has_legacy_checkpoint, has_new_checkpoint, load_any_checkpoint
from ..viz import render_morse_from_files
from ..viz.regions_of_attraction import render_cell_graph_roa


def _morse_dir_for(cfg: ExperimentConfig) -> Path:
    return cfg.paths.morse_dir


def _bounds_from_log(log_path: Path) -> tuple[list[float] | None, list[float] | None]:
    if not log_path.exists():
        return None, None
    lower: list[float] | None = None
    upper: list[float] | None = None
    for line in log_path.read_text().splitlines():
        if line.lower().startswith("lower bounds:"):
            lower = _parse_bounds(line)
        elif line.lower().startswith("upper bounds:"):
            upper = _parse_bounds(line)
    return lower, upper


def _parse_bounds(line: str) -> list[float]:
    payload = line.split(":", 1)[1].strip()
    payload = payload.strip("[]() ")
    return [float(x.strip()) for x in payload.split(",") if x.strip()]


def render_stage(
    cfg: ExperimentConfig,
    *,
    train_file: str = "train",
    verbose: bool = True,
    out_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Re-render Morse plots from saved DOT/CSV plus any system-specific figures.

    Reads source artifacts from ``cfg.paths.morse_dir`` and
    ``cfg.paths.output_dir``. When ``out_dir`` is provided, all rendered
    figures (Morse PDFs/PNGs and system-specific extras) are written under
    ``out_dir`` (``out_dir/MG/...`` and ``out_dir/figures/...``); when it is
    ``None``, output defaults to ``cfg.paths.output_dir``, matching the
    pre-replay-routing behavior.

    Returns ``{"skipped": <reason>}`` when the saved Morse artifacts are missing
    or empty (e.g. partial uploads), so a multi-seed sweep can keep going.
    """
    morse_dir = _morse_dir_for(cfg)
    dot_path = morse_dir / "morse_graph"
    csv_path = morse_dir / "morse_sets"
    if not dot_path.exists() or not csv_path.exists():
        if verbose:
            print(f"render: skipped {morse_dir} (missing morse_graph or morse_sets)")
        return {"skipped": f"missing morse_graph or morse_sets in {morse_dir}"}
    if dot_path.stat().st_size == 0 or csv_path.stat().st_size == 0:
        if verbose:
            print(f"render: skipped {morse_dir} (empty morse_graph or morse_sets)")
        return {"skipped": f"empty morse_graph or morse_sets in {morse_dir}"}

    bounds_lower, bounds_upper = _bounds_from_log(cfg.paths.output_dir / "mg_params_log.txt")

    write_root = Path(out_dir) if out_dir is not None else cfg.paths.output_dir
    figures = render_morse_from_files(
        morse_dir,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        out_dir=write_root / "MG",
    )
    rendered = [
        str(figures.morse_graph_pdf),
        str(figures.morse_graph_png),
        *map(str, figures.morse_sets_paths),
    ]

    roa = _render_roa_overlay(cfg, dot_path, csv_path, out_dir=write_root, verbose=verbose)
    if roa is not None:
        rendered.append(str(roa))

    extras = render_extras(cfg, train_file=train_file, verbose=verbose, out_dir=write_root)
    rendered.extend(extras)

    if verbose:
        print(f"render: {len(rendered)} file(s) under {write_root}")
    return {"figures": rendered}


def _render_roa_overlay(
    cfg: ExperimentConfig,
    dot_path: Path,
    csv_path: Path,
    *,
    out_dir: Path,
    verbose: bool,
) -> Path | None:
    """Cell-graph regions-of-attraction overlay, written to ``out_dir/MG/``.

    Currently 2D-only (the cell-graph backend raises for d!=2); silently skips
    if the latent map is higher-dimensional or the checkpoint is missing.
    """
    if cfg.arch.low_dims != 2:
        if verbose:
            print(f"render: skipping RoA overlay (latent dim {cfg.arch.low_dims} != 2)")
        return None
    model_dir = cfg.paths.output_dir / "models"
    if not (has_legacy_checkpoint(model_dir) or has_new_checkpoint(model_dir)):
        if verbose:
            print(f"render: skipping RoA overlay (no checkpoint at {model_dir})")
        return None
    model, _arch = load_any_checkpoint(model_dir, arch=cfg.arch)
    device = torch.device("cpu")
    model.to(device).eval()

    n_min = len(MorseGraph.from_dot(dot_path).minimal)
    title = (
        f"{cfg.system.name} — regions of attraction "
        f"({n_min} minimal Morse set{'s' if n_min != 1 else ''})"
    )
    out_path = Path(out_dir) / "MG" / "regions_of_attraction.png"
    return render_cell_graph_roa(
        dot_path, csv_path, model.latent_map, out_path, device=str(device), title=title
    )


def render_extras(
    cfg: ExperimentConfig,
    *,
    train_file: str = "train",
    verbose: bool = True,
    out_dir: Path | None = None,
) -> list[str]:
    """System-specific render hooks; safe no-op for systems without extras."""
    name = cfg.system.name
    if name == "leslie3d":
        return _render_leslie3d_extras(
            cfg,
            train_file=train_file,
            verbose=verbose,
            out_dir=out_dir,
        )
    return []


LESLIE3D_PERIODIC_PTS: dict[int, list[list[float]]] = {
    0: [
        [102.59382834, 4.62509476, 0.59276684],
        [6.47696572e-02, 7.18156798e01, 3.23756633e00],
        [1.20972812e00, 4.53387600e-02, 5.02709759e01],
        [6.60727793, 0.84680968, 0.03173713],
    ],
    1: [
        [20.09019989, 2.26201326, 21.10982997],
        [14.41254064, 14.06313992, 1.58340928],
        [43.08128567, 10.08877845, 9.84419795],
        [3.23144751, 30.15689997, 7.06214491],
    ],
}


def _make_encode_callable(encoder: torch.nn.Module, scaler, device: torch.device):
    @torch.no_grad()
    def _encode(points: np.ndarray) -> np.ndarray:
        scaled = scaler.transform(points)
        x = torch.as_tensor(scaled, dtype=torch.float32, device=device)
        return encoder(x).cpu().numpy()

    return _encode


def _make_advance_callable(latent_map: torch.nn.Module, device: torch.device):
    @torch.no_grad()
    def _advance(z: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(z, dtype=torch.float32, device=device)
        return latent_map(x).cpu().numpy()

    return _advance


def _render_leslie3d_extras(
    cfg: ExperimentConfig,
    *,
    train_file: str,
    verbose: bool,
    out_dir: Path | None = None,
) -> list[str]:
    """Render the latent-trajectory overlay (paper Fig. 1.214) from saved artifacts."""
    from ..viz import plot_latent_trajectory

    morse_dir = _morse_dir_for(cfg)
    morse_sets_path = morse_dir / "morse_sets"
    if not morse_sets_path.exists():
        return []

    model_dir = cfg.paths.output_dir / "models"
    if not (has_legacy_checkpoint(model_dir) or has_new_checkpoint(model_dir)):
        if verbose:
            print(f"  no checkpoint at {model_dir}; skipping latent trajectory")
        return []

    model, _arch = load_any_checkpoint(model_dir, arch=cfg.arch)
    device = torch.device("cpu")
    model.to(device)

    scaler_path = cfg.paths.scaler_path(train_file)
    if not scaler_path.exists():
        if verbose:
            print(f"  no scaler at {scaler_path}; skipping latent trajectory")
        return []
    scaler = joblib.load(scaler_path)

    morse_set_data = np.loadtxt(morse_sets_path, delimiter=",", ndmin=2)
    encode = _make_encode_callable(model.encoder, scaler, device)
    advance = _make_advance_callable(model.latent_map, device)

    figures_root = Path(out_dir) / "figures" if out_dir is not None else cfg.paths.figures_dir
    save_path = figures_root / "latent_trajectory.png"
    plot_latent_trajectory(
        morse_set_data=morse_set_data,
        periodic_pts=LESLIE3D_PERIODIC_PTS,
        encode=encode,
        advance_latent=advance,
        save_path=save_path,
        trajectory_steps=4,
    )
    return [str(save_path)]
