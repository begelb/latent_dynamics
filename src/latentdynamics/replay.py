"""Replay paper figures from saved artifacts, the easy way.

This is the notebook-facing entry point. A single call

    >>> from latentdynamics.replay import load_experiment
    >>> exp = load_experiment("leslie_2gen_contraction")
    >>> exp.show_morse_graph()
    >>> exp.show_morse_sets()

finds the experiment's config, loads its trained autoencoder, scaler, and the
CMGDB box bounds, and returns a :class:`ReplayExperiment` that re-renders the
paper's Morse graph and Morse sets from the saved ``MG/morse_graph`` (DOT) and
``MG/morse_sets`` (CSV) artifacts. No training and no CMGDB recompute are
invoked.

The goal is to make the replay notebooks as thin as CMGDB's own example
notebooks (``import CMGDB`` -> load -> plot): all path wrangling, seed/sweep
layout resolution, device selection, and checkpoint loading live here instead
of being copy-pasted into every notebook.

Path resolution reuses the pipeline's own helpers so a replayed experiment
points at exactly the same files the pipeline would produce. Relative config
paths are resolved against the package repo root (``code/``) so notebooks work
regardless of the current working directory.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from numpy.typing import NDArray

from .cli.pipeline import (
    _config_for_seed,
    _parse_mg_params_log,
    _resolve_device,
    _train_files_for,
)
from .config import load_config
from .config.schema import ArchConfig, ExperimentConfig
from .models.autoencoder import LatentDynamicsAutoencoder
from .training import load_any_checkpoint
from .viz import RenderedMorseFigures, plot_latent_trajectory, render_morse_from_files

# code/ -- the repo root the relative config paths are written against.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
# Where re-rendered PDFs/PNGs land (source trees are read-only).
DEFAULT_RENDER_ROOT = REPO_ROOT / "notebooks" / "rendered"


def _abs(path: str | Path) -> Path:
    """Resolve a (possibly relative) config path against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def repo_path(*parts: str | Path) -> Path:
    """Absolute path under the package repo root (``code/``).

    Convenience for notebooks that read data files directly, e.g.
    ``repo_path("data/coral/train_500.csv")``.
    """
    return REPO_ROOT.joinpath(*[str(part) for part in parts])


def show_image(path: str | Path, *, width: int | None = None) -> None:
    """Display an image inline under IPython; a no-op otherwise."""
    try:
        from IPython.display import Image, display
    except ImportError:
        return
    display(Image(filename=str(path), width=width))


def available_experiments() -> list[str]:
    """Config stems that :func:`load_experiment` accepts, e.g. ``leslie3d_example1``."""
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))


def resolve_config_path(config: str | Path) -> Path:
    """Map an experiment name (or a path) to its YAML config file."""
    p = Path(config)
    if p.suffix == ".yaml" and p.exists():
        return p
    candidate = CONFIGS_DIR / f"{Path(config).stem}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"no config for {config!r}; available: {', '.join(available_experiments())}"
    )


def _bounds_from_log(log_path: Path) -> tuple[list[float] | None, list[float] | None]:
    """Read the CMGDB lower/upper box bounds from a ``mg_params_log.txt``."""
    params = _parse_mg_params_log(log_path)
    if not params:
        return None, None

    def _parse(key: str) -> list[float] | None:
        raw = params.get(key)
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, list) else None

    return _parse("lower_bounds"), _parse("upper_bounds")


def _load_scaler(seed_cfg: ExperimentConfig, train_file: str) -> Any | None:
    """Load the joblib scaler for this cell, or ``None`` for raw-coordinate runs."""
    scaler_path = _abs(seed_cfg.paths.scaler_path(train_file))
    if not scaler_path.exists():
        return None
    # Archived scalers were pickled under an older scikit-learn; the version
    # mismatch is benign for a MinMaxScaler, so keep notebook output clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(scaler_path)


@dataclass(repr=False)
class ReplayExperiment:
    """A loaded experiment ready to re-render its paper figures.

    Attributes are populated by :func:`load_experiment`; notebooks normally only
    call the methods (:meth:`show_morse_graph`, :meth:`show_morse_sets`,
    :meth:`diagnostics`, and :meth:`encode` / :meth:`advance` for latent overlays).
    """

    name: str
    cfg: ExperimentConfig
    seed_cfg: ExperimentConfig
    seed: int | None
    train_file: str
    model: LatentDynamicsAutoencoder
    arch: ArchConfig
    scaler: Any | None
    device: torch.device
    _rendered: RenderedMorseFigures | None = field(default=None, repr=False)

    # -- locations -------------------------------------------------------- #
    @property
    def seed_dir(self) -> Path:
        """Directory holding this cell's ``models/``, ``MG/``, logs, metrics."""
        return _abs(self.seed_cfg.paths.output_dir)

    @property
    def morse_dir(self) -> Path:
        return self.seed_dir / "MG"

    @property
    def data_csv(self) -> Path:
        """The training CSV this run was built from (high/next-state columns)."""
        return _abs(self.cfg.paths.data_dir) / f"{self.train_file}.csv"

    def morse_bounds(self) -> tuple[list[float] | None, list[float] | None]:
        return _bounds_from_log(self.seed_dir / "mg_params_log.txt")

    # -- latent dynamics -------------------------------------------------- #
    @torch.no_grad()
    def encode(self, x_high: NDArray[np.floating] | Any) -> NDArray[np.float64]:
        """Map ambient points ``x`` (rows) into the latent space via the encoder."""
        x = np.asarray(x_high, dtype=np.float64)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return self.model.encoder(tensor).cpu().numpy()

    @torch.no_grad()
    def advance(self, z_low: NDArray[np.floating] | Any) -> NDArray[np.float64]:
        """Apply the latent map ``G`` once to latent points ``z`` (rows)."""
        tensor = torch.as_tensor(np.asarray(z_low), dtype=torch.float32, device=self.device)
        return self.model.latent_map(tensor).cpu().numpy()

    # -- Morse figures ---------------------------------------------------- #
    def render_morse(
        self, out_dir: str | Path | None = None, *, force: bool = False
    ) -> RenderedMorseFigures:
        """Re-render Morse graph + sets from the saved DOT/CSV (cached per call)."""
        if self._rendered is not None and not force:
            return self._rendered
        dot = self.morse_dir / "morse_graph"
        if not (dot.is_file() and dot.stat().st_size > 0):
            raise FileNotFoundError(
                f"{self.name}: no Morse artifacts at {self.morse_dir} "
                f"(the saved files may be 0-byte / not synced). "
                f"Pick a cell with non-empty artifacts, or recompute the run."
            )
        lower, upper = self.morse_bounds()
        out = Path(out_dir) if out_dir is not None else DEFAULT_RENDER_ROOT / self.name
        self._rendered = render_morse_from_files(
            self.morse_dir, bounds_lower=lower, bounds_upper=upper, out_dir=out
        )
        return self._rendered

    def show_morse_graph(self, out_dir: str | Path | None = None, *, width: int = 600) -> None:
        """Render and display the Morse-graph Hasse diagram inline.

        Displays only (like ``CMGDB.PlotMorseGraph``); use :meth:`render_morse`
        if you need the figure paths.
        """
        figs = self.render_morse(out_dir)
        show_image(figs.morse_graph_png, width=width)

    def show_morse_sets(self, out_dir: str | Path | None = None, *, width: int = 720) -> None:
        """Render and display the Morse sets (latent phase-space regions) inline."""
        figs = self.render_morse(out_dir)
        for png in (p for p in figs.morse_sets_paths if p.suffix == ".png"):
            show_image(png, width=width)

    def show_latent_trajectory(
        self,
        periodic_pts: dict[int, list[list[float]]],
        *,
        steps: int = 4,
        out_dir: str | Path | None = None,
        width: int = 720,
    ) -> None:
        """Overlay periodic-orbit latent trajectories on the (2D) Morse-set partition.

        ``periodic_pts`` maps a Morse label to ambient representative points
        (see ``latentdynamics.cli.render.LESLIE3D_PERIODIC_PTS``). Each orbit is
        encoded and pushed forward ``steps`` times under the latent map.
        """
        morse_csv = self.morse_dir / "morse_sets"
        morse_data = np.loadtxt(morse_csv, delimiter=",", ndmin=2)
        out = Path(out_dir) if out_dir is not None else DEFAULT_RENDER_ROOT / self.name
        out.mkdir(parents=True, exist_ok=True)
        path = plot_latent_trajectory(
            morse_set_data=morse_data,
            periodic_pts=periodic_pts,
            encode=self.encode,
            advance_latent=self.advance,
            save_path=out / "latent_trajectory.png",
            trajectory_steps=steps,
        )
        show_image(path, width=width)

    # -- provenance ------------------------------------------------------- #
    def diagnostics(self) -> dict[str, Any]:
        """Final losses, CMGDB parameters, and paper metrics saved with the run."""
        out: dict[str, Any] = {
            "experiment": self.name,
            "seed": self.seed,
            "train_file": self.train_file,
            "dims": f"{self.arch.high_dims} -> {self.arch.low_dims}",
            "seed_dir": str(self.seed_dir.relative_to(REPO_ROOT)),
        }
        losses = self.seed_dir / "final_losses.txt"
        if losses.is_file():
            out["final_losses"] = losses.read_text().strip()
        params = _parse_mg_params_log(self.seed_dir / "mg_params_log.txt")
        if params:
            out["mg_params"] = params
        metrics = self.seed_dir / "metrics.json"
        if metrics.is_file():
            text = metrics.read_text().strip()
            out["metrics"] = json.loads(text) if text else {}
        return out

    def __repr__(self) -> str:
        return (
            f"ReplayExperiment(name={self.name!r}, seed={self.seed}, "
            f"train_file={self.train_file!r}, dims={self.arch.high_dims}->{self.arch.low_dims}, "
            f"device={self.device.type!r})"
        )


def load_experiment(
    config: str | Path,
    *,
    seed: int | None = None,
    train_file: str | None = None,
    device: str | torch.device | None = None,
    output_dir: str | Path | None = None,
) -> ReplayExperiment:
    """Load a paper experiment for figure replay.

    Parameters
    ----------
    config:
        Experiment name (config stem, e.g. ``"leslie3d_example1"``) or a path to
        a YAML config.
    seed:
        Seed to replay. Defaults to the config's first seed, or the flat
        (no-``seed_k``) layout when the config declares no seeds.
    train_file:
        For sweep configs (e.g. coral), which ``train_<N>`` cell to load.
        Defaults to the first train file the config implies.
    device:
        Torch device override. Defaults to MPS > CUDA > CPU.
    output_dir:
        Override the config's artifact tree (and its scaler dir). Use when the
        run you want lives somewhere other than the config's ``output_dir`` --
        e.g. a local retrain under ``output/`` when the preserved tree is
        incomplete.
    """
    cfg_path = resolve_config_path(config)
    cfg = load_config(cfg_path)
    name = cfg.experiment_name or cfg_path.stem

    if output_dir is not None:
        cfg = cfg.model_copy(deep=True)
        cfg.paths.output_dir = Path(output_dir)
        cfg.paths.scaler_dir_override = None

    train_files = _train_files_for(cfg)
    if train_file is None:
        train_file = train_files[0]
    elif train_file not in train_files:
        raise ValueError(f"train_file {train_file!r} not in {train_files} for {name}")

    if seed is None:
        seed = cfg.seeds[0] if cfg.seeds else None

    seed_cfg = _config_for_seed(cfg, train_file=train_file, seed=seed)
    dev = _resolve_device(device)

    model_dir = _abs(seed_cfg.paths.model_dir)
    try:
        model, arch = load_any_checkpoint(model_dir, arch=cfg.arch, map_location=dev)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{name}: no usable checkpoint at {model_dir} "
            f"(artifacts may be 0-byte or not synced from the cluster). "
            f"Choose a cell that has non-empty checkpoints, or recompute. [{exc}]"
        ) from exc
    model.to(dev).eval()

    scaler = _load_scaler(seed_cfg, train_file)

    return ReplayExperiment(
        name=name,
        cfg=cfg,
        seed_cfg=seed_cfg,
        seed=seed,
        train_file=train_file,
        model=model,
        arch=arch,
        scaler=scaler,
        device=dev,
    )
