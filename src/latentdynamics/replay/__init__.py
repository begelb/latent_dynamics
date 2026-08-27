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

Two more verbs cover everything the notebooks need beyond replay:

    >>> exp.recompute_morse(subdiv=(10, 14, 20))   # CMGDB on the saved model
    >>> retrain("leslie3d_example1", overrides={"training": {"epochs": 300}})

Both write only under the playground tree (``output/notebooks/`` by default);
the preserved trees (``replay_sources/``, ``paper_figures/``) are never
touched.

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
import shutil
import time
import urllib.error
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from numpy.typing import NDArray

from .._paths import get_repo_root
from ..analysis.cmgdb_roa import compute_and_save_exact_roa
from ..analysis.morse import LatentBounds, compute_morse_graph
from ..cli.morse_graph import write_mg_params_log
from ..cli.pipeline import (
    _config_for_seed,
    _parse_mg_params_log,
    _resolve_device,
    _train_files_for,
)
from ..cli.pipeline import run as run_pipeline
from ..config import load_config
from ..config.loader import deep_merge
from ..config.schema import ArchConfig, CMGDBConfig, ExperimentConfig
from ..models.autoencoder import LatentDynamicsAutoencoder
from ..training import load_any_checkpoint
from ..viz import (
    RenderedMorseFigures,
    plot_latent_trajectory,
    render_morse_from_files,
    save_morse_graph_artifacts,
)
from .fetch import ArtifactsNotPublishedError, FetchError, fetch_artifacts
from .fetch import _normalize_experiment_name
from .fetch import fetch_bundle as fetch_bundle

# code/ -- the repo root the relative config paths are written against.
# On pip-install, this falls back to the cache dir.
REPO_ROOT = get_repo_root()
# Where re-rendered PDFs/PNGs land (source trees are read-only).
DEFAULT_RENDER_ROOT = REPO_ROOT / "notebooks" / "rendered"
# Where notebook experiments (Morse recomputes, retrains) write their artifacts.
DEFAULT_PLAYGROUND_ROOT = REPO_ROOT / "output" / "notebooks"
# Trees holding the preserved paper artifacts; nothing here may ever be written.
_PROTECTED_ROOTS = (REPO_ROOT / "replay_sources", REPO_ROOT / "paper_figures")


def _check_playground_dir(out_dir: Path) -> Path:
    resolved = out_dir.resolve()
    for protected in _PROTECTED_ROOTS:
        if resolved == protected or resolved.is_relative_to(protected):
            raise ValueError(
                f"refusing to write under {protected} (preserved paper artifacts; "
                f"fetched bundles are checksum-verified and kept read-only, so any "
                f"write would invalidate them); "
                f"use a directory under {DEFAULT_PLAYGROUND_ROOT} instead"
            )
    return out_dir


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
    from ..config.loader import _get_packaged_configs_dir

    packaged_dir = _get_packaged_configs_dir()
    return sorted(p.stem for p in packaged_dir.glob("*.yaml") if p.stem != "CONFIG_REFERENCE")


def resolve_config_path(config: str | Path) -> Path:
    """Map an experiment name (or a path) to its YAML config file."""
    from ..config.loader import _get_packaged_configs_dir

    p = Path(config)
    if p.suffix == ".yaml" and p.exists():
        return p
    packaged_dir = _get_packaged_configs_dir()
    candidate = packaged_dir / f"{Path(config).stem}.yaml"
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
    _render_key: tuple[Any, ...] | None = field(default=None, repr=False)

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

    def _default_render_dir(self) -> Path:
        """Where re-rendered figures land when no ``out_dir`` is given.

        Read-only/preserved runs render into ``notebooks/rendered/<name>/``;
        writable runs (playground recomputes, retrains) render next to their
        own artifacts under ``<seed_dir>/MG/``, matching the pipeline layout.
        """
        protected = any(
            self.seed_dir == root or self.seed_dir.is_relative_to(root)
            for root in _PROTECTED_ROOTS
        )
        if self.seed_cfg.paths.read_only or protected:
            return DEFAULT_RENDER_ROOT / self.name
        return self.morse_dir

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
        self,
        out_dir: str | Path | None = None,
        *,
        box_scale: float | dict[int, float] | str = "auto",
        box_scale_min_frac: float = 0.025,
        box_scale_max: float = 10.0,
        force: bool = False,
    ) -> RenderedMorseFigures:
        """Re-render Morse graph + sets from the saved DOT/CSV.

        ``box_scale`` inflates Morse-set boxes so tiny sets stay visible at
        paper figure size: a float (global factor), a ``{label: factor}`` dict,
        or ``"auto"`` (inflate only sets whose extent falls below
        ``box_scale_min_frac`` of the view span, capped at ``box_scale_max``).
        Results are cached per parameter combination; pass ``force=True`` to
        re-render regardless.
        """
        key = (
            None if out_dir is None else str(out_dir),
            repr(box_scale),
            box_scale_min_frac,
            box_scale_max,
        )
        if self._rendered is not None and self._render_key == key and not force:
            return self._rendered
        dot = self.morse_dir / "morse_graph"
        if not (dot.is_file() and dot.stat().st_size > 0):
            raise FileNotFoundError(
                f"{self.name}: no Morse artifacts at {self.morse_dir} "
                f"(the saved files may be 0-byte / not synced). "
                f"Pick a cell with non-empty artifacts, or recompute the run."
            )
        lower, upper = self.morse_bounds()
        out = Path(out_dir) if out_dir is not None else self._default_render_dir()
        self._rendered = render_morse_from_files(
            self.morse_dir,
            bounds_lower=lower,
            bounds_upper=upper,
            out_dir=out,
            box_scale=box_scale,
            box_scale_min_frac=box_scale_min_frac,
            box_scale_max=box_scale_max,
        )
        self._render_key = key
        return self._rendered

    def show_morse_graph(self, out_dir: str | Path | None = None, *, width: int = 600) -> None:
        """Render and display the Morse-graph Hasse diagram inline.

        Displays only (like ``CMGDB.PlotMorseGraph``); use :meth:`render_morse`
        if you need the figure paths.
        """
        figs = self.render_morse(out_dir)
        show_image(figs.morse_graph_png, width=width)

    def show_morse_sets(
        self,
        out_dir: str | Path | None = None,
        *,
        width: int = 720,
        box_scale: float | dict[int, float] | str = "auto",
        box_scale_min_frac: float = 0.025,
        box_scale_max: float = 10.0,
    ) -> None:
        """Render and display the Morse sets (latent phase-space regions) inline.

        See :meth:`render_morse` for the ``box_scale*`` knobs.
        """
        figs = self.render_morse(
            out_dir,
            box_scale=box_scale,
            box_scale_min_frac=box_scale_min_frac,
            box_scale_max=box_scale_max,
        )
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
        out = Path(out_dir) if out_dir is not None else self._default_render_dir()
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

    # -- CMGDB recompute --------------------------------------------------- #
    def recompute_morse(
        self,
        *,
        subdiv: tuple[int, int, int],
        cmgdb_overrides: Mapping[str, Any] | None = None,
        out_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        verbose: bool = True,
    ) -> ReplayExperiment:
        """Re-run CMGDB on this experiment's saved model at new subdivisions.

        No training happens: the loaded checkpoint and the replayed run's
        latent box bounds are reused, so the only thing that changes is the
        CMGDB grid schedule (plus anything in ``cmgdb_overrides``). Returns a
        new :class:`ReplayExperiment` pointed at the recomputed artifacts, so
        ``show_morse_graph`` / ``show_morse_sets`` / ``diagnostics`` work as
        usual. Artifacts land under ``output/notebooks/<name>/morse_<i>-<m>-<x>``;
        the preserved paper trees are never written.

        Small subdivisions like ``(10, 14, 20)`` give a fast qualitative
        preview. Coarse grids can merge nearby recurrent sets and change the
        Morse graph, so paper-quality results need the config's values.

        Parameters
        ----------
        subdiv:
            ``(subdiv_init, subdiv_min, subdiv_max)``, validated against the
            usual ordering constraint.
        cmgdb_overrides:
            Extra ``cmgdb`` fields merged on top and re-validated, e.g.
            ``{"padding": False}``, ``{"lower_bounds": [...], "upper_bounds":
            [...]}``, or ``{"compute_roa": True}``. ``compute_roa`` defaults
            to off here because the exact RoA costs far more than the Morse
            graph itself.
        out_dir:
            Override the playground directory (must not point into the
            preserved trees).
        device:
            Torch device override; moves this experiment's loaded model.
        """
        try:
            sub_init, sub_min, sub_max = (int(v) for v in subdiv)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"subdiv must be (subdiv_init, subdiv_min, subdiv_max); got {subdiv!r}"
            ) from exc

        raw = self.seed_cfg.cmgdb.model_dump()
        raw.update(
            subdiv_init=sub_init, subdiv_min=sub_min, subdiv_max=sub_max, compute_roa=False
        )
        if cmgdb_overrides:
            raw.update(dict(cmgdb_overrides))
        cmgdb_cfg = CMGDBConfig.model_validate(raw)
        if cmgdb_cfg.lower_bounds is not None and len(cmgdb_cfg.lower_bounds) != self.arch.low_dims:
            raise ValueError(
                f"cmgdb bounds have dim {len(cmgdb_cfg.lower_bounds)}; "
                f"latent dim is {self.arch.low_dims}"
            )

        label = f"morse_{sub_init}-{sub_min}-{sub_max}"
        if self.train_file != "train":
            label = f"{self.train_file}_{label}"
        if self.seed is not None:
            label = f"{label}_seed{self.seed}"
        out = _abs(out_dir) if out_dir is not None else DEFAULT_PLAYGROUND_ROOT / self.name / label
        _check_playground_dir(out)

        if cmgdb_cfg.lower_bounds is not None and cmgdb_cfg.upper_bounds is not None:
            lower, upper = cmgdb_cfg.lower_bounds, cmgdb_cfg.upper_bounds
            bounds_source = "config"
        else:
            lower, upper = self.morse_bounds()
            if lower is None or upper is None:
                raise FileNotFoundError(
                    f"{self.name}: no saved latent bounds at "
                    f"{self.seed_dir / 'mg_params_log.txt'}; pass explicit bounds via "
                    f"cmgdb_overrides={{'lower_bounds': [...], 'upper_bounds': [...]}}"
                )
            bounds_source = "replayed_run"
        bounds = LatentBounds(
            lower=np.asarray(lower, dtype=np.float64),
            upper=np.asarray(upper, dtype=np.float64),
        )

        dev = self.device if device is None else _resolve_device(device)
        if dev != self.device:
            self.model.to(dev)

        if verbose:
            print(
                f"CMGDB: subdiv=({sub_init},{sub_min},{sub_max}) "
                f"bounds[{bounds_source}] backend={cmgdb_cfg.box_map_backend}"
            )
        t0 = time.perf_counter()
        morse_graph, map_graph = compute_morse_graph(self.model, bounds, cmgdb_cfg, device=dev)
        duration_s = time.perf_counter() - t0

        dot_path, _csv_path = save_morse_graph_artifacts(morse_graph, out / "MG")
        if cmgdb_cfg.compute_roa:
            if self.arch.low_dims != 2:
                if verbose:
                    print(f"exact RoA: skipped (latent dim {self.arch.low_dims} != 2)")
            else:
                compute_and_save_exact_roa(
                    map_graph=map_graph,
                    cmgdb_morse_graph=morse_graph,
                    morse_graph_dot=dot_path,
                    out_dir=out / "MG",
                    lower_bounds=bounds.lower,
                    upper_bounds=bounds.upper,
                    max_vertices=cmgdb_cfg.roa_max_vertices,
                    collapse_to_lca=cmgdb_cfg.collapse_roa_to_lca,
                )
        write_mg_params_log(
            out,
            bounds=bounds,
            cmgdb_cfg=cmgdb_cfg,
            bounds_source=bounds_source,
            duration_s=duration_s,
        )
        if verbose:
            print(
                f"{morse_graph.num_vertices()} Morse sets in {duration_s:.1f}s -> "
                f"{out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}"
            )

        new_seed_cfg = self.seed_cfg.model_copy(deep=True)
        new_seed_cfg.paths.output_dir = out
        new_seed_cfg.paths.read_only = False
        new_seed_cfg.cmgdb = cmgdb_cfg
        return replace(self, seed_cfg=new_seed_cfg, device=dev, _rendered=None, _render_key=None)

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


def _seed_reference_models(name: str) -> bool:
    """Copy the tracked minimal checkpoint tree for *name* into replay_sources.

    ``artifacts/reference_models/<key>/`` mirrors the ``replay_sources/``
    layout and holds just the network weights, arch sidecar, and scaler --
    a few hundred kilobytes that ship with the repository so ``quick`` and
    ``morse`` recomputation works from a bare clone (Colab included) without
    the released artifact bundles. Existing files are never overwritten.
    Returns True if the experiment has a staged tree (whether or not any file
    needed copying).
    """
    root = get_repo_root()
    staged = root / "artifacts" / "reference_models" / _normalize_experiment_name(name)
    if not staged.is_dir():
        return False
    for source in staged.rglob("*"):
        if not source.is_file():
            continue
        target = root / "replay_sources" / source.relative_to(staged)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return True


def load_experiment(
    config: str | Path,
    *,
    seed: int | None = None,
    train_file: str | None = None,
    device: str | torch.device | None = None,
    output_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
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
    overrides:
        Nested config overrides merged before validation (see
        :func:`latentdynamics.config.load_config`). Safe for ``cmgdb`` fields;
        overriding ``arch`` breaks loading the saved checkpoint, and ``paths``
        overrides change where artifacts are looked up.
    """
    cfg_path = resolve_config_path(config)
    cfg = load_config(cfg_path, overrides=overrides)
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

    # If model directory doesn't exist, first materialize the minimal in-repo
    # checkpoints (weights + scaler, tracked under artifacts/reference_models/),
    # which are all quick/morse recomputation needs; fall back to the full
    # bundle fetch only if the experiment has no staged reference model.
    if not model_dir.exists() and _seed_reference_models(name):
        pass
    if not model_dir.exists():
        try:
            fetch_artifacts(name)
        except ValueError as e:
            # Unknown experiment name
            raise ValueError(f"{name}: {e}") from e
        except ArtifactsNotPublishedError:
            # The message already names the manual placement path.
            raise
        except FetchError as e:
            # Manifest problem or a bundle that failed verification
            raise RuntimeError(f"{name}: artifact fetch failed: {e}") from e
        except urllib.error.URLError as e:
            # Network error or download failed
            raise RuntimeError(
                f"{name}: failed to download artifacts: {e}"
            ) from e

    try:
        model, arch = load_any_checkpoint(
            model_dir,
            arch=cfg.arch,
            map_location=dev,
            legacy_root=REPO_ROOT / "legacy",
        )
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


def retrain(
    config: str | Path,
    *,
    seed: int | None = None,
    train_file: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    out_root: str | Path | None = None,
    stages: Iterable[str] | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> ReplayExperiment:
    """Run the training pipeline into an isolated playground directory.

    Loads ``config`` (with ``overrides`` merged and validated), redirects all
    outputs -- including freshly generated data and scalers -- to a timestamped
    directory under ``output/notebooks/<name>/``, runs the pipeline for one
    (train_file, seed) cell, and returns the resulting run as a
    :class:`ReplayExperiment`. The config's original artifact trees are never
    written.

    Parameters
    ----------
    config:
        Experiment name (config stem) or a path to a YAML config.
    seed:
        Training seed; defaults to the config's first seed.
    train_file:
        For sweep configs, which train-file cell to run; defaults to the first.
    overrides:
        Nested config overrides, e.g. ``{"training": {"epochs": 300},
        "cmgdb": {"subdiv_max": 20}}``. An explicit ``{"paths": {"data_dir":
        ...}}`` override reuses that data directory instead of regenerating
        data inside the run directory.
    out_root:
        Override the run directory (must not point into the preserved trees).
    stages:
        Pipeline stage subset (default: all of data/scale/train/diagnose/
        morse/render/metrics).
    device:
        Torch device override. Defaults to MPS > CUDA > CPU.
    """
    cfg_path = resolve_config_path(config)
    probe = load_config(cfg_path, overrides=overrides)
    name = probe.experiment_name or cfg_path.stem

    if out_root is not None:
        run_root = _abs(out_root)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_root = DEFAULT_PLAYGROUND_ROOT / name / f"retrain_{stamp}"
    _check_playground_dir(run_root)

    # Redirect every write into the run directory. Paths are made absolute so
    # the pipeline behaves the same regardless of the notebook's working dir.
    user_paths = dict((overrides or {}).get("paths") or {})
    path_overrides: dict[str, Any] = {
        "output_dir": str(run_root),
        "scaler_dir_override": None,
        "read_only": False,
    }
    if "data_dir" not in user_paths:
        path_overrides["data_dir"] = str(run_root / "data")
    eff_overrides = deep_merge(dict(overrides or {}), {"paths": path_overrides})
    cfg = load_config(cfg_path, overrides=eff_overrides)

    if seed is None:
        seed = cfg.seeds[0] if cfg.seeds else None
    cfg.seeds = [seed] if seed is not None else []

    train_files = _train_files_for(cfg)
    if train_file is None:
        train_file = train_files[0]
    elif train_file not in train_files:
        raise ValueError(f"train_file {train_file!r} not in {train_files} for {name}")
    cell_index = train_files.index(train_file) if len(train_files) > 1 else None

    if verbose:
        rel = run_root.relative_to(REPO_ROOT) if run_root.is_relative_to(REPO_ROOT) else run_root
        print(f"{name}: retraining into {rel} (seed={seed}, train_file={train_file!r})")
    run_pipeline(
        cfg,
        stages=stages,
        device=device,
        verbose=verbose,
        cell_index=cell_index,
    )

    return load_experiment(
        cfg_path,
        seed=seed,
        train_file=train_file,
        device=device,
        overrides=eff_overrides,
    )
