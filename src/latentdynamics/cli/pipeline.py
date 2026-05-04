"""Single config-driven pipeline orchestrator.

Stages:

============  =======================================================  ===========================
stage         what it does                                             persisted artefacts
============  =======================================================  ===========================
``data``      generate train/test trajectory CSVs                       ``data_dir/<train>.csv`` + metadata
``scale``     fit a MinMax scaler from a training CSV                   ``scaler_dir/<train>/scaler.gz``
``train``     train the autoencoder for one (config, seed)              ``models/autoencoder.{pt,json}``
``morse``     compute the Conley-Morse graph via CMGDB                  ``MG/morse_graph`` (DOT) + ``MG/morse_sets`` (CSV)
``render``    re-render plots from saved Morse artefacts                ``MG/{morse_graph,morse_sets}.{pdf,png}`` + experiment extras
``metrics``   compute paper-specific metrics from saved artefacts       ``metrics.json``
============  =======================================================  ===========================

Re-running ``render`` and ``metrics`` does **not** invoke CMGDB or training -
it reads the saved DOT / CSV / state_dict files only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..training import has_legacy_checkpoint, has_new_checkpoint

ALL_STAGES: tuple[str, ...] = ("data", "scale", "train", "morse", "render", "metrics")


@dataclass(frozen=True)
class PipelineCell:
    """One independently runnable experiment cell.

    A cell is the unit used by cluster arrays: one train-file basename, one
    seed (or ``None`` for legacy single-checkpoint layouts), and the output
    directory that receives model/Morse/render/metric artefacts.
    """

    index: int
    train_file: str
    seed: int | None
    output_dir: str


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Pick the best available accelerator: MPS > CUDA > CPU."""
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_subdir(seed: int) -> str:
    return f"seed_{seed}"


def _train_files_for(cfg: ExperimentConfig) -> list[str]:
    """Resolve the ``train_<N>``/``train`` basenames implied by the config."""
    if cfg.data.train_files is not None:
        return list(cfg.data.train_files)
    if isinstance(cfg.data.n_samples_train, list):
        return [f"train_{N}" for N in cfg.data.n_samples_train]
    return ["train"]


def _config_for_seed(cfg: ExperimentConfig, *, train_file: str | None, seed: int | None) -> ExperimentConfig:
    """Return a copy of cfg whose output_dir points at the seed (and size) subdir.

    Layout:
      single train file + single seed     -> ``output_dir/seed_<k>``
      multi train files + single seed     -> ``output_dir/<train_file>/seed_<k>``

    The scaler is shared across seeds, so we pin ``scaler_dir_override`` to the
    parent's resolved scaler dir before re-routing ``output_dir``.
    """
    new_cfg = cfg.model_copy(deep=True)
    if new_cfg.paths.scaler_dir_override is None:
        new_cfg.paths.scaler_dir_override = cfg.paths.scaler_dir
    parts: list[str] = []
    has_multi_train = (
        cfg.data.train_files is not None
        or isinstance(cfg.data.n_samples_train, list)
    )
    if has_multi_train and train_file is not None and train_file != "train":
        parts.append(train_file)
    if seed is not None:
        parts.append(_seed_subdir(seed))
    if parts:
        new_cfg.paths.output_dir = cfg.paths.output_dir.joinpath(*parts)
    return new_cfg


def _normalise_stages(stages: Iterable[str] | None) -> list[str]:
    if stages is None:
        return list(ALL_STAGES)
    requested = [s.strip().lower() for s in stages if s.strip()]
    unknown = [s for s in requested if s not in ALL_STAGES]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; valid: {list(ALL_STAGES)}")
    # preserve canonical order regardless of input order
    return [s for s in ALL_STAGES if s in requested]


def iter_cells(cfg: ExperimentConfig, *, max_seeds: int | None = None) -> list[PipelineCell]:
    """Enumerate the independent train-file/seed cells implied by ``cfg``."""
    train_files = _train_files_for(cfg)
    seeds: list[int | None] = (
        list(cfg.seeds[:max_seeds]) if max_seeds is not None else list(cfg.seeds)
    )
    if not seeds:
        seeds = [None]

    cells: list[PipelineCell] = []
    for train_file in train_files:
        for seed in seeds:
            seed_cfg = _config_for_seed(cfg, train_file=train_file, seed=seed)
            cells.append(
                PipelineCell(
                    index=len(cells),
                    train_file=train_file,
                    seed=seed,
                    output_dir=str(seed_cfg.paths.output_dir),
                )
            )
    return cells


def plan_cells(cfg: ExperimentConfig, *, max_seeds: int | None = None) -> list[dict]:
    """JSON-serialisable cell plan used by dry-runs and cluster submission."""
    return [asdict(cell) for cell in iter_cells(cfg, max_seeds=max_seeds)]


def _select_cells(
    cfg: ExperimentConfig,
    *,
    max_seeds: int | None,
    cell_index: int | None,
    expected_cells: int | None,
) -> list[PipelineCell]:
    cells = iter_cells(cfg, max_seeds=max_seeds)
    if expected_cells is not None and expected_cells != len(cells):
        raise ValueError(f"expected {expected_cells} cell(s), but config expands to {len(cells)}")
    if cell_index is None:
        return cells
    if cell_index < 0 or cell_index >= len(cells):
        raise IndexError(f"cell_index {cell_index} out of range for {len(cells)} cell(s)")
    return [cells[cell_index]]


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _data_complete(cfg: ExperimentConfig) -> bool:
    labels = [*_train_files_for(cfg), "test"]
    for label in labels:
        if not _nonempty_file(cfg.paths.data_dir / f"{label}.csv"):
            return False
        if not _nonempty_file(cfg.paths.data_dir / f"{label}_metadata.json"):
            return False
    return True


def _stage_complete(
    stage: str,
    cfg: ExperimentConfig,
    seed_cfg: ExperimentConfig,
    *,
    train_file: str,
) -> bool:
    """Best-effort artefact completeness check for resumable runs."""
    if stage == "data":
        return _data_complete(cfg)
    if stage == "scale":
        from .scale_data import scaler_is_current

        return scaler_is_current(cfg, train_file)
    if stage == "train":
        return has_new_checkpoint(seed_cfg.paths.model_dir) or has_legacy_checkpoint(seed_cfg.paths.model_dir)
    if stage == "morse":
        return (
            _nonempty_file(seed_cfg.paths.morse_dir / "morse_graph")
            and _nonempty_file(seed_cfg.paths.morse_dir / "morse_sets")
        )
    if stage == "render":
        return (
            _nonempty_file(seed_cfg.paths.morse_dir / "morse_graph.pdf")
            and _nonempty_file(seed_cfg.paths.morse_dir / "morse_graph.png")
            and _nonempty_file(seed_cfg.paths.morse_dir / "morse_sets.pdf")
            and _nonempty_file(seed_cfg.paths.morse_dir / "morse_sets.png")
        )
    if stage == "metrics":
        return _nonempty_file(seed_cfg.paths.output_dir / "metrics.json")
    raise ValueError(f"unknown stage {stage!r}")


def _skip_completed(
    stage: str,
    cfg: ExperimentConfig,
    seed_cfg: ExperimentConfig,
    *,
    train_file: str,
    enabled: bool,
    verbose: bool,
) -> bool:
    if not enabled:
        return False
    if not _stage_complete(stage, cfg, seed_cfg, train_file=train_file):
        return False
    if verbose:
        print(f"{stage}: skipped existing artefacts for {seed_cfg.paths.output_dir}")
    return True


def run_one(
    cfg: ExperimentConfig,
    *,
    stages: Iterable[str] | None = None,
    train_file: str = "train",
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
    skip_completed: bool = False,
) -> dict:
    """Run requested stages for one (config, train_file, seed) cell.

    The ``data`` stage always operates on the top-level ``cfg`` (data files
    are global). All other stages route to the seed-specific output subdir.
    """
    plan = _normalise_stages(stages)
    seed_cfg = _config_for_seed(cfg, train_file=train_file, seed=seed)
    dev = _resolve_device(device)

    summary: dict = {"stages": plan, "train_file": train_file, "seed": seed,
                     "output_dir": str(seed_cfg.paths.output_dir), "device": str(dev)}

    skipped: list[str] = []

    if "data" in plan and not _skip_completed(
        "data", cfg, seed_cfg, train_file=train_file, enabled=skip_completed, verbose=verbose
    ):
        from . import make_data as make_data_stage

        make_data_stage.run(cfg, verbose=verbose)
    elif "data" in plan:
        skipped.append("data")
    if "scale" in plan and not _skip_completed(
        "scale", cfg, seed_cfg, train_file=train_file, enabled=skip_completed, verbose=verbose
    ):
        from . import scale_data as scale_stage

        scale_stage.run(cfg, train_file, verbose=verbose)
    elif "scale" in plan:
        skipped.append("scale")
    if "train" in plan and not _skip_completed(
        "train", cfg, seed_cfg, train_file=train_file, enabled=skip_completed, verbose=verbose
    ):
        from . import train as train_stage

        train_stage.run(seed_cfg, train_file=train_file, seed=seed, device=dev, verbose=verbose)
    elif "train" in plan:
        skipped.append("train")
    if "morse" in plan and not _skip_completed(
        "morse", cfg, seed_cfg, train_file=train_file, enabled=skip_completed, verbose=verbose
    ):
        from . import morse_graph as morse_stage

        morse_stage.run(seed_cfg, train_file=train_file, device=dev, verbose=verbose)
    elif "morse" in plan:
        skipped.append("morse")
    if "render" in plan and not _skip_completed(
        "render", cfg, seed_cfg, train_file=train_file, enabled=skip_completed, verbose=verbose
    ):
        from .render import render_stage

        summary["render"] = render_stage(seed_cfg, train_file=train_file, verbose=verbose)
    elif "render" in plan:
        skipped.append("render")
    if "metrics" in plan and not _skip_completed(
        "metrics", cfg, seed_cfg, train_file=train_file, enabled=skip_completed, verbose=verbose
    ):
        from .metrics import metrics_stage

        summary["metrics"] = metrics_stage(seed_cfg, cfg, train_file=train_file, verbose=verbose)
    elif "metrics" in plan:
        skipped.append("metrics")

    if skipped:
        summary["skipped_stages"] = skipped
    from .provenance import write_run_manifest

    summary["manifest"] = str(
        write_run_manifest(
            seed_cfg,
            cfg,
            cell_summary=summary,
            stages=plan,
            train_file=train_file,
        )
    )

    return summary


def run(
    cfg: ExperimentConfig,
    *,
    stages: Iterable[str] | None = None,
    max_seeds: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
    cell_index: int | None = None,
    expected_cells: int | None = None,
    skip_completed: bool = False,
) -> list[dict]:
    """Run requested stages for every (train_file, seed) implied by ``cfg``."""
    plan = _normalise_stages(stages)
    cells = _select_cells(
        cfg,
        max_seeds=max_seeds,
        cell_index=cell_index,
        expected_cells=expected_cells,
    )
    dev = _resolve_device(device)
    results: list[dict] = []

    if "data" in plan and not _skip_completed(
        "data", cfg, _config_for_seed(cfg, train_file=None, seed=None),
        train_file="train", enabled=skip_completed, verbose=verbose
    ):
        from . import make_data as make_data_stage

        make_data_stage.run(cfg, verbose=verbose)

    scaled_train_files: set[str] = set()
    for cell in cells:
        train_file = cell.train_file
        seed = cell.seed
        seed_cfg = _config_for_seed(cfg, train_file=train_file, seed=seed)

        if "scale" in plan and train_file not in scaled_train_files:
            if _skip_completed(
                "scale", cfg, seed_cfg, train_file=train_file,
                enabled=skip_completed, verbose=verbose
            ):
                scaled_train_files.add(train_file)
            else:
                from . import scale_data as scale_stage

                scale_stage.run(cfg, train_file, verbose=verbose)
                scaled_train_files.add(train_file)

        cell_summary: dict = {
            "cell_index": cell.index,
            "train_file": train_file,
            "seed": seed,
            "device": str(dev),
            "output_dir": str(seed_cfg.paths.output_dir),
        }
        skipped: list[str] = []
        if "train" in plan and not _skip_completed(
            "train", cfg, seed_cfg, train_file=train_file,
            enabled=skip_completed, verbose=verbose
        ):
            from . import train as train_stage

            train_stage.run(seed_cfg, train_file=train_file, seed=seed, device=dev, verbose=verbose)
        elif "train" in plan:
            skipped.append("train")
        if "morse" in plan and not _skip_completed(
            "morse", cfg, seed_cfg, train_file=train_file,
            enabled=skip_completed, verbose=verbose
        ):
            from . import morse_graph as morse_stage

            morse_stage.run(seed_cfg, train_file=train_file, device=dev, verbose=verbose)
        elif "morse" in plan:
            skipped.append("morse")
        if "render" in plan and not _skip_completed(
            "render", cfg, seed_cfg, train_file=train_file,
            enabled=skip_completed, verbose=verbose
        ):
            from .render import render_stage

            cell_summary["render"] = render_stage(seed_cfg, train_file=train_file, verbose=verbose)
        elif "render" in plan:
            skipped.append("render")
        if "metrics" in plan and not _skip_completed(
            "metrics", cfg, seed_cfg, train_file=train_file,
            enabled=skip_completed, verbose=verbose
        ):
            from .metrics import metrics_stage

            cell_summary["metrics"] = metrics_stage(seed_cfg, cfg, train_file=train_file, verbose=verbose)
        elif "metrics" in plan:
            skipped.append("metrics")
        if skipped:
            cell_summary["skipped_stages"] = skipped
        from .provenance import write_run_manifest

        cell_summary["manifest"] = str(
            write_run_manifest(
                seed_cfg,
                cfg,
                cell_summary=cell_summary,
                stages=plan,
                train_file=train_file,
            )
        )
        results.append(cell_summary)

    return results
