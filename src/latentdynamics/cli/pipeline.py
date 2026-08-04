"""Single config-driven pipeline orchestrator.

Stages:

============  =======================================================  ===========================
stage         what it does                                             persisted artifacts
============  =======================================================  ===========================
``data``      generate train/test trajectory CSVs                       ``data_dir/<train>.csv`` + metadata
``scale``     fit a MinMax scaler from a training CSV                   ``scaler_dir/<train>/scaler.gz``
``train``     train the autoencoder for one (config, seed)              ``models/autoencoder.{pt,json}``
``morse``     compute the Conley-Morse graph via CMGDB                  ``MG/morse_graph`` (DOT) + ``MG/morse_sets`` (CSV)
``render``    re-render plots from saved Morse artifacts                ``MG/{morse_graph,morse_sets}.{pdf,png}`` + experiment extras
``metrics``   compute paper-specific metrics from saved artifacts       ``metrics.json``
============  =======================================================  ===========================

Re-running ``render`` and ``metrics`` does **not** invoke CMGDB or training -
it reads the saved DOT / CSV / state_dict files only.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..training import has_legacy_checkpoint, has_new_checkpoint

ALL_STAGES: tuple[str, ...] = (
    "data",
    "scale",
    "train",
    "diagnose",
    "morse",
    "render",
    "metrics",
)

# Stages that produce or could overwrite the irreplaceable paper artifacts
# (datasets, scalers, trained checkpoints, CMGDB DOT/CSV). ``render``,
# ``metrics``, and ``diagnose`` are derived from these and may run on
# read-only configs - their writes get redirected to the replay tree by
# ``_derived_output_dir`` rather than blocked.
WRITE_STAGES: frozenset[str] = frozenset({"data", "scale", "train", "morse"})

# Default replay root (relative to cwd). Used when ``cfg.paths.read_only`` is
# true and the caller did not supply ``replay_root``.
DEFAULT_REPLAY_ROOT: Path = Path("replay")


def _check_read_only(cfg: ExperimentConfig, plan: list[str], *, force_overwrite: bool) -> None:
    if cfg.paths.scaler_read_only and "scale" in plan:
        raise RuntimeError(
            "config sets paths.scaler_read_only=true; refusing the scale stage because "
            f"it would overwrite the protected scaler at {cfg.paths.scaler_path('train')}. "
            "Omit 'scale' from --stages."
        )
    if not cfg.paths.read_only or force_overwrite:
        return
    blocked = [s for s in plan if s in WRITE_STAGES]
    if blocked:
        raise RuntimeError(
            f"config sets paths.read_only=true; refusing to run write-stages "
            f"{blocked} on {cfg.paths.output_dir}. "
            f"Pass --force-overwrite to override (and consider whether you "
            f"really want to clobber preserved paper artifacts)."
        )


def _resolve_replay_root(
    cfg: ExperimentConfig,
    replay_root: Path | str | None,
    *,
    force_overwrite: bool = False,
) -> Path | None:
    """Return the replay root to use, or None if redirection is disabled.

    Redirection is active only when ``cfg.paths.read_only`` is true. If the
    caller supplied ``replay_root`` (either programmatically or via CLI), use
    it. Otherwise fall back to :data:`DEFAULT_REPLAY_ROOT`. Passing
    ``force_overwrite`` disables redirection so all stages write to the source
    tree intentionally.
    """
    if force_overwrite:
        return None
    if not cfg.paths.read_only:
        return None
    if replay_root is None:
        return DEFAULT_REPLAY_ROOT
    return Path(replay_root)


def _derived_output_dir(
    cfg: ExperimentConfig,
    seed_cfg: ExperimentConfig,
    *,
    replay_root: Path | None,
) -> Path:
    """Where derived stages (render/metrics/diagnose/manifest) write.

    For non-read-only configs, or when ``replay_root`` is None, this returns
    ``seed_cfg.paths.output_dir`` - the source location, matching the
    pre-replay-routing behavior.

    For read-only configs with a replay root, this returns
    ``replay_root / cfg.experiment_name / <rel>`` where ``<rel>`` is the
    relative path from ``cfg.paths.output_dir`` to ``seed_cfg.paths.output_dir``
    so multi-cell substructure (``train_file/seed_K``) is mirrored under the
    replay tree without touching source artifacts.
    """
    if replay_root is None:
        return seed_cfg.paths.output_dir
    rel = seed_cfg.paths.output_dir.relative_to(cfg.paths.output_dir)
    name = cfg.experiment_name or "unnamed"
    return Path(replay_root) / name / rel


@dataclass(frozen=True)
class PipelineCell:
    """One independently runnable experiment cell.

    A cell is the unit used by cluster arrays: one train-file basename, one
    seed (or ``None`` for legacy single-checkpoint layouts), and the output
    directory that receives model/Morse/render/metric artifacts.
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


def _config_for_seed(
    cfg: ExperimentConfig, *, train_file: str | None, seed: int | None
) -> ExperimentConfig:
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
    has_multi_train = cfg.data.train_files is not None or isinstance(cfg.data.n_samples_train, list)
    if has_multi_train and train_file is not None and train_file != "train":
        parts.append(train_file)
    if seed is not None:
        parts.append(_seed_subdir(seed))
    if parts:
        new_cfg.paths.output_dir = cfg.paths.output_dir.joinpath(*parts)
    return new_cfg


def _normalize_stages(stages: Iterable[str] | None) -> list[str]:
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
    """JSON-serializable cell plan used by dry-runs and cluster submission."""
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


def _parse_mg_params_log(path: Path) -> dict[str, str] | None:
    if not _nonempty_file(path):
        return None
    params: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        params[normalized_key] = value.strip()
    return params


def _log_int(params: dict[str, str], key: str) -> int | None:
    try:
        return int(params[key])
    except (KeyError, TypeError, ValueError):
        return None


def _log_float(params: dict[str, str], key: str) -> float | None:
    try:
        return float(params[key])
    except (KeyError, TypeError, ValueError):
        return None


def _log_bool(params: dict[str, str], key: str) -> bool | None:
    raw = params.get(key)
    if raw is None:
        return None
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    return None


def _log_float_list(params: dict[str, str], key: str) -> list[float] | None:
    raw = params.get(key)
    if raw is None:
        return None
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    try:
        values = [float(value) for value in parsed]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def _float_lists_close(left: list[float], right: list[float]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(left, right, strict=True)
    )


def _morse_params_log_matches_config(cfg: ExperimentConfig, output_dir: Path) -> bool:
    params = _parse_mg_params_log(output_dir / "mg_params_log.txt")
    if params is None:
        return False

    expected_ints = {
        "subdiv_init": cfg.cmgdb.subdiv_init,
        "subdiv_min": cfg.cmgdb.subdiv_min,
        "subdiv_max": cfg.cmgdb.subdiv_max,
        "subdiv_limit": cfg.cmgdb.subdiv_limit,
    }
    for key, expected in expected_ints.items():
        if _log_int(params, key) != int(expected):
            return False

    if _log_bool(params, "padding") != bool(cfg.cmgdb.padding):
        return False
    if params.get("box_map_backend") != cfg.cmgdb.box_map_backend:
        return False
    logged_bounds_role = params.get("bounds_data_role")
    if logged_bounds_role is not None and logged_bounds_role != cfg.cmgdb.bounds_data_role:
        return False
    if cfg.cmgdb.bounds_data_role != "train_and_validation_pairs" and (
        logged_bounds_role != cfg.cmgdb.bounds_data_role
    ):
        return False
    logged_precompute_subdiv = params.get("adaptive_precompute_subdiv")
    if (
        logged_precompute_subdiv is not None
        and logged_precompute_subdiv != cfg.cmgdb.adaptive_precompute_subdiv
    ):
        return False
    if cfg.cmgdb.adaptive_precompute_subdiv != "init" and (
        logged_precompute_subdiv != cfg.cmgdb.adaptive_precompute_subdiv
    ):
        return False
    logged_compute_roa = _log_bool(params, "compute_roa")
    if cfg.cmgdb.compute_roa and logged_compute_roa is not True:
        return False
    if logged_compute_roa is not None and logged_compute_roa != bool(cfg.cmgdb.compute_roa):
        return False

    lower = _log_float_list(params, "lower_bounds")
    upper = _log_float_list(params, "upper_bounds")
    if lower is None or upper is None:
        return False
    if len(lower) != cfg.arch.low_dims or len(upper) != cfg.arch.low_dims:
        return False

    bounds_source = params.get("bounds_source")
    if cfg.cmgdb.lower_bounds is not None and cfg.cmgdb.upper_bounds is not None:
        return (
            bounds_source == "config"
            and _float_lists_close(lower, [float(v) for v in cfg.cmgdb.lower_bounds])
            and _float_lists_close(upper, [float(v) for v in cfg.cmgdb.upper_bounds])
        )

    expected_bounds_source = (
        "encoded_train_pairs" if cfg.cmgdb.bounds_data_role == "train_pairs" else "encoded_data"
    )
    if bounds_source != expected_bounds_source:
        return False
    logged_epsilon = _log_float(params, "bounds_epsilon_frac")
    return logged_epsilon is not None and math.isclose(
        logged_epsilon,
        float(cfg.cmgdb.bounds_epsilon_frac),
        rel_tol=1e-12,
        abs_tol=0.0,
    )


def _data_complete(cfg: ExperimentConfig) -> bool:
    for label in _train_files_for(cfg):
        if not _nonempty_file(cfg.paths.data_dir / f"{label}.csv"):
            return False
        if not _nonempty_file(cfg.paths.data_dir / f"{label}_metadata.json"):
            return False
    return _nonempty_file(cfg.paths.data_dir / "val.csv") and _nonempty_file(
        cfg.paths.data_dir / "val_metadata.json"
    )


def _stage_complete(
    stage: str,
    cfg: ExperimentConfig,
    seed_cfg: ExperimentConfig,
    *,
    train_file: str,
    derived_dir: Path | None = None,
) -> bool:
    """Best-effort artifact completeness check for resumable runs.

    For derived stages (``diagnose``/``render``/``metrics``), completeness
    is checked against ``derived_dir`` when supplied; this matters under
    replay-routing, where source artifacts are tracked-and-stale yet the
    replay tree may be empty. Falls back to ``seed_cfg.paths.output_dir``
    for source-routed runs.
    """
    derived = derived_dir if derived_dir is not None else seed_cfg.paths.output_dir
    if stage == "data":
        return _data_complete(cfg)
    if stage == "scale":
        from .scale_data import scaler_is_current

        return scaler_is_current(cfg, train_file)
    if stage == "train":
        return has_new_checkpoint(seed_cfg.paths.model_dir) or has_legacy_checkpoint(
            seed_cfg.paths.model_dir
        )
    if stage == "diagnose":
        return _nonempty_file(derived / "diagnose.json")
    if stage == "morse":
        if cfg.cmgdb.compute_roa and not _nonempty_file(
            seed_cfg.paths.morse_dir / "regions_of_attraction_exact.npz"
        ):
            return False
        return (
            _nonempty_file(seed_cfg.paths.morse_dir / "morse_graph")
            and _nonempty_file(seed_cfg.paths.morse_dir / "morse_sets")
            and _morse_params_log_matches_config(cfg, seed_cfg.paths.output_dir)
        )
    if stage == "render":
        morse_render_dir = derived / "MG"
        return (
            _nonempty_file(morse_render_dir / "morse_graph.pdf")
            and _nonempty_file(morse_render_dir / "morse_graph.png")
            and _nonempty_file(morse_render_dir / "morse_sets.pdf")
            and _nonempty_file(morse_render_dir / "morse_sets.png")
        )
    if stage == "metrics":
        return _nonempty_file(derived / "metrics.json")
    raise ValueError(f"unknown stage {stage!r}")


def _skip_completed(
    stage: str,
    cfg: ExperimentConfig,
    seed_cfg: ExperimentConfig,
    *,
    train_file: str,
    enabled: bool,
    verbose: bool,
    derived_dir: Path | None = None,
) -> bool:
    if not enabled:
        return False
    if not _stage_complete(
        stage,
        cfg,
        seed_cfg,
        train_file=train_file,
        derived_dir=derived_dir,
    ):
        return False
    if verbose:
        target = derived_dir if derived_dir is not None else seed_cfg.paths.output_dir
        print(f"{stage}: skipped existing artifacts for {target}")
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
    force_overwrite: bool = False,
    replay_root: Path | str | None = None,
    figures: set[str] | None = None,
) -> dict:
    """Run requested stages for one (config, train_file, seed) cell.

    The ``data`` stage always operates on the top-level ``cfg`` (data files
    are global). All other stages route to the seed-specific output subdir.

    Under read-only configs, derived stages (``render``/``metrics``/``diagnose``
    and ``run_manifest``) read from the source ``seed_cfg.paths`` tree but
    write under ``replay_root / cfg.experiment_name / <rel>``; see
    :func:`_derived_output_dir`. Source-write stages remain blocked unless
    ``force_overwrite`` is passed.
    """
    plan = _normalize_stages(stages)
    _check_read_only(cfg, plan, force_overwrite=force_overwrite)
    seed_cfg = _config_for_seed(cfg, train_file=train_file, seed=seed)
    resolved_replay_root = _resolve_replay_root(
        cfg,
        replay_root,
        force_overwrite=force_overwrite,
    )
    derived_dir = _derived_output_dir(cfg, seed_cfg, replay_root=resolved_replay_root)
    dev = _resolve_device(device)

    summary: dict = {
        "stages": plan,
        "train_file": train_file,
        "seed": seed,
        "output_dir": str(seed_cfg.paths.output_dir),
        "device": str(dev),
    }
    if derived_dir != seed_cfg.paths.output_dir:
        summary["replay_dir"] = str(derived_dir)

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

        train_stage.run(
            seed_cfg,
            train_file=train_file,
            seed=seed,
            device=dev,
            verbose=verbose,
            force_overwrite=force_overwrite,
        )
    elif "train" in plan:
        skipped.append("train")
    if "diagnose" in plan and not _skip_completed(
        "diagnose",
        cfg,
        seed_cfg,
        train_file=train_file,
        enabled=skip_completed,
        verbose=verbose,
        derived_dir=derived_dir,
    ):
        from . import diagnose as diagnose_stage

        summary["diagnose"] = diagnose_stage.run(
            seed_cfg,
            train_file=train_file,
            device=dev,
            verbose=verbose,
            out_dir=derived_dir,
        )
    elif "diagnose" in plan:
        skipped.append("diagnose")
    if "morse" in plan and not _skip_completed(
        "morse", cfg, seed_cfg, train_file=train_file, enabled=skip_completed, verbose=verbose
    ):
        from . import morse_graph as morse_stage

        morse_stage.run(
            seed_cfg,
            train_file=train_file,
            device=dev,
            verbose=verbose,
            force_overwrite=force_overwrite,
        )
    elif "morse" in plan:
        skipped.append("morse")
    if "render" in plan and not _skip_completed(
        "render",
        cfg,
        seed_cfg,
        train_file=train_file,
        enabled=skip_completed,
        verbose=verbose,
        derived_dir=derived_dir,
    ):
        from .render import render_stage

        summary["render"] = render_stage(
            seed_cfg,
            train_file=train_file,
            device=dev,
            verbose=verbose,
            out_dir=derived_dir,
            figures=figures,
        )
    elif "render" in plan:
        skipped.append("render")
    if "metrics" in plan and not _skip_completed(
        "metrics",
        cfg,
        seed_cfg,
        train_file=train_file,
        enabled=skip_completed,
        verbose=verbose,
        derived_dir=derived_dir,
    ):
        from .metrics import metrics_stage

        summary["metrics"] = metrics_stage(
            seed_cfg,
            cfg,
            train_file=train_file,
            verbose=verbose,
            out_dir=derived_dir,
        )
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
            out_dir=derived_dir,
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
    force_overwrite: bool = False,
    replay_root: Path | str | None = None,
    figures: set[str] | None = None,
) -> list[dict]:
    """Run requested stages for every (train_file, seed) implied by ``cfg``.

    See :func:`run_one` for replay-routing semantics under read-only configs.
    """
    plan = _normalize_stages(stages)
    _check_read_only(cfg, plan, force_overwrite=force_overwrite)
    cells = _select_cells(
        cfg,
        max_seeds=max_seeds,
        cell_index=cell_index,
        expected_cells=expected_cells,
    )
    resolved_replay_root = _resolve_replay_root(
        cfg,
        replay_root,
        force_overwrite=force_overwrite,
    )
    dev = _resolve_device(device)
    results: list[dict] = []

    if "data" in plan and not _skip_completed(
        "data",
        cfg,
        _config_for_seed(cfg, train_file=None, seed=None),
        train_file="train",
        enabled=skip_completed,
        verbose=verbose,
    ):
        from . import make_data as make_data_stage

        make_data_stage.run(cfg, verbose=verbose)

    scaled_train_files: set[str] = set()
    for cell in cells:
        train_file = cell.train_file
        seed = cell.seed
        seed_cfg = _config_for_seed(cfg, train_file=train_file, seed=seed)
        derived_dir = _derived_output_dir(cfg, seed_cfg, replay_root=resolved_replay_root)

        if "scale" in plan and train_file not in scaled_train_files:
            if _skip_completed(
                "scale",
                cfg,
                seed_cfg,
                train_file=train_file,
                enabled=skip_completed,
                verbose=verbose,
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
        if derived_dir != seed_cfg.paths.output_dir:
            cell_summary["replay_dir"] = str(derived_dir)
        skipped: list[str] = []
        if "train" in plan and not _skip_completed(
            "train", cfg, seed_cfg, train_file=train_file, enabled=skip_completed, verbose=verbose
        ):
            from . import train as train_stage

            train_stage.run(
                seed_cfg,
                train_file=train_file,
                seed=seed,
                device=dev,
                verbose=verbose,
                force_overwrite=force_overwrite,
            )
        elif "train" in plan:
            skipped.append("train")
        if "diagnose" in plan and not _skip_completed(
            "diagnose",
            cfg,
            seed_cfg,
            train_file=train_file,
            enabled=skip_completed,
            verbose=verbose,
            derived_dir=derived_dir,
        ):
            from . import diagnose as diagnose_stage

            cell_summary["diagnose"] = diagnose_stage.run(
                seed_cfg,
                train_file=train_file,
                device=dev,
                verbose=verbose,
                out_dir=derived_dir,
            )
        elif "diagnose" in plan:
            skipped.append("diagnose")
        if "morse" in plan and not _skip_completed(
            "morse", cfg, seed_cfg, train_file=train_file, enabled=skip_completed, verbose=verbose
        ):
            from . import morse_graph as morse_stage

            morse_stage.run(
                seed_cfg,
                train_file=train_file,
                device=dev,
                verbose=verbose,
                force_overwrite=force_overwrite,
            )
        elif "morse" in plan:
            skipped.append("morse")
        if "render" in plan and not _skip_completed(
            "render",
            cfg,
            seed_cfg,
            train_file=train_file,
            enabled=skip_completed,
            verbose=verbose,
            derived_dir=derived_dir,
        ):
            from .render import render_stage

            cell_summary["render"] = render_stage(
                seed_cfg,
                train_file=train_file,
                device=dev,
                verbose=verbose,
                out_dir=derived_dir,
                figures=figures,
            )
        elif "render" in plan:
            skipped.append("render")
        if "metrics" in plan and not _skip_completed(
            "metrics",
            cfg,
            seed_cfg,
            train_file=train_file,
            enabled=skip_completed,
            verbose=verbose,
            derived_dir=derived_dir,
        ):
            from .metrics import metrics_stage

            cell_summary["metrics"] = metrics_stage(
                seed_cfg,
                cfg,
                train_file=train_file,
                verbose=verbose,
                out_dir=derived_dir,
            )
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
                out_dir=derived_dir,
            )
        )
        results.append(cell_summary)

    return results
