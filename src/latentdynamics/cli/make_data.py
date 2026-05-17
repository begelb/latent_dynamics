"""Build train/val trajectory CSVs and metadata from an experiment config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..sampling import build_strategy, sample_trajectories
from ..sampling.trajectories import TrajectoryDataset
from ..systems import build_system
from ..systems.base import DynamicalSystem


def _dataset_paths(label: str, data_dir: Path) -> tuple[Path, Path]:
    return data_dir / f"{label}.csv", data_dir / f"{label}_metadata.json"


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"stale existing dataset metadata at {path}: invalid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"stale existing dataset metadata at {path}: expected JSON object")
    return metadata


def _sampling_seed(cfg: ExperimentConfig, role: str) -> int:
    default_seed = 42 if role == "train" else 9999
    return int(getattr(cfg.data, f"{role}_seed", default_seed))


def _expected_metadata(
    cfg: ExperimentConfig,
    *,
    system: DynamicalSystem,
    n_samples: int | None,
    sampling_seed: int | None,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "system": type(system).__name__,
        "dimension": int(system.dim),
        "n_iterations": int(cfg.data.n_iterations),
        "skip_initial_steps": int(cfg.data.skip),
        "model_params": system.params,
        "sampling_method": cfg.data.sampling_method,
    }
    if n_samples is not None:
        expected["n_samples"] = int(n_samples)
    if sampling_seed is not None:
        expected["sampling_seed"] = int(sampling_seed)
    return expected


def _validate_metadata(
    *,
    label: str,
    metadata: dict[str, Any],
    expected: dict[str, Any],
    dataset_names: tuple[str, ...],
    roles: tuple[str, ...],
) -> None:
    mismatches: list[str] = []

    actual_name = metadata.get("dataset_name")
    if actual_name not in dataset_names:
        mismatches.append(f"dataset_name expected one of {dataset_names}, got {actual_name!r}")

    actual_role = metadata.get("role")
    if actual_role not in roles:
        mismatches.append(f"role expected one of {roles}, got {actual_role!r}")

    for key, value in expected.items():
        if key not in metadata:
            mismatches.append(f"{key} missing")
        elif metadata[key] != value:
            mismatches.append(f"{key} expected {value!r}, got {metadata[key]!r}")

    if mismatches:
        raise ValueError(f"stale existing dataset {label!r}: " + "; ".join(mismatches))


def _existing_dataset(
    label: str,
    data_dir: Path,
    *,
    cfg: ExperimentConfig | None = None,
    system: DynamicalSystem | None = None,
    role: str | None = None,
    n_samples: int | None = None,
    dataset_names: tuple[str, ...] | None = None,
    roles: tuple[str, ...] | None = None,
) -> bool:
    csv_path, meta_path = _dataset_paths(label, data_dir)
    csv_exists = csv_path.exists()
    meta_exists = meta_path.exists()
    if csv_exists and meta_exists:
        if cfg is not None and system is not None and role is not None:
            sampling_seed = (
                _sampling_seed(cfg, role)
                if cfg.data.sampling_method in {"uniform", "sobol"}
                else None
            )
            _validate_metadata(
                label=label,
                metadata=_load_metadata(meta_path),
                expected=_expected_metadata(
                    cfg,
                    system=system,
                    n_samples=n_samples,
                    sampling_seed=sampling_seed,
                ),
                dataset_names=dataset_names or (label,),
                roles=roles or (role,),
            )
        return True
    if csv_exists or meta_exists:
        raise FileExistsError(
            f"partial dataset for {label!r}: found csv={csv_exists}, "
            f"metadata={meta_exists}; refusing to overwrite saved data"
        )
    return False


def _existing_val_dataset(
    data_dir: Path,
    *,
    cfg: ExperimentConfig | None = None,
    system: DynamicalSystem | None = None,
) -> bool:
    """True if the ``val.csv`` pair already exists on disk."""
    return _existing_dataset(
        "val",
        data_dir,
        cfg=cfg,
        system=system,
        role="val" if cfg is not None else None,
        n_samples=cfg.data.n_samples_val if cfg is not None else None,
    )


def _emit(label: str, ds: TrajectoryDataset, data_dir: Path, *, verbose: bool) -> None:
    csv_path = data_dir / f"{label}.csv"
    meta_path = data_dir / f"{label}_metadata.json"
    ds.to_csv(csv_path)
    ds.save_metadata(meta_path)
    if verbose:
        print(f"wrote {csv_path} ({ds.X.shape[0]} pairs)")


def _train_labels(cfg: ExperimentConfig) -> list[tuple[int | None, str]]:
    if cfg.data.train_files is not None:
        labels = list(cfg.data.train_files)
        if isinstance(cfg.data.n_samples_train, list) and len(cfg.data.n_samples_train) == len(
            labels
        ):
            return [
                (int(n_samples), label)
                for n_samples, label in zip(cfg.data.n_samples_train, labels, strict=True)
            ]
        if isinstance(cfg.data.n_samples_train, int) and len(labels) == 1:
            return [(int(cfg.data.n_samples_train), labels[0])]
        return [(None, label) for label in labels]
    if isinstance(cfg.data.n_samples_train, list):
        return [(int(N), f"train_{N}") for N in cfg.data.n_samples_train]
    return [(int(cfg.data.n_samples_train), "train")]


def _validate_precomputed(
    train_labels: list[tuple[int | None, str]],
    data_dir: Path,
    *,
    cfg: ExperimentConfig,
    system: DynamicalSystem,
    verbose: bool,
) -> None:
    missing: list[str] = []
    for n_samples, label in train_labels:
        csv_path, meta_path = _dataset_paths(label, data_dir)
        if not csv_path.exists() or not meta_path.exists():
            missing.append(f"{csv_path} + {meta_path}")
            continue
        _existing_dataset(
            label,
            data_dir,
            cfg=cfg,
            system=system,
            role="train",
            n_samples=n_samples,
        )
    if missing:
        raise FileNotFoundError(
            "adaptive sampling is precomputed; missing saved dataset(s): " + "; ".join(missing)
        )
    if verbose:
        print(f"using {len(train_labels)} precomputed dataset(s) under {data_dir}")


def _validate_precomputed_val(
    data_dir: Path,
    *,
    cfg: ExperimentConfig,
    system: DynamicalSystem,
    verbose: bool,
) -> None:
    """Validate that a precomputed ``val.csv`` dataset exists."""
    val_csv, val_meta = _dataset_paths("val", data_dir)
    if val_csv.exists() and val_meta.exists():
        _existing_dataset(
            "val",
            data_dir,
            cfg=cfg,
            system=system,
            role="val",
            n_samples=cfg.data.n_samples_val,
        )
        if verbose:
            print(f"using precomputed val dataset under {data_dir}")
        return
    if val_csv.exists() or val_meta.exists():
        _existing_dataset("val", data_dir)
    raise FileNotFoundError(
        f"adaptive sampling is precomputed; missing validation dataset under {data_dir} "
        f"(expected val.csv + val_metadata.json)"
    )


def run(cfg: ExperimentConfig, *, verbose: bool = True) -> None:
    """Generate all train CSVs (one per train size) and the val CSV."""
    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)

    train_labels = _train_labels(cfg)
    system = build_system(cfg.system.name, cfg.system.params)
    if cfg.data.sampling_method == "adaptive":
        _validate_precomputed(
            train_labels,
            cfg.paths.data_dir,
            cfg=cfg,
            system=system,
            verbose=verbose,
        )
        _validate_precomputed_val(
            cfg.paths.data_dir,
            cfg=cfg,
            system=system,
            verbose=verbose,
        )
        return

    if verbose:
        print(f"system: {cfg.system.name} (dim={system.dim})")
        print(f"  lower_bounds: {system.lower_bounds.tolist()}")
        print(f"  upper_bounds: {system.upper_bounds.tolist()}")

    train_strategy = build_strategy(cfg.data.sampling_method, role="train", config=cfg.data)
    val_strategy = build_strategy(cfg.data.sampling_method, role="val", config=cfg.data)

    for n_samples, label in train_labels:
        if _existing_dataset(
            label,
            cfg.paths.data_dir,
            cfg=cfg,
            system=system,
            role="train",
            n_samples=n_samples,
        ):
            if verbose:
                print(f"kept existing {cfg.paths.data_dir / f'{label}.csv'}")
            continue
        if n_samples is None:
            raise ValueError(f"cannot generate custom train_file {label!r} without n_samples")
        ds = sample_trajectories(
            system=system,
            strategy=train_strategy,
            n_samples=n_samples,
            n_iterations=cfg.data.n_iterations,
            skip=cfg.data.skip,
            metadata_extra={
                "dataset_name": label,
                "sampling_method": cfg.data.sampling_method,
                "sampling_seed": _sampling_seed(cfg, "train"),
                "role": "train",
            },
        )
        _emit(label, ds, cfg.paths.data_dir, verbose=verbose)

    if _existing_val_dataset(cfg.paths.data_dir, cfg=cfg, system=system):
        if verbose:
            kept = cfg.paths.val_csv()
            print(f"kept existing {kept}")
        return
    val_ds = sample_trajectories(
        system=system,
        strategy=val_strategy,
        n_samples=cfg.data.n_samples_val,
        n_iterations=cfg.data.n_iterations,
        skip=cfg.data.skip,
        metadata_extra={
            "dataset_name": "val",
            "sampling_method": cfg.data.sampling_method,
            "sampling_seed": _sampling_seed(cfg, "val"),
            "role": "val",
        },
    )
    _emit("val", val_ds, cfg.paths.data_dir, verbose=verbose)
