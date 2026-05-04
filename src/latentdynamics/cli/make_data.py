"""Build train/test trajectory CSVs and metadata from an experiment config."""

from __future__ import annotations

from pathlib import Path

from ..config import ExperimentConfig
from ..sampling import build_strategy, sample_trajectories
from ..sampling.trajectories import TrajectoryDataset
from ..systems import build_system


def _dataset_paths(label: str, data_dir: Path) -> tuple[Path, Path]:
    return data_dir / f"{label}.csv", data_dir / f"{label}_metadata.json"


def _existing_dataset(label: str, data_dir: Path) -> bool:
    csv_path, meta_path = _dataset_paths(label, data_dir)
    csv_exists = csv_path.exists()
    meta_exists = meta_path.exists()
    if csv_exists and meta_exists:
        return True
    if csv_exists or meta_exists:
        raise FileExistsError(
            f"partial dataset for {label!r}: found csv={csv_exists}, "
            f"metadata={meta_exists}; refusing to overwrite saved data"
        )
    return False


def _emit(label: str, ds: TrajectoryDataset, data_dir: Path, *, verbose: bool) -> None:
    csv_path = data_dir / f"{label}.csv"
    meta_path = data_dir / f"{label}_metadata.json"
    ds.to_csv(csv_path)
    ds.save_metadata(meta_path)
    if verbose:
        print(f"wrote {csv_path} ({ds.X.shape[0]} pairs)")


def _train_labels(cfg: ExperimentConfig) -> list[tuple[int | None, str]]:
    if cfg.data.train_files is not None:
        return [(None, label) for label in cfg.data.train_files]
    if isinstance(cfg.data.n_samples_train, list):
        return [(int(N), f"train_{N}") for N in cfg.data.n_samples_train]
    return [(int(cfg.data.n_samples_train), "train")]


def _validate_precomputed(labels: list[str], data_dir: Path, *, verbose: bool) -> None:
    missing: list[str] = []
    for label in labels:
        csv_path, meta_path = _dataset_paths(label, data_dir)
        if not csv_path.exists() or not meta_path.exists():
            missing.append(f"{csv_path} + {meta_path}")
    if missing:
        raise FileNotFoundError(
            "adaptive sampling is precomputed; missing saved dataset(s): "
            + "; ".join(missing)
        )
    if verbose:
        print(f"using {len(labels)} precomputed dataset(s) under {data_dir}")


def run(cfg: ExperimentConfig, *, verbose: bool = True) -> None:
    """Generate all train CSVs (one per train size) and the test CSV."""
    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)

    train_labels = _train_labels(cfg)
    if cfg.data.sampling_method == "adaptive":
        _validate_precomputed(
            [label for _, label in train_labels] + ["test"],
            cfg.paths.data_dir,
            verbose=verbose,
        )
        return

    system = build_system(cfg.system.name, cfg.system.params)
    if verbose:
        print(f"system: {cfg.system.name} (dim={system.dim})")
        print(f"  lower_bounds: {system.lower_bounds.tolist()}")
        print(f"  upper_bounds: {system.upper_bounds.tolist()}")

    train_strategy = build_strategy(cfg.data.sampling_method, role="train", config=cfg.data)
    test_strategy = build_strategy(cfg.data.sampling_method, role="test", config=cfg.data)

    for n_samples, label in train_labels:
        if _existing_dataset(label, cfg.paths.data_dir):
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
                "role": "train",
            },
        )
        _emit(label, ds, cfg.paths.data_dir, verbose=verbose)

    if _existing_dataset("test", cfg.paths.data_dir):
        if verbose:
            print(f"kept existing {cfg.paths.data_dir / 'test.csv'}")
        return
    test_ds = sample_trajectories(
        system=system,
        strategy=test_strategy,
        n_samples=cfg.data.n_samples_test,
        n_iterations=cfg.data.n_iterations,
        skip=cfg.data.skip,
        metadata_extra={
            "dataset_name": "test",
            "sampling_method": cfg.data.sampling_method,
            "role": "test",
        },
    )
    _emit("test", test_ds, cfg.paths.data_dir, verbose=verbose)
