"""Reproduce the Chafee--Infante latent-dimension study.

This driver deliberately follows the archived reference computation whose
inputs ship under ``replay_sources/chafee_infante/reference_inputs``:

* the exact 30,000 one-step pairs are used without scaling;
* the architecture and fixed 4,000-epoch full-batch objective are unchanged;
* seed 0 is explicit for the new one- and three-dimensional models;
* bounds are inferred from encoded current *and* next states with 10% padding;
* CMGDB uses padding and ``subdiv_limit=10000``; and
* the 10,000 archived initial conditions and trajectory labels are reused for
  the basin table.

The CMGDB callback never evaluates a neural network.  A dense corner table is
persisted before the uniform computation.  For the 3-D adaptive computation,
the level-24 uniform Morse sets select level-24 cells in which 9x9x9 level-33
corner blocks are pre-evaluated and persisted.  Both scalar and batch CMGDB
callbacks subsequently perform table lookup only.

Every resumable stage has a schema-v2 marker containing a canonical direct-
input fingerprint and hashed output manifest.  A changed checkpoint, upstream
artifact, or semantic option invalidates reuse and descendants are explicitly
marked stale before recomputation.  Matching legacy artifacts can be adopted
once after their embedded hashes, dimensions, bounds, and resolutions pass the
same study invariants.

The canonical saved 1-D and 3-D checkpoints were trained on PyTorch's MPS
backend with seed 0.  The seed fixes the RNG streams but deterministic
algorithms are not forced, so bitwise-identical weights are not promised across
PyTorch versions, hardware, or CPU/MPS/CUDA backends.  This limitation and the
resolved runtime are recorded in ``study_config.json`` and training metadata.

The uniform graph is an auxiliary computation used to define the basins.  It
therefore uses ``ComputeMorseGraph`` and deliberately defers Conley-index
homology to the final adaptive graph.  This changes only node annotations:
the recurrent components, partial order, Morse boxes, map graph, and basin
classification are identical to ``ComputeConleyMorseGraph``.  The adaptive
stage requests Conley indices by default.  If CHomP raises an error, or if a
known pathological Smith-normal-form computation is bypassed explicitly with
``--adaptive-topology-only``, the adaptive artifacts are saved with an
explicit topology-only status.

Examples
--------
Run the complete 1-D study, resuming completed stages::

    python scripts/chafee_latent_dimension_study.py --dimensions 1

Run selected 3-D stages::

    python scripts/chafee_latent_dimension_study.py --dimensions 3 \
        --stages precompute-coarse uniform precompute-fine adaptive stats render \
        --cmgdb-reserve-edges 1200000000

The 3-D uniform graph has exactly 2^24 cells.  A locally built CMGDB with the
batched MapGraph cache is required.  ``--cmgdb-reserve-edges`` pre-allocates
the CSR edge buffer at roughly eight bytes per edge; it is a sizing hint, not
a ceiling, and the cache grows past it when the graph is larger.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import platform
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import CMGDB
import numpy as np
import torch
from numpy.typing import NDArray

from latentdynamics.analysis.basin_statistics import (
    OUTSIDE,
    cmgdb_morton_cell_indices,
    compute_chafee_basin_statistics,
)
from latentdynamics.analysis.hierarchical_precomputed import (
    HierarchicalPrecomputedBoxMap,
)
from latentdynamics._paths import get_repo_root
from latentdynamics.analysis.morse import LatentBounds, infer_latent_bounds
from latentdynamics.analysis.morse_graph_parser import MorseGraph
from latentdynamics.config import ArchConfig
from latentdynamics.training import load_checkpoint, train_reference_full_batch
from latentdynamics.viz import (
    render_morse_graph_from_dot,
    render_morse_set_projections_from_csv,
    render_morse_sets_3d_cubical_from_csv,
    render_morse_sets_from_csv,
    save_morse_graph_artifacts,
)

REPO_ROOT = get_repo_root()
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "replay_sources" / "chafee_infante" / "reference_inputs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "chafee_latent_dimension_study"

TRAIN_DATA_SHA256 = "890fea29a2d26b31b44bbc4fbd773af0ebf742b08893c27ddaf1bdbf87e30329"
TRAJECTORY_LABELS_SHA256 = (
    "f163b7427e50a4e4d08ab54c87cb5bd16592768edfe8432f019842416afbb145"
)
STABLE_ROOTS_SHA256 = "cae0222acb37ae9688e54cb2a1f42ac3777360e49b3919403ec1433363de0586"

HIGH_DIMENSION = 64
TRAINING_ROWS = 30_000
TRAJECTORY_ROWS = 10_000
SEED = 0
EPOCHS = 4_000
LEARNING_RATE = 0.003
BOUNDS_EPSILON_FRAC = 0.1
SUBDIV_LIMIT = 10_000
PADDING = True
EXPECTED_TRAJECTORY_LABEL_COUNTS = {-1: 3_909, 0: 2_138, 1: 3_953}
CANONICAL_TRAINING_BACKEND = "mps"
STAGE_PROVENANCE_SCHEMA_VERSION = 2
STAGE_ALGORITHM_VERSION = {
    "train": 1,
    "bounds": 1,
    "precompute-coarse": 1,
    "uniform": 2,
    "precompute-fine": 1,
    "adaptive": 3,
    "stats": 2,
    "render": 1,
}

STAGE_ORDER = (
    "train",
    "bounds",
    "precompute-coarse",
    "uniform",
    "precompute-fine",
    "adaptive",
    "stats",
    "render",
)
VALID_STAGES = frozenset(("validate", *STAGE_ORDER, "all"))
STAGE_DEPENDENCIES = {
    "train": (),
    "bounds": ("train",),
    "precompute-coarse": ("train", "bounds"),
    "uniform": ("train", "bounds", "precompute-coarse"),
    "precompute-fine": ("train", "bounds", "precompute-coarse"),
    "adaptive": ("train", "bounds", "precompute-fine"),
    "stats": ("uniform", "adaptive"),
    "render": ("bounds", "adaptive"),
}


@dataclass(frozen=True)
class Resolution:
    """Matched CMGDB total subdivision depths for one latent dimension."""

    dimension: int
    uniform_init: int
    uniform_min: int
    uniform_max: int
    adaptive_init: int
    adaptive_min: int
    adaptive_max: int

    @property
    def coarse_cells_per_axis(self) -> int:
        return 2 ** (self.uniform_max // self.dimension)

    @property
    def uniform_cells(self) -> int:
        return 2**self.uniform_max


RESOLUTIONS = {
    1: Resolution(
        dimension=1,
        uniform_init=8,
        uniform_min=8,
        uniform_max=8,
        adaptive_init=7,
        adaptive_min=8,
        adaptive_max=11,
    ),
    3: Resolution(
        dimension=3,
        uniform_init=24,
        uniform_min=24,
        uniform_max=24,
        adaptive_init=21,
        adaptive_min=24,
        adaptive_max=33,
    ),
}


@dataclass(frozen=True)
class ExactInputs:
    archive_dir: Path
    train_data: Path
    trajectory_labels: Path
    stable_roots: Path
    hashes: dict[str, str]
    sizes_bytes: dict[str, int]

    def provenance(self) -> dict[str, Any]:
        return {
            "archive_dir": str(self.archive_dir.resolve()),
            "files": {
                "train_data.csv": {
                    "path": str(self.train_data.resolve()),
                    "sha256": self.hashes["train_data.csv"],
                    "size_bytes": self.sizes_bytes["train_data.csv"],
                },
                "traj_attractors.pkl": {
                    "path": str(self.trajectory_labels.resolve()),
                    "sha256": self.hashes["traj_attractors.pkl"],
                    "size_bytes": self.sizes_bytes["traj_attractors.pkl"],
                },
                "stable_solutions.csv": {
                    "path": str(self.stable_roots.resolve()),
                    "sha256": self.hashes["stable_solutions.csv"],
                    "size_bytes": self.sizes_bytes["stable_solutions.csv"],
                },
            },
        }


@dataclass(frozen=True)
class DimensionPaths:
    output_root: Path
    dimension: int

    @property
    def run(self) -> Path:
        return self.output_root / f"latent_{self.dimension}d" / f"seed_{SEED}"

    @property
    def models(self) -> Path:
        return self.run / "models"

    @property
    def bounds(self) -> Path:
        return self.run / "bounds.json"

    @property
    def precomputed_root(self) -> Path:
        return self.run / "precomputed_map"

    @property
    def coarse_table(self) -> Path:
        level = RESOLUTIONS[self.dimension]
        in_place = self.run / (
            f"precomputed_level{level.uniform_max}_to{level.adaptive_max}"
        )
        split = self.precomputed_root / "coarse"
        if in_place.exists() or not split.exists():
            return in_place
        return split

    @property
    def hierarchical_table(self) -> Path:
        split = self.precomputed_root / "hierarchical"
        if split.exists() and not self.coarse_table.name.startswith("precomputed_level"):
            return split
        return self.coarse_table

    @property
    def uniform(self) -> Path:
        level = RESOLUTIONS[self.dimension].uniform_max
        return self.run / f"MG_uniform_s{level}"

    @property
    def adaptive(self) -> Path:
        return self.run / "MG_adaptive"

    @property
    def stats(self) -> Path:
        return self.run / "basin_statistics.json"

    @property
    def stage_dir(self) -> Path:
        return self.run / "stage_status"

    def stage_marker(self, stage: str) -> Path:
        return self.stage_dir / f"{stage}.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def verify_exact_inputs(archive_dir: Path) -> ExactInputs:
    archive = archive_dir.resolve()
    paths = {
        "train_data.csv": archive / "train_data.csv",
        "traj_attractors.pkl": archive / "traj_attractors.pkl",
        "stable_solutions.csv": archive / "stable_solutions.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing archived Chafee--Infante inputs: {missing}")

    hashes = {name: sha256_file(path) for name, path in paths.items()}
    expected = {
        "train_data.csv": TRAIN_DATA_SHA256,
        "traj_attractors.pkl": TRAJECTORY_LABELS_SHA256,
        "stable_solutions.csv": STABLE_ROOTS_SHA256,
    }
    mismatches = {
        name: {"expected": expected[name], "actual": hashes[name]}
        for name in expected
        if hashes[name] != expected[name]
    }
    if mismatches:
        raise ValueError(
            "archived Chafee--Infante input hash mismatch; refusing a non-comparable run: "
            f"{mismatches}"
        )

    sizes = {name: int(path.stat().st_size) for name, path in paths.items()}
    return ExactInputs(
        archive_dir=archive,
        train_data=paths["train_data.csv"],
        trajectory_labels=paths["traj_attractors.pkl"],
        stable_roots=paths["stable_solutions.csv"],
        hashes=hashes,
        sizes_bytes=sizes,
    )


def reference_architecture(latent_dimension: int) -> ArchConfig:
    if latent_dimension not in RESOLUTIONS:
        raise ValueError(f"supported latent dimensions are {sorted(RESOLUTIONS)}")
    return ArchConfig(
        high_dims=HIGH_DIMENSION,
        low_dims=latent_dimension,
        encoder={
            "hidden_shapes": [64, 32],
            "activation": "tanh",
            "out_activation": "none",
        },
        latent_map={
            "hidden_shapes": [32, 32],
            "activation": "tanh",
            "out_activation": "none",
        },
        decoder={
            "hidden_shapes": [32, 64],
            "activation": "tanh",
            "out_activation": "none",
        },
    )


def _load_training_pairs(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pairs = np.loadtxt(path, delimiter=",", dtype=np.float64)
    expected_shape = (TRAINING_ROWS, 2 * HIGH_DIMENSION)
    if pairs.shape != expected_shape:
        raise ValueError(f"{path} has shape {pairs.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(pairs)):
        raise ValueError(f"{path} contains non-finite values")
    return pairs[:, :HIGH_DIMENSION], pairs[:, HIGH_DIMENSION:]


def _load_stable_roots(path: Path) -> NDArray[np.float64]:
    roots = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if roots.shape != (2, HIGH_DIMENSION):
        raise ValueError(f"{path} has shape {roots.shape}; expected {(2, HIGH_DIMENSION)}")
    if roots[0, 0] >= 0.0 or roots[1, 0] <= 0.0:
        raise ValueError("stable_solutions.csv no longer has negative-root then positive-root order")
    return roots


def _load_trajectory_labels(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    with path.open("rb") as source:
        # This is a trusted local archive whose SHA256 is verified first.
        archived = pickle.load(source)
    if not isinstance(archived, dict) or len(archived) != TRAJECTORY_ROWS:
        raise ValueError(f"{path} must contain a {TRAJECTORY_ROWS}-entry dictionary")
    points = np.asarray([tuple(point) for point in archived], dtype=np.float64)
    labels = np.asarray(list(archived.values()), dtype=np.int64)
    if points.shape != (TRAJECTORY_ROWS, HIGH_DIMENSION):
        raise ValueError(
            f"archived trajectory initial points have shape {points.shape}; "
            f"expected {(TRAJECTORY_ROWS, HIGH_DIMENSION)}"
        )
    counts = {int(label): int(count) for label, count in Counter(labels.tolist()).items()}
    if counts != EXPECTED_TRAJECTORY_LABEL_COUNTS:
        raise ValueError(
            "archived trajectory-label counts changed: "
            f"got {counts}, expected {EXPECTED_TRAJECTORY_LABEL_COUNTS}"
        )
    return points, labels


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model(
    paths: DimensionPaths,
    *,
    device: torch.device,
) -> tuple[Any, ArchConfig]:
    model, arch = load_checkpoint(paths.models, map_location=device)
    expected = reference_architecture(paths.dimension)
    if arch != expected:
        raise ValueError(
            f"checkpoint architecture in {paths.models} is not the reference-faithful "
            f"{paths.dimension}-D architecture"
        )
    model = model.to(device)
    model.eval()
    return model, arch


def _encode_numpy(
    encoder: torch.nn.Module,
    values: NDArray[np.float64],
    *,
    device: torch.device,
    batch_size: int = 16_384,
) -> NDArray[np.float64]:
    rows: list[NDArray[np.float64]] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            tensor = torch.as_tensor(
                values[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            rows.append(encoder(tensor).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(rows, axis=0)


def _load_bounds(path: Path, dimension: int) -> LatentBounds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lower = np.asarray(payload["lower"], dtype=np.float64)
    upper = np.asarray(payload["upper"], dtype=np.float64)
    if lower.shape != (dimension,) or upper.shape != (dimension,):
        raise ValueError(f"{path} bounds do not have dimension {dimension}")
    if not np.all(lower < upper):
        raise ValueError(f"{path} does not satisfy lower < upper")
    epsilon = payload.get("epsilon_frac", payload.get("epsilon_fraction"))
    if epsilon is None or float(epsilon) != BOUNDS_EPSILON_FRAC:
        raise ValueError(f"{path} was not generated with epsilon_frac={BOUNDS_EPSILON_FRAC}")
    return LatentBounds(lower=lower, upper=upper)


def _morse_summary(dot_path: Path) -> dict[str, Any]:
    graph = MorseGraph.from_dot(dot_path)
    return {
        "nodes": len(graph.nodes),
        "edges": int(sum(len(targets) for targets in graph.edges.values())),
        "minimal_nodes": sorted(int(node) for node in graph.minimal),
        "is_bistable": len(graph.minimal) == 2,
    }


def _table_metadata(directory: Path) -> dict[str, Any]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    arrays = {}
    for name in ("coarse_values.npy", "active_coarse_indices.npy", "fine_block_values.npy"):
        path = directory / name
        if path.exists():
            array = np.load(path, mmap_mode="r")
            arrays[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "size_bytes": int(path.stat().st_size),
            }
    return {"metadata": metadata, "arrays": arrays}


def _json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cached_file_sha256(
    path: Path,
    cache: dict[tuple[str, int, int], str],
) -> str:
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    digest = cache.get(key)
    if digest is None:
        digest = sha256_file(path)
        cache[key] = digest
    return digest


def _artifact_signature(
    path: Path,
    *,
    run_root: Path,
    cache: dict[tuple[str, int, int], str],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        label = str(path.resolve().relative_to(run_root.resolve()))
    except ValueError:
        label = str(path.resolve())
    return {
        "path": label,
        "size_bytes": int(path.stat().st_size),
        "sha256": _cached_file_sha256(path, cache),
    }


def _stable_table_metadata(directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    keys = (
        "schema_version",
        "dimension",
        "lower",
        "upper",
        "coarse_subdiv",
        "fine_subdiv",
        "padding",
        "callback_neural_evaluations",
    )
    return {key: payload.get(key) for key in keys}


def _uniform_statistics_core(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "schema_version",
        "dimension",
        "seed",
        "method",
        "trajectory_data",
        "stable_roots",
        "cmgdb",
        "classification",
        "statistics",
    )
    return {key: payload.get(key) for key in keys}


def _stage_output_paths(paths: DimensionPaths, stage: str) -> dict[str, Path]:
    if stage == "train":
        return {
            "checkpoint": paths.models / "autoencoder.pt",
            "checkpoint_metadata": paths.models / "autoencoder.json",
            "training_summary": paths.run / "training_summary.json",
            "training_history": paths.run / "logs" / "history.json",
        }
    if stage == "bounds":
        return {"bounds": paths.bounds}
    if stage == "precompute-coarse":
        # Fine-block precomputation may update metadata in the same directory.
        # The dense coarse values themselves are immutable.
        return {"coarse_values": paths.coarse_table / "coarse_values.npy"}
    if stage == "uniform":
        result = {
            "morse_graph": paths.uniform / "morse_graph",
            "morse_sets": paths.uniform / "morse_sets",
        }
        query_artifact = paths.uniform / "reference_singleton_reachability_queries.npz"
        if query_artifact.is_file():
            result["strict_singleton_reachability_queries"] = query_artifact
        return result
    if stage == "precompute-fine":
        return {
            name.removesuffix(".npy"): paths.hierarchical_table / name
            for name in (
                "coarse_values.npy",
                "active_coarse_indices.npy",
                "fine_block_values.npy",
                "active_coarse_boxes.npy",
            )
        }
    if stage == "adaptive":
        return {
            "morse_graph": paths.adaptive / "morse_graph",
            "morse_sets": paths.adaptive / "morse_sets",
        }
    if stage == "stats":
        return {"basin_statistics": paths.stats}
    if stage == "render":
        result = {
            "morse_graph_pdf": paths.adaptive / "morse_graph.pdf",
            "morse_graph_png": paths.adaptive / "morse_graph.png",
        }
        if paths.dimension == 1:
            result.update(
                {
                    "morse_sets_pdf": paths.adaptive / "morse_sets.pdf",
                    "morse_sets_png": paths.adaptive / "morse_sets.png",
                }
            )
        else:
            result.update(
                {
                    f"morse_sets_z{i}_z{j}_{extension}": (
                        paths.adaptive / f"morse_sets_z{i}_z{j}.{extension}"
                    )
                    for i, j in ((1, 2), (1, 3), (2, 3))
                    for extension in ("pdf", "png")
                }
            )
            result.update(
                {
                    f"morse_sets_cubical_3d_{extension}": (
                        paths.adaptive / f"morse_sets_cubical_3d.{extension}"
                    )
                    for extension in ("pdf", "png")
                }
            )
        return result
    raise ValueError(f"unknown stage {stage!r}")


def _stage_output_manifest(
    paths: DimensionPaths,
    stage: str,
    *,
    cache: dict[tuple[str, int, int], str],
) -> dict[str, Any]:
    files = {
        name: _artifact_signature(path, run_root=paths.run, cache=cache)
        for name, path in _stage_output_paths(paths, stage).items()
    }
    components: dict[str, Any] = {}
    if stage in {"precompute-coarse", "precompute-fine"}:
        components["table_geometry"] = _stable_table_metadata(
            paths.coarse_table if stage == "precompute-coarse" else paths.hierarchical_table
        )
    if stage == "uniform":
        components["basin_statistics_core"] = _uniform_statistics_core(paths.stats)
    core = {"files": files, "components": components}
    return {**core, "fingerprint": _json_fingerprint(core)}


def _stage_dependencies(paths: DimensionPaths, stage: str) -> tuple[str, ...]:
    dependencies = STAGE_DEPENDENCIES[stage]
    if stage == "precompute-fine" and paths.dimension == 3:
        return (*dependencies, "uniform")
    return dependencies


def _load_stage_marker(paths: DimensionPaths, stage: str) -> dict[str, Any] | None:
    marker_path = paths.stage_marker(stage)
    if not marker_path.is_file():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _marker_output_fingerprint(paths: DimensionPaths, stage: str) -> str:
    marker = _load_stage_marker(paths, stage)
    try:
        return str(marker["provenance"]["outputs"]["fingerprint"])
    except (KeyError, TypeError):
        raise ValueError(
            f"{stage!r} has no validated output fingerprint for {paths.dimension}D"
        ) from None


def _cmgdb_binary_signature(
    paths: DimensionPaths,
    cache: dict[tuple[str, int, int], str],
) -> dict[str, Any]:
    module_name = getattr(CMGDB.ComputeMorseGraph, "__module__", "")
    module = __import__(module_name, fromlist=["__file__"])
    binary = Path(module.__file__).resolve()
    return _artifact_signature(binary, run_root=paths.run, cache=cache)


def _training_reproducibility_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "canonical_artifact_backend": CANONICAL_TRAINING_BACKEND,
        "resolved_backend": device.type,
        "seed": SEED,
        "seeded_rngs": ["python", "numpy", "torch_cpu", "torch_cuda_if_available"],
        "deterministic_algorithms_enforced": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "bitwise_reproducible_across_backends_or_runtime_versions": False,
        "limitation": (
            "Seed 0 fixes the initialized RNG streams, but deterministic PyTorch "
            "algorithms are not forced. MPS/CPU/CUDA kernels, hardware, and PyTorch "
            "versions may produce numerically different trained checkpoints."
        ),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }


def _stage_direct_inputs(
    paths: DimensionPaths,
    inputs: ExactInputs,
    stage: str,
    *,
    device: torch.device,
    batch_points: int | str,
    min_box_side_frac: float,
    adaptive_topology_only: bool,
    cache: dict[tuple[str, int, int], str],
) -> dict[str, Any]:
    resolution = RESOLUTIONS[paths.dimension]
    upstream = {
        dependency: _marker_output_fingerprint(paths, dependency)
        for dependency in _stage_dependencies(paths, stage)
    }
    study: dict[str, Any] = {
        "dimension": paths.dimension,
        "seed": SEED,
        "train_data_sha256": inputs.hashes["train_data.csv"],
    }
    if stage in {"uniform", "stats"}:
        study.update(
            {
                "trajectory_labels_sha256": inputs.hashes["traj_attractors.pkl"],
                "stable_roots_sha256": inputs.hashes["stable_solutions.csv"],
            }
        )

    if stage == "train":
        configuration: dict[str, Any] = {
            "architecture": reference_architecture(paths.dimension).model_dump(),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "objective": "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)",
            "optimizer": "Adam",
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "factor": 0.5,
                "patience": 100,
                "threshold": 1e-4,
                "threshold_mode": "rel",
                "min_lr": 1e-6,
            },
            "dtype": "float32",
            "full_batch": True,
            "device": str(device),
            "reproducibility": _training_reproducibility_metadata(device),
        }
    elif stage == "bounds":
        configuration = {
            "epsilon_frac": BOUNDS_EPSILON_FRAC,
            "source": "encoded current and next training states",
            "device": str(device),
        }
    elif stage == "precompute-coarse":
        configuration = {
            "coarse_subdiv": resolution.uniform_max,
            "fine_subdiv": resolution.adaptive_max,
            "padding": PADDING,
            "batch_points": batch_points,
            "device": str(device),
            "zero_neural_evaluations_during_cmgdb": True,
        }
    elif stage == "uniform":
        configuration = {
            "subdiv_init": resolution.uniform_init,
            "subdiv_min": resolution.uniform_min,
            "subdiv_max": resolution.uniform_max,
            "subdiv_limit": SUBDIV_LIMIT,
            "padding": PADDING,
            "routine": "CMGDB.ComputeMorseGraph",
            "basin_semantics": "complete reachable Morse set equals one singleton",
            "closed_boundary_priority": "negative then positive",
            "authoritative_reachability_artifact": (
                "reference_singleton_reachability_queries.npz"
            ),
            "legacy_minimal_lca_roa_artifacts_are_not_consumed": True,
            "encoding_device": str(device),
            "cmgdb_binary": _cmgdb_binary_signature(paths, cache),
        }
    elif stage == "precompute-fine":
        configuration = {
            "fine_subdiv": resolution.adaptive_max,
            "padding": PADDING,
            "selection": (
                "all level-8 cells"
                if paths.dimension == 1
                else "all cells in every level-24 uniform Morse set"
            ),
            "batch_points": batch_points,
            "device": str(device),
        }
        if paths.dimension == 3:
            configuration["uniform_morse_sets"] = _artifact_signature(
                paths.uniform / "morse_sets",
                run_root=paths.run,
                cache=cache,
            )
    elif stage == "adaptive":
        configuration = {
            "subdiv_init": resolution.adaptive_init,
            "subdiv_min": resolution.adaptive_min,
            "subdiv_max": resolution.adaptive_max,
            "subdiv_limit": SUBDIV_LIMIT,
            "padding": PADDING,
            "requested_topology_only": adaptive_topology_only,
            "conley_error_fallback": True,
            "cmgdb_binary": _cmgdb_binary_signature(paths, cache),
        }
    elif stage == "stats":
        configuration = {
            "uniform_precondition": "exactly two minimal Morse nodes",
            "adaptive_precondition": "exactly two minimal Morse nodes",
            "root_precondition": "two distinct uniform minimal-node basins",
            "statistics_algorithm": "condition on nonzero truth labels",
        }
    elif stage == "render":
        configuration = {
            "formats": ["pdf", "png"],
            "projection_min_box_side_frac": (
                min_box_side_frac if paths.dimension == 3 else None
            ),
            "projection_pairs": (
                [[0, 1], [0, 2], [1, 2]] if paths.dimension == 3 else None
            ),
            "cubical_3d_view": paths.dimension == 3,
            "cubical_3d_basename": (
                "morse_sets_cubical_3d" if paths.dimension == 3 else None
            ),
        }
    else:
        raise ValueError(f"unknown stage {stage!r}")

    return {
        "provenance_schema_version": STAGE_PROVENANCE_SCHEMA_VERSION,
        "stage_algorithm_version": STAGE_ALGORITHM_VERSION[stage],
        "study": study,
        "upstream_output_fingerprints": upstream,
        "configuration": configuration,
    }


def _stamp_stage_provenance(
    paths: DimensionPaths,
    stage: str,
    direct_inputs: dict[str, Any],
    *,
    cache: dict[tuple[str, int, int], str],
    legacy_adoption: bool = False,
) -> None:
    marker = _load_stage_marker(paths, stage) or {
        "stage": stage,
        "dimension": paths.dimension,
        "seed": SEED,
    }
    marker.pop("invalidated", None)
    marker["schema_version"] = STAGE_PROVENANCE_SCHEMA_VERSION
    marker["provenance"] = {
        "schema_version": STAGE_PROVENANCE_SCHEMA_VERSION,
        "direct_inputs": direct_inputs,
        "direct_input_fingerprint": _json_fingerprint(direct_inputs),
        "outputs": _stage_output_manifest(paths, stage, cache=cache),
        "legacy_artifacts_adopted_after_validation": legacy_adoption,
    }
    _write_json(paths.stage_marker(stage), marker)


def _write_stage_marker(
    paths: DimensionPaths,
    stage: str,
    payload: dict[str, Any],
) -> None:
    _write_json(
        paths.stage_marker(stage),
        {
            "schema_version": STAGE_PROVENANCE_SCHEMA_VERSION,
            "stage": stage,
            "dimension": paths.dimension,
            "seed": SEED,
            **payload,
        },
    )


def _stage_outputs_exist(paths: DimensionPaths, stage: str) -> bool:
    if stage == "train":
        return (paths.models / "autoencoder.pt").is_file() and (
            paths.models / "autoencoder.json"
        ).is_file()
    if stage == "bounds":
        return paths.bounds.is_file()
    if stage == "precompute-coarse":
        return all(
            (paths.coarse_table / name).is_file()
            for name in ("metadata.json", "coarse_values.npy")
        )
    if stage == "uniform":
        return (
            (paths.uniform / "morse_graph").is_file()
            and (paths.uniform / "morse_sets").is_file()
            and paths.stats.is_file()
        )
    if stage == "precompute-fine":
        return all(
            (paths.hierarchical_table / name).is_file()
            for name in (
                "metadata.json",
                "coarse_values.npy",
                "active_coarse_indices.npy",
                "fine_block_values.npy",
                "active_coarse_boxes.npy",
            )
        )
    if stage == "adaptive":
        return all((paths.adaptive / name).is_file() for name in ("morse_graph", "morse_sets"))
    if stage == "stats":
        return paths.stats.is_file() and paths.stage_marker("stats").is_file()
    if stage == "render":
        common = [
            paths.adaptive / "morse_graph.pdf",
            paths.adaptive / "morse_graph.png",
        ]
        if paths.dimension == 1:
            dimension_specific = [
                paths.adaptive / "morse_sets.pdf",
                paths.adaptive / "morse_sets.png",
            ]
        else:
            dimension_specific = [
                paths.adaptive / f"morse_sets_z{i}_z{j}.{extension}"
                for i, j in ((1, 2), (1, 3), (2, 3))
                for extension in ("pdf", "png")
            ]
            dimension_specific.extend(
                paths.adaptive / f"morse_sets_cubical_3d.{extension}"
                for extension in ("pdf", "png")
            )
        return all(path.is_file() for path in (*common, *dimension_specific))
    raise ValueError(f"unknown stage {stage!r}")


def _legacy_training_matches(
    paths: DimensionPaths,
    inputs: ExactInputs,
    *,
    device: torch.device,
) -> bool:
    summary_path = paths.run / "training_summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = summary.get("data", {})
    scheduler = summary.get("scheduler", {})
    return bool(
        # Summaries written before the training module was renamed carry the
        # older "marcio_full_batch" value; both label the same recipe.
        summary.get("training_method") in ("reference_full_batch", "marcio_full_batch")
        and int(summary.get("seed", -1)) == SEED
        and str(summary.get("device")) == str(device)
        and int(summary.get("epochs_completed", -1)) == EPOCHS
        and int(summary.get("checkpoint_epoch", -1)) == EPOCHS
        and summary.get("checkpoint_selection") == "final_epoch"
        and summary.get("objective") == "MSE(D(E(x)), x) + MSE(D(G(E(x))), y)"
        and float(summary.get("optimizer", {}).get("learning_rate", -1.0))
        == LEARNING_RATE
        and scheduler.get("name") == "ReduceLROnPlateau"
        and float(scheduler.get("factor", -1.0)) == 0.5
        and int(scheduler.get("patience", -1)) == 100
        and float(scheduler.get("threshold", -1.0)) == 1e-4
        and float(scheduler.get("min_lr", -1.0)) == 1e-6
        and summary.get("arch") == reference_architecture(paths.dimension).model_dump()
        and int(data.get("n_pairs", -1)) == TRAINING_ROWS
        and int(data.get("high_dims", -1)) == HIGH_DIMENSION
        and data.get("dtype") == "float32"
        and data.get("full_batch") is True
        and data.get("sha256") == inputs.hashes["train_data.csv"]
    )


def _legacy_bounds_match(paths: DimensionPaths, inputs: ExactInputs) -> bool:
    payload = json.loads(paths.bounds.read_text(encoding="utf-8"))
    bounds = _load_bounds(paths.bounds, paths.dimension)
    recorded_hash = payload.get("train_data_sha256", payload.get("source_sha256"))
    checkpoint = paths.models / "autoencoder.pt"
    return bool(
        recorded_hash == inputs.hashes["train_data.csv"]
        and bounds.lower.shape == (paths.dimension,)
        and bounds.upper.shape == (paths.dimension,)
        and paths.bounds.stat().st_mtime_ns >= checkpoint.stat().st_mtime_ns
    )


def _legacy_table_geometry_matches(
    paths: DimensionPaths,
    *,
    fine_required: bool,
) -> bool:
    resolution = RESOLUTIONS[paths.dimension]
    directory = paths.hierarchical_table if fine_required else paths.coarse_table
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    bounds = _load_bounds(paths.bounds, paths.dimension)
    coarse = np.load(directory / "coarse_values.npy", mmap_mode="r")
    expected_shape = (
        *(resolution.coarse_cells_per_axis + 1 for _ in range(paths.dimension)),
        paths.dimension,
    )
    if (
        coarse.shape != expected_shape
        or coarse.dtype != np.float64
        or int(metadata.get("dimension", -1)) != paths.dimension
        or int(metadata.get("coarse_subdiv", -1)) != resolution.uniform_max
        or int(metadata.get("fine_subdiv", -1)) != resolution.adaptive_max
        or metadata.get("padding") is not PADDING
        or not np.array_equal(np.asarray(metadata.get("lower")), bounds.lower)
        or not np.array_equal(np.asarray(metadata.get("upper")), bounds.upper)
    ):
        return False
    if not fine_required:
        return True
    active = np.load(directory / "active_coarse_indices.npy", mmap_mode="r")
    blocks = np.load(directory / "fine_block_values.npy", mmap_mode="r")
    active_boxes = np.load(directory / "active_coarse_boxes.npy", mmap_mode="r")
    fine_corners = 2 ** (
        (resolution.adaptive_max - resolution.uniform_max) // paths.dimension
    ) + 1
    return bool(
        metadata.get("has_fine_blocks") is True
        and active.ndim == 2
        and active.shape[1] == paths.dimension
        and active_boxes.shape == (active.shape[0], 2 * paths.dimension)
        and blocks.shape
        == (
            active.shape[0],
            *(fine_corners for _ in range(paths.dimension)),
            paths.dimension,
        )
        and int(metadata.get("n_active_coarse_cells", -1)) == active.shape[0]
    )


def _legacy_uniform_matches(paths: DimensionPaths, inputs: ExactInputs) -> bool:
    resolution = RESOLUTIONS[paths.dimension]
    graph = MorseGraph.from_dot(paths.uniform / "morse_graph")
    statistics = json.loads(paths.stats.read_text(encoding="utf-8"))
    attractors = sorted(int(node) for node in statistics.get("cmgdb", {}).get(
        "attractor_nodes", []
    ))
    trajectory = statistics.get("trajectory_data", {})
    roots = statistics.get("stable_roots", {})
    return bool(
        len(graph.minimal) == 2
        and sorted(int(node) for node in graph.minimal) == attractors
        and "singleton-all-reachable-Morse-set" in str(statistics.get("method", ""))
        and int(statistics.get("dimension", -1)) == paths.dimension
        and int(statistics.get("seed", -1)) == SEED
        and trajectory.get("sha256") == inputs.hashes["traj_attractors.pkl"]
        and int(trajectory.get("total", -1)) == TRAJECTORY_ROWS
        and roots.get("sha256") == inputs.hashes["stable_solutions.csv"]
        and statistics.get("cmgdb", {}).get("subdivisions")
        == [
            resolution.uniform_init,
            resolution.uniform_min,
            resolution.uniform_max,
        ]
        and int(statistics.get("cmgdb", {}).get("uniform_cells", -1))
        == resolution.uniform_cells
    )


def _legacy_adaptive_matches(
    paths: DimensionPaths,
    *,
    adaptive_topology_only: bool,
) -> bool:
    resolution = RESOLUTIONS[paths.dimension]
    graph = MorseGraph.from_dot(paths.adaptive / "morse_graph")
    if len(graph.minimal) != 2:
        return False
    summary_path = paths.adaptive / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, expected in (
            ("subdiv_init", resolution.adaptive_init),
            ("subdiv_min", resolution.adaptive_min),
            ("subdiv_max", resolution.adaptive_max),
            ("subdiv_limit", SUBDIV_LIMIT),
        ):
            if int(summary.get(key, -1)) != expected:
                return False
    marker = _load_stage_marker(paths, "adaptive")
    status = (marker or {}).get("conley", {}).get("status")
    if adaptive_topology_only:
        return status in {None, "explicit_topology_only_fallback"}
    return status != "explicit_topology_only_fallback"


def _legacy_stage_matches(
    paths: DimensionPaths,
    inputs: ExactInputs,
    stage: str,
    *,
    device: torch.device,
    batch_points: int | str,
    min_box_side_frac: float,
    adaptive_topology_only: bool,
) -> bool:
    try:
        if stage == "train":
            return _legacy_training_matches(paths, inputs, device=device)
        if stage == "bounds":
            return _legacy_bounds_match(paths, inputs)
        if stage == "precompute-coarse":
            return _legacy_table_geometry_matches(paths, fine_required=False)
        if stage == "uniform":
            return _legacy_uniform_matches(paths, inputs)
        if stage == "precompute-fine":
            return _legacy_table_geometry_matches(paths, fine_required=True)
        if stage == "adaptive":
            return _legacy_adaptive_matches(
                paths,
                adaptive_topology_only=adaptive_topology_only,
            )
        if stage == "stats":
            statistics = json.loads(paths.stats.read_text(encoding="utf-8"))
            adaptive = _morse_summary(paths.adaptive / "morse_graph")
            return bool(
                statistics.get("eligible_for_bistable_dimension_table") is True
                and statistics.get("uniform_is_bistable") is True
                and statistics.get("roots_define_two_distinct_attractor_basins") is True
                and adaptive["is_bistable"]
            )
        if stage == "render":
            marker = _load_stage_marker(paths, "render")
            recorded_floor = (marker or {}).get("projection_min_box_side_frac")
            return bool(
                paths.dimension == 1
                or recorded_floor is None
                or float(recorded_floor) == min_box_side_frac
            )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    raise ValueError(f"unknown stage {stage!r}")


def _stage_is_reusable(
    paths: DimensionPaths,
    inputs: ExactInputs,
    stage: str,
    *,
    device: torch.device,
    batch_points: int | str,
    min_box_side_frac: float,
    adaptive_topology_only: bool,
    cache: dict[tuple[str, int, int], str],
    memo: dict[str, bool],
) -> bool:
    cached = memo.get(stage)
    if cached is not None:
        return cached
    if not _stage_outputs_exist(paths, stage):
        memo[stage] = False
        return False
    for dependency in _stage_dependencies(paths, stage):
        if not _stage_is_reusable(
            paths,
            inputs,
            dependency,
            device=device,
            batch_points=batch_points,
            min_box_side_frac=min_box_side_frac,
            adaptive_topology_only=adaptive_topology_only,
            cache=cache,
            memo=memo,
        ):
            memo[stage] = False
            return False

    marker = _load_stage_marker(paths, stage)
    if marker and marker.get("invalidated"):
        memo[stage] = False
        return False
    direct_inputs = _stage_direct_inputs(
        paths,
        inputs,
        stage,
        device=device,
        batch_points=batch_points,
        min_box_side_frac=min_box_side_frac,
        adaptive_topology_only=adaptive_topology_only,
        cache=cache,
    )
    provenance = (marker or {}).get("provenance")
    if isinstance(provenance, dict):
        try:
            outputs = _stage_output_manifest(paths, stage, cache=cache)
            matches = bool(
                provenance.get("schema_version") == STAGE_PROVENANCE_SCHEMA_VERSION
                and provenance.get("direct_input_fingerprint")
                == _json_fingerprint(direct_inputs)
                and provenance.get("outputs", {}).get("fingerprint")
                == outputs["fingerprint"]
            )
        except (FileNotFoundError, OSError, ValueError):
            matches = False
        memo[stage] = matches
        return matches

    if not _legacy_stage_matches(
        paths,
        inputs,
        stage,
        device=device,
        batch_points=batch_points,
        min_box_side_frac=min_box_side_frac,
        adaptive_topology_only=adaptive_topology_only,
    ):
        memo[stage] = False
        return False
    _stamp_stage_provenance(
        paths,
        stage,
        direct_inputs,
        cache=cache,
        legacy_adoption=True,
    )
    memo[stage] = True
    return True


def _is_descendant(paths: DimensionPaths, candidate: str, ancestor: str) -> bool:
    pending = list(_stage_dependencies(paths, candidate))
    seen: set[str] = set()
    while pending:
        dependency = pending.pop()
        if dependency == ancestor:
            return True
        if dependency not in seen:
            seen.add(dependency)
            pending.extend(_stage_dependencies(paths, dependency))
    return False


def _invalidate_stage_and_descendants(paths: DimensionPaths, stage: str) -> None:
    for candidate in STAGE_ORDER:
        if candidate == stage or _is_descendant(paths, candidate, stage):
            _write_json(
                paths.stage_marker(candidate),
                {
                    "schema_version": STAGE_PROVENANCE_SCHEMA_VERSION,
                    "stage": candidate,
                    "dimension": paths.dimension,
                    "seed": SEED,
                    "invalidated": {
                        "by_stage": stage,
                        "reason": "upstream stage scheduled for recomputation",
                    },
                },
            )


def _uniform_graph_outputs_exist(paths: DimensionPaths) -> bool:
    return (paths.uniform / "morse_graph").is_file() and (
        paths.uniform / "morse_sets"
    ).is_file()


def _quarantine_legacy_roa_artifact(paths: DimensionPaths) -> Path | None:
    """Rename the obsolete minimal/LCA ROA archive so it cannot look authoritative."""

    source = paths.uniform / "regions_of_attraction_exact.npz"
    if not source.is_file():
        return None
    target = paths.uniform / "regions_of_attraction_legacy_minimal_lca.npz"
    source_digest = sha256_file(source)
    if target.exists():
        if not target.is_file() or sha256_file(target) != source_digest:
            raise FileExistsError(
                f"cannot quarantine {source}: incompatible target already exists at {target}"
            )
        source.unlink()
    else:
        source.replace(target)
    _write_json(
        paths.uniform / "regions_of_attraction_legacy_minimal_lca.json",
        {
            "schema_version": 1,
            "status": "legacy_not_used_by_dimension_study",
            "artifact": target.name,
            "sha256": source_digest,
            "legacy_method": (
                "minimal-attractor/LCA labels; not the strict complete-"
                "reachable-Morse-set singleton semantics"
            ),
            "authoritative_replacement": (
                "reference_singleton_reachability_queries.npz generated by the "
                "uniform stage"
            ),
        },
    )
    return target


def _require_stage_outputs(paths: DimensionPaths, stage: str) -> None:
    if not _stage_outputs_exist(paths, stage):
        raise FileNotFoundError(
            f"{stage!r} outputs are required for latent dimension {paths.dimension}; "
            f"run --stages {stage} first (run root: {paths.run})"
        )


def _run_training(
    paths: DimensionPaths,
    inputs: ExactInputs,
    *,
    device: torch.device,
    verbose: bool,
) -> None:
    x, y = _load_training_pairs(inputs.train_data)
    started = time.perf_counter()
    result = train_reference_full_batch(
        arch=reference_architecture(paths.dimension),
        x=x,
        y=y,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        device=device,
        output_dir=paths.run,
        scheduler_factor=0.5,
        scheduler_patience=100,
        scheduler_threshold=1e-4,
        scheduler_min_lr=1e-6,
        verbose=verbose,
    )
    _write_stage_marker(
        paths,
        "train",
        {
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
            "checkpoint": str(result.checkpoint_path),
            "final_loss": result.summary["final_epoch_train"]["loss_total"],
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "train_data_sha256": inputs.hashes["train_data.csv"],
            "reproducibility": _training_reproducibility_metadata(device),
        },
    )
    del result, x, y
    gc.collect()


def _run_bounds(
    paths: DimensionPaths,
    inputs: ExactInputs,
    *,
    device: torch.device,
) -> None:
    _require_stage_outputs(paths, "train")
    x, y = _load_training_pairs(inputs.train_data)
    model, _ = _load_model(paths, device=device)
    all_states = np.concatenate((x, y), axis=0)
    bounds = infer_latent_bounds(
        model.encoder,
        all_states,
        epsilon_frac=BOUNDS_EPSILON_FRAC,
        device=device,
    )
    payload = {
        "schema_version": 1,
        "dimension": paths.dimension,
        "lower": np.asarray(bounds.lower, dtype=np.float64).tolist(),
        "upper": np.asarray(bounds.upper, dtype=np.float64).tolist(),
        "epsilon_frac": BOUNDS_EPSILON_FRAC,
        "source": "encoder(train_data current states + next states)",
        "n_encoded_states": int(all_states.shape[0]),
        "train_data_sha256": inputs.hashes["train_data.csv"],
        "device": str(device),
    }
    _write_json(paths.bounds, payload)
    _write_stage_marker(paths, "bounds", payload)
    del model, all_states, x, y
    gc.collect()


def _run_precompute_coarse(
    paths: DimensionPaths,
    *,
    device: torch.device,
    batch_points: int | str,
) -> None:
    _require_stage_outputs(paths, "train")
    _require_stage_outputs(paths, "bounds")
    resolution = RESOLUTIONS[paths.dimension]
    bounds = _load_bounds(paths.bounds, paths.dimension)
    model, _ = _load_model(paths, device=device)
    started = time.perf_counter()
    box_map = HierarchicalPrecomputedBoxMap.precompute_coarse(
        model.latent_map,
        lower=bounds.lower,
        upper=bounds.upper,
        coarse_subdiv=resolution.uniform_max,
        fine_subdiv=resolution.adaptive_max,
        padding=PADDING,
        batch_points=batch_points,
        device=device,
    )
    box_map.save(paths.coarse_table)
    # A forced coarse rebuild intentionally invalidates any older sparse blocks.
    for stale_name in (
        "active_coarse_indices.npy",
        "fine_block_values.npy",
        "active_coarse_boxes.npy",
    ):
        (paths.coarse_table / stale_name).unlink(missing_ok=True)
    payload = {
        "duration_seconds": time.perf_counter() - started,
        "device": str(device),
        "batch_points": batch_points,
        "neural_evaluations_during_cmgdb": 0,
        **_table_metadata(paths.coarse_table),
    }
    _write_stage_marker(paths, "precompute-coarse", payload)
    del box_map, model
    gc.collect()


def _run_lookup_cmgdb(
    box_map: HierarchicalPrecomputedBoxMap,
    bounds: LatentBounds,
    *,
    subdiv_init: int,
    subdiv_min: int,
    subdiv_max: int,
    compute_conley: bool,
    fallback_to_topology_on_conley_error: bool = False,
) -> tuple[Any, Any, float, dict[str, Any]]:
    """Run lookup-only CMGDB and record the Conley-annotation status.

    ``ComputeConleyMorseGraph`` first computes exactly the same Morse and map
    graphs as ``ComputeMorseGraph`` and then attaches a Conley index to every
    Morse node.  Consequently, disabling or falling back from Conley affects
    only those annotations; topology, Morse boxes, and basin reachability are
    invariant.
    """

    model = CMGDB.Model(
        subdiv_min,
        subdiv_max,
        subdiv_init,
        SUBDIV_LIMIT,
        bounds.lower.tolist(),
        bounds.upper.tolist(),
        box_map,
    )
    if not hasattr(model, "set_batch_map"):
        raise RuntimeError(
            "this study requires CMGDB.Model.set_batch_map; install the pinned "
            "cmgdb fork wheel before running the large lookup-only computation"
        )
    model.set_batch_map(box_map.batch)
    started = time.perf_counter()
    if compute_conley:
        try:
            morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)
            conley_status: dict[str, Any] = {
                "requested": True,
                "computed": True,
                "status": "computed",
                "routine": "CMGDB.ComputeConleyMorseGraph",
            }
        except Exception as error:
            if not fallback_to_topology_on_conley_error:
                raise
            morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
            conley_status = {
                "requested": True,
                "computed": False,
                "status": "failed_then_topology_only",
                "routine": "CMGDB.ComputeMorseGraph",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
    else:
        morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
        conley_status = {
            "requested": False,
            "computed": False,
            "status": "topology_only",
            "routine": "CMGDB.ComputeMorseGraph",
        }
    duration = time.perf_counter() - started
    conley_status["topology_morse_sets_map_graph_and_basins_invariant"] = True
    has_cache = getattr(map_graph, "has_cache", None)
    if callable(has_cache) and not bool(has_cache()):
        raise RuntimeError(
            "CMGDB did not retain its batched MapGraph cache; refusing an implicit "
            "per-cell callback fallback"
        )
    return morse_graph, map_graph, duration, conley_status


def _map_graph_cache_metadata(map_graph: Any) -> dict[str, Any]:
    has_cache = getattr(map_graph, "has_cache", None)
    edge_count = getattr(map_graph, "num_cached_edges", None)
    return {
        "map_cells": int(map_graph.num_vertices()),
        "has_batch_cache": bool(has_cache()) if callable(has_cache) else None,
        "cached_edges": int(edge_count()) if callable(edge_count) else None,
        "CMGDB_MAPGRAPH_RESERVE_EDGES": os.environ.get(
            "CMGDB_MAPGRAPH_RESERVE_EDGES"
        ),
    }


@dataclass(frozen=True)
class UniformPointCells:
    """Closed-cell candidates for points on a uniform cyclic CMGDB grid."""

    flat_cell_ids: NDArray[np.int64]
    offsets: NDArray[np.int64]

    @property
    def n_points(self) -> int:
        return int(self.offsets.size - 1)

    def candidates(self, point_index: int) -> NDArray[np.int64]:
        return self.flat_cell_ids[
            int(self.offsets[point_index]) : int(self.offsets[point_index + 1])
        ]


def _uniform_point_cells(
    points: NDArray[np.float64],
    bounds: LatentBounds,
    resolution: Resolution,
) -> UniformPointCells:
    """Return every closed uniform cell containing each point.

    Interior points have one candidate. A point exactly on an internal grid
    face has candidates on both sides, matching the archived closed-box membership
    checks. Points outside the CMGDB domain have no candidates.
    """

    values = np.asarray(points, dtype=np.float64)
    dimension = resolution.dimension
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"points must have shape (n, {dimension}); got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must contain only finite values")

    n_per_axis = resolution.coarse_cells_per_axis
    shape = np.full(dimension, n_per_axis, dtype=np.int64)
    span = bounds.upper - bounds.lower
    outside = np.any(
        (values < bounds.lower[None, :]) | (values > bounds.upper[None, :]),
        axis=1,
    )
    flat_rows: list[NDArray[np.int64]] = []
    offsets = np.zeros(values.shape[0] + 1, dtype=np.int64)
    for point_index, point in enumerate(values):
        if outside[point_index]:
            offsets[point_index + 1] = offsets[point_index]
            continue
        clipped = np.clip(point, bounds.lower, bounds.upper)
        scaled = (clipped - bounds.lower) / span * shape
        base = np.minimum(np.floor(scaled).astype(np.int64), shape - 1)
        nearest = np.rint(scaled).astype(np.int64)
        reconstructed = bounds.lower + nearest * span / shape
        internal_boundary = (
            (nearest > 0)
            & (nearest < shape)
            & (clipped == reconstructed)
        )
        choices = [
            (int(nearest[axis] - 1), int(nearest[axis]))
            if internal_boundary[axis]
            else (int(base[axis]),)
            for axis in range(dimension)
        ]
        bins = np.asarray(list(product(*choices)), dtype=np.int64)
        cell_ids = np.unique(cmgdb_morton_cell_indices(bins, shape))
        flat_rows.append(cell_ids)
        offsets[point_index + 1] = offsets[point_index] + cell_ids.size

    flat = (
        np.concatenate(flat_rows).astype(np.int64, copy=False)
        if flat_rows
        else np.empty(0, dtype=np.int64)
    )
    return UniformPointCells(flat_cell_ids=flat, offsets=offsets)


def _native_singleton_reachability(
    map_graph: Any,
    morse_graph: Any,
    query_cell_ids: NDArray[np.int64],
) -> NDArray[np.int32]:
    query = np.asarray(query_cell_ids, dtype=np.int64)
    if query.ndim != 1:
        raise ValueError("query_cell_ids must be one-dimensional")
    native = getattr(CMGDB, "MorseSingletonReachability", None)
    if not callable(native):
        raise RuntimeError(
            "this study requires CMGDB.MorseSingletonReachability; rebuild the "
            "pinned cmgdb fork wheel before computing archive-equivalent basins"
        )
    result = native(map_graph, morse_graph, query)
    if (
        not isinstance(result, np.ndarray)
        or result.dtype != np.int32
        or result.shape != query.shape
        or not result.flags.c_contiguous
    ):
        raise TypeError(
            "CMGDB.MorseSingletonReachability must return a C-contiguous "
            f"int32 array shaped {query.shape}; got "
            f"{type(result).__name__}, {getattr(result, 'dtype', None)}, "
            f"{getattr(result, 'shape', None)}"
        )
    return result


def _point_basin_labels(
    singleton_by_candidate: NDArray[np.int32],
    cells: UniformPointCells,
    *,
    negative_attractor: int,
    positive_attractor: int,
) -> NDArray[np.int32]:
    """Apply the archived negative-basin-first closed-box classification."""

    labels = np.full(cells.n_points, OUTSIDE, dtype=np.int32)
    for point_index in range(cells.n_points):
        start = int(cells.offsets[point_index])
        stop = int(cells.offsets[point_index + 1])
        candidates = singleton_by_candidate[start:stop]
        if np.any(candidates == negative_attractor):
            labels[point_index] = negative_attractor
        elif np.any(candidates == positive_attractor):
            labels[point_index] = positive_attractor
    return labels


def _root_attractor_label(
    singleton_by_candidate: NDArray[np.int32],
    cells: UniformPointCells,
    point_index: int,
    attractors: Sequence[int],
) -> int:
    start = int(cells.offsets[point_index])
    stop = int(cells.offsets[point_index + 1])
    candidates = singleton_by_candidate[start:stop]
    matches = sorted({int(value) for value in candidates if int(value) in attractors})
    if len(matches) != 1:
        raise ValueError(
            f"encoded stable root {point_index} matches attractors {matches}; "
            "expected exactly one singleton-all-Morse basin"
        )
    return matches[0]


def _morse_attractors(morse_graph: Any) -> list[int]:
    """Return minimal Morse nodes after reading the edge relation only once.

    CMGDB's ``MorseGraph.adjacencies`` reconstructs the transitive reduction
    on every call.  That is prohibitively expensive for the large 3-D graph.
    A node is a sink in the unreduced reachability relation exactly when it is
    a sink in its transitive reduction, so prefer the cheaper unreduced edge
    list and fall back to one reduced-edge computation for older CMGDB builds.
    """

    vertices_method = getattr(morse_graph, "vertices", None)
    if callable(vertices_method):
        vertices = [int(vertex) for vertex in vertices_method()]
    else:
        vertices = list(range(int(morse_graph.num_vertices())))

    edge_method = getattr(morse_graph, "edges_unreduced", None)
    if not callable(edge_method):
        edge_method = getattr(morse_graph, "edges", None)
    if not callable(edge_method):
        raise TypeError("Morse graph must provide edges_unreduced() or edges()")

    non_attractors = {
        int(source)
        for source, target in edge_method()
        if int(source) != int(target)
    }
    return [vertex for vertex in vertices if vertex not in non_attractors]


def _require_exactly_two_minimal_attractors(morse_graph: Any) -> list[int]:
    """Enforce the archived bistable uniform-graph precondition."""

    attractors = _morse_attractors(morse_graph)
    if len(attractors) != 2:
        raise ValueError(
            "Archive-equivalent basin statistics require exactly two minimal "
            f"attracting Morse nodes; the uniform graph has {len(attractors)}: "
            f"{attractors}"
        )
    return attractors


def _compute_live_reference_statistics(
    paths: DimensionPaths,
    inputs: ExactInputs,
    *,
    device: torch.device,
    bounds: LatentBounds,
    resolution: Resolution,
    map_graph: Any,
    morse_graph: Any,
    attractors: Sequence[int] | None = None,
) -> tuple[dict[str, Any], float]:
    """Query only archived points while preserving the archived exact basin rule."""

    if attractors is None:
        attractors = _require_exactly_two_minimal_attractors(morse_graph)
    else:
        attractors = [int(node) for node in attractors]
        if len(attractors) != 2:
            raise ValueError(
                "Archive-equivalent basin statistics require exactly two minimal "
                f"attracting Morse nodes; got {attractors}"
            )

    points, truth = _load_trajectory_labels(inputs.trajectory_labels)
    roots = _load_stable_roots(inputs.stable_roots)
    model, _ = _load_model(paths, device=device)
    encoded_points = _encode_numpy(model.encoder, points, device=device)
    encoded_roots = _encode_numpy(model.encoder, roots, device=device)
    point_cells = _uniform_point_cells(encoded_points, bounds, resolution)
    root_cells = _uniform_point_cells(encoded_roots, bounds, resolution)

    all_candidate_ids = np.concatenate(
        (point_cells.flat_cell_ids, root_cells.flat_cell_ids)
    )
    unique_cell_ids, inverse = np.unique(all_candidate_ids, return_inverse=True)
    query_started = time.perf_counter()
    singleton_by_unique_cell = _native_singleton_reachability(
        map_graph,
        morse_graph,
        unique_cell_ids,
    )
    query_duration = time.perf_counter() - query_started
    singleton_by_candidate = singleton_by_unique_cell[inverse]
    split = point_cells.flat_cell_ids.size
    point_singletons = np.asarray(singleton_by_candidate[:split], dtype=np.int32)
    root_singletons = np.asarray(singleton_by_candidate[split:], dtype=np.int32)

    negative_attractor = _root_attractor_label(
        root_singletons,
        root_cells,
        0,
        attractors,
    )
    positive_attractor = _root_attractor_label(
        root_singletons,
        root_cells,
        1,
        attractors,
    )
    if negative_attractor == positive_attractor:
        raise ValueError("the encoded stable roots map to the same attracting Morse node")

    predicted = _point_basin_labels(
        point_singletons,
        point_cells,
        negative_attractor=negative_attractor,
        positive_attractor=positive_attractor,
    )
    statistics = compute_chafee_basin_statistics(
        truth,
        predicted,
        negative_basin_label=negative_attractor,
        positive_basin_label=positive_attractor,
    )
    query_path = paths.uniform / "reference_singleton_reachability_queries.npz"
    np.savez_compressed(
        query_path,
        queried_cell_ids=unique_cell_ids,
        singleton_node_by_queried_cell=singleton_by_unique_cell,
        point_candidate_cell_ids=point_cells.flat_cell_ids,
        point_candidate_offsets=point_cells.offsets,
        point_singleton_nodes=point_singletons,
        point_basin_labels=predicted,
        root_candidate_cell_ids=root_cells.flat_cell_ids,
        root_candidate_offsets=root_cells.offsets,
        root_singleton_nodes=root_singletons,
        encoded_stable_roots=encoded_roots,
    )
    np.save(paths.run / "trajectory_basin_labels.npy", predicted)
    np.save(paths.run / "encoded_stable_roots.npy", encoded_roots)
    count_by_label = {
        str(label): int(count)
        for label, count in sorted(Counter(predicted.tolist()).items())
    }
    payload = {
        "schema_version": 2,
        "dimension": paths.dimension,
        "seed": SEED,
        "method": (
            "Exact archived singleton-all-reachable-Morse-set basin semantics "
            "on uniform CMGDB graph"
        ),
        "trajectory_data": {
            "path": str(inputs.trajectory_labels.resolve()),
            "sha256": inputs.hashes["traj_attractors.pkl"],
            "total": TRAJECTORY_ROWS,
            "label_counts": {
                str(label): count
                for label, count in EXPECTED_TRAJECTORY_LABEL_COUNTS.items()
            },
        },
        "stable_roots": {
            "path": str(inputs.stable_roots.resolve()),
            "sha256": inputs.hashes["stable_solutions.csv"],
            "encoded": encoded_roots.tolist(),
            "uniform_candidate_cell_ids": [
                root_cells.candidates(index).tolist() for index in range(2)
            ],
            "negative_basin_label": negative_attractor,
            "positive_basin_label": positive_attractor,
        },
        "cmgdb": {
            "subdivisions": [
                resolution.uniform_init,
                resolution.uniform_min,
                resolution.uniform_max,
            ],
            "uniform_cells": resolution.uniform_cells,
            "morse_nodes": int(morse_graph.num_vertices()),
            "attractor_nodes": attractors,
            "queried_uniform_cells": int(unique_cell_ids.size),
        },
        "classification": {
            "rule": (
                "complete reachable Morse-node set equals exactly the "
                "corresponding singleton attractor"
            ),
            "native_query": "CMGDB.MorseSingletonReachability",
            "closed_cell_boundary_policy": (
                "negative basin first, then positive basin, matching the archived loop"
            ),
            "query_artifact": str(query_path.relative_to(paths.run)),
            "counts_by_point_label": count_by_label,
        },
        "statistics": {
            "total_trajectories": statistics.total_trajectories,
            "excluded_zero_trajectories": statistics.excluded_zero_trajectories,
            "conditioned_trajectories": statistics.conditioned_trajectories,
            "counts": statistics.counts(),
            "percentages": statistics.percentages(),
        },
    }
    _write_json(paths.stats, payload)
    del model, points, truth, roots, encoded_points, encoded_roots
    gc.collect()
    return payload, query_duration


def _run_uniform(
    paths: DimensionPaths,
    inputs: ExactInputs,
    *,
    device: torch.device,
) -> None:
    _require_stage_outputs(paths, "bounds")
    _require_stage_outputs(paths, "precompute-coarse")
    resolution = RESOLUTIONS[paths.dimension]
    bounds = _load_bounds(paths.bounds, paths.dimension)
    box_map = HierarchicalPrecomputedBoxMap.load(paths.coarse_table, mmap_mode="r")
    morse_graph, map_graph, duration, conley_status = _run_lookup_cmgdb(
        box_map,
        bounds,
        subdiv_init=resolution.uniform_init,
        subdiv_min=resolution.uniform_min,
        subdiv_max=resolution.uniform_max,
        compute_conley=False,
    )
    conley_status["status"] = "deferred_to_adaptive"
    conley_status["reason"] = (
        "the uniform graph is auxiliary to the basin computation; Conley "
        "homology is computed only for the final adaptive graph"
    )
    if int(map_graph.num_vertices()) != resolution.uniform_cells:
        raise ValueError(
            f"uniform CMGDB returned {int(map_graph.num_vertices())} cells; "
            f"expected {resolution.uniform_cells}"
        )
    attractors = _require_exactly_two_minimal_attractors(morse_graph)
    dot_path, csv_path = save_morse_graph_artifacts(morse_graph, paths.uniform)
    statistics_payload, query_duration = _compute_live_reference_statistics(
        paths,
        inputs,
        device=device,
        bounds=bounds,
        resolution=resolution,
        map_graph=map_graph,
        morse_graph=morse_graph,
        attractors=attractors,
    )
    payload = {
        "duration_seconds": duration,
        "reference_reachability_query_seconds": query_duration,
        "subdiv_init": resolution.uniform_init,
        "subdiv_min": resolution.uniform_min,
        "subdiv_max": resolution.uniform_max,
        "subdiv_limit": SUBDIV_LIMIT,
        "padding": PADDING,
        "callback": "persisted HierarchicalPrecomputedBoxMap coarse lookup",
        "callback_neural_evaluations": 0,
        "dot_path": str(dot_path),
        "morse_sets_path": str(csv_path),
        "basin_statistics_path": str(paths.stats),
        "basin_method": statistics_payload["method"],
        "conley": conley_status,
        **_map_graph_cache_metadata(map_graph),
        **_morse_summary(dot_path),
    }
    _write_stage_marker(paths, "uniform", payload)
    del box_map, morse_graph, map_graph
    gc.collect()


def _all_uniform_boxes(bounds: LatentBounds, resolution: Resolution) -> NDArray[np.float64]:
    if resolution.dimension != 1:
        raise ValueError("full-grid fine-block selection is used only for the 1-D study")
    n = resolution.coarse_cells_per_axis
    edges = np.linspace(bounds.lower[0], bounds.upper[0], n + 1, dtype=np.float64)
    return np.column_stack((edges[:-1], edges[1:]))


def _recurrent_uniform_boxes(
    morse_sets_path: Path,
    *,
    dimension: int,
    bounds: LatentBounds,
    resolution: Resolution,
) -> tuple[NDArray[np.float64], dict[int, int]]:
    data = np.loadtxt(morse_sets_path, delimiter=",", ndmin=2, dtype=np.float64)
    expected_columns = 2 * dimension + 1
    if data.ndim != 2 or data.shape[1] != expected_columns:
        raise ValueError(
            f"{morse_sets_path} has shape {data.shape}; expected (*, {expected_columns})"
        )
    labels_raw = data[:, -1]
    if not np.all(labels_raw == np.rint(labels_raw)):
        raise ValueError(f"{morse_sets_path} contains non-integer Morse labels")
    labels = labels_raw.astype(np.int64)
    boxes = np.unique(data[:, : 2 * dimension], axis=0)
    expected_width = (bounds.upper - bounds.lower) / resolution.coarse_cells_per_axis
    widths = boxes[:, dimension:] - boxes[:, :dimension]
    if not np.allclose(widths, expected_width, rtol=1e-7, atol=1e-11):
        raise ValueError(
            "the uniform pilot Morse sets contain cells that are not at the "
            f"level-{resolution.uniform_max} grid width"
        )
    counts = {int(label): int(count) for label, count in Counter(labels.tolist()).items()}
    return boxes, counts


def _run_precompute_fine(
    paths: DimensionPaths,
    *,
    device: torch.device,
    batch_points: int | str,
) -> None:
    _require_stage_outputs(paths, "train")
    _require_stage_outputs(paths, "bounds")
    _require_stage_outputs(paths, "precompute-coarse")
    resolution = RESOLUTIONS[paths.dimension]
    bounds = _load_bounds(paths.bounds, paths.dimension)
    if paths.dimension == 1:
        active_boxes = _all_uniform_boxes(bounds, resolution)
        selection = "all level-8 cells (complete 1-D fine-grid coverage)"
        morse_label_counts: dict[int, int] | None = None
    else:
        if not _uniform_graph_outputs_exist(paths):
            raise FileNotFoundError(
                f"uniform pilot Morse artifacts are required in {paths.uniform}"
            )
        active_boxes, morse_label_counts = _recurrent_uniform_boxes(
            paths.uniform / "morse_sets",
            dimension=paths.dimension,
            bounds=bounds,
            resolution=resolution,
        )
        selection = "union of every level-24 cell in every uniform-pilot Morse set"

    # Load into RAM before an in-place save so the existing 407 MB coarse file
    # can be safely reused without duplicating it on disk.
    box_map = HierarchicalPrecomputedBoxMap.load(paths.coarse_table, mmap_mode=None)
    model, _ = _load_model(paths, device=device)
    started = time.perf_counter()
    box_map.precompute_fine_blocks(
        model.latent_map,
        active_boxes,
        batch_points=batch_points,
        device=device,
    )
    box_map.save(paths.hierarchical_table)
    np.save(paths.hierarchical_table / "active_coarse_boxes.npy", active_boxes)
    payload = {
        "duration_seconds": time.perf_counter() - started,
        "device": str(device),
        "batch_points": batch_points,
        "selection": selection,
        "uniform_morse_label_box_counts": morse_label_counts,
        "n_active_coarse_cells": int(active_boxes.shape[0]),
        "fine_corners_per_active_axis": box_map.fine_cells_per_coarse_axis + 1,
        "neural_evaluations_during_cmgdb": 0,
        **_table_metadata(paths.hierarchical_table),
    }
    _write_stage_marker(paths, "precompute-fine", payload)
    del box_map, model, active_boxes
    gc.collect()


def _run_adaptive(
    paths: DimensionPaths,
    *,
    topology_only: bool = False,
) -> None:
    _require_stage_outputs(paths, "bounds")
    _require_stage_outputs(paths, "precompute-fine")
    resolution = RESOLUTIONS[paths.dimension]
    bounds = _load_bounds(paths.bounds, paths.dimension)
    box_map = HierarchicalPrecomputedBoxMap.load(paths.hierarchical_table, mmap_mode="r")
    morse_graph, map_graph, duration, conley_status = _run_lookup_cmgdb(
        box_map,
        bounds,
        subdiv_init=resolution.adaptive_init,
        subdiv_min=resolution.adaptive_min,
        subdiv_max=resolution.adaptive_max,
        compute_conley=not topology_only,
        fallback_to_topology_on_conley_error=True,
    )
    if topology_only:
        conley_status["status"] = "explicit_topology_only_fallback"
        conley_status["reason"] = (
            "requested with --adaptive-topology-only after identifying a "
            "pathological Conley-index Smith-normal-form computation"
        )
    dot_path, csv_path = save_morse_graph_artifacts(morse_graph, paths.adaptive)
    payload = {
        "duration_seconds": duration,
        "subdiv_init": resolution.adaptive_init,
        "subdiv_min": resolution.adaptive_min,
        "subdiv_max": resolution.adaptive_max,
        "subdiv_limit": SUBDIV_LIMIT,
        "padding": PADDING,
        "callback": "persisted HierarchicalPrecomputedBoxMap dense-coarse/sparse-fine lookup",
        "callback_neural_evaluations": 0,
        "dot_path": str(dot_path),
        "morse_sets_path": str(csv_path),
        "conley": conley_status,
        **_map_graph_cache_metadata(map_graph),
        **_morse_summary(dot_path),
    }
    _write_stage_marker(paths, "adaptive", payload)
    del box_map, morse_graph, map_graph
    gc.collect()


def _run_statistics(paths: DimensionPaths) -> None:
    _require_stage_outputs(paths, "uniform")
    _require_stage_outputs(paths, "adaptive")
    payload = json.loads(paths.stats.read_text(encoding="utf-8"))
    method = str(payload.get("method", ""))
    if "singleton-all-reachable-Morse-set" not in method:
        raise ValueError(
            f"{paths.stats} was not computed with the archived strict "
            "singleton-all-reachable-Morse-set semantics"
        )
    adaptive_summary = _morse_summary(paths.adaptive / "morse_graph")
    attractors = [int(node) for node in payload["cmgdb"]["attractor_nodes"]]
    root_labels = {
        int(payload["stable_roots"]["negative_basin_label"]),
        int(payload["stable_roots"]["positive_basin_label"]),
    }
    payload["uniform_is_bistable"] = len(attractors) == 2
    payload["adaptive_graph"] = adaptive_summary
    payload["roots_define_two_distinct_attractor_basins"] = (
        len(root_labels) == 2 and root_labels.issubset(set(attractors))
    )
    payload["eligible_for_bistable_dimension_table"] = bool(
        payload["uniform_is_bistable"]
        and adaptive_summary["is_bistable"]
        and payload["roots_define_two_distinct_attractor_basins"]
    )
    _write_json(paths.stats, payload)
    _write_stage_marker(paths, "stats", payload)


def _run_render(paths: DimensionPaths, *, min_box_side_frac: float) -> None:
    _require_stage_outputs(paths, "bounds")
    _require_stage_outputs(paths, "adaptive")
    bounds = _load_bounds(paths.bounds, paths.dimension)
    graph_paths = render_morse_graph_from_dot(
        paths.adaptive / "morse_graph",
        paths.adaptive,
        basename="morse_graph",
        formats=("pdf", "png"),
    )
    if paths.dimension == 1:
        set_outputs: Any = render_morse_sets_from_csv(
            paths.adaptive / "morse_sets",
            paths.adaptive,
            bounds_lower=bounds.lower,
            bounds_upper=bounds.upper,
            basename="morse_sets",
            formats=("pdf", "png"),
        )
        cubical_outputs: list[Path] | None = None
    else:
        set_outputs = render_morse_set_projections_from_csv(
            paths.adaptive / "morse_sets",
            paths.adaptive,
            bounds_lower=bounds.lower,
            bounds_upper=bounds.upper,
            basename="morse_sets",
            formats=("pdf", "png"),
            min_box_side_frac=min_box_side_frac,
        )
        cubical_outputs = render_morse_sets_3d_cubical_from_csv(
            paths.adaptive / "morse_sets",
            paths.adaptive,
            basename="morse_sets_cubical_3d",
            formats=("pdf", "png"),
        )
    payload = {
        "graph_outputs": [str(path) for path in graph_paths],
        "morse_set_outputs": (
            [str(path) for path in set_outputs]
            if isinstance(set_outputs, list)
            else {
                f"{pair[0]},{pair[1]}": [str(path) for path in outputs]
                for pair, outputs in set_outputs.items()
            }
        ),
        "projection_min_box_side_frac": (
            min_box_side_frac if paths.dimension == 3 else None
        ),
        "morse_sets_cubical_3d_outputs": (
            [str(path) for path in cubical_outputs]
            if cubical_outputs is not None
            else None
        ),
    }
    _write_stage_marker(paths, "render", payload)


def _parse_stages(tokens: Iterable[str]) -> tuple[str, ...]:
    expanded = [
        stage.strip()
        for token in tokens
        for stage in token.split(",")
        if stage.strip()
    ]
    unknown = sorted(set(expanded) - VALID_STAGES)
    if unknown:
        raise ValueError(f"unknown stages {unknown}; choose from {sorted(VALID_STAGES)}")
    if "all" in expanded:
        return STAGE_ORDER
    # Preserve dependency order, independent of command-line ordering.
    requested = set(expanded)
    return tuple(stage for stage in STAGE_ORDER if stage in requested)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _batch_points(value: str) -> int | str:
    return value if value == "auto" else _positive_int(value)


def _study_config(
    paths: DimensionPaths,
    inputs: ExactInputs,
    *,
    device: torch.device,
    batch_points: int | str,
    adaptive_topology_only: bool,
) -> dict[str, Any]:
    resolution = RESOLUTIONS[paths.dimension]
    return {
        "schema_version": STAGE_PROVENANCE_SCHEMA_VERSION,
        "dimension": paths.dimension,
        "seed": SEED,
        "architecture": reference_architecture(paths.dimension).model_dump(),
        "training": {
            "method": "reference-faithful fixed full batch",
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "factor": 0.5,
                "patience": 100,
                "threshold": 1e-4,
                "min_lr": 1e-6,
            },
            "reproducibility": _training_reproducibility_metadata(device),
        },
        "bounds_epsilon_frac": BOUNDS_EPSILON_FRAC,
        "cmgdb": {
            **asdict(resolution),
            "subdiv_limit": SUBDIV_LIMIT,
            "padding": PADDING,
            "uniform_cells": resolution.uniform_cells,
            "batch_points": batch_points,
            "callback_policy": "persisted lookup only; zero neural evaluations",
            "conley_policy": {
                "uniform": "deferred_to_adaptive",
                "adaptive": (
                    "explicit_topology_only_fallback"
                    if adaptive_topology_only
                    else "attempt_conley_with_topology_only_error_fallback"
                ),
            },
            "CMGDB_MAPGRAPH_RESERVE_EDGES": os.environ.get(
                "CMGDB_MAPGRAPH_RESERVE_EDGES"
            ),
        },
        "device": str(device),
        "inputs": inputs.provenance(),
    }


def run_dimension(
    dimension: int,
    *,
    stages: Sequence[str],
    inputs: ExactInputs,
    output_root: Path,
    device: torch.device,
    batch_points: int | str,
    force: bool,
    verbose_training: bool,
    min_box_side_frac: float,
    adaptive_topology_only: bool,
) -> None:
    paths = DimensionPaths(output_root=output_root.resolve(), dimension=dimension)
    paths.run.mkdir(parents=True, exist_ok=True)
    _quarantine_legacy_roa_artifact(paths)
    study_config = _study_config(
        paths,
        inputs,
        device=device,
        batch_points=batch_points,
        adaptive_topology_only=adaptive_topology_only,
    )
    _write_json(
        paths.run / "study_config.json",
        study_config,
    )
    runners = {
        "train": lambda: _run_training(
            paths,
            inputs,
            device=device,
            verbose=verbose_training,
        ),
        "bounds": lambda: _run_bounds(paths, inputs, device=device),
        "precompute-coarse": lambda: _run_precompute_coarse(
            paths,
            device=device,
            batch_points=batch_points,
        ),
        "uniform": lambda: _run_uniform(
            paths,
            inputs,
            device=device,
        ),
        "precompute-fine": lambda: _run_precompute_fine(
            paths,
            device=device,
            batch_points=batch_points,
        ),
        "adaptive": lambda: _run_adaptive(
            paths,
            topology_only=adaptive_topology_only,
        ),
        "stats": lambda: _run_statistics(paths),
        "render": lambda: _run_render(
            paths,
            min_box_side_frac=min_box_side_frac,
        ),
    }
    hash_cache: dict[tuple[str, int, int], str] = {}
    reusable_memo: dict[str, bool] = {}
    for stage in stages:
        reusable = False
        if not force:
            reusable = _stage_is_reusable(
                paths,
                inputs,
                stage,
                device=device,
                batch_points=batch_points,
                min_box_side_frac=min_box_side_frac,
                adaptive_topology_only=adaptive_topology_only,
                cache=hash_cache,
                memo=reusable_memo,
            )
        if reusable:
            print(f"[{dimension}D] skip completed stage: {stage}")
            continue
        stale_dependencies = [
            dependency
            for dependency in _stage_dependencies(paths, stage)
            if not _stage_is_reusable(
                paths,
                inputs,
                dependency,
                device=device,
                batch_points=batch_points,
                min_box_side_frac=min_box_side_frac,
                adaptive_topology_only=adaptive_topology_only,
                cache=hash_cache,
                memo=reusable_memo,
            )
        ]
        if stale_dependencies:
            raise RuntimeError(
                f"[{dimension}D] cannot run {stage!r}: prerequisite provenance is "
                f"missing or stale for {stale_dependencies}. Request those stages "
                "in the same invocation (or use --stages all)."
            )
        _invalidate_stage_and_descendants(paths, stage)
        reusable_memo.clear()
        print(f"[{dimension}D] start stage: {stage}")
        runners[stage]()
        direct_inputs = _stage_direct_inputs(
            paths,
            inputs,
            stage,
            device=device,
            batch_points=batch_points,
            min_box_side_frac=min_box_side_frac,
            adaptive_topology_only=adaptive_topology_only,
            cache=hash_cache,
        )
        _stamp_stage_provenance(
            paths,
            stage,
            direct_inputs,
            cache=hash_cache,
        )
        reusable_memo.clear()
        print(f"[{dimension}D] completed stage: {stage}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        choices=sorted(RESOLUTIONS),
        default=sorted(RESOLUTIONS),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        help=f"space- or comma-separated stages; choices: {', '.join(sorted(VALID_STAGES))}",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help=(
            "directory holding the archived reference inputs "
            "(train_data.csv, ci_model_weights.pth, stable_solutions.csv, "
            "traj_attractors.pkl)"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-points", type=_batch_points, default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet-training", action="store_true")
    parser.add_argument(
        "--adaptive-topology-only",
        action="store_true",
        help=(
            "bypass adaptive Conley-index homology after a known pathological "
            "Smith-normal-form run; artifacts record the explicit fallback"
        ),
    )
    parser.add_argument(
        "--cmgdb-reserve-edges",
        type=_positive_int,
        default=None,
        help="optionally sets CMGDB_MAPGRAPH_RESERVE_EDGES for the batched CSR cache",
    )
    parser.add_argument(
        "--projection-min-box-side-frac",
        type=float,
        default=0.0025,
        help="display-only visibility floor for 3-D pair projections",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.projection_min_box_side_frac < 0:
        raise ValueError("--projection-min-box-side-frac must be nonnegative")
    stages = _parse_stages(args.stages)
    inputs = verify_exact_inputs(args.reference_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "input_provenance.json", inputs.provenance())
    print(
        "verified exact archived inputs: "
        f"train={inputs.hashes['train_data.csv']}, "
        f"trajectories={inputs.hashes['traj_attractors.pkl']}, "
        f"roots={inputs.hashes['stable_solutions.csv']}"
    )
    if not stages:
        return 0

    if args.cmgdb_reserve_edges is not None:
        os.environ["CMGDB_MAPGRAPH_RESERVE_EDGES"] = str(args.cmgdb_reserve_edges)
    device = _resolve_device(args.device)
    for dimension in dict.fromkeys(args.dimensions):
        resolution = RESOLUTIONS[dimension]
        if args.cmgdb_max_vertices < resolution.uniform_cells:
            raise ValueError(
                f"--cmgdb-max-vertices={args.cmgdb_max_vertices} is below the "
                f"{dimension}D uniform cell count {resolution.uniform_cells}"
            )
        run_dimension(
            dimension,
            stages=stages,
            inputs=inputs,
            output_root=args.output_root,
            device=device,
            batch_points=args.batch_points,
            force=args.force,
            verbose_training=not args.quiet_training,
            min_box_side_frac=args.projection_min_box_side_frac,
            adaptive_topology_only=args.adaptive_topology_only,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
