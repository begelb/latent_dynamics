"""Audit Chafee--Infante errors for the persisted 1-D, 2-D, and 3-D models.

This script is deliberately render-free and computation-light.  It evaluates
the three saved checkpoints on the same 30,000 one-step pairs used by Marcio's
two-dimensional training run.  It does not train a model, integrate the PDE,
run CMGDB, or perform the dense residual search used by the paper.

The common finite-data audit reports both the two-term training objective and
the latent semiconjugacy residual.  The latter is conditioned on the persisted
minimal attracting blocks and identified by physical sign, so differing Morse
node numbers across latent dimensions cannot silently swap the two attractors.
The existing dense two-dimensional paper result is copied by reference into a
separate protocol section; no analogous dense search has been run for the
one- or three-dimensional models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from latentdynamics.training import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = CODE_ROOT / "output" / "chafee_latent_dimension_study"

TRAIN_DATA = PROJECT_ROOT / "archive" / "marcio" / "scripts" / "train_data.csv"
STABLE_ROOTS = (
    PROJECT_ROOT / "archive" / "marcio" / "scripts" / "stable_solutions.csv"
)
DENSE_D2_ROOT = (
    PROJECT_ROOT
    / "tmp"
    / "tolerance-evaluation-2026-07-25"
    / "results"
    / "chafee_infante_current"
)
DENSE_D2_RESULT = DENSE_D2_ROOT / "dense_sampling.json"
DEFAULT_OUTPUT = OUTPUT_ROOT / "residual_audit.json"

HIGH_DIMENSION = 64
N_PAIRS = 30_000
STEPS_PER_TRAJECTORY = 30
DEFAULT_BATCH_SIZE = 4_096

EXPECTED_SHA256 = {
    "train_data": "890fea29a2d26b31b44bbc4fbd773af0ebf742b08893c27ddaf1bdbf87e30329",
    "stable_roots": "cae0222acb37ae9688e54cb2a1f42ac3777360e49b3919403ec1433363de0586",
    "checkpoint_d1": "f2d1ad7dcc094e4565f25446e613d4b528261012810bb493ef70d1a3977c0f91",
    "checkpoint_d2": "ba47f9ff1ce06da658e052c46ca2e38d8db9a0ffe063ffd997c9d6f547218b79",
    "checkpoint_d3": "bdb0f8a69fe1358ab3d7f3bb2e69f6e6883f92fe83fe61e365bed3e04e1e2bab",
    "blocks_d1": "ff7b5b704974153e5d2c082c09407d437a488df7f3d5639e5df93816f34e6154",
    "blocks_d2_node0": "9d560a34779b203a4fd9a757769ae1997c1fef334b6f9907afe72b47d0b4d06f",
    "blocks_d2_node1": "b1ab3f12bbbb4fb78391033165a0e6621be8e2e2c48a79925f501aa71152163a",
    "blocks_d3": "14979bd3f3cf526e24a7a486822e0c48328b93bfc57d374cf0709682c2370919",
    "dense_d2_result": "1a900ce6215656ee5850c9b8075847cbada321d9ce62aac79a9d532e49902306",
}


@dataclass(frozen=True)
class DimensionSpec:
    """Persisted inputs and physical node mapping for one latent dimension."""

    dimension: int
    model_dir: Path
    checkpoint_sha_key: str
    physical_nodes: dict[str, int]
    block_kind: str
    block_sources: dict[int, Path]
    block_sha_keys: dict[int, str]


DIMENSION_SPECS = {
    1: DimensionSpec(
        dimension=1,
        model_dir=OUTPUT_ROOT / "latent_1d" / "seed_0" / "models",
        checkpoint_sha_key="checkpoint_d1",
        physical_nodes={"negative": 0, "positive": 1},
        block_kind="adaptive_morse_sets_csv",
        block_sources={
            0: OUTPUT_ROOT / "latent_1d" / "seed_0" / "MG_adaptive" / "morse_sets",
            1: OUTPUT_ROOT / "latent_1d" / "seed_0" / "MG_adaptive" / "morse_sets",
        },
        block_sha_keys={0: "blocks_d1", 1: "blocks_d1"},
    ),
    2: DimensionSpec(
        dimension=2,
        model_dir=CODE_ROOT
        / "replay_sources"
        / "chafee_infante"
        / "replay"
        / "models",
        checkpoint_sha_key="checkpoint_d2",
        physical_nodes={"negative": 1, "positive": 0},
        block_kind="dense_paper_protocol_live_block_npz",
        block_sources={
            0: DENSE_D2_ROOT / "block_0.npz",
            1: DENSE_D2_ROOT / "block_1.npz",
        },
        block_sha_keys={
            0: "blocks_d2_node0",
            1: "blocks_d2_node1",
        },
    ),
    3: DimensionSpec(
        dimension=3,
        model_dir=OUTPUT_ROOT / "latent_3d" / "seed_0" / "models",
        checkpoint_sha_key="checkpoint_d3",
        physical_nodes={"negative": 0, "positive": 1},
        block_kind="adaptive_morse_sets_csv",
        block_sources={
            0: OUTPUT_ROOT / "latent_3d" / "seed_0" / "MG_adaptive" / "morse_sets",
            1: OUTPUT_ROOT / "latent_3d" / "seed_0" / "MG_adaptive" / "morse_sets",
        },
        block_sha_keys={0: "blocks_d3", 1: "blocks_d3"},
    ),
}


@dataclass(frozen=True)
class ModelEvaluation:
    """Every finite-data tensor needed by the scalar and witness summaries."""

    reconstruction_error: Tensor
    prediction_error: Tensor
    encoded_current: Tensor
    encoded_next: Tensor
    predicted_latent: Tensor

    @property
    def latent_residual(self) -> Tensor:
        return self.predicted_latent - self.encoded_next

    @property
    def euclidean_latent_residual(self) -> Tensor:
        return torch.linalg.vector_norm(self.latent_residual, dim=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_sha256(path: Path, expected: str, *, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{description} SHA256 mismatch for {path}: "
            f"expected {expected}, observed {actual}"
        )
    return actual


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _source_record(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = _checked_sha256(
        path,
        expected_sha256,
        description="audited input",
    )
    return {
        "path": _relative_path(path),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
    }


def _load_training_pairs(
    path: Path,
) -> tuple[NDArray[np.float64], Tensor, Tensor]:
    raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if raw.shape != (N_PAIRS, 2 * HIGH_DIMENSION):
        raise ValueError(
            f"{path} has shape {raw.shape}; "
            f"expected {(N_PAIRS, 2 * HIGH_DIMENSION)}"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{path} contains non-finite values")
    x = torch.as_tensor(
        np.ascontiguousarray(raw[:, :HIGH_DIMENSION]),
        dtype=torch.float32,
    )
    y = torch.as_tensor(
        np.ascontiguousarray(raw[:, HIGH_DIMENSION:]),
        dtype=torch.float32,
    )
    return raw, x, y


def _load_stable_roots(path: Path) -> NDArray[np.float64]:
    roots = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if roots.shape != (2, HIGH_DIMENSION):
        raise ValueError(
            f"{path} has shape {roots.shape}; expected {(2, HIGH_DIMENSION)}"
        )
    if not np.all(np.isfinite(roots)):
        raise ValueError(f"{path} contains non-finite values")
    if roots[0, 0] >= 0.0 or roots[1, 0] <= 0.0:
        raise ValueError(
            "stable root rows must be ordered physical negative then positive"
        )
    return roots


def _load_csv_blocks(path: Path, dimension: int) -> dict[int, NDArray[np.float64]]:
    rows = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)
    if rows.shape[1] != 2 * dimension + 1:
        raise ValueError(
            f"{path} has shape {rows.shape}; expected {2 * dimension + 1} columns"
        )
    labels = np.rint(rows[:, -1]).astype(np.int64)
    if not np.array_equal(rows[:, -1], labels.astype(np.float64)):
        raise ValueError(f"{path} contains non-integral Morse labels")
    blocks: dict[int, NDArray[np.float64]] = {}
    for node in (0, 1):
        selected = np.ascontiguousarray(rows[labels == node, :-1])
        if selected.shape[0] == 0:
            raise ValueError(f"{path} contains no boxes for expected node {node}")
        if np.any(selected[:, :dimension] >= selected[:, dimension:]):
            raise ValueError(f"{path} contains an invalid box for node {node}")
        blocks[node] = selected
    return blocks


def _load_npz_block(path: Path, dimension: int) -> NDArray[np.float64]:
    with np.load(path) as archive:
        if "block_boxes" not in archive.files:
            raise ValueError(f"{path} does not contain block_boxes")
        boxes = np.asarray(archive["block_boxes"], dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 2 * dimension or boxes.shape[0] == 0:
        raise ValueError(
            f"{path} block_boxes has shape {boxes.shape}; expected (n, {2 * dimension})"
        )
    if not np.all(np.isfinite(boxes)):
        raise ValueError(f"{path} block_boxes contains non-finite values")
    if np.any(boxes[:, :dimension] >= boxes[:, dimension:]):
        raise ValueError(f"{path} contains an invalid block box")
    return np.ascontiguousarray(boxes)


def _load_blocks(
    spec: DimensionSpec,
) -> tuple[dict[int, NDArray[np.float64]], dict[int, dict[str, Any]]]:
    blocks: dict[int, NDArray[np.float64]] = {}
    sources: dict[int, dict[str, Any]] = {}
    if spec.block_kind == "adaptive_morse_sets_csv":
        path = spec.block_sources[0]
        expected = EXPECTED_SHA256[spec.block_sha_keys[0]]
        sources = {
            node: {
                **_source_record(path, expected),
                "kind": spec.block_kind,
            }
            for node in (0, 1)
        }
        blocks = _load_csv_blocks(path, spec.dimension)
    elif spec.block_kind == "dense_paper_protocol_live_block_npz":
        for node in (0, 1):
            path = spec.block_sources[node]
            sources[node] = {
                **_source_record(
                    path,
                    EXPECTED_SHA256[spec.block_sha_keys[node]],
                ),
                "kind": spec.block_kind,
            }
            blocks[node] = _load_npz_block(path, spec.dimension)
    else:
        raise ValueError(f"unsupported block kind {spec.block_kind!r}")
    return blocks, sources


def _evaluate_model(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    *,
    batch_size: int,
) -> ModelEvaluation:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    reconstruction_error: list[Tensor] = []
    prediction_error: list[Tensor] = []
    encoded_current: list[Tensor] = []
    encoded_next: list[Tensor] = []
    predicted_latent: list[Tensor] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            current = x[start:stop]
            next_state = y[start:stop]
            z_current = model.encoder(current)
            z_next = model.encoder(next_state)
            z_predicted = model.latent_map(z_current)
            current_reconstructed = model.decoder(z_current)
            next_predicted = model.decoder(z_predicted)
            reconstruction_error.append(
                (current_reconstructed - current).cpu()
            )
            prediction_error.append((next_predicted - next_state).cpu())
            encoded_current.append(z_current.cpu())
            encoded_next.append(z_next.cpu())
            predicted_latent.append(z_predicted.cpu())

    return ModelEvaluation(
        reconstruction_error=torch.cat(reconstruction_error),
        prediction_error=torch.cat(prediction_error),
        encoded_current=torch.cat(encoded_current),
        encoded_next=torch.cat(encoded_next),
        predicted_latent=torch.cat(predicted_latent),
    )


def _membership_mask(
    points: Tensor,
    boxes: NDArray[np.float64],
) -> Tensor:
    dimension = points.shape[1]
    if boxes.ndim != 2 or boxes.shape[1] != 2 * dimension:
        raise ValueError(
            f"boxes shape {boxes.shape} is incompatible with points {tuple(points.shape)}"
        )
    lower = torch.as_tensor(boxes[:, :dimension], dtype=points.dtype)
    upper = torch.as_tensor(boxes[:, dimension:], dtype=points.dtype)
    return torch.all(
        (points[:, None, :] >= lower[None, :, :])
        & (points[:, None, :] <= upper[None, :, :]),
        dim=2,
    ).any(dim=1)


def _witness(
    index: int,
    raw_pairs: NDArray[np.float64],
    evaluation: ModelEvaluation,
) -> dict[str, Any]:
    residual = evaluation.latent_residual[index]
    return {
        "row_index_zero_based": index,
        "trajectory_index_zero_based": index // STEPS_PER_TRAJECTORY,
        "step_index_zero_based": index % STEPS_PER_TRAJECTORY,
        "x_raw": raw_pairs[index, :HIGH_DIMENSION].tolist(),
        "f_x_raw": raw_pairs[index, HIGH_DIMENSION:].tolist(),
        "E_x": evaluation.encoded_current[index].tolist(),
        "g_E_x": evaluation.predicted_latent[index].tolist(),
        "E_f_x": evaluation.encoded_next[index].tolist(),
        "latent_residual_vector": residual.tolist(),
        "euclidean_residual": float(torch.linalg.vector_norm(residual)),
    }


def _summarize_training_metrics(
    evaluation: ModelEvaluation,
) -> dict[str, float]:
    loss_reconstruction = torch.mean(evaluation.reconstruction_error.square())
    loss_prediction = torch.mean(evaluation.prediction_error.square())
    loss_total = loss_reconstruction + loss_prediction
    loss_semiconjugacy = torch.mean(evaluation.latent_residual.square())
    return {
        "L1_reconstruction_mse": float(loss_reconstruction),
        "L2_decoded_one_step_prediction_mse": float(loss_prediction),
        "L1_plus_L2": float(loss_total),
        "L3_unconditioned_latent_semiconjugacy_mse": float(
            loss_semiconjugacy
        ),
    }


def _validate_physical_node_mapping(
    model: nn.Module,
    stable_roots: NDArray[np.float64],
    blocks: dict[int, NDArray[np.float64]],
    expected_mapping: dict[str, int],
) -> dict[str, Any]:
    roots = torch.as_tensor(stable_roots, dtype=torch.float32)
    with torch.inference_mode():
        encoded = model.encoder(roots).cpu()
    physical_rows = {"negative": 0, "positive": 1}
    memberships: dict[str, list[int]] = {}
    for physical, row in physical_rows.items():
        memberships[physical] = [
            node
            for node, boxes in blocks.items()
            if bool(_membership_mask(encoded[row : row + 1], boxes)[0])
        ]
        expected = [expected_mapping[physical]]
        if memberships[physical] != expected:
            raise ValueError(
                f"{physical} stable root belongs to nodes "
                f"{memberships[physical]}, expected {expected}"
            )
    if expected_mapping["negative"] == expected_mapping["positive"]:
        raise ValueError("physical stable roots must map to distinct Morse nodes")
    return {
        "expected_physical_to_node": expected_mapping,
        "encoded_roots": {
            physical: encoded[row].tolist()
            for physical, row in physical_rows.items()
        },
        "observed_node_memberships": memberships,
        "passed": True,
    }


def _summarize_stored_pair_residuals(
    raw_pairs: NDArray[np.float64],
    evaluation: ModelEvaluation,
    blocks: dict[int, NDArray[np.float64]],
    physical_nodes: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors = evaluation.euclidean_latent_residual
    physical: dict[str, Any] = {}
    for sign in ("negative", "positive"):
        node = physical_nodes[sign]
        mask = _membership_mask(evaluation.encoded_current, blocks[node])
        accepted_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if accepted_indices.numel() == 0:
            raise ValueError(
                f"no stored states encode into the {sign} node {node} block"
            )
        local = torch.argmax(errors[mask])
        index = int(accepted_indices[local])
        physical[sign] = {
            "morse_node": node,
            "block_box_count": int(blocks[node].shape[0]),
            "evaluated_pairs": int(errors.numel()),
            "accepted_pairs": int(accepted_indices.numel()),
            "max_euclidean_residual": float(errors[index]),
            "witness": _witness(index, raw_pairs, evaluation),
        }

    global_index = int(torch.argmax(errors))
    global_result = {
        "evaluated_pairs": int(errors.numel()),
        "max_euclidean_residual": float(errors[global_index]),
        "witness": _witness(global_index, raw_pairs, evaluation),
    }
    return physical, global_result


def _dense_d2_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("example") != "chafee_infante_current":
        raise ValueError(f"unexpected dense d=2 example in {path}")
    node_to_physical = {0: "positive", 1: "negative"}
    by_physical: dict[str, Any] = {}
    for node, physical in node_to_physical.items():
        node_payload = payload["nodes"][str(node)]
        residual = node_payload["residual"]
        by_physical[physical] = {
            "morse_node": node,
            "block_box_count": int(node_payload["n_boxes"]),
            "evaluated_candidates": int(residual["evaluated_samples"]),
            "accepted_candidates": int(residual["accepted_samples"]),
            "sampled_max_euclidean_residual": float(
                residual["sampled_maximum"]
            ),
            "witness": residual["witness"],
        }
    return {
        "name": "existing_dense_d2_paper_residual_protocol",
        "formula": (
            "max over accepted samples of "
            "||g(E(x)) - E(f(x))||_2"
        ),
        "metric": payload["metric"],
        "artifact": _source_record(
            path,
            EXPECTED_SHA256["dense_d2_result"],
        ),
        "paper_reference": {
            "path": "paper/main_KM.tex",
            "table_label": "tab:sampled_residual_tolerance",
        },
        "status_by_dimension": {
            "1": "not_run",
            "2": "existing_result_referenced_not_recomputed",
            "3": "not_run",
        },
        "explicit_scope_note": (
            "No dense d=1 or d=3 residual protocol has been run. "
            "The common 30,000-pair audit is a different, smaller protocol."
        ),
        "dimension_2_by_physical_attractor": by_physical,
    }


def run_audit(*, batch_size: int) -> dict[str, Any]:
    """Run the checksum-guarded finite-pair audit and return its JSON payload."""

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data_record = _source_record(
        TRAIN_DATA,
        EXPECTED_SHA256["train_data"],
    )
    roots_record = _source_record(
        STABLE_ROOTS,
        EXPECTED_SHA256["stable_roots"],
    )
    raw_pairs, x, y = _load_training_pairs(TRAIN_DATA)
    stable_roots = _load_stable_roots(STABLE_ROOTS)

    results: dict[str, Any] = {}
    for dimension in (1, 2, 3):
        spec = DIMENSION_SPECS[dimension]
        checkpoint = spec.model_dir / "autoencoder.pt"
        checkpoint_record = _source_record(
            checkpoint,
            EXPECTED_SHA256[spec.checkpoint_sha_key],
        )
        sidecar = spec.model_dir / "autoencoder.json"
        model, arch = load_checkpoint(spec.model_dir, map_location="cpu")
        if arch.high_dims != HIGH_DIMENSION or arch.low_dims != dimension:
            raise ValueError(
                f"d={dimension} checkpoint architecture is "
                f"{arch.high_dims}->{arch.low_dims}, expected "
                f"{HIGH_DIMENSION}->{dimension}"
            )
        blocks, block_sources = _load_blocks(spec)
        mapping_validation = _validate_physical_node_mapping(
            model,
            stable_roots,
            blocks,
            spec.physical_nodes,
        )
        evaluation = _evaluate_model(
            model,
            x,
            y,
            batch_size=batch_size,
        )
        block_residuals, global_residual = _summarize_stored_pair_residuals(
            raw_pairs,
            evaluation,
            blocks,
            spec.physical_nodes,
        )
        results[str(dimension)] = {
            "latent_dimension": dimension,
            "checkpoint": checkpoint_record,
            "checkpoint_sidecar": {
                "path": _relative_path(sidecar),
                "sha256": _sha256(sidecar),
                "size_bytes": sidecar.stat().st_size,
            },
            "block_sources_by_node": {
                str(node): block_sources[node]
                for node in sorted(block_sources)
            },
            "physical_node_mapping_validation": mapping_validation,
            "training_metrics_on_all_30000_pairs": (
                _summarize_training_metrics(evaluation)
            ),
            "attracting_block_residuals_by_physical_sign": block_residuals,
            "global_unconditioned_max_euclidean_latent_residual": (
                global_residual
            ),
        }

    d2_stored = results["2"][
        "attracting_block_residuals_by_physical_sign"
    ]
    dense_reference = _dense_d2_reference(DENSE_D2_RESULT)
    dense_by_physical = dense_reference[
        "dimension_2_by_physical_attractor"
    ]
    for physical in ("negative", "positive"):
        stored_source = json.loads(
            DENSE_D2_RESULT.read_text(encoding="utf-8")
        )["nodes"][str(d2_stored[physical]["morse_node"])]["residual"][
            "source_summaries"
        ]["replay_sources/chafee_infante/data/train.csv"]
        if (
            int(stored_source["accepted_samples"])
            != d2_stored[physical]["accepted_pairs"]
            or float(stored_source["max_euclidean_residual"])
            != d2_stored[physical]["max_euclidean_residual"]
        ):
            raise ValueError(
                f"d=2 stored-pair audit disagrees with dense artifact "
                f"for the {physical} block"
            )
        if dense_by_physical[physical]["morse_node"] != (
            d2_stored[physical]["morse_node"]
        ):
            raise ValueError(
                f"d=2 dense/stored node mismatch for {physical}"
            )

    return {
        "schema_version": 1,
        "generated_by": "code/scripts/audit_chafee_dimension_residuals.py",
        "scope": {
            "description": (
                "Render-free audit of persisted d=1,2,3 checkpoints on the "
                "same finite Marcio training pairs"
            ),
            "training_performed": False,
            "cmgdb_performed": False,
            "pde_integration_performed": False,
            "dense_residual_search_performed": False,
            "neural_evaluations": (
                "only batched inference on the 30,000 stored pairs and "
                "the two stored stable roots"
            ),
        },
        "metric_definitions": {
            "L1_reconstruction_mse": "mean((D(E(x)) - x)^2)",
            "L2_decoded_one_step_prediction_mse": (
                "mean((D(g(E(x))) - f(x))^2)"
            ),
            "L1_plus_L2": (
                "Marcio's two-term full-data training objective"
            ),
            "L3_unconditioned_latent_semiconjugacy_mse": (
                "mean((g(E(x)) - E(f(x)))^2); diagnostic only, "
                "not included in Marcio's training objective"
            ),
            "euclidean_latent_residual": (
                "||g(E(x)) - E(f(x))||_2 in stored latent coordinates"
            ),
        },
        "finite_stored_pair_protocol": {
            "data": data_record,
            "stable_roots": roots_record,
            "n_pairs": N_PAIRS,
            "ambient_dimension": HIGH_DIMENSION,
            "dtype": "float32 model inference",
            "device": "cpu",
            "batch_size": batch_size,
            "split": (
                "all training pairs; Marcio's computation had no held-out "
                "validation/test split"
            ),
            "block_membership": (
                "closed-box union membership of E(x) in the persisted "
                "minimal attracting block"
            ),
            "results_by_latent_dimension": results,
        },
        "dense_paper_protocol_reference": dense_reference,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = run_audit(batch_size=args.batch_size)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "dimensions": [1, 2, 3],
                "pairs_per_dimension": N_PAIRS,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
