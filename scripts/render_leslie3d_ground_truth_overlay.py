#!/usr/bin/env python3
"""Render encoded direct Leslie ground truth over any accepted-run Morse CSV.

This is the parameterized successor to the fixed max-30 report renderer.  It
loads the accepted ``autoencoder`` checkpoint and archived scaler from the
requested run/config, verifies the analyzer's learned-role assignments, and
uses every row of the learned Morse-set CSV to build a raster background.

The encoded display-cover centers are a visualization only.  They are not an
enclosing image of the direct three-dimensional boxes under the encoder and
do not constitute a semiconjugacy or Conley-index certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageDraw

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from latentdynamics.cli.provenance import (
    config_conflicts_with_manifest,
    hash_config_dict,
)
from latentdynamics.config import load_config
from latentdynamics.training import load_any_checkpoint
from latentdynamics.viz.style import PALETTE, apply_paper_style, style_latent_axes

CODE_ROOT = Path(__file__).resolve().parents[1]
PACKAGED_CONFIGS = CODE_ROOT / "src" / "latentdynamics" / "configs"

DEFAULT_DIRECT_DISPLAY_COVER = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "ground_truth"
    / "absorbing_B_uniform_level33_recurrent_closure"
    / "cubical_3d_level24_display_cover"
    / "morse_sets_level24_display_cover.csv"
)
DEFAULT_DIRECT_DISPLAY_MANIFEST = DEFAULT_DIRECT_DISPLAY_COVER.parent / "manifest.json"

# The report is intentionally restricted to the accepted learned chart/map and
# immutable direct-system inputs.  The Morse CSV, analyzer files, config, and
# run manifest are resolution-specific and are hashed dynamically below.
EXPECTED_ACCEPTED_HASHES = {
    "checkpoint": "9fbee2cde690d58d2413c0d3521763838abaeb493736120f7173035612da3f3d",
    "checkpoint_sidecar": "238363dc17364677f2226cbf6cceffd6b0d08cc5a2f7e69a1e8b74f742af64a7",
    "scaler": "bb908b946d259fd6aa6a716cc003f789631e21bc7c9aa0a6a64c09ac629aa5e1",
    "dataset_manifest": "658926337cc98e5e2d08ff9f442496c929e7400369deb7bc0382a6b73e87f5a1",
    "direct_display_manifest": "5dde411d7ca3a4b11ecdaabe5baf36c6c5d3e612daa4816774221f2c587d1b76",
    "direct_display_cover": "1726db63abffbc4a0984d61591f1fbacd0291187301271e71fdbed474c3c2b29",
}

OBJECT_ORDER = ("P0", "P1", "S2", "S4", "p_star", "origin")
OBJECT_STYLE = {
    "P0": {"color": "#FFB000", "marker": "o", "math": r"$P_0$"},
    "P1": {"color": "#DC267F", "marker": "s", "math": r"$P_1$"},
    "S2": {"color": "#FE6100", "marker": "^", "math": r"$S_2$"},
    "S4": {"color": "#648FFF", "marker": "D", "math": r"$S_4$"},
    "p_star": {"color": "#785EF0", "marker": "*", "math": r"$p_*$"},
    "origin": {"color": "#008080", "marker": "X", "math": "origin"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, observed_hash: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": observed_hash or _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _resolve_code_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (CODE_ROOT / path).resolve()


def _config_file(config_arg: str | Path) -> Path:
    candidate = Path(config_arg)
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()
    if candidate.suffix == ".yaml" and candidate.is_file():
        return candidate.resolve()
    stem = candidate.stem if candidate.suffix else candidate.name
    packaged = PACKAGED_CONFIGS / f"{stem}.yaml"
    if packaged.is_file():
        return packaged.resolve()
    raise FileNotFoundError(f"cannot resolve config file for {config_arg!r}")


def _require_hash(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"{role} hash mismatch for {path}: expected {expected}, observed {observed}"
        )
    return observed


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _validate_run_manifest(config: Any, run_root: Path) -> tuple[Path, dict[str, Any]]:
    path = run_root / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"run manifest is required for provenance: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded_config = manifest.get("config")
    if not isinstance(recorded_config, dict):
        raise RuntimeError("run manifest has no recorded config object")
    recorded_hash = manifest.get("config_hash")
    if recorded_hash != hash_config_dict(recorded_config):
        raise RuntimeError("run manifest config_hash does not match its recorded config")
    conflicts = config_conflicts_with_manifest(config, recorded_config)
    if conflicts:
        raise RuntimeError(
            "requested config conflicts with run manifest at: " + ", ".join(conflicts)
        )
    recorded_root = manifest.get("cell", {}).get("output_dir")
    if not isinstance(recorded_root, str) or not _same_path(
        _resolve_code_path(Path(recorded_root)), run_root
    ):
        raise RuntimeError("run manifest cell.output_dir does not identify --run-root")
    return path, manifest


def _validate_analyzer(
    summary_path: Path,
    encoded_path: Path,
    *,
    config: Any,
    run_root: Path,
    morse_sets_path: Path,
    checkpoint: Path,
    sidecar: Path,
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"analyzer summary is required; run analyze_leslie3d_invariant_aware.py: {summary_path}"
        )
    if not encoded_path.is_file():
        raise FileNotFoundError(f"analyzer encoded-point CSV is required: {encoded_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "numerical_experiment_not_a_topological_certificate":
        raise RuntimeError(f"unexpected analyzer status: {summary.get('status')!r}")
    if summary.get("experiment") != config.experiment_name:
        raise RuntimeError("analyzer experiment does not match the requested config")
    checkpoint_info = summary.get("checkpoint", {})
    if checkpoint_info.get("basename") != "autoencoder":
        raise RuntimeError("overlay accepts only the promoted/accepted autoencoder checkpoint")
    if not _same_path(Path(checkpoint_info.get("path", "")), checkpoint):
        raise RuntimeError("analyzer checkpoint path does not match the run checkpoint")
    if not _same_path(Path(checkpoint_info.get("architecture_sidecar", "")), sidecar):
        raise RuntimeError("analyzer architecture sidecar does not match the run sidecar")
    expected_morse_dir = morse_sets_path.parent
    if not _same_path(Path(summary.get("morse_directory", "")), expected_morse_dir):
        raise RuntimeError("analyzer Morse directory does not match the selected Morse CSV")
    bounds = summary.get("configured_cmgdb_bounds", {})
    if not np.array_equal(
        np.asarray(bounds.get("lower"), dtype=np.float64),
        np.asarray(config.cmgdb.lower_bounds, dtype=np.float64),
    ) or not np.array_equal(
        np.asarray(bounds.get("upper"), dtype=np.float64),
        np.asarray(config.cmgdb.upper_bounds, dtype=np.float64),
    ):
        raise RuntimeError("analyzer CMGDB bounds do not match the requested config")
    if not _same_path(run_root / "models" / "autoencoder.pt", checkpoint):
        raise RuntimeError("checkpoint must belong to --run-root")
    return summary


def _load_learned_morse_sets(path: Path) -> tuple[NDArray[np.float64], dict[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 5 or data.shape[0] == 0:
        raise ValueError(f"expected a non-empty 2-D Morse CSV with five columns, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("learned Morse CSV contains non-finite values")
    if np.any(data[:, :2] > data[:, 2:4]):
        raise ValueError("learned Morse CSV contains reversed box bounds")
    labels = np.rint(data[:, 4]).astype(np.int64)
    if not np.array_equal(data[:, 4], labels):
        raise ValueError("learned Morse CSV labels are not integral")
    unique, counts = np.unique(labels, return_counts=True)
    return data, {int(label): int(count) for label, count in zip(unique, counts, strict=True)}


def _load_encoded_analyzer_points(
    path: Path,
) -> dict[str, NDArray[np.float64]]:
    grouped: dict[str, list[tuple[int, tuple[float, float]]]] = {
        name: [] for name in OBJECT_ORDER
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"object", "phase", "z0", "z1"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"encoded analyzer CSV lacks columns {sorted(required)}")
        for row in reader:
            name = row["object"]
            if name in grouped:
                grouped[name].append(
                    (int(row["phase"]), (float(row["z0"]), float(row["z1"])))
                )
    result: dict[str, NDArray[np.float64]] = {}
    for name in OBJECT_ORDER:
        rows = sorted(grouped[name])
        if [phase for phase, _point in rows] != list(range(len(rows))) or not rows:
            raise ValueError(f"encoded analyzer CSV has incomplete phases for {name}")
        result[name] = np.asarray([point for _phase, point in rows], dtype=np.float64)
    return result


def _role_assignments(summary: dict[str, Any]) -> dict[str, int | None]:
    membership = summary.get("morse_membership")
    if not isinstance(membership, dict):
        raise RuntimeError("analyzer summary has no morse_membership object")
    assignments: dict[str, int | None] = {}
    for name in OBJECT_ORDER:
        item = membership.get(name)
        if not isinstance(item, dict) or "assigned_morse_node" not in item:
            raise RuntimeError(f"analyzer summary has no assignment for {name}")
        node = item["assigned_morse_node"]
        if node is not None and (isinstance(node, bool) or not isinstance(node, int)):
            raise RuntimeError(f"invalid learned Morse node assignment for {name}: {node!r}")
        assignments[name] = node
    return assignments


def _verify_analyzer_membership(
    data: NDArray[np.float64],
    encoded: dict[str, NDArray[np.float64]],
    summary: dict[str, Any],
    *,
    tolerance: float = 1e-14,
) -> None:
    """Reject a stale analyzer summary by recomputing exact-point membership."""
    labels = np.rint(data[:, 4]).astype(np.int64)
    membership = summary["morse_membership"]
    for name in OBJECT_ORDER:
        observed_phases: list[list[int]] = []
        for point in encoded[name]:
            mask = (
                (data[:, 0] - point[0] <= tolerance)
                & (data[:, 1] - point[1] <= tolerance)
                & (point[0] - data[:, 2] <= tolerance)
                & (point[1] - data[:, 3] <= tolerance)
            )
            observed_phases.append(sorted(np.unique(labels[mask]).astype(int).tolist()))
        recorded = membership[name].get("phase_containing_nodes")
        if observed_phases != recorded:
            raise RuntimeError(
                f"analyzer membership for {name} is stale relative to the selected Morse CSV: "
                f"recorded={recorded}, observed={observed_phases}"
            )


@torch.no_grad()
def _encode_points(
    model: torch.nn.Module,
    scaler: Any,
    points: NDArray[np.float64],
    *,
    batch_size: int = 16384,
) -> NDArray[np.float64]:
    scaled = scaler.transform(points)
    chunks: list[NDArray[np.float64]] = []
    for start in range(0, len(scaled), batch_size):
        values = torch.as_tensor(scaled[start : start + batch_size], dtype=torch.float32)
        chunks.append(model.encoder(values).cpu().numpy().astype(np.float64))
    return np.vstack(chunks)


def _load_and_encode_direct_objects(
    model: torch.nn.Module,
    scaler: Any,
    dataset_manifest: dict[str, Any],
    display_cover_path: Path,
) -> tuple[
    dict[str, NDArray[np.float64]],
    dict[str, NDArray[np.float64]],
    dict[str, int],
    dict[str, int],
]:
    known = dataset_manifest.get("known_objects", {})
    direct_node_to_object: dict[int, str] = {}
    exact_encoded: dict[str, NDArray[np.float64]] = {}
    for name in OBJECT_ORDER:
        if name not in known:
            raise RuntimeError(f"dataset manifest has no known object {name}")
        node = int(known[name]["expected_direct_node"])
        if node in direct_node_to_object:
            raise RuntimeError(f"duplicate expected direct node {node}")
        direct_node_to_object[node] = name
        points = np.asarray(known[name]["points"], dtype=np.float64)
        exact_encoded[name] = _encode_points(model, scaler, points)

    cover = np.loadtxt(display_cover_path, delimiter=",", ndmin=2, dtype=np.float64)
    if cover.ndim != 2 or cover.shape[1] != 7 or cover.shape[0] == 0:
        raise ValueError(f"expected non-empty seven-column direct display cover, got {cover.shape}")
    labels = np.rint(cover[:, 6]).astype(np.int64)
    if not np.array_equal(cover[:, 6], labels):
        raise ValueError("direct display-cover labels are not integral")
    if set(np.unique(labels).tolist()) != set(direct_node_to_object):
        raise RuntimeError("direct display-cover labels do not match dataset direct-node roles")
    centers = 0.5 * (cover[:, :3] + cover[:, 3:6])
    encoded_centers = _encode_points(model, scaler, centers)
    clouds = {
        direct_node_to_object[node]: encoded_centers[labels == node]
        for node in sorted(direct_node_to_object)
    }
    cover_counts = {name: len(clouds[name]) for name in OBJECT_ORDER}
    direct_nodes = {name: int(known[name]["expected_direct_node"]) for name in OBJECT_ORDER}
    return clouds, exact_encoded, cover_counts, direct_nodes


def _rasterize_morse_boxes(
    data: NDArray[np.float64],
    *,
    palette: Sequence[str] = PALETTE,
    max_pixels: int = 2600,
    alpha: int = 178,
) -> tuple[Image.Image, tuple[float, float, float, float], dict[str, Any]]:
    """Rasterize every input box once without constructing millions of patches."""
    if max_pixels < 512:
        raise ValueError("max_pixels must be at least 512")
    lower = data[:, :2].min(axis=0)
    upper = data[:, 2:4].max(axis=0)
    span = upper - lower
    if np.any(span <= 0.0):
        raise ValueError("Morse-set occupied extent must be positive in both axes")
    if span[0] >= span[1]:
        width = max_pixels
        height = max(512, round(max_pixels * span[1] / span[0]))
    else:
        height = max_pixels
        width = max(512, round(max_pixels * span[0] / span[1]))

    x0 = np.floor((data[:, 0] - lower[0]) / span[0] * width).astype(np.int32)
    x1 = np.ceil((data[:, 2] - lower[0]) / span[0] * width).astype(np.int32) - 1
    y0_from_bottom = np.floor((data[:, 1] - lower[1]) / span[1] * height).astype(np.int32)
    y1_from_bottom = np.ceil((data[:, 3] - lower[1]) / span[1] * height).astype(np.int32) - 1
    np.clip(x0, 0, width - 1, out=x0)
    np.clip(x1, 0, width - 1, out=x1)
    np.maximum(x1, x0, out=x1)
    np.clip(y0_from_bottom, 0, height - 1, out=y0_from_bottom)
    np.clip(y1_from_bottom, 0, height - 1, out=y1_from_bottom)
    np.maximum(y1_from_bottom, y0_from_bottom, out=y1_from_bottom)
    y0 = height - 1 - y1_from_bottom
    y1 = height - 1 - y0_from_bottom
    labels = np.rint(data[:, 4]).astype(np.int64)

    rgba = []
    for color in palette:
        value = matplotlib.colors.to_rgba(color)
        rgb = tuple(round(255 * channel) for channel in value[:3])
        rgba.append((*rgb, alpha))
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    rectangle = ImageDraw.Draw(image).rectangle
    for left, top, right, bottom, label in zip(
        x0, y0, x1, y1, labels, strict=True
    ):
        rectangle(
            (int(left), int(top), int(right), int(bottom)),
            fill=rgba[int(label) % len(rgba)],
        )
    extent = (float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1]))
    metadata = {
        "method": "full-row Pillow RGBA raster; no row sampling or aggregation",
        "input_rows": int(data.shape[0]),
        "rows_visited": int(data.shape[0]),
        "row_sampling": "none",
        "pixel_width": width,
        "pixel_height": height,
        "alpha_0_to_255": alpha,
        "minimum_visible_box": "one output pixel",
    }
    return image, extent, metadata


def _legend_handles(assignments: dict[str, int | None]) -> list[Line2D]:
    handles: list[Line2D] = []
    for name in OBJECT_ORDER:
        style = OBJECT_STYLE[name]
        node = assignments[name]
        learned = "not uniquely assigned" if node is None else rf"learned $M_{{{node}}}$"
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                markerfacecolor="white" if name != "origin" else style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=1.2,
                linewidth=0.0,
                label=f"{style['math']} → {learned}",
            )
        )
    return handles


def _atomic_savefig(
    fig: Any,
    path: Path,
    *,
    fmt: str,
    dpi: int,
    overwrite: bool,
) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=f".{fmt}", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        fig.savefig(temp_path, format=fmt, bbox_inches="tight", dpi=dpi)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temp_path, path)
        path.chmod(0o644)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any], *, overwrite: bool) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".json", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temp_path, path)
        path.chmod(0o644)
    finally:
        temp_path.unlink(missing_ok=True)


def render(
    config_arg: str | Path,
    run_root: Path,
    output_dir: Path,
    *,
    morse_sets_path: Path | None = None,
    analyzer_summary_path: Path | None = None,
    encoded_invariants_path: Path | None = None,
    direct_display_cover_path: Path = DEFAULT_DIRECT_DISPLAY_COVER,
    direct_display_manifest_path: Path = DEFAULT_DIRECT_DISPLAY_MANIFEST,
    max_background_pixels: int = 2600,
    dpi: int = 320,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_path = _config_file(config_arg)
    config = load_config(config_path)
    run_root = _resolve_code_path(run_root)
    output_dir = _resolve_code_path(output_dir)
    morse_sets_path = (
        run_root / "MG" / "morse_sets"
        if morse_sets_path is None
        else _resolve_code_path(morse_sets_path)
    )
    analyzer_summary_path = (
        run_root / "analysis" / "invariant_aware_summary.json"
        if analyzer_summary_path is None
        else _resolve_code_path(analyzer_summary_path)
    )
    encoded_invariants_path = (
        analyzer_summary_path.parent / "encoded_invariant_points.csv"
        if encoded_invariants_path is None
        else _resolve_code_path(encoded_invariants_path)
    )
    direct_display_cover_path = _resolve_code_path(direct_display_cover_path)
    direct_display_manifest_path = _resolve_code_path(direct_display_manifest_path)

    try:
        output_dir.relative_to(run_root)
    except ValueError:
        pass
    else:
        raise ValueError("--output-dir must be outside --run-root to keep the run immutable")

    checkpoint = run_root / "models" / "autoencoder.pt"
    sidecar = run_root / "models" / "autoencoder.json"
    data_dir = _resolve_code_path(config.paths.data_dir)
    dataset_manifest_path = data_dir / "dataset_manifest.json"
    scaler_path = _resolve_code_path(config.paths.scaler_path("train"))
    run_manifest_path, run_manifest = _validate_run_manifest(config, run_root)

    observed_static = {
        "checkpoint": _require_hash(
            checkpoint, EXPECTED_ACCEPTED_HASHES["checkpoint"], "accepted checkpoint"
        ),
        "checkpoint_sidecar": _require_hash(
            sidecar,
            EXPECTED_ACCEPTED_HASHES["checkpoint_sidecar"],
            "accepted checkpoint sidecar",
        ),
        "scaler": _require_hash(
            scaler_path, EXPECTED_ACCEPTED_HASHES["scaler"], "accepted scaler"
        ),
        "dataset_manifest": _require_hash(
            dataset_manifest_path,
            EXPECTED_ACCEPTED_HASHES["dataset_manifest"],
            "invariant-aware dataset manifest",
        ),
        "direct_display_manifest": _require_hash(
            direct_display_manifest_path,
            EXPECTED_ACCEPTED_HASHES["direct_display_manifest"],
            "direct display-cover manifest",
        ),
        "direct_display_cover": _require_hash(
            direct_display_cover_path,
            EXPECTED_ACCEPTED_HASHES["direct_display_cover"],
            "direct display-cover CSV",
        ),
    }
    if run_manifest.get("artifacts", {}).get("scaler_sha256") != observed_static["scaler"]:
        raise RuntimeError("run manifest scaler hash does not match the accepted scaler")

    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if sidecar_payload.get("arch") != config.arch.model_dump(mode="json"):
        raise RuntimeError("checkpoint architecture sidecar does not match the requested config")
    analyzer_summary = _validate_analyzer(
        analyzer_summary_path,
        encoded_invariants_path,
        config=config,
        run_root=run_root,
        morse_sets_path=morse_sets_path,
        checkpoint=checkpoint,
        sidecar=sidecar,
    )

    direct_manifest = json.loads(direct_display_manifest_path.read_text(encoding="utf-8"))
    if direct_manifest.get("cover", {}).get("csv", {}).get("sha256") != observed_static[
        "direct_display_cover"
    ]:
        raise RuntimeError("direct display manifest does not authenticate its display CSV")
    direct_raw_path = Path(direct_manifest["source"]["morse_sets"]["path"])
    if not direct_raw_path.is_file():
        direct_raw_path = (
            CODE_ROOT
            / "output"
            / "original_leslie"
            / "ground_truth"
            / "absorbing_B_i29_m33_M36_L10000"
            / "screen"
            / "MG"
            / "morse_sets"
        )
    direct_raw_hash = _sha256(direct_raw_path)
    if direct_raw_hash != direct_manifest["source"]["morse_sets"]["sha256"]:
        raise RuntimeError("direct raw Morse-set hash does not match the display manifest")

    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    model, _ = load_any_checkpoint(
        run_root / "models", arch=config.arch, basename="autoencoder"
    )
    model.eval().cpu()
    scaler = joblib.load(scaler_path)
    cover_encoded, exact_encoded, cover_counts, direct_nodes = (
        _load_and_encode_direct_objects(
            model, scaler, dataset_manifest, direct_display_cover_path
        )
    )
    analyzer_encoded = _load_encoded_analyzer_points(encoded_invariants_path)
    for name in OBJECT_ORDER:
        if analyzer_encoded[name].shape != exact_encoded[name].shape or not np.allclose(
            analyzer_encoded[name], exact_encoded[name], rtol=0.0, atol=5e-7
        ):
            maximum = (
                float(np.max(np.abs(analyzer_encoded[name] - exact_encoded[name])))
                if analyzer_encoded[name].shape == exact_encoded[name].shape
                else float("inf")
            )
            raise RuntimeError(
                f"freshly encoded exact phases disagree with analyzer CSV for {name}; max={maximum}"
            )

    morse_data, boxes_by_label = _load_learned_morse_sets(morse_sets_path)
    assignments = _role_assignments(analyzer_summary)
    _verify_analyzer_membership(morse_data, analyzer_encoded, analyzer_summary)
    present_labels = set(boxes_by_label)
    missing_assigned = sorted(
        node for node in assignments.values() if node is not None and node not in present_labels
    )
    if missing_assigned:
        raise RuntimeError(f"analyzer assignments reference absent Morse labels {missing_assigned}")

    tag = (
        f"s{config.cmgdb.subdiv_init}_m{config.cmgdb.subdiv_min}_"
        f"M{config.cmgdb.subdiv_max}_L{config.cmgdb.subdiv_limit}"
    )
    basename = f"direct_ground_truth_on_{tag}_morse_sets"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    provenance_path = output_dir / "direct_ground_truth_overlay_provenance.json"
    if not overwrite:
        existing = [path for path in (png_path, pdf_path, provenance_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to clobber existing report assets: "
                + ", ".join(str(path) for path in existing)
            )

    background, extent, raster_metadata = _rasterize_morse_boxes(
        morse_data, max_pixels=max_background_pixels
    )
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(10.8, 7.3), layout="constrained")
    ax.imshow(
        background,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        zorder=1,
        rasterized=True,
    )
    for name in OBJECT_ORDER:
        style = OBJECT_STYLE[name]
        cloud = cover_encoded[name]
        ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            s=6.0 if name == "P1" else 13.0,
            marker=".",
            color=style["color"],
            alpha=0.22 if name == "P1" else 0.48,
            linewidths=0.0,
            rasterized=True,
            zorder=18,
        )
        exact = exact_encoded[name]
        if len(exact) > 1:
            closed = np.vstack([exact, exact[0]])
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                color=style["color"],
                linewidth=0.9,
                alpha=0.75,
                zorder=24,
            )
        ax.scatter(
            exact[:, 0],
            exact[:, 1],
            s=58 if name != "p_star" else 110,
            marker=style["marker"],
            facecolor="white" if name != "origin" else style["color"],
            edgecolor=style["color"],
            linewidth=1.5,
            zorder=26,
        )

    all_overlay = np.vstack([*cover_encoded.values(), *exact_encoded.values()])
    lower = np.minimum(np.asarray([extent[0], extent[2]]), all_overlay.min(axis=0))
    upper = np.maximum(np.asarray([extent[1], extent[3]]), all_overlay.max(axis=0))
    padding = 0.035 * np.maximum(upper - lower, 1e-6)
    ax.set_xlim(float(lower[0] - padding[0]), float(upper[0] + padding[0]))
    ax.set_ylim(float(lower[1] - padding[1]), float(upper[1] + padding[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    style_latent_axes(ax, two_d=True)
    ax.set_title(
        "Leslie3D latent Morse sets "
        f"({config.cmgdb.subdiv_init}, {config.cmgdb.subdiv_min}, "
        f"{config.cmgdb.subdiv_max}; limit {config.cmgdb.subdiv_limit:,})\n"
        "with encoded direct-system ground truth",
        pad=12,
    )
    ax.legend(
        handles=_legend_handles(assignments),
        title="Direct role → analyzer-assigned learned node",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=1,
    )
    ax.text(
        0.01,
        0.015,
        f"Background: all {morse_data.shape[0]:,} saved learned Morse boxes (no row sampling)\n"
        "Dots: encoded centers of direct-map display-cover cells\n"
        "Outlined symbols: exact direct fixed/periodic phases",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 3.0},
        zorder=40,
    )

    _atomic_savefig(fig, png_path, fmt="png", dpi=dpi, overwrite=overwrite)
    _atomic_savefig(fig, pdf_path, fmt="pdf", dpi=dpi, overwrite=overwrite)
    plt.close(fig)

    learned_hash = _sha256(morse_sets_path)
    analyzer_summary_hash = _sha256(analyzer_summary_path)
    encoded_hash = _sha256(encoded_invariants_path)
    config_hash = _sha256(config_path)
    run_manifest_hash = _sha256(run_manifest_path)
    input_sha256 = {
        str(config_path): config_hash,
        str(run_manifest_path): run_manifest_hash,
        str(morse_sets_path): learned_hash,
        str(checkpoint): observed_static["checkpoint"],
        str(sidecar): observed_static["checkpoint_sidecar"],
        str(scaler_path): observed_static["scaler"],
        str(dataset_manifest_path): observed_static["dataset_manifest"],
        str(analyzer_summary_path): analyzer_summary_hash,
        str(encoded_invariants_path): encoded_hash,
        str(direct_display_manifest_path): observed_static["direct_display_manifest"],
        str(direct_display_cover_path): observed_static["direct_display_cover"],
        str(direct_raw_path.resolve()): direct_raw_hash,
    }
    outputs = {
        png_path.name: _artifact(png_path),
        pdf_path.name: _artifact(pdf_path),
    }
    provenance: dict[str, Any] = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "direct Leslie ground truth over parameterized accepted-run latent Morse sets",
        "configuration": config.experiment_name,
        "config_path": str(config_path),
        "run_root": str(run_root),
        "input_sha256": input_sha256,
        "accepted_model": {
            "checkpoint": _artifact(checkpoint, observed_hash=observed_static["checkpoint"]),
            "architecture_sidecar": _artifact(
                sidecar, observed_hash=observed_static["checkpoint_sidecar"]
            ),
            "hash_gate": "matches the accepted invariant-aware v2 smooth map",
        },
        "scaler": _artifact(scaler_path, observed_hash=observed_static["scaler"]),
        "cmgdb": {
            "subdiv_init": config.cmgdb.subdiv_init,
            "subdiv_min": config.cmgdb.subdiv_min,
            "subdiv_max": config.cmgdb.subdiv_max,
            "subdiv_limit": config.cmgdb.subdiv_limit,
            "lower_bounds": config.cmgdb.lower_bounds,
            "upper_bounds": config.cmgdb.upper_bounds,
            "padding": config.cmgdb.padding,
        },
        "learned_morse_sets": {
            **_artifact(morse_sets_path, observed_hash=learned_hash),
            "rows": int(morse_data.shape[0]),
            "boxes_by_label": {str(k): v for k, v in boxes_by_label.items()},
            "full_raw_csv_preserved_at_source": True,
            "source_was_not_modified": True,
        },
        "analyzer": {
            "summary": _artifact(analyzer_summary_path, observed_hash=analyzer_summary_hash),
            "encoded_invariant_points": _artifact(
                encoded_invariants_path, observed_hash=encoded_hash
            ),
            "status": analyzer_summary["status"],
            "learned_role_assignment": assignments,
            "membership_independently_rechecked_against_selected_csv": True,
        },
        "direct_display_cover": {
            "csv": _artifact(
                direct_display_cover_path,
                observed_hash=observed_static["direct_display_cover"],
            ),
            "manifest": _artifact(
                direct_display_manifest_path,
                observed_hash=observed_static["direct_display_manifest"],
            ),
            "raw_source_morse_sets": _artifact(
                direct_raw_path, observed_hash=direct_raw_hash
            ),
            "encoded_center_counts_by_object": cover_counts,
            "expected_direct_nodes_from_dataset_manifest": direct_nodes,
        },
        "render": raster_metadata,
        "method": {
            "learned_background": "every saved learned Morse box, rasterized at report resolution",
            "direct_set_overlay": (
                "centers of all level-24 display-cover cells, scaled and encoded by E"
            ),
            "exact_object_overlay": (
                "all 16 fixed/periodic phases from the invariant-aware dataset manifest, "
                "freshly encoded and checked against analyzer output"
            ),
            "learned_node_labels": "read from analyzer morse_membership; no node IDs hard-coded",
        },
        "limitations": [
            "encoded display-cover centers are sampled points, not enclosing images of 3-D boxes under E",
            "the display cover is render-only and is not a recomputed level-24 direct Morse decomposition",
            "the learned box background is pixel-rasterized for tractable PDF size, with at least one pixel per input box",
            "the overlay is numerical evidence and not a semiconjugacy or Conley-index certificate",
        ],
        "outputs": outputs,
    }
    _atomic_write_json(provenance_path, provenance, overwrite=overwrite)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="packaged config name or YAML path")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--morse-sets",
        type=Path,
        default=None,
        help="optional learned Morse CSV; defaults to RUN_ROOT/MG/morse_sets",
    )
    parser.add_argument("--analyzer-summary", type=Path, default=None)
    parser.add_argument("--encoded-invariants", type=Path, default=None)
    parser.add_argument("--max-background-pixels", type=int, default=2600)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only this script's three named outputs; default is non-clobbering",
    )
    args = parser.parse_args()
    result = render(
        args.config,
        args.run_root,
        args.output_dir,
        morse_sets_path=args.morse_sets,
        analyzer_summary_path=args.analyzer_summary,
        encoded_invariants_path=args.encoded_invariants,
        max_background_pixels=args.max_background_pixels,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
