#!/usr/bin/env python3
"""Create a self-contained leslie3d_example1 fixed-22 versus adaptive bundle.

Assembles the uniform (22,22,22) recomputation, the active adaptive
(23,23,27) replay artifacts, and the connection-complete 4/5 merge into one
directory with consistency gates, a machine-readable manifest, and per-file
checksums, then runs ``scripts/render_leslie3d_example1_figures.py`` to
produce the paper-ready panels inside the bundle.

Inputs: ``replay_sources/leslie3d_example1/`` (fetched artifacts) plus the
outputs of ``leslie3d_example1_coarsen_morse_graph.py``,
``leslie3d_example1_uniform_grid.py --depth 22``, and
``leslie3d_example1_uniform_sampled_metrics.py`` under ``--input-root``
(default ``output/leslie3d_example1_study``).  The bundle is written to
``--output`` (default ``<input-root>/fixed22_vs_adaptive``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

from latentdynamics._paths import get_repo_root


REPO_ROOT = get_repo_root()
SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVE = REPO_ROOT / "replay_sources" / "leslie3d_example1" / "spurious_attractor_ex"
DEFAULT_INPUT_ROOT = REPO_ROOT / "output" / "leslie3d_example1_study"
PROVENANCE_REFERENCE = (
    REPO_ROOT
    / "artifacts"
    / "reference_results"
    / "leslie3d_example1"
    / "fixed22_vs_adaptive"
    / "provenance"
    / "g1_provenance_full.md"
)
MIGRATION_RECORD = REPO_ROOT / "artifacts" / "provenance" / "MIGRATION_RECORD.json"
REFERENCE_METRICS = (
    REPO_ROOT
    / "artifacts"
    / "reference_results"
    / "leslie3d_example1"
    / "fixed22"
    / "residual_tolerance"
)

# Checksums of the checkpoint files recorded in the coauthor's private
# development archive (see provenance/g1_provenance_full.md). The shipped
# public bundle carries the single-file migration of those weights instead;
# either form is accepted, checked byte-for-byte against its record.
AUTHOR_ARCHIVE_MODEL_SHA256 = {
    "encoder.pt": "211f84456707afb254da1cfa05defa41137f90813f624af420722954763d1f4c",
    "decoder.pt": "25fb56dae18de6239bffefad463e0abed126535036f883be123f39ccfc4173e2",
    "dynamics.pt": "5d175395081ba8983ac863841fac83e9eab26e82de31121be572a2575b8ed955",
}

BUNDLED_SCRIPTS = (
    "leslie3d_example1_coarsen_morse_graph.py",
    "leslie3d_example1_uniform_grid.py",
    "leslie3d_example1_uniform_sampled_metrics.py",
    "leslie3d_example1_verify_closures.py",
    "render_leslie3d_example1_figures.py",
    "leslie3d_example1_package_bundle.py",
)

PALETTE = [
    "#FFB000",
    "#DC267F",
    "#648FFF",
    "#FE6100",
    "#785EF0",
    "#008080",
    "#FCC2E8",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def verified_model_sha256() -> tuple[dict[str, str], str]:
    """Check the active checkpoint against its recorded checksums.

    Accepts either the legacy three-file author checkpoint (checked against
    the author-archive checksums) or the shipped single-file migration
    (checked against ``artifacts/provenance/MIGRATION_RECORD.json``). Returns
    the observed per-file checksums and which form was found.
    """
    legacy_names = ("encoder.pt", "dynamics.pt", "decoder.pt")
    if all((ACTIVE / "models" / name).is_file() for name in legacy_names):
        observed = {name: sha256(ACTIVE / "models" / name) for name in legacy_names}
        if observed != {name: AUTHOR_ARCHIVE_MODEL_SHA256[name] for name in observed}:
            raise RuntimeError(
                "active model files do not match the recorded author-archive "
                f"checksums: {observed}"
            )
        return observed, "legacy_three_file"
    record = json.loads(MIGRATION_RECORD.read_text(encoding="utf-8"))
    expected = record["migrated"]["leslie3d_example1/spurious_attractor_ex"][
        "migrated_files"
    ]["autoencoder.pt"]["sha256"]
    observed = {"autoencoder.pt": sha256(ACTIVE / "models" / "autoencoder.pt")}
    if observed["autoencoder.pt"] != expected:
        raise RuntimeError(
            "active migrated checkpoint does not match the checksum recorded "
            f"in {recorded_path(MIGRATION_RECORD)}: {observed['autoencoder.pt']}"
        )
    return observed, "migrated_single_file"


def recorded_path(path: Path) -> str:
    """Repo-relative path when possible, otherwise the path as given."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def copy(source: Path, dest: Path, relative: str) -> Path:
    destination = dest / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def copy_optional(source: Path, dest: Path, relative: str) -> Path | None:
    if not source.is_file():
        print(f"optional artifact missing, skipped: {source}", flush=True)
        return None
    return copy(source, dest, relative)


def csv_counts(path: Path) -> dict[str, int]:
    counts: Counter[int] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            counts[int(float(row[-1]))] += 1
    return {str(label): counts[label] for label in sorted(counts)}


def graph_node_lines(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(\d+) \[", line)
        if match:
            result[int(match.group(1))] = line
    return result


def descendants(nodes: list[int], edges: list[list[int]]) -> dict[int, set[int]]:
    adjacency = {node: set() for node in nodes}
    for source, target in edges:
        adjacency[source].add(target)
    result: dict[int, set[int]] = {}
    for source in nodes:
        reached = {source}
        stack = [source]
        while stack:
            current = stack.pop()
            for target in adjacency[current]:
                if target not in reached:
                    reached.add(target)
                    stack.append(target)
        result[source] = reached
    return result


def write_fixed22_nontrivial(
    result: dict[str, Any], fixed: Path, dest: Path
) -> tuple[Path, Path]:
    graph = result["fixed_graph"]
    retained = sorted(
        int(node)
        for node, component in graph["components"].items()
        if any(value != "0" for value in component["conley_index"])
    )
    reachability = descendants(
        list(range(int(graph["node_count"]))),
        graph["edges"],
    )
    retained_set = set(retained)
    reduced_edges: list[tuple[int, int]] = []
    for source in retained:
        targets = (reachability[source] & retained_set) - {source}
        covers = sorted(
            target
            for target in targets
            if not any(
                target in reachability[other]
                for other in targets
                if other != target
            )
        )
        reduced_edges.extend((source, target) for target in covers)

    node_lines = graph_node_lines(fixed / "morse_graph_fixed22.dot")
    dot = dest / "uniform_22_22_22" / "nontrivial" / "morse_graph.dot"
    dot.parent.mkdir(parents=True, exist_ok=True)
    lines = ["digraph {"]
    for node in retained:
        line = node_lines[node]
        if node == 23:
            line = re.sub(r'fillcolor="[^"]+"', 'fillcolor="#008080"', line)
        lines.append(line)
    minima = sorted(int(node) for node in result["nontrivial_skeleton"]["minimal"])
    if minima:
        lines.append("{rank=same; " + " ".join(map(str, minima)) + " };")
    lines.extend(f"{source} -> {target};" for source, target in reduced_edges)
    lines.append("}")
    dot.write_text("\n".join(lines) + "\n", encoding="utf-8")
    png = dot.with_suffix(".png")
    subprocess.run(
        ["dot", "-Tpng", str(dot), "-o", str(png)],
        check=True,
    )
    return dot, png


def render_fixed22_sets(
    csv_path: Path, result: dict[str, Any], dest: Path
) -> tuple[Path, Path]:
    values = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    lower = np.asarray(result["bounds"]["lower"], dtype=np.float64)
    upper = np.asarray(result["bounds"]["upper"], dtype=np.float64)
    sizes = np.asarray(result["subdivision"]["sizes"], dtype=np.int64)
    widths = (upper - lower) / sizes
    indices = np.rint((values[:, :2] - lower) / widths).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= sizes):
        raise RuntimeError("fixed-22 Morse boxes do not align with the saved grid")
    labels = values[:, -1].astype(np.int64)
    raster = np.full((int(sizes[1]), int(sizes[0])), -1, dtype=np.int16)
    raster[indices[:, 1], indices[:, 0]] = labels
    masked = np.ma.masked_less(raster, 0)
    node_count = int(result["fixed_graph"]["node_count"])
    cmap = ListedColormap([PALETTE[node % len(PALETTE)] for node in range(node_count)])
    cmap.set_bad(alpha=0.0)
    norm = BoundaryNorm(np.arange(-0.5, node_count + 0.5), node_count)

    png = dest / "uniform_22_22_22" / "raw" / "morse_sets.png"
    pdf = png.with_suffix(".pdf")
    fig, ax = plt.subplots(figsize=(9.5, 9.0), constrained_layout=True)
    image = ax.imshow(
        masked,
        origin="lower",
        extent=[lower[0], upper[0], lower[1], upper[1]],
        interpolation="none",
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )
    ax.set_title("Uniform (22,22,22) raw Morse sets")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=np.arange(node_count),
        fraction=0.035,
        pad=0.025,
    )
    colorbar.set_label("raw Morse node")
    colorbar.ax.tick_params(labelsize=7)
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def write_fixed22_nontrivial_sets(
    csv_path: Path,
    result: dict[str, Any],
    dest: Path,
) -> tuple[Path, Path, Path, dict[str, int]]:
    retained = sorted(
        int(node)
        for node, component in result["fixed_graph"]["components"].items()
        if any(value != "0" for value in component["conley_index"])
    )
    retained_set = set(retained)
    destination = dest / "uniform_22_22_22" / "nontrivial" / "morse_sets.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open(encoding="utf-8") as source, destination.open(
        "w",
        encoding="utf-8",
    ) as target:
        for line in source:
            label = int(float(line.rsplit(",", 1)[-1]))
            if label in retained_set:
                target.write(line)

    counts = csv_counts(destination)
    expected = {
        str(node): int(result["fixed_graph"]["components"][str(node)]["cells"])
        for node in retained
    }
    if counts != expected:
        raise RuntimeError("nontrivial fixed-22 CSV counts do not match result.json")

    values = np.loadtxt(destination, delimiter=",", dtype=np.float64)
    lower = np.asarray(result["bounds"]["lower"], dtype=np.float64)
    upper = np.asarray(result["bounds"]["upper"], dtype=np.float64)
    sizes = np.asarray(result["subdivision"]["sizes"], dtype=np.int64)
    widths = (upper - lower) / sizes
    indices = np.rint((values[:, :2] - lower) / widths).astype(np.int64)
    labels = values[:, -1].astype(np.int64)
    ordinal = {node: index for index, node in enumerate(retained)}
    raster = np.full((int(sizes[1]), int(sizes[0])), -1, dtype=np.int8)
    raster[indices[:, 1], indices[:, 0]] = np.asarray(
        [ordinal[int(label)] for label in labels],
        dtype=np.int8,
    )
    masked = np.ma.masked_less(raster, 0)
    colors = ["#FFB000", "#DC267F", "#648FFF", "#008080"]
    cmap = ListedColormap(colors)
    cmap.set_bad(alpha=0.0)
    norm = BoundaryNorm(np.arange(-0.5, len(retained) + 0.5), len(retained))

    png = destination.with_suffix(".png")
    pdf = destination.with_suffix(".pdf")
    fig, ax = plt.subplots(figsize=(10.2, 8.6), constrained_layout=True)
    ax.imshow(
        masked,
        origin="lower",
        extent=[lower[0], upper[0], lower[1], upper[1]],
        interpolation="none",
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )
    ax.set_title("Uniform (22,22,22): original nontrivial Morse sets")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")

    node23 = values[labels == 23]
    zoom_lower = node23[:, :2].min(axis=0)
    zoom_upper = node23[:, 2:4].max(axis=0)
    zoom_pad = np.maximum((zoom_upper - zoom_lower) * 0.18, widths * 3)
    inset = ax.inset_axes([0.61, 0.57, 0.35, 0.35])
    inset.imshow(
        masked,
        origin="lower",
        extent=[lower[0], upper[0], lower[1], upper[1]],
        interpolation="none",
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )
    inset.set_xlim(zoom_lower[0] - zoom_pad[0], zoom_upper[0] + zoom_pad[0])
    inset.set_ylim(zoom_lower[1] - zoom_pad[1], zoom_upper[1] + zoom_pad[1])
    inset.set_title("node 23 zoom", fontsize=10)
    inset.tick_params(labelsize=7)
    for spine in inset.spines.values():
        spine.set_linewidth(1.2)
    ax.add_patch(
        Rectangle(
            zoom_lower,
            *(zoom_upper - zoom_lower),
            fill=False,
            edgecolor=colors[ordinal[23]],
            linewidth=1.2,
        )
    )

    labels_text = {
        0: r"0: $(x^4-1,0,0)$",
        1: r"1: $(x^2-1,0,0)$",
        9: r"9: $(0,x^4-1,0)$",
        23: r"23: $(0,x+1,0)$",
    }
    handles = [
        Patch(
            facecolor=colors[ordinal[node]],
            label=f"{labels_text[node]}  ({counts[str(node)]:,} cells)",
        )
        for node in retained
    ]
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    summary = {
        "filter": "original raw fixed-22 Morse components with nonzero Conley index",
        "merged": False,
        "source": "../raw/morse_sets.csv",
        "schema": ["x_lower", "y_lower", "x_upper", "y_upper", "morse_label"],
        "total_cells": sum(counts.values()),
        "nodes": {
            str(node): {
                "cells": counts[str(node)],
                "conley_index": result["fixed_graph"]["components"][str(node)]["conley_index"],
                "minimal": result["fixed_graph"]["components"][str(node)]["minimal"],
                "extent": result["fixed_graph"]["components"][str(node)]["extent"],
            }
            for node in retained
        },
    }
    summary_path = destination.parent / "morse_sets_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return destination, png, pdf, counts


def inventory(dest: Path, *, include_manifest: bool = False) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(dest.rglob("*")):
        excluded = (
            path.name.startswith(".")
            or path.name == "SHA256SUMS"
            or (path.name == "bundle_manifest.json" and not include_manifest)
        )
        if path.is_file() and not excluded:
            result[str(path.relative_to(dest))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return result


def main(input_root: Path, dest: Path) -> None:
    fixed = input_root / "fixed22"
    merged_dir = input_root / "coarsened_45"
    metrics = fixed / "residual_tolerance"

    # This directory is a generated, self-contained bundle. Rebuild it from a
    # clean slate so stale files can never enter the manifest or checksums.
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    fixed_result = json.loads((fixed / "result.json").read_text(encoding="utf-8"))
    merged_result = json.loads((merged_dir / "result.json").read_text(encoding="utf-8"))
    metric_result = json.loads(
        _first_existing(
            metrics / "sampled_residual_tolerance.json",
            REFERENCE_METRICS / "sampled_residual_tolerance.json",
        ).read_text(encoding="utf-8")
    )

    if fixed_result["subdivision"]["init"] != 22:
        raise RuntimeError("fixed source is not depth 22")
    if not fixed_result["coarsening"]["same_component"]:
        raise RuntimeError("fixed-22 point and period-two orbit are not in one raw component")
    if fixed_result["coarsening"]["added_connection_cells"] != 0:
        raise RuntimeError("fixed-22 identity quotient unexpectedly added cells")
    if merged_result["index_pairs"]["merged_45"]["conley_index"] != ["0", "x+1", "0"]:
        raise RuntimeError("unexpected merged 4/5 Conley index")

    model_sha256, checkpoint_form = verified_model_sha256()

    fixed_sets_source = fixed / "morse_sets_fixed22_raw.csv"
    if not fixed_sets_source.is_file():
        fixed_sets_source = fixed / "morse_sets_fixed22_connection_complete.csv"
    fixed_sets = copy(fixed_sets_source, dest, "uniform_22_22_22/raw/morse_sets.csv")
    copy(fixed / "morse_graph_fixed22.dot", dest, "uniform_22_22_22/raw/morse_graph.dot")
    copy_optional(
        fixed / "morse_graph_fixed22.png", dest, "uniform_22_22_22/raw/morse_graph.png"
    )
    copy(fixed / "result.json", dest, "uniform_22_22_22/result.json")
    skipped_reports: list[str] = []
    if copy_optional(fixed / "REPORT.md", dest, "uniform_22_22_22/REPORT.md") is None:
        skipped_reports.append("uniform_22_22_22/REPORT.md")
    residual_tolerance_inputs: dict[str, str] = {}
    for name in (
        "SUMMARY.md",
        "sampled_residual_tolerance.json",
        "tolerance_sampling.json",
        "forward_closure_verification.json",
    ):
        source = _first_existing(metrics / name, REFERENCE_METRICS / name)
        copy(source, dest, f"uniform_22_22_22/residual_tolerance/{name}")
        residual_tolerance_inputs[name] = recorded_path(source)
    write_fixed22_nontrivial(fixed_result, fixed, dest)
    render_fixed22_sets(fixed_sets, fixed_result, dest)
    _, _, _, fixed_nontrivial_counts = write_fixed22_nontrivial_sets(
        fixed_sets,
        fixed_result,
        dest,
    )

    original_sets = copy(
        ACTIVE / "MG" / "morse_sets",
        dest,
        "adaptive_23_23_27/raw/morse_sets.csv",
    )
    copy(ACTIVE / "MG" / "morse_graph", dest, "adaptive_23_23_27/raw/morse_graph.dot")
    copy_optional(
        ACTIVE / "MG" / "morse_graph.png", dest, "adaptive_23_23_27/raw/morse_graph.png"
    )
    copy_optional(
        ACTIVE / "MG" / "morse_graph.pdf", dest, "adaptive_23_23_27/raw/morse_graph.pdf"
    )
    copy_optional(
        ACTIVE / "MG" / "morse_sets.png", dest, "adaptive_23_23_27/raw/morse_sets.png"
    )
    copy_optional(
        ACTIVE / "MG" / "morse_sets.pdf", dest, "adaptive_23_23_27/raw/morse_sets.pdf"
    )

    copy(
        merged_dir / "morse_graph_coarse.dot",
        dest,
        "adaptive_23_23_27/merged_4_5/morse_graph.dot",
    )
    copy_optional(
        merged_dir / "morse_graph_coarse.png",
        dest,
        "adaptive_23_23_27/merged_4_5/morse_graph.png",
    )
    merged_sets = copy(
        merged_dir / "morse_sets_connection_complete.csv",
        dest,
        "adaptive_23_23_27/merged_4_5/morse_sets.csv",
    )
    copy(
        merged_dir / "morse_set_45_connection_complete.png",
        dest,
        "adaptive_23_23_27/merged_4_5/morse_set_4_5.png",
    )
    copy(merged_dir / "result.json", dest, "adaptive_23_23_27/merged_4_5/result.json")
    if (
        copy_optional(
            merged_dir / "REPORT.md", dest, "adaptive_23_23_27/merged_4_5/REPORT.md"
        )
        is None
    ):
        skipped_reports.append("adaptive_23_23_27/merged_4_5/REPORT.md")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "render_leslie3d_example1_figures.py"),
            "--output",
            str(dest),
        ],
        check=True,
    )

    copy(ACTIVE / "run_manifest.json", dest, "provenance/active_run_manifest.json")
    copy(ACTIVE / "mg_params_log.txt", dest, "provenance/active_mg_params_log.txt")
    copy(
        REPO_ROOT
        / "src"
        / "latentdynamics"
        / "configs"
        / "leslie3d_example1_replay.yaml",
        dest,
        "provenance/leslie3d_example1_replay.yaml",
    )
    copy_optional(PROVENANCE_REFERENCE, dest, "provenance/g1_provenance_full.md")
    for name in BUNDLED_SCRIPTS:
        copy(SCRIPT_DIR / name, dest, f"scripts/{name}")

    fixed_counts = csv_counts(fixed_sets)
    expected_fixed_counts = {
        str(node): int(component["cells"])
        for node, component in fixed_result["fixed_graph"]["components"].items()
    }
    if fixed_counts != expected_fixed_counts:
        raise RuntimeError("fixed-22 CSV label counts do not match result.json")
    original_counts = csv_counts(original_sets)
    merged_counts = csv_counts(merged_sets)

    concise_index = {
        "fine_nodes": [4, 5],
        "fine_indices": {
            "4": merged_result["index_pairs"]["fine_node_4"]["conley_index"],
            "5": merged_result["index_pairs"]["fine_node_5"]["conley_index"],
        },
        "literal_union_without_connections": {
            "cells": merged_result["index_pairs"]["literal_union_without_connections"]["morse_set_cells"],
            "pair_valid": merged_result["index_pairs"]["literal_union_without_connections"]["pair_valid"],
            "conley_index": merged_result["index_pairs"]["literal_union_without_connections"]["conley_index"],
        },
        "connection_complete_merge": {
            "node_4_cells": merged_result["connection_completion"]["node4_cells"],
            "node_5_cells": merged_result["connection_completion"]["node5_cells"],
            "added_connection_cells": merged_result["connection_completion"]["added_connection_cells"],
            "total_cells": merged_result["connection_completion"]["merged_cells"],
            "pair_valid": merged_result["index_pairs"]["merged_45"]["pair_valid"],
            "conley_index": merged_result["index_pairs"]["merged_45"]["conley_index"],
        },
        "coarse_graph": merged_result["coarse_graph"],
        "note": (
            "The merged calculation is a fresh live recomputation using the active "
            "checkpoint/config. Saved-vs-live nodes 4 and 5 match exactly; small "
            "13-cell differences occur only in nodes 0 and 1 and are recorded in result.json."
        ),
    }
    concise_path = dest / "adaptive_23_23_27" / "merged_4_5" / "merged_conley_index.json"
    concise_path.write_text(json.dumps(concise_index, indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Bundle leslie3d_example1 uniform (22,22,22), active adaptive "
            "(23,23,27), and connection-complete 4/5 merge artifacts."
        ),
        "provenance": {
            "checkpoint_origin": {
                "description": (
                    "Author-provided 2026-05-03 legacy three-file checkpoint; the "
                    "active replay model files are byte-identical to the recorded "
                    "author-archive checksums."
                    if checkpoint_form == "legacy_three_file"
                    else "Author-provided 2026-05-03 legacy three-file checkpoint, "
                    "shipped as its bitwise-equivalent single-file migration; the "
                    "active migrated checkpoint is byte-identical to the checksum "
                    "recorded in artifacts/provenance/MIGRATION_RECORD.json."
                ),
                "checkpoint_form": checkpoint_form,
                "active_path": recorded_path(ACTIVE / "models"),
                "sha256": model_sha256,
                "author_archive_sha256": dict(AUTHOR_ARCHIVE_MODEL_SHA256),
                "migration_record": recorded_path(MIGRATION_RECORD),
            },
            "active_original_morse_source": {
                "description": (
                    "Later wide-domain paper replay on the author-provided "
                    "checkpoint; not the narrower author-archive Morse cover."
                ),
                "path": recorded_path(ACTIVE),
                "nominal_subdivision": [23, 23, 27],
                "effective_morse_box_depth": 23,
                "bounds": {
                    "lower": fixed_result["bounds"]["lower"],
                    "upper": fixed_result["bounds"]["upper"],
                },
                "padding": True,
                "subdivision_limit": 10000,
            },
            "author_archive_morse_source": {
                "note": (
                    "The narrower run A cover in the coauthor's private "
                    "development archive is deliberately not substituted for the "
                    "active paper replay in this bundle; see "
                    "provenance/g1_provenance_full.md (checksums retained)."
                ),
            },
        },
        "uniform_22_22_22": {
            "grid_sizes": fixed_result["subdivision"]["sizes"],
            "cell_count": fixed_result["subdivision"]["cell_count"],
            "raw_node_count": fixed_result["fixed_graph"]["node_count"],
            "raw_minimal_nodes": fixed_result["fixed_graph"]["minimal"],
            "morse_set_rows_by_label": fixed_counts,
            "nontrivial_morse_set_rows_by_label": fixed_nontrivial_counts,
            "fixed_and_period_two_raw_node": fixed_result["distinguished_objects"]["fixed_point"]["node"],
            "existing_component_conley_index": fixed_result["coarsening"]["connection_complete"]["conley_index"],
            "merge_operation": "identity_existing_component",
            "added_connection_cells": 0,
            "residual_tolerance_inputs": residual_tolerance_inputs,
            "sampled_residual_tolerance": {
                "metric": metric_result["metric"],
                "nodes": {
                    node: {
                        "sampled_residual": data["residual"]["sampled_maximum"],
                        "sampled_tolerance": data["tolerance"]["sampled_minimum"],
                        "ratio": data["comparison"][
                            "sampled_residual_over_sampled_tolerance"
                        ],
                        "residual_candidates": data["residual"]["evaluated_samples"],
                        "accepted_residual_samples": data["residual"][
                            "accepted_samples"
                        ],
                        "tolerance_samples": data["tolerance"]["sample_count"],
                    }
                    for node, data in metric_result["nodes"].items()
                },
            },
        },
        "adaptive_23_23_27": {
            "raw_node_count": len(original_counts),
            "morse_set_rows_by_label": original_counts,
            "raw_graph_sha256": sha256(ACTIVE / "MG" / "morse_graph"),
            "raw_sets_sha256": sha256(ACTIVE / "MG" / "morse_sets"),
            "merged_4_5": {
                "morse_set_rows_by_label": merged_counts,
                "conley_index": concise_index["connection_complete_merge"]["conley_index"],
                "added_connection_cells": concise_index["connection_complete_merge"]["added_connection_cells"],
                "total_merged_cells": concise_index["connection_complete_merge"]["total_cells"],
                "display_color": "#785EF0",
                "display_color_semantic": "ground-truth period-2 saddle color",
            },
        },
        "paper_ready_figures": {
            "path": "paper_ready_no_legend",
            "intended_latex_width": "0.5\\textwidth",
            "minimum_box_side_fraction_full_views": 0.005,
            "zoom_views_use_exact_box_sizes": True,
            "legends": False,
            "titles": False,
            "numeric_ticks": False,
            "merged_4_5_display_color": "#785EF0",
        },
    }
    if skipped_reports:
        manifest["missing_optional_inputs"] = {
            "note": (
                "these REPORT.md files were absent from the inputs and are "
                "omitted from the bundle; rerunning the analysis scripts "
                "regenerates them"
            ),
            "paths": skipped_reports,
        }

    bundle_rel = recorded_path(dest)
    if checkpoint_form == "legacy_three_file":
        checkpoint_paragraph = (
            "The neural-network checkpoint is the author-provided 2026-05-03 legacy\n"
            "three-file checkpoint. The three active model files under\n"
            f"`{recorded_path(ACTIVE / 'models')}` are byte-identical to the checksums\n"
            "recorded for the coauthor's private development archive (see\n"
            "`provenance/g1_provenance_full.md`)."
        )
    else:
        checkpoint_paragraph = (
            "The neural-network checkpoint is the author-provided 2026-05-03 legacy\n"
            "three-file checkpoint, shipped as its bitwise-equivalent single-file\n"
            f"migration under `{recorded_path(ACTIVE / 'models')}`. The active migrated\n"
            "checkpoint is byte-identical to the checksum recorded in\n"
            "`artifacts/provenance/MIGRATION_RECORD.json`; the original author-archive\n"
            "checksums are retained in `bundle_manifest.json` and\n"
            "`provenance/g1_provenance_full.md`."
        )
    readme = f"""# leslie3d_example1: uniform 22 versus active adaptive run

## Where this run comes from

{checkpoint_paragraph}

The **original finer Morse graph used in this comparison is not the literal
author-archive Morse cover**. It is the later wide-domain paper replay stored
at `{recorded_path(ACTIVE)}` and computed on the same checkpoint. The literal
archived run A used narrower bounds. See `provenance/g1_provenance_full.md`
for the complete chain.

The active original is nominally `(subdiv_init, subdiv_min, subdiv_max) = (23,23,27)`, but every saved recurrent box is effectively at depth 23 because the 10,000 subdivision limit prevented refinement to 27. Relative to uniform depth 22, it has one additional split in the first latent coordinate.

## Folder map

```text
uniform_22_22_22/
  raw/morse_graph.dot,png       untouched 24-node graph
  raw/morse_sets.csv,png,pdf    all 89,449 raw boxes
  nontrivial/morse_graph.dot,png,pdf
                                 raw nonzero-index nodes only; nothing merged
  nontrivial/morse_sets.csv,png,pdf
                                 nodes 0, 1, 9, and 23 with a node-23 inset
  nontrivial/morse_sets_full_no_legend.png,pdf
                                 matched full view without the inset
  nontrivial/morse_set_23_zoom_no_legend.png,pdf
                                 exact-size local view with fixed/P2 markers
  nontrivial/morse_sets_summary.json
                                 counts, indices, and coordinate extents
  residual_tolerance/SUMMARY.md  concise sampled metric table
  residual_tolerance/sampled_residual_tolerance.json
                                 full residual/tolerance protocol and witnesses
  residual_tolerance/tolerance_sampling.json
                                 full latent tolerance search record
  residual_tolerance/forward_closure_verification.json
                                 exact map-graph closure replay
  result.json                   complete machine-readable computation

adaptive_23_23_27/
  raw/morse_graph.dot,png,pdf   active six-node original graph
  raw/morse_sets.csv,png,pdf    active 122,346-row original cover
  raw/morse_sets_no_legend.png,pdf
                                 half-textwidth paper rerender
  raw/morse_sets_with_separate_4_5_zoom_no_legend.png,pdf
                                 full original view with a marker-free inset
                                 separating sets 4 and 5 and connection cells
  merged_4_5/
    morse_graph.dot,png,pdf     five-node quotient graph; node is `[4,5]`
    morse_sets.csv              full connection-complete relabeling
    morse_sets_no_legend.png,pdf
                               full connection-complete set view
    morse_set_4_5.png,pdf       exact-size zoom of sets 4 and 5 only;
                                 no legend or fixed/P2 marker layer
    merged_conley_index.json    concise answer
    result.json                 full checks and diagnostics

provenance/                     source manifest, settings, config, full audit
scripts/                        frozen copies of the analysis/package scripts
paper_ready_no_legend/          all paper-ready PDFs plus PNG previews
bundle_manifest.json            counts, settings, provenance and hashes
SHA256SUMS                      integrity checks for every bundled file
```

## Main numerical result

For the active adaptive graph, nodes 4 and 5 contain 174 and 123 cells. Their literal 297-cell union is not a valid index pair. Adding the 25 internal connection cells gives a valid 322-cell coarse Morse set with Conley index:

```text
(0, x+1, 0)
```

The quotient graph has edges `2→0`, `2→1`, `3→1`, and `[4,5]→3`, with minima `0` and `1`.

The connection-complete merged node `[4,5]` is displayed in purple (`#785EF0`) in the coarse graph and full merged-set panel, matching the ground-truth period-2 saddle palette. The green/teal color remains reserved for the zero-associated object.

At uniform `(22,22,22)`, the fixed point and both period-two phases are already in raw node 23. No quotient and no added connection cells are needed; that existing 291-cell component has the same index `(0,x+1,0)`.

For the two minimal fixed-22 components that give bistability, the dense sampled comparison is:

| Node | Boxes | Sampled residual | Sampled tolerance | Ratio |
|---:|---:|---:|---:|---:|
| 0 | {metric_result['nodes']['0']['n_boxes']:,} | {metric_result['nodes']['0']['residual']['sampled_maximum']:.12g} | {metric_result['nodes']['0']['tolerance']['sampled_minimum']:.12g} | {metric_result['nodes']['0']['comparison']['sampled_residual_over_sampled_tolerance']:.6g} |
| 1 | {metric_result['nodes']['1']['n_boxes']:,} | {metric_result['nodes']['1']['residual']['sampled_maximum']:.12g} | {metric_result['nodes']['1']['tolerance']['sampled_minimum']:.12g} | {metric_result['nodes']['1']['comparison']['sampled_residual_over_sampled_tolerance']:.6g} |

The sampled residual exceeds the sampled tolerance for both components. This numerically contradicts the strict sufficient inequality on the evaluated blocks; it does not by itself classify either attractor as spurious. An exact cell-graph replay verifies that each minimal recurrent component equals its full forward closure. Node 23 is nonminimal and is excluded from this bistability comparison.

## Paper-ready figures

`paper_ready_no_legend/` contains matched original, merged, uniform-22, and
zoom figures. Copying panels into a manuscript's figure tree is a
manuscript-maintainer action performed outside this repository.

The full Morse-set panels are authored for `0.5\\textwidth` and use a display-only minimum box side of 0.5% of the corresponding axis span. The zooms use exact box sizes. No paper-ready panel has a legend, title, or numeric tick labels.

## File formats and access

- Graph `.png`/`.pdf`: open directly.
- Graph `.dot`: exact Graphviz node labels and Hasse edges; rerender with `dot -Tpng morse_graph.dot -o graph.png`.
- Morse-set `.csv`: headerless rows `x_lower,y_lower,x_upper,y_upper,morse_label`.
- `result.json`: complete graph, cell-count, index-pair, and Conley-index record.

From the repository root:

```bash
# Fixed-22 raw graph summary
jq '.fixed_graph | {{node_count, minimal, edges}}' \\
  {bundle_rel}/uniform_22_22_22/result.json

# Fixed point and period-two ownership at depth 22
jq '.distinguished_objects' \\
  {bundle_rel}/uniform_22_22_22/result.json

# Fixed-22 nontrivial Morse-set counts, indices, and extents
jq '.' \\
  {bundle_rel}/uniform_22_22_22/nontrivial/morse_sets_summary.json

# Fixed-22 sampled residual/tolerance comparison
jq '.nodes | with_entries(.value |= {{R_hat: .residual.sampled_maximum, tau_hat: .tolerance.sampled_minimum, ratio: .comparison.sampled_residual_over_sampled_tolerance}})' \\
  {bundle_rel}/uniform_22_22_22/residual_tolerance/sampled_residual_tolerance.json

# Concise merged Conley-index answer
jq '.' \\
  {bundle_rel}/adaptive_23_23_27/merged_4_5/merged_conley_index.json

# Full validity checks for the 4/5 merge
jq '.connection_completion, .index_pairs' \\
  {bundle_rel}/adaptive_23_23_27/merged_4_5/result.json
```

Python example from the repository root:

```python
import pandas as pd
from pathlib import Path

columns = ["x_lower", "y_lower", "x_upper", "y_upper", "morse_label"]
bundle = Path("{bundle_rel}")
boxes = pd.read_csv(bundle / "uniform_22_22_22/raw/morse_sets.csv", names=columns)
print(boxes.groupby("morse_label").size())
```

## Reproducibility note

The adaptive merged result came from a fresh live recomputation using the active checkpoint and recorded settings. Nodes 4 and 5 match the saved original boxes exactly. Nodes 0 and 1 differ by 1 and 12 cells respectively from the saved cover; this small backend-sensitive discrepancy is recorded under `.live_graph.matching` in `merged_4_5/result.json` and does not touch the merged region.
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")

    manifest["files"] = inventory(dest)
    manifest_path = dest / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    files = inventory(dest, include_manifest=True)
    sums = "\n".join(
        f"{metadata['sha256']}  {relative}"
        for relative, metadata in sorted(files.items())
    )
    (dest / "SHA256SUMS").write_text(sums + "\n", encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "study root holding coarsened_45/ and fixed22/ "
            "(default: output/leslie3d_example1_study)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="bundle destination (default: <input-root>/fixed22_vs_adaptive)",
    )
    arguments = parser.parse_args()
    destination = (
        arguments.output
        if arguments.output is not None
        else arguments.input_root / "fixed22_vs_adaptive"
    )
    main(arguments.input_root, destination)
