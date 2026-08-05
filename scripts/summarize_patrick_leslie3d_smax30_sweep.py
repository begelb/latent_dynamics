#!/usr/bin/env python3
"""Render, summarize, and report the Patrick Leslie3D max-30 CMGDB sweep."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Patch
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from latentdynamics.analysis.morse_graph_parser import MorseGraph
from latentdynamics.viz.morse_plots import (
    plot_morse_sets_from_csv,
    render_morse_graph_from_dot,
)
from latentdynamics.viz.style import PALETTE, apply_paper_style, style_latent_axes


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"
EXPERIMENT = "leslie3d_example2_patrick_cmgdb_initmin_sweep_max30_limit1000000_v1"
SWEEP_ROOT = CODE_ROOT / "output" / EXPERIMENT
SUMMARY_ROOT = SWEEP_ROOT / "summary"
PDF_ROOT = WORKSPACE_ROOT / "output" / "pdf" / "leslie3d_example2_patrick_smax30_sweep"
FINAL_PDF = PDF_ROOT / "leslie3d_example2_patrick_smax30_sweep_summary.pdf"
ARCHIVED_GRAPH = CODE_ROOT / "replay_sources" / "leslie3d_example2" / "MG" / "morse_graph"

LOWER = np.array([-0.37490714, -0.4695556], dtype=np.float64)
UPPER = np.array([0.3535685, 0.455769], dtype=np.float64)
# Keep the finest covers legible in the report's compact 2-by-3 layout.  This
# floor is display-only: raw box coordinates and all area calculations remain
# unchanged.
MIN_BOX_SIDE_FRAC = 0.004
EXPECTED_RUNS = (
    "run_00_init16_min18_max30",
    "run_01_init18_min20_max30",
    "run_02_init20_min22_max30",
    "run_03_init22_min24_max30",
    "run_04_init24_min26_max30",
    "run_05_init26_min28_max30",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _artifact_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(relative_to)),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _load_runs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sweep_path = SWEEP_ROOT / "sweep_summary.json"
    if not sweep_path.is_file():
        raise FileNotFoundError(f"run the sweep first: {sweep_path}")
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    if sweep.get("status") != "complete" or sweep.get("completed_run_count") != 6:
        raise ValueError(f"sweep is incomplete: {sweep_path}")
    by_id = {row["run_id"]: row for row in sweep["runs"]}
    missing = [run_id for run_id in EXPECTED_RUNS if run_id not in by_id]
    if missing:
        raise ValueError(f"sweep summary is missing expected runs: {missing}")
    runs = [by_id[run_id] for run_id in EXPECTED_RUNS]
    for row in runs:
        run_root = SWEEP_ROOT / "runs" / row["run_id"]
        marker = json.loads((run_root / "stage_morse_complete.json").read_text())
        if marker.get("status") != "complete":
            raise ValueError(f"invalid completion marker: {run_root}")
        for record in marker["artifacts"].values():
            artifact = run_root / record["path"]
            if _sha256(artifact) != record["sha256"]:
                raise ValueError(f"artifact hash mismatch: {artifact}")
    return sweep, runs


def _graph_signature(path: Path) -> tuple[tuple[tuple[int, int], ...], dict[int, str]]:
    graph = MorseGraph.from_dot(path)
    edges = tuple(
        sorted(
            (source, target)
            for source, targets in graph.edges.items()
            for target in targets
        )
    )
    return edges, {node: graph.labels.get(node, "") for node in graph.nodes}


def _labeled_graph_isomorphic(path_a: Path, path_b: Path) -> bool:
    edges_a, labels_a = _graph_signature(path_a)
    edges_b, labels_b = _graph_signature(path_b)
    if len(labels_a) != len(labels_b) or len(edges_a) != len(edges_b):
        return False
    if sorted(labels_a.values()) != sorted(labels_b.values()):
        return False
    nodes_a = sorted(labels_a)
    groups_b: dict[str, list[int]] = {}
    for node, label in labels_b.items():
        groups_b.setdefault(label, []).append(node)
    choices = [tuple(itertools.permutations(groups_b[labels_a[node]])) for node in nodes_a]
    grouped_a: dict[str, list[int]] = {}
    for node in nodes_a:
        grouped_a.setdefault(labels_a[node], []).append(node)
    group_labels = sorted(grouped_a)
    group_permutations = [
        list(itertools.permutations(sorted(groups_b[label])))
        for label in group_labels
    ]
    edge_set_b = set(edges_b)
    for selected in itertools.product(*group_permutations):
        mapping: dict[int, int] = {}
        for label, targets in zip(group_labels, selected, strict=True):
            for source, target in zip(sorted(grouped_a[label]), targets, strict=True):
                mapping[source] = target
        mapped_edges = {(mapping[source], mapping[target]) for source, target in edges_a}
        if mapped_edges == edge_set_b:
            return True
    return False


def _render_individual_outputs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    apply_paper_style()
    for row in runs:
        run_root = SWEEP_ROOT / "runs" / row["run_id"]
        mg_root = run_root / "MG"
        graph_paths = render_morse_graph_from_dot(
            mg_root / "morse_graph",
            mg_root,
            formats=("pdf", "png"),
            palette=PALETTE,
        )
        plot = plot_morse_sets_from_csv(
            mg_root / "morse_sets",
            bounds_lower=LOWER,
            bounds_upper=UPPER,
            paper_style=True,
            box_scale=1.0,
            min_box_side_frac=MIN_BOX_SIDE_FRAC,
        )
        plot.fig.set_size_inches(6.4, 5.2)
        params = row["parameters"]
        plot.ax.set_title(
            f"init={params['subdiv_init']}, min={params['subdiv_min']}, max=30",
            fontsize=13,
        )
        set_paths: list[Path] = []
        for suffix in ("pdf", "png"):
            path = mg_root / f"morse_sets.{suffix}"
            plot.fig.savefig(path, dpi=300, bbox_inches="tight")
            set_paths.append(path)
        plt.close(plot.fig)
        artifacts = {
            path.name: _artifact_record(path, relative_to=run_root)
            for path in (*graph_paths, *set_paths)
        }
        render_manifest = {
            "schema_version": 1,
            "run_id": row["run_id"],
            "morse_set_render": {
                "box_scale": 1.0,
                "min_box_side_frac": MIN_BOX_SIDE_FRAC,
                "floor_basis": "adaptive_occupied_view",
                "bounds_lower": LOWER.tolist(),
                "bounds_upper": UPPER.tolist(),
                "palette": PALETTE,
            },
            "artifacts": artifacts,
        }
        _write_json_atomic(run_root / "render_manifest.json", render_manifest)
        records.append(render_manifest)
    return records


def _crop_image(path: Path, *, padding: int = 20) -> np.ndarray:
    with PILImage.open(path).convert("RGB") as image:
        array = np.asarray(image)
    nonwhite = np.any(array < 248, axis=2)
    if not np.any(nonwhite):
        return array
    ys, xs = np.nonzero(nonwhite)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(array.shape[0], int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(array.shape[1], int(xs.max()) + padding + 1)
    return array[y0:y1, x0:x1]


def _build_graph_contact_sheet(runs: list[dict[str, Any]]) -> tuple[Path, Path]:
    apply_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), layout="constrained")
    for ax, row in zip(axes.flat, runs, strict=True):
        run_root = SWEEP_ROOT / "runs" / row["run_id"]
        image = _crop_image(run_root / "MG" / "morse_graph.png")
        ax.imshow(image)
        ax.axis("off")
        params = row["parameters"]
        graph = row["graph"]
        ax.set_title(
            f"init/min/max = {params['subdiv_init']}/{params['subdiv_min']}/30\n"
            f"{graph['node_count']} nodes, {graph['edge_count']} edges, "
            f"{len(graph['minimal_nodes'])} minimal",
            fontsize=11,
            pad=5,
        )
    fig.suptitle("Patrick latent dynamics: Morse graphs across subdivision ladders", fontsize=16)
    png = SUMMARY_ROOT / "graph_contact_sheet.png"
    pdf = SUMMARY_ROOT / "graph_contact_sheet.pdf"
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def _build_set_contact_sheet(runs: list[dict[str, Any]]) -> tuple[Path, Path]:
    apply_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), layout="constrained")
    all_labels: set[int] = set()
    for index, (ax, row) in enumerate(zip(axes.flat, runs, strict=True)):
        run_root = SWEEP_ROOT / "runs" / row["run_id"]
        csv_path = run_root / "MG" / "morse_sets"
        data = np.loadtxt(csv_path, delimiter=",", ndmin=2)
        if data.shape[1] != 5 or not np.all(np.isfinite(data)):
            raise ValueError(f"invalid 2-D Morse-set CSV: {csv_path}")
        min_width = MIN_BOX_SIDE_FRAC * float(UPPER[0] - LOWER[0])
        min_height = MIN_BOX_SIDE_FRAC * float(UPPER[1] - LOWER[1])
        rects: list[mpatches.Rectangle] = []
        facecolors: list[str] = []
        for box_lx, box_ly, box_ux, box_uy, label_value in data:
            label = int(label_value)
            width = max(float(box_ux - box_lx), min_width)
            height = max(float(box_uy - box_ly), min_height)
            center_x = 0.5 * float(box_lx + box_ux)
            center_y = 0.5 * float(box_ly + box_uy)
            rects.append(
                mpatches.Rectangle(
                    (center_x - 0.5 * width, center_y - 0.5 * height),
                    width,
                    height,
                )
            )
            facecolors.append(PALETTE[label % len(PALETTE)])
        ax.add_collection(
            PatchCollection(
                rects,
                facecolors=facecolors,
                edgecolors="none",
                rasterized=True,
            )
        )
        all_labels.update(int(value) for value in np.unique(data[:, -1]))
        ax.set_xlim(float(LOWER[0]), float(UPPER[0]))
        ax.set_ylim(float(LOWER[1]), float(UPPER[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        style_latent_axes(ax, two_d=True)
        params = row["parameters"]
        ax.set_title(
            f"init/min/max = {params['subdiv_init']}/{params['subdiv_min']}/30\n"
            f"{row['morse_sets']['row_count']:,} boxes",
            fontsize=11,
        )
        if index % 3:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        if index < 3:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
    handles = [
        Patch(
            facecolor=PALETTE[label % len(PALETTE)],
            edgecolor="none",
            label=f"Morse node {label}",
        )
        for label in sorted(all_labels)
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Patrick latent dynamics: Morse-set covers with a fixed visibility floor",
        fontsize=16,
    )
    png = SUMMARY_ROOT / "morse_set_contact_sheet.png"
    pdf = SUMMARY_ROOT / "morse_set_contact_sheet.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def _write_tables(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in runs:
        params = row["parameters"]
        graph = row["graph"]
        sets = row["morse_sets"]
        run_root = SWEEP_ROOT / "runs" / row["run_id"]
        archive_match = _labeled_graph_isomorphic(run_root / "MG" / "morse_graph", ARCHIVED_GRAPH)
        rows.append(
            {
                "run_id": row["run_id"],
                "subdiv_init": params["subdiv_init"],
                "subdiv_min": params["subdiv_min"],
                "subdiv_max": params["subdiv_max"],
                "subdiv_limit": params["subdiv_limit"],
                "node_count": graph["node_count"],
                "edge_count": graph["edge_count"],
                "minimal_count": len(graph["minimal_nodes"]),
                "minimal_nodes": ";".join(map(str, graph["minimal_nodes"])),
                "box_count": sets["row_count"],
                "summed_cover_area": sets["summed_cover_area"],
                "graph_seconds": row["timing"]["graph_seconds"],
                "peak_rss_gib": row["memory"]["sampled_peak_rss_bytes"] / (1024**3),
                "archived_labeled_graph_isomorphic": archive_match,
            }
        )
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_ROOT / "cells.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json_atomic(SUMMARY_ROOT / "cells.json", rows)
    return rows


def _build_trend_plot(rows: list[dict[str, Any]]) -> Path:
    apply_paper_style()
    labels = [f"{row['subdiv_init']}/{row['subdiv_min']}" for row in rows]
    x = np.arange(len(rows))
    runtime = np.asarray([row["graph_seconds"] for row in rows], dtype=float)
    area = np.asarray([row["summed_cover_area"] for row in rows], dtype=float)
    fig, left = plt.subplots(figsize=(9.5, 2.5), layout="constrained")
    right = left.twinx()
    left.plot(x, runtime, color="#16324F", marker="o", linewidth=2, label="graph time")
    right.plot(x, area, color="#007C83", marker="s", linewidth=2, label="cover area")
    left.set_xticks(x, labels)
    left.set_xlabel("subdiv_init / subdiv_min (subdiv_max = 30)")
    left.set_ylabel("Graph time (s)", color="#16324F")
    right.set_ylabel("Summed cover area", color="#007C83")
    left.tick_params(axis="y", colors="#16324F")
    right.tick_params(axis="y", colors="#007C83")
    left.grid(axis="y", alpha=0.2)
    path = SUMMARY_ROOT / "runtime_cover_area.png"
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


def _register_fonts() -> tuple[str, str]:
    regular_path = findfont(FontProperties(family="STIXGeneral", weight="normal"))
    bold_path = findfont(FontProperties(family="STIXGeneral", weight="bold"))
    pdfmetrics.registerFont(TTFont("STIXGeneralReport", regular_path))
    pdfmetrics.registerFont(TTFont("STIXGeneralReportBold", bold_path))
    return "STIXGeneralReport", "STIXGeneralReportBold"


def _build_pdf(
    sweep: dict[str, Any],
    rows: list[dict[str, Any]],
    graph_sheet: Path,
    set_sheet: Path,
    trend: Path,
) -> Path:
    PDF_ROOT.mkdir(parents=True, exist_ok=True)
    regular, bold = _register_fonts()
    navy = colors.HexColor("#16324F")
    teal = colors.HexColor("#007C83")
    ink = colors.HexColor("#18212B")
    grey = colors.HexColor("#6B7280")
    light = colors.HexColor("#F3F5F7")
    grid = colors.HexColor("#D5DBE1")
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "SweepTitle",
        parent=styles["Title"],
        fontName=bold,
        fontSize=21,
        leading=23,
        textColor=navy,
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "SweepSubtitle",
        parent=styles["BodyText"],
        fontName=regular,
        fontSize=9.5,
        leading=12,
        textColor=grey,
        spaceAfter=8,
    )
    heading = ParagraphStyle(
        "SweepHeading",
        parent=styles["Heading2"],
        fontName=bold,
        fontSize=14,
        leading=16,
        textColor=navy,
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "SweepBody",
        parent=styles["BodyText"],
        fontName=regular,
        fontSize=8.8,
        leading=11.2,
        textColor=ink,
        spaceAfter=5,
    )
    caption = ParagraphStyle(
        "SweepCaption",
        parent=styles["BodyText"],
        fontName=regular,
        fontSize=7.8,
        leading=9.5,
        textColor=grey,
        spaceBefore=4,
    )
    table_header = ParagraphStyle(
        "SweepTableHeader",
        parent=body,
        fontName=bold,
        fontSize=7.7,
        leading=9,
        textColor=colors.white,
        alignment=1,
        spaceAfter=0,
    )

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setTitle("Patrick Leslie3D CMGDB subdivision sweep")
        canvas.setAuthor("Latent Dynamics reproducibility workflow")
        canvas.setStrokeColor(grid)
        canvas.line(0.38 * inch, 0.33 * inch, 10.62 * inch, 0.33 * inch)
        canvas.setFont(regular, 7.2)
        canvas.setFillColor(grey)
        canvas.drawString(
            0.4 * inch,
            0.19 * inch,
            "Patrick archived latent map - controlled current-code recomputation",
        )
        canvas.drawRightString(10.6 * inch, 0.19 * inch, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(FINAL_PDF),
        pagesize=landscape(letter),
        leftMargin=0.42 * inch,
        rightMargin=0.42 * inch,
        topMargin=0.38 * inch,
        bottomMargin=0.43 * inch,
        title="Patrick Leslie3D CMGDB subdivision sweep",
        author="Latent Dynamics reproducibility workflow",
    )

    precompute = sweep["shared_precompute"]
    first_two = next((row for row in rows if row["minimal_count"] == 2), None)
    transition_text = (
        "No run produced two minimal Morse sets."
        if first_two is None
        else (
            "The first run in this ladder with two minimal Morse sets is "
            f"init/min = {first_two['subdiv_init']}/{first_two['subdiv_min']}."
        )
    )
    story: list[Any] = [
        Paragraph("Patrick Leslie3D latent dynamics: CMGDB subdivision sweep", title),
        Paragraph(
            "Six decompositions share one level-30 neural corner table. The archived "
            "checkpoint, latent bounds, corner rule, padding, maximum subdivision, and "
            "subdivision limit are fixed; only the initial and minimum subdivision depths vary.",
            subtitle,
        ),
        Paragraph("Run summary", heading),
        Paragraph(
            f"The shared table contains {precompute['corner_point_count']:,} corner points "
            f"and was built once in {precompute['duration_seconds']:.2f} s. "
            f"Every graph uses subdiv_max=30 and subdiv_limit=1,000,000. {transition_text}",
            body,
        ),
    ]
    table_data = [[
        Paragraph("init", table_header),
        Paragraph("min", table_header),
        Paragraph("nodes", table_header),
        Paragraph("edges", table_header),
        Paragraph("minimal", table_header),
        Paragraph("boxes", table_header),
        Paragraph("cover area", table_header),
        Paragraph("graph time", table_header),
        Paragraph("peak RSS", table_header),
        Paragraph("archived graph", table_header),
    ]]
    for row in rows:
        table_data.append(
            [
                str(row["subdiv_init"]),
                str(row["subdiv_min"]),
                str(row["node_count"]),
                str(row["edge_count"]),
                str(row["minimal_count"]),
                f"{row['box_count']:,}",
                f"{row['summed_cover_area']:.3e}",
                f"{row['graph_seconds']:.2f} s",
                f"{row['peak_rss_gib']:.2f} GiB",
                "match" if row["archived_labeled_graph_isomorphic"] else "different",
            ]
        )
    table = Table(
        table_data,
        colWidths=[
            value * inch
            for value in (0.43, 0.43, 0.52, 0.50, 0.58, 0.72, 0.82, 0.77, 0.72, 0.96)
        ],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), bold),
                ("FONTNAME", (0, 1), (-1, -1), regular),
                ("FONTSIZE", (0, 0), (-1, -1), 7.7),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
                ("GRID", (0, 0), (-1, -1), 0.35, grid),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 0.10 * inch),
            Image(str(trend), width=9.7 * inch, height=2.55 * inch),
            Paragraph(
                "Box counts are not physical size: adaptive runs may save cells at different "
                "depths. The table therefore reports both row count and summed cover area. "
                "The shared precompute time is reported once and is not included in each "
                "graph time.",
                caption,
            ),
            PageBreak(),
            Paragraph("Morse graph comparison", heading),
            Image(str(graph_sheet), width=9.65 * inch, height=6.04 * inch),
            Paragraph(
                "Graphs use the paper palette and native CMGDB node labels. Edge order in DOT "
                "serialization is ignored; the summary's archived comparison uses labeled-DAG "
                "isomorphism, allowing harmless node renumbering.",
                caption,
            ),
            PageBreak(),
            Paragraph("Morse-set comparison", heading),
            Image(str(set_sheet), width=9.46 * inch, height=5.92 * inch),
            Paragraph(
                "Common archived bounds; fixed 0.4% display-only minimum side length. Saved "
                "boxes and reported areas are unchanged. These are learned-map outer covers, "
                "not certified direct-system invariant images.",
                caption,
            ),
        ]
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(str(FINAL_PDF))
    if len(reader.pages) != 3:
        raise ValueError(f"summary PDF has {len(reader.pages)} pages; expected 3")
    return FINAL_PDF


def main() -> int:
    sweep, runs = _load_runs()
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    PDF_ROOT.mkdir(parents=True, exist_ok=True)
    render_records = _render_individual_outputs(runs)
    rows = _write_tables(runs)
    trend = _build_trend_plot(rows)
    graph_png, graph_pdf = _build_graph_contact_sheet(runs)
    set_png, set_pdf = _build_set_contact_sheet(runs)
    final_pdf = _build_pdf(sweep, rows, graph_png, set_png, trend)
    local_pdf = SUMMARY_ROOT / final_pdf.name
    shutil.copy2(final_pdf, local_pdf)
    outputs = {
        "summary_pdf": _artifact_record(final_pdf, relative_to=WORKSPACE_ROOT),
        "summary_pdf_copy": _artifact_record(local_pdf, relative_to=SWEEP_ROOT),
        "cells_csv": _artifact_record(SUMMARY_ROOT / "cells.csv", relative_to=SWEEP_ROOT),
        "cells_json": _artifact_record(SUMMARY_ROOT / "cells.json", relative_to=SWEEP_ROOT),
        "trend_png": _artifact_record(trend, relative_to=SWEEP_ROOT),
        "graph_contact_sheet_png": _artifact_record(graph_png, relative_to=SWEEP_ROOT),
        "graph_contact_sheet_pdf": _artifact_record(graph_pdf, relative_to=SWEEP_ROOT),
        "morse_set_contact_sheet_png": _artifact_record(set_png, relative_to=SWEEP_ROOT),
        "morse_set_contact_sheet_pdf": _artifact_record(set_pdf, relative_to=SWEEP_ROOT),
    }
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "sweep_plan_sha256": sweep["plan_sha256"],
        "paper_palette": PALETTE,
        "common_bounds": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
        "morse_set_render": {
            "box_scale": 1.0,
            "min_box_side_frac": MIN_BOX_SIDE_FRAC,
            "display_only": True,
            "individual_floor_basis": "adaptive_occupied_view",
            "contact_sheet_floor_basis": "common_archived_bounds_axis_span",
        },
        "individual_render_records": render_records,
        "outputs": outputs,
    }
    _write_json_atomic(SUMMARY_ROOT / "summary_manifest.json", manifest)
    print(json.dumps({"summary_pdf": str(final_pdf), "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
