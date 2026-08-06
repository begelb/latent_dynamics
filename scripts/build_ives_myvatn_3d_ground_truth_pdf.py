#!/usr/bin/env python3
"""Build a concise PDF supplement for the direct 3D Ives computation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = CODE_ROOT / "output" / "ives_myvatn_3d_ground_truth"
DEFAULT_FINE_RUN = OUTPUT_ROOT / "absorbing_v3_i18_m33_M39_L30000000_morse"
DEFAULT_REFINED_RUN = (
    OUTPUT_ROOT / "absorbing_refined_r1_i18_m30_M36_L10000000_morse"
)
DEFAULT_UNREFINED_COMPARISON = OUTPUT_ROOT / "comparison_levels27_30_33" / "comparison.json"
DEFAULT_REFINED_COMPARISON = (
    OUTPUT_ROOT / "comparison_refined_r1_levels27_30" / "comparison.json"
)
DEFAULT_STABILITY = OUTPUT_ROOT / "invariant_stability" / "stability.json"
DEFAULT_OUTPUT = CODE_ROOT / "output" / "pdf" / "ives_myvatn_3d_ground_truth_summary.pdf"

INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#596579")
BLUE = colors.HexColor("#315B8A")
LIGHT_BLUE = colors.HexColor("#E9F1FA")
LIGHT_GREEN = colors.HexColor("#E9F6EF")
LIGHT_AMBER = colors.HexColor("#FFF4DC")
GRID = colors.HexColor("#CFD6E0")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "GroundTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "GroundSubtitle",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=MUTED,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "GroundH1",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=INK,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "GroundH2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=BLUE,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "GroundBody",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.3,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "GroundSmall",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.2,
            textColor=MUTED,
        ),
        "mono": ParagraphStyle(
            "GroundMono",
            parent=sample["Code"],
            fontName="Courier",
            fontSize=6.7,
            leading=8.6,
            textColor=INK,
            leftIndent=8,
            rightIndent=8,
            spaceAfter=4,
        ),
        "figure": ParagraphStyle(
            "GroundFigure",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.5,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceBefore=5,
        ),
    }


def _footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(document.leftMargin, 0.48 * inch, letter[0] - document.rightMargin, 0.48 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(document.leftMargin, 0.31 * inch, "Direct Ives 3D numerical reference")
    canvas.drawRightString(
        letter[0] - document.rightMargin,
        0.31 * inch,
        f"Page {document.page}",
    )
    canvas.restoreState()


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _status_table(styles: dict[str, ParagraphStyle]) -> Table:
    rows = [
        [
            _paragraph("Stable high-resolution outer decomposition", styles["small"]),
            _paragraph("YES", styles["body"]),
        ],
        [
            _paragraph("Exact two-attractor Morse split certified", styles["small"]),
            _paragraph("NO - finite enclosure retains M1 -> M0", styles["body"]),
        ],
    ]
    table = Table(rows, colWidths=(3.55 * inch, 2.95 * inch), hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_GREEN),
                ("BACKGROUND", (0, 1), (-1, 1), LIGHT_AMBER),
                ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _parameter_table(manifest: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    params = manifest["system"]["parameters"]
    rows = [
        [_paragraph("Parameter", styles["small"]), _paragraph("Value", styles["small"])],
        ["r1", f"{params['r1']:.6g}"],
        ["r2", f"{params['r2']:.6g}"],
        ["c", f"{params['c']:.15g}"],
        ["d", f"{params['d']:.6g}"],
        ["p", f"{params['p']:.6g}"],
        ["q", f"{params['q']:.6g}"],
    ]
    table = Table(rows, colWidths=(1.15 * inch, 2.15 * inch), hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _resolution_rows(comparison: dict[str, Any], method: str) -> list[list[str]]:
    return [
        [
            method,
            str(run["source_level"]),
            f"{max(run['cell_width']):.5f}",
            f"{run['morse_box_count']:,}",
            f"{run['compute_seconds']:.1f}",
            f"{run['morse_nodes']}/{run['morse_edges']}/{run['sink_count']}",
        ]
        for run in comparison["runs"]
    ]


def _resolution_table(
    unrefined: dict[str, Any],
    refined: dict[str, Any],
) -> Table:
    rows = [
        ["Enclosure", "Level", "Max width", "Morse boxes", "CMGDB sec", "N/E/S"],
        *_resolution_rows(unrefined, "analytic"),
        *_resolution_rows(refined, "8-subbox"),
    ]
    table = Table(
        rows,
        colWidths=(1.0 * inch, 0.55 * inch, 0.82 * inch, 1.05 * inch, 0.78 * inch, 0.67 * inch),
        hAlign="LEFT",
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), (colors.white, colors.HexColor("#F7F9FC"))),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _fit_image(path: Path, width: float, height: float) -> Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = Image(str(path))
    image._restrictSize(width, height)
    return image


def build(
    fine_run: Path,
    refined_run: Path,
    unrefined_comparison_path: Path,
    refined_comparison_path: Path,
    stability_path: Path,
    output: Path,
) -> dict[str, Any]:
    fine_manifest = _load_json(fine_run / "manifest.json")
    _load_json(refined_run / "manifest.json")
    unrefined = _load_json(unrefined_comparison_path)
    refined = _load_json(refined_comparison_path)
    stability = _load_json(stability_path)
    fine_audit = _load_json(fine_run / "interval_enclosure_audit.json")

    primary_image = fine_run / "render_3d" / "morse_sets_cubical_3d.png"
    alternate_image = fine_run / "render_3d" / "morse_sets_cubical_3d_alternate.png"
    graph_image = fine_run / "morse_graph.png"
    render_manifest = _load_json(fine_run / "render_3d" / "manifest.json")

    output.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    document = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.62 * inch,
        title="Ives Lake Myvatn - Direct 3D Ground-Truth Reference",
        author="Codex",
        subject="Direct three-dimensional CMGDB reference and validation summary",
    )

    bounds = fine_manifest["system"]["bounds"]
    fine_record = unrefined["runs"][-1]
    refined_record = refined["runs"][-1]
    stable = stability["finite_difference_stability"]
    story: list[Any] = [
        _paragraph("Ives Lake Myvatn - Direct 3D Ground-Truth Reference", styles["title"]),
        _paragraph(
            "Original log10 midge-algae-detritus map, evaluated directly with CMGDB. "
            "No encoder, decoder, latent map, or learned weights enter this computation.",
            styles["subtitle"],
        ),
        _status_table(styles),
        Spacer(1, 10),
        _paragraph("Result", styles["h1"]),
        _paragraph(
            "The role-matched outer Morse graph is stable at levels 27, 30, and 33: "
            "two Morse nodes and one directed edge, M1 -> M0. All 12 phases of the "
            "known period-12 orbit lie uniquely in M1; the fixed point lies uniquely "
            "in sink M0. The same graph and exact fine-to-parent nesting persist under "
            "an independent 2x2x2 internal enclosure refinement at levels 27 and 30.",
            styles["body"],
        ),
        _paragraph(
            "Important interpretation: the return-map and fixed-point Jacobian audits "
            "show that both known objects are locally attracting. Therefore the finite "
            "outer edge from the cycle-containing component to M0 is retained as a "
            "documented enclosure limitation, not interpreted as physical instability "
            "of the period-12 orbit.",
            styles["body"],
        ),
        _paragraph("Model and absorbing domain", styles["h2"]),
    ]
    domain_text = (
        f"Trapping box lower = {bounds['lower']}; upper = {bounds['upper']}. "
        "A successive interval enclosure of the complete archived sampling box enters "
        f"this trapping box within {fine_manifest['system']['archived_sampling_absorption']['steps']} steps."
    )
    model_layout = Table(
        [
            [_parameter_table(fine_manifest, styles), _paragraph(domain_text, styles["body"])],
        ],
        colWidths=(3.35 * inch, 3.15 * inch),
        hAlign="LEFT",
    )
    model_layout.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 12),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.extend([model_layout, PageBreak()])

    story.extend(
        [
            _paragraph("Resolution and tightening evidence", styles["h1"]),
            _resolution_table(unrefined, refined),
            Spacer(1, 8),
            _paragraph(
                "N/E/S denotes Morse nodes, directed edges, and graph sinks. The "
                "level-33 analytic run evaluated 120,002,052 CMGDB rectangles; the "
                "refined level-30 run evaluated 25,180,362 CMGDB rectangles and "
                "201,442,896 internal child enclosures.",
                styles["small"],
            ),
            _paragraph("Cross-resolution checks", styles["h2"]),
            _paragraph(
                "For both 27 -> 30 and 30 -> 33, cell widths halve in every axis, "
                "the colored transitive reductions are isomorphic, and 100% of fine "
                "role cells map into same-role coarse parents. No saved Morse set "
                "touches the trapping-box boundary, and no immediate post-minimum "
                "subdivision-limit stop is guaranteed.",
                styles["body"],
            ),
            _paragraph("Morse graph", styles["h2"]),
            KeepTogether(
                [
                    _fit_image(graph_image, 2.6 * inch, 1.55 * inch),
                    _paragraph(
                        "M1 contains all period-12 phases. M0 contains the fixed point "
                        "and is the sole finite-grid sink.",
                        styles["figure"],
                    ),
                ]
            ),
            _paragraph("What 'fine enough' means here", styles["h2"]),
            _paragraph(
                "The computation is fine enough to serve as a reproducible stable "
                "outer-enclosure reference: the graph, roles, nesting, and boundary "
                "status agree across three successive analytic levels and two refined "
                "levels. It is not a formal arbitrary-precision proof and does not yet "
                "certify a separate sink for the period-12 attractor.",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _paragraph("Direct 3D Morse sets", styles["h1"]),
            _fit_image(primary_image, 6.75 * inch, 7.75 * inch),
            _paragraph(
                f"Exact source: {fine_record['morse_box_count']:,} uniform level-33 "
                f"cells. Display: {render_manifest['cover']['display_cell_count']:,} "
                "collision-free level-24 parent cells. White circles mark the 12 cycle "
                "phases; the star marks the fixed point. Coarsening is render-only.",
                styles["figure"],
            ),
            PageBreak(),
        ]
    )

    audit_total = fine_audit["total_points"]
    story.extend(
        [
            _paragraph("Alternate view and validation", styles["h1"]),
            _fit_image(alternate_image, 6.55 * inch, 4.9 * inch),
            _paragraph(
                "Alternate camera view of the same collision-free display cover.",
                styles["figure"],
            ),
            _paragraph("Validation summary", styles["h2"]),
            _paragraph(
                f"The interval challenge passed {audit_total:,} evaluated points over "
                "six dyadic levels, including all eight corners per sampled box and all "
                "13 invariant representatives. Every saved level-33 cell passed finite, "
                "integral-label, dyadic-alignment, uniqueness, domain-containment, and "
                "reference-membership checks.",
                styles["body"],
            ),
            _paragraph(
                f"The refined period-12 orbit closes to {stability['refinement']['period12_closure_residual']:.2e} "
                f"in max norm. Across four finite-difference scales, its largest return-map "
                f"spectral radius is {stable['period12_maximum_spectral_radius']:.6f}; the "
                f"fixed point maximum is {stable['fixed_point_maximum_spectral_radius']:.6f}. "
                "Both are below one.",
                styles["body"],
            ),
            _paragraph("Reproduce the selected runs", styles["h2"]),
            _paragraph(
                "python scripts/compute_ives_myvatn_3d_ground_truth.py --domain trapping "
                "--subdiv 18 33 39 --subdiv-limit 30000000",
                styles["mono"],
            ),
            _paragraph(
                "python scripts/compute_ives_myvatn_3d_ground_truth.py --domain trapping "
                "--interval-refinement 1 --subdiv 18 30 36 --subdiv-limit 10000000",
                styles["mono"],
            ),
            _paragraph(
                "Numerical scope: analytic monotone float64 enclosures. The selected "
                "level-33 run used one outward nextafter step; the independent "
                "eight-subbox checks used an eight-ULP outward guard. Both are stronger "
                "than corner sampling, but neither is an arbitrary-precision "
                "directed-rounding proof.",
                styles["small"],
            ),
        ]
    )

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    payload = {
        "schema_version": 1,
        "output": {
            "path": str(output.resolve()),
            "size_bytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "sources": {
            "fine_manifest": str((fine_run / "manifest.json").resolve()),
            "refined_manifest": str((refined_run / "manifest.json").resolve()),
            "unrefined_comparison": str(unrefined_comparison_path.resolve()),
            "refined_comparison": str(refined_comparison_path.resolve()),
            "stability": str(stability_path.resolve()),
            "primary_image": str(primary_image.resolve()),
            "alternate_image": str(alternate_image.resolve()),
        },
        "headline_metrics": {
            "fine_level": fine_record["source_level"],
            "fine_morse_boxes": fine_record["morse_box_count"],
            "refined_level": refined_record["source_level"],
            "refined_morse_boxes": refined_record["morse_box_count"],
        },
    }
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-run", type=Path, default=DEFAULT_FINE_RUN)
    parser.add_argument("--refined-run", type=Path, default=DEFAULT_REFINED_RUN)
    parser.add_argument(
        "--unrefined-comparison", type=Path, default=DEFAULT_UNREFINED_COMPARISON
    )
    parser.add_argument("--refined-comparison", type=Path, default=DEFAULT_REFINED_COMPARISON)
    parser.add_argument("--stability", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = build(
        args.fine_run.resolve(),
        args.refined_run.resolve(),
        args.unrefined_comparison.resolve(),
        args.refined_comparison.resolve(),
        args.stability.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(payload["output"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
