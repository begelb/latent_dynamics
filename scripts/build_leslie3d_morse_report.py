#!/usr/bin/env python3
"""Build the self-contained Leslie3D max-30 Morse audit report bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"
REPORT_ROOT = WORKSPACE_ROOT / "output" / "pdf" / "leslie3d_morse_report"
ASSET_ROOT = REPORT_ROOT / "assets"
PDF_PATH = REPORT_ROOT / "leslie3d_morse_report.pdf"
MARKDOWN_PATH = REPORT_ROOT / "report.md"
MANIFEST_PATH = REPORT_ROOT / "report_manifest.json"

MAX30_ROOT = (
    CODE_ROOT
    / "output"
    / "notebooks"
    / "leslie3d_invariant_aware_v2_smooth_max30"
    / "seed_20260809"
)
TRAINING_ROOT = (
    CODE_ROOT / "output" / "leslie3d_invariant_aware_v2_smooth" / "seed_20260809"
)
GROUND_TRUTH_ROOT = (
    CODE_ROOT
    / "output"
    / "original_leslie"
    / "ground_truth"
    / "absorbing_B_uniform_level33_recurrent_closure"
)

DATA_MANIFEST = CODE_ROOT / "data" / "leslie3d_invariant_aware_v2" / "dataset_manifest.json"
TRAINING_SUMMARY = TRAINING_ROOT / "smooth_topology_summary.json"
MAX30_SUMMARY = MAX30_ROOT / "analysis" / "invariant_aware_summary.json"
OVERLAY_PROVENANCE = ASSET_ROOT / "direct_ground_truth_overlay_provenance.json"

SOURCE_FIGURES = {
    "max30_morse_sets.png": MAX30_ROOT / "MG" / "morse_sets.png",
    "max30_morse_graph.png": MAX30_ROOT / "MG" / "morse_graph.png",
    "direct_ground_truth_morse_graph.png": (
        GROUND_TRUTH_ROOT / "paper_figure_pruned" / "morse_graph.png"
    ),
    "direct_ground_truth_3d_display_cover.png": (
        GROUND_TRUTH_ROOT
        / "cubical_3d_level24_display_cover"
        / "morse_sets_cubical_3d_labeled.png"
    ),
}

EXPECTED_INPUT_HASHES = {
    DATA_MANIFEST: "658926337cc98e5e2d08ff9f442496c929e7400369deb7bc0382a6b73e87f5a1",
    TRAINING_SUMMARY: "16f20b6fe689f34fb3cf4e85d826aa3e88b46cdc31d506cfeca0f078dde7252e",
    MAX30_SUMMARY: "3937584ce415b402b98d401241be208130619ea6a6811e44bdd01a3adb7412e1",
}

NAVY = colors.HexColor("#16324F")
TEAL = colors.HexColor("#007C83")
ORANGE = colors.HexColor("#E77D22")
RED = colors.HexColor("#A33A3A")
LIGHT_BLUE = colors.HexColor("#EAF1F7")
LIGHT_TEAL = colors.HexColor("#E8F4F3")
LIGHT_ORANGE = colors.HexColor("#FFF3E7")
LIGHT_RED = colors.HexColor("#F9EAEA")
MID_GREY = colors.HexColor("#6B7280")
LIGHT_GREY = colors.HexColor("#F3F5F7")
GRID_GREY = colors.HexColor("#D5DBE1")
INK = colors.HexColor("#18212B")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_INPUT_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"input hash mismatch for {path}: {actual} != {expected}")
        observed[str(path)] = actual
    if not OVERLAY_PROVENANCE.is_file():
        raise FileNotFoundError(
            f"render the report overlay before building the report: {OVERLAY_PROVENANCE}"
        )
    for path in SOURCE_FIGURES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[str(path)] = _sha256(path)
    observed[str(OVERLAY_PROVENANCE)] = _sha256(OVERLAY_PROVENANCE)
    return observed


def _copy_assets() -> dict[str, str]:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name, source in SOURCE_FIGURES.items():
        destination = ASSET_ROOT / name
        shutil.copy2(source, destination)
        copied[name] = _sha256(destination)
    overlay = ASSET_ROOT / "direct_ground_truth_on_max30_morse_sets.png"
    if not overlay.is_file():
        raise FileNotFoundError(overlay)
    copied[overlay.name] = _sha256(overlay)
    return copied


def _paragraph_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=25,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=MID_GREY,
            spaceAfter=4 * mm,
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=NAVY,
            spaceBefore=1 * mm,
            spaceAfter=3.5 * mm,
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=TEAL,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.2,
            textColor=INK,
            spaceAfter=2.2 * mm,
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=9.8,
            textColor=INK,
            spaceAfter=1.5 * mm,
        ),
        "caption": ParagraphStyle(
            "ReportCaption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=9.5,
            textColor=MID_GREY,
            alignment=TA_LEFT,
            spaceBefore=1.2 * mm,
            spaceAfter=2 * mm,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=11.7,
            leftIndent=4 * mm,
            firstLineIndent=-2.5 * mm,
            bulletIndent=0,
            textColor=INK,
            spaceAfter=1.4 * mm,
        ),
        "code": ParagraphStyle(
            "ReportCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.4,
            leading=9.4,
            textColor=INK,
            backColor=LIGHT_GREY,
            borderPadding=5,
            spaceAfter=2 * mm,
        ),
        "center_small": ParagraphStyle(
            "ReportCenterSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.2,
            textColor=INK,
            alignment=TA_CENTER,
        ),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _bullet(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, styles["bullet"], bulletText="-")


def _table(
    rows: list[list[Any]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    *,
    font_size: float = 7.6,
    header_color: colors.Color = NAVY,
    row_backgrounds: tuple[colors.Color, colors.Color] = (colors.white, LIGHT_GREY),
    alignments: dict[int, str] | None = None,
) -> Table:
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["small"],
        fontSize=font_size,
        leading=font_size + 2,
        spaceAfter=0,
    )
    header_style = ParagraphStyle(
        "TableHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    materialized: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        materialized.append(
            [
                value
                if hasattr(value, "wrap")
                else Paragraph(str(value), header_style if row_index == 0 else cell_style)
                for value in row
            ]
        )
    table = Table(materialized, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, header_color),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, GRID_GREY),
    ]
    for row_index in range(1, len(rows)):
        commands.append(
            ("BACKGROUND", (0, row_index), (-1, row_index), row_backgrounds[(row_index - 1) % 2])
        )
    for column, alignment in (alignments or {}).items():
        commands.append(("ALIGN", (column, 1), (column, -1), alignment))
    table.setStyle(TableStyle(commands))
    return table


def _image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _callout(
    text: str,
    styles: dict[str, ParagraphStyle],
    *,
    background: colors.Color,
    border: colors.Color,
) -> Table:
    paragraph = Paragraph(text, styles["body"])
    box = Table([[paragraph]], colWidths=[178 * mm], hAlign="LEFT")
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return box


def _header_footer(canvas: Canvas, page_number: int) -> None:
    canvas.saveState()
    page_width, page_height = A4
    canvas.setStrokeColor(GRID_GREY)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, page_height - 14 * mm, page_width - 18 * mm, page_height - 14 * mm)
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(18 * mm, page_height - 10.5 * mm, "Leslie3D invariant-aware Morse audit")
    canvas.drawRightString(
        page_width - 18 * mm,
        page_height - 10.5 * mm,
        "Numerical experiment - not a Conley certificate",
    )
    canvas.line(18 * mm, 13 * mm, page_width - 18 * mm, 13 * mm)
    canvas.drawString(18 * mm, 8.5 * mm, "Generated 2026-08-04")
    canvas.drawRightString(page_width - 18 * mm, 8.5 * mm, f"Page {page_number}")
    canvas.restoreState()


class ReportCanvas(Canvas):
    """Draw running furniture after the page content so it stays visible."""

    def showPage(self) -> None:  # noqa: N802 - ReportLab API name
        _header_footer(self, self.getPageNumber())
        super().showPage()


def _build_story(styles: dict[str, ParagraphStyle]) -> list[Any]:
    story: list[Any] = []
    max30_sets = ASSET_ROOT / "max30_morse_sets.png"
    max30_graph = ASSET_ROOT / "max30_morse_graph.png"
    direct_graph = ASSET_ROOT / "direct_ground_truth_morse_graph.png"
    direct_3d = ASSET_ROOT / "direct_ground_truth_3d_display_cover.png"
    overlay = ASSET_ROOT / "direct_ground_truth_on_max30_morse_sets.png"

    story.extend(
        [
            Spacer(1, 4 * mm),
            _p("Leslie3D invariant-aware latent dynamics", styles["title"]),
            _p(
                "Data construction, training, max-30 Morse sets and graph, and a direct-system ground-truth overlay",
                styles["subtitle"],
            ),
            _callout(
                "<b>Outcome.</b> The learned two-dimensional map recovers the six named fixed and periodic objects locally, with the intended periods and stability roles. The global Morse/Conley target is not recovered: the stable fine graph has eight rather than six nodes, contains an extra attractor and saddle, gives incorrect indices for P1 and S2, and omits the required S2 -> P1 relation.",
                styles,
                background=LIGHT_RED,
                border=RED,
            ),
            Spacer(1, 3 * mm),
            _p("Physical map", styles["h2"]),
            _p(
                "f(x1,x2,x3) = ((28.9 x1 + 29.8 x2 + 22 x3) exp[-0.1(x1+x2+x3)], 0.7 x1, 0.7 x2), on the absorbing box [0,110] x [0,77] x [0,54].",
                styles["code"],
            ),
        ]
    )
    image_row = Table(
        [
            [
                _image(max30_sets, 108 * mm, 92 * mm),
                _image(max30_graph, 58 * mm, 92 * mm),
            ],
            [
                _p("Max-30 learned latent Morse sets", styles["center_small"]),
                _p("Max-30 learned Morse graph", styles["center_small"]),
            ],
        ],
        colWidths=[114 * mm, 62 * mm],
        hAlign="LEFT",
    )
    image_row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    story.extend(
        [
            image_row,
            Spacer(1, 2 * mm),
            _bullet(
                "Max-28 and max-30 give the same eight-node graph. The three microscopic nodes at max-29 comprise only five origin-adjacent boxes and disappear at max-30.",
                styles,
            ),
            _bullet(
                "All 16 exact direct-system phases lie in unique latent Morse sets, but point membership does not determine the surrounding Conley index or global connection order.",
                styles,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Spacer(1, 4 * mm),
            _p("1. Ground-truth skeleton and data construction", styles["h1"]),
            _p(
                "Every data row is an exact analytic pair (x, f(x)); no interpolated pseudo-transition is used. The design deliberately combines the recurrent skeleton, local neighborhoods, transition tubes, and broad background coverage. Training and held-out banks are disjoint at the recorded 15-digit CSV representation.",
                styles["body"],
            ),
            _p("Known direct-system invariant objects", styles["h2"]),
            _table(
                [
                    ["Object", "Role", "Period", "Phases", "Direct node"],
                    ["P0", "stable orbit", "4", "4", "0"],
                    ["P1", "stable orbit", "4", "4", "1"],
                    ["S2", "saddle orbit", "2", "2", "2"],
                    ["S4", "saddle orbit", "4", "4", "3"],
                    ["p_star", "positive saddle fixed point", "1", "1", "4"],
                    ["origin", "unstable boundary fixed point", "1", "1", "5"],
                ],
                [22 * mm, 63 * mm, 20 * mm, 20 * mm, 25 * mm],
                styles,
                alignments={2: "CENTER", 3: "CENTER", 4: "CENTER"},
            ),
            Spacer(1, 3 * mm),
            _p("Version-2 pair inventory", styles["h2"]),
            _table(
                [
                    ["Component", "Train", "Held out", "Construction purpose"],
                    ["Exact recurrent phases", "8,192", "-", "16 phases repeated 512 times"],
                    ["Multiscale recurrent neighborhoods", "4,096", "1,013", "local behavior over seven radius decades"],
                    ["Balanced direct-Morse neighborhoods", "5,120", "1,280", "equal coverage of five non-origin direct nodes"],
                    ["Saddle tangent tubes", "6,144", "3,072", "96/48 true 64-step trajectories"],
                    ["Origin positive-cone fan", "1,024", "512", "16/8 true 64-step trajectories"],
                    ["Sobol background trajectories", "12,288", "6,144", "1,024/512 starts, 12 steps"],
                    ["Absorbing-box corners", "8", "-", "explicit physical support"],
                    ["Audited origin-p_star-S2 tubes", "8,640", "8,640", "27 independent 320-step trajectories per split"],
                    ["Total", "45,512", "20,661", "exact analytic successor pairs"],
                ],
                [61 * mm, 21 * mm, 22 * mm, 74 * mm],
                styles,
                font_size=7.2,
                alignments={1: "RIGHT", 2: "RIGHT"},
            ),
            Spacer(1, 3 * mm),
            _callout(
                "<b>Transition witness design.</b> Three strictly positive Sobol-discovered starts per split were expanded to a center and eight coordinatewise +/-0.1% corners. All 54 resulting trajectories begin in the saved origin cell, enter and leave p_star, then enter and leave S2 within 320 analytic steps. These are finite itinerary witnesses, not certified heteroclinic full orbits.",
                styles,
                background=LIGHT_ORANGE,
                border=ORANGE,
            ),
            Spacer(1, 3 * mm),
            _p("Direct-system reference computation", styles["h2"]),
        ]
    )
    direct_row = Table(
        [[_image(direct_3d, 78 * mm, 43 * mm), _image(direct_graph, 59 * mm, 43 * mm)]],
        colWidths=[99 * mm, 75 * mm],
    )
    direct_row.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.extend(
        [
            direct_row,
            _p(
                "Left: render-only level-24 parent-cell display cover of the six saved direct-map Morse sets. Right: the verified saved-set direct graph. The exact source contains 1,955,948 level-33 cells; the display cover has 10,498 cells.",
                styles["caption"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Spacer(1, 4 * mm),
            _p("2. Architecture and training", styles["h1"]),
            _p(
                "The primary version-2 experiment held the inherited encoder and decoder fixed so that changes could be attributed to the latent transition map rather than a moving coordinate system. The latent map was replaced by a smooth GELU network to make exact-autograd return-map derivatives available.",
                styles["body"],
            ),
            _p("Networks", styles["h2"]),
            _table(
                [
                    ["Component", "Architecture", "Hidden activation", "Output", "Primary training"],
                    ["Encoder E", "3-64-64-2", "ReLU", "tanh", "frozen"],
                    ["Latent map g", "2-64-64-2", "GELU", "tanh", "optimized"],
                    ["Decoder D", "2-64-64-3", "ReLU", "sigmoid", "frozen"],
                ],
                [32 * mm, 39 * mm, 39 * mm, 26 * mm, 34 * mm],
                styles,
                alignments={1: "CENTER", 2: "CENTER", 3: "CENTER", 4: "CENTER"},
            ),
            Spacer(1, 3 * mm),
            _p("Optimization recipe", styles["h2"]),
            _table(
                [
                    ["Item", "Value"],
                    ["Warm start", "accepted invariant-aware v1 GELU latent map"],
                    ["Optimizer/data", "Adam continuation, full batch of 45,512 exact pairs"],
                    ["Effective step", "2 x 10^-7"],
                    ["Replay weights", "reconstruction/prediction/semiconjugacy/cycle = 100/20/100/20"],
                    ["Component multipliers", "saddle tubes 4; origin fan 3; audited origin-p_star-S2 tubes 8"],
                    ["Topology terms", "trust 10; characteristic-polynomial 5; topology role margin 20; all 16 phase anchors"],
                    ["Duration/selection", "3,170 epochs run; epoch 669 selected; 133.94 seconds"],
                    ["Promoted checkpoint", "SHA-256 9fbee2cde690d58...5612da3f3d"],
                ],
                [48 * mm, 128 * mm],
                styles,
                font_size=7.7,
            ),
            Spacer(1, 3 * mm),
            _p("Held-out and local-dynamics diagnostics", styles["h2"]),
            _table(
                [
                    ["Diagnostic", "Selected value", "Interpretation"],
                    ["Reconstruction MSE", "2.41393 x 10^-4", "held-out mean"],
                    ["Prediction MSE", "2.74302 x 10^-4", "held-out mean"],
                    ["Semiconjugacy MSE", "5.64443 x 10^-5", "held-out mean"],
                    ["Latent cycle MSE", "2.14994 x 10^-5", "held-out mean"],
                    ["Maximum normalized anchor error", "5.49785 x 10^-5", "all 16 phases"],
                    ["Maximum characteristic relative error", "0.00246848", "period-1/2/4 monodromies"],
                    ["Global trust RMSE", "0.00943702", "distillation control"],
                    ["Local role gates", "all pass", "period, unstable dimension, orientation"],
                ],
                [67 * mm, 43 * mm, 66 * mm],
                styles,
                font_size=7.5,
                alignments={1: "RIGHT"},
            ),
            Spacer(1, 3 * mm),
            _callout(
                "<b>Encoder follow-up.</b> A guarded update of the final encoder and first decoder layers preserved physical accuracy and enlarged the tightest named-object separations by about 1-2.3%. Two subsequent 4,000-epoch map-repair stages still failed the fixed characteristic and long-horizon gates, so no jointly refined checkpoint was promoted. The max-30 audit in this report therefore uses the accepted primary checkpoint above.",
                styles,
                background=LIGHT_BLUE,
                border=NAVY,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Spacer(1, 4 * mm),
            _p("3. Morse computation and subdivision stability", styles["h1"]),
            _p(
                "All four box computations use the same accepted checkpoint, chart, and latent rectangle [-0.3409648, 0.2583292] x [-0.2952787, 0.4102896]. The box map evaluates neural images at corners with padding. It is a reproducible numerical outer-map heuristic, not an outward-rounded interval enclosure of the network over each box.",
                styles["body"],
            ),
            _p("Resolution ladder", styles["h2"]),
            _table(
                [
                    ["Subdivision (i,m,M)", "Nodes", "Morse boxes", "Outcome"],
                    ["(20,24,26)", "5", "108,982", "S2 and p_star merged"],
                    ["(24,27,28)", "8", "629,742", "all six roles separated; two extras remain"],
                    ["(25,28,29)", "11", "1,133,093", "three tiny origin-side nodes totaling five boxes"],
                    ["(25,28,30)", "8", "1,133,088", "returns exactly to stable max-28 graph"],
                ],
                [42 * mm, 22 * mm, 38 * mm, 74 * mm],
                styles,
                alignments={1: "CENTER", 2: "RIGHT"},
            ),
            Spacer(1, 3 * mm),
            _callout(
                "<b>Subdivision conclusion.</b> The max-29 three-node diamond is a grid micro-splitting: its five boxes cease to be recurrent at max-30. The substantive eight-node graph is stable from max-28 to max-30, including its two unwanted components and incorrect global order.",
                styles,
                background=LIGHT_TEAL,
                border=TEAL,
            ),
            Spacer(1, 3 * mm),
            _p("Stable max-30 graph", styles["h2"]),
            _p(
                "Edges: 2 -> 1, 3 -> 2, 4 -> 3, 5 -> 4, 5 -> 0, 6 -> 5, 7 -> 6. Minimal nodes: 0 and 1.",
                styles["code"],
            ),
            _p("Role and index audit", styles["h2"]),
            _table(
                [
                    ["Direct role", "Latent node", "Observed index", "Expected latent index", "Match"],
                    ["P0", "0", "(x^4-1,0,0)", "(x^4-1,0,0)", "yes"],
                    ["P1", "4", "(0,0,0)", "(x^4-1,0,0)", "no"],
                    ["S2", "2", "(0,0,0)", "(0,x^2+1,0)", "no"],
                    ["S4", "5", "(0,x^4-1,0)", "(0,x^4-1,0)", "yes"],
                    ["p_star", "3", "(0,x+1,0)", "(0,x+1,0)", "yes"],
                    ["origin", "7", "(0,0,0)", "(0,0,0)", "yes"],
                    ["extra sink", "1", "(x^2-1,0,0)", "none", "extra"],
                    ["extra saddle", "6", "(0,x-1,0)", "none", "extra"],
                ],
                [31 * mm, 24 * mm, 45 * mm, 51 * mm, 23 * mm],
                styles,
                font_size=7.15,
                alignments={1: "CENTER", 4: "CENTER"},
            ),
            Spacer(1, 3 * mm),
            _p("Ground truth versus learned order", styles["h2"]),
            _table(
                [
                    ["Direct-system reduced order", "Learned max-30 order"],
                    ["origin -> S4 and p_star", "origin -> extra saddle -> S4"],
                    ["p_star -> S2", "S4 -> P1 -> p_star -> S2"],
                    ["S2 -> P1", "S2 -> extra sink (required S2 -> P1 is absent)"],
                    ["S4 -> P0 and P1", "S4 -> P0 and P1"],
                ],
                [87 * mm, 89 * mm],
                styles,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Spacer(1, 4 * mm),
            _p("4. Direct ground truth over the learned Morse sets", styles["h1"]),
            _p(
                "The new plot below combines three layers: all saved learned max-30 latent Morse boxes; encoded centers of the 10,498-cell render-only display cover derived from the original 1,955,948 direct-map level-33 Morse boxes; and the 16 exact direct-system fixed/periodic phases as outlined symbols.",
                styles["body"],
            ),
            _image(overlay, 178 * mm, 120 * mm),
            _p(
                "Direct-system ground-truth overlay. The colored clouds show E(center) for the direct-map display-cover cells. The outlined symbols are the exact fixed and periodic phases. Center encoding is a sampled visual comparison: it is not a certified enclosure of E(M), and the display cover itself is not a recomputed level-24 decomposition.",
                styles["caption"],
            ),
            _p("Exact-phase membership", styles["h2"]),
            _table(
                [
                    ["Object", "Exact phases", "Assigned max-30 node", "All phases in one unique set"],
                    ["P0", "4", "0", "yes"],
                    ["P1", "4", "4", "yes"],
                    ["S2", "2", "2", "yes"],
                    ["S4", "4", "5", "yes"],
                    ["p_star", "1", "3", "yes"],
                    ["origin", "1", "7", "yes"],
                ],
                [38 * mm, 35 * mm, 52 * mm, 51 * mm],
                styles,
                alignments={1: "CENTER", 2: "CENTER", 3: "CENTER"},
            ),
            Spacer(1, 3 * mm),
            _callout(
                "<b>What the overlay establishes.</b> The encoder places every known direct recurrent phase in a distinct, uniquely assigned latent Morse set. <b>What it does not establish.</b> It does not remove the two extra latent recurrent components, fix the P1/S2 indices, restore S2 -> P1, or transport a valid three-dimensional index pair through a two-dimensional encoder.",
                styles,
                background=LIGHT_ORANGE,
                border=ORANGE,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Spacer(1, 4 * mm),
            _p("5. Interpretation, ideal-data statement, and limitations", styles["h1"]),
            _p("Numerical conclusion", styles["h2"]),
            _bullet(
                "The data and topology-aware loss recover the intended recurrent skeleton locally: the six named cycles have the intended periods, unstable dimensions, and orientations.",
                styles,
            ),
            _bullet(
                "Local correctness is not global correctness. An independent dense period-1/2/4 census finds 27 cycles: six intended, 16 extra inside the CMGDB rectangle, and five extra outside it.",
                styles,
            ),
            _bullet(
                "Higher subdivision resolves the max-29 micro-splitting but confirms the persistent eight-node topology mismatch.",
                styles,
            ),
            _p("Conditional ideal-data argument", styles["h2"]),
            _p(
                "Let K be compact and suppose a continuous encoder E, decoder D, and latent map g attain zero population reconstruction and semiconjugacy loss on a full-support measure: D(E(x)) = x and g(E(x)) = E(f(x)) for every x in K. Then E is injective and, because K is compact and the latent space Hausdorff, E is a homeomorphism from K to E(K). The restricted dynamics are conjugate on K.",
                styles["body"],
            ),
            _p(
                "Conley-index recovery still requires more: E must transport a valid index pair, the latent isolating neighborhood must contain exactly the encoded invariant set and no extra recurrence, and the quotient index maps must be conjugate (or a certified shift equivalence must be constructed). Semiconjugacy alone can collapse invariant sets and does not imply these conditions.",
                styles["body"],
            ),
            _p("Why this computation is not a certificate", styles["h2"]),
            _bullet(
                "Finite sampled data do not bound the residual supremum over an index pair or exclude off-support recurrent dynamics in the neural extension.",
                styles,
            ),
            _bullet(
                "A continuous 3-to-2 encoder cannot embed an open three-dimensional neighborhood, so ambient index-pair transport is not available from reconstruction on the invariant points alone.",
                styles,
            ),
            _bullet(
                "The neural box map uses corner evaluations with padding rather than an outward-rounded interval enclosure of the whole image of each box.",
                styles,
            ),
            _bullet(
                "The current objective does not construct quotient maps or verify the chain-level identities needed for shift equivalence.",
                styles,
            ),
            Spacer(1, 3 * mm),
            _callout(
                "<b>Final assessment.</b> The invariant-aware construction is successful as a local recurrent-skeleton learner and as a diagnostic demonstration of why excellent one-step, periodic-orbit, and sampled-transition fit do not by themselves recover the global Morse graph or Conley index.",
                styles,
                background=LIGHT_TEAL,
                border=TEAL,
            ),
            Spacer(1, 4 * mm),
            _p("Primary provenance", styles["h2"]),
            _table(
                [
                    ["Artifact", "Pinned identifier"],
                    ["Accepted model", "autoencoder.pt SHA-256 9fbee2cde690...da3f3d"],
                    ["V2 dataset manifest", "SHA-256 658926337cc9...e87f5a1"],
                    ["Max-30 raw Morse boxes", "SHA-256 faf4de21228c...b2fd63"],
                    ["Direct raw Morse boxes", "SHA-256 91dce56924eb...b6fcd"],
                    ["Report overlay provenance", "assets/direct_ground_truth_overlay_provenance.json"],
                    ["Machine-readable report manifest", "report_manifest.json"],
                ],
                [62 * mm, 114 * mm],
                styles,
                font_size=7.3,
            ),
        ]
    )
    return story


def _markdown() -> str:
    return """# Leslie3D invariant-aware latent dynamics: max-30 Morse audit

Generated 2026-08-04. This is a numerical experiment, not a Conley certificate.

## Outcome

The learned two-dimensional map recovers the six named fixed and periodic
objects locally, with their intended periods and stability roles. It does not
recover the global Morse/Conley target: the stable fine graph has eight rather
than six nodes, contains an extra attractor and saddle, gives incorrect indices
for `P1` and `S2`, and omits `S2 -> P1`.

![Max-30 learned Morse sets](assets/max30_morse_sets.png)

![Max-30 learned Morse graph](assets/max30_morse_graph.png)

## Physical system

```text
f(x1,x2,x3) = ((28.9*x1 + 29.8*x2 + 22*x3)*exp(-0.1*(x1+x2+x3)),
               0.7*x1,
               0.7*x2)
```

The absorbing box is `[0,110] x [0,77] x [0,54]`. The direct recurrent
inventory has 16 phases: stable period-four `P0` and `P1`, saddle period-two
`S2`, saddle period-four `S4`, the positive saddle fixed point `p_star`, and
the unstable boundary origin.

## Data construction

Every row is an exact analytic pair `(x, f(x))`; no interpolation is used.

| Component | Train | Held out | Purpose |
|---|---:|---:|---|
| Exact recurrent phases | 8,192 | - | 16 phases repeated 512 times |
| Multiscale recurrent neighborhoods | 4,096 | 1,013 | local recurrent geometry |
| Balanced direct-Morse neighborhoods | 5,120 | 1,280 | balance five non-origin direct nodes |
| Saddle tangent tubes | 6,144 | 3,072 | true 64-step trajectories |
| Origin positive-cone fan | 1,024 | 512 | boundary repeller coverage |
| Sobol background trajectories | 12,288 | 6,144 | broad off-skeleton coverage |
| Absorbing-box corners | 8 | - | explicit physical support |
| Audited origin-p_star-S2 tubes | 8,640 | 8,640 | 27 true 320-step trajectories per split |
| **Total** | **45,512** | **20,661** | |

The audited witnesses begin in the saved origin cell, enter and leave
`p_star`, then enter and leave `S2`. They are finite itinerary witnesses, not
certified heteroclinic full orbits.

## Architecture and training

| Component | Architecture | Activation/output | Primary status |
|---|---|---|---|
| Encoder `E` | 3-64-64-2 | ReLU/tanh | frozen |
| Latent map `g` | 2-64-64-2 | GELU/tanh | optimized |
| Decoder `D` | 2-64-64-3 | ReLU/sigmoid | frozen |

The accepted primary-v2 continuation used all 45,512 training pairs in one
batch, an effective Adam step of `2e-7`, and replay weights
`100/20/100/20` for reconstruction/prediction/semiconjugacy/cycle. Saddle
tubes, the origin fan, and audited tubes received component multipliers
`4/3/8`. Trust, fixed-phase, characteristic-polynomial, and topology-role
terms were also included. Training ran 3,170 epochs and selected epoch 669.
The promoted checkpoint SHA-256 is
`9fbee2cde690d58d2413c0d3521763838abaeb493736120f7173035612da3f3d`.

## Subdivision audit

| Subdivision `(i,m,M)` | Nodes | Morse boxes | Outcome |
|---|---:|---:|---|
| `(20,24,26)` | 5 | 108,982 | `S2` and `p_star` merged |
| `(24,27,28)` | 8 | 629,742 | six roles separated; two extras remain |
| `(25,28,29)` | 11 | 1,133,093 | five-box origin-side micro-splitting |
| `(25,28,30)` | 8 | 1,133,088 | returns to the max-28 graph |

The stable edges are `2->1, 3->2, 4->3, 5->4, 5->0, 6->5, 7->6`.
Assignments are `P0->0`, `P1->4`, `S2->2`, `S4->5`, `p_star->3`, and
`origin->7`. `P0`, `S4`, `p_star`, and the origin have the requested indices;
`P1` and `S2` do not. Node 1 is an extra attractor and node 6 an extra saddle.

## Direct-system ground truth overlay

![Direct ground truth on max-30 Morse sets](assets/direct_ground_truth_on_max30_morse_sets.png)

The background contains every learned max-30 latent Morse box. Colored clouds
are encoded centers of the 10,498-cell render-only display cover derived from
the 1,955,948 direct-map level-33 Morse boxes. Outlined symbols are the 16 exact
direct-system phases. Center encoding is a sampled visual comparison, not an
enclosing image of each three-dimensional box under `E`.

Every exact phase lies in one unique latent Morse set. This does not establish
index-pair transport, eliminate extra recurrence, fix the two incorrect
indices, or restore the missing `S2 -> P1` relation.

## Conditional ideal-data statement

If continuous `E`, `D`, and `g` attain zero population reconstruction and
semiconjugacy loss on a full-support measure over compact `K`, then
`D(E(x))=x` and `g(E(x))=E(f(x))` pointwise, so `E` is a homeomorphism from
`K` to `E(K)` and the restricted dynamics are conjugate. Conley-index recovery
still requires transport of a valid index pair, no extra or lost invariant
dynamics, and conjugate quotient index maps (or certified shift equivalence).
Semiconjugacy alone is insufficient.

See `report_manifest.json` and
`assets/direct_ground_truth_overlay_provenance.json` for full machine-readable
provenance and hashes.
"""


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    source_hashes = _verify_inputs()
    asset_hashes = _copy_assets()
    styles = _paragraph_styles()

    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=25 * mm,
        bottomMargin=17 * mm,
        title="Leslie3D invariant-aware latent dynamics: max-30 Morse audit",
        author="Codex",
        subject="Invariant-aware training, Morse sets, Morse graph, and direct-system overlay",
    )
    document.build(_build_story(styles), canvasmaker=ReportCanvas)

    MARKDOWN_PATH.write_text(_markdown(), encoding="utf-8")
    outputs = {
        str(PDF_PATH): {
            "sha256": _sha256(PDF_PATH),
            "size_bytes": PDF_PATH.stat().st_size,
        },
        str(MARKDOWN_PATH): {
            "sha256": _sha256(MARKDOWN_PATH),
            "size_bytes": MARKDOWN_PATH.stat().st_size,
        },
    }
    manifest = {
        "schema_version": 1,
        "title": "Leslie3D invariant-aware latent dynamics: max-30 Morse audit",
        "generated_date": "2026-08-04",
        "status": "numerical_experiment_not_a_conley_certificate",
        "source_sha256": source_hashes,
        "asset_sha256": asset_hashes,
        "outputs": outputs,
        "overlay_provenance": str(OVERLAY_PROVENANCE),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
