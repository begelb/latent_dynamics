#!/usr/bin/env python3
"""Build a deterministic PDF summary for the completed Ives 3x5 sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAGE_W, PAGE_H = letter
MARGIN = 42
CONTENT_W = PAGE_W - 2 * MARGIN

NAVY = colors.HexColor("#0B1F33")
NAVY_2 = colors.HexColor("#163A59")
INK = colors.HexColor("#172B3A")
MUTED = colors.HexColor("#5A6B78")
LINE = colors.HexColor("#D9E2E8")
PAPER = colors.HexColor("#F4F7F8")
WHITE = colors.white
TEAL = colors.HexColor("#18A999")
TEAL_DARK = colors.HexColor("#087F73")
TEAL_PALE = colors.HexColor("#DFF5F1")
ORANGE = colors.HexColor("#F4A340")
ORANGE_PALE = colors.HexColor("#FFF0DC")
RED = colors.HexColor("#C84A4A")
RED_PALE = colors.HexColor("#FBE7E7")
BLUE = colors.HexColor("#4B83C4")
BLUE_PALE = colors.HexColor("#E9F1FA")


class DeterministicCanvas(canvas.Canvas):
    """Canvas configured to avoid volatile PDF timestamps and IDs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["invariant"] = 1
        kwargs["pageCompression"] = 1
        super().__init__(*args, **kwargs)


class HorizontalBars:
    """Small table-backed horizontal bar chart."""

    def __init__(
        self,
        items: Iterable[tuple[str, int]],
        *,
        width: float = 244,
        bar_color: colors.Color = TEAL,
        max_value: int | None = None,
    ) -> None:
        self.items = list(items)
        self.width = width
        self.bar_color = bar_color
        self.max_value = max_value or max(value for _, value in self.items)

    def flowable(self, styles: dict[str, ParagraphStyle]) -> Table:
        rows: list[list[Any]] = []
        bar_width = self.width - 104
        for label, value in self.items:
            fraction = 0 if self.max_value == 0 else value / self.max_value
            filled = max(2, bar_width * fraction) if value else 0
            empty = max(0, bar_width - filled)
            bar = Table(
                [["", ""]],
                colWidths=[filled, empty],
                rowHeights=[8],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, 0), self.bar_color),
                        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E8EEF2")),
                        ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#D6E0E6")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ]
                ),
            )
            rows.append(
                [
                    Paragraph(escape(label), styles["chart_label"]),
                    bar,
                    Paragraph(str(value), styles["chart_value"]),
                ]
            )
        return Table(
            rows,
            colWidths=[76, bar_width, 24],
            rowHeights=[21] * len(rows),
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            ),
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_label(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}"


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}
    styles["body"] = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.2,
        textColor=INK,
        spaceAfter=7,
    )
    styles["small"] = ParagraphStyle(
        "Small",
        parent=styles["body"],
        fontSize=7.4,
        leading=10.2,
        textColor=MUTED,
        spaceAfter=3,
    )
    styles["tiny"] = ParagraphStyle(
        "Tiny",
        parent=styles["small"],
        fontSize=6.2,
        leading=8.2,
    )
    styles["title"] = ParagraphStyle(
        "CoverTitle",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=31,
        leading=34,
        textColor=WHITE,
        alignment=TA_LEFT,
        spaceAfter=12,
    )
    styles["cover_subtitle"] = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["body"],
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#BFD8E8"),
        spaceAfter=22,
    )
    styles["cover_body"] = ParagraphStyle(
        "CoverBody",
        parent=styles["body"],
        fontSize=10.3,
        leading=15,
        textColor=colors.HexColor("#E9F2F7"),
        spaceAfter=6,
    )
    styles["metric"] = ParagraphStyle(
        "Metric",
        parent=styles["body"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=23,
        alignment=TA_CENTER,
        textColor=WHITE,
        spaceAfter=0,
    )
    styles["metric_label"] = ParagraphStyle(
        "MetricLabel",
        parent=styles["small"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9,
        alignment=TA_CENTER,
        textColor=WHITE,
        spaceAfter=0,
    )
    styles["page_title"] = ParagraphStyle(
        "PageTitle",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=23,
        textColor=NAVY,
        spaceAfter=5,
    )
    styles["page_deck"] = ParagraphStyle(
        "PageDeck",
        parent=styles["body"],
        fontSize=10.2,
        leading=14.4,
        textColor=MUTED,
        spaceAfter=13,
    )
    styles["h2"] = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=NAVY_2,
        spaceBefore=4,
        spaceAfter=6,
    )
    styles["h3"] = ParagraphStyle(
        "H3",
        parent=styles["h2"],
        fontSize=9.2,
        leading=11,
        textColor=TEAL_DARK,
        spaceBefore=2,
        spaceAfter=4,
    )
    styles["callout"] = ParagraphStyle(
        "Callout",
        parent=styles["body"],
        fontSize=9.3,
        leading=13.4,
        textColor=NAVY,
        spaceAfter=0,
    )
    styles["table_header"] = ParagraphStyle(
        "TableHeader",
        parent=styles["small"],
        fontName="Helvetica-Bold",
        fontSize=6.6,
        leading=8,
        alignment=TA_CENTER,
        textColor=WHITE,
        spaceAfter=0,
    )
    styles["table_cell"] = ParagraphStyle(
        "TableCell",
        parent=styles["small"],
        fontSize=6.4,
        leading=7.8,
        alignment=TA_CENTER,
        textColor=INK,
        spaceAfter=0,
    )
    styles["matrix"] = ParagraphStyle(
        "Matrix",
        parent=styles["small"],
        fontName="Helvetica-Bold",
        fontSize=7.1,
        leading=9.2,
        alignment=TA_CENTER,
        textColor=INK,
        spaceAfter=0,
    )
    styles["chart_label"] = ParagraphStyle(
        "ChartLabel",
        parent=styles["small"],
        fontSize=7.2,
        leading=8,
        textColor=INK,
        spaceAfter=0,
    )
    styles["chart_value"] = ParagraphStyle(
        "ChartValue",
        parent=styles["chart_label"],
        fontName="Helvetica-Bold",
        alignment=TA_RIGHT,
    )
    styles["mono"] = ParagraphStyle(
        "Mono",
        parent=styles["small"],
        fontName="Courier",
        fontSize=6.6,
        leading=9,
        textColor=NAVY,
        wordWrap="CJK",
        spaceAfter=0,
    )
    styles["atlas_label"] = ParagraphStyle(
        "AtlasLabel",
        parent=styles["tiny"],
        fontName="Helvetica-Bold",
        fontSize=6.4,
        leading=7.5,
        alignment=TA_CENTER,
        textColor=NAVY_2,
        spaceAfter=2,
    )
    return styles


def body_page(canv: canvas.Canvas, doc: SimpleDocTemplate) -> None:
    canv.saveState()
    canv.setStrokeColor(LINE)
    canv.setLineWidth(0.6)
    canv.line(MARGIN, PAGE_H - 35, PAGE_W - MARGIN, PAGE_H - 35)
    canv.setFont("Helvetica-Bold", 7.2)
    canv.setFillColor(NAVY_2)
    canv.drawString(MARGIN, PAGE_H - 27, "IVES LAKE MYVATN 3x5 REPLICATION")
    canv.setFont("Helvetica", 7.2)
    canv.setFillColor(MUTED)
    canv.drawRightString(PAGE_W - MARGIN, PAGE_H - 27, "FINAL EVIDENCE SUMMARY")
    canv.line(MARGIN, 31, PAGE_W - MARGIN, 31)
    canv.setFont("Helvetica", 7)
    canv.drawString(MARGIN, 20, "latent-dynamics | generated from verified sweep artifacts")
    canv.drawRightString(PAGE_W - MARGIN, 20, f"{doc.page}")
    canv.restoreState()


def cover_page(canv: canvas.Canvas, doc: SimpleDocTemplate) -> None:
    canv.saveState()
    canv.setFillColor(NAVY)
    canv.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canv.setFillColor(TEAL)
    canv.rect(0, PAGE_H - 12, PAGE_W, 12, fill=1, stroke=0)
    canv.setFillColor(ORANGE)
    canv.circle(PAGE_W - 70, PAGE_H - 87, 31, fill=1, stroke=0)
    canv.setFillColor(NAVY_2)
    canv.circle(PAGE_W - 104, PAGE_H - 117, 18, fill=1, stroke=0)
    canv.setStrokeColor(colors.HexColor("#35536D"))
    canv.setLineWidth(0.7)
    canv.line(MARGIN, 31, PAGE_W - MARGIN, 31)
    canv.setFont("Helvetica", 7)
    canv.setFillColor(colors.HexColor("#AFC5D3"))
    canv.drawString(MARGIN, 20, "latent-dynamics | reproducible experiment report")
    canv.drawRightString(PAGE_W - MARGIN, 20, f"{doc.page}")
    canv.restoreState()


def page_title(
    title: str,
    deck: str,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    return [
        Paragraph(escape(title), styles["page_title"]),
        Paragraph(deck, styles["page_deck"]),
        HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=12),
    ]


def panel(
    content: list[Any],
    *,
    width: float,
    background: colors.Color = PAPER,
    border: colors.Color = LINE,
    padding: float = 10,
) -> Table:
    return Table(
        [[content]],
        colWidths=[width],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, border),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), padding),
                ("RIGHTPADDING", (0, 0), (-1, -1), padding),
                ("TOPPADDING", (0, 0), (-1, -1), padding),
                ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
            ]
        ),
    )


def bullet(text: str, styles: dict[str, ParagraphStyle], *, small: bool = False) -> Paragraph:
    style = styles["small"] if small else styles["body"]
    return Paragraph(text, style, bulletText="-")


def fit_image(path: Path, *, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def failed_gate_label(cell: dict[str, Any]) -> str:
    classification = cell["classification"]
    gates = (
        ("archive graph", classification["graph_shape_pass"]),
        ("fixed assignment", classification["fixed_assignment_pass"]),
        ("cycle assignment", classification["cycle_assignment_pass"]),
        ("sink separation", classification["distinct_sink_pass"]),
    )
    failed = [label for label, passed in gates if not passed]
    return ", ".join(failed) if failed else "none"


def gallery_row(
    *,
    cell: dict[str, Any],
    overlay_png: Path,
    graph_png: Path,
    styles: dict[str, ParagraphStyle],
) -> Table:
    graph = cell["morse_graph"]
    classification = cell["classification"]
    passed = bool(cell["machine_pass"])
    fixed_label = (
        f"M{classification['fixed_sink_id']}"
        if classification["fixed_sink_id"] is not None
        else "unassigned"
    )
    cycle_label = (
        f" in M{classification['cycle_sink_id']}"
        if classification["cycle_sink_id"] is not None
        else ""
    )
    result_color = "#087F73" if passed else "#C84A4A"
    status = "PASS" if passed else "VERIFIED FAIL"
    evidence = [
        Paragraph(
            f"<b>MODEL SEED {cell['model_seed']}</b><br/>"
            f'<font color="{result_color}"><b>{status}</b></font>',
            styles["h3"],
        ),
        Paragraph(
            f"N/E/S = {graph['n_nodes']}/{graph['n_edges']}/{len(graph['sink_ids'])}<br/>"
            f"{cell['morse_sets']['total_boxes']:,} saved Morse boxes<br/>"
            f"selected validation loss {cell['training']['selected_loss_total']:.2e}",
            styles["small"],
        ),
        Paragraph(
            f"<b>Fixed point:</b> {fixed_label}<br/>"
            f"<b>Period-12:</b> {classification['cycle_unique_target_count']}/12 "
            f"accepted{cycle_label}; {classification['cycle_conflicting_phase_count']} "
            f"conflicting; {classification['cycle_unassigned_count']} unassigned",
            styles["small"],
        ),
        Paragraph(
            ("All strict gates passed." if passed else f"Failed gates: {failed_gate_label(cell)}."),
            styles["tiny"],
        ),
    ]
    set_block = [
        Paragraph("LATENT MORSE SETS + REFINED ORBIT", styles["atlas_label"]),
        fit_image(overlay_png, max_width=164, max_height=158),
    ]
    graph_block = [
        Paragraph("MORSE GRAPH", styles["atlas_label"]),
        fit_image(graph_png, max_width=132, max_height=124),
    ]
    return Table(
        [[set_block, graph_block, evidence]],
        colWidths=[178, 146, 192],
        rowHeights=[184],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (1, 0), WHITE),
                ("BACKGROUND", (2, 0), (2, 0), TEAL_PALE if passed else RED_PALE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("LINEBEFORE", (1, 0), (2, 0), 0.5, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ALIGN", (0, 0), (1, 0), "CENTER"),
            ]
        ),
    )


def build_pdf(sweep_root: Path, output_path: Path) -> None:
    summary_dir = sweep_root / "summary"
    aggregate_path = summary_dir / "aggregate_summary.json"
    cells_path = summary_dir / "cells.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    detailed = json.loads(cells_path.read_text(encoding="utf-8"))
    overlay_dir = summary_dir / "figures" / "invariant_overlays"
    overlay_manifest_path = overlay_dir / "manifest.json"
    overlay_manifest = json.loads(overlay_manifest_path.read_text(encoding="utf-8"))
    cells = detailed["cells"]
    expected = aggregate["expected_design"]
    inventory = aggregate["inventory"]
    training = aggregate["training"]
    topology = aggregate["topology"]
    first_config = cells[0]["run_manifest"]["config"]
    data_cfg = first_config["data"]
    train_cfg = first_config["training"]
    cmgdb = first_config["cmgdb"]
    system = first_config["system"]["params"]

    assert aggregate["provisional"] is False
    assert inventory["n_complete_cells"] == 15
    assert inventory["n_verified_cells"] == 15
    assert len(cells) == 15
    assert overlay_manifest["sources"]["cells_json"]["sha256"] == sha256(cells_path)
    assert overlay_manifest["design"]["gallery_run_count"] == 15
    assert overlay_manifest["all_refined_memberships_equal_saved_memberships"] is True

    data_order = expected["data_seeds"]
    model_order = expected["model_seeds"]
    cell_by_key = {(cell["data_seed"], cell["model_seed"]): cell for cell in cells}
    pass_cells = [cell for cell in cells if cell["machine_pass"]]
    assert len(pass_cells) == 1
    winner = pass_cells[0]
    winner_key = (winner["data_seed"], winner["model_seed"])
    overlay_by_key = {
        (record["data_seed"], record["model_seed"]): record
        for record in overlay_manifest["renders"]
        if not record["detailed"]
    }
    assert set(overlay_by_key) == set(cell_by_key)
    winner_detail = next(
        record
        for record in overlay_manifest["renders"]
        if record["detailed"]
        and (record["data_seed"], record["model_seed"]) == winner_key
    )

    loss_ranked = sorted(cells, key=lambda cell: cell["training"]["selected_loss_total"])
    winner_loss_rank = next(i for i, cell in enumerate(loss_ranked, 1) if cell is winner)
    gate_counts = {
        "graph shape": sum(cell["classification"]["graph_shape_pass"] for cell in cells),
        "fixed point": sum(cell["classification"]["fixed_assignment_pass"] for cell in cells),
        "cycle assignment": sum(cell["classification"]["cycle_assignment_pass"] for cell in cells),
        "distinct sinks": sum(cell["classification"]["distinct_sink_pass"] for cell in cells),
    }
    sink_counts = Counter(len(cell["morse_graph"]["sink_ids"]) for cell in cells)

    graph_png = Path(winner_detail["outputs"]["morse_graph_300dpi_png"]["path"])
    sets_png = Path(winner_detail["outputs"]["png"]["path"])
    if not graph_png.exists() or not sets_png.exists():
        raise FileNotFoundError("successful-cell invariant overlay or graph render is missing")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=48,
        bottomMargin=42,
        title="Ives Lake Myvatn 3x5 Replication - Final Summary",
        author="latent-dynamics",
        subject="Final evidence summary for the Ives Lake Myvatn 15-cell replication",
        creator="scripts/build_ives_myvatn_3x5_pdf.py",
    )

    story: list[Any] = []

    # Page 1: executive outcome.
    story.extend(
        [
            Spacer(1, 76),
            Paragraph("Ives Lake Myvatn", styles["title"]),
            Paragraph("3x5 replication - final evidence summary", styles["cover_subtitle"]),
            HRFlowable(width=118, thickness=3, color=TEAL, hAlign="LEFT", spaceAfter=25),
        ]
    )
    metric_cells = []
    for value, label, color in (
        ("15/15", "CELLS COMPLETE AND VERIFIED", TEAL_DARK),
        ("1/15", "STRICT SCIENTIFIC PASSES", ORANGE),
        ("6.7%", "PASS RATE", BLUE),
    ):
        metric_cells.append(
            Table(
                [[Paragraph(value, styles["metric"])], [Paragraph(label, styles["metric_label"])]],
                colWidths=[CONTENT_W / 3 - 8],
                rowHeights=[35, 26],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), color),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                ),
            )
        )
    story.append(
        Table(
            [metric_cells],
            colWidths=[CONTENT_W / 3] * 3,
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            ),
        )
    )
    story.extend(
        [
            Spacer(1, 28),
            panel(
                [
                    Paragraph("PRIMARY FINDING", styles["metric_label"]),
                    Spacer(1, 6),
                    Paragraph(
                        "Only <b>data seed 2158 / model seed 2</b> recovered the archived "
                        "four-node branch-then-chain graph and separated the fixed point "
                        "from all 12 phases of the period-12 cycle into distinct sinks.",
                        styles["cover_body"],
                    ),
                ],
                width=CONTENT_W,
                background=NAVY_2,
                border=colors.HexColor("#35536D"),
                padding=15,
            ),
            Spacer(1, 18),
            Paragraph(
                "The result is complete, not provisional: every planned cell has a full "
                "artifact set and a hash-backed classification. The low pass frequency "
                "shows that the archived topological behavior is reproducible for this "
                "protocol, but sensitive to dataset and network initialization.",
                styles["cover_body"],
            ),
            Spacer(1, 38),
            Paragraph(
                f"Evidence generated {escape(utc_label(aggregate['generated_at_utc']))}. "
                "Model source: Ives et al. (2008), Nature 452, 84-87, "
                '<link href="https://doi.org/10.1038/nature06610" color="#7ED7CB">'
                "doi:10.1038/nature06610</link>.",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Page 2: protocol.
    story.extend(
        page_title(
            "Frozen experiment protocol",
            "Five independently sampled training datasets crossed with three model "
            "initializations. Every cell uses the same physical map, validation set, "
            "network family, and topology recipe.",
            styles,
        )
    )
    left = [
        Paragraph("DATA CONTRACT", styles["h2"]),
        bullet(f"{data_cfg['n_samples_train']:,} training initial conditions per dataset.", styles),
        bullet(f"{data_cfg['n_samples_val']:,} validation initial conditions; shared seed {data_cfg['val_seed']}.", styles),
        bullet(f"T={data_cfg['n_iterations']}, discard first {data_cfg['skip']} steps, retain 20 transitions per trajectory.", styles),
        bullet("20,000 training pairs and 4,000 validation pairs per dataset.", styles),
        bullet("Five training CSV files are pairwise distinct; validation files are byte-identical.", styles),
        Spacer(1, 7),
        Paragraph("SEEDS", styles["h2"]),
        Paragraph(
            "Data: <b>" + ", ".join(map(str, data_order)) + "</b><br/>"
            "Model: <b>" + ", ".join(map(str, model_order)) + "</b>",
            styles["body"],
        ),
        Spacer(1, 7),
        Paragraph("SYSTEM AND COORDINATES", styles["h2"]),
        Paragraph(
            "Log10 coordinates on [-3, -7.5, -3] to [1.5, 1.5, 1.5]. "
            f"r1={system['r1']}, r2={system['r2']}, c={fmt_float(system['c'], 6)}, "
            f"d={system['d']}, p={system['p']}, q={system['q']}.",
            styles["body"],
        ),
    ]
    right = [
        Paragraph("MODEL", styles["h2"]),
        panel(
            [
                Paragraph("ENCODER", styles["h3"]),
                Paragraph("3 -> 32 ReLU -> 2 tanh", styles["body"]),
                Paragraph("LATENT MAP", styles["h3"]),
                Paragraph("2 -> 64 ReLU x 5 -> 2 tanh", styles["body"]),
                Paragraph("DECODER", styles["h3"]),
                Paragraph("2 -> 32 ReLU -> 3 sigmoid", styles["body"]),
            ],
            width=247,
            background=BLUE_PALE,
            border=colors.HexColor("#B8CCE1"),
        ),
        Spacer(1, 10),
        Paragraph("TRAINING", styles["h2"]),
        bullet(f"Adam, learning rate {train_cfg['learning_rate']}, batch {train_cfg['batch_size']}.", styles),
        bullet(f"Maximum {train_cfg['epochs']} epochs; early-stopping patience {train_cfg['patience']}.", styles),
        bullet("Equal reconstruction, prediction, and semiconjugacy weights; no gradient clipping.", styles),
        Spacer(1, 5),
        Paragraph("TOPOLOGY", styles["h2"]),
        bullet(
            f"Adaptive precomputed box map; subdivision {cmgdb['subdiv_init']}/{cmgdb['subdiv_min']}/{cmgdb['subdiv_max']}, limit {cmgdb['subdiv_limit']:,}.",
            styles,
        ),
        bullet("64^3 system-grid latent bounds, include latent image, expand 10%, clip to [-1, 1]^2.", styles),
        bullet("Morse graph and Morse sets only; regions of attraction disabled.", styles),
    ]
    story.append(
        Table(
            [[left, right]],
            colWidths=[258, 258],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (0, 0), 0),
                    ("RIGHTPADDING", (0, 0), (0, 0), 10),
                    ("LEFTPADDING", (1, 0), (1, 0), 10),
                    ("RIGHTPADDING", (1, 0), (1, 0), 0),
                    ("LINEBEFORE", (1, 0), (1, 0), 0.7, LINE),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            ),
        )
    )
    story.extend(
        [
            Spacer(1, 12),
            panel(
                [
                    Paragraph("STRICT CELL PASS", styles["h2"]),
                    Paragraph(
                        "The observed directed graph must be isomorphic to the archived "
                        "4-node, 3-edge, 2-sink branch-then-chain. The fixed point must "
                        "belong uniquely to one sink; at least 11 of 12 cycle phases must "
                        "belong uniquely to one common other sink, with zero conflicting "
                        "sink assignments. Periods {1, 12} are diagnostic only.",
                        styles["callout"],
                    ),
                ],
                width=CONTENT_W,
                background=TEAL_PALE,
                border=colors.HexColor("#9DD9D0"),
                padding=11,
            ),
            PageBreak(),
        ]
    )

    # Page 3: complete grid.
    story.extend(
        page_title(
            "Cell-by-cell outcomes",
            "The only pass occurs in the first dataset with model seed 2. "
            "N/E/S denotes Morse nodes, directed edges, and sinks.",
            styles,
        )
    )
    matrix_rows: list[list[Any]] = [
        [Paragraph("DATA SEED", styles["table_header"])]
        + [Paragraph(f"MODEL {seed}", styles["table_header"]) for seed in model_order]
    ]
    matrix_backgrounds: list[tuple[str, tuple[int, int], tuple[int, int], colors.Color]] = []
    for row_idx, data_seed in enumerate(data_order, 1):
        row: list[Any] = [Paragraph(str(data_seed), styles["matrix"])]
        for col_idx, model_seed in enumerate(model_order, 1):
            cell = cell_by_key[(data_seed, model_seed)]
            graph = cell["morse_graph"]
            loss = cell["training"]["selected_loss_total"]
            status = "PASS" if cell["machine_pass"] else "FAIL"
            row.append(
                Paragraph(
                    f"{status}<br/>N{graph['n_nodes']} E{graph['n_edges']} "
                    f"S{len(graph['sink_ids'])}<br/>loss {loss:.2e}",
                    styles["matrix"],
                )
            )
            matrix_backgrounds.append(
                ("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), TEAL_PALE if cell["machine_pass"] else RED_PALE)
            )
        matrix_rows.append(row)
    matrix_style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY_2),
        ("BACKGROUND", (0, 1), (0, -1), PAPER),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        *matrix_backgrounds,
    ]
    story.append(
        Table(
            matrix_rows,
            colWidths=[78, 150, 150, 150],
            rowHeights=[22] + [44] * len(data_order),
            style=TableStyle(matrix_style),
        )
    )
    story.extend([Spacer(1, 11), Paragraph("EXACT CLASSIFICATION EVIDENCE", styles["h2"])])
    headers = ["data", "model", "result", "N", "E", "S", "fixed", "cycle", "phases", "conf.", "val loss"]
    detail_rows: list[list[Any]] = [[Paragraph(item, styles["table_header"]) for item in headers]]
    for data_seed in data_order:
        for model_seed in model_order:
            cell = cell_by_key[(data_seed, model_seed)]
            graph = cell["morse_graph"]
            cls = cell["classification"]
            values = [
                str(data_seed),
                str(model_seed),
                "PASS" if cell["machine_pass"] else "fail",
                str(graph["n_nodes"]),
                str(graph["n_edges"]),
                str(len(graph["sink_ids"])),
                cls["fixed_sink_id"] or "-",
                cls["cycle_sink_id"] or "-",
                f"{cls['cycle_unique_target_count']}/12",
                str(cls["cycle_conflicting_phase_count"]),
                f"{cell['training']['selected_loss_total']:.2e}",
            ]
            detail_rows.append([Paragraph(value, styles["table_cell"]) for value in values])
    widths = [42, 34, 42, 24, 24, 24, 39, 39, 44, 36, 70]
    detail_style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY_2),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAPER]),
    ]
    winner_row = 1 + data_order.index(winner_key[0]) * len(model_order) + model_order.index(winner_key[1])
    detail_style.extend(
        [
            ("BACKGROUND", (0, winner_row), (-1, winner_row), TEAL_PALE),
            ("TEXTCOLOR", (0, winner_row), (-1, winner_row), TEAL_DARK),
            ("LINEABOVE", (0, winner_row), (-1, winner_row), 1.0, TEAL),
            ("LINEBELOW", (0, winner_row), (-1, winner_row), 1.0, TEAL),
        ]
    )
    story.append(Table(detail_rows, colWidths=widths, repeatRows=1, style=TableStyle(detail_style)))
    story.extend(
        [
            Spacer(1, 5),
            Paragraph(
                "All 15 classifications are backed by complete artifact sets and SHA-256 "
                "records. A dash means the invariant object did not receive a valid unique "
                "sink assignment under the strict rule.",
                styles["tiny"],
            ),
            PageBreak(),
        ]
    )

    # Page 4: successful cell.
    story.extend(
        page_title(
            "The successful cell",
            f"Data seed {winner_key[0]}, model seed {winner_key[1]} is the single cell "
            "that satisfies every graph and invariant-membership gate.",
            styles,
        )
    )
    graph_image = fit_image(graph_png, max_width=238, max_height=156)
    top_evidence = [
        Paragraph("GRAPH EVIDENCE", styles["h2"]),
        bullet("4 nodes, 3 directed edges, 2 sinks.", styles),
        bullet("Exact node-ID-invariant match to the archived branch-then-chain.", styles),
        bullet("Observed edges: 3 -> 1, 3 -> 2, 1 -> 0.", styles),
        bullet("Sink 0 has inferred period 12; sink 2 has inferred period 1.", styles),
        Paragraph(
            "Conley periods agree with {1, 12}, although this diagnostic does not affect "
            "the machine-pass decision.",
            styles["small"],
        ),
    ]
    story.append(
        Table(
            [[graph_image, top_evidence]],
            colWidths=[250, 266],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND", (0, 0), (0, 0), WHITE),
                    ("BACKGROUND", (1, 0), (1, 0), BLUE_PALE),
                    ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                    ("LINEBEFORE", (1, 0), (1, 0), 0.6, LINE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            ),
        )
    )
    story.extend([Spacer(1, 12), Paragraph("LATENT MORSE SETS AND INVARIANT ASSIGNMENT", styles["h2"])])
    sets_image = fit_image(sets_png, max_width=249, max_height=249)
    membership = [
        Paragraph("PERIOD-12 CYCLE", styles["h3"]),
        Paragraph(
            "12/12 phases are uniquely assigned to sink 0. There are no unassigned, "
            "ambiguous, or conflicting phases.",
            styles["body"],
        ),
        Paragraph("FIXED POINT", styles["h3"]),
        Paragraph("Uniquely assigned to sink 2.", styles["body"]),
        Paragraph("SEPARATION", styles["h3"]),
        Paragraph("The cycle and fixed point occupy distinct sinks.", styles["body"]),
        Paragraph("TRAINING", styles["h3"]),
        Paragraph(
            f"Selected validation loss {winner['training']['selected_loss_total']:.3e}; "
            f"rank {winner_loss_rank} of 15 from lowest to highest. Best epoch "
            f"{winner['training']['best_epoch']} of {winner['training']['epochs_completed']} "
            f"completed; training time {winner['training']['train_duration_seconds']:.2f} s.",
            styles["body"],
        ),
        panel(
            [
                Paragraph(
                    "PASS: graph shape + fixed assignment + cycle assignment + distinct sinks",
                    styles["callout"],
                )
            ],
            width=245,
            background=TEAL_PALE,
            border=colors.HexColor("#9DD9D0"),
            padding=9,
        ),
    ]
    story.append(
        Table(
            [[sets_image, membership]],
            colWidths=[260, 256],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (0, 0), 5),
                    ("RIGHTPADDING", (0, 0), (0, 0), 10),
                    ("LEFTPADDING", (1, 0), (1, 0), 10),
                    ("RIGHTPADDING", (1, 0), (1, 0), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            ),
        )
    )
    story.extend(
        [
            Spacer(1, 5),
            Paragraph(
                "Numbered circles are the CPU encodings E(x_k) of the refined direct-system "
                "period-12 phases; the closed path gives direct-system phase order, not "
                "iteration of the learned latent map G. The star is the encoded refined "
                "fixed point. Background colors are every saved Morse set and correspond "
                "to the graph nodes shown above.",
                styles["tiny"],
            ),
            PageBreak(),
        ]
    )

    # Pages 5-9: all 15 cells, one data seed per page.
    for data_seed in data_order:
        story.extend(
            page_title(
                f"Complete run gallery - data seed {data_seed}",
                "Every model initialization is reported, including verified scientific "
                "failures. Each row pairs all saved latent Morse boxes with its saved graph.",
                styles,
            )
        )
        story.append(
            Paragraph(
                "White circles and the connecting line show the independently encoded "
                "refined period-12 phases; the star marks the encoded fixed point. Node "
                "IDs and colors are local to each paired set-and-graph display. The refined "
                f"orbit closes under the direct map to "
                f"{overlay_manifest['sources']['period12_closure_residual']:.2e}; all "
                "refined Morse memberships equal the recorded classifications.",
                styles["small"],
            )
        )
        story.append(Spacer(1, 3))
        for model_index, model_seed in enumerate(model_order):
            cell = cell_by_key[(data_seed, model_seed)]
            record = overlay_by_key[(data_seed, model_seed)]
            story.append(
                gallery_row(
                    cell=cell,
                    overlay_png=Path(record["outputs"]["png"]["path"]),
                    graph_png=Path(record["outputs"]["morse_graph_300dpi_png"]["path"]),
                    styles=styles,
                )
            )
            if model_index != len(model_order) - 1:
                story.append(Spacer(1, 7))
        story.append(PageBreak())

    # Page 10: robustness and interpretation.
    story.extend(
        page_title(
            "Robustness and failure anatomy",
            "The dominant failure mode is topological collapse to one sink. Training "
            "loss alone does not identify the archived two-attractor structure.",
            styles,
        )
    )
    node_items = sorted((str(key) + " nodes", value) for key, value in topology["node_count_distribution"].items())
    gate_items = list(gate_counts.items())
    charts = Table(
        [
            [
                [
                    Paragraph("MORSE NODE COUNT", styles["h2"]),
                    HorizontalBars(node_items, width=246, bar_color=BLUE, max_value=5).flowable(styles),
                ],
                [
                    Paragraph("CELLS CLEARING EACH GATE", styles["h2"]),
                    HorizontalBars(gate_items, width=246, bar_color=TEAL, max_value=15).flowable(styles),
                ],
            ]
        ],
        colWidths=[258, 258],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 10),
                ("LEFTPADDING", (1, 0), (1, 0), 10),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
                ("LINEBEFORE", (1, 0), (1, 0), 0.6, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        ),
    )
    story.append(charts)
    story.extend([Spacer(1, 12)])
    one_sink = sink_counts.get(1, 0)
    two_sink = sink_counts.get(2, 0)
    story.append(
        Table(
            [
                [
                    Paragraph(f"<b>{one_sink}/15</b><br/>one-sink cells", styles["callout"]),
                    Paragraph(f"<b>{two_sink}/15</b><br/>two-sink cells", styles["callout"]),
                    Paragraph("<b>3/15</b><br/>unique fixed-point assignments", styles["callout"]),
                    Paragraph("<b>1/15</b><br/>valid cycle assignments", styles["callout"]),
                ]
            ],
            colWidths=[129] * 4,
            rowHeights=[52],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), RED_PALE),
                    ("BACKGROUND", (1, 0), (1, 0), TEAL_PALE),
                    ("BACKGROUND", (2, 0), (2, 0), ORANGE_PALE),
                    ("BACKGROUND", (3, 0), (3, 0), BLUE_PALE),
                    ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                    ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            ),
        )
    )
    story.extend([Spacer(1, 15), Paragraph("VALIDATION LOSS IS NOT A TOPOLOGY SURROGATE", styles["h2"])])
    top_loss_rows: list[list[Any]] = [
        [Paragraph(item, styles["table_header"]) for item in ("loss rank", "data", "model", "selected val loss", "topology")]
    ]
    for rank, cell in enumerate(loss_ranked[:5], 1):
        top_loss_rows.append(
            [
                Paragraph(str(rank), styles["table_cell"]),
                Paragraph(str(cell["data_seed"]), styles["table_cell"]),
                Paragraph(str(cell["model_seed"]), styles["table_cell"]),
                Paragraph(f"{cell['training']['selected_loss_total']:.3e}", styles["table_cell"]),
                Paragraph("PASS" if cell["machine_pass"] else "fail", styles["table_cell"]),
            ]
        )
    top_loss_style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY_2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAPER]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    story.append(Table(top_loss_rows, colWidths=[74, 88, 74, 150, 130], style=TableStyle(top_loss_style)))
    story.extend(
        [
            Spacer(1, 10),
            panel(
                [
                    Paragraph("INTERPRETATION", styles["h2"]),
                    Paragraph(
                        f"The passing model has the {winner_loss_rank}rd-lowest selected "
                        "validation loss. The two lower-loss models both fail because they "
                        "produce only one sink. Consequently, predictive and reconstruction "
                        "fit are necessary model-quality signals, but they do not by themselves "
                        "certify the archived global dynamics.",
                        styles["callout"],
                    ),
                ],
                width=CONTENT_W,
                background=ORANGE_PALE,
                border=colors.HexColor("#E7BF85"),
                padding=11,
            ),
            Spacer(1, 10),
            Paragraph(
                f"Across all cells: mean selected validation loss "
                f"{training['selected_validation_losses']['loss_total']['mean']:.3e}; "
                f"median {training['selected_validation_losses']['loss_total']['median']:.3e}. "
                f"Mean training duration {training['train_duration_seconds']['mean']:.2f} s; "
                f"mean completed epochs {training['epochs_completed']['mean']:.1f}.",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Page 11: reproducibility and integrity.
    story.extend(
        page_title(
            "Reproducibility and artifact integrity",
            "The final result is traceable to the frozen launcher, config, source "
            "invariant points, per-cell manifests, and content hashes.",
            styles,
        )
    )
    integrity_rows = [
        ["Planned grid", "5 data seeds x 3 model seeds = 15 cells"],
        ["Complete / verified", f"{inventory['n_complete_cells']} / {inventory['n_verified_cells']}"],
        ["Invalid / issues", f"{inventory['n_invalid_cells']} / {inventory['n_issues']}"],
        ["Report displays", "15 Morse-set overlays + 15 Morse graphs (all cells)"],
        ["Aggregate status", "final (provisional = false)"],
        ["Aggregate SHA-256", sha256(aggregate_path)],
        ["Cells JSON SHA-256", sha256(cells_path)],
        ["Classification invariant SHA-256", aggregate["reference_invariant_points"]["file"]["sha256"]],
    ]
    integrity_table: list[list[Any]] = [
        [Paragraph("CHECK", styles["table_header"]), Paragraph("RECORDED VALUE", styles["table_header"])]
    ]
    for key, value in integrity_rows:
        style = styles["mono"] if "SHA-256" in key else styles["small"]
        integrity_table.append([Paragraph(escape(key), styles["small"]), Paragraph(escape(value), style)])
    story.append(
        Table(
            integrity_table,
            colWidths=[146, 370],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY_2),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAPER]),
                    ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            ),
        )
    )
    story.extend([Spacer(1, 13)])
    paths_panel = [
        Paragraph("REPRODUCE FROM code/", styles["h2"]),
        Paragraph("bash scripts/run_ives_myvatn_3x5.sh", styles["mono"]),
        Paragraph("python scripts/render_ives_myvatn_3x5_invariant_overlays.py", styles["mono"]),
        Paragraph("python scripts/build_ives_myvatn_3x5_pdf.py", styles["mono"]),
        Spacer(1, 4),
        Paragraph("PRIMARY EVIDENCE", styles["h2"]),
        Paragraph("Within output/ives_myvatn_seedsweep_3x5_v1/summary/:", styles["small"]),
        Paragraph("aggregate_summary.json | cells.json | cells.csv | SUMMARY.md", styles["mono"]),
        Paragraph("figures/invariant_overlays/manifest.json", styles["mono"]),
        Spacer(1, 4),
        Paragraph("FROZEN INPUTS", styles["h2"]),
        Paragraph("src/latentdynamics/configs/ives_myvatn.yaml", styles["mono"]),
        Paragraph("src/latentdynamics/reference_data/ives_myvatn_invariant_points.csv", styles["mono"]),
        Paragraph(
            "output/ives_myvatn_3d_ground_truth/invariant_stability/refined_invariant_points.csv",
            styles["mono"],
        ),
    ]
    story.append(panel(paths_panel, width=CONTENT_W, background=BLUE_PALE, border=colors.HexColor("#B8CCE1"), padding=8))
    story.extend([Spacer(1, 6)])
    story.append(
        panel(
            [
                Paragraph("CONTROLLER STATUS NOTE", styles["h2"]),
                Paragraph(
                    "A historical controller status line reported a false failure because CSV "
                    "integers were parsed as strings. The parser is corrected and regression-tested; "
                    "fresh strict verification passes all 15 artifact audits and records one "
                    "scientific pass. No scientific artifact was missing or stale.",
                    styles["callout"],
                ),
            ],
            width=CONTENT_W,
            background=ORANGE_PALE,
            border=colors.HexColor("#E7BF85"),
            padding=9,
        )
    )
    story.extend(
        [
            Spacer(1, 6),
            Paragraph(
                "Conclusion: this 1000-IC, 3x5 replication produced one strict recovery of "
                "the archived two-attractor topology. All 15 Morse-set and graph pairs are "
                "reported with the encoded refined orbit. The result is scientifically "
                "complete and artifact-verified; the rare pass emphasizes initialization "
                "sensitivity.",
                styles["body"],
            ),
        ]
    )

    doc.build(
        story,
        onFirstPage=cover_page,
        onLaterPages=body_page,
        canvasmaker=DeterministicCanvas,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("output/ives_myvatn_seedsweep_3x5_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/ives_myvatn_3x5_summary.pdf"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_pdf(args.sweep_root.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
