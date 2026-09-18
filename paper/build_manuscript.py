#!/usr/bin/env python3
"""Build the EarthArXiv-ready causal spatial correction manuscript."""

from __future__ import annotations

import html
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
OUTPUT = ROOT / "output" / "pdf"
MANUSCRIPT = PAPER / "manuscript.md"
PDF = OUTPUT / "causal-spatial-correction-etas-preprint-v1.0.pdf"

INK = "#15212b"
MUTED = "#586975"
TEAL = "#087f8c"
GREEN = "#2f855a"
GOLD = "#b7791f"
RED = "#b44545"
PALE = "#eef4f5"


def make_method_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    boxes = [
        (0.2, 2.55, 2.3, 1.2, "Frozen ETAS", "background + triggered", TEAL),
        (3.0, 3.0, 2.3, 0.95, "Renewal state", "magnitude-marked age", GOLD),
        (3.0, 1.55, 2.3, 0.95, "Frailty state", "observed / expected bg", GREEN),
        (5.85, 2.25, 2.15, 1.25, "Bounded tilt", "27.6% mixture cap", "#526d82"),
        (8.55, 2.25, 2.0, 1.25, "CH-008", "same total rate", RED),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.07",
            linewidth=1.5, edgecolor=color, facecolor="white"
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=12, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                fontsize=9.5, color=MUTED)

    arrows = [
        ((2.5, 3.15), (3.0, 3.45)),
        ((2.5, 2.85), (3.0, 2.05)),
        ((5.3, 3.45), (5.85, 3.05)),
        ((5.3, 2.05), (5.85, 2.65)),
        ((8.0, 2.88), (8.55, 2.88)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.3, color="#83939d"))
    ax.text(1.35, 1.45, "Triggered component passes through unchanged",
            ha="center", va="center", fontsize=9.5, color=MUTED)
    ax.add_patch(FancyArrowPatch((2.45, 1.45), (9.25, 2.22), connectionstyle="arc3,rad=-0.14",
                                 arrowstyle="-|>", mutation_scale=14, linewidth=1.3,
                                 color="#83939d"))
    ax.text(5.4, 0.48, "Causal daily update: only events before issue time enter state",
            ha="center", va="center", fontsize=10, color=INK, weight="bold")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_evidence_figure(path: Path) -> None:
    labels = ["California\nvalidation", "California\nlater period", "New Zealand\nexploratory", "Chile\nexploratory"]
    means = [0.0052134832, 0.0078646125, 0.0185613228, 0.0097184294]
    lo30 = [0.0019816550, 0.0050927317, 0.0120060366, 0.0068622690]
    hi30 = [0.0088799911, 0.0106231594, 0.0296706779, 0.0132194540]
    lo90 = [0.0018444070, 0.0053693552, 0.0118173829, 0.0063210820]
    hi90 = [0.0092161379, 0.0109144248, 0.0313572766, 0.0143484400]
    n = [5204, 3995, 2270, 1909]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.axhline(0, color="#6f7d85", linewidth=1)
    for i in x:
        ax.vlines(i, lo90[i], hi90[i], color="#9fb0ba", linewidth=2.5, zorder=1)
        ax.vlines(i, lo30[i], hi30[i], color=TEAL, linewidth=7, alpha=0.68, zorder=2)
        ax.scatter(i, means[i], s=70, color=INK, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(i, hi90[i] + 0.0015, f"N={n[i]:,}", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Information gain per earthquake (nats)")
    ax.set_ylim(-0.0015, 0.0355)
    ax.grid(axis="y", color="#dbe3e7", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Qualified pre-prospective evidence against frozen ETAS", loc="left",
                 fontsize=13, weight="bold", color=INK, pad=14)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_timeline_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.2))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    stages = [
        (0.15, 1.60, "Warm-up", "2007--2013\nunscored", "#9aaab3"),
        (1.95, 1.60, "Fit / selection", "2014--2018\nCalifornia", GOLD),
        (3.75, 1.60, "Validation", "2019--2022\nqualified", TEAL),
        (5.55, 1.60, "Later period", "2023--Aug 2026\nsupportive", GREEN),
        (7.35, 1.60, "Dry run", "1--14 Sep 2026\nnot claim", RED),
        (9.15, 1.60, "Prospective", "24 Sep 2026\n365 days", INK),
    ]
    for i, (x, width, label, sub, color) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((x, 1.36), width, 0.62,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=color, edgecolor="none", alpha=0.92))
        ax.text(x + width / 2, 1.67, label, ha="center", va="center",
                fontsize=9.2, color="white", weight="bold")
        ax.text(x + width / 2, 1.12, sub, ha="center", va="top",
                fontsize=8.1, color=MUTED, linespacing=1.35)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.04, 1.67),
                                         (stages[i + 1][0] - 0.04, 1.67),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.0, color="#8797a0"))
    ax.text(0.15, 2.70, "Evidence chronology", fontsize=13, weight="bold", color=INK)
    ax.text(0.15, 2.35, "Only the final stage is eligible for the formal prospective claim.",
            fontsize=9.3, color=MUTED)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_causal_method_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.7))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    boxes = [
        (0.15, 2.75, 2.05, 1.05, "Frozen ETAS", "daily count + support", TEAL),
        (2.65, 3.10, 2.05, 0.92, "Safe expert", "causal renewal", GREEN),
        (2.65, 1.72, 2.05, 0.92, "Event history", "256 causal events", GOLD),
        (5.20, 1.72, 2.15, 0.92, "Fast - slow MLP", "3 LORO members", "#526d82"),
        (5.20, 3.10, 2.15, 0.92, "Top 1% support", "50% restricted mix", RED),
        (7.85, 2.42, 1.65, 1.05, "BF20 gate", "prior days only", "#6b4f8a"),
        (9.90, 2.42, 1.15, 1.05, "Daily\nforecast", "same count", INK),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.07",
                                    linewidth=1.5, edgecolor=color, facecolor="white"))
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=10.2, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=8.3, color=MUTED)
    arrows = [
        ((2.20, 3.32), (2.65, 3.55)), ((2.20, 3.05), (5.20, 3.55)),
        ((4.70, 2.18), (5.20, 2.18)), ((7.35, 3.55), (7.85, 3.05)),
        ((7.35, 2.18), (7.85, 2.72)), ((9.50, 2.95), (9.90, 2.95)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.25, color="#84949e"))
    ax.text(5.60, 0.78, "Only spatial mass moves; total ETAS rate is invariant",
            ha="center", va="center", fontsize=10.3, color=INK, weight="bold")
    ax.text(5.60, 0.38, "Target-region outcomes are excluded from neural training, but retrospective replays remain development evidence",
            ha="center", va="center", fontsize=8.5, color=MUTED)
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_retrospective_figure(path: Path) -> None:
    labels = ["California", "New Zealand", "Chile", "Japan C"]
    means = [0.0094367, 0.0304595, 0.0107279, 0.0015031]
    lo30 = [0.0058836, 0.0194348, 0.0076018, 0.0010245]
    hi30 = [0.0131998, 0.0392234, 0.0145325, 0.0020978]
    lo90 = [0.0058590, 0.0196489, 0.0069351, 0.0010012]
    hi90 = [0.0130591, 0.0400194, 0.0158418, 0.0021665]
    events = [3619, 2270, 1909, 354]
    x = list(range(4))
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.axhline(0, color="#6f7d85", linewidth=1)
    for i in x:
        ax.vlines(i, lo90[i], hi90[i], color="#a8b7bf", linewidth=3, zorder=1)
        ax.vlines(i, lo30[i], hi30[i], color=TEAL, linewidth=8, alpha=0.72, zorder=2)
        ax.scatter(i, means[i], s=78, color=INK, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(i, max(hi30[i], hi90[i]) + 0.0014, f"N={events[i]:,}", ha="center",
                va="bottom", fontsize=8.8, color=MUTED)
    ax.set_xticks(x, labels)
    ax.set_ylabel("IGPE against frozen ETAS (nats)")
    ax.set_ylim(-0.0015, 0.044)
    ax.grid(axis="y", color="#dbe3e7", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Evidence-gated retrospective daily-grid replay", loc="left",
                 fontsize=13, weight="bold", color=INK, pad=13)
    ax.text(0.0, 1.01, "Thick: 30-day blocks   Thin: 90-day blocks   All target outcomes were opened",
            transform=ax.transAxes, fontsize=8.6, color=MUTED)
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_prospective_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    regions = [(0.25, "California", TEAL), (2.05, "New Zealand", GREEN),
               (3.85, "Chile", GOLD), (5.65, "Japan C", RED)]
    for x, label, color in regions:
        ax.add_patch(FancyBboxPatch((x, 2.55), 1.45, 0.72,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor="white", edgecolor=color, linewidth=1.5))
        ax.text(x + 0.725, 2.91, label, ha="center", va="center",
                fontsize=9, weight="bold", color=INK)
        ax.add_patch(FancyArrowPatch((x + 0.725, 2.55), (4.15, 2.03), arrowstyle="-|>",
                                     mutation_scale=10, linewidth=1.0, color="#8797a0"))
    ax.add_patch(FancyBboxPatch((3.35, 1.28), 2.15, 0.76,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=INK, edgecolor="none"))
    ax.text(4.425, 1.66, "Atomic daily issue", ha="center", va="center",
            fontsize=10, color="white", weight="bold")
    stages = [(6.10, "Provisional\nscore", "daily"), (7.70, "Final score", "+7 days"),
              (9.15, "Primary test", "day 365; N>=500")]
    prior = (5.50, 1.66)
    for x, title, sub in stages:
        ax.add_patch(FancyBboxPatch((x, 1.25), 1.25, 0.82,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor="white", edgecolor="#526d82", linewidth=1.4))
        ax.text(x + 0.625, 1.73, title, ha="center", va="center", fontsize=8.8,
                weight="bold", color=INK)
        ax.text(x + 0.625, 1.39, sub, ha="center", va="center", fontsize=7.8, color=MUTED)
        ax.add_patch(FancyArrowPatch(prior, (x, 1.66), arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.1, color="#8797a0"))
        prior = (x + 1.25, 1.66)
    ax.text(5.5, 0.45, "365 fixed target days  |  no backfill  |  no refit  |  no region exclusion",
            ha="center", va="center", fontsize=10, color=INK, weight="bold")
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def inline_markup(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"(https?://[^\s<]+)", r'<a href="\1" color="#087f8c">\1</a>', escaped)
    return escaped


def parse_markdown(text: str, styles: dict[str, ParagraphStyle]) -> list:
    lines = text.splitlines()
    story: list = []
    paragraph: list[str] = []
    bullets: list[str] = []
    numbered: list[str] = []
    table_rows: list[list[str]] = []
    seen_title = False
    front_matter = True

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(x.strip() for x in paragraph)), styles["Body"]))
            story.append(Spacer(1, 2.5 * mm))
            paragraph.clear()

    def flush_lists() -> None:
        nonlocal bullets, numbered
        if bullets:
            items = [ListItem(Paragraph(inline_markup(x), styles["List"]), leftIndent=4 * mm) for x in bullets]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=5 * mm, bulletFontSize=7))
            story.append(Spacer(1, 2 * mm))
            bullets = []
        if numbered:
            items = [ListItem(Paragraph(inline_markup(x), styles["List"]), leftIndent=4 * mm) for x in numbered]
            story.append(ListFlowable(items, bulletType="1", start="1", leftIndent=6 * mm))
            story.append(Spacer(1, 2 * mm))
            numbered = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        columns = len(table_rows[0])
        data = []
        for row_index, row in enumerate(table_rows):
            style = styles["TableHead"] if row_index == 0 else styles["TableCell"]
            data.append([Paragraph(inline_markup(cell), style) for cell in row])
        table = Table(data, colWidths=[166 * mm / columns] * columns,
                      repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f1f3")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK)),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(TEAL)),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d8e1e5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([Spacer(1, 1.5 * mm), table, Spacer(1, 3 * mm)])
        table_rows = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph(); flush_lists()
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                table_rows.append(cells)
            continue
        flush_table()
        image_match = re.match(r"!\[(.+)]\((.+)\)", stripped)
        if not stripped:
            flush_paragraph()
            flush_lists()
            continue
        if image_match:
            flush_paragraph(); flush_lists()
            caption, rel = image_match.groups()
            img_path = PAPER / rel
            image = Image(str(img_path), width=166 * mm, height=58 * mm, kind="proportional")
            cap = Paragraph(inline_markup(caption), styles["Caption"])
            story.extend([Spacer(1, 2 * mm), KeepTogether([image, Spacer(1, 1.5 * mm), cap]), Spacer(1, 3 * mm)])
            continue
        if stripped.startswith("# "):
            flush_paragraph(); flush_lists()
            if not seen_title:
                story.append(Spacer(1, 11 * mm))
                story.append(Paragraph(inline_markup(stripped[2:]), styles["Title"]))
                story.append(Spacer(1, 7 * mm))
                seen_title = True
            continue
        if stripped.startswith("## "):
            flush_paragraph(); flush_lists()
            front_matter = False
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(inline_markup(stripped[3:]), styles["H1"]))
            story.append(Spacer(1, 1.5 * mm))
            continue
        if stripped.startswith("### "):
            flush_paragraph(); flush_lists()
            story.append(Paragraph(inline_markup(stripped[4:]), styles["H2"]))
            continue
        if stripped.startswith("- "):
            flush_paragraph(); flush_lists() if numbered else None
            bullets.append(stripped[2:])
            continue
        match_num = re.match(r"\d+\.\s+(.+)", stripped)
        if match_num:
            flush_paragraph(); flush_lists() if bullets else None
            numbered.append(match_num.group(1))
            continue
        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) < 100:
            flush_paragraph(); flush_lists()
            story.append(Paragraph(inline_markup(stripped), styles["Meta"]))
            continue
        if front_matter and seen_title:
            flush_paragraph(); flush_lists()
            story.append(Paragraph(inline_markup(stripped.rstrip("  ")), styles["Meta"]))
            continue
        paragraph.append(stripped)
    flush_paragraph(); flush_lists(); flush_table()
    return story


class ManuscriptDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=22 * mm,
            rightMargin=22 * mm,
            topMargin=19 * mm,
            bottomMargin=19 * mm,
            title="Can a Causal Spatial Correction Improve ETAS Across Tectonic Regimes?",
            author="Saban Baris Boga",
            subject="Retrospective evidence and a frozen prospective earthquake-forecast test",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self._decorate))

    @staticmethod
    def _decorate(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d5dfe3"))
        canvas.setLineWidth(0.5)
        canvas.line(22 * mm, 14 * mm, 188 * mm, 14 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(22 * mm, 9 * mm, "Non-peer-reviewed preprint submitted to EarthArXiv · v1.0")
        canvas.drawRightString(188 * mm, 9 * mm, f"{doc.page}")
        canvas.restoreState()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=22, leading=26, textColor=colors.HexColor(INK),
                                alignment=TA_LEFT, spaceAfter=4 * mm),
        "CoverTitle": ParagraphStyle("CoverTitle", parent=base["Title"], fontName="Helvetica-Bold",
                                     fontSize=24, leading=29, textColor=colors.HexColor(INK),
                                     alignment=TA_LEFT, spaceAfter=8 * mm),
        "CoverKicker": ParagraphStyle("CoverKicker", parent=base["BodyText"], fontName="Helvetica-Bold",
                                      fontSize=10, leading=13, textColor=colors.HexColor(TEAL),
                                      alignment=TA_LEFT, spaceAfter=3 * mm),
        "CoverStatement": ParagraphStyle("CoverStatement", parent=base["BodyText"], fontName="Helvetica",
                                         fontSize=11, leading=15, textColor=colors.HexColor(INK),
                                         backColor=colors.HexColor(PALE), borderColor=colors.HexColor("#cbd9de"),
                                         borderWidth=0.7, borderPadding=10, spaceBefore=5 * mm,
                                         spaceAfter=7 * mm),
        "Meta": ParagraphStyle("Meta", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=9.3, leading=12.5, textColor=colors.HexColor(MUTED),
                               alignment=TA_LEFT, spaceAfter=1.2 * mm),
        "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold",
                             fontSize=14.2, leading=17, textColor=colors.HexColor(INK),
                             spaceBefore=5 * mm, spaceAfter=2 * mm, keepWithNext=True),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=11.2, leading=14, textColor=colors.HexColor(TEAL),
                             spaceBefore=3.5 * mm, spaceAfter=1.2 * mm, keepWithNext=True),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=9.35, leading=13.1, textColor=colors.HexColor(INK),
                               alignment=TA_JUSTIFY, allowWidows=0, allowOrphans=0),
        "List": ParagraphStyle("List", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=9.2, leading=12.6, textColor=colors.HexColor(INK),
                               alignment=TA_LEFT),
        "Caption": ParagraphStyle("Caption", parent=base["BodyText"], fontName="Helvetica-Oblique",
                                  fontSize=8.2, leading=10.5, textColor=colors.HexColor(MUTED),
                                  alignment=TA_LEFT, keepWithNext=False),
        "TableHead": ParagraphStyle("TableHead", parent=base["BodyText"], fontName="Helvetica-Bold",
                                    fontSize=7.4, leading=9.2, textColor=colors.HexColor(INK)),
        "TableCell": ParagraphStyle("TableCell", parent=base["BodyText"], fontName="Helvetica",
                                    fontSize=7.1, leading=9.0, textColor=colors.HexColor(INK)),
    }


def cover_sheet(s: dict[str, ParagraphStyle]) -> list:
    return [
        Spacer(1, 18 * mm),
        Paragraph("EARTHARXIV SUBMISSION COVERSHEET", s["CoverKicker"]),
        Paragraph(
            "Can a Causal Spatial Correction Improve ETAS Across Tectonic Regimes? "
            "Retrospective Evidence and a Frozen Prospective Test",
            s["CoverTitle"],
        ),
        Paragraph(
            "This manuscript is a <b>non-peer-reviewed preprint submitted to EarthArXiv</b>. "
            "It may be revised. The retrospective results are development evidence; the frozen "
            "prospective experiment had not produced a result as of 18 September 2026.",
            s["CoverStatement"],
        ),
        Paragraph("<b>Author</b>: Saban Baris Boga (sole author)", s["Meta"]),
        Paragraph("<b>Affiliation</b>: Independent Researcher, Adana, Turkiye", s["Meta"]),
        Paragraph(
            '<b>ORCID</b>: <a href="https://orcid.org/0009-0000-9076-946X" '
            'color="#087f8c">https://orcid.org/0009-0000-9076-946X</a>',
            s["Meta"],
        ),
        Paragraph("<b>Correspondence</b>: hello@bboga.com", s["Meta"]),
        Spacer(1, 6 * mm),
        Paragraph("<b>Version</b>: 1.0, 18 September 2026", s["Meta"]),
        Paragraph("<b>Submission type</b>: Research article / retrospective evidence and prospective protocol", s["Meta"]),
        Paragraph("<b>License</b>: Creative Commons Attribution 4.0 International (CC BY 4.0)", s["Meta"]),
        Paragraph(
            "<b>Keywords</b>: earthquake forecasting; ETAS; neural point process; "
            "information gain; prospective evaluation; CSEP; negative transfer",
            s["Meta"],
        ),
        Spacer(1, 8 * mm),
        Paragraph(
            "The formal 365-day prospective evaluation activates only when one atomic issue "
            "publishes all four regional forecasts for the same future UTC day. Pre-activation "
            "evidence is labeled according to its actual retrospective development status.",
            s["Body"],
        ),
        PageBreak(),
    ]


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    make_causal_method_figure(FIGURES / "causal-spatial-method.png")
    make_retrospective_figure(FIGURES / "retrospective-igpe.png")
    make_prospective_figure(FIGURES / "prospective-protocol.png")
    style_map = styles()
    story = cover_sheet(style_map) + parse_markdown(MANUSCRIPT.read_text(encoding="utf-8"), style_map)
    ManuscriptDocTemplate(str(PDF)).build(story)
    print(PDF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
