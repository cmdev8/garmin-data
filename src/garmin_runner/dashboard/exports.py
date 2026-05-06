from __future__ import annotations

import io
import json
import zipfile
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from garmin_runner.dashboard.data import DashboardData
from garmin_runner.dashboard.i18n import PAGE_TITLE
from garmin_runner.dashboard.i18n import plan_family_label
from garmin_runner.dashboard.visualizations import (
    candidate_summary_values,
    diminishing_returns_values,
    feature_importance_table,
    hr_zone_table,
    latest_weekly_statistics_table,
    ml_metric_values,
    pace_zone_table,
    plan_adjusted_prediction_table,
    plan_reason_labels,
    plan_schedule_table,
    plan_summary,
    rejection_summary_table,
    scoring_component_values,
    selected_candidate_row,
    temporal_driver_table,
    temporal_ml_values,
    top_plan_ids,
)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def json_to_bytes(obj: dict) -> bytes:
    return json.dumps(obj, indent=2, default=str).encode("utf-8")


def markdown_to_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def build_training_plan_pdf(data: DashboardData) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font_name = _pdf_font_name(TTFont, pdfmetrics)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        pageCompression=0,
        title="Személyes edzésterv",
        author=PAGE_TITLE,
    )
    styles = _pdf_styles(font_name, ParagraphStyle, getSampleStyleSheet, TA_CENTER, TA_RIGHT)
    story: list[Any] = []

    candidate = selected_candidate_row(data.candidate_plans, data.next_week_plan)
    kept = int(data.activities["keep"].sum()) if "keep" in data.activities and not data.activities.empty else 0

    story.extend(
        [
            Paragraph(PAGE_TITLE, styles["Title"]),
            Paragraph("Személyes edzésterv", styles["Subtitle"]),
            Spacer(1, 8),
            _pdf_table_from_dataframe(
                pd.DataFrame(
                    [
                        {"Mutató": "Elemzés ideje", "Érték": data.analyzed_at},
                        {"Mutató": "Feltöltött fájlok", "Érték": data.uploaded_file_count},
                        {"Mutató": "Megtartott futások", "Érték": kept},
                        {
                            "Mutató": "Kiválasztott edzésterv",
                            "Érték": f"{data.next_week_plan.get('selected_plan_id', 'n/a')} ({plan_family_label(data.next_week_plan.get('family', 'n/a'))})",
                        },
                        {"Mutató": "Időszak", "Érték": f"{data.next_week_plan.get('start_date', 'n/a')} - {data.next_week_plan.get('end_date', 'n/a')}"},
                        {"Mutató": "Horizont", "Érték": f"{data.next_week_plan.get('horizon_days', 'n/a')} nap"},
                    ]
                ),
                styles,
                col_widths=[42 * mm, 116 * mm],
                header=False,
            ),
            Spacer(1, 12),
        ]
    )

    summary_values = {**plan_summary(data.next_week_plan), **_selected_candidate_cards(candidate)}
    story.extend(_pdf_section_heading("Terv összefoglaló", styles))
    story.append(_pdf_key_value_cards(summary_values, styles))

    story.extend(_pdf_section_heading("Edzésnaptár", styles))
    story.append(
        _pdf_table_from_dataframe(
            plan_schedule_table(data.next_week_plan),
            styles,
            max_rows=35,
            columns=["Dátum", "Nap", "Edzés", "Intenzitás", "Idő (perc)", "Táv (km)", "futás-séta előírás", "Cél"],
            col_widths=[21 * mm, 22 * mm, 30 * mm, 22 * mm, 18 * mm, 17 * mm, 30 * mm, 38 * mm],
        )
    )

    story.extend(_pdf_section_heading("Miért ezt a tervet?", styles))
    reasons = plan_reason_labels(data.next_week_plan.get("why_selected") or [])
    if reasons:
        story.append(_pdf_bullets([str(reason) for reason in reasons], styles))
    else:
        story.append(Paragraph("Nincs külön indoklás.", styles["Body"]))
    warnings = data.next_week_plan.get("warnings") or []
    if warnings:
        story.append(Paragraph("<b>Figyelmeztetések</b>", styles["SmallHeading"]))
        story.append(_pdf_bullets([str(warning) for warning in warnings], styles, color=colors.HexColor("#9f1239")))

    debug = data.plan_optimizer_debug
    driver = _historical_driver_cards(debug)
    if driver:
        story.append(Paragraph("<b>Történeti improvement driver</b>", styles["SmallHeading"]))
        story.append(_pdf_key_value_cards(driver, styles, columns=3))
    story.append(
        _pdf_key_value_cards(
            {
                "Generált jelöltek": str(debug.get("generated_candidate_count", "n/a")),
                "Érvényes jelöltek": str(debug.get("valid_candidate_count", "n/a")),
                "Elutasított jelöltek": str(debug.get("rejected_candidate_count", "n/a")),
            },
            styles,
            columns=3,
        )
    )
    top_ids = top_plan_ids(debug)
    if top_ids:
        story.append(Paragraph("<b>Top 5 jelölt:</b> " + escape(", ".join(top_ids)), styles["Body"]))
    story.append(_pdf_table_from_dataframe(rejection_summary_table(debug), styles, max_rows=12, col_widths=[112 * mm, 28 * mm]))

    story.extend(_pdf_section_heading("Scoring components", styles))
    story.append(_pdf_key_value_cards(scoring_component_values(candidate), styles, columns=3))

    story.extend(_pdf_section_heading("Várható előnyök", styles))
    story.append(
        Paragraph(
            "A tervhatás becslés az optimizer adaptáció/fatigue/overload risk score-jaiból készül; nem új model run.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "A diminishing returns factor csökkenti a pozitív becsült improvementet, ha a feltöltött korábbi edzések szerint magasabb edzettségnél kisebb a marginális válasz.",
            styles["Body"],
        )
    )
    story.append(_pdf_key_value_cards(diminishing_returns_values(data.ml_result), styles, columns=3))
    factor = data.ml_result.get("predictions", {}).get("diminishing_returns_factor")
    story.append(
        _pdf_table_from_dataframe(
            plan_adjusted_prediction_table(data.predictions, candidate, data.next_week_plan.get("horizon_days"), factor),
            styles,
            max_rows=12,
            col_widths=[17 * mm, 25 * mm, 27 * mm, 24 * mm, 22 * mm, 25 * mm, 25 * mm, 20 * mm],
        )
    )

    story.extend(_pdf_section_heading("ML és feature importance", styles))
    story.append(_pdf_key_value_cards(ml_metric_values(data.ml_result), styles, columns=3))
    story.append(Paragraph("<b>Temporal ML explanation</b>", styles["SmallHeading"]))
    story.append(_pdf_key_value_cards(temporal_ml_values(data.ml_result), styles, columns=3))
    story.append(_pdf_table_from_dataframe(temporal_driver_table(data.ml_result, "positive"), styles, max_rows=6, col_widths=[92 * mm, 34 * mm, 34 * mm]))
    story.append(_pdf_table_from_dataframe(temporal_driver_table(data.ml_result, "negative"), styles, max_rows=6, col_widths=[92 * mm, 34 * mm, 34 * mm]))
    story.append(_pdf_table_from_dataframe(feature_importance_table(data.ml_result), styles, max_rows=12, col_widths=[118 * mm, 30 * mm]))

    story.extend(_pdf_section_heading("Friss heti training stats", styles))
    story.append(_pdf_table_from_dataframe(latest_weekly_statistics_table(data.weekly_features), styles, max_rows=16, col_widths=[76 * mm, 48 * mm]))

    story.extend(_pdf_section_heading("Pulzuszónák", styles))
    story.append(_pdf_table_from_dataframe(hr_zone_table(data.predictions), styles, max_rows=8, col_widths=[28 * mm, 42 * mm, 42 * mm]))

    story.extend(_pdf_section_heading("Tempózónák", styles))
    story.append(_pdf_table_from_dataframe(pace_zone_table(data.predictions), styles, max_rows=10, col_widths=[46 * mm, 44 * mm, 44 * mm]))

    story.extend(_pdf_section_heading("Adatvédelem", styles))
    story.append(Paragraph("A PDF nem tartalmaz nyers FIT fájlokat, nyers JSON dumpot vagy betanított modell artefaktumot.", styles["Body"]))

    doc.build(story, onFirstPage=_pdf_page_footer(font_name), onLaterPages=_pdf_page_footer(font_name))
    return buffer.getvalue()


def build_processed_results_zip(data: DashboardData) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.md", data.report_markdown or "")
        archive.writestr("predictions.json", json_to_bytes(data.predictions))
        archive.writestr("next_week_plan.json", json_to_bytes(data.next_week_plan))
        archive.writestr("ml_result.json", json_to_bytes(data.ml_result))
        archive.writestr("activity_features.csv", dataframe_to_csv_bytes(data.activity_features))
        archive.writestr("weekly_features.csv", dataframe_to_csv_bytes(data.weekly_features))
        archive.writestr("candidate_plans.csv", dataframe_to_csv_bytes(data.candidate_plans))
    return buffer.getvalue()


def _pdf_font_name(TTFont, pdfmetrics) -> str:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("DashboardSans", path))
                return "DashboardSans"
            except Exception:
                continue
    return "Helvetica"


def _pdf_styles(font_name: str, ParagraphStyle, getSampleStyleSheet, TA_CENTER, TA_RIGHT) -> dict[str, Any]:
    from reportlab.lib import colors

    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle("TitleHu", parent=base["Title"], fontName=font_name, fontSize=24, leading=28, textColor=colors.HexColor("#172033"), spaceAfter=4),
        "Subtitle": ParagraphStyle("SubtitleHu", parent=base["Normal"], fontName=font_name, fontSize=14, leading=18, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=10),
        "Section": ParagraphStyle("SectionHu", parent=base["Heading2"], fontName=font_name, fontSize=14, leading=17, textColor=colors.HexColor("#0f172a"), spaceBefore=12, spaceAfter=6),
        "SmallHeading": ParagraphStyle("SmallHeadingHu", parent=base["Heading4"], fontName=font_name, fontSize=10, leading=13, textColor=colors.HexColor("#334155"), spaceBefore=6, spaceAfter=3),
        "Body": ParagraphStyle("BodyHu", parent=base["BodyText"], fontName=font_name, fontSize=8.5, leading=11.5, textColor=colors.HexColor("#1f2937")),
        "BodyRight": ParagraphStyle("BodyRightHu", parent=base["BodyText"], fontName=font_name, fontSize=8.5, leading=11.5, alignment=TA_RIGHT),
        "TableHeader": ParagraphStyle("TableHeaderHu", parent=base["BodyText"], fontName=font_name, fontSize=7.5, leading=9.5, textColor=colors.white),
        "TableCell": ParagraphStyle("TableCellHu", parent=base["BodyText"], fontName=font_name, fontSize=7.2, leading=9.2, textColor=colors.HexColor("#111827")),
        "CardLabel": ParagraphStyle("CardLabelHu", parent=base["BodyText"], fontName=font_name, fontSize=7, leading=8.5, textColor=colors.HexColor("#64748b")),
        "CardValue": ParagraphStyle("CardValueHu", parent=base["BodyText"], fontName=font_name, fontSize=10, leading=12, textColor=colors.HexColor("#111827")),
    }


def _pdf_section_heading(title: str, styles: dict[str, Any]) -> list[Any]:
    from reportlab.platypus import Paragraph, Spacer

    return [Spacer(1, 6), Paragraph(title, styles["Section"])]


def _pdf_key_value_cards(values: dict[str, Any], styles: dict[str, Any], *, columns: int = 4):
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import KeepTogether, Paragraph, Table, TableStyle

    items = list(values.items()) or [("Adat", "n/a")]
    rows: list[list[Any]] = []
    for start in range(0, len(items), columns):
        row: list[Any] = []
        for label, value in items[start : start + columns]:
            row.append([Paragraph(_pdf_text(label), styles["CardLabel"]), Paragraph(_pdf_text(value), styles["CardValue"])])
        while len(row) < columns:
            row.append("")
        rows.append(row)
    table = Table(rows, colWidths=[(178 * mm) / columns] * columns, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#d8dee9")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e5e7eb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return KeepTogether([table])


def _pdf_bullets(items: list[str], styles: dict[str, Any], *, color: Any | None = None):
    from reportlab.platypus import Paragraph

    style = styles["Body"]
    bullet_color = "#9f1239" if color is not None else "#334155"
    text = "<br/>".join(f'<font color="{bullet_color}">•</font> {escape(item)}' for item in items)
    return Paragraph(text or "n/a", style)


def _pdf_table_from_dataframe(
    table: pd.DataFrame,
    styles: dict[str, Any],
    *,
    max_rows: int = 20,
    columns: list[str] | None = None,
    col_widths: list[Any] | None = None,
    header: bool = True,
):
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Table, TableStyle

    if table.empty:
        return Paragraph("Nincs megjeleníthető adat.", styles["Body"])
    selected_columns = [column for column in (columns or list(table.columns)) if column in table.columns]
    if not selected_columns:
        return Paragraph("Nincs megjeleníthető adat.", styles["Body"])
    visible = table[selected_columns].head(max_rows)
    rows: list[list[Any]] = []
    if header:
        rows.append([Paragraph(_pdf_text(column), styles["TableHeader"]) for column in selected_columns])
    for _, row in visible.iterrows():
        rows.append([Paragraph(_pdf_text(row.get(column, "")), styles["TableCell"]) for column in selected_columns])
    if len(table) > max_rows:
        rows.append([Paragraph(_pdf_text(f"... további {len(table) - max_rows} sor"), styles["TableCell"])] + [""] * (len(selected_columns) - 1))
    widths = col_widths or [(178 * mm) / len(selected_columns)] * len(selected_columns)
    flowable = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
        first_data_row = 1
    else:
        first_data_row = 0
    for row_index in range(first_data_row, len(rows)):
        if (row_index - first_data_row) % 2 == 1:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f8fafc")))
    flowable.setStyle(TableStyle(commands))
    return flowable


def _selected_candidate_cards(candidate: dict[str, Any]) -> dict[str, str]:
    values = candidate_summary_values(candidate)
    return {
        "Adaptáció": values.get("Adaptáció", "n/a"),
        "fatigue": values.get("fatigue", "n/a"),
        "overload risk": values.get("overload risk", "n/a"),
        "Történeti driver match": values.get("Történeti driver match", "n/a"),
    }


def _historical_driver_cards(debug: dict[str, Any]) -> dict[str, str]:
    driver = debug.get("historical_improvement_driver")
    if not isinstance(driver, dict) or driver.get("category") in {None, "neutral"}:
        return {}
    return {
        "Driver": str(driver.get("label") or "n/a"),
        "Kategória": str(driver.get("category") or "n/a"),
        "Forrás": str(driver.get("source_distance_range") or "n/a"),
    }


def _pdf_page_footer(font_name: str):
    def draw(canvas, doc) -> None:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm

        width, _ = A4
        canvas.saveState()
        canvas.setFont(font_name, 7)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(16 * mm, 9 * mm, "Adatvédelem: nyers FIT fájlok és betanított modell artefaktumok nincsenek a PDF-ben.")
        canvas.drawRightString(width - 16 * mm, 9 * mm, f"{doc.page}. oldal")
        canvas.restoreState()

    return draw


def _pdf_text(value: Any, limit: int = 140) -> str:
    text = str(value if value is not None else "n/a").replace("\n", " ").strip()
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return escape(text)
