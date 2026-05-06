from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from garmin_runner.planning.schema import HistoricalDriver


@dataclass(frozen=True)
class HistoricalImprovementDriver:
    category: HistoricalDriver = "neutral"
    score: float = 0.0
    confidence: float = 0.0
    source_distance_range: str | None = None
    label: str | None = None
    reason: str | None = "No reliable historical improvement driver."


QUALITY_TERMS = {
    "Kemény arány",
    "Aerob training effect",
    "Anaerob training effect",
}
LONG_RUN_TERMS = {"Hosszú futás", "long_run_distance_km"}
AEROBIC_TERMS = {"Táv", "Mozgásidő", "7 napos terhelés", "Fáradtságra korrigált terhelés", "Korrigált ACWR", "ACWR"}
RECOVERY_TERMS = {"Fáradtság", "Forma", "Túlterhelési kockázat"}
RUN_WALK_TERMS = {"Futás-séta fáradtsági szorzó", "Tervezett futás-séta", "Séta arány"}
SPEED_ECONOMY_TERMS = {
    "Lépésütem",
    "Max lépésütem",
    "Lépéshossz",
    "Vertikális oszcilláció",
    "Vertikális arány",
    "Talajkontakt idő",
    "Teljesítmény/pulzus hatékonyság",
    "Átlagteljesítmény",
    "Normalizált teljesítmény",
}
NON_ACTIONABLE_TERMS = {
    "Adatminőség",
    "GPS lefedettség",
    "Átlaghőmérséklet",
    "Maximum hőmérséklet",
    "Emelkedés/terep proxy",
    "Magasságtartomány",
    "HRV mintaszám",
    "HRV RMSSD",
}


def extract_historical_improvement_driver(performance_change: dict[str, Any] | None) -> HistoricalImprovementDriver:
    payload = performance_change or {}
    summaries = [row for row in _list_of_dicts(payload.get("summary")) if _is_improving(row)]
    if not summaries:
        return HistoricalImprovementDriver()
    best = max(summaries, key=_summary_priority)
    source = str(best.get("distance_range") or "")
    confidence = _clip(_float(best.get("confidence")))
    improvement = max(0.0, -_float(best.get("adjusted_change_s_per_km")))
    models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    model = models.get(source, {}) if isinstance(models, dict) else {}
    importance = model.get("feature_importance") if isinstance(model, dict) else {}

    for label, value in _ordered_importance(importance):
        category = _category_for_label(label)
        if category != "neutral":
            score = _clip(confidence * max(0.2, float(value)) * min(1.0, improvement / 12.0))
            return HistoricalImprovementDriver(
                category=category,
                score=score,
                confidence=confidence,
                source_distance_range=source,
                label=label,
                reason=f"{source} range improved after controlling for context; strongest actionable feature: {label}.",
            )

    for label in _primary_factor_labels(best.get("primary_controlled_factors")):
        category = _category_for_label(label)
        if category != "neutral":
            score = _clip(confidence * min(1.0, improvement / 12.0) * 0.5)
            return HistoricalImprovementDriver(
                category=category,
                score=score,
                confidence=confidence,
                source_distance_range=source,
                label=label,
                reason=f"{source} range improved; primary controlled factor points to {label}.",
            )

    return HistoricalImprovementDriver(
        confidence=confidence,
        source_distance_range=source,
        reason="Historical improvement was detected, but the leading factors were not actionable for planning.",
    )


def _is_improving(row: dict[str, Any]) -> bool:
    return _float(row.get("adjusted_change_s_per_km")) < -1.0 and _float(row.get("confidence")) >= 0.45


def _summary_priority(row: dict[str, Any]) -> float:
    return max(0.0, -_float(row.get("adjusted_change_s_per_km"))) * max(0.0, _float(row.get("confidence")))


def _ordered_importance(value: Any) -> list[tuple[str, float]]:
    if not isinstance(value, dict):
        return []
    rows: list[tuple[str, float]] = []
    for key, score in value.items():
        rows.append((str(key), _float(score)))
    return sorted(rows, key=lambda item: item[1], reverse=True)


def _primary_factor_labels(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip() and part.strip() != "n/a"]


def _category_for_label(label: str) -> HistoricalDriver:
    if label in NON_ACTIONABLE_TERMS:
        return "neutral"
    if label in QUALITY_TERMS or "training effect" in label or "Kemény" in label:
        return "quality"
    if label in LONG_RUN_TERMS or "Hosszú" in label:
        return "long_run"
    if label in RECOVERY_TERMS or "fatigue" in label or "overload" in label:
        return "recovery"
    if label in RUN_WALK_TERMS or "Futás-séta" in label:
        return "run_walk"
    if label in SPEED_ECONOMY_TERMS or "Lépésütem" in label or "Teljesítmény" in label or "Lépéshossz" in label:
        return "speed_economy"
    if label in AEROBIC_TERMS or "terhelés" in label or label == "Táv":
        return "aerobic_base"
    return "neutral"


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _float(value: Any) -> float:
    try:
        if value is None or value != value:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))
