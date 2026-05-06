from __future__ import annotations

from typing import Any


def run_walk_fatigue_multiplier(
    run_walk_type: Any,
    walk_fraction: Any = None,
    planned_run_walk_score: Any = None,
) -> float:
    planned_discount = _clip(0.30 * _float(walk_fraction) + 0.10 * _float(planned_run_walk_score), 0.0, 0.25)
    type_text = str(run_walk_type or "unknown")
    if type_text == "planned":
        multiplier = 1.0 - planned_discount
    elif type_text == "mixed":
        multiplier = 1.0 - 0.5 * planned_discount
    else:
        multiplier = 1.0
    return round(_clip(multiplier, 0.75, 1.0), 3)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _float(value: Any) -> float:
    try:
        if value is None or value != value:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
