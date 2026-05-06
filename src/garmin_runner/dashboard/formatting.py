from __future__ import annotations


def pace(seconds_per_km) -> str:
    if seconds_per_km is None:
        return "n/a"
    try:
        seconds = int(round(float(seconds_per_km)))
    except (TypeError, ValueError):
        return "n/a"
    return f"{seconds // 60}:{seconds % 60:02d}/km"


def number(value, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def percent(value, digits: int = 0, *, signed: bool = False) -> str:
    try:
        numeric = float(value) * 100
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if signed else ""
    return f"{numeric:{sign}.{digits}f}%"
