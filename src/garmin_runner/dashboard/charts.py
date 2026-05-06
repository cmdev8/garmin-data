from __future__ import annotations

import pandas as pd

from garmin_runner.dashboard.formatting import number, pace, percent


def prepare_weekly_chart_data(weekly_features: pd.DataFrame) -> pd.DataFrame:
    if weekly_features.empty:
        return weekly_features
    data = weekly_features.copy()
    if "date" in data:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


def prepare_activity_table(activity_features: pd.DataFrame) -> pd.DataFrame:
    if activity_features.empty:
        return activity_features
    rows = []
    for row in activity_features.to_dict("records"):
        rows.append(
            {
                "Dátum": row.get("date", ""),
                "Táv": f"{number(row.get('distance_km'))} km",
                "Mozgásidő": _minutes(row.get("moving_time_min")),
                "Átlagtempó": pace(row.get("avg_pace_s_per_km")),
                "Átlagpulzus": _unit(row.get("avg_hr"), "bpm", 0),
                "Szintemelkedés": _unit(row.get("elevation_gain"), "m", 0),
                "Futás-séta": row.get("run_walk_type", "n/a"),
                "data quality": percent(row.get("data_quality_score")),
                "activity_id": row.get("activity_id", ""),
            }
        )
    return pd.DataFrame(rows)


def line_chart(df: pd.DataFrame, x: str, y: str):
    import streamlit as st

    if df.empty or x not in df or y not in df:
        st.info("Nincs megjeleníthető diagramadat.")
        return
    st.line_chart(df.set_index(x)[y])


def _minutes(value) -> str:
    try:
        minutes = int(round(float(value)))
    except (TypeError, ValueError):
        return "n/a"
    return f"{minutes} perc"


def _unit(value, unit: str, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f} {unit}"
    except (TypeError, ValueError):
        return "n/a"
