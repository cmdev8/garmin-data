from __future__ import annotations


def render(data, config=None) -> None:
    import streamlit as st

    columns = [
        "date",
        "run_walk_type",
        "run_time_s",
        "walk_time_s",
        "number_of_walk_breaks",
        "run_pace_only_s_per_km",
        "forced_walk_score",
    ]
    available = [column for column in columns if column in data.activity_features]
    if available:
        st.dataframe(data.activity_features[available], width="stretch")
    else:
        st.info("Nincs elérhető futás-séta mutató.")
    if not data.segments.empty:
        st.dataframe(data.segments, width="stretch")
