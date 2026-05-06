from __future__ import annotations

from garmin_runner.dashboard.charts import line_chart, prepare_weekly_chart_data


def render(data, config=None) -> None:
    import streamlit as st

    weekly = prepare_weekly_chart_data(data.weekly_features)
    st.subheader("7 napos training load")
    line_chart(weekly, "date", "load_7d")
    st.subheader("Heti km")
    line_chart(weekly, "date", "distance_7d_km")
    st.dataframe(data.weekly_features, width="stretch")
