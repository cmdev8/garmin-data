from __future__ import annotations

from garmin_runner.dashboard.components import metric_grid
from garmin_runner.dashboard.data import DashboardData
from garmin_runner.dashboard.exports_ui import render_downloads
from garmin_runner.dashboard.i18n import PAGE_TITLE, plan_family_label
from garmin_runner.dashboard.state import reset_analysis_state


def render_header(data: DashboardData) -> None:
    import streamlit as st

    st.title(PAGE_TITLE)
    kept = int(data.activities["keep"].sum()) if "keep" in data.activities and not data.activities.empty else 0
    metric_grid(
        {
            "Feltöltött fájlok": data.uploaded_file_count,
            "Megtartott futások": kept,
            "ML modell": data.ml_result.get("model_type", "n/a"),
            "Terv": plan_family_label(data.next_week_plan.get("family", "n/a")),
        },
        columns=4,
    )


def render_dashboard_sidebar(data: DashboardData) -> None:
    import streamlit as st

    st.sidebar.caption(f"Elemzés ideje: {data.analyzed_at}")
    st.sidebar.caption(f"Feltöltött fájlok: {data.uploaded_file_count}")
    st.sidebar.caption(f"Ideiglenes fájlok törölve: {'igen' if data.temp_files_deleted else 'nem'}")
    if st.sidebar.button("Elemzés alaphelyzetbe állítása"):
        reset_analysis_state()
        st.rerun()
    with st.sidebar.expander("Letöltések"):
        render_downloads(data)
