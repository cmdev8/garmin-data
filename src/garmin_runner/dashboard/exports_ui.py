from __future__ import annotations

from garmin_runner.dashboard.data import DashboardData
from garmin_runner.dashboard.exports import build_training_plan_pdf


def render_downloads(data: DashboardData) -> None:
    import streamlit as st

    st.download_button(
        "PDF tervjelentés letöltése",
        build_training_plan_pdf(data),
        "training_plan_report.pdf",
        mime="application/pdf",
    )
