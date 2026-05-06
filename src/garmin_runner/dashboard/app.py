from __future__ import annotations

from garmin_runner.dashboard.analysis_runner import run_analysis_for_uploads
from garmin_runner.dashboard.config import load_dashboard_config
from garmin_runner.dashboard.layout import render_dashboard_sidebar, render_header
from garmin_runner.dashboard.views import (
    activities,
    algorithms,
    attribution,
    blocks,
    data_quality,
    diagnostics,
    diminishing_returns,
    fitness,
    overview,
    performance_change,
    predictions,
    training_load,
)
from garmin_runner.dashboard.state import init_state, reset_analysis_state
from garmin_runner.dashboard.theme import PAGE_TITLE
from garmin_runner.dashboard.upload import render_upload_panel
from garmin_runner.dashboard.validation import has_running_activities
from garmin_runner.dashboard.i18n import PAGES_HU


PAGES = {
    "Áttekintés": overview.render,
    "Adatminőség": data_quality.render,
    "Diagnosztika": diagnostics.render,
    "Edzések": activities.render,
    PAGES_HU["training_load"]: training_load.render,
    "Fitness / fáradtság": fitness.render,
    PAGES_HU["performance_change"]: performance_change.render,
    "Edzésblokkok": blocks.render,
    PAGES_HU["attribution"]: attribution.render,
    PAGES_HU["diminishing_returns"]: diminishing_returns.render,
    PAGES_HU["predictions_plan"]: predictions.render,
    PAGES_HU["algorithms"]: algorithms.render,
}

PRE_ANALYSIS_PAGES = {
    "Elemzés indítása": "upload",
    PAGES_HU["algorithms"]: "algorithms",
}


def main() -> None:
    import streamlit as st

    config = load_dashboard_config()
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    init_state()

    if not st.session_state.get("analysis_complete"):
        selected_page = st.sidebar.radio("Oldal", list(PRE_ANALYSIS_PAGES.keys()))
        if PRE_ANALYSIS_PAGES[selected_page] == "algorithms":
            algorithms.render(None, config)
            return

        st.title(PAGE_TITLE)
        request = render_upload_panel()
        if request is not None:
            with st.status("Feltöltött FIT fájlok elemzése...", expanded=True) as status:
                try:
                    data = run_analysis_for_uploads(request, keep_temp=config.dev_keep_temp)
                    if not has_running_activities(data):
                        status.update(label="Az elemzés kész, de nem maradt futóedzés.", state="complete")
                        st.warning("Nem találtam futóedzést. A nem futás jellegű fájlokat kiszűrtem.")
                    st.session_state["dashboard_data"] = data
                    st.session_state["analysis_complete"] = True
                    status.update(label="Elemzés kész", state="complete")
                    st.rerun()
                except Exception as exc:
                    status.update(label="Az elemzés sikertelen", state="error")
                    st.error("Az elemzés sikertelen. A feltöltött fájlokat az app nem tárolta el.")
                    if config.show_debug:
                        st.exception(exc)
        return

    data = st.session_state["dashboard_data"]
    render_dashboard_sidebar(data)
    render_header(data)
    selected_page = st.sidebar.radio("Oldal", list(PAGES.keys()))
    PAGES[selected_page](data, config)


if __name__ == "__main__":
    main()
