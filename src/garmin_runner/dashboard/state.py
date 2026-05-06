from __future__ import annotations


SESSION_KEYS = ["dashboard_data", "analysis_complete", "selected_activity_id", "date_min", "date_max"]


def init_state() -> None:
    import streamlit as st

    st.session_state.setdefault("analysis_complete", False)


def reset_analysis_state() -> None:
    import streamlit as st

    for key in SESSION_KEYS:
        st.session_state.pop(key, None)
