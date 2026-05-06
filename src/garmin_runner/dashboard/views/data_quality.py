from __future__ import annotations


def render(data, config=None) -> None:
    import streamlit as st

    kept = int(data.activities["keep"].sum()) if "keep" in data.activities and not data.activities.empty else 0
    rejected = len(data.activities) - kept
    st.metric("Feltöltött fájlok", data.uploaded_file_count)
    st.metric("Feldolgozott FIT fájlok", len(data.activities))
    st.metric("Megtartott futóedzések", kept)
    st.metric("Kiszűrt nem futó edzések", rejected)
    st.metric("Ideiglenes fájlok törölve", "igen" if data.temp_files_deleted else "nem")
    if data.warnings:
        st.subheader("Figyelmeztetések")
        for warning in data.warnings:
            st.warning(warning)
