from __future__ import annotations

from garmin_runner.dashboard.visualizations import (
    performance_change_chart_table,
    performance_change_distance_options,
    performance_change_importance_table,
    performance_change_metric_values,
    performance_change_model_notes,
    performance_change_observation_table,
    performance_change_summary_table,
)


def render(data, config=None) -> None:
    import streamlit as st

    st.subheader("Teljesítménytrend")
    st.caption(
        "A korrigált trend a tempót futás-séta, szintemelkedés proxy, fatigue, training load, pulzus, lépésütem és adatminőség mellett becsüli. "
        "A terep itt szintemelkedésből származó proxy, nem ismert felszíntípus."
    )

    performance_change = data.performance_change
    warnings = performance_change.get("warnings") or []
    for warning in warnings:
        st.warning(str(warning))

    options = performance_change_distance_options(performance_change)
    if not options:
        st.info("Nincs elég feltöltött adat a teljesítménytrend megjelenítéséhez.")
        return

    cols = st.columns(4)
    for column, (label, value) in zip(cols, performance_change_metric_values(performance_change).items()):
        column.metric(label, value)

    st.subheader("Távonkénti összefoglaló")
    summary = performance_change_summary_table(performance_change)
    if summary.empty:
        st.info("Nincs megjeleníthető összefoglaló.")
    else:
        st.dataframe(summary, width="stretch", hide_index=True)

    selected = st.selectbox("Táv kiválasztása", options)
    chart = performance_change_chart_table(performance_change, selected)
    if chart.empty:
        st.info("Nincs megjeleníthető trenddiagram.")
    else:
        st.caption("A diagram tempót mutat perc/km skálán; az alacsonyabb érték gyorsabb futást jelent.")
        st.line_chart(chart.set_index("Dátum")[["raw tempó (perc/km)", "Korrigált tempó (perc/km)"]])

    notes = performance_change_model_notes(performance_change, selected)
    note_cols = st.columns(3)
    for column, (label, value) in zip(note_cols, notes.items()):
        if value:
            column.metric(label, value)

    importance = performance_change_importance_table(performance_change, selected)
    st.subheader("Kontrollált feature-ök hatása")
    if importance.empty:
        fallback = notes.get("Fallback") or "A model nem adott stabil feature importance becslést."
        st.info(fallback)
    else:
        st.bar_chart(importance.set_index("feature")["feature importance"])
        st.dataframe(importance, width="stretch", hide_index=True)

    with st.expander("Megfigyelések részletei"):
        observations = performance_change_observation_table(performance_change, selected)
        if observations.empty:
            st.info("Nincs megjeleníthető megfigyelés.")
        else:
            st.dataframe(observations, width="stretch", hide_index=True)
