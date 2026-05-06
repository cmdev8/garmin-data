from __future__ import annotations

from garmin_runner.dashboard.charts import prepare_activity_table
from garmin_runner.dashboard.components import metric_grid
from garmin_runner.dashboard.filters import filter_by_activity_id
from garmin_runner.dashboard.visualizations import (
    activity_advanced_fit_table,
    activity_option_labels,
    activity_quality_notes,
    activity_record_chart_table,
    activity_segment_summary,
    activity_segment_table,
    selected_activity_metrics,
)


def render(data, config=None) -> None:
    import streamlit as st

    st.subheader("Edzések")
    table = prepare_activity_table(data.activity_features)
    if table.empty:
        st.info("Nincs megjeleníthető edzés.")
        return
    st.dataframe(table, width="stretch", hide_index=True)

    labels = activity_option_labels(data.activity_features)
    if not labels:
        st.info("Nincs kiválasztható edzés.")
        return
    selected_label = st.selectbox("Edzés kiválasztása", list(labels.values()))
    selected = next(activity_id for activity_id, label in labels.items() if label == selected_label)
    row = data.activity_features[data.activity_features["activity_id"].astype(str) == str(selected)].iloc[0].to_dict()

    st.markdown("### Kiválasztott edzés")
    for group, values in selected_activity_metrics(row).items():
        st.markdown(f"**{group}**")
        metric_grid(values, columns=4)

    _render_record_section(st, "Tempó és pulzus", data.records, selected, "pace_hr", "Nincs tempó vagy pulzus idősor ehhez az edzéshez.")
    _render_record_section(
        st,
        "Lépésütem, power és futódinamika",
        data.records,
        selected,
        "mechanics",
        "Nincs lépésütem, power vagy futódinamika idősor ehhez az edzéshez.",
    )
    _render_record_section(st, "Szint és környezet", data.records, selected, "environment", "Nincs szint vagy hőmérséklet idősor ehhez az edzéshez.")

    st.markdown("### Futás-séta")
    segment_summary = activity_segment_summary(data.segments, selected)
    segment_table = activity_segment_table(data.segments, selected)
    if segment_summary.empty and segment_table.empty:
        st.info("Nincs futás-séta szakaszadat ehhez az edzéshez.")
    else:
        if not segment_summary.empty:
            st.dataframe(segment_summary, width="stretch", hide_index=True)
        if not segment_table.empty:
            with st.expander("Szakaszok részletei"):
                st.dataframe(segment_table, width="stretch", hide_index=True)

    st.markdown("### Minőség és FIT mezők")
    for note in activity_quality_notes(row, data.records, selected):
        st.caption(note)
    advanced = activity_advanced_fit_table(row)
    if advanced.empty:
        st.info("Nincs további opcionális FIT mező ehhez az edzéshez.")
    else:
        with st.expander("Opcionális FIT mezők"):
            st.dataframe(advanced, width="stretch", hide_index=True)


def _render_record_section(st, title: str, records, activity_id: str, chart_type: str, empty_text: str) -> None:
    st.markdown(f"### {title}")
    filtered = filter_by_activity_id(records, activity_id)
    chart = activity_record_chart_table(filtered, activity_id, chart_type)
    if chart.empty:
        st.info(empty_text)
        return
    st.line_chart(chart.set_index("perc"))
