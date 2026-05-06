from __future__ import annotations

from garmin_runner.dashboard.components import metric_grid
from garmin_runner.dashboard.visualizations import (
    training_block_composition_table,
    training_block_detail_values,
    training_block_improvement_chart,
    training_block_improvement_table,
    training_block_summary_values,
    training_block_timeline_table,
    training_block_volume_chart,
)


def render(data, config=None) -> None:
    import altair as alt
    import streamlit as st

    st.subheader("Edzésblokkok")
    if data.training_blocks.empty:
        st.info("Nem találtam edzésblokkokat.")
        return

    metric_grid(training_block_summary_values(data.training_blocks), columns=5)

    st.subheader("Idővonal")
    timeline = training_block_timeline_table(data.training_blocks)
    if timeline.empty:
        st.info("Nincs megjeleníthető blokk-idővonal.")
    else:
        st.caption("Minden sor egy edzésblokkot jelöl; a színek a blokk típusát mutatják.")
        chart = timeline[["Kezdet", "Vég", "Blokk", "Típus", "Kategória", "Szín", "Időtartam (nap)", "heti km", "improvement"]].dropna(
            subset=["Kezdet", "Vég"]
        )
        if chart.empty:
            st.dataframe(timeline[["Szín ikon", "Blokk", "Típus", "Kezdet", "Vég"]], width="stretch", hide_index=True)
        else:
            chart_view = (
                alt.Chart(chart)
                .mark_bar(size=22)
                .encode(
                    x=alt.X("Kezdet:T", title="Kezdet"),
                    x2="Vég:T",
                    y=alt.Y("Blokk:N", title="Edzésblokk", sort=chart["Blokk"].tolist()),
                    color=alt.Color(
                        "Kategória:N",
                        title="Típus",
                        scale=alt.Scale(
                            domain=["recovery", "base", "threshold", "long_run_endurance", "overload", "unknown"],
                            range=["#2ca25f", "#3182bd", "#f28e2b", "#756bb1", "#de2d26", "#8c8c8c"],
                        ),
                    ),
                    tooltip=["Blokk:N", "Típus:N", "Kezdet:T", "Vég:T", "Időtartam (nap):Q", "heti km:Q", "improvement:Q"],
                )
                .properties(height=max(260, min(760, 26 * len(chart))))
            )
            st.altair_chart(chart_view, width="stretch")
            st.dataframe(timeline[["Szín ikon", "Blokk", "Típus", "Kezdet", "Vég", "Időtartam (nap)"]], width="stretch", hide_index=True)

    st.subheader("Training load")
    volume = training_block_volume_chart(data.training_blocks)
    if volume.empty:
        st.info("Nincs megjeleníthető training load adat.")
    else:
        st.bar_chart(volume.set_index("Blokk")["heti km"])

    composition = training_block_composition_table(data.training_blocks)
    if not composition.empty:
        st.dataframe(composition, width="stretch", hide_index=True)

    st.subheader("Improvement")
    improvement = training_block_improvement_chart(data.training_blocks)
    if improvement.empty:
        st.info("Nincs megjeleníthető improvement adat.")
    else:
        st.bar_chart(improvement.set_index("Blokk")["improvement"])
    improvement_table = training_block_improvement_table(data.training_blocks)
    if not improvement_table.empty:
        st.dataframe(improvement_table, width="stretch", hide_index=True)

    options = timeline["Blokk"].tolist() if not timeline.empty and "Blokk" in timeline else []
    if options:
        selected = str(st.selectbox("Blokk kiválasztása", options))
        detail_cols = st.columns(4)
        for index, (label, value) in enumerate(training_block_detail_values(data.training_blocks, selected).items()):
            detail_cols[index % 4].metric(label, value)

    with st.expander("Edzésblokkok részletei"):
        detail = timeline.drop(columns=["Szín"], errors="ignore")
        if detail.empty:
            st.info("Nincs megjeleníthető részlet.")
        else:
            st.dataframe(detail, width="stretch", hide_index=True)
