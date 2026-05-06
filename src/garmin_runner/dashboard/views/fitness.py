from __future__ import annotations


def render(data, config=None) -> None:
    import streamlit as st

    if data.fitness_state.empty:
        st.info("Nincs elérhető fitness/fatigue adat.")
        return
    chart = data.fitness_state.copy()
    chart["date"] = chart["date"].astype(str)
    for column in ["fitness", "fatigue", "form"]:
        if column in chart:
            labels = {"fitness": "fitness", "fatigue": "fatigue", "form": "form"}
            st.subheader(labels.get(column, column))
            st.line_chart(chart.set_index("date")[column])
