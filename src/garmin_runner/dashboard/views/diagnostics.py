from __future__ import annotations

from garmin_runner.dashboard.components import metric_grid
from garmin_runner.dashboard.visualizations import (
    diagnostic_actions_table,
    diagnostic_reasons_table,
    diagnostic_signal_chart,
    diagnostic_summary_values,
    performance_change_model_notes,
)


def render(data, config=None) -> None:
    import streamlit as st

    st.subheader("Diagnosztika")
    st.caption(
        "Coach-jellegű összefoglaló arról, hogy a feltöltött adatok alapján mi segítheti vagy ronthatja a teljesítményváltozást."
    )

    metric_grid(diagnostic_summary_values(data), columns=4)

    st.subheader("Miért változik a teljesítmény?")
    chart = diagnostic_signal_chart(data)
    if chart.empty:
        st.info("Nincs elég jel a teljesítményváltozás okainak rangsorolásához.")
    else:
        st.bar_chart(chart.set_index("Tényező")["score"])

    reasons = diagnostic_reasons_table(data)
    if reasons.empty:
        st.info("Nincs megjeleníthető diagnosztikai ok.")
    else:
        st.dataframe(reasons, width="stretch", hide_index=True)

    st.subheader("Mit érdemes tenni?")
    actions = diagnostic_actions_table(data)
    if actions.empty:
        st.info("Nincs konkrét javaslat.")
    else:
        st.dataframe(actions, width="stretch", hide_index=True)

    st.subheader("Bizonyíték és korlátok")
    warnings = list(data.warnings or [])
    ml_warnings = data.ml_result.get("warnings") or []
    for warning in warnings + [str(value) for value in ml_warnings]:
        st.warning(str(warning))

    models = data.performance_change.get("models") or {}
    if models:
        rows = []
        for distance, model in models.items():
            notes = performance_change_model_notes(data.performance_change, str(distance))
            rows.append(
                {
                    "Táv": distance,
                    "model": notes.get("model", "n/a"),
                    "confidence": notes.get("confidence", "n/a"),
                    "Terep": notes.get("Terep", ""),
                    "Fallback": notes.get("Fallback", ""),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("Nincs kontrollált teljesítménymodell a bizonyítékokhoz.")
