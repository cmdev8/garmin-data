from __future__ import annotations

from garmin_runner.dashboard.charts import line_chart, prepare_weekly_chart_data
from garmin_runner.dashboard.components import show_warnings
from garmin_runner.dashboard.visualizations import (
    diminishing_returns_values,
    plan_adjusted_prediction_table,
    plan_reason_labels,
    plan_schedule_table,
    plan_summary,
    selected_candidate_row,
)


def render(data, config=None) -> None:
    import streamlit as st

    show_warnings(data.warnings)
    st.subheader("Training load")
    weekly = prepare_weekly_chart_data(data.weekly_features)
    line_chart(weekly, "date", "load_7d")
    st.subheader("Kiválasztott terv")
    summary = plan_summary(data.next_week_plan)
    cols = st.columns(6)
    for column, (label, value) in zip(cols, summary.items()):
        column.metric(label, value)

    workout_schedule = plan_schedule_table(data.next_week_plan, include_rest=False)
    full_schedule = plan_schedule_table(data.next_week_plan)
    if workout_schedule.empty:
        st.info("Nincs megjeleníthető edzésterv.")
    else:
        st.markdown("**Edzésnapok**")
        st.dataframe(workout_schedule, width="stretch", hide_index=True)
        if len(full_schedule) > len(workout_schedule):
            with st.expander("Teljes naptár pihenőnapokkal"):
                st.dataframe(full_schedule, width="stretch", hide_index=True)

    why_selected = plan_reason_labels(data.next_week_plan.get("why_selected") or [])
    if why_selected:
        st.markdown("**Miért ezt választotta az optimalizáló?**")
        for reason in why_selected:
            st.markdown(f"- {reason}")

    plan_warnings = data.next_week_plan.get("warnings") or []
    show_warnings([str(warning) for warning in plan_warnings])

    st.subheader("Várható improvement")
    candidate = selected_candidate_row(data.candidate_plans, data.next_week_plan)
    factor = data.ml_result.get("predictions", {}).get("diminishing_returns_factor")
    st.metric("diminishing returns factor", diminishing_returns_values(data.ml_result)["diminishing returns factor"])
    st.caption("A magasabb edzettség történetileg kisebb improvementet adhat ugyanarra az edzésingerre; ez csak a pozitív becslést csökkenti.")
    benefits = plan_adjusted_prediction_table(data.predictions, candidate, data.next_week_plan.get("horizon_days"), factor)
    if benefits.empty:
        st.info("Nincs megjeleníthető várható előny.")
    else:
        st.dataframe(benefits, width="stretch", hide_index=True)
