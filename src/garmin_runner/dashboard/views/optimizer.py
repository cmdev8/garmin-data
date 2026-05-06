from __future__ import annotations

from garmin_runner.dashboard.visualizations import (
    candidate_plan_table,
    plan_adjusted_prediction_table,
    plan_schedule_table,
    plan_summary,
    rejection_summary_table,
    scoring_component_values,
    selected_candidate_row,
    top_candidate_scores,
    top_plan_ids,
)


def render(data, config=None) -> None:
    import streamlit as st

    st.subheader("Kiválasztott terv")
    summary = plan_summary(data.next_week_plan)
    cols = st.columns(6)
    for column, (label, value) in zip(cols, summary.items()):
        column.metric(label, value)

    schedule = plan_schedule_table(data.next_week_plan, include_rest=False)
    if schedule.empty:
        st.info("Nincs megjeleníthető edzésterv.")
    else:
        st.dataframe(schedule, width="stretch", hide_index=True)

    selected_candidate = selected_candidate_row(data.candidate_plans, data.next_week_plan)
    st.subheader("Scoring components")
    component_cols = st.columns(3)
    for index, (label, value) in enumerate(scoring_component_values(selected_candidate).items()):
        component_cols[index % 3].metric(label, value)

    st.subheader("Várható előnyök")
    benefits = plan_adjusted_prediction_table(data.predictions, selected_candidate, data.next_week_plan.get("horizon_days"))
    if benefits.empty:
        st.info("Nincs megjeleníthető várható előny.")
    else:
        st.dataframe(benefits, width="stretch", hide_index=True)

    st.subheader("Jelölt tervek")
    top_scores = top_candidate_scores(data.candidate_plans)
    if not top_scores.empty:
        st.bar_chart(top_scores.set_index("Terv")["score"])

    candidate_table = candidate_plan_table(data.candidate_plans)
    if candidate_table.empty:
        st.info("Nincs megjeleníthető jelölt terv.")
    else:
        st.dataframe(candidate_table, width="stretch", hide_index=True)

    st.subheader("Optimalizáló diagnosztika")
    debug = data.plan_optimizer_debug
    cols = st.columns(3)
    cols[0].metric("Generált jelöltek", debug.get("generated_candidate_count", "n/a"))
    cols[1].metric("Érvényes jelöltek", debug.get("valid_candidate_count", "n/a"))
    cols[2].metric("Elutasított jelöltek", debug.get("rejected_candidate_count", "n/a"))

    rejection_table = rejection_summary_table(debug)
    if rejection_table.empty:
        st.info("Nincs elutasítási összefoglaló.")
    else:
        st.dataframe(rejection_table, width="stretch", hide_index=True)

    top_ids = top_plan_ids(debug)
    if top_ids:
        st.markdown("**Legjobb jelölt tervek**")
        for index, plan_id in enumerate(top_ids, 1):
            st.markdown(f"{index}. {plan_id}")

    fallback_warning = debug.get("fallback_warning")
    if fallback_warning:
        st.warning(str(fallback_warning))
