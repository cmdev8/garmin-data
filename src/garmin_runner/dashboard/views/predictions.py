from __future__ import annotations

from typing import Any

from garmin_runner.dashboard.visualizations import (
    candidate_plan_table,
    candidate_row_by_plan_id,
    candidate_summary_values,
    diminishing_returns_values,
    hr_zone_table,
    pace_zone_table,
    plan_adjusted_prediction_table,
    plan_time_savings_chart,
    prediction_plan_default_index,
    prediction_plan_options,
    race_prediction_table,
    rejection_summary_table,
    scoring_component_values,
    top_plan_ids,
)


def render(data, config=None) -> None:
    import streamlit as st

    st.subheader("Prediction és edzésterv")
    selected_plan_id = data.next_week_plan.get("selected_plan_id")
    options = prediction_plan_options(data.candidate_plans, selected_plan_id)
    if not options:
        st.info("Nincs választható jelölt terv, ezért csak a baseline prediction látható.")
        race_table = race_prediction_table(data.predictions)
        if race_table.empty:
            st.info("Nincs megjeleníthető verseny prediction.")
        else:
            st.dataframe(race_table, width="stretch", hide_index=True)
    else:
        default_index = prediction_plan_default_index(options, selected_plan_id)
        chosen_plan_id = st.selectbox("Terv kiválasztása", options, index=default_index)
        candidate_row = candidate_row_by_plan_id(data.candidate_plans, chosen_plan_id)
        if "is_valid" in data.candidate_plans and not bool(data.candidate_plans["is_valid"].eq(True).any()):
            st.warning("Nincs érvényes jelölt terv, ezért minden választható tervet megjelenítek.")
        if candidate_row.get("is_valid") is False:
            st.warning("A kiválasztott terv nem érvényes; a becslés nem mutat pozitív javulást.")

        cols = st.columns(4)
        for index, (label, value) in enumerate(_compact_candidate_metrics(candidate_row).items()):
            cols[index % 4].metric(label, value)

        factor = data.ml_result.get("predictions", {}).get("diminishing_returns_factor")
        st.metric("diminishing returns factor", diminishing_returns_values(data.ml_result)["diminishing returns factor"])
        st.caption("A tervhatás becslés az optimizer adaptáció/fatigue/overload risk score-jaiból készül; nem új model run.")
        st.caption("Rövid, például 7 napos horizontnál a becsült hatás erősen korlátozott, és hosszabb távokon kisebb.")
        st.caption("A diminishing returns factor az alapján csökkenti az optimista predictiont, hogy a feltöltött előzményekben hogyan változott az improvement magasabb edzettségnél.")
        adjusted = plan_adjusted_prediction_table(data.predictions, candidate_row, data.next_week_plan.get("horizon_days"), factor)
        if adjusted.empty:
            st.info("Nincs megjeleníthető verseny prediction.")
        else:
            st.dataframe(adjusted, width="stretch", hide_index=True)
            savings = plan_time_savings_chart(data.predictions, candidate_row, data.next_week_plan.get("horizon_days"), factor)
            if not savings.empty:
                st.bar_chart(savings.set_index("Táv")["időnyereség (mp)"])

        st.subheader("Scoring components")
        component_cols = st.columns(3)
        for index, (label, value) in enumerate(scoring_component_values(candidate_row).items()):
            component_cols[index % 3].metric(label, value)

        _render_optimizer_diagnostics(data)

    st.subheader("Pulzuszónák")
    hr_table = hr_zone_table(data.predictions)
    if hr_table.empty:
        st.info("Nincs megjeleníthető pulzuszóna.")
    else:
        st.dataframe(hr_table, width="stretch", hide_index=True)

    st.subheader("Tempózónák")
    pace_table = pace_zone_table(data.predictions)
    if pace_table.empty:
        st.info("Nincs megjeleníthető tempózóna.")
    else:
        st.dataframe(pace_table, width="stretch", hide_index=True)


def _compact_candidate_metrics(candidate_row: dict[str, Any]) -> dict[str, str]:
    summary = candidate_summary_values(candidate_row)
    labels = ["Terv", "Család", "score", "Adaptáció", "fatigue", "overload risk", "Történeti driver match", "minőségi napok"]
    return {label: summary[label] for label in labels if label in summary}


def _render_optimizer_diagnostics(data: Any) -> None:
    import streamlit as st

    st.subheader("Optimizer diagnosztika")
    debug = data.plan_optimizer_debug
    cols = st.columns(3)
    cols[0].metric("Generált jelöltek", debug.get("generated_candidate_count", "n/a"))
    cols[1].metric("Érvényes jelöltek", debug.get("valid_candidate_count", "n/a"))
    cols[2].metric("Elutasított jelöltek", debug.get("rejected_candidate_count", "n/a"))

    top_ids = top_plan_ids(debug)
    if top_ids:
        st.markdown("**Legjobb jelölt tervek**")
        st.markdown(", ".join(top_ids))

    driver = debug.get("historical_improvement_driver")
    if isinstance(driver, dict) and driver.get("category") and driver.get("category") != "neutral":
        st.markdown("**Történeti improvement driver**")
        st.caption(
            f"{driver.get('label', 'n/a')} | kategória: {driver.get('category', 'n/a')} | forrás: {driver.get('source_distance_range', 'n/a')}"
        )

    rejection_table = rejection_summary_table(debug)
    if not rejection_table.empty:
        with st.expander("Elutasítási összefoglaló"):
            st.dataframe(rejection_table, width="stretch", hide_index=True)

    fallback_warning = debug.get("fallback_warning")
    if fallback_warning:
        st.warning(str(fallback_warning))

    candidate_table = candidate_plan_table(data.candidate_plans)
    if not candidate_table.empty:
        with st.expander("Jelölt tervek részletei"):
            st.dataframe(candidate_table, width="stretch", hide_index=True)
