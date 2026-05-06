from __future__ import annotations

from garmin_runner.dashboard.visualizations import (
    feature_importance_table,
    feature_names,
    ml_metric_values,
    ml_prediction_values,
    temporal_driver_table,
    temporal_ml_values,
)


def render(data, config=None) -> None:
    import streamlit as st

    st.subheader("Per-run ML model")
    metrics = ml_metric_values(data.ml_result)
    cols = st.columns(5)
    for column, (label, value) in zip(cols, metrics.items()):
        column.metric(label, value)

    st.subheader("ML prediction jelek")
    prediction_cols = st.columns(3)
    for column, (label, value) in zip(prediction_cols, ml_prediction_values(data.ml_result).items()):
        column.metric(label, value)

    warnings = data.ml_result.get("warnings") or []
    for warning in warnings:
        st.warning(str(warning))

    st.subheader("Temporal ML")
    temporal_cols = st.columns(5)
    for column, (label, value) in zip(temporal_cols, temporal_ml_values(data.ml_result).items()):
        column.metric(label, value)

    positive = temporal_driver_table(data.ml_result, "positive")
    negative = temporal_driver_table(data.ml_result, "negative")
    if not positive.empty or not negative.empty:
        left, right = st.columns(2)
        with left:
            st.markdown("**Pozitív temporal driverek**")
            if positive.empty:
                st.info("Nincs stabil pozitív temporal driver.")
            else:
                st.dataframe(positive, width="stretch", hide_index=True)
        with right:
            st.markdown("**Negatív temporal driverek**")
            if negative.empty:
                st.info("Nincs stabil negatív temporal driver.")
            else:
                st.dataframe(negative, width="stretch", hide_index=True)

    st.subheader("feature importance")
    importance = feature_importance_table(data.ml_result)
    if importance.empty:
        st.info("Ehhez a futáshoz nincs stabil feature importance becslés.")
    else:
        st.bar_chart(importance.set_index("feature")["feature importance"])
        st.dataframe(importance, width="stretch", hide_index=True)

    names = feature_names(data.ml_result)
    if names:
        with st.expander("Felhasznált feature-ök"):
            for feature in names:
                st.markdown(f"- `{feature}`")
