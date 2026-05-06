from __future__ import annotations

from typing import Any, cast
from types import SimpleNamespace

import pandas as pd

from garmin_runner.dashboard.visualizations import (
    activity_advanced_fit_table,
    activity_option_labels,
    activity_quality_notes,
    activity_record_chart_table,
    activity_segment_summary,
    activity_segment_table,
    candidate_plan_table,
    candidate_row_by_plan_id,
    candidate_summary_values,
    diagnostic_actions_table,
    diagnostic_reasons_table,
    diagnostic_signal_chart,
    diagnostic_summary_values,
    diminishing_returns_history_chart,
    diminishing_returns_history_context_chart,
    diminishing_returns_history_table,
    diminishing_returns_values,
    feature_importance_table,
    hr_zone_table,
    latest_weekly_statistics_table,
    ml_metric_values,
    ml_prediction_values,
    pace_zone_table,
    performance_change_chart_table,
    performance_change_distance_options,
    performance_change_importance_table,
    performance_change_metric_values,
    performance_change_model_notes,
    performance_change_observation_table,
    performance_change_summary_table,
    plan_schedule_table,
    plan_adjusted_prediction_table,
    plan_time_savings_chart,
    prediction_plan_default_index,
    prediction_plan_options,
    race_prediction_table,
    race_confidence_chart,
    rejection_summary_table,
    scoring_component_values,
    selected_activity_metrics,
    temporal_driver_table,
    temporal_ml_values,
    top_candidate_scores,
    training_block_composition_table,
    training_block_detail_values,
    training_block_improvement_chart,
    training_block_improvement_table,
    training_block_summary_values,
    training_block_timeline_table,
    training_block_volume_chart,
)


def test_activity_option_labels_and_metrics_format_selected_activity():
    activities = pd.DataFrame(
        [
            {
                "activity_id": "a1",
                "date": "2026-05-06",
                "distance_km": 10.0,
                "moving_time_min": 50.0,
                "elapsed_time_min": 55.0,
                "avg_pace_s_per_km": 300.0,
                "best_rolling_1km_pace": 250.0,
                "avg_hr": 145,
                "max_hr": 170,
                "min_hr": 90,
                "avg_cadence": 172,
                "elevation_gain": 120,
                "elevation_gain_per_km": 12,
                "avg_power_w": 240,
                "total_training_effect": 3.2,
                "run_walk_type": "not_run_walk",
                "walk_fraction": 0.0,
                "run_walk_fatigue_multiplier": 1.0,
                "data_quality_score": 0.75,
            }
        ]
    )

    labels = activity_option_labels(activities)
    metrics = selected_activity_metrics(activities.iloc[0].to_dict())

    assert "2026-05-06 | 10.0 km | 5:00/km | a1" == labels["a1"]
    assert metrics["Alapadatok"]["Átlagtempó"] == "5:00/km"
    assert metrics["Pulzus és intenzitás"]["Átlagpulzus"] == "145 bpm"
    assert metrics["Mechanika és környezet"]["Szintemelkedés/km"] == "12.0 m/km"
    assert metrics["Futás-séta és minőség"]["data quality"] == "75%"


def test_activity_record_charts_tolerate_missing_optional_signals():
    records = pd.DataFrame(
        [
            {"activity_id": "a1", "elapsed_s": 0, "pace_s_per_km": 300, "heart_rate_bpm": 140},
            {"activity_id": "a1", "elapsed_s": 60, "pace_s_per_km": 295, "heart_rate_bpm": 145},
            {"activity_id": "a2", "elapsed_s": 0, "pace_s_per_km": 400},
        ]
    )

    pace_hr = activity_record_chart_table(records, "a1", "pace_hr")
    mechanics = activity_record_chart_table(records, "a1", "mechanics")

    assert list(pace_hr.columns) == ["perc", "tempó (perc/km)", "pulzus (bpm)"]
    assert pace_hr.loc[0, "tempó (perc/km)"] == 5.0
    assert mechanics.empty
    assert activity_record_chart_table(pd.DataFrame(), "a1", "pace_hr").empty


def test_activity_segment_and_advanced_fit_helpers_filter_selected_activity():
    segments = pd.DataFrame(
        [
            {"activity_id": "a1", "segment_id": 1, "state": "easy_running", "duration_s": 300, "distance_m": 1000, "avg_pace_s_per_km": 300},
            {"activity_id": "a1", "segment_id": 2, "state": "walking", "duration_s": 60, "distance_m": 80},
            {"activity_id": "a2", "segment_id": 1, "state": "hard_running", "duration_s": 120, "distance_m": 500},
        ]
    )
    activity = {
        "total_work_kj": 600,
        "vertical_ratio": 8.4,
        "hrv_rmssd_ms": None,
        "data_quality_score": 0.5,
    }
    records = pd.DataFrame([{"activity_id": "a1", "elapsed_s": 0, "heart_rate_bpm": 140}])

    segment_table = activity_segment_table(segments, "a1")
    segment_summary = activity_segment_summary(segments, "a1")
    advanced = activity_advanced_fit_table(activity)
    notes = activity_quality_notes(activity, records, "a1")

    assert len(segment_table) == 2
    assert set(segment_summary["Típus"]) == {"laza futás", "séta"}
    assert "Munka" in set(advanced["Mutató"])
    assert any("data quality: 50%" in note for note in notes)


def test_prediction_tables_tolerate_missing_zone_data():
    predictions = {
        "race_predictions": {
            "5km": {
                "pace_s_per_km": 300,
                "estimated_time_s": 1500,
                "confidence": 0.7,
                "explanation": "Stabil becslés.",
            }
        }
    }

    race = race_prediction_table(predictions)
    confidence = race_confidence_chart(predictions)
    hr = hr_zone_table(predictions)
    pace = pace_zone_table(predictions)

    assert race.loc[0, "tempó"] == "5:00/km"
    assert race.loc[0, "Becsült idő"] == "25:00"
    assert race.loc[0, "confidence"] == "70%"
    assert confidence.loc[0, "confidence"] == 70.0
    assert hr.empty
    assert pace.empty


def test_prediction_tables_tolerate_empty_predictions():
    assert race_prediction_table({}).empty
    assert hr_zone_table({}).empty
    assert pace_zone_table({}).empty


def test_prediction_plan_options_default_to_selected_plan():
    candidates = pd.DataFrame(
        [
            {"plan_id": "a", "is_valid": True},
            {"plan_id": "b", "is_valid": True},
        ]
    )

    options = prediction_plan_options(candidates, "b")

    assert options == ["a", "b"]
    assert prediction_plan_default_index(options, "b") == 1


def test_prediction_plan_options_put_valid_candidates_first():
    candidates = pd.DataFrame(
        [
            {"plan_id": "invalid", "is_valid": False},
            {"plan_id": "valid", "is_valid": True},
        ]
    )

    assert prediction_plan_options(candidates, None) == ["valid", "invalid"]


def test_candidate_row_by_plan_id_returns_matching_row():
    candidates = pd.DataFrame([{"plan_id": "selected", "total_score": 0.7}])

    row = candidate_row_by_plan_id(candidates, "selected")

    assert row["plan_id"] == "selected"
    assert row["total_score"] == 0.7


def test_candidate_summary_and_scoring_components_use_percentages():
    row = {
        "plan_id": "selected",
        "family": "base",
        "total_score": 0.7,
        "expected_adaptation": 0.8,
        "expected_fatigue": 0.2,
        "overload_risk": 0.1,
        "race_specificity": 0.6,
        "missing_stimulus_score": 0.5,
        "constraint_penalty": 0.0,
    }

    summary = candidate_summary_values(row)
    components = scoring_component_values(row)

    assert summary["score"] == "0.700"
    assert summary["Adaptáció"] == "80%"
    assert summary["fatigue"] == "20%"
    assert summary["overload risk"] == "10%"
    assert components["expected adaptation"] == "80%"
    assert components["Versenyspecifikusság"] == "60%"
    assert components["Korlátbüntetés"] == "0%"


def test_plan_adjusted_predictions_missing_candidate_columns_are_baseline_only():
    predictions = {"race_predictions": {"5km": {"pace_s_per_km": 300, "estimated_time_s": 1500, "confidence": 0.6}}}

    table = plan_adjusted_prediction_table(predictions, {"plan_id": "minimal"})
    chart = plan_time_savings_chart(predictions, {"plan_id": "minimal"})

    assert table.loc[0, "baseline tempó"] == "5:00/km"
    assert table.loc[0, "terv utáni tempó"] == "5:00/km"
    assert table.loc[0, "improvement"] == "+0.00%"
    assert chart.loc[0, "időnyereség (mp)"] == 0.0


def test_plan_adjusted_predictions_show_horizon_scaled_improvement_for_adaptive_plan():
    predictions = {
        "race_predictions": {
            "5km": {"distance_km": 5.0, "pace_s_per_km": 300, "estimated_time_s": 1500, "confidence": 0.6},
            "42.2km": {"distance_km": 42.2, "pace_s_per_km": 360, "estimated_time_s": 15192, "confidence": 0.5},
        }
    }
    candidate = {
        "expected_adaptation": 0.9,
        "missing_stimulus_score": 0.8,
        "race_specificity": 0.8,
        "expected_fatigue": 0.1,
        "overload_risk": 0.1,
        "constraint_penalty": 0.0,
        "is_valid": True,
    }

    seven_day_table = plan_adjusted_prediction_table(predictions, candidate, horizon_days=7)
    full_horizon_table = plan_adjusted_prediction_table(predictions, candidate, horizon_days=28)
    seven_day_chart = plan_time_savings_chart(predictions, candidate, horizon_days=7)
    full_horizon_chart = plan_time_savings_chart(predictions, candidate, horizon_days=28)

    assert str(seven_day_table.at[0, "terv utáni idő"]) != str(seven_day_table.at[0, "baseline idő"])
    assert str(seven_day_table.at[0, "improvement"]) == "+0.41%"
    assert str(full_horizon_table.at[0, "improvement"]) == "+1.63%"
    assert str(seven_day_table.at[0, "improvement"]) != str(full_horizon_table.at[0, "improvement"])
    assert float(cast(Any, seven_day_chart.at[0, "időnyereség (mp)"])) > 0
    assert float(cast(Any, seven_day_chart.at[0, "időnyereség (mp)"])) < float(cast(Any, full_horizon_chart.at[0, "időnyereség (mp)"]))
    assert float(cast(Any, seven_day_chart.at[1, "időnyereség (mp)"])) < float(cast(Any, full_horizon_chart.at[1, "időnyereség (mp)"]))


def test_diminishing_returns_factor_reduces_positive_plan_benefit():
    predictions = {"race_predictions": {"5km": {"distance_km": 5.0, "pace_s_per_km": 300, "estimated_time_s": 1500, "confidence": 0.6}}}
    candidate = {
        "expected_adaptation": 0.9,
        "missing_stimulus_score": 0.8,
        "race_specificity": 0.8,
        "expected_fatigue": 0.1,
        "overload_risk": 0.1,
        "constraint_penalty": 0.0,
        "is_valid": True,
    }

    full_chart = plan_time_savings_chart(predictions, candidate, horizon_days=7)
    reduced_chart = plan_time_savings_chart(predictions, candidate, horizon_days=7, diminishing_returns_factor=0.5)
    reduced_table = plan_adjusted_prediction_table(predictions, candidate, horizon_days=7, diminishing_returns_factor=0.5)

    assert float(cast(Any, reduced_chart.at[0, "időnyereség (mp)"])) < float(cast(Any, full_chart.at[0, "időnyereség (mp)"]))
    assert reduced_table.loc[0, "improvement"] == "+0.26%"


def test_plan_adjusted_predictions_can_worsen_with_high_risk():
    predictions = {"race_predictions": {"5km": {"pace_s_per_km": 300, "estimated_time_s": 1500, "confidence": 0.6}}}
    candidate = {
        "expected_adaptation": 0.0,
        "missing_stimulus_score": 0.0,
        "race_specificity": 0.0,
        "expected_fatigue": 1.0,
        "overload_risk": 1.0,
        "constraint_penalty": 1.0,
        "is_valid": True,
    }

    table = plan_adjusted_prediction_table(predictions, candidate, horizon_days=7)
    chart = plan_time_savings_chart(predictions, candidate, horizon_days=7)
    reduced_chart = plan_time_savings_chart(predictions, candidate, horizon_days=7, diminishing_returns_factor=0.35)

    assert str(table.at[0, "improvement"]).startswith("-")
    assert float(cast(Any, chart.at[0, "időnyereség (mp)"])) < 0
    assert float(cast(Any, reduced_chart.at[0, "időnyereség (mp)"])) == float(cast(Any, chart.at[0, "időnyereség (mp)"]))


def test_invalid_plan_never_shows_positive_improvement():
    predictions = {"race_predictions": {"5km": {"pace_s_per_km": 300, "estimated_time_s": 1500, "confidence": 0.6}}}
    candidate = {
        "expected_adaptation": 1.0,
        "missing_stimulus_score": 1.0,
        "race_specificity": 1.0,
        "expected_fatigue": 0.0,
        "overload_risk": 0.0,
        "constraint_penalty": 0.0,
        "is_valid": False,
    }

    table = plan_adjusted_prediction_table(predictions, candidate)
    chart = plan_time_savings_chart(predictions, candidate)

    assert table.loc[0, "improvement"] == "+0.00%"
    assert chart.loc[0, "időnyereség (mp)"] == 0.0


def test_plan_adjusted_predictions_tolerate_empty_inputs():
    assert prediction_plan_options(pd.DataFrame(), "selected") == []
    assert candidate_row_by_plan_id(pd.DataFrame(), "selected") == {}
    assert plan_adjusted_prediction_table({}, {}).empty
    assert plan_time_savings_chart({}, {}).empty


def test_plan_schedule_tolerates_missing_run_walk_prescription():
    plan = {
        "days": [
            {
                "date": "2026-05-06",
                "day": "Wednesday",
                "workout_type": "rest",
                "intensity": "rest",
                "duration_min": 0,
                "distance_km": None,
                "purpose": "Absorb recent training",
            },
            {
                "date": "2026-05-07",
                "day": "Thursday",
                "workout_type": "easy",
                "intensity": "easy",
                "duration_min": 35,
                "distance_km": 5.0,
                "purpose": "Build aerobic base with low stress",
            }
        ]
    }

    table = plan_schedule_table(plan)
    workout_table = plan_schedule_table(plan, include_rest=False)

    assert table.loc[0, "Táv (km)"] == ""
    assert table.loc[0, "Cél"] == "az edzésterhelés feldolgozása"
    assert table.loc[1, "Nap"] == "Csütörtök"
    assert table.loc[1, "futás-séta előírás"] == ""
    assert table.loc[1, "Cél"] == "aerob alap építése alacsony stresszel"
    assert len(workout_table) == 1


def test_optimizer_visualization_helpers_tolerate_empty_data():
    assert candidate_plan_table(pd.DataFrame()).empty
    assert top_candidate_scores(pd.DataFrame()).empty
    assert rejection_summary_table({}).empty


def test_optimizer_visualization_helpers_summarize_candidates():
    candidates = pd.DataFrame(
        [
            {"plan_id": "a", "family": "base", "total_score": 0.4, "is_valid": True, "total_distance_km": 20, "overload_risk": 0.25},
            {"plan_id": "b", "family": "recovery", "total_score": 0.8, "is_valid": True, "total_distance_km": 12, "overload_risk": 0.1},
            {"plan_id": "c", "family": "vo2", "total_score": -1.0, "is_valid": False, "rejection_reasons": "fatigue"},
        ]
    )

    table = candidate_plan_table(candidates)
    scores = top_candidate_scores(candidates)
    rejections = rejection_summary_table({"rejection_summary": {"fatigue": 1}})

    assert "Terv azonosító" in table.columns
    assert table.loc[0, "score"] == 0.4
    assert table.loc[0, "overload risk"] == "25%"
    assert scores.iloc[0]["Terv"] == "b"
    assert rejections.loc[0, "Ok"] == "fatigue"


def test_ml_and_feature_values_use_percentages():
    ml = {
        "confidence": 0.4,
        "row_count": 10,
        "model_type": "DummyRegressor",
        "metrics": {"mae": 1.2, "validation_rows": 3, "diminishing_returns_beta": -0.01, "diminishing_returns_observations": 8},
        "predictions": {"performance_index_7d": 55.0, "expected_adaptation": 0.35, "overload_risk_modifier": 0.65, "diminishing_returns_factor": 0.8},
        "feature_importance": {"load_7d": 0.2},
    }

    assert ml_metric_values(ml)["confidence"] == "40%"
    assert ml_prediction_values(ml)["expected adaptation"] == "35%"
    assert ml_prediction_values(ml)["overload risk modifier"] == "65%"
    assert ml_prediction_values(ml)["diminishing returns factor"] == "80%"
    assert diminishing_returns_values(ml)["diminishing returns factor"] == "80%"
    assert diminishing_returns_values(ml)["Megfigyelések"] == "8"
    assert diminishing_returns_values(ml)["Max. korrekció"] == "35%"
    assert diminishing_returns_values(ml)["Állapot"] == "n/a"
    assert feature_importance_table(ml).loc[0, "feature importance"] == 20.0


def test_temporal_ml_helpers_format_status_and_drivers():
    ml = {
        "metrics": {
            "temporal_model_used": True,
            "lookback_days": [7, 14, 28],
            "validation_strategy": "TimeSeriesSplit",
            "fold_count": 3,
            "sequence_row_count": 24,
        },
        "positive_temporal_drivers": [{"feature": "fatigue_adjusted_load_7d", "coefficient": 0.12345, "strength": 0.4}],
        "negative_temporal_drivers": [{"feature": "overload_risk", "coefficient": -0.2, "strength": 0.2}],
    }

    values = temporal_ml_values(ml)
    positive = temporal_driver_table(ml, "positive")
    negative = temporal_driver_table(ml, "negative")

    assert values["Temporal model"] == "igen"
    assert values["Validation"] == "TimeSeriesSplit"
    assert positive.loc[0, "feature"] == "fatigue_adjusted_load_7d"
    assert negative.loc[0, "Koeficiens"] == "-0.2000"


def test_diminishing_returns_history_helpers_format_monthly_values():
    ml = {
        "metrics": {
            "diminishing_returns_history": [
                {
                    "month": "2026-01",
                    "factor": 0.8,
                    "observations": 9,
                    "beta": -0.0123,
                    "reference_response": 0.7,
                    "current_response": 0.56,
                    "raw_ratio": 0.8,
                    "latest_performance_index": 52.5,
                    "reason": "learned_from_history",
                }
            ]
        }
    }

    table = diminishing_returns_history_table(ml)
    chart = diminishing_returns_history_chart(ml)
    context = diminishing_returns_history_context_chart(ml)

    assert table.loc[0, "diminishing returns factor"] == "80%"
    assert table.loc[0, "performance index"] == "52.5"
    assert table.loc[0, "response ratio"] == "80%"
    assert table.loc[0, "Megfigyelések"] == 9
    assert table.loc[0, "Állapot"] == "Történetből becsült"
    assert chart.loc[0, "diminishing returns factor (%)"] == 80.0
    assert context.loc[0, "performance index"] == 52.5
    assert context.loc[0, "response ratio (%)"] == 80.0


def test_diminishing_returns_history_helpers_tolerate_empty_data():
    assert diminishing_returns_history_table({}).empty
    assert diminishing_returns_history_chart({}).empty
    assert diminishing_returns_history_context_chart({}).empty


def test_latest_weekly_statistics_formats_fractions_as_percentages():
    table = latest_weekly_statistics_table(
        pd.DataFrame(
            [
                {
                    "date": "2026-05-06",
                    "easy_fraction": 0.75,
                    "moderate_fraction": 0.15,
                    "hard_fraction": 0.10,
                    "acute_chronic_workload_ratio": 1.2,
                    "fatigue_adjusted_load_7d": 90.0,
                    "fatigue_adjusted_acute_chronic_workload_ratio": 1.05,
                }
            ]
        )
    )

    values = dict(zip(table["Mutató"], table["Érték"]))

    assert values["laza arány"] == "75%"
    assert values["Közepes arány"] == "15%"
    assert values["minőségi/kemény arány"] == "10%"
    assert values["ACWR"] == 1.2
    assert values["fatigue-adjusted training load"] == 90.0
    assert values["Korrigált ACWR"] == 1.05


def test_percentage_helpers_show_missing_values_as_na():
    assert candidate_summary_values({})["Adaptáció"] == "n/a"
    assert scoring_component_values({})["expected adaptation"] == "n/a"
    assert ml_metric_values({})["confidence"] == "n/a"


def test_performance_change_visualization_helpers_tolerate_empty_data():
    empty: dict[str, Any] = {"summary": [], "observations": [], "models": {}}

    assert performance_change_summary_table(empty).empty
    assert performance_change_distance_options(empty) == []
    assert performance_change_chart_table(empty, "5 km").empty
    assert performance_change_observation_table(empty, "5 km").empty
    assert performance_change_importance_table(empty, "5 km").empty


def _diagnostic_data(**overrides: Any) -> SimpleNamespace:
    data = {
        "performance_change": {"summary": [], "observations": [], "models": {}, "warnings": []},
        "training_blocks": pd.DataFrame(),
        "fitness_state": pd.DataFrame(),
        "weekly_features": pd.DataFrame(),
        "activity_features": pd.DataFrame(),
        "ml_result": {"confidence": 0.4, "feature_importance": {}, "warnings": []},
        "warnings": [],
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_diagnostic_helpers_tolerate_empty_data():
    data = _diagnostic_data()

    assert diagnostic_summary_values(data)["Állapot"] == "Kevés adat"
    assert diagnostic_reasons_table(data).empty
    assert not diagnostic_actions_table(data).empty
    assert diagnostic_signal_chart(data).empty


def test_diagnostic_helpers_explain_improvement_and_ml_factor():
    data = _diagnostic_data(
        performance_change={
            "summary": [
                {
                    "distance_range": "5 km",
                    "adjusted_change_s_per_km": -12.0,
                    "confidence": 0.7,
                    "primary_controlled_factors": "fatigue",
                }
            ],
            "observations": [],
            "models": {},
            "warnings": [],
        },
        ml_result={"confidence": 0.6, "feature_importance": {"load_7d": 0.4}, "warnings": []},
    )

    summary = diagnostic_summary_values(data)
    reasons = diagnostic_reasons_table(data)
    chart = diagnostic_signal_chart(data)

    assert summary["Állapot"] == "Javulás"
    assert "5 km" in summary["Fő javító tényező"]
    assert "ML feature importance" in set(reasons["Tényező"])
    assert not chart.empty


def test_diagnostic_helpers_explain_worsening_fatigue_and_run_walk_risk():
    data = _diagnostic_data(
        performance_change={
            "summary": [{"distance_range": "10 km", "adjusted_change_s_per_km": 15.0, "confidence": 0.65}],
            "observations": [],
            "models": {},
            "warnings": [],
        },
        fitness_state=pd.DataFrame([{"fatigue": 50.0, "form": -8.0, "overload_risk": 0.5}]),
        weekly_features=pd.DataFrame([{"hard_fraction": 0.3, "load_7d": 300.0, "fatigue_adjusted_load_7d": 300.0}]),
        activity_features=pd.DataFrame([{"forced_walk_score": 0.7, "run_walk_fatigue_multiplier": 1.0, "planned_run_walk_score": 0.0, "data_quality_score": 0.5}]),
    )

    reasons = diagnostic_reasons_table(data)
    actions = diagnostic_actions_table(data)

    assert diagnostic_summary_values(data)["Állapot"] == "Romlás"
    assert "fatigue / form" in set(reasons["Tényező"])
    assert "Kényszerű futás-séta" in set(reasons["Tényező"])
    assert "Adatminőség" in set(reasons["Tényező"])
    assert actions.iloc[0]["Prioritás"] == 1


def test_diagnostic_helpers_show_planned_run_walk_as_positive_signal():
    data = _diagnostic_data(
        performance_change={"summary": [{"distance_range": "5 km", "adjusted_change_s_per_km": -5.0, "confidence": 0.6}], "observations": [], "models": {}, "warnings": []},
        activity_features=pd.DataFrame([{"forced_walk_score": 0.0, "run_walk_fatigue_multiplier": 0.86, "planned_run_walk_score": 0.75, "data_quality_score": 1.0}]),
    )

    reasons = diagnostic_reasons_table(data)
    actions = diagnostic_actions_table(data)

    assert "Tervezett futás-séta" in set(reasons["Tényező"])
    assert any("futás-séta" in text for text in actions["Javaslat"])


def test_diagnostic_training_block_signals_are_specific_not_generic():
    data = _diagnostic_data(
        performance_change={
            "summary": [{"distance_range": "10 km", "adjusted_change_s_per_km": 8.0, "confidence": 0.62}],
            "observations": [],
            "models": {},
            "warnings": [],
        },
        training_blocks=pd.DataFrame(
            [
                {
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-21",
                    "block_type": "base",
                    "improvement_score": 1.5,
                    "confidence": 0.6,
                },
                {
                    "start_date": "2026-02-01",
                    "end_date": "2026-02-21",
                    "block_type": "overload",
                    "improvement_score": -2.0,
                    "confidence": 0.7,
                },
            ]
        ),
    )

    summary = diagnostic_summary_values(data)
    reasons = diagnostic_reasons_table(data)

    assert summary["Fő javító tényező"].startswith("Segítő edzésblokk:")
    assert summary["Fő rontó tényező"].startswith("Rontó edzésblokk:")
    assert "Edzésblokk" not in set(reasons["Tényező"])
    assert any("2026-01-01 - 2026-01-21" in evidence for evidence in reasons["Bizonyíték"])


def test_performance_change_visualization_helpers_format_data():
    payload = {
        "summary": [
            {
                "distance_range": "5 km",
                "raw_change_s_per_km": 12.0,
                "adjusted_change_s_per_km": -8.0,
                "percent_change": -0.025,
                "observation_count": 12,
                "model_type": "Ridge kontrollmodell",
                "confidence": 0.62,
                "primary_controlled_factors": "szintemelkedés proxy",
            }
        ],
        "observations": [
            {
                "date": "2026-01-01",
                "distance_range": "5 km",
                "observed_pace_s_per_km": 300.0,
                "adjusted_pace_s_per_km": 295.0,
                "run_walk_type": "forced",
                "elevation_gain_per_km": 12.0,
                "fatigue": 25.0,
                "acute_chronic_workload_ratio": 1.1,
            }
        ],
        "models": {
            "5 km": {
                "model_type": "Ridge kontrollmodell",
                "confidence": 0.62,
                "feature_importance": {"szintemelkedés proxy": 0.7, "fatigue": 0.3},
                "fallback_reason": "",
            }
        },
    }

    summary = performance_change_summary_table(payload)
    metrics = performance_change_metric_values(payload)
    chart = performance_change_chart_table(payload, "5 km")
    observations = performance_change_observation_table(payload, "5 km")
    importance = performance_change_importance_table(payload, "5 km")
    notes = performance_change_model_notes(payload, "5 km")

    assert summary.loc[0, "Korrigált változás"] == "-8.0 mp/km"
    assert summary.loc[0, "confidence"] == "62%"
    assert metrics["Legnagyobb improvement"].startswith("5 km")
    assert performance_change_distance_options(payload) == ["5 km"]
    assert chart.loc[0, "raw tempó (perc/km)"] == 5.0
    assert chart.loc[0, "Korrigált tempó (perc/km)"] == 4.92
    assert observations.loc[0, "raw tempó"] == "5:00/km"
    assert importance.loc[0, "feature"] == "szintemelkedés proxy"
    assert "proxy" in notes["Terep"]


def test_training_block_helpers_tolerate_empty_data():
    empty = pd.DataFrame()

    assert training_block_timeline_table(empty).empty
    assert training_block_composition_table(empty).empty
    assert training_block_improvement_table(empty).empty
    assert training_block_volume_chart(empty).empty
    assert training_block_improvement_chart(empty).empty
    assert training_block_summary_values(empty)["Blokkok"] == "0"
    assert training_block_detail_values(empty, "missing") == {}


def test_training_block_helpers_format_blocks_graphically():
    blocks = pd.DataFrame(
        [
            {
                "start_date": "2026-01-01",
                "end_date": "2026-01-07",
                "block_type": "base",
                "volume": 42.0,
                "easy_fraction": 0.8,
                "moderate_fraction": 0.15,
                "hard_fraction": 0.05,
                "performance_before": 30.0,
                "performance_after": 31.5,
                "improvement_score": 1.5,
                "likely_improvement_driver": "base",
                "confidence": 0.6,
            },
            {
                "start_date": "2026-01-08",
                "end_date": "2026-01-10",
                "block_type": "overload",
                "volume": 55.0,
                "easy_fraction": 0.5,
                "moderate_fraction": 0.2,
                "hard_fraction": 0.3,
                "performance_before": 31.5,
                "performance_after": 30.5,
                "improvement_score": -1.0,
                "likely_improvement_driver": "overload",
                "confidence": 0.35,
            },
        ]
    )

    timeline = training_block_timeline_table(blocks)
    summary = training_block_summary_values(blocks)
    composition = training_block_composition_table(blocks)
    improvement = training_block_improvement_table(blocks)
    volume_chart = training_block_volume_chart(blocks)
    improvement_chart = training_block_improvement_chart(blocks)
    detail = training_block_detail_values(blocks, "1. Alapozás")

    assert timeline.loc[0, "Típus"] == "Alapozás"
    assert timeline.loc[0, "Szín"] == "#3182bd"
    assert timeline.loc[1, "Szín"] == "#de2d26"
    assert composition.loc[0, "laza"] == "80%"
    assert improvement.loc[1, "improvement"] == "-1.00"
    assert summary["Legjobb improvement"].startswith("Alapozás")
    assert summary["Legnagyobb romlás"].startswith("Túlterhelés")
    assert not volume_chart.empty
    assert not improvement_chart.empty
    assert detail["Intenzitás"] == "laza 80%, közepes 15%, minőségi/kemény 5%"


def test_training_block_helpers_tolerate_missing_performance_fields():
    blocks = pd.DataFrame([{"start_date": "2026-01-01", "end_date": "2026-01-01", "block_type": "unknown"}])

    table = training_block_improvement_table(blocks)
    summary = training_block_summary_values(blocks)

    assert table.loc[0, "Előtte"] == "n/a"
    assert table.loc[0, "confidence"] == "n/a"
    assert summary["Átlagos confidence"] == "n/a"
