from __future__ import annotations

from pathlib import Path

from garmin_runner.features.activity_features import build_activity_features
from garmin_runner.features.daily_features import build_daily_features, build_weekly_features
from garmin_runner.fit.parser import parse_fit_file
from garmin_runner.models.ml import FEATURE_NAMES, train_per_run_model
from garmin_runner.models.performance_change import build_performance_change


def test_parser_extracts_expanded_fit_session_and_record_fields():
    activities, records, warnings = parse_fit_file(Path("test-data/2026-05-05-13-31-59.fit"))

    activity = activities[0]
    record = records[0]

    assert warnings == []
    assert activity.min_hr == 96.0
    assert activity.total_calories == 166.0
    assert activity.total_training_effect == 2.3
    assert activity.total_anaerobic_training_effect == 0.0
    assert activity.avg_temperature_c == 30.0
    assert activity.avg_power_w == 356.0
    assert activity.normalized_power_w == 365.0
    assert activity.total_work_kj is not None
    assert activity.avg_step_length_mm == 994.5
    assert activity.avg_vertical_oscillation_mm == 83.9
    assert activity.avg_vertical_ratio == 8.46
    assert activity.lap_count == 3
    assert activity.lap_pace_variability is not None
    assert record.temperature_c == 32.0
    assert record.fractional_cadence == 0.5
    assert record.activity_type == "running"
    assert record.accumulated_power_w == 104.0
    assert record.vertical_oscillation_mm == 57.5
    assert record.step_length_mm == 771.0
    assert record.vertical_ratio == 7.45


def test_expanded_activity_daily_and_weekly_features_are_populated():
    activities, records, _ = parse_fit_file(Path("test-data/2026-05-05-13-31-59.fit"))
    activities[0].keep = True

    activity_features = build_activity_features(activities, records, {})
    daily = build_daily_features(activity_features)
    weekly = build_weekly_features(daily)

    activity = activity_features[0]
    week = weekly[-1]

    assert activity["calories_per_km"] > 0
    assert activity["power_hr_efficiency"] > 0
    assert activity["stride_length_mm"] == 994.5
    assert activity["pace_variability"] is not None
    assert activity["gps_coverage"] > 0.9
    assert week["avg_power_w_7d_mean"] == activity["avg_power_w"]
    assert week["avg_temperature_c_7d_mean"] == activity["avg_temperature_c"]
    assert week["total_training_effect_7d_mean"] == activity["total_training_effect"]
    assert week["lap_pace_variability_7d_mean"] == activity["lap_pace_variability"]


def test_expanded_ml_features_train_with_sparse_optional_fields():
    weekly = [
        {
            "date": f"2026-01-{index + 1:02d}",
            "distance_7d_km": 20 + index,
            "load_7d": 100 + index,
            "avg_power_w_7d_mean": 250 + index,
            "avg_temperature_c_7d_mean": 18 + index % 3,
            "stride_length_mm_7d_mean": 1000 + index,
        }
        for index in range(14)
    ]
    fitness = [
        {"date": f"2026-01-{index + 1:02d}", "predicted_performance_index": 50 + index * 0.5}
        for index in range(21)
    ]

    result = train_per_run_model([], weekly, fitness)

    assert "avg_power_w_7d_mean" in FEATURE_NAMES
    assert "stride_length_mm_7d_mean" in result.feature_names
    assert result.model_type in {"TemporalRidgeCV", "TemporalHistGradientBoostingRegressor", "HistGradientBoostingRegressor", "DummyRegressor"}
    assert "trained_model" not in result.to_dict()


def test_performance_change_controls_expanded_fit_attributes():
    activity = {
        "activity_id": "a1",
        "date": "2026-01-01",
        "distance_km": 5.0,
        "moving_time_min": 25.0,
        "elapsed_time_min": 26.0,
        "best_rolling_5km_pace": 300.0,
        "avg_power_w": 280.0,
        "avg_temperature_c": 27.0,
        "stride_length_mm": 1020.0,
        "vertical_ratio": 8.8,
        "lap_pace_variability": 0.04,
        "hrv_rmssd_ms": 35.0,
    }

    result = build_performance_change([activity], [], [])
    controlled = result["models"]["5 km"]["controlled_variables"]
    observation = result["observations"][0]

    assert "Átlagteljesítmény" in controlled
    assert "Átlaghőmérséklet" in controlled
    assert "Futódinamika" not in controlled
    assert "Lépéshossz" in controlled
    assert "HRV RMSSD" in controlled
    assert observation["avg_power_w"] == 280.0
    assert observation["avg_temperature_c"] == 27.0
