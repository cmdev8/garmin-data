from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from garmin_runner.config import load_athlete_config
from garmin_runner.dashboard.analysis_runner import run_analysis_for_uploads
from garmin_runner.dashboard.upload import UploadRequest, UploadedFilePayload
from garmin_runner.models.ml import _aggregate_signed_temporal_features, _dataset_rows, _select_temporal_candidate, _temporal_dataset_rows, train_per_run_model
from garmin_runner.pipeline import AnalysisConfig, run_analysis


def test_uploaded_fit_bytes_can_be_analyzed_without_export_dir():
    content = Path("test-data/2026-05-05-13-31-59.fit").read_bytes()
    config = load_athlete_config()

    result = run_analysis(
        AnalysisConfig(
            athlete_config=config,
            uploaded_fit_files=[("activity.fit", content)],
            export_dir=None,
        )
    )

    assert result.summary()["kept_running_activities"] >= 1
    assert result.next_week_plan["selected_plan_id"]
    assert result.ml_result.training_run_id


def test_every_run_trains_fresh_ml_result():
    config = load_athlete_config()

    first = run_analysis(AnalysisConfig(athlete_config=config))
    second = run_analysis(AnalysisConfig(athlete_config=config))

    assert first.ml_result.training_run_id != second.ml_result.training_run_id
    assert first.ml_result.model_type
    assert second.ml_result.model_type


def test_no_inputs_returns_friendly_warning_and_plan():
    result = run_analysis(AnalysisConfig(athlete_config=load_athlete_config()))

    assert "No FIT inputs provided." in result.warnings
    assert result.next_week_plan["selected_plan_id"]


def test_dashboard_data_does_not_keep_raw_upload_bytes():
    content = Path("test-data/2026-05-05-13-31-59.fit").read_bytes()
    request = UploadRequest(
        fit_files=[UploadedFilePayload("activity.fit", "0001_activity.fit", content, len(content))],
        athlete_settings={"planning": {"horizon_days": 7}},
        run_walk_enabled=True,
    )

    data = run_analysis_for_uploads(request)

    assert not hasattr(data, "fit_files")
    assert not hasattr(data, "raw_bytes")
    assert data.uploaded_file_count == 1
    assert data.temp_files_deleted is True
    assert data.detailed_charts is True
    assert not data.records.empty
    assert not data.segments.empty


def test_ml_learns_diminishing_returns_from_history():
    weekly = [{"date": f"2026-01-{index + 1:02d}", "load_7d": 100.0} for index in range(20)]
    fitness = [
        {"date": f"2026-01-{index + 1:02d}", "predicted_performance_index": 40.0 + 22.0 * (1.0 - 0.9**index)}
        for index in range(27)
    ]

    result = train_per_run_model([], weekly, fitness)

    assert result.predictions["diminishing_returns_factor"] < 1.0
    assert result.predictions["diminishing_returns_factor"] >= 0.65
    assert result.metrics["diminishing_returns_beta"] < 0
    assert "diminishing_returns_raw_ratio" in result.metrics
    assert result.metrics["diminishing_returns_reason"] == "learned_from_history"


def test_ml_tracks_monthly_diminishing_returns_point_in_time():
    start = date(2026, 1, 1)
    weekly = [{"date": (start + timedelta(days=index)).isoformat(), "load_7d": 100.0} for index in range(75)]
    fitness = [
        {
            "date": (start + timedelta(days=index)).isoformat(),
            "predicted_performance_index": 40.0 + 24.0 * (1.0 - 0.94**index),
        }
        for index in range(82)
    ]

    result = train_per_run_model([], weekly, fitness)
    history = result.metrics["diminishing_returns_history"]
    january = next(row for row in history if row["month"] == "2026-01")
    march = next(row for row in history if row["month"] == "2026-03")

    assert january["observations"] == 24
    assert march["factor"] < 1.0
    assert march["factor"] >= 0.65
    assert result.predictions["diminishing_returns_factor"] < 1.0


def test_ml_monthly_diminishing_returns_marks_sparse_early_month_neutral():
    start = date(2026, 1, 28)
    weekly = [{"date": (start + timedelta(days=index)).isoformat(), "load_7d": 100.0} for index in range(40)]
    fitness = [
        {
            "date": (start + timedelta(days=index)).isoformat(),
            "predicted_performance_index": 40.0 + 18.0 * (1.0 - 0.92**index),
        }
        for index in range(47)
    ]

    result = train_per_run_model([], weekly, fitness)
    january = next(row for row in result.metrics["diminishing_returns_history"] if row["month"] == "2026-01")

    assert january["factor"] == 1.0
    assert january["reason"] == "insufficient_true_7d_history"


def test_ml_uses_neutral_diminishing_returns_for_insufficient_history():
    weekly = [{"date": f"2026-01-{index + 1:02d}", "load_7d": 100.0} for index in range(8)]
    fitness = [{"date": f"2026-01-{index + 1:02d}", "predicted_performance_index": 40.0 + index} for index in range(12)]

    result = train_per_run_model([], weekly, fitness)

    assert result.predictions["diminishing_returns_factor"] == 1.0
    assert result.metrics["diminishing_returns_reason"] == "insufficient_true_7d_history"


def test_ml_does_not_discount_non_diminishing_history():
    weekly = [{"date": f"2026-01-{index + 1:02d}", "load_7d": 100.0} for index in range(20)]
    fitness = [{"date": f"2026-01-{index + 1:02d}", "predicted_performance_index": 30.0 + index * index * 0.2} for index in range(27)]

    result = train_per_run_model([], weekly, fitness)

    assert result.predictions["diminishing_returns_factor"] == 1.0
    assert result.metrics["diminishing_returns_reason"] == "non_diminishing_history"


def test_temporal_dataset_uses_only_past_and_current_rows():
    weekly = [{"date": f"2026-01-{index + 1:02d}", "load_7d": float(index)} for index in range(14)]
    fitness = [{"date": f"2026-01-{index + 1:02d}", "predicted_performance_index": 50.0 + index} for index in range(21)]

    rows = _dataset_rows(weekly, fitness)
    temporal_rows, names = _temporal_dataset_rows(rows)
    first = temporal_rows[0]

    assert "load_7d__latest" in names
    assert first["load_7d__latest"] == 6.0
    assert first["load_7d__lag_7d"] == 0.0
    assert first["load_7d__mean_7d"] == 3.0
    assert first["target_performance_index"] == 63.0


def test_enough_history_uses_temporal_model_with_explanations():
    weekly = [
        {
            "date": (date(2026, 1, 1) + timedelta(days=index)).isoformat(),
            "load_7d": 100.0 + index * 2.0,
            "distance_7d_km": 20.0 + index,
            "fatigue_adjusted_load_7d": 95.0 + index * 1.8,
        }
        for index in range(45)
    ]
    fitness = [
        {
            "date": (date(2026, 1, 1) + timedelta(days=index)).isoformat(),
            "predicted_performance_index": 40.0 + index * 0.7,
        }
        for index in range(52)
    ]

    result = train_per_run_model([], weekly, fitness)

    assert result.metrics["temporal_model_used"] is True
    assert result.metrics["validation_strategy"] == "TimeSeriesSplit"
    assert result.metrics["fold_count"] >= 2
    assert result.model_type == "TemporalRidgeCV"
    assert result.positive_temporal_drivers
    assert result.temporal_feature_importance
    assert "trained_model" not in result.to_dict()


def test_temporal_model_selection_prefers_linear_when_close_and_nonlinear_when_clear():
    linear = {"model_type": "TemporalRidgeCV", "mae": 10.0}
    close_nonlinear = {"model_type": "TemporalHistGradientBoostingRegressor", "mae": 9.7}
    clear_nonlinear = {"model_type": "TemporalHistGradientBoostingRegressor", "mae": 8.0}

    assert _select_temporal_candidate([linear, close_nonlinear]) is linear
    assert _select_temporal_candidate([linear, clear_nonlinear]) is clear_nonlinear


def test_temporal_signed_features_aggregate_lagged_names():
    grouped = _aggregate_signed_temporal_features(
        ["load_7d__latest", "load_7d__slope_14d", "overload_risk__latest"],
        [0.4, 0.2, -0.3],
    )

    assert grouped["load_7d"] == 0.6000000000000001
    assert grouped["overload_risk"] == -0.3
