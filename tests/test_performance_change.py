from __future__ import annotations

from garmin_runner.models.performance_change import build_performance_change


def _activity(index: int, *, pace: float, elevation: float = 5.0, forced: float = 0.0, run_walk_type: str = "not_run_walk") -> dict:
    return {
        "activity_id": f"a{index}",
        "date": f"2026-01-{index + 1:02d}",
        "distance_km": 6.0,
        "moving_time_min": 30.0,
        "elapsed_time_min": 31.0,
        "best_rolling_5km_pace": pace,
        "avg_pace_s_per_km": pace,
        "avg_hr": 145,
        "avg_cadence": 170,
        "elevation_gain": elevation,
        "hr_drift_proxy": 1.0,
        "decoupling_proxy": 0.02,
        "data_quality_score": 1.0,
        "run_walk_type": run_walk_type,
        "walk_fraction": 0.0 if forced == 0 else 0.18,
        "forced_walk_score": forced,
        "planned_run_walk_score": 0.0,
        "run_walk_fatigue_multiplier": 1.0,
        "number_of_walk_breaks": 0 if forced == 0 else 5,
        "run_pace_only_s_per_km": pace,
    }


def _weekly(index: int) -> dict:
    return {
        "date": f"2026-01-{index + 1:02d}",
        "load_7d": 300,
        "fatigue_adjusted_load_7d": 270,
        "acute_chronic_workload_ratio": 1.0,
        "fatigue_adjusted_acute_chronic_workload_ratio": 0.9,
        "hard_fraction": 0.1,
        "long_run_distance_km": 10.0,
    }


def _fitness(index: int) -> dict:
    return {
        "date": f"2026-01-{index + 1:02d}",
        "fatigue": 20.0,
        "form": 5.0,
        "overload_risk": 0.1,
    }


def test_performance_change_keeps_improvement_after_controls():
    activities = [_activity(index, pace=310.0 - index * 1.2) for index in range(24)]

    result = build_performance_change(activities, [_weekly(i) for i in range(24)], [_fitness(i) for i in range(24)])
    summary = {row["distance_range"]: row for row in result["summary"]}

    assert summary["5 km"]["model_type"] == "HistGradientBoostingRegressor"
    assert summary["5 km"]["adjusted_change_s_per_km"] < 0
    assert summary["5 km"]["improvement_s_per_km"] > 0


def test_performance_change_controls_uphill_and_forced_run_walk_decline():
    activities = []
    for index in range(12):
        if index < 6:
            activities.append(_activity(index, pace=300.0, elevation=0.0, forced=0.0, run_walk_type="not_run_walk"))
        else:
            activities.append(_activity(index, pace=330.0, elevation=180.0, forced=0.8, run_walk_type="forced"))

    result = build_performance_change(activities, [_weekly(i) for i in range(12)], [_fitness(i) for i in range(12)])
    summary = {row["distance_range"]: row for row in result["summary"]}

    assert summary["5 km"]["model_type"] == "Ridge kontrollmodell"
    assert summary["5 km"]["raw_change_s_per_km"] > 0
    assert summary["5 km"]["adjusted_change_s_per_km"] < summary["5 km"]["raw_change_s_per_km"]


def test_performance_change_sparse_data_uses_raw_fallback():
    activities = [_activity(index, pace=300.0 - index) for index in range(4)]

    result = build_performance_change(activities, [_weekly(i) for i in range(4)], [_fitness(i) for i in range(4)])
    summary = {row["distance_range"]: row for row in result["summary"]}

    assert summary["5 km"]["model_type"] == "Nyers trend"
    assert summary["5 km"]["confidence"] <= 0.3
    assert "trained_model" not in result


def test_performance_change_maps_race_like_bins():
    activity = {
        **_activity(0, pace=300.0),
        "distance_km": 32.0,
        "best_rolling_1km_pace": 250.0,
        "best_rolling_5km_pace": 280.0,
        "best_rolling_10km_pace": 300.0,
        "avg_pace_s_per_km": 330.0,
    }

    result = build_performance_change([activity], [_weekly(0)], [_fitness(0)])
    ranges = {row["distance_range"] for row in result["observations"]}

    assert {"1 km", "5 km", "10 km", "42.2 km"}.issubset(ranges)


def test_performance_change_missing_optional_fields_does_not_crash():
    result = build_performance_change(
        [{"activity_id": "a", "date": "2026-01-01", "distance_km": 5.0, "best_rolling_5km_pace": 300.0}],
        [],
        [],
    )

    assert result["summary"][0]["distance_range"] == "5 km"


def test_performance_change_controls_adjusted_run_walk_fatigue_fields():
    result = build_performance_change([_activity(index, pace=300.0 - index) for index in range(8)], [_weekly(i) for i in range(8)], [_fitness(i) for i in range(8)])
    controlled = result["models"]["5 km"]["controlled_variables"]
    observation = result["observations"][0]

    assert "Futás-séta fáradtsági szorzó" in controlled
    assert "Fáradtságra korrigált terhelés" in controlled
    assert observation["fatigue_adjusted_load_7d"] == 270
