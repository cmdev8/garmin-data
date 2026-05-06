from __future__ import annotations

from garmin_runner.features.daily_features import build_daily_features, build_weekly_features
from garmin_runner.models.fitness_fatigue import build_fitness_state


def test_daily_and_weekly_load_use_run_walk_fatigue_adjustment():
    daily = build_daily_features(
        [
            {
                "date": "2026-05-01",
                "distance_km": 10.0,
                "moving_time_min": 70.0,
                "avg_hr": 135,
                "run_walk_type": "planned",
                "run_walk_fatigue_multiplier": 0.85,
            }
        ]
    )
    weekly = build_weekly_features(daily)

    assert daily[0]["load_estimate"] == 100.0
    assert daily[0]["fatigue_adjusted_load_estimate"] == 85.0
    assert daily[0]["run_walk_fatigue_multiplier"] == 0.85
    assert weekly[0]["load_7d"] == 100.0
    assert weekly[0]["fatigue_adjusted_load_7d"] == 85.0
    assert round(weekly[0]["fatigue_adjusted_acute_chronic_workload_ratio"], 3) == 0.143


def test_continuous_run_keeps_adjusted_load_equal_to_raw_load():
    daily = build_daily_features(
        [
            {
                "date": "2026-05-01",
                "distance_km": 10.0,
                "moving_time_min": 70.0,
                "avg_hr": 135,
                "run_walk_type": "not_run_walk",
                "run_walk_fatigue_multiplier": 1.0,
            }
        ]
    )

    assert daily[0]["fatigue_adjusted_load_estimate"] == daily[0]["load_estimate"]


def test_fitness_fatigue_uses_adjusted_load_for_fatigue_only():
    continuous = build_fitness_state(
        [{"date": "2026-05-01", "total_distance_km": 10.0, "load_estimate": 100.0, "fatigue_adjusted_load_estimate": 100.0}]
    )
    planned = build_fitness_state(
        [{"date": "2026-05-01", "total_distance_km": 10.0, "load_estimate": 100.0, "fatigue_adjusted_load_estimate": 85.0}]
    )

    assert planned[0]["fitness"] == continuous[0]["fitness"]
    assert planned[0]["form"] > continuous[0]["form"]
    assert planned[0]["fatigue"] < continuous[0]["fatigue"]
    assert planned[0]["fatigue_adjusted_daily_load"] == 85.0
