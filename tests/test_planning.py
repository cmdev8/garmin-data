from datetime import date, timedelta

from garmin_runner.config import load_athlete_config
from garmin_runner.planning.rules import generate_plan


def _activity_rows():
    start = date(2026, 5, 1)
    return [
        {
            "date": (start + timedelta(days=i)).isoformat(),
            "distance_km": 5.0,
            "moving_time_min": 35.0,
            "run_walk_type": "not_run_walk",
        }
        for i in range(7)
    ]


def test_7_day_plan_uses_default_horizon():
    config = load_athlete_config()

    plan = generate_plan(config, _activity_rows(), [], [])

    assert plan["horizon_days"] == 7
    assert len(plan["days"]) == 7
    assert plan["start_date"] == "2026-05-08"


def test_14_day_plan_has_14_dated_days():
    config = load_athlete_config(planning_horizon_days=14)

    plan = generate_plan(config, _activity_rows(), [], [])

    assert plan["horizon_days"] == 14
    assert len(plan["days"]) == 14
    assert plan["end_date"] == "2026-05-21"


def test_hard_day_limit_scales_with_horizon():
    config = load_athlete_config(planning_horizon_days=14)

    plan = generate_plan(config, _activity_rows(), [], [])

    assert plan["constraints_applied"]["max_hard_days"] == 4
    assert plan["summary"]["hard_days"] <= 4


def test_no_hard_days_back_to_back():
    config = load_athlete_config(planning_horizon_days=14)

    plan = generate_plan(config, _activity_rows(), [], [])
    intensities = [day["intensity"] for day in plan["days"]]

    assert all(not (a == "hard" and b == "hard") for a, b in zip(intensities, intensities[1:]))


def test_high_fatigue_reduces_hard_work():
    config = load_athlete_config(planning_horizon_days=10)
    fitness_state = [{"date": "2026-05-07", "fitness": 20.0, "fatigue": 40.0}]

    plan = generate_plan(config, _activity_rows(), [], fitness_state)

    assert plan["constraints_applied"]["high_fatigue_recovery_bias"] is True
    assert plan["summary"]["hard_days"] == 0
