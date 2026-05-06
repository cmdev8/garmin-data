from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from garmin_runner.cli import app
from garmin_runner.config import load_athlete_config
from garmin_runner.planning.candidates import CandidatePlanGenerator
from garmin_runner.planning.constraints import ConstraintValidator
from garmin_runner.planning.historical import extract_historical_improvement_driver
from garmin_runner.planning.optimizer import V2PlanOptimizer, run_v2_optimizer
from garmin_runner.planning.scoring import PlanScorer
from garmin_runner.planning.schema import AthleteState, CandidatePlan, HistoricalDriver, RunWalkType, WeakSystem, WorkoutType
from garmin_runner.planning.templates import make_workout


def make_state(
    *,
    date: str = "2026-05-05",
    target_distance_km: float | None = 10.0,
    target_date: str | None = None,
    current_fitness: float = 45.0,
    current_fatigue: float = 40.0,
    current_form: float = 5.0,
    overload_risk: float = 0.2,
    recent_7d_distance_km: float = 30.0,
    recent_28d_avg_distance_km: float = 28.0,
    recent_7d_load: float = 320.0,
    recent_28d_avg_load: float = 280.0,
    recent_easy_fraction: float = 0.8,
    recent_moderate_fraction: float = 0.1,
    recent_hard_fraction: float = 0.1,
    longest_recent_run_km: float = 10.0,
    longest_continuous_run_s: float | None = 3600.0,
    run_walk_enabled: bool = False,
    run_walk_type: RunWalkType = "not_run_walk",
    average_run_segment_duration_s: float | None = None,
    average_walk_segment_duration_s: float | None = None,
    forced_walk_score: float | None = None,
    weak_system: WeakSystem = "aerobic_base",
    available_days: list[str] | None = None,
    preferred_long_run_day: str = "Sunday",
    max_minutes_per_week: int | None = 300,
    max_volume_increase_pct: float = 10.0,
    max_hard_days_per_week: int = 2,
    horizon_days: int = 7,
    historical_improvement_driver: HistoricalDriver = "neutral",
    historical_improvement_driver_score: float = 0.0,
    historical_improvement_driver_confidence: float = 0.0,
    historical_improvement_driver_label: str | None = None,
) -> AthleteState:
    return AthleteState(
        date=date,
        target_distance_km=target_distance_km,
        target_date=target_date,
        current_fitness=current_fitness,
        current_fatigue=current_fatigue,
        current_form=current_form,
        overload_risk=overload_risk,
        recent_7d_distance_km=recent_7d_distance_km,
        recent_28d_avg_distance_km=recent_28d_avg_distance_km,
        recent_7d_load=recent_7d_load,
        recent_28d_avg_load=recent_28d_avg_load,
        recent_easy_fraction=recent_easy_fraction,
        recent_moderate_fraction=recent_moderate_fraction,
        recent_hard_fraction=recent_hard_fraction,
        longest_recent_run_km=longest_recent_run_km,
        longest_continuous_run_s=longest_continuous_run_s,
        run_walk_enabled=run_walk_enabled,
        run_walk_type=run_walk_type,
        average_run_segment_duration_s=average_run_segment_duration_s,
        average_walk_segment_duration_s=average_walk_segment_duration_s,
        forced_walk_score=forced_walk_score,
        weak_system=weak_system,
        available_days=available_days or ["Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"],
        preferred_long_run_day=preferred_long_run_day,
        max_minutes_per_week=max_minutes_per_week,
        max_volume_increase_pct=max_volume_increase_pct,
        max_hard_days_per_week=max_hard_days_per_week,
        horizon_days=horizon_days,
        historical_improvement_driver=historical_improvement_driver,
        historical_improvement_driver_score=historical_improvement_driver_score,
        historical_improvement_driver_confidence=historical_improvement_driver_confidence,
        historical_improvement_driver_source="5 km" if historical_improvement_driver != "neutral" else None,
        historical_improvement_driver_label=historical_improvement_driver_label,
    )


def make_plan(workouts) -> CandidatePlan:
    total_duration = sum(w.duration_min for w in workouts)
    total_distance = sum(w.distance_km or 0 for w in workouts)
    hard_days = sum(1 for w in workouts if w.workout_type in {"threshold", "vo2max", "speed"})
    long_run = max([w.distance_km or 0 for w in workouts if w.workout_type == "long_run"], default=0)
    moving = sum(w.duration_min for w in workouts if w.intensity != "rest")
    easy = sum(w.duration_min for w in workouts if w.intensity == "easy")
    moderate = sum(w.duration_min for w in workouts if w.intensity == "moderate")
    hard = sum(w.duration_min for w in workouts if w.intensity == "hard")
    return CandidatePlan(
        plan_id="manual_001",
        family="manual",
        workouts=workouts,
        total_duration_min=total_duration,
        total_distance_km=total_distance,
        total_load=sum(w.load_estimate for w in workouts),
        easy_fraction=easy / moving if moving else 0,
        moderate_fraction=moderate / moving if moving else 0,
        hard_fraction=hard / moving if moving else 0,
        hard_days=hard_days,
        long_run_km=long_run,
        has_recovery_day_after_hard=True,
    )


def workout(day: str, workout_type: WorkoutType, distance: float = 5.0):
    return make_workout(
        date=f"2026-05-{day}",
        day={"06": "Wednesday", "07": "Thursday", "08": "Friday", "09": "Saturday", "10": "Sunday"}[day],
        workout_type=workout_type,
        duration_min=40 if workout_type != "rest" else 0,
        distance_km=None if workout_type == "rest" else distance,
    )


def test_generates_at_least_5_candidate_plans():
    plans = CandidatePlanGenerator().generate(make_state())

    assert len(plans) >= 5


def test_rejects_plan_above_weekly_distance_cap():
    state = make_state(recent_7d_distance_km=20, recent_28d_avg_distance_km=20)
    plan = make_plan([workout("06", "easy", 40), workout("07", "easy", 40)])

    result = ConstraintValidator().validate(plan, state)

    assert result.is_valid is False
    assert "too_much_volume" in result.hard_reasons


def test_rejects_hard_days_back_to_back():
    state = make_state()
    plan = make_plan([workout("06", "threshold", 4), workout("07", "vo2max", 4)])

    result = ConstraintValidator().validate(plan, state)

    assert result.is_valid is False
    assert "hard_days_back_to_back" in result.hard_reasons


def test_rejects_vo2_when_fatigue_is_high():
    state = make_state(current_fatigue=85)
    plan = make_plan([workout("06", "vo2max", 4), workout("08", "easy", 4)])

    result = ConstraintValidator().validate(plan, state)

    assert result.is_valid is False
    assert "fatigue_too_high_for_vo2" in result.hard_reasons


def test_high_fatigue_selects_recovery_week():
    state = make_state(current_fatigue=85, overload_risk=0.7, weak_system="vo2max")

    result = V2PlanOptimizer().optimize(state)

    assert result.selected.family == "recovery_week"
    assert all(w.workout_type not in {"vo2max", "threshold"} for w in result.selected.workouts)


def test_high_fatigue_long_horizon_recovers_first_then_returns_to_easy_running():
    state = make_state(current_fatigue=85, overload_risk=0.7, weak_system="vo2max", horizon_days=28)

    result = V2PlanOptimizer().optimize(state)
    first_week = result.selected.workouts[:7]
    later_weeks = result.selected.workouts[7:]

    assert result.selected.family == "recovery_week"
    assert all(workout.workout_type not in {"vo2max", "threshold", "speed"} for workout in result.selected.workouts)
    assert all(workout.workout_type in {"rest", "recovery", "easy"} for workout in first_week)
    assert any(workout.workout_type == "long_run" for workout in later_weeks)
    assert sum(1 for workout in later_weeks if workout.workout_type != "rest") > sum(
        1 for workout in first_week if workout.workout_type != "rest"
    )


def test_threshold_weakness_selects_threshold_or_base_when_fatigue_is_moderate():
    state = make_state(current_fatigue=45, overload_risk=0.2, weak_system="threshold")

    result = V2PlanOptimizer().optimize(state)

    assert result.selected.family in {"threshold_focus_week", "base_build_week"}


def test_historical_quality_driver_maps_from_performance_change():
    payload = {
        "summary": [
            {
                "distance_range": "5 km",
                "adjusted_change_s_per_km": -6.0,
                "confidence": 0.7,
                "primary_controlled_factors": "Kemény arány",
            }
        ],
        "models": {"5 km": {"feature_importance": {"Kemény arány": 0.8, "Átlaghőmérséklet": 0.2}}},
    }

    driver = extract_historical_improvement_driver(payload)

    assert driver.category == "quality"
    assert driver.label == "Kemény arány"


def test_historical_non_actionable_driver_stays_neutral():
    payload = {
        "summary": [{"distance_range": "5 km", "adjusted_change_s_per_km": -6.0, "confidence": 0.7}],
        "models": {"5 km": {"feature_importance": {"Átlaghőmérséklet": 0.9, "GPS lefedettség": 0.1}}},
    }

    driver = extract_historical_improvement_driver(payload)

    assert driver.category == "neutral"


def test_planned_run_walk_selects_run_walk_progression():
    state = make_state(
        run_walk_enabled=True,
        run_walk_type="planned",
        average_run_segment_duration_s=240,
        average_walk_segment_duration_s=60,
        weak_system="aerobic_base",
    )

    result = V2PlanOptimizer().optimize(state)

    assert result.selected.family == "run_walk_progression_week"
    assert any(w.run_walk_prescription for w in result.selected.workouts)


def test_historical_quality_driver_can_select_harder_plan_when_safe():
    state = make_state(
        weak_system="aerobic_base",
        historical_improvement_driver="quality",
        historical_improvement_driver_score=0.95,
        historical_improvement_driver_confidence=0.8,
        historical_improvement_driver_label="Kemény arány",
    )

    result = V2PlanOptimizer().optimize(state)

    assert result.selected.family in {"threshold_focus_week", "vo2_focus_week", "speed_support_week"}
    assert any(w.workout_type in {"threshold", "vo2max", "speed"} for w in result.selected.workouts)
    assert result.selected.historical_driver_match > 0


def test_historical_quality_driver_does_not_override_high_fatigue():
    state = make_state(
        current_fatigue=85,
        overload_risk=0.7,
        weak_system="aerobic_base",
        historical_improvement_driver="quality",
        historical_improvement_driver_score=0.95,
        historical_improvement_driver_confidence=0.8,
        historical_improvement_driver_label="Kemény arány",
    )

    result = V2PlanOptimizer().optimize(state)

    assert result.selected.family == "recovery_week"
    assert all(w.workout_type not in {"threshold", "vo2max", "speed"} for w in result.selected.workouts)


def test_historical_long_run_driver_favors_long_run_focus():
    state = make_state(
        weak_system="aerobic_base",
        historical_improvement_driver="long_run",
        historical_improvement_driver_score=0.95,
        historical_improvement_driver_confidence=0.8,
        historical_improvement_driver_label="Hosszú futás",
    )

    result = V2PlanOptimizer().optimize(state)

    assert result.selected.family in {"long_run_focus_week", "base_build_week"}
    assert result.selected.historical_driver_match > 0


def test_disabled_run_walk_does_not_generate_run_walk_candidates_or_prescriptions():
    state = make_state(
        run_walk_enabled=False,
        run_walk_type="planned",
        average_run_segment_duration_s=240,
        average_walk_segment_duration_s=60,
        weak_system="aerobic_base",
    )

    plans = CandidatePlanGenerator().generate(state)
    result = V2PlanOptimizer().optimize(state)

    assert all(plan.family != "run_walk_progression_week" for plan in plans)
    assert all(not workout.run_walk_prescription for plan in plans for workout in plan.workouts)
    assert result.selected.family != "run_walk_progression_week"
    assert all(not workout.run_walk_prescription for workout in result.selected.workouts)


def test_run_walk_history_does_not_create_prescriptions_when_candidates_disabled():
    config = load_athlete_config()
    config.run_walk.enabled = True
    config.run_walk.allow_walk_run_candidates = False
    activity_features = [
        {
            "date": "2026-05-05",
            "run_walk_type": "forced",
            "average_run_segment_duration_s": 180,
            "average_walk_segment_duration_s": 91,
            "forced_walk_score": 0.8,
        }
    ]

    result = run_v2_optimizer(
        config=config,
        activity_features=activity_features,
        weekly_features=[{"date": "2026-05-05", "distance_7d_km": 30.0, "load_7d": 300.0, "easy_fraction": 0.8}],
        fitness_state=[{"date": "2026-05-05", "fitness": 45.0, "fatigue": 40.0, "form": 5.0, "overload_risk": 0.2}],
    )

    assert result["state"].run_walk_enabled is False
    assert all(not workout.get("run_walk_prescription") for workout in result["plan"]["days"])


def test_run_walk_prescriptions_require_explicit_candidate_toggle():
    config = load_athlete_config()
    config.run_walk.enabled = True
    config.run_walk.allow_walk_run_candidates = True
    activity_features = [
        {
            "date": "2026-05-05",
            "run_walk_type": "forced",
            "average_run_segment_duration_s": 180,
            "average_walk_segment_duration_s": 91,
            "forced_walk_score": 0.8,
        }
    ]

    result = run_v2_optimizer(
        config=config,
        activity_features=activity_features,
        weekly_features=[{"date": "2026-05-05", "distance_7d_km": 30.0, "load_7d": 300.0, "easy_fraction": 0.8}],
        fitness_state=[{"date": "2026-05-05", "fitness": 45.0, "fatigue": 40.0, "form": 5.0, "overload_risk": 0.2}],
    )

    assert result["state"].run_walk_enabled is True
    assert any(workout.get("run_walk_prescription") for workout in result["plan"]["days"])


def test_forced_run_walk_does_not_get_vo2_or_speed():
    state = make_state(
        run_walk_enabled=True,
        run_walk_type="forced",
        forced_walk_score=0.8,
        weak_system="vo2max",
        current_fatigue=45,
    )

    result = V2PlanOptimizer().optimize(state)

    assert all(w.workout_type not in {"vo2max", "speed"} for w in result.selected.workouts)
    assert any(w.run_walk_prescription for w in result.selected.workouts)
    prescribed = [w for w in result.selected.workouts if w.run_walk_prescription]
    assert all(w.run_walk_fatigue_multiplier >= 0.9 for w in prescribed)


def test_planned_run_walk_candidate_has_reduced_load():
    state = make_state(run_walk_enabled=True, run_walk_type="planned", average_run_segment_duration_s=240, average_walk_segment_duration_s=60)

    plans = CandidatePlanGenerator().generate(state)
    run_walk_plan = next(plan for plan in plans if plan.family == "run_walk_progression_week")
    prescribed = [workout for workout in run_walk_plan.workouts if workout.run_walk_prescription]

    assert prescribed
    assert all(workout.run_walk_fatigue_multiplier == 0.85 for workout in prescribed)
    assert all(workout.load_estimate < (workout.distance_km or 0.0) * 8.0 for workout in prescribed)


def test_planned_run_walk_expected_fatigue_is_lower_than_continuous_load():
    state = make_state()
    continuous = make_plan([workout("06", "easy", 5), workout("08", "easy", 5)])
    adjusted_workouts = [
        make_workout(date="2026-05-06", day="Wednesday", workout_type="easy", duration_min=40, distance_km=5, run_walk_fatigue_multiplier=0.85),
        make_workout(date="2026-05-08", day="Friday", workout_type="easy", duration_min=40, distance_km=5, run_walk_fatigue_multiplier=0.85),
    ]
    run_walk = make_plan(adjusted_workouts)
    scorer = PlanScorer()
    validation = ConstraintValidator().validate(continuous, state)

    continuous = scorer.score(continuous, state, validation)
    run_walk = scorer.score(run_walk, state, validation)

    assert run_walk.expected_fatigue < continuous.expected_fatigue


def test_does_not_schedule_workouts_on_unavailable_days():
    state = make_state(available_days=["Tuesday"])

    plans = CandidatePlanGenerator().generate(state)

    for plan in plans:
        assert all(w.workout_type == "rest" or w.day == "Tuesday" for w in plan.workouts)


def test_respects_max_minutes_per_week():
    state = make_state(max_minutes_per_week=75)

    result = V2PlanOptimizer().optimize(state)

    assert result.selected.total_duration_min <= 75


def test_horizon_longer_than_week_repeats_workouts_across_weeks():
    state = make_state(horizon_days=14)

    plan = next(plan for plan in CandidatePlanGenerator().generate(state) if plan.family == "base_build_week")
    workout_offsets = [
        index
        for index, workout in enumerate(plan.workouts)
        if workout.workout_type != "rest"
    ]

    assert len(plan.workouts) == 14
    assert any(offset < 7 for offset in workout_offsets)
    assert any(offset >= 7 for offset in workout_offsets)
    assert sum(1 for workout in plan.workouts if workout.workout_type == "long_run") == 2


def test_rejects_long_run_above_allowed_share():
    state = make_state()
    plan = make_plan([workout("06", "long_run", 6), workout("08", "easy", 4)])

    result = ConstraintValidator().validate(plan, state)

    assert result.is_valid is False
    assert "long_run_too_large" in result.hard_reasons


def test_cli_analyze_is_v2_only_and_exports_only_when_requested(tmp_path: Path):
    fit_dir = tmp_path / "fit"
    out_dir = tmp_path / "out"
    fit_dir.mkdir()

    result = CliRunner().invoke(app, ["analyze", "--fit-dir", str(fit_dir), "--export-dir", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert (out_dir / "candidate_plans.parquet").exists()
    assert (out_dir / "plan_optimizer_debug.json").exists()


def test_cli_no_longer_exposes_plan_command():
    result = CliRunner().invoke(app, ["plan", "--help"])

    assert result.exit_code != 0
