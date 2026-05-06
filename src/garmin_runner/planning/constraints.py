from __future__ import annotations

import math
from collections import Counter

from garmin_runner.planning.candidates import HIGH_FATIGUE, HIGH_FORCED_WALK_SCORE
from garmin_runner.planning.schema import AthleteState, CandidatePlan, ValidationResult


class ConstraintValidator:
    def validate(self, plan: CandidatePlan, state: AthleteState) -> ValidationResult:
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []
        penalty = 0.0

        max_distance = _allowed_distance(state)
        if plan.total_distance_km > max_distance + 0.05:
            hard_reasons.append("too_much_volume")

        max_hard_days = math.ceil(state.max_hard_days_per_week * state.horizon_days / 7)
        if plan.hard_days > max_hard_days:
            hard_reasons.append("too_many_hard_days")

        if _hard_days_back_to_back(plan):
            hard_reasons.append("hard_days_back_to_back")

        if _hard_after_long_run(plan):
            hard_reasons.append("hard_day_after_long_run")

        if state.current_fatigue > HIGH_FATIGUE and any(w.workout_type == "vo2max" for w in plan.workouts):
            hard_reasons.append("fatigue_too_high_for_vo2")

        if _workout_on_unavailable_day(plan, state):
            hard_reasons.append("workout_on_unavailable_day")

        max_minutes = _allowed_minutes(state)
        if max_minutes is not None and plan.total_duration_min > max_minutes:
            hard_reasons.append("too_much_duration")

        forced = state.run_walk_enabled and state.run_walk_type in {"forced", "mixed"} and (state.forced_walk_score or 0.0) >= HIGH_FORCED_WALK_SCORE
        if forced and any(w.workout_type in {"vo2max", "speed"} for w in plan.workouts):
            hard_reasons.append("forced_run_walk_gets_vo2_or_speed")

        if plan.total_distance_km > 0 and plan.long_run_km > 0.4 * plan.total_distance_km:
            hard_reasons.append("long_run_too_large")

        if plan.total_distance_km > 0 and plan.long_run_km > 0.35 * plan.total_distance_km:
            penalty += 0.2
            soft_reasons.append("long_run_share_above_35_percent")
        if plan.moderate_fraction > 0.25:
            penalty += 0.15
            soft_reasons.append("too_much_moderate_intensity")
        if not plan.has_recovery_day_after_hard:
            penalty += 0.2
            soft_reasons.append("no_recovery_after_hard_day")
        if plan.easy_fraction < 0.75 and plan.total_duration_min > 0:
            penalty += 0.15
            soft_reasons.append("too_little_easy_volume")
        if _load_jump(plan, state) > 0.2:
            penalty += min(0.3, _load_jump(plan, state))
            soft_reasons.append("excessive_load_jump")

        return ValidationResult(
            is_valid=not hard_reasons,
            hard_reasons=hard_reasons,
            penalty=min(1.0, penalty),
            soft_reasons=soft_reasons,
        )


def rejection_summary(plans: list[CandidatePlan]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for plan in plans:
        counter.update(plan.rejection_reasons)
    return dict(counter)


def _allowed_distance(state: AthleteState) -> float:
    scale = state.horizon_days / 7
    candidates = []
    if state.recent_28d_avg_distance_km > 0:
        candidates.append(state.recent_28d_avg_distance_km * 1.15 * scale)
    if state.recent_7d_distance_km > 0:
        candidates.append(state.recent_7d_distance_km * (1 + state.max_volume_increase_pct / 100.0) * scale)
    return min(candidates) if candidates else 12.0 * scale


def _allowed_minutes(state: AthleteState) -> int | None:
    if state.max_minutes_per_week is None:
        return None
    return math.floor(state.max_minutes_per_week * state.horizon_days / 7)


def _hard_days_back_to_back(plan: CandidatePlan) -> bool:
    return any(_is_hard(a) and _is_hard(b) for a, b in zip(plan.workouts, plan.workouts[1:]))


def _hard_after_long_run(plan: CandidatePlan) -> bool:
    return any(a.workout_type == "long_run" and _is_hard(b) for a, b in zip(plan.workouts, plan.workouts[1:]))


def _workout_on_unavailable_day(plan: CandidatePlan, state: AthleteState) -> bool:
    allowed = set(state.available_days)
    return any(workout.workout_type != "rest" and workout.day not in allowed for workout in plan.workouts)


def _is_hard(workout) -> bool:
    return workout.workout_type in {"threshold", "vo2max", "speed"}


def _load_jump(plan: CandidatePlan, state: AthleteState) -> float:
    scale = state.horizon_days / 7
    baseline = state.recent_7d_load * scale
    if baseline <= 0:
        return 0.0
    return max(0.0, (plan.total_load - baseline) / baseline)
