from __future__ import annotations

from datetime import date, timedelta
from statistics import mean
from typing import Literal, cast

from garmin_runner.config import WEEKDAYS
from garmin_runner.planning.schema import AthleteState, CandidatePlan, RunWalkType, WeakSystem, Workout, WorkoutType
from garmin_runner.planning.templates import make_workout, run_walk_prescription


CandidateWorkoutType = WorkoutType | Literal["run_walk", "run_walk_long"]
HIGH_FATIGUE = 70
HIGH_OVERLOAD_RISK = 0.65
HIGH_FORCED_WALK_SCORE = 0.60


class CandidatePlanGenerator:
    def generate(self, state: AthleteState) -> list[CandidatePlan]:
        baseline = _baseline_distance(state)
        normal_target = baseline * (1 + state.max_volume_increase_pct / 100.0)
        recovery_target = baseline * 0.65
        fatigue_target = baseline * 0.75
        cap = _volume_cap(state)
        if state.max_minutes_per_week is not None:
            cap = min(cap, (state.max_minutes_per_week * state.horizon_days / 7) / 7.5)

        if state.current_fatigue >= HIGH_FATIGUE or state.overload_risk >= HIGH_OVERLOAD_RISK:
            normal_target = min(normal_target, fatigue_target)

        families: list[tuple[str, float, list[CandidateWorkoutType]]] = [
            ("recovery_week", min(recovery_target, cap), ["recovery", "easy", "recovery"]),
            ("base_build_week", min(normal_target, cap), ["easy", "easy", "long_run", "easy"]),
            ("threshold_focus_week", min(normal_target, cap), ["threshold", "easy", "long_run", "easy"]),
            ("vo2_focus_week", min(normal_target * 0.95, cap), ["vo2max", "easy", "long_run", "easy"]),
            ("long_run_focus_week", min(normal_target, cap), ["easy", "easy", "long_run", "recovery"]),
            ("maintenance_week", min(baseline * 0.9, cap), ["easy", "recovery", "easy"]),
        ]
        if _speed_support_is_appropriate(state):
            families.insert(4, ("speed_support_week", min(normal_target * 0.9, cap), ["easy", "speed", "easy", "long_run"]))
        if state.run_walk_enabled and state.run_walk_type in {"planned", "forced", "mixed"}:
            families.insert(
                5,
                ("run_walk_progression_week", min(normal_target * 0.9, cap), ["run_walk", "easy", "run_walk_long", "recovery"]),
            )

        plans = []
        for index, (family, distance, template) in enumerate(families, 1):
            plans.append(self._make_plan(state, family, distance, template, index))

        if len(plans) < 10:
            plans.extend(
                [
                    self._make_plan(state, "base_build_week", min(normal_target * 0.85, cap), ["easy", "long_run", "easy"], 8),
                    self._make_plan(state, "threshold_focus_week", min(normal_target * 0.9, cap), ["easy", "threshold", "recovery", "easy"], 9),
                    self._make_plan(state, "long_run_focus_week", min(normal_target * 0.9, cap), ["recovery", "easy", "long_run"], 10),
                ]
            )
        return plans[:20]

    def _make_plan(
        self,
        state: AthleteState,
        family: str,
        target_distance: float,
        template: list[CandidateWorkoutType],
        ordinal: int,
    ) -> CandidatePlan:
        dates = _plan_dates(state)
        available_days = state.available_days or WEEKDAYS
        available_slots = [i for i, (_, day) in enumerate(dates) if day in available_days]
        workouts: list[Workout] = [
            make_workout(date=day_date, day=day, workout_type="rest", duration_min=0, distance_km=None)
            for day_date, day in dates
        ]

        if not available_slots:
            return _summarize(f"{family}_{ordinal:03d}", family, workouts, ["No available training days"])

        slot_plan = _assign_slots(template, available_slots, dates, state.preferred_long_run_day, state, family)
        distances = _distances_for_types(target_distance, [workout_type for _, workout_type in slot_plan])
        for (slot, workout_type), distance in zip(slot_plan, distances):
            date_text, day = dates[slot]
            prescription = _prescription(state, workout_type)
            actual_type: WorkoutType = "easy" if workout_type in {"run_walk", "run_walk_long"} else cast(WorkoutType, workout_type)
            duration = _duration_for(actual_type, distance)
            fatigue_multiplier = _candidate_run_walk_fatigue_multiplier(state, actual_type, prescription)
            workouts[slot] = make_workout(
                date=date_text,
                day=day,
                workout_type=actual_type,
                duration_min=duration,
                distance_km=distance,
                run_walk_prescription=prescription,
                run_walk_fatigue_multiplier=fatigue_multiplier,
            )

        return _summarize(f"{family}_{ordinal:03d}", family, workouts, [f"Candidate family: {family}"])


def _plan_dates(state: AthleteState) -> list[tuple[str, str]]:
    start = date.fromisoformat(state.date) + timedelta(days=1)
    return [((start + timedelta(days=i)).isoformat(), (start + timedelta(days=i)).strftime("%A")) for i in range(state.horizon_days)]


def _assign_slots(
    template: list[CandidateWorkoutType],
    available_slots: list[int],
    dates: list[tuple[str, str]],
    preferred_long_run_day: str,
    state: AthleteState,
    family: str,
) -> list[tuple[int, CandidateWorkoutType]]:
    assignments: list[tuple[int, CandidateWorkoutType]] = []
    for week_start in range(0, len(dates), 7):
        week_end = min(week_start + 7, len(dates))
        week_slots = [slot for slot in available_slots if week_start <= slot < week_end]
        if not week_slots:
            continue
        week_index = week_start // 7
        assignments.extend(_assign_week_slots(_template_for_week(template, state, family, week_index), week_slots, dates, preferred_long_run_day))
    return sorted(assignments)


def _template_for_week(
    template: list[CandidateWorkoutType], state: AthleteState, family: str, week_index: int
) -> list[CandidateWorkoutType]:
    high_recovery_need = state.current_fatigue >= HIGH_FATIGUE or state.overload_risk >= HIGH_OVERLOAD_RISK
    if not high_recovery_need:
        return template
    if family == "recovery_week":
        if week_index == 0:
            return ["recovery", "easy", "recovery"]
        return ["easy", "easy", "long_run", "recovery"]
    if week_index == 0:
        return ["recovery" if workout in {"threshold", "vo2max", "speed"} else workout for workout in template]
    return template


def _assign_week_slots(
    template: list[CandidateWorkoutType],
    week_slots: list[int],
    dates: list[tuple[str, str]],
    preferred_long_run_day: str,
) -> list[tuple[int, CandidateWorkoutType]]:
    assignments: list[tuple[int, CandidateWorkoutType]] = []
    remaining_template = list(template)
    long_workout: CandidateWorkoutType | None = None
    for workout in remaining_template:
        if workout in {"long_run", "run_walk_long"}:
            long_workout = workout
            break
    if long_workout is not None:
        preferred = [slot for slot in week_slots if dates[slot][1] == preferred_long_run_day]
        slot = (preferred or week_slots)[-1]
        assignments.append((slot, long_workout))
        remaining_template.remove(long_workout)

    assigned_slots = {slot for slot, _ in assignments}
    available_without_long = [slot for slot in week_slots if slot not in assigned_slots]
    for workout_type, slot in zip(remaining_template, available_without_long):
        assignments.append((slot, workout_type))
    return assignments


def _distances_for_types(target_distance: float, workout_types: list[CandidateWorkoutType]) -> list[float]:
    weights = []
    for workout_type in workout_types:
        if workout_type in {"long_run", "run_walk_long"}:
            weights.append(1.75)
        elif workout_type in {"threshold", "vo2max"}:
            weights.append(1.2)
        elif workout_type == "recovery":
            weights.append(0.65)
        else:
            weights.append(1.0)
    total = sum(weights) or 1.0
    return [round(max(0.0, target_distance) * weight / total, 2) for weight in weights]


def _duration_for(workout_type: WorkoutType, distance: float) -> int:
    pace_min_per_km = {
        "recovery": 8.0,
        "easy": 7.0,
        "long_run": 7.3,
        "threshold": 6.2,
        "vo2max": 6.0,
        "speed": 6.5,
    }.get(workout_type, 7.0)
    return max(15, int(round(distance * pace_min_per_km)))


def _prescription(state: AthleteState, workout_type: CandidateWorkoutType) -> str | None:
    if not state.run_walk_enabled:
        return None
    if workout_type not in {"run_walk", "run_walk_long", "easy", "long_run", "recovery"}:
        return None
    if state.run_walk_type == "planned":
        run_minutes = int((state.average_run_segment_duration_s or 240) / 60) + 1
        walk_seconds = int(state.average_walk_segment_duration_s or 60)
        reps = 8 if workout_type != "run_walk_long" else 10
        return run_walk_prescription(run_minutes, walk_seconds, reps)
    if state.run_walk_type in {"forced", "mixed"}:
        walk_seconds = int(max(90, state.average_walk_segment_duration_s or 90))
        reps = 6 if workout_type != "run_walk_long" else 8
        return run_walk_prescription(3, walk_seconds, reps, forced=True)
    return None


def _candidate_run_walk_fatigue_multiplier(state: AthleteState, workout_type: WorkoutType, prescription: str | None) -> float:
    if prescription is None or workout_type not in {"recovery", "easy", "long_run"}:
        return 1.0
    if state.run_walk_type == "planned":
        return 0.85
    if state.run_walk_type in {"forced", "mixed"}:
        return 0.90
    return 1.0


def _speed_support_is_appropriate(state: AthleteState) -> bool:
    if state.current_fatigue >= HIGH_FATIGUE or state.overload_risk >= HIGH_OVERLOAD_RISK:
        return False
    if state.run_walk_enabled and state.run_walk_type in {"forced", "mixed"} and (state.forced_walk_score or 0.0) >= HIGH_FORCED_WALK_SCORE:
        return False
    return state.historical_improvement_driver in {"quality", "speed_economy"} or state.weak_system == "speed" or (
        state.target_distance_km is not None and state.target_distance_km <= 5
    )


def _summarize(plan_id: str, family: str, workouts: list[Workout], explanation: list[str]) -> CandidatePlan:
    total_duration = sum(w.duration_min for w in workouts)
    total_distance = sum(w.distance_km or 0.0 for w in workouts)
    total_load = sum(w.load_estimate for w in workouts)
    hard_days = sum(1 for w in workouts if w.workout_type in {"threshold", "vo2max", "speed"})
    long_run_km = max([w.distance_km or 0.0 for w in workouts if w.workout_type == "long_run"], default=0.0)
    intensity_minutes = {
        "easy": sum(w.duration_min for w in workouts if w.intensity == "easy"),
        "moderate": sum(w.duration_min for w in workouts if w.intensity == "moderate"),
        "hard": sum(w.duration_min for w in workouts if w.intensity == "hard"),
    }
    moving = sum(intensity_minutes.values())
    return CandidatePlan(
        plan_id=plan_id,
        family=family,
        workouts=workouts,
        total_duration_min=total_duration,
        total_distance_km=round(total_distance, 2),
        total_load=round(total_load, 2),
        easy_fraction=intensity_minutes["easy"] / moving if moving else 0.0,
        moderate_fraction=intensity_minutes["moderate"] / moving if moving else 0.0,
        hard_fraction=intensity_minutes["hard"] / moving if moving else 0.0,
        hard_days=hard_days,
        long_run_km=long_run_km,
        has_recovery_day_after_hard=_has_recovery_after_hard(workouts),
        explanation=explanation,
    )


def _has_recovery_after_hard(workouts: list[Workout]) -> bool:
    for current, following in zip(workouts, workouts[1:]):
        if current.workout_type in {"threshold", "vo2max", "speed"}:
            return following.workout_type in {"rest", "recovery", "easy"}
    return True


def _baseline_distance(state: AthleteState) -> float:
    baseline = max(state.recent_7d_distance_km, state.recent_28d_avg_distance_km * 0.9)
    if baseline <= 0:
        baseline = 10.0
    return baseline * state.horizon_days / 7


def _volume_cap(state: AthleteState) -> float:
    scale = state.horizon_days / 7
    cap_candidates = []
    if state.recent_28d_avg_distance_km > 0:
        cap_candidates.append(state.recent_28d_avg_distance_km * 1.15 * scale)
    if state.recent_7d_distance_km > 0:
        cap_candidates.append(state.recent_7d_distance_km * (1 + state.max_volume_increase_pct / 100.0) * scale)
    return min(cap_candidates) if cap_candidates else 12.0 * scale


def infer_weak_system(weekly_features: list[dict], activity_features: list[dict]) -> WeakSystem:
    latest = weekly_features[-1] if weekly_features else {}
    if latest.get("easy_fraction", 0.0) < 0.65:
        return "recovery"
    if latest.get("long_run_distance_km", 0.0) < 0.45 * max(10.0, latest.get("distance_7d_km", 0.0)):
        return "long_run_endurance"
    best_5k = [r.get("best_rolling_5km_pace") for r in activity_features if r.get("best_rolling_5km_pace")]
    best_1k = [r.get("best_rolling_1km_pace") for r in activity_features if r.get("best_rolling_1km_pace")]
    if best_1k and not best_5k:
        return "threshold"
    return "aerobic_base"


def latest_run_walk_type(activity_features: list[dict]) -> RunWalkType:
    types = [row.get("run_walk_type") for row in activity_features if row.get("run_walk_type")]
    value = str(types[-1]) if types else "unknown"
    if value not in {"planned", "forced", "mixed", "not_run_walk", "unknown"}:
        value = "unknown"
    return cast(RunWalkType, value)


def mean_recent(activity_features: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in activity_features[-8:] if row.get(field) is not None]
    return mean(values) if values else None
