from __future__ import annotations

from garmin_runner.planning.candidates import HIGH_FATIGUE, HIGH_FORCED_WALK_SCORE
from garmin_runner.planning.schema import AthleteState, CandidatePlan, ValidationResult


SCORING_WEIGHTS = {
    "expected_adaptation": 2.0,
    "missing_stimulus_score": 1.5,
    "race_specificity": 1.0,
    "historical_driver_match": 0.8,
    "expected_fatigue": -2.0,
    "overload_risk": -3.0,
    "constraint_penalty": -2.0,
}


ADAPTATION = {
    "easy": 0.35,
    "long_run": 0.55,
    "threshold": 0.75,
    "vo2max": 0.70,
    "speed": 0.45,
    "recovery": 0.15,
    "rest": 0.05,
}


class PlanScorer:
    def __init__(self, ml_result=None):
        self.ml_result = ml_result

    def score(self, plan: CandidatePlan, state: AthleteState, validation: ValidationResult) -> CandidatePlan:
        plan.constraint_penalty = validation.penalty
        plan.expected_adaptation = _expected_adaptation(plan, state)
        plan.expected_fatigue = _expected_fatigue(plan, state)
        plan.overload_risk = _overload_risk(plan, state)
        if self.ml_result is not None and getattr(self.ml_result, "confidence", 0.0) >= 0.35:
            ml_predictions = getattr(self.ml_result, "predictions", {})
            ml_adaptation = ml_predictions.get("expected_adaptation")
            ml_overload = ml_predictions.get("overload_risk_modifier")
            if ml_adaptation is not None:
                plan.expected_adaptation = _clip(0.65 * plan.expected_adaptation + 0.35 * float(ml_adaptation))
            if ml_overload is not None:
                plan.overload_risk = _clip(0.75 * plan.overload_risk + 0.25 * float(ml_overload))
            plan.explanation.append("ML prediction blended into v2 score")
        plan.race_specificity = _race_specificity(plan, state)
        plan.missing_stimulus_score = _missing_stimulus_score(plan, state)
        plan.historical_driver_match = _historical_driver_match(plan, state)
        plan.total_score = (
            SCORING_WEIGHTS["expected_adaptation"] * plan.expected_adaptation
            + SCORING_WEIGHTS["missing_stimulus_score"] * plan.missing_stimulus_score
            + SCORING_WEIGHTS["race_specificity"] * plan.race_specificity
            + SCORING_WEIGHTS["historical_driver_match"] * plan.historical_driver_match
            + SCORING_WEIGHTS["expected_fatigue"] * plan.expected_fatigue
            + SCORING_WEIGHTS["overload_risk"] * plan.overload_risk
            + SCORING_WEIGHTS["constraint_penalty"] * plan.constraint_penalty
        )
        plan.explanation.extend(validation.soft_reasons)
        plan.explanation.extend(_selection_reasons(plan, state))
        return plan


def _expected_adaptation(plan: CandidatePlan, state: AthleteState) -> float:
    scores = [ADAPTATION.get(w.workout_type, 0.25) for w in plan.workouts]
    value = sum(scores) / max(1, len(scores))
    if _addresses_weak_system(plan, state):
        value += 0.15
    if state.current_fatigue > HIGH_FATIGUE and any(w.workout_type in {"threshold", "vo2max", "speed"} for w in plan.workouts):
        value -= 0.2
    if state.run_walk_enabled and state.run_walk_type == "planned" and plan.family == "run_walk_progression_week":
        value += 0.15
    return _clip(value)


def _expected_fatigue(plan: CandidatePlan, state: AthleteState) -> float:
    scale = state.horizon_days / 7
    load_ratio = plan.total_load / max(1.0, state.recent_28d_avg_load * scale or state.recent_7d_load * scale or 80.0 * scale)
    value = 0.2 * _clip(load_ratio) + 0.18 * _clip(plan.hard_days / max(1, state.max_hard_days_per_week + 1))
    if plan.total_distance_km and plan.long_run_km:
        value += 0.15 * _clip(plan.long_run_km / plan.total_distance_km / 0.4)
    value += 0.2 * _clip(state.current_fatigue / 100.0)
    value += 0.15 * _clip((state.forced_walk_score or 0.0))
    if plan.family == "recovery_week":
        value *= 0.65
    return _clip(value)


def _overload_risk(plan: CandidatePlan, state: AthleteState) -> float:
    scale = state.horizon_days / 7
    distance_jump = 0.0
    baseline_distance = state.recent_7d_distance_km * scale
    if baseline_distance > 0:
        distance_jump = max(0.0, (plan.total_distance_km - baseline_distance) / baseline_distance)
    load_jump = 0.0
    baseline_load = state.recent_7d_load * scale
    if baseline_load > 0:
        load_jump = max(0.0, (plan.total_load - baseline_load) / baseline_load)
    long_share = plan.long_run_km / plan.total_distance_km if plan.total_distance_km else 0.0
    value = (
        0.25 * _clip(distance_jump)
        + 0.2 * _clip(load_jump)
        + 0.15 * _clip(plan.hard_days / max(1, state.max_hard_days_per_week))
        + 0.15 * _clip(long_share / 0.4)
        + 0.2 * _clip(state.overload_risk)
        + 0.15 * _clip(state.forced_walk_score or 0.0)
    )
    if plan.family == "recovery_week":
        value *= 0.55
    return _clip(value)


def _race_specificity(plan: CandidatePlan, state: AthleteState) -> float:
    target = state.target_distance_km
    family = plan.family
    if target is None:
        return 0.65 if family in {"base_build_week", "maintenance_week"} else 0.45
    if target <= 5:
        preferred = {"vo2_focus_week", "threshold_focus_week", "speed_support_week"}
    elif target <= 10:
        preferred = {"threshold_focus_week", "vo2_focus_week", "base_build_week"}
    elif target <= 21.1:
        preferred = {"threshold_focus_week", "long_run_focus_week", "base_build_week"}
    else:
        preferred = {"long_run_focus_week", "base_build_week"}
    return 0.9 if family in preferred else 0.55 if family != "recovery_week" else 0.35


def _missing_stimulus_score(plan: CandidatePlan, state: AthleteState) -> float:
    if state.current_fatigue > HIGH_FATIGUE or state.overload_risk > 0.65:
        return 1.0 if plan.family == "recovery_week" else 0.25
    if state.run_walk_enabled and state.run_walk_type == "planned" and plan.family == "run_walk_progression_week":
        return 1.0
    if state.run_walk_enabled and state.run_walk_type in {"forced", "mixed"}:
        return 0.9 if plan.family in {"run_walk_progression_week", "recovery_week", "base_build_week"} else 0.2
    mapping = {
        "aerobic_base": "base_build_week",
        "threshold": "threshold_focus_week",
        "vo2max": "vo2_focus_week",
        "long_run_endurance": "long_run_focus_week",
        "recovery": "recovery_week",
        "speed": "speed_support_week",
        "unknown": "base_build_week",
    }
    return 1.0 if plan.family == mapping.get(state.weak_system) else 0.45


def _addresses_weak_system(plan: CandidatePlan, state: AthleteState) -> bool:
    return {
        "aerobic_base": "base_build_week",
        "threshold": "threshold_focus_week",
        "vo2max": "vo2_focus_week",
        "long_run_endurance": "long_run_focus_week",
        "recovery": "recovery_week",
        "speed": "speed_support_week",
    }.get(state.weak_system) == plan.family


def _historical_driver_match(plan: CandidatePlan, state: AthleteState) -> float:
    if state.historical_improvement_driver == "neutral" or state.historical_improvement_driver_confidence < 0.45:
        return 0.0
    if _historical_hard_signal_is_suppressed(plan, state):
        return 0.0
    family_matches = {
        "quality": {"threshold_focus_week", "vo2_focus_week", "speed_support_week"},
        "long_run": {"long_run_focus_week", "base_build_week"},
        "aerobic_base": {"base_build_week", "maintenance_week"},
        "recovery": {"recovery_week", "base_build_week"},
        "run_walk": {"run_walk_progression_week"} if state.run_walk_enabled else set(),
        "speed_economy": {"speed_support_week", "vo2_focus_week", "base_build_week"},
    }.get(state.historical_improvement_driver, set())
    if plan.family not in family_matches:
        return 0.0
    base = max(0.25, state.historical_improvement_driver_score)
    if state.historical_improvement_driver == "recovery" and state.current_fatigue < HIGH_FATIGUE and state.overload_risk < 0.5:
        base *= 0.5
    return _clip(base)


def _historical_hard_signal_is_suppressed(plan: CandidatePlan, state: AthleteState) -> bool:
    if state.historical_improvement_driver not in {"quality", "speed_economy"}:
        return False
    has_harder_run = any(w.workout_type in {"threshold", "vo2max", "speed"} for w in plan.workouts)
    if not has_harder_run:
        return False
    forced_walk = state.run_walk_enabled and state.run_walk_type in {"forced", "mixed"} and (state.forced_walk_score or 0.0) >= HIGH_FORCED_WALK_SCORE
    return state.current_fatigue >= HIGH_FATIGUE or state.overload_risk >= 0.65 or forced_walk


def _selection_reasons(plan: CandidatePlan, state: AthleteState) -> list[str]:
    reasons = []
    if _addresses_weak_system(plan, state):
        reasons.append(f"Addresses weak system: {state.weak_system}")
    if plan.hard_days <= state.max_hard_days_per_week:
        reasons.append("Keeps hard days within limit")
    if plan.has_recovery_day_after_hard:
        reasons.append("Includes recovery after quality session")
    if plan.family == "recovery_week":
        reasons.append("Prioritizes recovery and lower overload risk")
    if plan.family == "run_walk_progression_week":
        reasons.append("Includes run-walk-specific prescriptions")
    if plan.historical_driver_match > 0 and state.historical_improvement_driver_label:
        reasons.append(
            f"Historical improvement driver matched: {state.historical_improvement_driver_label} ({state.historical_improvement_driver_source})"
        )
    elif state.historical_improvement_driver in {"quality", "speed_economy"} and _historical_hard_signal_is_suppressed(plan, state):
        reasons.append("Historical hard-run signal suppressed by fatigue/overload safety rules")
    return reasons


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))
