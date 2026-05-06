from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


WeakSystem = Literal["aerobic_base", "threshold", "vo2max", "long_run_endurance", "recovery", "speed", "unknown"]
RunWalkType = Literal["planned", "forced", "mixed", "not_run_walk", "unknown"]
WorkoutType = Literal["rest", "recovery", "easy", "long_run", "threshold", "vo2max", "speed", "strength", "cross_train"]
Intensity = Literal["rest", "easy", "moderate", "hard"]
HistoricalDriver = Literal["quality", "long_run", "aerobic_base", "recovery", "run_walk", "speed_economy", "neutral"]


@dataclass
class AthleteState:
    date: str
    target_distance_km: float | None
    target_date: str | None
    current_fitness: float
    current_fatigue: float
    current_form: float
    overload_risk: float
    recent_7d_distance_km: float
    recent_28d_avg_distance_km: float
    recent_7d_load: float
    recent_28d_avg_load: float
    recent_easy_fraction: float
    recent_moderate_fraction: float
    recent_hard_fraction: float
    longest_recent_run_km: float
    longest_continuous_run_s: float | None
    run_walk_enabled: bool
    run_walk_type: RunWalkType
    average_run_segment_duration_s: float | None
    average_walk_segment_duration_s: float | None
    forced_walk_score: float | None
    weak_system: WeakSystem
    available_days: list[str]
    preferred_long_run_day: str
    max_minutes_per_week: int | None
    max_volume_increase_pct: float
    max_hard_days_per_week: int
    horizon_days: int = 7
    historical_improvement_driver: HistoricalDriver = "neutral"
    historical_improvement_driver_score: float = 0.0
    historical_improvement_driver_confidence: float = 0.0
    historical_improvement_driver_source: str | None = None
    historical_improvement_driver_label: str | None = None
    historical_improvement_driver_reason: str | None = None


@dataclass
class Workout:
    day: str
    workout_type: WorkoutType
    duration_min: int
    distance_km: float | None
    intensity: Intensity
    load_estimate: float
    description: str
    purpose: str
    run_walk_prescription: str | None = None
    run_walk_fatigue_multiplier: float = 1.0
    date: str | None = None


@dataclass
class CandidatePlan:
    plan_id: str
    family: str
    workouts: list[Workout]
    total_duration_min: int
    total_distance_km: float
    total_load: float
    easy_fraction: float
    moderate_fraction: float
    hard_fraction: float
    hard_days: int
    long_run_km: float
    has_recovery_day_after_hard: bool
    expected_adaptation: float = 0.0
    expected_fatigue: float = 0.0
    overload_risk: float = 0.0
    race_specificity: float = 0.0
    missing_stimulus_score: float = 0.0
    historical_driver_match: float = 0.0
    constraint_penalty: float = 0.0
    total_score: float = 0.0
    explanation: list[str] = field(default_factory=list)
    is_valid: bool = True
    rejection_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    why_selected: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    is_valid: bool
    hard_reasons: list[str]
    penalty: float
    soft_reasons: list[str]


@dataclass
class OptimizerResult:
    selected: CandidatePlan
    candidates: list[CandidatePlan]
    debug: dict


def workout_to_dict(workout: Workout) -> dict:
    data = asdict(workout)
    if data["distance_km"] is not None:
        data["distance_km"] = round(data["distance_km"], 2)
    return data


def candidate_to_row(plan: CandidatePlan) -> dict:
    return {
        "plan_id": plan.plan_id,
        "family": plan.family,
        "total_duration_min": plan.total_duration_min,
        "total_distance_km": round(plan.total_distance_km, 2),
        "total_load": round(plan.total_load, 2),
        "easy_fraction": round(plan.easy_fraction, 3),
        "moderate_fraction": round(plan.moderate_fraction, 3),
        "hard_fraction": round(plan.hard_fraction, 3),
        "hard_days": plan.hard_days,
        "long_run_km": round(plan.long_run_km, 2),
        "expected_adaptation": round(plan.expected_adaptation, 3),
        "expected_fatigue": round(plan.expected_fatigue, 3),
        "overload_risk": round(plan.overload_risk, 3),
        "race_specificity": round(plan.race_specificity, 3),
        "missing_stimulus_score": round(plan.missing_stimulus_score, 3),
        "historical_driver_match": round(plan.historical_driver_match, 3),
        "constraint_penalty": round(plan.constraint_penalty, 3),
        "total_score": round(plan.total_score, 3),
        "is_valid": plan.is_valid,
        "rejection_reasons": "; ".join(plan.rejection_reasons),
        "explanation": "; ".join(plan.explanation),
    }


def selected_plan_to_json(plan: CandidatePlan, state: AthleteState) -> dict:
    workouts = [workout_to_dict(workout) for workout in plan.workouts]
    dates = [w["date"] for w in workouts if w.get("date")]
    return {
        "selected_plan_id": plan.plan_id,
        "family": plan.family,
        "score": round(plan.total_score, 3),
        "horizon_days": state.horizon_days,
        "start_date": min(dates) if dates else None,
        "end_date": max(dates) if dates else None,
        "why_selected": plan.why_selected or plan.explanation[:4],
        "warnings": plan.warnings,
        "week": workouts,
        "days": workouts,
        "summary": {
            "total_duration_min": plan.total_duration_min,
            "total_distance_km": round(plan.total_distance_km, 2),
            "total_load": round(plan.total_load, 2),
            "hard_days": plan.hard_days,
            "long_run_km": round(plan.long_run_km, 2),
        },
        "historical_driver": {
            "category": state.historical_improvement_driver,
            "label": state.historical_improvement_driver_label,
            "score": round(state.historical_improvement_driver_score, 3),
            "confidence": round(state.historical_improvement_driver_confidence, 3),
            "source_distance_range": state.historical_improvement_driver_source,
            "reason": state.historical_improvement_driver_reason,
        },
    }
