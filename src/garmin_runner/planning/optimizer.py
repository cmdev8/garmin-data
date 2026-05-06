from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from garmin_runner.config import AthleteConfig, load_athlete_config
from garmin_runner.planning.candidates import CandidatePlanGenerator, infer_weak_system, latest_run_walk_type, mean_recent
from garmin_runner.planning.constraints import ConstraintValidator, rejection_summary
from garmin_runner.planning.historical import extract_historical_improvement_driver
from garmin_runner.planning.schema import (
    AthleteState,
    CandidatePlan,
    OptimizerResult,
    RunWalkType,
    WeakSystem,
    candidate_to_row,
    selected_plan_to_json,
)
from garmin_runner.planning.scoring import SCORING_WEIGHTS, PlanScorer
from garmin_runner.utils.io import read_table, write_json, write_table


class V2PlanOptimizer:
    def __init__(
        self,
        candidate_generator: CandidatePlanGenerator | None = None,
        validator: ConstraintValidator | None = None,
        scorer: PlanScorer | None = None,
    ):
        self.candidate_generator = candidate_generator or CandidatePlanGenerator()
        self.validator = validator or ConstraintValidator()
        self.scorer = scorer or PlanScorer()

    def optimize(self, state: AthleteState) -> OptimizerResult:
        candidates = self.candidate_generator.generate(state)
        evaluated: list[CandidatePlan] = []
        for plan in candidates:
            validation = self.validator.validate(plan, state)
            if not validation.is_valid:
                plan.is_valid = False
                plan.rejection_reasons = validation.hard_reasons
                plan.constraint_penalty = validation.penalty
                plan.explanation.extend(validation.soft_reasons)
                evaluated.append(plan)
                continue
            evaluated.append(self.scorer.score(plan, state, validation))

        valid = [plan for plan in evaluated if plan.is_valid]
        fallback_warning = None
        if not valid:
            selected = self._fallback_recovery(state)
            fallback_warning = "No valid candidates; selected fallback recovery plan."
            evaluated.append(selected)
        else:
            selected = max(valid, key=lambda p: p.total_score)

        selected.why_selected = selected.explanation[:5]
        historical_reasons = [reason for reason in selected.explanation if "Historical improvement driver" in reason]
        for reason in historical_reasons:
            if reason not in selected.why_selected:
                selected.why_selected.append(reason)
        debug = build_debug(state, evaluated, fallback_warning)
        if fallback_warning:
            selected.warnings.append(fallback_warning)
        return OptimizerResult(selected=selected, candidates=evaluated, debug=debug)

    def _fallback_recovery(self, state: AthleteState) -> CandidatePlan:
        generator = CandidatePlanGenerator()
        fallback = generator.generate(
            replace(
                state,
                current_fatigue=max(state.current_fatigue, 90.0),
                overload_risk=max(state.overload_risk, 0.9),
                weak_system="recovery",
            )
        )[0]
        fallback.plan_id = "fallback_recovery_001"
        fallback.family = "recovery_week"
        fallback.is_valid = True
        fallback.total_score = -1.0
        fallback.explanation.append("Fallback recovery week")
        return fallback


def build_athlete_state(
    *,
    config: AthleteConfig,
    activity_features: list[dict],
    weekly_features: list[dict],
    fitness_state: list[dict],
    performance_change: dict | None = None,
) -> AthleteState:
    latest_fitness = fitness_state[-1] if fitness_state else {}
    latest_week = weekly_features[-1] if weekly_features else {}
    recent_28d_avg_distance = _average_recent(weekly_features, "distance_7d_km")
    latest_load_7d = latest_week.get("fatigue_adjusted_load_7d") or latest_week.get("load_7d") or 0.0
    latest_load_28d = latest_week.get("fatigue_adjusted_load_28d") or latest_week.get("load_28d") or 0.0
    recent_28d_avg_load = (latest_load_28d or 0.0) / 4 if latest_week else _average_recent(weekly_features, "fatigue_adjusted_load_7d")
    latest_activity_date = _latest_date(activity_features, fitness_state)
    run_walk_type: RunWalkType = latest_run_walk_type(activity_features)
    weak_system: WeakSystem = infer_weak_system(weekly_features, activity_features)
    historical_driver = extract_historical_improvement_driver(performance_change)
    run_walk_planning_enabled = config.run_walk.enabled and config.run_walk.allow_walk_run_candidates
    return AthleteState(
        date=latest_activity_date,
        target_distance_km=config.goals.target_distance_km,
        target_date=config.goals.target_date,
        current_fitness=float(latest_fitness.get("fitness") or 0.0),
        current_fatigue=float(latest_fitness.get("fatigue") or 0.0),
        current_form=float(latest_fitness.get("form") or 0.0),
        overload_risk=float(latest_fitness.get("overload_risk") or 0.0),
        recent_7d_distance_km=float(latest_week.get("distance_7d_km") or 0.0),
        recent_28d_avg_distance_km=float(recent_28d_avg_distance or latest_week.get("distance_7d_km") or 0.0),
        recent_7d_load=float(latest_load_7d),
        recent_28d_avg_load=float(recent_28d_avg_load or latest_load_7d or 0.0),
        recent_easy_fraction=float(latest_week.get("easy_fraction") or 0.0),
        recent_moderate_fraction=float(latest_week.get("moderate_fraction") or 0.0),
        recent_hard_fraction=float(latest_week.get("hard_fraction") or 0.0),
        longest_recent_run_km=float(latest_week.get("long_run_distance_km") or 0.0),
        longest_continuous_run_s=latest_week.get("longest_continuous_run_s"),
        run_walk_enabled=run_walk_planning_enabled,
        run_walk_type=run_walk_type,
        average_run_segment_duration_s=mean_recent(activity_features, "average_run_segment_duration_s"),
        average_walk_segment_duration_s=mean_recent(activity_features, "average_walk_segment_duration_s"),
        forced_walk_score=mean_recent(activity_features, "forced_walk_score"),
        weak_system=weak_system,
        available_days=config.availability.available_days,
        preferred_long_run_day=config.availability.preferred_long_run_day,
        max_minutes_per_week=config.availability.max_minutes_per_week,
        max_volume_increase_pct=config.training.max_volume_increase_pct,
        max_hard_days_per_week=config.training.max_hard_days_per_week,
        horizon_days=config.planning.horizon_days,
        historical_improvement_driver=historical_driver.category,
        historical_improvement_driver_score=historical_driver.score,
        historical_improvement_driver_confidence=historical_driver.confidence,
        historical_improvement_driver_source=historical_driver.source_distance_range,
        historical_improvement_driver_label=historical_driver.label,
        historical_improvement_driver_reason=historical_driver.reason,
    )


def run_v2_optimizer(
    *,
    config: AthleteConfig,
    activity_features: list[dict],
    weekly_features: list[dict],
    fitness_state: list[dict],
    ml_result=None,
    performance_change: dict | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    state = build_athlete_state(
        config=config,
        activity_features=activity_features,
        weekly_features=weekly_features,
        fitness_state=fitness_state,
        performance_change=performance_change,
    )
    result = V2PlanOptimizer(scorer=PlanScorer(ml_result)).optimize(state)
    selected_json = selected_plan_to_json(result.selected, state)
    if out_dir is not None:
        out = Path(out_dir)
        write_table(out / "candidate_plans.parquet", [candidate_to_row(plan) for plan in result.candidates])
        write_json(out / "next_week_plan.json", selected_json)
        write_json(out / "plan_optimizer_debug.json", result.debug)
    return {
        "state": state,
        "result": result,
        "plan": selected_json,
        "candidate_rows": [candidate_to_row(plan) for plan in result.candidates],
        "debug": result.debug,
    }


def run_plan(
    *,
    input_dir: str | Path,
    athlete_config: str | Path | None = None,
    optimizer: str = "v2",
    planning_horizon_days: int | None = None,
) -> dict:
    if optimizer != "v2":
        raise ValueError("standalone plan command currently supports optimizer='v2'")
    config = load_athlete_config(athlete_config, planning_horizon_days=planning_horizon_days)
    root = Path(input_dir)
    return run_v2_optimizer(
        config=config,
        activity_features=read_table(root / "activity_features.parquet"),
        weekly_features=read_table(root / "weekly_features.parquet"),
        fitness_state=read_table(root / "fitness_state.parquet"),
        performance_change=_read_optional_json(root / "performance_change.json"),
        out_dir=root,
    )


def build_debug(state: AthleteState, candidates: list[CandidatePlan], fallback_warning: str | None = None) -> dict:
    valid = [plan for plan in candidates if plan.is_valid]
    rejected = [plan for plan in candidates if not plan.is_valid]
    top = sorted(valid, key=lambda plan: plan.total_score, reverse=True)[:5]
    debug = {
        "athlete_state": asdict(state),
        "generated_candidate_count": len(candidates),
        "valid_candidate_count": len(valid),
        "rejected_candidate_count": len(rejected),
        "top_5_plan_ids": [plan.plan_id for plan in top],
        "rejection_summary": rejection_summary(rejected),
        "scoring_weights": SCORING_WEIGHTS,
        "historical_improvement_driver": {
            "category": state.historical_improvement_driver,
            "label": state.historical_improvement_driver_label,
            "score": round(state.historical_improvement_driver_score, 3),
            "confidence": round(state.historical_improvement_driver_confidence, 3),
            "source_distance_range": state.historical_improvement_driver_source,
            "reason": state.historical_improvement_driver_reason,
        },
    }
    if fallback_warning:
        debug["fallback_warning"] = fallback_warning
    return debug


def _read_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _average_recent(rows: list[dict], field: str) -> float:
    values: list[float] = []
    for row in rows[-28:]:
        value = row.get(field)
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else 0.0


def _latest_date(activity_features: list[dict], fitness_state: list[dict]) -> str:
    dates: list[str] = []
    for row in activity_features:
        value = row.get("date")
        if value is not None:
            dates.append(str(value))
    for row in fitness_state:
        value = row.get("date")
        if value is not None:
            dates.append(str(value))
    if dates:
        return max(dates)
    from datetime import date

    return date.today().isoformat()
