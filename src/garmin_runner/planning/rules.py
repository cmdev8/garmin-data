from __future__ import annotations

import math
from datetime import date, timedelta

from garmin_runner.config import AthleteConfig


def generate_plan(
    config: AthleteConfig,
    activity_features: list[dict],
    weekly_features: list[dict],
    fitness_state: list[dict],
) -> dict:
    horizon = config.planning.horizon_days
    start = _start_date(activity_features)
    end = start + timedelta(days=horizon - 1)
    latest_week = weekly_features[-1] if weekly_features else {}
    latest_state = fitness_state[-1] if fitness_state else {}
    recent_distance = latest_week.get("distance_7d_km") or sum(r.get("distance_km") or 0.0 for r in activity_features[-7:])
    recent_minutes = latest_week.get("time_7d_min") or sum(r.get("moving_time_min") or 0.0 for r in activity_features[-7:])
    fatigue = latest_state.get("fatigue") or 0.0
    fitness = latest_state.get("fitness") or 0.0
    high_fatigue = fatigue > fitness * 1.15 and fatigue > 20

    hard_limit = math.ceil(config.training.max_hard_days_per_week * horizon / 7)
    hard_limit = max(0, hard_limit - (1 if high_fatigue and hard_limit > 0 else 0))
    volume_cap = recent_distance * (1 + config.training.max_volume_increase_pct / 100.0) * horizon / 7
    if recent_distance <= 0:
        volume_cap = max(5.0, 6.0 * horizon / 7)
    minute_cap = None
    if config.availability.max_minutes_per_week:
        minute_cap = config.availability.max_minutes_per_week * horizon / 7
    if recent_minutes <= 0:
        recent_minutes = minute_cap or 45.0 * horizon / 7

    days = []
    hard_days_used = 0
    total_distance = 0.0
    total_minutes = 0
    previous_hard = False
    available = set(config.availability.available_days)
    long_run_scheduled = False

    for offset in range(horizon):
        current = start + timedelta(days=offset)
        weekday = current.strftime("%A")
        remaining_days = horizon - offset
        remaining_available = max(1, _count_available(start, offset, horizon, available))
        is_available = weekday in available
        workout = _rest_day(current, "Unavailable day" if not is_available else "Rest")
        if is_available:
            remaining_distance = max(0.0, volume_cap - total_distance)
            remaining_minutes = max(0.0, (minute_cap if minute_cap is not None else recent_minutes * 1.1 * horizon / 7) - total_minutes)
            base_distance = remaining_distance / remaining_available if remaining_available else 0.0
            base_minutes = int(max(20, remaining_minutes / remaining_available)) if remaining_minutes else 0

            if high_fatigue:
                workout = _run_day(current, "recovery", base_minutes, base_distance * 0.75, "easy", config, run_walk=True)
            elif weekday == config.availability.preferred_long_run_day and remaining_days >= 1:
                workout = _run_day(current, "long_run", base_minutes + 10, base_distance * 1.35, "easy", config, run_walk=True)
                long_run_scheduled = True
            elif hard_days_used < hard_limit and not previous_hard and _should_place_hard(offset, horizon):
                workout = _run_day(current, "threshold", base_minutes, base_distance, "hard", config, run_walk=False)
                hard_days_used += 1
            else:
                workout = _run_day(current, "easy", base_minutes, base_distance, "easy", config, run_walk=True)

        workout["distance_km"] = round(min(workout["distance_km"], max(0.0, volume_cap - total_distance)), 2)
        if minute_cap is not None:
            workout["duration_min"] = int(min(workout["duration_min"], max(0.0, minute_cap - total_minutes)))
        total_distance += workout["distance_km"]
        total_minutes += workout["duration_min"]
        previous_hard = workout["intensity"] == "hard"
        days.append(workout)

    if not long_run_scheduled and not high_fatigue:
        _promote_last_available_to_long_run(days, config)

    return {
        "horizon_days": horizon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "days": days,
        "summary": {
            "total_distance_km": round(sum(d["distance_km"] for d in days), 2),
            "total_duration_min": sum(d["duration_min"] for d in days),
            "hard_days": sum(1 for d in days if d["intensity"] == "hard"),
            "available_training_days": sum(1 for d in days if d["workout_type"] != "rest"),
        },
        "constraints_applied": {
            "max_hard_days": hard_limit,
            "volume_cap_km": round(volume_cap, 2),
            "max_minutes": int(minute_cap) if minute_cap is not None else None,
            "high_fatigue_recovery_bias": high_fatigue,
            "available_days": config.availability.available_days,
        },
        "rationale": _rationale(high_fatigue, hard_limit, horizon),
    }


def _start_date(activity_features: list[dict]) -> date:
    dates = [date.fromisoformat(r["date"]) for r in activity_features if r.get("date")]
    return (max(dates) + timedelta(days=1)) if dates else date.today() + timedelta(days=1)


def _count_available(start: date, offset: int, horizon: int, available: set[str]) -> int:
    return sum(1 for i in range(offset, horizon) if (start + timedelta(days=i)).strftime("%A") in available)


def _should_place_hard(offset: int, horizon: int) -> bool:
    return offset in {1, 4, 8, 11, 15, 18, 22, 25} and offset < horizon


def _rest_day(day: date, purpose: str) -> dict:
    return {
        "date": day.isoformat(),
        "day": day.strftime("%A"),
        "workout_type": "rest",
        "duration_min": 0,
        "distance_km": 0.0,
        "intensity": "rest",
        "description": "Rest",
        "purpose": purpose,
        "run_walk_prescription": None,
    }


def _run_day(day: date, workout_type: str, minutes: int, distance: float, intensity: str, config: AthleteConfig, *, run_walk: bool) -> dict:
    prescription = None
    if config.run_walk.enabled and run_walk:
        prescription = "Use comfortable run-walk breaks; progress run duration only if form stays controlled."
    return {
        "date": day.isoformat(),
        "day": day.strftime("%A"),
        "workout_type": workout_type,
        "duration_min": max(0, int(minutes)),
        "distance_km": max(0.0, float(distance)),
        "intensity": intensity,
        "description": workout_type.replace("_", " ").title(),
        "purpose": _purpose(workout_type),
        "run_walk_prescription": prescription,
    }


def _purpose(workout_type: str) -> str:
    return {
        "recovery": "reduce fatigue while preserving routine",
        "easy": "aerobic base",
        "long_run": "long-run endurance",
        "threshold": "threshold stimulus without back-to-back hard days",
    }.get(workout_type, "training stimulus")


def _promote_last_available_to_long_run(days: list[dict], config: AthleteConfig) -> None:
    candidates = [day for day in days if day["workout_type"] not in {"rest", "threshold"}]
    if not candidates:
        return
    preferred = [day for day in candidates if day["day"] == config.availability.preferred_long_run_day]
    target = (preferred or candidates)[-1]
    target["workout_type"] = "long_run"
    target["description"] = "Long Run"
    target["purpose"] = "long-run endurance"
    target["distance_km"] = round(target["distance_km"] * 1.2, 2)


def _rationale(high_fatigue: bool, hard_limit: int, horizon: int) -> list[str]:
    rationale = [f"Generated for a {horizon}-day planning horizon."]
    rationale.append(f"Hard-day allowance scaled to {hard_limit} for the horizon.")
    if high_fatigue:
        rationale.append("Fatigue is high, so the plan emphasizes recovery and easy volume.")
    else:
        rationale.append("Plan balances aerobic volume with spaced quality work.")
    return rationale
