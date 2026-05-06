from __future__ import annotations

from garmin_runner.planning.schema import Intensity, Workout, WorkoutType


LOAD_MULTIPLIERS: dict[WorkoutType, float] = {
    "rest": 0.0,
    "recovery": 5.0,
    "easy": 8.0,
    "long_run": 9.0,
    "threshold": 13.0,
    "vo2max": 14.0,
    "speed": 11.0,
}


PURPOSES: dict[WorkoutType, str] = {
    "rest": "Absorb recent training",
    "recovery": "Reduce fatigue while preserving consistency",
    "easy": "Build aerobic base with low stress",
    "long_run": "Improve durability and long-run endurance",
    "threshold": "Improve sustainable speed",
    "vo2max": "Improve 1-10 km aerobic ceiling",
    "speed": "Improve neuromuscular economy",
}


DESCRIPTIONS: dict[WorkoutType, str] = {
    "rest": "Rest day",
    "recovery": "Short recovery run",
    "easy": "Easy aerobic run",
    "long_run": "Long easy run",
    "threshold": "Warm up, then controlled threshold intervals, cool down",
    "vo2max": "Warm up, then short VO2max intervals with equal easy recovery",
    "speed": "Easy run with short relaxed strides",
}


def make_workout(
    *,
    date: str,
    day: str,
    workout_type: WorkoutType,
    duration_min: int,
    distance_km: float | None,
    run_walk_prescription: str | None = None,
    run_walk_fatigue_multiplier: float = 1.0,
) -> Workout:
    intensity: Intensity = "rest"
    if workout_type in {"recovery", "easy", "long_run"}:
        intensity = "easy"
    elif workout_type == "speed":
        intensity = "moderate"
    elif workout_type in {"threshold", "vo2max"}:
        intensity = "hard"

    distance = None if distance_km is None else max(0.0, float(distance_km))
    load = 0.0 if distance is None else distance * LOAD_MULTIPLIERS.get(workout_type, 8.0)
    if workout_type in {"threshold", "vo2max", "speed"}:
        load += duration_min * 0.25
    fatigue_multiplier = min(1.0, max(0.75, float(run_walk_fatigue_multiplier)))
    load *= fatigue_multiplier

    return Workout(
        date=date,
        day=day,
        workout_type=workout_type,
        duration_min=max(0, int(duration_min)),
        distance_km=distance,
        intensity=intensity,
        load_estimate=round(load, 2),
        description=DESCRIPTIONS.get(workout_type, workout_type.replace("_", " ").title()),
        purpose=PURPOSES.get(workout_type, "Training stimulus"),
        run_walk_prescription=run_walk_prescription,
        run_walk_fatigue_multiplier=fatigue_multiplier,
    )


def run_walk_prescription(run_minutes: int, walk_seconds: int, reps: int, *, forced: bool = False) -> str:
    if forced:
        return f"{reps} x [{run_minutes} min easy run + {walk_seconds} sec walk], keep HR in Z2 and keep the route flat"
    return f"{reps} x [{run_minutes} min easy run + {walk_seconds} sec walk]"
