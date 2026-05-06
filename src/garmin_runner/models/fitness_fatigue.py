from __future__ import annotations


def build_fitness_state(daily_features: list[dict]) -> list[dict]:
    fitness = 0.0
    fatigue = 0.0
    rows = []
    for day in daily_features:
        load = day.get("load_estimate") or 0.0
        adjusted_load = day.get("fatigue_adjusted_load_estimate")
        if adjusted_load is None:
            adjusted_load = load
        intensity_bonus = (day.get("hard_time_min") or 0.0) * 0.8 + (day.get("moderate_time_min") or 0.0) * 0.3
        long_run_score = min(25.0, (day.get("total_distance_km") or 0.0) * 1.5)
        fitness_stimulus = load + intensity_bonus + long_run_score
        fatigue_stimulus = adjusted_load + intensity_bonus + long_run_score
        fitness = fitness * 0.965 + fitness_stimulus * 0.035
        fatigue = fatigue * 0.82 + fatigue_stimulus * 0.18
        form = fitness - fatigue
        rows.append(
            {
                "date": day["date"],
                "daily_load": load,
                "fatigue_adjusted_daily_load": adjusted_load,
                "fitness": fitness,
                "fatigue": fatigue,
                "form": form,
                "predicted_performance_index": max(1.0, fitness + form * 0.25),
                "overload_risk": min(1.0, max(0.0, (fatigue - fitness) / 50.0)),
            }
        )
    return rows
