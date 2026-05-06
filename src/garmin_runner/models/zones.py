from __future__ import annotations

from statistics import mean

from garmin_runner.config import AthleteConfig


def estimate_zones(config: AthleteConfig, activity_features: list[dict]) -> dict:
    return {
        "hr_zones": _hr_zones(config, activity_features),
        "pace_zones": _pace_zones(config, activity_features),
    }


def _hr_zones(config: AthleteConfig, activity_features: list[dict]) -> dict:
    threshold = config.physiology.lactate_threshold_hr
    max_hr = config.physiology.max_hr
    resting = config.physiology.resting_hr or 60
    if threshold is None:
        hard_hrs = [r["max_hr"] for r in activity_features if r.get("max_hr")]
        threshold = int((max(hard_hrs) * 0.9) if hard_hrs else ((max_hr or 190) * 0.88))
    if max_hr is None:
        max_hr = int(threshold / 0.9)
    return {
        "Z1": {"min_bpm": resting, "max_bpm": int(threshold * 0.80)},
        "Z2": {"min_bpm": int(threshold * 0.80), "max_bpm": int(threshold * 0.88)},
        "Z3": {"min_bpm": int(threshold * 0.88), "max_bpm": int(threshold * 0.95)},
        "Z4": {"min_bpm": int(threshold * 0.95), "max_bpm": int(threshold * 1.02)},
        "Z5": {"min_bpm": int(threshold * 1.02), "max_bpm": max_hr},
    }


def _pace_zones(config: AthleteConfig, activity_features: list[dict]) -> dict:
    threshold = config.physiology.lactate_threshold_pace_s_per_km
    if threshold is None:
        candidates_raw = [
            r.get("run_pace_only_s_per_km") or r.get("best_rolling_5km_pace") or r.get("avg_pace_s_per_km")
            for r in activity_features
        ]
        candidates = [float(c) for c in candidates_raw if c]
        threshold = mean(sorted(candidates)[: max(1, len(candidates) // 3)]) if candidates else 360.0
    threshold = float(threshold)
    return {
        "recovery": {"min_s_per_km": threshold * 1.35, "max_s_per_km": threshold * 1.20},
        "easy": {"min_s_per_km": threshold * 1.20, "max_s_per_km": threshold * 1.10},
        "steady": {"min_s_per_km": threshold * 1.10, "max_s_per_km": threshold * 1.03},
        "marathon": {"min_s_per_km": threshold * 1.08, "max_s_per_km": threshold * 1.00},
        "threshold": {"min_s_per_km": threshold * 1.03, "max_s_per_km": threshold * 0.97},
        "VO2": {"min_s_per_km": threshold * 0.97, "max_s_per_km": threshold * 0.90},
        "speed": {"min_s_per_km": threshold * 0.90, "max_s_per_km": threshold * 0.80},
    }
