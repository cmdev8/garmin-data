from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev

from garmin_runner.fit.schema import Activity, Record
from garmin_runner.run_walk.classifier import classify_run_walk
from garmin_runner.run_walk.fatigue import run_walk_fatigue_multiplier


def build_activity_features(
    activities: list[Activity],
    records: list[Record],
    run_walk_metrics_by_activity: dict[str, dict],
) -> list[dict]:
    by_activity: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_activity[record.activity_id].append(record)

    rows = []
    for activity in activities:
        if not activity.keep:
            continue
        activity_records = sorted(by_activity.get(activity.activity_id, []), key=lambda r: r.elapsed_s)
        metrics = run_walk_metrics_by_activity.get(activity.activity_id, {})
        run_walk_type, confidence, evidence = classify_run_walk(metrics)
        distance_m = activity.total_distance_m or _distance(activity_records)
        moving_s = activity.total_moving_s or _moving_time(activity_records)
        elapsed_s = activity.total_elapsed_s or (activity_records[-1].elapsed_s if activity_records else None)
        hrs = [r.heart_rate_bpm for r in activity_records if r.heart_rate_bpm is not None]
        cadences = [r.cadence_spm for r in activity_records if r.cadence_spm is not None]
        powers = [r.power_w for r in activity_records if r.power_w is not None]
        temperatures = [r.temperature_c for r in activity_records if r.temperature_c is not None]
        vertical_oscillations = [r.vertical_oscillation_mm for r in activity_records if r.vertical_oscillation_mm is not None]
        step_lengths = [r.step_length_mm for r in activity_records if r.step_length_mm is not None]
        vertical_ratios = [r.vertical_ratio for r in activity_records if r.vertical_ratio is not None]
        stance_times = [r.stance_time_ms for r in activity_records if r.stance_time_ms is not None]
        speeds = [r.speed_mps for r in activity_records if r.speed_mps is not None and r.speed_mps > 0]
        gps_records = [r for r in activity_records if r.position_lat is not None and r.position_long is not None]
        avg_hr = activity.avg_hr or (mean(hrs) if hrs else None)
        max_hr = activity.max_hr or (max(hrs) if hrs else None)
        min_hr = activity.min_hr or (min(hrs) if hrs else None)
        avg_power = activity.avg_power_w or (mean(powers) if powers else None)
        normalized_power = activity.normalized_power_w
        avg_cadence = activity.avg_cadence_spm or activity.avg_running_cadence_spm or (mean(cadences) if cadences else None)
        elevation_gain = activity.total_ascent_m or _elevation_gain(activity_records)
        elevation_loss = activity.total_descent_m
        altitude_range = _altitude_range(activity, activity_records)

        row = {
            "activity_id": activity.activity_id,
            "date": activity.start_time.date().isoformat() if activity.start_time else None,
            "start_time": activity.start_time.isoformat() if activity.start_time else None,
            "distance_km": (distance_m or 0.0) / 1000.0,
            "elapsed_time_min": (elapsed_s or 0.0) / 60.0,
            "moving_time_min": (moving_s or 0.0) / 60.0,
            "avg_pace_s_per_km": (moving_s / (distance_m / 1000.0)) if distance_m and moving_s else None,
            "best_rolling_1km_pace": _best_rolling_pace(activity_records, 1000.0),
            "best_rolling_5km_pace": _best_rolling_pace(activity_records, 5000.0),
            "best_rolling_10km_pace": _best_rolling_pace(activity_records, 10000.0),
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "min_hr": min_hr,
            "hr_reserve_proxy": (avg_hr - min_hr) if avg_hr is not None and min_hr is not None else None,
            "avg_max_hr_ratio": avg_hr / max_hr if avg_hr is not None and max_hr else None,
            "min_max_hr_ratio": min_hr / max_hr if min_hr is not None and max_hr else None,
            "avg_cadence": avg_cadence,
            "max_running_cadence": activity.max_running_cadence_spm or (max(cadences) if cadences else None),
            "elevation_gain": elevation_gain,
            "elevation_loss": elevation_loss,
            "elevation_gain_per_km": elevation_gain / (distance_m / 1000.0) if elevation_gain is not None and distance_m else None,
            "altitude_range_m": altitude_range,
            "total_calories": activity.total_calories,
            "calories_per_km": activity.total_calories / (distance_m / 1000.0) if activity.total_calories is not None and distance_m else None,
            "calories_per_min": activity.total_calories / (moving_s / 60.0) if activity.total_calories is not None and moving_s else None,
            "total_training_effect": activity.total_training_effect,
            "total_anaerobic_training_effect": activity.total_anaerobic_training_effect,
            "avg_temperature_c": activity.avg_temperature_c or (mean(temperatures) if temperatures else None),
            "max_temperature_c": activity.max_temperature_c or (max(temperatures) if temperatures else None),
            "avg_power_w": avg_power,
            "max_power_w": activity.max_power_w or (max(powers) if powers else None),
            "normalized_power_w": normalized_power,
            "power_variability": normalized_power / avg_power if normalized_power is not None and avg_power else _cv(powers),
            "total_work_kj": activity.total_work_kj,
            "work_kj_per_km": activity.total_work_kj / (distance_m / 1000.0) if activity.total_work_kj is not None and distance_m else None,
            "power_hr_efficiency": avg_power / avg_hr if avg_power is not None and avg_hr else None,
            "total_strides": activity.total_strides,
            "stride_length_mm": activity.avg_step_length_mm or (mean(step_lengths) if step_lengths else None),
            "vertical_oscillation_mm": activity.avg_vertical_oscillation_mm or (mean(vertical_oscillations) if vertical_oscillations else None),
            "vertical_ratio": activity.avg_vertical_ratio or (mean(vertical_ratios) if vertical_ratios else None),
            "stance_time_ms": activity.avg_stance_time_ms or (mean(stance_times) if stance_times else None),
            "pace_variability": _cv([1000.0 / speed for speed in speeds]),
            "hr_variability": _cv(hrs),
            "cadence_variability": _cv(cadences),
            "power_record_variability": _cv(powers),
            "lap_count": activity.lap_count,
            "lap_pace_variability": activity.lap_pace_variability,
            "lap_hr_variability": activity.lap_hr_variability,
            "lap_power_variability": activity.lap_power_variability,
            "gps_coverage": len(gps_records) / len(activity_records) if activity_records else 0.0,
            "hrv_count": activity.hrv_count,
            "hrv_median_rr_ms": activity.hrv_median_rr_ms,
            "hrv_rmssd_ms": activity.hrv_rmssd_ms,
            "hr_drift_proxy": _hr_drift(activity_records),
            "decoupling_proxy": _decoupling(activity_records),
            "data_quality_score": _data_quality(activity_records),
            "run_walk_type": run_walk_type,
            "run_walk_confidence": confidence,
            "run_walk_evidence": "; ".join(evidence),
        }
        row.update(metrics)
        row["run_walk_fatigue_multiplier"] = run_walk_fatigue_multiplier(
            run_walk_type,
            row.get("walk_fraction"),
            row.get("planned_run_walk_score"),
        )
        rows.append(row)
    return rows


def _distance(records: list[Record]) -> float:
    distances = [r.distance_m for r in records if r.distance_m is not None]
    return max(distances) - min(distances) if len(distances) >= 2 else 0.0


def _moving_time(records: list[Record]) -> float:
    if len(records) < 2:
        return 0.0
    moving = 0.0
    for previous, current in zip(records, records[1:]):
        if current.speed_mps is None or current.speed_mps > 0.25:
            moving += max(0.0, current.elapsed_s - previous.elapsed_s)
    return moving


def _best_rolling_pace(records: list[Record], window_m: float) -> float | None:
    with_distance = [r for r in records if r.distance_m is not None]
    best = None
    start = 0
    for end, record in enumerate(with_distance):
        while start < end and (record.distance_m or 0) - (with_distance[start].distance_m or 0) >= window_m:
            duration = record.elapsed_s - with_distance[start].elapsed_s
            pace = duration / (window_m / 1000.0) if duration > 0 else None
            if pace is not None:
                best = pace if best is None else min(best, pace)
            start += 1
    return best


def _elevation_gain(records: list[Record]) -> float | None:
    alts = [r.altitude_m for r in records if r.altitude_m is not None]
    if len(alts) < 2:
        return None
    return sum(max(0.0, b - a) for a, b in zip(alts, alts[1:]))


def _altitude_range(activity: Activity, records: list[Record]) -> float | None:
    if activity.min_altitude_m is not None and activity.max_altitude_m is not None:
        return max(0.0, activity.max_altitude_m - activity.min_altitude_m)
    alts = [r.altitude_m for r in records if r.altitude_m is not None]
    return max(alts) - min(alts) if len(alts) >= 2 else None


def _hr_drift(records: list[Record]) -> float | None:
    hrs = [r.heart_rate_bpm for r in records if r.heart_rate_bpm is not None]
    if len(hrs) < 10:
        return None
    half = len(hrs) // 2
    return mean(hrs[half:]) - mean(hrs[:half])


def _decoupling(records: list[Record]) -> float | None:
    first = [(r.speed_mps, r.heart_rate_bpm) for r in records[: len(records) // 2] if r.speed_mps is not None and r.heart_rate_bpm is not None]
    second = [(r.speed_mps, r.heart_rate_bpm) for r in records[len(records) // 2 :] if r.speed_mps is not None and r.heart_rate_bpm is not None]
    if not first or not second:
        return None
    first_ratio = mean([speed / hr for speed, hr in first])
    second_ratio = mean([speed / hr for speed, hr in second])
    return (first_ratio - second_ratio) / first_ratio if first_ratio else None


def _data_quality(records: list[Record]) -> float:
    if not records:
        return 0.0
    score = 0.25
    score += 0.25 if any(r.heart_rate_bpm is not None for r in records) else 0
    score += 0.25 if any(r.cadence_spm is not None for r in records) else 0
    score += 0.25 if any(r.distance_m is not None for r in records) else 0
    return score


def _cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = mean(values)
    return pstdev(values) / avg if avg else None
