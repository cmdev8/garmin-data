from __future__ import annotations

from datetime import date, timedelta
from statistics import mean, pstdev


DAILY_AVERAGE_FIELDS = [
    "hr_drift_proxy",
    "decoupling_proxy",
    "data_quality_score",
    "avg_hr",
    "max_hr",
    "min_hr",
    "hr_reserve_proxy",
    "avg_max_hr_ratio",
    "avg_cadence",
    "max_running_cadence",
    "avg_power_w",
    "max_power_w",
    "normalized_power_w",
    "power_variability",
    "power_hr_efficiency",
    "avg_temperature_c",
    "max_temperature_c",
    "elevation_gain_per_km",
    "altitude_range_m",
    "calories_per_km",
    "calories_per_min",
    "total_training_effect",
    "total_anaerobic_training_effect",
    "stride_length_mm",
    "vertical_oscillation_mm",
    "vertical_ratio",
    "stance_time_ms",
    "pace_variability",
    "hr_variability",
    "cadence_variability",
    "power_record_variability",
    "lap_pace_variability",
    "lap_hr_variability",
    "lap_power_variability",
    "gps_coverage",
    "hrv_median_rr_ms",
    "hrv_rmssd_ms",
]

DAILY_SUM_FIELDS = ["total_calories", "total_work_kj", "total_strides", "lap_count", "hrv_count"]

WEEKLY_MEAN_FIELDS = [
    "avg_hr",
    "max_hr",
    "min_hr",
    "hr_reserve_proxy",
    "hr_drift_proxy",
    "decoupling_proxy",
    "data_quality_score",
    "avg_cadence",
    "max_running_cadence",
    "avg_power_w",
    "normalized_power_w",
    "power_variability",
    "power_hr_efficiency",
    "avg_temperature_c",
    "max_temperature_c",
    "elevation_gain_per_km",
    "altitude_range_m",
    "calories_per_km",
    "calories_per_min",
    "total_training_effect",
    "total_anaerobic_training_effect",
    "stride_length_mm",
    "vertical_oscillation_mm",
    "vertical_ratio",
    "stance_time_ms",
    "pace_variability",
    "hr_variability",
    "cadence_variability",
    "power_record_variability",
    "lap_pace_variability",
    "lap_hr_variability",
    "lap_power_variability",
    "gps_coverage",
    "hrv_median_rr_ms",
    "hrv_rmssd_ms",
    "run_walk_fatigue_multiplier",
]

WEEKLY_SUM_FIELDS = ["total_calories", "total_work_kj", "total_strides", "lap_count", "hrv_count"]


def build_daily_features(activity_features: list[dict]) -> list[dict]:
    by_date: dict[str, list[dict]] = {}
    for row in activity_features:
        if row.get("date"):
            by_date.setdefault(row["date"], []).append(row)
    if not by_date:
        return []

    start = date.fromisoformat(min(by_date))
    end = date.fromisoformat(max(by_date))
    rows = []
    current = start
    while current <= end:
        key = current.isoformat()
        items = by_date.get(key, [])
        distance = sum(i.get("distance_km") or 0.0 for i in items)
        moving = sum(i.get("moving_time_min") or 0.0 for i in items)
        hard = sum((i.get("moving_time_min") or 0.0) for i in items if (i.get("avg_hr") or 0) >= 160)
        moderate = sum((i.get("moving_time_min") or 0.0) for i in items if 140 <= (i.get("avg_hr") or 0) < 160)
        easy = max(0.0, moving - hard - moderate)
        load = distance * 10.0 + hard * 1.5 + moderate
        adjusted_activity_loads = []
        for item in items:
            item_distance = item.get("distance_km") or 0.0
            item_moving = item.get("moving_time_min") or 0.0
            item_hard = item_moving if (item.get("avg_hr") or 0) >= 160 else 0.0
            item_moderate = item_moving if 140 <= (item.get("avg_hr") or 0) < 160 else 0.0
            item_load = item_distance * 10.0 + item_hard * 1.5 + item_moderate
            adjusted_activity_loads.append(item_load * (item.get("run_walk_fatigue_multiplier") or 1.0))
        adjusted_load = sum(adjusted_activity_loads)
        multiplier = adjusted_load / load if load > 0 else 1.0
        row = {
            "date": key,
            "total_distance_km": distance,
            "total_moving_time_min": moving,
            "total_running_time_min": sum((i.get("run_time_s") or 0.0) for i in items) / 60.0,
            "easy_time_min": easy,
            "moderate_time_min": moderate,
            "hard_time_min": hard,
            "load_estimate": load,
            "fatigue_adjusted_load_estimate": adjusted_load,
            "run_walk_fatigue_multiplier": multiplier,
            "longest_continuous_run_s": max([i.get("longest_continuous_run_s") or 0.0 for i in items], default=0.0),
        }
        for field in DAILY_AVERAGE_FIELDS:
            row[field] = _weighted_average(items, field)
        for field in DAILY_SUM_FIELDS:
            row[field] = sum(_number(i.get(field)) for i in items)
        rows.append(row)
        current += timedelta(days=1)
    return rows


def build_weekly_features(daily_features: list[dict]) -> list[dict]:
    rows = []
    for index, day in enumerate(daily_features):
        window7 = daily_features[max(0, index - 6) : index + 1]
        window14 = daily_features[max(0, index - 13) : index + 1]
        window28 = daily_features[max(0, index - 27) : index + 1]
        loads7 = [d.get("load_estimate") or 0.0 for d in window7]
        adjusted_loads7 = [d.get("fatigue_adjusted_load_estimate") or d.get("load_estimate") or 0.0 for d in window7]
        total_time = sum(d.get("total_moving_time_min") or 0.0 for d in window7)
        easy = sum(d.get("easy_time_min") or 0.0 for d in window7)
        moderate = sum(d.get("moderate_time_min") or 0.0 for d in window7)
        hard = sum(d.get("hard_time_min") or 0.0 for d in window7)
        chronic = sum(d.get("load_estimate") or 0.0 for d in window28) / max(1, len(window28)) * 7
        adjusted_chronic = sum(d.get("fatigue_adjusted_load_estimate") or d.get("load_estimate") or 0.0 for d in window28) / max(1, len(window28)) * 7
        acute = sum(loads7)
        adjusted_acute = sum(adjusted_loads7)
        row = {
            "date": day["date"],
            "distance_7d_km": sum(d.get("total_distance_km") or 0.0 for d in window7),
            "time_7d_min": total_time,
            "load_7d": acute,
            "load_14d": sum(d.get("load_estimate") or 0.0 for d in window14),
            "load_28d": sum(d.get("load_estimate") or 0.0 for d in window28),
            "fatigue_adjusted_load_7d": adjusted_acute,
            "fatigue_adjusted_load_14d": sum(d.get("fatigue_adjusted_load_estimate") or d.get("load_estimate") or 0.0 for d in window14),
            "fatigue_adjusted_load_28d": sum(d.get("fatigue_adjusted_load_estimate") or d.get("load_estimate") or 0.0 for d in window28),
            "acute_chronic_workload_ratio": acute / chronic if chronic else None,
            "fatigue_adjusted_acute_chronic_workload_ratio": adjusted_acute / adjusted_chronic if adjusted_chronic else None,
            "easy_fraction": easy / total_time if total_time else 0.0,
            "moderate_fraction": moderate / total_time if total_time else 0.0,
            "hard_fraction": hard / total_time if total_time else 0.0,
            "long_run_distance_km": max([d.get("total_distance_km") or 0.0 for d in window7], default=0.0),
            "longest_continuous_run_s": max([d.get("longest_continuous_run_s") or 0.0 for d in window7], default=0.0),
            "monotony": mean(loads7) / pstdev(loads7) if len(loads7) > 1 and pstdev(loads7) else 0.0,
            "strain": acute * (mean(loads7) / pstdev(loads7) if len(loads7) > 1 and pstdev(loads7) else 0.0),
        }
        for field in WEEKLY_MEAN_FIELDS:
            row[f"{field}_7d_mean"] = _daily_mean(window7, field)
            row[f"{field}_28d_mean"] = _daily_mean(window28, field)
        for field in WEEKLY_SUM_FIELDS:
            row[f"{field}_7d"] = sum(_number(d.get(field)) for d in window7)
            row[f"{field}_28d"] = sum(_number(d.get(field)) for d in window28)
        for field in ["max_hr", "max_power_w", "max_temperature_c", "elevation_gain_per_km", "altitude_range_m", "total_training_effect", "total_anaerobic_training_effect"]:
            row[f"{field}_7d_max"] = _daily_max(window7, field)
        rows.append(row)
    return rows


def _weighted_average(items: list[dict], field: str) -> float | None:
    weighted_values = []
    for item in items:
        value = _optional_number(item.get(field))
        if value is None:
            continue
        weight = max(1.0, _number(item.get("moving_time_min")))
        weighted_values.append((value, weight))
    total_weight = sum(weight for _, weight in weighted_values)
    return sum(value * weight for value, weight in weighted_values) / total_weight if total_weight else None


def _daily_mean(days: list[dict], field: str) -> float | None:
    values = [_optional_number(day.get(field)) for day in days]
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def _daily_max(days: list[dict], field: str) -> float | None:
    values = [_optional_number(day.get(field)) for day in days]
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None


def _number(value) -> float:
    optional = _optional_number(value)
    return optional if optional is not None else 0.0


def _optional_number(value) -> float | None:
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
