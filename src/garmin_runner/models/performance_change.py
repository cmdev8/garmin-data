from __future__ import annotations

from statistics import median
from typing import Any

import pandas as pd


RACE_LIKE_RANGES = [
    ("1 km", "best_rolling_1km_pace", 1.0),
    ("5 km", "best_rolling_5km_pace", 5.0),
    ("10 km", "best_rolling_10km_pace", 10.0),
]

NUMERIC_CONTROL_FEATURES = [
    "distance_km",
    "moving_time_min",
    "elapsed_moving_ratio",
    "avg_hr",
    "min_hr",
    "max_hr",
    "hr_reserve_proxy",
    "avg_max_hr_ratio",
    "avg_cadence",
    "max_running_cadence",
    "elevation_gain_per_km",
    "altitude_range_m",
    "hr_drift_proxy",
    "decoupling_proxy",
    "data_quality_score",
    "avg_temperature_c",
    "max_temperature_c",
    "total_calories",
    "calories_per_km",
    "calories_per_min",
    "total_training_effect",
    "total_anaerobic_training_effect",
    "avg_power_w",
    "max_power_w",
    "normalized_power_w",
    "power_variability",
    "total_work_kj",
    "work_kj_per_km",
    "power_hr_efficiency",
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
    "hrv_count",
    "hrv_rmssd_ms",
    "walk_fraction",
    "run_walk_fatigue_multiplier",
    "forced_walk_score",
    "planned_run_walk_score",
    "number_of_walk_breaks",
    "run_pace_only_s_per_km",
    "fatigue",
    "form",
    "overload_risk",
    "load_7d",
    "fatigue_adjusted_load_7d",
    "acute_chronic_workload_ratio",
    "fatigue_adjusted_acute_chronic_workload_ratio",
    "hard_fraction",
    "long_run_distance_km",
]

CONTROL_LABELS = {
    "distance_km": "Táv",
    "moving_time_min": "Mozgásidő",
    "elapsed_moving_ratio": "Megállások aránya",
    "avg_hr": "Átlagpulzus",
    "min_hr": "Minimum pulzus",
    "max_hr": "Maximum pulzus",
    "hr_reserve_proxy": "Pulzustartalék proxy",
    "avg_max_hr_ratio": "Átlag/max pulzus arány",
    "avg_cadence": "Lépésütem",
    "max_running_cadence": "Max lépésütem",
    "elevation_gain_per_km": "Emelkedés/terep proxy",
    "altitude_range_m": "Magasságtartomány",
    "hr_drift_proxy": "Pulzusdrift",
    "decoupling_proxy": "Pulzus-tempó szétcsatolódás",
    "data_quality_score": "Adatminőség",
    "avg_temperature_c": "Átlaghőmérséklet",
    "max_temperature_c": "Maximum hőmérséklet",
    "total_calories": "Kalória",
    "calories_per_km": "Kalória/km",
    "calories_per_min": "Kalória/perc",
    "total_training_effect": "Aerob training effect",
    "total_anaerobic_training_effect": "Anaerob training effect",
    "avg_power_w": "Átlagteljesítmény",
    "max_power_w": "Maximum teljesítmény",
    "normalized_power_w": "Normalizált teljesítmény",
    "power_variability": "Teljesítmény variabilitás",
    "total_work_kj": "Munka",
    "work_kj_per_km": "Munka/km",
    "power_hr_efficiency": "Teljesítmény/pulzus hatékonyság",
    "stride_length_mm": "Lépéshossz",
    "vertical_oscillation_mm": "Vertikális oszcilláció",
    "vertical_ratio": "Vertikális arány",
    "stance_time_ms": "Talajkontakt idő",
    "pace_variability": "Tempó variabilitás",
    "hr_variability": "Pulzus variabilitás",
    "cadence_variability": "Lépésütem variabilitás",
    "power_record_variability": "Teljesítmény rekordvariabilitás",
    "lap_pace_variability": "Körtempó variabilitás",
    "lap_hr_variability": "Körpulzus variabilitás",
    "lap_power_variability": "Körteljesítmény variabilitás",
    "gps_coverage": "GPS lefedettség",
    "hrv_count": "HRV mintaszám",
    "hrv_rmssd_ms": "HRV RMSSD",
    "walk_fraction": "Séta arány",
    "run_walk_fatigue_multiplier": "Futás-séta fáradtsági szorzó",
    "forced_walk_score": "Kényszerű futás-séta",
    "planned_run_walk_score": "Tervezett futás-séta",
    "number_of_walk_breaks": "Sétaszünetek",
    "run_pace_only_s_per_km": "Csak futó tempó",
    "fatigue": "Fáradtság",
    "form": "Forma",
    "overload_risk": "Túlterhelési kockázat",
    "load_7d": "7 napos terhelés",
    "fatigue_adjusted_load_7d": "Fáradtságra korrigált terhelés",
    "acute_chronic_workload_ratio": "ACWR",
    "fatigue_adjusted_acute_chronic_workload_ratio": "Korrigált ACWR",
    "hard_fraction": "Kemény arány",
    "long_run_distance_km": "Hosszú futás",
}


def build_performance_change(
    activity_features: list[dict],
    weekly_features: list[dict],
    fitness_state: list[dict],
) -> dict[str, Any]:
    observations = _build_observations(activity_features, weekly_features, fitness_state)
    if not observations:
        return {"summary": [], "observations": [], "models": {}, "warnings": ["Nincs elég megjeleníthető teljesítményadat."]}

    adjusted: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    models: dict[str, dict[str, Any]] = {}
    for distance_range in _range_order():
        rows = [row for row in observations if row["distance_range"] == distance_range]
        if not rows:
            continue
        result = _fit_range(rows)
        adjusted.extend(result["observations"])
        summaries.append(result["summary"])
        models[distance_range] = result["model"]

    return {
        "summary": summaries,
        "observations": sorted(adjusted, key=lambda row: (row.get("date") or "", row.get("distance_range") or "")),
        "models": models,
        "warnings": [],
    }


def _build_observations(activity_features: list[dict], weekly_features: list[dict], fitness_state: list[dict]) -> list[dict[str, Any]]:
    weekly_by_date = {str(row.get("date")): row for row in weekly_features if row.get("date")}
    fitness_by_date = {str(row.get("date")): row for row in fitness_state if row.get("date")}
    rows: list[dict[str, Any]] = []
    for activity in activity_features:
        date = activity.get("date")
        if not date:
            continue
        weekly = weekly_by_date.get(str(date), {})
        fitness = fitness_by_date.get(str(date), {})
        for distance_range, field, distance_km in RACE_LIKE_RANGES:
            pace_value = _float_or_none(activity.get(field))
            if pace_value is not None and pace_value > 0:
                rows.append(_observation(activity, weekly, fitness, distance_range, distance_km, pace_value, field))
        activity_distance = _float(activity.get("distance_km"))
        activity_pace = _float_or_none(activity.get("avg_pace_s_per_km"))
        if activity_pace is None or activity_pace <= 0:
            continue
        if 18.0 <= activity_distance <= 25.0:
            rows.append(_observation(activity, weekly, fitness, "21.1 km", 21.1, activity_pace, "avg_pace_s_per_km"))
        if activity_distance >= 30.0:
            rows.append(_observation(activity, weekly, fitness, "42.2 km", 42.2, activity_pace, "avg_pace_s_per_km"))
    return rows


def _observation(
    activity: dict[str, Any],
    weekly: dict[str, Any],
    fitness: dict[str, Any],
    distance_range: str,
    evidence_distance_km: float,
    pace_s_per_km: float,
    source_field: str,
) -> dict[str, Any]:
    distance = _float(activity.get("distance_km"))
    moving = _float(activity.get("moving_time_min"))
    elapsed = _float(activity.get("elapsed_time_min"))
    elevation = _float(activity.get("elevation_gain"))
    return {
        "activity_id": activity.get("activity_id"),
        "date": activity.get("date"),
        "distance_range": distance_range,
        "source_field": source_field,
        "evidence_distance_km": evidence_distance_km,
        "observed_pace_s_per_km": pace_s_per_km,
        "distance_km": distance,
        "moving_time_min": moving,
        "elapsed_moving_ratio": elapsed / moving if moving > 0 else 1.0,
        "avg_hr": activity.get("avg_hr"),
        "min_hr": activity.get("min_hr"),
        "max_hr": activity.get("max_hr"),
        "hr_reserve_proxy": activity.get("hr_reserve_proxy"),
        "avg_max_hr_ratio": activity.get("avg_max_hr_ratio"),
        "avg_cadence": activity.get("avg_cadence"),
        "max_running_cadence": activity.get("max_running_cadence"),
        "elevation_gain_per_km": elevation / distance if distance > 0 else 0.0,
        "altitude_range_m": activity.get("altitude_range_m"),
        "hr_drift_proxy": activity.get("hr_drift_proxy"),
        "decoupling_proxy": activity.get("decoupling_proxy"),
        "data_quality_score": activity.get("data_quality_score"),
        "avg_temperature_c": activity.get("avg_temperature_c"),
        "max_temperature_c": activity.get("max_temperature_c"),
        "total_calories": activity.get("total_calories"),
        "calories_per_km": activity.get("calories_per_km"),
        "calories_per_min": activity.get("calories_per_min"),
        "total_training_effect": activity.get("total_training_effect"),
        "total_anaerobic_training_effect": activity.get("total_anaerobic_training_effect"),
        "avg_power_w": activity.get("avg_power_w"),
        "max_power_w": activity.get("max_power_w"),
        "normalized_power_w": activity.get("normalized_power_w"),
        "power_variability": activity.get("power_variability"),
        "total_work_kj": activity.get("total_work_kj"),
        "work_kj_per_km": activity.get("work_kj_per_km"),
        "power_hr_efficiency": activity.get("power_hr_efficiency"),
        "stride_length_mm": activity.get("stride_length_mm"),
        "vertical_oscillation_mm": activity.get("vertical_oscillation_mm"),
        "vertical_ratio": activity.get("vertical_ratio"),
        "stance_time_ms": activity.get("stance_time_ms"),
        "pace_variability": activity.get("pace_variability"),
        "hr_variability": activity.get("hr_variability"),
        "cadence_variability": activity.get("cadence_variability"),
        "power_record_variability": activity.get("power_record_variability"),
        "lap_pace_variability": activity.get("lap_pace_variability"),
        "lap_hr_variability": activity.get("lap_hr_variability"),
        "lap_power_variability": activity.get("lap_power_variability"),
        "gps_coverage": activity.get("gps_coverage"),
        "hrv_count": activity.get("hrv_count"),
        "hrv_rmssd_ms": activity.get("hrv_rmssd_ms"),
        "run_walk_type": activity.get("run_walk_type") or "unknown",
        "walk_fraction": activity.get("walk_fraction"),
        "run_walk_fatigue_multiplier": activity.get("run_walk_fatigue_multiplier"),
        "forced_walk_score": activity.get("forced_walk_score"),
        "planned_run_walk_score": activity.get("planned_run_walk_score"),
        "number_of_walk_breaks": activity.get("number_of_walk_breaks"),
        "run_pace_only_s_per_km": activity.get("run_pace_only_s_per_km"),
        "fatigue": fitness.get("fatigue"),
        "form": fitness.get("form"),
        "overload_risk": fitness.get("overload_risk"),
        "load_7d": weekly.get("load_7d"),
        "fatigue_adjusted_load_7d": weekly.get("fatigue_adjusted_load_7d"),
        "acute_chronic_workload_ratio": weekly.get("acute_chronic_workload_ratio"),
        "fatigue_adjusted_acute_chronic_workload_ratio": weekly.get("fatigue_adjusted_acute_chronic_workload_ratio"),
        "hard_fraction": weekly.get("hard_fraction"),
        "long_run_distance_km": weekly.get("long_run_distance_km"),
    }


def _fit_range(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    target = table["observed_pace_s_per_km"].astype(float)
    model_type = "Nyers trend"
    confidence = min(0.3, len(table) / 8.0 * 0.3)
    fallback_reason = "Kevés megfigyelés; nincs kontrollált modell."
    importance: dict[str, float] = {}
    predicted = pd.Series([float(target.median())] * len(table))

    if len(table) >= 8 and float(target.std(ddof=0)) > 1e-6:
        features = _feature_table(table)
        try:
            if len(table) >= 20:
                from sklearn.ensemble import HistGradientBoostingRegressor
                from sklearn.inspection import permutation_importance
                from sklearn.metrics import mean_absolute_error

                model = HistGradientBoostingRegressor(random_state=42, max_iter=80)
                model.fit(features, target)
                predicted = pd.Series(model.predict(features))
                mae = float(mean_absolute_error(target, predicted))
                model_type = "HistGradientBoostingRegressor"
                confidence = _confidence(len(table), mae, float(target.median()))
                fallback_reason = ""
                result = permutation_importance(model, features, target, n_repeats=3, random_state=42)
                importance = _importance_dict(features.columns.tolist(), getattr(result, "importances_mean", []))
            else:
                from sklearn.compose import ColumnTransformer
                from sklearn.linear_model import Ridge
                from sklearn.metrics import mean_absolute_error
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import OneHotEncoder, StandardScaler

                numeric_columns = [column for column in NUMERIC_CONTROL_FEATURES if column in table]
                categorical_columns = ["run_walk_type"]
                model_table = table[numeric_columns + categorical_columns].copy()
                for column in numeric_columns:
                    model_table[column] = pd.to_numeric(model_table[column], errors="coerce").fillna(0.0)
                model_table["run_walk_type"] = model_table["run_walk_type"].fillna("unknown").astype(str)
                preprocessor = ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), numeric_columns),
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
                    ]
                )
                model = make_pipeline(preprocessor, Ridge(alpha=1.0))
                model.fit(model_table, target)
                predicted = pd.Series(model.predict(model_table))
                mae = float(mean_absolute_error(target, predicted))
                model_type = "Ridge kontrollmodell"
                confidence = _confidence(len(table), mae, float(target.median()))
                fallback_reason = ""
                ridge = model.named_steps["ridge"]
                coefficients = [abs(float(value)) for value in list(getattr(ridge, "coef_", []))[: len(numeric_columns)]]
                importance = _importance_dict(numeric_columns, coefficients)
        except Exception as exc:
            fallback_reason = f"Kontrollált modell nem futott: {exc}"

    median_observed = float(target.median())
    table["model_predicted_pace_s_per_km"] = predicted
    table["adjusted_pace_s_per_km"] = median_observed + (target - predicted)
    raw_change = _window_change(table["observed_pace_s_per_km"].tolist())
    adjusted_change = _window_change(table["adjusted_pace_s_per_km"].tolist())
    baseline = _first_window_median(table["adjusted_pace_s_per_km"].tolist())
    percent_change = adjusted_change / baseline if baseline else 0.0
    primary_factors = _primary_factors(importance)
    summary = {
        "distance_range": rows[0]["distance_range"],
        "observation_count": len(table),
        "raw_change_s_per_km": round(raw_change, 1),
        "adjusted_change_s_per_km": round(adjusted_change, 1),
        "improvement_s_per_km": round(-adjusted_change, 1),
        "percent_change": round(percent_change, 4),
        "model_type": model_type,
        "confidence": round(confidence, 3),
        "primary_controlled_factors": primary_factors,
        "fallback_reason": fallback_reason,
    }
    model = {
        "model_type": model_type,
        "confidence": round(confidence, 3),
        "feature_importance": importance,
        "fallback_reason": fallback_reason,
        "controlled_variables": [CONTROL_LABELS.get(name, name) for name in NUMERIC_CONTROL_FEATURES] + ["Futás-séta típus"],
    }
    return {"summary": summary, "observations": table.to_dict("records"), "model": model}


def _feature_table(table: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()
    for column in NUMERIC_CONTROL_FEATURES:
        source = table[column] if column in table else pd.Series([0.0] * len(table))
        values = pd.to_numeric(source, errors="coerce")
        features[column] = values.fillna(0.0)
    run_walk = table["run_walk_type"].fillna("unknown").astype(str) if "run_walk_type" in table else pd.Series(["unknown"] * len(table))
    dummies = pd.get_dummies(run_walk, prefix="run_walk_type")
    return pd.concat([features, dummies], axis=1)


def _window_change(values: list[float]) -> float:
    return _last_window_median(values) - _first_window_median(values)


def _first_window_median(values: list[float]) -> float:
    size = _window_size(values)
    return float(median(values[:size])) if values else 0.0


def _last_window_median(values: list[float]) -> float:
    size = _window_size(values)
    return float(median(values[-size:])) if values else 0.0


def _window_size(values: list[float]) -> int:
    return max(2, int(len(values) * 0.3)) if len(values) >= 4 else max(1, len(values))


def _confidence(count: int, mae: float, median_pace: float) -> float:
    history = min(1.0, count / 25.0)
    fit = max(0.0, 1.0 - mae / max(30.0, median_pace * 0.12))
    return min(0.85, max(0.2, 0.2 + history * 0.35 + fit * 0.3))


def _importance_dict(names: list[str], values: Any) -> dict[str, float]:
    raw = {CONTROL_LABELS.get(name, name): max(0.0, _float(value)) for name, value in zip(names, values)}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {name: round(value / total, 4) for name, value in sorted(raw.items(), key=lambda item: item[1], reverse=True)}


def _primary_factors(importance: dict[str, float]) -> str:
    if not importance:
        return "n/a"
    return ", ".join(list(importance)[:3])


def _range_order() -> list[str]:
    return ["1 km", "5 km", "10 km", "21.1 km", "42.2 km"]


def _float(value: Any) -> float:
    try:
        if value is None or value != value:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
