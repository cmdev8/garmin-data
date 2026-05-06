from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from garmin_runner.dashboard.formatting import number, pace, percent
from garmin_runner.dashboard.i18n import hu_day, plan_family_label


def plan_summary(plan: dict[str, Any]) -> dict[str, str]:
    summary = _dict(plan.get("summary"))
    return {
        "score": number(plan.get("score"), 3),
        "Időtáv": f"{plan.get('horizon_days', 'n/a')} nap",
        "Táv": f"{number(summary.get('total_distance_km'))} km",
        "Idő": f"{summary.get('total_duration_min', 'n/a')} perc",
        "minőségi napok": str(summary.get("hard_days", "n/a")),
        "hosszú futás": f"{number(summary.get('long_run_km'))} km",
    }


def plan_schedule_table(plan: dict[str, Any], *, include_rest: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for workout in _list_of_dicts(plan.get("days") or plan.get("week")):
        if not include_rest and workout.get("workout_type") == "rest":
            continue
        rows.append(
            {
                "Dátum": workout.get("date", ""),
                "Nap": hu_day(str(workout.get("day", ""))),
                "Edzés": _workout_label(str(workout.get("workout_type", ""))),
                "Intenzitás": _intensity_label(str(workout.get("intensity", ""))),
                "Idő (perc)": workout.get("duration_min", 0),
                "Táv (km)": _distance_cell(workout.get("distance_km")),
                "futás-séta előírás": _prescription_label(workout.get("run_walk_prescription")),
                "Cél": _purpose_label(workout.get("purpose")),
            }
        )
    return pd.DataFrame(rows)


def plan_reason_labels(reasons: Any) -> list[str]:
    return [_reason_label(str(reason)) for reason in _list(reasons)]


def race_prediction_table(predictions: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, prediction in _dict(predictions.get("race_predictions")).items():
        if not isinstance(prediction, dict):
            continue
        rows.append(
            {
                "Táv": label,
                "tempó": pace(prediction.get("pace_s_per_km")),
                "Becsült idő": duration(prediction.get("estimated_time_s")),
                "confidence": percent(prediction.get("confidence")),
                "Magyarázat": prediction.get("explanation", ""),
            }
        )
    return pd.DataFrame(rows)


def race_confidence_chart(predictions: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, prediction in _dict(predictions.get("race_predictions")).items():
        if isinstance(prediction, dict):
            rows.append({"Táv": label, "confidence": round(_float(prediction.get("confidence")) * 100, 1)})
    return pd.DataFrame(rows)


def prediction_plan_options(candidate_plans: pd.DataFrame, selected_plan_id: str | None) -> list[str]:
    if candidate_plans.empty or "plan_id" not in candidate_plans:
        return []
    table = candidate_plans.copy()
    table["_plan_id"] = table["plan_id"].astype(str)
    if "is_valid" in table:
        valid = table[table["is_valid"] == True]  # noqa: E712
        invalid = table[table["is_valid"] != True]  # noqa: E712
        ordered = pd.concat([valid, invalid], ignore_index=True)
    else:
        ordered = table
    options = list(dict.fromkeys(str(plan_id) for plan_id in ordered["_plan_id"].tolist()))
    return options


def prediction_plan_default_index(options: list[str], selected_plan_id: str | None) -> int:
    if selected_plan_id is not None and selected_plan_id in options:
        return options.index(selected_plan_id)
    return 0


def candidate_row_by_plan_id(candidate_plans: pd.DataFrame, plan_id: str | None) -> dict[str, Any]:
    if candidate_plans.empty or plan_id is None or "plan_id" not in candidate_plans:
        return {}
    matches = candidate_plans[candidate_plans["plan_id"].astype(str) == str(plan_id)]
    if matches.empty:
        return {}
    return {str(key): value for key, value in matches.iloc[0].to_dict().items()}


def selected_candidate_row(candidate_plans: pd.DataFrame, plan: dict[str, Any]) -> dict[str, Any]:
    plan_id = plan.get("selected_plan_id")
    return candidate_row_by_plan_id(candidate_plans, str(plan_id) if plan_id is not None else None)


def candidate_summary_values(candidate_row: dict[str, Any]) -> dict[str, str]:
    return {
        "Terv": str(candidate_row.get("plan_id", "n/a")),
        "Család": plan_family_label(candidate_row.get("family", "n/a")),
        "score": number(candidate_row.get("total_score"), 3),
        "Adaptáció": percent(candidate_row.get("expected_adaptation")),
        "fatigue": percent(candidate_row.get("expected_fatigue")),
        "overload risk": percent(candidate_row.get("overload_risk")),
        "Történeti driver match": percent(candidate_row.get("historical_driver_match")),
        "Táv": f"{number(candidate_row.get('total_distance_km'))} km",
        "Idő": f"{candidate_row.get('total_duration_min', 'n/a')} perc",
        "minőségi napok": str(candidate_row.get("hard_days", "n/a")),
    }


def scoring_component_values(candidate_row: dict[str, Any]) -> dict[str, str]:
    return {
        "expected adaptation": percent(candidate_row.get("expected_adaptation")),
        "expected fatigue": percent(candidate_row.get("expected_fatigue")),
        "overload risk": percent(candidate_row.get("overload_risk")),
        "Versenyspecifikusság": percent(candidate_row.get("race_specificity")),
        "Hiányzó inger": percent(candidate_row.get("missing_stimulus_score")),
        "Történeti driver match": percent(candidate_row.get("historical_driver_match")),
        "Korlátbüntetés": percent(candidate_row.get("constraint_penalty")),
    }


def plan_adjusted_prediction_table(
    predictions: dict[str, Any],
    candidate_row: dict[str, Any],
    horizon_days: int | None = None,
    diminishing_returns_factor: Any = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, prediction in _dict(predictions.get("race_predictions")).items():
        if not isinstance(prediction, dict):
            continue
        baseline_pace = _float_or_none(prediction.get("pace_s_per_km"))
        baseline_time = _float_or_none(prediction.get("estimated_time_s"))
        if baseline_pace is None or baseline_time is None:
            continue
        distance_km = _float_or_none(prediction.get("distance_km")) or _distance_from_label(str(label))
        improvement_pct = _plan_improvement_pct(candidate_row, horizon_days, distance_km, diminishing_returns_factor)
        plan_pace = baseline_pace * (1 - improvement_pct)
        plan_time = baseline_time * (1 - improvement_pct)
        time_saved = baseline_time - plan_time
        rows.append(
            {
                "Táv": label,
                "baseline tempó": pace(baseline_pace),
                "terv utáni tempó": pace(plan_pace),
                "improvement": _percent(improvement_pct),
                "baseline idő": duration(baseline_time),
                "terv utáni idő": duration(plan_time),
                "időnyereség": _signed_duration(time_saved),
                "confidence": percent(prediction.get("confidence")),
            }
        )
    return pd.DataFrame(rows)


def plan_time_savings_chart(
    predictions: dict[str, Any],
    candidate_row: dict[str, Any],
    horizon_days: int | None = None,
    diminishing_returns_factor: Any = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, prediction in _dict(predictions.get("race_predictions")).items():
        if isinstance(prediction, dict):
            baseline_time = _float_or_none(prediction.get("estimated_time_s"))
            if baseline_time is not None:
                distance_km = _float_or_none(prediction.get("distance_km")) or _distance_from_label(str(label))
                improvement_pct = _plan_improvement_pct(candidate_row, horizon_days, distance_km, diminishing_returns_factor)
                rows.append({"Táv": label, "időnyereség (mp)": round(baseline_time * improvement_pct, 1)})
    return pd.DataFrame(rows)


def hr_zone_table(predictions: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for zone, values in _dict(predictions.get("hr_zones")).items():
        if isinstance(values, dict):
            rows.append({"pulzuszóna": zone, "Minimum (bpm)": values.get("min_bpm"), "Maximum (bpm)": values.get("max_bpm")})
    return pd.DataFrame(rows)


def pace_zone_table(predictions: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for zone, values in _dict(predictions.get("pace_zones")).items():
        if isinstance(values, dict):
            rows.append(
                {
                    "tempózóna": _pace_zone_label(zone),
                    "Lassabb határ": pace(values.get("min_s_per_km")),
                    "Gyorsabb határ": pace(values.get("max_s_per_km")),
                }
            )
    return pd.DataFrame(rows)


def candidate_plan_table(candidate_plans: pd.DataFrame) -> pd.DataFrame:
    if candidate_plans.empty:
        return pd.DataFrame()
    columns = [
        "plan_id",
        "family",
        "total_score",
        "is_valid",
        "total_distance_km",
        "total_duration_min",
        "hard_days",
        "overload_risk",
        "historical_driver_match",
        "rejection_reasons",
    ]
    table = candidate_plans[[column for column in columns if column in candidate_plans.columns]].copy()
    if "overload_risk" in table:
        table["overload_risk"] = table["overload_risk"].apply(percent)
    if "historical_driver_match" in table:
        table["historical_driver_match"] = table["historical_driver_match"].apply(percent)
    if "family" in table:
        table["family"] = table["family"].apply(plan_family_label)
    return table.rename(
        columns={
            "plan_id": "Terv azonosító",
            "family": "Tervtípus",
            "total_score": "score",
            "is_valid": "Érvényes",
            "total_distance_km": "Táv (km)",
            "total_duration_min": "Idő (perc)",
            "hard_days": "minőségi napok",
            "overload_risk": "overload risk",
            "historical_driver_match": "Történeti driver match",
            "rejection_reasons": "Elutasítás oka",
        }
    )


def top_candidate_scores(candidate_plans: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if candidate_plans.empty or "total_score" not in candidate_plans or "plan_id" not in candidate_plans:
        return pd.DataFrame()
    table = candidate_plans.copy()
    if "is_valid" in table:
        table = table[table["is_valid"] == True]  # noqa: E712
    if table.empty:
        return pd.DataFrame()
    return table.sort_values("total_score", ascending=False).head(limit)[["plan_id", "total_score"]].rename(
        columns={"plan_id": "Terv", "total_score": "score"}
    )


def rejection_summary_table(debug: dict[str, Any]) -> pd.DataFrame:
    rows = [{"Ok": reason, "Darab": count} for reason, count in _dict(debug.get("rejection_summary")).items()]
    return pd.DataFrame(rows)


def top_plan_ids(debug: dict[str, Any]) -> list[str]:
    return [str(plan_id) for plan_id in _list(debug.get("top_5_plan_ids"))]


def ml_metric_values(ml_result: dict[str, Any]) -> dict[str, str]:
    metrics = _dict(ml_result.get("metrics"))
    return {
        "model": str(ml_result.get("model_type", "n/a")),
        "Tanító sorok": str(ml_result.get("row_count", "n/a")),
        "confidence": percent(ml_result.get("confidence")),
        "MAE": number(metrics.get("mae"), 3),
        "Validációs sorok": str(metrics.get("validation_rows", "n/a")),
    }


def ml_prediction_values(ml_result: dict[str, Any]) -> dict[str, str]:
    predictions = _dict(ml_result.get("predictions"))
    return {
        "7 napos performance index": number(predictions.get("performance_index_7d"), 2),
        "expected adaptation": percent(predictions.get("expected_adaptation")),
        "overload risk modifier": percent(predictions.get("overload_risk_modifier")),
        "diminishing returns factor": percent(predictions.get("diminishing_returns_factor")),
    }


def diminishing_returns_values(ml_result: dict[str, Any]) -> dict[str, str]:
    metrics = _dict(ml_result.get("metrics"))
    predictions = _dict(ml_result.get("predictions"))
    return {
        "diminishing returns factor": percent(predictions.get("diminishing_returns_factor")),
        "Megfigyelések": str(metrics.get("diminishing_returns_observations", "n/a")),
        "Max. korrekció": "35%",
        "Állapot": _diminishing_returns_reason_label(metrics.get("diminishing_returns_reason")),
    }


def diminishing_returns_history_table(ml_result: dict[str, Any]) -> pd.DataFrame:
    metrics = _dict(ml_result.get("metrics"))
    rows: list[dict[str, Any]] = []
    for row in _list_of_dicts(metrics.get("diminishing_returns_history")):
        rows.append(
            {
                "Hónap": row.get("month", ""),
                "diminishing returns factor": percent(row.get("factor")),
                "performance index": number(row.get("latest_performance_index"), 1),
                "response ratio": percent(_diminishing_returns_raw_ratio(row)),
                "Megfigyelések": row.get("observations", "n/a"),
                "Állapot": _diminishing_returns_reason_label(row.get("reason")),
            }
        )
    return pd.DataFrame(rows)


def diminishing_returns_history_chart(ml_result: dict[str, Any]) -> pd.DataFrame:
    metrics = _dict(ml_result.get("metrics"))
    rows: list[dict[str, Any]] = []
    for row in _list_of_dicts(metrics.get("diminishing_returns_history")):
        factor = _float_or_none(row.get("factor"))
        if factor is not None:
            rows.append({"Hónap": row.get("month", ""), "diminishing returns factor (%)": round(factor * 100, 1)})
    return pd.DataFrame(rows)


def diminishing_returns_history_context_chart(ml_result: dict[str, Any]) -> pd.DataFrame:
    metrics = _dict(ml_result.get("metrics"))
    rows: list[dict[str, Any]] = []
    for row in _list_of_dicts(metrics.get("diminishing_returns_history")):
        performance = _float_or_none(row.get("latest_performance_index"))
        ratio = _diminishing_returns_raw_ratio(row)
        values: dict[str, Any] = {"Hónap": row.get("month", "")}
        if performance is not None:
            values["performance index"] = round(performance, 1)
        if ratio is not None:
            values["response ratio (%)"] = round(ratio * 100.0, 1)
        if len(values) > 1:
            rows.append(values)
    return pd.DataFrame(rows)


def feature_importance_table(ml_result: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"feature": feature, "feature importance": round(_float(score) * 100, 1)}
        for feature, score in _dict(ml_result.get("feature_importance")).items()
        if _float(score) > 0
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("feature importance", ascending=False)


def temporal_ml_values(ml_result: dict[str, Any]) -> dict[str, str]:
    metrics = _dict(ml_result.get("metrics"))
    return {
        "Temporal model": "igen" if metrics.get("temporal_model_used") else "nem",
        "Lookback": ", ".join(str(value) for value in _list(metrics.get("lookback_days"))) or "n/a",
        "Validation": str(metrics.get("validation_strategy", "n/a")),
        "Foldok": str(metrics.get("fold_count", "n/a")),
        "Temporal sorok": str(metrics.get("sequence_row_count", "n/a")),
    }


def temporal_driver_table(ml_result: dict[str, Any], direction: str = "positive") -> pd.DataFrame:
    key = "positive_temporal_drivers" if direction == "positive" else "negative_temporal_drivers"
    rows = [
        {"feature": str(row.get("feature", "n/a")), "Koeficiens": number(row.get("coefficient"), 4), "Erősség": number(row.get("strength"), 4)}
        for row in _list_of_dicts(ml_result.get(key))
    ]
    return pd.DataFrame(rows)


def feature_names(ml_result: dict[str, Any]) -> list[str]:
    return [str(feature) for feature in _list(ml_result.get("feature_names"))]


def activity_option_labels(activity_features: pd.DataFrame) -> dict[str, str]:
    if activity_features.empty or "activity_id" not in activity_features:
        return {}
    labels = {}
    for row in activity_features.to_dict("records"):
        activity_id = str(row.get("activity_id", ""))
        if not activity_id:
            continue
        date_text = str(row.get("date") or "dátum nélkül")
        distance_text = f"{number(row.get('distance_km'))} km"
        pace_text = pace(row.get("avg_pace_s_per_km"))
        labels[activity_id] = f"{date_text} | {distance_text} | {pace_text} | {activity_id}"
    return labels


def selected_activity_metrics(activity_row: Mapping[Any, Any]) -> dict[str, dict[str, str]]:
    return {
        "Alapadatok": {
            "Táv": f"{number(activity_row.get('distance_km'))} km",
            "Mozgásidő": duration(_minutes_to_seconds(activity_row.get("moving_time_min"))),
            "Eltelt idő": duration(_minutes_to_seconds(activity_row.get("elapsed_time_min"))),
            "Átlagtempó": pace(activity_row.get("avg_pace_s_per_km")),
            "Legjobb 1 km": pace(activity_row.get("best_rolling_1km_pace")),
            "Legjobb 5 km": pace(activity_row.get("best_rolling_5km_pace")),
            "Legjobb 10 km": pace(activity_row.get("best_rolling_10km_pace")),
        },
        "Pulzus és intenzitás": {
            "Átlagpulzus": _unit(activity_row.get("avg_hr"), "bpm", 0),
            "Max pulzus": _unit(activity_row.get("max_hr"), "bpm", 0),
            "Min pulzus": _unit(activity_row.get("min_hr"), "bpm", 0),
            "HR drift": _unit(activity_row.get("hr_drift_proxy"), "bpm", 1),
            "Decoupling": percent(activity_row.get("decoupling_proxy")),
            "Aerob training effect": number(activity_row.get("total_training_effect"), 1),
            "Anaerob training effect": number(activity_row.get("total_anaerobic_training_effect"), 1),
        },
        "Mechanika és környezet": {
            "Lépésütem": _unit(activity_row.get("avg_cadence"), "spm", 0),
            "Szintemelkedés": _unit(activity_row.get("elevation_gain"), "m", 0),
            "Szintemelkedés/km": _unit(activity_row.get("elevation_gain_per_km"), "m/km", 1),
            "Átlag power": _unit(activity_row.get("avg_power_w"), "W", 0),
            "normalized power": _unit(activity_row.get("normalized_power_w"), "W", 0),
            "Hőmérséklet": _unit(activity_row.get("avg_temperature_c"), "°C", 1),
            "Kalória": _unit(activity_row.get("total_calories"), "kcal", 0),
        },
        "Futás-séta és minőség": {
            "Futás-séta típus": str(activity_row.get("run_walk_type", "n/a")),
            "Séta arány": percent(activity_row.get("walk_fraction")),
            "Tervezett score": percent(activity_row.get("planned_run_walk_score")),
            "Kényszerű score": percent(activity_row.get("forced_walk_score")),
            "Fáradtsági szorzó": percent(activity_row.get("run_walk_fatigue_multiplier")),
            "data quality": percent(activity_row.get("data_quality_score")),
            "GPS lefedettség": percent(activity_row.get("gps_coverage")),
        },
    }


def activity_record_chart_table(records: pd.DataFrame, activity_id: str, chart_type: str) -> pd.DataFrame:
    selected = _activity_records(records, activity_id)
    if selected.empty or "elapsed_s" not in selected:
        return pd.DataFrame()
    table = pd.DataFrame({"perc": pd.to_numeric(selected["elapsed_s"], errors="coerce") / 60.0})
    if chart_type == "pace_hr":
        _add_chart_column(table, selected, "pace_s_per_km", "tempó (perc/km)", transform=_pace_minutes)
        _add_chart_column(table, selected, "heart_rate_bpm", "pulzus (bpm)")
    elif chart_type == "mechanics":
        _add_chart_column(table, selected, "cadence_spm", "lépésütem (spm)")
        _add_chart_column(table, selected, "power_w", "power (W)")
        _add_chart_column(table, selected, "vertical_ratio", "vertikális arány")
        _add_chart_column(table, selected, "stance_time_ms", "talajkontakt idő (ms)")
    elif chart_type == "environment":
        _add_chart_column(table, selected, "altitude_m", "magasság (m)")
        _add_chart_column(table, selected, "temperature_c", "hőmérséklet (°C)")
    table = table.dropna(axis=1, how="all").dropna(subset=["perc"])
    return table if len(table.columns) > 1 else pd.DataFrame()


def activity_segment_table(segments: pd.DataFrame, activity_id: str) -> pd.DataFrame:
    selected = _activity_records(segments, activity_id)
    if selected.empty:
        return pd.DataFrame()
    rows = []
    for row in selected.to_dict("records"):
        rows.append(
            {
                "Szakasz": row.get("segment_id", ""),
                "Típus": _segment_state_label(str(row.get("state", "unknown"))),
                "Idő": duration(row.get("duration_s")),
                "Táv": f"{number((_float_or_none(row.get('distance_m')) or 0.0) / 1000.0)} km",
                "Tempó": pace(row.get("avg_pace_s_per_km")),
                "Átlagpulzus": _unit(row.get("avg_hr"), "bpm", 0),
                "Lépésütem": _unit(row.get("avg_cadence_spm"), "spm", 0),
                "Szint": _unit(row.get("elevation_gain_m"), "m", 0),
            }
        )
    return pd.DataFrame(rows)


def activity_segment_summary(segments: pd.DataFrame, activity_id: str) -> pd.DataFrame:
    selected = _activity_records(segments, activity_id)
    if selected.empty or "state" not in selected:
        return pd.DataFrame()
    rows = []
    total_duration = _numeric_column(selected, "duration_s").fillna(0).sum()
    for state, group in selected.groupby("state", dropna=False):
        duration_s = _numeric_column(group, "duration_s").fillna(0).sum()
        distance_m = _numeric_column(group, "distance_m").fillna(0).sum()
        rows.append(
            {
                "Típus": _segment_state_label(str(state)),
                "Idő": duration(duration_s),
                "Arány": percent(duration_s / total_duration if total_duration else None),
                "Táv": f"{number(distance_m / 1000.0)} km",
            }
        )
    return pd.DataFrame(rows)


def activity_advanced_fit_table(activity_row: dict[str, Any]) -> pd.DataFrame:
    fields = [
        ("Kalória/km", "calories_per_km", "kcal/km", 1),
        ("Kalória/perc", "calories_per_min", "kcal/perc", 1),
        ("Max power", "max_power_w", "W", 0),
        ("Power variability", "power_variability", "", 2),
        ("Munka", "total_work_kj", "kJ", 1),
        ("Munka/km", "work_kj_per_km", "kJ/km", 1),
        ("Power/HR hatékonyság", "power_hr_efficiency", "", 2),
        ("Lépésszám", "total_strides", "", 0),
        ("Lépéshossz", "stride_length_mm", "mm", 0),
        ("Vertikális oszcilláció", "vertical_oscillation_mm", "mm", 1),
        ("Vertikális arány", "vertical_ratio", "", 2),
        ("Talajkontakt idő", "stance_time_ms", "ms", 0),
        ("Tempó variability", "pace_variability", "", 2),
        ("HR variability", "hr_variability", "", 2),
        ("Lépésütem variability", "cadence_variability", "", 2),
        ("Power record variability", "power_record_variability", "", 2),
        ("Körök száma", "lap_count", "", 0),
        ("Körtempó variability", "lap_pace_variability", "", 2),
        ("Kör HR variability", "lap_hr_variability", "", 2),
        ("Kör power variability", "lap_power_variability", "", 2),
        ("HRV minták", "hrv_count", "", 0),
        ("HRV medián RR", "hrv_median_rr_ms", "ms", 0),
        ("HRV RMSSD", "hrv_rmssd_ms", "ms", 1),
        ("Magasságtartomány", "altitude_range_m", "m", 0),
        ("Max hőmérséklet", "max_temperature_c", "°C", 1),
    ]
    rows = []
    for label, field, unit, digits in fields:
        value = _float_or_none(activity_row.get(field))
        if value is None:
            continue
        rows.append({"Mutató": label, "Érték": _unit(value, unit, digits) if unit else number(value, digits)})
    return pd.DataFrame(rows)


def activity_quality_notes(activity_row: dict[str, Any], records: pd.DataFrame, activity_id: str) -> list[str]:
    notes = []
    selected = _activity_records(records, activity_id)
    if selected.empty:
        notes.append("Nincs részletes idősor ehhez az edzéshez.")
        return notes
    expected = {
        "heart_rate_bpm": "pulzus",
        "cadence_spm": "lépésütem",
        "power_w": "power",
        "altitude_m": "magasság",
        "temperature_c": "hőmérséklet",
        "vertical_ratio": "futódinamika",
    }
    missing = [label for column, label in expected.items() if column not in selected or pd.to_numeric(selected[column], errors="coerce").dropna().empty]
    if missing:
        notes.append("Hiányzó vagy ritka idősor: " + ", ".join(missing) + ".")
    quality = percent(activity_row.get("data_quality_score"))
    notes.append(f"data quality: {quality}.")
    return notes


def diagnostic_summary_values(data: Any) -> dict[str, str]:
    reasons = _diagnostic_reason_rows(data)
    positive = [row for row in reasons if row["Irány"] == "Javító"]
    negative = [row for row in reasons if row["Irány"] == "Rontó"]
    changes = _performance_changes(data)
    if not changes:
        status = "Kevés adat"
    else:
        improving = sum(1 for row in changes if row["change"] < -1.0)
        worsening = sum(1 for row in changes if row["change"] > 1.0)
        if improving > worsening:
            status = "Javulás"
        elif worsening > improving:
            status = "Romlás"
        else:
            status = "Vegyes"
    confidence_values = [_float(row.get("confidence")) for row in _performance_summaries(data) if row.get("confidence") is not None]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else _float(_dict(_data_attr(data, "ml_result", {})).get("confidence"))
    return {
        "Állapot": status,
        "Fő javító tényező": _top_factor(positive),
        "Fő rontó tényező": _top_factor(negative),
        "confidence": percent(confidence),
    }


def diagnostic_reasons_table(data: Any) -> pd.DataFrame:
    rows = [{key: row[key] for key in ["Irány", "Tényező", "Bizonyíték", "Hatás", "confidence"]} for row in _diagnostic_reason_rows(data)]
    return pd.DataFrame(rows)


def diagnostic_actions_table(data: Any) -> pd.DataFrame:
    reasons = _diagnostic_reason_rows(data)
    factors = {str(row["Tényező"]) for row in reasons if row["Irány"] == "Rontó"}
    positive_factors = {str(row["Tényező"]) for row in reasons if row["Irány"] == "Javító"}
    rows: list[dict[str, Any]] = []
    if "fatigue / form" in factors or "overload risk" in factors:
        rows.append({"Prioritás": 1, "Javaslat": "Regeneráló hét vagy könnyített blokk", "Miért": "A fatigue, form vagy overload risk rontja a frissességet.", "Kapcsolódó adat": "fitness/fatigue"})
    if "Kényszerű futás-séta" in factors:
        rows.append({"Prioritás": 2, "Javaslat": "Sík, Z2 intenzitású futás-séta kontrollált szakaszokkal", "Miért": "A kényszerű séták fáradtságot vagy túl erős kezdést jelezhetnek.", "Kapcsolódó adat": "forced_walk_score"})
    if "intenzitáseloszlás" in factors:
        rows.append({"Prioritás": 3, "Javaslat": "Csökkentsd a minőségi napok arányát", "Miért": "A túl sok közepes/kemény intenzitás elfedheti az adaptációt.", "Kapcsolódó adat": "hard_fraction"})
    if "Adatminőség" in factors:
        rows.append({"Prioritás": 4, "Javaslat": "Javítsd a mérési minőséget", "Miért": "Hiányos HR/GPS/lépésütem mellett bizonytalanabb a diagnózis.", "Kapcsolódó adat": "data_quality_score"})
    if "Tervezett futás-séta" in positive_factors:
        rows.append({"Prioritás": 5, "Javaslat": "Tartsd meg a tervezett futás-séta struktúrát", "Miért": "A feltöltött adatokban ez segítette a fáradtság kontrollját.", "Kapcsolódó adat": "run_walk_fatigue_multiplier"})
    if not rows and _performance_changes(data):
        rows.append({"Prioritás": 1, "Javaslat": "Tartsd a jelenlegi terhelési mintát, kis lépésekben emelve", "Miért": "Nem látszik kritikus negatív diagnosztikai jel.", "Kapcsolódó adat": "korrigált teljesítménytrend"})
    if not rows:
        rows.append({"Prioritás": 1, "Javaslat": "Tölts fel több futóedzést", "Miért": "Kevés megfigyelésből nem lehet stabil okokat becsülni.", "Kapcsolódó adat": "megfigyelésszám"})
    return pd.DataFrame(rows).sort_values("Prioritás").reset_index(drop=True)


def diagnostic_signal_chart(data: Any) -> pd.DataFrame:
    rows = [
        {"Tényező": row["Tényező"], "score": round(row["_score"], 3), "Irány": row["Irány"]}
        for row in _diagnostic_reason_rows(data)
        if row["Irány"] in {"Javító", "Rontó"}
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", key=lambda series: series.abs(), ascending=False).head(8)


def latest_weekly_statistics_table(weekly_features: pd.DataFrame) -> pd.DataFrame:
    if weekly_features.empty:
        return pd.DataFrame()
    latest = {str(key): value for key, value in weekly_features.iloc[-1].to_dict().items()}
    fields = [
        ("Dátum", "date"),
        ("7 napos km", "distance_7d_km"),
        ("7 napos idő (perc)", "time_7d_min"),
        ("7 napos training load", "load_7d"),
        ("fatigue-adjusted training load", "fatigue_adjusted_load_7d"),
        ("14 napos training load", "load_14d"),
        ("14 napos korrigált training load", "fatigue_adjusted_load_14d"),
        ("28 napos training load", "load_28d"),
        ("28 napos korrigált training load", "fatigue_adjusted_load_28d"),
        ("laza arány", "easy_fraction"),
        ("Közepes arány", "moderate_fraction"),
        ("minőségi/kemény arány", "hard_fraction"),
        ("hosszú futás (km)", "long_run_distance_km"),
        ("ACWR", "acute_chronic_workload_ratio"),
        ("Korrigált ACWR", "fatigue_adjusted_acute_chronic_workload_ratio"),
        ("Átlag power (W)", "avg_power_w_7d_mean"),
        ("normalized power (W)", "normalized_power_w_7d_mean"),
        ("Átlaghőmérséklet (°C)", "avg_temperature_c_7d_mean"),
        ("Aerob training effect", "total_training_effect_7d_mean"),
        ("Anaerob training effect", "total_anaerobic_training_effect_7d_mean"),
        ("tempó variability", "pace_variability_7d_mean"),
        ("szintemelkedés proxy", "elevation_gain_per_km_7d_mean"),
        ("HRV RMSSD", "hrv_rmssd_ms_7d_mean"),
    ]
    percent_fields = {"easy_fraction", "moderate_fraction", "hard_fraction"}
    rows = [
        {"Mutató": label, "Érték": percent(latest.get(field)) if field in percent_fields else latest.get(field, "n/a")}
        for label, field in fields
        if field in latest
    ]
    return pd.DataFrame(rows)


def _diagnostic_reason_rows(data: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _performance_summaries(data):
        change = _float_or_none(item.get("adjusted_change_s_per_km"))
        if change is None:
            continue
        confidence = _float(item.get("confidence"))
        distance = item.get("distance_range", "n/a")
        factor = item.get("primary_controlled_factors") or "kontrollált trend"
        if change < -1.0:
            rows.append(_diagnostic_row("Javító", f"{distance} teljesítménytrend", f"Korrigált tempó {abs(change):.1f} mp/km javulás; kontroll: {factor}", abs(change) / 30 * confidence, confidence))
        elif change > 1.0:
            rows.append(_diagnostic_row("Rontó", f"{distance} teljesítménytrend", f"Korrigált tempó {change:.1f} mp/km romlás; kontroll: {factor}", -change / 30 * confidence, confidence))
    rows.extend(_training_block_reasons(data))
    rows.extend(_fatigue_reasons(data))
    rows.extend(_run_walk_reasons(data))
    rows.extend(_activity_context_reasons(data))
    rows.extend(_ml_and_quality_reasons(data))
    return sorted(rows, key=lambda row: abs(float(row["_score"])), reverse=True)


def _diagnostic_row(direction: str, factor: str, evidence: str, score: float, confidence: float | None = None) -> dict[str, Any]:
    return {
        "Irány": direction,
        "Tényező": factor,
        "Bizonyíték": evidence,
        "Hatás": _signed_number(score),
        "confidence": percent(confidence if confidence is not None else min(1.0, abs(score))),
        "_score": round(max(-1.0, min(1.0, score)), 3),
    }


def _performance_summaries(data: Any) -> list[dict[str, Any]]:
    return _list_of_dicts(_dict(_data_attr(data, "performance_change", {})).get("summary"))


def _performance_changes(data: Any) -> list[dict[str, float]]:
    rows = []
    for item in _performance_summaries(data):
        change = _float_or_none(item.get("adjusted_change_s_per_km"))
        if change is not None:
            rows.append({"change": change, "confidence": _float(item.get("confidence"))})
    return rows


def _training_block_reasons(data: Any) -> list[dict[str, Any]]:
    table = _data_attr(data, "training_blocks", pd.DataFrame())
    if not isinstance(table, pd.DataFrame) or table.empty or "improvement_score" not in table:
        return []
    rows = []
    best = table.sort_values("improvement_score", ascending=False, na_position="last").iloc[0].to_dict()
    worst = table.sort_values("improvement_score", ascending=True, na_position="last").iloc[0].to_dict()
    if _float(best.get("improvement_score")) > 0:
        label = _block_type_label(str(best.get("block_type", "unknown")))
        rows.append(
            _diagnostic_row(
                "Javító",
                f"Segítő edzésblokk: {label}",
                f"{_block_range_text(best)} {label} blokk után javulás látszik",
                _float(best.get("improvement_score")) / 5,
                _float_or_none(best.get("confidence")),
            )
        )
    if _float(worst.get("improvement_score")) < 0:
        label = _block_type_label(str(worst.get("block_type", "unknown")))
        rows.append(
            _diagnostic_row(
                "Rontó",
                f"Rontó edzésblokk: {label}",
                f"{_block_range_text(worst)} {label} blokk után romlás látszik",
                _float(worst.get("improvement_score")) / 5,
                _float_or_none(worst.get("confidence")),
            )
        )
    return rows


def _fatigue_reasons(data: Any) -> list[dict[str, Any]]:
    fitness = _data_attr(data, "fitness_state", pd.DataFrame())
    weekly = _data_attr(data, "weekly_features", pd.DataFrame())
    rows: list[dict[str, Any]] = []
    if isinstance(fitness, pd.DataFrame) and not fitness.empty:
        latest = fitness.iloc[-1].to_dict()
        fatigue = _float(latest.get("fatigue"))
        form = _float(latest.get("form"))
        overload = _float(latest.get("overload_risk"))
        if form < 0 or fatigue > 35:
            rows.append(_diagnostic_row("Rontó", "fatigue / form", f"form {form:.1f}, fatigue {fatigue:.1f}", -min(1.0, (abs(min(form, 0)) + fatigue) / 100), overload or 0.6))
        if overload > 0.35:
            rows.append(_diagnostic_row("Rontó", "overload risk", f"overload risk {percent(overload)}", -overload, overload))
    if isinstance(weekly, pd.DataFrame) and not weekly.empty:
        latest = weekly.iloc[-1].to_dict()
        hard = _float(latest.get("hard_fraction"))
        if hard > 0.22:
            rows.append(_diagnostic_row("Rontó", "intenzitáseloszlás", f"minőségi/kemény arány {percent(hard)} az utolsó 7 napban", -hard, 0.65))
        adjusted = _float_or_none(latest.get("fatigue_adjusted_load_7d"))
        raw = _float_or_none(latest.get("load_7d"))
        if adjusted is not None and raw is not None and raw > 0 and adjusted < raw * 0.95:
            rows.append(_diagnostic_row("Javító", "fatigue-adjusted training load", f"A korrigált training load alacsonyabb: {number(adjusted, 1)} vs {number(raw, 1)}", (raw - adjusted) / raw, 0.7))
    return rows


def _run_walk_reasons(data: Any) -> list[dict[str, Any]]:
    activity = _data_attr(data, "activity_features", pd.DataFrame())
    if not isinstance(activity, pd.DataFrame) or activity.empty:
        return []
    rows = []
    forced = _series_mean(activity, "forced_walk_score")
    multiplier = _series_mean(activity, "run_walk_fatigue_multiplier")
    planned = _series_mean(activity, "planned_run_walk_score")
    if multiplier is not None and multiplier < 0.98 and (planned or 0.0) > 0.2:
        rows.append(_diagnostic_row("Javító", "Tervezett futás-séta", f"Átlagos fáradtsági szorzó {percent(multiplier)}", 1.0 - multiplier, planned or 0.6))
    if forced is not None and forced > 0.45:
        rows.append(_diagnostic_row("Rontó", "Kényszerű futás-séta", f"Kényszerű futás-séta pontszám {percent(forced)}", -forced, forced))
    return rows


def _activity_context_reasons(data: Any) -> list[dict[str, Any]]:
    activity = _data_attr(data, "activity_features", pd.DataFrame())
    if not isinstance(activity, pd.DataFrame) or activity.empty:
        return []
    rows: list[dict[str, Any]] = []
    temperature = _series_mean(activity, "avg_temperature_c")
    if temperature is not None and temperature >= 24:
        rows.append(_diagnostic_row("Rontó", "Hőség / környezet", f"Átlagos edzéshőmérséklet {number(temperature, 1)} °C", -min(0.7, (temperature - 20.0) / 20.0), 0.55))
    elevation = _series_mean(activity, "elevation_gain_per_km")
    if elevation is not None and elevation >= 12:
        rows.append(_diagnostic_row("Korlát", "Emelkedés/terep proxy", f"Átlagos emelkedés {number(elevation, 1)} m/km; ez felszín helyett terep-proxy", -min(0.6, elevation / 60.0), 0.55))
    forced = _series_mean(activity, "forced_walk_score") or 0.0
    efficiency = _series_mean(activity, "power_hr_efficiency")
    if efficiency is not None and efficiency > 0 and forced < 0.3:
        rows.append(_diagnostic_row("Korlát", "Teljesítmény/pulzus hatékonyság", f"Átlagos power/HR proxy {number(efficiency, 2)}", min(0.5, efficiency / 5.0), 0.5))
    dynamics = _series_mean(activity, "vertical_ratio")
    if dynamics is not None:
        rows.append(_diagnostic_row("Korlát", "Futódinamika", f"Vertikális arány átlaga {number(dynamics, 2)}", -min(0.35, max(0.0, dynamics - 8.5) / 8.0), 0.45))
    return rows


def _ml_and_quality_reasons(data: Any) -> list[dict[str, Any]]:
    rows = []
    ml = _dict(_data_attr(data, "ml_result", {}))
    importance = _dict(ml.get("feature_importance"))
    if importance:
        feature, score = max(importance.items(), key=lambda item: _float(item[1]))
        rows.append(_diagnostic_row("Korlát", "ML feature importance", f"A legerősebb feature: {feature}", _float(score), _float(ml.get("confidence"))))
    positive = _list_of_dicts(ml.get("positive_temporal_drivers"))
    negative = _list_of_dicts(ml.get("negative_temporal_drivers"))
    if positive:
        driver = positive[0]
        rows.append(_diagnostic_row("Javító", "Temporal ML driver", f"Pozitív temporal feature: {driver.get('feature')}", _float(driver.get("strength")), _float(ml.get("confidence"))))
    if negative:
        driver = negative[0]
        rows.append(_diagnostic_row("Rontó", "Temporal ML driver", f"Negatív temporal feature: {driver.get('feature')}", -_float(driver.get("strength")), _float(ml.get("confidence"))))
    warnings = _list(_data_attr(data, "warnings", [])) + _list(ml.get("warnings"))
    if warnings:
        rows.append(_diagnostic_row("Korlát", "Adat / model figyelmeztetés", str(warnings[0]), -0.25, 0.5))
    activity = _data_attr(data, "activity_features", pd.DataFrame())
    quality = _series_mean(activity, "data_quality_score") if isinstance(activity, pd.DataFrame) else None
    if quality is not None and quality < 0.6:
        rows.append(_diagnostic_row("Rontó", "Adatminőség", f"Átlagos adatminőség {percent(quality)}", -(1.0 - quality), 0.55))
    return rows


def _series_mean(table: pd.DataFrame, column: str) -> float | None:
    if column not in table or table.empty:
        return None
    values = pd.to_numeric(table[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _top_factor(rows: list[dict[str, Any]]) -> str:
    return str(rows[0]["Tényező"]) if rows else "n/a"


def _data_attr(data: Any, name: str, default: Any = None) -> Any:
    return getattr(data, name, default)


def training_block_timeline_table(training_blocks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if training_blocks.empty:
        return pd.DataFrame()
    for index, row in enumerate(training_blocks.reset_index(drop=True).to_dict("records")):
        block_type = str(row.get("block_type", "unknown"))
        start = _to_timestamp(row.get("start_date"))
        end = _to_timestamp(row.get("end_date"))
        duration = _block_duration_days(start, end)
        rows.append(
            {
                "Blokk": f"{index + 1}. {_block_type_label(block_type)}",
                "Típus": _block_type_label(block_type),
                "Kategória": block_type,
                "Szín": _block_type_color(block_type),
                "Szín ikon": _block_type_icon(block_type),
                "Kezdet": start,
                "Vég": end,
                "Időtartam (nap)": duration,
                "heti km": _float_or_none(row.get("volume")),
                "laza": _float_or_none(row.get("easy_fraction")),
                "Közepes": _float_or_none(row.get("moderate_fraction")),
                "minőségi/kemény": _float_or_none(row.get("hard_fraction")),
                "improvement": _float_or_none(row.get("improvement_score")),
                "confidence": _float_or_none(row.get("confidence")),
                "Valószínű driver": _block_type_label(str(row.get("likely_improvement_driver", block_type))),
            }
        )
    return pd.DataFrame(rows)


def training_block_summary_values(training_blocks: pd.DataFrame) -> dict[str, str]:
    timeline = training_block_timeline_table(training_blocks)
    if timeline.empty:
        return {
            "Blokkok": "0",
            "Leghosszabb blokk": "n/a",
            "Legjobb improvement": "n/a",
            "Legnagyobb romlás": "n/a",
            "Átlagos confidence": "n/a",
        }
    best = timeline.sort_values("improvement", ascending=False, na_position="last").iloc[0]
    worst = timeline.sort_values("improvement", ascending=True, na_position="last").iloc[0]
    longest = timeline.sort_values("Időtartam (nap)", ascending=False, na_position="last").iloc[0]
    return {
        "Blokkok": str(len(timeline)),
        "Leghosszabb blokk": f"{longest['Típus']} ({longest['Időtartam (nap)']} nap)",
        "Legjobb improvement": f"{best['Típus']} ({_signed_number(best['improvement'])})",
        "Legnagyobb romlás": f"{worst['Típus']} ({_signed_number(worst['improvement'])})",
        "Átlagos confidence": percent(_float_or_none(timeline["confidence"].mean())),
    }


def training_block_composition_table(training_blocks: pd.DataFrame) -> pd.DataFrame:
    timeline = training_block_timeline_table(training_blocks)
    if timeline.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Blokk": timeline["Blokk"],
            "Típus": timeline["Típus"],
            "laza": timeline["laza"].apply(percent),
            "Közepes": timeline["Közepes"].apply(percent),
            "minőségi/kemény": timeline["minőségi/kemény"].apply(percent),
            "heti km": timeline["heti km"].apply(lambda value: number(value, 1)),
        }
    )


def training_block_improvement_table(training_blocks: pd.DataFrame) -> pd.DataFrame:
    if training_blocks.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(training_blocks.reset_index(drop=True).to_dict("records")):
        block_type = str(row.get("block_type", "unknown"))
        rows.append(
            {
                "Blokk": f"{index + 1}. {_block_type_label(block_type)}",
                "Előtte": number(row.get("performance_before"), 2),
                "Utána": number(row.get("performance_after"), 2),
                "improvement": _signed_number(row.get("improvement_score")),
                "Valószínű driver": _block_type_label(str(row.get("likely_improvement_driver", block_type))),
                "confidence": percent(row.get("confidence")),
            }
        )
    return pd.DataFrame(rows)


def training_block_volume_chart(training_blocks: pd.DataFrame) -> pd.DataFrame:
    timeline = training_block_timeline_table(training_blocks)
    if timeline.empty:
        return pd.DataFrame()
    return timeline[["Blokk", "heti km"]].dropna()


def training_block_improvement_chart(training_blocks: pd.DataFrame) -> pd.DataFrame:
    timeline = training_block_timeline_table(training_blocks)
    if timeline.empty:
        return pd.DataFrame()
    return timeline[["Blokk", "improvement"]].dropna()


def training_block_detail_values(training_blocks: pd.DataFrame, block_label: str) -> dict[str, str]:
    timeline = training_block_timeline_table(training_blocks)
    if timeline.empty:
        return {}
    matches = timeline[timeline["Blokk"] == block_label]
    row = matches.iloc[0] if not matches.empty else timeline.iloc[0]
    return {
        "Típus": str(row.get("Típus", "n/a")),
        "Időszak": f"{_date_text(row.get('Kezdet'))} - {_date_text(row.get('Vég'))}",
        "Időtartam": f"{row.get('Időtartam (nap)', 'n/a')} nap",
        "heti km": f"{number(row.get('heti km'), 1)} km/hét",
        "improvement": _signed_number(row.get("improvement")),
        "confidence": percent(row.get("confidence")),
        "Driver": str(row.get("Valószínű driver", "n/a")),
        "Intenzitás": f"laza {percent(row.get('laza'))}, közepes {percent(row.get('Közepes'))}, minőségi/kemény {percent(row.get('minőségi/kemény'))}",
    }


def performance_change_summary_table(performance_change: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in _list_of_dicts(_dict(performance_change).get("summary")):
        rows.append(
            {
                "Táv": row.get("distance_range", ""),
                "raw változás": _pace_delta(row.get("raw_change_s_per_km")),
                "Korrigált változás": _pace_delta(row.get("adjusted_change_s_per_km")),
                "improvement": _pace_delta(-_float(row.get("adjusted_change_s_per_km"))),
                "Változás %": percent(row.get("percent_change"), digits=1, signed=True),
                "Megfigyelések": row.get("observation_count", 0),
                "model": row.get("model_type", "n/a"),
                "confidence": percent(row.get("confidence")),
                "Kontrollált fő factors": row.get("primary_controlled_factors", "n/a"),
            }
        )
    return pd.DataFrame(rows)


def performance_change_metric_values(performance_change: dict[str, Any]) -> dict[str, str]:
    summaries = _list_of_dicts(_dict(performance_change).get("summary"))
    observations = _list_of_dicts(_dict(performance_change).get("observations"))
    usable = [row for row in summaries if _float_or_none(row.get("adjusted_change_s_per_km")) is not None]
    if usable:
        best = min(usable, key=lambda row: _float(row.get("adjusted_change_s_per_km")))
        worst = max(usable, key=lambda row: _float(row.get("adjusted_change_s_per_km")))
        best_text = f"{best.get('distance_range', 'n/a')} ({_pace_delta(-_float(best.get('adjusted_change_s_per_km')))})"
        worst_text = f"{worst.get('distance_range', 'n/a')} ({_pace_delta(-_float(worst.get('adjusted_change_s_per_km')))})"
    else:
        best_text = "n/a"
        worst_text = "n/a"
    return {
        "Legnagyobb improvement": best_text,
        "Legnagyobb romlás": worst_text,
        "Használható távok": str(len(summaries)),
        "Megfigyelések": str(len(observations)),
    }


def performance_change_distance_options(performance_change: dict[str, Any]) -> list[str]:
    return [str(row.get("distance_range")) for row in _list_of_dicts(_dict(performance_change).get("summary")) if row.get("distance_range")]


def performance_change_chart_table(performance_change: dict[str, Any], distance_range: str) -> pd.DataFrame:
    rows = [
        {
            "Dátum": row.get("date"),
            "raw tempó (perc/km)": _pace_minutes(row.get("observed_pace_s_per_km")),
            "Korrigált tempó (perc/km)": _pace_minutes(row.get("adjusted_pace_s_per_km")),
        }
        for row in _list_of_dicts(_dict(performance_change).get("observations"))
        if row.get("distance_range") == distance_range
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["Dátum"] = pd.to_datetime(table["Dátum"], errors="coerce")
    return table.dropna(subset=["Dátum"]).sort_values("Dátum")


def performance_change_observation_table(performance_change: dict[str, Any], distance_range: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in _list_of_dicts(_dict(performance_change).get("observations")):
        if row.get("distance_range") != distance_range:
            continue
        rows.append(
            {
                "Dátum": row.get("date"),
                "raw tempó": pace(row.get("observed_pace_s_per_km")),
                "Korrigált tempó": pace(row.get("adjusted_pace_s_per_km")),
                "Futás-séta": row.get("run_walk_type", "unknown"),
                "szintemelkedés proxy": f"{number(row.get('elevation_gain_per_km'), 1)} m/km",
                "fatigue": number(row.get("fatigue"), 1),
                "ACWR": number(row.get("acute_chronic_workload_ratio"), 2),
            }
        )
    return pd.DataFrame(rows)


def performance_change_importance_table(performance_change: dict[str, Any], distance_range: str) -> pd.DataFrame:
    model = _dict(_dict(performance_change).get("models")).get(distance_range, {})
    importance = _dict(_dict(model).get("feature_importance"))
    rows = [{"feature": key, "feature importance": round(_float(value) * 100, 1)} for key, value in importance.items()]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("feature importance", ascending=False)


def performance_change_model_notes(performance_change: dict[str, Any], distance_range: str) -> dict[str, str]:
    model = _dict(_dict(performance_change).get("models")).get(distance_range, {})
    return {
        "model": str(_dict(model).get("model_type", "n/a")),
        "confidence": percent(_dict(model).get("confidence")),
        "Fallback": str(_dict(model).get("fallback_reason") or ""),
        "Terep": "szintemelkedés proxy, mert a FIT fájl nem ad megbízható felszíntípust.",
    }


def duration(seconds_value: Any) -> str:
    try:
        seconds = int(round(float(seconds_value)))
    except (TypeError, ValueError):
        return "n/a"
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _plan_improvement_pct(
    candidate_row: dict[str, Any],
    horizon_days: int | None,
    distance_km: float | None,
    diminishing_returns_factor: Any,
) -> float:
    if not candidate_row:
        return 0.0
    benefit = (
        0.035 * _float(candidate_row.get("expected_adaptation"))
        + 0.015 * _float(candidate_row.get("missing_stimulus_score"))
        + 0.010 * _float(candidate_row.get("race_specificity"))
    )
    cost = (
        0.020 * _float(candidate_row.get("expected_fatigue"))
        + 0.030 * _float(candidate_row.get("overload_risk"))
        + 0.020 * _float(candidate_row.get("constraint_penalty"))
    )
    raw_improvement = max(-0.03, min(0.06, benefit - cost))
    improvement = raw_improvement * _horizon_effect_scale(horizon_days) * _distance_effect_scale(distance_km)
    if improvement > 0:
        improvement *= _diminishing_returns_factor(diminishing_returns_factor)
    if candidate_row.get("is_valid") is False:
        improvement = min(0.0, improvement)
    return improvement


def _diminishing_returns_factor(value: Any) -> float:
    if value is None:
        return 1.0
    return max(0.65, min(1.0, _float(value)))


def _diminishing_returns_reason_label(reason: Any) -> str:
    return {
        "learned_from_history": "Történetből becsült",
        "insufficient_true_7d_history": "Kevés lezárt 7 napos minta",
        "too_little_fitness_variation": "Kevés edzettségi változás",
        "non_diminishing_history": "Nem látszik csökkenő hozam",
        "non_positive_reference_response": "Instabil referencia",
        "fallback": "Fallback",
    }.get(str(reason), "n/a")


def _diminishing_returns_raw_ratio(row: Mapping[str, Any]) -> float | None:
    raw_ratio = _float_or_none(row.get("raw_ratio"))
    if raw_ratio is not None:
        return raw_ratio
    current = _float_or_none(row.get("current_response"))
    reference = _float_or_none(row.get("reference_response"))
    if current is None or reference is None or reference <= 0:
        return None
    return current / reference


def _horizon_effect_scale(horizon_days: int | None) -> float:
    days = 7 if horizon_days is None else max(1, min(28, int(horizon_days)))
    return min(1.0, days / 28.0)


def _distance_effect_scale(distance_km: float | None) -> float:
    if distance_km is None:
        return 0.35
    if distance_km <= 1.5:
        return 0.45
    if distance_km <= 5:
        return 0.35
    if distance_km <= 10:
        return 0.30
    if distance_km <= 21.1:
        return 0.20
    return 0.12


def _distance_from_label(label: str) -> float | None:
    normalized = label.lower().replace("km", "").strip()
    try:
        return float(normalized)
    except ValueError:
        return None


def _percent(value: float) -> str:
    return percent(value, digits=2, signed=True)


def _signed_duration(seconds_value: float) -> str:
    sign = "+" if seconds_value >= 0 else "-"
    return f"{sign}{duration(abs(seconds_value))}"


def _pace_delta(value: Any) -> str:
    seconds = _float_or_none(value)
    if seconds is None:
        return "n/a"
    sign = "+" if seconds >= 0 else "-"
    return f"{sign}{abs(seconds):.1f} mp/km"


def _pace_minutes(value: Any) -> float | None:
    seconds = _float_or_none(value)
    if seconds is None:
        return None
    return round(seconds / 60.0, 2)


def _minutes_to_seconds(value: Any) -> float | None:
    minutes = _float_or_none(value)
    return None if minutes is None else minutes * 60.0


def _unit(value: Any, unit: str, digits: int = 1) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "n/a"
    rendered = number(numeric, digits)
    return f"{rendered} {unit}".strip()


def _activity_records(table: pd.DataFrame, activity_id: str) -> pd.DataFrame:
    if table.empty or "activity_id" not in table:
        return pd.DataFrame()
    return table[table["activity_id"].astype(str) == str(activity_id)].copy()


def _numeric_column(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table:
        return pd.Series(dtype=float)
    return pd.to_numeric(table[column], errors="coerce")


def _add_chart_column(
    target: pd.DataFrame,
    source: pd.DataFrame,
    source_column: str,
    label: str,
    *,
    transform=None,
) -> None:
    if source_column not in source:
        return
    values = pd.to_numeric(source[source_column], errors="coerce")
    if transform is not None:
        values = values.apply(transform)
    if not values.dropna().empty:
        target[label] = values


def _signed_number(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:+.2f}"


def _block_duration_days(start: Any, end: Any) -> int | None:
    if pd.isna(start) or pd.isna(end):
        return None
    return int(max(1, (end - start).days + 1))


def _block_range_text(block: Mapping[Any, Any]) -> str:
    start = str(block.get("start_date") or "").strip()
    end = str(block.get("end_date") or "").strip()
    if start and end:
        return f"{start} - {end}:"
    if start:
        return f"{start}:"
    return ""


def _to_timestamp(value: Any) -> Any:
    return pd.Timestamp(value) if value is not None else pd.NaT


def _date_text(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return value.date().isoformat()
    except AttributeError:
        return str(value)


def _block_type_label(value: str) -> str:
    return {
        "recovery": "Regeneráló blokk",
        "base": "Alapozás",
        "threshold": "Küszöb",
        "long_run_endurance": "Hosszú futás",
        "overload": "Túlterhelés",
        "unknown": "Ismeretlen",
    }.get(value, value.replace("_", " ").title())


def _block_type_color(value: str) -> str:
    return {
        "recovery": "#2ca25f",
        "base": "#3182bd",
        "threshold": "#f28e2b",
        "long_run_endurance": "#756bb1",
        "overload": "#de2d26",
        "unknown": "#8c8c8c",
    }.get(value, "#8c8c8c")


def _block_type_icon(value: str) -> str:
    return {
        "recovery": "🟩",
        "base": "🟦",
        "threshold": "🟧",
        "long_run_endurance": "🟪",
        "overload": "🟥",
        "unknown": "⬜",
    }.get(value, "⬜")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
        if numeric != numeric:
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _workout_label(value: str) -> str:
    return {
        "rest": "Pihenő",
        "recovery": "regeneráló futás",
        "easy": "laza futás",
        "long_run": "Hosszú futás",
        "threshold": "küszöbedzés",
        "vo2max": "VO2max edzés",
        "speed": "résztáv / VO2max",
        "strength": "Erősítés",
        "cross_train": "Keresztedzés",
    }.get(value, value.replace("_", " ").title())


def _intensity_label(value: str) -> str:
    return {"rest": "Pihenő", "easy": "laza", "moderate": "közepes", "hard": "minőségi/kemény"}.get(value, value)


def _distance_cell(value: Any) -> str:
    numeric = _float_or_none(value)
    return "" if numeric is None or numeric <= 0 else number(numeric)


def _purpose_label(value: Any) -> str:
    text = str(value or "").strip()
    return {
        "Absorb recent training": "az edzésterhelés feldolgozása",
        "Reduce fatigue while preserving consistency": "regeneráció, a futóritmus megtartásával",
        "Build aerobic base with low stress": "aerob alap építése alacsony stresszel",
        "Improve durability and long-run endurance": "állóképesség és hosszúfutás-tűrés fejlesztése",
        "Improve sustainable speed": "küszöbtempó és tartós sebesség fejlesztése",
        "Improve 1-10 km aerobic ceiling": "VO2max és 1-10 km-es aerob plafon fejlesztése",
        "Improve neuromuscular economy": "futómozgás gazdaságossága, rövid repülők",
        "Training stimulus": "edzésinger",
    }.get(text, text)


def _reason_label(value: str) -> str:
    if value.startswith("Addresses weak system:"):
        weak_system = value.split(":", 1)[1].strip().replace("_", " ")
        return f"A fókusz a jelenlegi gyenge láncszemre esik: {weak_system}."
    if value.startswith("Candidate family:"):
        family = value.split(":", 1)[1].strip()
        return f"Jelölt tervtípus: {plan_family_label(family)}."
    if value.startswith("Historical improvement driver matched:"):
        driver = value.split(":", 1)[1].strip()
        return f"A történeti improvement driver illeszkedik a tervhez: {driver}."
    return {
        "Keeps hard days within limit": "A minőségi napok száma a megadott limit alatt marad.",
        "Includes recovery after quality session": "A minőségi edzés után van regeneráló nap.",
        "Prioritizes recovery and lower overload risk": "A regenerációt és alacsonyabb overload risket priorizálja.",
        "Includes run-walk-specific prescriptions": "Futás-séta specifikus előírásokat tartalmaz.",
        "Historical hard-run signal suppressed by fatigue/overload safety rules": "A történeti hard-run jel safety okból nem növelte a minőségi edzések súlyát.",
        "ML prediction blended into optimizer score": "Az ML prediction bekerült az optimizer score számításába.",
        "Fallback recovery week": "Biztonsági regeneráló hét.",
        "No available training days": "Nincs elérhető edzésnap.",
        "long_run_share_above_35_percent": "A hosszú futás aránya 35% felett lenne.",
        "too_much_moderate_intensity": "Túl sok lenne a közepes intenzitású futás.",
        "no_recovery_after_hard_day": "Hiányzik a regeneráció minőségi nap után.",
        "too_little_easy_volume": "Kevés lenne a laza futás aránya.",
        "excessive_load_jump": "Túl nagy lenne a training load ugrása.",
    }.get(value, value.replace("_", " "))


def _prescription_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("easy run", "laza futás")
    text = text.replace("sec walk", "mp séta")
    text = text.replace("min ", "perc ")
    text = text.replace(", keep HR in Z2 and keep the route flat", ", pulzus Z2-ben, lehetőleg sík útvonalon")
    return text


def _pace_zone_label(value: str) -> str:
    return {
        "recovery": "regeneráló",
        "easy": "laza",
        "steady": "Egyenletes",
        "marathon": "Maraton",
        "threshold": "küszöb",
        "VO2": "VO2",
        "speed": "Gyorsaság",
    }.get(value, value)


def _segment_state_label(value: str) -> str:
    return {
        "stopped": "állás",
        "walking": "séta",
        "recovery_jog": "regeneráló kocogás",
        "easy_running": "laza futás",
        "steady_running": "egyenletes futás",
        "hard_running": "kemény futás",
        "unknown": "ismeretlen",
    }.get(value, value.replace("_", " "))
