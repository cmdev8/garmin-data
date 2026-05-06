from __future__ import annotations

import hashlib
import io
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median, pstdev
from typing import Any

from garmin_runner.config import AthleteConfig
from garmin_runner.fit.filters import classify_activity
from garmin_runner.fit.schema import Activity, Record


def parse_fit_directory(fit_dir: str | Path, config: AthleteConfig) -> tuple[list[Activity], list[Record], list[str]]:
    root = Path(fit_dir)
    activities: list[Activity] = []
    records: list[Record] = []
    warnings: list[str] = []

    for path in sorted(root.rglob("*.fit")):
        parsed_activities, parsed_records, parsed_warnings = parse_fit_file(path)
        warnings.extend(parsed_warnings)
        by_activity: dict[str, list[Record]] = defaultdict(list)
        for record in parsed_records:
            by_activity[record.activity_id].append(record)
        for activity in parsed_activities:
            classify_activity(activity, by_activity.get(activity.activity_id, []), config)
        activities.extend(parsed_activities)
        records.extend(parsed_records)

    return activities, records, warnings


def parse_fit_payloads(payloads: list[tuple[str, bytes]], config: AthleteConfig) -> tuple[list[Activity], list[Record], list[str]]:
    activities: list[Activity] = []
    records: list[Record] = []
    warnings: list[str] = []
    for original_name, content in payloads:
        parsed_activities, parsed_records, parsed_warnings = parse_fit_bytes(original_name, content)
        warnings.extend(parsed_warnings)
        by_activity: dict[str, list[Record]] = defaultdict(list)
        for record in parsed_records:
            by_activity[record.activity_id].append(record)
        for activity in parsed_activities:
            classify_activity(activity, by_activity.get(activity.activity_id, []), config)
        activities.extend(parsed_activities)
        records.extend(parsed_records)
    return activities, records, warnings


def parse_fit_file(path: str | Path) -> tuple[list[Activity], list[Record], list[str]]:
    fit_path = Path(path)
    return _parse_fit_source(str(fit_path), str(fit_path))


def parse_fit_bytes(original_name: str, content: bytes) -> tuple[list[Activity], list[Record], list[str]]:
    return _parse_fit_source(io.BytesIO(content), original_name)


def _parse_fit_source(source, source_name: str) -> tuple[list[Activity], list[Record], list[str]]:
    try:
        from fitparse import FitFile
    except Exception as exc:
        return [_fallback_activity(source_name, f"fitparse unavailable: {exc}")], [], [f"{source_name}: fitparse unavailable"]

    warnings: list[str] = []
    try:
        fit = FitFile(source)
        session_messages = [_message_values(m) for m in fit.get_messages("session")]
        records_raw = [_message_values(m) for m in fit.get_messages("record")]
        laps_raw = [_message_values(m) for m in fit.get_messages("lap")]
        hrv_raw = [_message_values(m) for m in fit.get_messages("hrv")]
    except Exception as exc:
        activity = _fallback_activity(source_name, f"corrupted FIT skipped: {exc}")
        activity.keep = False
        activity.activity_type_confidence = "rejected"
        return [activity], [], [f"{source_name}: corrupted FIT skipped: {exc}"]

    if not session_messages:
        session_messages = [{}]
        warnings.append(f"{source_name}: no session messages found")
    if len(session_messages) > 1:
        warnings.append(f"{source_name}: multiple sessions found; records assigned to first session")

    base_id = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:12]
    activities: list[Activity] = []
    all_records: list[Record] = []
    for index, session in enumerate(session_messages):
        activity_id = f"{base_id}-{index + 1}"
        start = _dt(session.get("start_time") or session.get("timestamp"))
        lap_summary = _lap_summary(laps_raw) if index == 0 else {}
        hrv_summary = _hrv_summary(hrv_raw) if index == 0 else {}
        activity = Activity(
            activity_id=activity_id,
            source_file=source_name,
            start_time=start,
            sport=_string_value(session.get("sport")),
            sub_sport=_string_value(session.get("sub_sport")),
            keep=False,
            keep_reason="not classified",
            activity_type_confidence="rejected",
            activity_type_reason="not classified",
            total_distance_m=_float(session.get("total_distance")),
            total_elapsed_s=_float(session.get("total_elapsed_time")),
            total_timer_s=_float(session.get("total_timer_time")),
            total_moving_s=_float(session.get("total_moving_time")),
            avg_hr=_float(session.get("avg_heart_rate")),
            max_hr=_float(session.get("max_heart_rate")),
            min_hr=_float(session.get("min_heart_rate")),
            avg_speed_mps=_float(session.get("avg_speed") or session.get("enhanced_avg_speed")),
            max_speed_mps=_float(session.get("max_speed") or session.get("enhanced_max_speed")),
            avg_cadence_spm=_cadence_with_fraction(session.get("avg_cadence"), session.get("avg_fractional_cadence")),
            total_ascent_m=_float(session.get("total_ascent")),
            total_descent_m=_float(session.get("total_descent")),
            min_altitude_m=_float(session.get("min_altitude") or session.get("enhanced_min_altitude")),
            max_altitude_m=_float(session.get("max_altitude") or session.get("enhanced_max_altitude")),
            total_calories=_float(session.get("total_calories")),
            total_training_effect=_float(session.get("total_training_effect")),
            total_anaerobic_training_effect=_float(session.get("total_anaerobic_training_effect")),
            avg_temperature_c=_float(session.get("avg_temperature")),
            max_temperature_c=_float(session.get("max_temperature")),
            avg_power_w=_float(session.get("avg_power")),
            max_power_w=_float(session.get("max_power")),
            normalized_power_w=_float(session.get("normalized_power")),
            total_work_kj=_work_kj(session.get("total_work")),
            total_strides=_float(session.get("total_strides")),
            avg_running_cadence_spm=_cadence_with_fraction(session.get("avg_running_cadence") or session.get("avg_cadence"), session.get("avg_fractional_cadence")),
            max_running_cadence_spm=_cadence_with_fraction(session.get("max_running_cadence") or session.get("max_cadence"), session.get("max_fractional_cadence")),
            avg_step_length_mm=_float(session.get("avg_step_length")),
            avg_vertical_oscillation_mm=_float(session.get("avg_vertical_oscillation")),
            avg_vertical_ratio=_float(session.get("avg_vertical_ratio")),
            avg_stance_time_ms=_float(session.get("avg_stance_time")),
            lap_count=int(lap_summary.get("lap_count") or 0),
            lap_pace_variability=_float(lap_summary.get("lap_pace_variability")),
            lap_hr_variability=_float(lap_summary.get("lap_hr_variability")),
            lap_power_variability=_float(lap_summary.get("lap_power_variability")),
            hrv_count=int(hrv_summary.get("hrv_count") or 0),
            hrv_median_rr_ms=_float(hrv_summary.get("hrv_median_rr_ms")),
            hrv_rmssd_ms=_float(hrv_summary.get("hrv_rmssd_ms")),
            warnings="; ".join(warnings),
        )
        activities.append(activity)

        if index == 0:
            all_records.extend(_records(activity_id, records_raw, start))

    return activities, all_records, warnings


def _message_values(message) -> dict[str, Any]:
    values = {}
    for field in message:
        values[field.name] = field.value
    return values


def _records(activity_id: str, raw_records: list[dict[str, Any]], start_time: datetime | None) -> list[Record]:
    records: list[Record] = []
    first_timestamp: datetime | None = None
    last_distance: float | None = None
    for raw in raw_records:
        timestamp = _dt(raw.get("timestamp"))
        if first_timestamp is None:
            first_timestamp = timestamp or start_time
        elapsed = 0.0
        if timestamp and first_timestamp:
            elapsed = max(0.0, (timestamp - first_timestamp).total_seconds())
        distance = _float(raw.get("distance"))
        speed = _float(raw.get("speed") or raw.get("enhanced_speed"))
        if speed is None and distance is not None and last_distance is not None and records:
            dt = max(1.0, elapsed - records[-1].elapsed_s)
            speed = max(0.0, (distance - last_distance) / dt)
        if distance is not None:
            last_distance = distance
        records.append(
            Record(
                activity_id=activity_id,
                timestamp=timestamp,
                elapsed_s=elapsed,
                distance_m=distance,
                speed_mps=speed,
                pace_s_per_km=1000.0 / speed if speed and speed > 0 else None,
                heart_rate_bpm=_float(raw.get("heart_rate")),
                cadence_spm=_cadence_with_fraction(raw.get("cadence"), raw.get("fractional_cadence")),
                altitude_m=_float(raw.get("altitude") or raw.get("enhanced_altitude")),
                position_lat=_semicircles_to_degrees(raw.get("position_lat")),
                position_long=_semicircles_to_degrees(raw.get("position_long")),
                power_w=_float(raw.get("power")),
                accumulated_power_w=_float(raw.get("accumulated_power")),
                temperature_c=_float(raw.get("temperature")),
                fractional_cadence=_float(raw.get("fractional_cadence")),
                activity_type=_string_value(raw.get("activity_type")),
                vertical_oscillation_mm=_float(raw.get("vertical_oscillation")),
                step_length_mm=_float(raw.get("step_length")),
                vertical_ratio=_float(raw.get("vertical_ratio")),
                stance_time_ms=_float(raw.get("stance_time")),
            )
        )
    return records


def _lap_summary(raw_laps: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not raw_laps:
        return {"lap_count": 0}
    paces: list[float] = []
    hrs: list[float] = []
    powers: list[float] = []
    for lap in raw_laps:
        distance = _float(lap.get("total_distance"))
        timer = _float(lap.get("total_timer_time"))
        if distance and distance > 0 and timer and timer > 0:
            paces.append(timer / (distance / 1000.0))
        hr = _float(lap.get("avg_heart_rate"))
        if hr is not None:
            hrs.append(hr)
        power = _float(lap.get("avg_power") or lap.get("normalized_power"))
        if power is not None:
            powers.append(power)
    return {
        "lap_count": len(raw_laps),
        "lap_pace_variability": _cv(paces),
        "lap_hr_variability": _cv(hrs),
        "lap_power_variability": _cv(powers),
    }


def _hrv_summary(raw_hrv: list[dict[str, Any]]) -> dict[str, float | int | None]:
    rr_s: list[float] = []
    for row in raw_hrv:
        raw = row.get("time")
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        for value in values:
            number = _float(value)
            if number is not None and number > 0:
                rr_s.append(number)
    if not rr_s:
        return {"hrv_count": 0}
    diffs_ms = [abs(b - a) * 1000.0 for a, b in zip(rr_s, rr_s[1:])]
    rmssd = (sum(value * value for value in diffs_ms) / len(diffs_ms)) ** 0.5 if diffs_ms else None
    return {
        "hrv_count": len(rr_s),
        "hrv_median_rr_ms": median(rr_s) * 1000.0,
        "hrv_rmssd_ms": rmssd,
    }


def _fallback_activity(path: str | Path, reason: str) -> Activity:
    return Activity(
        activity_id=hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12],
        source_file=str(path),
        start_time=None,
        sport=None,
        sub_sport=None,
        keep=False,
        keep_reason=reason,
        activity_type_confidence="rejected",
        activity_type_reason=reason,
        warnings=reason,
    )


def _dt(value) -> datetime | None:
    return value if isinstance(value, datetime) else None


def _float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _cadence(value) -> float | None:
    number = _float(value)
    return number * 2.0 if number is not None and number < 120 else number


def _cadence_with_fraction(value, fraction) -> float | None:
    number = _float(value)
    if number is None:
        return None
    fractional = _float(fraction) or 0.0
    return _cadence(number + fractional)


def _work_kj(value) -> float | None:
    number = _float(value)
    return None if number is None else number / 1000.0


def _cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return pstdev(values) / avg if avg else None


def _string_value(value) -> str | None:
    if value is None:
        return None
    return str(value).lower()


def _semicircles_to_degrees(value) -> float | None:
    number = _float(value)
    if number is None:
        return None
    return number * (180.0 / 2**31)
