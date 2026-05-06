from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal


ActivityConfidence = Literal["high", "medium", "low", "rejected"]
SegmentState = Literal[
    "stopped",
    "walking",
    "recovery_jog",
    "easy_running",
    "steady_running",
    "hard_running",
    "unknown",
]


@dataclass
class Activity:
    activity_id: str
    source_file: str
    start_time: datetime | None
    sport: str | None
    sub_sport: str | None
    keep: bool
    keep_reason: str
    activity_type_confidence: ActivityConfidence
    activity_type_reason: str
    total_distance_m: float | None = None
    total_elapsed_s: float | None = None
    total_timer_s: float | None = None
    total_moving_s: float | None = None
    avg_hr: float | None = None
    max_hr: float | None = None
    min_hr: float | None = None
    avg_speed_mps: float | None = None
    max_speed_mps: float | None = None
    avg_cadence_spm: float | None = None
    total_ascent_m: float | None = None
    total_descent_m: float | None = None
    min_altitude_m: float | None = None
    max_altitude_m: float | None = None
    total_calories: float | None = None
    total_training_effect: float | None = None
    total_anaerobic_training_effect: float | None = None
    avg_temperature_c: float | None = None
    max_temperature_c: float | None = None
    avg_power_w: float | None = None
    max_power_w: float | None = None
    normalized_power_w: float | None = None
    total_work_kj: float | None = None
    total_strides: float | None = None
    avg_running_cadence_spm: float | None = None
    max_running_cadence_spm: float | None = None
    avg_step_length_mm: float | None = None
    avg_vertical_oscillation_mm: float | None = None
    avg_vertical_ratio: float | None = None
    avg_stance_time_ms: float | None = None
    lap_count: int = 0
    lap_pace_variability: float | None = None
    lap_hr_variability: float | None = None
    lap_power_variability: float | None = None
    hrv_count: int = 0
    hrv_median_rr_ms: float | None = None
    hrv_rmssd_ms: float | None = None
    warnings: str = ""


@dataclass
class Record:
    activity_id: str
    timestamp: datetime | None
    elapsed_s: float
    distance_m: float | None = None
    speed_mps: float | None = None
    pace_s_per_km: float | None = None
    heart_rate_bpm: float | None = None
    cadence_spm: float | None = None
    altitude_m: float | None = None
    position_lat: float | None = None
    position_long: float | None = None
    power_w: float | None = None
    accumulated_power_w: float | None = None
    temperature_c: float | None = None
    fractional_cadence: float | None = None
    activity_type: str | None = None
    vertical_oscillation_mm: float | None = None
    step_length_mm: float | None = None
    vertical_ratio: float | None = None
    stance_time_ms: float | None = None


@dataclass
class Segment:
    activity_id: str
    segment_id: int
    start_time: datetime | None
    end_time: datetime | None
    duration_s: float
    distance_m: float
    state: SegmentState
    avg_speed_mps: float | None = None
    avg_pace_s_per_km: float | None = None
    avg_hr: float | None = None
    max_hr: float | None = None
    avg_cadence_spm: float | None = None
    elevation_gain_m: float | None = None
    hr_drop_from_previous_run: float | None = None


def row(obj) -> dict:
    data = asdict(obj) if hasattr(obj, "__dataclass_fields__") else dict(obj)
    for key, value in list(data.items()):
        if isinstance(value, datetime):
            data[key] = value.isoformat()
    return data
