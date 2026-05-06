from __future__ import annotations

from collections import defaultdict
from statistics import mean

from garmin_runner.config import AthleteConfig
from garmin_runner.fit.schema import Record, Segment, SegmentState


def segment_records(records: list[Record], config: AthleteConfig) -> list[Segment]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.activity_id].append(record)

    segments: list[Segment] = []
    for activity_id, activity_records in grouped.items():
        activity_records = sorted(activity_records, key=lambda r: r.elapsed_s)
        segments.extend(_segment_activity(activity_id, activity_records, config))
    return segments


def _segment_activity(activity_id: str, records: list[Record], config: AthleteConfig) -> list[Segment]:
    if not records:
        return []
    labels: list[SegmentState] = [_label_record(record) for record in records]
    min_duration = max(1, config.run_walk.min_segment_duration_s)
    raw: list[tuple[int, int, SegmentState]] = []
    start = 0
    for index in range(1, len(records)):
        if labels[index] != labels[start]:
            raw.append((start, index - 1, labels[start]))
            start = index
    raw.append((start, len(records) - 1, labels[start]))

    merged = _merge_short(raw, records, min_duration)
    return [_make_segment(activity_id, segment_id, records, s, e, state) for segment_id, (s, e, state) in enumerate(merged, 1)]


def _label_record(record: Record) -> SegmentState:
    speed = record.speed_mps
    cadence = record.cadence_spm
    hr = record.heart_rate_bpm
    if speed is not None and speed < 0.25:
        return "stopped"
    if speed is None and cadence is None:
        return "unknown"
    if (cadence is not None and cadence < 125) or (speed is not None and speed < 1.8):
        return "walking"
    if speed is not None and speed >= 4.2:
        return "hard_running"
    if hr is not None and hr >= 165 and speed is not None and speed >= 3.2:
        return "hard_running"
    if speed is not None and speed < 2.4:
        return "recovery_jog"
    if speed is not None and speed < 3.4:
        return "easy_running"
    return "steady_running"


def _merge_short(
    segments: list[tuple[int, int, SegmentState]], records: list[Record], min_duration: int
) -> list[tuple[int, int, SegmentState]]:
    if len(segments) <= 1:
        return segments
    result: list[tuple[int, int, SegmentState]] = []
    for segment in segments:
        start, end, state = segment
        duration = max(0.0, records[end].elapsed_s - records[start].elapsed_s)
        if result and duration < min_duration:
            prev_start, prev_end, prev_state = result[-1]
            result[-1] = (prev_start, end, prev_state)
        else:
            result.append(segment)
    return result


def _make_segment(
    activity_id: str, segment_id: int, records: list[Record], start: int, end: int, state: SegmentState
) -> Segment:
    subset = records[start : end + 1]
    distance = _distance(subset)
    speeds = [r.speed_mps for r in subset if r.speed_mps is not None]
    hrs = [r.heart_rate_bpm for r in subset if r.heart_rate_bpm is not None]
    cadences = [r.cadence_spm for r in subset if r.cadence_spm is not None]
    elevation_gain = _elevation_gain(subset)
    avg_speed = mean(speeds) if speeds else None
    return Segment(
        activity_id=activity_id,
        segment_id=segment_id,
        start_time=subset[0].timestamp,
        end_time=subset[-1].timestamp,
        duration_s=max(0.0, subset[-1].elapsed_s - subset[0].elapsed_s),
        distance_m=distance,
        state=state,
        avg_speed_mps=avg_speed,
        avg_pace_s_per_km=1000.0 / avg_speed if avg_speed and avg_speed > 0 else None,
        avg_hr=mean(hrs) if hrs else None,
        max_hr=max(hrs) if hrs else None,
        avg_cadence_spm=mean(cadences) if cadences else None,
        elevation_gain_m=elevation_gain,
    )


def _distance(records: list[Record]) -> float:
    distances = [r.distance_m for r in records if r.distance_m is not None]
    if len(distances) >= 2:
        return max(0.0, distances[-1] - distances[0])
    speeds = [r.speed_mps for r in records if r.speed_mps is not None]
    duration = max(0.0, records[-1].elapsed_s - records[0].elapsed_s)
    return (mean(speeds) * duration) if speeds else 0.0


def _elevation_gain(records: list[Record]) -> float | None:
    alts = [r.altitude_m for r in records if r.altitude_m is not None]
    if len(alts) < 2:
        return None
    return sum(max(0.0, b - a) for a, b in zip(alts, alts[1:]))
