from __future__ import annotations

from collections import defaultdict
from statistics import mean

from garmin_runner.fit.schema import Segment


RUN_STATES = {"recovery_jog", "easy_running", "steady_running", "hard_running"}


def run_walk_metrics(segments: list[Segment]) -> dict[str, dict]:
    grouped: dict[str, list[Segment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.activity_id].append(segment)

    return {activity_id: _metrics(sorted(items, key=lambda s: s.segment_id)) for activity_id, items in grouped.items()}


def _metrics(segments: list[Segment]) -> dict:
    run_segments = [s for s in segments if s.state in RUN_STATES]
    walk_segments = [s for s in segments if s.state == "walking"]
    stopped_segments = [s for s in segments if s.state == "stopped"]
    total_time = sum(s.duration_s for s in segments) or 0.0
    run_time = sum(s.duration_s for s in run_segments)
    walk_time = sum(s.duration_s for s in walk_segments)
    stopped_time = sum(s.duration_s for s in stopped_segments)
    run_distance = sum(s.distance_m for s in run_segments)
    walk_distance = sum(s.distance_m for s in walk_segments)
    run_paces = [s.avg_pace_s_per_km for s in run_segments if s.avg_pace_s_per_km is not None]
    walk_paces = [s.avg_pace_s_per_km for s in walk_segments if s.avg_pace_s_per_km is not None]

    pace_decay = _decay([s.avg_pace_s_per_km for s in run_segments if s.avg_pace_s_per_km is not None])
    cadence_decay = _decay([s.avg_cadence_spm for s in run_segments if s.avg_cadence_spm is not None], higher_is_worse=False)
    forced_walk_score = min(1.0, max(0.0, pace_decay / 120.0 + cadence_decay / 30.0 + _duration_decay(run_segments)))
    planned_score = _regularity_score(run_segments) * _regularity_score(walk_segments) if walk_segments else 0.0

    return {
        "run_time_s": run_time,
        "walk_time_s": walk_time,
        "stopped_time_s": stopped_time,
        "run_distance_m": run_distance,
        "walk_distance_m": walk_distance,
        "run_fraction": run_time / total_time if total_time else 0.0,
        "walk_fraction": walk_time / total_time if total_time else 0.0,
        "number_of_walk_breaks": len(walk_segments),
        "average_run_segment_duration_s": mean([s.duration_s for s in run_segments]) if run_segments else None,
        "average_walk_segment_duration_s": mean([s.duration_s for s in walk_segments]) if walk_segments else None,
        "longest_continuous_run_s": max([s.duration_s for s in run_segments], default=0.0),
        "run_pace_only_s_per_km": mean(run_paces) if run_paces else None,
        "walk_pace_only_s_per_km": mean(walk_paces) if walk_paces else None,
        "hr_recovery_during_walks": _hr_recovery(segments),
        "pace_decay_across_run_segments": pace_decay,
        "cadence_decay_across_run_segments": cadence_decay,
        "forced_walk_score": forced_walk_score,
        "planned_run_walk_score": planned_score,
    }


def _decay(values: list[float], higher_is_worse: bool = True) -> float:
    if len(values) < 2:
        return 0.0
    first = mean(values[: max(1, len(values) // 3)])
    last = mean(values[-max(1, len(values) // 3) :])
    return max(0.0, last - first if higher_is_worse else first - last)


def _duration_decay(segments: list[Segment]) -> float:
    durations = [s.duration_s for s in segments]
    if len(durations) < 3:
        return 0.0
    return 0.3 if durations[-1] < durations[0] * 0.75 else 0.0


def _regularity_score(segments: list[Segment]) -> float:
    durations = [s.duration_s for s in segments if s.duration_s > 0]
    if len(durations) < 2:
        return 0.0
    avg = mean(durations)
    if avg == 0:
        return 0.0
    deviation = mean([abs(d - avg) for d in durations]) / avg
    return max(0.0, min(1.0, 1.0 - deviation))


def _hr_recovery(segments: list[Segment]) -> float | None:
    drops = []
    for previous, current in zip(segments, segments[1:]):
        if previous.state in RUN_STATES and current.state == "walking" and previous.max_hr and current.avg_hr:
            drops.append(previous.max_hr - current.avg_hr)
    return mean(drops) if drops else None
