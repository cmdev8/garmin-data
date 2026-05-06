from __future__ import annotations

from statistics import median

from garmin_runner.config import AthleteConfig
from garmin_runner.fit.schema import Activity, ActivityConfidence, Record


RUNNING_SPORTS = {"running"}
REJECT_SPORTS = {
    "cycling",
    "swimming",
    "strength_training",
    "cardio_training",
    "rowing",
    "elliptical",
    "skiing",
    "golf",
    "hiking",
    "multisport",
}


def classify_activity(activity: Activity, records: list[Record], config: AthleteConfig) -> Activity:
    sport = (activity.sport or "").lower()
    sub_sport = (activity.sub_sport or "").lower()

    if sport in RUNNING_SPORTS or sub_sport in RUNNING_SPORTS:
        activity.keep = True
        activity.keep_reason = "session sport indicates running"
        activity.activity_type_confidence = "high"
        activity.activity_type_reason = activity.keep_reason
        return activity

    if sport == "walking":
        if config.run_walk.allow_walk_run_candidates and _movement_looks_run_walk(records):
            activity.keep = True
            activity.keep_reason = "walking metadata with running-like movement"
            activity.activity_type_confidence = "low"
        else:
            activity.keep = False
            activity.keep_reason = "walking-only activity rejected by default"
            activity.activity_type_confidence = "rejected"
        activity.activity_type_reason = activity.keep_reason
        return activity

    if sport in REJECT_SPORTS:
        activity.keep = False
        activity.keep_reason = f"non-running sport rejected: {sport}"
        activity.activity_type_confidence = "rejected"
        activity.activity_type_reason = activity.keep_reason
        return activity

    if not sport or sport in {"unknown", "generic"}:
        keep, reason, confidence = _fallback(records)
        activity.keep = keep
        activity.keep_reason = reason
        activity.activity_type_confidence = confidence
        activity.activity_type_reason = reason
        return activity

    activity.keep = False
    activity.keep_reason = f"unsupported sport rejected: {sport}"
    activity.activity_type_confidence = "rejected"
    activity.activity_type_reason = activity.keep_reason
    return activity


def _movement_looks_run_walk(records: list[Record]) -> bool:
    speeds = [r.speed_mps for r in records if r.speed_mps is not None and r.speed_mps > 0]
    cadences = [r.cadence_spm for r in records if r.cadence_spm is not None and r.cadence_spm > 0]
    if not speeds:
        return False
    return max(speeds) >= 2.2 or bool(cadences and max(cadences) >= 140)


def _fallback(records: list[Record]) -> tuple[bool, str, ActivityConfidence]:
    if not records:
        return False, "no meaningful movement records", "rejected"

    distances = [r.distance_m for r in records if r.distance_m is not None]
    total_distance = max(distances) - min(distances) if len(distances) >= 2 else 0.0
    elapsed = max((r.elapsed_s for r in records), default=0.0)
    speeds = [r.speed_mps for r in records if r.speed_mps is not None and r.speed_mps > 0]
    cadence = [r.cadence_spm for r in records if r.cadence_spm is not None and r.cadence_spm > 0]
    has_gps = any(r.position_lat is not None and r.position_long is not None for r in records)

    if total_distance < 500 or elapsed < 300:
        return False, "fallback rejected: too short", "rejected"
    if not speeds:
        return False, "fallback rejected: missing speed evidence", "rejected"

    median_speed = median(speeds)
    median_cadence = median(cadence) if cadence else None
    if median_speed > 8.0 and not (median_cadence and median_cadence >= 130):
        return False, "fallback rejected: speed pattern looks non-running", "rejected"
    if not has_gps and not cadence:
        return False, "fallback rejected: no GPS or cadence evidence", "rejected"
    if 1.0 <= median_speed <= 5.5:
        return True, "fallback kept: plausible walk/run movement", "medium"
    return False, "fallback rejected: implausible walk/run movement", "rejected"
