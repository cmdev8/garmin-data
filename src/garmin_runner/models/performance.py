from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from garmin_runner.config import AthleteConfig
from garmin_runner.models.zones import estimate_zones


DISTANCES = [1.0, 5.0, 10.0, 21.1, 42.2]
HALF_MAX_SCARCITY_PENALTY = 0.12
MARATHON_MAX_SCARCITY_PENALTY = 0.25
HALF_ENOUGH_EVIDENCE_COUNT = 3
MARATHON_ENOUGH_PRIMARY_COUNT = 3


@dataclass(frozen=True)
class EvidenceProfile:
    evidence_count: int
    comparable_distance_count: int
    longest_run_km: float
    evidence_factor: float
    scarcity_multiplier: float
    long_distance_evidence_pace_s_per_km: float | None
    has_enough_direct_evidence: bool


def build_predictions(config: AthleteConfig, activity_features: list[dict], fitness_state: list[dict]) -> dict:
    zones = estimate_zones(config, activity_features)
    evidence_pace = _evidence_pace(activity_features)
    performance_index = fitness_state[-1]["predicted_performance_index"] if fitness_state else 1.0
    long_run = max([r.get("distance_km") or 0.0 for r in activity_features], default=0.0)
    predictions = {}
    for distance in DISTANCES:
        profile = _evidence_profile(distance, activity_features, long_run)
        pace = _riegel(evidence_pace, 5.0, distance)
        if distance >= 21.1:
            durability = max(1.0, 1.12 - min(0.12, long_run / (distance * 10.0)))
            pace *= durability
            if profile.long_distance_evidence_pace_s_per_km is not None and not profile.has_enough_direct_evidence:
                pace = max(pace, profile.long_distance_evidence_pace_s_per_km)
            pace *= profile.scarcity_multiplier
        confidence = _confidence(distance, activity_features, profile)
        predictions[f"{distance:g}km"] = {
            "distance_km": distance,
            "pace_s_per_km": pace,
            "estimated_time_s": pace * distance,
            "confidence": confidence,
            "explanation": _explanation(distance, confidence, performance_index, profile),
            "evidence_count": profile.evidence_count,
            "comparable_distance_count": profile.comparable_distance_count,
            "longest_run_km": round(profile.longest_run_km, 2),
            "evidence_factor": round(profile.evidence_factor, 3),
            "scarcity_multiplier": round(profile.scarcity_multiplier, 3),
            "long_distance_evidence_pace_s_per_km": profile.long_distance_evidence_pace_s_per_km,
        }
    return {
        "performance_index": performance_index,
        "race_predictions": predictions,
        **zones,
    }


def _evidence_pace(activity_features: list[dict]) -> float:
    raw_candidates = []
    for row in activity_features:
        raw_candidates.extend(
            [
                row.get("run_pace_only_s_per_km"),
                row.get("best_rolling_5km_pace"),
                row.get("avg_pace_s_per_km"),
            ]
        )
    candidates: list[float] = [float(c) for c in raw_candidates if c and c > 0]
    return mean(sorted(candidates)[: max(1, min(5, len(candidates)))]) if candidates else 360.0


def _evidence_profile(distance: float, rows: list[dict], longest_run_km: float) -> EvidenceProfile:
    if distance < 21.1:
        count = _short_distance_evidence_count(distance, rows)
        return EvidenceProfile(
            evidence_count=count,
            comparable_distance_count=count,
            longest_run_km=longest_run_km,
            evidence_factor=1.0,
            scarcity_multiplier=1.0,
            long_distance_evidence_pace_s_per_km=None,
            has_enough_direct_evidence=count > 0,
        )

    if distance < 42.2:
        comparable = [_activity_avg_pace(row) for row in rows if 18.0 <= _distance(row) <= 25.0]
        comparable_paces = [pace for pace in comparable if pace is not None]
        frequency_factor = min(1.0, len(comparable_paces) / HALF_ENOUGH_EVIDENCE_COUNT)
        distance_factor = min(1.0, longest_run_km / distance) if distance > 0 else 0.0
        evidence_factor = min(frequency_factor, distance_factor)
        return EvidenceProfile(
            evidence_count=len(comparable_paces),
            comparable_distance_count=len(comparable_paces),
            longest_run_km=longest_run_km,
            evidence_factor=evidence_factor,
            scarcity_multiplier=1.0 + HALF_MAX_SCARCITY_PENALTY * (1.0 - evidence_factor),
            long_distance_evidence_pace_s_per_km=min(comparable_paces) if comparable_paces else None,
            has_enough_direct_evidence=len(comparable_paces) >= HALF_ENOUGH_EVIDENCE_COUNT and distance_factor >= 0.95,
        )

    primary = [_activity_avg_pace(row) for row in rows if _distance(row) >= 30.0]
    support = [_activity_avg_pace(row) for row in rows if 18.0 <= _distance(row) < 30.0]
    primary_paces = [pace for pace in primary if pace is not None]
    support_paces = [pace for pace in support if pace is not None]
    comparable_paces = primary_paces + support_paces
    frequency_factor = min(1.0, len(primary_paces) / MARATHON_ENOUGH_PRIMARY_COUNT)
    distance_factor = min(1.0, longest_run_km / distance) if distance > 0 else 0.0
    evidence_factor = min(frequency_factor, distance_factor)
    return EvidenceProfile(
        evidence_count=len(primary_paces),
        comparable_distance_count=len(comparable_paces),
        longest_run_km=longest_run_km,
        evidence_factor=evidence_factor,
        scarcity_multiplier=1.0 + MARATHON_MAX_SCARCITY_PENALTY * (1.0 - evidence_factor),
        long_distance_evidence_pace_s_per_km=min(comparable_paces) if comparable_paces else None,
        has_enough_direct_evidence=len(primary_paces) >= MARATHON_ENOUGH_PRIMARY_COUNT and distance_factor >= 0.85,
    )


def _short_distance_evidence_count(distance: float, rows: list[dict]) -> int:
    field = {
        1.0: "best_rolling_1km_pace",
        5.0: "best_rolling_5km_pace",
        10.0: "best_rolling_10km_pace",
    }.get(distance)
    if field is None:
        return 0
    return sum(1 for row in rows if _float_or_none(row.get(field)) is not None)


def _riegel(pace: float, evidence_distance_km: float, target_distance_km: float) -> float:
    total = pace * evidence_distance_km
    predicted_total = total * (target_distance_km / evidence_distance_km) ** 1.06
    return predicted_total / target_distance_km


def _confidence(distance: float, rows: list[dict], profile: EvidenceProfile) -> float:
    if not rows:
        return 0.1
    confidence = min(0.85, 0.25 + len(rows) * 0.02)
    if distance >= 21.1:
        confidence *= max(0.15, profile.evidence_factor)
    return round(confidence, 3)


def _explanation(distance: float, confidence: float, performance_index: float, profile: EvidenceProfile) -> str:
    text = f"Deterministic estimate from recent pace evidence and performance index {performance_index:.1f}."
    if distance >= 21.1:
        if profile.long_distance_evidence_pace_s_per_km is not None and not profile.has_enough_direct_evidence:
            text += " Pace is not allowed to be faster than sparse comparable long-run evidence."
        if profile.scarcity_multiplier > 1.001:
            text += (
                " Long-distance estimate is discounted because historical frequency near this distance is sparse "
                f"(evidence factor {profile.evidence_factor:.2f}, longest run {profile.longest_run_km:.1f} km)."
            )
    if confidence < 0.4:
        text += " Treat this as a low-confidence baseline."
    return text


def _distance(row: dict[str, Any]) -> float:
    return _float_or_none(row.get("distance_km")) or 0.0


def _activity_avg_pace(row: dict[str, Any]) -> float | None:
    pace = _float_or_none(row.get("run_pace_only_s_per_km")) or _float_or_none(row.get("avg_pace_s_per_km"))
    return pace if pace is not None and pace > 0 else None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value != value:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
