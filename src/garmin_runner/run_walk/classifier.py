from __future__ import annotations


def classify_run_walk(metrics: dict) -> tuple[str, float, list[str]]:
    breaks = metrics.get("number_of_walk_breaks", 0) or 0
    if breaks == 0:
        return "not_run_walk", 1.0, ["no walk breaks detected"]

    planned = metrics.get("planned_run_walk_score") or 0.0
    forced = metrics.get("forced_walk_score") or 0.0
    evidence = []
    if planned >= 0.65:
        evidence.append("regular run/walk timing")
    if forced >= 0.45:
        evidence.append("late-session pace, cadence, or duration decay")

    if planned >= 0.65 and forced < 0.45:
        return "planned", planned, evidence
    if forced >= 0.45 and planned < 0.55:
        return "forced", forced, evidence
    return "mixed", max(planned, forced, 0.4), evidence or ["mixed run/walk evidence"]
