from __future__ import annotations


def build_report(
    *,
    activities: list,
    warnings: list[str],
    fitness_state: list[dict],
    training_blocks: list[dict],
    predictions: dict,
    plan: dict,
    optimizer: str = "v1",
) -> str:
    kept = [a for a in activities if a.keep]
    rejected = [a for a in activities if not a.keep]
    latest = fitness_state[-1] if fitness_state else {}
    best_block = max(training_blocks, key=lambda b: b.get("improvement_score") or 0.0, default={})
    plan_title = "Next Week Plan" if plan.get("horizon_days") == 7 else "Training Plan"
    plan_days = plan.get("days") or plan.get("week") or []
    lines = [
        "# Garmin Running Analysis",
        "",
        "## Input Summary",
        f"- files parsed: {len(activities)}",
        f"- running activities kept: {len(kept)}",
        f"- non-running activities rejected: {len(rejected)}",
        f"- warnings: {len(warnings)}",
        "",
        "## Current State",
        f"- fitness: {latest.get('fitness', 0):.1f}",
        f"- fatigue: {latest.get('fatigue', 0):.1f}",
        f"- form: {latest.get('form', 0):.1f}",
        f"- overload risk: {latest.get('overload_risk', 0):.2f}",
        "",
        "## Run-Walk-Run Analysis",
        "- Run-walk sessions use running-only pace for performance signals when available.",
        "- Average pace is retained only as a logistics statistic for run-walk sessions.",
        "",
        "## Best Improvement Period",
        f"- dates: {best_block.get('start_date')} to {best_block.get('end_date')}",
        f"- block type: {best_block.get('block_type', 'unknown')}",
        f"- likely driver: {best_block.get('likely_improvement_driver', 'unknown')}",
        f"- confidence: {best_block.get('confidence', 0):.2f}",
        "",
        "## Race Predictions",
    ]
    for key, value in predictions.get("race_predictions", {}).items():
        lines.append(f"- {key}: {value['pace_s_per_km']:.0f} s/km, confidence {value['confidence']:.2f}")
    lines.extend(["", "## HR Zones"])
    for key, value in predictions.get("hr_zones", {}).items():
        lines.append(f"- {key}: {value['min_bpm']} to {value['max_bpm']} bpm")
    lines.extend(["", "## Pace Zones"])
    for key, value in predictions.get("pace_zones", {}).items():
        lines.append(f"- {key}: {value['min_s_per_km']:.0f} to {value['max_s_per_km']:.0f} s/km")
    lines.extend(["", f"## {plan_title}"])
    if optimizer == "v2" or plan.get("selected_plan_id"):
        lines.extend(
            [
                f"- optimizer: {optimizer}",
                f"- selected family: {plan.get('family', 'unknown')}",
                f"- score: {plan.get('score', 0):.2f}",
            ]
        )
        for reason in plan.get("why_selected", []):
            lines.append(f"- rationale: {reason}")
        for warning in plan.get("warnings", []):
            lines.append(f"- warning: {warning}")
    for day in plan_days:
        lines.append(f"- {day['date']} {day['day']}: {day['description']} ({day['duration_min']} min)")
    return "\n".join(lines) + "\n"
