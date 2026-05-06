from __future__ import annotations


def classify_training_blocks(weekly_features: list[dict], fitness_state: list[dict]) -> list[dict]:
    if not weekly_features:
        return []
    state_by_date = {row["date"]: row for row in fitness_state}
    blocks = []
    current_type = "unknown"
    current: list[dict] = []
    for row in weekly_features:
        block_type = _block_type(row)
        if current and block_type != current_type:
            blocks.append(_summarize(current_type, current, state_by_date))
            current = []
        current_type = block_type
        current.append(row)
    if current:
        blocks.append(_summarize(current_type, current, state_by_date))
    return blocks


def _block_type(row: dict) -> str:
    acwr = row.get("acute_chronic_workload_ratio") or 0.0
    hard = row.get("hard_fraction") or 0.0
    long_run = row.get("long_run_distance_km") or 0.0
    load = row.get("load_7d") or 0.0
    if load == 0:
        return "recovery"
    if acwr > 1.35:
        return "overload"
    if hard > 0.22:
        return "threshold"
    if long_run >= 12:
        return "long_run_endurance"
    if load > 0:
        return "base"
    return "unknown"


def _summarize(block_type: str, rows: list[dict], state_by_date: dict[str, dict]) -> dict:
    before = state_by_date.get(rows[0]["date"], {}).get("predicted_performance_index")
    after = state_by_date.get(rows[-1]["date"], {}).get("predicted_performance_index")
    improvement = (after - before) if before is not None and after is not None else 0.0
    return {
        "start_date": rows[0]["date"],
        "end_date": rows[-1]["date"],
        "block_type": block_type,
        "volume": sum(r.get("distance_7d_km") or 0.0 for r in rows) / max(1, len(rows)),
        "easy_fraction": sum(r.get("easy_fraction") or 0.0 for r in rows) / max(1, len(rows)),
        "moderate_fraction": sum(r.get("moderate_fraction") or 0.0 for r in rows) / max(1, len(rows)),
        "hard_fraction": sum(r.get("hard_fraction") or 0.0 for r in rows) / max(1, len(rows)),
        "performance_before": before,
        "performance_after": after,
        "improvement_score": improvement,
        "likely_improvement_driver": block_type,
        "confidence": 0.6 if len(rows) >= 7 else 0.35,
    }
