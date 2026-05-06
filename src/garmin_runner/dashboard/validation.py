from __future__ import annotations


def has_running_activities(data) -> bool:
    return not data.activities.empty and "keep" in data.activities and bool(data.activities["keep"].sum())
