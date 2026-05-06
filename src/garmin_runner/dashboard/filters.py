from __future__ import annotations


def filter_by_activity_id(df, activity_id):
    if df.empty or "activity_id" not in df:
        return df
    return df[df["activity_id"] == activity_id]
