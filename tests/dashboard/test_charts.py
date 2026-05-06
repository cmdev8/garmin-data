from __future__ import annotations

import pandas as pd

from garmin_runner.dashboard.charts import prepare_activity_table, prepare_weekly_chart_data


def test_prepare_weekly_chart_data_tolerates_empty():
    data = prepare_weekly_chart_data(pd.DataFrame())

    assert data.empty


def test_prepare_activity_table_tolerates_missing_columns():
    data = prepare_activity_table(pd.DataFrame([{"activity_id": "a1"}]))

    assert data.loc[0, "activity_id"] == "a1"
    assert data.loc[0, "Átlagtempó"] == "n/a"
