from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from garmin_runner.fit.schema import row
from garmin_runner.pipeline import AnalysisResult


@dataclass
class DashboardData:
    source: str
    activities: pd.DataFrame
    records: pd.DataFrame
    segments: pd.DataFrame
    activity_features: pd.DataFrame
    daily_features: pd.DataFrame
    weekly_features: pd.DataFrame
    fitness_state: pd.DataFrame
    training_blocks: pd.DataFrame
    performance_change: dict
    predictions: dict
    next_week_plan: dict
    candidate_plans: pd.DataFrame
    plan_optimizer_debug: dict
    ml_result: dict
    report_markdown: str | None
    warnings: list[str]
    uploaded_file_count: int
    analyzed_at: str
    detailed_charts: bool
    temp_files_deleted: bool = True


def dashboard_data_from_analysis(result: AnalysisResult, *, uploaded_file_count: int, detailed_charts: bool) -> DashboardData:
    return DashboardData(
        source="uploaded_files",
        activities=pd.DataFrame([row(activity) for activity in result.activities]),
        records=pd.DataFrame([row(record) for record in result.records]) if detailed_charts else pd.DataFrame(),
        segments=pd.DataFrame([row(segment) for segment in result.segments]) if detailed_charts else pd.DataFrame(),
        activity_features=pd.DataFrame(result.activity_features),
        daily_features=pd.DataFrame(result.daily_features),
        weekly_features=pd.DataFrame(result.weekly_features),
        fitness_state=pd.DataFrame(result.fitness_state),
        training_blocks=pd.DataFrame(result.training_blocks),
        performance_change=result.performance_change,
        predictions=result.predictions,
        next_week_plan=result.next_week_plan,
        candidate_plans=pd.DataFrame(result.candidate_plans),
        plan_optimizer_debug=result.plan_optimizer_debug,
        ml_result=result.ml_result.to_dict(),
        report_markdown=result.report_markdown,
        warnings=result.warnings,
        uploaded_file_count=uploaded_file_count,
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        detailed_charts=detailed_charts,
        temp_files_deleted=True,
    )
