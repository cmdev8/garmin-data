from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from garmin_runner.config import AthleteConfig, load_athlete_config
from garmin_runner.features.activity_features import build_activity_features
from garmin_runner.features.daily_features import build_daily_features, build_weekly_features
from garmin_runner.fit.parser import parse_fit_directory, parse_fit_payloads
from garmin_runner.fit.schema import Activity, Record, Segment, row
from garmin_runner.models.fitness_fatigue import build_fitness_state
from garmin_runner.models.ml import MLResult, train_per_run_model
from garmin_runner.models.performance import build_predictions
from garmin_runner.models.performance_change import build_performance_change
from garmin_runner.models.training_blocks import classify_training_blocks
from garmin_runner.planning.optimizer import run_v2_optimizer
from garmin_runner.reporting.markdown_report import build_report
from garmin_runner.run_walk.metrics import run_walk_metrics
from garmin_runner.run_walk.segmenter import segment_records
from garmin_runner.utils.io import write_json, write_markdown, write_table


@dataclass
class AnalysisConfig:
    athlete_config: AthleteConfig
    fit_dir: Path | None = None
    uploaded_fit_files: list[tuple[str, bytes]] = field(default_factory=list)
    export_dir: Path | None = None
    include_records: bool = True


@dataclass
class AnalysisResult:
    activities: list[Activity]
    records: list[Record]
    segments: list[Segment]
    activity_features: list[dict]
    daily_features: list[dict]
    weekly_features: list[dict]
    fitness_state: list[dict]
    training_blocks: list[dict]
    performance_change: dict
    predictions: dict
    ml_result: MLResult
    optimizer_result: dict
    next_week_plan: dict
    candidate_plans: list[dict]
    plan_optimizer_debug: dict
    report_markdown: str
    warnings: list[str]

    def summary(self) -> dict:
        kept = [activity for activity in self.activities if activity.keep]
        return {
            "activities": len(self.activities),
            "kept_running_activities": len(kept),
            "records": len(self.records),
            "segments": len(self.segments),
            "ml_model_type": self.ml_result.model_type,
            "ml_training_run_id": self.ml_result.training_run_id,
            "selected_plan": self.next_week_plan.get("selected_plan_id"),
        }


def run_analysis(config: AnalysisConfig | None = None, **legacy_kwargs) -> AnalysisResult:
    """Run a full in-memory analysis.

    The returned object owns all processed outputs. Nothing is written unless
    ``export_dir`` is explicitly provided for developer/debug compatibility.
    """
    if config is None:
        config = _config_from_legacy_kwargs(**legacy_kwargs)
    activities, records, warnings = _parse_inputs(config)
    kept_activity_ids = {activity.activity_id for activity in activities if activity.keep}
    kept_records = [record for record in records if record.activity_id in kept_activity_ids]
    segments = segment_records(kept_records, config.athlete_config)
    metrics = run_walk_metrics(segments)
    activity_features = build_activity_features(activities, kept_records, metrics)
    daily_features = build_daily_features(activity_features)
    weekly_features = build_weekly_features(daily_features)
    fitness_state = build_fitness_state(daily_features)
    predictions = build_predictions(config.athlete_config, activity_features, fitness_state)
    training_blocks = classify_training_blocks(weekly_features, fitness_state)
    performance_change = build_performance_change(activity_features, weekly_features, fitness_state)
    ml_result = train_per_run_model(daily_features, weekly_features, fitness_state)
    optimizer_payload = run_v2_optimizer(
        config=config.athlete_config,
        activity_features=activity_features,
        weekly_features=weekly_features,
        fitness_state=fitness_state,
        ml_result=ml_result,
        performance_change=performance_change,
        out_dir=None,
    )
    report = build_report(
        activities=activities,
        warnings=warnings + ml_result.warnings,
        fitness_state=fitness_state,
        training_blocks=training_blocks,
        predictions=predictions,
        plan=optimizer_payload["plan"],
        optimizer="default",
    )
    result = AnalysisResult(
        activities=activities,
        records=kept_records if config.include_records else [],
        segments=segments if config.include_records else [],
        activity_features=activity_features,
        daily_features=daily_features,
        weekly_features=weekly_features,
        fitness_state=fitness_state,
        training_blocks=training_blocks,
        performance_change=performance_change,
        predictions=predictions,
        ml_result=ml_result,
        optimizer_result=optimizer_payload,
        next_week_plan=optimizer_payload["plan"],
        candidate_plans=optimizer_payload["candidate_rows"],
        plan_optimizer_debug=optimizer_payload["debug"],
        report_markdown=report,
        warnings=warnings + ml_result.warnings,
    )
    if config.export_dir is not None:
        export_analysis_result(result, config.export_dir)
    return result


def export_analysis_result(result: AnalysisResult, out_dir: str | Path) -> None:
    """Explicit developer export for CLI/debug use. Dashboard does not call this."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_table(out / "activities.parquet", [row(a) for a in result.activities])
    write_table(out / "records.parquet", [row(r) for r in result.records])
    write_table(out / "segments.parquet", [row(s) for s in result.segments])
    write_table(out / "activity_features.parquet", result.activity_features)
    write_table(out / "daily_features.parquet", result.daily_features)
    write_table(out / "weekly_features.parquet", result.weekly_features)
    write_table(out / "fitness_state.parquet", result.fitness_state)
    write_table(out / "training_blocks.parquet", result.training_blocks)
    write_table(out / "candidate_plans.parquet", result.candidate_plans)
    write_json(out / "performance_change.json", result.performance_change)
    write_json(out / "predictions.json", result.predictions)
    write_json(out / "next_week_plan.json", result.next_week_plan)
    write_json(out / "plan_optimizer_debug.json", result.plan_optimizer_debug)
    write_json(out / "ml_result.json", result.ml_result.to_dict())
    write_markdown(out / "report.md", result.report_markdown)


def _parse_inputs(config: AnalysisConfig) -> tuple[list[Activity], list[Record], list[str]]:
    if config.uploaded_fit_files:
        return parse_fit_payloads(config.uploaded_fit_files, config.athlete_config)
    if config.fit_dir is not None:
        return parse_fit_directory(config.fit_dir, config.athlete_config)
    return [], [], ["No FIT inputs provided."]


def _config_from_legacy_kwargs(**kwargs) -> AnalysisConfig:
    athlete_config = load_athlete_config(
        kwargs.get("athlete_config"),
        planning_horizon_days=kwargs.get("planning_horizon_days"),
        include_walk_run_candidates=kwargs.get("include_walk_run_candidates", False),
    )
    return AnalysisConfig(
        athlete_config=athlete_config,
        fit_dir=Path(kwargs["fit_dir"]) if kwargs.get("fit_dir") else None,
        export_dir=Path(kwargs["out_dir"]) if kwargs.get("out_dir") else None,
    )
