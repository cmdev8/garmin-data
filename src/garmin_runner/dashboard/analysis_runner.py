from __future__ import annotations

from garmin_runner.config import load_athlete_config_from_mapping
from garmin_runner.dashboard.data import DashboardData, dashboard_data_from_analysis
from garmin_runner.dashboard.upload import UploadRequest
from garmin_runner.pipeline import AnalysisConfig, run_analysis


def run_analysis_for_uploads(upload_request: UploadRequest, keep_temp: bool = False) -> DashboardData:
    # keep_temp is accepted for CLI/spec compatibility. The upload pipeline is
    # byte/in-memory first and does not create a workspace to preserve.
    del keep_temp
    config = _config_from_upload(upload_request)
    result = run_analysis(
        AnalysisConfig(
            athlete_config=config,
            uploaded_fit_files=[(file.safe_name, file.content) for file in upload_request.fit_files],
            include_records=True,
        )
    )
    return dashboard_data_from_analysis(
        result,
        uploaded_file_count=len(upload_request.fit_files),
        detailed_charts=True,
    )


def _config_from_upload(upload_request: UploadRequest):
    config = load_athlete_config_from_dict(upload_request.athlete_settings)
    config.run_walk.enabled = bool(upload_request.run_walk_enabled)
    if not config.run_walk.enabled:
        config.run_walk.allow_walk_run_candidates = False
    return config


def load_athlete_config_from_dict(data: dict):
    return load_athlete_config_from_mapping(data)
