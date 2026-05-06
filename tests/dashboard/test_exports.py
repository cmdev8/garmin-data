from __future__ import annotations

import inspect
import io
import zipfile

import pandas as pd

from garmin_runner.dashboard.data import DashboardData
from garmin_runner.dashboard import exports
from garmin_runner.dashboard.exports import build_processed_results_zip, build_training_plan_pdf
from garmin_runner.dashboard.exports_ui import render_downloads


def _dashboard_data(
    *,
    next_week_plan: dict | None = None,
    candidate_plans: pd.DataFrame | None = None,
    predictions: dict | None = None,
    plan_optimizer_debug: dict | None = None,
    ml_result: dict | None = None,
) -> DashboardData:
    return DashboardData(
        source="uploaded_files",
        activities=pd.DataFrame([{"keep": True}]),
        records=pd.DataFrame(),
        segments=pd.DataFrame(),
        activity_features=pd.DataFrame([{"distance_km": 5}]),
        daily_features=pd.DataFrame(),
        weekly_features=pd.DataFrame([{"date": "2026-05-06", "distance_7d_km": 42, "load_7d": 320}]),
        fitness_state=pd.DataFrame(),
        training_blocks=pd.DataFrame(),
        performance_change={"summary": [], "observations": [], "models": {}, "warnings": []},
        predictions=predictions
        or {
            "race_predictions": {"5km": {"pace_s_per_km": 300, "estimated_time_s": 1500, "confidence": 0.7}},
            "hr_zones": {"Z1": {"min_bpm": 60, "max_bpm": 130}},
            "pace_zones": {"easy": {"min_s_per_km": 420, "max_s_per_km": 390}},
        },
        next_week_plan=next_week_plan
        or {
            "selected_plan_id": "plan_pdf_001",
            "family": "base_build_week",
            "score": 0.9,
            "horizon_days": 7,
            "start_date": "2026-05-07",
            "end_date": "2026-05-13",
            "why_selected": ["Keeps hard days within limit"],
            "warnings": [],
            "days": [
                {
                    "date": "2026-05-07",
                    "day": "Thursday",
                    "workout_type": "easy",
                    "intensity": "easy",
                    "duration_min": 35,
                    "distance_km": 5,
                    "purpose": "Build aerobic base",
                }
            ],
            "summary": {"total_duration_min": 35, "total_distance_km": 5, "total_load": 40, "hard_days": 0, "long_run_km": 0},
        },
        candidate_plans=candidate_plans
        if candidate_plans is not None
        else pd.DataFrame(
            [
                {
                    "plan_id": "plan_pdf_001",
                    "family": "base_build_week",
                    "total_score": 0.9,
                    "expected_adaptation": 0.8,
                    "expected_fatigue": 0.2,
                    "overload_risk": 0.1,
                    "race_specificity": 0.8,
                    "missing_stimulus_score": 0.7,
                    "constraint_penalty": 0.0,
                    "is_valid": True,
                }
            ]
        ),
        plan_optimizer_debug=plan_optimizer_debug
        or {"generated_candidate_count": 10, "valid_candidate_count": 5, "rejected_candidate_count": 5, "top_5_plan_ids": ["plan_pdf_001"], "rejection_summary": {"fatigue": 2}},
        ml_result=ml_result
        or {
            "model_type": "DummyRegressor",
            "row_count": 3,
            "confidence": 0.4,
            "metrics": {
                "mae": 1.2,
                "validation_rows": 3,
                "diminishing_returns_observations": 8,
                "diminishing_returns_beta": -0.01,
                "diminishing_returns_reference_response": 0.8,
                "diminishing_returns_current_response": 0.6,
                "diminishing_returns_reason": "learned_from_history",
            },
            "predictions": {"diminishing_returns_factor": 0.75},
            "feature_importance": {"load_7d": 0.2},
        },
        report_markdown="# Report",
        warnings=[],
        uploaded_file_count=1,
        analyzed_at="2026-05-06T00:00:00+00:00",
        detailed_charts=False,
    )


def test_processed_zip_excludes_raw_fit_files():
    data = _dashboard_data(predictions={"race_predictions": {}}, next_week_plan={"selected_plan_id": "plan"}, candidate_plans=pd.DataFrame([{"plan_id": "plan"}]), plan_optimizer_debug={}, ml_result={"model_type": "DummyRegressor"})

    payload = build_processed_results_zip(data)

    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        names = archive.namelist()
    assert "activity_features.csv" in names
    assert all(not name.endswith(".fit") for name in names)
    assert all("model" not in name or name == "ml_result.json" for name in names)


def test_training_plan_pdf_contains_selected_plan_explanation_and_privacy_note():
    pdf = build_training_plan_pdf(_dashboard_data())

    assert pdf.startswith(b"%PDF")
    assert b"plan_pdf_001" in pdf
    assert b"80%" in pdf
    assert b"40%" in pdf
    source = inspect.getsource(exports.build_training_plan_pdf)
    assert "plan_family_label" in inspect.getsource(exports)
    assert "plan_reason_labels" in source
    assert "Várható előnyök" in source
    assert "diminishing returns factor" in source
    assert "ML és feature importance" in source
    assert "Személyes edzésterv" in source
    assert "Miért ezt a tervet?" in source
    assert "Adatvédelem" in source
    assert "SimpleDocTemplate" in inspect.getsource(exports)
    assert "TableStyle" in inspect.getsource(exports)
    assert " | ".join(["A", "B"]) not in inspect.getsource(exports.build_training_plan_pdf)
    assert b"FIT" in pdf


def test_training_plan_pdf_tolerates_missing_optional_data():
    data = _dashboard_data(
        next_week_plan={"selected_plan_id": "missing", "family": "unknown", "days": [], "summary": {}},
        candidate_plans=pd.DataFrame(),
        predictions={},
        plan_optimizer_debug={},
        ml_result={"metrics": {}, "feature_importance": {}},
    )

    pdf = build_training_plan_pdf(data)

    assert pdf.startswith(b"%PDF")
    assert b"missing" in pdf


def test_download_ui_has_single_pdf_button():
    source = inspect.getsource(render_downloads)

    assert source.count("st.download_button") == 1
    assert "PDF tervjelentés letöltése" in source
    assert "build_training_plan_pdf" in source
    assert "Jelentés letöltése" not in source
    assert "Előrejelzések letöltése" not in source
    assert "Edzésterv letöltése" not in source
    assert "Edzésjellemzők CSV" not in source
    assert "Heti jellemzők CSV" not in source
    assert "Feldolgozott eredmények ZIP" not in source
