from __future__ import annotations

import inspect
from pathlib import Path

from garmin_runner.dashboard.app import PAGES, PRE_ANALYSIS_PAGES
from garmin_runner.dashboard.views import activities, algorithms, blocks, diagnostics, diminishing_returns, performance_change, predictions
from garmin_runner.dashboard.upload import render_upload_panel, validate_uploaded_files


def test_algorithms_page_registered():
    assert "Dokumentáció" in PAGES
    assert PAGES["Dokumentáció"] is algorithms.render


def test_run_walk_page_removed_from_navigation():
    assert "Futás-séta" not in PAGES


def test_prediction_and_optimizer_pages_are_combined():
    assert "Prediction és edzésterv" in PAGES
    assert PAGES["Prediction és edzésterv"] is predictions.render
    assert "Előrejelzések" not in PAGES
    assert "Tervoptimalizáló" not in PAGES


def test_performance_change_page_registered():
    assert "Teljesítménytrend" in PAGES
    assert PAGES["Teljesítménytrend"] is performance_change.render


def test_diminishing_returns_page_registered():
    assert "Diminishing returns" in PAGES
    assert PAGES["Diminishing returns"] is diminishing_returns.render


def test_diagnostics_page_registered():
    assert "Diagnosztika" in PAGES
    assert PAGES["Diagnosztika"] is diagnostics.render


def test_algorithms_available_before_analysis():
    assert "Elemzés indítása" in PRE_ANALYSIS_PAGES
    assert "Dokumentáció" in PRE_ANALYSIS_PAGES
    assert PRE_ANALYSIS_PAGES["Dokumentáció"] == "algorithms"


def test_algorithms_page_contains_core_equations():
    equations = "\n".join(algorithms.ALGORITHM_EQUATIONS)

    assert "fitness_t = 0.965 * fitness_{t-1} + 0.035 * stimulus_t" in equations
    assert "T2 = T1 * (D2 / D1)^1.06" in equations
    assert "score = 2.0 * expected_adaptation" in equations
    assert "confidence = clamp(0.25 + 0.45 * min(0.9, n/45)" in equations
    assert "TimeSeriesSplit: train_index < validation_index" in equations
    assert "response_i = max(0, performance_index_{i+7} - performance_index_i)" in equations
    assert "diminishing_returns_factor = clamp" in equations
    assert "run_walk_fatigue_multiplier = clamp" in equations
    assert "fatigue_adjusted_load = load * run_walk_fatigue_multiplier" in equations


def test_algorithms_render_accepts_no_data():
    algorithms.render(None)


def test_algorithms_uses_mathjax_equations():
    source = inspect.getsource(algorithms)

    assert "LATEX_EQUATIONS" in source
    assert "st.latex" in source
    assert "diminishing\\_returns\\_factor" in source


def test_hist_gradient_boosting_documented():
    source = inspect.getsource(algorithms)

    assert "HistGradientBoostingRegressor" in source
    assert "histogram binning" in source
    assert "gradient boosting" in source
    assert "max_iter=80" in source
    assert "75/25 validation split" not in source
    assert "kronologikus holdout" in source
    assert "MAE" in source
    assert "permutation importance" in source
    assert "DummyRegressor" in source
    assert "RidgeCV" in source
    assert "TimeSeriesSplit" in source
    assert "koefficiens" in source
    assert "diminishing returns" in source
    assert "diminishing returns factor" in source
    assert "Futás-séta fáradtsági korrekció" in source
    assert "Kényszerű sétaszüneteknél nincs" in source


def test_algorithms_documents_grounded_prediction_and_ml_constants():
    source = inspect.getsource(algorithms)

    assert "18–25 km" in source
    assert ">=30 km" in source
    assert "18–30 km" in source
    assert "12%" in source
    assert "25%" in source
    assert "scarcity_multiplier" in source
    assert "TimeSeriesSplit" in source
    assert "RidgeCV" in source
    assert "DummyRegressor" in source
    assert "random split" in source
    assert "nem a fő temporal út" in source


def test_no_raw_json_in_dashboard_views():
    view_dir = Path("src/garmin_runner/dashboard/views")
    offenders = [path for path in view_dir.glob("*.py") if "st.json" in path.read_text(encoding="utf-8")]

    assert offenders == []


def test_activities_page_source_uses_full_selected_activity_visuals():
    source = inspect.getsource(activities)

    assert "Tempó és pulzus" in source
    assert "Lépésütem, power és futódinamika" in source
    assert "Szint és környezet" in source
    assert "Futás-séta" in source
    assert "Minőség és FIT mezők" in source
    assert "activity_record_chart_table" in source
    assert "activity_advanced_fit_table" in source
    assert "st.json" not in source


def test_training_blocks_page_source_uses_graphical_helpers():
    source = inspect.getsource(blocks)

    assert "Edzésblokkok" in source
    assert "Idővonal" in source
    assert "Training load" in source
    assert "Improvement" in source
    assert "Edzésblokkok részletei" in source
    assert "training_block_timeline_table" in source
    assert "st.bar_chart" in source
    assert "st.json" not in source
    assert "st.dataframe(data.training_blocks" not in source


def test_performance_change_page_source_is_hungarian_and_curated():
    source = inspect.getsource(performance_change)

    assert "Teljesítménytrend" in source
    assert "Táv kiválasztása" in source
    assert "perc/km" in source
    assert "szintemelkedés proxy" in source
    assert "st.json" not in source


def test_diagnostics_page_source_is_hungarian_and_curated():
    source = inspect.getsource(diagnostics)

    assert "Diagnosztika" in source
    assert "Miért változik a teljesítmény?" in source
    assert "Mit érdemes tenni?" in source
    assert "Bizonyíték és korlátok" in source
    assert "st.json" not in source


def test_diminishing_returns_page_source_is_hungarian_and_curated():
    source = inspect.getsource(diminishing_returns)

    assert "Diminishing returns" in source
    assert "diminishing returns factor" in source or "diminishing_returns_values" in source
    assert "Havi diminishing returns factor" in source
    assert "konzervatívan korlátozott" in source
    assert "nem jobb teljesítményt jelent" in source
    assert "kevesebb korrekció kell" in source
    assert "st.json" not in source


def test_prediction_page_has_plan_selector():
    source = inspect.getsource(predictions)

    assert "Terv kiválasztása" in source
    assert "st.selectbox" in source
    assert "st.json" not in source


def test_combined_prediction_plan_page_has_no_schedule_duplication():
    source = inspect.getsource(predictions)

    assert "Jelölt tervek részletei" in source
    assert "plan_schedule_table" not in source
    assert "hr_zone_table" in source
    assert "pace_zone_table" in source
    assert "plan_adjusted_prediction_table" in source
    assert "scoring_component_values" in source


def test_upload_ui_primary_labels_are_hungarian():
    source = inspect.getsource(render_upload_panel)

    assert "Garmin FIT fájlok feltöltése" in source
    assert "Célverseny távja" in source
    assert "Feltöltött fájlok elemzése" in source
    assert "Részletes diagram mód" not in source
    assert "Csak összefoglaló" not in source
    assert "Upload Garmin FIT files" not in source
    assert "Analyze uploaded files" not in source


def test_upload_validation_errors_are_hungarian():
    errors = validate_uploaded_files([])

    assert errors == ["Tölts fel legalább egy .fit fájlt."]


def test_readme_describes_pdf_and_diminishing_returns():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "training_plan_report.pdf" in readme
    assert "diminishing returns factor" in readme


def test_readme_describes_current_grounded_behavior():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "TemporalRidgeCV" in readme
    assert "TimeSeriesSplit" in readme
    assert "HistGradientBoostingRegressor" in readme
    assert "DummyRegressor" in readme
    assert "scarcity discount" in readme
    assert "18–25 km" in readme
    assert ">=30 km" in readme
    assert "raw FIT files are not stored" in readme
    assert "trained model artifacts are not written" in readme
    assert "Run-walk prescriptions are generated only when" in readme
    assert "--export-dir" in readme
