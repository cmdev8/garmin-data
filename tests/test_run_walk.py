from datetime import datetime, timedelta

from garmin_runner.config import load_athlete_config
from garmin_runner.fit.schema import Record
from garmin_runner.run_walk.classifier import classify_run_walk
from garmin_runner.run_walk.fatigue import run_walk_fatigue_multiplier
from garmin_runner.run_walk.metrics import run_walk_metrics
from garmin_runner.run_walk.segmenter import segment_records


def _planned_4_1_records():
    start = datetime(2026, 5, 1, 12, 0)
    records = []
    elapsed = 0
    distance = 0.0
    for _ in range(3):
        for _ in range(0, 240, 10):
            distance += 28
            records.append(Record("a1", start + timedelta(seconds=elapsed), elapsed, distance, 2.8, cadence_spm=165))
            elapsed += 10
        for _ in range(0, 60, 10):
            distance += 9
            records.append(Record("a1", start + timedelta(seconds=elapsed), elapsed, distance, 0.9, cadence_spm=110))
            elapsed += 10
    return records


def test_planned_run_walk_segments_and_classifies():
    segments = segment_records(_planned_4_1_records(), load_athlete_config())

    metrics = run_walk_metrics(segments)["a1"]
    workout_type, confidence, evidence = classify_run_walk(metrics)

    assert metrics["number_of_walk_breaks"] >= 2
    assert workout_type in {"planned", "mixed"}
    assert confidence > 0
    assert evidence


def test_planned_run_walk_gets_fatigue_discount():
    multiplier = run_walk_fatigue_multiplier("planned", walk_fraction=0.2, planned_run_walk_score=0.8)

    assert multiplier < 1.0
    assert multiplier >= 0.75


def test_forced_run_walk_gets_no_fatigue_discount():
    multiplier = run_walk_fatigue_multiplier("forced", walk_fraction=0.2, planned_run_walk_score=0.8)

    assert multiplier == 1.0
