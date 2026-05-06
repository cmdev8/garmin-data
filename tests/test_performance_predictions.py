from __future__ import annotations

from garmin_runner.config import load_athlete_config
from garmin_runner.models.performance import build_predictions


def test_marathon_prediction_is_discounted_when_only_half_marathon_evidence_exists():
    activities = [
        {"distance_km": 21.1, "avg_pace_s_per_km": 347.0, "best_rolling_5km_pace": 300.0},
        {"distance_km": 21.2, "avg_pace_s_per_km": 352.0, "best_rolling_5km_pace": 302.0},
    ]

    predictions = build_predictions(load_athlete_config(), activities, [])
    marathon = predictions["race_predictions"]["42.2km"]

    assert marathon["pace_s_per_km"] > 347.0
    assert marathon["long_distance_evidence_pace_s_per_km"] == 347.0
    assert marathon["evidence_count"] == 0
    assert marathon["comparable_distance_count"] == 2
    assert marathon["scarcity_multiplier"] > 1.0
    assert "sparse comparable long-run evidence" in marathon["explanation"]


def test_marathon_scarcity_penalty_disappears_with_direct_marathon_evidence():
    activities = [
        {"distance_km": 42.2, "avg_pace_s_per_km": 360.0, "best_rolling_5km_pace": 300.0},
        {"distance_km": 40.0, "avg_pace_s_per_km": 365.0, "best_rolling_5km_pace": 302.0},
        {"distance_km": 35.0, "avg_pace_s_per_km": 370.0, "best_rolling_5km_pace": 304.0},
    ]

    predictions = build_predictions(load_athlete_config(), activities, [])
    marathon = predictions["race_predictions"]["42.2km"]

    assert marathon["evidence_count"] == 3
    assert marathon["evidence_factor"] == 1.0
    assert marathon["scarcity_multiplier"] == 1.0
    assert marathon["confidence"] > 0.3


def test_no_long_runs_lowers_long_distance_confidence_and_slows_prediction():
    activities = [{"distance_km": 5.0, "avg_pace_s_per_km": 300.0, "best_rolling_5km_pace": 295.0} for _ in range(6)]

    predictions = build_predictions(load_athlete_config(), activities, [])
    half = predictions["race_predictions"]["21.1km"]
    marathon = predictions["race_predictions"]["42.2km"]

    assert half["confidence"] < 0.1
    assert marathon["confidence"] < 0.1
    assert half["scarcity_multiplier"] == 1.12
    assert marathon["scarcity_multiplier"] == 1.25


def test_short_predictions_are_not_penalized_by_missing_long_distance_evidence():
    activities = [{"distance_km": 5.0, "avg_pace_s_per_km": 300.0, "best_rolling_5km_pace": 295.0}]

    predictions = build_predictions(load_athlete_config(), activities, [])
    five_k = predictions["race_predictions"]["5km"]

    assert five_k["scarcity_multiplier"] == 1.0
    assert five_k["evidence_factor"] == 1.0
    assert five_k["long_distance_evidence_pace_s_per_km"] is None


def test_prediction_payload_contains_long_distance_evidence_metadata():
    predictions = build_predictions(load_athlete_config(), [], [])
    prediction = predictions["race_predictions"]["42.2km"]

    assert {
        "evidence_count",
        "comparable_distance_count",
        "longest_run_km",
        "evidence_factor",
        "scarcity_multiplier",
        "long_distance_evidence_pace_s_per_km",
    }.issubset(prediction)
