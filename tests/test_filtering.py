from garmin_runner.config import load_athlete_config
from garmin_runner.fit.filters import classify_activity
from garmin_runner.fit.schema import Activity, Record


def _activity(sport):
    return Activity(
        activity_id="a1",
        source_file="fixture.fit",
        start_time=None,
        sport=sport,
        sub_sport=None,
        keep=False,
        keep_reason="",
        activity_type_confidence="rejected",
        activity_type_reason="",
    )


def test_running_activity_is_kept():
    activity = classify_activity(_activity("running"), [], load_athlete_config())

    assert activity.keep is True
    assert activity.activity_type_confidence == "high"


def test_cycling_activity_is_rejected():
    activity = classify_activity(_activity("cycling"), [], load_athlete_config())

    assert activity.keep is False
    assert activity.activity_type_confidence == "rejected"


def test_walking_rejected_by_default():
    records = [Record("a1", None, 0, speed_mps=2.5, cadence_spm=150)]

    activity = classify_activity(_activity("walking"), records, load_athlete_config())

    assert activity.keep is False


def test_walking_can_be_walk_run_candidate_with_flag():
    records = [Record("a1", None, 0, speed_mps=2.5, cadence_spm=150)]

    activity = classify_activity(
        _activity("walking"),
        records,
        load_athlete_config(include_walk_run_candidates=True),
    )

    assert activity.keep is True
    assert activity.activity_type_confidence == "low"
