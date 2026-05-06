from pathlib import Path

import pytest

from garmin_runner.config import load_athlete_config


def test_default_horizon_is_7_days():
    config = load_athlete_config()

    assert config.planning.horizon_days == 7


def test_yaml_horizon_overrides_default(tmp_path: Path):
    config_path = tmp_path / "athlete.yaml"
    config_path.write_text("planning:\n  horizon_days: 14\n", encoding="utf-8")

    config = load_athlete_config(config_path)

    assert config.planning.horizon_days == 14


def test_cli_horizon_overrides_yaml(tmp_path: Path):
    config_path = tmp_path / "athlete.yaml"
    config_path.write_text("planning:\n  horizon_days: 10\n", encoding="utf-8")

    config = load_athlete_config(config_path, planning_horizon_days=3)

    assert config.planning.horizon_days == 3


@pytest.mark.parametrize("value", [0, 29])
def test_invalid_horizon_fails(value: int):
    with pytest.raises(ValueError, match="planning.horizon_days"):
        load_athlete_config(planning_horizon_days=value)
