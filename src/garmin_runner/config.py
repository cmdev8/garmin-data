from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
T = TypeVar("T")


@dataclass
class AthleteSection:
    name: str = "athlete"
    timezone: str = "UTC"


@dataclass
class GoalsSection:
    target_distance_km: float | None = 10.0
    target_date: str | None = None
    priority: str = "improve_endurance"


@dataclass
class AvailabilitySection:
    days_per_week: int = 4
    available_days: list[str] = field(default_factory=lambda: ["Tuesday", "Thursday", "Saturday", "Sunday"])
    preferred_long_run_day: str = "Sunday"
    max_minutes_per_week: int | None = 300


@dataclass
class PhysiologySection:
    max_hr: int | None = None
    resting_hr: int | None = None
    lactate_threshold_hr: int | None = None
    lactate_threshold_pace_s_per_km: float | None = None


@dataclass
class TrainingSection:
    max_volume_increase_pct: float = 10.0
    max_hard_days_per_week: int = 2
    deload_every_n_weeks: int = 4


@dataclass
class RunWalkSection:
    enabled: bool = True
    allow_walk_run_candidates: bool = False
    min_segment_duration_s: int = 10
    default_prescription: str = "progress_run_duration"


@dataclass
class PlanningSection:
    horizon_days: int = 7


@dataclass
class AthleteConfig:
    athlete: AthleteSection = field(default_factory=AthleteSection)
    goals: GoalsSection = field(default_factory=GoalsSection)
    availability: AvailabilitySection = field(default_factory=AvailabilitySection)
    physiology: PhysiologySection = field(default_factory=PhysiologySection)
    training: TrainingSection = field(default_factory=TrainingSection)
    run_walk: RunWalkSection = field(default_factory=RunWalkSection)
    planning: PlanningSection = field(default_factory=PlanningSection)


def _section(cls: type[T], data: dict[str, Any] | None) -> T:
    defaults = cls()
    if not data:
        return defaults
    allowed = {item.name for item in fields(cast(Any, cls))}
    values = {k: v for k, v in data.items() if k in allowed}
    return cls(**{**defaults.__dict__, **values})


def validate_config(config: AthleteConfig) -> AthleteConfig:
    horizon = config.planning.horizon_days
    if not isinstance(horizon, int) or horizon < 1 or horizon > 28:
        raise ValueError("planning.horizon_days must be an integer from 1 to 28")

    days = [day for day in config.availability.available_days if day in WEEKDAYS]
    if not days:
        days = WEEKDAYS[: max(1, min(7, config.availability.days_per_week))]
    config.availability.available_days = days

    if config.availability.preferred_long_run_day not in WEEKDAYS:
        config.availability.preferred_long_run_day = "Sunday"

    config.training.max_hard_days_per_week = max(0, int(config.training.max_hard_days_per_week))
    config.training.max_volume_increase_pct = max(0.0, float(config.training.max_volume_increase_pct))
    return config


def load_athlete_config(
    path: str | Path | None = None,
    *,
    planning_horizon_days: int | None = None,
    include_walk_run_candidates: bool | None = None,
) -> AthleteConfig:
    raw: dict[str, Any] = {}
    if path:
        config_path = Path(path)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError("athlete config must be a YAML mapping")
            raw = loaded

    return load_athlete_config_from_mapping(
        raw,
        planning_horizon_days=planning_horizon_days,
        include_walk_run_candidates=include_walk_run_candidates,
    )


def load_athlete_config_from_mapping(
    raw: dict[str, Any] | None,
    *,
    planning_horizon_days: int | None = None,
    include_walk_run_candidates: bool | None = None,
) -> AthleteConfig:
    raw = raw or {}
    config = AthleteConfig(
        athlete=_section(AthleteSection, raw.get("athlete")),
        goals=_section(GoalsSection, raw.get("goals")),
        availability=_section(AvailabilitySection, raw.get("availability")),
        physiology=_section(PhysiologySection, raw.get("physiology")),
        training=_section(TrainingSection, raw.get("training")),
        run_walk=_section(RunWalkSection, raw.get("run_walk")),
        planning=_section(PlanningSection, raw.get("planning")),
    )
    if planning_horizon_days is not None:
        config.planning.horizon_days = planning_horizon_days
    if include_walk_run_candidates is not None:
        config.run_walk.allow_walk_run_candidates = include_walk_run_candidates
    return validate_config(config)
