from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class DashboardConfig:
    dev_keep_temp: bool = False
    show_debug: bool = False


def load_dashboard_config() -> DashboardConfig:
    return DashboardConfig(
        dev_keep_temp=os.environ.get("GARMIN_RUNNER_DEV_KEEP_TEMP") == "1",
        show_debug=os.environ.get("GARMIN_RUNNER_DEBUG") == "1",
    )
