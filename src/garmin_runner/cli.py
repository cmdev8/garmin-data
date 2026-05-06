from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from garmin_runner.config import load_athlete_config
from garmin_runner.pipeline import AnalysisConfig, run_analysis

app = typer.Typer(no_args_is_help=True)


@app.callback()
def root() -> None:
    """Garmin Runner command line interface."""


@app.command()
def analyze(
    fit_dir: Annotated[Path, typer.Option(help="Folder containing Garmin .fit files.")],
    athlete_config: Annotated[Path | None, typer.Option(help="Optional athlete YAML config.")] = None,
    export_dir: Annotated[
        Path | None,
        typer.Option(help="Optional developer export directory. Dashboard runs stay in memory."),
    ] = None,
    include_walk_run_candidates: Annotated[
        bool,
        typer.Option(help="Allow low-confidence walking activities that look like run-walk candidates."),
    ] = False,
    planning_horizon_days: Annotated[
        int | None,
        typer.Option(help="Override planning.horizon_days from athlete config. Valid range: 1-28."),
    ] = None,
) -> None:
    """Run ML analysis. Results stay in memory unless --export-dir is set."""
    config = load_athlete_config(
        athlete_config,
        planning_horizon_days=planning_horizon_days,
        include_walk_run_candidates=include_walk_run_candidates,
    )
    result = run_analysis(
        AnalysisConfig(
            athlete_config=config,
            fit_dir=fit_dir,
            export_dir=export_dir,
        )
    )
    typer.echo(result.summary())


@app.command()
def dashboard(
    dev_keep_temp: Annotated[
        bool,
        typer.Option(help="Debug only: preserve temporary compatibility workspaces if any are used."),
    ] = False,
) -> None:
    """Launch the Streamlit upload dashboard."""
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    env = dict(os.environ)
    if dev_keep_temp:
        env["GARMIN_RUNNER_DEV_KEEP_TEMP"] = "1"
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True, shell=False, env=env)
    except ModuleNotFoundError as exc:
        raise typer.BadParameter("Streamlit is not installed. Install project dependencies in the venv.") from exc


def main(argv: list[str] | None = None) -> None:
    app(args=argv)


if __name__ == "__main__":
    main()
