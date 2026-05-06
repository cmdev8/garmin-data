# Garmin Runner

Dashboard-first Garmin FIT running analysis with per-run sklearn ML and a v2-only training plan optimizer.

## Streamlit Dashboard

Launch the dashboard:

```bash
python -m garmin_runner dashboard
```

or:

```bash
streamlit run src/garmin_runner/dashboard/app.py
```

Upload one or more Garmin `.fit` files in the browser. Athlete settings such as planning horizon, goal distance, availability, fatigue sensitivity, and run-walk planning preferences are entered in the UI rather than uploaded as `athlete.yaml`.

The dashboard parses uploaded FIT bytes in memory: raw FIT files are not stored by the app, and trained model artifacts are not written to disk. Processed dashboard data lives in Streamlit session state until reset.

The UI exposes one user download:

- `training_plan_report.pdf`

The PDF is a coach-style summary of the optimizer-selected plan: schedule, score components, expected benefits, ML diagnostics, zones, plan-quality checks, and rationale. It excludes raw FIT files, raw JSON dumps, and trained model artifacts.

## Machine Learning And Predictions

Every analysis run trains a fresh in-memory sklearn model.

When enough chronological history exists, the primary path builds temporal weekly samples from 7, 14, and 28 day lookbacks and trains `TemporalRidgeCV` with `TimeSeriesSplit` validation. A temporal `HistGradientBoostingRegressor` candidate is considered only with larger history and must beat the linear model clearly on chronological `MAE`. If the temporal path is not usable, the pipeline falls back to tabular `HistGradientBoostingRegressor`, then `DummyRegressor`, then deterministic fallback values.

Baseline race predictions use observed pace evidence and the Riegel equation. Longer predictions are conservative when history is sparse:

- Half marathon uses comparable `18–25 km` efforts, with 3 comparable runs treated as enough evidence.
- Marathon uses direct `>=30 km` efforts, with `18–30 km` efforts as supporting evidence; 3 direct long runs are treated as enough evidence.
- Sparse long-distance history applies a scarcity discount, lowers confidence, and can enforce a comparable-distance pace floor. This prevents optimistic extrapolations such as a fast marathon prediction from only a couple of slower half-marathon-length runs.

Plan-followed prediction changes are estimates, not guarantees. They combine optimizer score components, horizon/race-distance scaling, long-distance evidence, and the athlete's learned `diminishing returns factor`. Positive expected gains are reduced when the uploaded history shows smaller marginal response at higher fitness; fatigue or overload downside is not softened.

## Training Plan Optimizer

The dashboard uses the v2 optimizer only. It generates candidate plans for the configured horizon, validates hard safety constraints, scores valid candidates, and selects the highest-scoring safe plan.

The optimizer considers:

- progressive overload and week-to-week volume caps;
- easy-volume dominance, hard/easy spacing, hard-day limits, and long-run share;
- current `fitness`, `fatigue`, `form`, `overload risk`, and fatigue-adjusted training load;
- goal-distance specificity;
- historical improvement drivers from the controlled performance-trend analysis;
- ML expected adaptation and overload modifiers when model `confidence` is high enough.

The historical improvement driver is a soft signal. It can favor threshold, VO2max, speed-support, long-run, base, recovery, or run-walk-safe candidates when current safety allows it, but it never overrides hard constraints.

Run-walk prescriptions are generated only when run-walk support and run-walk candidate planning are explicitly enabled. Forced walk breaks in history are treated as fatigue/pacing risk, not as a reason to add faster workouts.

## Developer CLI

The `analyze` command is for local debugging. It runs the same in-memory pipeline. It keeps results in memory unless `--export-dir` is explicitly provided.

```bash
python -m garmin_runner analyze \
  --fit-dir ./test-data \
  --athlete-config ./athlete.example.yaml
```

Optional explicit export for debugging:

```bash
python -m garmin_runner analyze \
  --fit-dir ./test-data \
  --export-dir ./output \
  --planning-horizon-days 14
```

`--export-dir` can write processed parquet/json/markdown artifacts for development inspection. It is not part of the normal dashboard workflow and does not persist trained model objects.

## Limitations

- Predictions and plan benefits are model outputs, not race guarantees.
- Terrain is represented through elevation, grade, route, temperature, and quality proxies; FIT uploads do not provide reliable surface type.
- Optional FIT fields such as power, running dynamics, temperature, HRV, or advanced training effect depend on device support and may be missing.
- The optimizer plans 1-28 days at a time. It is not a full season-periodization coach.
- The app does not diagnose injury, illness, nutrition, sleep, or life stress directly unless those signals appear indirectly in the uploaded training data.
