from __future__ import annotations

from calendar import monthrange
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from statistics import mean
from typing import Any
from uuid import uuid4


@dataclass
class MLResult:
    training_run_id: str
    model_type: str
    row_count: int
    feature_names: list[str]
    metrics: dict
    predictions: dict
    confidence: float
    feature_importance: dict[str, float]
    warnings: list[str]
    positive_temporal_drivers: list[dict[str, Any]] = field(default_factory=list)
    negative_temporal_drivers: list[dict[str, Any]] = field(default_factory=list)
    temporal_feature_importance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


FEATURE_NAMES = [
    "distance_7d_km",
    "time_7d_min",
    "load_7d",
    "load_14d",
    "load_28d",
    "easy_fraction",
    "moderate_fraction",
    "hard_fraction",
    "long_run_distance_km",
    "longest_continuous_run_s",
    "acute_chronic_workload_ratio",
    "fatigue_adjusted_load_7d",
    "fatigue_adjusted_acute_chronic_workload_ratio",
    "avg_hr_7d_mean",
    "max_hr_7d_max",
    "min_hr_7d_mean",
    "hr_reserve_proxy_7d_mean",
    "hr_drift_proxy_7d_mean",
    "decoupling_proxy_7d_mean",
    "avg_cadence_7d_mean",
    "avg_power_w_7d_mean",
    "normalized_power_w_7d_mean",
    "power_variability_7d_mean",
    "power_hr_efficiency_7d_mean",
    "avg_temperature_c_7d_mean",
    "max_temperature_c_7d_max",
    "elevation_gain_per_km_7d_mean",
    "altitude_range_m_7d_mean",
    "calories_per_km_7d_mean",
    "total_calories_7d",
    "total_training_effect_7d_mean",
    "total_anaerobic_training_effect_7d_mean",
    "stride_length_mm_7d_mean",
    "vertical_oscillation_mm_7d_mean",
    "vertical_ratio_7d_mean",
    "stance_time_ms_7d_mean",
    "pace_variability_7d_mean",
    "hr_variability_7d_mean",
    "cadence_variability_7d_mean",
    "power_record_variability_7d_mean",
    "lap_pace_variability_7d_mean",
    "gps_coverage_7d_mean",
    "hrv_count_7d",
    "hrv_rmssd_ms_7d_mean",
    "run_walk_fatigue_multiplier_7d_mean",
]

TEMPORAL_WINDOWS = [7, 14, 28]
TEMPORAL_MIN_ROWS = 18
TEMPORAL_NONLINEAR_MIN_ROWS = 32
LINEAR_CLOSE_MAE_RATIO = 1.05


def train_per_run_model(daily_features: list[dict], weekly_features: list[dict], fitness_state: list[dict]) -> MLResult:
    """Train a fresh, in-memory model for every analysis run.

    The fitted model object is intentionally not returned. Dashboard/session state
    receives only metrics and predictions.
    """
    rows = _dataset_rows(weekly_features, fitness_state)
    diminishing_returns = _diminishing_returns_payload(weekly_features, fitness_state)
    warnings: list[str] = []
    training_run_id = uuid4().hex
    if not rows:
        warnings.append("Not enough history for supervised ML; using deterministic fallback prediction.")
        return _fallback_result(training_run_id, warnings, diminishing_returns)

    X = [[_number(row.get(name)) for name in FEATURE_NAMES] for row in rows]
    y = [_number(row.get("target_performance_index")) for row in rows]
    latest_x = X[-1]

    try:
        temporal_result = _train_temporal_model(rows, diminishing_returns, training_run_id, warnings)
        if temporal_result is not None:
            return temporal_result

        if len(rows) >= 12 and len(set(y)) > 1:
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import mean_absolute_error

            split = max(1, int(len(rows) * 0.75))
            model = HistGradientBoostingRegressor(random_state=42, max_iter=80)
            model.fit(X[:split], y[:split])
            validation_x = X[split:] or X[:split]
            validation_y = y[split:] or y[:split]
            validation_pred = model.predict(validation_x)
            mae = float(mean_absolute_error(validation_y, validation_pred))
            prediction = float(model.predict([latest_x])[0])
            confidence = _confidence(len(rows), mae)
            importance = {}
            if len(validation_x) >= 4:
                result = permutation_importance(model, validation_x, validation_y, n_repeats=3, random_state=42)
                importances = getattr(result, "importances_mean", [])
                importance = {
                    name: float(max(0.0, score))
                    for name, score in zip(FEATURE_NAMES, importances)
                }
            return MLResult(
                training_run_id=training_run_id,
                model_type="HistGradientBoostingRegressor",
                row_count=len(rows),
                feature_names=FEATURE_NAMES,
                metrics={**_temporal_fallback_metrics("not_enough_temporal_rows"), "mae": mae, "validation_rows": len(validation_x), **diminishing_returns["metrics"]},
                predictions={**_prediction_payload(prediction, confidence), **diminishing_returns["predictions"]},
                confidence=confidence,
                feature_importance=importance,
                warnings=warnings,
            )

        from sklearn.dummy import DummyRegressor
        from sklearn.metrics import mean_absolute_error

        model = DummyRegressor(strategy="mean")
        model.fit(X, y)
        prediction = float(model.predict([latest_x])[0])
        mae = float(mean_absolute_error(y, model.predict(X))) if y else 0.0
        warnings.append("Limited history; trained DummyRegressor fallback for this run.")
        confidence = min(0.45, 0.15 + len(rows) * 0.03)
        return MLResult(
            training_run_id=training_run_id,
            model_type="DummyRegressor",
            row_count=len(rows),
            feature_names=FEATURE_NAMES,
            metrics={**_temporal_fallback_metrics("not_enough_supervised_history"), "mae": mae, "validation_rows": len(rows), **diminishing_returns["metrics"]},
            predictions={**_prediction_payload(prediction, confidence), **diminishing_returns["predictions"]},
            confidence=confidence,
            feature_importance={},
            warnings=warnings,
        )
    except Exception as exc:
        warnings.append(f"ML training fallback used: {exc}")
        return _fallback_result(training_run_id, warnings, diminishing_returns)


def _dataset_rows(weekly_features: list[dict], fitness_state: list[dict]) -> list[dict]:
    state_by_date = {row.get("date"): row for row in fitness_state}
    rows = []
    for index, week in enumerate(weekly_features):
        date = week.get("date")
        future = fitness_state[min(index + 7, len(fitness_state) - 1)] if fitness_state else state_by_date.get(date, {})
        target = future.get("predicted_performance_index") if future else None
        if target is None:
            current = state_by_date.get(date, {})
            target = current.get("predicted_performance_index")
        if target is None:
            continue
        row = dict(week)
        row["target_performance_index"] = target
        rows.append(row)
    return rows


def _train_temporal_model(
    rows: list[dict],
    diminishing_returns: dict,
    training_run_id: str,
    warnings: list[str],
) -> MLResult | None:
    temporal_rows, temporal_feature_names = _temporal_dataset_rows(rows)
    if len(temporal_rows) < TEMPORAL_MIN_ROWS:
        return None
    y = [_number(row["target_performance_index"]) for row in temporal_rows]
    if len(set(y)) <= 1:
        return None

    X = [[_number(row.get(name)) for name in temporal_feature_names] for row in temporal_rows]
    latest_x = X[-1]
    folds = _time_series_fold_count(len(temporal_rows))
    if folds < 2:
        return None

    try:
        linear = _fit_temporal_linear(X, y, latest_x, folds, temporal_feature_names)
        candidates = [linear]
        if len(temporal_rows) >= TEMPORAL_NONLINEAR_MIN_ROWS:
            nonlinear = _fit_temporal_nonlinear(X, y, latest_x, folds, temporal_feature_names)
            if nonlinear is not None:
                candidates.append(nonlinear)
        selected = _select_temporal_candidate(candidates)
        confidence = _confidence(len(temporal_rows), selected["mae"])
        metrics = {
            "mae": selected["mae"],
            "validation_rows": selected["validation_rows"],
            "temporal_model_used": True,
            "lookback_days": TEMPORAL_WINDOWS,
            "sequence_row_count": len(temporal_rows),
            "sequence_feature_count": len(temporal_feature_names),
            "validation_strategy": "TimeSeriesSplit",
            "fold_count": folds,
            "temporal_fallback_reason": "",
            **diminishing_returns["metrics"],
        }
        return MLResult(
            training_run_id=training_run_id,
            model_type=selected["model_type"],
            row_count=len(temporal_rows),
            feature_names=temporal_feature_names,
            metrics=metrics,
            predictions={**_prediction_payload(float(selected["prediction"]), confidence), **diminishing_returns["predictions"]},
            confidence=confidence,
            feature_importance=selected["feature_importance"],
            warnings=warnings,
            positive_temporal_drivers=selected["positive_drivers"],
            negative_temporal_drivers=selected["negative_drivers"],
            temporal_feature_importance=selected["temporal_feature_importance"],
        )
    except Exception as exc:
        warnings.append(f"Temporal ML fallback used: {exc}")
        return None


def _temporal_dataset_rows(rows: list[dict]) -> tuple[list[dict[str, float]], list[str]]:
    temporal_rows: list[dict[str, float]] = []
    feature_names: list[str] = []
    for index, row in enumerate(rows):
        if index < min(TEMPORAL_WINDOWS) - 1:
            continue
        features: dict[str, float] = {}
        for base in FEATURE_NAMES:
            values = [_number(past.get(base)) for past in rows[: index + 1]]
            features[f"{base}__latest"] = values[-1]
            for window in TEMPORAL_WINDOWS:
                if len(values) < window:
                    continue
                window_values = values[-window:]
                features[f"{base}__lag_{window}d"] = window_values[0]
                features[f"{base}__mean_{window}d"] = mean(window_values)
                features[f"{base}__min_{window}d"] = min(window_values)
                features[f"{base}__max_{window}d"] = max(window_values)
                features[f"{base}__slope_{window}d"] = _slope(window_values)
                features[f"{base}__volatility_{window}d"] = _std(window_values)
                features[f"{base}__delta_{window}d"] = window_values[-1] - window_values[0]
                if len(values) >= window * 2:
                    prior = values[-window * 2 : -window]
                    features[f"{base}__recent_vs_prior_{window}d"] = mean(window_values) - mean(prior)
        features["target_performance_index"] = _number(row.get("target_performance_index"))
        temporal_rows.append(features)
        for name in features:
            if name != "target_performance_index" and name not in feature_names:
                feature_names.append(name)
    for row in temporal_rows:
        for name in feature_names:
            row.setdefault(name, 0.0)
    return temporal_rows, feature_names


def _fit_temporal_linear(X: list[list[float]], y: list[float], latest_x: list[float], folds: int, feature_names: list[str]) -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    validation_y: list[float] = []
    validation_pred: list[float] = []
    splitter = TimeSeriesSplit(n_splits=folds)
    for train_index, test_index in splitter.split(X):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))
        train_x = [X[index] for index in train_index]
        train_y = [y[index] for index in train_index]
        test_x = [X[index] for index in test_index]
        test_y = [y[index] for index in test_index]
        model.fit(train_x, train_y)
        validation_y.extend(test_y)
        validation_pred.extend(float(value) for value in model.predict(test_x))
    mae = float(mean_absolute_error(validation_y, validation_pred))
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))
    model.fit(X, y)
    ridge = model.named_steps["ridgecv"]
    coefficients = [float(value) for value in getattr(ridge, "coef_", [])]
    signed = _aggregate_signed_temporal_features(feature_names, coefficients)
    absolute_importance = _normalize_importance({feature: abs(score) for feature, score in signed.items()})
    return {
        "model_type": "TemporalRidgeCV",
        "mae": mae,
        "validation_rows": len(validation_y),
        "prediction": float(model.predict([latest_x])[0]),
        "feature_importance": absolute_importance,
        "temporal_feature_importance": absolute_importance,
        "positive_drivers": _temporal_driver_rows(signed, positive=True),
        "negative_drivers": _temporal_driver_rows(signed, positive=False),
    }


def _fit_temporal_nonlinear(X: list[list[float]], y: list[float], latest_x: list[float], folds: int, feature_names: list[str]) -> dict | None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import TimeSeriesSplit

    validation_y: list[float] = []
    validation_pred: list[float] = []
    splitter = TimeSeriesSplit(n_splits=folds)
    for train_index, test_index in splitter.split(X):
        model = HistGradientBoostingRegressor(random_state=42, max_iter=80)
        train_x = [X[index] for index in train_index]
        train_y = [y[index] for index in train_index]
        test_x = [X[index] for index in test_index]
        test_y = [y[index] for index in test_index]
        model.fit(train_x, train_y)
        validation_y.extend(test_y)
        validation_pred.extend(float(value) for value in model.predict(test_x))
    if not validation_y:
        return None
    mae = float(mean_absolute_error(validation_y, validation_pred))
    model = HistGradientBoostingRegressor(random_state=42, max_iter=80)
    model.fit(X, y)
    importance: dict[str, float] = {}
    if len(X) >= 8:
        result = permutation_importance(model, X, y, n_repeats=3, random_state=42)
        importance = _normalize_importance(_aggregate_temporal_importance(feature_names, getattr(result, "importances_mean", [])))
    return {
        "model_type": "TemporalHistGradientBoostingRegressor",
        "mae": mae,
        "validation_rows": len(validation_y),
        "prediction": float(model.predict([latest_x])[0]),
        "feature_importance": importance,
        "temporal_feature_importance": importance,
        "positive_drivers": [],
        "negative_drivers": [],
    }


def _select_temporal_candidate(candidates: list[dict]) -> dict:
    linear = next(candidate for candidate in candidates if candidate["model_type"] == "TemporalRidgeCV")
    best = min(candidates, key=lambda candidate: candidate["mae"])
    if best is not linear and best["mae"] <= linear["mae"] / LINEAR_CLOSE_MAE_RATIO:
        return best
    return linear


def _time_series_fold_count(row_count: int) -> int:
    return max(0, min(5, row_count // 6))


def _aggregate_signed_temporal_features(feature_names: list[str], values: list[float]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for name, value in zip(feature_names, values):
        grouped[_base_temporal_feature_name(name)] = grouped.get(_base_temporal_feature_name(name), 0.0) + float(value)
    return grouped


def _aggregate_temporal_importance(feature_names: list[str], values: Any) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for name, value in zip(feature_names, values):
        base = _base_temporal_feature_name(name)
        grouped[base] = grouped.get(base, 0.0) + max(0.0, float(value))
    return grouped


def _base_temporal_feature_name(name: str) -> str:
    return name.split("__", 1)[0]


def _normalize_importance(values: dict[str, float]) -> dict[str, float]:
    clean = {name: max(0.0, float(value)) for name, value in values.items() if value and value > 0}
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {name: round(value / total, 6) for name, value in sorted(clean.items(), key=lambda item: item[1], reverse=True)}


def _temporal_driver_rows(signed: dict[str, float], *, positive: bool, limit: int = 8) -> list[dict[str, float | str]]:
    items = [(name, value) for name, value in signed.items() if (value > 0 if positive else value < 0)]
    ordered = sorted(items, key=lambda item: abs(item[1]), reverse=True)[:limit]
    return [{"feature": name, "coefficient": round(value, 6), "strength": round(abs(value), 6)} for name, value in ordered]


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2
    y_mean = mean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    if denominator <= 0:
        return 0.0
    return sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values)) / denominator


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    value_mean = mean(values)
    return (sum((value - value_mean) ** 2 for value in values) / len(values)) ** 0.5


def _fallback_result(training_run_id: str, warnings: list[str], diminishing_returns: dict | None = None) -> MLResult:
    diminishing_returns = diminishing_returns or _default_diminishing_returns("fallback")
    return MLResult(
        training_run_id=training_run_id,
        model_type="DeterministicFallback",
        row_count=0,
        feature_names=FEATURE_NAMES,
        metrics={**_temporal_fallback_metrics("fallback"), "mae": None, "validation_rows": 0, **diminishing_returns["metrics"]},
        predictions={**_prediction_payload(1.0, 0.1), **diminishing_returns["predictions"]},
        confidence=0.1,
        feature_importance={},
        warnings=warnings,
    )


def _temporal_fallback_metrics(reason: str) -> dict:
    return {
        "temporal_model_used": False,
        "lookback_days": TEMPORAL_WINDOWS,
        "sequence_row_count": 0,
        "sequence_feature_count": 0,
        "validation_strategy": "none",
        "fold_count": 0,
        "temporal_fallback_reason": reason,
    }


def _prediction_payload(performance_index: float, confidence: float) -> dict:
    return {
        "performance_index_7d": performance_index,
        "expected_adaptation": min(1.0, max(0.0, performance_index / 100.0)),
        "overload_risk_modifier": max(0.0, 1.0 - confidence),
    }


def _diminishing_returns_payload(weekly_features: list[dict], fitness_state: list[dict]) -> dict:
    observations = _diminishing_return_observations(weekly_features, fitness_state)
    payload = _diminishing_returns_from_observations(observations)
    payload["metrics"]["diminishing_returns_history"] = _monthly_diminishing_returns_history(observations, fitness_state)
    return payload


def _diminishing_returns_from_observations(observations: list[dict[str, Any]], latest_performance_index: float | None = None) -> dict:
    performances = [row["performance_index"] for row in observations]
    responses = [row["response"] for row in observations]
    if len(observations) < 6:
        return _default_diminishing_returns("insufficient_true_7d_history", len(observations))
    if max(performances) - min(performances) < 1.0:
        return _default_diminishing_returns("too_little_fitness_variation", len(observations))

    mean_performance = mean(performances)
    mean_response = mean(responses)
    denominator = sum((value - mean_performance) ** 2 for value in performances)
    if denominator <= 1e-9:
        return _default_diminishing_returns("too_little_fitness_variation", len(observations))

    beta = sum((performance - mean_performance) * (response - mean_response) for performance, response in zip(performances, responses)) / denominator
    alpha = mean_response - beta * mean_performance
    if beta >= 0:
        return _default_diminishing_returns("non_diminishing_history", len(observations), alpha, beta)

    reference_response = alpha + beta * _percentile25(performances)
    current_performance = performances[-1] if latest_performance_index is None else latest_performance_index
    current_response = alpha + beta * current_performance
    if reference_response <= 0:
        return _default_diminishing_returns("non_positive_reference_response", len(observations), alpha, beta, reference_response, current_response)

    raw_ratio = current_response / reference_response
    factor = _realistic_diminishing_returns_factor(raw_ratio)
    return {
        "predictions": {"diminishing_returns_factor": round(factor, 3)},
        "metrics": {
            "diminishing_returns_observations": len(observations),
            "diminishing_returns_alpha": round(alpha, 6),
            "diminishing_returns_beta": round(beta, 6),
            "diminishing_returns_reference_response": round(reference_response, 6),
            "diminishing_returns_current_response": round(current_response, 6),
            "diminishing_returns_raw_ratio": round(raw_ratio, 6),
            "diminishing_returns_reason": "learned_from_history",
        },
    }


def _diminishing_return_observations(weekly_features: list[dict], fitness_state: list[dict]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for index, week in enumerate(weekly_features):
        future_index = index + 7
        if future_index >= len(fitness_state):
            continue
        current = fitness_state[index] if index < len(fitness_state) else {}
        future = fitness_state[future_index]
        performance = _number(current.get("predicted_performance_index"))
        future_performance = _number(future.get("predicted_performance_index"))
        load = _number(week.get("load_7d"))
        if performance <= 0 or future_performance <= 0:
            continue
        observation_date = _parse_date(week.get("date") or current.get("date"))
        future_date = _parse_date(future.get("date"))
        if observation_date is None or future_date is None:
            continue
        response = max(0.0, future_performance - performance) / max(0.25, load / 100.0)
        observations.append({"date": observation_date, "future_date": future_date, "performance_index": performance, "response": response})
    return observations


def _monthly_diminishing_returns_history(observations: list[dict[str, Any]], fitness_state: list[dict]) -> list[dict[str, Any]]:
    dated_performances: list[tuple[date, float]] = []
    for row in fitness_state:
        value = _number(row.get("predicted_performance_index"))
        row_date = _parse_date(row.get("date"))
        if value > 0 and row_date is not None:
            dated_performances.append((row_date, value))
    if not dated_performances and not observations:
        return []
    month_ends = _month_ends([row_date for row_date, _ in dated_performances] + [row["future_date"] for row in observations])
    history: list[dict[str, Any]] = []
    for month_end in month_ends:
        available = [row for row in observations if row["future_date"] <= month_end]
        latest_performance = _latest_performance_index(dated_performances, month_end)
        payload = _diminishing_returns_from_observations(available, latest_performance)
        metrics = payload["metrics"]
        history.append(
            {
                "month": month_end.strftime("%Y-%m"),
                "factor": payload["predictions"]["diminishing_returns_factor"],
                "observations": metrics.get("diminishing_returns_observations", 0),
                "alpha": metrics.get("diminishing_returns_alpha"),
                "beta": metrics.get("diminishing_returns_beta"),
                "reference_response": metrics.get("diminishing_returns_reference_response"),
                "current_response": metrics.get("diminishing_returns_current_response"),
                "raw_ratio": metrics.get("diminishing_returns_raw_ratio"),
                "latest_performance_index": round(latest_performance, 6) if latest_performance is not None else None,
                "reason": metrics.get("diminishing_returns_reason"),
            }
        )
    return history


def _month_ends(dates: list[date]) -> list[date]:
    months = sorted({(value.year, value.month) for value in dates})
    return [date(year, month, monthrange(year, month)[1]) for year, month in months]


def _latest_performance_index(dated_performances: list[tuple[date, float]], month_end: date) -> float | None:
    values = [value for row_date, value in dated_performances if row_date <= month_end]
    return values[-1] if values else None


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:10]).date()
        except ValueError:
            return None
    return None


def _default_diminishing_returns(
    reason: str,
    observations: int = 0,
    alpha: float | None = None,
    beta: float | None = None,
    reference_response: float | None = None,
    current_response: float | None = None,
) -> dict:
    return {
        "predictions": {"diminishing_returns_factor": 1.0},
        "metrics": {
            "diminishing_returns_observations": observations,
            "diminishing_returns_alpha": alpha,
            "diminishing_returns_beta": beta,
            "diminishing_returns_reference_response": reference_response,
            "diminishing_returns_current_response": current_response,
            "diminishing_returns_raw_ratio": None,
            "diminishing_returns_reason": reason,
        },
    }


def _percentile25(values: list[float]) -> float:
    ordered = sorted(values)
    index = int((len(ordered) - 1) * 0.25)
    return ordered[index]


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _realistic_diminishing_returns_factor(raw_ratio: float) -> float:
    """Shrink a noisy linear response ratio into a coach-facing plan modifier.

    The baseline plan benefit is already small and horizon-scaled, so the
    history factor should be a conservative dampener, not a hard veto. Even a
    very low or slightly negative fitted response means "be cautious" rather
    than "erase two thirds of expected adaptation".
    """
    bounded_ratio = _clip(raw_ratio, 0.0, 1.0)
    return _clip(1.0 - 0.35 * (1.0 - bounded_ratio), 0.65, 1.0)


def _confidence(row_count: int, mae: float) -> float:
    history = min(0.9, row_count / 45.0)
    error = max(0.0, 1.0 - mae / 25.0)
    return round(max(0.1, min(0.9, 0.25 + 0.45 * history + 0.3 * error)), 3)


def _number(value) -> float:
    try:
        if value is None or value != value:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
