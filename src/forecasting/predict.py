"""
predict.py
==========
Loads the trained freight forecasting model and produces recursive
multi-step forecasts, market-direction signals, and AI-assisted
(non-binding) chartering suggestions.

Primary entry point: :func:`forecast_freight`.

IMPORTANT LIMITATION — read before using this module's output for anything
real: the underlying data is the Baltic Dry Index (BDI), a broad dry-bulk
market indicator built from multiple standardized shipping routes. It is
NOT the freight rate for any specific route (e.g. Australia -> Paradip),
and forecasts here should not be presented as such. A production version of
FreightAI should replace or augment BDI with route-specific freight-rate
data. This limitation string is included in every result returned by
:func:`forecast_freight` so downstream consumers (e.g. the dashboard) can
surface it directly.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.forecasting.features import build_feature_row, infer_step_frequency
from src.forecasting.train import (
    FEATURE_COLUMNS_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
)

logger = logging.getLogger("freightai.forecasting.predict")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

# Forecast horizons this module was explicitly asked to support.
SUPPORTED_HORIZONS = (7, 30, 90)

# Horizons at or below this many days are treated as the model's normal,
# supported operating range. Beyond it, recursive forecasting compounds its
# own prediction errors step over step, so results are explicitly labeled
# experimental rather than presented with false confidence.
RELIABLE_HORIZON_DAYS = 30

# Minimum absolute percentage move (forecast end vs. last known value)
# before a trend is called INCREASING/DECREASING rather than STABLE.
TREND_THRESHOLD_PCT = 3.0

BDI_LIMITATION_NOTE = (
    "This forecast is derived from the Baltic Dry Index (BDI), a broad "
    "dry-bulk market indicator aggregated across multiple standardized "
    "shipping routes. It is NOT the freight rate for any specific route "
    "(e.g. Australia -> Paradip) and should not be presented as such. A "
    "production version of FreightAI should replace or augment BDI with "
    "route-specific freight-rate data."
)

ADVICE_DISCLAIMER = (
    "The market_signal and recommendation below are AI-assisted outputs "
    "generated from historical statistical patterns. They are NOT "
    "guaranteed financial or chartering advice, and should be reviewed by "
    "a qualified chartering/commercial professional before any decision is made."
)


class ArtifactsNotFoundError(RuntimeError):
    """Raised when the trained model / metrics / feature contract aren't on disk yet."""


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_artifacts(
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
):
    """Load the trained model, its metrics report, and its feature-column
    contract from ``models/``. Raises :class:`ArtifactsNotFoundError` with a
    clear, actionable message if training hasn't been run yet.
    """
    missing = [p for p in (model_path, metrics_path, feature_columns_path) if not p.exists()]
    if missing:
        raise ArtifactsNotFoundError(
            "Trained model artifacts not found: "
            f"{[str(p) for p in missing]}. Run `python -m src.forecasting.train` first."
        )

    model = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text())
    feature_contract = json.loads(feature_columns_path.read_text())
    return model, metrics, feature_contract


def load_history(data_path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned freight history used as the seed for recursive
    forecasting. Raises a clear error if unavailable.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find '{data_path}'. Run the data pipeline module first."
        )
    df = pd.read_csv(data_path)
    if "date" not in df.columns or "freight_rate" not in df.columns:
        raise ValueError(
            f"'{data_path}' is missing 'date'/'freight_rate' columns "
            "(expected standardized schema from the data pipeline module)."
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "freight_rate"]).sort_values("date").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"'{data_path}' has no usable rows after parsing.")
    return df


# ---------------------------------------------------------------------------
# Recursive multi-step forecasting
# ---------------------------------------------------------------------------

def _recursive_forecast(
    model, history_df: pd.DataFrame, feature_columns: List[str], n_days: int
) -> List[Dict]:
    """Step forward ``n_days`` observations, at each step building features
    from history-so-far (real + previously-predicted values only — never
    the value being predicted) and appending the new prediction to history
    before moving to the next step.
    """
    history = history_df.set_index("date")["freight_rate"].astype(float).copy()
    step = infer_step_frequency(history_df["date"])
    last_date = history.index.max()

    forecasts: List[Dict] = []
    current_date = last_date
    for _ in range(n_days):
        current_date = current_date + step
        row = build_feature_row(history, current_date)
        X_row = pd.DataFrame([row], columns=feature_columns)
        pred = float(model.predict(X_row)[0])
        forecasts.append({"date": current_date, "predicted_freight_rate": pred})
        history.loc[current_date] = pred

    return forecasts


# ---------------------------------------------------------------------------
# Market signal + recommendation
# ---------------------------------------------------------------------------

def _classify_trend(last_actual: float, forecast_end_value: float) -> str:
    if last_actual == 0 or pd.isna(last_actual):
        return "STABLE"
    pct_change = (forecast_end_value - last_actual) / last_actual * 100
    if pct_change > TREND_THRESHOLD_PCT:
        return "INCREASING"
    if pct_change < -TREND_THRESHOLD_PCT:
        return "DECREASING"
    return "STABLE"


def _recommend(signal: str) -> str:
    return {
        "INCREASING": "BUY/CHARTER NOW",
        "DECREASING": "WAIT",
        "STABLE": "MONITOR",
    }[signal]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def forecast_freight(
    days: int = 30,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
    data_path: Path = PROCESSED_DATA_PATH,
) -> Dict:
    """Forecast freight market conditions ``days`` ahead of the latest known
    data point, and derive a market-direction signal and an AI-assisted
    (non-binding) chartering suggestion.

    Returns a structured, JSON-serializable dict — designed to be consumed
    directly by the (not-yet-built) Streamlit dashboard:

        {
            "generated_at": iso-timestamp,
            "model_used": str,
            "horizon_days": int,
            "historical_last_date": "YYYY-MM-DD",
            "historical_last_value": float,
            "forecast": [{"date": "YYYY-MM-DD", "predicted_freight_rate": float}, ...],
            "market_signal": "INCREASING" | "STABLE" | "DECREASING",
            "recommendation": "BUY/CHARTER NOW" | "WAIT" | "MONITOR",
            "experimental": bool,
            "experimental_reason": str | None,
            "validation_metrics": {...},   # selected model's validation performance
            "disclaimer": str,
            "limitations": [str, ...],
        }

    Raises :class:`ArtifactsNotFoundError` if the model hasn't been trained
    yet, and standard ``FileNotFoundError``/``ValueError`` if the processed
    freight dataset is missing or malformed. Callers (e.g. the dashboard)
    should catch these and show a friendly message rather than let them
    propagate raw.
    """
    if not isinstance(days, int) or days <= 0:
        raise ValueError(f"'days' must be a positive integer, got {days!r}.")

    model, metrics_report, feature_contract = load_artifacts(
        model_path, metrics_path, feature_columns_path
    )
    feature_columns = feature_contract["feature_columns"]
    history_df = load_history(data_path)

    forecast_points = _recursive_forecast(model, history_df, feature_columns, days)

    last_actual_date = history_df["date"].iloc[-1]
    last_actual_value = float(history_df["freight_rate"].iloc[-1])
    forecast_end_value = forecast_points[-1]["predicted_freight_rate"]

    market_signal = _classify_trend(last_actual_value, forecast_end_value)
    recommendation = _recommend(market_signal)

    selected_model_name = metrics_report.get("selected_model", "unknown")
    validation_metrics = (
        metrics_report.get("metrics", {}).get(selected_model_name, {}).get("validation")
    )

    experimental = False
    experimental_reasons = []
    if days > RELIABLE_HORIZON_DAYS:
        experimental = True
        experimental_reasons.append(
            f"Horizon of {days} days exceeds the {RELIABLE_HORIZON_DAYS}-day range this "
            "model was validated over; recursive forecasting compounds prediction "
            "error at longer horizons."
        )
    if selected_model_name == "moving_average_baseline":
        experimental = True
        experimental_reasons.append(
            "The selected model is the naive moving-average baseline, not a fitted "
            "ML model — treat any forecast from it as indicative only."
        )
    if validation_metrics and validation_metrics.get("r2") is not None and validation_metrics["r2"] < 0:
        experimental = True
        experimental_reasons.append(
            "The selected model's validation R2 is negative (worse than predicting "
            "the mean), so forecasts should be treated cautiously."
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used": selected_model_name,
        "horizon_days": days,
        "historical_last_date": str(pd.Timestamp(last_actual_date).date()),
        "historical_last_value": last_actual_value,
        "forecast": [
            {"date": str(pd.Timestamp(p["date"]).date()), "predicted_freight_rate": round(p["predicted_freight_rate"], 3)}
            for p in forecast_points
        ],
        "market_signal": market_signal,
        "recommendation": recommendation,
        "experimental": experimental,
        "experimental_reason": " ".join(experimental_reasons) if experimental_reasons else None,
        "validation_metrics": validation_metrics,
        "disclaimer": ADVICE_DISCLAIMER,
        "limitations": [BDI_LIMITATION_NOTE],
    }
    return result


def forecast_7_days(**kwargs) -> Dict:
    """Convenience wrapper: :func:`forecast_freight` with a 7-day horizon."""
    return forecast_freight(days=7, **kwargs)


def forecast_30_days(**kwargs) -> Dict:
    """Convenience wrapper: :func:`forecast_freight` with a 30-day horizon."""
    return forecast_freight(days=30, **kwargs)


def forecast_90_days(**kwargs) -> Dict:
    """Convenience wrapper: :func:`forecast_freight` with a 90-day horizon
    (always returned as ``experimental``; see :data:`RELIABLE_HORIZON_DAYS`).
    """
    return forecast_freight(days=90, **kwargs)


if __name__ == "__main__":
    for horizon in SUPPORTED_HORIZONS:
        try:
            res = forecast_freight(days=horizon)
            print(f"\n=== {horizon}-day forecast (experimental={res['experimental']}) ===")
            print(f"Model: {res['model_used']}  Signal: {res['market_signal']}  "
                  f"Recommendation: {res['recommendation']}")
            print(f"Last known ({res['historical_last_date']}): {res['historical_last_value']:.2f}  "
                  f"-> Day {horizon} forecast: {res['forecast'][-1]['predicted_freight_rate']:.2f}")
        except ArtifactsNotFoundError as exc:
            print(exc)
            break
