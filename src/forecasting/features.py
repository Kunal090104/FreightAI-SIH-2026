"""
features.py
============
Feature engineering shared by ``train.py`` (historical feature frame) and
``predict.py`` (recursive multi-step forecasting). Keeping this logic in one
place guarantees the features used at prediction time are computed exactly
the same way as at training time.

Leakage rule enforced throughout: every feature for the row that predicts
``freight_rate`` at time *t* is built only from values strictly before *t*
(lags and rolling stats are computed on the series shifted by one step
before any rolling window is applied).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# Canonical, ordered list of model input features. This exact order is
# persisted to models/feature_columns.json so predict.py can reconstruct
# feature vectors identically regardless of dict ordering elsewhere.
FEATURE_COLUMNS: List[str] = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_30",
    "month",
    "quarter",
    "day_of_week",
]

TARGET_COLUMN = "freight_rate"
DATE_COLUMN = "date"

# Longest lookback any feature needs (lag_30 / rolling_mean_30) — this is
# how much trailing history predict.py must keep available before it can
# compute a full, non-NaN feature row.
MIN_HISTORY_REQUIRED = 30


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full historical feature frame from a clean freight dataframe.

    Parameters
    ----------
    df:
        Must contain ``date`` and ``freight_rate`` columns (the standardized
        schema produced by the data-pipeline module).

    Returns
    -------
    A dataframe sorted chronologically, with all columns in
    :data:`FEATURE_COLUMNS` plus ``date`` and ``freight_rate``, with the
    leading rows that don't have enough history (first ~30) dropped.
    """
    missing = {DATE_COLUMN, TARGET_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(
            f"Input dataframe is missing required column(s): {sorted(missing)}. "
            "Expected the standardized schema from the data pipeline module "
            "(columns: date, freight_rate)."
        )

    out = df[[DATE_COLUMN, TARGET_COLUMN]].copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
    out = out.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)

    series = out[TARGET_COLUMN]

    # Lags: value 1 / 7 / 14 / 30 steps back — strictly historical by construction.
    out["lag_1"] = series.shift(1)
    out["lag_7"] = series.shift(7)
    out["lag_14"] = series.shift(14)
    out["lag_30"] = series.shift(30)

    # Rolling stats: computed on the series shifted by 1 BEFORE rolling, so
    # the window for row t is [t-7 .. t-1], never including t itself.
    shifted = series.shift(1)
    out["rolling_mean_7"] = shifted.rolling(window=7).mean()
    out["rolling_mean_14"] = shifted.rolling(window=14).mean()
    out["rolling_mean_30"] = shifted.rolling(window=30).mean()
    out["rolling_std_7"] = shifted.rolling(window=7).std()
    out["rolling_std_30"] = shifted.rolling(window=30).std()

    # Calendar features derived from the target date itself (not leakage —
    # the calendar date of the row being predicted is always known in advance).
    out["month"] = out[DATE_COLUMN].dt.month
    out["quarter"] = out[DATE_COLUMN].dt.quarter
    out["day_of_week"] = out[DATE_COLUMN].dt.dayofweek

    out = out.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return out


def infer_step_frequency(dates: pd.Series) -> pd.Timedelta:
    """Infer the typical spacing between consecutive observations (e.g. 1 day
    for daily BDI data, or a longer gap if the series only has business-day
    or weekly points). Falls back to 1 day if it can't be determined.
    """
    dates = pd.to_datetime(pd.Series(dates)).sort_values().reset_index(drop=True)
    if len(dates) < 2:
        return pd.Timedelta(days=1)
    diffs = dates.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(days=1)
    median_diff = diffs.median()
    return median_diff if median_diff > pd.Timedelta(0) else pd.Timedelta(days=1)


def build_feature_row(history: pd.Series, target_date: pd.Timestamp) -> Dict[str, float]:
    """Build one feature row (as a plain dict, in :data:`FEATURE_COLUMNS`
    order) for predicting ``freight_rate`` at ``target_date``, given
    ``history`` — a chronologically-ordered ``pd.Series`` of known/previously
    -predicted freight_rate values, indexed up to (but not including)
    ``target_date``.

    This is the recursive-forecasting counterpart to
    :func:`build_feature_frame`'s row-wise logic: it must compute features
    identically, using only values already in ``history``.
    """
    values = history.to_numpy(dtype=float)
    n = len(values)

    def _lag(k: int) -> float:
        return float(values[-k]) if n >= k else np.nan

    def _rolling_mean(window: int) -> float:
        w = values[-window:] if n >= window else values
        return float(np.mean(w)) if len(w) > 0 else np.nan

    def _rolling_std(window: int) -> float:
        w = values[-window:] if n >= window else values
        return float(np.std(w, ddof=1)) if len(w) > 1 else 0.0

    row = {
        "lag_1": _lag(1),
        "lag_7": _lag(7),
        "lag_14": _lag(14),
        "lag_30": _lag(30),
        "rolling_mean_7": _rolling_mean(7),
        "rolling_mean_14": _rolling_mean(14),
        "rolling_mean_30": _rolling_mean(30),
        "rolling_std_7": _rolling_std(7),
        "rolling_std_30": _rolling_std(30),
        "month": int(target_date.month),
        "quarter": int(target_date.quarter),
        "day_of_week": int(target_date.dayofweek),
    }
    return row
