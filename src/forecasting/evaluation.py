"""
evaluation.py
=============
Metric computation and model comparison utilities for the freight
forecasting engine. Pure functions only — no training or I/O here, so any
module (training, dashboard, notebooks) can reuse them.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Mean Absolute Percentage Error, as a percentage (0-100+ scale).

    Rows where ``y_true`` is zero are excluded (undefined percentage error).
    Returns ``None`` if every row had a zero true value.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Coefficient of determination (R²). Returns ``None`` when it isn't a
    meaningful statistic (fewer than 2 samples, or zero variance in y_true).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return None
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return None
    return float(1 - ss_res / ss_tot)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute the full metric set (MAE, RMSE, MAPE, R²) for one set of
    predictions, rounded for readability. Used identically for train,
    validation, and test splits so results are directly comparable.
    """
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 4),
        "mape": (
            round(mean_absolute_percentage_error(y_true, y_pred), 4)
            if mean_absolute_percentage_error(y_true, y_pred) is not None else None
        ),
        "r2": (
            round(r_squared(y_true, y_pred), 4)
            if r_squared(y_true, y_pred) is not None else None
        ),
        "n_samples": int(len(np.asarray(y_true))),
    }


def compare_models(
    metrics_by_model: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    split: str = "validation",
    metric: str = "mae",
) -> List[str]:
    """Rank model names by a given metric on a given split (ascending — lower
    is better for mae/rmse/mape). Models missing that split/metric are
    excluded, never crash the comparison.

    Returns model names ordered best-first.
    """
    scored = []
    for name, splits in metrics_by_model.items():
        value = splits.get(split, {}).get(metric)
        if value is not None:
            scored.append((name, value))
    scored.sort(key=lambda pair: pair[1])
    return [name for name, _ in scored]


def select_best_model(
    metrics_by_model: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    split: str = "validation",
    metric: str = "mae",
) -> Optional[str]:
    """Return the single best model name by ``metric`` on ``split``, or
    ``None`` if no model produced a usable score.
    """
    ranked = compare_models(metrics_by_model, split=split, metric=metric)
    return ranked[0] if ranked else None
