"""
train.py
========
Trains and selects the freight-rate forecasting model.

Pipeline: load ``data/processed/freight_clean.csv`` -> engineer features
(``features.build_feature_frame``) -> chronological 70/15/15 split -> train
every candidate model -> evaluate on train/validation/test
(``evaluation.py``) -> select the best model by validation MAE -> persist
the model, its metrics, and its feature-column contract to ``models/``.

Run directly with:
    python -m src.forecasting.train
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.forecasting.evaluation import evaluate_predictions, select_best_model
from src.forecasting.features import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_frame
from src.forecasting.models import MovingAverageBaseline, TENSORFLOW_AVAILABLE

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

if TENSORFLOW_AVAILABLE:
    from src.forecasting.models import LSTMTabularRegressor

logger = logging.getLogger("freightai.forecasting.train")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "freight_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "freight_model.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 is TEST_FRAC

# LSTM is only trained if there are at least this many training rows —
# otherwise it's skipped rather than fit on data too thin to justify it.
MIN_LSTM_TRAIN_SAMPLES = 500

# A dataset this small can't support a meaningful chronological 70/15/15
# split with a 30-day feature lookback baked in.
MIN_ROWS_FOR_TRAINING = 60


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed_freight(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned freight dataset produced by the data pipeline module.

    Raises a clear, actionable error (rather than a raw pandas traceback) if
    the file is missing or doesn't have the expected standardized schema.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{path}'. Run the data pipeline module first "
            "(e.g. `python -m src.data.pipeline`) to produce freight_clean.csv."
        )
    df = pd.read_csv(path)
    missing = {"date", "freight_rate"} - set(df.columns)
    if missing:
        raise ValueError(
            f"'{path}' is missing expected column(s) {sorted(missing)}. "
            "Expected the standardized schema from the data pipeline module "
            "(columns: date, freight_rate)."
        )
    return df


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a chronologically-sorted dataframe into train/validation/test
    by row position — never randomly. The dataframe must already be sorted
    by date (``build_feature_frame`` guarantees this).
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

def get_candidate_models(n_train_samples: int) -> Tuple[Dict[str, object], Dict[str, str]]:
    """Build the dict of candidate models to train, plus a dict of reasons
    for any model that was intentionally skipped (missing dependency, or
    not enough data to justify it).
    """
    models: Dict[str, object] = {
        "moving_average_baseline": MovingAverageBaseline(feature_name="rolling_mean_7"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
    }
    skip_notes: Dict[str, str] = {}

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
        )
    else:
        skip_notes["xgboost"] = "xgboost is not installed — skipped ('XGBoost if available')."

    if TENSORFLOW_AVAILABLE and n_train_samples >= MIN_LSTM_TRAIN_SAMPLES:
        models["lstm"] = LSTMTabularRegressor(n_features=len(FEATURE_COLUMNS))
    elif not TENSORFLOW_AVAILABLE:
        skip_notes["lstm"] = "tensorflow/keras is not installed — skipped."
    else:
        skip_notes["lstm"] = (
            f"insufficient training data ({n_train_samples} rows < "
            f"{MIN_LSTM_TRAIN_SAMPLES} minimum) — skipped rather than used for appearance."
        )

    return models, skip_notes


# ---------------------------------------------------------------------------
# Training + evaluation orchestration
# ---------------------------------------------------------------------------

def train_and_select_model(data_path: Path = PROCESSED_DATA_PATH) -> Dict:
    """Run the full training pipeline and return a results dict:
        {
            "metrics": {model_name: {"train": {...}, "validation": {...}, "test": {...}}},
            "skipped": {model_name: reason},
            "selected_model": model_name,
            "models": {model_name: fitted_model_object},
            "split_sizes": {"train": n, "validation": n, "test": n},
        }
    Does not save anything to disk — see :func:`run_training_pipeline` for that.
    """
    raw_df = load_processed_freight(data_path)
    feature_df = build_feature_frame(raw_df)

    if len(feature_df) < MIN_ROWS_FOR_TRAINING:
        raise ValueError(
            f"Only {len(feature_df)} usable rows remain after feature engineering "
            f"(need >= {MIN_ROWS_FOR_TRAINING} for a meaningful 70/15/15 split). "
            "Provide more historical freight data."
        )

    train_df, val_df, test_df = chronological_split(feature_df)
    logger.info(
        "Chronological split -> train=%d, validation=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    candidate_models, skip_notes = get_candidate_models(n_train_samples=len(train_df))

    metrics: Dict[str, Dict] = {}
    fitted_models: Dict[str, object] = {}

    for name, model in candidate_models.items():
        try:
            logger.info("Training model: %s", name)
            model.fit(X_train, y_train)
            metrics[name] = {
                "train": evaluate_predictions(y_train, model.predict(X_train)),
                "validation": evaluate_predictions(y_val, model.predict(X_val)),
                "test": evaluate_predictions(y_test, model.predict(X_test)),
            }
            fitted_models[name] = model
        except Exception as exc:  # noqa: BLE001 - one bad model must not kill the run
            logger.error("Model '%s' failed to train/evaluate: %s", name, exc)
            skip_notes[name] = f"failed during training/evaluation: {exc}"

    if not metrics:
        raise RuntimeError("Every candidate model failed to train. See logs above for details.")

    selected_model = select_best_model(metrics, split="validation", metric="mae")
    logger.info("Selected model (best validation MAE): %s", selected_model)

    return {
        "metrics": metrics,
        "skipped": skip_notes,
        "selected_model": selected_model,
        "models": fitted_models,
        "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifacts(results: Dict, models_dir: Path = MODELS_DIR) -> None:
    """Persist the selected model, full metrics comparison, and the feature
    column contract to ``models/``.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    selected_name = results["selected_model"]
    selected_model = results["models"][selected_name]
    joblib.dump(selected_model, MODEL_PATH)
    logger.info("Saved selected model ('%s') -> %s", selected_name, MODEL_PATH)

    metrics_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_name,
        "selection_criterion": "lowest validation MAE",
        "split_sizes": results["split_sizes"],
        "metrics": results["metrics"],
        "skipped_models": results["skipped"],
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2, default=str))
    logger.info("Saved metrics -> %s", METRICS_PATH)

    feature_payload = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "date_column": "date",
    }
    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_payload, indent=2))
    logger.info("Saved feature column contract -> %s", FEATURE_COLUMNS_PATH)


def run_training_pipeline(data_path: Path = PROCESSED_DATA_PATH, models_dir: Path = MODELS_DIR) -> Dict:
    """Full entry point: train, select, save. Returns the same results dict
    as :func:`train_and_select_model` (without the fitted model objects, to
    keep the return value JSON-friendly for callers that don't need them).
    """
    results = train_and_select_model(data_path)
    save_artifacts(results, models_dir=models_dir)
    return {
        "selected_model": results["selected_model"],
        "metrics": results["metrics"],
        "skipped": results["skipped"],
        "split_sizes": results["split_sizes"],
    }


if __name__ == "__main__":
    summary = run_training_pipeline()
    print("\n=== FreightAI Forecasting — Training Summary ===")
    print(f"Selected model: {summary['selected_model']}")
    print(f"Split sizes: {summary['split_sizes']}")
    print("\nValidation metrics by model:")
    for name, m in summary["metrics"].items():
        print(f"  {name:28s} MAE={m['validation']['mae']:.3f}  "
              f"RMSE={m['validation']['rmse']:.3f}  "
              f"MAPE={m['validation']['mape']}  R2={m['validation']['r2']}")
    if summary["skipped"]:
        print("\nSkipped models:")
        for name, reason in summary["skipped"].items():
            print(f"  {name}: {reason}")
