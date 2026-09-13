"""
loader.py
=========
Data-loading and dataset-discovery utilities for the FreightAI pipeline.

Responsibilities of this module (and only this module):
    * Locate raw CSV files under ``data/raw/`` for each known FreightAI dataset.
    * Load a CSV safely, without crashing if the file is missing, empty, or malformed.
    * Auto-detect the date column and the primary numeric "value" column for a
      dataset (this is used heavily for the freight/BDI dataset, whose Kaggle
      filename and column names are not standardized).
    * Standardize column names (snake_case) and, for the freight dataset,
      rename the detected columns to the canonical ``date`` / ``freight_rate``
      schema.

Nothing in this module cleans, imputes, or transforms *values* — that is the
job of ``preprocessing.py``. Keeping the two concerns separate is what makes
this pipeline reusable by later FreightAI modules (dashboard, ML forecasting,
vessel/port engines, etc.).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("freightai.loader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# src/data/loader.py -> parents[2] is the FreightAI/ project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# The seven datasets this module supports. Any of them may legitimately be
# absent from data/raw/ — the pipeline must keep working regardless.
DATASET_KEYS: List[str] = [
    "freight",
    "vessels",
    "ports",
    "commodities",
    "weather",
    "economic",
    "congestion",
    "routes",
]

# Keywords used to *discover* a dataset's file when it isn't named exactly
# "<key>.csv" (this matters most for "freight", since Kaggle BDI dumps come
# under all sorts of names, e.g. "Baltic Dry Index Historical Data.csv").
DATASET_FILENAME_KEYWORDS: Dict[str, List[str]] = {
    "freight": ["freight", "bdi", "baltic", "dry_index", "dryindex"],
    "vessels": ["vessel", "ship", "fleet"],
    "ports": ["updatedpub150", "ports", "port_data", "world_port", "wpi"],
    "commodities": ["commodit", "cargo"],
    "weather": ["weather", "climate", "wind", "storm"],
    "economic": ["econom", "gdp", "trade", "macro"],
    "congestion": ["congest", "queue", "waiting", "berth"],
    "routes": ["route", "routes", "distance", "searoute"],
}

# Keywords used for auto-detecting columns inside a dataframe.
DATE_COLUMN_KEYWORDS = ["date", "time", "period", "month", "year", "timestamp"]
FREIGHT_VALUE_COLUMN_KEYWORDS = [
    "bdi", "baltic", "dry_index", "dryindex", "freight_rate", "freight",
    "index", "rate", "price", "value", "close", "closing",
]


# ---------------------------------------------------------------------------
# Column-name standardization
# ---------------------------------------------------------------------------

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with snake_case, whitespace-free column names.

    Example: "Baltic Dry Index" -> "baltic_dry_index", " Close Price " -> "close_price".
    """
    df = df.copy()
    new_cols = []
    for col in df.columns:
        col = str(col).strip()
        col = re.sub(r"[^\w\s]", "", col)          # drop punctuation
        col = re.sub(r"\s+", "_", col)             # spaces -> underscores
        col = col.lower()
        col = re.sub(r"_+", "_", col).strip("_")
        new_cols.append(col if col else "unnamed")
    df.columns = new_cols
    return df


# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------

def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Guess which column of ``df`` holds the date/time value.

    Strategy:
        1. Prefer a column whose *name* contains a date-like keyword.
        2. Otherwise, test every column: try to parse it as a datetime and
           pick the column with the highest successful-parse ratio (as long
           as that ratio clears a reasonable threshold).
    """
    if df.empty:
        return None

    # 1. Name-based match, in keyword priority order.
    lower_cols = {c: c.lower() for c in df.columns}
    for keyword in DATE_COLUMN_KEYWORDS:
        for col, low in lower_cols.items():
            if keyword in low:
                return col

    # 2. Content-based fallback: try parsing each column as a date.
    best_col, best_ratio = None, 0.0
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            continue
        ratio = parsed.notna().mean() if len(parsed) else 0.0
        if ratio > best_ratio:
            best_col, best_ratio = col, ratio

    return best_col if best_ratio >= 0.8 else None


def detect_freight_value_column(df: pd.DataFrame, date_col: Optional[str] = None) -> Optional[str]:
    """Guess which column of ``df`` holds the BDI / freight-rate value.

    Strategy:
        1. Prefer a numeric-*looking* column whose name contains a
           freight/BDI-related keyword, in keyword priority order.
        2. Otherwise, fall back to the numeric column (excluding the date
           column) with the highest variance, on the assumption that the
           "real" index/price series will vary more than an incidental
           numeric column (e.g. an id or a volume flag).
    """
    candidates = [c for c in df.columns if c != date_col]
    if not candidates:
        return None

    def _is_numeric_like(series: pd.Series) -> bool:
        coerced = pd.to_numeric(
            series.astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        return coerced.notna().mean() >= 0.8

    # 1. Keyword match among numeric-like columns.
    lower_cols = {c: c.lower() for c in candidates}
    for keyword in FREIGHT_VALUE_COLUMN_KEYWORDS:
        for col, low in lower_cols.items():
            if keyword in low and _is_numeric_like(df[col]):
                return col

    # 2. Fallback: highest-variance numeric column.
    best_col, best_var = None, -1.0
    for col in candidates:
        coerced = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        if coerced.notna().mean() < 0.5:
            continue
        var = coerced.var()
        if pd.notna(var) and var > best_var:
            best_col, best_var = col, var

    return best_col


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_dataset_file(dataset_key: str, raw_dir: Path = RAW_DATA_DIR) -> Optional[Path]:
    """Locate the raw CSV file for ``dataset_key`` inside ``raw_dir``.

    Resolution order:
        1. Exact filename match: ``<dataset_key>.csv``.
        2. Any ``*.csv`` file whose name contains one of the dataset's known
           keywords (handles arbitrary Kaggle-style filenames).

    Returns ``None`` (never raises) if nothing suitable is found.
    """
    if dataset_key not in DATASET_KEYS:
        raise ValueError(f"Unknown dataset_key '{dataset_key}'. Expected one of {DATASET_KEYS}.")

    if not raw_dir.exists():
        logger.warning("Raw data directory does not exist: %s", raw_dir)
        return None

    exact = raw_dir / f"{dataset_key}.csv"
    if exact.exists():
        return exact

    keywords = DATASET_FILENAME_KEYWORDS.get(dataset_key, [dataset_key])
    for csv_path in sorted(raw_dir.glob("*.csv")):
        name = csv_path.stem.lower()
        if any(kw in name for kw in keywords):
            return csv_path

    return None


# ---------------------------------------------------------------------------
# Safe CSV loading
# ---------------------------------------------------------------------------

def load_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load a CSV file into a DataFrame, never raising on a bad/missing file.

    Returns ``None`` if ``path`` is ``None``, the file doesn't exist, is
    empty, or fails to parse. All failure modes are logged, not raised, so a
    single missing/corrupt dataset never crashes the pipeline.
    """
    if path is None:
        return None
    try:
        if not Path(path).exists():
            logger.warning("File not found: %s", path)
            return None
        df = pd.read_csv(path)
        if df.empty:
            logger.warning("File is empty: %s", path)
            return None
        return df
    except pd.errors.EmptyDataError:
        logger.warning("No data / unparsable CSV: %s", path)
        return None
    except pd.errors.ParserError as exc:
        logger.error("CSV parse error in %s: %s", path, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - deliberately broad: never crash the pipeline
        logger.error("Unexpected error loading %s: %s", path, exc)
        return None


def load_dataset(dataset_key: str, raw_dir: Path = RAW_DATA_DIR) -> Optional[pd.DataFrame]:
    """Discover and load the raw dataset for ``dataset_key``.

    Applies :func:`standardize_column_names` to whatever is loaded. For the
    ``"freight"`` dataset specifically, also auto-detects the date and value
    columns and renames them to the canonical ``date`` / ``freight_rate``
    schema (see :func:`standardize_freight_schema`).

    Returns ``None`` if the dataset file cannot be found or loaded — callers
    must handle that case gracefully (this is intentional: the whole point
    is that a missing dataset does not stop the rest of the pipeline).
    """
    path = find_dataset_file(dataset_key, raw_dir=raw_dir)
    if path is None:
        logger.warning("No raw file found for dataset '%s' in %s", dataset_key, raw_dir)
        return None

    df = load_csv(path)
    if df is None:
        return None

    df = standardize_column_names(df)
    logger.info("Loaded '%s' from %s (%d rows, %d cols)", dataset_key, path.name, *df.shape)

    if dataset_key == "freight":
        df = standardize_freight_schema(df)

    return df


def standardize_freight_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the auto-detected date/value columns of a freight dataframe to
    the canonical schema: ``date``, ``freight_rate``.

    Raises no exception if detection fails for one side; instead logs a
    warning and returns the dataframe with as much standardization as could
    be done. Downstream preprocessing is responsible for validating that the
    required columns actually exist before proceeding.
    """
    date_col = detect_date_column(df)
    value_col = detect_freight_value_column(df, date_col=date_col)

    rename_map = {}
    if date_col is not None:
        rename_map[date_col] = "date"
    else:
        logger.warning("Could not auto-detect a date column in freight dataset.")

    if value_col is not None:
        rename_map[value_col] = "freight_rate"
    else:
        logger.warning("Could not auto-detect a BDI/freight value column in freight dataset.")

    df = df.rename(columns=rename_map)

    keep_cols = [c for c in ["date", "freight_rate"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in keep_cols]
    return df[keep_cols + other_cols]


def load_all_datasets(raw_dir: Path = RAW_DATA_DIR) -> Dict[str, Optional[pd.DataFrame]]:
    """Load every dataset in :data:`DATASET_KEYS`.

    Returns a dict mapping dataset key -> DataFrame (or ``None`` if that
    dataset's raw file was missing/unreadable). Never raises.
    """
    return {key: load_dataset(key, raw_dir=raw_dir) for key in DATASET_KEYS}
