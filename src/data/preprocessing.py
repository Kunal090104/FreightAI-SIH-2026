from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("freightai.preprocessing")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )


DATASET_DATE_COLUMNS: Dict[str, Optional[str]] = {
    "freight": "date",
    "vessels": None,
    "ports": None,
    "commodities": "date",
    "weather": "date",
    "economic": "date",
    "congestion": "date",
    "routes": None,
}

NON_NEGATIVE_KEYWORDS = [
    "rate",
    "price",
    "speed",
    "wait",
    "queue",
    "draft",
    "loa",
    "beam",
    "dwt",
    "capacity",
    "quantity",
    "volume",
    "count",
    "days",
    "hours",
    "distance",
    "age",
    "tonnage",
    "length",
    "width",
    "depth",
    "duration",
    "index",
    "congestion",
    "berths",
    "cost",
    "fee",
]

NEGATIVE_ALLOWED_KEYWORDS = [
    "temperature",
    "temp",
    "change",
    "growth",
    "delta",
    "anomaly",
    "return",
    "variation",
    "diff",
    "lat",
    "lon",
    "longitude",
    "latitude",
]

MAX_FILLABLE_GAP = 5


def convert_date_column(
    df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    if date_col not in df.columns:
        return df

    df = df.copy()
    before_valid = df[date_col].notna().sum()

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    after_valid = df[date_col].notna().sum()

    if after_valid < before_valid:
        logger.warning(
            "%d value(s) in '%s' could not be parsed as dates and became NaT",
            before_valid - after_valid,
            date_col,
        )

    return df


def sort_chronologically(
    df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    if date_col not in df.columns:
        return df

    return df.sort_values(
        by=date_col,
        kind="mergesort",
    ).reset_index(drop=True)


def remove_duplicate_rows(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    before = len(df)

    df = df.drop_duplicates(
        subset=subset,
        keep="first",
    ).reset_index(drop=True)

    removed = before - len(df)

    if removed:
        logger.info(
            "Removed %d duplicate row(s)",
            removed,
        )

    return df, removed


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    invalid_counts: Dict[str, int] = {}

    for col in columns:
        if col not in df.columns:
            continue

        before_na = df[col].isna().sum()

        cleaned = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(
                r"[^\d\.\-eE]",
                "",
                regex=True,
            )
            .replace("", np.nan)
        )

        df[col] = pd.to_numeric(
            cleaned,
            errors="coerce",
        )

        after_na = df[col].isna().sum()
        newly_invalid = after_na - before_na

        if newly_invalid > 0:
            logger.warning(
                "%d invalid numeric value(s) in '%s' coerced to NaN",
                newly_invalid,
                col,
            )

        invalid_counts[col] = int(newly_invalid)

    return df, invalid_counts


def infer_non_negative_columns(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> List[str]:
    result = []

    for col in numeric_cols:
        low = col.lower()

        if any(k in low for k in NEGATIVE_ALLOWED_KEYWORDS):
            continue

        if any(k in low for k in NON_NEGATIVE_KEYWORDS):
            result.append(col)

    return result


def flag_impossible_negatives(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    negative_counts: Dict[str, int] = {}

    for col in columns:
        if col not in df.columns:
            continue

        mask = df[col] < 0
        n_negative = int(mask.sum())

        negative_counts[col] = n_negative

        if n_negative:
            logger.warning(
                "%d impossible negative value(s) in '%s' set to NaN",
                n_negative,
                col,
            )
            df.loc[mask, col] = np.nan

    return df, negative_counts


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    max_gap: int = MAX_FILLABLE_GAP,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    df = df.copy()

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

    is_time_series = (
        date_col is not None
        and date_col in df.columns
    )

    report: Dict[str, Dict[str, int]] = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue

        missing_before = int(df[col].isna().sum())

        if missing_before == 0:
            report[col] = {
                "missing_before": 0,
                "missing_after": 0,
            }
            continue

        if is_time_series:
            df[col] = df[col].interpolate(
                method="linear",
                limit=max_gap,
                limit_area="inside",
            )

            df[col] = df[col].ffill(
                limit=max_gap,
            )

            df[col] = df[col].bfill(
                limit=max_gap,
            )

        else:
            median_val = df[col].median()

            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)

        missing_after = int(df[col].isna().sum())

        report[col] = {
            "missing_before": missing_before,
            "missing_after": missing_after,
        }

        logger.info(
            "Column '%s': missing %d -> %d after cleaning",
            col,
            missing_before,
            missing_after,
        )

    return df, report


def detect_outliers_iqr(
    series: pd.Series,
    k: float = 1.5,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().sum() < 4:
        return pd.Series(
            False,
            index=series.index,
        )

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0 or pd.isna(iqr):
        return pd.Series(
            False,
            index=series.index,
        )

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return (numeric < lower) | (numeric > upper)


def detect_outliers_rolling(
    series: pd.Series,
    window: int = 30,
    n_std: float = 3.0,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().sum() < window:
        return pd.Series(
            False,
            index=series.index,
        )

    min_periods = max(
        3,
        window // 3,
    )

    rolling_mean = numeric.rolling(
        window=window,
        min_periods=min_periods,
    ).mean()

    rolling_std = numeric.rolling(
        window=window,
        min_periods=min_periods,
    ).std()

    with np.errstate(invalid="ignore"):
        z = (
            numeric - rolling_mean
        ) / rolling_std.replace(
            0,
            np.nan,
        )

    return z.abs() > n_std


def flag_outliers(
    df: pd.DataFrame,
    value_col: str,
    method: str = "iqr",
    **kwargs,
) -> Tuple[pd.DataFrame, int]:
    df = df.copy()
    flag_col = f"{value_col}_outlier"

    if value_col not in df.columns:
        df[flag_col] = False
        return df, 0

    if method == "rolling":
        mask = detect_outliers_rolling(
            df[value_col],
            **kwargs,
        )
    else:
        mask = detect_outliers_iqr(
            df[value_col],
            **kwargs,
        )

    df[flag_col] = mask.fillna(False)
    n_outliers = int(df[flag_col].sum())

    if n_outliers:
        logger.info(
            "Flagged %d outlier(s) in '%s' using '%s' method (not removed)",
            n_outliers,
            value_col,
            method,
        )

    return df, n_outliers


def check_data_types(
    df: pd.DataFrame,
) -> Dict[str, str]:
    return {
        col: str(dtype)
        for col, dtype in df.dtypes.items()
    }


def clean_dataset(
    dataset_key: str,
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    if df is None or df.empty:
        return df, {
            "dataset": dataset_key,
            "status": "skipped_empty_or_missing",
        }

    stats: Dict = {
        "dataset": dataset_key,
        "rows_before": len(df),
    }

    date_col = DATASET_DATE_COLUMNS.get(
        dataset_key
    )

    if date_col and date_col in df.columns:
        df = convert_date_column(
            df,
            date_col,
        )

        df = sort_chronologically(
            df,
            date_col,
        )

    df, n_dupes = remove_duplicate_rows(df)
    stats["duplicates_removed"] = n_dupes

    numeric_cols = [
        c
        for c in df.select_dtypes(
            include=[np.number, "object"]
        ).columns
        if c != date_col
    ]

    already_numeric = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if (
        dataset_key == "freight"
        and "freight_rate" in df.columns
        and "freight_rate" not in already_numeric
    ):
        already_numeric.append("freight_rate")

    df, invalid_counts = coerce_numeric_columns(
        df,
        already_numeric,
    )

    stats["invalid_numeric_values"] = invalid_counts

    non_negative_cols = infer_non_negative_columns(
        df,
        already_numeric,
    )

    df, negative_counts = flag_impossible_negatives(
        df,
        non_negative_cols,
    )

    stats["impossible_negative_values"] = negative_counts

    df, missing_report = handle_missing_values(
        df,
        numeric_cols=already_numeric,
        date_col=date_col,
    )

    stats["missing_values"] = missing_report

    primary_value_col = (
        "freight_rate"
        if "freight_rate" in df.columns
        else (
            already_numeric[0]
            if already_numeric
            else None
        )
    )

    if primary_value_col:
        method = (
            "rolling"
            if date_col
            else "iqr"
        )

        df, n_outliers = flag_outliers(
            df,
            primary_value_col,
            method=method,
        )

        stats["outliers"] = {
            "column": primary_value_col,
            "method": method,
            "count": n_outliers,
        }

    else:
        stats["outliers"] = {
            "column": None,
            "method": None,
            "count": 0,
        }

    stats["dtypes"] = check_data_types(df)
    stats["rows_after"] = len(df)

    return df, stats