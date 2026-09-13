from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def _numeric_and_categorical_columns(
    df: pd.DataFrame,
) -> Dict[str, List[str]]:
    numeric_cols = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_cols = [
        c for c in numeric_cols
        if df[c].dtype != bool
    ]

    categorical_cols = [
        c
        for c in df.columns
        if c not in numeric_cols
        and df[c].dtype != "datetime64[ns]"
    ]

    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }


def _date_range(
    df: pd.DataFrame,
    date_col: Optional[str],
) -> Dict[str, Optional[str]]:
    if not date_col or date_col not in df.columns:
        return {
            "start": None,
            "end": None,
        }

    parsed = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    if parsed.notna().sum() == 0:
        return {
            "start": None,
            "end": None,
        }

    return {
        "start": str(parsed.min().date()),
        "end": str(parsed.max().date()),
    }


def generate_quality_report(
    df: Optional[pd.DataFrame],
    dataset_name: str,
    date_col: Optional[str] = None,
    outlier_column: Optional[str] = None,
    cleaning_stats: Optional[Dict] = None,
) -> Dict:
    if df is None:
        return {
            "dataset": dataset_name,
            "available": False,
            "rows": 0,
            "columns": 0,
            "missing_values_total": 0,
            "missing_values_by_column": {},
            "duplicate_rows": 0,
            "numeric_columns": [],
            "categorical_columns": [],
            "date_range": {
                "start": None,
                "end": None,
            },
            "outlier_count": 0,
        }

    col_kinds = _numeric_and_categorical_columns(df)

    missing_by_col = df.isna().sum()

    missing_by_col = {
        k: int(v)
        for k, v in missing_by_col.items()
        if v > 0
    }

    outlier_count = 0

    if (
        outlier_column
        and outlier_column in df.columns
    ):
        outlier_count = int(
            df[outlier_column].sum()
        )

    elif (
        cleaning_stats
        and isinstance(
            cleaning_stats.get("outliers"),
            dict,
        )
    ):
        outlier_count = int(
            cleaning_stats["outliers"].get(
                "count",
                0,
            )
            or 0
        )

    report: Dict = {
        "dataset": dataset_name,
        "available": True,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values_total": int(
            df.isna().sum().sum()
        ),
        "missing_values_by_column": missing_by_col,
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "numeric_columns": col_kinds[
            "numeric_columns"
        ],
        "categorical_columns": col_kinds[
            "categorical_columns"
        ],
        "date_range": _date_range(
            df,
            date_col,
        ),
        "outlier_count": outlier_count,
    }

    if cleaning_stats:
        report["cleaning_stats"] = cleaning_stats

    return report


def generate_full_report(
    reports: Dict[str, Dict],
) -> pd.DataFrame:
    rows = []

    for name, r in reports.items():
        rows.append({
            "dataset": name,
            "available": r.get(
                "available",
                False,
            ),
            "rows": r.get(
                "rows",
                0,
            ),
            "columns": r.get(
                "columns",
                0,
            ),
            "missing_values_total": r.get(
                "missing_values_total",
                0,
            ),
            "duplicate_rows": r.get(
                "duplicate_rows",
                0,
            ),
            "numeric_columns": len(
                r.get(
                    "numeric_columns",
                    [],
                )
            ),
            "categorical_columns": len(
                r.get(
                    "categorical_columns",
                    [],
                )
            ),
            "date_start": r.get(
                "date_range",
                {},
            ).get("start"),
            "date_end": r.get(
                "date_range",
                {},
            ).get("end"),
            "outlier_count": r.get(
                "outlier_count",
                0,
            ),
        })

    return pd.DataFrame(rows)