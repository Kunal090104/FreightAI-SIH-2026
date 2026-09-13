from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data.loader import (
    DATASET_KEYS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    load_all_datasets,
)
from src.data.preprocessing import DATASET_DATE_COLUMNS, clean_dataset
from src.data.quality_report import (
    generate_full_report,
    generate_quality_report,
)

logger = logging.getLogger("freightai.pipeline")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )


def save_processed(
    df: Optional[pd.DataFrame],
    dataset_key: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Optional[Path]:
    if df is None or df.empty:
        logger.info(
            "Skipping save for '%s': no cleaned data available",
            dataset_key,
        )
        return None

    try:
        processed_dir.mkdir(parents=True, exist_ok=True)

        out_path = processed_dir / f"{dataset_key}_clean.csv"

        df.to_csv(
            out_path,
            index=False,
        )

        logger.info(
            "Saved cleaned '%s' -> %s (%d rows)",
            dataset_key,
            out_path,
            len(df),
        )

        return out_path

    except Exception as exc:
        logger.error(
            "Failed to save processed data for '%s': %s",
            dataset_key,
            exc,
        )
        return None


def run_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Dict:
    raw = load_all_datasets(raw_dir=raw_dir)

    cleaned: Dict[str, Optional[pd.DataFrame]] = {}
    stats: Dict[str, Dict] = {}
    reports: Dict[str, Dict] = {}
    saved_paths: Dict[str, Optional[Path]] = {}

    for key in DATASET_KEYS:
        raw_df = raw.get(key)

        if raw_df is None:
            logger.warning(
                "Dataset '%s' unavailable — skipping clean/save/report for it.",
                key,
            )

            cleaned[key] = None
            stats[key] = {
                "dataset": key,
                "status": "missing_raw_file",
            }
            reports[key] = generate_quality_report(
                None,
                key,
            )
            saved_paths[key] = None
            continue

        clean_df, clean_stats = clean_dataset(
            key,
            raw_df,
        )

        cleaned[key] = clean_df
        stats[key] = clean_stats

        date_col = DATASET_DATE_COLUMNS.get(key)

        outlier_col = clean_stats.get(
            "outliers",
            {},
        ).get("column")

        outlier_flag_col = (
            f"{outlier_col}_outlier"
            if outlier_col
            else None
        )

        reports[key] = generate_quality_report(
            clean_df,
            key,
            date_col=date_col,
            outlier_column=outlier_flag_col,
            cleaning_stats=clean_stats,
        )

        saved_paths[key] = save_processed(
            clean_df,
            key,
            processed_dir=processed_dir,
        )

    summary = generate_full_report(reports)

    return {
        "cleaned": cleaned,
        "stats": stats,
        "reports": reports,
        "summary": summary,
        "saved_paths": saved_paths,
    }


if __name__ == "__main__":
    result = run_pipeline()

    print("\n=== FreightAI Data Quality Summary ===")
    print(
        result["summary"].to_string(
            index=False,
        )
    )