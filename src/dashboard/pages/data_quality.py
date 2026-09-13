from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from src.dashboard.state import load_data_quality_overview
from src.data.pipeline import RAW_DATA_DIR


def render(_analysis: Optional[Dict]) -> None:
    st.title("Data Quality")
    st.caption(
        "Reflects the datasets currently in data/processed/. To (re)generate them, run "
        "`python -m src.data.pipeline` after placing raw files in data/raw/."
    )
    st.divider()

    try:
        df = load_data_quality_overview()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load the data quality overview: {exc}")
        return

    available_count = int(df["available"].sum())
    st.metric("Datasets available", f"{available_count} / {len(df)}")

    display_df = df.copy()
    display_df["available"] = display_df["available"].map({True: "✅", False: "❌"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    missing = df[~df["available"]]["dataset"].tolist()

    if missing:
        st.info(
            f"No processed data found for: {', '.join(missing)}. Place the matching raw CSV(s) in "
            f"`{RAW_DATA_DIR}` and re-run the data pipeline to enable these."
        )

    with st.expander("What counts as REAL vs DEMO data in this dashboard?"):
        st.markdown(
            "- **REAL DATA** — sourced from your processed datasets, a trained model, or a matched "
            "route/port record.\n"
            "- **DEMO DATA** — a clearly-labeled prototype reference value used only when real data "
            "isn't available for that specific field (e.g. typical port dimensions).\n"
            "- **MIXED DATA** — some fields for that section are real, others are reference/default.\n"
            "- **NO DATA AVAILABLE** — the relevant dataset is missing entirely; the dashboard never "
            "fabricates a substitute (this applies especially to weather and route-distance data)."
        )