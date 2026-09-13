from __future__ import annotations

import json
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from src.dashboard.state import forecast_data_status
from src.dashboard.ui_helpers import (
    plot_freight_history_and_forecast,
    render_error,
    render_status_badge,
)
from src.forecasting.predict import forecast_freight, ArtifactsNotFoundError
from src.forecasting.train import METRICS_PATH, PROCESSED_DATA_PATH


def _load_history() -> Optional[pd.DataFrame]:
    if not PROCESSED_DATA_PATH.exists():
        return None

    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(
            subset=["date", "freight_rate"]
        ).sort_values("date")

    except Exception:
        return None


def render(analysis: Optional[Dict]) -> None:
    st.title("Freight Forecast")
    render_status_badge(forecast_data_status())
    st.divider()

    history = _load_history()

    if history is None:
        render_error(
            "No processed freight history found.",
            "Run the data pipeline first: `python -m src.data.pipeline`",
        )
        return

    st.subheader("Historical Freight Rate")

    st.plotly_chart(
        plot_freight_history_and_forecast(
            history,
            None,
            "Historical Freight Rate",
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Forecast")

    horizons = [7, 30, 90]
    results = {}
    error_msg = None

    for h in horizons:
        try:
            results[h] = forecast_freight(days=h)

        except ArtifactsNotFoundError as exc:
            error_msg = str(exc)
            break

        except Exception as exc:
            error_msg = f"Forecast failed for a {h}-day horizon: {exc}"
            break

    if error_msg:
        render_error(
            error_msg,
            "Train the forecasting model: `python -m src.forecasting.train`",
        )
        return

    default_forecast = results[30]

    st.plotly_chart(
        plot_freight_history_and_forecast(
            history.tail(180),
            default_forecast["forecast"],
            "History (last 180) + 30-Day Forecast",
        ),
        use_container_width=True,
    )

    cols = st.columns(3)

    for col, h in zip(cols, horizons):
        r = results[h]

        with col:
            st.metric(
                f"{h}-day forecast",
                f"{r['forecast'][-1]['predicted_freight_rate']:,.1f}",
                f"{r['market_signal']}"
                + (" (experimental)" if r["experimental"] else ""),
            )

            if r["experimental"]:
                st.caption(f"⚠️ {r['experimental_reason']}")

    st.caption(default_forecast["disclaimer"])

    for limitation in default_forecast["limitations"]:
        st.caption(f"ℹ️ {limitation}")

    st.divider()
    st.subheader("Model Performance")

    if not METRICS_PATH.exists():
        render_error(
            "No model metrics found.",
            "Train the forecasting model: `python -m src.forecasting.train`",
        )
        return

    try:
        metrics = json.loads(METRICS_PATH.read_text())

    except Exception as exc:
        render_error(f"Could not read model metrics: {exc}")
        return

    selected = metrics.get("selected_model")

    st.markdown(
        f"**Selected model:** `{selected}` "
        f"(selection criterion: {metrics.get('selection_criterion')})"
    )

    rows = []

    for name, m in metrics.get("metrics", {}).items():
        val = m.get("validation", {})

        rows.append({
            "model": name,
            "MAE": val.get("mae"),
            "RMSE": val.get("rmse"),
            "MAPE (%)": val.get("mape"),
            "R²": val.get("r2"),
            "selected": "✅" if name == selected else "",
        })

    if rows:
        df = pd.DataFrame(rows).sort_values("MAE")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

    skipped = metrics.get("skipped_models", {})

    if skipped:
        with st.expander("Models not trained"):
            for name, reason in skipped.items():
                st.caption(f"**{name}**: {reason}")