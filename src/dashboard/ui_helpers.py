"""
ui_helpers.py
=============
Small Streamlit-specific presentation helpers shared across dashboard
pages: data-status badges, a "friendly error" renderer, and Plotly chart
builders. Nothing here computes FreightAI results — that all lives in
``src/dashboard/state.py`` and the underlying engines; this module only
renders what it's given.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

STATUS_COLORS = {
    "REAL": "#16a34a",
    "DEMO": "#f59e0b",
    "MIXED": "#eab308",
    "NO DATA": "#6b7280",
}

STATUS_LABELS = {
    "REAL": "REAL DATA",
    "DEMO": "DEMO DATA",
    "MIXED": "MIXED DATA",
    "NO DATA": "NO DATA AVAILABLE",
}


def status_badge(status: str, prefix: Optional[str] = None) -> str:
    """Return an inline HTML badge string for a REAL/DEMO/MIXED/NO DATA status."""
    color = STATUS_COLORS.get(status, "#6b7280")
    label = STATUS_LABELS.get(status, status)
    text = f"{prefix}: {label}" if prefix else label
    return (
        f'<span style="background-color:{color}22;color:{color};border:1px solid {color};'
        f'padding:2px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;white-space:nowrap;">'
        f'{text}</span>'
    )


def render_status_badge(status: str, prefix: Optional[str] = None) -> None:
    st.markdown(status_badge(status, prefix), unsafe_allow_html=True)


def render_error(message: str, hint: Optional[str] = None) -> None:
    """Consistent, non-crashing error presentation for a failed engine call."""
    st.warning(message)
    if hint:
        st.caption(hint)


def render_engine_result_or_error(slot: Dict, empty_hint: str) -> bool:
    """Given a ``{"result": ..., "error": ...}`` slot from ``state.py``,
    render a warning if it failed and return whether the caller should
    proceed to render the actual result (``True`` only if ``result`` is present).
    """
    if slot is None or slot.get("result") is None:
        render_error(slot.get("error") or "No data available for this section.", empty_hint)
        return False
    return True


def level_color(level: Optional[str]) -> str:
    return {"LOW": "#16a34a", "MEDIUM": "#f59e0b", "HIGH": "#dc2626"}.get(level, "#6b7280")


def render_level_pill(level: Optional[str], label: str) -> None:
    color = level_color(level)
    text = level or "N/A"
    st.markdown(
        f'<div style="border:1px solid {color};border-radius:10px;padding:10px 14px;text-align:center;">'
        f'<div style="font-size:0.78rem;color:#6b7280;">{label}</div>'
        f'<div style="font-size:1.3rem;font-weight:700;color:{color};">{text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def plot_freight_history_and_forecast(
    history_df: Optional[pd.DataFrame],
    forecast_points: Optional[list],
    title: str = "Freight Rate — History & Forecast",
) -> go.Figure:
    """Build a Plotly chart of historical freight rate plus (optionally) an
    overlaid forecast line. Handles either input being missing/empty.
    """
    fig = go.Figure()
    if history_df is not None and not history_df.empty:
        fig.add_trace(go.Scatter(
            x=history_df["date"], y=history_df["freight_rate"],
            mode="lines", name="Historical", line=dict(color="#2563eb"),
        ))
    if forecast_points:
        dates = [p["date"] for p in forecast_points]
        rates = [p["predicted_freight_rate"] for p in forecast_points]
        fig.add_trace(go.Scatter(
            x=dates, y=rates, mode="lines+markers", name="Forecast",
            line=dict(color="#dc2626", dash="dash"),
        ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Freight Rate",
        margin=dict(l=10, r=10, t=40, b=10), height=420, legend=dict(orientation="h", y=1.1),
    )
    return fig


def plot_bar_scores(labels: list, values: list, title: str, highlight: Optional[str] = None) -> go.Figure:
    colors = ["#2563eb" if lab != highlight else "#16a34a" for lab in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="auto"))
    fig.update_layout(title=title, yaxis_title="Score (0-100)", margin=dict(l=10, r=10, t=40, b=10), height=380)
    return fig
