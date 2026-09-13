"""Risk Analysis page — Market/Port/Weather/Operational + Overall risk,
with plain-English reasons behind every score, from risk_engine."""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from src.dashboard.ui_helpers import render_engine_result_or_error, render_level_pill

CATEGORY_LABELS = {
    "market_risk": "Market Risk",
    "port_risk": "Port Risk",
    "weather_risk": "Weather Risk",
    "operational_risk": "Operational Risk",
}


def _render_category(key: str, category: Dict) -> None:
    with st.container(border=True):
        render_level_pill(category["level"], CATEGORY_LABELS[key])
        if category["score"] is not None:
            st.caption(f"Score: {category['score']:.1f} / 100")
        st.markdown("**Reasons:**")
        for reason in category["reasons"]:
            st.write(f"- {reason}")


def render(analysis: Optional[Dict]) -> None:
    st.title("Risk Analysis")

    if analysis is None:
        st.info("Run an analysis from the sidebar to see the risk assessment.")
        return

    risk_slot = analysis["risk"]
    if not render_engine_result_or_error(risk_slot, "A vessel recommendation is required before risk can be assessed."):
        return

    result = risk_slot["result"]

    st.markdown("### Overall Risk")
    render_level_pill(result["overall_risk"], "Overall Risk")
    if result["risk_score"] is not None:
        st.caption(f"Combined score: {result['risk_score']:.1f} / 100")
    if result["excluded_categories"]:
        st.caption(f"Excluded from overall score (no usable data): {', '.join(result['excluded_categories'])}")

    st.divider()
    cols = st.columns(4)
    for col, key in zip(cols, CATEGORY_LABELS):
        with col:
            _render_category(key, result[key])

    st.divider()
    st.subheader("All reasons")
    for reason in result["risk_reasons"]:
        st.write(f"- {reason}")

    st.caption(result["disclaimer"])
