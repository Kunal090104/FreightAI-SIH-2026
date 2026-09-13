"""Charter Strategy page — compares Spot / Short-Term / Medium-Term / COA
on score plus the qualitative characteristics from the original strategy
definitions (flexibility, price protection, market exposure, commitment),
then shows charter_optimizer's final recommendation."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from src.dashboard.html_components import decision_factors_card
from src.dashboard.ui_helpers import plot_bar_scores, render_engine_result_or_error
from src.optimization.charter_optimizer import STRATEGY_DISPLAY_NAMES

# Qualitative profile per strategy — presentation labels only, mirroring the
# same structural characterization documented in charter_optimizer.py's own
# STRATEGY_PROS_CONS (these describe what each strategy type IS, not a
# per-user-input computation, so they're shown as fixed reference labels
# alongside the engine's actual computed score for this specific case).
QUALITATIVE_PROFILE = {
    "SPOT": {"Flexibility": "High", "Price Protection": "None", "Market Exposure": "High", "Commitment": "Low"},
    "SHORT_TERM": {"Flexibility": "Moderate", "Price Protection": "Moderate", "Market Exposure": "Moderate", "Commitment": "Moderate"},
    "MEDIUM_TERM": {"Flexibility": "Low", "Price Protection": "High", "Market Exposure": "Low", "Commitment": "High"},
    "COA": {"Flexibility": "Very Low", "Price Protection": "High", "Market Exposure": "Low", "Commitment": "Very High"},
}


def render(analysis: Optional[Dict]) -> None:
    st.title("Charter Strategy")

    if analysis is None:
        st.info("Run an analysis from the sidebar to see the charter strategy comparison.")
        return

    charter_slot = analysis["charter"]
    if not render_engine_result_or_error(charter_slot, "Charter strategy needs cargo/contract inputs at minimum."):
        return

    result = charter_slot["result"]
    recommended = result["recommended_strategy"]

    st.subheader("Strategy Comparison")
    rows = []
    for key, display in STRATEGY_DISPLAY_NAMES.items():
        profile = QUALITATIVE_PROFILE[key]
        rows.append({
            "Strategy": display,
            "Score": result["scores"][key],
            "Flexibility": profile["Flexibility"],
            "Price Protection": profile["Price Protection"],
            "Market Exposure": profile["Market Exposure"],
            "Commitment": profile["Commitment"],
        })
    df = pd.DataFrame(rows).set_index("Strategy")

    def _highlight(row: pd.Series) -> list:
        color = "background-color:#16a34a22;font-weight:600;" if row.name == STRATEGY_DISPLAY_NAMES[recommended] else ""
        return [color] * len(row)

    st.dataframe(df.style.apply(_highlight, axis=1), use_container_width=True)
    st.caption("Flexibility/Price Protection/Market Exposure/Commitment are structural characteristics of each "
               "strategy type (prototype reference labels); Score is computed for this specific case.")

    st.plotly_chart(
        plot_bar_scores(list(df.index), df["Score"].tolist(), "Charter Strategy Scores",
                         highlight=STRATEGY_DISPLAY_NAMES[recommended]),
        use_container_width=True,
    )

    st.divider()
    st.markdown(f"### Final Recommendation: {result['recommended_strategy_display']}")
    st.markdown(f"**Confidence:** {result['confidence']}")
    st.info(result["summary_reason"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Advantages**")
        for a in result["advantages"]:
            st.write(f"✅ {a}")
    with col2:
        st.markdown("**Disadvantages**")
        for d in result["disadvantages"]:
            st.write(f"⚠️ {d}")

    with st.expander("Full reasoning for every strategy"):
        for key, display in STRATEGY_DISPLAY_NAMES.items():
            st.markdown(f"**{display}** ({result['scores'][key]})")
            for reason in result["reasons"][key]:
                st.write(f"- {reason}")

    factors = {
        k: v
        for k, v in result["decision_factors"].items()
        if k != "vessel_incompatibility_escalation"
    }

    st.markdown(
        decision_factors_card(factors),
        unsafe_allow_html=True,
    )

    st.caption(result["disclaimer"])
