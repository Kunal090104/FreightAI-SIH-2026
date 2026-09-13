"""Vessel Recommendation page — per-class comparison table with the
recommended vessel highlighted, sourced entirely from vessel_optimizer."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from src.dashboard.state import vessel_data_status
from src.dashboard.ui_helpers import (
    plot_bar_scores, render_engine_result_or_error, render_status_badge,
)


def _spec_caption(spec: Dict) -> str:
    source_note = "fleet data" if spec.get("data_source") == "fleet_data" else "reference spec"
    if spec.get("data_source") == "fleet_data":
        source_note += f", {spec.get('n_vessels')} vessel(s)"
    return (
        f"DWT {spec.get('dwt'):,.0f} t · LOA {spec.get('loa')} m · Beam {spec.get('beam')} m · "
        f"Draft {spec.get('draft')} m · Speed {spec.get('speed')} kn ({source_note})"
    )


def render(analysis: Optional[Dict]) -> None:
    st.title("Vessel Recommendation")

    if analysis is None:
        st.info("Run an analysis from the sidebar to see vessel recommendations.")
        return

    vessel_slot = analysis["vessel"]
    if not render_engine_result_or_error(vessel_slot, "Check that cargo quantity and ports are valid."):
        return

    result = vessel_slot["result"]
    render_status_badge(vessel_data_status(result))
    st.divider()

    recommended = result["recommended_vessel_type"]
    st.markdown(f"### Recommended: **{recommended}**")
    if not result["recommended_compatible"]:
        st.warning(
            f"{recommended} is not port-compatible for this route. "
            + (f"Consider **{result['alternative_vessel_type']}** instead." if result["alternative_vessel_type"]
               else "No evaluated class is fully port-compatible for this route.")
        )
    for reason in result["reasons"]:
        st.caption(f"• {reason}")

    st.divider()
    st.subheader("Vessel Type Comparison")

    rows = []
    for cls, ev in result["evaluations"].items():
        comps = ev["components"]
        rows.append({
            "Vessel Type": cls,
            "Cargo Fit": comps["cargo_capacity"]["score"],
            "Port Fit": comps["port_compatibility"]["score"],
            "Risk": comps["operational_risk"]["score"],
            "Score": ev["suitability_score"],
        })
    df = pd.DataFrame(rows).set_index("Vessel Type")

    def _highlight(row: pd.Series) -> list:
        color = "background-color:#16a34a22;font-weight:600;" if row.name == recommended else ""
        return [color] * len(row)

    st.dataframe(df.style.apply(_highlight, axis=1), use_container_width=True)
    st.plotly_chart(
        plot_bar_scores(list(df.index), df["Score"].tolist(), "Suitability Score by Vessel Type", highlight=recommended),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Details by vessel type")
    for cls, ev in result["evaluations"].items():
        with st.expander(f"{cls}{' (recommended)' if cls == recommended else ''} — score {ev['suitability_score']}"):
            st.caption(_spec_caption(ev["vessel_spec"]))
            for reason in ev["reasons"]:
                st.write(f"- {reason}")

    st.caption(result["disclaimer"])
