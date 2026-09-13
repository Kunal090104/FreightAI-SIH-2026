"""Port Compatibility page — vessel-vs-port parameter table plus port
operations context (congestion, waiting, turnaround, handling capacity)."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from src.dashboard.state import port_data_status
from src.dashboard.html_components import data_sources_card
from src.dashboard.ui_helpers import render_engine_result_or_error, render_status_badge

DISPLAY_PARAMS = ["draft", "loa", "beam", "dwt"]
PARAM_LABELS = {"draft": "Draft", "loa": "LOA", "beam": "Beam", "dwt": "DWT"}
STATUS_ICON = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "UNKNOWN": "❓ UNKNOWN"}


def render(analysis: Optional[Dict]) -> None:
    st.title("Port Compatibility")

    if analysis is None:
        st.info("Run an analysis from the sidebar to see port compatibility.")
        return

    port_slot = analysis["port"]
    if not render_engine_result_or_error(port_slot, "A vessel recommendation is required before port compatibility can be checked."):
        return

    result = port_slot["result"]
    port_spec = result["port"]
    render_status_badge(port_data_status(result))
    st.divider()

    st.markdown(f"### {port_spec['port_name']} — " + ("✅ Compatible" if result["compatible"] else "❌ Not Compatible"))
    if not result["verified"]:
        st.caption("⚠️ One or more limits could not be fully verified — see UNKNOWN rows below.")

    rows = []
    for entry in result["compatibility_table"]:
        if entry["parameter"] in DISPLAY_PARAMS:
            rows.append({
                "Parameter": PARAM_LABELS[entry["parameter"]],
                "Vessel": entry["vessel_value"],
                "Port Limit": entry["port_limit"],
                "Status": STATUS_ICON.get(entry["status"], entry["status"]),
            })
    df = pd.DataFrame(rows).set_index("Parameter")
    st.table(df)

    if result["warnings"]:
        st.subheader("Warnings")
        for w in result["warnings"]:
            st.warning(w)

    st.divider()
    st.subheader("Port Operations Context")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Congestion", port_spec.get("congestion_level", "—"))
    with col2:
        wait = port_spec.get("avg_waiting_hours")
        st.metric("Avg. Waiting Time", f"{wait:.0f} h" if wait is not None else "—")
    with col3:
        turn = port_spec.get("avg_turnaround_hours")
        st.metric("Avg. Turnaround", f"{turn:.0f} h" if turn is not None else "—")
    with col4:
        rate = port_spec.get("cargo_handling_rate")
        st.metric("Handling Capacity", f"{rate:,.0f} t/day" if rate is not None else "—")

    if port_spec.get("tidal_restriction"):
        st.info("This port has tidal access restrictions — berthing/departure windows are tide-dependent.")
    if port_spec.get("note"):
        st.caption(port_spec["note"])


    sources = port_spec.get("field_sources", {})

    st.markdown(
        data_sources_card(sources),
        unsafe_allow_html=True,
    )