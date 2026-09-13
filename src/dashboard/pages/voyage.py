"""Voyage Analysis page — distance, sailing time, port time breakdown,
idle time, and total voyage duration, straight from voyage_calculator."""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from src.dashboard.state import routes_data_status
from src.dashboard.ui_helpers import render_engine_result_or_error, render_status_badge


def _fmt_hours(h: Optional[float]) -> str:
    return f"{h:,.1f} h" if h is not None else "—"


def render(analysis: Optional[Dict]) -> None:
    st.title("Voyage Analysis")

    if analysis is None:
        st.info("Run an analysis from the sidebar to see voyage analysis.")
        return

    voyage_slot = analysis["voyage"]
    if not render_engine_result_or_error(voyage_slot, "A vessel recommendation is required before voyage analysis can run."):
        return

    result = voyage_slot["result"]
    render_status_badge(routes_data_status(result), prefix="Route distance")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if result["distance_nm"] is not None:
            st.metric("Distance", f"{result['distance_nm']:,.0f} nm")
        else:
            st.metric("Distance", "—")
            st.warning(result["distance_status"])
    with col2:
        st.metric("Sailing Time", _fmt_hours(result["sailing_time_hours"]))

    st.divider()
    st.subheader("Port Time Breakdown")
    pt = result["port_time"]
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        st.metric("Waiting", _fmt_hours(pt["waiting_time_hours"]))
    with col4:
        st.metric("Handling", _fmt_hours(pt["cargo_handling_time_hours"]))
    with col5:
        st.metric("Turnaround", _fmt_hours(pt["turnaround_time_hours"]))
    with col6:
        st.metric("Total Port Time", _fmt_hours(pt["total_port_time_hours"]))

    st.divider()
    st.subheader("Idle Time Estimate")
    idle = result["idle_time"]
    col7, col8, col9, col10 = st.columns(4)
    with col7:
        st.metric("Waiting", _fmt_hours(idle["waiting_time_hours"]))
    with col8:
        st.metric("Congestion Delay", _fmt_hours(idle["congestion_delay_hours"]))
    with col9:
        st.metric("Weather Delay", _fmt_hours(idle["weather_delay_hours"]))
    with col10:
        st.metric("Total Idle Time", _fmt_hours(idle["total_estimated_idle_time_hours"]))
    st.caption(idle["note"])

    st.divider()
    st.subheader("Total Voyage Duration")
    if result["estimated_total_voyage_time_hours"] is not None:
        st.markdown(
            f"### {result['estimated_total_voyage_time_hours']:,.1f} hours "
            f"({result['estimated_total_voyage_time_days']:,.1f} days)"
        )
    else:
        st.warning("Total voyage duration unavailable — " + result["distance_status"])

    if result["warnings"]:
        st.subheader("Warnings")
        for w in result["warnings"]:
            st.warning(w)
