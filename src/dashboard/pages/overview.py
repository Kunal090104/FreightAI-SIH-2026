"""Dashboard (overview) page — the at-a-glance summary + Final Recommendation Card."""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from src.dashboard.html_components import recommendation_card
from src.dashboard.state import (
    congestion_data_status,
    forecast_data_status,
    overall_data_status,
    port_data_status,
    routes_data_status,
    vessel_data_status,
    weather_data_status,
)
from src.dashboard.ui_helpers import (
    render_error,
    render_level_pill,
    render_status_badge,
)


def _metric_card(
    label: str,
    value: str,
    sub: Optional[str] = None,
) -> None:
    """Render a dashboard metric using Streamlit's native metric component."""

    st.metric(
        label,
        value,
        help=sub,
    )


def render(analysis: Optional[Dict]) -> None:
    st.title("FreightAI")
    st.caption("AI-Powered Bulk Freight & Chartering Intelligence")

    if analysis is None:
        st.info(
            "Fill in the Cargo and Contract details in the sidebar, "
            "then click **Run Analysis** to begin."
        )
        return

    render_status_badge(
        overall_data_status(analysis),
        prefix="Overall data",
    )

    st.divider()

    forecast = analysis["forecast"]["result"]
    vessel = analysis["vessel"]["result"]
    port = analysis["port"]["result"]
    risk = analysis["risk"]["result"]
    charter = analysis["charter"]["result"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Current Freight")

        if forecast:
            _metric_card(
                "Current BDI-based rate",
                f"{forecast['historical_last_value']:,.1f}",
                f"as of {forecast['historical_last_date']}",
            )
        else:
            render_error(
                analysis["forecast"]["error"],
                "Train the forecasting model: "
                "`python -m src.forecasting.train`",
            )

    with col2:
        st.subheader("Predicted Freight")

        if forecast:
            last_point = forecast["forecast"][-1]

            _metric_card(
                f"{forecast['horizon_days']}-day forecast",
                f"{last_point['predicted_freight_rate']:,.1f}",
                (
                    "Experimental"
                    if forecast["experimental"]
                    else "Within supported range"
                ),
            )
        else:
            st.write("—")

    with col3:
        st.subheader("Market Trend")

        if forecast:
            trend_icon = {
                "INCREASING": "📈",
                "STABLE": "➖",
                "DECREASING": "📉",
            }.get(
                forecast["market_signal"],
                "",
            )

            st.markdown(
                f"### {trend_icon} {forecast['market_signal']}"
            )
        else:
            st.write("—")

    st.divider()

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        with st.container(border=True):
            st.subheader("Market Signal")

            if forecast:
                st.markdown(
                    f"**{forecast['recommendation']}**"
                )
            else:
                st.write("—")

    with col5:
        with st.container(border=True):
            st.subheader("Recommended Vessel")

            if vessel:
                st.markdown(
                    f"### {vessel['recommended_vessel_type']}"
                )

                render_status_badge(
                    vessel_data_status(vessel)
                )
            else:
                st.write("—")

    with col6:
        with st.container(border=True):
            st.subheader("Port Compatibility")

            if port:
                if port["compatible"]:
                    st.markdown("✅ **Compatible**")
                else:
                    st.markdown("❌ **Not Compatible**")
            else:
                render_error(
                    analysis["port"]["error"],
                    None,
                )

    with col7:
        with st.container(border=True):
            st.subheader("Risk")

            render_level_pill(
                risk["overall_risk"] if risk else None,
                "Overall Risk",
            )

    st.subheader("Recommended Charter")

    if charter:
        st.markdown(
            f"### {charter['recommended_strategy_display']}"
        )
        st.caption(charter["summary_reason"])
    else:
        render_error(
            analysis["charter"]["error"],
            None,
        )

    st.divider()

    st.subheader("Data sources behind this analysis")

    badges = {
        "Forecast model": forecast_data_status(),
        "Vessel fleet": vessel_data_status(vessel),
        "Port limits": port_data_status(port),
        "Route distance": routes_data_status(
            analysis["voyage"]["result"]
        ),
        "Weather": weather_data_status(risk),
        "Congestion": congestion_data_status(risk),
    }

    cols = st.columns(len(badges))

    for col, (label, status) in zip(
        cols,
        badges.items(),
    ):
        with col:
            st.caption(label)
            render_status_badge(status)

    st.divider()

    _render_final_recommendation_card(analysis)


def _render_final_recommendation_card(
    analysis: Dict,
) -> None:
    """Render the final FreightAI recommendation."""

    forecast = analysis["forecast"]["result"]
    vessel = analysis["vessel"]["result"]
    port = analysis["port"]["result"]
    risk = analysis["risk"]["result"]
    charter = analysis["charter"]["result"]
    inputs = analysis["inputs"]

    rows = [
        (
            "Market Entry",
            (
                forecast["recommendation"]
                if forecast
                else "Unavailable — forecasting model not trained"
            ),
        ),
        (
            "Vessel",
            (
                vessel["recommended_vessel_type"]
                if vessel
                else "Unavailable"
            ),
        ),
        (
            "Destination",
            inputs.destination_port,
        ),
        (
            "Port Compatibility",
            (
                (
                    "Compatible"
                    if port["compatible"]
                    else "Not Compatible"
                )
                if port
                else "Unavailable"
            ),
        ),
        (
            "Risk",
            (
                risk["overall_risk"]
                if risk
                else "Unavailable"
            ),
        ),
        (
            "Charter Strategy",
            (
                charter["recommended_strategy_display"]
                if charter
                else "Unavailable"
            ),
        ),
        (
            "Reason",
            (
                charter["summary_reason"]
                if charter
                else (
                    "Insufficient data to generate a "
                    "recommendation reason."
                )
            ),
        ),
    ]

    st.markdown(
        recommendation_card(rows),
        unsafe_allow_html=True,
    )

    if (
        port is not None
        and not port["compatible"]
        and vessel
        and vessel.get("alternative_vessel_type")
    ):
        st.info(
            f"Note: the recommended vessel isn't "
            f"port-compatible at {inputs.destination_port} — "
            f"consider **{vessel['alternative_vessel_type']}** instead."
        )