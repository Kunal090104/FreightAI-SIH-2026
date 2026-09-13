from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import streamlit as st

from src.dashboard.pages import (
    charter,
    data_quality,
    forecast,
    overview,
    port,
    risk,
    vessel,
    voyage,
)
from src.dashboard.state import UserInputs, run_full_analysis
from src.optimization.port_optimizer import SUPPORTED_PORTS

st.set_page_config(
    page_title="FreightAI",
    page_icon="🚢",
    layout="wide",
)

PAGES = {
    "Dashboard": overview,
    "Freight Forecast": forecast,
    "Vessel Recommendation": vessel,
    "Port Compatibility": port,
    "Voyage Analysis": voyage,
    "Risk Analysis": risk,
    "Charter Strategy": charter,
    "Data Quality": data_quality,
}

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None


def load_css() -> None:
    css_path = Path("assets") / "styles.css"

    if not css_path.exists():
        st.error(f"CSS file not found: {css_path}")
        return

    css = css_path.read_text(encoding="utf-8")

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    st.sidebar.title("🚢 FreightAI")
    st.sidebar.caption("AI-Powered Bulk Freight & Chartering Intelligence")
    st.sidebar.divider()

    with st.sidebar.form("cargo_contract_form"):
        st.markdown("#### Cargo")

        commodity = st.text_input(
            "Commodity",
            value="Iron Ore",
        )

        cargo_quantity = st.number_input(
            "Quantity (metric tons)",
            min_value=1.0,
            value=75_000.0,
            step=1000.0,
        )

        origin_country = st.text_input(
            "Origin country",
            value="Australia",
        )

        origin_port = st.text_input(
            "Origin port",
            value="Port Hedland",
        )

        destination_port = st.selectbox(
            "Destination port",
            options=SUPPORTED_PORTS,
        )

        required_delivery_date = st.date_input(
            "Required delivery date",
            value=date.today() + timedelta(days=30),
        )

        st.markdown("#### Contract")

        contract_duration_months = st.number_input(
            "Contract duration (months)",
            min_value=0.0,
            value=6.0,
            step=1.0,
        )

        expected_number_of_voyages = st.number_input(
            "Expected voyages",
            min_value=1,
            value=1,
            step=1,
        )

        submitted = st.form_submit_button(
            "Run Analysis",
            use_container_width=True,
        )

    if submitted:
        if not origin_port.strip():
            st.sidebar.error("Origin port is required.")

        elif cargo_quantity <= 0:
            st.sidebar.error("Cargo quantity must be positive.")

        else:
            inputs = UserInputs(
                commodity=commodity.strip() or "Unspecified",
                cargo_quantity=float(cargo_quantity),
                origin_country=origin_country.strip(),
                origin_port=origin_port.strip(),
                destination_port=destination_port,
                required_delivery_date=required_delivery_date,
                contract_duration_months=float(contract_duration_months),
                expected_number_of_voyages=int(expected_number_of_voyages),
            )

            try:
                with st.spinner("Running FreightAI analysis..."):
                    st.session_state.analysis = run_full_analysis(inputs)

                st.session_state.analysis_error = None

            except Exception as exc:
                st.session_state.analysis = None
                st.session_state.analysis_error = str(exc)

    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        list(PAGES.keys()),
    )

    return page


def main() -> None:
    load_css()

    page = _render_sidebar()

    if st.session_state.analysis_error:
        st.error(
            f"Analysis failed unexpectedly: "
            f"{st.session_state.analysis_error}"
        )

        st.caption(
            "Try adjusting your inputs, or check the "
            "Data Quality page for missing datasets."
        )

    PAGES[page].render(
        st.session_state.analysis
    )


if __name__ == "__main__":
    main()