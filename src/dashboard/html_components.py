"""
FreightAI — Reusable HTML Components

This file contains reusable HTML snippets/components used by
the Streamlit dashboard.

Keep presentation HTML here instead of putting large HTML
blocks directly inside app.py or individual pages.
"""

from __future__ import annotations

from textwrap import dedent


def metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    status: str = "",
) -> str:
    """Create a reusable metric card."""

    status_html = ""

    if status:
        status_html = f"""
        <div class="metric-status">
            {status}
        </div>
        """

    subtitle_html = ""

    if subtitle:
        subtitle_html = f"""
        <div class="metric-subtitle">
            {subtitle}
        </div>
        """

    return f"""
    <div class="metric-card">
        <div class="metric-title">
            {title}
        </div>

        <div class="metric-value">
            {value}
        </div>

        {subtitle_html}

        {status_html}
    </div>
    """


def status_badge(
    text: str,
    status: str = "neutral",
) -> str:
    """Create a small status badge."""

    return f"""
    <span class="status-badge status-{status}">
        {text}
    </span>
    """


def section_header(
    title: str,
    subtitle: str = "",
) -> str:
    """Create a reusable section heading."""

    subtitle_html = ""

    if subtitle:
        subtitle_html = f"""
        <div class="section-subtitle">
            {subtitle}
        </div>
        """

    return f"""
    <div class="section-header">
        <div class="section-title">
            {title}
        </div>

        {subtitle_html}
    </div>
    """


def info_card(
    title: str,
    value: str,
    description: str = "",
) -> str:
    """Create a generic information card."""

    description_html = ""

    if description:
        description_html = f"""
        <div class="info-description">
            {description}
        </div>
        """

    return f"""
    <div class="info-card">
        <div class="info-title">
            {title}
        </div>

        <div class="info-value">
            {value}
        </div>

        {description_html}
    </div>
    """


def alert_card(
    title: str,
    message: str,
    severity: str = "warning",
) -> str:
    """Create an alert/exception card."""

    return f"""
    <div class="alert-card alert-{severity}">
        <div class="alert-title">
            {title}
        </div>

        <div class="alert-message">
            {message}
        </div>
    </div>
    """
def recommendation_card(rows: list[tuple[str, str]]) -> str:
    """Create the FreightAI final recommendation card."""

    rows_html = ""

    for label, value in rows:
        rows_html += (
            '<div class="recommendation-row">'
            f'<span class="recommendation-label">{label}</span>'
            f'<span class="recommendation-value">{value}</span>'
            '</div>'
        )

    return (
        '<div class="recommendation-card">'
        '<h3 class="recommendation-title">📦 FREIGHTAI RECOMMENDATION</h3>'
        f'{rows_html}'
        '</div>'
    )
def data_sources_card(sources: dict) -> str:
    """Create a clean field-level data sources card."""

    rows_html = ""

    for field, source in sources.items():
        field_name = field.replace("_", " ").title()

        rows_html += (
            '<div class="data-source-row">'
            f'<span class="data-source-field">{field_name}</span>'
            f'<span class="data-source-value">{source}</span>'
            '</div>'
        )

    if not rows_html:
        rows_html = (
            '<div class="data-source-empty">'
            'No field-level source information available.'
            '</div>'
        )

    return (
        '<div class="data-sources-card">'
        '<div class="data-sources-title">🔍 Field-level Data Sources</div>'
        '<div class="data-sources-subtitle">'
        'Data provenance for the port compatibility analysis'
        '</div>'
        f'{rows_html}'
        '</div>'
    )
def decision_factors_card(factors: dict) -> str:
    """Create a clean decision-factors card."""

    rows_html = ""

    for key, value in factors.items():
        field_name = key.replace("_", " ").title()

        if isinstance(value, float):
            display_value = f"{value:,.2f}"
        elif isinstance(value, int):
            display_value = f"{value:,}"
        elif isinstance(value, bool):
            display_value = "Yes" if value else "No"
        else:
            display_value = str(value)

        rows_html += (
            '<div class="decision-factor-row">'
            f'<span class="decision-factor-label">{field_name}</span>'
            f'<span class="decision-factor-value">{display_value}</span>'
            '</div>'
        )

    if not rows_html:
        rows_html = (
            '<div class="data-source-empty">'
            'No decision factors available.'
            '</div>'
        )

    return (
        '<div class="decision-factors-card">'
        '<div class="decision-factors-title">🧠 Decision Factors Used</div>'
        '<div class="decision-factors-subtitle">'
        'Inputs considered by the charter strategy decision engine'
        '</div>'
        f'{rows_html}'
        '</div>'
    )