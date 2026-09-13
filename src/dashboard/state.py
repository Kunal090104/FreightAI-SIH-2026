"""
state.py
========
Streamlit-independent orchestration layer for the FreightAI dashboard.

Everything in this module can be imported and unit-tested without
Streamlit installed — it only calls the real FreightAI engines (forecasting,
optimization, risk, database) and returns plain dicts. ``app.py`` and the
page modules under ``src/dashboard/pages/`` are the only places that touch
``streamlit`` itself; this module is the seam that keeps the dashboard
"just a UI" over logic that already exists and is fully reusable elsewhere
(a CLI, a notebook, tests) without dragging Streamlit along.

No ML/optimization logic is reimplemented here — every number in the
returned dicts comes from calling ``src.forecasting``, ``src.optimization``,
``src.risk``, or ``src.database`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data.pipeline import DATASET_KEYS, PROCESSED_DATA_DIR
from src.data.preprocessing import DATASET_DATE_COLUMNS
from src.data.quality_report import generate_quality_report
from src.database.database import (
    initialize_database, insert_data, port_spec_to_db_row, save_forecast_result, vessel_spec_to_db_row,
)
from src.forecasting.predict import forecast_freight
from src.forecasting.train import FEATURE_COLUMNS_PATH, METRICS_PATH, MODEL_PATH, PROCESSED_DATA_PATH
from src.optimization.charter_optimizer import recommend_charter_strategy
from src.optimization.port_optimizer import SUPPORTED_PORTS, compare_vessel_to_port
from src.optimization.vessel_optimizer import recommend_vessel_class
from src.optimization.voyage_calculator import calculate_voyage
from src.risk.risk_engine import assess_voyage_risk

MIN_FORECAST_HORIZON_DAYS = 1
MAX_FORECAST_HORIZON_DAYS = 90


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------

@dataclass
class UserInputs:
    """Everything the input form collects, in one place."""
    commodity: str = "Iron Ore"
    cargo_quantity: float = 75_000.0
    origin_country: str = ""
    origin_port: str = ""
    destination_port: str = SUPPORTED_PORTS[0]
    required_delivery_date: Optional[date] = None
    contract_duration_months: float = 6.0
    expected_number_of_voyages: int = 1


def compute_forecast_horizon_days(required_delivery_date: Optional[date], today: Optional[date] = None) -> Dict:
    """Turn a delivery date into a forecast horizon (days), clipped to the
    range the forecasting engine actually supports [1, 90]. Never silently
    extrapolates beyond what ``forecast_freight`` was designed for.
    """
    today = today or date.today()
    if required_delivery_date is None:
        return {"horizon_days": 30, "note": "No delivery date provided — defaulting to a 30-day forecast horizon."}

    raw_days = (required_delivery_date - today).days
    if raw_days < MIN_FORECAST_HORIZON_DAYS:
        return {
            "horizon_days": MIN_FORECAST_HORIZON_DAYS,
            "note": "Delivery date is today or in the past — using the minimum 1-day forecast horizon.",
        }
    if raw_days > MAX_FORECAST_HORIZON_DAYS:
        return {
            "horizon_days": MAX_FORECAST_HORIZON_DAYS,
            "note": (
                f"Delivery date is {raw_days} days out, beyond the 90-day horizon the forecasting "
                "engine supports — showing the maximum supported 90-day (experimental) forecast instead."
            ),
        }
    return {"horizon_days": raw_days, "note": None}


# ---------------------------------------------------------------------------
# Safe-call wrapper — every engine call goes through this so one failing
# engine (missing data, untrained model, bad input) never crashes the app.
# ---------------------------------------------------------------------------

def _safe_call(fn, *args, **kwargs) -> Dict:
    try:
        return {"result": fn(*args, **kwargs), "error": None}
    except Exception as exc:  # noqa: BLE001 — deliberately broad: this is the app's crash firewall
        return {"result": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(inputs: UserInputs) -> Dict:
    """Run every FreightAI engine for the given inputs and return a single
    dict the dashboard pages can read from — computed once per form
    submission, not on every Streamlit rerun.
    """
    horizon = compute_forecast_horizon_days(inputs.required_delivery_date)

    forecast = _safe_call(forecast_freight, days=horizon["horizon_days"])
    if forecast["result"] is not None:
        try:
            initialize_database()
            save_forecast_result(forecast["result"])
        except Exception:  # noqa: BLE001 — logging a prediction must never block the dashboard
            pass

    vessel = _safe_call(
        recommend_vessel_class,
        cargo_quantity=inputs.cargo_quantity, commodity=inputs.commodity,
        origin=inputs.origin_port, destination=inputs.destination_port,
    )

    vessel_spec = None
    if vessel["result"] is not None:
        rec_type = vessel["result"]["recommended_vessel_type"]
        vessel_spec = vessel["result"]["evaluations"][rec_type]["vessel_spec"]

    port = {"result": None, "error": "Vessel recommendation unavailable — port compatibility needs a vessel spec."}
    voyage = {"result": None, "error": "Vessel recommendation unavailable — voyage analysis needs a vessel spec."}
    risk = {"result": None, "error": "Vessel recommendation unavailable — risk assessment needs a vessel spec."}

    if vessel_spec is not None:
        port = _safe_call(compare_vessel_to_port, vessel_spec, inputs.destination_port)
        voyage = _safe_call(
            calculate_voyage, vessel_spec, inputs.origin_port, inputs.destination_port, inputs.cargo_quantity,
        )
        risk = _safe_call(
            assess_voyage_risk, vessel_spec, inputs.origin_port, inputs.destination_port, inputs.cargo_quantity,
        )
        try:
            initialize_database()
            insert_data("vessels", vessel_spec_to_db_row(
                vessel_spec, vessel_name=f"Recommended {vessel['result']['recommended_vessel_type']}",
                vessel_type=vessel["result"]["recommended_vessel_type"],
            ))
            if port["result"] is not None:
                insert_data("ports", port_spec_to_db_row(port["result"]["port"]))
        except Exception:  # noqa: BLE001 — persistence must never block the dashboard
            pass

    charter = _safe_call(
        recommend_charter_strategy,
        cargo_quantity=inputs.cargo_quantity,
        contract_duration_months=inputs.contract_duration_months,
        expected_number_of_voyages=inputs.expected_number_of_voyages,
        forecast_result=forecast["result"],
        vessel_recommendation=vessel["result"],
        risk_result=risk["result"],
    )

    return {
        "inputs": inputs,
        "horizon": horizon,
        "forecast": forecast,
        "vessel": vessel,
        "vessel_spec": vessel_spec,
        "port": port,
        "voyage": voyage,
        "risk": risk,
        "charter": charter,
        "computed_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Data status — REAL / DEMO / MIXED / NO DATA, surfaced from provenance
# metadata the engines already return (data_source / field_sources), never
# invented by the dashboard.
# ---------------------------------------------------------------------------

def _classify_source(label: Optional[str]) -> str:
    if not label:
        return "NO DATA"
    low = str(label).lower()
    if "reference" in low or "default" in low:
        return "DEMO"
    return "REAL"


def aggregate_status(labels: List[str]) -> str:
    """Combine several per-field source labels into one overall status."""
    statuses = {_classify_source(l) for l in labels if l}
    statuses.discard("NO DATA")
    if not statuses:
        return "NO DATA"
    if statuses == {"REAL"}:
        return "REAL"
    if statuses == {"DEMO"}:
        return "DEMO"
    return "MIXED"


def vessel_data_status(vessel_result: Optional[Dict]) -> str:
    if not vessel_result:
        return "NO DATA"
    rec_type = vessel_result["recommended_vessel_type"]
    source = vessel_result["evaluations"][rec_type]["vessel_spec"].get("data_source")
    return "REAL" if source == "fleet_data" else "DEMO"


def port_data_status(port_result: Optional[Dict]) -> str:
    if not port_result:
        return "NO DATA"
    field_sources = port_result.get("port", {}).get("field_sources", {})
    return aggregate_status(list(field_sources.values()))


def routes_data_status(voyage_result: Optional[Dict]) -> str:
    if not voyage_result:
        return "NO DATA"
    return "REAL" if voyage_result.get("distance_status") == "OK" else "NO DATA"


def weather_data_status(risk_result: Optional[Dict]) -> str:
    if not risk_result:
        return "NO DATA"
    return "REAL" if risk_result.get("weather_risk", {}).get("score") is not None else "NO DATA"


def congestion_data_status(risk_result: Optional[Dict]) -> str:
    if not risk_result:
        return "NO DATA"
    source = risk_result.get("port_risk", {}).get("details", {}).get("congestion_source") or ""
    if source.startswith("congestion_clean.csv"):
        return "REAL"
    if "reference_default" in source:
        return "DEMO"
    if "ports_clean.csv" in source:
        return "REAL"
    return "NO DATA"


def forecast_data_status() -> str:
    """REAL only if there's a trained model AND real processed freight
    history behind it — never a demo fallback (the forecasting engine has
    no synthetic-data mode; if it's not trained, there is simply no forecast).
    """
    if MODEL_PATH.exists() and METRICS_PATH.exists() and FEATURE_COLUMNS_PATH.exists() and PROCESSED_DATA_PATH.exists():
        return "REAL"
    return "NO DATA"


def overall_data_status(analysis: Dict) -> str:
    """One combined badge for the top of the dashboard."""
    labels = [
        forecast_data_status(),
        vessel_data_status(analysis.get("vessel", {}).get("result")),
        port_data_status(analysis.get("port", {}).get("result")),
    ]
    non_empty = [l for l in labels if l != "NO DATA"]
    if not non_empty:
        return "NO DATA"
    if set(non_empty) == {"REAL"}:
        return "REAL"
    if set(non_empty) == {"DEMO"}:
        return "DEMO"
    return "MIXED"


# ---------------------------------------------------------------------------
# Data Quality page support — reads existing processed CSVs and reuses the
# real quality_report module; never re-runs the mutating pipeline itself
# (that stays a deliberate, explicit CLI action: `python -m src.data.pipeline`).
# ---------------------------------------------------------------------------

def load_data_quality_overview(processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    reports = {}
    for key in DATASET_KEYS:
        path = processed_dir / f"{key}_clean.csv"
        df = None
        if path.exists():
            try:
                df = pd.read_csv(path)
            except Exception:  # noqa: BLE001
                df = None
        reports[key] = generate_quality_report(df, key, date_col=DATASET_DATE_COLUMNS.get(key))

    rows = []
    for key, r in reports.items():
        rows.append({
            "dataset": key,
            "available": r["available"],
            "rows": r["rows"],
            "columns": r["columns"],
            "missing_values": r["missing_values_total"],
            "duplicate_rows": r["duplicate_rows"],
            "date_range": f"{r['date_range']['start']} to {r['date_range']['end']}" if r["date_range"]["start"] else "—",
            "outlier_count": r["outlier_count"],
        })
    return pd.DataFrame(rows)
