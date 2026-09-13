"""
risk_engine.py
==============
Risk Assessment Engine for FreightAI.

Calculates an explainable, 0-100 operational risk score for a proposed
bulk cargo voyage, broken into four categories:

    1. Market Risk        — freight volatility, forecast uncertainty, recent freight movement
    2. Port Risk           — congestion, waiting time, infrastructure compatibility, tidal restrictions
    3. Weather Risk        — wind, rainfall, wave height, storm indicator
    4. Operational Risk    — vessel-port mismatch, high turnaround/waiting time, vessel suitability

This module is the "capstone" that composes the other FreightAI engines
already built: ``forecasting/`` (market risk), ``optimization/port_optimizer.py``
and ``optimization/voyage_calculator.py`` (port + operational risk), and
``optimization/vessel_optimizer.py`` (vessel-suitability risk). It does not
build the dashboard.

------------------------------------------------------------------------
Explainability
------------------------------------------------------------------------
Every category returns a list of plain-English ``reasons`` (e.g. "High
port congestion", "High forecast uncertainty", "Vessel draft close to
port limit") alongside its numeric score — never a bare number.

------------------------------------------------------------------------
Never invent weather or congestion data
------------------------------------------------------------------------
Weather risk is computed ONLY from ``data/processed/weather_clean.csv``.
If that file is missing, or has no record for the destination port, weather
risk is returned as unavailable (``score: None``) with an explicit reason
— never filled in with a plausible-looking fake number. The same applies
to the dedicated ``data/processed/congestion_clean.csv`` dataset for port
congestion: if present, its real data is used; if not, the engine falls
back to the destination port's ``congestion_level`` field from
``port_optimizer`` — which is itself always labeled as either real
(from ``ports_clean.csv``) or an explicitly-flagged prototype reference
value (see ``port_optimizer.REFERENCE_PORT_SPECS``) — so the provenance
of every congestion figure is always visible to the caller, never presented
as more certain than it is.

Any category that ends up with no usable data is excluded from the overall
weighted score (the remaining categories' weights are renormalized) rather
than silently defaulting to a fabricated mid-range value.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.forecasting.train import METRICS_PATH, PROCESSED_DATA_PATH as FREIGHT_HISTORY_PATH
from src.optimization.port_optimizer import PORTS_PATH, compare_vessel_to_port
from src.optimization.vessel_optimizer import classify_dwt, evaluate_cargo_capacity
from src.optimization.voyage_calculator import ROUTES_PATH, calculate_voyage

logger = logging.getLogger("freightai.risk.risk_engine")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEATHER_PATH = PROJECT_ROOT / "data" / "processed" / "weather_clean.csv"
CONGESTION_PATH = PROJECT_ROOT / "data" / "processed" / "congestion_clean.csv"

# ---------------------------------------------------------------------------
# PROTOTYPE weights — configurable, not a validated commercial risk model.
# ---------------------------------------------------------------------------

PROTOTYPE_CATEGORY_WEIGHTS: Dict[str, float] = {
    "market_risk": 0.30,
    "port_risk": 0.25,
    "weather_risk": 0.20,
    "operational_risk": 0.25,
}
assert abs(sum(PROTOTYPE_CATEGORY_WEIGHTS.values()) - 1.0) < 1e-9

MARKET_SUBWEIGHTS = {"freight_volatility": 0.40, "forecast_uncertainty": 0.35, "recent_freight_movement": 0.25}
PORT_SUBWEIGHTS = {"congestion": 0.35, "waiting_time": 0.25, "infrastructure_compatibility": 0.30, "tidal_restriction": 0.10}
WEATHER_SUBWEIGHTS = {"wind": 0.30, "rainfall": 0.20, "wave_height": 0.30, "storm_indicator": 0.20}
OPERATIONAL_SUBWEIGHTS = {"vessel_port_mismatch": 0.35, "turnaround_time": 0.20, "waiting_time": 0.20, "vessel_suitability": 0.25}

# Overall 0-100 score -> LOW/MEDIUM/HIGH bands. Configurable.
RISK_LEVEL_THRESHOLDS = {"LOW": 33, "MEDIUM": 66}  # score <= LOW -> "LOW"; <= MEDIUM -> "MEDIUM"; else "HIGH"

# PROTOTYPE thresholds used to turn a raw metric into a 0-100 risk score.
# (medium_threshold, high_threshold) — see _risk_from_thresholds().
THRESHOLDS = {
    "freight_volatility_cv_pct": (7.0, 15.0),
    "recent_movement_pct": (5.0, 12.0),
    "forecast_mape_pct": (8.0, 15.0),
    "waiting_hours": (18.0, 36.0),
    "turnaround_hours": (30.0, 48.0),
    "wind_knots": (21.0, 34.0),        # ~Beaufort 6 / Beaufort 8 (gale)
    "rainfall_mm": (25.0, 50.0),
    "wave_height_m": (2.5, 4.0),
}
NEAR_LIMIT_MARGIN_PCT = 10.0  # vessel value within this % of a port limit -> "close to limit" flag
CONGESTION_LEVEL_SCORE = {"LOW": 20.0, "MEDIUM": 55.0, "HIGH": 90.0}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _risk_from_thresholds(value: Optional[float], medium_thresh: float, high_thresh: float) -> Optional[float]:
    """Map a raw metric to a 0-100 risk score using two PROTOTYPE
    breakpoints: 0 at value=0, ~40 at the medium threshold, ~75 at the
    high threshold, approaching 100 well beyond it. Monotonic, continuous.
    """
    if value is None or pd.isna(value):
        return None
    value = abs(float(value))
    if value <= 0:
        return 0.0
    if value <= medium_thresh:
        return round(value / medium_thresh * 40.0, 1)
    if value <= high_thresh:
        return round(40.0 + (value - medium_thresh) / (high_thresh - medium_thresh) * 35.0, 1)
    extra = min(1.0, (value - high_thresh) / high_thresh)
    return round(75.0 + extra * 25.0, 1)


def weighted_average(scores: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    """Weighted average of ``scores`` by ``weights``, skipping any ``None``
    entries and renormalizing the remaining weights so they still sum to 1.
    Returns ``None`` if every score is unavailable.
    """
    available = {k: v for k, v in scores.items() if v is not None and k in weights}
    total_weight = sum(weights[k] for k in available)
    if not available or total_weight == 0:
        return None
    return round(sum(scores[k] * weights[k] for k in available) / total_weight, 2)


def classify_level(score: Optional[float]) -> Optional[str]:
    """Map a 0-100 score to LOW / MEDIUM / HIGH using :data:`RISK_LEVEL_THRESHOLDS`."""
    if score is None:
        return None
    if score <= RISK_LEVEL_THRESHOLDS["LOW"]:
        return "LOW"
    if score <= RISK_LEVEL_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "HIGH"


def _find_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    lower_map = {c: str(c).lower() for c in df.columns}
    for kw in keywords:
        for col, low in lower_map.items():
            if kw in low:
                return col
    return None


def _filter_latest_by_port(df: pd.DataFrame, port_name: str, port_keywords: List[str], date_keywords: List[str]) -> Optional[pd.Series]:
    """Find the most recent row of ``df`` matching ``port_name`` (by a
    keyword-detected port/location column). Returns ``None`` if no port
    column or no matching row is found.
    """
    port_col = _find_column(df, port_keywords)
    if port_col is None:
        return None
    target = port_name.strip().lower()
    mask = df[port_col].astype(str).str.strip().str.lower().str.contains(target, na=False, regex=False)
    matches = df[mask]
    if matches.empty:
        return None
    date_col = _find_column(matches, date_keywords)
    if date_col is not None:
        parsed_dates = pd.to_datetime(matches[date_col], errors="coerce")
        if parsed_dates.notna().any():
            matches = matches.assign(_parsed_date=parsed_dates).sort_values("_parsed_date")
    return matches.iloc[-1]


# ===========================================================================
# 1. Market Risk
# ===========================================================================

def assess_market_risk(
    freight_path: Path = FREIGHT_HISTORY_PATH,
    metrics_path: Path = METRICS_PATH,
    lookback_days: int = 30,
) -> Dict:
    """Market risk from real historical freight data (never invented) plus,
    if available, the trained forecasting model's own validation-set MAPE
    as a forecast-uncertainty proxy.
    """
    reasons: List[str] = []
    components: Dict[str, Optional[float]] = {
        "freight_volatility": None, "forecast_uncertainty": None, "recent_freight_movement": None,
    }
    details: Dict[str, Optional[float]] = {}

    if not freight_path.exists():
        reasons.append(
            "Freight history unavailable — market risk not assessed. "
            "Run the data pipeline module to produce freight_clean.csv."
        )
        return {"score": None, "level": None, "reasons": reasons, "components": components, "details": details}

    try:
        df = pd.read_csv(freight_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "freight_rate"]).sort_values("date").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"Could not read freight history ({exc}) — market risk not assessed.")
        return {"score": None, "level": None, "reasons": reasons, "components": components, "details": details}

    if df.empty:
        reasons.append("Freight history is empty — market risk not assessed.")
        return {"score": None, "level": None, "reasons": reasons, "components": components, "details": details}

    # --- freight volatility: coefficient of variation over the lookback window
    recent = df.tail(max(lookback_days, 7))
    mean_val, std_val = recent["freight_rate"].mean(), recent["freight_rate"].std()
    cv_pct = (std_val / mean_val * 100) if mean_val else None
    components["freight_volatility"] = _risk_from_thresholds(cv_pct, *THRESHOLDS["freight_volatility_cv_pct"])
    details["freight_volatility_cv_pct"] = round(cv_pct, 2) if cv_pct is not None else None
    if cv_pct is not None:
        med, high = THRESHOLDS["freight_volatility_cv_pct"]
        if cv_pct >= high:
            reasons.append(f"High freight-rate volatility over the last {lookback_days} days (CV={cv_pct:.1f}%).")
        elif cv_pct >= med:
            reasons.append(f"Moderate freight-rate volatility over the last {lookback_days} days (CV={cv_pct:.1f}%).")

    # --- recent freight movement: % change over the lookback window
    movement_pct = None
    if len(df) > lookback_days:
        past_val = df["freight_rate"].iloc[-lookback_days - 1]
        last_val = df["freight_rate"].iloc[-1]
        if past_val:
            movement_pct = (last_val - past_val) / past_val * 100
    components["recent_freight_movement"] = _risk_from_thresholds(movement_pct, *THRESHOLDS["recent_movement_pct"])
    details["recent_freight_movement_pct"] = round(movement_pct, 2) if movement_pct is not None else None
    if movement_pct is not None:
        med, high = THRESHOLDS["recent_movement_pct"]
        if abs(movement_pct) >= high:
            direction = "increase" if movement_pct > 0 else "decrease"
            reasons.append(f"Sharp recent freight-rate {direction} of {movement_pct:+.1f}% over the last {lookback_days} days.")
        elif abs(movement_pct) >= med:
            reasons.append(f"Moderate recent freight-rate movement of {movement_pct:+.1f}% over the last {lookback_days} days.")

    # --- forecast uncertainty: selected model's own validation MAPE, if trained
    mape = None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            selected = metrics.get("selected_model")
            mape = metrics.get("metrics", {}).get(selected, {}).get("validation", {}).get("mape")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read model metrics: %s", exc)
    components["forecast_uncertainty"] = _risk_from_thresholds(mape, *THRESHOLDS["forecast_mape_pct"])
    details["forecast_validation_mape_pct"] = mape
    if mape is not None:
        med, high = THRESHOLDS["forecast_mape_pct"]
        if mape >= high:
            reasons.append(f"High forecast uncertainty (selected forecasting model's validation MAPE = {mape:.1f}%).")
        elif mape >= med:
            reasons.append(f"Moderate forecast uncertainty (selected forecasting model's validation MAPE = {mape:.1f}%).")
    else:
        reasons.append(
            "Forecast uncertainty not assessed — no trained forecasting model found "
            "(run `python -m src.forecasting.train`)."
        )

    score = weighted_average(components, MARKET_SUBWEIGHTS)
    if not reasons:
        reasons.append("No significant market-risk indicators detected.")

    return {"score": score, "level": classify_level(score), "reasons": reasons, "components": components, "details": details}


# ===========================================================================
# 2. Port Risk
# ===========================================================================

def assess_port_risk(
    vessel_spec: Dict,
    destination_port: str,
    congestion_path: Path = CONGESTION_PATH,
    ports_path: Path = PORTS_PATH,
) -> Dict:
    """Port risk from real congestion data when available
    (``congestion_clean.csv``), the port's compatibility table (real
    vessel-vs-limit comparison), and its waiting-time/tidal-restriction
    fields (from ``ports_clean.csv`` if present, else the explicitly-labeled
    prototype reference specs in ``port_optimizer``).
    """
    reasons: List[str] = []
    port_result = compare_vessel_to_port(vessel_spec, destination_port, ports_path=ports_path)
    port_spec = port_result["port"]

    # --- congestion: prefer the dedicated congestion dataset; never invent it
    congestion_score, congestion_source = None, None
    if congestion_path.exists():
        try:
            cdf = pd.read_csv(congestion_path)
            row = _filter_latest_by_port(
                cdf, destination_port,
                port_keywords=["port_name", "port", "location"],
                date_keywords=["date", "time"],
            )
            if row is not None:
                level_col = _find_column(cdf, ["congestion_level", "congestion_category"])
                index_col = _find_column(cdf, ["congestion_index", "congestion_score", "queue", "vessels_waiting"])
                if level_col is not None and pd.notna(row.get(level_col)):
                    congestion_score = CONGESTION_LEVEL_SCORE.get(str(row[level_col]).strip().upper())
                    congestion_source = "congestion_clean.csv"
                elif index_col is not None and pd.notna(row.get(index_col)):
                    # Treat an index/queue-length column as a 0-100-ish raw signal via thresholds.
                    congestion_score = _risk_from_thresholds(pd.to_numeric(row[index_col], errors="coerce"), 5, 15)
                    congestion_source = "congestion_clean.csv"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read congestion data: %s", exc)

    if congestion_score is None:
        # Fall back to the port spec's own congestion_level (already labeled
        # real vs. reference in port_optimizer's field_sources).
        level = port_spec.get("congestion_level")
        congestion_score = CONGESTION_LEVEL_SCORE.get(level)
        source_label = port_spec.get("field_sources", {}).get("congestion_level", "reference_default")
        congestion_source = f"port spec ({source_label})"

    if congestion_score is not None and congestion_score >= CONGESTION_LEVEL_SCORE["HIGH"]:
        reasons.append(f"High port congestion at {port_spec['port_name']} (source: {congestion_source}).")
    elif congestion_score is not None and congestion_score >= CONGESTION_LEVEL_SCORE["MEDIUM"]:
        reasons.append(f"Moderate port congestion at {port_spec['port_name']} (source: {congestion_source}).")

    # --- waiting time
    waiting_hours = port_spec.get("avg_waiting_hours")
    waiting_score = _risk_from_thresholds(waiting_hours, *THRESHOLDS["waiting_hours"])
    if waiting_score is not None:
        med, high = THRESHOLDS["waiting_hours"]
        if waiting_hours >= high:
            reasons.append(f"High average waiting time at {port_spec['port_name']} (~{waiting_hours:.0f}h).")
        elif waiting_hours >= med:
            reasons.append(f"Moderate average waiting time at {port_spec['port_name']} (~{waiting_hours:.0f}h).")

    # --- infrastructure compatibility (real vessel-vs-port comparison)
    hard_checks = [row for row in port_result["compatibility_table"] if row["parameter"] in
                   ("draft", "loa", "beam", "dwt", "channel_depth")]
    n_fail = sum(1 for r in hard_checks if r["status"] == "FAIL")
    n_unknown = sum(1 for r in hard_checks if r["status"] == "UNKNOWN")
    infra_score = min(100.0, n_fail * 40.0 + n_unknown * 10.0)
    for r in hard_checks:
        if r["status"] == "FAIL":
            reasons.append(f"Vessel {r['parameter']} exceeds {port_spec['port_name']}'s limit ({r['vessel_value']} > {r['port_limit']}).")
        elif r["status"] == "PASS" and r["port_limit"]:
            margin_pct = (r["port_limit"] - r["vessel_value"]) / r["port_limit"] * 100
            if 0 <= margin_pct <= NEAR_LIMIT_MARGIN_PCT:
                reasons.append(f"Vessel {r['parameter']} close to {port_spec['port_name']}'s limit ({r['vessel_value']} vs. {r['port_limit']}, margin {margin_pct:.1f}%).")

    # --- tidal restriction
    tidal_score = 60.0 if port_spec.get("tidal_restriction") else 0.0
    if port_spec.get("tidal_restriction"):
        reasons.append(f"{port_spec['port_name']} has tidal access restrictions affecting berthing windows.")

    components = {
        "congestion": congestion_score,
        "waiting_time": waiting_score,
        "infrastructure_compatibility": infra_score,
        "tidal_restriction": tidal_score,
    }
    score = weighted_average(components, PORT_SUBWEIGHTS)
    if not reasons:
        reasons.append("No significant port-risk indicators detected.")

    return {
        "score": score, "level": classify_level(score), "reasons": reasons,
        "components": components,
        "details": {"port_compatibility": port_result, "congestion_source": congestion_source},
    }


# ===========================================================================
# 3. Weather Risk
# ===========================================================================

def assess_weather_risk(destination_port: str, weather_path: Path = WEATHER_PATH) -> Dict:
    """Weather risk computed ONLY from ``data/processed/weather_clean.csv``.
    Never invents wind/rainfall/wave/storm data — returns ``score: None``
    with an explicit reason if the dataset or a matching port record isn't
    available.
    """
    reasons: List[str] = []
    components: Dict[str, Optional[float]] = {"wind": None, "rainfall": None, "wave_height": None, "storm_indicator": None}
    details: Dict = {}

    if not weather_path.exists():
        reasons.append(
            "Weather data unavailable — weather risk not assessed. "
            "Provide data/processed/weather_clean.csv (via the data pipeline module) to enable this category."
        )
        return {"score": None, "level": None, "reasons": reasons, "components": components, "details": details}

    try:
        wdf = pd.read_csv(weather_path)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"Could not read weather data ({exc}) — weather risk not assessed.")
        return {"score": None, "level": None, "reasons": reasons, "components": components, "details": details}

    row = _filter_latest_by_port(
        wdf, destination_port,
        port_keywords=["port_name", "port", "location", "station"],
        date_keywords=["date", "time"],
    )
    if row is None:
        reasons.append(f"No weather records found for {destination_port} — weather risk not assessed.")
        return {"score": None, "level": None, "reasons": reasons, "components": components, "details": details}

    wind_col = _find_column(wdf, ["wind_speed", "wind"])
    rain_col = _find_column(wdf, ["rainfall", "precipitation", "rain"])
    wave_col = _find_column(wdf, ["wave_height", "wave"])
    storm_col = _find_column(wdf, ["storm", "cyclone"])

    if wind_col is not None and pd.notna(row.get(wind_col)):
        wind = float(pd.to_numeric(row[wind_col], errors="coerce"))
        components["wind"] = _risk_from_thresholds(wind, *THRESHOLDS["wind_knots"])
        details["wind_knots"] = wind
        med, high = THRESHOLDS["wind_knots"]
        if wind >= high:
            reasons.append(f"High wind speed at {destination_port} (~{wind:.0f} kn).")
        elif wind >= med:
            reasons.append(f"Moderate wind speed at {destination_port} (~{wind:.0f} kn).")

    if rain_col is not None and pd.notna(row.get(rain_col)):
        rain = float(pd.to_numeric(row[rain_col], errors="coerce"))
        components["rainfall"] = _risk_from_thresholds(rain, *THRESHOLDS["rainfall_mm"])
        details["rainfall_mm"] = rain
        med, high = THRESHOLDS["rainfall_mm"]
        if rain >= high:
            reasons.append(f"Heavy rainfall at {destination_port} (~{rain:.0f} mm).")
        elif rain >= med:
            reasons.append(f"Moderate rainfall at {destination_port} (~{rain:.0f} mm).")

    if wave_col is not None and pd.notna(row.get(wave_col)):
        wave = float(pd.to_numeric(row[wave_col], errors="coerce"))
        components["wave_height"] = _risk_from_thresholds(wave, *THRESHOLDS["wave_height_m"])
        details["wave_height_m"] = wave
        med, high = THRESHOLDS["wave_height_m"]
        if wave >= high:
            reasons.append(f"High wave height at {destination_port} (~{wave:.1f} m).")
        elif wave >= med:
            reasons.append(f"Moderate wave height at {destination_port} (~{wave:.1f} m).")

    if storm_col is not None and pd.notna(row.get(storm_col)):
        raw = row[storm_col]
        is_storm = bool(raw) if not isinstance(raw, str) else raw.strip().lower() in ("1", "true", "yes", "y")
        components["storm_indicator"] = 95.0 if is_storm else 0.0
        details["storm_indicator"] = is_storm
        if is_storm:
            reasons.append(f"Active storm/cyclone indicator flagged for {destination_port}.")

    if all(v is None for v in components.values()):
        reasons.append(
            f"Weather record found for {destination_port} but no recognizable wind/rainfall/wave/storm "
            "columns were detected — weather risk not assessed."
        )
        return {"score": None, "level": None, "reasons": reasons, "components": components, "details": details}

    score = weighted_average(components, WEATHER_SUBWEIGHTS)
    if not reasons:
        reasons.append("No significant weather-risk indicators detected.")

    return {"score": score, "level": classify_level(score), "reasons": reasons, "components": components, "details": details}


# ===========================================================================
# 4. Operational Risk
# ===========================================================================

def assess_operational_risk(
    vessel_spec: Dict,
    destination_port: str,
    cargo_quantity: float,
    origin: str,
    routes_path: Path = ROUTES_PATH,
    ports_path: Path = PORTS_PATH,
) -> Dict:
    """Operational risk from vessel-port mismatch (compatibility fails),
    high turnaround/waiting time, and vessel-suitability (cargo-capacity
    utilization) — reusing ``voyage_calculator``/``vessel_optimizer`` so
    these numbers are computed exactly once, consistently, across FreightAI.
    """
    reasons: List[str] = []
    voyage = calculate_voyage(vessel_spec, origin, destination_port, cargo_quantity,
                               routes_path=routes_path, ports_path=ports_path)
    port_result = voyage["port_compatibility"]
    port_spec = port_result["port"]

    hard_checks = [row for row in port_result["compatibility_table"] if row["parameter"] in
                   ("draft", "loa", "beam", "dwt", "channel_depth")]
    n_fail = sum(1 for r in hard_checks if r["status"] == "FAIL")
    mismatch_score = min(100.0, n_fail * 40.0)
    if n_fail:
        reasons.append(f"Vessel-port mismatch: {n_fail} parameter(s) exceed {port_spec['port_name']}'s limits.")

    turnaround_hours = port_spec.get("avg_turnaround_hours")
    turnaround_score = _risk_from_thresholds(turnaround_hours, *THRESHOLDS["turnaround_hours"])
    if turnaround_score is not None:
        med, high = THRESHOLDS["turnaround_hours"]
        if turnaround_hours >= high:
            reasons.append(f"High expected turnaround time at {port_spec['port_name']} (~{turnaround_hours:.0f}h).")
        elif turnaround_hours >= med:
            reasons.append(f"Moderate expected turnaround time at {port_spec['port_name']} (~{turnaround_hours:.0f}h).")

    waiting_hours = port_spec.get("avg_waiting_hours")
    waiting_score = _risk_from_thresholds(waiting_hours, *THRESHOLDS["waiting_hours"])
    if waiting_score is not None:
        med, high = THRESHOLDS["waiting_hours"]
        if waiting_hours >= high:
            reasons.append(f"High expected waiting time at {port_spec['port_name']} (~{waiting_hours:.0f}h).")

    dwt = vessel_spec.get("dwt")
    vessel_class = classify_dwt(dwt) if dwt else None
    cargo_eval = evaluate_cargo_capacity(cargo_quantity, dwt) if dwt else {"score": None, "note": "Vessel DWT unknown."}
    suitability_score = (100.0 - cargo_eval["score"]) if cargo_eval.get("score") is not None else None
    if suitability_score is not None and cargo_eval.get("feasible") is False:
        reasons.append(f"Vessel suitability: {cargo_eval['note']}")
    elif suitability_score is not None and suitability_score >= 40:
        reasons.append(f"Vessel suitability concern ({vessel_class or 'unclassified'} class): {cargo_eval['note']}")

    components = {
        "vessel_port_mismatch": mismatch_score,
        "turnaround_time": turnaround_score,
        "waiting_time": waiting_score,
        "vessel_suitability": suitability_score,
    }
    score = weighted_average(components, OPERATIONAL_SUBWEIGHTS)
    if not reasons:
        reasons.append("No significant operational-risk indicators detected.")

    return {
        "score": score, "level": classify_level(score), "reasons": reasons,
        "components": components,
        "details": {"vessel_class": vessel_class, "cargo_capacity": cargo_eval, "voyage": voyage},
    }


# ===========================================================================
# Overall assessment
# ===========================================================================

def assess_voyage_risk(
    vessel_spec: Dict,
    origin: str,
    destination_port: str,
    cargo_quantity: float,
    weights: Optional[Dict[str, float]] = None,
    lookback_days: int = 30,
    freight_path: Path = FREIGHT_HISTORY_PATH,
    metrics_path: Path = METRICS_PATH,
    ports_path: Path = PORTS_PATH,
    weather_path: Path = WEATHER_PATH,
    congestion_path: Path = CONGESTION_PATH,
    routes_path: Path = ROUTES_PATH,
) -> Dict:
    """Run all four risk categories and combine them into an overall,
    explainable voyage risk assessment.

    Parameters
    ----------
    vessel_spec: dict with ``dwt``, ``loa``, ``beam``, ``draft``, ``speed``.
    origin, destination_port, cargo_quantity: as in the other optimization engines.
    weights: optional override of :data:`PROTOTYPE_CATEGORY_WEIGHTS`
        (same keys, need not sum to 1.0 — renormalized automatically).

    Returns
    -------
    {
        "market_risk": {...}, "port_risk": {...}, "weather_risk": {...}, "operational_risk": {...},
        "overall_risk": "LOW" | "MEDIUM" | "HIGH" | None,
        "risk_score": float | None,          # 0-100
        "risk_reasons": [str, ...],           # combined, deduplicated, across all categories
        "weights_used": {...},                 # labeled prototype
        "excluded_categories": [str, ...],      # categories with no usable data
        "disclaimer": str,
    }
    """
    weights = weights or PROTOTYPE_CATEGORY_WEIGHTS

    market_risk = assess_market_risk(freight_path=freight_path, metrics_path=metrics_path, lookback_days=lookback_days)
    port_risk = assess_port_risk(vessel_spec, destination_port, congestion_path=congestion_path, ports_path=ports_path)
    weather_risk = assess_weather_risk(destination_port, weather_path=weather_path)
    operational_risk = assess_operational_risk(vessel_spec, destination_port, cargo_quantity, origin,
                                                routes_path=routes_path, ports_path=ports_path)

    category_scores = {
        "market_risk": market_risk["score"],
        "port_risk": port_risk["score"],
        "weather_risk": weather_risk["score"],
        "operational_risk": operational_risk["score"],
    }
    risk_score = weighted_average(category_scores, weights)
    overall_risk = classify_level(risk_score)
    excluded = [k for k, v in category_scores.items() if v is None]

    risk_reasons: List[str] = []
    for cat in (market_risk, port_risk, weather_risk, operational_risk):
        for reason in cat["reasons"]:
            if reason not in risk_reasons:
                risk_reasons.append(reason)
    if excluded:
        risk_reasons.append(
            f"Overall score excludes: {', '.join(excluded)} (no usable data) — remaining category weights were renormalized."
        )

    return {
        "market_risk": market_risk,
        "port_risk": port_risk,
        "weather_risk": weather_risk,
        "operational_risk": operational_risk,
        "overall_risk": overall_risk,
        "risk_score": risk_score,
        "risk_reasons": risk_reasons,
        "weights_used": {**weights, "_label": "PROTOTYPE_CATEGORY_WEIGHTS — configurable, not a validated commercial risk model"},
        "excluded_categories": excluded,
        "disclaimer": (
            "This is a prototype, explainable risk-scoring model using configurable heuristic "
            "weights and thresholds. Weather and congestion figures are used only when sourced "
            "from real data (data/processed/weather_clean.csv, congestion_clean.csv) or from "
            "clearly-labeled reference port specs — never fabricated. It is not a substitute for "
            "professional voyage risk assessment, weather routing, or underwriting analysis."
        ),
    }


if __name__ == "__main__":
    import json as _json

    demo_vessel = {"dwt": 82_000, "loa": 225.0, "beam": 32.3, "draft": 14.0, "speed": 14.2}
    result = assess_voyage_risk(
        vessel_spec=demo_vessel, origin="Port Hedland",
        destination_port="Paradip", cargo_quantity=75_000,
    )
    print("\n=== FreightAI Risk Assessment Engine — demo ===")
    print(f"Overall risk: {result['overall_risk']}  (score={result['risk_score']})")
    for cat in ("market_risk", "port_risk", "weather_risk", "operational_risk"):
        c = result[cat]
        print(f"  {cat:18s} score={c['score']}  level={c['level']}")
    print("\nReasons:")
    for r in result["risk_reasons"]:
        print(f"  - {r}")
