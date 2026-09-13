"""
voyage_calculator.py
=====================
Voyage Analysis Engine for FreightAI.

Computes sailing time, port time, idle time, and total estimated voyage
duration for a vessel calling at one of the seven supported Indian East
Coast ports (see ``port_optimizer.py``), and folds in that port's
compatibility table so a single call returns everything the (future)
dashboard needs for one vessel/route/cargo combination.

Formulas (as specified):
    Sailing Time            = Distance / Vessel Speed
    Total Port Time         = Waiting Time + Cargo Handling Time + Turnaround Time
    Estimated Total Voyage Time = Sailing Time + Total Port Time

------------------------------------------------------------------------
On distance — never invented
------------------------------------------------------------------------
Route distance is read ONLY from ``data/raw/routes.csv`` (columns:
``origin``, ``destination``, ``distance_nm``). This module does not
estimate, geocode, or otherwise invent a real-world nautical-mile
distance. If no matching route row is found, every distance-dependent
result (sailing time, total voyage time) is returned as ``None`` and the
result explicitly displays the string ``"Route distance unavailable."``
Port-time and idle-time components — which don't depend on distance —
are still computed and returned.

------------------------------------------------------------------------
On idle time — prototype estimate
------------------------------------------------------------------------
Waiting time comes from the port's ``avg_waiting_hours`` reference/actual
figure. Congestion delay and weather delay are simplified, clearly-labeled
PROTOTYPE placeholders (a congestion-level-to-hours lookup, and a flat
seasonal weather buffer) — not a live congestion feed or a weather
forecast. Both are configurable parameters on :func:`estimate_idle_time`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.optimization.port_optimizer import PORTS_PATH, compare_vessel_to_port

logger = logging.getLogger("freightai.optimization.voyage_calculator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTES_PATH = PROJECT_ROOT / "data" / "raw" / "routes.csv"

ROUTE_DISTANCE_UNAVAILABLE_MSG = "Route distance unavailable."

# PROTOTYPE placeholders — configurable, not live data feeds.
CONGESTION_DELAY_HOURS = {"LOW": 0.0, "MEDIUM": 6.0, "HIGH": 18.0}
DEFAULT_WEATHER_DELAY_HOURS = 8.0  # flat seasonal buffer; override per call if you have better info


# ---------------------------------------------------------------------------
# Route distance (never invented — CSV-only)
# ---------------------------------------------------------------------------

def load_route_distance(origin: str, destination: str, routes_path: Path = ROUTES_PATH) -> Optional[float]:
    """Look up the nautical-mile distance for ``origin`` -> ``destination``
    in ``data/raw/routes.csv``. Returns ``None`` (never raises, never
    estimates) if the file or the specific route row doesn't exist.
    """
    if not routes_path.exists():
        logger.info("Routes file not found at %s — route distance unavailable.", routes_path)
        return None
    try:
        df = pd.read_csv(routes_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read routes file: %s", exc)
        return None

    required = {"origin", "destination", "distance_nm"}
    cols_lower = {c.lower().strip(): c for c in df.columns}
    if not required.issubset(cols_lower.keys()):
        logger.warning("routes.csv missing required columns %s — found %s.", required, list(df.columns))
        return None

    origin_col, dest_col, dist_col = cols_lower["origin"], cols_lower["destination"], cols_lower["distance_nm"]
    o, d = str(origin).strip().lower(), str(destination).strip().lower()

    mask = (
        df[origin_col].astype(str).str.strip().str.lower() == o
    ) & (
        df[dest_col].astype(str).str.strip().str.lower() == d
    )
    matches = df[mask]
    if matches.empty:
        logger.info("No route row for '%s' -> '%s' in routes.csv — distance unavailable.", origin, destination)
        return None

    distance = pd.to_numeric(matches.iloc[0][dist_col], errors="coerce")
    return float(distance) if pd.notna(distance) else None


# ---------------------------------------------------------------------------
# Time components
# ---------------------------------------------------------------------------

def calculate_sailing_time_hours(distance_nm: Optional[float], vessel_speed_knots: Optional[float]) -> Optional[float]:
    """Sailing Time = Distance / Vessel Speed (knots = nautical miles/hour,
    so the result is already in hours). Returns ``None`` if distance is
    unavailable or speed is missing/non-positive.
    """
    if distance_nm is None:
        return None
    if vessel_speed_knots is None or vessel_speed_knots <= 0:
        logger.warning("Vessel speed missing/invalid — cannot compute sailing time.")
        return None
    return round(distance_nm / vessel_speed_knots, 2)


def calculate_cargo_handling_time_hours(cargo_quantity: float, cargo_handling_rate_tpd: Optional[float]) -> Optional[float]:
    """Cargo Handling Time, in hours, from a port's handling rate
    (metric tons/day). Returns ``None`` if the rate is missing/non-positive.
    """
    if cargo_handling_rate_tpd is None or cargo_handling_rate_tpd <= 0:
        logger.warning("Cargo handling rate missing/invalid — cannot compute handling time.")
        return None
    return round(cargo_quantity / cargo_handling_rate_tpd * 24.0, 2)


def estimate_idle_time(
    port_spec: Dict,
    congestion_delay_hours: Optional[Dict[str, float]] = None,
    weather_delay_hours: float = DEFAULT_WEATHER_DELAY_HOURS,
) -> Dict:
    """Estimate idle time at port: waiting time (from the port spec) plus
    two PROTOTYPE delay buffers — congestion delay (looked up from the
    port's congestion_level) and a flat weather delay. See module
    docstring: these are configurable placeholders, not live feeds.
    """
    congestion_map = congestion_delay_hours or CONGESTION_DELAY_HOURS
    waiting_hours = float(port_spec.get("avg_waiting_hours") or 0.0)
    congestion_level = port_spec.get("congestion_level", "MEDIUM")
    congestion_delay = float(congestion_map.get(congestion_level, congestion_map.get("MEDIUM", 6.0)))

    total_idle = round(waiting_hours + congestion_delay + weather_delay_hours, 2)
    return {
        "waiting_time_hours": round(waiting_hours, 2),
        "congestion_delay_hours": round(congestion_delay, 2),
        "congestion_level": congestion_level,
        "weather_delay_hours": round(weather_delay_hours, 2),
        "total_estimated_idle_time_hours": total_idle,
        "note": (
            "PROTOTYPE estimate: congestion_delay is a lookup by congestion_level, not a live "
            "vessel-traffic feed; weather_delay is a flat seasonal placeholder, not a forecast."
        ),
    }


# ---------------------------------------------------------------------------
# Full voyage calculation
# ---------------------------------------------------------------------------

def calculate_voyage(
    vessel_spec: Dict,
    origin: str,
    destination_port: str,
    cargo_quantity: float,
    routes_path: Path = ROUTES_PATH,
    ports_path: Path = PORTS_PATH,
    weather_delay_hours: float = DEFAULT_WEATHER_DELAY_HOURS,
) -> Dict:
    """Run the full port-compatibility + voyage-time analysis for one
    vessel/route/cargo combination.

    Parameters
    ----------
    vessel_spec: dict with ``dwt``, ``loa``, ``beam``, ``draft``, ``speed``
        (same shape as produced by ``vessel_optimizer.build_class_profiles()``).
    origin: origin port/place name — matched against ``routes.csv``.
    destination_port: one of ``port_optimizer.SUPPORTED_PORTS``.
    cargo_quantity: metric tons.

    Returns
    -------
    {
        "port_compatibility": {...compare_vessel_to_port() result...},
        "distance_nm": float | None,
        "distance_status": "OK" | "Route distance unavailable.",
        "sailing_time_hours": float | None,
        "port_time": {
            "waiting_time_hours": float,
            "cargo_handling_time_hours": float | None,
            "turnaround_time_hours": float,
            "total_port_time_hours": float | None,
        },
        "idle_time": {...estimate_idle_time() result...},
        "estimated_total_voyage_time_hours": float | None,
        "estimated_total_voyage_time_days": float | None,
        "warnings": [str, ...],
    }
    """
    if cargo_quantity is None or cargo_quantity <= 0:
        raise ValueError(f"cargo_quantity must be a positive number, got {cargo_quantity!r}.")
    if not origin or not destination_port:
        raise ValueError("Both 'origin' and 'destination_port' are required.")

    port_result = compare_vessel_to_port(vessel_spec, destination_port, ports_path=ports_path)
    port_spec = port_result["port"]
    warnings = list(port_result["warnings"])

    distance_nm = load_route_distance(origin, destination_port, routes_path=routes_path)
    distance_status = "OK" if distance_nm is not None else ROUTE_DISTANCE_UNAVAILABLE_MSG
    if distance_nm is None:
        warnings.append(
            f"{ROUTE_DISTANCE_UNAVAILABLE_MSG} No route row found for '{origin}' -> "
            f"'{destination_port}' in data/raw/routes.csv. Sailing time and total voyage "
            "time cannot be calculated without a real distance."
        )

    sailing_time = calculate_sailing_time_hours(distance_nm, vessel_spec.get("speed"))
    if distance_nm is not None and sailing_time is None:
        warnings.append("Vessel speed missing/invalid — sailing time could not be calculated.")

    waiting_time = float(port_spec.get("avg_waiting_hours") or 0.0)
    turnaround_time = float(port_spec.get("avg_turnaround_hours") or 0.0)
    handling_time = calculate_cargo_handling_time_hours(cargo_quantity, port_spec.get("cargo_handling_rate"))
    if handling_time is None:
        warnings.append("Cargo handling rate unavailable for this port — handling time could not be calculated.")

    total_port_time = None
    if handling_time is not None:
        total_port_time = round(waiting_time + handling_time + turnaround_time, 2)

    idle_time = estimate_idle_time(port_spec, weather_delay_hours=weather_delay_hours)

    total_voyage_time = None
    if sailing_time is not None and total_port_time is not None:
        total_voyage_time = round(sailing_time + total_port_time, 2)

    if not port_result["compatible"]:
        warnings.append(
            f"Vessel is NOT fully compatible with {port_spec['port_name']} — see port_compatibility.compatibility_table."
        )

    return {
        "inputs": {
            "origin": origin,
            "destination_port": port_spec["port_name"],
            "cargo_quantity": cargo_quantity,
        },
        "port_compatibility": port_result,
        "distance_nm": distance_nm,
        "distance_status": distance_status,
        "sailing_time_hours": sailing_time,
        "port_time": {
            "waiting_time_hours": round(waiting_time, 2),
            "cargo_handling_time_hours": handling_time,
            "turnaround_time_hours": round(turnaround_time, 2),
            "total_port_time_hours": total_port_time,
        },
        "idle_time": idle_time,
        "estimated_total_voyage_time_hours": total_voyage_time,
        "estimated_total_voyage_time_days": round(total_voyage_time / 24.0, 2) if total_voyage_time is not None else None,
        "warnings": warnings,
    }


if __name__ == "__main__":
    import json

    demo_vessel = {"dwt": 82_000, "loa": 225.0, "beam": 32.3, "draft": 14.0, "speed": 14.2}
    result = calculate_voyage(
        vessel_spec=demo_vessel,
        origin="Port Hedland",
        destination_port="Paradip",
        cargo_quantity=75_000,
    )
    print("\n=== FreightAI Voyage Calculator — demo ===")
    print(f"Distance: {result['distance_nm']} nm ({result['distance_status']})")
    print(f"Sailing time: {result['sailing_time_hours']} h")
    print(f"Port time: {result['port_time']}")
    print(f"Idle time: {result['idle_time']['total_estimated_idle_time_hours']} h")
    print(f"Total voyage time: {result['estimated_total_voyage_time_hours']} h "
          f"({result['estimated_total_voyage_time_days']} days)")
    if result["warnings"]:
        print("\nWarnings:")
        for w in result["warnings"]:
            print(f"  - {w}")
