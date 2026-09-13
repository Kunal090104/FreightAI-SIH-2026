"""
port_optimizer.py
==================
Port Compatibility Engine for FreightAI — Indian East Coast destination ports.

Supports exactly seven destination ports:
    Paradip, Visakhapatnam, Gangavaram, Gopalpur, Dhamra, Sagar-Sandheads, Haldia

Compares a vessel's specifications against each port's operating limits and
returns a parameter-by-parameter PASS/FAIL/INFO table, so
``voyage_calculator.py`` (and, later, the dashboard) can build on a single
source of port truth.

------------------------------------------------------------------------
Data sourcing and honesty about precision
------------------------------------------------------------------------
Two data sources are used, in priority order:

    1. ``data/processed/ports_clean.csv`` — if present and a row matches
       the requested port (by name), its columns are auto-detected by
       keyword (same approach as the rest of FreightAI's data modules) and
       used, field by field.
    2. :data:`REFERENCE_PORT_SPECS` — indicative figures compiled from
       public port-authority/port-reference sources, used for any field
       not found in (1). These are reasonable planning-level approximations
       (channel depths in particular change with ongoing dredging projects)
       — NOT a substitute for the current official Notice to Mariners /
       port authority pilotage data before an actual voyage decision.

Every returned port spec carries a ``data_source`` per field group so
callers always know which figures are "your data" vs. "prototype reference".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("freightai.optimization.port_optimizer")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTS_PATH = PROJECT_ROOT / "data" / "processed" / "ports_clean.csv"

# ---------------------------------------------------------------------------
# Supported destination ports
# ---------------------------------------------------------------------------

SUPPORTED_PORTS: List[str] = [
    "Paradip",
    "Visakhapatnam",
    "Gangavaram",
    "Gopalpur",
    "Dhamra",
    "Sagar-Sandheads",
    "Haldia",
]

# Indicative reference specs — see module docstring. All linear dimensions
# in metres, DWT in metric tons, cargo_handling_rate in metric tons/day,
# turnaround/waiting in hours. Compiled from public port-authority and
# port-reference sources; treat as planning-level, not authoritative.
REFERENCE_PORT_SPECS: Dict[str, Dict] = {
    "Paradip": {
        "max_draft": 16.5, "max_loa": 300.0, "max_beam": 46.0, "max_dwt": 155_000,
        "channel_depth": 18.7, "berth_count": 18, "cargo_handling_rate": 35_000,
        "avg_turnaround_hours": 36, "avg_waiting_hours": 18,
        "tidal_restriction": False, "congestion_level": "MEDIUM",
        "note": "Major deep-water port; Capesize-enabled since dredging upgrades.",
    },
    "Visakhapatnam": {
        "max_draft": 17.0, "max_loa": 300.0, "max_beam": 45.0, "max_dwt": 150_000,
        "channel_depth": 18.0, "berth_count": 28, "cargo_handling_rate": 30_000,
        "avg_turnaround_hours": 40, "avg_waiting_hours": 20,
        "tidal_restriction": False, "congestion_level": "HIGH",
        "note": "One of India's busiest major ports (inner + outer harbour); higher congestion is typical.",
    },
    "Gangavaram": {
        "max_draft": 18.5, "max_loa": 300.0, "max_beam": 48.0, "max_dwt": 200_000,
        "channel_depth": 20.0, "berth_count": 9, "cargo_handling_rate": 40_000,
        "avg_turnaround_hours": 30, "avg_waiting_hours": 10,
        "tidal_restriction": False, "congestion_level": "LOW",
        "note": "India's deepest privately-run port; Super-Capesize capable. Some sources report up to ~21m draft.",
    },
    "Gopalpur": {
        "max_draft": 14.5, "max_loa": 230.0, "max_beam": 32.3, "max_dwt": 95_000,
        "channel_depth": 15.5, "berth_count": 3, "cargo_handling_rate": 20_000,
        "avg_turnaround_hours": 30, "avg_waiting_hours": 8,
        "tidal_restriction": False, "congestion_level": "LOW",
        "note": "Smaller all-weather port; effectively Panamax-scale, not Capesize.",
    },
    "Dhamra": {
        "max_draft": 18.0, "max_loa": 290.0, "max_beam": 47.0, "max_dwt": 180_000,
        "channel_depth": 19.0, "berth_count": 8, "cargo_handling_rate": 38_000,
        "avg_turnaround_hours": 32, "avg_waiting_hours": 12,
        "tidal_restriction": False, "congestion_level": "LOW",
        "note": "Deep-water, storm-sheltered port; Capesize-capable.",
    },
    "Sagar-Sandheads": {
        "max_draft": 20.0, "max_loa": 330.0, "max_beam": 50.0, "max_dwt": 200_000,
        "channel_depth": 20.0, "berth_count": 0, "cargo_handling_rate": 15_000,
        "avg_turnaround_hours": 48, "avg_waiting_hours": 24,
        "tidal_restriction": True, "congestion_level": "MEDIUM",
        "note": (
            "NOT a conventional berthing port — an open-water anchorage/lightering point "
            "(ship-to-ship transfer) serving Haldia/Kolkata for vessels too large for the "
            "Hooghly river approach. 'berth_count' is 0 by design; handling is via STS/floating cranes."
        ),
    },
    "Haldia": {
        "max_draft": 9.1, "max_loa": 240.0, "max_beam": 32.26, "max_dwt": 50_000,
        "channel_depth": 12.5, "berth_count": 12, "cargo_handling_rate": 18_000,
        "avg_turnaround_hours": 48, "avg_waiting_hours": 30,
        "tidal_restriction": True, "congestion_level": "HIGH",
        "note": (
            "Shallow river/tide-gate dock on the Hooghly; larger vessels (e.g. Panamax) can only "
            "call partially loaded (~40-50% capacity) and must transit locks on the tide."
        ),
    },
}

# Fields treated as hard PASS/FAIL compatibility checks against a vessel.
COMPATIBILITY_FIELDS = ["draft", "loa", "beam", "dwt", "channel_depth"]

# Fields reported for context/operational planning, not pass/fail.
INFO_FIELDS = [
    "berth_count", "cargo_handling_rate", "avg_turnaround_hours",
    "avg_waiting_hours", "tidal_restriction", "congestion_level",
]

VESSEL_FIELD_TO_PORT_FIELD = {
    "draft": "max_draft",
    "loa": "max_loa",
    "beam": "max_beam",
    "dwt": "max_dwt",
    "channel_depth": "channel_depth",  # vessel draft vs. channel depth (safety margin)
}

PORT_COLUMN_KEYWORDS = {
    "name": ["port_name", "port", "name"],
    "max_draft": ["max_draft", "draft", "draught"],
    "max_loa": ["max_loa", "max_length", "loa", "length"],
    "max_beam": ["max_beam", "beam", "width"],
    "max_dwt": ["max_dwt", "max_capacity", "dwt", "capacity"],
    "channel_depth": ["channel_depth", "channel", "approach_depth"],
    "berth_count": ["berth_count", "berths", "no_of_berths"],
    "cargo_handling_rate": ["cargo_handling_rate", "handling_rate", "loading_rate"],
    "avg_turnaround_hours": ["turnaround"],
    "avg_waiting_hours": ["waiting"],
    "tidal_restriction": ["tidal", "tide"],
    "congestion_level": ["congestion"],
}


def _find_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    lower_map = {c: str(c).lower() for c in df.columns}
    for kw in keywords:
        for col, low in lower_map.items():
            if kw in low:
                return col
    return None


def _detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    detected: Dict[str, Optional[str]] = {}
    used_cols = set()
    for field, keywords in PORT_COLUMN_KEYWORDS.items():
        remaining = [c for c in df.columns if c not in used_cols]
        col = _find_column(df[remaining] if remaining else df, keywords)
        detected[field] = col
        if col:
            used_cols.add(col)
    return detected


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ports_data(path: Path = PORTS_PATH) -> Optional[pd.DataFrame]:
    """Load ``ports_clean.csv`` if present. Returns ``None`` (never raises)
    if missing/empty/unreadable — callers fall back to reference specs.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load ports data: %s", exc)
        return None


def _lookup_row(ports_df: pd.DataFrame, port_name: str, name_col: str) -> Optional[pd.Series]:
    names = ports_df[name_col].astype(str).str.strip().str.lower()
    target = port_name.strip().lower()
    match_idx = names[names == target].index
    if len(match_idx) == 0:
        match_idx = names[names.str.contains(target, na=False, regex=False)].index
    return ports_df.loc[match_idx[0]] if len(match_idx) else None


# ---------------------------------------------------------------------------
# Public spec lookup
# ---------------------------------------------------------------------------

def get_port_spec(port_name: str, ports_path: Path = PORTS_PATH) -> Dict:
    """Return the full spec for a supported destination port, merging
    ``ports_clean.csv`` (preferred, field by field) with
    :data:`REFERENCE_PORT_SPECS` (fallback).

    Raises ``ValueError`` if ``port_name`` is not one of
    :data:`SUPPORTED_PORTS` — this module only supports the seven named
    Indian East Coast ports, by design.
    """
    canonical = _canonicalize_port_name(port_name)
    if canonical is None:
        raise ValueError(
            f"Unsupported destination port '{port_name}'. Supported ports: {SUPPORTED_PORTS}."
        )

    reference = REFERENCE_PORT_SPECS[canonical]
    spec = dict(reference)
    spec["port_name"] = canonical
    field_sources = {field: "reference_default" for field in reference if field != "note"}

    ports_df = load_ports_data(ports_path)
    if ports_df is not None:
        cols = _detect_columns(ports_df)
        name_col = cols.get("name")
        if name_col is not None:
            row = _lookup_row(ports_df, canonical, name_col)
            if row is not None:
                for field in field_sources:
                    col = cols.get(field)
                    if col is None or col not in row.index:
                        continue
                    val = row.get(col)
                    if pd.isna(val):
                        continue
                    if field in ("tidal_restriction",):
                        spec[field] = bool(val) if not isinstance(val, str) else val.strip().lower() in ("1", "true", "yes", "y")
                    elif field == "congestion_level":
                        spec[field] = str(val).strip().upper()
                    else:
                        parsed = pd.to_numeric(val, errors="coerce")
                        if pd.notna(parsed):
                            spec[field] = float(parsed)
                    field_sources[field] = "ports_clean.csv"

    spec["field_sources"] = field_sources
    return spec


def _canonicalize_port_name(port_name: str) -> Optional[str]:
    if not port_name:
        return None
    normalized = str(port_name).strip().lower().replace("_", "-").replace(" ", "-")
    for supported in SUPPORTED_PORTS:
        if supported.lower().replace(" ", "-") == normalized:
            return supported
    # tolerate "Sagar Sandheads" / "Sandheads" / "Sagar" style variants
    if "sagar" in normalized or "sandhead" in normalized:
        return "Sagar-Sandheads"
    return None


def list_supported_ports() -> List[str]:
    """Return the list of destination ports this module supports."""
    return list(SUPPORTED_PORTS)


# ---------------------------------------------------------------------------
# Vessel-vs-port compatibility table
# ---------------------------------------------------------------------------

def _pass_fail(vessel_value: Optional[float], limit: Optional[float]) -> str:
    if vessel_value is None or limit is None or pd.isna(vessel_value) or pd.isna(limit):
        return "UNKNOWN"
    return "PASS" if vessel_value <= limit else "FAIL"


def compare_vessel_to_port(vessel_spec: Dict, port_name: str, ports_path: Path = PORTS_PATH) -> Dict:
    """Compare a vessel's specifications against a destination port's
    limits and return a full parameter-by-parameter table.

    Parameters
    ----------
    vessel_spec:
        Dict with (a subset of) keys ``dwt``, ``loa``, ``beam``, ``draft``
        (metres/metric tons) — the same shape produced by
        ``vessel_optimizer.build_class_profiles()``.
    port_name:
        One of :data:`SUPPORTED_PORTS`.

    Returns
    -------
    {
        "port": {...full port spec, incl. field_sources...},
        "compatibility_table": [
            {"parameter": "draft", "vessel_value": .., "port_limit": .., "status": "PASS"},
            ...,   # draft, loa, beam, dwt, channel_depth  (PASS/FAIL/UNKNOWN)
            {"parameter": "berth_count", "value": .., "status": "INFO"},
            ...,   # informational port attributes, no vessel comparison
        ],
        "compatible": bool,       # True only if every hard check is PASS (no FAIL)
        "verified": bool,         # True only if no UNKNOWN among hard checks
        "warnings": [str, ...],
    }
    """
    port_spec = get_port_spec(port_name, ports_path=ports_path)
    table: List[Dict] = []
    warnings: List[str] = []

    for field in COMPATIBILITY_FIELDS:
        vessel_val = vessel_spec.get(field)
        port_field = VESSEL_FIELD_TO_PORT_FIELD[field]
        port_limit = port_spec.get(port_field)
        # channel_depth is compared against vessel draft (a safety-margin check)
        compare_val = vessel_spec.get("draft") if field == "channel_depth" else vessel_val
        status = _pass_fail(compare_val, port_limit)
        table.append({
            "parameter": field,
            "vessel_value": compare_val,
            "port_limit": port_limit,
            "status": status,
        })
        if status == "FAIL":
            warnings.append(
                f"Vessel {field} ({compare_val}) exceeds {port_spec['port_name']}'s {port_field} ({port_limit})."
            )

    for field in INFO_FIELDS:
        table.append({"parameter": field, "value": port_spec.get(field), "status": "INFO"})

    if port_spec.get("tidal_restriction"):
        warnings.append(
            f"{port_spec['port_name']} has tidal access restrictions — berthing/departure windows are tide-dependent."
        )
    if port_spec.get("congestion_level") == "HIGH":
        warnings.append(f"{port_spec['port_name']} currently has a HIGH congestion level — expect longer waiting time.")
    if port_spec["port_name"] == "Sagar-Sandheads":
        warnings.append(
            "Sagar-Sandheads is an anchorage/lightering point, not a conventional berth — "
            "cargo handling occurs via ship-to-ship transfer."
        )

    hard_checks = [row for row in table if row["parameter"] in COMPATIBILITY_FIELDS]
    compatible = all(row["status"] != "FAIL" for row in hard_checks)
    verified = all(row["status"] != "UNKNOWN" for row in hard_checks)

    return {
        "port": port_spec,
        "compatibility_table": table,
        "compatible": compatible,
        "verified": verified,
        "warnings": warnings,
    }


if __name__ == "__main__":
    import json

    demo_vessel = {"dwt": 82_000, "loa": 225.0, "beam": 32.3, "draft": 14.0, "speed": 14.2}
    print("\n=== FreightAI Port Compatibility Engine — demo (Panamax vs. all 7 ports) ===")
    for port in list_supported_ports():
        result = compare_vessel_to_port(demo_vessel, port)
        print(f"{port:16s} compatible={result['compatible']!s:5s} verified={result['verified']!s:5s} "
              f"warnings={len(result['warnings'])}")
    print("\nFull detail for Haldia:")
    print(json.dumps(compare_vessel_to_port(demo_vessel, "Haldia"), indent=2, default=str))
