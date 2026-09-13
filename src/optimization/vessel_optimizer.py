"""
vessel_optimizer.py
====================
Vessel Optimization Engine for FreightAI.

Recommends the most suitable bulk carrier class — Handysize, Supramax,
Panamax, or Capesize — for a given cargo quantity, commodity, origin port,
and destination port, based on:

    * cargo capacity suitability     (does the vessel fit the cargo well?)
    * port compatibility             (draft / LOA / beam / DWT vs. port limits)
    * estimated voyage efficiency    (cost + speed + turnaround proxies)
    * operational risk               (vessel-age based proxy)

This module does NOT build the dashboard. It returns plain, JSON-friendly
Python data (dicts) so a future dashboard module can render it directly.

------------------------------------------------------------------------
IMPORTANT — prototype scope
------------------------------------------------------------------------
The suitability score is a weighted blend of simplified heuristic
components (see PROTOTYPE_WEIGHTS below). It is a decision-SUPPORT
prototype, not a full voyage-estimation or naval-architecture engine —
voyage cost, turnaround, and operational risk are approximated from
typical class-level assumptions and vessel age, not from real bunker
prices, port tariffs, or berth schedules. Every result carries a
`disclaimer` field saying so. The weights themselves are explicitly
labeled prototype and are trivially reconfigurable (see
:func:`recommend_vessel_class`'s ``weights`` parameter).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("freightai.optimization.vessel_optimizer")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VESSELS_PATH = PROJECT_ROOT / "data" / "processed" / "vessels_clean.csv"
PORTS_PATH = PROJECT_ROOT / "data" / "processed" / "ports_clean.csv"
COMMODITIES_PATH = PROJECT_ROOT / "data" / "processed" / "commodities_clean.csv"

# ---------------------------------------------------------------------------
# Vessel classes and reference (typical/industry-standard) specs
# ---------------------------------------------------------------------------

VESSEL_CLASSES: List[str] = ["Handysize", "Supramax", "Panamax", "Capesize"]

# DWT bucket boundaries (metric tons) used to classify individual vessels
# from the fleet dataset into one of the four supported classes.
CLASS_DWT_RANGES = {
    "Handysize": (10_000, 40_000),
    "Supramax": (40_000, 60_000),
    "Panamax": (60_000, 100_000),
    "Capesize": (100_000, 220_000),
}

# Typical/reference specs per class — used ONLY as a fallback when the
# fleet dataset (vessels_clean.csv) is missing or has no vessel of that
# class. Clearly a prototype approximation, not vessel-specific data.
REFERENCE_CLASS_SPECS = {
    "Handysize": {"dwt": 35_000, "loa": 180.0, "beam": 30.0, "draft": 10.5, "speed": 13.5, "year_built": None},
    "Supramax": {"dwt": 58_000, "loa": 190.0, "beam": 32.3, "draft": 12.0, "speed": 14.0, "year_built": None},
    "Panamax": {"dwt": 82_000, "loa": 225.0, "beam": 32.3, "draft": 14.0, "speed": 14.2, "year_built": None},
    "Capesize": {"dwt": 180_000, "loa": 292.0, "beam": 45.0, "draft": 18.0, "speed": 14.5, "year_built": None},
}

# Class-level "typical" economics used by the voyage-cost and turnaround
# proxies (0-100 scale, higher = better). These represent well-known
# rules of thumb in dry-bulk shipping (bigger ships = better economies of
# scale per tonne-mile, but slower/more restricted port turnaround) and
# are NOT calculated from real bunker/port-tariff/berth data.
CLASS_COST_EFFICIENCY_BASE = {"Handysize": 70, "Supramax": 78, "Panamax": 85, "Capesize": 92}
CLASS_TURNAROUND_BASE = {"Handysize": 90, "Supramax": 82, "Panamax": 75, "Capesize": 65}

# ---------------------------------------------------------------------------
# Suitability weights — PROTOTYPE, configurable
# ---------------------------------------------------------------------------
# These weights are an explicit prototype/starting point, not a validated
# commercial model. Pass a custom `weights` dict (same keys, summing to 1.0)
# to recommend_vessel_class() to override them.
PROTOTYPE_WEIGHTS: Dict[str, float] = {
    "port_compatibility": 0.30,
    "cargo_capacity": 0.25,
    "voyage_cost": 0.20,
    "speed": 0.10,
    "turnaround": 0.10,
    "operational_risk": 0.05,
}
assert abs(sum(PROTOTYPE_WEIGHTS.values()) - 1.0) < 1e-9, "PROTOTYPE_WEIGHTS must sum to 1.0"


# ---------------------------------------------------------------------------
# Generic column auto-detection (vessels_clean.csv / ports_clean.csv have
# no fixed schema beyond snake_case — the data pipeline module only
# standardizes column *names*, it doesn't rename them to a fixed contract
# the way it does for freight_rate).
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Return the first column of ``df`` whose lowercase name contains one
    of ``keywords`` (checked in priority order). ``None`` if no match.
    """
    lower_map = {c: str(c).lower() for c in df.columns}
    for kw in keywords:
        for col, low in lower_map.items():
            if kw in low:
                return col
    return None


VESSEL_COLUMN_KEYWORDS = {
    "name": ["vessel_name", "ship_name", "name"],
    "dwt": ["dwt", "deadweight"],
    "loa": ["loa", "length_overall", "length"],
    "beam": ["beam", "width"],
    "draft": ["draft", "draught"],
    "speed": ["speed", "knots"],
    "year_built": ["year_built", "build_year", "built", "year"],
}

PORT_COLUMN_KEYWORDS = {
    "name": ["port_name", "port", "name"],
    "max_draft": ["max_draft", "draft", "draught"],
    "max_loa": ["max_loa", "max_length", "loa", "length"],
    "max_beam": ["max_beam", "beam", "width"],
    "max_dwt": ["max_dwt", "max_capacity", "dwt", "capacity"],
}


def _detect_columns(df: pd.DataFrame, keyword_map: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    """Apply :func:`_find_column` for every logical field in ``keyword_map``,
    avoiding assigning the same physical column to two different fields.
    """
    detected: Dict[str, Optional[str]] = {}
    used_cols = set()
    for field, keywords in keyword_map.items():
        remaining_cols = [c for c in df.columns if c not in used_cols]
        sub_df = df[remaining_cols] if remaining_cols else df
        col = _find_column(sub_df, keywords)
        detected[field] = col
        if col:
            used_cols.add(col)
    return detected


# ---------------------------------------------------------------------------
# Data loading (never crashes — missing files degrade to reference data)
# ---------------------------------------------------------------------------

def load_vessels(path: Path = VESSELS_PATH) -> Optional[pd.DataFrame]:
    """Load the cleaned vessel fleet dataset. Returns ``None`` (not an
    exception) if it's missing/empty — callers fall back to reference specs.
    """
    if not path.exists():
        logger.warning("Vessel fleet dataset not found at %s — using reference class specs only.", path)
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load vessel fleet dataset: %s", exc)
        return None


def load_ports(path: Path = PORTS_PATH) -> Optional[pd.DataFrame]:
    """Load the cleaned ports dataset. Returns ``None`` if missing/empty —
    callers mark port-compatibility checks as unverifiable rather than crash.
    """
    if not path.exists():
        logger.warning("Ports dataset not found at %s — port compatibility cannot be verified.", path)
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load ports dataset: %s", exc)
        return None


def load_commodities(path: Path = COMMODITIES_PATH) -> Optional[pd.DataFrame]:
    """Load the cleaned commodities dataset, if present. Optional — only
    used to refine cargo-capacity notes (e.g. a stowage-factor column) when
    available; core logic works fine without it.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load commodities dataset: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Fleet -> vessel-class profiles
# ---------------------------------------------------------------------------

def classify_dwt(dwt: float) -> Optional[str]:
    """Map a DWT value to one of the four supported vessel classes."""
    for cls, (low, high) in CLASS_DWT_RANGES.items():
        if low <= dwt < high:
            return cls
    if dwt >= CLASS_DWT_RANGES["Capesize"][0]:
        return "Capesize"
    return None


def build_class_profiles(vessels_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
    """Build one representative spec per vessel class.

    If real fleet data is available, each class's spec is the median of
    that class's vessels in the fleet (``data_source: "fleet_data"``,
    with ``n_vessels`` reported). If the fleet has no vessel of a given
    class (or no fleet data at all), that class falls back to
    :data:`REFERENCE_CLASS_SPECS` (``data_source: "reference_default"``).
    """
    profiles: Dict[str, Dict] = {}

    fleet_by_class: Dict[str, List[Dict]] = {cls: [] for cls in VESSEL_CLASSES}
    cols: Dict[str, Optional[str]] = {}

    if vessels_df is not None:
        cols = _detect_columns(vessels_df, VESSEL_COLUMN_KEYWORDS)
        dwt_col = cols.get("dwt")
        if dwt_col is None:
            logger.warning("Could not detect a DWT column in vessels_clean.csv — cannot classify fleet.")
        else:
            for _, row in vessels_df.iterrows():
                dwt_val = pd.to_numeric(row.get(dwt_col), errors="coerce")
                if pd.isna(dwt_val):
                    continue
                vessel_class = classify_dwt(float(dwt_val))
                if vessel_class is None:
                    continue
                fleet_by_class[vessel_class].append({
                    "dwt": float(dwt_val),
                    "loa": pd.to_numeric(row.get(cols.get("loa")), errors="coerce") if cols.get("loa") else np.nan,
                    "beam": pd.to_numeric(row.get(cols.get("beam")), errors="coerce") if cols.get("beam") else np.nan,
                    "draft": pd.to_numeric(row.get(cols.get("draft")), errors="coerce") if cols.get("draft") else np.nan,
                    "speed": pd.to_numeric(row.get(cols.get("speed")), errors="coerce") if cols.get("speed") else np.nan,
                    "year_built": pd.to_numeric(row.get(cols.get("year_built")), errors="coerce") if cols.get("year_built") else np.nan,
                })

    for cls in VESSEL_CLASSES:
        vessels_in_class = fleet_by_class.get(cls, [])
        if vessels_in_class:
            df = pd.DataFrame(vessels_in_class)
            profiles[cls] = {
                "dwt": float(df["dwt"].median()),
                "loa": float(df["loa"].median()) if df["loa"].notna().any() else REFERENCE_CLASS_SPECS[cls]["loa"],
                "beam": float(df["beam"].median()) if df["beam"].notna().any() else REFERENCE_CLASS_SPECS[cls]["beam"],
                "draft": float(df["draft"].median()) if df["draft"].notna().any() else REFERENCE_CLASS_SPECS[cls]["draft"],
                "speed": float(df["speed"].median()) if df["speed"].notna().any() else REFERENCE_CLASS_SPECS[cls]["speed"],
                "year_built": float(df["year_built"].median()) if df["year_built"].notna().any() else None,
                "data_source": "fleet_data",
                "n_vessels": int(len(df)),
            }
        else:
            profiles[cls] = {**REFERENCE_CLASS_SPECS[cls], "data_source": "reference_default", "n_vessels": 0}

    return profiles


# ---------------------------------------------------------------------------
# Port spec lookup
# ---------------------------------------------------------------------------

def find_port_spec(ports_df: Optional[pd.DataFrame], port_name: str) -> Optional[Dict]:
    """Look up a port's max draft/LOA/beam/DWT by (case-insensitive,
    substring-tolerant) name match. Returns ``None`` if the ports dataset
    is unavailable or the port can't be found — callers must treat that as
    "unverifiable", not "incompatible".
    """
    if ports_df is None or not port_name:
        return None

    cols = _detect_columns(ports_df, PORT_COLUMN_KEYWORDS)
    name_col = cols.get("name")
    if name_col is None:
        logger.warning("Could not detect a port-name column in ports_clean.csv.")
        return None

    target = str(port_name).strip().lower()
    names = ports_df[name_col].astype(str).str.strip().str.lower()

    match_idx = names[names == target].index
    if len(match_idx) == 0:
        match_idx = names[names.str.contains(target, na=False, regex=False)].index
    if len(match_idx) == 0:
        logger.warning("Port '%s' not found in ports_clean.csv.", port_name)
        return None

    row = ports_df.loc[match_idx[0]]

    def _val(field):
        col = cols.get(field)
        if col is None:
            return None
        v = pd.to_numeric(row.get(col), errors="coerce")
        return float(v) if pd.notna(v) else None

    return {
        "port_name": str(row.get(name_col)),
        "max_draft": _val("max_draft"),
        "max_loa": _val("max_loa"),
        "max_beam": _val("max_beam"),
        "max_dwt": _val("max_dwt"),
    }


# ---------------------------------------------------------------------------
# Suitability component evaluators (each returns a 0-100 score + details)
# ---------------------------------------------------------------------------

def evaluate_port_param(vessel_value: Optional[float], port_max: Optional[float]) -> str:
    """PASS / FAIL / UNKNOWN for a single vessel-vs-port-limit parameter."""
    if vessel_value is None or port_max is None:
        return "UNKNOWN"
    return "PASS" if vessel_value <= port_max else "FAIL"


def evaluate_port_compatibility_for_port(vessel_spec: Dict, port_spec: Optional[Dict]) -> Dict:
    """Evaluate one port (origin or destination) against a vessel spec.

    Returns per-parameter PASS/FAIL/UNKNOWN plus an overall status for
    this single port (compatible = no FAIL among the four parameters).
    """
    if port_spec is None:
        # No known limits means no known FAILURE either — this is
        # "unverified", not "incompatible". Compatibility stays True
        # (optimistic) but verified=False makes the uncertainty explicit,
        # and the score penalty for UNKNOWN checks still applies below.
        return {
            "port_name": None,
            "verified": False,
            "checks": {p: "UNKNOWN" for p in ("draft", "loa", "beam", "dwt")},
            "compatible": True,
            "note": "Port not found in ports_clean.csv (or ports dataset unavailable) — compatibility unverified.",
        }

    checks = {
        "draft": evaluate_port_param(vessel_spec.get("draft"), port_spec.get("max_draft")),
        "loa": evaluate_port_param(vessel_spec.get("loa"), port_spec.get("max_loa")),
        "beam": evaluate_port_param(vessel_spec.get("beam"), port_spec.get("max_beam")),
        "dwt": evaluate_port_param(vessel_spec.get("dwt"), port_spec.get("max_dwt")),
    }
    has_fail = "FAIL" in checks.values()
    has_unknown = "UNKNOWN" in checks.values()
    return {
        "port_name": port_spec.get("port_name"),
        "verified": not has_unknown,
        "checks": checks,
        "compatible": (not has_fail) if not has_unknown else (not has_fail),
        "note": None if not has_unknown else "One or more port limits unavailable — treated as not blocking, but unverified.",
    }


def evaluate_port_compatibility(vessel_spec: Dict, origin_spec: Optional[Dict], destination_spec: Optional[Dict]) -> Dict:
    """Combine origin + destination port checks. The vessel must clear
    BOTH ports to be considered port-compatible overall.
    """
    origin_eval = evaluate_port_compatibility_for_port(vessel_spec, origin_spec)
    destination_eval = evaluate_port_compatibility_for_port(vessel_spec, destination_spec)

    compatible = bool(origin_eval["compatible"]) and bool(destination_eval["compatible"])
    n_fail = sum(1 for e in (origin_eval, destination_eval) for v in e["checks"].values() if v == "FAIL")
    n_unknown = sum(1 for e in (origin_eval, destination_eval) for v in e["checks"].values() if v == "UNKNOWN")

    score = 100 - (n_fail * 25) - (n_unknown * 5)
    score = max(0, min(100, score))

    return {
        "score": score,
        "compatible": compatible,
        "origin": origin_eval,
        "destination": destination_eval,
    }


def evaluate_cargo_capacity(cargo_quantity: float, vessel_dwt: float) -> Dict:
    """Score how well the cargo quantity fits the vessel's DWT.

    Prototype heuristic: cargo exceeding DWT is infeasible (score 0).
    Utilization in [0.85, 0.98] is treated as the efficient sweet spot
    (score 100); utilization just under 0.85 or just over 0.98 is
    penalized more gently than a hard cutoff.
    """
    if vessel_dwt is None or vessel_dwt <= 0:
        return {"score": 0, "utilization": None, "feasible": False, "note": "Vessel DWT unknown."}

    if cargo_quantity > vessel_dwt:
        return {
            "score": 0, "utilization": round(cargo_quantity / vessel_dwt, 3), "feasible": False,
            "note": f"Cargo quantity ({cargo_quantity:,.0f} t) exceeds vessel DWT ({vessel_dwt:,.0f} t).",
        }

    utilization = cargo_quantity / vessel_dwt
    if 0.85 <= utilization <= 0.98:
        score = 100.0
        note = "Cargo quantity efficiently utilizes vessel capacity."
    elif utilization > 0.98:
        score = 90.0
        note = "Cargo quantity nearly maxes out vessel capacity — little margin."
    else:
        # Below the efficient range: score decays toward a floor as
        # utilization drops (an under-filled large ship is feasible but inefficient).
        score = max(20.0, 100.0 - (0.85 - utilization) / 0.85 * 80.0)
        note = "Vessel is larger than needed for this cargo quantity — capacity is under-utilized."

    return {"score": round(score, 1), "utilization": round(utilization, 3), "feasible": True, "note": note}


def evaluate_voyage_cost(vessel_class: str, utilization: Optional[float]) -> Dict:
    """Prototype voyage-cost-efficiency proxy: blends a class-level
    economies-of-scale baseline with how well the cargo utilizes the ship
    (a half-empty ship is less cost-efficient per tonne regardless of class).
    NOT derived from real bunker prices, freight-cost models, or route distance.
    """
    base = CLASS_COST_EFFICIENCY_BASE.get(vessel_class, 75)
    utilization_component = (utilization or 0) * 100
    score = round(0.6 * base + 0.4 * min(100.0, utilization_component), 1)
    return {
        "score": score,
        "note": (
            f"Prototype proxy: {vessel_class} class economies-of-scale baseline "
            f"({base}/100) blended with cargo utilization — not a real cost model."
        ),
    }


def evaluate_speed(vessel_speed: Optional[float]) -> Dict:
    """Score vessel service speed on a simple 10-16 knot reference band
    typical of dry-bulk carriers. Higher speed = shorter voyage time."""
    if vessel_speed is None or pd.isna(vessel_speed):
        return {"score": 50.0, "note": "Vessel speed unknown — neutral default score applied."}
    score = (float(vessel_speed) - 10.0) / (16.0 - 10.0) * 100.0
    score = round(max(0.0, min(100.0, score)), 1)
    return {"score": score, "note": f"Based on service speed of {vessel_speed:.1f} knots (reference band: 10-16 kn)."}


def evaluate_turnaround(vessel_class: str) -> Dict:
    """Prototype port-turnaround proxy from typical class-level assumptions
    (smaller vessels generally turn around faster). NOT derived from real
    berth schedules or port operations data for the specific origin/destination.
    """
    score = CLASS_TURNAROUND_BASE.get(vessel_class, 75)
    return {
        "score": score,
        "note": f"Prototype proxy based on typical {vessel_class}-class port turnaround expectations.",
    }


def evaluate_operational_risk(year_built: Optional[float], reference_year: Optional[int] = None) -> Dict:
    """Prototype operational-risk proxy based on vessel age. Older vessels
    are treated as marginally higher risk (more prone to mechanical issues,
    stricter port state control scrutiny) — a simplification; it does not
    account for actual maintenance/inspection/incident history.
    """
    reference_year = reference_year or date.today().year
    if year_built is None or pd.isna(year_built):
        return {"score": 70.0, "age_years": None, "note": "Vessel build year unknown — neutral default score applied."}

    age = reference_year - int(year_built)
    if age <= 10:
        score = 100.0
    elif age <= 20:
        score = 100.0 - (age - 10) * 4.0  # down to 60 at age 20
    else:
        score = max(30.0, 60.0 - (age - 20) * 2.0)
    return {"score": round(score, 1), "age_years": age, "note": f"Vessel age proxy: ~{age} years old."}


# ---------------------------------------------------------------------------
# Full per-class evaluation + recommendation
# ---------------------------------------------------------------------------

def evaluate_vessel_class(
    vessel_class: str,
    vessel_spec: Dict,
    cargo_quantity: float,
    origin_port_spec: Optional[Dict],
    destination_port_spec: Optional[Dict],
    weights: Dict[str, float],
) -> Dict:
    """Run every suitability component for one vessel class and combine
    them into a single weighted suitability score.
    """
    port = evaluate_port_compatibility(vessel_spec, origin_port_spec, destination_port_spec)
    cargo = evaluate_cargo_capacity(cargo_quantity, vessel_spec.get("dwt"))
    voyage_cost = evaluate_voyage_cost(vessel_class, cargo.get("utilization"))
    speed = evaluate_speed(vessel_spec.get("speed"))
    turnaround = evaluate_turnaround(vessel_class)
    risk = evaluate_operational_risk(vessel_spec.get("year_built"))

    component_scores = {
        "port_compatibility": port["score"],
        "cargo_capacity": cargo["score"],
        "voyage_cost": voyage_cost["score"],
        "speed": speed["score"],
        "turnaround": turnaround["score"],
        "operational_risk": risk["score"],
    }
    suitability_score = round(sum(component_scores[k] * weights[k] for k in weights), 2)

    reasons: List[str] = []
    if not cargo.get("feasible", True):
        reasons.append(cargo["note"])
    elif cargo.get("note"):
        reasons.append(cargo["note"])
    if port["compatible"] is False:
        failed = [
            f"{side}:{param}" for side, ev in (("origin", port["origin"]), ("destination", port["destination"]))
            for param, status in ev["checks"].items() if status == "FAIL"
        ]
        reasons.append(f"Port compatibility FAILED on: {', '.join(failed) if failed else 'unspecified parameter(s)'}.")
    elif not port["origin"]["verified"] or not port["destination"]["verified"]:
        reasons.append("Port compatibility could not be fully verified (missing port data).")
    else:
        reasons.append("Vessel clears origin and destination port draft/LOA/beam/DWT limits.")
    reasons.append(risk["note"])

    return {
        "vessel_class": vessel_class,
        "vessel_spec": vessel_spec,
        "suitability_score": suitability_score,
        "compatible": port["compatible"],
        "components": {
            "port_compatibility": port,
            "cargo_capacity": cargo,
            "voyage_cost": voyage_cost,
            "speed": speed,
            "turnaround": turnaround,
            "operational_risk": risk,
        },
        "reasons": reasons,
    }


def recommend_vessel_class(
    cargo_quantity: float,
    commodity: str,
    origin: str,
    destination: str,
    weights: Optional[Dict[str, float]] = None,
    vessels_path: Path = VESSELS_PATH,
    ports_path: Path = PORTS_PATH,
) -> Dict:
    """Recommend the most suitable bulk carrier class for a shipment.

    Parameters
    ----------
    cargo_quantity: cargo weight in metric tons.
    commodity: commodity name (used for context/notes; see module docstring
        for the scope of commodity-aware logic in this prototype).
    origin, destination: port names, matched against ``ports_clean.csv``.
    weights: optional override for :data:`PROTOTYPE_WEIGHTS` (same keys,
        should sum to 1.0). Explicitly supported so this scoring model
        stays configurable rather than hard-coded.

    Returns
    -------
    A structured dict:
        {
            "inputs": {...},
            "weights_used": {...},                # labeled prototype
            "evaluations": {class_name: {...}, ...},   # all 4 classes, always
            "recommended_vessel_type": str,             # highest-scoring class overall
            "recommended_compatible": bool,
            "alternative_vessel_type": str | None,      # best compatible class, if the top pick isn't compatible
            "reasons": [str, ...],
            "data_sources": {...},
            "disclaimer": str,
        }

    Never raises on missing vessels_clean.csv / ports_clean.csv — falls
    back to reference class specs and flags port compatibility as
    unverified instead.
    """
    if cargo_quantity is None or cargo_quantity <= 0:
        raise ValueError(f"cargo_quantity must be a positive number, got {cargo_quantity!r}.")
    if not origin or not destination:
        raise ValueError("Both 'origin' and 'destination' port names are required.")

    weights = weights or PROTOTYPE_WEIGHTS
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0, got {sum(weights.values())}.")

    vessels_df = load_vessels(vessels_path)
    ports_df = load_ports(ports_path)

    class_profiles = build_class_profiles(vessels_df)
    origin_spec = find_port_spec(ports_df, origin)
    destination_spec = find_port_spec(ports_df, destination)

    evaluations: Dict[str, Dict] = {}
    for cls in VESSEL_CLASSES:
        evaluations[cls] = evaluate_vessel_class(
            cls, class_profiles[cls], cargo_quantity, origin_spec, destination_spec, weights,
        )

    ranked = sorted(evaluations.values(), key=lambda e: e["suitability_score"], reverse=True)
    top_pick = ranked[0]

    alternative = None
    if not top_pick["compatible"]:
        compatible_candidates = [e for e in ranked if e["compatible"]]
        if compatible_candidates:
            alternative = compatible_candidates[0]["vessel_class"]

    overall_reasons = [
        f"{top_pick['vessel_class']} scored highest overall "
        f"({top_pick['suitability_score']}/100 under the current prototype weights).",
    ]
    if not top_pick["compatible"]:
        overall_reasons.append(
            f"{top_pick['vessel_class']} is NOT port-compatible for this origin/destination."
        )
        if alternative:
            overall_reasons.append(f"Recommending {alternative} as a compatible alternative instead.")
        else:
            overall_reasons.append("No evaluated vessel class is fully port-compatible for this route.")

    return {
        "inputs": {
            "cargo_quantity": cargo_quantity,
            "commodity": commodity,
            "origin": origin,
            "destination": destination,
        },
        "weights_used": {**weights, "_label": "PROTOTYPE_WEIGHTS — configurable, not a validated commercial model"},
        "evaluations": evaluations,
        "recommended_vessel_type": top_pick["vessel_class"],
        "recommended_compatible": top_pick["compatible"],
        "alternative_vessel_type": alternative,
        "reasons": overall_reasons,
        "data_sources": {
            "vessels": "fleet_data" if vessels_df is not None else "reference_default (vessels_clean.csv not found)",
            "origin_port": "ports_clean.csv" if origin_spec else "unavailable — compatibility unverified",
            "destination_port": "ports_clean.csv" if destination_spec else "unavailable — compatibility unverified",
        },
        "disclaimer": (
            "This is a prototype decision-support score using simplified, configurable "
            "heuristic weights (see weights_used) and typical class-level assumptions for "
            "voyage cost, turnaround, and operational risk. It is not a substitute for full "
            "voyage estimation, chartering, or naval-architecture analysis."
        ),
    }


if __name__ == "__main__":
    import json

    demo = recommend_vessel_class(
        cargo_quantity=55_000,
        commodity="Iron Ore",
        origin="Port Hedland",
        destination="Paradip",
    )
    print("\n=== FreightAI Vessel Optimization Engine — demo ===")
    print(f"Recommended: {demo['recommended_vessel_type']} "
          f"(compatible={demo['recommended_compatible']}, "
          f"alternative={demo['alternative_vessel_type']})")
    for cls, ev in demo["evaluations"].items():
        print(f"  {cls:10s} score={ev['suitability_score']:6.2f}  compatible={ev['compatible']}  "
              f"source={ev['vessel_spec'].get('data_source')}")
    print("\nFull result:")
    print(json.dumps(demo, indent=2, default=str))
