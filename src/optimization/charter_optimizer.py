"""
charter_optimizer.py
=====================
Charter Strategy Engine for FreightAI.

Recommends the most suitable chartering strategy — Spot, Short-Term,
Medium-Term, or Multiple-Voyage/COA — by combining outputs from the other
FreightAI engines (Freight Forecasting, Vessel Optimization, Port
Optimization, Risk) with direct market/cargo inputs.

Does not build the dashboard. Returns plain, JSON-friendly Python data.

------------------------------------------------------------------------
Prototype decision rules — not universal industry rules
------------------------------------------------------------------------
The scoring system below encodes commonly-cited, directionally-sensible
chartering heuristics (e.g. "lock in a longer charter ahead of a forecast
rate increase", "a Contract of Affreightment suits predictable repeat
voyages"). It is a transparent, additive point system — every point gained
or lost by a strategy is attached to a plain-English reason — but it is a
PROTOTYPE decision-support tool, not a validated commercial chartering
model. Real chartering decisions also depend on counterparty relationships,
financing, charter-party terms, and market intelligence this module has no
access to.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.forecasting.predict import TREND_THRESHOLD_PCT
from src.risk.risk_engine import THRESHOLDS as RISK_THRESHOLDS
from src.risk.risk_engine import _risk_from_thresholds, classify_level

# ---------------------------------------------------------------------------
# Supported strategies
# ---------------------------------------------------------------------------

STRATEGIES = ["SPOT", "SHORT_TERM", "MEDIUM_TERM", "COA"]

STRATEGY_DISPLAY_NAMES = {
    "SPOT": "Spot Charter",
    "SHORT_TERM": "Short-Term Charter",
    "MEDIUM_TERM": "Medium-Term Charter",
    "COA": "Multiple-Voyage / COA",
}

# Static, prototype advantages/disadvantages — general chartering
# characteristics, not derived from the specific inputs of any one call.
STRATEGY_PROS_CONS = {
    "SPOT": {
        "advantages": [
            "Maximum flexibility — no ongoing obligation beyond the single voyage",
            "Full ability to benefit if rates fall before the next fixture",
            "No long-term capital or counterparty commitment",
        ],
        "disadvantages": [
            "Full exposure to rising rates on every future fixture",
            "No guaranteed vessel availability when needed",
            "Rate must be re-negotiated for every voyage",
        ],
    },
    "SHORT_TERM": {
        "advantages": [
            "Some price protection against short-term rate swings",
            "Retains meaningful flexibility versus a longer commitment",
            "Secures vessel availability for the charter period",
        ],
        "disadvantages": [
            "Less price certainty than a medium-term or COA arrangement",
            "Still exposed to the market at each renewal",
            "May be too short to justify for genuinely repeat, high-volume cargo",
        ],
    },
    "MEDIUM_TERM": {
        "advantages": [
            "Higher price certainty over a longer horizon",
            "Secures vessel capacity ahead of a tightening market",
            "Reduces administrative overhead of repeated fixtures",
        ],
        "disadvantages": [
            "Lower flexibility — committed even if the market moves favorably",
            "Higher exposure if the forecast direction turns out wrong",
            "Not justified for a single or small cargo requirement",
        ],
    },
    "COA": {
        "advantages": [
            "Greatest planning certainty for repeated, predictable voyages",
            "Locks in capacity and (often) pricing across many future voyages",
            "Efficient for high, predictable cargo volumes over time",
        ],
        "disadvantages": [
            "Highest commitment and lowest flexibility of the four options",
            "Requires predictable cargo volume and voyage count to be worthwhile",
            "Costly to unwind if cargo requirements change materially",
        ],
    },
}

# ---------------------------------------------------------------------------
# PROTOTYPE scoring constants — configurable
# ---------------------------------------------------------------------------

BASELINE_SCORE = 50.0

POINTS = {
    "direction_strong": 18.0,
    "uncertainty_strong": 15.0,
    "uncertainty_mild": 8.0,
    "voyage_pattern_strong": 22.0,
    "voyage_pattern_mild": 10.0,
    "duration_match": 13.0,
    "volatility_strong": 12.0,
    "operational_risk_strong": 12.0,
    "operational_risk_mild": 6.0,
    "availability_strong": 10.0,
    "availability_mild": 6.0,
}

FACTOR_CATEGORIES = [
    "direction", "uncertainty", "voyage_pattern", "duration",
    "volatility", "operational_risk", "vessel_availability",
]


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _normalize_level(value) -> Optional[str]:
    """Normalize a level input that may already be 'LOW'/'MEDIUM'/'HIGH', or
    a raw 0-100 numeric risk score, into 'LOW'/'MEDIUM'/'HIGH'. Returns
    ``None`` for anything unrecognizable/missing.
    """
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().upper()
        return v if v in ("LOW", "MEDIUM", "HIGH") else None
    try:
        return classify_level(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_availability(value) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().upper()
    if v in ("SCARCE", "LIMITED", "TIGHT"):
        return "SCARCE"
    if v in ("ABUNDANT", "PLENTIFUL", "AVAILABLE", "HIGH"):
        return "ABUNDANT"
    if v in ("MODERATE", "MEDIUM", "NORMAL"):
        return "MODERATE"
    return None


# ---------------------------------------------------------------------------
# Resolve decision factors from direct inputs + upstream engine outputs
# ---------------------------------------------------------------------------

def resolve_decision_factors(
    cargo_quantity: float,
    contract_duration_months: Optional[float] = None,
    expected_number_of_voyages: Optional[int] = None,
    current_freight_rate: Optional[float] = None,
    predicted_freight_rate: Optional[float] = None,
    freight_volatility: Optional[float] = None,
    vessel_availability: Optional[str] = None,
    port_congestion: Optional[str] = None,
    operational_risk: Optional[str] = None,
    forecast_result: Optional[Dict] = None,
    vessel_recommendation: Optional[Dict] = None,
    port_evaluation: Optional[Dict] = None,
    risk_result: Optional[Dict] = None,
) -> Dict:
    """Combine directly-supplied inputs with upstream FreightAI engine
    outputs into one normalized decision-factor dict. Direct arguments
    always take priority over a value derived from an engine output.
    Anything that can't be resolved from either source is left as ``None``
    (treated as neutral / not scored) rather than guessed.
    """
    factors: Dict = {
        "cargo_quantity": cargo_quantity,
        "contract_duration_months": contract_duration_months,
        "expected_number_of_voyages": expected_number_of_voyages,
    }

    resolved_current = current_freight_rate
    resolved_predicted = predicted_freight_rate
    if forecast_result:
        if resolved_current is None:
            resolved_current = forecast_result.get("historical_last_value")
        if resolved_predicted is None and forecast_result.get("forecast"):
            resolved_predicted = forecast_result["forecast"][-1].get("predicted_freight_rate")
    factors["current_freight_rate"] = resolved_current
    factors["predicted_freight_rate"] = resolved_predicted

    direction = None
    if forecast_result and forecast_result.get("market_signal"):
        direction = forecast_result["market_signal"]
    elif resolved_current and resolved_predicted:
        pct_change = (resolved_predicted - resolved_current) / resolved_current * 100
        if pct_change > TREND_THRESHOLD_PCT:
            direction = "INCREASING"
        elif pct_change < -TREND_THRESHOLD_PCT:
            direction = "DECREASING"
        else:
            direction = "STABLE"
    factors["forecast_direction"] = direction
    factors["pct_change"] = (
        round((resolved_predicted - resolved_current) / resolved_current * 100, 2)
        if resolved_current and resolved_predicted else None
    )

    uncertainty_level = None
    if risk_result:
        uncertainty_level = _normalize_level(
            risk_result.get("market_risk", {}).get("components", {}).get("forecast_uncertainty")
        )
    elif forecast_result is not None:
        uncertainty_level = "HIGH" if forecast_result.get("experimental") else "MEDIUM"
    factors["forecast_uncertainty_level"] = uncertainty_level

    volatility_level = None
    if risk_result:
        volatility_level = _normalize_level(
            risk_result.get("market_risk", {}).get("components", {}).get("freight_volatility")
        )
    elif freight_volatility is not None:
        volatility_level = classify_level(_risk_from_thresholds(freight_volatility, *RISK_THRESHOLDS["freight_volatility_cv_pct"]))
    factors["freight_volatility_level"] = volatility_level

    op_risk_level = None
    if risk_result:
        op_risk_level = risk_result.get("operational_risk", {}).get("level")
    elif operational_risk is not None:
        op_risk_level = _normalize_level(operational_risk)
    if vessel_recommendation is not None and vessel_recommendation.get("recommended_compatible") is False:
        escalation = {"LOW": "MEDIUM", "MEDIUM": "HIGH", "HIGH": "HIGH", None: "MEDIUM"}
        op_risk_level = escalation.get(op_risk_level, op_risk_level)
        factors["vessel_incompatibility_escalation"] = True
    else:
        factors["vessel_incompatibility_escalation"] = False
    factors["operational_risk_level"] = op_risk_level

    congestion_level = None
    if risk_result:
        congestion_level = _normalize_level(
            risk_result.get("port_risk", {}).get("components", {}).get("congestion")
        )
    elif port_evaluation is not None:
        congestion_level = _normalize_level(port_evaluation.get("port", {}).get("congestion_level"))
    elif port_congestion is not None:
        congestion_level = _normalize_level(port_congestion)
    factors["port_congestion_level"] = congestion_level

    factors["vessel_availability_level"] = _normalize_availability(vessel_availability)

    return factors


# ---------------------------------------------------------------------------
# Per-strategy scoring rules — each returns (score, [(delta, reason), ...])
# ---------------------------------------------------------------------------

def _score_spot(f: Dict, w: Dict[str, float]) -> Tuple[float, List[Tuple[float, str]]]:
    deltas: List[Tuple[float, str]] = []

    if f["forecast_direction"] == "DECREASING":
        deltas.append((POINTS["direction_strong"] * w["direction"],
                        "Forecast indicates decreasing freight rates — spot fixtures let you capture lower rates as they emerge rather than locking in now."))
    elif f["forecast_direction"] == "INCREASING":
        deltas.append((-POINTS["direction_strong"] * w["direction"],
                        "Forecast indicates increasing freight rates — spot exposure risks paying more on each future fixture."))

    if f["forecast_uncertainty_level"] == "HIGH":
        deltas.append((POINTS["uncertainty_strong"] * w["uncertainty"],
                        "High forecast uncertainty favors staying flexible rather than committing to a longer charter."))
    elif f["forecast_uncertainty_level"] == "LOW":
        deltas.append((-POINTS["uncertainty_mild"] * w["uncertainty"],
                        "Low forecast uncertainty means there's less need for spot's flexibility."))

    voyages = f.get("expected_number_of_voyages")
    if voyages is not None:
        if voyages <= 1:
            deltas.append((POINTS["voyage_pattern_mild"] * w["voyage_pattern"],
                            "A single voyage requirement fits a one-off spot fixture well."))
        else:
            deltas.append((-POINTS["voyage_pattern_strong"] * w["voyage_pattern"],
                            "Multiple expected voyages are better served by a repeat-business arrangement than repeated spot fixtures."))

    duration = f.get("contract_duration_months")
    if duration is not None:
        if duration <= 1:
            deltas.append((POINTS["duration_match"] * w["duration"],
                            "No meaningful coverage period is required, matching spot's no-commitment nature."))
        elif duration >= 6:
            deltas.append((-POINTS["duration_match"] * w["duration"],
                            "A multi-month coverage requirement doesn't suit single spot fixtures."))

    if f["freight_volatility_level"] == "HIGH":
        deltas.append((-POINTS["volatility_strong"] * w["volatility"],
                        "High freight-rate volatility means spot fixtures carry significant price-uncertainty risk."))

    if f["operational_risk_level"] == "HIGH":
        deltas.append((POINTS["operational_risk_mild"] * w["operational_risk"],
                        "High operational risk favors avoiding a long-term commitment to a single risky port/vessel combination."))

    if f["vessel_availability_level"] == "SCARCE":
        deltas.append((-POINTS["availability_strong"] * w["vessel_availability"],
                        "Scarce vessel availability makes reliably securing capacity via spot fixtures harder."))
    elif f["vessel_availability_level"] == "ABUNDANT":
        deltas.append((POINTS["availability_mild"] * w["vessel_availability"],
                        "Abundant vessel availability supports relying on spot fixtures as needed."))

    score = _clip(BASELINE_SCORE + sum(d for d, _ in deltas))
    return score, deltas


def _score_short_term(f: Dict, w: Dict[str, float]) -> Tuple[float, List[Tuple[float, str]]]:
    deltas: List[Tuple[float, str]] = []

    if f["forecast_direction"] == "INCREASING":
        deltas.append((POINTS["direction_strong"] * 0.7 * w["direction"],
                        "Forecast indicates increasing freight rates — a short-term charter provides price protection while retaining flexibility."))
    elif f["forecast_direction"] == "DECREASING":
        deltas.append((-POINTS["direction_strong"] * 0.45 * w["direction"],
                        "Forecast indicates decreasing freight rates, so even a short commitment risks locking in a rate that's about to fall."))
    elif f["forecast_direction"] == "STABLE":
        deltas.append((POINTS["uncertainty_mild"] * 0.5 * w["direction"],
                        "A stable freight-rate forecast makes a short-term charter a balanced default choice."))

    if f["forecast_uncertainty_level"] == "HIGH":
        deltas.append((POINTS["uncertainty_mild"] * 0.5 * w["uncertainty"],
                        "Short-term exposure limits the downside of high forecast uncertainty compared to a longer commitment."))

    voyages = f.get("expected_number_of_voyages")
    if voyages is not None:
        if 2 <= voyages <= 4:
            deltas.append((POINTS["voyage_pattern_mild"] * w["voyage_pattern"],
                            "A handful of expected voyages fits comfortably within a short-term charter period."))
        elif voyages > 4:
            deltas.append((-POINTS["voyage_pattern_mild"] * 0.6 * w["voyage_pattern"],
                            "A higher number of expected voyages may be more efficiently covered by a longer-term or COA arrangement."))
        elif voyages <= 1:
            deltas.append((POINTS["voyage_pattern_mild"] * 0.4 * w["voyage_pattern"],
                            "A single voyage can still be reasonably covered by a short-term charter if some coverage period is desired."))

    duration = f.get("contract_duration_months")
    if duration is not None:
        if 1 < duration <= 6:
            deltas.append((POINTS["duration_match"] * w["duration"],
                            "The required coverage period fits a short-term charter well."))
        elif duration > 6:
            deltas.append((-POINTS["duration_match"] * 0.5 * w["duration"],
                            "The required coverage period is longer than a typical short-term charter."))

    if f["freight_volatility_level"] == "HIGH":
        deltas.append((POINTS["volatility_strong"] * 0.7 * w["volatility"],
                        "A short-term charter offers some price protection against volatile spot rates without a long-term commitment."))

    if f["operational_risk_level"] == "HIGH":
        deltas.append((POINTS["operational_risk_mild"] * 0.5 * w["operational_risk"],
                        "Moderate commitment length keeps exposure to elevated operational risk contained."))

    if f["vessel_availability_level"] == "SCARCE":
        deltas.append((POINTS["availability_mild"] * w["vessel_availability"],
                        "A short-term charter offers more certainty of capacity than relying on spot fixtures alone."))

    score = _clip(BASELINE_SCORE + sum(d for d, _ in deltas))
    return score, deltas


def _score_medium_term(f: Dict, w: Dict[str, float]) -> Tuple[float, List[Tuple[float, str]]]:
    deltas: List[Tuple[float, str]] = []

    if f["forecast_direction"] == "INCREASING":
        deltas.append((POINTS["direction_strong"] * w["direction"],
                        "Locking in a medium-term charter secures today's lower rate ahead of the forecast increase."))
    elif f["forecast_direction"] == "DECREASING":
        deltas.append((-POINTS["direction_strong"] * w["direction"],
                        "Committing medium-term while rates are forecast to fall risks locking in a rate higher than future alternatives."))

    if f["forecast_uncertainty_level"] == "HIGH":
        deltas.append((-POINTS["uncertainty_strong"] * w["uncertainty"],
                        "High forecast uncertainty makes a longer rate commitment risky in either direction."))
    elif f["forecast_uncertainty_level"] == "LOW":
        deltas.append((POINTS["uncertainty_strong"] * 0.8 * w["uncertainty"],
                        "Low forecast uncertainty supports confidently locking in price over a longer period."))

    voyages = f.get("expected_number_of_voyages")
    if voyages is not None:
        if voyages <= 1:
            deltas.append((-POINTS["voyage_pattern_strong"] * 0.6 * w["voyage_pattern"],
                            "A single voyage does not justify a medium-term commitment."))
        elif 2 <= voyages <= 6:
            deltas.append((POINTS["voyage_pattern_mild"] * w["voyage_pattern"],
                            "A moderate number of expected voyages fits a medium-term charter's coverage period."))

    duration = f.get("contract_duration_months")
    if duration is not None:
        if 6 <= duration <= 12:
            deltas.append((POINTS["duration_match"] * w["duration"],
                            "The required coverage period matches a typical medium-term charter."))
        elif duration < 3:
            deltas.append((-POINTS["duration_match"] * w["duration"],
                            "The required coverage period is too short to justify a medium-term commitment."))

    if f["freight_volatility_level"] == "HIGH":
        deltas.append((POINTS["volatility_strong"] * w["volatility"],
                        "Locking in price via a medium-term charter protects against high freight-rate volatility over a longer horizon."))

    if f["operational_risk_level"] == "HIGH":
        deltas.append((-POINTS["operational_risk_strong"] * w["operational_risk"],
                        "High operational risk makes a longer commitment to this route/vessel combination riskier."))

    if f["vessel_availability_level"] == "SCARCE":
        deltas.append((POINTS["availability_strong"] * w["vessel_availability"],
                        "Securing a medium-term charter locks in capacity ahead of a tightening vessel market."))
    elif f["vessel_availability_level"] == "ABUNDANT":
        deltas.append((-POINTS["availability_mild"] * w["vessel_availability"],
                        "With abundant vessel availability, a medium-term commitment gives up flexibility for little added security."))

    score = _clip(BASELINE_SCORE + sum(d for d, _ in deltas))
    return score, deltas


def _score_coa(f: Dict, w: Dict[str, float]) -> Tuple[float, List[Tuple[float, str]]]:
    deltas: List[Tuple[float, str]] = []

    if f["forecast_direction"] == "INCREASING":
        deltas.append((POINTS["direction_strong"] * 0.55 * w["direction"],
                        "Forecast rate increases favor locking in coverage across multiple future voyages now."))
    elif f["forecast_direction"] == "DECREASING":
        deltas.append((-POINTS["direction_strong"] * 0.55 * w["direction"],
                        "Committing to a COA while rates are forecast to fall risks locking in unfavorable pricing across many voyages."))

    if f["forecast_uncertainty_level"] == "HIGH":
        deltas.append((-POINTS["uncertainty_strong"] * 0.8 * w["uncertainty"],
                        "High forecast uncertainty is especially risky to lock in across many future voyages."))
    elif f["forecast_uncertainty_level"] == "LOW":
        deltas.append((POINTS["uncertainty_strong"] * 0.6 * w["uncertainty"],
                        "Low forecast uncertainty supports the confidence needed for a multi-voyage commitment."))

    voyages = f.get("expected_number_of_voyages")
    if voyages is not None:
        if voyages >= 3:
            deltas.append((POINTS["voyage_pattern_strong"] * w["voyage_pattern"],
                            "Multiple, predictable voyages of similar cargo make a Contract of Affreightment the most efficient structure."))
        else:
            deltas.append((-POINTS["voyage_pattern_strong"] * w["voyage_pattern"],
                            "A Contract of Affreightment isn't justified without multiple repeat voyages."))
    else:
        deltas.append((-POINTS["voyage_pattern_mild"] * w["voyage_pattern"],
                        "Expected number of voyages wasn't provided — a COA is hard to justify without a predictable voyage pattern."))

    duration = f.get("contract_duration_months")
    if duration is not None:
        if duration >= 12:
            deltas.append((POINTS["duration_match"] * w["duration"],
                            "A long required coverage period matches a COA's typical planning horizon."))
        elif duration < 6:
            deltas.append((-POINTS["duration_match"] * w["duration"],
                            "The required coverage period is short relative to a typical COA horizon."))

    if f["freight_volatility_level"] == "HIGH":
        deltas.append((POINTS["volatility_strong"] * 0.8 * w["volatility"],
                        "Locking in terms across multiple voyages hedges against high freight-rate volatility."))

    if f["operational_risk_level"] == "HIGH":
        deltas.append((-POINTS["operational_risk_strong"] * w["operational_risk"],
                        "High operational risk makes committing to many future voyages on this route riskier."))

    if f["vessel_availability_level"] == "SCARCE":
        deltas.append((POINTS["availability_strong"] * 1.1 * w["vessel_availability"],
                        "A COA guarantees capacity across many future voyages — most valuable exactly when vessels are scarce."))

    score = _clip(BASELINE_SCORE + sum(d for d, _ in deltas))
    return score, deltas


_SCORERS = {"SPOT": _score_spot, "SHORT_TERM": _score_short_term, "MEDIUM_TERM": _score_medium_term, "COA": _score_coa}


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def _compute_confidence(scores: Dict[str, float], factors: Dict) -> str:
    """Confidence in the recommendation: based on (a) how clearly the top
    strategy beats the runner-up, and (b) how many of the decision factors
    were actually resolved (vs. left neutral/unknown).
    """
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else 0

    factor_keys = [
        "forecast_direction", "forecast_uncertainty_level", "freight_volatility_level",
        "operational_risk_level", "vessel_availability_level",
        "expected_number_of_voyages", "contract_duration_months",
    ]
    resolved = sum(1 for k in factor_keys if factors.get(k) is not None)
    completeness = resolved / len(factor_keys)

    if margin >= 15 and completeness >= 0.6:
        return "HIGH"
    if margin >= 7 and completeness >= 0.4:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def recommend_charter_strategy(
    cargo_quantity: float,
    contract_duration_months: Optional[float] = None,
    expected_number_of_voyages: Optional[int] = None,
    current_freight_rate: Optional[float] = None,
    predicted_freight_rate: Optional[float] = None,
    freight_volatility: Optional[float] = None,
    vessel_availability: Optional[str] = None,
    port_congestion: Optional[str] = None,
    operational_risk: Optional[str] = None,
    forecast_result: Optional[Dict] = None,
    vessel_recommendation: Optional[Dict] = None,
    port_evaluation: Optional[Dict] = None,
    risk_result: Optional[Dict] = None,
    factor_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Recommend a chartering strategy for a proposed voyage/cargo.

    Every decision factor can come from either a direct argument or an
    upstream FreightAI engine's output (``forecast_result`` from
    ``forecasting.predict.forecast_freight()``, ``vessel_recommendation``
    from ``optimization.vessel_optimizer.recommend_vessel_class()``,
    ``port_evaluation`` from ``optimization.port_optimizer`` /
    ``voyage_calculator``, ``risk_result`` from
    ``risk.risk_engine.assess_voyage_risk()``) — direct arguments always
    take priority. Anything resolvable from neither source is left neutral
    (not scored, not guessed).

    ``factor_weights``: optional dict with keys matching
    :data:`FACTOR_CATEGORIES` (direction, uncertainty, voyage_pattern,
    duration, volatility, operational_risk, vessel_availability), each a
    multiplier (default 1.0) on that category's point contributions —
    the PROTOTYPE scoring rules are trivially reconfigurable this way.

    Returns
    -------
    {
        "recommended_strategy": "SHORT_TERM",
        "recommended_strategy_display": "Short-Term Charter",
        "scores": {"SPOT": .., "SHORT_TERM": .., "MEDIUM_TERM": .., "COA": ..},
        "reasons": {"SPOT": [...], ...},           # every scored reason, per strategy
        "summary_reason": str,                       # the winning strategy's top 1-2 reasons, as prose
        "advantages": [...], "disadvantages": [...],  # for the recommended strategy
        "confidence": "LOW" | "MEDIUM" | "HIGH",
        "decision_factors": {...},                    # fully resolved factors used
        "disclaimer": str,
    }
    """
    if cargo_quantity is None or cargo_quantity <= 0:
        raise ValueError(f"cargo_quantity must be a positive number, got {cargo_quantity!r}.")

    weights = {cat: 1.0 for cat in FACTOR_CATEGORIES}
    if factor_weights:
        weights.update(factor_weights)

    factors = resolve_decision_factors(
        cargo_quantity=cargo_quantity,
        contract_duration_months=contract_duration_months,
        expected_number_of_voyages=expected_number_of_voyages,
        current_freight_rate=current_freight_rate,
        predicted_freight_rate=predicted_freight_rate,
        freight_volatility=freight_volatility,
        vessel_availability=vessel_availability,
        port_congestion=port_congestion,
        operational_risk=operational_risk,
        forecast_result=forecast_result,
        vessel_recommendation=vessel_recommendation,
        port_evaluation=port_evaluation,
        risk_result=risk_result,
    )

    scores: Dict[str, float] = {}
    reasons: Dict[str, List[str]] = {}
    deltas_by_strategy: Dict[str, List[Tuple[float, str]]] = {}
    for strategy in STRATEGIES:
        score, deltas = _SCORERS[strategy](factors, weights)
        scores[strategy] = round(score, 1)
        deltas_by_strategy[strategy] = deltas
        reasons[strategy] = [r for _, r in deltas] or ["No strong signal either way for this strategy given the inputs provided."]

    recommended = max(scores, key=scores.get)
    top_deltas = sorted(deltas_by_strategy[recommended], key=lambda t: abs(t[0]), reverse=True)
    summary_reason = " ".join(r for _, r in top_deltas[:2]) or reasons[recommended][0]

    confidence = _compute_confidence(scores, factors)

    return {
        "recommended_strategy": recommended,
        "recommended_strategy_display": STRATEGY_DISPLAY_NAMES[recommended],
        "scores": scores,
        "reasons": reasons,
        "summary_reason": summary_reason,
        "advantages": STRATEGY_PROS_CONS[recommended]["advantages"],
        "disadvantages": STRATEGY_PROS_CONS[recommended]["disadvantages"],
        "confidence": confidence,
        "decision_factors": factors,
        "weights_used": {**weights, "_label": "PROTOTYPE scoring weights — configurable, not universal industry rules"},
        "disclaimer": (
            "This is a prototype, explainable chartering-strategy recommendation using "
            "configurable heuristic rules. It is not a substitute for professional chartering, "
            "legal, or commercial judgment, and does not account for counterparty relationships, "
            "financing, or specific charter-party terms."
        ),
    }


if __name__ == "__main__":
    result = recommend_charter_strategy(
        cargo_quantity=210_000,
        contract_duration_months=9,
        expected_number_of_voyages=3,
        current_freight_rate=1500,
        predicted_freight_rate=1680,
        freight_volatility=12.5,
        vessel_availability="MODERATE",
        operational_risk="LOW",
    )
    print("\n=== FreightAI Charter Strategy Engine — demo ===")
    print(f"RECOMMENDED STRATEGY:\n{result['recommended_strategy_display'].upper()}")
    print(f"\nReason:\n\"{result['summary_reason']}\"")
    print(f"\nConfidence: {result['confidence']}")
    print("\nScores:")
    for s, v in result["scores"].items():
        print(f"  {STRATEGY_DISPLAY_NAMES[s]:24s} {v}")
    print("\nAdvantages:")
    for a in result["advantages"]:
        print(f"  + {a}")
    print("Disadvantages:")
    for d in result["disadvantages"]:
        print(f"  - {d}")
