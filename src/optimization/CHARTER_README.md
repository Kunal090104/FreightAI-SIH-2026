# FreightAI — Charter Strategy Engine

Recommends the most suitable chartering strategy — **Spot, Short-Term,
Medium-Term, or Multiple-Voyage/COA** — by combining outputs from every
other FreightAI engine with direct market/cargo inputs.

File: `src/optimization/charter_optimizer.py`
Entry point: `recommend_charter_strategy(cargo_quantity, ...)`
Does not build the dashboard — returns plain, JSON-friendly Python dicts.

## Inputs — direct values OR upstream engine outputs

Every decision factor can be supplied either directly, or pulled
automatically from an upstream engine's result (direct values always win):

| Direct argument | Or derived from |
|---|---|
| `current_freight_rate`, `predicted_freight_rate` | `forecast_result` (`forecasting.predict.forecast_freight()`) |
| forecast direction | `forecast_result["market_signal"]`, or computed from current/predicted rate |
| `freight_volatility` (CV %) | `risk_result["market_risk"]["components"]["freight_volatility"]` |
| forecast uncertainty | `risk_result["market_risk"]["components"]["forecast_uncertainty"]`, or `forecast_result["experimental"]` |
| `operational_risk` | `risk_result["operational_risk"]["level"]` (escalated one notch if `vessel_recommendation["recommended_compatible"]` is `False`) |
| `port_congestion` | `risk_result["port_risk"]["components"]["congestion"]`, or a `port_evaluation` dict |
| `vessel_availability` | direct only — no engine currently produces this |
| `cargo_quantity`, `contract_duration_months`, `expected_number_of_voyages` | direct only |

Anything resolvable from neither source is left neutral (not scored, not guessed).

## Decision factors considered

Current freight level & forecast direction, forecast uncertainty, cargo
volume, number of voyages, price certainty, flexibility, market exposure,
operational risk (port congestion, vessel-port mismatch, waiting/turnaround
time folded in via `risk_result`).

## Strategy scoring — transparent, additive, PROTOTYPE

Each strategy starts at a baseline score of 50 and every applicable rule
adds or subtracts points, each with an attached plain-English reason — so
nothing is a bare number. For example:

* **Spot**: rewarded by a falling-rate forecast, high volatility avoidance
  isn't relevant (spot IS the volatility), single voyage, short/no coverage
  period; penalized by a rising-rate forecast, multiple voyages, scarce
  vessel availability.
* **Short-Term**: moderate version of the above — gets a meaningful (not
  maximal) boost from a rising-rate forecast ("price protection while
  retaining flexibility"), fits 2-4 voyages and 1-6 month coverage.
* **Medium-Term**: strongly rewarded by a rising-rate forecast + low
  forecast uncertainty + 6-12 month coverage; penalized by high forecast
  uncertainty or a single voyage.
* **Multiple-Voyage/COA**: primarily driven by **voyage count ≥ 3** and
  long (≥12 month) coverage — "useful when cargo volume and voyage
  requirements are predictable," per the spec; penalized hard when voyage
  count is low or unknown.

**These are prototype decision rules, not universal industry rules** — see
the module's `POINTS` dict and `factor_weights` argument to reconfigure.

## Output

```python
{
    "recommended_strategy": "SHORT_TERM",
    "recommended_strategy_display": "Short-Term Charter",
    "scores": {"SPOT": 10.0, "SHORT_TERM": 85.6, "MEDIUM_TERM": 78.0, "COA": 68.9},
    "reasons": {"SPOT": [...], "SHORT_TERM": [...], ...},   # every scored reason, per strategy
    "summary_reason": "...",                                  # top reasons for the winner, as prose
    "advantages": [...], "disadvantages": [...],               # for the recommended strategy
    "confidence": "LOW" | "MEDIUM" | "HIGH",
    "decision_factors": {...},                                  # fully resolved factors used
    "weights_used": {..., "_label": "PROTOTYPE scoring weights..."},
    "disclaimer": "...",
}
```

`confidence` reflects both the score margin between the top two strategies
and how many decision factors were actually resolved (vs. left neutral).

## Example — reproducing the spec's worked example

```python
from src.optimization.charter_optimizer import recommend_charter_strategy

result = recommend_charter_strategy(
    cargo_quantity=120_000, contract_duration_months=4, expected_number_of_voyages=3,
    current_freight_rate=1400, predicted_freight_rate=1500,   # +7.1% -> INCREASING
    freight_volatility=9, vessel_availability="MODERATE", operational_risk="LOW",
)
result["recommended_strategy_display"]   # "Short-Term Charter"
result["summary_reason"]
# "The required coverage period fits a short-term charter well. Forecast indicates
#  increasing freight rates — a short-term charter provides price protection while
#  retaining flexibility."
```

## Example — wired to the other engines directly

```python
from src.forecasting.predict import forecast_freight
from src.risk.risk_engine import assess_voyage_risk
from src.optimization.charter_optimizer import recommend_charter_strategy

forecast = forecast_freight(days=30)
vessel = {"dwt": 82_000, "loa": 225.0, "beam": 32.3, "draft": 14.0, "speed": 14.2}
risk = assess_voyage_risk(vessel, origin="Port Hedland", destination_port="Paradip", cargo_quantity=75_000)

result = recommend_charter_strategy(
    cargo_quantity=75_000, contract_duration_months=6, expected_number_of_voyages=4,
    forecast_result=forecast, risk_result=risk, vessel_availability="SCARCE",
)
```

## Run the built-in demo

```bash
python -m src.optimization.charter_optimizer
```
Works standalone with only direct inputs — no upstream engine outputs required.

## Limitations

A transparent, prototype heuristic scoring system — not a validated
commercial chartering model. Does not account for counterparty
relationships, financing, specific charter-party terms, or market
intelligence outside the inputs provided.
