# FreightAI — Risk Assessment Engine

Calculates an explainable, 0-100 operational risk score for a proposed
bulk cargo voyage, across four categories, by composing the other
FreightAI engines already built.

File: `src/risk/risk_engine.py`
Entry point: `assess_voyage_risk(vessel_spec, origin, destination_port, cargo_quantity, weights=None)`

Does not build the dashboard — returns plain, JSON-friendly Python dicts.

## Risk categories

| Category | Signals used | Source |
|---|---|---|
| **Market Risk** | freight volatility (CV over trailing window), recent freight-rate movement (%), forecast uncertainty (validation MAPE) | `data/processed/freight_clean.csv` + `models/model_metrics.json` (real data; each excluded — not guessed — if missing) |
| **Port Risk** | congestion, waiting time, infrastructure compatibility, tidal restriction | `data/processed/congestion_clean.csv` (preferred) or the port's own (labeled real/reference) spec; `port_optimizer.compare_vessel_to_port` |
| **Weather Risk** | wind, rainfall, wave height, storm indicator | `data/processed/weather_clean.csv` **only** |
| **Operational Risk** | vessel-port mismatch, high turnaround/waiting time, vessel suitability | `voyage_calculator.calculate_voyage` + `vessel_optimizer.evaluate_cargo_capacity` |

## Never invents weather or congestion data

Weather risk is computed **only** from `data/processed/weather_clean.csv`.
If that file doesn't exist, or has no row for the destination port, weather
risk comes back as `score: None` with an explicit reason — never a
plausible-looking fake wind/rain/wave number.

Port congestion prefers the dedicated `data/processed/congestion_clean.csv`
dataset (real data, if present). If that's not available, it falls back to
the destination port's `congestion_level` from `port_optimizer` — which is
itself always labeled as either real (from `ports_clean.csv`) or an
explicitly-flagged prototype reference value — so you always know the
provenance of every congestion figure used.

**Any category with no usable data is excluded from the overall score**
(remaining category weights are renormalized) rather than filled in with a
fabricated mid-range value. `excluded_categories` in the result lists
which ones, if any.

## Suitability score — prototype, configurable weights

```python
PROTOTYPE_CATEGORY_WEIGHTS = {
    "market_risk": 0.30,
    "port_risk": 0.25,
    "weather_risk": 0.20,
    "operational_risk": 0.25,
}
```
Override via `assess_voyage_risk(..., weights=your_weights)`. Overall
0-100 score maps to `LOW` (≤33) / `MEDIUM` (≤66) / `HIGH` (>66) via
`RISK_LEVEL_THRESHOLDS`, also configurable.

## Explainability

Every category returns a `reasons` list of plain-English findings, e.g.:
```
"High port congestion at Paradip (source: congestion_clean.csv)."
"Vessel draft close to Paradip's limit (14.0 vs. 14.3, margin 2.1%)."
"Sharp recent freight-rate increase of +39.6% over the last 30 days."
"Active storm/cyclone indicator flagged for Paradip."
```
`risk_reasons` at the top level is the deduplicated union across all four
categories, plus a note about any excluded category.

## Output

```python
{
    "market_risk": {"score": .., "level": .., "reasons": [...], "components": {...}, "details": {...}},
    "port_risk": {...}, "weather_risk": {...}, "operational_risk": {...},
    "overall_risk": "LOW" | "MEDIUM" | "HIGH" | None,
    "risk_score": float | None,
    "risk_reasons": [...],
    "weights_used": {..., "_label": "PROTOTYPE_CATEGORY_WEIGHTS — configurable..."},
    "excluded_categories": [...],
    "disclaimer": "...",
}
```

## Example usage

```python
from src.risk.risk_engine import assess_voyage_risk

vessel = {"dwt": 82_000, "loa": 225.0, "beam": 32.3, "draft": 14.0, "speed": 14.2}
result = assess_voyage_risk(
    vessel_spec=vessel, origin="Port Hedland",
    destination_port="Paradip", cargo_quantity=75_000,
)
result["overall_risk"], result["risk_score"]
result["risk_reasons"]
```

## Run the built-in demo

```bash
python -m src.risk.risk_engine
```
Works with no data files present (market/weather categories report
`score: None` with clear reasons and are excluded from the overall score).

## Limitations

Prototype, explainable heuristic scoring — thresholds and weights are
reasonable starting points, not a validated commercial or actuarial risk
model. Treat the score and level as decision support, not a substitute for
professional voyage risk assessment, weather routing, or underwriting analysis.
