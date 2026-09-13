# FreightAI — Vessel Optimization Engine

Recommends the most suitable bulk carrier class — **Handysize, Supramax,
Panamax, or Capesize** — for a shipment, given cargo quantity, commodity,
origin port, and destination port.

File: `src/optimization/vessel_optimizer.py`
Entry point: `recommend_vessel_class(cargo_quantity, commodity, origin, destination, weights=None)`

Does not build the dashboard — returns plain, JSON-friendly Python dicts for
a future dashboard module to render.

## Inputs

* `data/processed/vessels_clean.csv` — fleet data (any column names; the
  module auto-detects DWT/LOA/beam/draft/speed/year-built columns by
  keyword, the same way the data-pipeline module auto-detects freight
  columns). **Not required** — if missing, each vessel class falls back to
  typical/reference specs (clearly flagged `data_source: "reference_default"`).
* `data/processed/ports_clean.csv` — port limits (max draft/LOA/beam/DWT),
  same auto-detection approach. **Not required** — if missing, or if a
  named port isn't found, port-compatibility checks are marked `UNKNOWN`/
  `verified: false` rather than crashing or silently assuming compatibility
  is impossible.
* User inputs: `cargo_quantity` (metric tons), `commodity`, `origin`,
  `destination` (port names).

## Port compatibility

For **both** the origin and destination port, each vessel class is checked
against:

```
vessel draft <= port max draft
vessel LOA   <= port max LOA
vessel beam  <= port max beam
vessel DWT   <= port max DWT
```

Every parameter is reported as `PASS` / `FAIL` / `UNKNOWN` (`UNKNOWN` when
the relevant port data isn't available — this is treated as *unverified*,
not as an automatic failure, but it does reduce the port-compatibility
score and is called out in `reasons`). A vessel class is only
`compatible: true` overall if it clears both ports with no `FAIL`.

## Suitability score — prototype, configurable weights

```python
PROTOTYPE_WEIGHTS = {
    "port_compatibility": 0.30,
    "cargo_capacity":     0.25,
    "voyage_cost":        0.20,
    "speed":              0.10,
    "turnaround":         0.10,
    "operational_risk":   0.05,
}
```

These are explicitly a **prototype starting point**, not a validated
commercial model — pass your own `weights` dict (same keys, summing to
1.0) to `recommend_vessel_class(..., weights=your_weights)` to override
them.

Component scoring (all 0-100, prototype heuristics — see docstrings in
`vessel_optimizer.py` for full detail):

| Component | Basis |
|---|---|
| `port_compatibility` | PASS/FAIL/UNKNOWN checks above |
| `cargo_capacity` | how well cargo quantity utilizes vessel DWT (85-98% = ideal) |
| `voyage_cost` | class-level economies-of-scale baseline blended with utilization — **not** a real bunker/freight cost model |
| `speed` | vessel service speed vs. a 10-16 knot reference band |
| `turnaround` | typical class-level port-turnaround assumption — **not** derived from real berth data |
| `operational_risk` | vessel-age proxy (older = marginally higher risk) |

## Output

`recommend_vessel_class(...)` always evaluates all four classes and returns:

```python
{
    "inputs": {...},
    "weights_used": {... , "_label": "PROTOTYPE_WEIGHTS — configurable..."},
    "evaluations": {
        "Handysize": {"suitability_score": ..., "compatible": ..., "components": {...}, "reasons": [...]},
        "Supramax":  {...}, "Panamax": {...}, "Capesize": {...},
    },
    "recommended_vessel_type": "Supramax",
    "recommended_compatible": True,
    "alternative_vessel_type": None,   # populated if the top pick isn't port-compatible
    "reasons": [...],
    "data_sources": {...},
    "disclaimer": "This is a prototype decision-support score ...",
}
```

If the highest-scoring class isn't port-compatible, `alternative_vessel_type`
is set to the best-scoring class that IS compatible (or `None` if none are).

## Example usage

```python
from src.optimization.vessel_optimizer import recommend_vessel_class

result = recommend_vessel_class(
    cargo_quantity=55_000,
    commodity="Iron Ore",
    origin="Port Hedland",
    destination="Paradip",
)
print(result["recommended_vessel_type"], result["recommended_compatible"])
print(result["evaluations"]["Panamax"]["components"]["port_compatibility"]["destination"]["checks"])
# {'draft': 'FAIL', 'loa': 'PASS', 'beam': 'PASS', 'dwt': 'PASS'}
```

## Run the built-in demo

```bash
python -m src.optimization.vessel_optimizer
```
Works even with no `vessels_clean.csv`/`ports_clean.csv` present (uses
reference specs and flags ports as unverified).

## Limitations

This is a decision-support **prototype**: voyage cost, turnaround, and
operational risk are simplified class-level/age-based proxies, not derived
from real bunker prices, port tariffs, berth schedules, or maintenance
records. Treat the suitability score as directional guidance, not a
substitute for full voyage estimation or chartering analysis.
