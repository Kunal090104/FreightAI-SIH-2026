# FreightAI — Port Compatibility & Voyage Analysis Engine

Determines whether a selected vessel can efficiently operate between an
origin and one of the seven supported Indian East Coast destination ports,
and estimates the voyage timeline.

Files: `src/optimization/port_optimizer.py`, `src/optimization/voyage_calculator.py`
Does not build the dashboard — returns plain, JSON-friendly Python dicts.

## Supported destination ports

Paradip, Visakhapatnam, Gangavaram, Gopalpur, Dhamra, **Sagar-Sandheads**,
Haldia.

`Sagar-Sandheads` is modeled honestly as what it actually is: an open-water
anchorage/lightering point (ship-to-ship transfer) serving the Hooghly
river approach to Haldia/Kolkata, not a conventional berthing port —
`berth_count` is 0 by design.

## Port data — two sources, clearly labeled

1. **`data/processed/ports_clean.csv`**, if present — columns are
   auto-detected by keyword (same pattern used elsewhere in FreightAI for
   schema-flexible datasets), and any field found there is used.
2. **`REFERENCE_PORT_SPECS`** (in `port_optimizer.py`) — indicative figures
   compiled from public port-authority/port-reference sources, used for
   any field not found in (1). These are planning-level approximations
   (channel depths shift with ongoing dredging projects) — **not** a
   substitute for the current official Notice to Mariners / port pilotage
   data before a real voyage decision.

Every port spec returned includes a `field_sources` map so callers always
know, field by field, whether a number came from your data or from the
reference defaults.

## Port compatibility check

`port_optimizer.compare_vessel_to_port(vessel_spec, port_name)` checks:

```
vessel draft          <= port max draft
vessel LOA             <= port max LOA
vessel beam             <= port max beam
vessel DWT               <= port max DWT
vessel draft          <= port channel depth   (safety-margin check)
```

Each returns `PASS` / `FAIL` / `UNKNOWN` (unknown when a value is missing —
never silently treated as a pass or a fail). Informational port attributes
(berth count, cargo handling rate, average turnaround/waiting hours, tidal
restriction, congestion level) are included in the same table with status
`INFO` for context, not compared against the vessel.

## Voyage calculation

`voyage_calculator.calculate_voyage(vessel_spec, origin, destination_port, cargo_quantity)`:

```
Sailing Time              = Distance / Vessel Speed
Total Port Time           = Waiting Time + Cargo Handling Time + Turnaround Time
Estimated Total Voyage Time = Sailing Time + Total Port Time
```

* **Distance** comes only from `data/raw/routes.csv` (`origin`,
  `destination`, `distance_nm`). **No real-world distance is ever
  invented or estimated.** If no matching row exists, `distance_status`
  is the literal string **`"Route distance unavailable."`**, and
  `sailing_time_hours` / `estimated_total_voyage_time_hours` are `None` —
  everything else (port time, idle time) is still computed and returned.
* **Cargo Handling Time** = cargo_quantity ÷ port's cargo handling rate
  (metric tons/day) × 24.
* **Turnaround Time** and **Waiting Time** come from the port spec.

### Idle time (prototype estimate)

`estimate_idle_time()` reports:

| Component | Basis |
|---|---|
| `waiting_time_hours` | port's average waiting hours |
| `congestion_delay_hours` | PROTOTYPE lookup by congestion level (LOW=0h, MEDIUM=6h, HIGH=18h) — not a live traffic feed |
| `weather_delay_hours` | PROTOTYPE flat seasonal buffer (default 8h) — not a weather forecast; override via the `weather_delay_hours` argument |
| `total_estimated_idle_time_hours` | sum of the above |

This idle-time figure is reported **separately** for risk visibility and
is **not** double-counted into `estimated_total_voyage_time_hours` (which
uses the formula-specified `waiting_time_hours` only, once).

## Output shape

```python
{
    "port_compatibility": {
        "port": {...spec + field_sources...},
        "compatibility_table": [...PASS/FAIL/UNKNOWN + INFO rows...],
        "compatible": bool, "verified": bool, "warnings": [...],
    },
    "distance_nm": float | None,
    "distance_status": "OK" | "Route distance unavailable.",
    "sailing_time_hours": float | None,
    "port_time": {
        "waiting_time_hours": ..., "cargo_handling_time_hours": ...,
        "turnaround_time_hours": ..., "total_port_time_hours": ...,
    },
    "idle_time": {...},
    "estimated_total_voyage_time_hours": float | None,
    "estimated_total_voyage_time_days": float | None,
    "warnings": [str, ...],
}
```

## Example usage

```python
from src.optimization.voyage_calculator import calculate_voyage

vessel = {"dwt": 82_000, "loa": 225.0, "beam": 32.3, "draft": 14.0, "speed": 14.2}
result = calculate_voyage(
    vessel_spec=vessel, origin="Port Hedland",
    destination_port="Paradip", cargo_quantity=75_000,
)
result["distance_status"]                     # "OK" or "Route distance unavailable."
result["estimated_total_voyage_time_days"]
result["port_compatibility"]["compatibility_table"]
```

### `data/raw/routes.csv` format

```csv
origin,destination,distance_nm
Port Hedland,Paradip,4150
Port Hedland,Haldia,4400
Santos,Visakhapatnam,8700
```

## Run the built-in demos

```bash
python -m src.optimization.port_optimizer      # Panamax vs. all 7 ports
python -m src.optimization.voyage_calculator     # full voyage calc, Port Hedland -> Paradip
```
Both work with no data files present (reference specs; distance shows "unavailable").

## Limitations

Reference port specs are indicative, not authoritative — verify against
current official port-authority data before real decisions. Congestion
and weather delay are simplified prototype placeholders, not live feeds.
Route distance is never invented; without `routes.csv` coverage, sailing
time and total voyage time are honestly reported as unavailable rather
than guessed.
