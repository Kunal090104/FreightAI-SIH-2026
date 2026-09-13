# FreightAI — Database Layer

A small, framework-agnostic SQLite persistence layer for FreightAI.

File: `src/database/database.py`
Database: `database/freightai.db` (created by `initialize_database()`)

Built on nothing but the standard library (`sqlite3`) plus `pandas` for
convenient result sets — **no Streamlit or other UI import, ever**. The
dashboard module (built separately) will import and call the functions
here; this module never imports anything UI-related, so it can be tested,
scripted, or reused completely on its own.

## Tables

| Table | Columns | Write behavior |
|---|---|---|
| `freight_rates` | `id, date, freight_rate` | upsert on `date` (UNIQUE) |
| `vessels` | `vessel_id, vessel_name, vessel_type, dwt, loa, beam, draft, speed, year_built` | upsert on `vessel_id` |
| `ports` | `port_name, max_draft, max_loa, max_beam, max_dwt, channel_depth, berth_count, cargo_handling_rate, turnaround_hours, waiting_hours, congestion_level` | upsert on `port_name` (PK) |
| `commodities` | `date, commodity, price, demand_index, import_volume, export_volume, origin_country` | upsert on `(date, commodity, origin_country)` |
| `predictions` | `id, prediction_date, forecast_horizon, predicted_rate, trend, model` | **append-only** — every forecast run is kept as its own history row, never overwritten |

`id`/`vessel_id` are `INTEGER PRIMARY KEY` — omit them when inserting and
SQLite autoincrements; upserts (`INSERT OR REPLACE`) still work correctly
via each table's natural key (SQLite deletes the conflicting row and
reinserts, so `id` may change across an upsert — the natural key is the
real identity, not the surrogate id).

## Functions

```python
from src.database.database import (
    initialize_database, insert_data, query_data,
    save_prediction, get_latest_prediction, save_forecast_result,
)

initialize_database()                      # creates database/freightai.db + all tables; idempotent

insert_data("freight_rates", {"date": "2026-09-01", "freight_rate": 1450.5})
insert_data("freight_rates", some_dataframe)   # dict, list[dict], or DataFrame all work

df = query_data("freight_rates", filters={"date": "2026-09-01"})         # -> pandas DataFrame
rows = query_data("predictions", filters={"forecast_horizon": [7, 30]},   # IN-list filtering
                   order_by="prediction_date", ascending=False, as_dataframe=False)

pred_id = save_prediction("2026-09-10", 30, 1520.3, trend="INCREASING", model="linear_regression")
latest = get_latest_prediction(forecast_horizon=30)   # most recent matching row, or None

# Convenience: log a forecasting-engine result directly
from src.forecasting.predict import forecast_freight
save_forecast_result(forecast_freight(days=30))
```

## Safety

- Every table/column name is validated against the schema before being
  interpolated into SQL; every value is passed as a parameterized query
  argument — no string-built SQL from user input.
- `insert_data()` rejects unknown column names with a clear `ValueError`
  rather than silently dropping data (catches typos early).
- All writes go through a single context-managed connection
  (`get_connection()`) that commits on success and rolls back on any
  exception — one seam to change if the backend ever moves off SQLite.

## Run the built-in demo

```bash
python -m src.database.database
```
Runs entirely against a temporary database file (not `database/freightai.db`)
and cleans up after itself — safe to run anytime without touching real data.
