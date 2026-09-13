"""
database.py
============
SQLite database layer for FreightAI.

Provides a small, framework-agnostic persistence layer for the five core
FreightAI tables: ``freight_rates``, ``vessels``, ``ports``, ``commodities``,
and ``predictions``. Built on nothing but the Python standard library
(``sqlite3``) plus ``pandas`` for convenient result sets — it does not
import Streamlit or any other UI framework, and never will: the dashboard
module (built separately) is expected to import and call the functions
here, never the other way around.

Database file: ``database/freightai.db`` (created on first
:func:`initialize_database` call; safe to call repeatedly).

------------------------------------------------------------------------
Design notes
------------------------------------------------------------------------
* SQLite today, by design ("Use SQLite initially") — every function takes
  a ``db_path`` argument and talks to the database only through
  :func:`get_connection`, so swapping the backend later (e.g. Postgres)
  means changing one connection helper, not every call site.
* ``freight_rates``, ``vessels``, ``ports``, and ``commodities`` are
  upserted (``INSERT OR REPLACE``) on their natural key, so re-running the
  data pipeline or re-importing a vessel/port spec updates the existing
  row instead of duplicating it.
* ``predictions`` is an append-only log (plain ``INSERT``) — every forecast
  run is kept as its own historical row, never overwritten.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "freightai.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

TABLE_COLUMNS: Dict[str, List[str]] = {
    "freight_rates": ["id", "date", "freight_rate"],
    "vessels": ["vessel_id", "vessel_name", "vessel_type", "dwt", "loa", "beam", "draft", "speed", "year_built"],
    "ports": [
        "port_name", "max_draft", "max_loa", "max_beam", "max_dwt", "channel_depth",
        "berth_count", "cargo_handling_rate", "turnaround_hours", "waiting_hours", "congestion_level",
    ],
    "commodities": ["date", "commodity", "price", "demand_index", "import_volume", "export_volume", "origin_country"],
    "predictions": ["id", "prediction_date", "forecast_horizon", "predicted_rate", "trend", "model"],
}

# How insert_data() resolves a conflict on the table's natural key.
# "REPLACE" = upsert (INSERT OR REPLACE); "INSERT" = always append (a log).
_INSERT_MODE: Dict[str, str] = {
    "freight_rates": "REPLACE",
    "vessels": "REPLACE",
    "ports": "REPLACE",
    "commodities": "REPLACE",
    "predictions": "INSERT",
}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS freight_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    freight_rate REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_freight_rates_date ON freight_rates(date);

CREATE TABLE IF NOT EXISTS vessels (
    vessel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    vessel_name TEXT NOT NULL,
    vessel_type TEXT,
    dwt REAL,
    loa REAL,
    beam REAL,
    draft REAL,
    speed REAL,
    year_built INTEGER
);

CREATE TABLE IF NOT EXISTS ports (
    port_name TEXT PRIMARY KEY,
    max_draft REAL,
    max_loa REAL,
    max_beam REAL,
    max_dwt REAL,
    channel_depth REAL,
    berth_count INTEGER,
    cargo_handling_rate REAL,
    turnaround_hours REAL,
    waiting_hours REAL,
    congestion_level TEXT
);

CREATE TABLE IF NOT EXISTS commodities (
    date TEXT NOT NULL,
    commodity TEXT NOT NULL,
    price REAL,
    demand_index REAL,
    import_volume REAL,
    export_volume REAL,
    origin_country TEXT,
    UNIQUE (date, commodity, origin_country)
);
CREATE INDEX IF NOT EXISTS idx_commodities_date ON commodities(date);
CREATE INDEX IF NOT EXISTS idx_commodities_commodity ON commodities(commodity);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date TEXT NOT NULL,
    forecast_horizon INTEGER NOT NULL,
    predicted_rate REAL NOT NULL,
    trend TEXT,
    model TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON predictions(forecast_horizon);
"""


# ---------------------------------------------------------------------------
# Connection handling
# ---------------------------------------------------------------------------

@contextmanager
def get_connection(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Context-managed SQLite connection: rows come back as ``sqlite3.Row``
    (dict-like access), commits on success, rolls back on any exception,
    and always closes the connection. This is the only way any function in
    this module touches the database — the single seam to change if the
    backend ever moves off SQLite.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _validate_table(table_name: str) -> None:
    if table_name not in TABLE_COLUMNS:
        raise ValueError(f"Unknown table '{table_name}'. Valid tables: {sorted(TABLE_COLUMNS)}.")


# ---------------------------------------------------------------------------
# initialize_database()
# ---------------------------------------------------------------------------

def initialize_database(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
    """Create ``database/freightai.db`` and all five FreightAI tables
    (plus supporting indexes) if they don't already exist. Idempotent —
    safe to call at the start of every session/script.
    """
    with get_connection(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)


# ---------------------------------------------------------------------------
# insert_data()
# ---------------------------------------------------------------------------

def _normalize_records(records: Union[Dict, List[Dict], "pd.DataFrame"]) -> List[Dict]:
    if isinstance(records, pd.DataFrame):
        return records.where(pd.notna(records), None).to_dict(orient="records")
    if isinstance(records, dict):
        return [records]
    if isinstance(records, list):
        return records
    raise TypeError(f"records must be a dict, list of dicts, or pandas DataFrame, got {type(records)!r}.")


def insert_data(
    table_name: str,
    records: Union[Dict, List[Dict], "pd.DataFrame"],
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> int:
    """Insert (or, for ``freight_rates``/``vessels``/``ports``/``commodities``,
    upsert-by-natural-key) one or more records into ``table_name``.

    Parameters
    ----------
    table_name: one of ``freight_rates``, ``vessels``, ``ports``,
        ``commodities``, ``predictions``.
    records: a single dict, a list of dicts, or a pandas DataFrame. Any
        column not present in a record is stored as ``NULL``; any key
        present that isn't a real column for this table raises
        ``ValueError`` (fail fast on typos rather than silently dropping data).

    Returns
    -------
    The number of rows written.
    """
    _validate_table(table_name)
    rows = _normalize_records(records)
    if not rows:
        return 0

    columns = TABLE_COLUMNS[table_name]
    valid_cols = set(columns)
    for row in rows:
        unknown = set(row.keys()) - valid_cols
        if unknown:
            raise ValueError(f"Unknown column(s) for table '{table_name}': {sorted(unknown)}. Valid columns: {columns}.")

    verb = "INSERT OR REPLACE" if _INSERT_MODE[table_name] == "REPLACE" else "INSERT"
    placeholders = ", ".join("?" for _ in columns)
    sql = f"{verb} INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    values = [tuple(row.get(col) for col in columns) for row in rows]

    with get_connection(db_path) as conn:
        cursor = conn.executemany(sql, values)
        return cursor.rowcount if cursor.rowcount != -1 else len(rows)


# ---------------------------------------------------------------------------
# query_data()
# ---------------------------------------------------------------------------

def query_data(
    table_name: str,
    filters: Optional[Dict[str, Any]] = None,
    columns: Optional[List[str]] = None,
    order_by: Optional[str] = None,
    ascending: bool = True,
    limit: Optional[int] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    as_dataframe: bool = True,
) -> Union["pd.DataFrame", List[Dict]]:
    """Query rows from ``table_name`` with simple equality filtering.

    Parameters
    ----------
    filters: dict of ``{column: value}`` — only equality filtering is
        supported (kept intentionally simple and injection-safe via
        parameterized queries). Pass a list/tuple as a value to filter
        with ``IN (...)``.
    columns: subset of columns to return (default: all).
    order_by: a single column name to sort by.
    limit: max rows to return.
    as_dataframe: return a ``pandas.DataFrame`` (default) or a list of dicts.

    All table/column names are validated against :data:`TABLE_COLUMNS`
    before being interpolated into SQL — values are always parameterized.
    """
    _validate_table(table_name)
    valid_cols = TABLE_COLUMNS[table_name]

    select_cols = columns or valid_cols
    unknown_select = set(select_cols) - set(valid_cols)
    if unknown_select:
        raise ValueError(f"Unknown column(s) for table '{table_name}': {sorted(unknown_select)}.")

    sql = f"SELECT {', '.join(select_cols)} FROM {table_name}"
    params: List[Any] = []

    if filters:
        unknown_filter = set(filters.keys()) - set(valid_cols)
        if unknown_filter:
            raise ValueError(f"Unknown filter column(s) for table '{table_name}': {sorted(unknown_filter)}.")
        clauses = []
        for col, val in filters.items():
            if isinstance(val, (list, tuple, set)):
                val = list(val)
                clauses.append(f"{col} IN ({', '.join('?' for _ in val)})")
                params.extend(val)
            else:
                clauses.append(f"{col} = ?")
                params.append(val)
        sql += " WHERE " + " AND ".join(clauses)

    if order_by:
        if order_by not in valid_cols:
            raise ValueError(f"Unknown order_by column '{order_by}' for table '{table_name}'.")
        sql += f" ORDER BY {order_by} {'ASC' if ascending else 'DESC'}"

    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    return df if as_dataframe else df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Adapters: optimization-engine dicts -> this schema's column names
# ---------------------------------------------------------------------------
#
# port_optimizer.get_port_spec() and vessel_optimizer's per-class vessel_spec
# use field names that don't line up 1:1 with the ``ports``/``vessels``
# tables below (the two were built from separate task specs). Passing either
# dict straight into insert_data() raises "unknown column" for
# avg_turnaround_hours/avg_waiting_hours/tidal_restriction/data_source/
# n_vessels. These two adapters are the single place that translation
# happens, so insert_data() itself never has to special-case either engine.

def port_spec_to_db_row(port_spec: Dict) -> Dict:
    """Map a ``port_optimizer.get_port_spec()`` result onto the ``ports``
    table's column names. ``tidal_restriction`` has no column in this
    schema (not part of the original spec) and is intentionally dropped —
    it's still available from the engine directly if needed.
    """
    return {
        "port_name": port_spec.get("port_name"),
        "max_draft": port_spec.get("max_draft"),
        "max_loa": port_spec.get("max_loa"),
        "max_beam": port_spec.get("max_beam"),
        "max_dwt": port_spec.get("max_dwt"),
        "channel_depth": port_spec.get("channel_depth"),
        "berth_count": port_spec.get("berth_count"),
        "cargo_handling_rate": port_spec.get("cargo_handling_rate"),
        "turnaround_hours": port_spec.get("avg_turnaround_hours"),
        "waiting_hours": port_spec.get("avg_waiting_hours"),
        "congestion_level": port_spec.get("congestion_level"),
    }


def vessel_spec_to_db_row(vessel_spec: Dict, vessel_name: Optional[str] = None, vessel_type: Optional[str] = None) -> Dict:
    """Map a ``vessel_optimizer`` vessel_spec (``dwt``/``loa``/``beam``/
    ``draft``/``speed``/``year_built`` plus provenance fields) onto the
    ``vessels`` table's column names. ``data_source``/``n_vessels`` are
    provenance metadata, not physical specs, and have no column here.
    """
    return {
        "vessel_name": vessel_name or "Unnamed vessel",
        "vessel_type": vessel_type,
        "dwt": vessel_spec.get("dwt"),
        "loa": vessel_spec.get("loa"),
        "beam": vessel_spec.get("beam"),
        "draft": vessel_spec.get("draft"),
        "speed": vessel_spec.get("speed"),
        "year_built": vessel_spec.get("year_built"),
    }


# ---------------------------------------------------------------------------
# save_prediction() / get_latest_prediction()
# ---------------------------------------------------------------------------

def save_prediction(
    prediction_date: str,
    forecast_horizon: int,
    predicted_rate: float,
    trend: Optional[str] = None,
    model: Optional[str] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> int:
    """Append one row to the ``predictions`` log (never overwritten —
    every forecast run is kept as its own history entry).

    Parameters
    ----------
    prediction_date: the date the prediction was generated (``YYYY-MM-DD``).
    forecast_horizon: forecast horizon in days (e.g. 7, 30, 90).
    predicted_rate: the predicted freight rate for that horizon.
    trend: e.g. ``"INCREASING"`` / ``"STABLE"`` / ``"DECREASING"``.
    model: name of the model that produced the prediction.

    Returns
    -------
    The new row's ``id``.
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO predictions (prediction_date, forecast_horizon, predicted_rate, trend, model) "
            "VALUES (?, ?, ?, ?, ?)",
            (prediction_date, forecast_horizon, predicted_rate, trend, model),
        )
        return cursor.lastrowid


def save_forecast_result(
    forecast_result: Dict,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> int:
    """Convenience wrapper around :func:`save_prediction` that logs the
    output of ``forecasting.predict.forecast_freight()`` directly, so the
    forecasting engine and the database layer don't need any shared
    knowledge beyond that dict's shape.
    """
    prediction_date = forecast_result.get("generated_at", datetime.now(timezone.utc).isoformat())[:10]
    forecast_horizon = forecast_result["horizon_days"]
    predicted_rate = forecast_result["forecast"][-1]["predicted_freight_rate"]
    trend = forecast_result.get("market_signal")
    model = forecast_result.get("model_used")
    return save_prediction(prediction_date, forecast_horizon, predicted_rate, trend, model, db_path=db_path)


def get_latest_prediction(
    forecast_horizon: Optional[int] = None,
    model: Optional[str] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> Optional[Dict]:
    """Return the most recently generated prediction (optionally filtered
    by ``forecast_horizon`` and/or ``model``) as a dict, or ``None`` if the
    ``predictions`` table has no matching rows.
    """
    filters: Dict[str, Any] = {}
    if forecast_horizon is not None:
        filters["forecast_horizon"] = forecast_horizon
    if model is not None:
        filters["model"] = model

    rows = query_data(
        "predictions", filters=filters or None,
        order_by="id", ascending=False, limit=1,
        db_path=db_path, as_dataframe=False,
    )
    return rows[0] if rows else None


if __name__ == "__main__":
    import tempfile

    demo_path = Path(tempfile.gettempdir()) / "freightai_database_demo.db"
    demo_path.unlink(missing_ok=True)

    print("\n=== FreightAI Database Layer — demo ===")
    initialize_database(demo_path)
    print(f"Initialized database at {demo_path}")

    insert_data("freight_rates", [
        {"date": "2026-09-01", "freight_rate": 1450.5},
        {"date": "2026-09-02", "freight_rate": 1462.0},
    ], db_path=demo_path)
    insert_data("vessels", {"vessel_name": "MV Example", "vessel_type": "Panamax", "dwt": 82000,
                             "loa": 225.0, "beam": 32.3, "draft": 14.0, "speed": 14.2, "year_built": 2015},
                db_path=demo_path)
    insert_data("ports", {"port_name": "Paradip", "max_draft": 16.5, "max_loa": 300.0, "max_beam": 46.0,
                           "max_dwt": 155000, "channel_depth": 18.7, "berth_count": 18,
                           "cargo_handling_rate": 35000, "turnaround_hours": 36, "waiting_hours": 18,
                           "congestion_level": "MEDIUM"}, db_path=demo_path)

    df = query_data("freight_rates", order_by="date", db_path=demo_path)
    print(f"\nfreight_rates:\n{df}")

    pred_id = save_prediction("2026-09-10", 30, 1520.3, trend="INCREASING", model="linear_regression", db_path=demo_path)
    print(f"\nSaved prediction id={pred_id}")

    latest = get_latest_prediction(db_path=demo_path)
    print(f"Latest prediction: {latest}")

    demo_path.unlink(missing_ok=True)
