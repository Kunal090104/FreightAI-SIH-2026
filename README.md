# FreightAI

**AI-Powered Bulk Freight & Chartering Intelligence**

An end-to-end decision-support prototype for dry-bulk freight/chartering:
data pipeline → freight forecasting → vessel/port/voyage optimization →
risk assessment → charter strategy → a Streamlit dashboard tying it all
together. Every module is independently importable and testable; the
dashboard is a thin UI layer that calls the real engines — it never
reimplements or fakes any of the underlying logic.

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard works even with no data at all — it falls back to clearly
labeled reference/demo values (or an honest "no data" message) rather than
crashing or pretending to have real data it doesn't. For real results:

```bash
# 1. Drop raw CSVs into data/raw/ (freight, vessels, ports, weather, etc.)
python -m src.data.pipeline           # clean + standardize -> data/processed/

# 2. Train the forecasting model
python -m src.forecasting.train        # -> models/freight_model.pkl, model_metrics.json

# 3. (optional) add data/raw/routes.csv for real sailing-distance calculations
```

## Project structure

```
FreightAI/
├── app.py                        # Streamlit entry point (streamlit run app.py)
├── data/{raw,processed}/          # raw inputs / cleaned outputs
├── database/freightai.db           # SQLite database (created on first run)
├── models/                          # trained forecasting model + metrics
├── requirements.txt
└── src/
    ├── data/                       # loading, cleaning, quality reporting
    ├── forecasting/                 # feature engineering, training, prediction
    ├── optimization/
    │   ├── vessel_optimizer.py       # Handysize/Supramax/Panamax/Capesize recommendation
    │   ├── port_optimizer.py          # 7 Indian East Coast ports compatibility
    │   ├── voyage_calculator.py        # sailing/port/idle time, total voyage duration
    │   └── charter_optimizer.py         # Spot/Short-Term/Medium-Term/COA strategy
    ├── risk/risk_engine.py            # Market/Port/Weather/Operational risk, explainable
    ├── database/database.py            # SQLite persistence layer (no UI dependency)
    └── dashboard/                       # Streamlit-independent orchestration + pages
        ├── state.py                      # run_full_analysis(), data-status detection
        ├── ui_helpers.py                  # badges, Plotly charts
        └── pages/                          # one module per sidebar page
```

Each module folder has its own README with full details, limitations, and
example usage — see `src/*/README.md`.

## Dashboard pages

Dashboard · Freight Forecast · Vessel Recommendation · Port Compatibility ·
Voyage Analysis · Risk Analysis · Charter Strategy · Data Quality

Fill in the Cargo/Contract form in the sidebar and click **Run Analysis**
to populate every page from one consistent set of inputs.

## Data status

Every section that could be running on placeholder data displays a clear
**REAL DATA** / **DEMO DATA** / **MIXED DATA** / **NO DATA AVAILABLE**
badge, derived from the provenance metadata the engines already track
(never guessed by the dashboard itself). Demo data is never presented as
real.

## Error handling

The dashboard is designed to never crash on missing datasets, an untrained
model, missing columns, invalid input, or unavailable port/route/weather
data — it shows a specific, actionable message instead (e.g. "Train the
forecasting model: `python -m src.forecasting.train`").

## Limitations

This is a prototype decision-support system. Forecasts, risk scores, and
charter recommendations use configurable heuristics and models that are
explicitly documented as prototypes throughout the codebase — see each
module's README for specifics (e.g. the Baltic Dry Index is a broad
market indicator, not a route-specific freight rate).
