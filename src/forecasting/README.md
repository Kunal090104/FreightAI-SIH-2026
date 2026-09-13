# FreightAI — Freight Forecasting Engine

This module trains and serves ML/time-series models that forecast future
dry-bulk freight market conditions from historical Baltic Dry Index (BDI)
data. It consumes the output of the data pipeline module
(`data/processed/freight_clean.csv`) and does **not** build the dashboard —
that's a separate, later module.

## ⚠️ Important limitation — read this first

**The Baltic Dry Index (BDI) is a broad dry-bulk market indicator**,
aggregated across many standardized shipping routes (Capesize, Panamax,
Supramax across benchmark routes worldwide). It is **not** the freight rate
for any specific route — for example, it is **not** the Australia → Paradip
rate, or any other single origin/destination pair.

Every forecast produced by this module carries a `limitations` field
restating this explicitly. **A production version of FreightAI should
replace or augment BDI with route-specific freight-rate data** before its
output is used for real chartering decisions on a specific route.

Similarly, the `market_signal` and `recommendation` fields are AI-assisted,
statistically-derived outputs — **not guaranteed financial or chartering
advice**. Every forecast result includes a `disclaimer` field saying so.

## Files

```
src/forecasting/
├── features.py     # shared feature engineering (train + predict use the same logic)
├── models.py        # MovingAverageBaseline + optional LSTMTabularRegressor
├── train.py          # loads data, trains all candidate models, selects & saves the best
├── predict.py        # loads the saved model, exposes forecast_freight()
└── evaluation.py     # MAE / RMSE / MAPE / R2 + model comparison utilities

models/                        # created by train.py
├── freight_model.pkl           # the selected model (joblib)
├── model_metrics.json           # full train/val/test metrics for every model tried
└── feature_columns.json         # the exact feature list/order the model expects
```

## Feature engineering

From `date` and `freight_rate` alone, twelve features are built:
`lag_1`, `lag_7`, `lag_14`, `lag_30`, `rolling_mean_7`, `rolling_mean_14`,
`rolling_mean_30`, `rolling_std_7`, `rolling_std_30`, `month`, `quarter`,
`day_of_week`.

**No data leakage**: every lag/rolling feature for the row predicting day
*t* is built only from values strictly before *t* (rolling windows are
computed on the series shifted by one step first). The first ~30 rows of
history are dropped since they can't have a full `lag_30`/`rolling_mean_30`.

## Models

| Model | Always trained? |
|---|---|
| Moving Average baseline (trailing 7-day mean) | Yes |
| Linear Regression | Yes |
| Random Forest | Yes |
| XGBoost | Only if `xgboost` is installed |
| LSTM | Only if `tensorflow` is installed **and** there are ≥500 training rows |

The baseline is trained and reported alongside the ML models on purpose —
if a fancier model can't beat "just average the last 7 days," that's
worth knowing. XGBoost/LSTM are skipped (with a logged, JSON-recorded
reason) rather than forced, per the "don't use LSTM for appearance" rule.

## Time-series split

Chronological only — **never shuffled**: 70% train / 15% validation / 15% test,
by row order after sorting by date. The best model is selected by the
**lowest validation MAE**.

## Evaluation

MAE, RMSE, MAPE, and R² (where meaningful — R²/MAPE are reported as `null`
when they're undefined, e.g. a validation split with a zero true value)
are computed for every model on train/validation/test and saved to
`models/model_metrics.json` for full comparison and audit.

## Forecasting & experimental labeling

`forecast_freight(days=N)` recursively forecasts N steps ahead — each
predicted value is fed back in as history for the next step's lag/rolling
features. Supported horizons: 7, 30, and 90 days.

Because recursive forecasting compounds its own error at longer horizons,
results are marked `"experimental": true` (with a human-readable reason)
whenever:
- the horizon exceeds 30 days (so **90-day forecasts are always experimental**),
- the selected model is the naive moving-average baseline, or
- the selected model's validation R² is negative.

## Market signal & recommendation

Based on the percentage move between the last known value and the forecast
horizon's end value (±3% threshold):

| Signal | Recommendation |
|---|---|
| INCREASING | BUY/CHARTER NOW |
| STABLE | MONITOR |
| DECREASING | WAIT |

Again: **AI-assisted, not guaranteed advice** — see `disclaimer` in every result.

## How to run training

```bash
python -m src.forecasting.train
```
Requires `data/processed/freight_clean.csv` to already exist (run the data
pipeline module first). Prints a comparison of every model's validation
metrics and which model was selected, and writes the three files under
`models/`.

## Example usage

```python
from src.forecasting.predict import forecast_freight

result = forecast_freight(days=30)
print(result["market_signal"], result["recommendation"])
print(result["forecast"][-1])          # {"date": "...", "predicted_freight_rate": ...}
print(result["limitations"])           # BDI route-specificity caveat
print(result["experimental"])          # False for 30-day (unless model is weak)

# Convenience wrappers
from src.forecasting.predict import forecast_7_days, forecast_30_days, forecast_90_days
forecast_90_days()                     # always experimental=True
```

The returned dict is plain JSON-serializable data (dates as ISO strings,
floats, bools) — ready for the Streamlit dashboard module to consume
directly, once that module is built.

## Dependencies

```
pandas, numpy, scikit-learn, joblib   # required
xgboost                                 # optional — enables XGBoost candidate
tensorflow                              # optional — enables LSTM candidate (also needs enough data)
```
See `requirements.txt`.
