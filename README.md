# Portuguese Housing Price Forecasts — Streamlit app

Interactive front-end for the time-series study of the median bank appraisal
value of housing (€/m²) across Portugal and its nine NUTS-II regions.

## How it works

The app does **no modelling**. All numbers come from `artifacts/`, which are
produced once by `export_artifacts.py` — it reproduces the data preparation and
the final per-region models (SARIMA / SARIMAX) from `code.ipynb`.

```
export_artifacts.py  ──reads──▶  *.csv (INE data)
                     ──writes─▶  artifacts/{history,forecasts,metrics,acf_pacf}.csv + model_specs.json
app.py               ──reads──▶  artifacts/   (Streamlit + Plotly)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# 1. Generate artifacts (re-run only when CSVs or model specs change)
python export_artifacts.py

# 2. Launch the app
streamlit run app.py
```

## Views

- **Overview** — all-region series, current ranking, national headline forecast.
- **Region explorer** — pick a region; history + 12-month forecast with 95% CI,
  plus which model was selected.
- **Model comparison** — chosen model vs an ARIMA(2,2,2) baseline on a 12-month
  holdout (MAE / RMSE / MAPE) and the SARIMA-vs-SARIMAX verdict.

## Input data

`avaliacao_bancaria_total.csv` (target), `inflation.csv`,
`mortgage_interest_rate.csv`, `nr_new_houses.csv`, `unemployment_rate.csv`.
