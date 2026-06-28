"""
Export artifacts for the Streamlit app.

Reproduces the data preparation (notebook cell 18) and the FINAL per-region
models (cell 23), plus a fair ARIMA(2,2,2)-baseline holdout comparison
(adapted from cell 22), and writes everything the app needs into ./artifacts/.

The app does NO modelling — it only reads these files. Re-run this script
whenever the input CSVs or model specs change:

    python export_artifacts.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings("ignore")

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Presentation metadata (lifted from the notebook)
# ─────────────────────────────────────────────────────────────────────────────
colors = {
    "Portugal": "#1a1a2e",
    "Norte": "#e94560",
    "Centro": "#f5a623",
    "Oeste e Vale do Tejo": "#2ecc71",
    "Grande Lisboa": "#3498db",
    "Península de Setúbal": "#9b59b6",
    "Alentejo": "#e67e22",
    "Algarve": "#1abc9c",
    "Região Autónoma dos Açores": "#c0392b",
    "Região Autónoma da Madeira": "#34495e",
}

short_names = {
    "Portugal": "Portugal",
    "Norte": "Norte",
    "Centro": "Centro",
    "Oeste e Vale do Tejo": "Oeste e Vale do Tejo",
    "Grande Lisboa": "Grande Lisboa",
    "Península de Setúbal": "Península de Setúbal",
    "Alentejo": "Alentejo",
    "Algarve": "Algarve",
    "Região Autónoma dos Açores": "R.A. Açores",
    "Região Autónoma da Madeira": "R.A. Madeira",
}

# Final model specifications chosen in the report (notebook cell 23)
sarima_specs = {
    "Portugal": {"r_order": (1, 1, 2), "s_order": (0, 0, 1, 12)},
    "Norte": {"r_order": (1, 1, 2), "s_order": (0, 0, 1, 12)},
    "Centro": {"r_order": (1, 1, 2), "s_order": (0, 0, 1, 12)},
    "Península de Setúbal": {"r_order": (1, 1, 2), "s_order": (0, 0, 1, 12)},
    "Alentejo": {"r_order": (2, 1, 2), "s_order": (1, 0, 0, 12)},
    "Algarve": {"r_order": (2, 1, 2), "s_order": (1, 0, 1, 12)},
    "Região Autónoma dos Açores": {"r_order": (2, 1, 2), "s_order": (0, 0, 1, 12)},
}

sarimax_specs = {
    "Oeste e Vale do Tejo": {"r_order": (1, 1, 1), "s_order": (1, 0, 0, 12)},
    "Grande Lisboa": {"r_order": (1, 1, 1), "s_order": (1, 0, 0, 12)},
    "Região Autónoma da Madeira": {"r_order": (1, 1, 2), "s_order": (1, 0, 0, 12)},
}

no_inflation = ["Oeste e Vale do Tejo", "Península de Setúbal"]
FORECAST_STEPS = 12
HOLDOUT = 12
ARIMA_BASELINE = (2, 2, 2)
ACF_LAGS = 24
# Supply scenarios: scale "new houses built" relative to its latest value.
SUPPLY_LEVELS = [-0.50, -0.30, -0.15, 0.0, 0.15, 0.30, 0.50]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load + prepare data (notebook cell 18)
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(
        "avaliacao_bancaria_total.csv",
        index_col="date", parse_dates=True, encoding="latin-1",
    ).astype(float)
    df = df.drop(columns=["Continente"], errors="ignore")
    regions = df.columns.tolist()

    inflation = pd.read_csv(
        "inflation.csv", index_col="date", parse_dates=True, encoding="latin-1"
    ).astype(float)[regions]["2011-01-01":]

    mortgage_raw = pd.read_csv(
        "mortgage_interest_rate.csv",
        index_col="date", parse_dates=True, encoding="latin-1",
    )["2011-01-01":]
    mortgage = pd.DataFrame(index=mortgage_raw.index)
    for reg in regions:
        mortgage[reg] = mortgage_raw["Portugal"]

    new_houses_q = pd.read_csv(
        "nr_new_houses.csv", index_col="date", parse_dates=True, encoding="latin-1"
    ).astype(float)[regions]["2011-01-01":]
    new_houses = new_houses_q.resample("MS").asfreq().interpolate(method="linear")

    unemploy_q = pd.read_csv(
        "unemployment_rate.csv", index_col="date", parse_dates=True, encoding="latin-1"
    ).astype(float)[regions]["2011-01-01":]
    unemploy = unemploy_q.resample("MS").asfreq().interpolate(method="linear")

    end_date = min(
        new_houses.index.max(), unemploy.index.max(),
        inflation.index.max(), mortgage.index.max(), df.index.max(),
    )
    return df, inflation, mortgage, new_houses, unemploy, end_date, regions


def build_exog(col, inflation, mortgage, new_houses, unemploy):
    cols = {
        "mortgage": mortgage[col],
        "new_houses": new_houses[col],
        "unemployment": unemploy[col],
    }
    if col not in no_inflation:
        cols = {"inflation": inflation[col], **cols}
    return pd.DataFrame(cols)


def fit_sample(col, df, inflation, mortgage, new_houses, unemploy, end_date):
    """Return (y_fit, exog_fit) on the common, complete sample for a SARIMAX region."""
    exog = build_exog(col, inflation, mortgage, new_houses, unemploy)
    y = df[col]
    common = y.index.intersection(exog.index)
    common = common[common <= end_date]
    y_fit, exog_fit = y.loc[common], exog.loc[common]
    valid = y_fit.notna() & exog_fit.notna().all(axis=1)
    return y_fit[valid], exog_fit[valid]


def fit_sample_full(col, df, inflation, mortgage, new_houses, unemploy):
    """(y, exog) on the FULL target sample for a SARIMAX region.

    Exog series end before the target (e.g. new houses stops in Oct 2025), so
    the regressors are held flat at their last observed value to cover the rest
    of the target sample (and back-filled for any leading gap). This anchors the
    SARIMAX forecast at the target's last month, matching the SARIMA forecasts
    and avoiding a shifted anchor across regions.
    """
    exog = build_exog(col, inflation, mortgage, new_houses, unemploy)
    y = df[col].dropna()
    exog_full = exog.reindex(y.index).ffill().bfill()
    return y, exog_full


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fit final models + 12-month forecasts (notebook cell 23)
# ─────────────────────────────────────────────────────────────────────────────
def fit_final(df, inflation, mortgage, new_houses, unemploy, end_date, regions):
    forecast_rows = []
    specs_out = {}

    for col in regions:
        if col in sarima_specs:
            spec = sarima_specs[col]
            y = df[col]
            model = SARIMAX(y, order=spec["r_order"],
                            seasonal_order=spec["s_order"]).fit(disp=False)
            fc = model.get_forecast(steps=FORECAST_STEPS)
            mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
            mtype, exog_vars = "SARIMA", []
        else:
            spec = sarimax_specs[col]
            y, exog_fit = fit_sample_full(col, df, inflation, mortgage,
                                          new_houses, unemploy)
            model = SARIMAX(y, exog=exog_fit, order=spec["r_order"],
                            seasonal_order=spec["s_order"]).fit(disp=False)
            future_exog = pd.DataFrame(
                [exog_fit.iloc[-1].values] * FORECAST_STEPS,
                columns=exog_fit.columns,
            )
            fc = model.get_forecast(steps=FORECAST_STEPS, exog=future_exog)
            mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
            mtype, exog_vars = "SARIMAX", list(exog_fit.columns)

        last_date = y.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=FORECAST_STEPS, freq="MS",
        )
        for d, m, lo, hi in zip(future_dates, mean.values,
                                ci.iloc[:, 0].values, ci.iloc[:, 1].values):
            forecast_rows.append({
                "region": col, "short_name": short_names[col],
                "date": d, "point": round(float(m), 1),
                "lower95": round(float(lo), 1), "upper95": round(float(hi), 1),
                "model": mtype,
            })

        first, last = mean.iloc[0], mean.iloc[-1]
        pct = (last / y.iloc[-1] - 1) * 100
        specs_out[col] = {
            "short_name": short_names[col],
            "type": mtype,
            "order": list(spec["r_order"]),
            "seasonal_order": list(spec["s_order"]),
            "exog": exog_vars,
            "aic": round(float(model.aic), 1),
            "anchor_date": last_date.strftime("%Y-%m"),
            "anchor_value": round(float(y.iloc[-1]), 1),
            "forecast_start": future_dates[0].strftime("%Y-%m"),
            "forecast_end": future_dates[-1].strftime("%Y-%m"),
            "forecast_end_value": round(float(last), 1),
            "pct_change_12m": round(float(pct), 1),
        }
        print(f"  {short_names[col]:<24} {mtype:<8} AIC={model.aic:8.1f}  "
              f"12m {pct:+.1f}%")

    return pd.DataFrame(forecast_rows), specs_out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Holdout comparison: final model vs ARIMA(2,2,2) baseline (adapted cell 22)
# ─────────────────────────────────────────────────────────────────────────────
def holdout_metrics(df, inflation, mortgage, new_houses, unemploy, end_date, regions):
    rows = []
    for col in regions:
        is_sarimax = col in sarimax_specs
        if is_sarimax:
            spec = sarimax_specs[col]
            y_full, exog_full = fit_sample(col, df, inflation, mortgage,
                                           new_houses, unemploy, end_date)
        else:
            spec = sarima_specs[col]
            y_full, exog_full = df[col], None

        if len(y_full) <= HOLDOUT + 10:
            continue

        y_train, y_test = y_full.iloc[:-HOLDOUT], y_full.iloc[-HOLDOUT:]

        # ARIMA(2,2,2) baseline — same training window, no exog
        arima = ARIMA(y_train, order=ARIMA_BASELINE).fit()
        arima_f = arima.forecast(steps=HOLDOUT)

        # Final chosen model
        if is_sarimax:
            ex_tr, ex_te = exog_full.iloc[:-HOLDOUT], exog_full.iloc[-HOLDOUT:]
            final = SARIMAX(y_train, exog=ex_tr, order=spec["r_order"],
                            seasonal_order=spec["s_order"]).fit(disp=False)
            final_f = final.forecast(steps=HOLDOUT, exog=ex_te)
            label = "SARIMAX"
        else:
            final = SARIMAX(y_train, order=spec["r_order"],
                            seasonal_order=spec["s_order"]).fit(disp=False)
            final_f = final.forecast(steps=HOLDOUT)
            label = "SARIMA"

        def metrics(pred):
            mae = mean_absolute_error(y_test, pred)
            rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
            mape = float(np.mean(np.abs((y_test - pred) / y_test)) * 100)
            return round(mae, 1), round(rmse, 1), round(mape, 2)

        a_mae, a_rmse, a_mape = metrics(arima_f)
        f_mae, f_rmse, f_mape = metrics(final_f)
        rows.append({
            "region": col, "short_name": short_names[col],
            "chosen_model": label,
            "arima_mae": a_mae, "arima_rmse": a_rmse, "arima_mape": a_mape,
            "final_mae": f_mae, "final_rmse": f_rmse, "final_mape": f_mape,
            "better": "Chosen" if f_mape < a_mape else "ARIMA(2,2,2)",
        })
        print(f"  {short_names[col]:<24} {label:<8} "
              f"MAPE chosen={f_mape:5.2f}%  ARIMA={a_mape:5.2f}%")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Housing-supply scenarios for the SARIMAX regions
#     Re-run each final SARIMAX forecast with "new houses built" scaled by
#     ±X% versus its latest value (all other exogenous drivers held flat).
#     ARIMA/SARIMA regions have no supply lever, so they are excluded.
# ─────────────────────────────────────────────────────────────────────────────
def supply_scenarios(df, inflation, mortgage, new_houses, unemploy):
    rows = []
    supply_meta = {}
    for col in sarimax_specs:
        spec = sarimax_specs[col]
        y, exog = fit_sample_full(col, df, inflation, mortgage, new_houses, unemploy)
        model = SARIMAX(y, exog=exog, order=spec["r_order"],
                        seasonal_order=spec["s_order"]).fit(disp=False)

        base_last = exog.iloc[-1].copy()
        anchor = float(y.iloc[-1])
        last_date = y.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=FORECAST_STEPS, freq="MS",
        )

        for lv in SUPPLY_LEVELS:
            future_exog = pd.DataFrame(
                [base_last.values] * FORECAST_STEPS, columns=exog.columns
            )
            future_exog["new_houses"] = base_last["new_houses"] * (1 + lv)
            mean = model.get_forecast(steps=FORECAST_STEPS, exog=future_exog).predicted_mean
            for d, m in zip(future_dates, mean.values):
                rows.append({
                    "region": col, "short_name": short_names[col],
                    "supply_pct": int(round(lv * 100)),
                    "date": d, "point": round(float(m), 1),
                })

        coef = float(model.params.get("new_houses", float("nan")))
        supply_meta[col] = {
            "short_name": short_names[col],
            "coef_per_house": round(coef, 5),
            "eur_per_10k_houses": round(coef * 10_000, 0),
            "new_houses_baseline": round(float(base_last["new_houses"]), 0),
            "anchor_value": round(anchor, 1),
        }
        print(f"  {short_names[col]:<24} new_houses coef={coef:+.4f}  "
              f"(≈ {coef * 10_000:+,.0f} €/m² per 10k homes)")
    return pd.DataFrame(rows), supply_meta


# ─────────────────────────────────────────────────────────────────────────────
# 4. ACF / PACF for model identification (notebook cells 8 [d=2] and 9 [d=1])
# ─────────────────────────────────────────────────────────────────────────────
def acf_pacf_table(df, regions):
    """ACF/PACF values + 95% confidence bands (centred at zero, as in plot_acf)
    for the first- and second-differenced series, lags 0–24."""
    rows = []
    for col in regions:
        for d in (1, 2):
            series = df[col]
            for _ in range(d):
                series = series.diff()
            series = series.dropna()

            a, a_ci = acf(series, nlags=ACF_LAGS, alpha=0.05)
            p, p_ci = pacf(series, nlags=ACF_LAGS, alpha=0.05)
            for kind, vals, ci in (("ACF", a, a_ci), ("PACF", p, p_ci)):
                for lag in range(len(vals)):
                    rows.append({
                        "region": col,
                        "short_name": short_names[col],
                        "diff": d,
                        "kind": kind,
                        "lag": lag,
                        "value": round(float(vals[lag]), 4),
                        # band relative to zero, matching statsmodels plot_acf
                        "lo": round(float(ci[lag, 0] - vals[lag]), 4),
                        "hi": round(float(ci[lag, 1] - vals[lag]), 4),
                    })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data ...")
    df, inflation, mortgage, new_houses, unemploy, end_date, regions = load_data()
    print(f"  price series: {df.index.min():%Y-%m} → {df.index.max():%Y-%m} "
          f"({len(df)} obs), common-sample end {end_date:%Y-%m}\n")

    # History
    hist = df.copy()
    hist.index.name = "date"
    hist.to_csv(ART / "history.csv")

    print("Computing ACF / PACF (d=1, d=2) ...")
    acf_pacf_table(df, regions).to_csv(ART / "acf_pacf.csv", index=False)

    print("Fitting final models + forecasts ...")
    fc_df, specs = fit_final(df, inflation, mortgage, new_houses,
                             unemploy, end_date, regions)
    fc_df.to_csv(ART / "forecasts.csv", index=False)

    print("\nHoldout comparison vs ARIMA(2,2,2) ...")
    met = holdout_metrics(df, inflation, mortgage, new_houses,
                          unemploy, end_date, regions)
    met.to_csv(ART / "metrics.csv", index=False)

    print("\nHousing-supply scenarios (SARIMAX regions) ...")
    scen_df, supply_meta = supply_scenarios(df, inflation, mortgage,
                                            new_houses, unemploy)
    scen_df.to_csv(ART / "scenarios.csv", index=False)

    sarimax_regions = [short_names[c] for c in sarimax_specs]
    meta = {
        "specs": specs,
        "sarimax_regions": sarimax_regions,
        "supply": supply_meta,
        "supply_levels": [int(round(l * 100)) for l in SUPPLY_LEVELS],
        "n_regions": len(regions),
        "verdict": (
            f"SARIMAX was selected over SARIMA in {len(sarimax_specs)} of "
            f"{len(regions)} regions: {', '.join(sarimax_regions)}. "
            "Elsewhere the simpler SARIMA model was kept."
        ),
        "colors": colors,
        "short_names": short_names,
    }
    (ART / "model_specs.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"\nArtifacts written to {ART.resolve()}/")
    for f in sorted(ART.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
