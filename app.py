"""
Housing Price Dynamics in Portuguese Regions — interactive presentation app.

Reads pre-computed artifacts from ./artifacts/ (produced by export_artifacts.py)
and presents the historical series, 12-month forecasts, and model comparison.
The app does NO modelling.

    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# ✏️  EDIT ME — presentation cover details (shown on the Introduction page).
#     Fill these in by hand; they are not part of the analysis.
# ─────────────────────────────────────────────────────────────────────────────
COURSE = "‹ course name ›"
FACULTY = "‹ faculty / university ›"
GROUP_MEMBERS = "‹ group member names ›"
# ─────────────────────────────────────────────────────────────────────────────

ART = Path(__file__).parent / "artifacts"

st.set_page_config(
    page_title="Portuguese Housing Forecasts",
    page_icon="🏠",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_artifacts():
    if not ART.exists():
        return None
    history = pd.read_csv(ART / "history.csv", index_col="date", parse_dates=True)
    forecasts = pd.read_csv(ART / "forecasts.csv", parse_dates=["date"])
    metrics = pd.read_csv(ART / "metrics.csv")
    meta = json.loads((ART / "model_specs.json").read_text())
    return history, forecasts, metrics, meta


data = load_artifacts()
if data is None:
    st.error(
        "No `artifacts/` folder found. Run `python export_artifacts.py` first "
        "to generate the data the app reads."
    )
    st.stop()

history, forecasts, metrics, meta = data
colors = meta["colors"]
short_names = meta["short_names"]
specs = meta["specs"]
regions = list(history.columns)
# region -> short name and reverse
short_to_region = {short_names[r]: r for r in regions}

PLOT_BG = "#fafafa"


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def style_fig(fig, ytitle="€/m²", height=460):
    fig.update_layout(
        height=height,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(color="#444", size=13),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ddd",
                    borderwidth=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", linecolor="#ddd")
    fig.update_yaxes(title=ytitle, showgrid=True, gridcolor="#eee",
                     linecolor="#ddd", tickformat=",")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Header + navigation
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏠 Housing Price Dynamics in Portuguese Regions")
st.caption(
    "Median bank appraisal value of housing (€/m²), Jan 2011 – Mar 2026 · "
    "Source: INE · Time-series forecasting with ARIMA / SARIMA / SARIMAX"
)

st.sidebar.markdown("**Context**")
view = st.sidebar.radio(
    "View",
    [
        "Introduction",
        "Data",
        "Methodology",
        "Overview",
        "Region explorer",
        "Model comparison",
        "Key findings",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{meta['n_regions']}** series · 12-month forecasts")
st.sidebar.caption(meta["verdict"])


# ─────────────────────────────────────────────────────────────────────────────
# VIEW — Introduction
# ─────────────────────────────────────────────────────────────────────────────
if view == "Introduction":
    st.subheader("Housing Price Dynamics in Portuguese Regions: Time Series Forecasting")
    st.markdown(
        f"**Course:** {COURSE}  \n"
        f"**Faculty:** {FACULTY}  \n"
        f"**Group:** {GROUP_MEMBERS}"
    )
    st.divider()
    st.markdown(
        "The Portuguese housing market has become a topic of great discussion in "
        "Portugal but also internationally. It has undergone severe transformations "
        "over the past decade. Following the large contraction of 2010–2013 due to "
        "the 2008 financial crisis, housing prices have been increasing at a fast "
        "and sustained pace."
    )
    st.markdown(
        "This study applies two time-series modelling frameworks — **ARIMA / SARIMA** "
        "and **SARIMAX** — to the monthly *median bank appraisal value of housing* "
        "(€/m²) for ten Portuguese regions (the Portugal aggregate plus the nine "
        "NUTS-II regions) over **January 2011 to March 2026**, and produces "
        "**12-month-ahead forecasts** with 95% confidence intervals."
    )
    st.caption("Source: Instituto Nacional de Estatística (INE).")

    st.info(
        "Use the **Context** menu on the left to walk through the data and method, "
        "or jump straight to **Overview**, **Region explorer** and **Model "
        "comparison** to explore the results interactively."
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIEW — Data
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Data":
    st.subheader("Data")
    st.markdown(
        "The primary dataset is the **median bank appraisal value of housing per "
        "square meter** collected by INE — the median value attributed to "
        "residential properties by financial institutions, in €/m², published "
        "monthly. It covers the Portugal aggregate and the nine NUTS-II regions: "
        "Norte, Centro, Oeste e Vale do Tejo, Grande Lisboa, Península de Setúbal, "
        "Alentejo, Algarve, R.A. Açores and R.A. Madeira. The series runs from "
        "January 2011 to March 2026 — **183 monthly observations per region, with "
        "no missing values**."
    )
    st.markdown(
        "A set of macroeconomic and supply-side variables is added to give context "
        "to the price dynamics, in accordance with the literature:"
    )

    var_table = pd.DataFrame(
        [
            ["Median bank appraisal value", "Median bank appraisal value in €/m²",
             "INE", "Monthly", "2011–2026"],
            ["Inflation rate", "Consumer price inflation",
             "INE", "Monthly", "2011–2026"],
            ["Mortgage interest rate", "Implied rate on housing credit",
             "INE", "Monthly", "2012–2026"],
            ["New houses completed", "Number of new completed houses",
             "INE", "Quarterly", "2011–2026"],
            ["Unemployment rate", "Regional unemployment rate",
             "INE", "Quarterly", "2011–2026"],
        ],
        columns=["Variable", "Description", "Source", "Frequency", "Period"],
    )
    st.markdown("**Table 1 — Variable description**")
    st.dataframe(var_table, width="stretch", hide_index=True)

    st.markdown("##### Why these variables")
    st.markdown(
        "- **Inflation rate** — affects prices through two channels: it lowers the "
        "real cost of mortgage debt (making homeownership relatively more "
        "attractive) and raises the replacement cost of housing through higher "
        "construction costs (upward pressure on prices). Regional inflation for "
        "Oeste e Vale do Tejo and Península de Setúbal is unavailable for most of "
        "the sample, so models for these two regions exclude inflation.\n"
        "- **Mortgage interest rate** — available only at the national level and "
        "applied uniformly across regions. Rising rates increase mortgage costs, "
        "reducing demand and pressuring prices down.\n"
        "- **Number of new houses completed** — obtained quarterly and interpolated "
        "to monthly frequency (a straight line drawn between adjacent quarterly "
        "values). More new supply generally exerts downward pressure on prices.\n"
        "- **Unemployment rate** — obtained quarterly and interpolated to monthly. "
        "A proxy for labour-market conditions and household income uncertainty; "
        "higher unemployment reduces demand and tends to lower prices."
    )
    st.caption(
        "Other determinants mentioned in the literature — demographics and "
        "urbanization, internal/external migration, the concentration of economic "
        "activity in Lisbon, and foreign investment and tourism — could also explain "
        "price dynamics, but are outside the scope of this study."
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIEW — Methodology
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Methodology":
    st.subheader("Methodology")
    st.caption(
        "Two modelling frameworks applied to the monthly regional series over "
        "Jan 2011 – Mar 2026 (or Oct 2025, depending on the analysis)."
    )

    st.markdown("##### 1 · Stationarity testing")
    st.markdown(
        "Each series was tested for stationarity with the **Augmented Dickey-Fuller "
        "(ADF)** and **KPSS** tests; the integration order *d* was set by the number "
        "of differences needed for confirmed stationarity under both. Seasonality "
        "was checked through additive decomposition with a 12-month period. All ten "
        "series were non-stationary in levels; results conflicted at the first "
        "difference, so the **second difference** was used, where both tests confirm "
        "stationarity."
    )

    st.markdown("##### 2 · Univariate model — ARIMA")
    st.markdown(
        "Following the **Box-Jenkins** methodology, *p* and *q* were identified from "
        "the ACF/PACF of the differenced series. Six candidate ARIMA(p,2,q) "
        "specifications (*p,q ∈ {0,1,2}*) were estimated by maximum likelihood and "
        "compared by **AIC**; **ARIMA(2,2,2)** had the lowest AIC for nine of the ten "
        "series and was adopted for all ten for consistency. The Ljung-Box test then "
        "flagged significant residual autocorrelation at lag 12, motivating a "
        "seasonal model."
    )

    st.markdown("##### 3 · Seasonal model — SARIMA")
    st.markdown(
        "A full grid search over **SARIMA(p,1,q)(P,D,Q,12)** models (*p,q ∈ {0,1,2}*, "
        "*P,D,Q ∈ {0,1}*) was run. Because a valid model needs white-noise residuals, "
        "AIC was treated as secondary to passing the **Ljung-Box** test: the "
        "lowest-AIC model that passed was chosen. Residual diagnostics (time plot, "
        "ACF/PACF, histogram, Q-Q plot) confirmed approximately white-noise residuals."
    )

    st.markdown("##### 4 · Multivariate model — SARIMAX")
    st.markdown(
        "The framework was extended with four exogenous variables (inflation, "
        "mortgage rate, new housing completions, unemployment). A grid search over "
        "**36 SARIMAX(p,1,q)(P,D,Q,12)** specifications per region was conducted, with "
        "the same selection rule (pass Ljung-Box first, then lowest AIC). Inflation "
        "was excluded for Oeste e Vale do Tejo and Península de Setúbal."
    )

    st.markdown("##### 5 · Model comparison & forecast evaluation")
    st.markdown(
        "For a fair comparison, **ARIMA(2,2,2)** was refitted on the identical "
        "estimation sample used for each SARIMAX model. Out-of-sample accuracy was "
        "evaluated over a **12-month holdout** using **MAE**, **RMSE** and **MAPE**. "
        "These accuracy measures did not drive model selection, to avoid data leakage."
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Overview":
    pt = specs["Portugal"]
    latest = history.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("National value (Mar 2026)", f"€{pt['anchor_value']:,.0f}/m²")
    c2.metric("National 12-month forecast", f"€{pt['forecast_end_value']:,.0f}/m²",
              f"{pt['pct_change_12m']:+.1f}%")
    c3.metric("Most expensive region",
              max(short_to_region, key=lambda s: latest[short_to_region[s]]),
              f"€{latest.max():,.0f}/m²")

    st.subheader("Median appraisal value by region")
    fig = go.Figure()
    for r in regions:
        is_pt = r == "Portugal"
        fig.add_trace(go.Scatter(
            x=history.index, y=history[r], name=short_names[r],
            line=dict(color=colors[r], width=3.4 if is_pt else 1.6),
            opacity=1.0 if is_pt else 0.8,
        ))
    st.plotly_chart(style_fig(fig, height=560), width="stretch")

    st.subheader("Current value ranking (Mar 2026)")
    ranked = latest.sort_values(ascending=True)
    bar = go.Figure(go.Bar(
        x=ranked.values, y=[short_names[r] for r in ranked.index],
        orientation="h", marker_color=[colors[r] for r in ranked.index],
        text=[f"€{v:,.0f}" for v in ranked.values], textposition="auto",
    ))
    bar.update_layout(height=440)
    st.plotly_chart(style_fig(bar, ytitle="", height=440), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2 — Region explorer
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Region explorer":
    sel_short = st.selectbox("Region", [short_names[r] for r in regions])
    region = short_to_region[sel_short]
    spec = specs[region]
    color = colors[region]

    model_str = f"{spec['type']}{tuple(spec['order'])}{tuple(spec['seasonal_order'])}"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", spec["type"], help=model_str)
    c2.metric(f"Value ({spec['anchor_date']})", f"€{spec['anchor_value']:,.0f}")
    c3.metric(f"Forecast ({spec['forecast_end']})",
              f"€{spec['forecast_end_value']:,.0f}", f"{spec['pct_change_12m']:+.1f}%")
    c4.metric("AIC", f"{spec['aic']:,.1f}")

    fc = forecasts[forecasts["region"] == region].sort_values("date")
    hist = history[region].iloc[-48:]
    anchor_date = pd.to_datetime(spec["anchor_date"])
    anchor_val = spec["anchor_value"]

    # Connect last observed point to the forecast line / band
    fc_dates = [anchor_date] + list(fc["date"])
    fc_point = [anchor_val] + list(fc["point"])
    fc_lo = [anchor_val] + list(fc["lower95"])
    fc_hi = [anchor_val] + list(fc["upper95"])

    fig = go.Figure()
    # 95% CI band
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1], y=fc_hi + fc_lo[::-1],
        fill="toself", fillcolor=_rgba(color, 0.15), line=dict(width=0),
        hoverinfo="skip", showlegend=True, name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, name="Historical",
        line=dict(color=color, width=2.4),
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_point, name="Forecast",
        line=dict(color=color, width=2.4, dash="dash"),
    ))
    fig.add_vline(x=anchor_date, line=dict(color="#aaa", width=1, dash="dot"))
    fig.update_layout(title=f"{sel_short} — {model_str}")
    st.plotly_chart(style_fig(fig, height=520), width="stretch")

    if spec["exog"]:
        st.caption("Exogenous variables: " + ", ".join(spec["exog"]))

    with st.expander("Forecast values"):
        show = fc[["date", "point", "lower95", "upper95"]].copy()
        show["date"] = show["date"].dt.strftime("%Y-%m")
        show.columns = ["Month", "Forecast (€/m²)", "Lower 95%", "Upper 95%"]
        st.dataframe(show, width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 3 — Model comparison
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Model comparison":
    st.subheader("Final model vs ARIMA(2,2,2) baseline — 12-month holdout")
    st.caption(
        "Each region's chosen model (SARIMA or SARIMAX) is compared against a "
        "plain ARIMA(2,2,2) refit on the identical training window. "
        "MAE / RMSE in €/m², MAPE in %."
    )

    tbl = metrics.copy()
    tbl = tbl[[
        "short_name", "chosen_model",
        "final_mae", "arima_mae",
        "final_rmse", "arima_rmse",
        "final_mape", "arima_mape", "better",
    ]]
    tbl.columns = [
        "Region", "Chosen",
        "Chosen MAE", "ARIMA MAE",
        "Chosen RMSE", "ARIMA RMSE",
        "Chosen MAPE", "ARIMA MAPE", "Better",
    ]
    st.dataframe(
        tbl.style.format({
            "Chosen MAE": "{:.1f}", "ARIMA MAE": "{:.1f}",
            "Chosen RMSE": "{:.1f}", "ARIMA RMSE": "{:.1f}",
            "Chosen MAPE": "{:.2f}%", "ARIMA MAPE": "{:.2f}%",
        }),
        width="stretch", hide_index=True,
    )

    st.info(meta["verdict"])

    st.subheader("MAPE by region — chosen model vs ARIMA(2,2,2)")
    m = metrics.sort_values("short_name")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m["short_name"], y=m["final_mape"],
                         name="Chosen model", marker_color="#3498db"))
    fig.add_trace(go.Bar(x=m["short_name"], y=m["arima_mape"],
                         name="ARIMA(2,2,2)", marker_color="#e94560"))
    fig.update_layout(barmode="group")
    st.plotly_chart(style_fig(fig, ytitle="MAPE (%)", height=460),
                    width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# VIEW — Key findings
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Key findings":
    st.subheader("Key findings")

    st.markdown("##### Forecasts (12 months, Mar 2026 → Mar 2027)")
    st.markdown(
        "- All regions except **Alentejo** are projected to continue their upward "
        "trajectory; nationally, prices are expected to keep rising fast — a "
        "**+13.4% year-on-year** increase.\n"
        "- **Grande Lisboa** remains the most expensive region by a substantial "
        "margin: €3,278/m² → **€3,511/m² (+7.1%)**.\n"
        "- **Península de Setúbal** shows the **largest increase of all regions, "
        "+17.2%** (€2,653/m² → €3,110/m²), likely overtaking Algarve for second "
        "place — possibly spillover from buyers priced out of Grande Lisboa.\n"
        "- **Algarve** reaches €2,983/m² (**+5%**).\n"
        "- **Alentejo** stays the most affordable and roughly flat (~€1,351–€1,359/m²); "
        "**Centro** rises modestly to €1,392/m² (**+2.1%**) and **R.A. Açores** under 1%."
    )

    st.markdown("##### Model selection")
    st.markdown(
        "- **SARIMAX** was selected over SARIMA in **3 of 10 regions** — Oeste e Vale "
        "do Tejo, Grande Lisboa and R.A. Madeira — with small AIC improvements of "
        "4–4.8 points; elsewhere the simpler **SARIMA** was kept.\n"
        "- On out-of-sample accuracy, **SARIMA outperforms SARIMAX in 8 of 10 "
        "regions** — so in some markets the exogenous variables add explanatory power "
        "but not predictive accuracy.\n"
        "- The **number of new houses built** is the strongest predictor among the "
        "variables used. In Grande Lisboa it is significant at 1%: ceteris paribus, "
        "+10,000 new houses lowers prices by about **€896/m²**."
    )

    st.markdown("##### Policy scenario — housing supply")
    st.markdown(
        "Beyond the baseline (exogenous variables held at their last known value), "
        "forecasts were also run for **+15%, +30% and +50%** in new houses built — "
        "reflecting the Portuguese government's announced housing fiscal package "
        "(reduced VAT, among other measures). A +50% supply increase moves Grande "
        "Lisboa and Oeste e Vale do Tejo prices by only ~1%, highlighting a market "
        "dominated by **price momentum** and short-run growth that is highly "
        "**inelastic to construction**."
    )

    st.warning(
        "**Limitations:** forecast confidence intervals are wide, which limits "
        "precise inference, and the univariate SARIMA forecasts do not incorporate "
        "economic variables."
    )
