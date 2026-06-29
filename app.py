"""
Housing Price Dynamics in Portuguese Regions — interactive presentation app.

Reads pre-computed artifacts from ./artifacts/ (produced by export_artifacts.py)
and presents the historical series, 12-month forecasts, and housing-supply
scenarios for the SARIMAX regions. The app does NO modelling.

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
COURSE = "Time Series (2025/2026)"
FACULTY = "Faculty of Science - University of Lisbon"
GROUP_MEMBERS = "Carlos da Cruz, Mercè Cortes, Giovanna Chaves, Emmanuel Nkata"
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
    scenarios = pd.read_csv(ART / "scenarios.csv", parse_dates=["date"])
    acf_pacf = pd.read_csv(ART / "acf_pacf.csv")
    meta = json.loads((ART / "model_specs.json").read_text())
    return history, forecasts, scenarios, acf_pacf, meta


data = load_artifacts()
if data is None:
    st.error(
        "No `artifacts/` folder found. Run `python export_artifacts.py` first "
        "to generate the data the app reads."
    )
    st.stop()

history, forecasts, scenarios, acf_pacf, meta = data
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


def correlogram(sub, title, color):
    """Interactive ACF/PACF stem plot with a 95% confidence band (around zero)."""
    sub = sub.sort_values("lag")
    lags = sub["lag"].tolist()
    vals = sub["value"].tolist()
    lo, hi = sub["lo"].tolist(), sub["hi"].tolist()

    fig = go.Figure()
    # 95% significance band (centred on zero)
    fig.add_trace(go.Scatter(
        x=lags + lags[::-1], y=hi + lo[::-1], fill="toself",
        fillcolor="rgba(120,120,120,0.12)", line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    ))
    # stems
    for lg, v in zip(lags, vals):
        fig.add_shape(type="line", x0=lg, x1=lg, y0=0, y1=v,
                      line=dict(color=color, width=1.6))
    # markers (carry the hover)
    fig.add_trace(go.Scatter(
        x=lags, y=vals, mode="markers", marker=dict(color=color, size=7),
        showlegend=False, hovertemplate="lag %{x}<br>%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="black", width=0.8))
    fig.update_layout(title=title)
    fig = style_fig(fig, ytitle="", height=320)
    fig.update_xaxes(title="Lag", dtick=6)
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
        "Overview",
        "Methodology",
        "Region explorer",
        "Housing supply impact",
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
        "or jump straight to **Overview**, **Region explorer** and **Housing supply "
        "impact** to explore the results interactively."
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

    st.markdown("##### 2 · ARIMA")
    st.markdown(
        "Following the **Box-Jenkins** methodology, *p* and *q* were identified from "
        "the ACF/PACF of the differenced series. Nine candidate ARIMA(p,2,q) "
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

    st.markdown("##### 4 · SARIMAX")
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

    st.divider()
    st.markdown("##### ACF / PACF")
    st.caption(
        "The identification plots behind steps 2–3, up to 24 lags. "
        "**d = 2** is used to identify the ARIMA(p,2,q) orders (report Fig. 4); "
        "**d = 1** for the SARIMA grid search (report Fig. 5). Bars reaching outside "
        "the shaded 95% band indicate statistically significant autocorrelation."
    )
    c1, c2 = st.columns([3, 2])
    sel_short = c1.selectbox("Region", [short_names[r] for r in regions],
                             key="acf_region")
    diff = c2.radio("Differencing", [2, 1], horizontal=True,
                    format_func=lambda d: f"d = {d}", key="acf_diff")
    region = short_to_region[sel_short]
    sub = acf_pacf[(acf_pacf["region"] == region) & (acf_pacf["diff"] == diff)]

    pc, ac = st.columns(2)
    pc.plotly_chart(
        correlogram(sub[sub["kind"] == "PACF"], "PACF", "#e94560"),
        width="stretch",
    )
    ac.plotly_chart(
        correlogram(sub[sub["kind"] == "ACF"], "ACF", "#3498db"),
        width="stretch",
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
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", spec["type"], help=model_str)
    c2.metric(f"Value ({spec['anchor_date']})", f"€{spec['anchor_value']:,.0f}")
    c3.metric(f"Forecast ({spec['forecast_end']})",
              f"€{spec['forecast_end_value']:,.0f}", f"{spec['pct_change_12m']:+.1f}%")

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
# VIEW 3 — Housing supply impact (SARIMAX regions only)
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Housing supply impact":
    st.subheader("SARIMAX scenarios")
    st.caption(
        "Only the three **SARIMAX** regions use *number of new houses built* as a "
        "driver, so only they can answer this. Each model's 12-month forecast is "
        "re-run with new houses scaled from **−50% to +50%** versus the latest value "
        "(all other drivers held fixed). Redder = less building, bluer = more building."
    )

    # Diverging palette over the supply levels (−50 … 0 … +50).
    SUPPLY_COLORS = {
        -50: "#b2182b", -30: "#d6604d", -15: "#f4a582",
        0: "#444444",
        15: "#92c5de", 30: "#4393c3", 50: "#2166ac",
    }
    sx_regions = [r for r in regions if specs[r]["type"] == "SARIMAX"]
    levels = sorted(scenarios["supply_pct"].unique())

    def supply_panel(region, show_legend):
        sc = scenarios[scenarios["region"] == region]
        anchor_date = pd.to_datetime(specs[region]["anchor_date"])
        anchor_val = specs[region]["anchor_value"]
        hist = history[region].iloc[-24:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist.values, name="History",
            line=dict(color="#999", width=2), showlegend=False,
        ))
        for lv in levels:
            s = sc[sc["supply_pct"] == lv].sort_values("date")
            x = [anchor_date] + list(s["date"])
            y = [anchor_val] + list(s["point"])
            is_base = lv == 0
            fig.add_trace(go.Scatter(
                x=x, y=y,
                name="baseline" if is_base else f"{lv:+d}% supply",
                line=dict(
                    color=SUPPLY_COLORS.get(lv, "#888"),
                    width=3.2 if is_base else 1.6,
                    dash="solid" if is_base else "dot",
                ),
                showlegend=show_legend,
            ))
        fig.add_vline(x=anchor_date, line=dict(color="#bbb", width=1, dash="dot"))
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.0,
                        xanchor="left", x=0, font=dict(size=10),
                        bgcolor="rgba(0,0,0,0)", borderwidth=0),
        )
        return style_fig(fig, height=380)

    cols = st.columns(len(sx_regions))
    for c, region in zip(cols, sx_regions):
        c.markdown(f"**{short_names[region]}**")
        c.plotly_chart(supply_panel(region, show_legend=(region == sx_regions[0])),
                       width="stretch")

    st.subheader("Supply sensitivity — 12-month price change vs building more")
    st.caption(
        "Each line is the projected change from today's value at the 12-month "
        "horizon, as new-house supply is scaled. A **downward** slope means more "
        "building cools prices."
    )
    fig2 = go.Figure()
    for region in sx_regions:
        sc = scenarios[scenarios["region"] == region]
        anchor = specs[region]["anchor_value"]
        xs, ys = [], []
        for lv in levels:
            end = sc[sc["supply_pct"] == lv].sort_values("date")["point"].iloc[-1]
            xs.append(lv)
            ys.append((end / anchor - 1) * 100)
        fig2.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=short_names[region],
            line=dict(color=colors[region], width=2.6),
            marker=dict(size=7),
        ))
    fig2.add_vline(x=0, line=dict(color="#bbb", width=1, dash="dot"))
    fig2 = style_fig(fig2, ytitle="12-month price change (%)", height=420)
    fig2.update_xaxes(title="Change in new houses built (%)", ticksuffix="%")
    st.plotly_chart(fig2, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# VIEW — Key findings
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Key findings":
    st.subheader("Key findings")

    st.markdown("##### Forecasts (12 months, Mar 2026 → Mar 2027)")
    st.markdown(
        "- National prices rise **+13.4%** year-on-year.\n"
        "- All regions are projected to rise except **Alentejo**, which stays roughly "
        "flat (€1,351–€1,359/m²) and remains the most affordable.\n"
        "- **Grande Lisboa** is the most expensive region: €3,278 → **€3,755/m² (+14.5%)**.\n"
        "- **Península de Setúbal** has the largest increase, **+17.2%** "
        "(€2,653 → €3,110/m²), moving ahead of Algarve into second place.\n"
        "- **Algarve**: €2,840 → **€2,983/m² (+5.0%)**.\n"
        "- **Centro**: €1,363 → **€1,392/m² (+2.1%)**; **R.A. Açores** under +1%."
    )

    st.markdown("##### Model selection")
    st.markdown(
        "- **SARIMAX** was selected over SARIMA in **3 of 10 regions** (Oeste e Vale "
        "do Tejo, Grande Lisboa, R.A. Madeira), with AIC improvements of 4–4.8 "
        "points; SARIMA was kept elsewhere.\n"
        "- On the 12-month holdout, **SARIMA is more accurate in 8 of 10 regions**; "
        "only **Portugal** and **Oeste e Vale do Tejo** are better with SARIMAX.\n"
        "- The **number of new houses built** is the strongest predictor among the "
        "exogenous variables. In Grande Lisboa it is significant at 1%: ceteris "
        "paribus, +1,000 new houses lowers prices by **€89.6/m²**."
    )

    st.markdown("##### Housing-supply scenarios")
    st.markdown(
        "Forecasts were run for **+15%, +30% and +50%** in new houses built, beyond "
        "the baseline (exogenous variables held at their last known value):\n"
        "- At +50% vs. baseline, **Grande Lisboa** and **Oeste e Vale do Tejo** fall "
        "by about **1%**.\n"
        "- **R.A. Madeira** rises as supply increases."
    )

    st.warning(
        "**Limitations:** forecast confidence intervals are wide, and the univariate "
        "SARIMA forecasts do not incorporate economic variables."
    )
