
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data" / "processed" / "flows.csv"
CONFIG = ROOT / "config.yaml"

st.set_page_config(
    page_title="Zecathon | Fund Flow Intelligence",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.3rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(115deg, #10243a, #174c63);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 2rem;}
    .hero p {margin: .45rem 0 0 0; color: #d8e8f0;}
    .note {font-size: .88rem; color: #657386;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load project files
# ---------------------------------------------------------
@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_config():
    with open(CONFIG, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


try:
    cfg = load_config()
except Exception as exc:
    st.error(f"Could not read config.yaml: {exc}")
    st.stop()

forecasts = load_csv(str(OUTPUTS / "final_forecasts.csv"))
all_forecasts = load_csv(str(OUTPUTS / "final_forecasts_all_models.csv"))
metrics = load_csv(str(OUTPUTS / "metrics_extended.csv"))
policy = load_csv(str(OUTPUTS / "policy_extended.csv"))
flows = load_csv(str(DATA))

if forecasts.empty:
    st.error("Missing outputs/final_forecasts.csv. Run python run_forecast.py first.")
    st.stop()

forecasts["target_month"] = pd.to_datetime(forecasts["target_month"])
forecasts["point_forecast_cr"] = pd.to_numeric(
    forecasts["point_forecast_cr"], errors="coerce"
)

for col in ["lo80_cr", "hi80_cr", "lo95_cr", "hi95_cr"]:
    forecasts[col] = pd.to_numeric(forecasts[col], errors="coerce")

if not metrics.empty:
    for col in ["step", "MASE", "cov80", "cov95", "IS80", "IS95"]:
        if col in metrics:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")

if not policy.empty:
    for col in ["total_cost", "mean_cost", "mean_buffer",
                "months_with_shortfall", "n_months"]:
        if col in policy:
            policy[col] = pd.to_numeric(policy[col], errors="coerce")

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.title("Dashboard controls")

categories = sorted(forecasts["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox(
    "Fund category",
    categories,
    index=0,
)

models_available = sorted(forecasts["model"].dropna().unique().tolist())
selected_model = st.sidebar.selectbox(
    "Forecast model",
    models_available,
    index=models_available.index("seasonal_naive")
    if "seasonal_naive" in models_available else 0,
)

show_95 = st.sidebar.checkbox("Show 95% prediction interval", value=True)
show_80 = st.sidebar.checkbox("Show 80% prediction interval", value=True)

cutoff = cfg.get("data_cutoff", "Not configured")
lag = cfg.get("publication_lag_months", "Not configured")
horizon = cfg.get("horizon", 3)
costs = cfg.get("costs", {})

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Fund Flow Intelligence</h1>
      <p>Zecathon 2026 · PS3 Financial Forecasting ·
      Liquid and Overnight Mutual Fund Categories</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Decision-support prototype | Monthly net-flow forecasts, "
    "uncertainty, model validation and cash-buffer policy."
)

# ---------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------
cat_forecasts = forecasts[
    (forecasts["category"] == selected_category)
    & (forecasts["model"] == selected_model)
].sort_values("target_month")

if cat_forecasts.empty:
    st.warning("No forecasts available for this selection.")
    st.stop()

last_train = (
    cat_forecasts["last_training_month"].iloc[0]
    if "last_training_month" in cat_forecasts.columns else "Unknown"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Fund category", selected_category.title())
m2.metric("Forecast model", selected_model.replace("_", " ").title())
m3.metric("Latest training month", str(last_train))
m4.metric("Forecast horizon", f"{horizon} months")

st.divider()

# ---------------------------------------------------------
# Forecast visualization
# ---------------------------------------------------------
st.subheader("1. Three-month net-flow forecast")
st.caption(
    "Values are in the source dataset's net_flow_cr units. "
    "Negative forecasts indicate net outflows; positive forecasts indicate net inflows."
)

fig = go.Figure()
x = cat_forecasts["target_month"]

if show_95:
    fig.add_trace(go.Scatter(
        x=x,
        y=cat_forecasts["hi95_cr"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        name="95% upper",
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=cat_forecasts["lo95_cr"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(80, 130, 190, 0.14)",
        name="95% prediction interval",
        hovertemplate="Lower bound: %{y:,.0f}<extra></extra>",
    ))

if show_80:
    fig.add_trace(go.Scatter(
        x=x,
        y=cat_forecasts["hi80_cr"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        name="80% upper",
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=cat_forecasts["lo80_cr"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(30, 110, 160, 0.25)",
        name="80% prediction interval",
        hovertemplate="Lower bound: %{y:,.0f}<extra></extra>",
    ))

fig.add_trace(go.Scatter(
    x=x,
    y=cat_forecasts["point_forecast_cr"],
    mode="lines+markers",
    name="Point forecast",
    line=dict(width=3),
    marker=dict(size=9),
    customdata=cat_forecasts[["lo80_cr", "hi80_cr", "lo95_cr", "hi95_cr"]]
    .to_numpy(),
    hovertemplate=(
        "%{x|%b %Y}<br>Forecast: %{y:,.0f}"
        "<br>80% interval: %{customdata[0]:,.0f} to %{customdata[1]:,.0f}"
        "<br>95% interval: %{customdata[2]:,.0f} to %{customdata[3]:,.0f}"
        "<extra></extra>"
    ),
))

fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.update_layout(
    height=430,
    xaxis_title="Target month",
    yaxis_title="Net flow (source units)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Forecast table**")
display_forecasts = cat_forecasts[
    ["target_month", "point_forecast_cr",
     "lo80_cr", "hi80_cr", "lo95_cr", "hi95_cr"]
].copy()
display_forecasts["target_month"] = (
    display_forecasts["target_month"].dt.strftime("%B %Y")
)
display_forecasts = display_forecasts.rename(columns={
    "target_month": "Target month",
    "point_forecast_cr": "Point forecast",
    "lo80_cr": "80% lower",
    "hi80_cr": "80% upper",
    "lo95_cr": "95% lower",
    "hi95_cr": "95% upper",
})
st.dataframe(
    display_forecasts.style.format({
        "Point forecast": "{:,.0f}",
        "80% lower": "{:,.0f}",
        "80% upper": "{:,.0f}",
        "95% lower": "{:,.0f}",
        "95% upper": "{:,.0f}",
    }),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------
# Historical flow context
# ---------------------------------------------------------
if not flows.empty and {"month", "category", "net_flow_cr"}.issubset(flows.columns):
    st.subheader("2. Historical monthly net flows")
    hist = flows.copy()
    hist["month"] = pd.to_datetime(hist["month"])
    hist = hist[hist["category"] == selected_category].sort_values("month")

    if not hist.empty:
        history_fig = go.Figure()
        history_fig.add_trace(go.Scatter(
            x=hist["month"],
            y=hist["net_flow_cr"],
            mode="lines",
            name="Observed net flow",
            line=dict(width=1.8),
        ))
        history_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        history_fig.update_layout(
            height=320,
            xaxis_title="Month",
            yaxis_title="Net flow (source units)",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(history_fig, use_container_width=True)

# ---------------------------------------------------------
# Model performance
# ---------------------------------------------------------
st.subheader("3. Rolling-origin model validation")
st.caption(
    "Metrics are calculated from historical backtests, not from the future forecast. "
    "Lower MASE and interval score are preferable; coverage should be interpreted "
    "against nominal 80% and 95% levels."
)

if not metrics.empty:
    cat_metrics = metrics[metrics["category"] == selected_category].copy()

    if not cat_metrics.empty:
        metric_summary = (
            cat_metrics.groupby(["model", "variant"], as_index=False)
            .agg(
                Mean_MASE=("MASE", "mean"),
                Mean_80_coverage=("cov80", "mean"),
                Mean_95_coverage=("cov95", "mean"),
                Mean_80_interval_score=("IS80", "mean"),
                Mean_95_interval_score=("IS95", "mean"),
            )
            .sort_values("Mean_MASE")
        )

        col1, col2 = st.columns(2)

        with col1:
            mase_fig = go.Figure()
            for variant in metric_summary["variant"].dropna().unique():
                sub = metric_summary[metric_summary["variant"] == variant]
                mase_fig.add_trace(go.Bar(
                    x=sub["model"],
                    y=sub["Mean_MASE"],
                    name=str(variant),
                ))
            mase_fig.update_layout(
                title="Mean MASE by model",
                barmode="group",
                height=360,
                xaxis_title="Model",
                yaxis_title="Mean MASE",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(mase_fig, use_container_width=True)

        with col2:
            coverage_fig = go.Figure()
            coverage_fig.add_hline(
                y=0.80, line_dash="dash", line_color="gray",
                annotation_text="80% target",
            )
            coverage_fig.add_hline(
                y=0.95, line_dash="dot", line_color="gray",
                annotation_text="95% target",
            )
            for variant in metric_summary["variant"].dropna().unique():
                sub = metric_summary[metric_summary["variant"] == variant]
                coverage_fig.add_trace(go.Bar(
                    x=sub["model"],
                    y=sub["Mean_80_coverage"],
                    name=f"{variant} — 80%",
                ))
                coverage_fig.add_trace(go.Bar(
                    x=sub["model"],
                    y=sub["Mean_95_coverage"],
                    name=f"{variant} — 95%",
                ))
            coverage_fig.update_layout(
                title="Mean prediction-interval coverage",
                barmode="group",
                height=360,
                yaxis=dict(tickformat=".0%", range=[0, 1.05]),
                xaxis_title="Model",
                yaxis_title="Empirical coverage",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(coverage_fig, use_container_width=True)

        st.dataframe(
            metric_summary.style.format({
                "Mean_MASE": "{:.3f}",
                "Mean_80_coverage": "{:.1%}",
                "Mean_95_coverage": "{:.1%}",
                "Mean_80_interval_score": "{:,.1f}",
                "Mean_95_interval_score": "{:,.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Mean coverage aggregates across forecast steps and variants. "
            "Individual-step results may differ; consult outputs/metrics_extended.csv."
        )
    else:
        st.info("No model metrics found for this category.")
else:
    st.warning("outputs/metrics_extended.csv is missing.")

# ---------------------------------------------------------
# Cash-buffer policy
# ---------------------------------------------------------
st.subheader("4. Cash-buffer policy comparison")
st.caption(
    "Policy costs are historical backtest scores under the configured cost function. "
    "They are not guaranteed future savings or actual rupee costs unless the units "
    "and rates have been economically calibrated."
)

if not policy.empty:
    cat_policy = policy[policy["category"] == selected_category].copy()

    if not cat_policy.empty:
        cat_policy = cat_policy.sort_values("total_cost")
        top_policy = cat_policy.head(10).copy()

        policy_fig = go.Figure()
        policy_fig.add_trace(go.Bar(
            x=top_policy["total_cost"],
            y=top_policy["rule"].astype(str) + " | " +
              top_policy["model"].astype(str) + " | " +
              top_policy["variant"].astype(str),
            orientation="h",
            name="Backtested total cost",
            hovertemplate="Rule: %{y}<br>Total cost: %{x:,.0f}<extra></extra>",
        ))
        policy_fig.update_layout(
            title="Ten lowest-cost tested policies",
            height=450,
            xaxis_title="Total backtested cost (project units)",
            yaxis_title="Policy / model / variant",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(policy_fig, use_container_width=True)

        best = cat_policy.iloc[0]
        a, b, c = st.columns(3)
        a.metric("Lowest tested cost", f"{best['total_cost']:,.0f}")
        b.metric("Rule", str(best["rule"]))
        c.metric("Months with shortfall",
                 f"{best['months_with_shortfall']:.0f} / {best['n_months']:.0f}")

        st.dataframe(cat_policy, use_container_width=True, hide_index=True)
    else:
        st.info("No policy results found for this category.")
else:
    st.warning("outputs/policy_extended.csv is missing.")

# ---------------------------------------------------------
# Methodology and limitations
# ---------------------------------------------------------
st.subheader("5. Methodology, assumptions and limitations")

left, right = st.columns(2)

with left:
    st.markdown("**Forecast controls**")
    st.write(f"- Configured cutoff: `{cutoff}`")
    st.write(f"- Publication lag: `{lag}` month(s)")
    st.write(f"- Forecast horizon: `{horizon}` months")
    st.write(f"- Last eligible training month in selected output: `{last_train}`")
    st.write("- Selected final forecast: the saved forecast file for the chosen model.")

    st.markdown("**Cost assumptions**")
    st.write(f"- Idle cash rate: `{costs.get('idle_rate', 'Not specified')}`")
    st.write(f"- Shortfall rate: `{costs.get('shortfall_rate', 'Not specified')}`")
    st.write(
        f"- Shortfall penalty multiplier: "
        f"`{costs.get('shortfall_multiplier', 'Not specified')}`"
    )

with right:
    st.markdown("**Interpretation safeguards**")
    st.write(
        "- Forecast intervals express model uncertainty; they are not guarantees."
    )
    st.write(
        "- Historical interval coverage does not guarantee future calibration."
    )
    st.write(
        "- A two-month lag is a configured assumption; verify actual source release dates."
    )
    st.write(
        "- The current model set includes seasonal naive, ETS and tax-dummy SARIMAX. "
        "The tax dummy does not replace measured banking liquidity, money-market yields "
        "or market-return data."
    )
    st.write(
        "- Buffer-policy rankings depend on the project's chosen cost assumptions."
    )

if not all_forecasts.empty:
    st.download_button(
        "Download all model forecasts (CSV)",
        data=all_forecasts.to_csv(index=False).encode("utf-8"),
        file_name="final_forecasts_all_models.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    '<p class="note">Zecathon 2026 · PS3 Financial Forecasting · '
    'Decision-support prototype. Validate data availability and operational '
    'cost assumptions before using forecasts for live fund positioning.</p>',
    unsafe_allow_html=True,
)
