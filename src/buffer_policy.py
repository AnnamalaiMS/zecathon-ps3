"""Cash-buffer policy. Cost = idle_rate*max(B-O,0) + mult*shortfall_rate*max(O-B,0),
where O is the realised month's net outflow (max(-net_flow, 0)).
With per-rupee costs the optimal buffer is the q* quantile of the outflow
distribution, q* = mult*p / (mult*p + d)  (newsvendor critical ratio)."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import available_as_of
from .models import Z80


def critical_quantile(idle_rate, shortfall_rate, mult):
    return mult * shortfall_rate / (mult * shortfall_rate + idle_rate)


def policy_cost(buffer, outflow, idle_rate, shortfall_rate, mult):
    buffer, outflow = np.asarray(buffer, float), np.asarray(outflow, float)
    return idle_rate * np.maximum(buffer - outflow, 0) + mult * shortfall_rate * np.maximum(outflow - buffer, 0)


def buffer_from_forecast(mean, lo80, hi80, q):
    """Normal approximation fitted to the 80% interval (an assumption to state in the report)."""
    sigma = (np.asarray(hi80) - np.asarray(lo80)) / (2 * Z80)
    net_q = np.asarray(mean) + norm.ppf(1 - q) * sigma  # net-flow quantile = outflow q-quantile, sign flipped
    return np.maximum(-net_q, 0.0)


def evaluate_policies(df, bt, lag_months, idle_rate, shortfall_rate, mult, window=12):
    """Evaluate buffers for the month of each origin (step 1) against its realised outflow."""
    q = critical_quantile(idle_rate, shortfall_rate, mult)
    s1 = bt[bt.step == 1].dropna(subset=["actual"]).copy()
    outflow = np.maximum(-s1["actual"].values, 0)
    hist_max, hist_mean = [], []
    for o in s1["origin"]:
        known = np.maximum(-available_as_of(df, o, lag_months)["net_flow_cr"].tail(window).values, 0)
        hist_max.append(known.max())
        hist_mean.append(known.mean())
    rules = {
        f"forecast_quantile_q{q:.2f}": buffer_from_forecast(s1["mean"], s1["lo80"], s1["hi80"], q),
        "point_forecast_only": np.maximum(-s1["mean"].values, 0),
        f"trailing_{window}m_max": np.array(hist_max),
        f"trailing_{window}m_mean": np.array(hist_mean),
    }
    rows = []
    for name, b in rules.items():
        c = policy_cost(b, outflow, idle_rate, shortfall_rate, mult)
        rows.append({"rule": name, "total_cost": c.sum(), "mean_cost": c.mean(),
                     "months_with_shortfall": int((outflow > b).sum()), "mean_buffer": float(np.mean(b)), "n_months": len(c)})
    return pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)
