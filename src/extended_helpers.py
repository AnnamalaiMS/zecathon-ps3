"""Extended helpers for Zecathon PS3.

All forecasting features are constructed from observations strictly before the
target month. Conformal calibration uses only backtest errors whose target
months would have been published by the current forecast origin.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .features import available_as_of, month_start, last_known_month_end
from .metrics import summarize
from .buffer_policy import critical_quantile, policy_cost, buffer_from_forecast


def _feature_row(y, target_month):
    """Build target-month features using history ending before target_month."""
    y = np.asarray(y, dtype=float)
    t = pd.Timestamp(target_month)
    def lag(k):
        return float(y[-k]) if len(y) >= k else np.nan
    def mean(k):
        return float(np.mean(y[-k:])) if len(y) >= k else np.nan
    def std(k):
        return float(np.std(y[-k:], ddof=1)) if len(y) >= k + 1 else np.nan
    return {
        "month": float(t.month),
        "quarter_end": float(t.month in (3, 6, 9, 12)),
        "lag1": lag(1), "lag2": lag(2), "lag3": lag(3),
        "lag6": lag(6), "lag12": lag(12), "lag13": lag(13),
        "mean3": mean(3), "mean6": mean(6), "mean12": mean(12),
        "std6": std(6), "std12": std(12),
    }


def quantile_gbm(train_df, n_steps):
    """Recursive multi-step quantile GBM forecast; output matches src.models."""
    train = train_df.sort_values("month").reset_index(drop=True)
    y = train["net_flow_cr"].astype(float).to_numpy()
    months = pd.DatetimeIndex(train["month"])
    if len(y) < 30:
        # A deterministic seasonal-naive fallback keeps the interface usable
        # on short warm-up windows; report the fallback honestly if it occurs.
        from .models import seasonal_naive
        return seasonal_naive(train, n_steps)

    X, target = [], []
    # Supervised one-step samples; each feature row uses only earlier targets.
    for i in range(24, len(y)):
        X.append(_feature_row(y[:i], months[i]))
        target.append(y[i])
    X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target = np.asarray(target, dtype=float)

    models = {}
    for key, loss, alpha in [
        ("mean", "squared_error", None),
        ("q10", "quantile", 0.10),
        ("q90", "quantile", 0.90),
        ("q025", "quantile", 0.025),
        ("q975", "quantile", 0.975),
    ]:
        kwargs = {"loss": loss, "n_estimators": 100, "max_depth": 2,
                  "learning_rate": 0.03, "min_samples_leaf": max(2, min(6, len(target)//10)),
                  "random_state": 42}
        if alpha is not None:
            kwargs["alpha"] = alpha
        model = GradientBoostingRegressor(**kwargs)
        model.fit(X, target)
        models[key] = model

    history = list(y)
    last_month = months[-1]
    rows = []
    for step in range(1, n_steps + 1):
        target_month = last_month + pd.offsets.MonthEnd(step)
        feat = pd.DataFrame([_feature_row(history, target_month)]).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)
        pred = {k: float(m.predict(feat)[0]) for k, m in models.items()}
        lo80, hi80 = sorted((pred["q10"], pred["q90"]))
        lo95, hi95 = sorted((pred["q025"], pred["q975"]))
        mean = pred["mean"]
        # Keep nested intervals even if independently trained quantiles cross.
        lo95 = min(lo95, lo80, mean)
        hi95 = max(hi95, hi80, mean)
        lo80 = min(lo80, mean)
        hi80 = max(hi80, mean)
        rows.append({"mean": mean, "lo80": lo80, "hi80": hi80,
                     "lo95": lo95, "hi95": hi95})
        history.append(mean)
    return pd.DataFrame(rows)


def conformalize(bt_all, lag_months, scored_origins):
    """Expand intervals using only prior errors available at each scored origin.

    Uses absolute residual quantiles by forecast step. If fewer than 8 eligible
    residuals exist, raw intervals are retained for that row.
    """
    bt = bt_all.copy()
    scored = {month_start(o) for o in scored_origins}
    bt["origin"] = pd.to_datetime(bt["origin"]).map(month_start)
    bt["target_month"] = pd.to_datetime(bt["target_month"])
    out = bt[bt["origin"].isin(scored)].copy()
    for idx, row in out.iterrows():
        origin, step = row["origin"], int(row["step"])
        cutoff = last_known_month_end(origin, lag_months)
        eligible = bt[
            (bt["origin"] < origin) &
            (bt["step"].astype(int) == step) &
            (bt["target_month"] <= cutoff) &
            bt["actual"].notna()
        ].copy()
        if len(eligible) < 8:
            continue
        resid = np.abs(eligible["actual"].to_numpy(float) -
                       eligible["mean"].to_numpy(float))
        # Conservative finite-sample empirical quantiles.
        c80 = float(np.quantile(resid, min(1.0, np.ceil((len(resid)+1)*0.80)/len(resid))))
        c95 = float(np.quantile(resid, min(1.0, np.ceil((len(resid)+1)*0.95)/len(resid))))
        # Expand only; never narrow the raw interval.
        e80 = max(0.0, c80 - max(float(row["hi80"] - row["mean"]),
                                 float(row["mean"] - row["lo80"])))
        e95 = max(0.0, c95 - max(float(row["hi95"] - row["mean"]),
                                 float(row["mean"] - row["lo95"])))
        out.loc[idx, "lo80"] = float(row["lo80"]) - e80
        out.loc[idx, "hi80"] = float(row["hi80"]) + e80
        out.loc[idx, "lo95"] = float(row["lo95"]) - e95
        out.loc[idx, "hi95"] = float(row["hi95"]) + e95
    return out.reset_index(drop=True)


def monthly_costs(df, bt, lag_months, idle_rate, shortfall_rate, mult):
    """Per-origin step-1 costs for competing cash-buffer rules."""
    s1 = bt[bt["step"] == 1].dropna(subset=["actual"]).copy().sort_values("origin")
    outflow = np.maximum(-s1["actual"].to_numpy(float), 0.0)
    q = critical_quantile(idle_rate, shortfall_rate, mult)
    quantile = buffer_from_forecast(s1["mean"], s1["lo80"], s1["hi80"], q)
    point = np.maximum(-s1["mean"].to_numpy(float), 0.0)
    trailing = []
    for origin in s1["origin"]:
        known = available_as_of(df, origin, lag_months)["net_flow_cr"].tail(12).to_numpy(float)
        trailing.append(np.mean(np.maximum(-known, 0.0)) if len(known) else 0.0)
    return pd.DataFrame({
        "origin": s1["origin"].to_numpy(),
        "quantile_rule": policy_cost(quantile, outflow, idle_rate, shortfall_rate, mult),
        "point_forecast_only": policy_cost(point, outflow, idle_rate, shortfall_rate, mult),
        "trailing_12m_mean": policy_cost(np.asarray(trailing), outflow, idle_rate, shortfall_rate, mult),
    })


def quantile_sweep(bt, idle_rate, shortfall_rate, mult):
    s1 = bt[bt["step"] == 1].dropna(subset=["actual"])
    if s1.empty:
        return pd.DataFrame(columns=["q", "total_cost", "mean_cost"])
    outflow = np.maximum(-s1["actual"].to_numpy(float), 0.0)
    rows = []
    for q in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        b = buffer_from_forecast(s1["mean"], s1["lo80"], s1["hi80"], q)
        c = policy_cost(b, outflow, idle_rate, shortfall_rate, mult)
        rows.append({"q": q, "total_cost": float(c.sum()), "mean_cost": float(c.mean())})
    return pd.DataFrame(rows)


def ratio_sweep(df, bt, lag_months, idle_rate, shortfall_rate):
    s1 = bt[bt["step"] == 1].dropna(subset=["actual"])
    if s1.empty:
        return pd.DataFrame(columns=["shortfall_multiplier", "q_star", "total_cost", "mean_cost"])
    outflow = np.maximum(-s1["actual"].to_numpy(float), 0.0)
    rows = []
    for mult in (1.0, 2.0, 3.0, 4.0, 5.0):
        q = critical_quantile(idle_rate, shortfall_rate, mult)
        b = buffer_from_forecast(s1["mean"], s1["lo80"], s1["hi80"], q)
        c = policy_cost(b, outflow, idle_rate, shortfall_rate, mult)
        rows.append({"shortfall_multiplier": mult, "q_star": q,
                     "total_cost": float(c.sum()), "mean_cost": float(c.mean())})
    return pd.DataFrame(rows)


def bootstrap_diff(a, b, n_boot=5000, seed=42):
    """Paired bootstrap of monthly cost differences; negative favors rule a."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    if n == 0:
        return {"mean_diff": np.nan, "ci90_low": np.nan, "ci90_high": np.nan,
                "months_a_cheaper": 0, "n_months": 0}
    d = a[:n] - b[:n]
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(n_boot)])
    return {"mean_diff": float(d.mean()),
            "ci90_low": float(np.quantile(means, 0.05)),
            "ci90_high": float(np.quantile(means, 0.95)),
            "months_a_cheaper": int((d < 0).sum()), "n_months": int(n)}
