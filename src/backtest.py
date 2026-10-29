import pandas as pd
from .features import available_as_of, month_start, last_known_month_end


def make_origins(last_actual_month, n_origins, horizon):
    """Origins whose full horizon has realised actuals, so every step is scored."""
    last_period = pd.Timestamp(last_actual_month).to_period("M") - (horizon - 1)
    return [p.to_timestamp() for p in pd.period_range(end=last_period, periods=n_origins, freq="M")]


def rolling_backtest(df, model_fn, origins, horizon, lag_months, min_train=24):
    """df: columns month (month-end), net_flow_cr for ONE category.
    At each origin the model is refit on data published by then only."""
    rows = []
    for o in origins:
        o = month_start(o)
        train = available_as_of(df, o, lag_months)
        if len(train) < min_train:
            raise ValueError(f"Only {len(train)} months of training data at origin {o.date()}; need {min_train}.")
        assert train["month"].max() <= last_known_month_end(o, lag_months)
        n_steps = lag_months + horizon - 1
        fc = model_fn(train, n_steps).iloc[lag_months - 1:].reset_index(drop=True)
        fc["origin"] = o
        fc["step"] = range(1, horizon + 1)
        fc["target_month"] = [(o + pd.DateOffset(months=h)) + pd.offsets.MonthEnd(0) for h in range(horizon)]
        rows.append(fc)
    bt = pd.concat(rows, ignore_index=True)
    actual = df[["month", "net_flow_cr"]].rename(columns={"month": "target_month", "net_flow_cr": "actual"})
    return bt.merge(actual, on="target_month", how="left")
