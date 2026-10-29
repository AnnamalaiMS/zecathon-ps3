"""Forecasters. Each takes (train_df, n_steps) and returns a DataFrame with
columns mean, lo80, hi80, lo95, hi95, one row per month ahead.
train_df has columns `month` (month-end) and `net_flow_cr`."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import MONTH_END, quarter_end_flag

Z80 = norm.ppf(0.90)
Z95 = norm.ppf(0.975)


def _series(train_df):
    s = train_df.sort_values("month").set_index("month")["net_flow_cr"].astype(float)
    expected = pd.date_range(s.index[0], periods=len(s), freq=MONTH_END)
    if not s.index.equals(expected):
        raise ValueError("Training series has missing or duplicate months; fix data first.")
    s.index = expected
    return s


def _future_index(s, n):
    return pd.date_range(s.index[-1] + pd.offsets.MonthEnd(1), periods=n, freq=MONTH_END)


def _pack(mean, lo80, hi80, lo95, hi95):
    return pd.DataFrame({"mean": np.asarray(mean), "lo80": np.asarray(lo80), "hi80": np.asarray(hi80),
                         "lo95": np.asarray(lo95), "hi95": np.asarray(hi95)})


def seasonal_naive(train_df, n_steps):
    y = _series(train_df).values
    if len(y) < 18:
        raise ValueError("Need at least 18 months for seasonal naive.")
    sigma = np.std(y[12:] - y[:-12], ddof=1)
    k = np.arange(1, n_steps + 1)
    mean = np.array([y[len(y) - 12 + ((i - 1) % 12)] for i in k])
    sd = sigma * np.sqrt(np.ceil(k / 12))
    return _pack(mean, mean - Z80 * sd, mean + Z80 * sd, mean - Z95 * sd, mean + Z95 * sd)


def ets(train_df, n_steps):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    s = _series(train_df)
    scale = s.std() or 1.0
    ys = s / scale
    res = None
    if len(ys) >= 30:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ETSModel(ys, error="add", trend="add", damped_trend=True, seasonal="add",
                               seasonal_periods=12, initialization_method="estimated").fit(disp=False)
        except Exception:
            res = None
    if res is None:  # short history or failed fit: non-seasonal damped ETS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ETSModel(ys, error="add", trend="add", damped_trend=True,
                           initialization_method="estimated").fit(disp=False)
    pred = res.get_prediction(start=len(ys), end=len(ys) + n_steps - 1)
    f80, f95 = pred.summary_frame(alpha=0.2), pred.summary_frame(alpha=0.05)
    return _pack(f80["mean"].values * scale, f80["pi_lower"].values * scale, f80["pi_upper"].values * scale,
                 f95["pi_lower"].values * scale, f95["pi_upper"].values * scale)


def sarimax(train_df, n_steps):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    s = _series(train_df)
    scale = s.std() or 1.0
    ys = s / scale
    exog = quarter_end_flag(s.index).reshape(-1, 1)
    fut = _future_index(s, n_steps)
    exog_f = quarter_end_flag(fut).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAX(ys, exog=exog, order=(1, 0, 1), seasonal_order=(1, 0, 0, 12), trend="c").fit(disp=False, maxiter=200)
        f = m.get_forecast(n_steps, exog=exog_f)
        c80, c95 = f.conf_int(alpha=0.2), f.conf_int(alpha=0.05)
    return _pack(f.predicted_mean.values * scale, c80.iloc[:, 0].values * scale, c80.iloc[:, 1].values * scale,
                 c95.iloc[:, 0].values * scale, c95.iloc[:, 1].values * scale)


MODELS = {"seasonal_naive": seasonal_naive, "ets": ets, "sarimax_taxdummy": sarimax}
