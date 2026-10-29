"""Leakage tests. These use SYNTHETIC data for testing only; no result in the report may come from here."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import make_origins, rolling_backtest
from src.features import available_as_of, last_known_month_end
from src.models import MODELS


def synth(n=84, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2018-01-31", periods=n, freq="ME")
    base = rng.normal(0, 300, n)
    qe = np.isin(months.month, [3, 6, 9, 12])
    return pd.DataFrame({"month": months, "net_flow_cr": base - 1500 * qe})


def test_last_known_month_with_lag_two():
    assert last_known_month_end("2026-05-01", 2) == pd.Timestamp("2026-03-31")
    assert last_known_month_end("2026-05-20", 2) == pd.Timestamp("2026-03-31")


def test_nothing_after_cutoff_visible():
    df = synth()
    tr = available_as_of(df, "2022-05-01", 2)
    assert tr["month"].max() == pd.Timestamp("2022-03-31")


def test_poisoned_future_changes_nothing_in_training():
    df = synth()
    df2 = df.copy()
    df2.loc[df2["month"] > "2022-03-31", "net_flow_cr"] = 1e9
    assert available_as_of(df, "2022-05-01", 2).equals(available_as_of(df2, "2022-05-01", 2))


@pytest.mark.parametrize("name", list(MODELS))
def test_poisoned_future_changes_no_forecast(name):
    df = synth()
    origins = make_origins(df["month"].max(), 3, 3)
    a = rolling_backtest(df, MODELS[name], origins, 3, 2)
    df2 = df.copy()
    poison_from = pd.Timestamp(origins[0]) - pd.DateOffset(months=1)  # strictly after the first origin's cutoff
    df2.loc[df2["month"] > last_known_month_end(origins[0], 2), "net_flow_cr"] = 1e9
    b = rolling_backtest(df2.assign(), MODELS[name], origins[:1], 3, 2)
    a1 = a[a.origin == origins[0]][["mean", "lo80", "hi80", "lo95", "hi95"]].reset_index(drop=True)
    b1 = b[["mean", "lo80", "hi80", "lo95", "hi95"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a1, b1)


def test_targets_are_correct_months():
    df = synth()
    origins = make_origins(df["month"].max(), 2, 3)
    bt = rolling_backtest(df, MODELS["seasonal_naive"], origins, 3, 2)
    first = bt[bt.origin == origins[0]]
    expect = [(pd.Timestamp(origins[0]) + pd.DateOffset(months=h)) + pd.offsets.MonthEnd(0) for h in range(3)]
    assert list(first.target_month) == expect
    assert first["actual"].notna().all()
