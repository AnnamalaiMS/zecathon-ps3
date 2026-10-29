"""Point-in-time data access. Every model sees data through this module only.

Convention: an *origin* is the first day of the month in which the forecast is
made (first-of-the-month positioning). AMFI publishes month m-1 data around the
8th-10th of month m, so on the 1st of month m the latest usable month is m-2.
That is `publication_lag_months: 2` in config.yaml.
"""
import pandas as pd

MONTH_END = "ME"


def month_start(origin):
    return pd.Timestamp(origin).to_period("M").to_timestamp()


def last_known_month_end(origin, lag_months):
    o = month_start(origin)
    return (o - pd.DateOffset(months=lag_months)) + pd.offsets.MonthEnd(0)


def available_as_of(df, origin, lag_months, date_col="month"):
    """Rows whose month had been published on or before `origin`."""
    cutoff = last_known_month_end(origin, lag_months)
    out = df[df[date_col] <= cutoff].copy()
    assert out.empty or out[date_col].max() <= cutoff
    return out


def quarter_end_flag(month_ends):
    """1 for Mar/Jun/Sep/Dec (advance-tax months). A calendar fact, so it is
    the one kind of feature that may legitimately be 'known in advance'."""
    idx = pd.DatetimeIndex(month_ends)
    return idx.month.isin([3, 6, 9, 12]).astype(float)
