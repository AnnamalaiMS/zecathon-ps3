import numpy as np
import pandas as pd


def mase_scale(train_series, m=12):
    y = np.asarray(train_series, dtype=float)
    return np.mean(np.abs(y[m:] - y[:-m]))


def mase(actual, mean, scale):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(mean))) / scale)


def coverage(actual, lo, hi):
    a = np.asarray(actual)
    return float(np.mean((a >= np.asarray(lo)) & (a <= np.asarray(hi))))


def interval_score(actual, lo, hi, alpha):
    a, lo, hi = map(np.asarray, (actual, lo, hi))
    return float(np.mean((hi - lo) + (2 / alpha) * np.maximum(lo - a, 0) + (2 / alpha) * np.maximum(a - hi, 0)))


def summarize(bt, scale):
    out = []
    for step, g in bt.dropna(subset=["actual"]).groupby("step"):
        out.append({"step": step, "n": len(g),
                    "MASE": mase(g.actual, g["mean"], scale),
                    "cov80": coverage(g.actual, g.lo80, g.hi80),
                    "cov95": coverage(g.actual, g.lo95, g.hi95),
                    "IS80": interval_score(g.actual, g.lo80, g.hi80, 0.20),
                    "IS95": interval_score(g.actual, g.lo95, g.hi95, 0.05)})
    return pd.DataFrame(out)
