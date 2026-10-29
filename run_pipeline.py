"""Run everything: python run_pipeline.py
Reads  data/processed/flows.csv  (columns: month, category, net_flow_cr[, aum_cr])
Writes outputs/*.csv and outputs/*.png. Every number in the report comes from here."""
import argparse
import os
import numpy as np
import pandas as pd
import yaml

from src.backtest import make_origins, rolling_backtest
from src.buffer_policy import critical_quantile, evaluate_policies
from src.features import available_as_of
from src.metrics import mase_scale, summarize
from src.models import MODELS


def load_flows(path, cfg):
    flows = pd.read_csv(path, parse_dates=["month"])
    flows["month"] = flows["month"] + pd.offsets.MonthEnd(0)
    cutoff = str(cfg.get("data_cutoff", ""))
    if cutoff and cutoff[0].isdigit() and "Y" not in cutoff:
        if flows["month"].max() > pd.Timestamp(cutoff):
            raise ValueError("Data extends past the declared data_cutoff.")
    else:
        print("WARNING: data_cutoff in config.yaml is still a placeholder.")
    return flows


def run(flows, cfg, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(cfg.get("seed", 42))
    H, N, LAG = cfg["horizon"], cfg["n_origins"], cfg["publication_lag_months"]
    c = cfg["costs"]
    q = critical_quantile(c["idle_rate"], c["shortfall_rate"], c["shortfall_multiplier"])
    all_metrics, all_policy = [], []
    for cat in cfg["categories"]:
        df = flows[flows.category == cat][["month", "net_flow_cr"]].sort_values("month").reset_index(drop=True)
        if df.empty:
            print(f"No data for category {cat}; skipping.")
            continue
        origins = make_origins(df["month"].max(), N, H)
        scale = mase_scale(available_as_of(df, origins[0], LAG)["net_flow_cr"])
        for name, fn in MODELS.items():
            try:
                bt = rolling_backtest(df, fn, origins, H, LAG)
            except Exception as e:  # report honestly instead of hiding a failure
                print(f"[{cat}/{name}] failed: {e}")
                continue
            bt.to_csv(f"{outdir}/backtest_{cat}_{name}.csv", index=False)
            m = summarize(bt, scale)
            m.insert(0, "model", name)
            m.insert(0, "category", cat)
            all_metrics.append(m)
            p = evaluate_policies(df, bt, LAG, c["idle_rate"], c["shortfall_rate"], c["shortfall_multiplier"])
            p.insert(0, "forecast_model", name)
            p.insert(0, "category", cat)
            all_policy.append(p)
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    policy = pd.concat(all_policy, ignore_index=True) if all_policy else pd.DataFrame()
    metrics.to_csv(f"{outdir}/metrics.csv", index=False)
    policy.to_csv(f"{outdir}/policy_costs.csv", index=False)
    print(f"Critical quantile q* = {q:.3f}")
    return metrics, policy


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", default="data/processed/flows.csv")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    metrics, policy = run(load_flows(a.flows, cfg), cfg)
    pd.set_option("display.width", 200)
    print(metrics.round(3).to_string(index=False))
    print(policy.round(1).to_string(index=False))
