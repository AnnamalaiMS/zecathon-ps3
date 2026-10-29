"""python run_extended.py   -> adds ML model, calibrated intervals and robustness checks.
Writes outputs/metrics_extended.csv, policy_extended.csv, sweep_quantile.csv, sweep_ratio.csv, bootstrap.csv"""
import os
import numpy as np
import pandas as pd
import yaml

from run_pipeline import load_flows
from src.backtest import make_origins, rolling_backtest
from src.buffer_policy import critical_quantile, evaluate_policies
from src.extras import (bootstrap_diff, conformalize, monthly_costs, quantile_gbm, quantile_sweep, ratio_sweep)
from src.features import available_as_of
from src.metrics import mase_scale, summarize
from src.models import MODELS

EXTRA = 18  # unscored warm-up origins used only to build calibration errors


def main(flows_path="data/processed/flows.csv", config="config.yaml", outdir="outputs"):
    cfg = yaml.safe_load(open(config))
    flows = load_flows(flows_path, cfg)
    os.makedirs(outdir, exist_ok=True)
    H, N, LAG = cfg["horizon"], cfg["n_origins"], cfg["publication_lag_months"]
    c = cfg["costs"]
    idle, short, mult = c["idle_rate"], c["shortfall_rate"], c["shortfall_multiplier"]
    models = dict(MODELS, quantile_gbm=quantile_gbm)
    met, pol, swq, swr, boot = [], [], [], [], []
    for cat in cfg["categories"]:
        df = flows[flows.category == cat][["month", "net_flow_cr", "aum_cr"]].sort_values("month").reset_index(drop=True)
        scored = make_origins(df["month"].max(), N, H)
        ext = make_origins(df["month"].max(), N + EXTRA, H)
        scale = mase_scale(available_as_of(df, scored[0], LAG)["net_flow_cr"])
        for name, fn in models.items():
            try:
                bt_all = rolling_backtest(df, fn, ext, H, LAG)
            except Exception as e:
                print(f"[{cat}/{name}] failed: {e}")
                continue
            variants = {"raw": bt_all[bt_all.origin.isin(scored)].reset_index(drop=True),
                        "conformal": conformalize(bt_all, LAG, scored)}
            variants["conformal"].to_csv(f"{outdir}/bt_{cat}_{name}_conformal.csv", index=False)
            for var, bt in variants.items():
                m = summarize(bt, scale)
                m.insert(0, "variant", var); m.insert(0, "model", name); m.insert(0, "category", cat)
                met.append(m)
                p = evaluate_policies(df, bt, LAG, idle, short, mult)
                p.insert(0, "variant", var); p.insert(0, "model", name); p.insert(0, "category", cat)
                pol.append(p)
                s = quantile_sweep(bt, idle, short, mult)
                s.insert(0, "variant", var); s.insert(0, "model", name); s.insert(0, "category", cat)
                swq.append(s)
                r = ratio_sweep(df, bt, LAG, idle, short)
                r.insert(0, "variant", var); r.insert(0, "model", name); r.insert(0, "category", cat)
                swr.append(r)
                mc = monthly_costs(df, bt, LAG, idle, short, mult)
                for other in ("point_forecast_only", "trailing_12m_mean"):
                    b = bootstrap_diff(mc["quantile_rule"], mc[other])
                    boot.append({"category": cat, "model": name, "variant": var, "quantile_rule_vs": other, **b})
            print(f"done {cat}/{name}")
    pd.concat(met).to_csv(f"{outdir}/metrics_extended.csv", index=False)
    pd.concat(pol).to_csv(f"{outdir}/policy_extended.csv", index=False)
    pd.concat(swq).to_csv(f"{outdir}/sweep_quantile.csv", index=False)
    pd.concat(swr).to_csv(f"{outdir}/sweep_ratio.csv", index=False)
    pd.DataFrame(boot).to_csv(f"{outdir}/bootstrap.csv", index=False)
    pd.set_option("display.width", 220)
    m = pd.concat(met)
    print("\n=== accuracy and calibration (step 1..3) ===")
    print(m.round(3).to_string(index=False))
    print("\n=== buffer policy total cost (equal rates, mult=%s, q*=%.2f) ===" % (mult, critical_quantile(idle, short, mult)))
    print(pd.concat(pol).round(0).to_string(index=False))
    print("\n=== bootstrap: quantile rule minus other rule, per-month cost (negative = quantile rule cheaper) ===")
    print(pd.DataFrame(boot).round(1).to_string(index=False))


if __name__ == "__main__":
    main()