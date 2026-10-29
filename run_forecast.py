
"""Generate final three-month forecasts under the competition cutoff."""
import os
import numpy as np
import pandas as pd
import yaml

from run_pipeline import load_flows
from src.features import available_as_of
from src.models import MODELS


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cutoff = pd.Timestamp(cfg["data_cutoff"])
    lag = int(cfg["publication_lag_months"])
    horizon = int(cfg["horizon"])

    if horizon != 3:
        raise ValueError("The competition requires a three-month forecast horizon.")

    flows = load_flows("data/processed/flows.csv", cfg)
    os.makedirs("outputs", exist_ok=True)

    # Forecast the three calendar months after the cutoff month.
    first_target = cutoff.to_period("M") + 1
    target_months = pd.period_range(first_target, periods=horizon, freq="M")
    target_ends = target_months.to_timestamp(how="end").normalize()

    all_forecasts = []

    for category in cfg["categories"]:
        df = (
            flows.loc[flows["category"] == category,
                      ["month", "net_flow_cr"]]
            .sort_values("month")
            .reset_index(drop=True)
        )

        if df.empty:
            raise ValueError(f"No observations found for {category}.")

        # Enforce the information available under the configured lag.
        train = available_as_of(df, cutoff, lag)

        if len(train) < 24:
            raise ValueError(
                f"Insufficient eligible training data for {category}: "
                f"{len(train)} observations."
            )

        last_train = train["month"].max()
        if last_train >= target_ends[0]:
            raise ValueError("Training data overlaps the forecast horizon.")

        # Forecast all intervening months, then retain Nov 2026–Jan 2027.
        offset = (
            target_months[0].ordinal
            - last_train.to_period("M").ordinal
            - 1
        )
        n_steps = offset + horizon

        for model_name, model_fn in MODELS.items():
            prediction = model_fn(train, n_steps).iloc[
                offset:offset + horizon
            ].reset_index(drop=True)

            if len(prediction) != horizon:
                raise ValueError(
                    f"{category}/{model_name}: unexpected forecast length."
                )

            result = pd.DataFrame({
                "category": category,
                "model": model_name,
                "forecast_origin_cutoff": cutoff.date().isoformat(),
                "publication_lag_months": lag,
                "last_training_month": last_train.date().isoformat(),
                "target_month": target_ends.strftime("%Y-%m-%d"),
                "point_forecast_cr": prediction["mean"].to_numpy(),
                "lo80_cr": prediction["lo80"].to_numpy(),
                "hi80_cr": prediction["hi80"].to_numpy(),
                "lo95_cr": prediction["lo95"].to_numpy(),
                "hi95_cr": prediction["hi95"].to_numpy(),
            })

            all_forecasts.append(result)

    forecasts = pd.concat(all_forecasts, ignore_index=True)

    # Sanity checks on interval ordering.
    assert (forecasts["lo95_cr"] <= forecasts["lo80_cr"]).all()
    assert (forecasts["lo80_cr"] <= forecasts["point_forecast_cr"]).all()
    assert (forecasts["point_forecast_cr"] <= forecasts["hi80_cr"]).all()
    assert (forecasts["hi80_cr"] <= forecasts["hi95_cr"]).all()

    forecasts.to_csv(
        "outputs/final_forecasts_all_models.csv", index=False
    )

    # Seasonal naive was the strongest point-forecast baseline in prior tests.
    selected = forecasts.loc[
        forecasts["model"] == "seasonal_naive"
    ].copy()
    selected.to_csv("outputs/final_forecasts.csv", index=False)

    print("\nForecast cutoff:", cutoff.date())
    print("Target months:", ", ".join(target_months.astype(str)))
    print("Publication lag:", lag, "months")
    print("\nSelected forecasts:")
    print(selected[
        ["category", "target_month", "point_forecast_cr",
         "lo80_cr", "hi80_cr", "lo95_cr", "hi95_cr"]
    ].to_string(index=False))
    print("\nSaved:")
    print("outputs/final_forecasts.csv")
    print("outputs/final_forecasts_all_models.csv")


if __name__ == "__main__":
    main()