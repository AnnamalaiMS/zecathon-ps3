\# Zecathon PS3 — Mutual Fund Flow Forecasting and Cash Buffer Optimization



A forecasting and decision-support project for India's liquid and overnight mutual fund categories. The project combines monthly net-flow forecasting, uncertainty intervals, historical backtesting, and cash-buffer policy evaluation in an interactive dashboard.



\## Project Objectives



\- Forecast monthly net flows for liquid and overnight fund categories over a three-month horizon.

\- Compare forecasting approaches using rolling-origin backtesting.

\- Generate 80% and 95% prediction intervals.

\- Evaluate cash-buffer policies against idle-cash costs and shortfall penalties.

\- Present forecasts, validation metrics, and policy comparisons in an interactive dashboard.



\## Methodology



The project includes seasonal-naive forecasting, Exponential Smoothing (ETS), and SARIMAX with a tax-period indicator. Extended experiments also evaluate quantile Gradient Boosting and conformal interval adjustments.



Models are evaluated using historical forecast origins and error metrics, including Mean Absolute Scaled Error (MASE). The buffer policy is evaluated using configurable idle-cash and shortfall costs.



\## Project Structure



```text

zecathon-ps3/

├── config.yaml

├── dashboard.py

├── run\_pipeline.py

├── run\_forecast.py

├── run\_extended.py

├── requirements.txt

├── data/

│   ├── raw/

│   │   └── SOURCES.md

│   └── processed/

│       └── flows.csv

├── outputs/

├── src/

└── tests/

```



\## Setup



Python 3.11 or a compatible version supported by the pinned dependencies is recommended.



Create and activate a virtual environment:



```bash

python -m venv .venv

```



Windows:



```cmd

.venv\\Scripts\\activate

```



Install dependencies:



```bash

python -m pip install -r requirements.txt

```



\## Run the Dashboard



```bash

python -m streamlit run dashboard.py

```



Open the local URL displayed in the terminal, usually `http://localhost:8501`.



\## Reproduce Forecasts and Experiments



Run the final forecasting script:



```bash

python run\_forecast.py

```



Run the extended backtesting and policy experiments:



```bash

python run\_extended.py

```



The pipeline and experiment scripts may have different input-data requirements. Check their source and configuration before running them independently.



\## Test



```bash

python -m pytest

```



\## Data Sources



Monthly source data is documented in \[`data/raw/SOURCES.md`](data/raw/SOURCES.md). Consult the source provider's applicable terms before redistributing downloaded files. The repository should retain source attribution and clear data-acquisition instructions.



\## Important Limitations



\- Forecasts are experimental decision-support outputs, not guaranteed predictions.

\- Historical backtest performance does not guarantee future performance.

\- Cash-buffer costs depend on configurable assumptions and should not be interpreted as realized monetary savings without calibration.

\- Banking liquidity, money-market yields, and market-return variables should not be described as model inputs unless they are actually incorporated and validated.

\- Confirm the source dataset's units and publication timing before interpreting forecast magnitudes.



\## Responsible Use



Use the forecasts alongside appropriate liquidity controls, independent validation, and human review. Do not use experimental outputs as the sole basis for financial decisions.
