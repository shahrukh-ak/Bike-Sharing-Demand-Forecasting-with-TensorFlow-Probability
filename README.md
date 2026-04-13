# Bike Sharing Demand Forecasting with TensorFlow Probability

Forecasts daily bike rental demand using a fully Bayesian structural time series model built with TensorFlow Probability. The model decomposes demand into interpretable components: external regressors (weather, calendar), weekly and monthly seasonality, a local linear trend, and an autoregressive term. Fitting is performed with Hamiltonian Monte Carlo. Includes RMSE evaluation on a 31-day test set and rolling forecast cross-validation.

## Business Context

A bike sharing operator needs accurate daily demand forecasts to decide how many bikes to deploy across stations, when to perform maintenance, and how to price rides dynamically. Structural decomposition makes the model interpretable, so operations teams can understand which factors drive demand spikes and which represent long-run trends.

## Dataset

`Daily Bike Sharing.csv` contains two years of daily records (2011-2012) with columns including `dteday` (date), `cnt` (total rentals), `holiday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, and `windspeed`.

## Methodology

**Preprocessing:**
- `dteday` set as a parsed datetime index with daily frequency
- Target column `cnt` renamed to `y`
- `weathersit` one-hot encoded with `drop_first=True`

**Model Components (TFP Structural Time Series):**

| Component | Role |
|-----------|------|
| `LinearRegression` | Weather and calendar regressors |
| `LocalLinearTrend` | Captures smooth level and growth changes |
| `Autoregressive(1)` | Short-term serial dependency |
| `Seasonal(7)` | Day-of-week demand patterns |
| `Seasonal(12)` | Month-of-year seasonality with variable-length seasons |

**Fitting:** Hamiltonian Monte Carlo (HMC) via `tfp.sts.fit_with_hmc` with variational inference as the warmup initialisation.

**Evaluation:** Forecast mean over the 31-day test set compared to actuals with RMSE. Visualisation of training data, test actuals, and predictions.

**Cross-Validation:** Rolling forecast CV with `initial=540 days`, `horizon=31 days`, `period=90 days`. Each fold refits the model independently from scratch.

## Project Structure

```
22_tfp_bike_demand_forecasting/
├── tfp_bike_demand.py  # Full pipeline
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
tensorflow
tensorflow-probability
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Place `Daily Bike Sharing.csv` in the same directory and run:

```bash
python tfp_bike_demand.py
```

Outputs: `demand_series.png`, `demand_forecast.png`, and printed RMSE and cross-validation results.

## Notes

HMC fitting is computationally expensive. The default parameters (`num_results=10`, `num_warmup_steps=10`, `num_variational_steps=20`) are kept low for practical runtimes. Increasing these values improves posterior quality at the cost of longer run times. The rolling CV runs multiple HMC fits sequentially and may take 20-40 minutes on CPU. TensorFlow is required at version 2.x; TFP version should match the installed TF version.
