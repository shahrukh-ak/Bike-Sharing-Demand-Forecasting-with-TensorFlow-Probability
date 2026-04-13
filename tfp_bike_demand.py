"""
Bike Sharing Demand Forecasting with TensorFlow Probability
============================================================
Forecasts daily bike rental demand using a structural time series
model built with TensorFlow Probability (TFP). The model combines
linear regression (weather/calendar regressors), a local linear
trend, an autoregressive component, and weekly and monthly seasonal
effects. Fitting is performed with Hamiltonian Monte Carlo. Includes
train/test evaluation (RMSE) and rolling forecast cross-validation.

Dataset: Daily Bike Sharing.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_probability as tfp
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error


# ── Data Loading and Preprocessing ───────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the bike sharing dataset with 'dteday' as a parsed datetime index.
    Select the target and feature columns, rename cnt to y,
    set daily frequency, and one-hot encode the weathersit variable.
    """
    df_raw = pd.read_csv(filepath, index_col="dteday", parse_dates=True)

    df = df_raw.loc[:, [
        "cnt", "holiday", "workingday", "weathersit",
        "temp", "atemp", "hum", "windspeed"
    ]].copy()

    df = df.rename(columns={"cnt": "y"})
    df = df.asfreq("D")

    weathersit_dummies = pd.get_dummies(df["weathersit"], drop_first=True)
    df = pd.concat([df.drop(columns="weathersit"), weathersit_dummies], axis=1)

    print(f"Shape: {df.shape}")
    print(df.head())
    return df


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_demand(df: pd.DataFrame):
    """Line plot of daily bike demand with a styled Seaborn theme."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y="y",
                 color="steelblue", label="Bike Sharing Count")
    plt.title("Daily Bike Sharing Demand", fontsize=16)
    plt.xlabel("Date", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.legend(title="Legend", fontsize=11)
    plt.tight_layout()
    plt.savefig("demand_series.png", dpi=150)
    plt.show()
    print("Saved: demand_series.png")


# ── Train / Test Split ────────────────────────────────────────────────────────

def train_test_split_ts(df: pd.DataFrame, test_days: int = 31) -> tuple:
    """Hold out the last N days as the test set."""
    training = df.iloc[:-test_days]
    test     = df.iloc[-test_days:]
    print(f"Training: {len(training)} days  |  Test: {len(test)} days")
    return training, test


# ── TFP Model Components ──────────────────────────────────────────────────────

def build_model_components(df: pd.DataFrame, y: pd.Series) -> dict:
    """
    Construct the five structural time series components:
    linear regressors, weekly seasonality, monthly seasonality,
    autoregressive (order=1), and local linear trend.
    """
    exog = np.asmatrix(df.iloc[:, 1:].astype(np.float64))
    regressors = tfp.sts.LinearRegression(design_matrix=exog, name="regressors")

    trend = tfp.sts.LocalLinearTrend(observed_time_series=y, name="trend")

    autoregressive = tfp.sts.Autoregressive(
        order=1, observed_time_series=y, name="autoregressive"
    )

    weekday_effect = tfp.sts.Seasonal(
        num_seasons=7, num_steps_per_season=1,
        observed_time_series=y, name="weekday_effect"
    )

    num_days_per_month = np.array([
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],  # 2011
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],  # 2012
    ])
    monthly_effect = tfp.sts.Seasonal(
        num_seasons=12, num_steps_per_season=num_days_per_month,
        observed_time_series=y, name="monthly_effect"
    )

    return {
        "regressors":    regressors,
        "weekday_effect": weekday_effect,
        "monthly_effect": monthly_effect,
        "autoregressive": autoregressive,
        "trend":          trend,
    }


def assemble_model(components: dict, y: pd.Series) -> tfp.sts.Sum:
    """Combine all components into a TFP Sum structural model."""
    model = tfp.sts.Sum(
        list(components.values()),
        observed_time_series=y,
    )
    return model


# ── Model Fitting ─────────────────────────────────────────────────────────────

def fit_with_hmc(model, y: pd.Series,
                 num_results: int = 10,
                 num_warmup_steps: int = 10,
                 num_variational_steps: int = 20):
    """
    Fit the structural time series model using Hamiltonian Monte Carlo.
    Returns posterior samples (mcmc) and kernel diagnostic results.
    """
    print("Fitting model with HMC (this may take several minutes)...")
    mcmc, kernel_results = tfp.sts.fit_with_hmc(
        model=model,
        observed_time_series=y,
        num_results=num_results,
        num_warmup_steps=num_warmup_steps,
        num_variational_steps=num_variational_steps,
    )
    print("HMC fitting complete.")
    return mcmc, kernel_results


# ── Forecasting and Evaluation ────────────────────────────────────────────────

def generate_forecast(model, y: pd.Series, mcmc,
                      test: pd.DataFrame) -> pd.Series:
    """Generate predictions for the test period and return as a Series."""
    forecast_dist = tfp.sts.forecast(
        model=model,
        observed_time_series=y,
        parameter_samples=mcmc,
        num_steps_forecast=len(test),
    )
    predictions = pd.Series(
        forecast_dist.mean()[:, 0].numpy(),
        index=test.index,
        name="TFP",
    )
    return predictions


def evaluate(test: pd.DataFrame, predictions: pd.Series):
    """Compute and print RMSE on the test set."""
    rmse = mean_squared_error(test["y"], predictions, squared=False)
    print(f"\nTest RMSE: {rmse:.4f}")
    return rmse


def plot_forecast(training: pd.DataFrame, test: pd.DataFrame,
                  predictions: pd.Series, start_plot: str = "2012-07-01"):
    """Visualise training data, actual test values, and TFP predictions."""
    training["y"][start_plot:].plot(figsize=(10, 6), legend=True, label="Training")
    test["y"].plot(legend=True, label="Actual")
    predictions.plot(legend=True, linestyle="--", label="TFP Forecast")
    plt.title("Bike Demand Forecast – TFP Structural Model")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("demand_forecast.png", dpi=150)
    plt.show()
    print("Saved: demand_forecast.png")


# ── Rolling Forecast Cross-Validation ────────────────────────────────────────

def create_fit_sts_model(train_data: pd.DataFrame, full_df: pd.DataFrame):
    """Helper that builds and fits a TFP STS model for cross-validation folds."""
    y_train = train_data["y"].astype(np.float64)
    exog    = np.asmatrix(full_df.iloc[:, 1:].astype(np.float64))

    trend          = tfp.sts.LocalLinearTrend(observed_time_series=y_train, name="trend")
    autoregressive = tfp.sts.Autoregressive(order=1, observed_time_series=y_train,
                                             name="autoregressive")
    weekday_effect = tfp.sts.Seasonal(num_seasons=7, num_steps_per_season=1,
                                       observed_time_series=y_train, name="weekday_effect")
    regressors     = tfp.sts.LinearRegression(design_matrix=exog, name="regressors")
    model          = tfp.sts.Sum(
        [regressors, weekday_effect, autoregressive, trend],
        observed_time_series=y_train,
    )

    mcmc, _ = tfp.sts.fit_with_hmc(
        model=model, observed_time_series=y_train,
        num_results=10, num_warmup_steps=10, num_variational_steps=20,
    )
    return model, mcmc


def rolling_forecast_cv(dataset: pd.DataFrame, initial: str,
                         horizon: str, period: str) -> pd.DataFrame:
    """
    Rolling forecast cross-validation for TFP STS models.
    Returns a DataFrame of RMSE per fold.
    """
    initial_days = pd.to_timedelta(initial).days
    horizon_days = pd.to_timedelta(horizon).days
    period_days  = pd.to_timedelta(period).days

    n          = len(dataset)
    rmse_list  = []
    dates      = []
    start_idx  = initial_days

    while start_idx + horizon_days <= n:
        train_data    = dataset.iloc[:start_idx]
        val_data      = dataset.iloc[start_idx: start_idx + horizon_days]
        cutoff_date   = dataset.index[start_idx]

        print(f"Fold cutoff: {cutoff_date.date()}")
        try:
            model, mcmc = create_fit_sts_model(train_data, dataset)
            y_train     = train_data["y"].astype(np.float64)
            forecast_dist = tfp.sts.forecast(model=model, observed_time_series=y_train,
                                              parameter_samples=mcmc,
                                              num_steps_forecast=horizon_days)
            preds = forecast_dist.mean()[:, 0].numpy()
            rmse  = float(mean_squared_error(val_data["y"].values, preds, squared=False))
        except Exception as exc:
            print(f"  Fold failed: {exc}")
            rmse = float("nan")

        rmse_list.append(rmse)
        dates.append(cutoff_date)
        start_idx += period_days

    results = pd.DataFrame({"cutoff": dates, "rmse": rmse_list})
    print("\nCross-Validation Results:")
    print(results.to_string())
    return results


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "Daily Bike Sharing.csv"

    df = load_data(DATA_PATH)
    plot_demand(df)

    training, test = train_test_split_ts(df, test_days=31)

    y_train    = training["y"].astype(np.float64)
    components = build_model_components(df, y_train)
    model      = assemble_model(components, y_train)

    mcmc, _ = fit_with_hmc(model, y_train)

    predictions = generate_forecast(model, y_train, mcmc, test)
    evaluate(test, predictions)
    plot_forecast(training, test, predictions)

    print("\nRunning rolling forecast cross-validation...")
    cv_results = rolling_forecast_cv(
        dataset=df, initial="540 days", horizon="31 days", period="90 days"
    )
