"""
Lab: Time Series Forecasting with ARIMA
=========================================

Course-wide good practices used in this lab:
  1. Dependencies listed in requirements.txt (see also GOOD_PRACTICES.md)
  2. Readable output via f-strings (spacing, labels, decimal precision)
  3. Docstrings on functions that are called more than once
  4. Standard end-of-lab checklist (see bottom of this file)

Run this script section by section (recommended: in Positron, run cell
by cell using the '# %%' markers), or top to bottom as a single script.
"""
# %%
# ============================================================================
# Environment activation
#
# 1. Activate it (needed every time you start a new terminal session):
#    conda activate econ5371
#
# 2. Install required packages/dependencies
#    pip install -r requirements.txt
#
# ============================================================================

# Step 1: Set up your path

# Edit the path below to match where this lab's folder lives on your
# own computer
'''
We use a fixed path here, rather than trying to detect it automatically,
because automatic detection (e.g., via __file__) only works when running
this file as a whole script -- it breaks when running cell-by-cell in an
interactive console or notebook, which is a common way to work through a lab.
'''
LAB_FOLDER = r"\Users\ncachanosky\OneDrive\Research\GitHub\ECON-5371-lab\lab_2"

import os

os.chdir(LAB_FOLDER)

# Step 2: Import the installed packages.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 3: Other general settings

# Reproducibility
np.random.seed(42)

# Display settings
pd.set_option('display.precision', 2)


# %% A note on functions, before we start
#
# Two small functions appear below: run_adf_test() and
# fit_and_summarize(). Both get called more than once in this script
# -- run_adf_test() twice (on the level series and the differenced
# series), fit_and_summarize() three times (once per candidate ARIMA
# order). Wrapping repeated code in a function means:
#   - You write the formatting logic once, not three times
#   - A fix or change only has to happen in one place
#   - The repeated pattern (test, fit, print a summary) is visible as
#     a single named step, rather than three near-identical blocks
#
# Not every piece of code needs to be a function -- a block that runs
# exactly once (like the forecast plot later in this script) is often
# easier to read left inline, where you can see it top to bottom
# without jumping to a definition elsewhere in the file. The rule of
# thumb: if you find yourself about to copy-paste a block and change
# one value, that's usually the signal to make it a function instead.


# %% 1. Load and Visualize the Data

df = pd.read_csv("widget_sales.csv", parse_dates=["date"])
series = df.set_index("date")["widget_sales_index"]
series.index.freq = "MS"

print(f"{'='*50}")
print(f"Widget Sales Index -- Loaded Series")
print(f"{'='*50}")
print(f"{'Observations:':<20}{len(series):>10}")
print(f"{'Date range:':<20}{str(series.index.min().date()):>10} to {series.index.max().date()}")
print(f"{'Mean:':<20}{series.mean():>10.2f}")
print(f"{'Std. Dev.:':<20}{series.std():>10.2f}")
print(f"{'='*50}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series.index, series.values)
ax.set_title("Widget Sales Index (Monthly)")
ax.set_xlabel("Date")
ax.set_ylabel("Index Value")
plt.tight_layout()
plt.show()

# Discussion: Does this series look stationary? What features do you
# see (trend? seasonality?) that we'll need to address before fitting
# an ARIMA model?


# %% 2. Test for Stationarity
# We use the Augmented Dickey-Fuller (ADF) test. The null hypothesis is
# that the series has a unit root (i.e., is *not* stationary).
#
# This runs twice below (levels, then differenced) with identical
# formatting each time -- exactly the kind of repetition worth
# wrapping in a function, per the note above.

def run_adf_test(series, label="Series"):
    """Run the Augmented Dickey-Fuller test and print a formatted summary.

    Parameters
    ----------
    series : pd.Series
        The time series to test.
    label : str, default "Series"
        A short name for the series, used in the printed output.

    Returns
    -------
    float
        The ADF test p-value.
    """
    result = adfuller(series.dropna())
    p_value = result[1]

    print(f"\n{'-'*50}")
    print(f"ADF Test: {label}")
    print(f"{'-'*50}")
    print(f"{'Test statistic:':<20}{result[0]:>10.4f}")
    print(f"{'p-value:':<20}{p_value:>10.4f}")
    print(f"{'Stationary?':<20}{'Yes' if p_value < 0.05 else 'No':>10}")
    print(f"{'-'*50}")

    return p_value


p_value = run_adf_test(series, label="Original Series")

# If not stationary, difference the series and re-test
series_diff = series.diff().dropna()
p_value_diff = run_adf_test(series_diff, label="First-Differenced Series")


# %% 3. Identify Candidate Orders with ACF / PACF

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series_diff, ax=axes[0], lags=24)
plot_pacf(series_diff, ax=axes[1], lags=24)
axes[0].set_title("ACF -- Differenced Series")
axes[1].set_title("PACF -- Differenced Series")
plt.tight_layout()
plt.show()

# Discussion: Based on the ACF and PACF plots, what candidate (p, d, q)
# orders seem reasonable to try? Recall: PACF cutoff suggests AR order;
# ACF cutoff suggests MA order.


# %% 4. Fit Candidate Models
#
# This runs three times below, once per candidate order, with
# identical fitting and printing logic each time -- again, repetition
# worth wrapping in a function.

def fit_and_summarize(series, order):
    """Fit an ARIMA model and print a formatted summary of key metrics.

    Parameters
    ----------
    series : pd.Series
        The time series to model (levels, not pre-differenced --
        ARIMA handles differencing internally via the `d` term).
    order : tuple of int
        The (p, d, q) order of the ARIMA model.

    Returns
    -------
    statsmodels ARIMAResults
        The fitted model results object.
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()

    print(f"\n{'='*40}")
    print(f"ARIMA{order} -- Model Summary")
    print(f"{'='*40}")
    print(f"{'AIC:':<12}{fitted.aic:>10.2f}")
    print(f"{'BIC:':<12}{fitted.bic:>10.2f}")
    print(f"{'Log-Lik:':<12}{fitted.llf:>10.2f}")
    print(f"{'='*40}")

    return fitted


candidate_orders = [(1, 1, 1), (2, 1, 1), (1, 1, 2)]
fitted_models = {}

for order in candidate_orders:
    fitted_models[order] = fit_and_summarize(series, order)


# %% 5. Check Residuals
# Pick your best candidate model (lowest AIC) and check that its
# residuals resemble white noise.

best_order = min(fitted_models, key=lambda o: fitted_models[o].aic)
best_model = fitted_models[best_order]

print(f"Best model by AIC: ARIMA{best_order}")

fig = best_model.plot_diagnostics(figsize=(10, 6))
plt.tight_layout()
plt.show()


# %% 6. Forecast
#
# Unlike the two functions above, this block runs exactly once -- there
# is no repetition to remove by wrapping it in a function. Left inline,
# you can read the forecast, the plot, and the table top to bottom in
# one pass, without jumping to a definition elsewhere in the file.

forecast_horizon = 12
forecast_result = best_model.get_forecast(steps=forecast_horizon)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series.index, series.values, label="Observed")
ax.plot(forecast_mean.index, forecast_mean.values,
        label="Forecast", color="firebrick")
ax.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color="firebrick", alpha=0.15, label="95% CI")
ax.set_title(f"Widget Sales Index -- {forecast_horizon}-Month Forecast")
ax.legend()
plt.tight_layout()
plt.show()

forecast_table = pd.concat([forecast_mean.rename("forecast"), conf_int], axis=1)

print(f"\n{'='*72}")
print(f"12-Month Forecast -- ARIMA{best_order}")
print(f"{'='*72}")
print(forecast_table)
print(f"{'='*72}")


# %% 7. Compare Models

print(f"\n{'='*50}")
print(f"{'Model':<15}{'AIC':>12}{'BIC':>12}")
print(f"{'-'*50}")
for order, fitted in fitted_models.items():
    label = f"ARIMA{order}"
    print(f"{label:<15}{fitted.aic:>12.2f}{fitted.bic:>12.2f}")
print(f"{'='*50}")

# Discussion: Which model do AIC and BIC favor? Do they agree? If not,
# which criterion would you trust more here, and why?


# %% Lab Complete -- Before You Leave

print("\n" + "="*65)
print("LAB COMPLETE -- BEFORE YOU LEAVE:")
print("="*65)
print("1. Save this script")
print("2. Commit your changes:")
print('     git add .')
print('     git commit -m "Complete forecasting lab"')
print('     git push')
print("3. Update README.md if you added new files or changed structure")
print("="*65)