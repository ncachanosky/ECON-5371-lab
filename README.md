# Lab 1: Time Series Forecasting with ARIMA

In this lab, you'll build and evaluate ARIMA models to forecast a real
economic time series, using the tools and workflow we've used in prior
labs (Quarto, Python, Git).

## Learning Objectives

By the end of this lab, you will be able to:
- Test a time series for stationarity and apply differencing when needed
- Interpret ACF and PACF plots to select candidate model orders
- Fit an ARIMA model in Python and evaluate its residuals
- Generate and visualize out-of-sample forecasts
- Compare model fit using AIC/BIC

## Prerequisites

Before this lab, you should have:
- Completed Lab 0 (environment setup) and Lab [X] (intro to Python for
  time series / pandas review)
- A working Python environment with:
  - `statsmodels`
  - `pandas`
  - `matplotlib`

## Folder Contents

| File | Description |
|---|---|
| `lab_1/gdp_synthetic.csv` | The time series data we'll use |
| `lab_1.py` | Python code |
