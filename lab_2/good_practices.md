# Good Practices -- Forecasting Lab

Starting this lab, we hold code to standards beyond "does it work."
These habits cost little now and save real time later -- for you, for
teammates, and for anyone (including future you) who has to reread
this code in three months.

## 1. List your dependencies in `requirements.txt`

Every package you `import` should also appear in `requirements.txt`,
so anyone can recreate your environment with one command:

```bash
pip install -r requirements.txt
```

This lab's `requirements.txt`:

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
statsmodels>=0.14
```

If you add a new import while working, add it here too -- before you
forget.

## 2. Readable output with f-strings

Use f-strings instead of comma-separated `print()` calls. This gives
you control over spacing, labels, and decimal precision, and makes
output genuinely easier to read at a glance.

```python
# Avoid:
print("AIC:", aic_value)

# Prefer:
print(f"AIC: {aic_value:.2f}")
```

A few formatting building blocks worth knowing:

| Syntax | Effect | Example |
|---|---|---|
| `:.2f` | Round to 2 decimal places | `f"{3.14159:.2f}"` → `3.14` |
| `:<12` | Left-align, pad to width 12 | `f"{'AIC:':<12}"` |
| `:>10` | Right-align, pad to width 10 | `f"{842.17:>10.2f}"` |
| `:,` | Thousands separator | `f"{1234567:,}"` → `1,234,567` |

Putting it together, a labeled summary block:

```python
print(f"{'='*40}")
print(f"ARIMA{order} -- Model Summary")
print(f"{'='*40}")
print(f"{'AIC:':<12}{aic:>10.2f}")
print(f"{'BIC:':<12}{bic:>10.2f}")
print(f"{'='*40}")
```

## 3. Document your functions

Every function should have at least a one-line docstring explaining
what it does. If it takes non-obvious parameters or returns something,
document those too (NumPy-style docstrings, as below, are a common
convention in scientific Python):

```python
def calculate_growth_rate(series, periods=1):
    """Calculate growth rate for a time series.

    Parameters
    ----------
    series : pd.Series
        Time series data (e.g., an index or price level).
    periods : int, default 1
        Number of periods to shift for the growth calculation.

    Returns
    -------
    pd.Series
        Growth rate as a percentage change.
    """
    return series.pct_change(periods=periods) * 100
```

A one-line docstring is fine for simple helper functions:

```python
def load_series(filepath):
    """Load the time series CSV and return it indexed by date."""
    ...
```

## 4. End every lab the same way

Before leaving, every lab session should end with the same three
steps. Make this muscle memory:

1. Save and re-render your notebook
2. Commit your changes:
   ```bash
   git add .
   git commit -m "Complete forecasting lab"
   git push
   ```
3. Update `README.md` if you added new files or changed the project
   structure

---

*This file will grow across the semester as we introduce new
practices. Check back if you want a running reference of everything
we've covered so far.*