"""
functions.py -- Lab 3 (Structural Breaks)

Why a separate file?
---------------------
Lab 2 introduced functions to avoid repeating code that ran more than
once within a single script. This lab takes that one step further:
functions that are useful on their own -- not just to avoid repetition
within one script, but as reusable tools you might want in a future
lab, or in your own project -- get moved into their own file.

lab_3.py imports what it needs from here with:

    from functions import run_chow_test, run_qlr_test

This keeps lab_3.py focused on the analysis itself (load data, run
tests, interpret output), while the mechanics of *how* each test is
computed live in one place, separate from the narrative of the lab.
If you ever need a Chow test in a different script, you can copy this
file over rather than hunting through old lab scripts for the code.
"""

import numpy as np
from scipy import stats


def run_chow_test(series, break_index, label=""):
    """Run a Chow test for a single, known break point (mean shift only).

    Tests whether the series has the same mean before and after
    `break_index`, against the alternative that the mean differs.

    Parameters
    ----------
    series : array-like
        The time series to test (as a 1-D array or pandas Series).
    break_index : int
        The index (0-based) at which to split the sample. This is the
        *hypothesized* break date -- Chow requires you to specify it.
    label : str, default ""
        A short description of the break date, used in printed output.

    Returns
    -------
    dict
        Keys: "F_stat", "p_value", "break_index".
    """
    y = np.asarray(series)
    n = len(y)

    SSR_full = np.sum((y - y.mean()) ** 2)

    # Split the series into two pieces at break_index:
    #    y1 is every observation BEFORE the hypothesized break
    #    y2 is every observation AFTER the hypothesized break
    y1, y2 = y[:break_index], y[break_index:]
    SSR1 = np.sum((y1 - y1.mean()) ** 2)   
    SSR2 = np.sum((y2 - y2.mean()) ** 2)
    SSR_restricted = SSR1 + SSR2

    k = 1  # one parameter (the mean) estimated per regime
    F_stat = ((SSR_full - SSR_restricted) / k) / (SSR_restricted / (n - 2 * k))
    p_value = 1 - stats.f.cdf(F_stat, k, n - 2 * k)

    print(f"\n{'-'*50}")
    print(f"Chow test{': ' + label if label else ''}")
    print(f"{'-'*50}")
    print(f"{'Break index:':<20}{break_index:>10}")
    print(f"{'F-statistic:':<20}{F_stat:>10.4f}")
    print(f"{'p-value:':<20}{p_value:>10.6f}")
    print(f"{'-'*50}")

    return {"F_stat": F_stat, "p_value": p_value, "break_index": break_index}


def run_qlr_test(series, trim=0.15):
    """Run a Quandt Likelihood Ratio (QLR) test for an unknown break point.

    Computes the Chow F-statistic at every candidate break point within
    the trimmed sample, and returns the largest one -- the QLR
    statistic -- along with the break point that produced it.

    Parameters
    ----------
    series : array-like
        The time series to test.
    trim : float, default 0.15
        Fraction of the sample excluded from both ends of the search.
        Standard practice (Andrews, 1993) trims 15% from each side,
        since the test is unreliable too close to the sample edges.

    Returns
    -------
    dict
        Keys: "F_stat" (the max, i.e. the QLR statistic), "break_index"
        (the candidate that produced it), "search_range" (the (lo, hi)
        indices actually searched).
    """
    y = np.asarray(series)
    n = len(y)

    lo = int(n * trim)           # Beginning of search domain
    hi = int(n * (1 - trim))     # End of search domain

    best_F = -np.inf
    best_t = None
    for t in range(lo, hi):
        y1, y2 = y[:t], y[t:]
        SSR_full = np.sum((y - y.mean()) ** 2)
        SSR1 = np.sum((y1 - y1.mean()) ** 2)
        SSR2 = np.sum((y2 - y2.mean()) ** 2)
        SSR_restricted = SSR1 + SSR2
        k = 1
        F = ((SSR_full - SSR_restricted) / k) / (SSR_restricted / (n - 2 * k))
        if F > best_F:
            best_F = F
            best_t = t

    print(f"\n{'-'*50}")
    print(f"QLR test")
    print(f"{'-'*50}")
    print(f"{'Search range:':<20}indices {lo} to {hi}")
    print(f"{'Max F-statistic:':<20}{best_F:>10.4f}")
    print(f"{'At break index:':<20}{best_t:>10}")
    print(f"{'-'*50}")
    print("Note: QLR's max F-statistic does not follow a standard F")
    print("distribution (the Davies problem) -- critical values must")
    print("come from tables built for this test, not a plain F-table.")

    return {"F_stat": best_F, "break_index": best_t, "search_range": (lo, hi)}