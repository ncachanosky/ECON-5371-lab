"""
Lab 3: Structural Breaks
=========================

Applies the Chow test, the QLR test, and Bai-Perron multiple-break
detection to a synthetic monthly inflation series with two genuine
regime shifts.

New this lab: the Chow and QLR test functions live in a separate file,
functions.py, in this same folder. See the docstring at the top of
that file for why.

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


#Set the lab folder location.
LAB_FOLDER = r"\Users\ncachanosky\OneDrive\Research\GitHub\ECON-5371-lab\lab_3"

import os

os.chdir(LAB_FOLDER)

# Import packages, including our own functions.py.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

from functions import run_chow_test, run_qlr_test

pd.set_option('display.precision', 2)


# %% 1. Load and Visualize the Data

df = pd.read_csv("inflation_synthetic.csv", parse_dates=["date"])
series = df.set_index("date")["inflation_rate"]
series.index.freq = "MS"

print(f"{'='*50}")
print(f"Inflation Rate -- Loaded Series")
print(f"{'='*50}")
print(f"{'Observations:':<20}{len(series):>10}")
print(f"{'Date range:':<20}{str(series.index.min().date()):>10} to {series.index.max().date()}")
print(f"{'Mean:':<20}{series.mean():>10.2f}")
print(f"{'Std. Dev.:':<20}{series.std():>10.2f}")
print(f"{'='*50}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series.index, series.values, color="tab:blue", linewidth=1.2)
ax.set_title("Synthetic Inflation Rate (Monthly)")
ax.set_xlabel("Date")
ax.set_ylabel("Inflation Rate (%)")
plt.tight_layout()
plt.show()

# Discussion: Does the mean of this series look constant over time, or
# do you see one or more points where its behavior seems to shift?


# %% 2. Chow Test -- A Known Break Date
#
# Suppose we have a specific reason to suspect a break at a particular
# date -- 2016-01, say. The Chow test asks: does the series' mean
# differ before and after this exact point?
#
# H0: no break -- the mean is the same in both subsamples.

break_date = "2016-01-01"
break_index = series.index.get_loc(break_date)

chow_result = run_chow_test(series.values, break_index, label=f"break at {break_date}")

# Discussion: Given the F-statistic and p-value above, do we reject H0?
# What does that tell us -- and what does it NOT tell us, if we hadn't
# already known to look at this specific date?


# %% 3. QLR Test -- An Unknown Break Date
#
# In practice we rarely know the break date in advance. The QLR test
# searches across candidate break points and reports the one that
# produces the largest Chow F-statistic.

qlr_result = run_qlr_test(series.values, trim=0.15)

qlr_break_date = series.index[qlr_result["break_index"]].date()
print(f"QLR's best break point corresponds to: {qlr_break_date}")

# Discussion: Does QLR find the same break date we assumed in Section
# 2 -- without being told where to look? What does it mean that QLR's
# F-statistic at its best point matches the Chow F-statistic from
# Section 2?


# %% 4. Bai-Perron -- Multiple Break Detection
#
# Chow and QLR are both single-break tools: QLR finds the ONE break
# point with the strongest evidence, even if the series has more than
# one regime change. The Bai-Perron approach (implemented here via the
# ruptures package) searches for multiple breaks at once.

algo = rpt.Pelt(model="l2", min_size=12).fit(series.values)
breakpoints = algo.predict(pen=10)

# The last value ruptures returns is always the series length, not a
# real break -- drop it before interpreting the results.
detected_breaks = breakpoints[:-1]

print(f"\n{'-'*50}")
print(f"Bai-Perron (via ruptures)")
print(f"{'-'*50}")
print(f"{'Breaks detected:':<20}{len(detected_breaks):>10}")
for bp in detected_breaks:
    print(f"  index {bp:>4} -> {series.index[bp].date()}")
print(f"{'-'*50}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series.index, series.values, color="tab:blue", linewidth=1.2)
for bp in detected_breaks:
    ax.axvline(series.index[bp], color="firebrick", linestyle="--", linewidth=1)
ax.set_title("Inflation Rate with Bai-Perron Breakpoints")
ax.set_xlabel("Date")
ax.set_ylabel("Inflation Rate (%)")
plt.tight_layout()
plt.show()

# Discussion: How many breaks did Bai-Perron find? Does it agree with
# QLR on at least one of them? Is there a break here that neither Chow
# nor QLR could have told us about -- and why not, given how each test
# is built?


# %% Lab Complete -- Before You Leave

print("\n" + "="*65)
print("LAB COMPLETE -- BEFORE YOU LEAVE:")
print("="*65)
print("1. Save this script")
print("2. Commit your changes:")
print('     git add .')
print('     git commit -m "Lab 3: structural breaks"')
print('     git push')
print("3. Update README.md if you added new files or changed structure")
print("="*65)