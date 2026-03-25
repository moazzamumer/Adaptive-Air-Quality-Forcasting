
"""# Phase 1: Data Preparation

### Load Data & Parse Datetime
"""

import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("data/beijing.csv")

# Combine year, month, day, hour into datetime
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

# Set datetime as index
df.set_index('datetime', inplace=True)

# Drop redundant columns
df.drop(columns=['year', 'month', 'day', 'hour', 'zone', 'city',
                 'longitude', 'latitude', 'date', 'aqi'], inplace=True)

# Sort chronologically
df = df.sort_index()

df.tail()

"""### Data Splitting (MUST come before any outlier handling to prevent leakage)"""

# 90/10 split (time-based, no shuffle) — done FIRST, before any statistics are computed
n_total = len(df)
n_test  = int(n_total * 0.10)
n_train = n_total - n_test

df_train = df.iloc[:n_train].copy().reset_index()
df_test  = df.iloc[n_train:].copy().reset_index()

# Sanity check
print("Total rows:", n_total)
print("Train rows:", len(df_train))
print("Test  rows:", len(df_test))
print("Sum check: ", len(df_train) + len(df_test))

"""### Save Clean (Pre-Winsorization) Splits

These files are the primary input for Regime 1 and Regime 2.
Winsorization is NOT applied here — each regime script handles it
at the appropriate granularity:

  - Regime 1 (walk-forward): fits bounds from the expanding training
    window at every iteration, so each test-week forecast is clipped
    with bounds that only use data available at that point.

  - Regime 2 (frozen model): fits bounds once on df_train_initial and
    applies those same frozen bounds to every test-week slice.

This is the methodologically correct approach for a rolling evaluation:
no future data pollutes any preprocessing decision.
"""

clean_train_path = "data/beijing_train_clean.csv"
clean_test_path  = "data/beijing_test_clean.csv"

df_train.to_csv(clean_train_path, index=False)
df_test.to_csv(clean_test_path,  index=False)

print(f"Saved (clean, unwinsorized): {clean_train_path}")
print(f"Saved (clean, unwinsorized): {clean_test_path}")

"""### Winsorization Bounds (Documentation / Reproducibility)

Compute and store the 1%–99% bounds from the training set only.
These are saved for:
  - Paper reproducibility (reviewers can verify the exact bounds used)
  - Regime 2, which freezes these bounds together with the model

NOTE: Regime 1 will RECOMPUTE bounds from its expanding training window
at each walk-forward iteration — it does NOT use this file for clipping.
"""

numeric_cols = df_train.select_dtypes(include="number").columns
lower_q, upper_q = 0.01, 0.99

wins_low  = df_train[numeric_cols].quantile(lower_q)
wins_high = df_train[numeric_cols].quantile(upper_q)

wins_bounds = pd.DataFrame({'low': wins_low, 'high': wins_high})
wins_bounds.to_csv("data/winsorization_bounds.csv")

print(f"\nWinsorization bounds (from {len(df_train)} training rows) saved to "
      f"data/winsorization_bounds.csv")
print(wins_bounds)
