"""# Phase 2: Feature Selection

### Mutual Information (MI)
"""

from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import pandas as pd

# Load clean (unwinsorized) training split produced by phase1.py
import numpy as np
df = pd.read_csv("data/beijing_train_clean.csv", parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Apply train-only winsorization BEFORE feature selection so that MI/mRMR
# scores reflect the same data distribution the models actually receive.
# Bounds are fitted from this same training set — no test data involved.
numeric_cols = df.select_dtypes(include='number').columns
wins_low  = df[numeric_cols].quantile(0.01)
wins_high = df[numeric_cols].quantile(0.99)
df[numeric_cols] = df[numeric_cols].clip(lower=wins_low, upper=wins_high, axis=1)

df.set_index('datetime', inplace=True)

features = ['co', 'no', 'no2', 'o3', 'so2', 'nh3', 'temperature', 'dewpt']

X = df[features]
y_pm25 = df['pm2_5']
y_pm10 = df['pm10']

# Mutual Information for PM2.5
mi_pm25 = mutual_info_regression(X, y_pm25)
mi_scores_pm25 = pd.Series(mi_pm25, index=features).sort_values(ascending=False)

# Mutual Information for PM10
mi_pm10 = mutual_info_regression(X, y_pm10)
mi_scores_pm10 = pd.Series(mi_pm10, index=features).sort_values(ascending=False)

# -----------------------------------------------------------
# MAKE BOTH PLOTS IN ONE FIGURE (UP/DOWN COLLAGE)
# -----------------------------------------------------------

plt.figure(figsize=(12, 10))  # big figure

# --- PM2.5 subplot ---
plt.subplot(2, 1, 1)
mi_scores_pm25.plot(kind='bar')
plt.title("Mutual Information Scores for PM2.5", fontsize=25)
plt.ylabel("MI Score", fontsize=18)
plt.xticks(fontsize=15, rotation=0)
plt.yticks(fontsize=15)

# --- PM10 subplot ---
plt.subplot(2, 1, 2)
mi_scores_pm10.plot(kind='bar')
plt.title("Mutual Information Scores for PM10", fontsize=25)
plt.ylabel("MI Score", fontsize=18)
plt.xticks(fontsize=15, rotation=0)
plt.yticks(fontsize=15)

plt.tight_layout()
plt.show()

"""### mRMR (Minimum Redundancy Maximum Relevance)"""

import pymrmr

# pymrmr requires all numerical data and no NaNs
df_mrmr = df[features + ['pm2_5', 'pm10']].dropna()
df_mrmr = df_mrmr.reset_index(drop=True)

# mRMR for PM2.5 (returns top 5 features)
mrmr_pm25 = pymrmr.mRMR(df_mrmr.drop(columns=['pm10']).rename(columns={'pm2_5': 'target'}), 'MIQ', 5)

# mRMR for PM10
mrmr_pm10 = pymrmr.mRMR(df_mrmr.drop(columns=['pm2_5']).rename(columns={'pm10': 'target'}), 'MIQ', 5)

print("Top mRMR Features for PM2.5:", mrmr_pm25)
print("Top mRMR Features for PM10:", mrmr_pm10)

