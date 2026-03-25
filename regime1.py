"""# Regime 1: Weekly Walk-forward Refit (Enhanced)

**Features:**
- **3 Models**: NeuralProphet, FBProphet, SARIMAX
- **Targets**: pm2.5
- **Data Split**: 90/10 Train/Test
- **Regressors**: 'no', 'no2', 'co', 'so2'
- **Strategy**: First iteration uses 90% training data, then add observed test weeks
- **Output**: Weekly plots for each model-week combination
"""

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# Create directory for weekly plots
regime1_weekly_plots_dir = "regime1_weekly_plots"
os.makedirs(regime1_weekly_plots_dir, exist_ok=True)

# Load CLEAN (unwinsorized) splits produced by phase1.py.
# Winsorization is handled per walk-forward window below — this is the
# methodologically correct approach for rolling evaluation:
#   - bounds are re-estimated from the expanding training window at every step
#   - the next test week is clipped with those same bounds
#   - no future data ever influences any preprocessing decision
df_train_initial = pd.read_csv("data/beijing_train_clean.csv", parse_dates=['datetime'])
df_test          = pd.read_csv("data/beijing_test_clean.csv",  parse_dates=['datetime'])
df_train_initial = df_train_initial.sort_values('datetime').reset_index(drop=True)
df_test          = df_test.sort_values('datetime').reset_index(drop=True)

# Walk-forward parameters
print('='*60)
print('REGIME 1: WEEKLY WALK-FORWARD REFIT')
print('='*60)
print(f'\nData Split:')
print(f'  Train rows: {len(df_train_initial)} (90%)')
print(f'  Train days: {len(df_train_initial) / 24:.1f}')
print(f'  Test rows: {len(df_test)} (10%)')
print(f'  Test days: {len(df_test) / 24:.1f}')

# Walk-forward parameters
step_size = 168  # 7 days * 24 hours
forecast_horizon = 168
num_train_weeks = len(df_train_initial) // step_size
print(f'  Number of train weeks: {num_train_weeks}')
num_test_weeks = len(df_test) // step_size
print(f'  Number of test weeks: {num_test_weeks}')

# Regressors
regressors = ['no', 'no2', 'co', 'so2']
print(f'  Regressors: {regressors}')

# Storage
results_regime1 = {}

"""## Model 1: NeuralProphet"""

# NeuralProphet - PM2.5
print('\n' + '='*60)
print('NeuralProphet - PM2.5')
print('='*60)
import time
start_time = time.perf_counter()

all_forecasts = []
all_actuals = []

for week_idx in range(num_test_weeks):
    test_start_idx = week_idx * step_size
    if test_start_idx == 0:
        df_train_current = df_train_initial.copy()
    else:
        df_train_current = pd.concat([df_train_initial, df_test.iloc[:test_start_idx]], ignore_index=True)

    train_days = len(df_train_current) / 24
    print(f'\nWeek {week_idx + 1}/{num_test_weeks}')
    print(f'  Train: {len(df_train_current)} rows ({train_days:.1f} days)')

    # ── Per-window leakage-safe winsorization ────────────────────────────────
    # Fit bounds from the CURRENT expanding training window (all data seen so far).
    # Apply the same bounds to this week's test slice — no future data used.
    numeric_cols_w = df_train_current.select_dtypes(include='number').columns
    wins_low_w  = df_train_current[numeric_cols_w].quantile(0.01)
    wins_high_w = df_train_current[numeric_cols_w].quantile(0.99)
    df_train_current[numeric_cols_w] = df_train_current[numeric_cols_w].clip(
        lower=wins_low_w, upper=wins_high_w, axis=1
    )
    test_week = df_test.iloc[test_start_idx:test_start_idx + step_size].copy()
    test_week[numeric_cols_w] = test_week[numeric_cols_w].clip(
        lower=wins_low_w, upper=wins_high_w, axis=1
    )
    # ─────────────────────────────────────────────────────────────────────────

    # Prepare training data
    train_data = df_train_current[['datetime', 'pm2_5'] + regressors].copy()
    train_data.columns = ['ds', 'y'] + regressors
    train_data = train_data.sort_values("ds").set_index("ds")
    train_data = train_data.asfreq("H")  # forces continuous hourly index
    train_data[['y'] + regressors] = train_data[['y'] + regressors].ffill().bfill()
    train_data = train_data.reset_index()

    # Fit scaler ONLY on current training window
    scaler = StandardScaler()
    scaler.fit(train_data[regressors])

    # Transform training data
    train_data[regressors] = scaler.transform(train_data[regressors])

    # Initialize model with drop_missing=True
    model = NeuralProphet(
        n_lags=168,
        n_forecasts=forecast_horizon,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        epochs=30,
        batch_size=128,
        learning_rate=0.001,
        drop_missing=False,
        impute_missing=True
    )
    for reg in regressors:
        model.add_future_regressor(reg)  # PP: actual future values will be provided

    cols = ['ds','y'] + regressors
    print("After reindex NaNs:\n", train_data[['y'] + regressors].isna().sum())

    model.fit(train_data, freq='H')

    # Perfect Prognosis: pass the actual (scaled) next-week regressor values
    # as regressors_df to make_future_dataframe — this is the required NeuralProphet
    # API for add_future_regressor() (cannot be injected by mutating future_df).
    n_future = min(forecast_horizon, len(test_week))
    future_regs_scaled = scaler.transform(test_week[regressors].values[:n_future])
    # Pad with last known value if final week is shorter than forecast_horizon
    if n_future < forecast_horizon:
        pad = np.repeat(future_regs_scaled[-1:], forecast_horizon - n_future, axis=0)
        future_regs_scaled = np.vstack([future_regs_scaled, pad])
    regressors_df = pd.DataFrame(future_regs_scaled, columns=regressors)

    future_df = model.make_future_dataframe(
        train_data, periods=forecast_horizon,
        n_historic_predictions=True,
        regressors_df=regressors_df
    )

    forecast = model.predict(future_df)

    # Extract forecasts - get the last forecast_horizon predictions
    yhat_cols = [f"yhat{i}" for i in range(1, forecast_horizon + 1)]

    # Forecast origin = last observed timestamp in train_data
    origin_ds = train_data["ds"].max()

    origin_row = forecast.loc[forecast["ds"] == origin_ds, yhat_cols]
    if origin_row.empty:
        print("  Warning: origin row not found in forecast.")
        continue

    week_forecast = origin_row.iloc[0].to_numpy()

    # Get actuals
    test_end_idx = min(test_start_idx + step_size, len(df_test))
    week_actual = test_week['pm2_5'].values[:168]

    # Align lengths
    min_len = min(len(week_actual), len(week_forecast))
    week_actual = week_actual[:min_len]
    week_forecast = week_forecast[:min_len]

    # Remove NaN values from both arrays (keep indices aligned)
    valid_mask = ~(np.isnan(week_forecast) | np.isnan(week_actual))
    week_forecast_clean = week_forecast[valid_mask]
    week_actual_clean = week_actual[valid_mask]

    if len(week_forecast_clean) == 0:
        print(f'  Warning: All forecasts are NaN for this week, skipping')
        continue

    # Weekly plot
    plt.figure(figsize=(12, 5))
    plt.plot(week_actual_clean, label='Actual', color='black', linewidth=2)
    plt.plot(week_forecast_clean, label='Forecast', color='blue', linewidth=2, alpha=0.7)
    plt.title(f'NeuralProphet - PM2.5 - Week {week_idx + 1}', fontweight='bold', fontsize=14)
    plt.xlabel('Hour')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{regime1_weekly_plots_dir}/np_pm25_w{week_idx + 1}.png', dpi=150, bbox_inches='tight')
    plt.close()

    all_forecasts.extend(week_forecast_clean)
    all_actuals.extend(week_actual_clean)
    print(f'  Week MAE: {mean_absolute_error(week_actual_clean, week_forecast_clean):.2f}')
    print(f'  Valid predictions: {len(week_forecast_clean)}/{len(week_forecast)}')

all_forecasts = np.array(all_forecasts)
all_actuals = np.array(all_actuals)
results_regime1['NeuralProphet_pm2.5'] = {
    'forecasts': all_forecasts, 'actuals': all_actuals,
    'mae': mean_absolute_error(all_actuals, all_forecasts),
    'rmse': np.sqrt(mean_squared_error(all_actuals, all_forecasts))
}
print(f"\nOverall MAE: {results_regime1['NeuralProphet_pm2.5']['mae']:.2f}")
print(f"Overall RMSE: {results_regime1['NeuralProphet_pm2.5']['rmse']:.2f}")

end_time = time.perf_counter()

elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = elapsed % 60

print(f"Execution Time: {minutes} min {seconds:.2f} sec")



# FBProphet - PM2.5
print('\n' + '='*60)
print('FBProphet - PM2.5')
print('='*60)
import time
start_time = time.perf_counter()

all_forecasts = []
all_actuals = []

for week_idx in range(num_test_weeks):
    test_start_idx = week_idx * step_size
    if test_start_idx == 0:
        df_train_current = df_train_initial.copy()
    else:
        df_train_current = pd.concat([df_train_initial, df_test.iloc[:test_start_idx]], ignore_index=True)

    train_days = len(df_train_current) / 24
    print(f'\nWeek {week_idx + 1}/{num_test_weeks}')
    print(f'  Train: {len(df_train_current)} rows ({train_days:.1f} days)')

    # ── Per-window leakage-safe winsorization ────────────────────────────────
    numeric_cols_w = df_train_current.select_dtypes(include='number').columns
    wins_low_w  = df_train_current[numeric_cols_w].quantile(0.01)
    wins_high_w = df_train_current[numeric_cols_w].quantile(0.99)
    df_train_current[numeric_cols_w] = df_train_current[numeric_cols_w].clip(
        lower=wins_low_w, upper=wins_high_w, axis=1
    )
    test_week = df_test.iloc[test_start_idx:test_start_idx + step_size].copy()
    test_week[numeric_cols_w] = test_week[numeric_cols_w].clip(
        lower=wins_low_w, upper=wins_high_w, axis=1
    )
    # ─────────────────────────────────────────────────────────────────────────

    # ── Per-window leakage-safe scaling ───────────────────────────────────
    # Fit StandardScaler on training regressors only (same window as above).
    # Apply the same scaler to the next test week's known regressor values.
    scaler_w = StandardScaler()
    df_train_current[regressors] = scaler_w.fit_transform(df_train_current[regressors])
    test_week[regressors] = scaler_w.transform(test_week[regressors])
    # ─────────────────────────────────────────────────────────────────────────

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                   seasonality_mode='multiplicative')
    for reg in regressors:
        model.add_regressor(reg)

    train_data = df_train_current[['datetime', 'pm2_5'] + regressors].copy()
    train_data.columns = ['ds', 'y'] + regressors
    model.fit(train_data)

    # Create future dataframe (includes historical + forecast periods)
    future_df = model.make_future_dataframe(periods=forecast_horizon, freq='H')

    test_end_idx = min(test_start_idx + forecast_horizon, len(df_test))
    df_full = pd.concat([df_train_current, test_week], ignore_index=True)

    # Add regressors to future_df (need full length matching future_df)
    for reg in regressors:
        future_df[reg] = df_full[reg].values[:len(future_df)]

    forecast = model.predict(future_df)
    week_forecast = forecast.iloc[-forecast_horizon:]['yhat'].values
    week_actual = test_week['pm2_5'].values[:forecast_horizon]

    plt.figure(figsize=(12, 5))
    plt.plot(week_actual, label='Actual', color='black', linewidth=2)
    plt.plot(week_forecast[:len(week_actual)], label='Forecast', color='green', linewidth=2, alpha=0.7)
    plt.title(f'FBProphet - PM2.5 - Week {week_idx + 1}', fontweight='bold', fontsize=14)
    plt.xlabel('Hour')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{regime1_weekly_plots_dir}/fbp_ds_0/fbp_pm25_w{week_idx + 1}.png', dpi=150, bbox_inches='tight')
    plt.close()

    all_forecasts.extend(week_forecast[:len(week_actual)])
    all_actuals.extend(week_actual)
    print(f'  Week MAE: {mean_absolute_error(week_actual, week_forecast[:len(week_actual)]):.2f}')

all_forecasts = np.array(all_forecasts)
all_actuals = np.array(all_actuals)
results_regime1['FBProphet_ds0_pm2.5'] = {
    'forecasts': all_forecasts, 'actuals': all_actuals,
    'mae': mean_absolute_error(all_actuals, all_forecasts),
    'rmse': np.sqrt(mean_squared_error(all_actuals, all_forecasts))
}
print(f"\nOverall MAE: {results_regime1['FBProphet_ds0_pm2.5']['mae']:.2f}")
print(f"Overall RMSE: {results_regime1['FBProphet_ds0_pm2.5']['rmse']:.2f}")

end_time = time.perf_counter()

elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = elapsed % 60

print(f"Execution Time: {minutes} min {seconds:.2f} sec")

# FBProphet - PM2.5
print('\n' + '='*60)
print('FBProphet - PM2.5')
print('='*60)
import time
start_time = time.perf_counter()

all_forecasts = []
all_actuals = []

for week_idx in range(num_test_weeks):
    test_start_idx = week_idx * step_size
    if test_start_idx == 0:
        df_train_current = df_train_initial.copy()
    else:
        df_train_current = pd.concat([df_train_initial, df_test.iloc[:test_start_idx]], ignore_index=True)

    train_days = len(df_train_current) / 24
    print(f'\nWeek {week_idx + 1}/{num_test_weeks}')
    print(f'  Train: {len(df_train_current)} rows ({train_days:.1f} days)')

    # ── Per-window leakage-safe winsorization ────────────────────────────────
    numeric_cols_w = df_train_current.select_dtypes(include='number').columns
    wins_low_w  = df_train_current[numeric_cols_w].quantile(0.01)
    wins_high_w = df_train_current[numeric_cols_w].quantile(0.99)
    df_train_current[numeric_cols_w] = df_train_current[numeric_cols_w].clip(
        lower=wins_low_w, upper=wins_high_w, axis=1
    )
    test_week = df_test.iloc[test_start_idx:test_start_idx + step_size].copy()
    test_week[numeric_cols_w] = test_week[numeric_cols_w].clip(
        lower=wins_low_w, upper=wins_high_w, axis=1
    )
    # ─────────────────────────────────────────────────────────────────────────

    # ── Per-window leakage-safe scaling ───────────────────────────────────
    scaler_w = StandardScaler()
    df_train_current[regressors] = scaler_w.fit_transform(df_train_current[regressors])
    test_week[regressors] = scaler_w.transform(test_week[regressors])
    # ─────────────────────────────────────────────────────────────────────────

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,
                   seasonality_mode='multiplicative')
    for reg in regressors:
        model.add_regressor(reg)

    train_data = df_train_current[['datetime', 'pm2_5'] + regressors].copy()
    train_data.columns = ['ds', 'y'] + regressors
    model.fit(train_data)

    # Create future dataframe (includes historical + forecast periods)
    future_df = model.make_future_dataframe(periods=forecast_horizon, freq='H')

    # Use pre-winsorized test slice for regressors
    test_end_idx = min(test_start_idx + forecast_horizon, len(df_test))
    df_full = pd.concat([df_train_current, test_week], ignore_index=True)

    # Add regressors to future_df (need full length matching future_df)
    for reg in regressors:
        future_df[reg] = df_full[reg].values[:len(future_df)]

    forecast = model.predict(future_df)
    week_forecast = forecast.iloc[-forecast_horizon:]['yhat'].values
    week_actual = test_week['pm2_5'].values[:forecast_horizon]

    plt.figure(figsize=(12, 5))
    plt.plot(week_actual, label='Actual', color='black', linewidth=2)
    plt.plot(week_forecast[:len(week_actual)], label='Forecast', color='green', linewidth=2, alpha=0.7)
    plt.title(f'FBProphet - PM2.5 - Week {week_idx + 1}', fontweight='bold', fontsize=14)
    plt.xlabel('Hour')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{regime1_weekly_plots_dir}/fbp_ds_1/fbp_pm25_w{week_idx + 1}.png', dpi=150, bbox_inches='tight')
    plt.close()

    all_forecasts.extend(week_forecast[:len(week_actual)])
    all_actuals.extend(week_actual)
    print(f'  Week MAE: {mean_absolute_error(week_actual, week_forecast[:len(week_actual)]):.2f}')

all_forecasts = np.array(all_forecasts)
all_actuals = np.array(all_actuals)
results_regime1['FBProphet_pm2.5'] = {
    'forecasts': all_forecasts, 'actuals': all_actuals,
    'mae': mean_absolute_error(all_actuals, all_forecasts),
    'rmse': np.sqrt(mean_squared_error(all_actuals, all_forecasts))
}
print(f"\nOverall MAE: {results_regime1['FBProphet_pm2.5']['mae']:.2f}")
print(f"Overall RMSE: {results_regime1['FBProphet_pm2.5']['rmse']:.2f}")

end_time = time.perf_counter()

elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = elapsed % 60

print(f"Execution Time: {minutes} min {seconds:.2f} sec")

"""## Model 3: SARIMAX"""

# SARIMAX - PM2.5
print('\n' + '='*60)
print('SARIMAX - PM2.5')
print('='*60)
import time
start_time = time.perf_counter()

all_forecasts = []
all_actuals = []

previous_params = None
for week_idx in range(num_test_weeks):
    test_start_idx = week_idx * step_size
    if test_start_idx == 0:
        df_train_current = df_train_initial.copy()
    else:
        df_train_current = pd.concat([df_train_initial, df_test.iloc[:test_start_idx]], ignore_index=True)

    train_days = len(df_train_current) / 24
    print(f'\nWeek {week_idx + 1}/{num_test_weeks}')
    print(f'  Train: {len(df_train_current)} rows ({train_days:.1f} days)')

    # ── Per-window leakage-safe winsorization ────────────────────────────────
    numeric_cols_w = df_train_current.select_dtypes(include='number').columns
    wins_low_w  = df_train_current[numeric_cols_w].quantile(0.01)
    wins_high_w = df_train_current[numeric_cols_w].quantile(0.99)
    df_train_current[numeric_cols_w] = df_train_current[numeric_cols_w].clip(
        lower=wins_low_w, upper=wins_high_w, axis=1
    )
    test_week = df_test.iloc[test_start_idx:test_start_idx + step_size].copy()
    test_week[numeric_cols_w] = test_week[numeric_cols_w].clip(
        lower=wins_low_w, upper=wins_high_w, axis=1
    )
    # ─────────────────────────────────────────────────────────────────────────

    # ── Per-window leakage-safe scaling ───────────────────────────────────
    scaler_w = StandardScaler()
    df_train_current[regressors] = scaler_w.fit_transform(df_train_current[regressors])
    test_week[regressors] = scaler_w.transform(test_week[regressors])
    # ─────────────────────────────────────────────────────────────────────────

    train_series = df_train_current['pm2_5'].values
    train_exog = df_train_current[regressors].values

    model = SARIMAX(train_series, exog=train_exog, order=(1, 1, 1),
                   seasonal_order=(1, 1, 1, 24),
                   enforce_stationarity=False, enforce_invertibility=False)

    if previous_params is None:
        model_fit = model.fit(
            disp=False,
            method='lbfgs',
            maxiter=50
        )
    else:
        model_fit = model.fit(
            start_params=previous_params,
            disp=False,
            method='lbfgs',
            maxiter=30
        )

    previous_params = model_fit.params

    # Use scaled test slice for SARIMAX exog (scaled with per-window scaler)
    forecast_exog = test_week[regressors].values
    week_forecast = model_fit.forecast(steps=len(forecast_exog), exog=forecast_exog)
    week_actual = test_week['pm2_5'].values

    plt.figure(figsize=(12, 5))
    plt.plot(week_actual, label='Actual', color='black', linewidth=2)
    plt.plot(week_forecast[:len(week_actual)], label='Forecast', color='red', linewidth=2, alpha=0.7)
    plt.title(f'SARIMAX - PM2.5 - Week {week_idx + 1}', fontweight='bold', fontsize=14)
    plt.xlabel('Hour')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{regime1_weekly_plots_dir}/sarimax/sarimax_pm25_w{week_idx + 1}.png', dpi=150, bbox_inches='tight')
    plt.close()

    all_forecasts.extend(week_forecast[:len(week_actual)])
    all_actuals.extend(week_actual)
    print(f'  Week MAE: {mean_absolute_error(week_actual, week_forecast[:len(week_actual)]):.2f}')

all_forecasts = np.array(all_forecasts)
all_actuals = np.array(all_actuals)
results_regime1['SARIMAX_pm2.5'] = {
    'forecasts': all_forecasts, 'actuals': all_actuals,
    'mae': mean_absolute_error(all_actuals, all_forecasts),
    'rmse': np.sqrt(mean_squared_error(all_actuals, all_forecasts))
}
print(f"\nOverall MAE: {results_regime1['SARIMAX_pm2.5']['mae']:.2f}")
print(f"Overall RMSE: {results_regime1['SARIMAX_pm2.5']['rmse']:.2f}")

end_time = time.perf_counter()

elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = elapsed % 60

print(f"Execution Time: {minutes} min {seconds:.2f} sec")

"""## Results Comparison"""

# Comparison tables
print('\n' + '='*60)
print('RESULTS COMPARISON')
print('='*60)

pm25_comp = pd.DataFrame({
    'Model': ['NeuralProphet', 'FBProphet (DS=0)', 'FBProphet (DS=1)', 'SARIMAX'],
    'MAE': [results_regime1['NeuralProphet_pm2.5']['mae'],
           results_regime1['FBProphet_ds0_pm2.5']['mae'],
           results_regime1['FBProphet_pm2.5']['mae'],
           results_regime1['SARIMAX_pm2.5']['mae']],
    'RMSE': [results_regime1['NeuralProphet_pm2.5']['rmse'],
            results_regime1['FBProphet_ds0_pm2.5']['rmse'],
            results_regime1['FBProphet_pm2.5']['rmse'],
            results_regime1['SARIMAX_pm2.5']['rmse']]
}).sort_values('MAE')

print('\n--- PM2.5 Results ---')
print(pm25_comp.to_string(index=False))

pm25_comp.to_csv('./regime1_pm25_comparison.csv', index=False)

# Overall visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('Regime 1: Weekly Walk-forward Refit (Enhanced)', fontsize=16, fontweight='bold')

models_pm25 = ['NeuralProphet_pm2.5', 'FBProphet_pm2.5', 'SARIMAX_pm2.5']
models_pm10 = ['NeuralProphet_pm10', 'FBProphet_pm10', 'SARIMAX_pm10']
colors = ['blue', 'green', 'red']

for idx, (model_key, color) in enumerate(zip(models_pm25, colors)):
    ax = axes[0, idx]
    result = results_regime1[model_key]
    ax.plot(result['actuals'], label='Actual', alpha=0.7, color='black', linewidth=1)
    ax.plot(result['forecasts'], label='Forecast', alpha=0.7, color=color, linewidth=1)
    ax.set_title(f"{model_key.replace('_', ' ').upper()}\nMAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}", fontweight='bold')
    ax.set_ylabel('PM2.5')
    ax.legend()
    ax.grid(True, alpha=0.3)

for idx, (model_key, color) in enumerate(zip(models_pm10, colors)):
    ax = axes[1, idx]
    result = results_regime1[model_key]
    ax.plot(result['actuals'], label='Actual', alpha=0.7, color='black', linewidth=1)
    ax.plot(result['forecasts'], label='Forecast', alpha=0.7, color=color, linewidth=1)
    ax.set_title(f"{model_key.replace('_', ' ').upper()}\nMAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}", fontweight='bold')
    ax.set_xlabel('Hour')
    ax.set_ylabel('PM10')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./regime1_overall.png', dpi=300, bbox_inches='tight')
plt.show()

print('\nSaved: ./regime1_overall.png')
print('Weekly plots saved in: ./regime1_weekly_plots/')

"""# Regime 2: Frozen Model + Online Residual Correction

**Strategy:**
- Train each base model **once** on initial 90% data (frozen model)
- Generate weekly forecasts using the frozen model
- Apply **EWMA (Exponentially Weighted Moving Average)** bias correction based on recent residuals
- Corrected forecast: `ŷ_final = ŷ_base + b_t`

**EWMA Bias Update:**
- `b_t = α × e_t + (1-α) × b_{t-1}`
- Where `e_t = actual - predicted` (residual)
- `α = 0.3` (weight for recent errors)

**Benefits:**
- No expensive retraining
- Adaptive to recent trends via residual correction
- Constant memory footprint

"""

