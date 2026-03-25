# Regime 2 Setup
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
regime2_weekly_plots_dir = "regime2_weekly_plots"
os.makedirs(regime2_weekly_plots_dir, exist_ok=True)

# Load CLEAN (unwinsorized) splits produced by phase1.py.
# Regime 2 fits winsorization bounds ONCE on the initial training set
# (frozen alongside the model) and applies those same bounds to every
# test-week slice — consistent with the frozen-model philosophy.
df_train_initial = pd.read_csv("data/beijing_train_clean.csv", parse_dates=['datetime'])
df_test          = pd.read_csv("data/beijing_test_clean.csv",  parse_dates=['datetime'])
df_train_initial = df_train_initial.sort_values('datetime').reset_index(drop=True)
df_test          = df_test.sort_values('datetime').reset_index(drop=True)

# Walk-forward / evaluation parameters (must match regime1.py)
step_size        = 168          # 7 days × 24 hours
forecast_horizon = 168
num_test_weeks   = len(df_test) // step_size
regressors       = ['no', 'no2', 'co', 'so2']

print(f'Test weeks: {num_test_weeks}  |  Regressors: {regressors}')

print('='*60)
print('REGIME 2: FROZEN MODEL + EWMA RESIDUAL CORRECTION')
print('='*60)

# EWMA parameter
alpha = 0.3  # Weight for recent residuals

# Storage for Regime 2 results
results_regime2 = {}

"""## Model 1: NeuralProphet with EWMA

"""

# NeuralProphet - PM2.5 (Frozen + EWMA)
print('\n' + '='*60)
print('NeuralProphet - PM2.5 (Frozen + EWMA)')
print('='*60)
import time
start_time = time.perf_counter()

# ── Frozen winsorization bounds (computed once from initial train) ─────────
# These bounds are frozen alongside the model: the same clip is applied
# to every test-week slice throughout the evaluation phase.
numeric_cols_r2 = df_train_initial.select_dtypes(include='number').columns
wins_low_r2  = df_train_initial[numeric_cols_r2].quantile(0.01)
wins_high_r2 = df_train_initial[numeric_cols_r2].quantile(0.99)

df_train_np = df_train_initial.copy()
df_train_np[numeric_cols_r2] = df_train_np[numeric_cols_r2].clip(
    lower=wins_low_r2, upper=wins_high_r2, axis=1
)
print(f"Frozen winsorization bounds fitted on {len(df_train_np)} training rows.")
# ─────────────────────────────────────────────────────────────────────────

# Prepare training data
train_data = df_train_np[['datetime', 'pm2_5'] + regressors].copy()
train_data.columns = ['ds', 'y'] + regressors
train_data = train_data.sort_values("ds").set_index("ds")
train_data = train_data.asfreq("H")  # forces continuous hourly index
train_data[['y'] + regressors] = train_data[['y'] + regressors].ffill().bfill()
train_data = train_data.reset_index()

scaler = StandardScaler()
train_data[regressors] = scaler.fit_transform(train_data[regressors])

print(f'\nTraining frozen model on {len(train_data)} rows ({len(train_data)/24:.1f} days)')

model_frozen = NeuralProphet(
    n_lags=168, n_forecasts=168,  # Forecast 1 week ahead
    yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,
    epochs=30, batch_size=128, learning_rate=0.001, drop_missing=False, impute_missing=True
)
for reg in regressors:
    model_frozen.add_future_regressor(reg)  # PP: actual future values will be provided

model_frozen.fit(train_data, freq='H')
print('Frozen model training complete!')

# Weekly forecasting with EWMA correction
# Use ACTUAL observed data as context, but never retrain the model
all_forecasts_base = []
all_forecasts_corrected = []
all_actuals = []
bias = 0.0

for week_idx in range(num_test_weeks):
    test_start_idx = week_idx * step_size
    test_end_idx = min(test_start_idx + step_size, len(df_test))

    print(f'\nWeek {week_idx + 1}/{num_test_weeks}')

    # Apply frozen bounds to this week's test slice
    test_week = df_test.iloc[test_start_idx:test_end_idx].copy()
    test_week[numeric_cols_r2] = test_week[numeric_cols_r2].clip(
        lower=wins_low_r2, upper=wins_high_r2, axis=1
    )

    # Create context data: training + all observed test data up to this week
    if test_start_idx == 0:
        context_data = train_data.copy()
    else:
        observed_test = df_test.iloc[:test_start_idx].copy()
        observed_test[numeric_cols_r2] = observed_test[numeric_cols_r2].clip(
            lower=wins_low_r2, upper=wins_high_r2, axis=1
        )
        observed_test = observed_test[['datetime', 'pm2_5'] + regressors].copy()
        observed_test.columns = ['ds', 'y'] + regressors
        observed_test = observed_test.sort_values("ds").set_index("ds").asfreq("H")
        observed_test[['y'] + regressors] = observed_test[['y'] + regressors].ffill().bfill()
        observed_test = observed_test.reset_index()

        # Scale ONLY observed_test (train_data is already scaled)
        observed_test[regressors] = scaler.transform(observed_test[regressors])

        # Concatenate scaled train + scaled observed_test
        context_data = pd.concat([train_data, observed_test], ignore_index=True)

    # Generate forecast for this week using frozen model (no retraining)
    try:
        H = 168
        yhat_cols = [f"yhat{i}" for i in range(1, H + 1)]

        # Perfect Prognosis: pass the actual (scaled) next-week regressor values
        # as regressors_df — required NeuralProphet API for add_future_regressor().
        n_future = min(H, len(test_week))
        future_regs_scaled = scaler.transform(test_week[regressors].values[:n_future])
        if n_future < H:
            pad = np.repeat(future_regs_scaled[-1:], H - n_future, axis=0)
            future_regs_scaled = np.vstack([future_regs_scaled, pad])
        regressors_df = pd.DataFrame(future_regs_scaled, columns=regressors)

        future_df = model_frozen.make_future_dataframe(
            context_data, periods=H,
            n_historic_predictions=True,
            regressors_df=regressors_df
        )

        forecast = model_frozen.predict(future_df)

        # Forecast origin is last observed timestamp in context_data
        origin_ds = context_data["ds"].max()

        origin_row = forecast.loc[forecast["ds"] == origin_ds, yhat_cols]
        if origin_row.empty:
            print("  Warning: origin row not found in forecast.")
            continue

        week_forecast_base = origin_row.iloc[0].to_numpy()

    except Exception as e:
        print(f'  Error generating forecast: {e}')
        continue

    # Get actuals from the pre-winsorized test slice
    week_actual = test_week['pm2_5'].values

    # Align lengths
    min_len = min(len(week_actual), len(week_forecast_base))
    week_forecast_base = week_forecast_base[:min_len]
    week_actual = week_actual[:min_len]

    # Filter NaN
    valid_mask = ~(np.isnan(week_forecast_base) | np.isnan(week_actual))
    week_forecast_base_clean = week_forecast_base[valid_mask]
    week_actual_clean = week_actual[valid_mask]

    if len(week_forecast_base_clean) == 0:
        print(f'  Warning: All forecasts are NaN, skipping')
        continue

    # Apply EWMA bias correction
    week_forecast_corrected = week_forecast_base_clean + bias

    # Compute residuals and update bias
    residuals = week_actual_clean - week_forecast_base_clean
    mean_residual = np.mean(residuals)
    bias = alpha * mean_residual + (1 - alpha) * bias

    # Weekly plot
    plt.figure(figsize=(14, 5))
    plt.plot(week_actual_clean, label='Actual', color='black', linewidth=2)
    plt.plot(week_forecast_base_clean, label='Base Forecast', color='lightblue', linewidth=2, alpha=0.7, linestyle='--')
    plt.plot(week_forecast_corrected, label='Corrected (EWMA)', color='blue', linewidth=2, alpha=0.9)
    plt.title(f'NeuralProphet - PM2.5 - Week {week_idx + 1} (Bias={bias:.2f})', fontweight='bold', fontsize=14)
    plt.xlabel('Hour')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{regime2_weekly_plots_dir}/np_ds_1/np_pm25_w{week_idx + 1}.png', dpi=150, bbox_inches='tight')
    plt.close()

    all_forecasts_base.extend(week_forecast_base_clean)
    all_forecasts_corrected.extend(week_forecast_corrected)
    all_actuals.extend(week_actual_clean)

    mae_base = mean_absolute_error(week_actual_clean, week_forecast_base_clean)
    mae_corrected = mean_absolute_error(week_actual_clean, week_forecast_corrected)
    print(f'  Base MAE: {mae_base:.2f} | Corrected MAE: {mae_corrected:.2f} | Improvement: {mae_base - mae_corrected:.2f}')

# Overall metrics
all_forecasts_base = np.array(all_forecasts_base)
all_forecasts_corrected = np.array(all_forecasts_corrected)
all_actuals = np.array(all_actuals)

results_regime2['NeuralProphet_pm2.5'] = {
    'forecasts': all_forecasts_corrected,
    'actuals': all_actuals,
    'mae': mean_absolute_error(all_actuals, all_forecasts_corrected),
    'rmse': np.sqrt(mean_squared_error(all_actuals, all_forecasts_corrected)),
    'mae_base': mean_absolute_error(all_actuals, all_forecasts_base),
    'rmse_base': np.sqrt(mean_squared_error(all_actuals, all_forecasts_base))
}

print(f"\nOverall Base MAE: {results_regime2['NeuralProphet_pm2.5']['mae_base']:.2f}")
print(f"Overall Corrected MAE: {results_regime2['NeuralProphet_pm2.5']['mae']:.2f}")
print(f"Improvement in MAE: {results_regime2['NeuralProphet_pm2.5']['mae_base'] - results_regime2['NeuralProphet_pm2.5']['mae']:.2f}")
print(f"Overall Base RMSE: {results_regime2['NeuralProphet_pm2.5']['rmse_base']:.2f}")
print(f"Overall Corrected RMSE: {results_regime2['NeuralProphet_pm2.5']['rmse']:.2f}")

end_time = time.perf_counter()

elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = elapsed % 60

print(f"Execution Time: {minutes} min {seconds:.2f} sec")

"""## Model 2: FBProphet with EWMA

"""

# FBProphet - PM2.5 (Frozen + EWMA)
print('\n' + '='*60)
print('FBProphet - PM2.5 (Frozen + EWMA)')
print('='*60)
import time
start_time = time.perf_counter()

# ── Frozen winsorization bounds (computed once from initial train) ─────────
numeric_cols_r2 = df_train_initial.select_dtypes(include='number').columns
wins_low_r2  = df_train_initial[numeric_cols_r2].quantile(0.01)
wins_high_r2 = df_train_initial[numeric_cols_r2].quantile(0.99)

df_train_fbp = df_train_initial.copy()
df_train_fbp[numeric_cols_r2] = df_train_fbp[numeric_cols_r2].clip(
    lower=wins_low_r2, upper=wins_high_r2, axis=1
)
print(f"Frozen winsorization bounds fitted on {len(df_train_fbp)} training rows.")
# ─────────────────────────────────────────────────────────────────────────

# ── Frozen leakage-safe scaling ───────────────────────────────────────
# Fit scaler once on initial training regressors; frozen alongside the model.
scaler_fbp = StandardScaler()
df_train_fbp[regressors] = scaler_fbp.fit_transform(df_train_fbp[regressors])
# ─────────────────────────────────────────────────────────────────────────

# Train frozen model
train_data = df_train_fbp[['datetime', 'pm2_5'] + regressors].copy()
train_data.columns = ['ds', 'y'] + regressors

print(f'\nTraining frozen model on {len(train_data)} rows ({len(train_data)/24:.1f} days)')

model_frozen = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,
                       )
for reg in regressors:
    model_frozen.add_regressor(reg)

model_frozen.fit(train_data)
print('Frozen model training complete!')

# Weekly forecasting with EWMA
all_forecasts_base = []
all_forecasts_corrected = []
all_actuals = []
bias = 0.0

for week_idx in range(num_test_weeks):
    test_start_idx = week_idx * step_size
    test_end_idx = min(test_start_idx + step_size, len(df_test))

    print(f'\nWeek {week_idx + 1}/{num_test_weeks}')

    # Apply frozen bounds and frozen scaler to this week's test slice
    test_week = df_test.iloc[test_start_idx:test_end_idx].copy()
    test_week[numeric_cols_r2] = test_week[numeric_cols_r2].clip(
        lower=wins_low_r2, upper=wins_high_r2, axis=1
    )
    test_week[regressors] = scaler_fbp.transform(test_week[regressors])

    test_week_df = test_week[['datetime'] + regressors].rename(columns={'datetime': 'ds'})
    future_df = test_week_df

    forecast = model_frozen.predict(future_df)
    week_forecast_base = forecast.iloc[-forecast_horizon:]['yhat'].values

    week_actual = test_week['pm2_5'].values

    min_len = min(len(week_actual), len(week_forecast_base))
    week_actual = week_actual[:min_len]
    week_forecast_base = week_forecast_base[:min_len]

    # Apply EWMA correction
    week_forecast_corrected = week_forecast_base + bias
    residuals = week_actual - week_forecast_base
    mean_residual = np.mean(residuals)
    bias = alpha * mean_residual + (1 - alpha) * bias

    plt.figure(figsize=(14, 5))
    plt.plot(week_actual, label='Actual', color='black', linewidth=2)
    plt.plot(week_forecast_base, label='Base Forecast', color='lightgreen', linewidth=2, alpha=0.7, linestyle='--')
    plt.plot(week_forecast_corrected, label='Corrected (EWMA)', color='green', linewidth=2, alpha=0.9)
    plt.title(f'FBProphet - PM2.5 - Week {week_idx + 1} (Bias={bias:.2f})', fontweight='bold', fontsize=14)
    plt.xlabel('Hour')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{regime2_weekly_plots_dir}/fbp_ds_1/fbp_pm25_w{week_idx + 1}.png', dpi=150, bbox_inches='tight')
    plt.close()

    all_forecasts_base.extend(week_forecast_base)
    all_forecasts_corrected.extend(week_forecast_corrected)
    all_actuals.extend(week_actual)

    mae_base = mean_absolute_error(week_actual, week_forecast_base)
    mae_corrected = mean_absolute_error(week_actual, week_forecast_corrected)
    print(f'  Base MAE: {mae_base:.2f} | Corrected MAE: {mae_corrected:.2f} | Improvement: {mae_base - mae_corrected:.2f}')

all_forecasts_base = np.array(all_forecasts_base)
all_forecasts_corrected = np.array(all_forecasts_corrected)
all_actuals = np.array(all_actuals)

results_regime2['FBProphet_pm2.5'] = {
    'forecasts': all_forecasts_corrected,
    'actuals': all_actuals,
    'mae': mean_absolute_error(all_actuals, all_forecasts_corrected),
    'rmse': np.sqrt(mean_squared_error(all_actuals, all_forecasts_corrected)),
    'mae_base': mean_absolute_error(all_actuals, all_forecasts_base),
    'rmse_base': np.sqrt(mean_squared_error(all_actuals, all_forecasts_base))
}

print(f"\nOverall Base MAE: {results_regime2['FBProphet_pm2.5']['mae_base']:.2f}")
print(f"Overall Corrected MAE: {results_regime2['FBProphet_pm2.5']['mae']:.2f}")
print(f"Improvement in MAE: {results_regime2['FBProphet_pm2.5']['mae_base'] - results_regime2['FBProphet_pm2.5']['mae']:.2f}")
print(f"Overall Base RMSE: {results_regime2['FBProphet_pm2.5']['rmse_base']:.2f}")
print(f"Overall Corrected RMSE: {results_regime2['FBProphet_pm2.5']['rmse']:.2f}")

end_time = time.perf_counter()

elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = elapsed % 60

print(f"Execution Time: {minutes} min {seconds:.2f} sec")

"""## Model 3: SARIMAX with EWMA

"""

# SARIMAX - PM2.5 (Frozen + EWMA)
print('\n' + '='*60)
print('SARIMAX - PM2.5 (Frozen + EWMA)')
print('='*60)
import time
start_time = time.perf_counter()

# ── Frozen winsorization bounds (computed once from initial train) ─────────
numeric_cols_r2 = df_train_initial.select_dtypes(include='number').columns
wins_low_r2  = df_train_initial[numeric_cols_r2].quantile(0.01)
wins_high_r2 = df_train_initial[numeric_cols_r2].quantile(0.99)

df_train_sx = df_train_initial.copy()
df_train_sx[numeric_cols_r2] = df_train_sx[numeric_cols_r2].clip(
    lower=wins_low_r2, upper=wins_high_r2, axis=1
)
print(f"Frozen winsorization bounds fitted on {len(df_train_sx)} training rows.")
# ─────────────────────────────────────────────────────────────────────────

# ── Frozen leakage-safe scaling ───────────────────────────────────────
scaler_sx = StandardScaler()
df_train_sx[regressors] = scaler_sx.fit_transform(df_train_sx[regressors])
# ─────────────────────────────────────────────────────────────────────────

# Train frozen model
print(f'\nTraining frozen model on {len(df_train_sx)} rows ({len(df_train_sx)/24:.1f} days)')

train_series = df_train_sx['pm2_5'].values
train_exog = df_train_sx[regressors].values

model_frozen = SARIMAX(train_series, exog=train_exog, order=(1, 1, 1),
                       seasonal_order=(1, 1, 1, 24),
                       enforce_stationarity=False, enforce_invertibility=False)
model_fit = model_frozen.fit(disp=False)
print('Frozen model training complete!')

# Weekly forecasting with EWMA
all_forecasts_base = []
all_forecasts_corrected = []
all_actuals = []
bias = 0.0

for week_idx in range(num_test_weeks):
    test_start_idx = week_idx * step_size
    test_end_idx = min(test_start_idx + step_size, len(df_test))

    print(f'\nWeek {week_idx + 1}/{num_test_weeks}')

    # Apply frozen bounds and frozen scaler to this week's test slice
    test_week = df_test.iloc[test_start_idx:test_end_idx].copy()
    test_week[numeric_cols_r2] = test_week[numeric_cols_r2].clip(
        lower=wins_low_r2, upper=wins_high_r2, axis=1
    )
    test_week[regressors] = scaler_sx.transform(test_week[regressors])

    # Generate base forecast using scaled test exog
    forecast_exog = test_week[regressors].values
    week_forecast_base = model_fit.forecast(steps=len(forecast_exog), exog=forecast_exog)

    week_actual = test_week['pm2_5'].values

    # Apply EWMA correction
    week_forecast_corrected = week_forecast_base + bias
    residuals = week_actual - week_forecast_base
    mean_residual = np.mean(residuals)
    bias = alpha * mean_residual + (1 - alpha) * bias

    plt.figure(figsize=(14, 5))
    plt.plot(week_actual, label='Actual', color='black', linewidth=2)
    plt.plot(week_forecast_base, label='Base Forecast', color='lightcoral', linewidth=2, alpha=0.7, linestyle='--')
    plt.plot(week_forecast_corrected, label='Corrected (EWMA)', color='red', linewidth=2, alpha=0.9)
    plt.title(f'SARIMAX - PM2.5 - Week {week_idx + 1} (Bias={bias:.2f})', fontweight='bold', fontsize=14)
    plt.xlabel('Hour')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{regime2_weekly_plots_dir}/sarimax/sarimax_pm25_w{week_idx + 1}.png', dpi=150, bbox_inches='tight')
    plt.close()

    all_forecasts_base.extend(week_forecast_base)
    all_forecasts_corrected.extend(week_forecast_corrected)
    all_actuals.extend(week_actual)

    mae_base = mean_absolute_error(week_actual, week_forecast_base)
    mae_corrected = mean_absolute_error(week_actual, week_forecast_corrected)
    print(f'  Base MAE: {mae_base:.2f} | Corrected MAE: {mae_corrected:.2f} | Improvement: {mae_base - mae_corrected:.2f}')

all_forecasts_base = np.array(all_forecasts_base)
all_forecasts_corrected = np.array(all_forecasts_corrected)
all_actuals = np.array(all_actuals)

results_regime2['SARIMAX_pm2.5'] = {
    'forecasts': all_forecasts_corrected,
    'actuals': all_actuals,
    'mae': mean_absolute_error(all_actuals, all_forecasts_corrected),
    'rmse': np.sqrt(mean_squared_error(all_actuals, all_forecasts_corrected)),
    'mae_base': mean_absolute_error(all_actuals, all_forecasts_base),
    'rmse_base': np.sqrt(mean_squared_error(all_actuals, all_forecasts_base))
}

print(f"\nOverall Base MAE: {results_regime2['SARIMAX_pm2.5']['mae_base']:.2f}")
print(f"Overall Corrected MAE: {results_regime2['SARIMAX_pm2.5']['mae']:.2f}")
print(f"Improvement in MAE: {results_regime2['SARIMAX_pm2.5']['mae_base'] - results_regime2['SARIMAX_pm2.5']['mae']:.2f}")
print(f"Overall Base RMSE: {results_regime2['SARIMAX_pm2.5']['rmse_base']:.2f}")
print(f"Overall Corrected RMSE: {results_regime2['SARIMAX_pm2.5']['rmse']:.2f}")

end_time = time.perf_counter()

elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = elapsed % 60

print(f"Execution Time: {minutes} min {seconds:.2f} sec")

"""## Results Comparison

"""

# Regime 2 Results Comparison
print('\n' + '='*60)
print('REGIME 2 RESULTS COMPARISON')
print('='*60)

# PM2.5 Comparison
pm25_comp = pd.DataFrame({
    'Model': ['NeuralProphet', 'FBProphet', 'SARIMAX'],
    'MAE (Base)': [
        results_regime2['NeuralProphet_pm2.5']['mae_base'],
        results_regime2['FBProphet_pm2.5']['mae_base'],
        results_regime2['SARIMAX_pm2.5']['mae_base']
    ],
    'MAE (EWMA)': [
        results_regime2['NeuralProphet_pm2.5']['mae'],
        results_regime2['FBProphet_pm2.5']['mae'],
        results_regime2['SARIMAX_pm2.5']['mae']
    ],
    'RMSE (Base)': [
        results_regime2['NeuralProphet_pm2.5']['rmse_base'],
        results_regime2['FBProphet_pm2.5']['rmse_base'],
        results_regime2['SARIMAX_pm2.5']['rmse_base']
    ],
    'RMSE (EWMA)': [
        results_regime2['NeuralProphet_pm2.5']['rmse'],
        results_regime2['FBProphet_pm2.5']['rmse'],
        results_regime2['SARIMAX_pm2.5']['rmse']
    ]
})
pm25_comp['MAE Improvement'] = pm25_comp['MAE (Base)'] - pm25_comp['MAE (EWMA)']
pm25_comp = pm25_comp.sort_values('MAE (EWMA)')

# # PM10 Comparison
# pm10_comp = pd.DataFrame({
#     'Model': ['NeuralProphet', 'FBProphet', 'SARIMAX'],
#     'MAE (Base)': [
#         results_regime2['NeuralProphet_pm10']['mae_base'],
#         results_regime2['FBProphet_pm10']['mae_base'],
#         results_regime2['SARIMAX_pm10']['mae_base']
#     ],
#     'MAE (EWMA)': [
#         results_regime2['NeuralProphet_pm10']['mae'],
#         results_regime2['FBProphet_pm10']['mae'],
#         results_regime2['SARIMAX_pm10']['mae']
#     ],
#     'RMSE (Base)': [
#         results_regime2['NeuralProphet_pm10']['rmse_base'],
#         results_regime2['FBProphet_pm10']['rmse_base'],
#         results_regime2['SARIMAX_pm10']['rmse_base']
#     ],
#     'RMSE (EWMA)': [
#         results_regime2['NeuralProphet_pm10']['rmse'],
#         results_regime2['FBProphet_pm10']['rmse'],
#         results_regime2['SARIMAX_pm10']['rmse']
#     ]
# })
# pm10_comp['MAE Improvement'] = pm10_comp['MAE (Base)'] - pm10_comp['MAE (EWMA)']
# pm10_comp = pm10_comp.sort_values('MAE (EWMA)')

print('\n--- PM2.5 Results ---')
print(pm25_comp.to_string(index=False))
# print('\n--- PM10 Results ---')
# print(pm10_comp.to_string(index=False))

# Save comparisons
# pm25_comp.to_csv('./regime2_pm25_comparison.csv', index=False)
# pm10_comp.to_csv('./regime2_pm10_comparison.csv', index=False)
# print('\nComparison tables saved to CSV files')

# Regime 2 Overall Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Regime 2: Frozen Model + EWMA Residual Correction — PM2.5', fontsize=16, fontweight='bold')

models_pm25 = ['NeuralProphet_pm2.5', 'FBProphet_pm2.5', 'SARIMAX_pm2.5']
colors = ['blue', 'green', 'red']

for idx, (model_key, color) in enumerate(zip(models_pm25, colors)):
    ax = axes[idx]
    result = results_regime2[model_key]
    ax.plot(result['actuals'], label='Actual', alpha=0.7, color='black', linewidth=1)
    ax.plot(result['forecasts'], label='Forecast (EWMA)', alpha=0.7, color=color, linewidth=1)
    ax.set_title(f"{model_key.replace('_', ' ').upper()}", fontweight='bold')
    ax.set_ylabel('PM2.5')
    ax.legend()
    ax.grid(True, alpha=0.3)

# PM10 plots
# for idx, (model_key, color) in enumerate(zip(models_pm10, colors)):
#     ax = axes[1, idx]
#     result = results_regime2[model_key]
#     ax.plot(result['actuals'], label='Actual', alpha=0.7, color='black', linewidth=1)
#     ax.plot(result['forecasts'], label='Forecast (EWMA)', alpha=0.7, color=color, linewidth=1)
#     ax.set_title(f"{model_key.replace('_', ' ').upper()}\\nMAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}\\nImprovement: {result['mae_base'] - result['mae']:.2f}",
#                 fontweight='bold')
#     ax.set_xlabel('Hour')
#     ax.set_ylabel('PM10')
#     ax.legend()
#     ax.grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig('./regime2_overall.png', dpi=300, bbox_inches='tight')
plt.show()

#print('\nSaved: ./regime2_overall.png')
#print('Weekly plots saved in: ./regime2_weekly_plots/')