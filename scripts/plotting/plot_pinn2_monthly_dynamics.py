import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300
})

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
results_master = os.path.join(base_dir, "paper_v2", "results", "master_suite")
results_retrain2 = os.path.join(base_dir, "paper_v2", "results", "retrain_pinn2", "1_multiseed")
figures_dir = os.path.join(base_dir, "paper_v2", "figures", "review_figures")
os.makedirs(figures_dir, exist_ok=True)

seeds = ["42", "123", "2024", "7", "999"]
dates = pd.date_range(start="2018-01-01", periods=96, freq="MS")

# Storage
b_levels = []
b_deltas = []
p2_levels = []
p2_deltas = []

actual_level = None
actual_delta = None

for s in seeds:
    # Baseline
    b_csv = os.path.join(results_master, "1_multiseed", "baseline", "1m", f"seed_{s}", "prediction_data.csv")
    if os.path.exists(b_csv):
        df_b = pd.read_csv(b_csv)
        b_levels.append(df_b['Predicted_Level'][:96].values)
        b_deltas.append(df_b['Predicted_DeltaH'][:96].values)
        if actual_level is None:
            actual_level = df_b['Actual_Level'][:96].values
            actual_delta = df_b['Actual_DeltaH'][:96].values

    # PINN-2 (Budyko Retrained)
    p2_csv = os.path.join(results_retrain2, "pinn2", "1m", f"seed_{s}", "prediction_data.csv")
    if os.path.exists(p2_csv):
        df_p = pd.read_csv(p2_csv)
        p2_levels.append(df_p['Predicted_Level'][:96].values)
        p2_deltas.append(df_p['Predicted_DeltaH'][:96].values)

b_levels = np.array(b_levels)
b_deltas = np.array(b_deltas)
p2_levels = np.array(p2_levels)
p2_deltas = np.array(p2_deltas)

# Means and Std Devs
b_level_mean = b_levels.mean(axis=0)
b_delta_mean = b_deltas.mean(axis=0)

p2_level_mean = p2_levels.mean(axis=0)
p2_level_std = p2_levels.std(axis=0)
p2_delta_mean = p2_deltas.mean(axis=0)
p2_delta_std = p2_deltas.std(axis=0)

# Create 2-Panel Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# 1. Absolute Water Level Plot (Top Panel)
ax1.plot(dates, actual_level, color='black', linewidth=1.8, alpha=0.85, label='Actual Altimetry (DAHITI)', zorder=10)
ax1.plot(dates, b_level_mean, color='#d62728', linewidth=1.4, linestyle='--', alpha=0.8, label='Pure Data Baseline LSTM (Seed Mean)')
ax1.plot(dates, p2_level_mean, color='#2ca02c', linewidth=1.8, alpha=0.85, label='PINN-2 (Budyko Partitioning) (Seed Mean)')
ax1.fill_between(dates, p2_level_mean - p2_level_std, p2_level_mean + p2_level_std, color='#2ca02c', alpha=0.15, label='PINN-2 Variance Band')

# Plot individual seeds as light background lines
for i in range(len(seeds)):
    ax1.plot(dates, p2_levels[i], color='#2ca02c', alpha=0.25, linewidth=0.6)

ax1.set_title('Monthly Absolute Sea Level Elevation (H) Predictions (2018–2026)', fontsize=14, fontweight='bold', pad=10)
ax1.set_ylabel('Absolute Level Elevation (m)', fontweight='bold')
ax1.xaxis.set_major_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.legend(loc='upper right', frameon=True)
ax1.grid(True, linestyle=':', alpha=0.6)

# 2. Water Level Change Dynamics Plot (Bottom Panel)
ax2.plot(dates, actual_delta, color='black', linewidth=1.4, alpha=0.85, label='Actual Delta H', zorder=10)
ax2.plot(dates, b_delta_mean, color='#d62728', linewidth=1.1, linestyle='--', alpha=0.8, label='Pure Data Baseline LSTM (Seed Mean)')
ax2.plot(dates, p2_delta_mean, color='#2ca02c', linewidth=1.4, alpha=0.85, label='PINN-2 (Budyko Partitioning) (Seed Mean)')
ax2.fill_between(dates, p2_delta_mean - p2_delta_std, p2_delta_mean + p2_delta_std, color='#2ca02c', alpha=0.15, label='PINN-2 Variance Band')

# Plot individual seeds as light background lines
for i in range(len(seeds)):
    ax2.plot(dates, p2_deltas[i], color='#2ca02c', alpha=0.25, linewidth=0.6)

ax2.set_title('Monthly Sea Level Change Dynamics (Delta H) Predictions (2018–2026)', fontsize=14, fontweight='bold', pad=10)
ax2.set_xlabel('Time Interval (Years)', fontweight='bold')
ax2.set_ylabel('Sea Level Change Delta H (m)', fontweight='bold')
ax2.xaxis.set_major_locator(mdates.YearLocator(1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.legend(loc='upper right', frameon=True)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
out_plot = os.path.join(figures_dir, "fig4_8_pinn2_monthly_dynamics.png")
plt.savefig(out_plot, dpi=300)
plt.close()

print(f"Monthly Dynamics comparison plot successfully saved to {out_plot}!")
