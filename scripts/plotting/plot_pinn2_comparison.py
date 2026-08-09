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
results_retrain2 = os.path.join(base_dir, "paper_v2", "results", "retrain_pinn2", "1_multiseed")
figures_dir = os.path.join(base_dir, "paper_v2", "figures", "review_figures")
os.makedirs(figures_dir, exist_ok=True)

seeds = ["42", "123", "2024", "7", "999"]
dates = pd.date_range(start="2018-01-01", periods=96, freq="MS")

# Load predictions
p2_runs = []
p2r_runs = []
actual = None

for s in seeds:
    # Pure Budyko (without River)
    p2_csv = os.path.join(results_retrain2, "pinn2", "1m", f"seed_{s}", "prediction_data.csv")
    if os.path.exists(p2_csv):
        df = pd.read_csv(p2_csv)
        p2_runs.append(df['Predicted_Level'][:96].values)
        if actual is None:
            actual = df['Actual_Level'][:96].values
            
    # Budyko + Volga Inflow (with River)
    p2r_csv = os.path.join(results_retrain2, "pinn2_river", "1m", f"seed_{s}", "prediction_data.csv")
    if os.path.exists(p2r_csv):
        df = pd.read_csv(p2r_csv)
        p2r_runs.append(df['Predicted_Level'][:96].values)

if not p2_runs or not p2r_runs:
    print("Error: Could not find training outputs for both PINN-2 and PINN-2v2.")
    exit(1)

p2_runs = np.array(p2_runs)
p2r_runs = np.array(p2r_runs)

# Compute means and standard deviations
p2_mean = p2_runs.mean(axis=0)
p2_std = p2_runs.std(axis=0)

p2r_mean = p2r_runs.mean(axis=0)
p2r_std = p2r_runs.std(axis=0)

# Generate Plot
plt.figure(figsize=(14, 8))

# Plot Actual Altimetry
plt.plot(dates, actual, color='black', linewidth=3.0, alpha=0.8, label='Actual Altimetry (DAHITI)', zorder=10)

# Plot Pure Budyko (Without River)
plt.plot(dates, p2_mean, color='#ff7f0e', linewidth=2.5, linestyle='--', alpha=0.85, label='PINN-2 (Pure Budyko - No Volga Inflow)')
plt.fill_between(dates, p2_mean - p2_std, p2_mean + p2_std, color='#ff7f0e', alpha=0.15, label='PINN-2 Variance Band')

# Plot Budyko with River (PINN-2v2)
plt.plot(dates, p2r_mean, color='#9467bd', linewidth=2.5, alpha=0.85, label='PINN-2v2 (Coupled Volga River Inflow)')
plt.fill_between(dates, p2r_mean - p2r_std, p2r_mean + p2r_std, color='#9467bd', alpha=0.15, label='PINN-2v2 Variance Band')

# Individual seed runs as light background lines for scientific depth
for i in range(len(seeds)):
    plt.plot(dates, p2_runs[i], color='#ff7f0e', alpha=0.1, linewidth=1.0)
    plt.plot(dates, p2r_runs[i], color='#9467bd', alpha=0.1, linewidth=1.0)

# Highlight drought drift region
plt.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2026-01-01'), color='gray', alpha=0.08, label='Extreme Drought Window (2022-2026)')

plt.title('Impact of Volga River Coupling on PINN-2 Catchment Predictions (2018–2026)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Test Period (Years)', fontsize=14, fontweight='bold')
plt.ylabel('Baltic Sea Level Elevation (m)', fontsize=14, fontweight='bold')
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.ylim(-28.8, -27.2)
plt.legend(loc='upper right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
out_plot = os.path.join(figures_dir, "fig4_7_pinn2_river_impact.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
plt.close()

print(f"Direct PINN-2 comparison plot successfully saved to {out_plot}!")
