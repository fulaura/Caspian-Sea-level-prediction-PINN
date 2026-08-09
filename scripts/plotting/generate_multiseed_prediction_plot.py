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
results_master = os.path.join(base_dir, "paper_v2", "results", "master_suite", "1_multiseed")
results_retrain1 = os.path.join(base_dir, "paper_v2", "results", "retrain_pinn1", "1_multiseed")
results_retrain2 = os.path.join(base_dir, "paper_v2", "results", "retrain_pinn2", "1_multiseed")
figures_dir = os.path.join(base_dir, "paper_v2", "figures")
os.makedirs(figures_dir, exist_ok=True)

seeds = ["42", "123", "2024", "7", "999"]

# Determine if the user has trained PINN-2v2 (Budyko with River Inflow)
river_pinn2_trained = os.path.exists(os.path.join(results_retrain2, "pinn2_river", "1m", "seed_42", "prediction_data.csv"))
retrained_pinn2_exists = os.path.exists(os.path.join(results_retrain2, "pinn2", "1m", "seed_42", "prediction_data.csv"))

models = {}
models["baseline"] = {"name": "Baseline LSTM", "color": "#d62728", "source": results_master}

# PINN-2 Pure Budyko (use retrained if exists, otherwise fallback to master)
p2_source = results_retrain2 if retrained_pinn2_exists else results_master
models["pinn2"] = {"name": "PINN-2 (Pure Budyko)", "color": "#ff7f0e", "source": p2_source}

# PINN-2v2 Budyko with River Inflow (only plot if trained)
if river_pinn2_trained:
    models["pinn2_river"] = {"name": "PINN-2v2 (Budyko + River Inflow)", "color": "#9467bd", "source": results_retrain2} # Purple

# PINN-1 Mass Balance
models["pinn1"] = {"name": "PINN-1 (Mass Balance)", "color": "#2ca02c", "source": results_retrain1}

print(f"Loading test predictions across 5 seeds... (PINN-2v2 trained: {river_pinn2_trained})")
dates = pd.date_range(start="2018-01-01", periods=96, freq="MS")

n_panels = len(models)
fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=True)
if n_panels == 1:
    axes = [axes]

for i, (mod_key, mod_info) in enumerate(models.items()):
    ax = axes[i]
    actual_plotted = False
    all_preds = []
    
    for s_idx, seed in enumerate(seeds):
        csv_p = os.path.join(mod_info["source"], mod_key, "1m", f"seed_{seed}", "prediction_data.csv")
        if os.path.exists(csv_p):
            df = pd.read_csv(csv_p)
            if not actual_plotted:
                ax.plot(dates, df['Actual_Level'][:96], color='black', linewidth=3.0, label='Actual Altimetry (DAHITI)', zorder=10)
                actual_plotted = True
            
            alpha_val = 0.4 if s_idx > 0 else 0.8
            lbl = f"{mod_info['name']} Trajectories" if s_idx == 0 else "_nolegend_"
            ax.plot(dates, df['Predicted_Level'][:96], color=mod_info['color'], alpha=alpha_val, linewidth=1.5, label=lbl)
            all_preds.append(df['Predicted_Level'][:96].values)
            
    if all_preds:
        all_preds_np = np.array(all_preds)
        mean_pred = all_preds_np.mean(axis=0)
        std_pred = all_preds_np.std(axis=0)
        ax.plot(dates, mean_pred, color=mod_info['color'], linewidth=2.5, linestyle='--', label=f"{mod_info['name']} (5-Seed Mean)")
        ax.fill_between(dates, mean_pred - std_pred, mean_pred + std_pred, color=mod_info['color'], alpha=0.15, label=r"$\pm 1\sigma$ Variance Band")
        
    ax.set_title(f"{mod_info['name']} - Multi-Seed Trajectories vs Actual Level", fontweight='bold', pad=10)
    ax.set_ylabel("Caspian Sea Level (m)", fontweight='bold')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(-28.9, -27.2)

axes[-1].set_xlabel("Test Period (Years 2018–2026)", fontweight='bold')
fig.suptitle("Multi-Seed Caspian Sea Level Forecast Trajectories (2018–2026)", y=0.98, fontsize=20, fontweight='bold')
plt.tight_layout()
out_p = os.path.join(figures_dir, "fig4_5_multiseed_trajectories.png")
plt.savefig(out_p, dpi=300, bbox_inches='tight')
plt.close()

print(f"Multi-seed trajectory prediction plot successfully generated in paper_v2/figures/fig4_5_multiseed_trajectories.png with {n_panels} panels!")
