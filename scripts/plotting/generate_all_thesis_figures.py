import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Set professional academic visual style
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
results_retrain1 = os.path.join(base_dir, "paper_v2", "results", "retrain_pinn1")
results_retrain2 = os.path.join(base_dir, "paper_v2", "results", "retrain_pinn2")
figures_dir = os.path.join(base_dir, "paper_v2", "figures")
os.makedirs(figures_dir, exist_ok=True)

# -------------------------------------------------------------------------
# Figure 4.1: Lambda Sensitivity Curve (from master_suite)
# -------------------------------------------------------------------------
print("Generating Figure 4.1: Lambda Sensitivity Curve...")
sens_csv = os.path.join(results_master, "3_sensitivity", "sensitivity_summary.csv")
if os.path.exists(sens_csv):
    df_sens = pd.read_csv(sens_csv)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = '#1f77b4' # deep blue
    ax1.set_xlabel(r'Physical Loss Weight ($\lambda_{\text{phys}}$)', fontweight='bold')
    ax1.set_ylabel('Test RMSE (m)', color=color, fontweight='bold')
    line1 = ax1.plot(df_sens['lambda_phys'], df_sens['rmse'], color=color, marker='o', linewidth=2.5, label='Test RMSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.set_xticks([0.1, 0.5, 1.0, 5.0])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax2 = ax1.twinx()  
    color = '#2ca02c' # emerald green
    ax2.set_ylabel('Explained Variance ($R^2$)', color=color, fontweight='bold')
    line2 = ax2.plot(df_sens['lambda_phys'], df_sens['r2'], color=color, marker='s', linewidth=2.5, linestyle='--', label=r'Test $R^2$')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(r'Impact of Physical Regularization ($\lambda_{\text{phys}}$) on Out-of-Sample Skill', pad=15, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_1_lambda_sensitivity.png"), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------------
# Figure 4.2: Multi-Seed Robustness & Variance (All Retrained Across 5 Seeds!)
# -------------------------------------------------------------------------
print("Generating Figure 4.2: Multi-Seed Robustness & Variance...")
resolutions = ["1m", "10d"]
models = {
    "baseline": {"name": "Baseline LSTM", "source": results_master},
    "pinn1": {"name": "PINN-1 (Mass)", "source": results_retrain1},
    "pinn2": {"name": "PINN-2 (Pure Budyko)", "source": results_retrain2},
    "pinn2_river": {"name": "PINN-2v2 (Budyko+River)", "source": results_retrain2}
}
multi_data = []

for res in resolutions:
    for mod_key, mod_info in models.items():
        csv_p = os.path.join(mod_info["source"], "1_multiseed", mod_key, res, "multiseed_summary.csv")
        if os.path.exists(csv_p):
            df = pd.read_csv(csv_p)
            for val in df['rmse']:
                multi_data.append({'Resolution': res.upper(), 'Model': mod_info['name'], 'RMSE': val})

if multi_data:
    df_multi = pd.DataFrame(multi_data)
    df_1m = df_multi[df_multi['Resolution'] == '1M']
    
    plt.figure(figsize=(9, 5))
    ax = sns.boxplot(x='Model', y='RMSE', data=df_1m, palette='Set2', width=0.4, boxprops=dict(alpha=.8))
    sns.stripplot(x='Model', y='RMSE', data=df_1m, color='black', size=8, jitter=0.1, alpha=0.7)
    
    plt.title('Multi-Seed Forecasting Stability Across 5 Random Seeds (Monthly 1M)', pad=15, fontweight='bold')
    plt.ylabel('Test RMSE (m)', fontweight='bold')
    plt.xlabel('Model Architecture', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_2_multiseed_variance.png"), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------------
# Figure 4.3: Expanding-Window Walk-Forward Validation (from master_suite)
# -------------------------------------------------------------------------
print("Generating Figure 4.3: Expanding-Window Walk-Forward Validation...")
wf_csv = os.path.join(results_master, "6_walk_forward_thermodynamic", "full", "walk_forward_thermodynamic_summary.csv")
if os.path.exists(wf_csv):
    df_wf = pd.read_csv(wf_csv)
    df_wf['Model'] = df_wf['model'].str.upper()
    df_wf['Fold'] = df_wf['fold'].replace({'Fold1_2010': 'Fold 1 (Train <= 2010)', 'Fold2_2015': 'Fold 2 (Train <= 2015)', 'Fold3_2020': 'Fold 3 (Train <= 2020)'})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Fold', y='r2', hue='Model', data=df_wf, palette='viridis')
    plt.title('Expanding Window Walk-Forward Validation Across Climate Regimes', pad=15, fontweight='bold')
    plt.ylabel('Explained Variance ($R^2$)', fontweight='bold')
    plt.xlabel('Historical Climate Window', fontweight='bold')
    plt.ylim(0.45, 0.65)
    plt.legend(title='Model Architecture', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_3_walk_forward_r2.png"), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------------
# Figure 4.4: Bayesian Epistemic Uncertainty Plot (from retrained pinn1)
# -------------------------------------------------------------------------
print("Generating Figure 4.4: Bayesian Epistemic Uncertainty Plot...")
mc_csv = os.path.join(results_retrain1, "2_mc_dropout", "mc_uncertainty_bands.csv")
if os.path.exists(mc_csv):
    df_mc = pd.read_csv(mc_csv)
    n_steps = len(df_mc)
    dates = pd.date_range(start="2018-01-01", periods=n_steps, freq="MS")
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, df_mc["Actual_H"], label="Actual Level (Altimetry)", color="#31a354", linewidth=2.5)
    plt.plot(dates, df_mc["Mean_H"], label="MC Predicted Mean", color="#756bb1", linewidth=2.5, linestyle="--")
    
    start_lvl = df_mc["Actual_H"].iloc[0]
    lower_cum = start_lvl + np.cumsum(df_mc["Lower_95"])
    upper_cum = start_lvl + np.cumsum(df_mc["Upper_95"])
    
    plt.fill_between(dates, lower_cum, upper_cum, color="#bcbddc", alpha=0.4, label="95% Epistemic Confidence Band")
    plt.title("Monte Carlo Dropout Epistemic Uncertainty Intervals (PINN-1 Monthly)", fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("Caspian Sea Level (m)", fontweight="bold")
    plt.xlabel("Test Period (Years 2018-2026)", fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_4_mc_uncertainty.png"), dpi=300, bbox_inches='tight')
    plt.close()

print("All 4 publication-grade figures successfully generated in paper_v2/figures/!")
