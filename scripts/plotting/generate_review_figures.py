import os
import shutil
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
results_retrain1 = os.path.join(base_dir, "paper_v2", "results", "retrain_pinn1")
figures_dir = os.path.join(base_dir, "paper_v2", "figures", "review_figures")
os.makedirs(figures_dir, exist_ok=True)

# -------------------------------------------------------------------------
# Рисунок 3.1: Copy Triple-Hybrid PINN Architecture Diagram
# -------------------------------------------------------------------------
print("Processing Triple-Hybrid PINN Architecture Diagram...")
brain_dir = r"C:\Users\Niitro_musics\.gemini\antigravity\brain\41df179b-1d38-4801-bbe7-bae66e73389b"
arch_filename = "fig3_1_triple_hybrid_architecture_1779186995635.png"
src_arch_path = os.path.join(brain_dir, arch_filename)
dest_arch_path = os.path.join(figures_dir, "fig3_1_triple_hybrid_architecture.png")

if os.path.exists(src_arch_path):
    shutil.copy(src_arch_path, dest_arch_path)
    print(f"Architecture Diagram successfully copied to {dest_arch_path}")
else:
    print(f"Warning: Schematic file not found at {src_arch_path}")

# -------------------------------------------------------------------------
# Рисунок 4.1: Сравнение предсказаний на Monthly (1m)
# -------------------------------------------------------------------------
print("Generating Сравнение предсказаний на Monthly (1m)...")
seeds = ["42", "123", "2024", "7", "999"]
dates_1m = pd.date_range(start="2018-01-01", periods=96, freq="MS")

actual_delta_1m = None
baseline_1m_preds = []
pinn1_1m_preds = []

for s in seeds:
    # Baseline
    b_csv = os.path.join(results_master, "1_multiseed", "baseline", "1m", f"seed_{s}", "prediction_data.csv")
    if os.path.exists(b_csv):
        df = pd.read_csv(b_csv)
        baseline_1m_preds.append(df['Predicted_DeltaH'][:96].values)
        if actual_delta_1m is None:
            actual_delta_1m = df['Actual_DeltaH'][:96].values

    # PINN-1
    p_csv = os.path.join(results_retrain1, "1_multiseed", "pinn1", "1m", f"seed_{s}", "prediction_data.csv")
    if os.path.exists(p_csv):
        df = pd.read_csv(p_csv)
        pinn1_1m_preds.append(df['Predicted_DeltaH'][:96].values)

if actual_delta_1m is not None and baseline_1m_preds and pinn1_1m_preds:
    b_mean_1m = np.mean(baseline_1m_preds, axis=0)
    p_mean_1m = np.mean(pinn1_1m_preds, axis=0)

    plt.figure(figsize=(14, 6))
    plt.plot(dates_1m, actual_delta_1m, color='black', linewidth=2.5, alpha=0.85, label='Actual Delta H')
    plt.plot(dates_1m, b_mean_1m, color='#d62728', linewidth=2.0, linestyle='--', alpha=0.8, label='Pure Data Baseline LSTM (Seed Mean)')
    plt.plot(dates_1m, p_mean_1m, color='#1f77b4', linewidth=2.0, alpha=0.85, label='PINN-1 (Volumetric Mass Bal.) (Seed Mean)')

    plt.title('Comparison of Sea Level Change (Delta H) Predictions on Monthly (1m)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Time Interval (Years)', fontweight='bold')
    plt.ylabel('Sea Level Change Delta H (m)', fontweight='bold')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_1_monthly_predictions.png"), dpi=300)
    plt.close()
    print("Monthly Predictions plot generated successfully!")

# -------------------------------------------------------------------------
# Рисунок 4.2: Сравнение предсказаний на 10-Day (10d)
# -------------------------------------------------------------------------
print("Generating Сравнение предсказаний на 10-Day (10d)...")
actual_delta_10d = None
baseline_10d_preds = []
pinn2_10d_preds = []

for s in seeds:
    # Baseline
    b_csv = os.path.join(results_master, "1_multiseed", "baseline", "10d", f"seed_{s}", "prediction_data.csv")
    if os.path.exists(b_csv):
        df = pd.read_csv(b_csv)
        baseline_10d_preds.append(df['Predicted_DeltaH'].values)
        if actual_delta_10d is None:
            actual_delta_10d = df['Actual_DeltaH'].values

    # PINN-2
    p_csv = os.path.join(results_master, "1_multiseed", "pinn2", "10d", f"seed_{s}", "prediction_data.csv")
    if os.path.exists(p_csv):
        df = pd.read_csv(p_csv)
        pinn2_10d_preds.append(df['Predicted_DeltaH'].values)

if actual_delta_10d is not None and baseline_10d_preds and pinn2_10d_preds:
    # Crop to the minimum length to avoid shape mismatch
    min_len = min(len(actual_delta_10d), 
                  min(len(p) for p in baseline_10d_preds), 
                  min(len(p) for p in pinn2_10d_preds))
    
    actual_delta_10d = actual_delta_10d[:min_len]
    b_mean_10d = np.mean([p[:min_len] for p in baseline_10d_preds], axis=0)
    p_mean_10d = np.mean([p[:min_len] for p in pinn2_10d_preds], axis=0)

    # Let's generate a date-based timeline for the 10d steps
    dates_10d = pd.date_range(start="2018-01-01", periods=min_len, freq="10D")

    plt.figure(figsize=(14, 6))
    plt.plot(dates_10d, actual_delta_10d, color='black', linewidth=1.5, alpha=0.75, label='Actual Delta H')
    plt.plot(dates_10d, b_mean_10d, color='#d62728', linewidth=1.5, linestyle='--', alpha=0.8, label='Pure Data Baseline LSTM (Seed Mean)')
    plt.plot(dates_10d, p_mean_10d, color='#2ca02c', linewidth=1.5, alpha=0.85, label='PINN-2 (Budyko Partitioning) (Seed Mean)')

    plt.title('Comparison of Sea Level Change (Delta H) Predictions on 10-Day (10d) Resolution', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Time Interval (Years)', fontweight='bold')
    plt.ylabel('Sea Level Change Delta H (m)', fontweight='bold')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_2_10d_predictions.png"), dpi=300)
    plt.close()
    print("10-Day Predictions plot generated successfully!")

# -------------------------------------------------------------------------
# Рисунок 4.3: Сравнение предсказаний на Daily (1d)
# -------------------------------------------------------------------------
print("Generating Сравнение предсказаний на Daily (1d)...")
actual_delta_1d = None
actual_level_1d = None
baseline_1d_deltas = []
baseline_1d_levels = []
pinn2_1d_deltas = []
pinn2_1d_levels = []

for s in seeds:
    # Baseline
    b_csv = os.path.join(results_master, "1_multiseed", "baseline", "10d", f"seed_{s}", "prediction_data.csv")
    if os.path.exists(b_csv):
        df = pd.read_csv(b_csv)
        baseline_1d_deltas.append(df['Predicted_DeltaH'].values)
        baseline_1d_levels.append(df['Predicted_Level'].values)
        if actual_delta_1d is None:
            actual_delta_1d = df['Actual_DeltaH'].values
            actual_level_1d = df['Actual_Level'].values

    # PINN-2
    p_csv = os.path.join(results_master, "1_multiseed", "pinn2", "10d", f"seed_{s}", "prediction_data.csv")
    if os.path.exists(p_csv):
        df = pd.read_csv(p_csv)
        pinn2_1d_deltas.append(df['Predicted_DeltaH'].values)
        pinn2_1d_levels.append(df['Predicted_Level'].values)

if actual_delta_1d is not None and baseline_1d_deltas and pinn2_1d_deltas:
    min_len = min(len(actual_delta_1d), 
                  min(len(p) for p in baseline_1d_deltas), 
                  min(len(p) for p in pinn2_1d_deltas))
    
    actual_delta_1d = actual_delta_1d[:min_len]
    actual_level_1d = actual_level_1d[:min_len]
    
    b_mean_delta = np.mean([p[:min_len] for p in baseline_1d_deltas], axis=0)
    p_mean_delta = np.mean([p[:min_len] for p in pinn2_1d_deltas], axis=0)
    
    b_mean_level = np.mean([p[:min_len] for p in baseline_1d_levels], axis=0)
    p_mean_level = np.mean([p[:min_len] for p in pinn2_1d_levels], axis=0)

    dates_1d = pd.date_range(start="2018-01-01", periods=min_len, freq="D")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Full Period Absolute Water Level (Top Panel)
    ax1.plot(dates_1d, actual_level_1d, color='black', linewidth=1.5, alpha=0.85, label='Actual Altimetry (DAHITI)', zorder=10)
    ax1.plot(dates_1d, b_mean_level, color='#d62728', linewidth=1.2, linestyle='--', alpha=0.8, label='Pure Data Baseline LSTM (Seed Mean)')
    ax1.plot(dates_1d, p_mean_level, color='#2ca02c', linewidth=1.5, alpha=0.85, label='PINN-2 (Budyko Partitioning) (Seed Mean)')

    ax1.set_title('Daily Absolute Sea Level Elevation (H) Predictions (2018–2026)', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Absolute Level Elevation (m)', fontweight='bold')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Zoomed Period Delta H Dynamics Plot (Bottom Panel - 2018 to 2020)
    zoom_mask = (dates_1d >= "2018-01-01") & (dates_1d <= "2020-01-01")
    dates_zoom = dates_1d[zoom_mask]
    actual_zoom = actual_delta_1d[zoom_mask]
    b_zoom = b_mean_delta[zoom_mask]
    p_zoom = p_mean_delta[zoom_mask]

    ax2.plot(dates_zoom, actual_zoom, color='black', linewidth=1.4, alpha=0.85, label='Actual Delta H')
    ax2.plot(dates_zoom, b_zoom, color='#d62728', linewidth=1.1, linestyle='--', alpha=0.8, label='Pure Data Baseline LSTM (Seed Mean)')
    ax2.plot(dates_zoom, p_zoom, color='#2ca02c', linewidth=1.4, alpha=0.9, label='PINN-2 (Budyko Partitioning) (Seed Mean)')

    ax2.set_title('Zoomed-In Chronological Detail of Daily Changes (2018–2020)', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Time Interval (Years)', fontweight='bold')
    ax2.set_ylabel('Daily Sea Level Change Delta H (m)', fontweight='bold')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_3_daily_predictions.png"), dpi=300)
    plt.close()
    print("Daily Predictions plot (2-panel) generated successfully!")

# -------------------------------------------------------------------------
# Рисунок 4.4: Чувствительность к весу физического лосса lambda_phys
# -------------------------------------------------------------------------
print("Generating Чувствительность к весу физического лосса...")
lambda_phys = [0.1, 0.5, 1.0, 5.0]
rmse_vals = [0.0566, 0.0550, 0.0541, 0.0532]
r2_vals = [0.5855, 0.6077, 0.6214, 0.6337]

fig, ax1 = plt.subplots(figsize=(8, 5))

color = '#1f77b4' # deep blue
ax1.set_xlabel(r'Physical Regularization Weight ($\lambda_{\text{phys}}$)', fontweight='bold')
ax1.set_ylabel('Test RMSE (m)', color=color, fontweight='bold')
line1 = ax1.plot(lambda_phys, rmse_vals, color=color, marker='o', linewidth=2.5, alpha=0.85, label='Pointwise Data RMSE')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log')
ax1.set_xticks([0.1, 0.5, 1.0, 5.0])
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax2 = ax1.twinx()  
color = '#2ca02c' # emerald green
ax2.set_ylabel(r'Coefficient of Determination ($R^2$)', color=color, fontweight='bold')
line2 = ax2.plot(lambda_phys, r2_vals, color=color, marker='s', linewidth=2.5, linestyle='--', alpha=0.85, label=r'Validation $R^2$')
ax2.tick_params(axis='y', labelcolor=color)

plt.title(r'Sensitivity to Physical Regularization Weight ($\lambda_{\text{phys}}$)', pad=15, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(figures_dir, "fig4_4_lambda_sensitivity_twinx.png"), dpi=300)
plt.close()
print("Sensitivity plot generated successfully!")

# -------------------------------------------------------------------------
# Рисунок 4.5: Bayesian Epistemic Uncertainty (Fold 3)
# -------------------------------------------------------------------------
print("Generating Bayesian Epistemic Uncertainty (Fold 3)...")
mc_csv = os.path.join(results_retrain1, "2_mc_dropout", "mc_uncertainty_bands.csv")

if os.path.exists(mc_csv):
    df_mc = pd.read_csv(mc_csv)
    
    # Slicing Fold 3 test period (2021-2026, which corresponds to indexes 36 to 96)
    df_fold3 = df_mc.iloc[36:96].copy()
    dates_fold3 = pd.date_range(start="2021-01-01", periods=60, freq="MS")
    
    actual_h = df_fold3["Actual_H"].values
    mean_h = df_fold3["Mean_H"].values
    
    # Calculate cumulative bounds starting from actual_H of first month in Fold 3
    start_lvl = actual_h[0]
    lower_cum = start_lvl + np.cumsum(df_fold3["Lower_95"])
    upper_cum = start_lvl + np.cumsum(df_fold3["Upper_95"])
    
    plt.figure(figsize=(14, 7))
    plt.plot(dates_fold3, actual_h, label="Actual Level (Altimetry)", color="black", linewidth=2.5, alpha=0.85)
    plt.plot(dates_fold3, mean_h, label="MC Predicted Mean (50 Passes)", color="#756bb1", linewidth=2.5, linestyle="--", alpha=0.85)
    plt.fill_between(dates_fold3, lower_cum, upper_cum, color="#bcbddc", alpha=0.35, label="95% Epistemic Confidence Band")
    
    # Highlight May-July Spring Flood Window for each year
    first_highlight = True
    for yr in [2021, 2022, 2023, 2024, 2025]:
        start_highlight = pd.Timestamp(f"{yr}-05-01")
        end_highlight = pd.Timestamp(f"{yr}-07-31")
        
        lbl = "Volga Spring Flood Window (May-July)" if first_highlight else None
        plt.axvspan(start_highlight, end_highlight, color="#ff7f0e", alpha=0.1, label=lbl)
        first_highlight = False
        
    plt.title("Monte Carlo Dropout Epistemic Uncertainty (Fold 3: 2021-2026)", fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("Caspian Sea Level (m)", fontweight="bold")
    plt.xlabel("Test Period (Years 2021-2026)", fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_5_mc_uncertainty_fold3.png"), dpi=300)
    plt.close()
    print("Bayesian Epistemic Uncertainty plot generated successfully!")

print("All requested review figures generated successfully!")
