import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
results_retrain1 = os.path.join(base_dir, "paper_v2", "results", "retrain_pinn1", "1_multiseed")
results_retrain2 = os.path.join(base_dir, "paper_v2", "results", "retrain_pinn2", "1_multiseed")
figures_dir = os.path.join(base_dir, "paper_v2", "figures")
os.makedirs(figures_dir, exist_ok=True)

seeds = ["42", "123", "2024", "7", "999"]

# Load PINN-1 1M history
p1_tot, p1_data, p1_phys = [], [], []
for s in seeds:
    csv_p = os.path.join(results_retrain1, "pinn1", "1m", f"seed_{s}", "history.csv")
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p)
        p1_tot.append(df['train_tot_loss'].values)
        p1_data.append(df['train_data_mse'].values)
        p1_phys.append(df['train_phys_loss'].values)

# Load PINN-2 1M history
p2_tot, p2_data, p2_phys = [], [], []
for s in seeds:
    csv_p = os.path.join(results_retrain2, "pinn2", "1m", f"seed_{s}", "history.csv")
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p)
        p2_tot.append(df['train_tot_loss'].values)
        p2_data.append(df['train_data_mse'].values)
        p2_phys.append(df['train_phys_loss'].values)

# Load PINN-2v2 1M history
p2r_tot, p2r_data, p2r_phys = [], [], []
for s in seeds:
    csv_p = os.path.join(results_retrain2, "pinn2_river", "1m", f"seed_{s}", "history.csv")
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p)
        p2r_tot.append(df['train_tot_loss'].values)
        p2r_data.append(df['train_data_mse'].values)
        p2r_phys.append(df['train_phys_loss'].values)

print("Plotting training loss curves...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: PINN-1 (Mass Balance) Convergence
if p1_data and p1_phys:
    p1_data_arr = np.array(p1_data)
    p1_phys_arr = np.array(p1_phys)
    epochs = np.arange(1, p1_data_arr.shape[1] + 1)
    
    # Left subplot (PINN-1)
    ax1 = axes[0]
    # Data MSE
    mean_d = p1_data_arr.mean(axis=0)
    std_d = p1_data_arr.std(axis=0)
    line1 = ax1.plot(epochs, mean_d, color='#1f77b4', linewidth=2.5, label='Data Loss (Normalized MSE)')
    ax1.fill_between(epochs, mean_d - std_d, mean_d + std_d, color='#1f77b4', alpha=0.15)
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Data Loss (MSE)', color='#1f77b4', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_yscale('log')
    
    # Twin axis for Physics
    ax1_twin = ax1.twinx()
    mean_p = p1_phys_arr.mean(axis=0)
    std_p = p1_phys_arr.std(axis=0)
    line2 = ax1_twin.plot(epochs, mean_p, color='#2ca02c', linewidth=2.5, linestyle='--', label='Physics Loss (Water Balance)')
    ax1_twin.fill_between(epochs, mean_p - std_p, mean_p + std_p, color='#2ca02c', alpha=0.15)
    ax1_twin.set_ylabel('Physics Loss', color='#2ca02c', fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='#2ca02c')
    ax1_twin.set_yscale('log')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax1.set_title('PINN-1 (Mass Balance) Training Dynamics', fontweight='bold', pad=12)

# Panel 2: PINN-2 vs PINN-2v2 Physics Loss Convergence
if p2_phys and p2r_phys:
    p2_phys_arr = np.array(p2_phys)
    p2r_phys_arr = np.array(p2r_phys)
    epochs = np.arange(1, p2_phys_arr.shape[1] + 1)
    
    ax2 = axes[1]
    
    # Pure Budyko (PINN-2)
    mean_p2 = p2_phys_arr.mean(axis=0)
    std_p2 = p2_phys_arr.std(axis=0)
    ax2.plot(epochs, mean_p2, color='#ff7f0e', linewidth=2.5, label='PINN-2 (Pure Budyko)')
    ax2.fill_between(epochs, mean_p2 - std_p2, mean_p2 + std_p2, color='#ff7f0e', alpha=0.15)
    
    # Budyko + River Inflow (PINN-2v2)
    mean_p2r = p2r_phys_arr.mean(axis=0)
    std_p2r = p2r_phys_arr.std(axis=0)
    ax2.plot(epochs, mean_p2r, color='#9467bd', linewidth=2.5, linestyle='--', label='PINN-2v2 (Budyko + River Inflow)')
    ax2.fill_between(epochs, mean_p2r - std_p2r, mean_p2r + std_p2r, color='#9467bd', alpha=0.15)
    
    ax2.set_xlabel('Epochs', fontweight='bold')
    ax2.set_ylabel('Physics Loss Magnitude', fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax2.set_title('Budyko Physics Loss Stabilization comparison', fontweight='bold', pad=12)

plt.suptitle("Physics-Informed Neural Networks - Multi-Seed Loss Convergence Profiles", y=0.98, fontsize=18, fontweight='bold')
plt.tight_layout()
out_p = os.path.join(figures_dir, "fig4_6_loss_curves_convergence.png")
plt.savefig(out_p, dpi=300, bbox_inches='tight')
plt.close()

print(f"Loss curves successfully saved to paper_v2/figures/fig4_6_loss_curves_convergence.png!")
