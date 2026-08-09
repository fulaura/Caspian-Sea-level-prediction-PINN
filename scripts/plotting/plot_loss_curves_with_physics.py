import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BASE_DIR = os.path.join(base_dir_abs, "paper_v2", "results")
RETRAIN_DIR_P1 = os.path.join(BASE_DIR, "retrain_pinn1", "interval_comparison_retrain", "1m")
RETRAIN_DIR_P2 = os.path.join(BASE_DIR, "retrain_pinn2", "interval_comparison_retrain", "1m")
BASE_DIR_BL = os.path.join(BASE_DIR, "master_suite", "interval_comparison", "1m")
OUT_PLOT = os.path.join(BASE_DIR, "loss_curves_with_physics.png")

def plot_physics_curves():
    print("=== Generating Master Loss Convergence Curves with Physics vs Data ===")
    
    b_path = os.path.join(BASE_DIR_BL, "history.csv")
    p1_path = os.path.join(RETRAIN_DIR_P1, "history.csv")
    p2_path = os.path.join(RETRAIN_DIR_P2, "history.csv")
    
    b_df = pd.read_csv(b_path) if os.path.exists(b_path) else None
    p1_df = pd.read_csv(p1_path) if os.path.exists(p1_path) else None
    p2_df = pd.read_csv(p2_path) if os.path.exists(p2_path) else None
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("Disentangled Training & Validation Loss Convergence across Models (1 Month Resolution)", fontsize=18, fontweight='bold', y=1.04)
    
    # 1. Baseline (Data-Only LSTM)
    if b_df is not None:
        ax = axes[0]
        epochs = range(1, len(b_df) + 1)
        ax.plot(epochs, b_df['train'], label='Train MSE (Normalized)', color='#2b8cbe', linewidth=2.5)
        ax.plot(epochs, b_df['val'], label='Validation MSE (Normalized)', color='#e34a33', linewidth=2.5)
        ax.set_title("Ablation Baseline (Data-Only LSTM)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Epochs", fontsize=12, fontweight='bold')
        ax.set_ylabel("Normalized Mean Squared Error", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=11)
        
    # 2. PINN-1 (Water Balance Conservation)
    if p1_df is not None:
        ax_left = axes[1]
        ax_right = ax_left.twinx()
        epochs = range(1, len(p1_df) + 1)
        
        # Left axis: Data Loss & Validation Loss (Normalized MSE)
        l1 = ax_left.plot(epochs, p1_df['train_data'], label='Train Data MSE (Normalized)', color='#31a354', linewidth=2.5)
        l2 = ax_left.plot(epochs, p1_df['val_mse'], label='Validation Data MSE (Normalized)', color='#756bb1', linewidth=2.5, linestyle='--')
        
        # Right axis: Physics Loss (Residual of Water Balance Equation)
        l3 = ax_right.plot(epochs, p1_df['phys'], label='Physics Loss (Water Balance Residual)', color='#e6550d', linewidth=2.5, linestyle=':')
        
        ax_left.set_title("PINN-1 (Water Balance Conservation)", fontsize=14, fontweight='bold')
        ax_left.set_xlabel("Epochs", fontsize=12, fontweight='bold')
        ax_left.set_ylabel("Normalized Mean Squared Error", fontsize=12, fontweight='bold', color='#31a354')
        ax_right.set_ylabel("Physics Loss Residual", fontsize=12, fontweight='bold', color='#e6550d')
        
        ax_left.tick_params(axis='y', labelcolor='#31a354')
        ax_right.tick_params(axis='y', labelcolor='#e6550d')
        ax_left.grid(True, linestyle=':', alpha=0.6)
        
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc='upper right', fontsize=11)
        
    # 3. PINN-2 (Budyko Hydro-Climatic Partitioning)
    if p2_df is not None:
        ax_left = axes[2]
        ax_right = ax_left.twinx()
        epochs = range(1, len(p2_df) + 1)
        
        # Left axis: Total Loss & Train Data Loss
        l1 = ax_left.plot(epochs, p2_df['train_total'], label='Total Weighted Training Loss', color='#1f77b4', linewidth=2.5)
        l2 = ax_left.plot(epochs, p2_df['train_data'], label='Train Data MSE (Normalized)', color='#2ca02c', linewidth=2)
        
        # Right axis: Raw Validation MSE & Physics Loss
        l3 = ax_right.plot(epochs, p2_df['val_raw'], label='Validation Data MSE (m²)', color='#ff7f0e', linewidth=2.5, linestyle='--')
        l4 = ax_right.plot(epochs, p2_df['train_phys'], label='Physics Loss (Budyko Residual)', color='#d62728', linewidth=2.5, linestyle=':')
        
        ax_left.set_title("PINN-2 (Budyko Hydro-Climatic Partitioning)", fontsize=14, fontweight='bold')
        ax_left.set_xlabel("Epochs", fontsize=12, fontweight='bold')
        ax_left.set_ylabel("Loss (Negative Log-Variance Weighted)", fontsize=12, fontweight='bold', color='#1f77b4')
        ax_right.set_ylabel("Validation MSE (m²) / Physics Loss", fontsize=12, fontweight='bold', color='#ff7f0e')
        
        ax_left.tick_params(axis='y', labelcolor='#1f77b4')
        ax_right.tick_params(axis='y', labelcolor='#ff7f0e')
        ax_left.grid(True, linestyle=':', alpha=0.6)
        
        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc='upper right', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated disentangled loss curves:\n -> {OUT_PLOT}\n")

if __name__ == "__main__":
    plot_physics_curves()
