import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "paper_v2/results"
OUT_PLOT = os.path.join(BASE_DIR, "all_models_loss_curves.png")

def plot_all_curves():
    print("=== Generating Enhanced Dual-Axis Loss Curves ===")
    
    b_path = os.path.join(BASE_DIR, "baseline", "interval_comparison", "1m", "history.csv")
    p1_path = os.path.join(BASE_DIR, "pinn1", "interval_comparison", "1m", "history.csv")
    p2_path = os.path.join(BASE_DIR, "pinn2", "interval_comparison", "1m", "history.csv")
    
    b_df = pd.read_csv(b_path) if os.path.exists(b_path) else None
    p1_df = pd.read_csv(p1_path) if os.path.exists(p1_path) else None
    p2_df = pd.read_csv(p2_path) if os.path.exists(p2_path) else None
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("Training & Validation Loss Convergence across Models (1 Month Resolution)", fontsize=18, fontweight='bold', y=1.03)
    
    # 1. Baseline (Single Axis since both train and val are on the same normalized scale)
    if b_df is not None:
        ax = axes[0]
        epochs = range(1, len(b_df) + 1)
        l1 = ax.plot(epochs, b_df['train'], label='Train MSE (Normalized)', color='#2b8cbe', linewidth=2)
        l2 = ax.plot(epochs, b_df['val'], label='Validation MSE (Normalized)', color='#e34a33', linewidth=2.5)
        ax.set_title("Ablation Baseline (Data-Only LSTM)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Epochs", fontsize=12, fontweight='bold')
        ax.set_ylabel("Normalized Mean Squared Error", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=11)
        
    # 2. PINN-1 (Dual Axis: Left for Total Negative Log Loss, Right for Raw Validation MSE)
    if p1_df is not None:
        ax_left = axes[1]
        ax_right = ax_left.twinx()
        
        epochs = range(1, len(p1_df) + 1)
        l1 = ax_left.plot(epochs, p1_df['train_total'], label='Total Training Loss (Left Axis)', color='#31a354', linewidth=2.5)
        l2 = ax_right.plot(epochs, p1_df['val_mse'], label='Validation Data MSE (Right Axis)', color='#756bb1', linewidth=2.5, linestyle='--')
        
        ax_left.set_title("PINN-1 (Water Balance Conservation)", fontsize=14, fontweight='bold')
        ax_left.set_xlabel("Epochs", fontsize=12, fontweight='bold')
        ax_left.set_ylabel("Total Loss (Log-Variance Weighted)", fontsize=12, fontweight='bold', color='#31a354')
        ax_right.set_ylabel("Validation Data MSE (m²)", fontsize=12, fontweight='bold', color='#756bb1')
        
        ax_left.tick_params(axis='y', labelcolor='#31a354')
        ax_right.tick_params(axis='y', labelcolor='#756bb1')
        ax_left.grid(True, linestyle=':', alpha=0.6)
        
        # Combine legends
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc='upper right', fontsize=11)
        
    # 3. PINN-2 (Dual Axis: Left for Total Negative Log Loss, Right for Raw Validation MSE)
    if p2_df is not None:
        ax_left = axes[2]
        ax_right = ax_left.twinx()
        
        epochs = range(1, len(p2_df) + 1)
        l1 = ax_left.plot(epochs, p2_df['train_total'], label='Total Training Loss (Left Axis)', color='#1f77b4', linewidth=2.5)
        l2 = ax_right.plot(epochs, p2_df['val_raw'], label='Validation Data MSE (Right Axis)', color='#ff7f0e', linewidth=2.5, linestyle='--')
        
        ax_left.set_title("PINN-2 (Budyko Hydro-Climatic Partitioning)", fontsize=14, fontweight='bold')
        ax_left.set_xlabel("Epochs", fontsize=12, fontweight='bold')
        ax_left.set_ylabel("Total Loss (Log-Variance Weighted)", fontsize=12, fontweight='bold', color='#1f77b4')
        ax_right.set_ylabel("Validation Data MSE (m²)", fontsize=12, fontweight='bold', color='#ff7f0e')
        
        ax_left.tick_params(axis='y', labelcolor='#1f77b4')
        ax_right.tick_params(axis='y', labelcolor='#ff7f0e')
        ax_left.grid(True, linestyle=':', alpha=0.6)
        
        # Combine legends
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc='upper right', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated enhanced dual-axis loss curves:\n -> {OUT_PLOT}\n")

if __name__ == "__main__":
    plot_all_curves()
