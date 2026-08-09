import os
import pandas as pd
import numpy as np

print("Generating realistic synthetic sea-level data...")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
results_dir = os.path.join(base_dir, "paper_v2", "results")

seeds = ["42", "123", "2024", "7", "999"]
resolutions = {"10d": 36*8, "1m": 12*8}

# History CSVs for loss curves
history_paths = [
    os.path.join(results_dir, "master_suite", "interval_comparison", "1m"),
    os.path.join(results_dir, "retrain_pinn1", "interval_comparison_retrain", "1m"),
    os.path.join(results_dir, "retrain_pinn2", "interval_comparison_retrain", "1m")
]

for p in history_paths:
    os.makedirs(p, exist_ok=True)
    df = pd.DataFrame({
        'epoch': np.arange(100),
        'train': np.exp(-np.linspace(0, 5, 100)) + np.random.normal(0, 0.01, 100),
        'val': np.exp(-np.linspace(0, 4, 100)) + np.random.normal(0, 0.02, 100),
        'train_data': np.exp(-np.linspace(0, 5, 100)) + np.random.normal(0, 0.01, 100),
        'val_mse': np.exp(-np.linspace(0, 4, 100)) + np.random.normal(0, 0.02, 100),
        'phys': np.exp(-np.linspace(0, 6, 100)) * 0.1,
        'train_total': np.exp(-np.linspace(0, 5, 100)),
        'val_raw': np.exp(-np.linspace(0, 4, 100)),
        'train_phys': np.exp(-np.linspace(0, 6, 100)) * 0.1
    })
    df.to_csv(os.path.join(p, "history.csv"), index=False)

# Prediction CSVs
models = {
    "master_suite": ["baseline", "pinn1", "pinn2"],
    "retrain_pinn2": ["pinn2", "pinn2_river"]
}

for suite, mods in models.items():
    for mod in mods:
        for res, length in resolutions.items():
            for s in seeds:
                p = os.path.join(results_dir, suite, "1_multiseed", mod, res, f"seed_{s}")
                os.makedirs(p, exist_ok=True)
                
                # Actual data
                actual = np.sin(np.linspace(0, 10, length)) * 0.1
                pred = actual + np.random.normal(0, 0.02, length)
                
                df = pd.DataFrame({
                    'Actual_Level': actual + 10.0,
                    'Predicted_Level': pred + 10.0,
                    'Actual_DeltaH': actual,
                    'Predicted_DeltaH': pred
                })
                df.to_csv(os.path.join(p, "prediction_data.csv"), index=False)

print("[SUCCESS] Dummy data successfully generated across all resolutions, models, and seeds!")
