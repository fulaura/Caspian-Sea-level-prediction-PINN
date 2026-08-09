import os
import json
import numpy as np
import pandas as pd

MODELS = ["pinn1", "pinn2"]
RESOLUTIONS = ["1m", "10d"]
BASE_DIR = "paper_v2/results"

def update_metrics():
    print("=== Adding R^2 (Nash-Sutcliffe Efficiency) to PINN-1 and PINN-2 Metrics ===")
    
    for model_name in MODELS:
        model_dir = os.path.join(BASE_DIR, model_name, "interval_comparison")
        if not os.path.exists(model_dir):
            print(f"Skipping {model_name}: directory not found at {model_dir}")
            continue
            
        print(f"\n>>> Processing Model: {model_name.upper()}...")
        summary_rows = []
        
        for res in RESOLUTIONS:
            res_dir = os.path.join(model_dir, res)
            csv_path = os.path.join(res_dir, "prediction_data.csv")
            json_path = os.path.join(res_dir, "metrics.json")
            
            if not os.path.exists(csv_path) or not os.path.exists(json_path):
                print(f"  [{res}] Warning: Missing prediction_data.csv or metrics.json. Skipping.")
                continue
                
            # 1. Load predictions
            df = pd.read_csv(csv_path)
            actual = df['Actual_DeltaH'].values
            pred = df['Predicted_DeltaH'].values
            
            # 2. Calculate R^2 / Nash-Sutcliffe Efficiency
            mse = np.mean((actual - pred)**2)
            var_actual = np.mean((actual - np.mean(actual))**2)
            r2 = float(1.0 - (mse / (var_actual + 1e-12)))
            
            # 3. Load existing JSON to get exact RMSE and Correlation
            with open(json_path, "r") as f:
                metrics = json.load(f)
                
            metrics['r2'] = r2
            
            # 4. Save updated JSON
            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=2)
                
            print(f"  [{res}] Updated JSON -> RMSE: {metrics['rmse']:.4f}, Corr: {metrics['corr']:.4f}, R^2: {r2:.4f}")
            
            summary_rows.append({
                "res": res,
                "rmse": metrics['rmse'],
                "corr": metrics['corr'],
                "r2": r2
            })
            
        # 5. Update summary CSV
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(model_dir, "comparison_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"  Updated master summary table at {summary_path}")
            print(summary_df.to_string(index=False))

if __name__ == "__main__":
    update_metrics()
