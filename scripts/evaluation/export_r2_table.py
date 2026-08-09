import os
import json
import pandas as pd

RESOLUTIONS = ["1m", "10d"]
BASE_DIR = "paper_v2/results"
OUT_CSV = os.path.join(BASE_DIR, "pinn1_vs_pinn2_r2_comparison.csv")

def export_r2_table():
    print("=== Generating PINN-1 vs PINN-2 R^2 Summary CSV ===")
    
    rows = []
    for res in RESOLUTIONS:
        p1_path = os.path.join(BASE_DIR, "pinn1", "interval_comparison", res, "metrics.json")
        p2_path = os.path.join(BASE_DIR, "pinn2", "interval_comparison", res, "metrics.json")
        
        p1_r2, p2_r2 = None, None
        
        if os.path.exists(p1_path):
            with open(p1_path, "r") as f:
                data = json.load(f)
                p1_r2 = round(data.get("r2", 0.0), 4)
                
        if os.path.exists(p2_path):
            with open(p2_path, "r") as f:
                data = json.load(f)
                p2_r2 = round(data.get("r2", 0.0), 4)
                
        rows.append({
            "Resolution": res,
            "PINN1_R2": p1_r2,
            "PINN2_R2": p2_r2
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    
    print(f"\nSuccessfully exported comparison CSV to:\n -> {OUT_CSV}\n")
    print(df.to_string(index=False))

if __name__ == "__main__":
    export_r2_table()
