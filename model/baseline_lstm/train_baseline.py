import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from time import time
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline_model import DataOnlyBaselineLSTM

# ── Configuration ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_END_YEAR = 2017
OUT_ROOT = os.path.join(os.getcwd(), "antigravity", "results", "baseline", "interval_comparison")
os.makedirs(OUT_ROOT, exist_ok=True)

RESOLUTIONS = {
    "1m": {"path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_monthly.nc")), "seq_len": 12, "batch_size": 16, "epochs": 600, "lr": 1e-4, "patience": 100}
}

VOLGA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "volga_discharge.csv"))

# ── Data Loading Helpers ──────────────────────────────────────────────────
volga_df = pd.read_csv(VOLGA_PATH)
volga_df['time'] = pd.to_datetime(volga_df['time'])
volga_df = volga_df.set_index('time')

class BaselineDataset(Dataset):
    def __init__(self, ds, volga_data, seq_len=6, train_end_year=2017):
        self.times = ds.time.values
        self.seq_len = seq_len
        self.vars = ['tp', 't2m', 'ssr', 'str', 'sp', 'e', 'd2m', 'u10', 'v10', 'Qs_acc', 'Qsb_acc', 'SWE_inst']
        
        volga_interp = volga_data.reindex(pd.to_datetime(self.times), method='nearest').fillna(8000.0)
        self.volga_vals = volga_interp['volga_q'].values.astype(np.float32)
        
        feature_list = [ds[v].values.astype(np.float32) for v in self.vars]
        volga_grid = np.array([np.full_like(feature_list[0][0], q) for q in self.volga_vals], dtype=np.float32)
        feature_list.append(volga_grid)
        self.data_raw = np.stack(feature_list, axis=1)
        self.targets_raw = ds.delta_water_level.values.astype(np.float32)

        years = pd.to_datetime(self.times).year
        train_mask = years <= train_end_year
        
        self.stats = {
            'mean': np.nanmean(self.data_raw[train_mask], axis=(0, 2, 3), keepdims=True),
            'std':  np.nanstd(self.data_raw[train_mask],  axis=(0, 2, 3), keepdims=True),
            't_mean': float(np.nanmean(self.targets_raw[train_mask])),
            't_std':  float(np.nanstd(self.targets_raw[train_mask]))
        }
            
        self.data = (self.data_raw - self.stats['mean']) / (self.stats['std'] + 1e-8)
        self.data = np.nan_to_num(self.data)
        self.targets_norm = (self.targets_raw - self.stats['t_mean']) / (self.stats['t_std'] + 1e-8)
        self.targets_norm = np.nan_to_num(self.targets_norm)

    def __len__(self):
        return len(self.times) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        return (
            torch.from_numpy(x),
            torch.tensor([self.targets_norm[idx + self.seq_len - 1]]),
            torch.tensor([self.targets_raw[idx + self.seq_len - 1]])
        )

# ── Training Function ─────────────────────────────────────────────────────
def train_resolution(res_name, cfg):
    print(f"\n{'='*60}\n>>> Training Baseline LSTM Resolution: {res_name}\n{'='*60}", flush=True)
    seq_len = cfg['seq_len']
    res_dir = os.path.join(OUT_ROOT, res_name)
    os.makedirs(res_dir, exist_ok=True)
    
    ds = xr.open_dataset(cfg['path'])
    full_ds = BaselineDataset(ds, volga_df, seq_len=seq_len)
    
    years = pd.to_datetime(ds.time.values).year
    train_indices = [i for i in range(len(full_ds)) if years[i + seq_len - 1] <= TRAIN_END_YEAR]
    test_indices = [i for i in range(len(full_ds)) if years[i + seq_len - 1] > TRAIN_END_YEAR]
    
    train_loader = DataLoader(torch.utils.data.Subset(full_ds, train_indices), batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(torch.utils.data.Subset(full_ds, test_indices), batch_size=cfg['batch_size'], shuffle=False)
    
    model = DataOnlyBaselineLSTM(in_channels=13, seq_len=seq_len).to(DEVICE)
    weights_path = os.path.join(res_dir, "best_model.pth")
    t_mean, t_std = full_ds.stats['t_mean'], full_ds.stats['t_std']
    
    if os.path.exists(weights_path):
        print(f"Found existing weights at {weights_path}! Skipping training loop and running evaluation...", flush=True)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_cnt = 0
        history = {'train': [], 'val': []}
        
        print(f"Dataset Split: {len(train_indices)} train sequences, {len(test_indices)} test sequences.", flush=True)
        
        for epoch in range(cfg['epochs']):
            model.train()
            train_loss = 0
            for x, y_norm, _ in train_loader:
                x, y_norm = x.to(DEVICE), y_norm.to(DEVICE)
                optimizer.zero_grad()
                p_norm = model(x)
                loss = criterion(p_norm, y_norm)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            scheduler.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y_norm, _ in test_loader:
                    x, y_norm = x.to(DEVICE), y_norm.to(DEVICE)
                    p_norm = model(x)
                    val_loss += criterion(p_norm, y_norm).item()
                    
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(test_loader)
            history['train'].append(avg_train)
            history['val'].append(avg_val)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Ep {epoch+1:3d}/{cfg['epochs']} | Train MSE: {avg_train:.4f} | Val MSE: {avg_val:.4f}", flush=True)
                
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_cnt = 0
                torch.save(model.state_dict(), weights_path)
            else:
                patience_cnt += 1
                if patience_cnt >= cfg['patience']:
                    print(f"Early stopping at epoch {epoch+1}", flush=True)
                    break

        pd.DataFrame(history).to_csv(os.path.join(res_dir, "history.csv"), index=False)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, _, y_raw in test_loader:
            x = x.to(DEVICE)
            p_norm = model(x)
            p_raw = (p_norm.cpu() * t_std + t_mean).numpy()
            all_preds.append(p_raw)
            all_targets.append(y_raw.numpy())
            
    preds = np.concatenate(all_preds).flatten()
    actual = np.concatenate(all_targets).flatten()
    
    rmse = np.sqrt(np.mean((preds - actual)**2))
    corr = np.corrcoef(preds, actual)[0, 1]
    var_actual = np.mean((actual - np.mean(actual))**2)
    r2 = float(1.0 - (rmse**2 / (var_actual + 1e-12)))
    
    start_level = ds.water_level.values[test_indices[0] + seq_len - 1]
    actual_abs = start_level + np.cumsum(actual)
    pred_abs = start_level + np.cumsum(preds)
    
    df_out = pd.DataFrame({"Actual_DeltaH": actual, "Predicted_DeltaH": preds, "Actual_H": actual_abs, "Predicted_H": pred_abs})
    df_out.to_csv(os.path.join(res_dir, "prediction_data.csv"), index=False)
    
    metrics = {"rmse": float(rmse), "corr": float(corr), "r2": float(r2), "res": res_name}
    with open(os.path.join(res_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n[{res_name}] Final Baseline Metrics -> RMSE: {rmse:.4f}, Corr: {corr:.4f}, R^2: {r2:.4f}\n", flush=True)
    
    # Get exact timestamps for the test period to show Years on X-axis
    test_dates = pd.to_datetime(ds.time.values)[[i + seq_len - 1 for i in test_indices]]
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(test_dates, actual, label='Actual Delta H', alpha=0.7, color='#2b8cbe', linewidth=2)
    plt.plot(test_dates, preds, label='Baseline Predicted Delta H', alpha=0.9, color='#e34a33', linewidth=2, linestyle='--')
    plt.title(f"Baseline Data-Only LSTM ({res_name}) - Monthly Water Level Change ΔH (RMSE: {rmse:.4f} m, R²: {r2:.4f})", fontsize=14, fontweight='bold')
    plt.xlabel("Year", fontsize=12, fontweight='bold')
    plt.ylabel("Monthly Level Change ΔH (m)", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.subplot(2, 1, 2)
    plt.plot(test_dates, actual_abs, label='Actual Water Level (Baltic System)', color='#31a354', linewidth=2.5)
    plt.plot(test_dates, pred_abs, label='Baseline Reconstructed Level', color='#756bb1', linestyle='--', linewidth=2.5)
    plt.title(f"Baseline Data-Only LSTM ({res_name}) - 8-Year Absolute Water Level Reconstruction", fontsize=14, fontweight='bold')
    plt.xlabel("Year", fontsize=12, fontweight='bold')
    plt.ylabel("Caspian Sea Surface Level (m)", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "comparison_plots.png"), dpi=300)
    plt.close()
    
    return metrics

if __name__ == "__main__":
    start_time = time()
    summary = []
    for res in ["1m"]:
        try:
            m = train_resolution(res, RESOLUTIONS[res])
            summary.append(m)
        except Exception as e:
            print(f"Error training {res}: {e}", flush=True)
            
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(OUT_ROOT, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n=== Final Monthly Baseline Summary ===", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\nTotal elapsed time: {timedelta(seconds=time()-start_time)}", flush=True)
