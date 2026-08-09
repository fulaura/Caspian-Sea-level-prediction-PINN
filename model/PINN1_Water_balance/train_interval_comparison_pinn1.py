import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from time import time
from collections import OrderedDict

# Import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pinn1_water_balance import WaterBalancePINN1

# ── Configuration ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_END_YEAR = 2017
OUT_ROOT = "antigravity/results/pinn1/interval_comparison_retrain"
os.makedirs(OUT_ROOT, exist_ok=True)

RESOLUTIONS = {
    "1m": {"path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_monthly.nc")), "dt": 30.42 * 86400},
    "10d": {"path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_10day.nc")), "dt": 10.0 * 86400},
    "1d": {"path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_daily.nc")), "dt": 1.0 * 86400}
}

# Hyperparameters per resolution
HYPARAMS = {
    "1m": {"epochs": 600, "batch_size": 16, "lr": 1e-4, "patience": 100, "seq_len": 12, "lambda_phys": 100.0},
    "10d": {"epochs": 300, "batch_size": 32, "lr": 2e-4, "patience": 50, "seq_len": 30, "lambda_phys": 50.0},
    "1d": {"epochs": 100, "batch_size": 128, "lr": 1e-3, "patience": 20, "seq_len": 30, "lambda_phys": 20.0} 
}

# ── Data Loading Helpers ──────────────────────────────────────────────────
volga_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "volga_discharge.csv")))
volga_df['time'] = pd.to_datetime(volga_df['time'])
volga_df = volga_df.set_index('time')

class PINN1Dataset(Dataset):
    def __init__(self, ds, volga_data, seq_len=6, train_end_year=2017, stats=None):
        self.ds = ds
        self.times = ds.time.values
        self.seq_len = seq_len
        self.vars = ['tp', 't2m', 'ssr', 'str', 'sp', 'e', 'd2m', 'u10', 'v10', 'Qs_acc', 'Qsb_acc', 'SWE_inst']
        
        # Land-sea mask
        lsm_raw = ds.lsm.values
        if lsm_raw.ndim == 3: lsm_raw = lsm_raw[0]
        self.basin_mask = lsm_raw > 0.5
        self.lake_mask = lsm_raw <= 0.5
        
        # Pre-calculate physics drivers
        self.p_lake = ds.tp.where(self.lake_mask).mean(dim=['latitude', 'longitude']).values.astype(np.float32)
        self.e_lake = (-ds.e).where(self.lake_mask).mean(dim=['latitude', 'longitude']).values.astype(np.float32)
        self.p_basin = ds.tp.where(self.basin_mask).mean(dim=['latitude', 'longitude']).values.astype(np.float32)
        self.pet_basin = (ds.ssr.where(self.basin_mask).mean(dim=['latitude', 'longitude']).values.astype(np.float32) / 2.5e9)
        self.pet_basin = np.clip(self.pet_basin, a_min=1e-3, a_max=None).astype(np.float32)
        
        volga_interp = volga_data.reindex(pd.to_datetime(self.times), method='nearest').fillna(8000.0)
        self.volga_vals = volga_interp['volga_q'].values.astype(np.float32)
        
        # Gridded features
        feature_list = [ds[v].values.astype(np.float32) for v in self.vars]
        volga_grid = np.array([np.full_like(feature_list[0][0], q) for q in self.volga_vals], dtype=np.float32)
        feature_list.append(volga_grid)
        self.data_raw = np.stack(feature_list, axis=1)
        self.targets_raw = ds.delta_water_level.values.astype(np.float32)

        if stats is None:
            years = pd.to_datetime(self.times).year
            train_mask = years <= train_end_year
            
            self.stats = {
                'mean': np.nanmean(self.data_raw[train_mask], axis=(0, 2, 3), keepdims=True),
                'std':  np.nanstd(self.data_raw[train_mask],  axis=(0, 2, 3), keepdims=True),
                't_mean': float(np.nanmean(self.targets_raw[train_mask])),
                't_std':  float(np.nanstd(self.targets_raw[train_mask]))
            }
        else:
            self.stats = stats
            
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
            torch.tensor([self.targets_raw[idx + self.seq_len - 1]]),
            torch.tensor([self.p_lake[idx + self.seq_len - 1]]),
            torch.tensor([self.e_lake[idx + self.seq_len - 1]]),
            torch.tensor([self.volga_vals[idx + self.seq_len - 1]]),
            torch.tensor([self.p_basin[idx + self.seq_len - 1]]),
            torch.tensor([self.pet_basin[idx + self.seq_len - 1]])
        )

# ── Training Function ─────────────────────────────────────────────────────
def train_resolution(res_name, res_config):
    print(f"\n{'='*60}\n>>> Training PINN-1 Resolution: {res_name}\n{'='*60}", flush=True)
    hp = HYPARAMS[res_name]
    nc_path = res_config['path']
    dt_val = res_config['dt']
    seq_len = hp['seq_len']
    
    res_dir = os.path.join(OUT_ROOT, res_name)
    os.makedirs(res_dir, exist_ok=True)
    
    ds = xr.open_dataset(nc_path)
    if res_name == "1d":
        total_len = len(ds.time)
        ds = ds.isel(time=slice(total_len - 5000, total_len)) 
    
    full_ds = PINN1Dataset(ds, volga_df, seq_len=seq_len)
    
    years = pd.to_datetime(ds.time.values).year
    train_indices = [i for i in range(len(full_ds)) if years[i + seq_len - 1] <= TRAIN_END_YEAR]
    test_indices = [i for i in range(len(full_ds)) if years[i + seq_len - 1] > TRAIN_END_YEAR]
    
    train_loader = DataLoader(torch.utils.data.Subset(full_ds, train_indices), batch_size=hp['batch_size'], shuffle=True)
    test_loader = DataLoader(torch.utils.data.Subset(full_ds, test_indices), batch_size=hp['batch_size'], shuffle=False)
    
    model = WaterBalancePINN1(in_channels=13, seq_len=seq_len).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp['epochs'])
    
    t_mean, t_std = full_ds.stats['t_mean'], full_ds.stats['t_std']
    best_val_loss = float('inf')
    patience_cnt = 0
    history = {'train_data': [], 'val_mse': [], 'phys': [], 'alpha': [], 'beta': [], 'gamma': []}
    
    print(f"Dataset Split: {len(train_indices)} train sequences, {len(test_indices)} test sequences.", flush=True)
    
    for epoch in range(hp['epochs']):
        model.train()
        train_loss, phys_loss_total = 0, 0
        a_list, b_list, g_list = [], [], []
        
        for x, y_norm, y_raw, pl, el, qv, pb, petb in train_loader:
            x, y_norm, y_raw = x.to(DEVICE), y_norm.to(DEVICE), y_raw.to(DEVICE)
            pl, el, qv, pb, petb = pl.to(DEVICE), el.to(DEVICE), qv.to(DEVICE), pb.to(DEVICE), petb.to(DEVICE)
            
            optimizer.zero_grad()
            dh_pred_norm, a, b, g, lv_data, lv_phys = model(x)
            dh_pred_raw = dh_pred_norm * t_std + t_mean
            
            loss_data = nn.MSELoss()(dh_pred_norm, y_norm)
            loss_phys, _ = model.water_balance_loss(dh_pred_raw, a, b, g, pl, el, qv, pb, petb, dt_val)
            
            loss = loss_data + hp['lambda_phys'] * loss_phys
            prior_penalty = 5.0 * (torch.pow(a - 1.0, 2).mean() + torch.pow(b - 1.0, 2).mean())
            loss += prior_penalty
            mean_bias = 50.0 * torch.abs(dh_pred_norm.mean() - y_norm.mean())
            loss += mean_bias
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss_data.item()
            phys_loss_total += loss_phys.item()
            a_list.append(a.mean().item())
            b_list.append(b.mean().item())
            g_list.append(g.mean().item())
            
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y_norm, y_raw, *_ in test_loader:
                x, y_norm = x.to(DEVICE), y_norm.to(DEVICE)
                p_norm, *_ = model(x)
                val_loss += nn.MSELoss()(p_norm, y_norm).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        avg_phys = phys_loss_total / len(train_loader)
        
        history['train_data'].append(avg_train)
        history['val_mse'].append(avg_val)
        history['phys'].append(avg_phys)
        history['alpha'].append(np.mean(a_list))
        history['beta'].append(np.mean(b_list))
        history['gamma'].append(np.mean(g_list))
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Ep {epoch+1:3d}/{hp['epochs']} | Data: {avg_train:.4f} | Phys: {avg_phys:.4f} | Val MSE: {avg_val:.4f} | a={history['alpha'][-1]:.2f} b={history['beta'][-1]:.2f}", flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(res_dir, "best_model.pth"))
        else:
            patience_cnt += 1
            if patience_cnt >= hp['patience']:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

    pd.DataFrame(history).to_csv(os.path.join(res_dir, "history.csv"), index=False)

    model.load_state_dict(torch.load(os.path.join(res_dir, "best_model.pth")))
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y_norm, y_raw, *_ in test_loader:
            x, y_norm = x.to(DEVICE), y_norm.to(DEVICE)
            p_norm, *_ = model(x)
            p_raw = (p_norm.cpu() * t_std + t_mean).numpy()
            all_preds.append(p_raw)
            all_targets.append(y_raw.numpy())
            
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    rmse = np.sqrt(np.mean((preds - targets)**2))
    corr = np.corrcoef(preds, targets)[0, 1]
    var_actual = np.mean((targets - np.mean(targets))**2)
    r2 = float(1.0 - (rmse**2 / (var_actual + 1e-12)))
    
    start_level = ds.water_level.values[test_indices[0] + seq_len - 1]
    actual_abs = start_level + np.cumsum(targets)
    pred_abs = start_level + np.cumsum(preds)
    
    plot_df = pd.DataFrame({
        'Actual_DeltaH': targets,
        'Predicted_DeltaH': preds,
        'Actual_Level': actual_abs,
        'Predicted_Level': pred_abs
    })
    plot_df.to_csv(os.path.join(res_dir, "prediction_data.csv"), index=False)
    
    metrics = {"rmse": float(rmse), "corr": float(corr), "r2": float(r2), "res": res_name}
    with open(os.path.join(res_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[{res_name}] PINN-1 Metrics -> RMSE: {rmse:.4f}, Corr: {corr:.4f}, R^2: {r2:.4f}\n", flush=True)

    test_dates = pd.to_datetime(ds.time.values)[[i + seq_len - 1 for i in test_indices]]
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(test_dates, targets, label='Actual Delta H', alpha=0.7, color='#2b8cbe', linewidth=2)
    plt.plot(test_dates, preds, label='PINN-1 Predicted Delta H', color='#e34a33', linewidth=2, linestyle='--')
    plt.title(f"PINN-1 Water Balance ({res_name}) - Water Level Change ΔH (RMSE: {rmse:.4f} m, R²: {r2:.4f})", fontsize=14, fontweight='bold')
    plt.xlabel("Year", fontsize=12, fontweight='bold')
    plt.ylabel("Monthly Level Change ΔH (m)", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.subplot(2, 1, 2)
    plt.plot(test_dates, actual_abs, label='Actual Water Level (Baltic System)', color='#31a354', linewidth=2.5)
    plt.plot(test_dates, pred_abs, label='PINN-1 Reconstructed Level', color='#756bb1', linestyle='--', linewidth=2.5)
    plt.title(f"PINN-1 Water Balance ({res_name}) - Absolute Water Level Reconstruction", fontsize=14, fontweight='bold')
    plt.xlabel("Year", fontsize=12, fontweight='bold')
    plt.ylabel("Caspian Sea Surface Level (m)", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "comparison_plots.png"), dpi=300)
    plt.close()
    
    return metrics

# ── Main Loop ─────────────────────────────────────────────────────────────
start_time = time()
summary = []
for res in ["1m"]: 
    try:
        res_info = train_resolution(res, RESOLUTIONS[res])
        summary.append(res_info)
    except Exception as e:
        print(f"Error training {res}: {e}")

# Save final comparison summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUT_ROOT, "comparison_summary.csv"), index=False)

print(f"\nDone! Total time: {timedelta(seconds=time()-start_time)}")
