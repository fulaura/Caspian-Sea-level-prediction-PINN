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
from datetime import datetime
from time import time
from datetime import timedelta

# Import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pinn2_budyko import BudykoPINN2_v21

# ── Configuration ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_END_YEAR = 2017
SEQ_LEN = 6        
OUT_ROOT = "antigravity/results/pinn2/interval_comparison_v21_style"
os.makedirs(OUT_ROOT, exist_ok=True)

RESOLUTIONS = {
    "1m": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_monthly.nc")),
    "10d": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_10day.nc")),
    "1d": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_daily.nc"))
}

# Hyperparameters per resolution - 1m increased to 300 epochs per v21
HYPARAMS = {
    "1m": {"epochs": 300, "batch_size": 16, "lr": 3e-4},
    "10d": {"epochs": 150, "batch_size": 32, "lr": 3e-4},
    "1d": {"epochs": 50, "batch_size": 128, "lr": 1e-3}
}

# ── Data Loading Helpers ──────────────────────────────────────────────────
lsm_raw = xr.open_dataset(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'lsm.area-subset.47.5.54.5.36.5.45.5.nc')))

volga_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "volga_discharge.csv")))
volga_df['time'] = pd.to_datetime(volga_df['time'])
volga_df = volga_df.set_index('time')

class BudykoDataset(Dataset):
    def __init__(self, xarray_ds, volga_data, p_b_s, pet_b_s, p_l_s, e_l_s, dh_s, seq_len=6, stats=None):
        self.ds = xarray_ds
        self.times = xarray_ds.time.values
        self.seq_len = seq_len
        self.vars = ['tp', 't2m', 'ssr', 'str', 'sp', 'e', 'd2m', 'u10', 'v10', 'Qs_acc', 'Qsb_acc', 'SWE_inst']
        
        # Volga interpolation
        volga_interp = volga_data.reindex(pd.to_datetime(self.times), method='nearest').fillna(8000.0)
        self.volga_vals = volga_interp['volga_q'].values
        
        self.targets = dh_s.values
        self.p_b, self.pet_b = p_b_s.values, pet_b_s.values
        self.p_l, self.e_l = p_l_s.values, e_l_s.values

        # Decide whether to load in memory
        self.in_memory = False
        if len(self.times) < 5000: # Approx < 2GB for monthly/10d
            print("Loading dataset into memory...")
            data_list = []
            for v in self.vars:
                data_list.append(self.ds[v].values)
            v_spatial = self.volga_vals[:, None, None] * np.ones_like(data_list[0][0])
            self.data = np.stack(data_list + [v_spatial], axis=1)
            self.in_memory = True

        if stats is None:
            if self.in_memory:
                self.stats = {'mean': np.nanmean(self.data, axis=(0, 2, 3), keepdims=True), 
                              'std': np.nanstd(self.data, axis=(0, 2, 3), keepdims=True)}
            else:
                subset_indices = np.random.choice(len(self.times), min(200, len(self.times)), replace=False)
                subset_data = []
                for v in self.vars:
                    subset_data.append(self.ds[v].isel(time=subset_indices).values)
                v_subset = self.volga_vals[subset_indices]
                v_spatial = v_subset[:, None, None] * np.ones_like(subset_data[0][0])
                subset_data.append(v_spatial)
                subset_data = np.stack(subset_data, axis=1)
                self.stats = {'mean': np.nanmean(subset_data, axis=(0, 2, 3), keepdims=True), 
                              'std': np.nanstd(subset_data, axis=(0, 2, 3), keepdims=True)}
            self.target_mean, self.target_std = np.mean(self.targets), np.std(self.targets)
        else:
            self.stats = stats; self.target_mean, self.target_std = stats['target_mean'], stats['target_std']
        
        self.targets_raw = self.targets
        self.targets_norm = (self.targets - self.target_mean) / (self.target_std + 1e-8)
        self.valid_indices = [i for i in range(len(self.times)) if i >= seq_len - 1]

    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx):
        f_idx = self.valid_indices[idx]
        
        if self.in_memory:
            x = self.data[f_idx - self.seq_len + 1 : f_idx + 1]
        else:
            # Slice xarray
            slice_ds = self.ds.isel(time=slice(f_idx - self.seq_len + 1, f_idx + 1))
            x_list = []
            for v in self.vars:
                x_list.append(slice_ds[v].values)
                
            v_slice = self.volga_vals[f_idx - self.seq_len + 1 : f_idx + 1]
            volga_spatial = v_slice[:, None, None] * np.ones_like(x_list[0][0])
            x = np.stack(x_list + [volga_spatial], axis=1)

        x = np.nan_to_num((x - self.stats['mean']) / (self.stats['std'] + 1e-8))
        
        y_norm = self.targets_norm[f_idx]
        y_raw = self.targets_raw[f_idx]
        return (torch.FloatTensor(x), torch.FloatTensor([y_norm]), torch.FloatTensor([y_raw]),
                torch.FloatTensor([self.p_b[f_idx]]), torch.FloatTensor([self.pet_b[f_idx]]), 
                torch.FloatTensor([self.p_l[f_idx]]), torch.FloatTensor([self.e_l[f_idx]]))

# ── Training Function ────────────────────────────────────────────────────
def train_resolution(res_name, ds_path):
    print(f"\n>>> Training Resolution: {res_name} ({ds_path}) [v21 STYLE]")
    out_dir = os.path.join(OUT_ROOT, res_name)
    os.makedirs(out_dir, exist_ok=True)
    
    cfg = HYPARAMS[res_name]
    PATIENCE = 50 # Early stopping patience from v21
    
    # Load DS
    if res_name == "1d":
        ds = xr.open_dataset(ds_path, chunks={'time': 500})
        ds = ds.isel(time=slice(-500, None))
    else:
        ds = xr.open_dataset(ds_path)
    
    # Align masks
    lsm_aligned = lsm_raw.lsm.interp_like(ds, method='nearest')
    basin_mask = lsm_aligned.values[0] > 0.5
    lake_mask = lsm_aligned.values[0] <= 0.5
    
    # Prepare basin/lake series
    p_b = ds.tp.where(basin_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
    pet_b = (ds.ssr.where(basin_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0) / 2.5e9).clip(lower=0.001)
    p_l = ds.tp.where(lake_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
    e_l = -ds.e.where(lake_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
    
    water_level = ds.water_level.to_series().ffill().fillna(0)
    delta_h = water_level.diff().fillna(0)
    
    # Dataset
    full_ds = BudykoDataset(ds, volga_df, p_b, pet_b, p_l, e_l, delta_h, seq_len=SEQ_LEN)
    
    # Split
    n_samples = len(full_ds)
    if res_name == "1d":
        train_size = int(0.8 * n_samples)
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, n_samples))
    else:
        all_years = pd.to_datetime(ds.time.values).year
        train_indices = [i for i, v in enumerate(full_ds.valid_indices) if all_years[v] <= TRAIN_END_YEAR]
        test_indices = [i for i, v in enumerate(full_ds.valid_indices) if all_years[v] > TRAIN_END_YEAR]

    train_loader = DataLoader(torch.utils.data.Subset(full_ds, train_indices), batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(torch.utils.data.Subset(full_ds, test_indices), batch_size=cfg['batch_size'], shuffle=False)

    # Model
    model = BudykoPINN2_v21(in_channels=13, seq_len=SEQ_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    t_mean, t_std = full_ds.target_mean, full_ds.target_std
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_total': [], 'val_raw': []}

    for epoch in range(cfg['epochs']):
        model.train()
        train_l = 0
        for batch in train_loader:
            x, y_norm, y_raw, p_b_b, pet_b_b, p_l_b, e_l_b = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            
            pred_dh_norm, n_t, scale_t = model(x)
            pred_dh_raw = pred_dh_norm * t_std + t_mean
            
            loss_data = nn.functional.mse_loss(pred_dh_norm, y_norm)
            loss_phys = model.budyko_loss(pred_dh_raw, n_t, scale_t, p_b_b, pet_b_b, p_l_b, e_l_b)
            
            precision_data = torch.exp(-model.log_vars[0])
            precision_phys = torch.exp(-model.log_vars[1])
            total_loss = 0.5 * precision_data * loss_data + 0.5 * precision_phys * loss_phys + 0.5 * model.log_vars[0] + 0.5 * model.log_vars[1]
            
            total_loss.backward()
            
            # v21 ADVANTAGE: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_l += total_loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_l = 0
        with torch.no_grad():
            for batch in test_loader:
                x, _, y_raw, *_ = [b.to(DEVICE) for b in batch]
                p_norm, _, _ = model(x)
                p_raw = p_norm * t_std + t_mean
                val_l += nn.functional.mse_loss(p_raw, y_raw).item()
        
        avg_train = train_l / len(train_loader)
        avg_val = val_l / len(test_loader)
        history['train_total'].append(avg_train)
        history['val_raw'].append(avg_val)
        
        # v21 ADVANTAGE: Early Stopping logic
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{cfg['epochs']} | Train Tot: {avg_train:.4f} | Val MSE(m2): {avg_val:.6f} | Patience: {patience_counter}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state: model.load_state_dict(best_model_state)
    
    # Final Metrics
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, _, y_raw, *_ = [b.to(DEVICE) for b in batch]
            p_norm, _, _ = model(x)
            p_raw = p_norm * t_std + t_mean
            test_preds.append(p_raw.cpu().numpy().flatten())
            test_targets.append(y_raw.cpu().numpy().flatten())
            
    p_raw_arr = np.concatenate(test_preds)
    t_raw_arr = np.concatenate(test_targets)
    rmse = np.sqrt(np.mean((p_raw_arr - t_raw_arr)**2))
    corr = np.corrcoef(p_raw_arr, t_raw_arr)[0, 1]
    
    start_level = water_level.iloc[test_indices[0] + SEQ_LEN - 1]
    actual_abs = start_level + np.cumsum(t_raw_arr)
    pred_abs = start_level + np.cumsum(p_raw_arr)
    
    results = {"rmse": float(rmse), "corr": float(corr), "res": res_name}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f: json.dump(results, f, indent=2)
    
    # Save Model
    torch.save(model.state_dict(), os.path.join(out_dir, "model_weights.pth"))
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(t_raw_arr, label='Actual Delta H', alpha=0.5, color='#3498db')
    ax1.plot(p_raw_arr, label='Predicted Delta H', color='#e74c3c', linewidth=1.5)
    ax1.set_title(f"Resolution: {res_name} (v21 Style) - Delta H (RMSE: {rmse:.4f})")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(actual_abs, label='Actual Water Level', color='#2ecc71', linewidth=2)
    ax2.plot(pred_abs, label='PINN Predicted Level', color='#e67e22', linestyle='--', linewidth=2)
    ax2.set_title(f"Resolution: {res_name} (v21 Style) - Absolute Water Level Reconstruction")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_plots.png"))
    plt.close()
    
    return results

# ── Main Execution ────────────────────────────────────────────────────────
start_time = time()
summary = []
for res in ["10d", "1m"]: 
    try:
        res_info = train_resolution(res, RESOLUTIONS[res])
        summary.append(res_info)
    except Exception as e:
        print(f"Error training {res}: {e}")

try:
    print("\n>>> Training Resolution: 1d (Daily) [v21 STYLE]...")
    res_info = train_resolution("1d", RESOLUTIONS["1d"])
    summary.append(res_info)
except Exception as e:
    print(f"Error training 1d: {e}")

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUT_ROOT, "comparison_summary.csv"), index=False)
print("\n=== Final Summary (v21 Style) ===")
print(summary_df)

end_time = time()
print("Total training time:", timedelta(seconds=end_time - start_time))
