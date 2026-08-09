import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from time import time
from datetime import timedelta

# Ensure imports work regardless of where script is run
base_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model"))
sys.path.append(os.path.join(base_model_dir, "baseline_lstm"))
from baseline_model import DataOnlyBaselineLSTM

sys.path.append(os.path.join(base_model_dir, "interval_comparison"))
from pinn2_budyko import BudykoPINN2_v21

sys.path.append(os.path.join(base_model_dir, "PINN1_Water_balance"))
from pinn1_water_balance import WaterBalancePINN1

# ── Configuration & Reference Hyperparameters ─────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_END_YEAR = 2017
SEQ_LEN = 6        # Strictly aligned with train_interval_comparison.py
WEIGHT_DECAY = 0.05

SEEDS = [42, 123, 2024, 7, 999]
ORDERED_RESOLUTIONS = ["1m", "10d"]
ORDERED_MODELS = ["baseline", "pinn2", "pinn1"]

OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "multiseed_evaluation"))

DRY_RUN = "--dry-run" in sys.argv

RESOLUTIONS = {
    "1m": {"path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_monthly.nc")), "dt": 30.42 * 86400, "epochs": 2 if DRY_RUN else 250, "batch_size": 16, "lr": 3e-4, "lambda_pinn1": 100.0},
    "10d": {"path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_10day.nc")), "dt": 10.0 * 86400, "epochs": 2 if DRY_RUN else 150, "batch_size": 32, "lr": 3e-4, "lambda_pinn1": 50.0},

}

if DRY_RUN:
    SEEDS = [11]  # Only one seed for dry run
    print("\n[WARNING] RUNNING IN DRY-RUN MODE (1 seed, 2 epochs per model)\n")

VOLGA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "volga_discharge.csv"))
LSM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lsm.area-subset.47.5.54.5.36.5.45.5.nc"))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Optimization 1: Enable cuDNN kernel benchmarking for fastest C++ execution
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# ── Unified Pre-Tensorized Data Loading Helper ────────────────────────────
class UnifiedMultiSeedDataset(Dataset):
    def __init__(self, ds, volga_data, p_b_s, pet_b_s, p_l_s, e_l_s, seq_len=6, train_end_year=2017):
        self.times = ds.time.values
        self.seq_len = seq_len
        self.vars = ['tp', 't2m', 'ssr', 'str', 'sp', 'e', 'd2m', 'u10', 'v10', 'Qs_acc', 'Qsb_acc', 'SWE_inst']
        
        # Volga interpolation
        volga_interp = volga_data.reindex(pd.to_datetime(self.times), method='nearest').fillna(8000.0)
        volga_vals_np = volga_interp['volga_q'].values.astype(np.float32)
        
        p_b_np, pet_b_np = p_b_s.values.astype(np.float32), pet_b_s.values.astype(np.float32)
        p_l_np, e_l_np = p_l_s.values.astype(np.float32), e_l_s.values.astype(np.float32)

        print("Loading and pre-tensorizing dataset features in RAM...", flush=True)
        T = len(self.times)
        Lat, Lon = ds[self.vars[0]].shape[1], ds[self.vars[0]].shape[2]
        data_raw_np = np.empty((T, 13, Lat, Lon), dtype=np.float32)
        
        for i, v in enumerate(self.vars):
            # Load in chunks to prevent HDF5 memory fragmentation errors on Windows
            for t in range(0, T, 500):
                data_raw_np[t:t+500, i, :, :] = ds[v][t:t+500].values.astype(np.float32)
            
        data_raw_np[:, 12, :, :] = volga_vals_np[:, None, None]
        
        water_level_np = ds.water_level.to_series().ffill().fillna(0)
        targets_raw_np = water_level_np.diff().fillna(0).values.astype(np.float32)
        self.water_level = water_level_np.values.astype(np.float32)

        years = pd.to_datetime(self.times).year
        train_mask = years <= train_end_year
        
        self.stats = {
            'mean': np.nanmean(data_raw_np[train_mask], axis=(0, 2, 3), keepdims=True),
            'std':  np.nanstd(data_raw_np[train_mask],  axis=(0, 2, 3), keepdims=True),
            't_mean': float(np.nanmean(targets_raw_np[train_mask])),
            't_std':  float(np.nanstd(targets_raw_np[train_mask]))
        }
            
        data_np = np.nan_to_num((data_raw_np - self.stats['mean']) / (self.stats['std'] + 1e-8))
        targets_norm_np = np.nan_to_num((targets_raw_np - self.stats['t_mean']) / (self.stats['t_std'] + 1e-8))
        self.valid_indices = [i for i in range(len(self.times)) if i >= seq_len - 1]

        # Optimization 2: Pre-convert everything into PyTorch Tensors in RAM
        self.data = torch.from_numpy(data_np).float()
        self.targets_norm = torch.from_numpy(targets_norm_np).float().unsqueeze(1)
        self.targets_raw = torch.from_numpy(targets_raw_np).float().unsqueeze(1)
        
        self.p_b = torch.from_numpy(p_b_s.values.astype(np.float32)).float().unsqueeze(1)
        self.pet_b = torch.from_numpy(pet_b_s.values.astype(np.float32)).float().unsqueeze(1)
        self.p_l = torch.from_numpy(p_l_s.values.astype(np.float32)).float().unsqueeze(1)
        self.e_l = torch.from_numpy(e_l_s.values.astype(np.float32)).float().unsqueeze(1)
        self.volga_vals = torch.from_numpy(volga_vals_np).float().unsqueeze(1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        f_idx = self.valid_indices[idx]
        return (
            self.data[f_idx - self.seq_len + 1 : f_idx + 1],
            self.targets_norm[f_idx],
            self.targets_raw[f_idx],
            self.p_b[f_idx],
            self.pet_b[f_idx],
            self.p_l[f_idx],
            self.e_l[f_idx],
            self.volga_vals[f_idx],
            f_idx
        )

# ── Optimized Training Engine (AMP + Pinned Memory) ───────────────────────
def train_eval_seed(model_name, seed, full_ds, train_loader, test_loader, cfg, model_dir):
    print(f"\n{'='*50}\n>>> Training {model_name.upper()} (Seed: {seed})\n{'='*50}", flush=True)
    set_seed(seed)
    seed_dir = os.path.join(model_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    weights_path = os.path.join(seed_dir, "model_weights.pth")
    
    t_mean, t_std = full_ds.stats['t_mean'], full_ds.stats['t_std']
    epochs, lr = cfg['epochs'], cfg['lr']
    dt_val = cfg['dt']

    if model_name == "baseline":
        model = DataOnlyBaselineLSTM(in_channels=13, seq_len=SEQ_LEN).to(DEVICE)
    elif model_name == "pinn2":
        model = BudykoPINN2_v21(in_channels=13, seq_len=SEQ_LEN).to(DEVICE)
    elif model_name == "pinn1":
        model = WaterBalancePINN1(in_channels=13, seq_len=SEQ_LEN).to(DEVICE)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Optimization 3: Enable GradScaler for Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    
    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'train_data_mse': [], 'train_phys_loss': [], 'val_mse_raw': []}

    for epoch in range(epochs):
        model.train()
        train_tot_loss, train_d_loss, train_p_loss = 0, 0, 0
        
        for batch in train_loader:
            # Asynchronous non-blocking DMA transfer to GPU
            x, y_norm, y_raw, pb, petb, pl, el, qv = [b.to(DEVICE, non_blocking=True) for b in batch[:-1]]
            optimizer.zero_grad()
            
            # Optimization 4: FP16 Autocast for GPU Tensor Cores
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                if model_name == "baseline":
                    pred_dh_norm = model(x)
                    loss_data = F.mse_loss(pred_dh_norm, y_norm)
                    total_loss = loss_data
                    loss_phys_val = 0.0
                elif model_name == "pinn2":
                    pred_dh_norm, n_t, scale_t = model(x)
                    pred_dh_raw = pred_dh_norm * t_std + t_mean
                    loss_data = F.mse_loss(pred_dh_norm, y_norm)
                    loss_phys = model.budyko_loss(pred_dh_raw, n_t, scale_t, pb, petb, pl, el)
                    
                    prec_data = torch.exp(-model.log_vars[0])
                    prec_phys = torch.exp(-model.log_vars[1])
                    total_loss = 0.5 * prec_data * loss_data + 0.5 * prec_phys * loss_phys + 0.5 * model.log_vars[0] + 0.5 * model.log_vars[1]
                    loss_phys_val = loss_phys.item()
                elif model_name == "pinn1":
                    pred_dh_norm, a, b, g, lv_data, lv_phys = model(x)
                    pred_dh_raw = pred_dh_norm * t_std + t_mean
                    loss_data = F.mse_loss(pred_dh_norm, y_norm)
                    loss_phys, _ = model.water_balance_loss(pred_dh_raw, a, b, g, pl, el, qv, pb, petb, dt_val)
                    
                    total_loss = loss_data + cfg['lambda_pinn1'] * loss_phys
                    prior_pen = 5.0 * (torch.pow(a - 1.0, 2).mean() + torch.pow(b - 1.0, 2).mean())
                    mean_bias = 50.0 * torch.abs(pred_dh_norm.mean() - y_norm.mean())
                    total_loss = total_loss + prior_pen + mean_bias
                    loss_phys_val = loss_phys.item()

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_tot_loss += total_loss.item()
            train_d_loss += loss_data.item()
            train_p_loss += loss_phys_val
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                x, _, y_raw = [b.to(DEVICE, non_blocking=True) for b in batch[:3]]
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                    if model_name == "baseline": p_norm = model(x)
                    else: p_norm, *_ = model(x)
                p_raw = p_norm * t_std + t_mean
                val_loss += F.mse_loss(p_raw, y_raw).item()
                
        avg_tot = train_tot_loss / len(train_loader)
        avg_d = train_d_loss / len(train_loader)
        avg_p = train_p_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        
        history['train_loss'].append(avg_tot)
        history['train_data_mse'].append(avg_d)
        history['train_phys_loss'].append(avg_p)
        history['val_mse_raw'].append(avg_val)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()
            
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Ep {epoch+1:3d}/{epochs} | Tot Loss: {avg_tot:.4f} | Data MSE: {avg_d:.4f} | Phys Loss: {avg_p:.4f} | Val MSE: {avg_val:.6f}", flush=True)

    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), weights_path)
    pd.DataFrame(history).to_csv(os.path.join(seed_dir, "history.csv"), index=False)
    
    # ── Final Test Metrics ──
    model.eval()
    all_preds, all_targets, f_indices = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x, _, y_raw = [b.to(DEVICE, non_blocking=True) for b in batch[:3]]
            idx = batch[-1]
            if model_name == "baseline": p_norm = model(x)
            else: p_norm, *_ = model(x)
            p_raw = (p_norm.cpu() * t_std + t_mean).numpy().flatten()
            all_preds.append(p_raw)
            all_targets.append(y_raw.cpu().numpy().flatten())
            f_indices.append(idx.cpu().numpy().flatten())
            
    preds = np.concatenate(all_preds)
    actual = np.concatenate(all_targets)
    test_valid_idx = np.concatenate(f_indices)
    
    rmse = np.sqrt(np.mean((preds - actual)**2))
    corr = np.corrcoef(preds, actual)[0, 1]
    var_actual = np.mean((actual - np.mean(actual))**2)
    r2 = float(1.0 - (rmse**2 / (var_actual + 1e-12)))
    
    start_h = full_ds.water_level[test_valid_idx[0]]
    actual_abs = start_h + np.cumsum(actual)
    pred_abs = start_h + np.cumsum(preds)
    
    df_pred = pd.DataFrame({
        "Actual_DeltaH": actual, "Predicted_DeltaH": preds,
        "Actual_Level": actual_abs, "Predicted_Level": pred_abs
    })
    df_pred.to_csv(os.path.join(seed_dir, "prediction_data.csv"), index=False)
    
    metrics = {"seed": seed, "rmse": float(rmse), "corr": float(corr), "r2": float(r2)}
    with open(os.path.join(seed_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[{model_name.upper()} | Seed {seed}] -> RMSE: {rmse:.4f} m | R^2: {r2:.4f} | Corr: {corr:.4f}", flush=True)
    return metrics

def run_all_benchmarks():
    total_start_time = time()
    
    print(f"Loading LSM mask {LSM_PATH}...", flush=True)
    lsm_raw = xr.open_dataset(LSM_PATH)
    volga_df = pd.read_csv(VOLGA_PATH)
    volga_df['time'] = pd.to_datetime(volga_df['time'])
    volga_df = volga_df.set_index('time')
    
    for res_name in ORDERED_RESOLUTIONS:
        cfg = RESOLUTIONS[res_name]
        ds_path = cfg['path']
        print(f"\n{'#'*70}\n🚀 RUNNING RESOLUTION: {res_name.upper()} ({ds_path})\n{'#'*70}", flush=True)
        
        ds = xr.open_dataset(ds_path)
            
        lsm_aligned = lsm_raw.lsm.interp_like(ds, method='nearest')
        basin_mask = lsm_aligned.values[0] > 0.5
        lake_mask = lsm_aligned.values[0] <= 0.5
        
        p_b = ds.tp.where(basin_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
        pet_b = (ds.ssr.where(basin_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0) / 2.5e9).clip(lower=0.001)
        p_l = ds.tp.where(lake_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
        e_l = -ds.e.where(lake_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)

        full_ds = UnifiedMultiSeedDataset(ds, volga_df, p_b, pet_b, p_l, e_l, seq_len=SEQ_LEN)
        
        years = pd.to_datetime(ds.time.values).year
        train_indices = [i for i, v in enumerate(full_ds.valid_indices) if years[v] <= TRAIN_END_YEAR]
        test_indices = [i for i, v in enumerate(full_ds.valid_indices) if years[v] > TRAIN_END_YEAR]
        
        # Optimization 3: pin_memory=True for lightning-fast host-to-GPU transfer
        train_loader = DataLoader(torch.utils.data.Subset(full_ds, train_indices), batch_size=cfg['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(torch.utils.data.Subset(full_ds, test_indices), batch_size=cfg['batch_size'], shuffle=False, pin_memory=True)
        
        for model_name in ORDERED_MODELS:
            print(f"\n{'-'*60}\n⚡ Evaluating Model: {model_name.upper()} at Resolution: {res_name}\n{'-'*60}", flush=True)
            model_dir = os.path.join(OUT_ROOT, model_name, res_name)
            os.makedirs(model_dir, exist_ok=True)
            
            model_start_t = time()
            all_metrics = []
            for s in SEEDS:
                m = train_eval_seed(model_name, s, full_ds, train_loader, test_loader, cfg, model_dir)
                all_metrics.append(m)
                
            df_summary = pd.DataFrame(all_metrics)
            summary_path = os.path.join(model_dir, "multiseed_summary.csv")
            df_summary.to_csv(summary_path, index=False)
            
            mean_rmse, std_rmse = df_summary['rmse'].mean(), df_summary['rmse'].std()
            mean_r2, std_r2 = df_summary['r2'].mean(), df_summary['r2'].std()
            mean_corr, std_corr = df_summary['corr'].mean(), df_summary['corr'].std()
            
            print(f"\n🏆 CONSOLIDATED RESULT ({model_name.upper()} | {res_name} | 5 Seeds):")
            print(f"RMSE (m)   : {mean_rmse:.4f} ± {std_rmse:.4f}")
            print(f"R^2 Score  : {mean_r2:.4f} ± {std_r2:.4f}")
            print(f"Correlation: {mean_corr:.4f} ± {std_corr:.4f}")
            print(f"Elapsed for {model_name}/{res_name}: {timedelta(seconds=time() - model_start_t)}\n{'-'*60}")
            
    print(f"\n{'='*70}\n🎉 ALL BENCHMARKS COMPLETED SUCCESSFULLY! Total Time: {timedelta(seconds=time() - total_start_time)}\n{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Multi-seed Suite")
    parser.add_argument("--run-all", action="store_true", default=True, help="Run full benchmark suite")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (fast, 2 epochs)")
    args, unknown = parser.parse_known_args()
    
    if args.run_all:
        run_all_benchmarks()
