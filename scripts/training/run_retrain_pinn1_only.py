import os
import sys
import json
import math
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
import matplotlib.pyplot as plt

# Ensure imports work
base_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model"))
sys.path.append(os.path.join(base_model_dir, "PINN1v2"))
from pinn1v2_water_balance import WaterBalancePINN1v2, DynamicAreaWaterBalancePINN1v2, _softplus_inv

# ── Configuration & Reference Hyperparameters ─────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_END_YEAR = 2017
SEQ_LEN = 6        
WEIGHT_DECAY = 0.05

SEEDS = [42, 123, 2024, 7, 999]
ORDERED_RESOLUTIONS = ["1m", "10d"]

OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "retrain_pinn1"))

DRY_RUN = "--dry-run" in sys.argv

RESOLUTIONS = {
    "1m": {"path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_monthly.nc")), "dt": 30.42 * 86400, "epochs": 2 if DRY_RUN else 250, "batch_size": 16, "lr": 3e-4},
    "10d": {"path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_10day.nc")), "dt": 10.0 * 86400, "epochs": 2 if DRY_RUN else 150, "batch_size": 32, "lr": 3e-4},

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
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

class UnifiedMasterDataset(Dataset):
    def __init__(self, ds, volga_data, p_b_s, pet_b_s, p_l_s, e_l_s, seq_len=6, train_end_year=2017):
        self.times = ds.time.values
        self.seq_len = seq_len
        self.vars = ['tp', 't2m', 'ssr', 'str', 'sp', 'e', 'd2m', 'u10', 'v10', 'Qs_acc', 'Qsb_acc', 'SWE_inst']
        
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

        self.data = torch.from_numpy(data_np).float()
        self.targets_norm = torch.from_numpy(targets_norm_np).float().unsqueeze(1)
        self.targets_raw = torch.from_numpy(targets_raw_np).float().unsqueeze(1)
        self.p_b = torch.from_numpy(p_b_np).float().unsqueeze(1)
        self.pet_b = torch.from_numpy(pet_b_np).float().unsqueeze(1)
        self.p_l = torch.from_numpy(p_l_np).float().unsqueeze(1)
        self.e_l = torch.from_numpy(e_l_np).float().unsqueeze(1)
        self.volga_vals = torch.from_numpy(volga_vals_np).float().unsqueeze(1)
        self.lake_h = torch.from_numpy(self.water_level).float().unsqueeze(1)

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
            self.lake_h[f_idx],
            f_idx
        )

# ── General Training & Evaluation Engine ──────────────────────────────────
def run_standard_train_eval(model, train_loader, test_loader, full_ds, epochs, lr, dt_val, save_dir, model_type="pinn1", no_residual=False, lambda_phys=100.0, dry_run=False):
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "model_weights.pth")
    t_mean, t_std = full_ds.stats['t_mean'], full_ds.stats['t_std']
    
    if dry_run: epochs = 1
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))
    
    best_val_loss = float('inf')
    best_state = None
    history = {'train_tot_loss': [], 'train_data_mse': [], 'train_phys_loss': [], 'val_mse': []}

    for epoch in range(epochs):
        model.train()
        train_tot, train_d, train_p = 0, 0, 0
        for batch_idx, batch in enumerate(train_loader):
            if dry_run and batch_idx >= 2: break
            
            x, y_norm, y_raw, pb, petb, pl, el, qv, lh = [b.to(DEVICE, non_blocking=True) for b in batch[:-1]]
            optimizer.zero_grad()
            
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                if model_type == "pinn1":
                    pred_dh, a, b, g = model(x)
                    if no_residual: g = g * 0.0
                    loss_d = F.mse_loss(pred_dh, y_norm)
                    loss_p, _ = model.water_balance_loss(pred_dh * t_std + t_mean, a, b, g, pl, el, qv, pb, petb, dt_val)
                    prec_d, prec_p = torch.exp(-model.log_vars[0]), torch.exp(-model.log_vars[1])
                    total_loss = 0.5 * (prec_d * loss_d + prec_p * loss_p + model.log_vars.sum()) + 0.1 * (torch.pow(a - 1.0, 2).mean() + torch.pow(b - 1.0, 2).mean())
                elif model_type == "dynamic":
                    pred_dh, a, b, g = model(x)
                    loss_d = F.mse_loss(pred_dh, y_norm)
                    loss_p = model.dynamic_water_balance_loss(pred_dh * t_std + t_mean, a, b, g, pl, el, qv, pb, petb, lh, dt_val)
                    prec_d, prec_p = torch.exp(-model.log_vars[0]), torch.exp(-model.log_vars[1])
                    total_loss = 0.5 * (prec_d * loss_d + prec_p * loss_p + model.log_vars.sum())

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_tot += total_loss.item()
            train_d += loss_d.item()
            train_p += loss_p.item() if isinstance(loss_p, torch.Tensor) else loss_p
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if dry_run and batch_idx >= 2: break
                x, _, y_raw = [b.to(DEVICE, non_blocking=True) for b in batch[:3]]
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                    pred_dh, *_ = model(x)
                val_loss += F.mse_loss(pred_dh * t_std + t_mean, y_raw).item()
                
        n_batches = min(len(train_loader), 2 if dry_run else len(train_loader))
        avg_val = val_loss / min(len(test_loader), 2 if dry_run else len(test_loader))
        
        history['train_tot_loss'].append(train_tot / n_batches)
        history['train_data_mse'].append(train_d / n_batches)
        history['train_phys_loss'].append(train_p / n_batches)
        history['val_mse'].append(avg_val)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()
            
        if (epoch + 1) % 50 == 0 or epoch == 0 or dry_run:
            print(f"[{model_type.upper()}] Ep {epoch+1:3d}/{epochs} | Tot Loss: {history['train_tot_loss'][-1]:.4f} | Data MSE: {history['train_data_mse'][-1]:.4f} | Phys Loss: {history['train_phys_loss'][-1]:.4f} | Val MSE: {avg_val:.6f}", flush=True)

    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), weights_path)
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "history.csv"), index=False)
    
    # Test Evaluation
    model.eval()
    all_preds, all_targets, f_indices = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if dry_run and batch_idx >= 2: break
            x, _, y_raw = [b.to(DEVICE, non_blocking=True) for b in batch[:3]]
            idx = batch[-1]
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                p_norm, *_ = model(x)
            all_preds.append((p_norm.cpu() * t_std + t_mean).numpy().flatten())
            all_targets.append(y_raw.cpu().numpy().flatten())
            f_indices.append(idx.cpu().numpy().flatten())
            
    preds = np.concatenate(all_preds)
    actual = np.concatenate(all_targets)
    test_idx = np.concatenate(f_indices)
    
    rmse = np.sqrt(np.mean((preds - actual)**2))
    corr = np.corrcoef(preds, actual)[0, 1]
    r2 = float(1.0 - (rmse**2 / (np.var(actual) + 1e-12)))
    
    start_h = full_ds.water_level[test_idx[0]]
    df_pred = pd.DataFrame({
        "Actual_DeltaH": actual, "Predicted_DeltaH": preds,
        "Actual_Level": start_h + np.cumsum(actual), "Predicted_Level": start_h + np.cumsum(preds)
    })
    df_pred.to_csv(os.path.join(save_dir, "prediction_data.csv"), index=False)
    
    metrics = {"rmse": float(rmse), "corr": float(corr), "r2": float(r2)}
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Final Test [{model_type.upper()}] -> RMSE: {rmse:.4f} m | R^2: {r2:.4f} | Corr: {corr:.4f}", flush=True)
    return metrics, model

# ── Master Execution Suite ────────────────────────────────────────────────
def run_master_pinn1_suite(dry_run=False):
    total_start = time()
    if dry_run:
        print(f"\n{'='*70}\nDRY RUN / SMOKE TEST MODE ACTIVE (1 epoch, 2 batches)\n{'='*70}", flush=True)
        
    print(f"Loading LSM mask {LSM_PATH}...", flush=True)
    lsm_raw = xr.open_dataset(LSM_PATH)
    volga_df = pd.read_csv(VOLGA_PATH).set_index('time')
    volga_df.index = pd.to_datetime(volga_df.index)

    cfg_1m = RESOLUTIONS["1m"]
    ds_1m = xr.open_dataset(cfg_1m['path'])
    lsm_aligned = lsm_raw.lsm.interp_like(ds_1m, method='nearest')
    b_mask, l_mask = lsm_aligned.values[0] > 0.5, lsm_aligned.values[0] <= 0.5
    
    pb = ds_1m.tp.where(b_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
    petb = (ds_1m.ssr.where(b_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0) / 2.5e9).clip(lower=0.001)
    pl = ds_1m.tp.where(l_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
    el = -ds_1m.e.where(l_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
    
    full_ds_1m = UnifiedMasterDataset(ds_1m, volga_df, pb, petb, pl, el, seq_len=SEQ_LEN)
    years_1m = pd.to_datetime(ds_1m.time.values).year
    tr_idx_1m = [i for i, v in enumerate(full_ds_1m.valid_indices) if years_1m[v] <= TRAIN_END_YEAR]
    te_idx_1m = [i for i, v in enumerate(full_ds_1m.valid_indices) if years_1m[v] > TRAIN_END_YEAR]
    
    train_ld_1m = DataLoader(torch.utils.data.Subset(full_ds_1m, tr_idx_1m), batch_size=16, shuffle=True, pin_memory=True)
    test_ld_1m  = DataLoader(torch.utils.data.Subset(full_ds_1m, te_idx_1m), batch_size=16, shuffle=False, pin_memory=True)

    # 1. Multi-Seed Benchmark for PINN-1
    print(f"\n{'='*70}\nEXPERIMENT 1: MULTI-SEED BENCHMARK (PINN-1 ONLY)\n{'='*70}", flush=True)
    exp1_dir = os.path.join(OUT_ROOT, ("dryrun_" if dry_run else "") + "1_multiseed", "pinn1")
    
    for res in ORDERED_RESOLUTIONS:
        cfg = RESOLUTIONS[res]
        ds = xr.open_dataset(cfg['path'])

        lsm_a = lsm_raw.lsm.interp_like(ds, method='nearest')
        bm, lm = lsm_a.values[0] > 0.5, lsm_a.values[0] <= 0.5
        
        p_b = ds.tp.where(bm).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
        pet_b = (ds.ssr.where(bm).mean(dim=['latitude', 'longitude']).to_series().fillna(0) / 2.5e9).clip(lower=0.001)
        p_l = ds.tp.where(lm).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
        e_l = -ds.e.where(lm).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
        
        fds = UnifiedMasterDataset(ds, volga_df, p_b, pet_b, p_l, e_l, seq_len=SEQ_LEN)
        yrs = pd.to_datetime(ds.time.values).year
        tr_i = [i for i, v in enumerate(fds.valid_indices) if yrs[v] <= TRAIN_END_YEAR]
        te_i = [i for i, v in enumerate(fds.valid_indices) if yrs[v] > TRAIN_END_YEAR]
        tr_ld = DataLoader(torch.utils.data.Subset(fds, tr_i), batch_size=cfg['batch_size'], shuffle=True, pin_memory=True)
        te_ld = DataLoader(torch.utils.data.Subset(fds, te_i), batch_size=cfg['batch_size'], shuffle=False, pin_memory=True)
        
        print(f"\nMultiseed PINN1 -> Res: {res}", flush=True)
        res_dir = os.path.join(exp1_dir, res)
        all_m = []
        seed_list = [11] if dry_run else SEEDS
        for s in seed_list:
            set_seed(s)
            mdl = WaterBalancePINN1v2(13, seq_len=SEQ_LEN).to(DEVICE)
            m, _ = run_standard_train_eval(mdl, tr_ld, te_ld, fds, cfg['epochs'], cfg['lr'], cfg['dt'], os.path.join(res_dir, f"seed_{s}"), "pinn1", dry_run=dry_run)
            m['seed'] = s
            all_m.append(m)
        pd.DataFrame(all_m).to_csv(os.path.join(res_dir, "multiseed_summary.csv"), index=False)

    # 2. Monte Carlo Dropout Uncertainty (PINN-1)
    print(f"\n{'='*70}\nEXPERIMENT 2: MONTE CARLO DROPOUT UNCERTAINTY BOUNDS (PINN-1)\n{'='*70}", flush=True)
    exp2_dir = os.path.join(OUT_ROOT, ("dryrun_" if dry_run else "") + "2_mc_dropout")
    set_seed(42)
    best_p1 = WaterBalancePINN1v2(13, seq_len=SEQ_LEN).to(DEVICE)
    _, best_p1 = run_standard_train_eval(best_p1, train_ld_1m, test_ld_1m, full_ds_1m, 250, 3e-4, cfg_1m['dt'], exp2_dir, "pinn1", dry_run=dry_run)
    
    best_p1.train() # Enable dropout
    mc_preds, mc_preds_raw = [], []
    with torch.no_grad():
        n_passes = 2 if dry_run else 50
        for _ in range(n_passes):
            pass_preds, pass_preds_raw = [], []
            for batch_idx, batch in enumerate(test_ld_1m):
                if dry_run and batch_idx >= 2: break
                x, _, y_raw = [b.to(DEVICE, non_blocking=True) for b in batch[:3]]
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                    p_norm, *_ = best_p1(x)
                pass_preds.append((p_norm.cpu() * full_ds_1m.stats['t_std'] + full_ds_1m.stats['t_mean']).numpy().flatten())
                pass_preds_raw.append(y_raw.cpu().numpy().flatten())
            mc_preds.append(np.concatenate(pass_preds))
            if not mc_preds_raw: mc_preds_raw.append(np.concatenate(pass_preds_raw))
            
    mc_np = np.array(mc_preds)
    mean_preds = mc_np.mean(axis=0)
    std_preds  = mc_np.std(axis=0)
    actual_vals = mc_preds_raw[0]
    
    df_mc = pd.DataFrame({
        "Actual_DeltaH": actual_vals, "Mean_DeltaH": mean_preds, "Std_DeltaH": std_preds,
        "Lower_95": mean_preds - 1.96*std_preds, "Upper_95": mean_preds + 1.96*std_preds,
        "Actual_H": full_ds_1m.water_level[te_idx_1m[0]] + np.cumsum(actual_vals),
        "Mean_H":   full_ds_1m.water_level[te_idx_1m[0]] + np.cumsum(mean_preds)
    })
    df_mc.to_csv(os.path.join(exp2_dir, "mc_uncertainty_bands.csv"), index=False)

    # 3. Sensitivity Analysis on lambda_phys (PINN-1)
    print(f"\n{'='*70}\nEXPERIMENT 3: SENSITIVITY ANALYSIS (lambda = 0.1, 0.5, 1.0, 5.0)\n{'='*70}", flush=True)
    exp3_dir = os.path.join(OUT_ROOT, ("dryrun_" if dry_run else "") + "3_sensitivity")
    lambdas = [0.1] if dry_run else [0.1, 0.5, 1.0, 5.0]
    sens_metrics = []
    
    for l_val in lambdas:
        print(f"\nRunning Sensitivity -> lambda_phys = {l_val}", flush=True)
        set_seed(42)
        mdl = WaterBalancePINN1v2(13, seq_len=SEQ_LEN).to(DEVICE)
        m, _ = run_standard_train_eval(mdl, train_ld_1m, test_ld_1m, full_ds_1m, 250, 3e-4, cfg_1m['dt'], os.path.join(exp3_dir, f"lambda_{l_val}"), "pinn1", lambda_phys=l_val, dry_run=dry_run)
        m['lambda_phys'] = l_val
        sens_metrics.append(m)
    pd.DataFrame(sens_metrics).to_csv(os.path.join(exp3_dir, "sensitivity_summary.csv"), index=False)

    # 4. Clean Ablation Study (PINN-1 No Residual Basin Runoff)
    print(f"\n{'='*70}\nEXPERIMENT 4: CLEAN ABLATION (PINN-1 Without Residual Basin Inflow)\n{'='*70}", flush=True)
    exp4_dir = os.path.join(OUT_ROOT, ("dryrun_" if dry_run else "") + "4_ablation_no_residual")
    set_seed(42)
    mdl = WaterBalancePINN1v2(13, seq_len=SEQ_LEN).to(DEVICE)
    run_standard_train_eval(mdl, train_ld_1m, test_ld_1m, full_ds_1m, 250, 3e-4, cfg_1m['dt'], exp4_dir, "pinn1", no_residual=True, dry_run=dry_run)

    # 5. Dynamic Lake Surface Area A(H)
    print(f"\n{'='*70}\nEXPERIMENT 5: DYNAMIC LAKE SURFACE AREA A(H)\n{'='*70}", flush=True)
    exp5_dir = os.path.join(OUT_ROOT, ("dryrun_" if dry_run else "") + "5_dynamic_area")
    set_seed(42)
    mdl = DynamicAreaWaterBalancePINN1v2(13, seq_len=SEQ_LEN).to(DEVICE)
    run_standard_train_eval(mdl, train_ld_1m, test_ld_1m, full_ds_1m, 250, 3e-4, cfg_1m['dt'], exp5_dir, "dynamic", dry_run=dry_run)

    print(f"\n{'='*70}\nRETRAINING OF PINN1 SATELLITE SUITE COMPLETED in {timedelta(seconds=time() - total_start)}!\n{'='*70}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain PINN-1 Only Suite")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch smoke test")
    args, _ = parser.parse_known_args()
    run_master_pinn1_suite(dry_run=args.dry_run)
