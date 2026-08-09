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

# Ensure imports work
base_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model"))
sys.path.append(os.path.join(base_model_dir, "baseline_lstm"))
from baseline_model import DataOnlyBaselineLSTM

sys.path.append(os.path.join(base_model_dir, "interval_comparison"))
from pinn2_budyko import BudykoPINN2_v21

sys.path.append(os.path.join(base_model_dir, "PINN1_Water_balance"))
from pinn1_water_balance import WaterBalancePINN1, _softplus_inv

sys.path.append(os.path.join(base_model_dir, "pinn3"))
from pinn3_energy import EnergyPINN3, saturation_slope, psychrometric_constant, vapour_pressure

# ── Configuration & Reference Hyperparameters ─────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 6        
WEIGHT_DECAY = 0.05

OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "master_suite", "6_walk_forward_thermodynamic"))

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_monthly.nc"))
VOLGA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "volga_discharge.csv"))
LSM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lsm.area-subset.47.5.54.5.36.5.45.5.nc"))
A_LAKE_M2_REF = 3.71e11
DT_MONTHLY = 30.42 * 86400

MODELS_TO_TEST = ["pinn2", "pinn3", "hybrid"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# ── Unified Pre-Tensorized Data Loading Helper ────────────────────────────
class UnifiedMasterDataset(Dataset):
    def __init__(self, ds, volga_data, p_b_s, pet_b_s, p_l_s, e_l_s, seq_len=6, train_end_year=2017):
        self.times = ds.time.values
        self.seq_len = seq_len
        self.vars = ['tp', 't2m', 'ssr', 'str', 'sp', 'e', 'd2m', 'u10', 'v10', 'Qs_acc', 'Qsb_acc', 'SWE_inst']
        
        volga_interp = volga_data.reindex(pd.to_datetime(self.times), method='nearest').fillna(8000.0)
        volga_vals_np = volga_interp['volga_q'].values.astype(np.float32)
        
        p_b_np, pet_b_np = p_b_s.values.astype(np.float32), pet_b_s.values.astype(np.float32)
        p_l_np, e_l_np = p_l_s.values.astype(np.float32), e_l_s.values.astype(np.float32)

        # Penman PINN-3 specific variables over the lake
        lsm_raw = xr.open_dataset(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lsm.area-subset.47.5.54.5.36.5.45.5.nc")))
        lsm_aligned = lsm_raw.lsm.interp_like(ds, method='nearest')
        lake_mask = lsm_aligned.values[0] <= 0.5
        
        t_lk = ds.t2m.where(lake_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(273.15).values.astype(np.float32) - 273.15
        td_lk = ds.d2m.where(lake_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(273.15).values.astype(np.float32) - 273.15
        sp_lk = ds.sp.where(lake_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(101325.0).values.astype(np.float32) / 1000.0
        u10_lk = ds.u10.where(lake_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0.0).values.astype(np.float32)
        v10_lk = ds.v10.where(lake_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0.0).values.astype(np.float32)
        w_lk = np.sqrt(u10_lk**2 + v10_lk**2)
        
        ssr_lk = ds.ssr.where(lake_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0.0).values.astype(np.float32)
        str_lk = ds.str.where(lake_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0.0).values.astype(np.float32)
        rn_lk = np.clip((ssr_lk + str_lk) / 2.45e9, a_min=0.0001, a_max=None)
        
        p3_vars_np = np.stack([t_lk, td_lk, rn_lk, w_lk, sp_lk], axis=1)

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
        self.p3_vars = torch.from_numpy(p3_vars_np).float()

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
            self.p3_vars[f_idx],
            f_idx
        )

class MasterTripleHybridPINN(nn.Module):
    """Master Triple-Hybrid PINN combining Volumetric Mass Balance + Budyko Runoff + Penman Energy Balance."""
    def __init__(self, in_channels=13, hidden_dim=64, seq_len=6):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.lstm = nn.LSTM(1024, hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Dropout(0.5), nn.Linear(64, 8)
        )
        # 4 task loss weights: Data loss, Mass balance loss, Budyko loss, Penman loss
        self.log_vars = nn.Parameter(torch.tensor([0.0, -1.0, -1.0, -1.0]))
        with torch.no_grad():
            b = self.regressor[-1].bias
            b[0], b[1], b[2], b[3] = 0.0, _softplus_inv(1.0), _softplus_inv(1.0), _softplus_inv(0.3)
            b[4], b[5] = _softplus_inv(1.5), _softplus_inv(0.1)
            b[6], b[7] = _softplus_inv(0.26), _softplus_inv(1.0) # alpha_PT offset (+1.0), C_wind

    def forward(self, x):
        bs, sl, c, h, w = x.shape
        feat = self.encoder(x.view(bs * sl, c, h, w)).view(bs, sl, -1)
        lstm_out, _ = self.lstm(feat)
        out = self.regressor(lstm_out[:, -1, :])
        
        dh_pred = out[:, 0:1]
        a_E, b_v, g_o = F.softplus(out[:, 1:2]), F.softplus(out[:, 2:3]), F.softplus(out[:, 3:4])
        n_t = F.softplus(out[:, 4:5]) + 1.0
        scale_t = F.softplus(out[:, 5:6])
        alpha_PT = F.softplus(out[:, 6:7]) + 1.0
        C_wind = F.softplus(out[:, 7:8])
        return dh_pred, a_E, b_v, g_o, n_t, scale_t, alpha_PT, C_wind

    def hybrid_physics_loss(self, dh_pred, a_E, b_v, g_o, n_t, scale_t, alpha_PT, C_wind,
                            P_lake, E_lake, Q_volga, P_basin, PET_basin,
                            T_lake, Td_lake, Rn_lake_eq, U_lake, P_kpa_lake, dt_sec):
        # 1. PINN-1 Volumetric Mass Balance
        lake_mass = P_lake - a_E * E_lake
        Q_in_m = (b_v * Q_volga * dt_sec) / A_LAKE_M2_REF
        runoff_mass = g_o * (P_basin - PET_basin).clamp(min=0.0)
        dh_mass = (lake_mass + Q_in_m + runoff_mass).view_as(dh_pred)
        loss_mass = F.mse_loss(dh_pred, dh_mass)
        
        # 2. PINN-2 Budyko Hydroclimatic Runoff Partitioning
        phi = (PET_basin / (P_basin + 1e-8)).clamp(0.01, 20.0)
        phi_n = torch.pow(phi, n_t)
        et_ratio = phi / torch.pow(1.0 + phi_n, 1.0 / n_t)
        runoff_budyko = (P_basin - et_ratio * P_basin) * scale_t
        dh_budyko = (runoff_budyko + lake_mass).view_as(dh_pred)
        loss_budyko = F.mse_loss(dh_pred, dh_budyko)
        
        # 3. PINN-3 Penman / Priestley-Taylor Aerodynamic Energy Balance
        delta_T = saturation_slope(T_lake)
        gamma_p = psychrometric_constant(P_kpa_lake)
        es = vapour_pressure(T_lake)
        ea = vapour_pressure(Td_lake)
        VPD = (es - ea).clamp(min=0.0)
        denom = (delta_T + gamma_p).clamp(min=1e-6)
        
        E_rad = alpha_PT * (delta_T / denom) * Rn_lake_eq
        f_U = 1.0 + 0.536 * U_lake
        E_aero = C_wind * (gamma_p / denom) * f_U * VPD * 1e-3
        E_pen = E_rad + E_aero
        dh_penman = (P_lake - E_pen + runoff_mass).view_as(dh_pred)
        loss_penman = F.mse_loss(dh_pred, dh_penman)
        
        return loss_mass, loss_budyko, loss_penman

# ── General Training & Evaluation Engine ──────────────────────────────────
def run_standard_train_eval(model, train_loader, test_loader, full_ds, epochs, lr, dt_val, save_dir, model_type="pinn2", lambda_phys=100.0, dry_run=False):
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
            
            x, y_norm, y_raw, pb, petb, pl, el, qv, lh, p3_vars = [b.to(DEVICE, non_blocking=True) for b in batch[:-1]]
            optimizer.zero_grad()
            
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                if model_type == "pinn2":
                    pred_dh, n_t, scale_t = model(x)
                    loss_d = F.mse_loss(pred_dh, y_norm)
                    loss_p = model.budyko_loss(pred_dh * t_std + t_mean, n_t, scale_t, pb, petb, pl, el)
                    total_loss = loss_d + lambda_phys * loss_p
                elif model_type == "pinn3":
                    pred_dh, alpha_PT, C_wind, gamma_runoff = model(x)
                    loss_d = F.mse_loss(pred_dh, y_norm)
                    t_lk, td_lk, rn_lk, u_lk, sp_lk = p3_vars[:, 0:1], p3_vars[:, 1:2], p3_vars[:, 2:3], p3_vars[:, 3:4], p3_vars[:, 4:5]
                    loss_p, _ = model.penman_loss(
                        pred_dh * t_std + t_mean, alpha_PT, C_wind, gamma_runoff,
                        t_lk, td_lk, rn_lk, u_lk, sp_lk, pl, pb, petb
                    )
                    total_loss = loss_d + lambda_phys * loss_p
                elif model_type == "hybrid":
                    pred_dh, a_E, b_v, g_o, n_t, s_t, alpha_PT, C_wind = model(x)
                    loss_d = F.mse_loss(pred_dh, y_norm)
                    t_lk, td_lk, rn_lk, u_lk, sp_lk = p3_vars[:, 0:1], p3_vars[:, 1:2], p3_vars[:, 2:3], p3_vars[:, 3:4], p3_vars[:, 4:5]
                    l_mass, l_bud, l_pen = model.hybrid_physics_loss(
                        pred_dh * t_std + t_mean, a_E, b_v, g_o, n_t, s_t, alpha_PT, C_wind,
                        pl, el, qv, pb, petb, t_lk, td_lk, rn_lk, u_lk, sp_lk, dt_val
                    )
                    loss_p = (l_mass + l_bud + l_pen) / 3.0
                    prec_d, prec_m, prec_b, prec_p = torch.exp(-model.log_vars[0]), torch.exp(-model.log_vars[1]), torch.exp(-model.log_vars[2]), torch.exp(-model.log_vars[3])
                    total_loss = 0.5 * (prec_d * loss_d + prec_m * l_mass + prec_b * l_bud + prec_p * l_pen + model.log_vars.sum())

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
                    if model_type == "pinn3": pred_dh, *_ = model(x)
                    elif model_type == "hybrid": pred_dh, *_ = model(x)
                    else: pred_dh, *_ = model(x)
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
            if model_type == "pinn3": p_norm, *_ = model(x)
            elif model_type == "hybrid": p_norm, *_ = model(x)
            else: p_norm, *_ = model(x)
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
        
    print(f"🏁 Final Test [{model_type.upper()}] -> RMSE: {rmse:.4f} m | R^2: {r2:.4f} | Corr: {corr:.4f}", flush=True)
    return metrics, model

def run_walk_forward_thermodynamic(dry_run=False):
    total_start = time()
    if dry_run:
        print(f"\n{'='*70}\n⚠️ DRY RUN / SMOKE TEST MODE ACTIVE (1 epoch, 2 batches)\n{'='*70}", flush=True)
        
    print(f"Loading LSM mask {LSM_PATH}...", flush=True)
    lsm_raw = xr.open_dataset(LSM_PATH)
    volga_df = pd.read_csv(VOLGA_PATH).set_index('time')
    volga_df.index = pd.to_datetime(volga_df.index)

    ds_1m = xr.open_dataset(DATA_PATH)
    lsm_aligned = lsm_raw.lsm.interp_like(ds_1m, method='nearest')
    b_mask, l_mask = lsm_aligned.values[0] > 0.5, lsm_aligned.values[0] <= 0.5
    lsm_raw.close()
    
    pb = ds_1m.tp.where(b_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
    petb = (ds_1m.ssr.where(b_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0) / 2.5e9).clip(lower=0.001)
    pl = ds_1m.tp.where(l_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
    el = -ds_1m.e.where(l_mask).mean(dim=['latitude', 'longitude']).to_series().fillna(0)
    
    full_ds_1m = UnifiedMasterDataset(ds_1m, volga_df, pb, petb, pl, el, seq_len=SEQ_LEN)
    years_1m = pd.to_datetime(ds_1m.time.values).year

    folds = [{"name": "Fold1_2010", "tr_end": 2010, "te_end": 2015}] if dry_run else [
        {"name": "Fold1_2010", "tr_end": 2010, "te_end": 2015},
        {"name": "Fold2_2015", "tr_end": 2015, "te_end": 2020},
        {"name": "Fold3_2020", "tr_end": 2020, "te_end": 2025}
    ]
    
    all_results = []
    out_dir = os.path.join(OUT_ROOT, "dryrun" if dry_run else "full")
    os.makedirs(out_dir, exist_ok=True)
    
    for f in folds:
        print(f"\n{'='*70}\n🚀 RUNNING WALK-FORWARD FOLD: {f['name']} (Train <= {f['tr_end']}, Test <= {f['te_end']})\n{'='*70}", flush=True)
        fold_dir = os.path.join(out_dir, f['name'])
        
        tr_i = [i for i, v in enumerate(full_ds_1m.valid_indices) if years_1m[v] <= f['tr_end']]
        te_i = [i for i, v in enumerate(full_ds_1m.valid_indices) if f['tr_end'] < years_1m[v] <= f['te_end']]
        tr_ld = DataLoader(torch.utils.data.Subset(full_ds_1m, tr_i), batch_size=16, shuffle=True, pin_memory=True)
        te_ld = DataLoader(torch.utils.data.Subset(full_ds_1m, te_i), batch_size=16, shuffle=False, pin_memory=True)
        
        for mod_name in MODELS_TO_TEST:
            print(f"\n⚡ Evaluating Model: {mod_name.upper()} on {f['name']}", flush=True)
            set_seed(42)
            if mod_name == "pinn2": mdl = BudykoPINN2_v21(13, seq_len=SEQ_LEN).to(DEVICE)
            elif mod_name == "pinn3": mdl = EnergyPINN3(13, seq_len=SEQ_LEN).to(DEVICE)
            elif mod_name == "hybrid": mdl = MasterTripleHybridPINN(13, seq_len=SEQ_LEN).to(DEVICE)
            
            m, _ = run_standard_train_eval(mdl, tr_ld, te_ld, full_ds_1m, 200, 3e-4, DT_MONTHLY, os.path.join(fold_dir, mod_name), mod_name, dry_run=dry_run)
            m['fold'] = f['name']
            m['model'] = mod_name
            all_results.append(m)
            
        pd.DataFrame(all_results).to_csv(os.path.join(out_dir, "walk_forward_thermodynamic_summary.csv"), index=False)
        
    print(f"\n{'='*70}\n🎉 WALK-FORWARD EVALUATION (PINN-2 Budyko vs PINN-3 Penman vs Triple-Hybrid) COMPLETED in {timedelta(seconds=time() - total_start)}!\n{'='*70}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-Forward Thermodynamic Suite")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch smoke test")
    args, _ = parser.parse_known_args()
    run_walk_forward_thermodynamic(dry_run=args.dry_run)
