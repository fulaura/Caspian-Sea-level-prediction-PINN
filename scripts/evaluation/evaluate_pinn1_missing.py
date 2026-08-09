import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(os.path.join(os.getcwd(), "paper_v2", "model", "PINN1_Water_balance"))
from pinn1_water_balance import WaterBalancePINN1

DEVICE = torch.device("cpu")
SEQ_LEN = 30
OUT_ROOT = "paper_v2/results/pinn1/interval_comparison"
DS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_daily.nc"))
VOLGA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "volga_discharge.csv"))

def load_robust_pinn1(weights_path, seq_len):
    model = WaterBalancePINN1(in_channels=13, seq_len=seq_len).to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    
    is_4head = False
    for k, v in state_dict.items():
        if "output_layer" in k or "regressor.3" in k:
            if v.shape[0] == 4:
                is_4head = True
                break
                
    if is_4head:
        model.regressor.output_layer = torch.nn.Linear(64, 4).to(DEVICE)
        head_type = "4-Head Checkpoint (Mapped)"
    else:
        head_type = "6-Head Checkpoint (Uncertainty)"
        
    mapping = {
        "encoder.0": "encoder.conv1",
        "encoder.3": "encoder.conv2",
        "regressor.0": "regressor.fc1",
        "regressor.3": "regressor.output_layer"
    }
    new_sd = {}
    for k, v in state_dict.items():
        matched = False
        for old, new in mapping.items():
            if k.startswith(old):
                new_sd[k.replace(old, new)] = v
                matched = True
                break
        if not matched:
            new_sd[k] = v
            
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model, head_type

def evaluate_1d_fast():
    print("=== Superfast Memory-Optimized PINN-1 1d Evaluation on CPU ===")
    res_dir = os.path.join(OUT_ROOT, "1d")
    weights_path = os.path.join(res_dir, "best_model.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(res_dir, "model_weights.pth")

    model, head_type = load_robust_pinn1(weights_path, SEQ_LEN)
    print(f">>> Checkpoint Loaded: {head_type}")

    print(f"Opening dataset {DS_PATH}...")
    ds = xr.open_dataset(DS_PATH)
    volga_df = pd.read_csv(VOLGA_PATH)
    volga_df['time'] = pd.to_datetime(volga_df['time'])
    volga_df = volga_df.set_index('time')
    
    water_level = ds.water_level.to_series().ffill().fillna(0)
    delta_h = water_level.diff().fillna(0)
    
    print("Calculating training stats...")
    ds_train = ds.isel(time=slice(0, 500))
    vars_list = ['tp', 't2m', 'ssr', 'str', 'sp', 'e', 'd2m', 'u10', 'v10', 'Qs_acc', 'Qsb_acc', 'SWE_inst']
    
    train_volga = volga_df.reindex(pd.to_datetime(ds_train.time.values), method='nearest').fillna(8000.0)['volga_q'].values
    train_list = [ds_train[v].values.astype(np.float32) for v in vars_list]
    v_spatial_tr = train_volga[:, None, None] * np.ones_like(train_list[0][0])
    data_tr_stack = np.stack(train_list + [v_spatial_tr], axis=1)
    
    stats = {
        'mean': np.nanmean(data_tr_stack, axis=(0, 2, 3), keepdims=True),
        'std': np.nanstd(data_tr_stack, axis=(0, 2, 3), keepdims=True),
        'target_mean': float(np.mean(delta_h.iloc[:500])),
        'target_std': float(np.std(delta_h.iloc[:500]))
    }
    
    print("Loading test slice variable-wise into memory...")
    ds_eval = ds.isel(time=slice(-3500, None))
    eval_volga = volga_df.reindex(pd.to_datetime(ds_eval.time.values), method='nearest').fillna(8000.0)['volga_q'].values
    eval_list = [ds_eval[v].values.astype(np.float32) for v in vars_list]
    v_spatial_ev = eval_volga[:, None, None] * np.ones_like(eval_list[0][0])
    data_ev_stack = np.stack(eval_list + [v_spatial_ev], axis=1)
    
    eval_times = pd.to_datetime(ds_eval.time.values)
    eval_delta_h = delta_h.reindex(eval_times).values
    eval_water_level = water_level.reindex(eval_times).values
    
    test_start_date = pd.Timestamp('2018-01-01')
    first_test_idx = np.where(eval_times >= test_start_date)[0][0]
    n_test_steps = len(eval_times) - first_test_idx
    
    all_preds = []
    all_targets = []
    
    print(f"Running sequential inference over {n_test_steps} steps (Zero memory overhead)...")
    model.eval()
    with torch.no_grad():
        for i in range(n_test_steps):
            mem_idx = first_test_idx + i
            x = data_ev_stack[mem_idx - SEQ_LEN + 1 : mem_idx + 1]
            x = np.nan_to_num((x - stats['mean']) / (stats['std'] + 1e-8))
            x_tensor = torch.FloatTensor(x).unsqueeze(0).to(DEVICE)
            
            p_norm = model(x_tensor)[0].item()
            p_raw = p_norm * stats['target_std'] + stats['target_mean']
            all_preds.append(p_raw)
            all_targets.append(eval_delta_h[mem_idx])
            if (i + 1) % 500 == 0:
                print(f"  Step {i+1}/{n_test_steps} done.")
            
    preds = np.array(all_preds)
    actual = np.array(all_targets)
    
    rmse = np.sqrt(np.mean((preds - actual)**2))
    corr = np.corrcoef(preds, actual)[0, 1]
    var_actual = np.mean((actual - np.mean(actual))**2)
    r2 = float(1.0 - (rmse**2 / (var_actual + 1e-12)))
    
    start_level = eval_water_level[first_test_idx]
    actual_abs = start_level + np.cumsum(actual)
    pred_abs = start_level + np.cumsum(preds)
    
    df_out = pd.DataFrame({"Actual_DeltaH": actual, "Predicted_DeltaH": preds, "Actual_H": actual_abs, "Predicted_H": pred_abs})
    df_out.to_csv(os.path.join(res_dir, "prediction_data.csv"), index=False)
    
    metrics = {"rmse": float(rmse), "corr": float(corr), "r2": float(r2), "res": "1d"}
    with open(os.path.join(res_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[1d] Metrics -> RMSE: {rmse:.4f}, Corr: {corr:.4f}, R^2: {r2:.4f}")
    
    master_rows = []
    for r in ["1m", "10d"]:
        m_path = os.path.join(OUT_ROOT, r, "metrics.json")
        if os.path.exists(m_path):
            with open(m_path, "r") as f:
                master_rows.append(json.load(f))
                
    summary_df = pd.DataFrame(master_rows)
    summary_df.to_csv(os.path.join(OUT_ROOT, "comparison_summary.csv"), index=False)
    print("\n=== Final Master Summary Table for PINN-1 ===")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_1d_fast()
