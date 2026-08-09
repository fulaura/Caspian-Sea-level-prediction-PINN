import os
import sys
import torch
import xarray as xr
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add model path
sys.path.append(os.path.join(os.getcwd(), "antigravity", "model", "interval_comparison"))
from pinn2_budyko import BudykoPINN2_v21

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 6
WEIGHTS_PATH = "antigravity/results/pinn2/interval_comparison/1d/model_weights.pth"
DS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_daily.nc"))
VOLGA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "volga_discharge.csv"))
LSM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'lsm.area-subset.47.5.54.5.36.5.45.5.nc'))
OUT_DIR = "antigravity/results/pinn2/interval_comparison/1d"
OUT_PATH = os.path.join(OUT_DIR, "full_test_predictions_2018_2026.csv")

class BudykoInferenceDataset(Dataset):
    def __init__(self, ds, volga_data, delta_h, stats, indices, seq_len=6):
        self.ds = ds
        self.volga_data = volga_data
        self.delta_h = delta_h.values
        self.stats = stats
        self.indices = indices
        self.seq_len = seq_len
        self.vars = ['tp', 't2m', 'ssr', 'str', 'sp', 'e', 'd2m', 'u10', 'v10', 'Qs_acc', 'Qsb_acc', 'SWE_inst']
        self.volga_all = volga_data.reindex(pd.to_datetime(ds.time.values), method='nearest').fillna(8000.0)['volga_q'].values

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        f_idx = self.indices[idx]
        
        # Lazy slice xarray to avoid OOM
        slice_ds = self.ds.isel(time=slice(f_idx - self.seq_len + 1, f_idx + 1))
        x_list = []
        for v in self.vars:
            x_list.append(slice_ds[v].values)
            
        v_slice = self.volga_all[f_idx - self.seq_len + 1 : f_idx + 1]
        volga_spatial = v_slice[:, None, None] * np.ones_like(x_list[0][0])
        x = np.stack(x_list + [volga_spatial], axis=1)
        
        x = np.nan_to_num((x - self.stats['mean']) / (self.stats['std'] + 1e-8))
        
        return torch.FloatTensor(x), torch.FloatTensor([self.delta_h[f_idx]])

def run_inference():
    print(f"Loading model weights from {WEIGHTS_PATH}...")
    model = BudykoPINN2_v21(in_channels=13, seq_len=SEQ_LEN).to(DEVICE)
    
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    # Map old Sequential keys to new named keys
    mapping = {
        "encoder.0": "encoder.conv1",
        "encoder.3": "encoder.conv2",
        "regressor.0": "regressor.fc1",
        "regressor.3": "regressor.output_layer"
    }
    new_state_dict = {}
    for k, v in state_dict.items():
        matched = False
        for old, new in mapping.items():
            if k.startswith(old):
                new_key = k.replace(old, new)
                new_state_dict[new_key] = v
                matched = True
                break
        if not matched:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()

    print(f"Opening dataset {DS_PATH}...")
    ds = xr.open_dataset(DS_PATH)
    volga_df = pd.read_csv(VOLGA_PATH)
    volga_df['time'] = pd.to_datetime(volga_df['time'])
    volga_df = volga_df.set_index('time')
    
    water_level = ds.water_level.to_series().ffill().fillna(0)
    delta_h = water_level.diff().fillna(0)

    # 1. Re-calculate Training Stats (on last 500 days as per training script)
    print("Re-calculating training statistics for normalization...")
    ds_train = ds.isel(time=slice(-500, None))
    vars = ['tp', 't2m', 'ssr', 'str', 'sp', 'e', 'd2m', 'u10', 'v10', 'Qs_acc', 'Qsb_acc', 'SWE_inst']
    volga_train = volga_df.reindex(pd.to_datetime(ds_train.time.values), method='nearest').fillna(8000.0)['volga_q'].values
    
    train_data_list = []
    for v in vars:
        train_data_list.append(ds_train[v].values)
    v_spatial = volga_train[:, None, None] * np.ones_like(train_data_list[0][0])
    data_train_stack = np.stack(train_data_list + [v_spatial], axis=1)
    
    stats = {
        'mean': np.nanmean(data_train_stack, axis=(0, 2, 3), keepdims=True),
        'std': np.nanstd(data_train_stack, axis=(0, 2, 3), keepdims=True),
        'target_mean': float(np.mean(delta_h.iloc[-500:])),
        'target_std': float(np.std(delta_h.iloc[-500:]))
    }

    # 2. Identify Test Period (2018 onwards)
    all_times = pd.to_datetime(ds.time.values)
    # We need SEQ_LEN-1 context before the first test step
    test_start_date = pd.Timestamp('2018-01-01')
    first_test_idx = np.where(all_times >= test_start_date)[0][0]
    
    # Pre-load gridded data for test period + buffer into memory to speed up
    print("Pre-loading test set data into memory variable-wise...")
    start_buffer_idx = first_test_idx - SEQ_LEN + 1
    
    test_data_list = []
    for v in vars:
        print(f"  Loading {v}...")
        val = ds[v].isel(time=slice(start_buffer_idx, None)).values
        test_data_list.append(val)
    
    ds_test_times = ds.time.isel(time=slice(start_buffer_idx, None)).values
    test_volga = volga_df.reindex(pd.to_datetime(ds_test_times), method='nearest').fillna(8000.0)['volga_q'].values
    v_spatial = test_volga[:, None, None] * np.ones_like(test_data_list[0][0])
    test_data_memory = np.stack(test_data_list + [v_spatial], axis=1)
    
    # Map back the indices relative to test_data_memory
    # test_indices[0] in original -> idx SEQ_LEN-1 in test_data_memory
    n_test_steps = len(all_times) - first_test_idx
    
    all_preds = []
    all_targets = []
    
    print(f"Starting inference for {n_test_steps} steps...")
    for i in tqdm(range(n_test_steps)):
        idx_in_mem = i + SEQ_LEN - 1
        x = test_data_memory[idx_in_mem - SEQ_LEN + 1 : idx_in_mem + 1]
        x = np.nan_to_num((x - stats['mean']) / (stats['std'] + 1e-8))
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            p_norm, _, _ = model(x_tensor)
            p_raw = p_norm.item() * stats['target_std'] + stats['target_mean']
            all_preds.append(p_raw)
            all_targets.append(delta_h.iloc[first_test_idx + i])

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    # 3. Save results
    print(f"Saving results to {OUT_PATH}...")
    df_res = pd.DataFrame({
        'Date': all_times[first_test_idx : first_test_idx + n_test_steps],
        'Actual_DeltaH': targets,
        'Predicted_DeltaH': preds
    })
    
    # Calculate absolute levels
    start_level = water_level.iloc[first_test_idx]
    df_res['Actual_Level'] = water_level.iloc[first_test_idx : first_test_idx + n_test_steps].values
    df_res['Predicted_Level'] = start_level + df_res['Predicted_DeltaH'].cumsum()
    
    df_res.to_csv(OUT_PATH, index=False)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((preds - targets)**2))
    corr = np.corrcoef(preds, targets)[0, 1]
    print(f"Full Test Set Metrics (2018-2026): RMSE={rmse:.6f}, Correlation={corr:.4f}")

    # 4. Plotting
    import matplotlib.pyplot as plt
    print("Generating plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Delta H
    ax1.plot(df_res['Date'], df_res['Actual_DeltaH'], label='Actual Daily Delta H', color='skyblue', alpha=0.5)
    ax1.plot(df_res['Date'], df_res['Predicted_DeltaH'], label='Predicted Daily Delta H', color='red', alpha=0.8)
    ax1.set_title(f"PINN-2 Daily Delta H Prediction (2018-2026)\nRMSE: {rmse:.4f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Absolute Level
    ax2.plot(df_res['Date'], df_res['Actual_Level'], label='Actual Water Level', color='green', linewidth=2)
    ax2.plot(df_res['Date'], df_res['Predicted_Level'], label='PINN-2 Reconstructed Level', color='orange', linestyle='--', linewidth=2)
    ax2.set_title("Absolute Water Level Reconstruction (Cumulative)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "full_test_plots.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    run_inference()
