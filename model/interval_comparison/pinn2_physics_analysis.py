import os
import sys
import torch
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "antigravity", "model", "interval_comparison"))
from pinn2_budyko import BudykoPINN2_v21

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 6

RESOLUTIONS = {
    "1m": {"nc": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_monthly.nc")), "dir": "antigravity/results/pinn2/interval_comparison/1m"},
    "10d": {"nc": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_10day.nc")), "dir": "antigravity/results/pinn2/interval_comparison/10d"},
    "1d": {"nc": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "merged_grid_dataset_daily.nc")), "dir": "antigravity/results/pinn2/interval_comparison/1d"}
}

VOLGA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "volga_discharge.csv"))
vars_list = ['tp', 't2m', 'ssr', 'str', 'sp', 'e', 'd2m', 'u10', 'v10', 'Qs_acc', 'Qsb_acc', 'SWE_inst']

def load_model_robustly(weights_path):
    model = BudykoPINN2_v21(in_channels=13, seq_len=SEQ_LEN).to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    
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
    return model

def analyze_all():
    print("=== Starting PINN-2 Physics & Uncertainty Analysis ===")
    
    volga_df = pd.read_csv(VOLGA_PATH)
    volga_df['time'] = pd.to_datetime(volga_df['time'])
    volga_df = volga_df.set_index('time')
    
    uncertainty_results = []
    physics_trajectories = {}
    
    for res_name, paths in RESOLUTIONS.items():
        weights_path = os.path.join(paths['dir'], "model_weights.pth")
        if not os.path.exists(weights_path):
            print(f"Skipping {res_name}: weights not found at {weights_path}")
            continue
            
        print(f"\n>>> Analyzing {res_name} resolution...")
        model = load_model_robustly(weights_path)
        
        # 1. Uncertainty Weighting Analysis
        log_var_data = model.log_vars[0].item()
        log_var_phys = model.log_vars[1].item()
        prec_data = np.exp(-log_var_data)
        prec_phys = np.exp(-log_var_phys)
        total_prec = prec_data + prec_phys
        
        phys_share = (prec_phys / total_prec) * 100.0
        data_share = (prec_data / total_prec) * 100.0
        
        print(f"  Converged Log-Variances: Data={log_var_data:.4f}, Phys={log_var_phys:.4f}")
        print(f"  Effective Precision (1/sigma^2): Data={prec_data:.4f} ({data_share:.1f}%), Phys={prec_phys:.4f} ({phys_share:.1f}%)")
        
        uncertainty_results.append({
            "Resolution": res_name,
            "Log_Var_Data": log_var_data,
            "Log_Var_Phys": log_var_phys,
            "Precision_Data": prec_data,
            "Precision_Phys": prec_phys,
            "Data_Weight_%": data_share,
            "Phys_Weight_%": phys_share
        })
        
        # 2. Extract Learned Budyko Trajectories over Test Period
        ds_path = paths['nc']
        if res_name == "1d":
            ds = xr.open_dataset(ds_path)
            all_times = pd.to_datetime(ds.time.values)
            first_test_idx = np.where(all_times >= pd.Timestamp('2018-01-01'))[0][0]
            # Take a 1000-day representative slice of test period for 1d to keep memory light
            start_buffer_idx = first_test_idx - SEQ_LEN + 1
            sub_ds = ds.isel(time=slice(start_buffer_idx, start_buffer_idx + 1000))
        else:
            ds = xr.open_dataset(ds_path)
            all_times = pd.to_datetime(ds.time.values)
            first_test_idx = np.where(all_times >= pd.Timestamp('2018-01-01'))[0][0]
            start_buffer_idx = first_test_idx - SEQ_LEN + 1
            sub_ds = ds.isel(time=slice(start_buffer_idx, None))
            
        sub_times = sub_ds.time.values
        test_dates = pd.to_datetime(sub_times)[SEQ_LEN-1:]

        
        # Calculate stats from whole DS or sample for normalization
        sample_ds = ds if res_name != "1d" else ds.isel(time=slice(0, 500))
        sample_data = []
        for v in vars_list: sample_data.append(sample_ds[v].values)
        sample_volga = volga_df.reindex(pd.to_datetime(sample_ds.time.values), method='nearest').fillna(8000.0)['volga_q'].values
        sample_data.append(sample_volga[:, None, None] * np.ones_like(sample_data[0][0]))
        sample_stack = np.stack(sample_data, axis=1)
        mean_stat = np.nanmean(sample_stack, axis=(0, 2, 3), keepdims=True)
        std_stat = np.nanstd(sample_stack, axis=(0, 2, 3), keepdims=True)
        
        # Load test slice
        test_data = []
        for v in vars_list: test_data.append(sub_ds[v].values)
        test_volga = volga_df.reindex(pd.to_datetime(sub_times), method='nearest').fillna(8000.0)['volga_q'].values
        test_data.append(test_volga[:, None, None] * np.ones_like(test_data[0][0]))
        test_stack = np.stack(test_data, axis=1)
        
        n_steps = len(test_dates)
        n_t_list, scale_t_list = [], []
        
        with torch.no_grad():
            for i in range(n_steps):
                x = test_stack[i : i + SEQ_LEN]
                x_norm = np.nan_to_num((x - mean_stat) / (std_stat + 1e-8))
                x_tensor = torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)
                
                _, n_t, scale_t = model(x_tensor)
                n_t_list.append(n_t.item())
                scale_t_list.append(scale_t.item())
                
        traj_df = pd.DataFrame({"Date": test_dates, "n_t": n_t_list, "scale_t": scale_t_list})
        out_csv = os.path.join(paths['dir'], "learned_budyko_params.csv")
        traj_df.to_csv(out_csv, index=False)
        physics_trajectories[res_name] = traj_df
        print(f"  Saved Budyko parameters to {out_csv} (Mean n_t={np.mean(n_t_list):.3f}, scale_t={np.mean(scale_t_list):.3f})")
        
    # Save uncertainty summary
    unc_df = pd.DataFrame(uncertainty_results)
    summary_path = "antigravity/results/pinn2/interval_comparison/uncertainty_weights_summary.csv"
    unc_df.to_csv(summary_path, index=False)
    print(f"\nSaved Uncertainty Summary to {summary_path}")
    print(unc_df.to_string(index=False))
    
    # 3. Create Master Figure
    print("\nGenerating master thesis figure...")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    
    # Subplot 1: Boxplots of n_t
    ax1 = fig.add_subplot(gs[0, 0])
    box_data = [physics_trajectories[r]['n_t'].values for r in ['1m', '10d', '1d'] if r in physics_trajectories]
    labels = [r for r in ['1m', '10d', '1d'] if r in physics_trajectories]
    ax1.boxplot(box_data, labels=labels, patch_artist=True, boxprops=dict(facecolor='#a29bfe', color='#6c5ce7'))
    ax1.set_title("Learned Budyko Parameter ($n_t$) Across Resolutions", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Budyko Parameter $n_t$ (Theoretical Catchment Property)")
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Bar chart of Precision Weights
    ax2 = fig.add_subplot(gs[0, 1])
    res_list = unc_df['Resolution'].values
    d_weights = unc_df['Data_Weight_%'].values
    p_weights = unc_df['Phys_Weight_%'].values
    x_pos = np.arange(len(res_list))
    width = 0.35
    ax2.bar(x_pos - width/2, d_weights, width, label='Data Precision Weight (%)', color='#0984e3')
    ax2.bar(x_pos + width/2, p_weights, width, label='Physics Precision Weight (%)', color='#00b894')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(res_list)
    ax2.set_title("Adaptive Loss Weighting Balance (Soft-Attention)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Effective Precision Share (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Timeseries of n_t & scale_t (1m and 10d)
    ax3 = fig.add_subplot(gs[1, :])
    if '1m' in physics_trajectories:
        ax3.plot(physics_trajectories['1m']['Date'], physics_trajectories['1m']['n_t'], label='1m Resolution ($n_t$)', color='#e84393', linewidth=2)
    if '10d' in physics_trajectories:
        ax3.plot(physics_trajectories['10d']['Date'], physics_trajectories['10d']['n_t'], label='10d Resolution ($n_t$)', color='#fdcb6e', linewidth=1.5, alpha=0.8)
    ax3.set_title("Timeseries of Learned Budyko Catchment Parameter over Test Period (2018-2026)", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Parameter Value ($n_t$)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "antigravity/results/pinn2/interval_comparison/pinn2_physics_analysis.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Master plot successfully saved to {plot_path}")

if __name__ == "__main__":
    analyze_all()
