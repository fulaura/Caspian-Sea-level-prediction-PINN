"""
PINN-2v2: Budyko-Constrained Lake Water Level Prediction with Measured River Inflow
==================================================================================
This upgraded architecture uses Volga river discharge as a direct physical inflow
constraint, while using the Budyko formulation to partition precip/PET for the
remaining, ungauged portion of the catchment basin.
==================================================================================
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

A_LAKE_M2_REF = 3.71e11

def _softplus_inv(x):
    return math.log(math.exp(x) - 1.0)

class BudykoRiverPINN2(nn.Module):
    def __init__(self, in_channels=13, hidden_dim=64, seq_len=6):
        super(BudykoRiverPINN2, self).__init__()
        self.seq_len = seq_len
        
        # Spatial Feature Extractor (CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Temporal Aggregator (LSTM)
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_dim, num_layers=2, 
                            batch_first=True, dropout=0.5)
        
        # Prediction Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 4) # [dh_norm, beta_volga, n_t, scale_t]
        )

        # Adaptive Loss Weights: log_vars[0] -> data, log_vars[1] -> phys
        self.log_vars = nn.Parameter(torch.tensor([0.0, -2.0]))

        # Initialize biases to realistic physical values
        with torch.no_grad():
            b = self.regressor[-1].bias
            b[0] = 0.0                  # dh_norm
            b[1] = _softplus_inv(1.0)   # beta_volga
            b[2] = _softplus_inv(0.8)   # n_t (softplus + 1.0 -> 1.8)
            b[3] = _softplus_inv(0.1)   # scale_t

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.encoder(x_reshaped)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(spatial_features)
        last_features = lstm_out[:, -1, :]
        
        outputs = self.regressor(last_features)
        
        dh_pred = outputs[:, 0:1]
        beta_volga = F.softplus(outputs[:, 1:2])
        n_t = F.softplus(outputs[:, 2:3]) + 1.0
        scale_t = F.softplus(outputs[:, 3:4])
        
        return dh_pred, beta_volga, n_t, scale_t

    def budyko_river_loss(self, dh_pred, beta_volga, n_t, scale_t, P_lake, E_lake, Q_volga, P_basin, PET_basin, dt_seconds):
        # 1. Lake mass balance (direct precip and evaporation)
        lake_contrib = P_lake - E_lake
        
        # 2. Measured Volga Inflow (scaled)
        Q_in_m = (beta_volga * Q_volga * dt_seconds) / A_LAKE_M2_REF
        
        # 3. Residual basin runoff using Budyko curve (representing unmeasured basin runoff)
        phi = PET_basin / (P_basin + 1e-8)
        phi = phi.clamp(0.01, 20.0)
        phi_n = torch.pow(phi, n_t)
        et_ratio_theory = phi / torch.pow(1.0 + phi_n, 1.0 / n_t)
        
        ET_budyko = et_ratio_theory * P_basin
        runoff_budyko = (P_basin - ET_budyko) * scale_t
        
        # 4. Total physical water level change
        dh_phys = lake_contrib + Q_in_m + runoff_budyko
        dh_phys = dh_phys.view_as(dh_pred)

        return F.mse_loss(dh_pred, dh_phys)
