"""
PINN-1 v2: Upgraded Water Balance / Mass Conservation PINN
=========================================================================
Architecture: CNN-LSTM with ReLU activations (per user specification).
Physics: Enforces volumetric water balance with Bayesian Homoscedastic Uncertainty Weighting.

Equations:
    Δh_phys = (P_lake − α_E · E_lake) + β_volga · Q_volga · Δt / A_lake 
            + γ_other · max(0, P_basin − PET_basin)
=========================================================================
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

A_LAKE_M2_REF = 3.71e11

def _softplus_inv(y):
    return math.log(math.expm1(y))

class WaterBalancePINN1v2(nn.Module):
    """
    Upgraded PINN-1 with ReLU activations and Bayesian Homoscedastic Loss Weighting.
    Outputs: [dh_pred, a_E, b_v, g_o]
    """
    def __init__(self, in_channels=13, hidden_dim=64, seq_len=6):
        super().__init__()
        self.seq_len = seq_len

        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU()),
            ('avg_pool', nn.AdaptiveAvgPool2d((4, 4)))
        ]))

        self.lstm = nn.LSTM(
            input_size=1024, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, dropout=0.5,
        )

        self.regressor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_dim, 64)),
            ('relu_fc', nn.ReLU()),
            ('dropout_fc', nn.Dropout(0.5)),
            ('output_layer', nn.Linear(64, 4))
        ]))

        # Learnable log variances for data loss and physics loss
        self.log_vars = nn.Parameter(torch.tensor([0.0, -2.0]))

        with torch.no_grad():
            b = self.regressor[-1].bias
            b[0] = 0.0
            b[1] = _softplus_inv(1.0) # alpha_E -> 1.0
            b[2] = _softplus_inv(1.0) # beta_volga -> 1.0
            b[3] = _softplus_inv(0.3) # gamma_other -> 0.3

    def forward(self, x):
        bs, sl, c, h, w = x.shape
        feat = self.encoder(x.view(bs * sl, c, h, w)).view(bs, sl, -1)
        lstm_out, _ = self.lstm(feat)
        out = self.regressor(lstm_out[:, -1, :])

        dh_pred     = out[:, 0:1]
        alpha_E     = F.softplus(out[:, 1:2])
        beta_volga  = F.softplus(out[:, 2:3])
        gamma_other = F.softplus(out[:, 3:4])
        
        return dh_pred, alpha_E, beta_volga, gamma_other

    def water_balance_loss(self, dh_pred, alpha_E, beta_volga, gamma_other, P_lake, E_lake, Q_volga, P_basin, PET_basin, dt_seconds):
        lake_contrib = P_lake - alpha_E * E_lake
        Q_in_m = beta_volga * Q_volga * dt_seconds / A_LAKE_M2_REF
        runoff_contrib = gamma_other * (P_basin - PET_basin).clamp(min=0.0)

        dh_phys = (lake_contrib + Q_in_m + runoff_contrib).view_as(dh_pred)
        return F.mse_loss(dh_pred, dh_phys), {
            'lake_contrib': float(lake_contrib.detach().mean().item()),
            'Q_in_m':       float(Q_in_m.detach().mean().item()),
            'runoff':       float(runoff_contrib.detach().mean().item()),
            'dh_phys':      float(dh_phys.detach().mean().item()),
        }

class DynamicAreaWaterBalancePINN1v2(nn.Module):
    """
    Upgraded PINN-1 with ReLU activations and dynamic bathymetric surface area curve A(H).
    """
    def __init__(self, in_channels=13, hidden_dim=64, seq_len=6):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.lstm = nn.LSTM(1024, hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 4)
        )
        self.log_vars = nn.Parameter(torch.tensor([0.0, -2.0]))

        with torch.no_grad():
            b = self.regressor[-1].bias
            b[0], b[1], b[2], b[3] = 0.0, _softplus_inv(1.0), _softplus_inv(1.0), _softplus_inv(0.3)

    def forward(self, x):
        bs, sl, c, h, w = x.shape
        feat = self.encoder(x.view(bs * sl, c, h, w)).view(bs, sl, -1)
        lstm_out, _ = self.lstm(feat)
        out = self.regressor(lstm_out[:, -1, :])
        return out[:, 0:1], F.softplus(out[:, 1:2]), F.softplus(out[:, 2:3]), F.softplus(out[:, 3:4])

    def dynamic_water_balance_loss(self, dh_pred, a_E, b_v, g_o, P_lake, E_lake, Q_volga, P_basin, PET_basin, lake_h, dt_sec):
        area_dynamic = A_LAKE_M2_REF + 1.5e10 * (lake_h - (-27.0))
        area_dynamic = area_dynamic.clamp(min=3.0e11, max=4.2e11)
        
        lake_contrib = P_lake - a_E * E_lake
        Q_in_m = (b_v * Q_volga * dt_sec) / area_dynamic
        runoff_contrib = g_o * (P_basin - PET_basin).clamp(min=0.0)
        
        dh_phys = (lake_contrib + Q_in_m + runoff_contrib).view_as(dh_pred)
        return F.mse_loss(dh_pred, dh_phys)
