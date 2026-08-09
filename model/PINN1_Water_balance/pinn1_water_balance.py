"""
PINN-1: Water Balance / Mass Conservation PINN
==================================================
The most fundamental physics constraint of the three PINNs in this study.
For a closed basin like the Caspian:

    dh/dt = (P_lake − α_E · E_lake)            ← lake surface fluxes
          + β_volga · Q_volga · Δt / A_lake    ← Volga inflow
          + γ_other · max(0, P_basin − PET_basin)  ← residual basin runoff
                                                    (Ural, Terek, groundwater)

This PINN does NOT model evaporation (that's PINN-3's job — Penman) and does
NOT partition runoff (that's PINN-2's job — Budyko). It only enforces that
the level change equals the sum of measured inflows and outflows, with three
small correction factors learned from data.

The three corrections:
    α_E      : ERA5 evaporation bias correction (≈ 1.0 if ERA5 is unbiased)
    β_volga  : fraction of Volga discharge that actually reaches the lake
               (groundwater interception, channel losses; ≈ 1.0 if clean)
    γ_other  : residual basin contribution scaling (Ural, Terek, smaller
               tributaries that aren't in `volga_discharge.csv`)

Why this matters for the defense:
    PINN-1 is the simplest of the three.  Its R² should be high precisely
    BECAUSE mass conservation is the most universally true constraint —
    there is nothing to mis-model.  PINN-1 sets the ceiling that the
    more elaborate PINN-2 and PINN-3 need to match.
==================================================
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Caspian geometry
A_LAKE_M2     = 3.71e11           # surface area ≈ 371,000 km²
SEC_PER_MONTH = 30.42 * 86400     # 2.628e6 s


def _softplus_inv(y):
    """Inverse softplus: given desired output y > 0, return bias b such that softplus(b) = y."""
    return math.log(math.expm1(y))


class WaterBalancePINN1(nn.Module):
    """
    CNN-LSTM with 4 heads:  [dh_pred, α_E, β_volga, γ_other]
    Architecture identical to PINN-2 (Budyko) and PINN-3 (Penman) so the
    three-PINN comparison is governed purely by the physics choice.
    """

    def __init__(self, in_channels=13, hidden_dim=64, seq_len=6):
        super().__init__()
        self.seq_len = seq_len

        # 1. Spatial Feature Extractor (CNN)
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU()),
            ('avg_pool', nn.AdaptiveAvgPool2d((4, 4)))
        ]))

        # 2. Temporal Aggregator (LSTM)
        self.lstm = nn.LSTM(
            input_size=1024, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, dropout=0.5,
        )

        # 3. Prediction Head — 6 outputs: 
        # [dh_pred, alpha_E, beta_volga, gamma_other, log_var_data, log_var_phys]
        self.regressor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_dim, 64)),
            ('relu_fc', nn.ReLU()),
            ('dropout_fc', nn.Dropout(0.5)),
            ('output_layer', nn.Linear(64, 6))
        ]))

        # Bias init: physics corrections start near their prior values
        with torch.no_grad():
            b = self.regressor[-1].bias
            b[0] = 0.0                          # dh_pred — symmetric around 0
            b[1] = _softplus_inv(1.0)           # α_E      → 1.0  (ERA5 unbiased)
            b[2] = _softplus_inv(1.0)           # β_volga  → 1.0  (full transfer)
            b[3] = _softplus_inv(0.3)           # γ_other  → 0.3  (small residual)

    def forward(self, x):
        bs, sl, c, h, w = x.shape
        feat = self.encoder(x.view(bs * sl, c, h, w))
        feat = feat.view(bs, sl, -1)
        lstm_out, _ = self.lstm(feat)
        out = self.regressor(lstm_out[:, -1, :])

        dh_pred     = out[:, 0:1]
        alpha_E     = F.softplus(out[:, 1:2])      # ≥ 0
        beta_volga  = F.softplus(out[:, 2:3])      # ≥ 0
        gamma_other = F.softplus(out[:, 3:4])      # ≥ 0
        log_var_data = out[:, 4:5]
        log_var_phys = out[:, 5:6]
        
        return dh_pred, alpha_E, beta_volga, gamma_other, log_var_data, log_var_phys

    # ------------------------------------------------------------------
    # Mass conservation physics loss
    # ------------------------------------------------------------------
    def water_balance_loss(
        self,
        dh_pred,
        alpha_E, beta_volga, gamma_other,
        P_lake,          # precipitation on lake surface     [m / step]
        E_lake,          # ERA5 evaporation from lake        [m / step]
        Q_volga,         # Volga discharge                   [m³ / s]
        P_basin,         # basin precipitation               [m / step]
        PET_basin,       # basin potential ET                [m / step]
        dt_seconds,      # time step in seconds
    ):
        """
        Soft constraint that drives the network's predicted Δh toward the Δh
        implied by mass conservation.
        """

        # 1. Lake surface: direct precipitation minus evaporation
        lake_contrib = P_lake - alpha_E * E_lake

        # 2. Volga inflow → m/step over the lake area
        Q_in_m = beta_volga * Q_volga * dt_seconds / A_LAKE_M2

        # 3. Residual basin contribution
        runoff_contrib = gamma_other * (P_basin - PET_basin).clamp(min=0.0)

        # Mass conservation
        dh_phys = lake_contrib + Q_in_m + runoff_contrib
        dh_phys = dh_phys.view_as(dh_pred)

        loss = F.mse_loss(dh_pred, dh_phys)
        diag = {
            'lake_contrib': float(lake_contrib.detach().mean().item()),
            'Q_in_m':       float(Q_in_m.detach().mean().item()),
            'runoff':       float(runoff_contrib.detach().mean().item()),
            'dh_phys':      float(dh_phys.detach().mean().item()),
        }
        return loss, diag
