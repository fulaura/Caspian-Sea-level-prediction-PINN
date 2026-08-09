"""
PINN-3: Energy / Penman-Constrained Lake Water Level Prediction (v20 Deep Learning)
====================================================================================
Architecture: CNN-LSTM + Heavy Regularization (matched to PINN-2 for fair comparison)
Physics:      Penman / Priestley-Taylor energy balance for open-water evaporation,
              with dynamic learnable parameters (alpha_PT, C_wind, gamma_runoff).

Soft-PINN formulation:
    Δ(T)        = 4098 · 0.6108·exp(17.27·T/(T+237.3)) / (T+237.3)²   [kPa/°C]
    γ(P)        = 0.665e-3 · P                                         [kPa/°C, P in kPa]
    e_s(T)      = 0.6108 · exp(17.27·T/(T+237.3))                      [kPa]
    e_a(T_d)    = 0.6108 · exp(17.27·T_d/(T_d+237.3))                  [kPa]
    VPD         = e_s − e_a                                            [kPa]
    E_rad       = α_PT · Δ/(Δ+γ) · Rn_eq                               [m/month water equiv.]
    E_aero      = C_wind · γ/(Δ+γ) · f(U) · VPD                        [m/month water equiv.]
    E_penman    = E_rad + E_aero                                       [m/month]
    dh_phys     = (P_lake − E_penman) + γ_runoff · (P_basin − PET_basin)

The three physics parameters (α_PT, C_wind, γ_runoff) are produced by the same
network head, so they adapt month-by-month to the climate state.
====================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Penman helper functions (vectorised, unit-safe) ──────────────────────
def saturation_slope(T_celsius):
    """
    Slope of saturation vapour pressure curve  Δ  [kPa/°C]
    T in degrees Celsius.
    """
    es = 0.6108 * torch.exp(17.27 * T_celsius / (T_celsius + 237.3))
    delta = 4098.0 * es / (T_celsius + 237.3).pow(2)
    return delta


def psychrometric_constant(P_kpa):
    """
    Psychrometric constant γ  [kPa/°C]
    P in kPa.
    """
    return 0.665e-3 * P_kpa


def vapour_pressure(T_celsius):
    """Saturation/actual vapour pressure  [kPa]  (Tetens)."""
    return 0.6108 * torch.exp(17.27 * T_celsius / (T_celsius + 237.3))


# ── PINN-3 Model ─────────────────────────────────────────────────────────
class EnergyPINN3(nn.Module):
    """
    CNN-LSTM with three physics-parameter heads + one prediction head.
    Architecture identical to PINN-2 (Budyko) so results are directly comparable.
    """

    def __init__(self, in_channels=13, hidden_dim=64, seq_len=6):
        super().__init__()
        self.seq_len = seq_len

        # 1. Spatial Feature Extractor (CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # 2. Temporal Aggregator (LSTM)
        self.lstm = nn.LSTM(
            input_size=1024, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, dropout=0.5,
        )

        # 3. Prediction Head — outputs 4 quantities:
        #    [0] dh_pred, [1] alpha_PT, [2] C_wind, [3] gamma_runoff
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.encoder(x_reshaped)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(spatial_features)
        last_features = lstm_out[:, -1, :]
        outputs = self.regressor(last_features)

        # head [0]: water level change (raw scalar, can be negative)
        dh_pred = outputs[:, 0:1]

        # head [1]: Priestley-Taylor coefficient.
        #          Free-water reference value ≈ 1.26.  Range pinned to [1.0, ~3].
        alpha_PT = F.softplus(outputs[:, 1:2]) + 1.0

        # head [2]: aerodynamic / wind-function coefficient (positive).
        C_wind = F.softplus(outputs[:, 2:3])

        # head [3]: basin-runoff coupling factor (positive scaling).
        gamma_runoff = F.softplus(outputs[:, 3:4])

        return dh_pred, alpha_PT, C_wind, gamma_runoff

    # ------------------------------------------------------------------
    # Penman energy-balance physics loss
    # ------------------------------------------------------------------
    def penman_loss(
        self,
        dh_pred,
        alpha_PT, C_wind, gamma_runoff,
        T_lake,         # 2 m air-temp over the lake     [°C]
        Td_lake,        # 2 m dewpoint over the lake     [°C]
        Rn_lake_eq,     # net radiation lake, water-eq.  [m/month]
        U_lake,         # 10 m wind speed over the lake  [m/s]
        P_kpa_lake,     # surface pressure over the lake [kPa]
        P_lake,         # precipitation over the lake    [m/month]
        P_basin,        # precipitation over the basin   [m/month]
        PET_basin,      # potential ET over the basin    [m/month]
    ):
        """
        Soft constraint that drives the network's predicted Δh toward the
        Δh implied by the Penman / Priestley-Taylor energy balance.

        All scalar fluxes must be in m/month (water-equivalent depth) so the
        residual lives on the same scale as the target Δh.
        """

        # Energy / aerodynamic terms ---------------------------------------
        delta_T = saturation_slope(T_lake)                        # kPa/°C
        gamma_p = psychrometric_constant(P_kpa_lake)              # kPa/°C
        es      = vapour_pressure(T_lake)
        ea      = vapour_pressure(Td_lake)
        VPD     = (es - ea).clamp(min=0.0)                        # kPa

        denom   = (delta_T + gamma_p).clamp(min=1e-6)

        # Radiation-driven part (Priestley-Taylor)
        E_rad   = alpha_PT * (delta_T / denom) * Rn_lake_eq       # m/month

        # Aerodynamic / wind part (dimensionless wind function 1+0.536·U,
        # multiplied by a learnable transfer coefficient C_wind so the
        # physical scale is set adaptively without forcing a fixed constant)
        f_U     = 1.0 + 0.536 * U_lake
        E_aero  = C_wind * (gamma_p / denom) * f_U * VPD * 1e-3   # kPa→ m/month

        E_pen   = E_rad + E_aero                                  # m/month

        # Water balance ----------------------------------------------------
        runoff_contrib = gamma_runoff * (P_basin - PET_basin).clamp(min=0.0)
        dh_phys        = (P_lake - E_pen) + runoff_contrib

        dh_phys        = dh_phys.view_as(dh_pred)
        return F.mse_loss(dh_pred, dh_phys), {
            'E_pen':  E_pen.detach().mean().item(),
            'E_rad':  E_rad.detach().mean().item(),
            'E_aero': E_aero.detach().mean().item(),
            'VPD':    VPD.detach().mean().item(),
        }
