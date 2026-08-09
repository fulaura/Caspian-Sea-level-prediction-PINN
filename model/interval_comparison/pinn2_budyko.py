"""
PINN-2: Budyko-Constrained Lake Water Level Prediction (v21 Soft Attention)
=========================================================================
Architecture: CNN-LSTM with SiLU activation
Physics: Soft-PINN with dynamic parameters (n_t, scale_t) and Uncertainty Weighting
=========================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class BudykoPINN2_v21(nn.Module):
    def __init__(self, in_channels=13, hidden_dim=64, seq_len=6):
        super(BudykoPINN2_v21, self).__init__()
        self.seq_len = seq_len
        
        # 1. Spatial Feature Extractor (CNN)
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)),
            ('silu1', nn.SiLU()),
            ('pool1', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ('silu2', nn.SiLU()),
            ('avg_pool', nn.AdaptiveAvgPool2d((4, 4)))
        ]))
        
        # 2. Temporal Aggregator (LSTM)
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_dim, num_layers=2, 
                            batch_first=True, dropout=0.5)
        
        # 3. Prediction Head
        self.regressor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_dim, 64)),
            ('silu_fc', nn.SiLU()),
            ('dropout_fc', nn.Dropout(0.5)),
            ('output_layer', nn.Linear(64, 3))
        ]))

        # 4. Adaptive Loss Weights
        # Initializing log(var) for Data Loss and Physics Loss
        # log_vars[0] -> data, log_vars[1] -> phys
        self.log_vars = nn.Parameter(torch.tensor([0.0, -2.0]))

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.encoder(x_reshaped)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(spatial_features)
        last_features = lstm_out[:, -1, :]
        
        outputs = self.regressor(last_features)
        
        dh_pred = outputs[:, 0:1]
        n_t = F.softplus(outputs[:, 1:2]) + 1.0 
        scale_t = F.softplus(outputs[:, 2:3]) 
        
        return dh_pred, n_t, scale_t

    def budyko_loss(self, dh_pred, n_t, scale_t, P_basin, PET_basin, P_lake, E_lake):
        phi = PET_basin / (P_basin + 1e-8)
        phi = phi.clamp(0.01, 20.0)

        phi_n = torch.pow(phi, n_t)
        et_ratio_theory = phi / torch.pow(1.0 + phi_n, 1.0 / n_t)
        
        ET_budyko = et_ratio_theory * P_basin
        runoff_contrib = (P_basin - ET_budyko) * scale_t
        lake_contrib = P_lake - E_lake

        dh_phys = runoff_contrib + lake_contrib
        dh_phys = dh_phys.view_as(dh_pred)

        return F.mse_loss(dh_pred, dh_phys)
