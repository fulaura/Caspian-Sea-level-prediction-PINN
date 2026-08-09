import torch
import torch.nn as nn

class DataOnlyBaselineLSTM(nn.Module):
    """
    Ablation Baseline: Data-Only CNN-LSTM.
    Identical CNN-LSTM feature extractor as PINN-1, PINN-2, and PINN-3,
    but with exactly 1 output head (dh_pred) and no physics constraints.
    """
    def __init__(self, in_channels=13, hidden_dim=64, seq_len=6):
        super().__init__()
        self.seq_len = seq_len
        
        # Spatial Feature Extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Temporal Encoder
        self.lstm = nn.LSTM(1024, hidden_dim, batch_first=True, num_layers=2)
        
        # Single Head Regressor
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)  # Only predicting dh_pred
        )
        
    def forward(self, x):
        bs, sl, c, h, w = x.shape
        feat = self.encoder(x.view(bs * sl, c, h, w))
        feat = feat.view(bs, sl, -1)
        lstm_out, _ = self.lstm(feat)
        dh_pred = self.regressor(lstm_out[:, -1, :])
        return dh_pred
