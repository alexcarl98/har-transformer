import torch
import torch.nn as nn
import math

class HARWindowDataset(torch.utils.data.Dataset):
    def __init__(self, X, X_meta, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X_meta = torch.tensor(X_meta, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_meta[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
    
class NormalizeAccel(nn.Module):
    def __init__(self, accel_range=(-15, 15)):
        super().__init__()
        self.min_val = accel_range[0]
        self.max_val = accel_range[1]
    
    def forward(self, x):
        return 2 * (x - self.min_val) / (self.max_val - self.min_val) - 1



# === Transformer Model for HAR ===
class AccelTransformer(nn.Module):
    def __init__(self, d_model=128, fc_hidden_dim=128, 
                 in_seq_dim=3, in_meta_dim=3, nhead=4, 
                 num_layers=2, dropout=0.1, num_classes=6,
                 accel_range=(-15, 15)):
        super().__init__()
        # self.seq_proj = nn.Linear(in_seq_dim, d_model)             # Input: (batch, seq_len, 3)
            # NormalizeAccel(accel_range),  # Keep input normalization
            # nn.BatchNorm1d(in_seq_dim),
        self.normalize = nn.BatchNorm1d(in_seq_dim)
        
        self.seq_proj = nn.Sequential(
            nn.Linear(in_seq_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=128, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.meta_proj = nn.Linear(in_meta_dim, d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes)
        )


    def forward(self, x_seq, x_meta):
        """
        x_seq: (batch, seq_len=5, 3)
        x_meta: (batch, n_meta_features)
        """
        # batch_size, seq_len, _ = x_seq.shape
        x= x_seq.transpose(1,2)
        x = self.normalize(x)
        x= x.transpose(1,2)

        x = self.seq_proj(x)         # (batch, seq_len, d_model)

        # x = self.seq_proj(x_seq)         # (batch, seq_len, d_model)
        x = self.pos_encoder(x)               # (batch, seq_len, d_model)

        x = x.permute(1, 0, 2)                # (seq_len, batch, d_model)
        x = self.transformer_encoder(x)       # (seq_len, batch, d_model)
        x = x.mean(dim=0)                     # (batch, d_model)

        meta = self.meta_proj(x_meta)        # (batch, d_model)

        combined = torch.cat([x, meta], dim=1)   # (batch, d_model * 2)

        return self.classifier(combined)      # (batch, num_classes)
