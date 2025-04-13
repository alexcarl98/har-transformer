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

    def combine_with(self, other):
        # print(f"Combining datasets of sizes: {len(self)} and {len(other)}")
        X = torch.cat([self.X, other.X], dim=0)
        X_meta = torch.cat([self.X_meta, other.X_meta], dim=0)
        y = torch.cat([self.y, other.y], dim=0)
        result = HARWindowDataset(X, X_meta, y)
        # print(f"Result size: {len(result)}")
        return result
    
    @classmethod
    def decouple_combine(cls, har_list: list['HARWindowDataset']):
        # print(f"\nDEBUG decouple_combine:")
        # print(f"Number of sensors to combine: {len(har_list)}")
        first = har_list[0]
        # print(f"First sensor dataset size: {len(first)}")
        
        for i in range(1, len(har_list)):
            # print(f"Combining with sensor {i}, size: {len(har_list[i])}")
            first = first.combine_with(har_list[i])
            # print(f"Combined size: {len(first)}")
        return first



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)
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


# === Transformer Model for HAR ===
class AccelTransformer(nn.Module):
    def __init__(self, d_model=128, fc_hidden_dim=128, 
                 in_seq_dim=3, in_meta_dim=3, nhead=4, 
                 num_layers=2, dropout=0.1, num_classes=6,
                 accel_range=(-15, 15), 
                 use_metadata=False,
                 use_vec_mag=True
                 ):
        super().__init__()
        
        self.use_metadata = use_metadata
        self.use_vec_mag = use_vec_mag
        if use_vec_mag:
            in_seq_dim += 1
        
        if use_metadata:
            self.seq_proj = nn.Sequential(
                nn.Linear(in_seq_dim, d_model//2),
                nn.ReLU(),
                nn.Linear(d_model//2, d_model)
            )
        else:
            self.seq_proj = nn.Linear(in_seq_dim, d_model)

        self.normalize = nn.BatchNorm1d(in_seq_dim)

        assert d_model % 2 == 0, "d_model must be even"
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead,
                                                   dim_feedforward=128, 
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers)

        # Only create metadata layers if using metadata
        if use_metadata:
            meta_hidden_dim = d_model
            self.meta_proj = nn.Sequential(
                nn.Linear(in_meta_dim, d_model//2),
                nn.ReLU(),
                nn.Linear(d_model//2, d_model)
            )
            combined_dim = d_model + meta_hidden_dim
        else:
            combined_dim = d_model

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes)
        )

    def forward(self, x_seq, x_meta=None):
        """
        x_seq: (batch, seq_len=5, 3)
        x_meta: (batch, n_meta_features) or None if use_metadata=False
        """

        if self.use_vec_mag:
            magnitude = torch.norm(x_seq, dim=2, keepdim=True)
            x_seq = torch.cat([x_seq, magnitude], dim=2)
            

        x = x_seq.transpose(1, 2)
        x = self.normalize(x)
        x = x.transpose(1, 2)
        
        x = self.seq_proj(x)
        x = self.pos_encoder(x)

        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)

        if self.use_metadata:
            assert x_meta is not None, "Metadata expected but not provided"
            meta = self.meta_proj(x_meta)
            x = torch.cat([x, meta], dim=1)

        return self.classifier(x)


class XYZLSTM(nn.Module):
    def __init__(self, in_seq_dim=3, hidden_dim=128, num_classes=6):
        super().__init__()
        self.normalize = nn.LayerNorm(in_seq_dim)
        
        self.lstm = nn.LSTM(input_size=in_seq_dim, hidden_size=hidden_dim, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, x_meta=None):
        x = self.normalize(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])
        
        

# import torch
# import torch.nn as nn
# import math

class CNNTRNPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CNNTransformerHAR(nn.Module):
    def __init__(self, seq_len=100, in_seq_dim=3, in_meta_dim=3, num_classes=6,
                 d_model=128, cnn_out_channels=64, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()

        # CNN: (batch, 3, seq_len) -> (batch, cnn_out_channels, new_seq_len)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_seq_dim, out_channels=cnn_out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_out_channels, d_model, kernel_size=5, padding=2),  # -> (batch, d_model, seq_len)
            nn.ReLU()
        )

        self.pos_encoder = CNNTRNPositionalEncoding(d_model=d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.meta_proj = nn.Linear(in_meta_dim, d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_seq, x_meta):
        """
        x_seq: (batch, seq_len, 3)
        x_meta: (batch, in_meta_dim)
        """
        x_seq = x_seq.permute(0, 2, 1)       # (batch, 3, seq_len)
        x = self.cnn(x_seq)                  # (batch, d_model, seq_len)
        x = x.permute(0, 2, 1)               # (batch, seq_len, d_model)

        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)               # (seq_len, batch, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)                    # (batch, d_model)

        x_meta = self.meta_proj(x_meta)      # (batch, d_model)
        x_combined = torch.cat([x, x_meta], dim=1)
        return self.classifier(x_combined)   # (batch, num_classes)
