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
                 num_layers=2, dropout=0.1, num_classes=6):
        super().__init__()
        
        self.normalize = nn.BatchNorm1d(in_seq_dim)
        self.seq_proj = nn.Sequential(
            nn.Linear(in_seq_dim, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )

        # Make sure d_model is divisible by 2 for the positional encoding
        assert d_model % 2 == 0, "d_model must be even"
        self.pos_encoder = PositionalEncoding(d_model)  # Changed from d_model//2 to d_model

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead,
                                                   dim_feedforward=128, 
                                                   dropout=dropout)
                                                #    dropout=dropout,
                                                #    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers)

        # Simplified meta projection to match dimensions
        meta_hidden_dim = 16  # or even smaller, like 8

        self.meta_proj = nn.Linear(in_meta_dim, meta_hidden_dim)
        combined_dim = d_model + meta_hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes)
        )

    def forward(self, x_seq, x_meta):
        """
        x_seq: (batch, seq_len=5, 3)
        x_meta: (batch, n_meta_features)
        """
        x = x_seq.transpose(1, 2)  # (batch, 3, seq_len)
        x = self.normalize(x)      # BatchNorm1d expects (batch, channels, length)
        x = x.transpose(1, 2)      # (batch, seq_len, 3)
        
        x = self.seq_proj(x)         # (batch, seq_len, d_model)
        x = self.pos_encoder(x)      # (batch, seq_len, d_model)

        x = x.permute(1, 0, 2)       # (seq_len, batch, d_model)
        x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        x = x.mean(dim=0)            # (batch, d_model)

        meta = self.meta_proj(x_meta)  # (batch, meta_hidden_dim)

        combined = torch.cat([x, meta], dim=1)  # (batch, d_model + meta_hidden_dim)

        return self.classifier(combined)  # (batch, num_classes)
    


'''
model: &base_model
  data_dir: "raw_data/"
  out_data_dir: "processed_data/"
  output_dir: "doc/latex/figure/"
  model_out_dir: "models/"
  random_seed: 42
  # ==== Data Processing ====
  sensor_loc:
    - "waist"
    - "ankle"
    - "wrist"
  ft_col:
    - "x"
    - "y"
    - "z"

  extracted_features:
    - "std"
  classes:
    - "downstairs"
    - "jog_treadmill"
    - "upstairs"
    - "walk_treadmill"
  window_size: 100
  stride: 5
  save_pkl: False

transformer:
  <<: *base_model
  test_size: 0.3
  warmup_ratio: 0.08  # percentage of total steps for warmup
  min_lr_ratio: 0.0  # minimum learning rate as a fraction of initial lr
  batch_size: 64
  patience: 20
  learning_rate: 0.0015
  weight_decay: 0.001
  epochs: 17
  d_model: 6
  fc_hidden_dim: 128
  nhead: 6
  num_layers: 2
  dropout: 0.1
  load_model_path: ''

    # NOTE:
    # self.head_dim * num_heads == self.embed_dim

'''