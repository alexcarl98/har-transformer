import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# Parameters
subject_id = "002"
data_dir = "har_data"
sensor = "ankle"
axis = "y"
window_length = 25  # Must be odd and <= size of signal
polyorder = 3       # Polynomial order for smoothing

# Load the data
full_path = os.path.join(data_dir, f"{subject_id}.csv")
data = pd.read_csv(full_path)
data['time'] = pd.to_datetime(data['time'])

# Filter to a single activity segment for clarity
activity_changes = data['activity'].ne(data['activity'].shift()).cumsum()
activity_groups = data.groupby(activity_changes)
sample_segment = [group for _, group in activity_groups if group['activity'].iloc[0] == 'walk_sidewalk'][0]

# Extract signal and time
signal_raw = sample_segment[f"{sensor}_{axis}"].values
time_rel = (sample_segment['time'] - sample_segment['time'].iloc[0]).dt.total_seconds()

# Apply Savitzky-Golay filter
signal_smooth = savgol_filter(signal_raw, window_length=window_length, polyorder=polyorder)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(time_rel, signal_raw, color='gray', alpha=0.5, label='Raw Signal')
plt.plot(time_rel, signal_smooth, color='blue', linewidth=2, label='Smoothed Signal (Savitzky-Golay)')
plt.title(f'{sensor.capitalize()} Sensor ({axis}-axis) - Savitzky-Golay Filter Demo')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


################################################################

'''
import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Literal

import numpy as np
import pandas as pd
import torch
from datetime import datetime

@dataclass
class DataConfig:
    data_dir: str = "raw_data/"
    middle_percentage: float = 0.85
    encoding_method: Literal["one_hot", "label_encoding"] = "one_hot"
    sensor_loc: List[str] = field(default_factory=lambda: ["waist", "ankle", "wrist"])
    ft_col: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    window_size: int = 100
    stride: int = 10


@dataclass
class TConfig:
    """Configuration for the experiment."""
    data_dir: str = "raw_data/"
    out_data_dir: str = "processed_data/"
    output_dir: str = "doc/latex/figure/"
    model_out_dir: str = "models/"
    random_seed: int = 42
    sensor_loc: List[str] = field(default_factory=lambda: ["waist", "ankle", "wrist"])
    ft_col: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    extracted_features: List[str] = field(default_factory=lambda: ["mean", "std"])
    classes: List[str] = field(default_factory=lambda: ["downstairs", "jog_treadmill", "upstairs", "walk_mixed", "walk_sidewalk", "walk_treadmill"])
    window_size: int = 100
    stride: int = 10
    save_pkl: bool = False
    test_size: float = 0.4
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.0
    batch_size: int = 16
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 20
    d_model: int = 128
    fc_hidden_dim: int = 128
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    load_model_path: str = ''

    def __post_init__(self):
        self.time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_sub_dir = f"run_{self.time_stamp}"
        self.figure_out_dir = os.path.join(self.model_out_dir, self.model_sub_dir)
        self.weights_out_dir = os.path.join(self.model_out_dir, self.model_sub_dir, "weights")
        self.encoder_dict = {label: idx for idx, label in enumerate(self.classes)}
        self.decoder_dict = {idx: label for idx, label in enumerate(self.classes)}

        if self.load_model_path:
            assert os.path.exists(self.load_model_path), f"Model file {self.load_model_path} does not exist."
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.exists(self.out_data_dir):
            os.makedirs(self.out_data_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.model_out_dir):
            os.makedirs(self.model_out_dir)

        os.makedirs(self.figure_out_dir, exist_ok=True)
        os.makedirs(self.weights_out_dir, exist_ok=True)

    
    @property
    def in_seq_dim(self):
        return len(self.ft_col)

    @property
    def in_meta_dim(self):
        return len(self.extracted_features)*self.in_seq_dim

    @property
    def num_classes(self):
        return len(self.classes)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config['transformer'])

    @classmethod
    def create_sweep_dict(cls):
        """Creates a wandb sweep configuration dictionary based on TConfig parameters."""
        return {
            "program": "src/train.py",
            "method": "bayes",
            "metric": {
                "goal": "maximize",
                "name": "val_f1"
            },
            "parameters": {
                "batch_size": {
                    "distribution": "int_uniform",
                    "min": 16,
                    "max": 128
                },
                "d_model": {
                    "distribution": "categorical",
                    "values": [32, 64, 128]
                },
                "dropout": {
                    "distribution": "uniform",
                    "min": 0.05,
                    "max": 0.2
                },
                "epochs": {
                    "distribution": "int_uniform",
                    "min": 10,
                    "max": 80
                },
                "fc_hidden_dim": {
                    "distribution": "int_uniform",
                    "min": 64,
                    "max": 512
                },
                "learning_rate": {
                    "distribution": "uniform",
                    "min": 0.0005,
                    "max": 0.002
                },
                "nhead": {
                    "distribution": "categorical",
                    "values": [2, 4, 8]
                },
                "stride": {
                    "distribution": "int_uniform",
                    "min": 3,
                    "max": 40
                },
                "test_size": {
                    "distribution": "uniform",
                    "min": 0.2,
                    "max": 0.4
                },
                "weight_decay": {
                    "distribution": "uniform",
                    "min": 0.0001,
                    "max": 0.01
                },
                "window_size": {
                    "distribution": "int_uniform",
                    "min": 50,
                    "max": 200
                }
            },
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 10
            }
        }

    @classmethod
    def from_wandb_config(cls, wandb_config, yaml_config):
        """Creates a TConfig instance from wandb sweep config and base yaml config."""
        base_config = yaml_config['transformer'].copy()
        # Update base config with sweep parameters
        base_config.update(wandb_config)
        return cls(**base_config)


if __name__ == "__main__":
    config = TConfig.from_yaml("config.yml")
    print(config)
    print(config.figure_out_dir)
    print(config.weights_out_dir)

'''

####################################################

'''
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
                                                   dim_feedforward=d_model*4, 
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
'''

##########################

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
    # - "vm"
  extracted_features:
    # - "mean"
    - "std"
  classes:
    - "downstairs"
    - "jog_treadmill"
    - "upstairs"
    - "walk_treadmill"
    # - "walk_mixed"
    # - "walk_sidewalk"
  window_size: 100
  stride: 5
  save_pkl: False
  
# ==== Model Training ====
random_forest:
  <<: *base_model
  n_estimators: 100
  max_depth: 10
  min_samples_split: 2
  min_samples_leaf: 1

transformer:
  <<: *base_model
  test_size: 0.34
  warmup_ratio: 0.08  # percentage of total steps for warmup
  min_lr_ratio: 0.0  # minimum learning rate as a fraction of initial lr
  batch_size: 108
  patience: 20
  learning_rate: 0.01
  weight_decay: 0.00
  epochs: 17
  d_model: 128
  fc_hidden_dim: 128
  nhead: 8
  num_layers: 2
  dropout: 0.11094459161151084
  load_model_path: ''

    # NOTE:
    # self.head_dim * num_heads == self.embed_dim

'''