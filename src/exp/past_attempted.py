import torch
import torch.nn as nn
import torch.fft
import math
import numpy as np
from scipy import signal

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
