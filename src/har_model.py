import torch
import torch.nn as nn
import torch.fft
import math
import numpy as np
from scipy import signal

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


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.rel_pos_embedding = nn.Parameter(torch.randn(2 * max_len - 1, d_model))
        positions = torch.arange(max_len).unsqueeze(1)
        rel_positions = positions - positions.T + max_len - 1
        self.register_buffer('rel_positions', rel_positions)
        
    def forward(self, x):
        """x: (batch_size, seq_len, d_model)"""
        seq_len = x.size(1)
        rel_pos = self.rel_pos_embedding[self.rel_positions[:seq_len, :seq_len]]
        return x, rel_pos  # Used differently in attention mechanism

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x):
        """x: (batch_size, seq_len, d_model)"""
        return x + self.pos_embedding[:, :x.size(1), :]


# === Transformer Model for HAR ===
class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, seq_features, stat_features):
        # seq_features: (seq_len, batch, d_model)
        # stat_features: (batch, d_model)
        stat_features = stat_features.unsqueeze(0)  # (1, batch, d_model)
        attended, _ = self.attn(stat_features, seq_features, seq_features)
        return self.norm(attended + stat_features).squeeze(0)

class AccelTransformer(nn.Module):
    def __init__(self, d_model=128, fc_hidden_dim=128, 
                 in_seq_dim=3, in_meta_dim=3, nhead=4, 
                 num_layers=2, dropout=0.1, num_classes=6,
                 accel_range=(-15, 15), 
                 use_metadata=False,
                 use_vec_mag=True,
                 use_stats=True,
                 use_fft=True
                 ):
        super().__init__()
        
        self.use_metadata = use_metadata
        self.use_vec_mag = use_vec_mag
        if use_vec_mag:
            in_seq_dim += 1
        self.use_stats = use_stats
        self.use_fft = use_fft
        
        if use_metadata or use_stats or use_fft:
            self.seq_proj = nn.Sequential(
                nn.Linear(in_seq_dim, d_model//2),
                nn.ReLU(),
                nn.Linear(d_model//2, d_model)
            )
        else:
            self.seq_proj = nn.Linear(in_seq_dim, d_model)

        self.normalize = nn.BatchNorm1d(in_seq_dim)

        assert d_model % 2 == 0, "d_model must be even"
        # self.pos_encoder = PositionalEncoding(d_model)
        # self.pos_encoder = RelativePositionalEncoding(d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model)

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

        if use_stats:
            stats_input_dim = 18
            combined_dim += d_model
            print(f"Stats input dim: {stats_input_dim}")
            print(f"d_model: {d_model}")
            self.stats_proj = nn.Sequential(
                nn.Linear(stats_input_dim, d_model//2),
                nn.ReLU(),
                nn.Linear(d_model//2, d_model)
            )
        if use_fft:
            # TODO: it's hardcoded right now, will need to change later
            combined_dim += d_model
            self.fft_proj = nn.Sequential(
                nn.Linear(30, d_model//2),
                nn.ReLU(),
                nn.Linear(d_model//2, d_model)
            )

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes)
        )

        self.cross_attention = CrossAttention(d_model, nhead)

    def extract_statistical_features(self, x_seq):
        """
        Input: x_seq shape (batch, seq_len, channels)
        Output: shape (batch, channels * 6)
        """
        # Get stats for all channels
        mean = x_seq.mean(dim=1)  # (batch, channels)
        std = x_seq.std(dim=1)    # (batch, channels)
        max_ = x_seq.max(dim=1).values  # (batch, channels)
        min_ = x_seq.min(dim=1).values  # (batch, channels)
        
        # Fix skewness and kurtosis calculations
        centered = x_seq - mean.unsqueeze(1)  # (batch, seq_len, channels)
        skewness = centered.pow(3).mean(dim=1) / (std + 1e-5).pow(3)  # (batch, channels)
        kurt = centered.pow(4).mean(dim=1) / (std + 1e-5).pow(4)      # (batch, channels)
        
        # All tensors should now be (batch, channels)
        stats = torch.cat([mean, std, max_, min_, skewness, kurt], dim=1)
        return stats  # (batch, channels * 6)

    def extract_fft_features(self, x_seq, k=10):
        """
        Input: x_seq shape (batch, seq_len, 3)
        Output: shape (batch, 3 * k)
        """
        # rfft returns half the spectrum (real), useful for magnitude features
        fft = torch.fft.rfft(x_seq, dim=1)  # shape: (batch, freq_bins, 3)
        power = torch.abs(fft)             # magnitude spectrum

        # Take top-k frequency magnitudes
        topk_vals, _ = power.topk(k, dim=1)  # (batch, k, 3)
        return topk_vals.flatten(start_dim=1)  # (batch, 3 * k)

    def apply_butterworth_filter(self, x_seq, cutoff=10.0, fs=100.0, order=4):
        """
        Apply Butterworth low-pass filter to remove high-frequency noise
        
        Args:
            x_seq: Input tensor (batch, seq_len, channels)
            cutoff: Cutoff frequency in Hz
            fs: Sampling frequency in Hz
            order: Filter order
        """
        # Design filter
        nyq = 0.5 * fs
        normalized_cutoff = cutoff / nyq
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        
        # Convert to numpy for filtering
        x_np = x_seq.cpu().numpy()
        x_filtered = np.zeros_like(x_np)
        
        # Apply filter to each channel of each batch
        for i in range(x_np.shape[0]):  # batch
            for j in range(x_np.shape[2]):  # channels
                x_filtered[i, :, j] = signal.filtfilt(b, a, x_np[i, :, j])
        
        return torch.from_numpy(x_filtered).to(x_seq.device)

    def apply_median_filter(self, x_seq, kernel_size=5):
        """
        Apply median filter to remove spike noise
        
        Args:
            x_seq: Input tensor (batch, seq_len, channels)
            kernel_size: Size of the median filter kernel
        """
        # Convert to numpy for filtering
        x_np = x_seq.cpu().numpy()
        x_filtered = np.zeros_like(x_np)
        
        # Apply filter to each channel of each batch
        for i in range(x_np.shape[0]):  # batch
            for j in range(x_np.shape[2]):  # channels
                x_filtered[i, :, j] = signal.medfilt(x_np[i, :, j], kernel_size)
        
        return torch.from_numpy(x_filtered).to(x_seq.device)

    def extract_robust_statistical_features(self, x_seq):
        """
        Extract robust statistical features less sensitive to noise
        
        Args:
            x_seq: Input tensor (batch, seq_len, channels)
        """
        # Median (more robust than mean)
        median = torch.median(x_seq, dim=1).values
        
        # IQR (more robust than std)
        q75 = torch.quantile(x_seq, 0.75, dim=1)
        q25 = torch.quantile(x_seq, 0.25, dim=1)
        iqr = q75 - q25
        
        # Trimmed mean (removes outliers)
        sorted_seq, _ = torch.sort(x_seq, dim=1)
        trim_size = int(0.1 * x_seq.shape[1])  # 10% trim
        trimmed_mean = torch.mean(sorted_seq[:, trim_size:-trim_size, :], dim=1)
        
        # MAD (Median Absolute Deviation)
        mad = torch.median(torch.abs(x_seq - median.unsqueeze(1)), dim=1).values
        
        # Combine features
        stats = torch.cat([median, iqr, trimmed_mean, mad, q25, q75], dim=1)
        return stats

    def forward(self, x_seq, x_meta=None):
        """
        Forward pass with noise reduction
        """
        # Apply noise reduction
        x_seq = self.apply_butterworth_filter(x_seq)
        x_seq = self.apply_median_filter(x_seq)
        
        if self.use_stats:
            x_stats = self.extract_robust_statistical_features(x_seq)
        if self.use_fft:
            x_fft = self.extract_fft_features(x_seq)
            
        if self.use_vec_mag:
            magnitude = torch.norm(x_seq, dim=2, keepdim=True)
            x_seq = torch.cat([x_seq, magnitude], dim=2)        # (batch, seq_len, 4)

        # let's assume use_vec_mag is True...
        x = x_seq.transpose(1, 2)        # (batch, 4, seq_len) : transpose axes
        x = self.normalize(x)            # (batch, 4, seq_len) : normalize across axes
        x = x.transpose(1, 2)            # (batch, seq_len, 4) : transpose axes back
        
        x = self.seq_proj(x)             # (batch, seq_len, d_model) : project to d_model
        x = self.pos_encoder(x)          # (batch, seq_len, d_model)

        x = x.permute(1, 0, 2)          # (seq_len, batch, d_model) : permute axes, would be unecessary when batch_first=True
        x = self.transformer_encoder(x) # (seq_len, batch, d_model)
        x = x.mean(dim=0)               # (batch, d_model) : dim=1 if batch_first=True

        if self.use_metadata:
            assert x_meta is not None, "Metadata expected but not provided"
            meta = self.meta_proj(x_meta)
            x = torch.cat([x, meta], dim=1)
        
        if self.use_stats:
            x_stats = self.stats_proj(x_stats)
            # x_stats = self.cross_attention(x, x_stats)
            x = torch.cat([x, x_stats], dim=1)
        if self.use_fft:
            x_fft = self.fft_proj(x_fft)
            x = torch.cat([x, x_fft], dim=1)

        return self.classifier(x)
