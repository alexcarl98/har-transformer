import yaml
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Union
import torch
import torch.nn as nn
import numpy as np
import math

@dataclass
class DataConfig:
    data_dir: str = "raw_data/"
    out_data_dir: str = "processed_data/"
    sensor_loc: List[str] = field(default_factory=lambda: ["waist", "ankle", "wrist"])
    ft_col: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    extracted_features: List[str] = field(default_factory=lambda: ["mean", "std"])
    classes: List[str] = field(default_factory=lambda: ["downstairs", "jog_treadmill", "upstairs", "walk_treadmill"])
    window_size: int = 100
    stride: int = 5
    batch_size: int = 2
    save_pkl: bool = False

    def __post_init__(self):
        # Several variable dimensions for this architecture, we need to calculate them here
        self.n_features = len(self.ft_col) * len(self.sensor_loc)
        
        self.stats_dim = len(self.extracted_features)
        self.stats_dim += 1 if "skewness" in self.extracted_features else 0
        self.stats_dim += 1 if "kurt" in self.extracted_features else 0
        self.stats_dim *= self.n_features

        print(f"{self.stats_dim=}")
        print(f"{self.n_features=}")
        self.n_classes = len(self.classes)
        self.n_window = self.window_size
        self.n_stride = self.stride
    
    @property
    def data_shape(self):
        return (self.batch_size, self.window_size, self.n_features)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Handle odd-dimensional input properly
        pe[:, 0::2] = torch.sin(position * div_term[:d_model//2])
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:(d_model-1)//2])
            pe[:, -1] = torch.sin(position.squeeze(-1) * div_term[-1])
            
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class TorchStatsPipeline(nn.Module):
    def __init__(self, attributes: List[str], n_features: int):
        super(TorchStatsPipeline, self).__init__()
        # self.config = config
        self.n_features = n_features
        self.stats_dim = len(attributes)
        self.stats_dim += 1 if "skewness" in attributes else 0
        self.stats_dim += 1 if "kurt" in attributes else 0
        self.stats_dim *= self.n_features

        self.pipeline = self.create_pipeline(attributes)
        # Add normalization for both raw and statistical features
        self.stats_norm = nn.LayerNorm(self.get_feature_dim())
        
    def get_feature_dim(self):
        """Calculate total feature dimension"""
        return self.stats_dim

    def create_pipeline(self, attributes: List[str]):
        pipeline = []
        for attr in attributes:
            if attr == 'mean':
                pipeline.append(('mean', lambda x, d=1: x.mean(dim=d)))
            elif attr == 'std':
                pipeline.append(('std', lambda x, d=1: x.std(dim=d)))
            elif attr == 'max':
                pipeline.append(('max', lambda x, d=1: x.max(dim=d).values))
            elif attr == 'min':
                pipeline.append(('min', lambda x, d=1: x.min(dim=d).values))
            elif attr == 'range':
                pipeline.append(('range', lambda x, d=1: x.max(dim=d).values - x.min(dim=d).values))
            elif attr == 'median':
                pipeline.append(('median', lambda x, d=1: x.median(dim=d).values))
            elif attr == 'skewness':
                def skewness(x, d=1):
                    mean = x.mean(dim=d, keepdim=True)
                    std = x.std(dim=d, keepdim=True)
                    skew = ((x - mean) ** 3).mean(dim=d) / (std ** 3)
                    return skew  # Will return (batch_size, 3) for x,y,z axes
                pipeline.append(('skewness', skewness))
            elif attr == 'kurt':
                def kurtosis(x, d=1):
                    mean = x.mean(dim=d, keepdim=True)
                    std = x.std(dim=d, keepdim=True)
                    kurt = ((x - mean) ** 4).mean(dim=d) / (std ** 4)
                    return kurt  # Will return (batch_size, 3) for x,y,z axes
                pipeline.append(('kurtosis', kurtosis))
            elif attr == 'q75':
                pipeline.append(('q75', lambda x, d=1: torch.quantile(x, 0.75, dim=d)))
            elif attr == 'q25':
                pipeline.append(('q25', lambda x, d=1: torch.quantile(x, 0.25, dim=d)))
            elif attr == 'iqr':
                pipeline.append(('iqr', lambda x, d=1: torch.quantile(x, 0.75, dim=d) - 
                                                     torch.quantile(x, 0.25, dim=d)))
            elif attr == 'mad':
                pipeline.append(('mad', lambda x, d=1: (x - x.mean(dim=d, keepdim=True)).abs().mean(dim=d)))
            elif attr == 'zero_crossing':
                def zero_crossing(x, d=1):
                    signs = x.sign()
                    sign_changes = signs[:, 1:] * signs[:, :-1]
                    crossings = (sign_changes == -1.0).sum(dim=1)
                    return crossings
                pipeline.append(('zero_crossing', zero_crossing))
        return pipeline

    def forward(self, x: torch.Tensor, return_dict: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply all statistical features to the input tensor
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            return_dict: If True, returns dictionary of features
            
        Returns:
            If return_dict: Dictionary of statistical features
            Else: Tensor of shape (batch_size, num_statistical_features)
        """
        # Calculate statistical features
        features = {}
        for name, func in self.pipeline:
            feat = func(x)
            if len(feat.shape) > 2:
                feat = feat.reshape(feat.shape[0], -1)
            features[name] = feat
            
        if return_dict:
            return features
            
        # Concatenate and normalize statistical features
        stats_features = torch.cat(list(features.values()), dim=-1)
        stats_features = self.stats_norm(stats_features)
            
        return stats_features

class HybridTransformer(nn.Module):
    def __init__(self, seq_length, args):
        super().__init__()
        feature_dim, stats_dim = args.n_features, args.stats_dim
        self.raw_norm = nn.LayerNorm(feature_dim)
        self.pos_encoder = PositionalEncoding(feature_dim)
        self.stats_pipeline = TorchStatsPipeline(args.extracted_features, args.n_features)
        # Process raw sequence
        self.seq_transformer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=1,  # Changed to 1 since feature_dim=3 can't be divided by 3
            dim_feedforward=64,
            batch_first=True
        )
        
        # Combine sequence and statistical features
        self.fusion_layer = nn.Linear(feature_dim + stats_dim, args.n_classes)
        
    def forward(self, raw_seq):
        # Add positional encoding to raw sequence
        stats_features = self.stats_pipeline(raw_seq)
        raw_seq = self.raw_norm(raw_seq)
        seq_encoded = self.pos_encoder(raw_seq)
        
        # Process raw sequence
        seq_features = self.seq_transformer(seq_encoded)
        
        # Get sequence representation
        seq_repr = seq_features.mean(dim=1)  # (batch_size, feature_dim)
        
        # Concatenate with statistical features
        combined = torch.cat([seq_repr, stats_features], dim=-1)
        
        # Fuse features
        output = self.fusion_layer(combined)
        return output

def test_pipeline():
    config = DataConfig(
        sensor_loc = ["waist", "wrist"],
        ft_col = ["x", "y", "z"],
        extracted_features=["mean", "std", "max", "min", 
                                        #   "range", 
                                        #   "median", 
                                        #   "skewness", 
                                          "kurt", 
                                          "q75", 
                                          "q25", 
                                          "iqr", 
                                          "mad", "zero_crossing"])
    
    # NOTE: I will eventually use this but for simplicity, I will use a smaller example
    # example_data_sz = (config.batch_size,config.window_size,config.n_features)
    # example_data_sz = (2,8,6)
    # Example data
    data = torch.rand(size=config.data_shape, dtype=torch.float32)
    print(f"{data.shape=}")
    # data = torch.tensor([
    #     [[1, 2, 1],
    #      [4, 5, 6],
    #      [1, 8, 9],
    #      [-1, 8, 9],
    #      [1, 8, 9],
    #      [4, 8, 9],
    #      [1, 8, 9],
    #      [-1, 5, 6]],

    #     [[1, 2, 1],
    #      [1, 5, 6],
    #      [2, 8, 9],
    #      [2, 8, 9],
    #      [1, 5, 6],
    #      [1, 5, 6],
    #      [2, 8, 9],
    #      [1, 5, 6]]
    # ], dtype=torch.float32)
    # print(f"{data.shape=}")
    # exit()
    
    # pipeline = TorchStatsPipeline(config)
    # stats_features = pipeline(data)
    
    # print("Statistical features shape:", stats_features.shape)
    
    # Create and test transformer
    model = HybridTransformer(
        seq_length=data.size(1),
        args=config
    )
    
    output = model(data)
    print("Transformer output shape:", output.shape)
    
    # Print detailed features
    # features_dict = pipeline(data, return_dict=True)
    # for name, value in features_dict.items():
    #     print(f"\n{name}:")
    #     print(value)

if __name__ == "__main__":
    test_pipeline()