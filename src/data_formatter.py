import yaml
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Union
import torch
import torch.nn as nn
import numpy as np
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
    save_pkl: bool = False

class TorchStatsPipeline(nn.Module):
    def __init__(self, config: DataConfig):
        super(TorchStatsPipeline, self).__init__()
        self.config = config
        self.pipeline = self.create_pipeline(config.extracted_features)

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
                pipeline.append(('skewness', lambda x, d=1: ((x - x.mean(dim=d, keepdim=True)) ** 3).mean(dim=d) / 
                                                          (x.std(dim=d, keepdim=True) ** 3)))
            elif attr == 'kurt':
                pipeline.append(('kurtosis', lambda x, d=1: ((x - x.mean(dim=d, keepdim=True)) ** 4).mean(dim=d) / 
                                                          (x.std(dim=d, keepdim=True) ** 4)))
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
                    # For shape (batch, seq_len, features)
                    signs = x.sign()
                    # We need to detect when signs change from positive to negative or vice versa
                    # Multiply adjacent signs and look for -1 (indicating pos->neg or neg->pos)
                    sign_changes = signs[:, 1:] * signs[:, :-1]
                    # We want strictly -1 to catch actual crossings
                    crossings = (sign_changes == -1.0).sum(dim=1)
                    return crossings
                pipeline.append(('zero_crossing', zero_crossing))
        return pipeline

    def forward(self, x: torch.Tensor, return_dict: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply all statistical features to the input tensor
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            return_dict: If True, returns dictionary of features, else returns concatenated tensor
            
        Returns:
            If return_dict:
                Dictionary of feature names and their computed values
            Else:
                Tensor of shape (batch_size, num_features * num_statistical_features)
        """
        features = {}
        for name, func in self.pipeline:
            feat = func(x)
            # Handle different feature shapes
            if len(feat.shape) > 2:
                feat = feat.reshape(feat.shape[0], -1)
            features[name] = feat
        
        if return_dict:
            return features
        
        # Concatenate all features along the feature dimension
        return torch.cat(list(features.values()), dim=-1)

if __name__ == "__main__":
    config = DataConfig(extracted_features=["mean", "std", "max", "min", 
                                            "range", "median", "skewness", 
                                            "kurt", "q75", "q25", "iqr", 
                                            "mad", "zero_crossing"])
    # Fixed tensor definition with proper comma
    data = torch.tensor([
        [
        [1, 2, 1],
        [4, 5, 6],
        [1, 8, 9],
        [-1, 8, 9],
        [1, 8, 9],
        [4, 8, 9],
        [1, 8, 9],
        [-1, 5, 6]
        ],

        [[1, 2, 1],
        [1, 5, 6],
        [2, 8, 9],  # Added missing comma
        [2, 8, 9],  # Added missing comma
        [1, 5, 6],
        [1, 5, 6],
        [2, 8, 9],  # Added missing comma
        [1, 5, 6]]
    ], dtype=torch.float32)
    
    print(f"{data.shape=}")
    print(f"{data=}")
    print("\nComputing features...")
    
    pipeline = TorchStatsPipeline(config)
    features_tensor = pipeline(data)
    
    print(f"{features_tensor.shape=}")
    print(f"{features_tensor=}")
    # print("\nCombined features shape:", features_tensor.shape)
    
    # For inspection/debugging
    # features_dict = pipeline(data, return_dict=True)
    # print("\nIndividual features:")
    # for name, value in features_dict.items():
    #     print(f"\n{name}:")
    #     print(value)
    
    # Example neural network usage
    linear_layer = nn.Linear(features_tensor.shape[1], 64)
    output = linear_layer(features_tensor)