import torch
import torch.nn as nn
import math
from typing import Union, Dict
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


class TorchStatsPipeline(nn.Module):
    def __init__(self, attributes: list[str], n_features: int):
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

    def create_pipeline(self, attributes: list[str]):
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

class SensorPatches(nn.Module):
    def __init__(self, in_channels, projection_dim, patch_size, stride):
        super().__init__()
        self.projection = nn.Conv1d(
            in_channels=in_channels,
            out_channels=projection_dim,
            kernel_size=patch_size,
            stride=stride,
            padding='valid'
        )
    
    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        x = self.projection(x)  # (batch, projection_dim, n_patches)
        return x.transpose(1, 2)  # (batch, n_patches, projection_dim)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_patches):
        super().__init__()
        self.position_embedding = nn.Embedding(max_patches, d_model)
        
    def forward(self, x):
        # x: (batch, n_patches, d_model)
        positions = torch.arange(x.size(1), device=x.device)
        pos_encoding = self.position_embedding(positions)
        return x + pos_encoding

class AccelTransformerV1(nn.Module):
    def __init__(self, d_model=128, fc_hidden_dim=128, 
                 in_channels=3, in_meta_dim=3, nhead=4, 
                 num_layers=2, dropout=0.1, num_classes=6,
                 patch_size=16, stride=8, torch_stats_pipeline: TorchStatsPipeline = None):
        super().__init__()
        
        # Calculate expected sequence length after convolution
        self.max_patches = ((100 - patch_size) // stride) + 1
        
        # Project input sequences into patches
        self.patch_embedding = SensorPatches(
            in_channels=in_channels,
            projection_dim=d_model,
            patch_size=patch_size,
            stride=stride
        )
        
        # Make sure d_model is divisible by nhead
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        # Position encoding for the actual number of patches
        self.pos_encoder = LearnablePositionalEncoding(d_model, self.max_patches + 1)  # +1 for cls token
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Metadata projection
        if torch_stats_pipeline is not None:
            self.stats = torch_stats_pipeline
            stats_dim= self.stats.get_feature_dim()
            d_model += 6*stats_dim

            # self.meta_proj = 
            
            # nn.Linear(self.stats.get_feature_dim(), 3*self.stats.get_feature_dim())
            self.meta_proj = nn.Sequential(
                nn.Linear(self.stats.get_feature_dim(), 3*self.stats.get_feature_dim()),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(3*self.stats.get_feature_dim(), 6*self.stats.get_feature_dim())
            )
            
            # nn.Linear(self.stats.get_feature_dim(), 3*self.stats.get_feature_dim())
        else:
            self.stats = None
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, fc_hidden_dim),  # *2 for meta concat
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes)
        )

    def forward(self, x_seq, x_meta):
        # Print shapes for debugging
        batch_size = x_seq.size(0)
        # print(f"Input shape: {x_seq.shape}")
        
        # Create patches
        x = self.patch_embedding(x_seq)
        # print(f"After patching: {x.shape}")
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(f"After adding CLS token: {x.shape}")
        
        # Add position encoding
        x = self.pos_encoder(x)
        # print(f"After position encoding: {x.shape}")
        
        # Transform
        x = self.transformer_encoder(x)
        
        # Get class token output
        x = x[:, 0]
        
        # Process metadata
        if self.stats is not None:
            x_stats = self.stats(x_seq)
            x_stats = self.meta_proj(x_stats)
            x = torch.cat((x, x_stats), dim=1)
        # meta = self.meta_proj(x_meta)
        
        # Combine and classify
        # combined = torch.cat([x, meta], dim=1)
        # return self.classifier(combined)
        return self.classifier(x)
