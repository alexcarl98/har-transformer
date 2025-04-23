import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.fft import fft
from dataclasses import dataclass, field
import yaml
from typing import List
from model_version.v1 import HARWindowDatasetV1, TorchStatsPipeline
import os
from tqdm import tqdm
import torch


@dataclass
class RFConfig:
    """Configuration for the experiment."""
    data_dir: str = "raw_data/"
    out_data_dir: str = "processed_data/"
    output_dir: str = "doc/latex/figure/"
    model_out_dir: str = "models/"
    random_seed: int = 42
    sensor_loc: List[str] = field(default_factory=lambda: ["waist", "ankle", "wrist"])
    ft_col: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    extracted_features: List[str] = field(default_factory=lambda: ["mean", "std"])
    classes: List[str] = field(default_factory=lambda: ["downstairs", "jog_treadmill", "upstairs", "walk_treadmill"])
    window_size: int = 100
    stride: int = 10
    save_pkl: bool = False
    test_size: float = 0.4
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1

    def __post_init__(self):
        self.encoder_dict = {label: idx for idx, label in enumerate(self.classes)}
        self.decoder_dict = {idx: label for idx, label in enumerate(self.classes)}

    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config['random_forest'])


def process_for_rf(dataset: HARWindowDatasetV1):
    # Keep data on GPU if it's already there, or move it
    X = dataset.X.cuda() if torch.cuda.is_available() else dataset.X
    y = dataset.y
    _, window_size, _ = X.shape

    # Process entire batch at once instead of window by window
    # Compute basic statistics across the window dimension (dim=1)
    means = X.mean(dim=1)
    stds = X.std(dim=1)
    mins = X.min(dim=1)[0]
    maxs = X.max(dim=1)[0]
    ranges = maxs - mins
    medians = X.median(dim=1)[0]
    
    # Mean absolute deviation
    mad = (X - X.mean(dim=1, keepdim=True)).abs().mean(dim=1)
    
    # Frequency domain features (process in batches)
    fft_values = torch.fft.fft(X, dim=1)
    fft_magnitude = torch.abs(fft_values)[:, :window_size//2, :]
    freq_means = fft_magnitude.mean(dim=1)
    freq_stds = fft_magnitude.std(dim=1)
    freq_energy = (fft_magnitude.pow(2).sum(dim=1)) / window_size

    # Move everything to CPU and convert to numpy for DataFrame creation
    features_dict = {
        f'acc_{axis}_{stat}': tensor.cpu().numpy()
        for axis_idx, axis in enumerate(['x', 'y', 'z'])
        for stat, tensor in [
            ('mean', means[:, axis_idx]),
            ('std', stds[:, axis_idx]),
            ('min', mins[:, axis_idx]),
            ('max', maxs[:, axis_idx]),
            ('range', ranges[:, axis_idx]),
            ('median', medians[:, axis_idx]),
            ('mad', mad[:, axis_idx]),
            ('freq_mean', freq_means[:, axis_idx]),
            ('freq_std', freq_stds[:, axis_idx]),
            ('freq_energy', freq_energy[:, axis_idx])
        ]
    }

    # For skewness and kurtosis, we'll still use scipy as PyTorch doesn't have direct equivalents
    # But we'll process them in batches on CPU
    X_numpy = X.cpu().numpy()
    for axis_idx, axis in enumerate(['x', 'y', 'z']):
        features_dict[f'acc_{axis}_skew'] = stats.skew(X_numpy[:, :, axis_idx], axis=1)
        features_dict[f'acc_{axis}_kurtosis'] = stats.kurtosis(X_numpy[:, :, axis_idx], axis=1)

    # Create DataFrame from dictionary
    feature_df = pd.DataFrame(features_dict)
    
    return feature_df, pd.Series(y.numpy())



def main():
    args = RFConfig.from_yaml("config.yml")
    print(args)

    try:
        print("Loading preprocessed datasets...")
        train_data = torch.load(f'{args.out_data_dir}train_data.pt')
        val_data = torch.load(f'{args.out_data_dir}val_data.pt')
        rf_train_data = train_data.combine_with(val_data)
        test_data = torch.load(f'{args.out_data_dir}test_data.pt')
        print("Datasets loaded successfully!")
    except FileNotFoundError:
        raise FileNotFoundError("Preprocessed datasets not found. Run `src/train.py` first to generate preprocessed datasets.")

    print(f"{train_data.X.shape=}")
    print(f"{train_data.y.shape=}")
    print()
    print(f"{val_data.X.shape=}")
    print(f"{val_data.y.shape=}")
    print()
    print(f"{rf_train_data.X.shape=}")
    print(f"{rf_train_data.y.shape=}")
    print()
    print(f"{test_data.X.shape=}")
    print(f"{test_data.y.shape=}")

    rf_X, rf_y = process_for_rf(rf_train_data)
    print(rf_X.head())
    print(rf_y.head())

if __name__ == "__main__":
    main()