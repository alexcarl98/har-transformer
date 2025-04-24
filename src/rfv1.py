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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from config import Config
from data import GeneralDataLoader

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

def print_classification_results(y_test, y_pred):
    """Print classification report and return confusion matrix."""
    print("\nClassification Report (Raw Data):")
    print(classification_report(y_test, y_pred))
    # return confusion_matrix(y_test, y_pred)


def train_rf_with_grid_search(X, y, args):
    """
    Train Random Forest with GridSearchCV using a minimal approach
    """
    # Minimal parameter grid
    param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Initialize base model with minimal settings
    base_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=args.random_seed,
        verbose=1,
        n_jobs=1  # Force single core for base estimator
    )
    
    # Initialize GridSearchCV with minimal settings
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        verbose=1,
        n_jobs=1,  # Force single core for grid search
        error_score='raise'
    )
    
    try:
        # Fit GridSearchCV
        print("Starting Grid Search...")
        grid_search.fit(X, y)
        
        print("\nBest parameters found:")
        print(grid_search.best_params_)
        print("\nBest cross-validation score:")
        print(f"F1-macro: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
        
    except Exception as e:
        print(f"Error during grid search: {str(e)}")
        print("Falling back to default Random Forest...")
        
        # Fallback to basic model if grid search fails
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=args.random_seed,
            verbose=1,
            n_jobs=1  # Force single core
        )
        rf_model.fit(X, y)
        return rf_model

def main():
    config = Config.from_yaml("config.yml")
    data_loader = GeneralDataLoader.from_yaml(config.get_data_config_path())
    args = config.random_forest
    try:
        print("Loading preprocessed datasets...")
        train_data = data_loader.get_har_dataset('train')
        val_data = data_loader.get_har_dataset('val')
        rf_train_data = train_data.combine_with(val_data)
        test_data = data_loader.get_har_dataset('test')
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

    # Save the processed data to a pickle file
    print(f"Training:")
    # rf_model = train_rf_with_grid_search(rf_X, rf_y, args)
    rf_model = RandomForestClassifier(n_estimators=args.n_estimators, 
                                      random_state=42, 
                                      max_depth=args.max_depth,
                                      min_samples_split=args.min_samples_split,
                                      verbose=2)
    rf_model.fit(rf_X, rf_y)
    print(f"Training complete")

    print(f"Testing:")
    rf_X_test, rf_y_test = process_for_rf(test_data)
    rf_y_pred = rf_model.predict(rf_X_test)

    print_classification_results(rf_y_test, rf_y_pred)

if __name__ == "__main__":
    main()