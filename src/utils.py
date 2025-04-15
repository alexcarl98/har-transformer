import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Literal

import numpy as np
import pandas as pd
import torch
from datetime import datetime

# "downstairs", "jog_treadmill", "upstairs", "walk_mixed", "walk_sidewalk", "walk_treadmill"
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
    classes: List[str] = field(default_factory=lambda: ["downstairs", "jog_treadmill", "upstairs", "walk_treadmill"])
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
        sweep_dict = {
            "program": "src/train.py",
            "method": "bayes",
            "metric": {
                "goal": "minimize",
                "name": "val_loss"
            },
            "parameters": {
                "batch_size": {
                    "distribution": "categorical",
                    "values": [8,16, 32, 64, 128]
                },
                "d_model": {
                    "distribution": "categorical",
                    "values": [8, 16, 32, 64, 128]
                },
                "dropout": {
                    "distribution": "uniform",
                    "min": 0.05,
                    "max": 0.2
                },
                "epochs": {
                    "distribution": "categorical",
                    "values": [10, 20, 30, 40, 50, 60, 70, 80]
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
                    "value": 50
                },
                "test_size": {
                    "value": 0.3
                },
                "weight_decay": {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 0.01
                },
                "window_size": {
                    "value": 100
                }
            },
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 10
            }
        }
        return sweep_dict

    @classmethod
    def save_sweep_config(cls, output_path: str = "sweep_config.yaml"):
        """Saves the wandb sweep configuration to a YAML file.
        
        Args:
            output_path (str): Path where to save the sweep configuration file.
        """
        sweep_dict = cls.create_sweep_dict()
        with open(output_path, "w") as f:
            yaml.dump(sweep_dict, f, default_flow_style=False)




@dataclass
class RandomForestConfig:
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
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1

    def __post_init__(self):
        self.encoder_dict = {label: idx for idx, label in enumerate(self.classes)}
        self.decoder_dict = {idx: label for idx, label in enumerate(self.classes)}

    @property
    def num_classes(self):
        return len(self.classes)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config['random_forest'])



if __name__ == "__main__":
    # Add these lines to generate the sweep config file
    TConfig.save_sweep_config("sweep_config.yaml")
    print("Sweep configuration saved to sweep_config.yaml")
    
    config = TConfig.from_yaml("config.yml")
    print(config)
    print(config.figure_out_dir)
    print(config.weights_out_dir)