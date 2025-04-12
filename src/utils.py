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


if __name__ == "__main__":
    config = TConfig.from_yaml("config.yml")
    print(config)
    print(config.figure_out_dir)
    print(config.weights_out_dir)