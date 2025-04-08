import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Literal

import numpy as np
import pandas as pd
import torch

@dataclass
class TConfig:
    """Configuration for the experiment."""
    data_dir: str = "raw_data/"
    output_dir: str = "doc/latex/figure/"
    random_seed: int = 42
    sensor_loc: List[str] = field(default_factory=lambda: ["waist", "ankle", "wrist"])
    window_size: int = 100
    stride: int = 10
    test_size: float = 0.4
    num_classes: int = 6
    batch_size: int = 16
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 20
    d_model: int = 128
    fc_hidden_dim: int = 128
    in_seq_dim: int = 3
    in_meta_dim: int = 3
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    load_model_path: str = ''

    def __postinit__(self):
        if self.load_model_path:
            assert os.path.exists(self.load_model_path), f"Model file {self.load_model_path} does not exist."
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


if __name__ == "__main__":
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    # print(config)
    print(config['transformer'])