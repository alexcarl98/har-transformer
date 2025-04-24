from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
import os
from datetime import datetime
import hashlib
import json


@dataclass
class RFConfig:
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1

@dataclass
class TConfig:
    extracted_features: List[str] = field(default_factory=lambda: ["mean", "std"])
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

@dataclass
class WandBConfig:
    mode: str
    entity: str
    project: str
    model_versioning: bool

@dataclass
class OutputPathsConfig:
    base_path: str
    data_settings: str
    run_id: str = None
    
    def __post_init__(self):
        # Generate unique run ID if not provided
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create run directory structure
        self.run_dir = os.path.join(self.base_path, f"run_{self.run_id}")
        self.wandb_dir = os.path.join(self.run_dir, "wandb")
        self.plots_dir = os.path.join(self.run_dir, "plots")
        self.models_dir = os.path.join(self.run_dir, "models")
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        
        # Create all directories
        for dir_path in [self.run_dir, self.wandb_dir, self.plots_dir, 
                        self.models_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_plot_path(self, model_name: str, plot_name: str) -> str:
        """Get path for a specific plot"""
        return os.path.join(self.plots_dir, f"{model_name}_{plot_name}.png")
    
    def get_model_path(self, model_name: str) -> str:
        """Get path for saving model weights"""
        return os.path.join(self.models_dir, f"{model_name}.pkl")
    
    def get_metrics_path(self, model_name: str) -> str:
        """Get path for saving model metrics"""
        return os.path.join(self.metrics_dir, f"{model_name}_metrics.json")

@dataclass
class Config:
    output_paths: OutputPathsConfig
    models_tested: List[str]
    evaluation_metrics: List[str]
    wandb: WandBConfig
    data: DataConfig
    random_forest: Optional[RFConfig] = None
    transformer: Optional[TConfig] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create output paths configuration
        output_paths = OutputPathsConfig(base_path=config_dict['path'], data_settings=config_dict['data_settings'])
        
        # Rest of the configuration loading...
        wandb_config = WandBConfig(**config_dict['wandb'])
        data_config = DataConfig(**config_dict['data'])
        
        model_configs = {}
        
        if 'transformer' in config_dict['models_tested']:
            model_configs['transformer'] = TConfig(**config_dict['transformer'])
        
        if 'random_forest' in config_dict['models_tested']:
            model_configs['random_forest'] = RFConfig(**config_dict['random_forest'])
        
        return cls(
            output_paths=output_paths,
            models_tested=config_dict['models_tested'],
            evaluation_metrics=config_dict['evaluation_metrics'],
            wandb=wandb_config,
            data=data_config,
            **model_configs
        )
    

    def get_transformer_params(self):
        return {
            'model_name': self.transformer.base_model,
            'freeze': self.transformer.freeze,
            'batch_size': self.transformer.batch_size,
            'n_epochs': self.data.epochs,
            'learning_rate': self.transformer.learning_rate,
        }