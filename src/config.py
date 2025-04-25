from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any
import yaml
import os
from datetime import datetime
import hashlib
import json
from simple_download import download_data
from query_helpers import SubjectPartition, BiometricsData
# from data import DataConfig
import warnings

@dataclass
class PartitionConfig:
    """Configuration for dataset partitions."""
    mapping: Dict[str, List[str]]
    min_sample_count: int

    @classmethod
    def from_yaml(cls, partition_dict: Dict[str, Any], activities: List[str], bio_data: BiometricsData) -> 'PartitionConfig':
        """
        Create PartitionConfig from YAML configuration.
        
        Args:
            partition_dict: Dictionary containing partition configurations
            activities: List of activity names to consider
            bio_data: BiometricsData instance for subject information
        """
        # Skip special configuration sections
        ignore_keys = {"cross_set_configs", "data"}
        
        # Process each partition
        processed_partitions = {}
        for partition_name, _ in partition_dict.items():
            if partition_name in ignore_keys:
                continue
                
            partition = SubjectPartition.from_dict(
                partition_dict=partition_dict,
                partition_name=partition_name
            )
            subjects = partition.get_subjects_str(bio_data)
            
            if not subjects:
                warnings.warn(f"Partition '{partition_name}' returned no subjects and will be skipped.")
                continue
                
            processed_partitions[partition_name] = subjects

        if not processed_partitions:
            raise ValueError("No valid partitions found. All partitions returned empty lists.")

        # Calculate minimum sample count across all subjects
        all_subjects = list(set(
            subject 
            for subjects in processed_partitions.values() 
            for subject in subjects
        ))
        _, _, min_sample_count = bio_data.get_abs_min_sample_window(
            all_subjects, 
            activities
        )

        return cls(
            mapping=processed_partitions,
            min_sample_count=min_sample_count
        )

    def get_partition(self, name: str) -> List[str]:
        """Get subjects for a specific partition."""
        if name not in self.mapping:
            raise ValueError(f"Partition '{name}' not found. Available partitions: {list(self.mapping.keys())}")
        return self.mapping[name]

    def get_all_subjects(self) -> List[str]:
        """Get all unique subjects across all partitions."""
        return list(set(
            subject 
            for subjects in self.mapping.values() 
            for subject in subjects
        ))

@dataclass
class DataConfig:
    dataset_url: str
    raw_dir: str
    processed_dir: str
    dataset_name: str
    balance_setting: Literal["min_sample", "subject", "None"]
    window_center_percentage: float
    train_on_sensors: List[str]
    test_on_sensors: List[str]
    classes: List[str]
    test_size: float
    val_size: float
    window_size: int
    stride: int
    ft_col: List[str]
    partitions: Optional[PartitionConfig] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DataConfig':
        """Load data configuration from a YAML file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if 'data' not in config:
            raise ValueError(f"No 'data' section found in {yaml_path}")
        
        # Create DataConfig first without partitions
        data_config = cls(**config['data'], partitions=None)
        
        # Then create and set partitions if partition section exists
        if 'partition' in config:
            bio_data = BiometricsData.from_csv(
                os.path.join(data_config.raw_dir, '000_biometrics.csv')
            )
            
            data_config.partitions = PartitionConfig.from_yaml(
                partition_dict=config['partition'],
                activities=data_config.classes,
                bio_data=bio_data
            )
        
        return data_config

    def __post_init__(self):
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
            download_data(self.dataset_url, self.raw_dir)

        """Validate configuration parameters"""
        # Validate balance_setting
        valid_balance_options = {"min_sample", "subject", "None"}
        if self.balance_setting not in valid_balance_options:
            raise ValueError(f"balance_setting must be one of {valid_balance_options}")
            
        # Validate sensors
        valid_sensors = {"ankle", "wrist", "waist"}
        if not set(self.train_on_sensors).issubset(valid_sensors):
            raise ValueError(f"train_on_sensors contains invalid sensors. Valid sensors: {valid_sensors}")
        if not set(self.test_on_sensors).issubset(valid_sensors):
            raise ValueError(f"test_on_sensors contains invalid sensors. Valid sensors: {valid_sensors}")
            
        # Validate split sizes
        if not 0 <= self.test_size <= 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 <= self.val_size <= 1:
            raise ValueError("val_size must be between 0 and 1")
        if self.test_size + self.val_size >= 1:
            raise ValueError("test_size + val_size must be less than 1")
    
        # Validate window size and stride
        if self.window_size <= 0:
            raise ValueError("window_size must be greater than 0")
        if self.stride <= 0:
            raise ValueError("stride must be greater than 0")
        if self.window_size <= self.stride:
            raise ValueError("window_size must be greater than stride")
    
        # Validate ft_col
        valid_ft_col = {"x", "y", "z", "vm", "time"}
        if not set(self.ft_col).issubset(valid_ft_col):
            raise ValueError(f"ft_col contains invalid features. Valid features: {valid_ft_col}")
        

    def csv_info(self):
        """Return formatted header and row for CSV output."""
        fields = {
            'balance_setting': self.balance_setting,
            'train_on_sensors': '"'+','.join(self.train_on_sensors)+'"',
            'test_on_sensors': '"'+','.join(self.test_on_sensors)+'"',
            'classes': '"'+','.join(self.classes)+'"',
            'test_size': self.test_size,
            'val_size': self.val_size,
            'window_size': self.window_size,
            'stride': self.stride,
            'ft_col': '"'+','.join(self.ft_col)+'"'
        }
        
        # Create header and row without self. references
        header = ','.join(fields.keys())
        row = ','.join(str(value) for value in fields.values())
        return header, row
    
    @property
    def in_seq_dim(self):
        return len(self.ft_col)

    @property
    def num_classes(self):
        return len(self.classes)
    
    def get_bio_data(self):
        bio_path = os.path.join(self.raw_dir, '000_biometrics.csv')
        return BiometricsData.from_csv(bio_path)

@dataclass
class RFConfig:
    extracted_features: List[str] = field(default_factory=lambda: ["mean", "std"])
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
    patch_size: int = 16
    kernel_stride: int = 8

@dataclass
class WandBConfig:
    mode: str
    entity: str
    project: str
    model_versioning: bool

@dataclass
class OutputPathsConfig:
    base_path: str
    run_id: str = None
    
    def __post_init__(self):
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

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
    
    def clean(self):
        for root, dirs, files in os.walk(self.base_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    # Check if directory is empty (no files and no non-empty subdirectories)
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                except OSError:
                    # Skip if directory can't be removed (e.g., permission issues or not empty)
                    continue
        

@dataclass
class Config:
    output_paths: OutputPathsConfig
    data: DataConfig
    models_tested: List[str]
    evaluation_metrics: List[str]
    wandb: WandBConfig
    random_forest: Optional[RFConfig] = None
    transformer: Optional[TConfig] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # TODO:put the loading code from `data.py` into here.

        # Create output paths configuration
        output_paths = OutputPathsConfig(base_path=config_dict['base_output_dir'])

        # Create data configuration
        data = DataConfig.from_yaml(yaml_path)

        
        # Rest of the configuration loading...
        wandb_config = WandBConfig(**config_dict['wandb'])
        # data_config = DataConfig(**config_dict['data'])
        
        model_configs = {}
        
        if 'transformer' in config_dict['models_tested']:
            model_configs['transformer'] = TConfig(**config_dict['transformer'])

        if 'random_forest' in config_dict['models_tested']:
            model_configs['random_forest'] = RFConfig(**config_dict['random_forest'])
        
        return cls(
            output_paths=output_paths,
            data=data,
            models_tested=config_dict['models_tested'],
            evaluation_metrics=config_dict['evaluation_metrics'],
            wandb=wandb_config,
            **model_configs
        )
    

    def get_transformer_params(self):
        return {
            'd_model': self.transformer.d_model,
            'fc_hidden_dim': self.transformer.fc_hidden_dim,
            'in_channels': self.data.in_seq_dim,
            'nhead': self.transformer.nhead,
            'num_layers': self.transformer.num_layers,
            'dropout': self.transformer.dropout,
            'num_classes': self.data.num_classes,
            'patch_size': self.transformer.patch_size,
            'kernel_stride': self.transformer.kernel_stride,
            'window_size': self.data.window_size,
            'extracted_features': self.transformer.extracted_features
        }
    
    def get_data_config_path(self):
        return self.output_paths.data_settings
    
    def get_data_loader_params(self):
        return{
            'data_config': self.data,
            'bio_data': self.data.get_bio_data(),
            'subject_partitions': self.data.partitions.mapping,
            'min_sample_count': self.data.partitions.min_sample_count
        }
    

    


if __name__ == "__main__":
    import data
    config = Config.from_yaml('config.yml')
    print(config.data.partitions)
    print(config.data.partitions.get_all_subjects())
