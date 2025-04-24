import yaml
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Literal
import pandas as pd
import hashlib
import json
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch
from simple_download import download_data

class HarDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def combine_with(self, other):
        X = torch.cat([self.X, other.X], dim=0)
        y = torch.cat([self.y, other.y], dim=0)
        result = HarDataset(X, y)
        return result
    
    @classmethod
    def decouple_combine(cls, har_list: list['HarDataset']):
        first = har_list[0]
        
        for i in range(1, len(har_list)):
            first = first.combine_with(har_list[i])
        return first

@dataclass
class BiometricsData:
    df: pd.DataFrame
    
    @classmethod
    def from_csv(cls, filepath: str = 'har_data/000_biometrics.csv') -> 'BiometricsData':
        """Load biometrics data from CSV file"""
        df = pd.read_csv(filepath)
        return cls(df)
    
    def query_subjects(self, **conditions) -> List[int]:
        """
        Query subjects based on multiple conditions. Supports both exact matches and callable conditions.
        
        Examples:
        >>> bio_data.query_subjects(sex=1, age=25)  # male, age 25
        >>> bio_data.query_subjects(acl_injury=1)    # subjects with ACL injury
        >>> bio_data.query_subjects(dominant_hand='Right')
        >>> bio_data.query_subjects(age=lambda x: x > 30)  # subjects over 30
        >>> bio_data.query_subjects(height=lambda x: 170 <= x <= 190)  # height between 170-190cm
        """
        mask = pd.Series(True, index=self.df.index)
        
        for column, value in conditions.items():
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in biometrics data")
            
            if callable(value):
                # Apply the callable function to the column
                mask &= self.df[column].apply(value)
            else:
                # Exact match comparison
                mask &= (self.df[column] == value)
            
        return self.df[mask]['ID'].tolist()
    
    def get_subject_data(self, subject_id: int) -> Optional[pd.Series]:
        """Get all biometrics data for a specific subject"""
        subject_data = self.df[self.df['ID'] == subject_id]
        return subject_data.iloc[0] if not subject_data.empty else None


    def get_subject_min_sample_window(self, subject_id: str, activities: List[str], threshold: int = 1) -> tuple[str, str, int]:
        """
        Find the activity with the smallest non-zero sample count for a specific subject.
        
        Args:
            subject_id: Subject ID as string (e.g., '001')
            activities: List of activity names without '_count' suffix
            threshold: Minimum number of samples to consider (default: 1)
            
        Returns:
            tuple: (subject_id, activity_name, count) or (subject_id, None, 0) if no valid counts found
        """
        activity_counts = [f"{activity}_count" for activity in activities]
        row = self.df[self.df['ID'] == int(subject_id)][activity_counts].iloc[0]
        
        # Filter activities with counts >= threshold
        valid_activities = [act for act in activities if row[f"{act}_count"] >= threshold]
        
        if not valid_activities:
            return subject_id, None, 0
            
        min_activity = min(valid_activities, key=lambda act: row[f"{act}_count"])
        min_count = row[f"{min_activity}_count"]
        
        return subject_id, min_activity, min_count
    
    def get_abs_min_sample_window(self, partition: List[str], activities: List[str], threshold: int = 1) -> tuple[str, str, int]:
        """Find the subject ID and activity with the smallest sample count across all subjects."""
        min_result = (None, None, float('inf'))
        
        for subject_id in partition:
            result = self.get_subject_min_sample_window(subject_id, activities, threshold)
            if result[2] < min_result[2]:  # Compare counts
                min_result = result
                
        return min_result


    def get_subject_windows(self, subject_id: str, activities: List[str], threshold: int = 1, 
                            wp: float = 1.0, balance: bool = True, 
                            min_count_override: Optional[int] = None) -> List[tuple[str, tuple[int, int]]]:
        """Get window indices for each activity of a subject, either balanced or unbalanced."""
        activity_counts = [f"{activity}_count" for activity in activities]
        row = self.df[self.df['ID'] == int(subject_id)][activity_counts].iloc[0]
        windows = {}
        
        if not balance:
            # Normal windows: each activity uses wp% of its total samples
            for activity in activities:
                count = row[f"{activity}_count"]
                if count >= threshold:
                    window_size = int(count * wp)
                    middle_idx = count // 2
                    half_window = window_size // 2
                    start_idx = middle_idx - half_window
                    end_idx = middle_idx + half_window
                    windows[activity] = (start_idx, end_idx)
        else:
            if min_count_override is not None:
                # Override windows: use partition's minimum count * wp
                window_size = int(min_count_override * wp)
            else:
                # Balanced windows: use subject's minimum count * wp
                _, _, min_count = self.get_subject_min_sample_window(subject_id, activities, threshold)
                window_size = int(min_count * wp)
            
            for activity in activities:
                count = row[f"{activity}_count"]
                if count >= threshold:
                    middle_idx = count // 2
                    half_window = window_size // 2
                    start_idx = middle_idx - half_window
                    end_idx = middle_idx + half_window
                    windows[activity] = (start_idx, end_idx)
        return windows


@dataclass
class SubjectPartition:
    # Required parameter
    activities_required: List[str]  # Must be specified for every partition
    
    # Optional parameters
    sensors_required: Optional[List[str]] = None  # ['ankle', 'wrist', 'waist']
    sensors_excluded: Optional[List[str]] = None  # for testing sensor invariance
    has_jogging: Optional[bool] = None  # specifically for jogging class
    min_samples_per_activity: Optional[float] = None  # minimum samples for each activity
    age_range: Optional[tuple[float, float]] = None
    injury_free: Optional[bool] = None
    
    def __post_init__(self):
        """Validate that activities_required contains valid activities"""
        valid_activities = {'downstairs', 'upstairs', 'walk_treadmill', 
                          'walk_mixed', 'walk_sidewalk', 'jog_treadmill'}
        invalid_activities = set(self.activities_required) - valid_activities
        if invalid_activities:
            raise ValueError(f"Invalid activities specified: {invalid_activities}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str, partition_name: str) -> 'SubjectPartition':
        """Load a partition configuration from a YAML file"""
        with open(yaml_path, 'r') as f:
            partitions = yaml.safe_load(f)
            
        if partition_name not in partitions:
            raise ValueError(f"Partition '{partition_name}' not found in {yaml_path}")
            
        # Handle age_range tuple if present
        if 'age_range' in partitions[partition_name]:
            partitions[partition_name]['age_range'] = tuple(partitions[partition_name]['age_range'])
            
        return cls(**partitions[partition_name])
    
    def get_subjects(self, bio_data: BiometricsData) -> List[int]:
        """Get subject IDs matching all specified criteria"""
        conditions = {}
        
        # Sensor requirements
        if self.sensors_required:
            for sensor in self.sensors_required:
                conditions[f'has_{sensor}'] = 1
                
        if self.sensors_excluded:
            for sensor in self.sensors_excluded:
                conditions[f'has_{sensor}'] = 0
        
        # Jogging activity requirement
        if self.has_jogging is False:
            conditions['jog_treadmill_count'] = 0
        elif self.has_jogging is True:
            conditions['jog_treadmill_count'] = lambda x: x > 0
            
        # Minimum samples per activity
        if self.min_samples_per_activity is not None:
            # Use activities_required directly since it's now a required parameter
            activities = self.activities_required.copy()
            
            # Remove jogging from requirements if has_jogging is False
            if self.has_jogging is False and 'jog_treadmill' in activities:
                activities.remove('jog_treadmill')
                
            # Add minimum sample requirements for each activity
            for activity in activities:
                conditions[f'{activity}_count'] = lambda x: x >= self.min_samples_per_activity
        
        # Age range filter
        if self.age_range is not None:
            min_age, max_age = self.age_range
            conditions['age'] = lambda x: min_age <= x <= max_age
            
        # Injury filter
        if self.injury_free:
            conditions['acl_injury'] = 0
            conditions['current_injuries'] = lambda x: pd.isna(x)
            conditions['past_injuries'] = lambda x: pd.isna(x)
            
        return bio_data.query_subjects(**conditions)

    def get_subjects_str(self, bio_data: BiometricsData) -> List[str]:
        """Get subject IDs matching all specified criteria"""
        subjects = self.get_subjects(bio_data)
        return [f"{subject:03d}" for subject in subjects]

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

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DataConfig':
        """Load data configuration from a YAML file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if 'data' not in config:
            raise ValueError(f"No 'data' section found in {yaml_path}")
            
        return cls(**config['data'])

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



def obtain_standard_partitions(bio_data: BiometricsData, activity_list: List[str], yaml_path: str) -> List[List[str]]:
    with open(yaml_path, 'r') as f:
        partitions = yaml.safe_load(f)

    ignore = ["cross_set_configs", 'data']
    parts = {}

    for key in partitions.keys():
        if key in ignore:
            continue
        parts[key] = SubjectPartition.from_yaml(
            yaml_path=yaml_path,
            partition_name=key
        ).get_subjects_str(bio_data)

    all_subjects = list(set([subject for partition in parts.values() for subject in partition]))
    _,_, min_sample_count = bio_data.get_abs_min_sample_window(all_subjects, activity_list)
    
    return parts, min_sample_count


class GeneralDataLoader:
    def __init__(self, data_config: DataConfig, bio_data: BiometricsData, subject_partitions: Dict[str, List[str]], min_sample_count: int = None):
        self.data_config = data_config
        self.parts = subject_partitions
        self.subject_partitions = list(subject_partitions.values())
        self.bio_data = bio_data
        self.encoder_dict = {label: idx for idx, label in enumerate(self.data_config.classes)}
        self.decoder_dict = {idx: label for idx, label in enumerate(self.data_config.classes)}
        self.num_classes = len(self.data_config.classes)
        self.hash_value = hashlib.sha256(
            json.dumps({
                'config': asdict(self.data_config),
                'partitions': self.subject_partitions
            }, sort_keys=True).encode()
        ).hexdigest()
        if self.data_config.processed_dir is not None:
            os.makedirs(self.data_config.processed_dir, exist_ok=True)

        if self.data_config.balance_setting == "min_sample" and min_sample_count is not None:
            self.min_sample_count = min_sample_count
        else:
            self.min_sample_count = None

        self.data_save_path = os.path.join(self.data_config.processed_dir, f"data-{self.hash_value}.pkl")
        if os.path.exists(self.data_save_path):
            print(f"Loading data from {self.data_save_path}")
            self.data = pickle.load(open(self.data_save_path, 'rb'))
        else:
            print(f"Processing new data config and saving to {self.data_save_path}")
            self.data = self.process_partitions()
            pickle.dump(self.data, open(self.data_save_path, 'wb'))

    def process_features_labels(self, subject_id: str, subject_windows: Dict[str, tuple[int, int]], 
                              sensor_locs: List[str]) -> tuple[List[np.ndarray], List[int]]:
        """Process features and labels for a single subject."""
        X, y = [], []
        file_path = os.path.join(self.data_config.raw_dir, f"{subject_id}.csv")
        feature_cols = [f'{sensor_loc}_{ft}' for ft in self.data_config.ft_col 
                       for sensor_loc in sensor_locs]
        
        # Load and sort data
        df = pd.read_csv(file_path, parse_dates=['time']).sort_values('time')
        df['class_change'] = (df['activity'] != df['activity'].shift()).cumsum()
        
        # Process each activity segment
        for _, group in df.groupby('class_change'):
            activity = group['activity'].iloc[0]
            if activity not in self.data_config.classes or len(group) < self.data_config.window_size:
                continue
                
            if activity in subject_windows:
                start_idx, end_idx = map(int, subject_windows[activity])  # Convert to integers
                features = group[feature_cols].values[start_idx:end_idx]
                
                # Sliding window
                for i in range(0, len(features) - self.data_config.window_size + 1, 
                             self.data_config.stride):
                    window = features[i:i + self.data_config.window_size]
                    X.append(window)
                    y.append(self.encoder_dict[activity])
        
        return X, y

    def process_partition(self, partition: List[str], sensor_locs: List[str]) -> tuple[np.ndarray, np.ndarray]:
        """Process all subjects in a partition."""
        window_args = {
            'activities': self.data_config.classes,
            'wp': self.data_config.window_center_percentage,
            'balance': self.data_config.balance_setting in ["subject", "min_sample"]
        }
        
        # Handle different balancing strategies
        if self.data_config.balance_setting == "min_sample":
            window_args['min_count_override'] = self.min_sample_count
        
        # Process each subject
        X_all, y_all = [], []
        for subject in partition:
            subject_windows = self.bio_data.get_subject_windows(subject, **window_args)
            for sensor in sensor_locs:
                try:

                    X, y = self.process_features_labels(subject, subject_windows, [sensor])
                    X_all.extend(X)
                    y_all.extend(y)
                except Exception as e:
                    print(f"{e=}")

        return np.array(X_all), np.array(y_all)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'GeneralDataLoader':
        
        data_config = DataConfig.from_yaml(yaml_path)
        bio_path = os.path.join(data_config.raw_dir, '000_biometrics.csv')
        bio_data = BiometricsData.from_csv(bio_path)
        parts, min_sample_count = obtain_standard_partitions(bio_data, data_config.classes, yaml_path)
        
        return cls(data_config, bio_data, parts, min_sample_count)

    def write_to_csv(self):
        """Write configuration information to CSV file."""
        file_path = os.path.join(self.data_config.processed_dir, "overview.csv")
        config_header, config_row = self.data_config.csv_info()
        num_partitions = len(self.subject_partitions)
        num_subjects = sum(len(partition) for partition in self.subject_partitions)
        
        # Combine all fields
        full_header = f"hash_value,{config_header},num_partitions,num_subjects"
        full_row = f"{self.hash_value},{config_row},{num_partitions},{num_subjects}"
        
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(f"{full_header}\n")
                f.write(f"{full_row}\n")
        else:
            with open(file_path, 'a') as f:
                f.write(f"{full_row}\n")

    def process_partitions(self) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Process all partitions and split into train/val/test sets."""
        datasets = {
            'train_X': [], 'train_y': [],
            'val_X': [], 'val_y': [],
            'test_X': [], 'test_y': []
        }
        
        for partition in self.subject_partitions:
            # Split subjects into train/val/test
            n_subjects = len(partition)
            indices = np.arange(n_subjects)
            temp_indices, test_indices = train_test_split(
                indices, test_size=self.data_config.test_size, random_state=42)
            
            val_size_adjusted = self.data_config.val_size / (1 - self.data_config.test_size)
            train_indices, val_indices = train_test_split(
                temp_indices, test_size=val_size_adjusted, random_state=42)
            
            # Process each split
            splits = {
                'train': (train_indices, self.data_config.train_on_sensors),
                'val': (val_indices, self.data_config.train_on_sensors),
                'test': (test_indices, self.data_config.test_on_sensors)
            }
            
            for split_name, (split_indices, sensors) in splits.items():
                split_partition = [partition[i] for i in split_indices]
                X, y = self.process_partition(split_partition, sensors)
                datasets[f'{split_name}_X'].append(X)
                datasets[f'{split_name}_y'].append(y)
        

        self.write_to_csv()
        # Concatenate all partitions
        return {
            'train': (np.concatenate(datasets['train_X']), np.concatenate(datasets['train_y'])),
            'val': (np.concatenate(datasets['val_X']), np.concatenate(datasets['val_y'])),
            'test': (np.concatenate(datasets['test_X']), np.concatenate(datasets['test_y']))
        }
    
    def get_Xy(self, split: Literal['train', 'val', 'test']) -> tuple[np.ndarray, np.ndarray]:
        X, y = self.data[split]
        return X, y
    
    def get_har_dataset(self, split: Literal['train', 'val', 'test']) -> HarDataset:
        X, y = self.get_Xy(split)
        return HarDataset(X, y)
    



# Example usage:
if __name__ == "__main__":
    data_loader = GeneralDataLoader.from_yaml('cfg_data.yml')

    X, y = data_loader.get_Xy('train')
    print(f"{X.shape=}")
    print(f"{y.shape=}")
    print(f"{y=}")

    # har_dataset = data_loader.get_har_dataset('train')
    # print(f"{har_dataset=}")
    # print(f"{har_dataset.X.shape=}")
    # print(f"{har_dataset.y.shape=}")
    # print(f"{har_dataset.y=}")