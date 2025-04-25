import yaml
import os
from dataclasses import asdict
from typing import List, Dict, Literal
import pandas as pd
import hashlib
import json
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch
from config import DataConfig
from query_helpers import BiometricsData, SubjectPartition

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

        # print(f"{self.data_config.test_on_sensors=}")
        # print(f"{self.data_config.train_on_sensors=}")
        # print(f"{self.hash_value=}")
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
        
        return cls(data_config, bio_data, data_config.partitions.mapping, data_config.partitions.min_sample_count)

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
    data_loader = GeneralDataLoader.from_yaml('config.yml')
    

    # X, y = data_loader.get_Xy('train')
    # print(f"{X.shape=}")
    # print(f"{y.shape=}")
    # print(f"{y=}")

    # har_dataset = data_loader.get_har_dataset('train')
    # print(f"{har_dataset=}")
    # print(f"{har_dataset.X.shape=}")
    # print(f"{har_dataset.y.shape=}")
    # print(f"{har_dataset.y=}")