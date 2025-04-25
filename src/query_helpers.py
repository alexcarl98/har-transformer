import yaml
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd


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
    def from_dict(cls, partition_dict: Dict[str, Any], partition_name: str) -> 'SubjectPartition':
        if partition_name not in partition_dict:
            raise ValueError(f"Partition '{partition_name}' not found in {partition_dict}")
            
        # Handle age_range tuple if present
        if 'age_range' in partition_dict[partition_name]:
            partition_dict[partition_name]['age_range'] = tuple(partition_dict[partition_name]['age_range'])
        
        return cls(**partition_dict[partition_name])
    
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
