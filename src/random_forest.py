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
from utils import RandomForestConfig

TARGET_COLUMN = 'activity'
SENSOR_LOCS=['ankle']
dataset_numbers = ['001', '004', '008','010','011','012',
                   '015','016','017', '018', '019', '020',
                   '021','022','024','025', '031', '032', '033', 
                   '035','036', '039', '041'
                   ]

ENCODER_DICT = {
    'downstairs': [1, 0, 0, 0, 0, 0],
    'jog_treadmill': [0, 1, 0, 0, 0, 0],
    'upstairs': [0, 0, 1, 0, 0, 0],
    'walk_mixed': [0, 0, 0, 1, 0, 0],
    'walk_sidewalk': [0, 0, 0, 0, 1, 0],
    'walk_treadmill': [0, 0, 0, 0, 0, 1]
}
ALT_ENCODER_DICT = {
    'downstairs': [1, 0, 0, 0],
    'jog_treadmill': [0, 1, 0, 0],
    'upstairs': [0, 0, 1, 0],
    'walk_treadmill': [0, 0, 0, 1],
}

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def process_activity_chunks(df, columns_to_keep, chunk_size=1500):
    """Process data into chunks for each activity with separators."""
    filtered_df = df[columns_to_keep].copy()
    # unique_activities = df[TARGET_COLUMN].unique()
    selected_chunks = []

    for activity in ALT_ENCODER_DICT.keys():  # Using ENCODER_DICT to ensure we process all activities
        try:
            activity_data = filtered_df[filtered_df[TARGET_COLUMN] == activity].copy()
            middle_index = len(activity_data) // 2
            start_index = max(0, middle_index - chunk_size//2)
            end_index = min(len(activity_data), middle_index + chunk_size//2)
            selected_data = activity_data.iloc[start_index:end_index].copy()

            if len(selected_data) < chunk_size:
                print(f"Warning: Activity '{activity}' has only {len(selected_data)} data points (less than {chunk_size})")

            selected_chunks.append(selected_data)
        except Exception as e:
            print(f"Error processing activity '{activity}': {e}")

    # Remove the final concatenation of selected_chunks[:-1] which drops the last chunk
    final_df = pd.concat(selected_chunks)
    final_df.reset_index(drop=True, inplace=True)
    return final_df

def print_dataset_info(df):
    """Print information about the dataset."""
    print("\nTotal rows in final dataset:", len(df))
    activity_counts = df['activity'].value_counts()
    print("\nActivity distribution:")
    print(activity_counts)

# def train_random_forest(X, y, test_size=0.3, random_state=42):
#     """Train random forest model and return predictions and metrics."""
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )

#     # Train model
#     print("Training Random Forest on raw accelerometer data...")
#     rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
#     rf_model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = rf_model.predict(X_test)
    
#     return rf_model, X_test, y_test, y_pred

def print_classification_results(y_test, y_pred):
    """Print classification report and return confusion matrix."""
    print("\nClassification Report (Raw Data):")
    print(classification_report(y_test, y_pred))
    # return confusion_matrix(y_test, y_pred)

def encode_activities(y, encoder_dict):
    """
    Encode activity labels using the provided encoder dictionary.
    Returns numpy array of encoded labels.
    """
    return np.array([encoder_dict[activity] for activity in y])


def standardize_columns(df, sensor_loc):
    """
    Standardize accelerometer column names to acc_x, acc_y, acc_z regardless of sensor location.
    """
    return df.rename(columns={
        f'{sensor_loc}_x': 'acc_x',
        f'{sensor_loc}_y': 'acc_y',
        f'{sensor_loc}_z': 'acc_z'
    })



def combine_datasets(dataset_numbers, train_datasets, sensor_locs=['ankle'], xft=False):
    """
    Load and combine multiple datasets, separating into train and test based on dataset numbers.
    Returns (X_train, X_test, y_train, y_test).
    """
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    feature_names = []
    # Track which activities are present in each dataset
    # activity_presence = {dataset: set() for dataset in dataset_numbers}
    
    for i, dataset_number in enumerate(dataset_numbers):
        path = f'har_data/{dataset_number}.csv'
        df = load_data(path)
        
        # Log activities in this dataset
        # activities_in_dataset = df['activity'].unique()
        # activity_presence[dataset_number].update(activities_in_dataset)
        # print(f"\nDataset {dataset_number} contains activities: {activities_in_dataset}")
        
        # Extract features and labels
        for sensor_loc in sensor_locs:
            columns_to_keep = ['time', f'{sensor_loc}_x', f'{sensor_loc}_y', f'{sensor_loc}_z', 'activity']
            processed_df = process_activity_chunks(df, columns_to_keep)
            processed_df = standardize_columns(processed_df, sensor_loc)
            # print(processed_df.head())
            if xft:
                features, labels = extract_features(processed_df, sensor_loc='acc')
                if i == 0:
                    feature_names = features.columns
            else:
                features = processed_df[['acc_x', 'acc_y', 'acc_z']].values
                labels = processed_df['activity'].values
                if i == 0:
                    feature_names = ['acc_x', 'acc_y', 'acc_z']
            
            # Split based on dataset number
            if dataset_number in train_datasets:
                train_features.append(features)
                train_labels.append(labels)
            else:
                test_features.append(features)
                test_labels.append(labels)
    
    # Print summary of missing activities
    # print("\nActivity presence summary:")
    # all_activities = set(ALT_ENCODER_DICT.keys())
    # for dataset, activities in activity_presence.items():
    #     missing = all_activities - activities
    #     if missing:
    #         print(f"Dataset {dataset} is missing activities: {missing}")
    
    # Combine train and test datasets separately
    X_train = np.vstack(train_features)
    y_train = np.concatenate(train_labels)
    X_test = np.vstack(test_features)
    y_test = np.concatenate(test_labels)
    
    # Print final class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nTraining set class distribution:")
    for activity, count in zip(unique, counts):
        print(f"{activity}: {count}")
    
    unique, counts = np.unique(y_test, return_counts=True)
    print("\nTest set class distribution:")
    for activity, count in zip(unique, counts):
        print(f"{activity}: {count}")
    
    return X_train, X_test, y_train, y_test, feature_names



def extract_features(data, sensor_loc='ankle',window_size=100, overlap=50):
    """Extract time and frequency domain features from accelerometer data"""
    features = []
    labels = []

    # Get unique activities (excluding separator activity)
    activities = data[data['activity'] != 'separator']['activity'].unique()

    for activity in activities:
        # Get data for this activity
        activity_data = data[data['activity'] == activity].copy()

        # Sliding window with overlap
        step = window_size - overlap
        for i in range(0, len(activity_data) - window_size, step):
            window = activity_data.iloc[i:i+window_size]

            # Extract features for each axis
            feature_row = {}

            for axis in [f'{sensor_loc}_x', f'{sensor_loc}_y', f'{sensor_loc}_z']:
                values = window[axis].values

                # Time domain features
                feature_row[f'{axis}_mean'] = np.mean(values)
                feature_row[f'{axis}_std'] = np.std(values)
                feature_row[f'{axis}_min'] = np.min(values)
                feature_row[f'{axis}_max'] = np.max(values)
                feature_row[f'{axis}_range'] = np.max(values) - np.min(values)
                feature_row[f'{axis}_median'] = np.median(values)
                feature_row[f'{axis}_mad'] = np.mean(np.abs(values - np.mean(values)))  # Mean absolute deviation
                feature_row[f'{axis}_skew'] = stats.skew(values)
                feature_row[f'{axis}_kurtosis'] = stats.kurtosis(values)

                # Frequency domain features
                fft_values = fft(values)
                fft_magnitude = np.abs(fft_values)[:window_size//2]

                # feature_row[f'{axis}_freq_max'] = np.argmax(fft_magnitude)
                feature_row[f'{axis}_freq_mean'] = np.mean(fft_magnitude)
                feature_row[f'{axis}_freq_std'] = np.std(fft_magnitude)
                feature_row[f'{axis}_freq_energy'] = np.sum(fft_magnitude**2) / window_size

            # Add additional features using multiple axes
            axis_x = window[f'{sensor_loc}_x'].values
            axis_y = window[f'{sensor_loc}_y'].values
            axis_z = window[f'{sensor_loc}_z'].values

            # Magnitude
            magnitude = np.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
            feature_row['magnitude_mean'] = np.mean(magnitude)
            feature_row['magnitude_std'] = np.std(magnitude)

            features.append(feature_row)
            labels.append(activity)

    # Convert to DataFrame
    feature_df = pd.DataFrame(features)

    return feature_df, pd.Series(labels)

def apply_pca(X_train, X_test, n_components=0.95):
    """
    Apply PCA to the training and test data.
    Args:
        X_train: Training features
        X_test: Test features
        n_components: Either number of components or variance ratio to preserve
    Returns:
        X_train_pca, X_test_pca, pca_model
    """
    # First standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Print variance explained
    print(f"\nNumber of components selected: {pca.n_components_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")
    
    return X_train_pca, X_test_pca, pca


def main():
    args = RandomForestConfig.from_yaml('config.yml')
    # Define dataset splits (example: 80% train, 20% test)
    np.random.seed(args.random_seed)
    
    all_datasets = np.array(dataset_numbers)
    np.random.shuffle(all_datasets)
    train_size = int((1-args.test_size) * len(all_datasets))
    train_datasets = all_datasets[:train_size]
    
    # Get train/test split data
    X_train, X_test, y_train, y_test, feature_names = combine_datasets(
        dataset_numbers, 
        train_datasets,
        sensor_locs=args.sensor_loc,
        xft=True
    )
    
    # Train model and get predictions
    print(f"Training on subjects: {train_datasets}")
    print(f"Testing on subjects: {[d for d in dataset_numbers if d not in train_datasets]}")
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    # X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test)
    # rf_model_pca = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf_model_pca.fit(X_train_pca, y_train)
    # y_pred_pca = rf_model_pca.predict(X_test_pca)
    
    # Get results
    print_classification_results(y_test, y_pred)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance)

if __name__ == "__main__":
    main()