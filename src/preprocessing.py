import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from scipy.fft import fft
from constants import WINDOW_SIZE, STRIDE, FEATURES_COL, LABELS_COL, TIME_COL, RANDOM_SEED, TEST_SIZE, SZ_META_DATA
from sklearn.model_selection import train_test_split
π = np.pi

def encode_labels(y):
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)  # Converts to array of integers like [0, 1, 0, 2, ...]
    encoder_dict = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    decoder_dict = {idx: label for label, idx in encoder_dict.items()}
    return y_int, encoder_dict, decoder_dict

def extract_window_signal_features(window):
    fft_values = fft(window)
    fft_mag = np.abs(fft_values)[:WINDOW_SIZE//2]

    mean_mag = list(np.mean(window, axis=0))
    std_mag = list(np.std(window, axis=0))

    freq_mean = list(np.mean(fft_mag, axis=0))
    freq_std = list(np.std(fft_mag, axis=0))
    freq_energy = list(np.mean(fft_mag**2, axis=0))

    extracted = [*mean_mag, *std_mag, *freq_mean, *freq_std, *freq_energy]
    assert len(extracted) == SZ_META_DATA
    return extracted

def load_and_process_data(file_path):
    # Load CSV
    df = pd.read_csv(file_path, parse_dates=[TIME_COL])

    # Optional: Sort by time if needed
    df = df.sort_values(TIME_COL)

    # Identify contiguous class blocks
    df['class_change'] = (df[LABELS_COL] != df[LABELS_COL].shift()).cumsum()

    # Store results
    X_windows = []
    X_meta = []
    y_labels = []

    # Process each contiguous block
    for _, group in df.groupby('class_change'):
        if len(group) >= WINDOW_SIZE:
            features = group[FEATURES_COL].values
            label = group[LABELS_COL].iloc[0]
            
            # Sliding window
            for i in range(0, len(features) - WINDOW_SIZE + 1, STRIDE):
                window = features[i:i+WINDOW_SIZE]
                meta_data = extract_window_signal_features(window)

                X_windows.append(window)
                X_meta.append(meta_data)
                y_labels.append(label)

    # Convert to array
    X = np.array(X_windows)  # shape: (num_windows, 5, 3)
    X_meta = np.array(X_meta)  # (n_windows, number_of_meta_features)
    y = np.array(y_labels)                      # (n_windows,)
    return X, X_meta, y

def split_data(X, X_meta, y_encoded, test_size=TEST_SIZE):
    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=test_size, 
        random_state=RANDOM_SEED, stratify=y_encoded)

    X_train, X_test = X[idx_train], X[idx_test]
    X_meta_train, X_meta_test = X_meta[idx_train], X_meta[idx_test]
    y_train, y_test = y_encoded[idx_train], y_encoded[idx_test]

    return X_train, X_meta_train, y_train, X_test, X_meta_test, y_test






def zero_crossing(df, column_name):
    df[f"{column_name}_zero_crossing"] = df[column_name].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    return df


def derive_periodic_features(t, period):
    ω = (2*π) / period
    return np.sin(ω*t), np.cos(ω*t)

# def tensors_equal(new_data, old_data):
#     for i, (new, old) in enumerate(zip(new_data, old_data)):
#         if isinstance(new, torch.Tensor):
#             if not torch.equal(new, old):
#                 # Find where they differ
#                 differences = (new != old)
#                 for row in range(differences.shape[0]):
#                     for col in range(differences.shape[1]):
#                         if differences[row, col]:
#                             print(f"Difference in tensor {i}:")
#                             print(f"Row {row}, Column {col}")
#                             print(f"New value: {new[row, col]}")
#                             print(f"Old value: {old[row, col]}")
#                             print("---")
#                 return False
#         elif new != old:
#             print(f"Non-tensor difference in position {i}:")
#             print(f"New value: {new}")
#             print(f"Old value: {old}")
#             return False
#     return True