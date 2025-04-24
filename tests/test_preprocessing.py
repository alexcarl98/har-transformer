import sys
import os
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent))

print(sys.path)
import pytest
import numpy as np
import pandas as pd
from src.preprocessing import (
    encode_labels,
    extract_window_signal_features,
    load_and_process_data,
    split_data,
    zero_crossing,
    derive_periodic_features
)
from src.exp.wip.constants import WINDOW_SIZE, SZ_META_DATA

@pytest.fixture
def sample_data():
    # Create a small sample DataFrame for testing
    times = pd.date_range(start='2024-01-01', periods=100, freq='100ms')
    data = {
        'timestamp': times,
        'waist_x': np.sin(np.linspace(0, 4*np.pi, 100)),
        'waist_y': np.cos(np.linspace(0, 4*np.pi, 100)),
        'waist_z': np.random.normal(0, 1, 100),
        'activity': ['walking'] * 50 + ['running'] * 50
    }
    return pd.DataFrame(data)

def test_encode_labels():
    labels = ['walking', 'running', 'walking', 'sitting']
    y_int, encoder_dict, decoder_dict = encode_labels(labels)
    
    assert len(y_int) == 4
    assert isinstance(encoder_dict, dict)
    assert isinstance(decoder_dict, dict)
    assert all(isinstance(x, np.integer) for x in y_int)
    assert decoder_dict[encoder_dict['walking']] == 'walking'

def test_extract_window_signal_features():
    # Create a sample window of shape (WINDOW_SIZE, 3)
    window = np.random.normal(0, 1, (WINDOW_SIZE, 3))
    
    features = extract_window_signal_features(window)
    
    assert len(features) == SZ_META_DATA
    assert isinstance(features, list)
    assert all(isinstance(x, (float, np.float64)) for x in features)

def test_load_and_process_data(sample_data, tmp_path):
    # Save sample data to temporary file
    temp_file = tmp_path / "test_data.csv"
    sample_data.to_csv(temp_file, index=False)
    
    X, X_meta, y = load_and_process_data(temp_file)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(X_meta, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[2] == 3  # 3 features (x, y, z)
    assert X.shape[1] == WINDOW_SIZE
    assert len(X) == len(y)
    assert len(X_meta) == len(y)

def test_split_data():
    # Create dummy data
    X = np.random.normal(0, 1, (100, WINDOW_SIZE, 3))
    X_meta = np.random.normal(0, 1, (100, SZ_META_DATA))
    y = np.array([0] * 50 + [1] * 50)
    
    X_train, X_meta_train, y_train, X_test, X_meta_test, y_test = split_data(X, X_meta, y)
    
    assert len(X_train) > len(X_test)
    assert len(X_meta_train) == len(y_train)
    assert len(X_meta_test) == len(y_test)
    # Check if stratification worked
    assert sum(y_train == 0) > 0 and sum(y_train == 1) > 0
    assert sum(y_test == 0) > 0 and sum(y_test == 1) > 0

def test_zero_crossing():
    df = pd.DataFrame({
        'signal': [1, 2, -1, -2, 1, 0, -1]
    })
    
    result = zero_crossing(df, 'signal')
    assert 'signal_zero_crossing' in result.columns
    assert all(x in [-1, 0, 1] for x in result['signal_zero_crossing'])

def test_derive_periodic_features():
    t = np.array([0, 0.5, 1.0])
    period = 2.0
    
    sin_vals, cos_vals = derive_periodic_features(t, period)
    
    assert len(sin_vals) == len(t)
    assert len(cos_vals) == len(t)
    assert np.allclose(sin_vals**2 + cos_vals**2, 1.0)  # Verify unit circle property