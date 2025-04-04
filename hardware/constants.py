RANDOM_SEED = 42

# ===== Raw Data Processing =====
dataset_numbers = ['001', '002', '004', '008','010','011','012',
                   '013','015','016','017', '018', '019', '020',
                   '021','022','024','025','026','027', '028',
                   '029', '030', '031', '032', '033', '034','035',
                   '036', '037', '038', '039', '040', '041']
data_dir = "HAR_data/"
URL_BASE = "https://raw.githubusercontent.com/Har-Lab/HumanActivityData/refs/heads/main/data/labeled_activity_data/"


# ===== Intermediate Data Processing =====
## dataframe processing
sensor_loc = 'waist'
FEATURES_COL = [f'{sensor_loc}_x', f'{sensor_loc}_y', f'{sensor_loc}_z']
LABELS_COL = ['activity']
TIME_COL = 'time'
## feature extraction
WINDOW_SIZE = 100   # 100Hz
STRIDE = 95

# ===== Model Training =====
## model training
TEST_SIZE = 0.2
BATCH_SIZE = 64
WEIGHT_DECAY = 0.0
EPOCHS = 2
LEARNING_RATE = 1e-3