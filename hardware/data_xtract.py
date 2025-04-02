# Data Extraction
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==== Data Processing ====
FILE_PATH = "HAR_data/unproc.csv"
FEATURES_COL = ['waist_x', 'waist_y', 'waist_z']
LABELS_COL = ['activity']
TIME_COL = 'time'
WINDOW_SIZE = 5
STRIDE = 2
TEST_SIZE = 0.2
BATCH_SIZE = 64

# ==== Training ====
EPOCHS = 2
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_labels(y):
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)  # Converts to array of integers like [0, 1, 0, 2, ...]
    encoder_dict = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    decoder_dict = {idx: label for label, idx in encoder_dict.items()}
    return y_int, encoder_dict, decoder_dict

def load_and_process_data(file_path=FILE_PATH):
    # Load CSV
    df = pd.read_csv(file_path, parse_dates=[TIME_COL])

    # Optional: Sort by time if needed
    df = df.sort_values(TIME_COL)

    # Compute magnitude
    df['magnitude'] = np.sqrt((df[FEATURES_COL]**2).sum(axis=1))

    # Identify contiguous class blocks
    df['class_change'] = (df[LABELS_COL[0]] != df[LABELS_COL[0]].shift()).cumsum()

    # print(df)
    # Store results
    X_windows = []
    X_meta = []
    y_labels = []

    # Process each contiguous block
    for _, group in df.groupby('class_change'):
        if len(group) >= WINDOW_SIZE:
            features = group[FEATURES_COL].values
            mag_sqrd = group['magnitude'].values**2
            label = group[LABELS_COL].iloc[0]
            
            # Sliding window
            for i in range(0, len(features) - WINDOW_SIZE + 1, STRIDE):
                window = features[i:i+WINDOW_SIZE]
                mag_sum = np.sum(mag_sqrd[i:i+WINDOW_SIZE])

                X_windows.append(window)
                X_meta.append(mag_sum)
                y_labels.append(label)

    # Convert to array
    X = np.array(X_windows)  # shape: (num_windows, 5, 3)
    X_meta = np.array(X_meta).reshape(-1, 1)  # (n_windows, 1)
    y = np.array(y_labels)                      # (n_windows,)
    return X, X_meta, y

## Transformer Model
class HARWindowDataset(torch.utils.data.Dataset):
    def __init__(self, X, X_meta, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X_meta = torch.tensor(X_meta, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_meta[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

# === Transformer Model for HAR ===
class AccelTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1, num_classes=6):
        super().__init__()
        self.seq_embedding = nn.Linear(3, d_model)             # Input: (batch, seq_len, 3)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=128, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.energy_proj = nn.Linear(1, d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_seq, x_energy):
        """
        x_seq: (batch, seq_len=5, 3)
        x_energy: (batch, 1)
        """
        x = self.seq_embedding(x_seq)         # (batch, seq_len, d_model)
        x = self.pos_encoder(x)               # (batch, seq_len, d_model)

        x = x.permute(1, 0, 2)                # (seq_len, batch, d_model)
        x = self.transformer_encoder(x)       # (seq_len, batch, d_model)
        x = x.mean(dim=0)                     # (batch, d_model)

        e = self.energy_proj(x_energy)        # (batch, d_model)

        combined = torch.cat([x, e], dim=1)   # (batch, d_model * 2)

        return self.classifier(combined)      # (batch, num_classes)


X, X_meta, y = load_and_process_data()
y_int, encoder_dict, decoder_dict = encode_labels(y)

idx_train, idx_test = train_test_split(
    np.arange(len(X)), test_size=TEST_SIZE, 
    random_state=RANDOM_SEED, stratify=y_int)

X_train, X_test = X[idx_train], X[idx_test]
X_meta_train, X_meta_test = X_meta[idx_train], X_meta[idx_test]
y_train, y_test = y_int[idx_train], y_int[idx_test]

train_dataset = HARWindowDataset(X_train, X_meta_train, y_train)
test_dataset = HARWindowDataset(X_test, X_meta_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# === Model, loss, optimizer ===
model = AccelTransformer(num_classes=len(encoder_dict)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# === Training loop ===

print(f"{DEVICE=}")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x_seq, x_meta, y) in enumerate(train_loader):
        x_seq, x_meta, y = x_seq.to(DEVICE), x_meta.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x_seq, x_meta)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

        # Print occasionally
        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch+1}] Batch {batch_idx}: Loss = {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")