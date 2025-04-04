# Data Extraction
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from scipy.fft import fft
from constants import *

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==== Data Processing ====
FILE_PATH = "https://raw.githubusercontent.com/Har-Lab/HumanActivityData/refs/heads/main/data/labeled_activity_data/016_labeled.csv"

raw_data_urls = [f"{data_dir}{num}.csv" for num in dataset_numbers]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    freq_energy = list(np.mean(fft_mag**2, axis=0))

    extracted = [*mean_mag, *std_mag, *freq_energy]
    return extracted

def load_and_process_data(file_path=FILE_PATH):
    # Load CSV
    df = pd.read_csv(file_path, parse_dates=[TIME_COL])

    # Optional: Sort by time if needed
    df = df.sort_values(TIME_COL)

    # Compute magnitude
    # df['magnitude'] = np.sqrt((df[FEATURES_COL]**2).sum(axis=1))

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
    def __init__(self, d_model=64, n_seq_features=3, n_meta_features=3, nhead=4, num_layers=2, dropout=0.1, num_classes=6):
        super().__init__()
        self.seq_embedding = nn.Linear(n_seq_features, d_model)             # Input: (batch, seq_len, 3)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=128, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # self.energy_proj = nn.Linear(1, d_model)
        self.meta_proj = nn.Linear(n_meta_features, d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_seq, x_meta):
        """
        x_seq: (batch, seq_len=5, 3)
        x_meta: (batch, n_meta_features)
        """
        x = self.seq_embedding(x_seq)         # (batch, seq_len, d_model)
        x = self.pos_encoder(x)               # (batch, seq_len, d_model)

        x = x.permute(1, 0, 2)                # (seq_len, batch, d_model)
        x = self.transformer_encoder(x)       # (seq_len, batch, d_model)
        x = x.mean(dim=0)                     # (batch, d_model)

        meta = self.meta_proj(x_meta)        # (batch, d_model)

        combined = torch.cat([x, meta], dim=1)   # (batch, d_model * 2)

        return self.classifier(combined)      # (batch, num_classes)


if __name__ == "__main__":

    X_all = []
    X_meta_all = []
    y_all = []

    for file_path in tqdm(raw_data_urls):
        X, X_meta, y = load_and_process_data(file_path)
        X_all.append(X)
        X_meta_all.append(X_meta)
        y_all.append(y)
        

    X = np.concatenate(X_all, axis=0)
    X_meta = np.concatenate(X_meta_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    y_int, encoder_dict, decoder_dict = encode_labels(y)

    print("X shape:", X.shape)
    print("X_meta shape:", X_meta.shape)
    print("y shape:", y.shape)
    print("Classes:", np.unique(y))
    print("Encoder dict:", encoder_dict)
    print("Decoder dict:", decoder_dict)

    # X_train, X_meta_train, y_train, X_test, X_meta_test, y_test = split_data(X, X_meta, y_int)
    X_train, X_meta_train, y_train, X_temp, X_meta_temp, y_temp = split_data(X, X_meta, y_int)
    X_val, X_meta_val, y_val, X_test, X_meta_test, y_test = split_data(X_temp, X_meta_temp, y_temp, 0.5)


    train_dataset = HARWindowDataset(X_train, X_meta_train, y_train)
    val_dataset = HARWindowDataset(X_val, X_meta_val, y_val)
    test_dataset = HARWindowDataset(X_test, X_meta_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # === Model, loss, optimizer ===
    model = AccelTransformer(
        num_classes=len(encoder_dict),
        n_seq_features=X.shape[-1],
        n_meta_features=X_meta.shape[-1]
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # === Training loop ===
    best_val_loss = float('inf')
    best_model_state = None
    patience=10
    patience_counter = 0

    print(f"{DEVICE=}")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f"===(Training)===")
        for batch_idx, (x_seq, x_meta, y_true) in enumerate(tqdm(train_loader)):
            x_seq, x_meta, y_true = x_seq.to(DEVICE), x_meta.to(DEVICE), y_true.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x_seq, x_meta)

            loss = criterion(outputs, y_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_true).sum().item()
            total += y_true.size(0)

        avg_train_loss = train_loss / len(train_loader)
        print(f'Training Loss: {avg_train_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        print(f"===(Validation)===")
        with torch.no_grad():
            for batch_idx, (x_seq, x_meta, y_true) in enumerate(tqdm(val_loader)):
                x_seq, x_meta, y_true = x_seq.to(DEVICE), x_meta.to(DEVICE), y_true.to(DEVICE)
                outputs = model(x_seq, x_meta)
                loss = criterion(outputs, y_true)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'accel_transformer.pth')
            print(f'New best model saved! Validation Loss: {avg_val_loss:.4f}')
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    eval_metrics = {}
    with torch.no_grad():
        total_loss = 0
        predictions = []
        actuals = []
        
        for batch_idx, (x_seq, x_meta, y_true) in enumerate(train_loader):
            x_seq, x_meta, y_true = x_seq.to(DEVICE), x_meta.to(DEVICE), y_true.to(DEVICE)
            outputs = model(x_seq, x_meta)
            loss = criterion(outputs, y_true)
            total_loss += loss.item()
            
            # Convert outputs to class predictions
            pred_classes = torch.argmax(outputs, dim=1)
            true_classes = y_true
            
            # Store predictions and actual values
            predictions.extend(pred_classes.cpu().numpy())
            actuals.extend(true_classes.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        precision, recall, f1_score, _ = precision_recall_fscore_support(actuals, predictions)
        cm = confusion_matrix(actuals, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=decoder_dict.values(),
                    yticklabels=decoder_dict.values())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Raw Data)')
        plt.savefig('transformer_confusion_matrix.png')
        print("===Evaluation Metrics===")
        print("class\t\tprec.\trecall\tf1-score")
        for i, value in enumerate(decoder_dict.values()):
            print(f"{value}\t{precision[i]:.4f}\t{recall[i]:.4f}\t{f1_score[i]:.4f}")

        print()    
        print(f"avg\t\t{np.mean(precision):.4f}\t{np.mean(recall):.4f}\t{np.mean(f1_score):.4f}")
        

        
        avg_test_loss = total_loss / len(test_loader)
        print(f"Average Test Loss: {avg_test_loss}")
        eval_metrics = {
            'test_loss': avg_test_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        }
