import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from scipy.fft import fft
from torch.utils.data import DataLoader, TensorDataset
import random
import wandb

π = np.pi
LOG_OUTPUTS   = False
RANDOM_SEED   = 0xBEEF
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 2

if LOG_OUTPUTS:
    run = wandb.init(
        entity="alex-alvarez1903-loyola-marymount-university",
        project="HAR_transformer",
        config={
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "architecture": "Transformer",
            "dataset": "HAR-labs-Accelerometer-Dataset-016",
            "num_epochs": NUM_EPOCHS,
            "random_seed": RANDOM_SEED
        }
    )

def derive_periodic_features(t, period):
    ω = (2*π) / period
    return np.sin(ω*t), np.cos(ω*t)

def format_time(time_col):
    '''
    This is a helper function to format a time column when
    The time is dependend on shorter periods than longer ones
    '''
    time_col = pd.to_datetime(time_col)

    minute = time_col.dt.minute
    second = time_col.dt.second
    microsecond = time_col.dt.microsecond

    microsecond = microsecond // 1000
    microsecond_period = 1000000//1000 # 1000 possible values
    # We also can derive the sin and cos of the minute, second, and microsecond
    sin_minute, cos_minute = derive_periodic_features(minute, 60)
    sin_second, cos_second = derive_periodic_features(second, 60)
    sin_microsecond, cos_microsecond = derive_periodic_features(microsecond, microsecond_period)

    # Now we can concatenate the sin and cos of the minute, second, and microsecond
    time_df = pd.DataFrame({
        "minute": minute.astype(np.int8),
        "second": second.astype(np.int8),
        "microsecond": microsecond.astype(np.int16),
        "sin_minute": sin_minute,
        "cos_minute": cos_minute,
        "sin_second": sin_second,
        "cos_second": cos_second,
        "sin_microsecond": sin_microsecond,
        "cos_microsecond": cos_microsecond
    })

    return time_df
    
def remove_sensor_data(df, sensor):
    df.drop(columns=[f"{sensor}_x", f"{sensor}_y", 
                     f"{sensor}_z", f"{sensor}_vm"], 
                     inplace=True)
    return df


def create_new_data(file_path: str = ""):
    target_value = "activity"
    if file_path == "":
        har_dataset = pd.read_csv('https://raw.githubusercontent.com/Har-Lab/HumanActivityData/refs/heads/main/data/labeled_activity_data/016_labeled.csv' ) 
    else:
        har_dataset = pd.read_csv(file_path)
    
    # Person never changes in this dataset, let's drop it
    if "person" in har_dataset.columns:
        har_dataset = har_dataset.drop(columns=["person"])
    
    # For now just using the waist sensor
    columns_with_sensors = har_dataset.filter(regex='ankle|wrist').columns

    # trying to use only one accelerometer sensor
    if columns_with_sensors.size > 1:
        har_dataset = remove_sensor_data(har_dataset, "wrist")
        har_dataset = remove_sensor_data(har_dataset, "ankle")
    
    # Format time column to datetime and get categorical time features
    time_df = create_time_features(har_dataset["time"], components=["minute", "second", "microsecond"])
    har_dataset = pd.concat([har_dataset, time_df], axis=1)
    har_dataset.drop(columns=["time"], inplace=True)
    
    # Get indices for categorical time columns
    min_idx = har_dataset.columns.get_loc("minute")
    sec_idx = har_dataset.columns.get_loc("second")
    usec_idx = har_dataset.columns.get_loc("microsecond")
    
    # One-hot encode the activity labels first
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    target_col = har_dataset[target_value]
    encoded_activities = encoder.fit_transform(target_col.values.reshape(-1, 1))
    print("\nActivity to One-Hot Encoding Mapping:")
    for i, activity in enumerate(encoder.categories_[0]):
        one_hot = np.zeros(len(encoder.categories_[0]))
        one_hot[i] = 1
        print(f"{activity}: {one_hot}")
    
    # Remove the target column before scaling
    features = har_dataset.drop(columns=[target_value])
    
    # Convert features to numpy array
    X = features.to_numpy()
    y = encoded_activities

    # Scale only the numerical columns (accelerometer data)
    numerical_indices = [i for i in range(X.shape[1]) if i not in [min_idx, sec_idx, usec_idx]]
    scaler = StandardScaler()
    X[:, numerical_indices] = scaler.fit_transform(X[:, numerical_indices])
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Convert time columns to long dtype for embedding layers
    X_tensor[:, min_idx] = X_tensor[:, min_idx].long()
    X_tensor[:, sec_idx] = X_tensor[:, sec_idx].long()
    X_tensor[:, usec_idx] = X_tensor[:, usec_idx].long()
    
    # Convert labels to tensor
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Store categorical column indices for embedding layers
    categorical_indices = [min_idx, sec_idx, usec_idx]
    
    stratify_labels = y_tensor.argmax(dim=1)

    to_be_pickled = train_test_split(X_tensor, y_tensor, 
                          test_size=0.2, 
                          random_state=RANDOM_SEED,
                          stratify=stratify_labels) + [categorical_indices]
    return to_be_pickled


def determine_num_trials(dataset_size: int) -> int:
    value = 20 - 4*np.log10(dataset_size)
    return max(3, min(10, np.floor(value)))

class TransformerModel(nn.Module):
    def __init__(self, input_dim, h_m_d_idx, embed_dim=32, num_heads=4, num_layers=2, num_classes=1, is_regression=True):
        super(TransformerModel, self).__init__()
        self.cat_feats = h_m_d_idx
        self.num_feats = [i for i in range(input_dim) if i not in self.cat_feats]
        # 🔹 Embedding layers for categorical time-based features
        self.minute_embedding = nn.Embedding(60, embed_dim)  # 7 days in a week
        self.second_embedding = nn.Embedding(60, embed_dim)  # 12 months in a year
        self.microsecond_embedding = nn.Embedding(1000, embed_dim)  # 24 hours in a day
        
        # 🔹 Linear projection for numerical features (excluding categorical ones)
        self.embedding = nn.Linear(input_dim - len(self.cat_feats), embed_dim)  # Exclude day_of_week and month

        # 🔹 Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=128, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 🔹 Final classification/regression head
        self.fc = nn.Linear(embed_dim, num_classes)
        self.is_regression = is_regression
        # self.dropout = nn.Dropout(0.2)  # Add dropout

    def forward(self, x):
        minute = x[:, self.cat_feats[0]].long()
        second = x[:, self.cat_feats[2]].long()
        microsecond = x[:, self.cat_feats[1]].long()
        
        # Add validation
        if torch.max(minute) >= 60 or torch.min(minute) < 0:
            raise ValueError(f"Minute values must be between 0-59, got range: {torch.min(minute)}-{torch.max(minute)}")
        if torch.max(second) >= 60 or torch.min(second) < 0:
            raise ValueError(f"Second values must be between 0-59, got range: {torch.min(second)}-{torch.max(second)}")
        if torch.max(microsecond) >= 1000 or torch.min(microsecond) < 0:
            raise ValueError(f"Millisecond values must be between 0-999, got range: {torch.min(microsecond)}-{torch.max(microsecond)}")
        
        # Get all indices except categorical features
        numerical_features = x[:, self.num_feats]

        # Apply embeddings to categorical features
        minute_embedded = self.minute_embedding(minute)
        second_embedded = self.second_embedding(second)
        microsecond_embedded = self.microsecond_embedding(microsecond)
        
        # Apply linear projection to numerical features
        num_embedded = self.embedding(numerical_features)

        # Combine all features
        x = num_embedded + minute_embedded+second_embedded+microsecond_embedded
        
        # Pass through Transformer
        x = self.transformer(x.unsqueeze(1))  # Add sequence dimension
        x = x.mean(dim=1)  # Aggregate over sequence dimension

        return self.fc(x).squeeze(-1) if self.is_regression else self.fc(x)


def get_folds(k: int, X_train):
    """
    Generate dynamically scaled k-fold splits based on dataset size.

    Parameters:
    --------------------
        k -- int, number of folds
        data -- Data object containing X (features) and y (labels)

    Returns:
    --------------------
        List of k-fold splitters for multiple trials
    """
    dataset_size = len(X_train)  # Total dataset size
    n_trials = determine_num_trials(dataset_size)  # Determine trials dynamically
    fold_type = KFold

    # Generate multiple KFold instances (one per trial)
    return [fold_type(n_splits=k, shuffle=True, random_state=i) for i in range(n_trials)]



def train_transformer(X_train, y_train, X_test, y_test, 
                      model, criterion, optimizer, num_epochs, 
                      device, patience=10):
    # Split into train and validation
    labels = ["downstairs", "jog_treadmill", "upstairs", 
              "walk_mixed", "walk_sidewalk", "walk_treadmill"]
    
    
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    x_val, x_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED)
    
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Check if this is the best model
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
            }, 'best_model_checkpoint.pth')
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
        
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            # Convert outputs to class predictions
            pred_classes = torch.argmax(outputs, dim=1)
            true_classes = torch.argmax(y_batch, dim=1)
            
            # Store predictions and actual values
            predictions.extend(pred_classes.cpu().numpy())
            actuals.extend(true_classes.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        precision, recall, f1_score, _ = precision_recall_fscore_support(actuals, predictions, average="macro")
        cm = confusion_matrix(actuals, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Raw Data)')
        plt.savefig('transformer_confusion_matrix.png')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1_score:.4f}")
        
        avg_test_loss = total_loss / len(test_loader)
        print(f"Average Test Loss: {avg_test_loss:.4f}")
        eval_metrics = {
            'test_loss': avg_test_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        }
        if LOG_OUTPUTS:
            run.log(eval_metrics)
    
    return model, best_val_loss, eval_metrics





def main():
    k = 10
    num_classes = 6
    num_epochs = NUM_EPOCHS
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    X_train, X_test, y_train, y_test, h_m_d_idx = create_new_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(X_train[0:2])
    print(y_train[0:2])
    # print(h_m_d_idx)
    number_of_kfolds_trials = determine_num_trials(X_train.shape[0])
    print(number_of_kfolds_trials)
    kfs = get_folds(10, X_train)
    is_regression = False

    print(f"{is_regression=}")
    print(f"{num_classes=}")
    input_dimension = X_train.shape[1]
    print(f"{input_dimension=}")

    # Set model type
    model = TransformerModel(input_dim=input_dimension, h_m_d_idx=h_m_d_idx, 
                             num_classes=num_classes, is_regression=is_regression).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model, best_val_loss, metrics = train_transformer(X_train, y_train, X_test, y_test, 
                      model, criterion, optimizer, num_epochs, 
                      device, patience=10)
    
    if LOG_OUTPUTS:
        run.finish()


def custom_data_test():
    labels = ["downstairs", "jog_treadmill", "upstairs", 
              "walk_mixed", "walk_sidewalk", "walk_treadmill"]
    k = 10
    num_classes = 6
    X_train, X_test, y_train, y_test, h_m_d_idx = create_new_data(file_path="HAR_data/my_walking_data.csv")

    

    
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # initialize Model:
    model = TransformerModel(input_dim=X_train.shape[1], h_m_d_idx=h_m_d_idx, 
                             num_classes=num_classes, is_regression=False).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    checkpoint = torch.load("best_model_checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    # load existing model:
    # model.load_state_dict(torch.load("best_model_checkpoint.pth"))
    model.eval()
    with torch.no_grad():
        outputs = model(X_train)
        predictions = torch.argmax(outputs, dim=1)

        labeled = []
        for i in range(len(predictions)):
            activity = labels[predictions[i]]
            labeled.append(activity)
            print(f"{i}: {activity}")
        outputs = model(X_train)
        predictions = torch.argmax(outputs, dim=1)

        labeled = []
        for i in range(len(predictions)):
            activity = labels[predictions[i]]
            labeled.append(activity)
            print(f"{i}: {activity}")


if __name__ == "__main__":
    main()
    # custom_data_test()