# Data Extraction
import torch
import torch.nn as nn
from torch.optim import Adam
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import random
from tqdm import tqdm
from constants import *
from preprocessing import load_and_process_data, split_data, encode_labels
from har_model import AccelTransformer, HARWindowDataset

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==== Data Processing ====
raw_data_urls = [f"{data_dir}{num}.csv" for num in dataset_numbers]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    sensor_locs = ['ankle', 'waist', 'wrist']
    X_all = []
    X_meta_all = []
    y_all = []

    for file_path in tqdm(raw_data_urls):
        for sensor_loc in sensor_locs:
            try:
                X, X_meta, y = load_and_process_data(file_path, sensor_loc)
            except Exception as e:
                print(f"Error processing {file_path} with {sensor_loc}: {e}")
                continue
            X_all.append(X)
            X_meta_all.append(X_meta)
            y_all.append(y)
        

    X = np.concatenate(X_all, axis=0)
    X_meta = np.concatenate(X_meta_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    y_int, encoder_dict, decoder_dict = encode_labels(y)

    assert X.shape[-1] == SZ_SEQ_DATA
    assert X_meta.shape[-1] == SZ_META_DATA
    assert len(encoder_dict) == NUM_CLS

    print("X shape:", X.shape)
    print("X_meta shape:", X_meta.shape)
    print("y shape:", y.shape)
    print("Classes:", np.unique(y))
    # print("Encoder dict:", encoder_dict)
    # print("Decoder dict:", decoder_dict)

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
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_val_loss = float('inf')
    
    if LOAD_PREVIOUS_MODEL:
        checkpoint = torch.load("accel_transformer.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['val_loss']

    criterion = nn.CrossEntropyLoss()

    # === Training loop ===
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
