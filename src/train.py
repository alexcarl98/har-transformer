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
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import load_and_process_data, split_data, encode_labels
from har_model import AccelTransformer, HARWindowDataset, CNNTransformerHAR, XYZLSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import logging
from utils import TConfig
import yaml
import wandb
from datetime import datetime
from torchinfo import summary
DEBUG_MODE = False
run = None

ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_BLUE = "\033[94m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_MAGENTA = "\033[95m"
ANSI_RESET = "\033[0m"

if not DEBUG_MODE:
    run = wandb.init(
        entity="alex-alvarez1903-loyola-marymount-university",
        project="HAR-PosTransformer",
    )


# ==== Data Processing ====

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def val_batchX_labels(x_seq, x_meta, y_true, y_pred, decoder_dict, args, current_number, num_samples=9, name=''):
    """
    Visualize sample predictions from a batch alongside their signals in a 3x3 grid
    
    Args:
        x_seq: tensor of shape (batch_size, seq_len, features)
        x_meta: tensor of shape (batch_size, meta_features)
        y_true: tensor of true labels
        y_pred: tensor of predicted labels
        decoder_dict: dictionary mapping indices to class names
        args: configuration arguments
        num_samples: number of samples to visualize (default: 9 for 3x3 grid)
    """
    # Convert tensors to numpy arrays
    x_seq = x_seq.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Randomly select samples
    batch_size = x_seq.shape[0]
    sample_indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)
    
    # Create a grid of subplots (3x3)
    nrows = ncols = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot each sample
    for plot_idx, sample_idx in enumerate(sample_indices):
        ax = axes_flat[plot_idx]
        
        # Get the sequence data for this sample
        sequence = x_seq[sample_idx]  # shape: (seq_len, features)
        true_label = decoder_dict[y_true[sample_idx]]
        pred_label = decoder_dict[y_pred[sample_idx]]
        
        # Calculate time points
        time_points = np.arange(sequence.shape[0])
        
        # Plot each feature (x, y, z, vm if available)
        colors = ['red', 'green', 'blue', 'purple']
        feature_names = args.ft_col
        
        for i, (feature, color) in enumerate(zip(feature_names, colors)):
            signal_data = sequence[:, i]
            ax.plot(time_points, signal_data, color=color, 
                   label=f'{feature.upper()}-axis', linewidth=1, alpha=0.7)
        
        # Customize the subplot
        ax.set_title(f'True: {true_label}\nPred: {pred_label}',
                    color='green' if true_label == pred_label else 'red')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    # Remove any empty subplots
    for idx in range(plot_idx + 1, len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    
    plt.suptitle('Sample Predictions from Validation Batch', fontsize=16, y=1.02)
    plt.tight_layout()
    if name:
        full_path = f'{args.figure_out_dir}/{name}_batch_predictions_{current_number}.png'
    else:
        full_path = f'{args.figure_out_dir}/batch_predictions_{current_number}.png'
    
    # Save and log to wandb if not in debug mode
    plt.savefig(full_path, bbox_inches='tight')
    if not DEBUG_MODE:
        run.log({
            f"{name}_batch_predictions_{current_number}": wandb.Image(
                plt.imread(full_path),
                caption="Sample Predictions from Validation Batch"
            )
        })
    plt.close()

def plot_roc_curves(true, all_outputs, decoder_dict, name="model", graph_path=''):
    """
    Plot ROC curves for each class
    
    Args:
        true: numpy array of true labels
        all_outputs: torch tensor of model outputs
        decoder_dict: dictionary mapping indices to class names
        name: name prefix for saving the plot
    """
    # Binarize the labels for ROC curve
    n_classes = len(decoder_dict)
    # Convert classes to consecutive integers starting from 0
    class_mapping = {c: i for i, c in enumerate(sorted(decoder_dict.values()))}
    true_mapped = np.array([class_mapping[decoder_dict[t]] for t in true])
    true_bin = label_binarize(true_mapped, classes=range(n_classes))
    
    # Get prediction probabilities for all data
    pred_proba = torch.softmax(all_outputs, dim=1).numpy()
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, n_classes))
    
    for i, (class_idx, label) in enumerate(sorted(decoder_dict.items())):
        fpr, tpr, _ = roc_curve(true_bin[:, class_mapping[label]], pred_proba[:, class_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    
    # Save and log the ROC curve
    plt.savefig(f'{graph_path}/{name}_roc_curves.png')
    roc_im = plt.imread(f'{graph_path}/{name}_roc_curves.png')
    if not DEBUG_MODE:
        run.log({
            f"{name}_roc_curves": wandb.Image(roc_im, caption=f"{name} ROC Curves")
        })
    plt.close()


def evaluate_model(model, data_loader, criterion, name="model",verbose=False, graph_path=''):
    model.eval()
    total_loss = 0.0
    predictions = []
    true = []
    all_outputs = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for batch_idx, (x_seq, x_meta, y_true) in enumerate(pbar):
            x_seq, x_meta, y_true = x_seq.to(DEVICE), x_meta.to(DEVICE), y_true.to(DEVICE)
            outputs = model(x_seq, x_meta)
            loss = criterion(outputs, y_true)
            
            # Accumulate batch loss
            total_loss += loss.item() * x_seq.size(0)
            
            # Get predictions
            pred_classes = torch.argmax(outputs, dim=1)
            predictions.extend(pred_classes.cpu().numpy())
            true.extend(y_true.cpu().numpy())
            
            # Store all outputs
            all_outputs.append(outputs.cpu())
            
            # Calculate accuracy
            correct += (pred_classes == y_true).sum().item()
            total += y_true.size(0)
            current_accuracy = 100. * correct / total
            
            # Update progress bar
            current_avg_loss = total_loss / ((batch_idx + 1) * x_seq.size(0))
            if batch_idx % 40 == 0 or batch_idx == len(data_loader) - 1:
                pbar.set_description(f"Loss: {current_avg_loss:.4f}, Acc: {current_accuracy:.2f}%")

    # After the loop, concatenate all outputs
    all_outputs = torch.cat(all_outputs, dim=0)

    # Calculate validation metrics
    predictions = np.array(predictions)
    true = np.array(true)
    precision, recall, f1, _ = precision_recall_fscore_support(true, predictions, average='macro')
    avg_loss = total_loss / len(data_loader.dataset)

    if verbose:
        print(classification_report(true, predictions))

    if graph_path:
        report = classification_report(true, predictions, output_dict=True)
        # Log metrics for each class
        metrics = {}
        for class_name, class_metrics in report.items():
            if isinstance(class_metrics, dict):  # Skip 'accuracy' which isn't a dict
                for metric_name, value in class_metrics.items():
                    metrics[f"{class_name}/{metric_name}"] = value
        
        # Log overall metrics
        metrics.update({
            'overall/accuracy': report['accuracy'],
            'overall/macro_avg_precision': report['macro avg']['precision'],
            'overall/macro_avg_recall': report['macro avg']['recall'],
            'overall/macro_avg_f1': report['macro avg']['f1-score'],
            'overall/weighted_avg_precision': report['weighted avg']['precision'],
            'overall/weighted_avg_recall': report['weighted avg']['recall'],
            'overall/weighted_avg_f1': report['weighted avg']['f1-score'],
        })
        # Log to wandb
        if not DEBUG_MODE:
            run.log(metrics)

        # Calculate both raw and normalized confusion matrices
        cm = confusion_matrix(true, predictions)
        cm_norm = confusion_matrix(true, predictions, normalize='true')  # normalize by row (true labels)

        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=decoder_dict.values(),
                    yticklabels=decoder_dict.values(), ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title('Confusion Matrix (Raw Counts)')

        # Normalized values (as percentages)
        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',  # .1% formats as percentage with 1 decimal
                    xticklabels=decoder_dict.values(),
                    yticklabels=decoder_dict.values(), ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        ax2.set_title('Confusion Matrix (Normalized by True Label)')

        plt.tight_layout()
        cm_path = f'{graph_path}/{name}_confusion_matrix.png'
        plt.savefig(cm_path, bbox_inches='tight', dpi=300)
        im = plt.imread(cm_path)
        if not DEBUG_MODE:
            run.log({
                f"{name}_confusion_matrix": wandb.Image(im, caption=f"{name} Confusion Matrix")
            })
        # Plot ROC curves
        plot_roc_curves(true, all_outputs, decoder_dict, name, graph_path)
        
        idx_1= len(data_loader)//4
        idx_2= len(data_loader)//2
        idx_3= 3*len(data_loader)//4
        idxs= [idx_1, idx_2, idx_3]
        counter = 0
        
        for batch_idx, (x_seq, x_meta, y_true) in enumerate(data_loader):
            if batch_idx in idxs:
                x_seq, x_meta, y_true = x_seq.to(DEVICE), x_meta.to(DEVICE), y_true.to(DEVICE)
                outputs = model(x_seq, x_meta)
                pred_classes = torch.argmax(outputs, dim=1)
                val_batchX_labels(x_seq, x_meta, y_true, pred_classes, decoder_dict,
                                  args, counter+1, num_samples=9, name=name)
                break
        plt.close()
    
    return avg_loss, f1


def save_model(epoch, model_state, optimizer_state, 
               train_loss, val_loss, val_f1, args, 
               name="model"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_f1': val_f1,
    }, f"{args.weights_out_dir}/{name}.pth")

def partition_across_subjects(data_paths, args):
    all_subjects = [] 

    for file_path in tqdm(data_paths):
        subject_data = [] 
        
        for sensor_loc in args.sensor_loc:
            try:
                X, X_meta, y = load_and_process_data(file_path, args, sensor_loc)
                temp = y.tolist()
                y_encoded= np.array([args.encoder_dict[label[0]] for label in temp])
                subject_data.append(HARWindowDataset(X, X_meta, y_encoded))
            except Exception as e:
                print(f"Error processing {file_path} with {sensor_loc}: {e}")
                continue
        all_subjects.append(subject_data)

    def alt_format_data(all_subjects, indices):
        # Start with first subject's data
        current_har_subject_data = HARWindowDataset.decouple_combine(all_subjects[indices[0]])
        
        # Combine with remaining subjects
        for i in indices[1:]:
            next_subject_data = HARWindowDataset.decouple_combine(all_subjects[i])
            current_har_subject_data = current_har_subject_data.combine_with(next_subject_data)
        
        return current_har_subject_data
    

    # Split subjects into train/val/test before concatenating
    n_subjects = len(all_subjects)
    indices = np.arange(n_subjects)
    
    # First split: 60% train, 40% temp
    train_indices, temp_indices = train_test_split(
        indices, test_size=args.test_size, random_state=args.random_seed
    )
    
    # Second split: 20% val, 20% test (from the 40% temp)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=args.random_seed
    )

    train_data = alt_format_data(all_subjects, train_indices)
    val_data = alt_format_data(all_subjects, val_indices)
    test_data = alt_format_data(all_subjects, test_indices)
    
    return train_data, val_data, test_data

def partition_across_sensors(data_paths, args):
    X_all = []
    X_meta_all = []
    y_all = []

    for file_path in tqdm(data_paths):
        for sensor_loc in args.sensor_loc:
            try:
                X, X_meta, y = load_and_process_data(file_path, args, sensor_loc)
                X_all.append(X)
                X_meta_all.append(X_meta)
                y_all.append(y)
            except Exception as e:
                print(f"Error processing {file_path} with {sensor_loc}: {e}")
                continue

    # Split sensor streams into train/val/test before concatenating
    n_sensors = len(X_all)
    indices = np.arange(n_sensors)
    
    # First split: 60% train, 40% temp
    train_indices, temp_indices = train_test_split(
        indices, test_size=args.test_size, random_state=args.random_seed
    )
    
    # Second split: 20% val, 20% test (from the 40% temp)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=args.random_seed)
    
    def format_data(X_all, X_meta_all, y_all, indices, args):
        X = np.concatenate([X_all[i] for i in indices], axis=0)
        X_meta = np.concatenate([X_meta_all[i] for i in indices], axis=0)
        y = np.concatenate([y_all[i] for i in indices], axis=0).ravel()
        y_int = np.array([args.encoder_dict[label] for label in y])
        return HARWindowDataset(X, X_meta, y_int)
    
    training_data = format_data(X_all, X_meta_all, y_all, train_indices, args)
    val_data = format_data(X_all, X_meta_all, y_all, val_indices, args)
    test_data = format_data(X_all, X_meta_all, y_all, test_indices, args)

    return training_data, val_data, test_data

from torch.optim.lr_scheduler import LambdaLR
import math

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """
    Creates a learning rate scheduler with linear warmup and cosine decay
    
    Args:
        optimizer: The optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

def train_model(args, train_loader, val_data, model, optimizer, criterion, device):
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    current_epoch = 0
    last_avg_loss = 0.0
    last_f1 = 0.0

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)  # 10% of total steps for warmup
    # scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    if args.load_model_path:
        checkpoint = torch.load(args.load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_f1 = checkpoint['val_f1']

    for epoch in range(args.epochs):
        # Training phase
        current_epoch += 1
        model.train()
        train_loss = 0.0
        current_avg_loss = 0.0
        current_accuracy = 0.0
        predictions = []
        true = []
        correct = 0
        total = 0
        print(f'Epoch {current_epoch}/{args.epochs}:')
        print(f"===(Training)===")
        pbar = tqdm(train_loader)
        # print("Model weights:", model.classifier[0].weight.data.shape)
        # exit()
        for batch_idx, (x_seq, x_meta, y_true) in enumerate(pbar):
            x_seq, x_meta, y_true = x_seq.to(DEVICE), x_meta.to(DEVICE), y_true.to(DEVICE)

            # Training specific steps
            optimizer.zero_grad()
            outputs = model(x_seq, x_meta)
            # if batch_idx % 200 == 0:
            #     print("Model Weights Before:", model.classifier[0].weight.data)
            loss = criterion(outputs, y_true)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # if batch_idx % 200 == 0:
            #     print("Model Weights After:", model.classifier[0].weight.data)
            # Consistent prediction handling
            pred_classes = torch.argmax(outputs, dim=1)
            predictions.extend(pred_classes.cpu().numpy())
            true.extend(y_true.cpu().numpy())
            
            # Accumulate batch loss (consistent with eval loop)
            train_loss += loss.item() * x_seq.size(0)
            total += y_true.size(0)
            
            # Update progress bar
            current_avg_loss = train_loss / ((batch_idx + 1) * x_seq.size(0))
            # current_lr = scheduler.get_last_lr()[0]

            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                current_accuracy = 100. * accuracy_score(true, predictions)
                pbar.set_description(f"Loss: {current_avg_loss:.4f}, Acc: {current_accuracy:.2f}%")
                # pbar.set_description(f"Loss: {current_avg_loss:.4f}, Acc: {current_accuracy:.2f}%, Lr: {current_lr:.6f}")

        avg_train_loss = train_loss / total
        train_accuracy = 100. * correct / total
        # exit()

        # Validation phase
        print(f"===(Validation)===")
        avg_val_loss, f1 = evaluate_model(model, val_loader, criterion)
        print(f"Validation - Avg Loss: {avg_val_loss:.4f}, F1 Score: {f1:.4f}")

        if not DEBUG_MODE:
            run.log({
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_f1": f1,
            })

        # Save best model based on F1 score
        if f1 > best_val_f1:
            best_val_f1 = f1

            best_model_state = model.state_dict().copy()
            patience_counter = 0
            # Save the model
            save_model(current_epoch, model.state_dict(), optimizer.state_dict(), 
                       avg_train_loss, avg_val_loss, f1, args, 
                       name="best_accel_transformer")
            print(f'New best model saved! Validation F1: {f1:.4f}')
        else:
            patience_counter += 1
        
        last_avg_loss = avg_val_loss
        last_f1 = f1
        # Early stopping check
        if patience_counter >= args.patience:
            print(f'Early stopping triggered after {current_epoch} epochs')
            break
        print()
    
    save_model(current_epoch, model.state_dict(), optimizer.state_dict(), 
               avg_train_loss, last_avg_loss, last_f1, args, 
               name="last_accel_transformer")


if __name__ == "__main__":
    ADD_NOISY_DATA = False
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    
    if not DEBUG_MODE:
        run.config.update(config['transformer'])
        with open("sweep.yaml", "r") as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep_config)
    else:
        print(f"{ANSI_CYAN}DEBUG MODE: Will not log to wandb{ANSI_RESET}")

    args = TConfig(**config['transformer'])
    # if DEBUG_MODE:
    #     dataset_numbers = debug_set
    raw_data_paths = [f"{args.data_dir}{num}.csv" for num in dataset_numbers]
    incomplete_data_paths = [f"{args.data_dir}{num}.csv" for num in incomplete]

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    decoder_dict = args.decoder_dict

    train_data, val_data, test_data = partition_across_subjects(raw_data_paths, args)
    
    if ADD_NOISY_DATA:
        noise_train_data, noise_val_data, noise_test_data = partition_across_sensors(incomplete_data_paths, args)
        
        train_data = train_data.combine_with(noise_train_data)
        val_data = val_data.combine_with(noise_val_data)
        test_data = test_data.combine_with(noise_test_data)

    print("X_train shape:", train_data.X.shape)
    print("X_val shape:", val_data.X.shape)
    print("X_test shape:", test_data.X.shape)
    print("Classes:", np.unique(train_data.y))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)


    # === Model, loss, optimizer ===
    model = AccelTransformer(
        d_model=args.d_model,
        fc_hidden_dim=args.fc_hidden_dim,
        num_classes=args.num_classes,
        in_seq_dim=args.in_seq_dim,
        in_meta_dim=args.in_meta_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(DEVICE)
    input_size = [
        (args.batch_size, args.window_size, args.in_seq_dim),  # x_seq shape
        (args.batch_size, args.in_meta_dim),                   # x_meta shape
    ]

    summary(model, input_size)
    # model = XYZLSTM(
    #     in_seq_dim=args.in_seq_dim,
    #     hidden_dim=args.fc_hidden_dim,
    #     num_classes=args.num_classes
    # ).to(DEVICE)
    # model = AccelTransformer(
    #     d_model=args.d_model,
    #     fc_hidden_dim=args.fc_hidden_dim,
    #     num_classes=args.num_classes,
    #     in_seq_dim=args.in_seq_dim,
    #     in_meta_dim=args.in_meta_dim,
    #     nhead=args.nhead,
    #     num_layers=args.num_layers,
    #     dropout=args.dropout
    # ).to(DEVICE)
    # model = CNNTransformerHAR(
    #     num_classes=args.num_classes,
    #     seq_len=args.window_size,
    #     in_seq_dim=args.in_seq_dim,
    #     in_meta_dim=args.in_meta_dim,
    #     nhead=args.nhead,
    #     num_layers=args.num_layers,
    #     dropout=args.dropout
    # ).to(DEVICE)
        # d_model=args.d_model,
        # cnn_out_channels=args.cnn_out_channels,
        # fc_hidden_dim=args.fc_hidden_dim,
    optimizer = Adam(model.parameters(),
                      lr=args.learning_rate, 
                      weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    train_model(args, train_loader, val_loader, model, optimizer, criterion, DEVICE)
    

    print("testing most recent model:")

    checkpoint = torch.load(f"{args.weights_out_dir}/last_accel_transformer.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"===(Testing)===")
    test_avg_loss, test_f1 = evaluate_model(model, test_loader, criterion, 
                                            name=f"last_model_ep{checkpoint['epoch']}", 
                                            verbose=True, graph_path=args.figure_out_dir)

    print("testing best model:")
    checkpoint = torch.load(f"{args.weights_out_dir}/best_accel_transformer.pth")

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"===(Testing)===")
    best_model_avg_loss, best_model_f1 = evaluate_model(model, test_loader, criterion, 
                                                        name=f"best_model_ep{checkpoint['epoch']}", 
                                                        verbose=True, graph_path=args.figure_out_dir)

    if not DEBUG_MODE:
        run.finish()

    exit()