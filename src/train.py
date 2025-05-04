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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import model_version.v1 as v1
import wandb
import os
from config import Config
from data import GeneralDataLoader
from util import WandBLogger
from typing import Optional

ANSI_BLUE = "\033[94m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_MAGENTA = "\033[95m"
ANSI_RESET = "\033[0m"

# ==== Data Processing ====

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_batch_examples(x_seq, true, predictions, class_names, name="model", graph_path='', feature_names=None, num_samples=9):
    """
    Visualize sample predictions alongside their signals in a grid
    
    Args:
        x_seq: numpy array of shape (batch_size, seq_len, features)
        true: numpy array of true labels
        predictions: numpy array of predicted labels
        class_names: list of class names
        name: prefix for saving the plot
        graph_path: path to save the plot
        feature_names: list of feature names for plotting
        num_samples: number of samples to visualize (default: 9 for 3x3 grid)
    """
    # Select random samples
    batch_size = len(true)
    sample_indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)
    
    # Create grid
    nrows = ncols = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
    axes_flat = axes.flatten()
    
    # Plot each sample
    for plot_idx, sample_idx in enumerate(sample_indices):
        ax = axes_flat[plot_idx]
        sequence = x_seq[sample_idx]
        true_label = class_names[true[sample_idx]]
        pred_label = class_names[predictions[sample_idx]]
        
        # Plot features
        time_points = np.arange(sequence.shape[0])
        colors = ['red', 'green', 'blue', 'purple']
        
        for i, (feature, color) in enumerate(zip(feature_names, colors)):
            ax.plot(time_points, sequence[:, i], color=color, 
                   label=feature, linewidth=1, alpha=0.7)
        
        # Customize subplot
        ax.set_title(f'True: {true_label}\nPred: {pred_label}',
                    color='green' if true_label == pred_label else 'red')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')
    
    # Remove empty subplots
    for idx in range(plot_idx + 1, len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    
    plt.tight_layout()
    plt.savefig(f'{graph_path}/{name}_batch_examples.png', bbox_inches='tight', dpi=150)
    plt.close()

def plot_roc_curves(true, all_outputs, class_names, name="model", graph_path=''):
    """
    Plot ROC curves for each class
    
    Args:
        true: numpy array of true labels (integers)
        all_outputs: torch tensor of model outputs
        class_names: list of class names as strings
        name: name prefix for saving the plot
        graph_path: path to save the graph
    """
    # Binarize the labels for ROC curve
    n_classes = len(class_names)
    
    # No need to map if already integers
    true_bin = label_binarize(true, classes=range(n_classes))
    
    # Get prediction probabilities for all data
    pred_proba = torch.softmax(all_outputs, dim=1).numpy()
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, n_classes))
    
    for i, class_name in enumerate(sorted(class_names)):
        fpr, tpr, _ = roc_curve(true_bin[:, i], pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_name} (AUC = {roc_auc:.2f})')
    
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
    plt.close()


def plot_confusion_matrix(true, predictions, class_names, name="model", graph_path=''):
    cm = confusion_matrix(true, predictions)
    cm_norm = confusion_matrix(true, predictions, normalize='true')

    # Reduced figure size and adjusted font sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Smaller figure size
    
    # Set font sizes
    plt.rcParams.update({
        'font.size': 8,           # Base font size
        'axes.titlesize': 10,     # Title font size
        'axes.labelsize': 9,      # Axis label size
    })

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names, ax=ax1,
                annot_kws={'size': 8})  # Annotation font size
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (Raw Counts)')

    # Normalized values
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names, ax=ax2,
                annot_kws={'size': 8})  # Annotation font size
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (Normalized by True Label)')

    plt.tight_layout()
    cm_path = f'{graph_path}/{name}_confusion_matrix.png'
    plt.savefig(cm_path, 
                bbox_inches='tight', 
                dpi=150)  # Lower DPI
    im = plt.imread(cm_path)
    plt.close()  # Close the figure to free memory


def evaluate_model(model, data_loader, criterion, name="model", verbose=False, graph_path='', class_names=None, feature_names=None, logger=None):
    model.eval()
    total_loss = 0.0
    predictions = []
    true = []
    all_outputs = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for batch_idx, (x_seq, y_true) in enumerate(pbar):
            x_seq, y_true = x_seq.to(DEVICE), y_true.to(DEVICE)
            outputs = model(x_seq)
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

    _, _, f1, _ = precision_recall_fscore_support(true, predictions, average='macro')
    avg_loss = total_loss / len(data_loader.dataset)

    if verbose:
        print(classification_report(true, predictions, target_names=class_names))

    if graph_path:
        # Plot metrics
        plot_confusion_matrix(true, predictions, class_names, name, graph_path)
        plot_roc_curves(true, all_outputs, class_names, name, graph_path)
        
        # Get one batch for example plotting
        x_batch, y_batch = next(iter(data_loader))
        with torch.no_grad():
            outputs = model(x_batch.to(DEVICE))
            pred_batch = torch.argmax(outputs, dim=1)
            
        # Plot examples from this batch
        plot_batch_examples(x_batch.cpu().numpy(), 
                          y_batch.cpu().numpy(),
                          pred_batch.cpu().numpy(),
                          class_names, name, graph_path, feature_names)
    
    if logger:        
        # Log plots if graph_path exists
        if graph_path:
            logger.log_metrics({
                f"{name}/{logger.cm_save_dir}": wandb.Image(f'{graph_path}/{name}_confusion_matrix.png'),
                f"{name}/{logger.roc_save_dir}": wandb.Image(f'{graph_path}/{name}_roc_curves.png'),
                f"{name}/{logger.batch_save_dir}": wandb.Image(f'{graph_path}/{name}_batch_examples.png')
            })

    return avg_loss, f1


def save_model(epoch, model_state, optimizer_state, 
               train_loss, val_loss, val_f1, 
               name="model",model_out_path=''):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_f1': val_f1,
    }, f"{model_out_path}/{name}.pth")

def train_model(args, train_loader, val_data, model, optimizer, criterion, model_out_path='', logger=None):
    best_val_f1 = 0.0
    patience_counter = 0
    current_epoch = 0
    last_avg_loss = 0.0
    last_f1 = 0.0

    if args.load_model_path:
        checkpoint = torch.load(args.load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_f1 = checkpoint['val_f1']

    for _ in range(args.epochs):
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

        for batch_idx, (x_seq, y_true) in enumerate(pbar):
            x_seq, y_true = x_seq.to(DEVICE),  y_true.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x_seq)
            loss = criterion(outputs, y_true)
            loss.backward()
            optimizer.step()

            pred_classes = torch.argmax(outputs, dim=1)
            predictions.extend(pred_classes.cpu().numpy())
            true.extend(y_true.cpu().numpy())
            
            train_loss += loss.item() * x_seq.size(0)
            total += y_true.size(0)
            
            current_avg_loss = train_loss / ((batch_idx + 1) * x_seq.size(0))

            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                current_accuracy = 100. * accuracy_score(true, predictions)
                pbar.set_description(f"Loss: {current_avg_loss:.4f}, Acc: {current_accuracy:.2f}%")

        avg_train_loss = train_loss / total
        train_accuracy = 100. * correct / total

        # Log training metrics

        # Validation phase
        print(f"===(Validation)===")
        avg_val_loss, f1 = evaluate_model(
            model, val_data, criterion, 
            logger=logger
        )
        if logger:
            metrics = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_f1": f1,
            }
            logger.log_metrics(metrics)

        # Save best F1 model
        if f1 > best_val_f1:
            patience_counter = 0
        else:
            patience_counter += 1

        if f1 > best_val_f1:
            best_val_f1 = f1
            save_model(current_epoch, model.state_dict(), optimizer.state_dict(), 
                      avg_train_loss, avg_val_loss, f1, 
                      name="best_f1", model_out_path=model_out_path)
            if logger:
                logger.log_model_artifact("best_f1", f"{model_out_path}/best_f1.pth")
            print(f'New best F1 model saved! Validation F1: {f1:.4f}')

        # Early stopping check
        if patience_counter >= args.patience:
            print(f'Early stopping triggered after {current_epoch} epochs')
            break
        print()
    
    save_model(current_epoch, model.state_dict(), optimizer.state_dict(), 
               avg_train_loss, last_avg_loss, last_f1, 
               name="last",model_out_path=model_out_path)

def set_all_seeds(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate_and_save_metrics(config, name, model, test_loader, criterion, logger=None):
    model_dir = config.output_paths.models_dir
    output_path = config.output_paths.plots_dir
    classes = config.data.classes
    ft_col = config.data.ft_col
    checkpoint = torch.load(f"{model_dir}/{name}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return evaluate_model(model, test_loader, criterion, 
                                            name=name, 
                                            verbose=True, graph_path=output_path,
                                            class_names=classes, feature_names=ft_col,
                                            logger=logger)


def train_main(config: Config, logger: Optional[WandBLogger] = None) -> Optional[WandBLogger]:
    set_all_seeds(42)
    
    # Initialize WandB logger if wandb is enabled
    if logger is None and config.wandb.mode != 'disabled':
        logger = WandBLogger(config)
    
    data_loader = GeneralDataLoader(**config.get_data_loader_params())
    train_data = data_loader.get_har_dataset('train')
    val_data = data_loader.get_har_dataset('val')
    print("X_train shape:", train_data.X.shape)
    print("X_val shape:", val_data.X.shape)
    print("Classes:", np.unique(train_data.y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.transformer.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.transformer.batch_size)
    model_dir = config.output_paths.models_dir

    # === Model, loss, optimizer ===
    model = v1.AccelTransformerV1(**config.get_transformer_params()).to(DEVICE)
    print(model)

    optimizer = Adam(model.parameters(),
                    lr=config.transformer.learning_rate, 
                    weight_decay=config.transformer.weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    train_model(
        config.transformer, train_loader, val_loader, 
        model, optimizer, criterion, model_dir,
        logger=logger
    )
    
    return logger

def test_main(config: Config, logger: Optional[WandBLogger] = None):
    # Use existing logger or create new one if None
    if logger is None and config.wandb.mode != 'disabled':
        logger = WandBLogger(config)
    

    data_loader = GeneralDataLoader(**config.get_data_loader_params())
    test_data = data_loader.get_har_dataset('test')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.transformer.batch_size)

    model = v1.AccelTransformerV1(**config.get_transformer_params()).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    print("testing most recent model:")
    print(f"===(Testing)===")
    evaluate_and_save_metrics(
        config, 'last', model, test_loader, criterion,
        logger=logger
    )

    print("testing f1 best model:")
    evaluate_and_save_metrics(
        config, 'best_f1', model, test_loader, criterion,
        logger=logger
    )
    
    # Only finish the logger if we created it in this function
    if logger is not None and config.wandb.mode != 'disabled':
        logger.finish()

if __name__ == "__main__":
    set_all_seeds(42)
    config = Config.from_yaml('config.yml')
    logger = WandBLogger(config, job_type='train')
    
    # Run training and get logger
    logger = train_main(config, logger)
    
    # Pass logger to test_main
    test_main(config, logger)
    
    config.output_paths.clean()

    exit()