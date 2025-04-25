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


ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_BLUE = "\033[94m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_MAGENTA = "\033[95m"
ANSI_RESET = "\033[0m"


# ==== Data Processing ====

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def val_batchX_labels(x_seq, y_true, y_pred, class_names, current_number, num_samples=9, name='', feature_names=None, plot_dir=''):
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
        true_label = class_names[y_true[sample_idx]]
        pred_label = class_names[y_pred[sample_idx]]
        
        # Calculate time points
        time_points = np.arange(sequence.shape[0])
        
        # Plot each feature (x, y, z, vm if available)
        colors = ['red', 'green', 'blue', 'purple']
        
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
        full_path = f'{plot_dir}/{name}_batch_predictions_{current_number}.png'
    else:
        full_path = f'{plot_dir}/batch_predictions_{current_number}.png'
    
    # Save and log to wandb if not in debug mode
    plt.savefig(full_path, bbox_inches='tight')
    
    # run.log({
    #     f"{name}_batch_predictions_{current_number}": wandb.Image(
    #         plt.imread(full_path),
    #         caption="Sample Predictions from Validation Batch"
    #     )
    # })
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


def evaluate_model(model, data_loader, criterion, name="model",verbose=False, graph_path='',class_names=None,feature_names=None):
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

    precision, recall, f1, _ = precision_recall_fscore_support(true, predictions, average='macro')
    avg_loss = total_loss / len(data_loader.dataset)

    if verbose:
        print(classification_report(true, predictions, target_names=class_names))

    if graph_path:
        # results = classification_report(true, predictions, target_names=decoder_dict, output_dict=True)
        # print(results)
        # exit()
        # Calculate both raw and normalized confusion matrices
        plot_confusion_matrix(true, predictions, class_names, name, graph_path)
        # Plot ROC curves
        plot_roc_curves(true, all_outputs, class_names, name, graph_path)
        
        idx_1= len(data_loader)//4
        idx_2= len(data_loader)//2
        idx_3= 3*len(data_loader)//4
        idxs= [idx_1, idx_2, idx_3]
        counter = 0
        
        for batch_idx, (x_seq, y_true) in enumerate(data_loader):
            if batch_idx in idxs:
                x_seq,  y_true = x_seq.to(DEVICE), y_true.to(DEVICE)
                outputs = model(x_seq)
                pred_classes = torch.argmax(outputs, dim=1)
                val_batchX_labels(x_seq, y_true, pred_classes, class_names,
                                   counter+1, num_samples=9, name=name, feature_names=feature_names, plot_dir=graph_path)
                break
        plt.close()
    
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

def train_model(args, train_loader, val_data, model, optimizer, criterion, model_out_path=''):
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    current_epoch = 0
    last_avg_loss = 0.0
    last_f1 = 0.0

    if args.load_model_path:
        checkpoint = torch.load(args.load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_f1 = checkpoint['val_f1']
        best_val_loss = checkpoint.get('val_loss', float('inf'))

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

        # Validation phase
        print(f"===(Validation)===")
        avg_val_loss, f1 = evaluate_model(model, val_data, criterion)
        print(f"Validation - Avg Loss: {avg_val_loss:.4f}, F1 Score: {f1:.4f}")

        # run.log({
        #     "train_loss": avg_train_loss,
        #     "train_accuracy": train_accuracy,
        #     "val_loss": avg_val_loss,
        #     "val_f1": f1,
        # })

        # Save best F1 model
        if f1 > best_val_f1:
            patience_counter = 0
        else:
            patience_counter += 1

        if f1 > best_val_f1:
            best_val_f1 = f1
            save_model(current_epoch, model.state_dict(), optimizer.state_dict(), 
                      avg_train_loss, avg_val_loss, f1, 
                      name="best_f1",model_out_path=model_out_path)
            print(f'New best F1 model saved! Validation F1: {f1:.4f}')

        # # Save best loss model
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     save_model(current_epoch, model.state_dict(), optimizer.state_dict(), 
        #               avg_train_loss, avg_val_loss, f1, 
        #               name="best_loss",model_out_path=model_out_path)
        #     print(f'New best loss model saved! Validation Loss: {avg_val_loss:.4f}')
        
        last_avg_loss = avg_val_loss
        last_f1 = f1

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


def evaluate_and_save_metrics(name, model, test_loader, criterion, output_path, classes, ft_col):
    checkpoint = torch.load(f"{model_dir}/{name}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return evaluate_model(model, test_loader, criterion, 
                                            name=name, 
                                            verbose=True, graph_path=output_path,
                                            class_names=classes, feature_names=ft_col)

if __name__ == "__main__":
    # Set seed before any other operations
    set_all_seeds(42)

    config = Config.from_yaml('config.yml')
    data_loader = GeneralDataLoader.from_yaml(config.get_data_config_path())

    # run = wandb.init(
    #     entity=config.wandb.entity,
    #     project=config.wandb.project,
    #     config=config.transformer,
    #     mode=config.wandb.mode,
    # )
    # decoder_dict = data_loader.decoder_dict

    train_data = data_loader.get_har_dataset('train')
    val_data = data_loader.get_har_dataset('val')
    test_data = data_loader.get_har_dataset('test')


    print("X_train shape:", train_data.X.shape)
    print("X_val shape:", val_data.X.shape)
    print("X_test shape:", test_data.X.shape)
    print("Classes:", np.unique(train_data.y))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.transformer.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.transformer.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.transformer.batch_size)
    stats_pp = None
    model_dir = config.output_paths.models_dir

    


    # === Model, loss, optimizer ===
    model = v1.AccelTransformerV1(
        d_model=config.transformer.d_model,
        fc_hidden_dim=config.transformer.fc_hidden_dim,
        num_classes=len(data_loader.data_config.classes),
        in_channels=len(data_loader.data_config.ft_col),
        # in_meta_dim=config.in_meta_dim,
        nhead=config.transformer.nhead,
        # num_layers=args.num_layers,
        dropout=config.transformer.dropout,
        patch_size=16,
        kernel_stride=4,
        window_size=data_loader.data_config.window_size,
        extracted_features=config.transformer.extracted_features
    ).to(DEVICE)

    print(model)

    optimizer = Adam(model.parameters(),
                      lr=config.transformer.learning_rate, 
                      weight_decay=config.transformer.weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    train_model(config.transformer, train_loader, val_loader, model, optimizer, criterion, model_dir)
    
    print("testing most recent model:")

    plot_dir = config.output_paths.plots_dir
    classNames = data_loader.data_config.classes
    ft_col = data_loader.data_config.ft_col
    print(f"===(Testing)===")
    evaluate_and_save_metrics('last', model, test_loader, criterion, plot_dir, classNames, ft_col)

    print("testing f1 best model:")
    evaluate_and_save_metrics('best_f1', model, test_loader, criterion, plot_dir, classNames, ft_col)


    # run.finish()

    config.output_paths.clean()

    exit()