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
from har_model import AccelTransformer, HARWindowDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import logging
from utils import TConfig
import yaml
import wandb

DEBUG_MODE = True
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
raw_data_urls = [f"{data_dir}{num}.csv" for num in dataset_numbers]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, data_loader, criterion, name="model",verbose=False, graph=False):
    model.eval()
    total_loss = 0.0
    predictions = []
    true = []
    all_outputs = []
    print(f"===(Validation)===")
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
            
            # Update progress bar
            current_avg_loss = total_loss / ((batch_idx + 1) * x_seq.size(0))
            pbar.set_description(f"Loss: {current_avg_loss:.4f}")

    # After the loop, concatenate all outputs
    all_outputs = torch.cat(all_outputs, dim=0)

    # Calculate validation metrics
    predictions = np.array(predictions)
    true = np.array(true)
    precision, recall, f1, _ = precision_recall_fscore_support(true, predictions, average='macro')
    avg_loss = total_loss / len(data_loader.dataset)

    if verbose:
        print(classification_report(true, predictions))

    if graph:
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

        # Log confusion matrix
        cm = confusion_matrix(true, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=decoder_dict.values(),
                    yticklabels=decoder_dict.values())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Raw Data)')
        plt.savefig(f'{name}_confusion_matrix.png')
        im = plt.imread(f'{name}_confusion_matrix.png')
        if not DEBUG_MODE:
            run.log({
                f"{name}_confusion_matrix": wandb.Image(im, caption=f"{name} Confusion Matrix")
            })

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
        plt.savefig(f'{name}_roc_curves.png')
        roc_im = plt.imread(f'{name}_roc_curves.png')
        if not DEBUG_MODE:
            run.log({
                f"{name}_roc_curves": wandb.Image(roc_im, caption=f"{name} ROC Curves")
            })
        plt.close()
        
    
    return avg_loss, f1
    # print(f"Validation - Avg Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}")
    
    


if __name__ == "__main__":
    
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    if not DEBUG_MODE:
        run.config.update(config['transformer'])
    else:
        print(f"{ANSI_CYAN}DEBUG MODE: Will not log to wandb{ANSI_RESET}")

    args = TConfig(**config['transformer'])

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    
    X_all = []
    X_meta_all = []
    y_all = []

    for file_path in tqdm(raw_data_urls):
        for sensor_loc in args.sensor_loc:
            try:
                X, X_meta, y = load_and_process_data(file_path, args, sensor_loc)
            except Exception as e:
                print(f"Error processing {file_path} with {sensor_loc}: {e}")
                continue
            X_all.append(X)
            X_meta_all.append(X_meta)
            y_all.append(y)

    # Split subjects into train/val/test before concatenating
    n_subjects = len(X_all)
    indices = np.arange(n_subjects)
    
    # First split: 60% train, 40% temp
    train_indices, temp_indices = train_test_split(
        indices, test_size=args.test_size, random_state=args.random_seed
    )
    
    # Second split: 20% val, 20% test (from the 40% temp)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=args.random_seed
    )
    
    # Concatenate data for each split
    X_train = np.concatenate([X_all[i] for i in train_indices], axis=0)
    X_meta_train = np.concatenate([X_meta_all[i] for i in train_indices], axis=0)
    y_train = np.concatenate([y_all[i] for i in train_indices], axis=0).ravel()
    
    X_val = np.concatenate([X_all[i] for i in val_indices], axis=0)
    X_meta_val = np.concatenate([X_meta_all[i] for i in val_indices], axis=0)
    y_val = np.concatenate([y_all[i] for i in val_indices], axis=0).ravel()
    
    X_test = np.concatenate([X_all[i] for i in test_indices], axis=0)
    X_meta_test = np.concatenate([X_meta_all[i] for i in test_indices], axis=0)
    y_test = np.concatenate([y_all[i] for i in test_indices], axis=0).ravel()
    
    # Encode labels after splitting
    y_train_int, encoder_dict, decoder_dict = encode_labels(y_train)
    y_val_int = np.array([encoder_dict[label] for label in y_val])
    y_test_int = np.array([encoder_dict[label] for label in y_test])

    assert X_train.shape[-1] == args.in_seq_dim
    assert X_meta_train.shape[-1] == args.in_meta_dim
    assert len(encoder_dict) == args.num_classes

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    print("Classes:", np.unique(y_train))

    train_dataset = HARWindowDataset(X_train, X_meta_train, y_train_int)
    val_dataset = HARWindowDataset(X_val, X_meta_val, y_val_int)
    test_dataset = HARWindowDataset(X_test, X_meta_test, y_test_int)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    


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
    optimizer = Adam(model.parameters(),
                      lr=args.learning_rate, 
                      weight_decay=args.weight_decay)
    
    best_val_f1 = 0.0


    
    if args.load_model_path:
        checkpoint = torch.load(args.load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_f1 = checkpoint['val_f1']

    criterion = nn.CrossEntropyLoss()
    
    
    # === Training loop ===
    best_model_state = None
    patience_counter = 0
    last_epoch = 0
    print(f"{DEVICE=}")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        current_avg_loss = 0.0
        current_accuracy = 0.0
        predictions = []
        true = []
        correct = 0
        total = 0
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f"===(Training)===")
        pbar = tqdm(train_loader)
        for batch_idx, (x_seq, x_meta, y_true) in enumerate(pbar):
            x_seq, x_meta, y_true = x_seq.to(DEVICE), x_meta.to(DEVICE), y_true.to(DEVICE)

            # Training specific steps
            optimizer.zero_grad()
            outputs = model(x_seq, x_meta)
            loss = criterion(outputs, y_true)
            loss.backward()
            optimizer.step()

            # Consistent prediction handling
            pred_classes = torch.argmax(outputs, dim=1)
            predictions.extend(pred_classes.cpu().numpy())
            true.extend(y_true.cpu().numpy())
            
            # Accumulate batch loss (consistent with eval loop)
            train_loss += loss.item() * x_seq.size(0)
            total += y_true.size(0)
            
            # Update progress bar
            current_avg_loss = train_loss / ((batch_idx + 1) * x_seq.size(0))
            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                current_accuracy = 100. * accuracy_score(true, predictions)
                pbar.set_description(f"Loss: {current_avg_loss:.4f}, Acc: {current_accuracy:.2f}%")

        avg_train_loss = train_loss / total
        train_accuracy = 100. * correct / total

        # Validation phase
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_f1': f1,
            }, 'best_accel_transformer.pth')
            print(f'New best model saved! Validation F1: {f1:.4f}')
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= args.patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_f1': f1,
            }, 'last_accel_transformer.pth')
            last_epoch = epoch
            break
        print()

    print("testing most recent model:")
    last_epoch = args.epochs if last_epoch == 0 else last_epoch

    last_avg_loss, last_f1 = evaluate_model(model, test_loader, criterion, name=f"last_model_ep{last_epoch}", verbose=True, graph=True)
    if last_epoch != args.epochs:
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': last_avg_loss,
            'val_f1': last_f1,
        }, 'last_accel_transformer.pth')
            
    print("testing best model:")
    checkpoint = torch.load("best_accel_transformer.pth")

    model.load_state_dict(checkpoint['model_state_dict'])
    best_model_avg_loss, best_model_f1 = evaluate_model(model, test_loader, criterion, name=f"best_model_ep{checkpoint['epoch']}", verbose=True, graph=True)

    if not DEBUG_MODE:
        run.finish()

    exit()

    # Load the best model
    # if best_model_state is not None:
    #     model.load_state_dict(best_model_state)
    
    model.eval()
    eval_metrics = {}
    with torch.no_grad():
        total_loss = 0
        predictions = []
        actuals = []
        correct = 0
        total = 0
        
        for batch_idx, (x_seq, x_meta, y_true) in enumerate(test_loader):
            x_seq, x_meta, y_true = x_seq.to(DEVICE), x_meta.to(DEVICE), y_true.to(DEVICE)
            outputs = model(x_seq, x_meta)
            loss = criterion(outputs, y_true)
            total_loss += loss.item()
            
            # Convert outputs to class predictions
            pred_classes = torch.argmax(outputs, dim=1)
            true_classes = y_true
            
            # Calculate accuracy
            correct += (pred_classes == true_classes).sum().item()
            total += true_classes.size(0)
            
            # Store predictions and actual values
            predictions.extend(pred_classes.cpu().numpy())
            actuals.extend(true_classes.cpu().numpy())
        
        test_accuracy = 100. * correct / total
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
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
