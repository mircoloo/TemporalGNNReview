from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
                confusion_matrix, classification_report, roc_auc_score,
                f1_score, matthews_corrcoef, accuracy_score, 
                precision_score, recall_score, mean_absolute_error,
                mean_squared_error, r2_score
            )

class BaseModelRunner:
    def __init__(self, model, device, market_name=''):
        self.model = model
        self.device = device
        self.market_name = market_name
    def train(self, train_loader, val_loader, **kwargs):
        raise NotImplementedError
    def test(self, test_loader, **kwargs):
        raise NotImplementedError
    
    
def evaluate_decorator(func):
        """
        Decorator to evaluate the model on a given dataset with comprehensive metrics.
        """
        def wrapper(self, *args, **kwargs):
            
            
            print(f"\n{'='*80}")
            print(f"Evaluating {self.model_name} on {self.market_name} dataset...")
            
            # Call the original test function
            results = func(self, *args, **kwargs)
            # Check if results contain predictions and targets
            print(f"Results contain predictions and targets: {isinstance(results, dict) and 'preds' in results and 'targets' in results}")
            if isinstance(results, dict) and 'preds' in results and 'targets' in results:
                preds = results['preds']
                targets = results['targets']
                
                # Determine if classification or regression based on unique values
                unique_targets = np.unique(targets)
                is_binary_classification = len(unique_targets) <= 2
                print(f"Detected {'binary classification' if is_binary_classification else 'regression'} task based on target values.")
                if is_binary_classification:
                    # Classification metrics
                    print("targets.shape:", targets.shape, "preds.shape:", preds.shape)
                    acc = accuracy_score(targets, preds)
                    prec = precision_score(targets, preds, zero_division=0)
                    rec = recall_score(targets, preds, zero_division=0)
                    f1 = f1_score(targets, preds, zero_division=0)
                    mcc = matthews_corrcoef(targets, preds)
                    
                    # Generate confusion matrix
                    cm = confusion_matrix(targets, preds)
                    
                    # Try to calculate ROC AUC if probabilities are available
                    auc_roc = None
                    if 'probs' in results and len(unique_targets) == 2:
                        try:
                            auc_roc = roc_auc_score(targets, results['probs'])
                        except:
                            pass
                    
                    # Calculate class distribution
                    class_dist = np.bincount(targets.astype(int))
                    class_dist_pct = class_dist / np.sum(class_dist) * 100
                    
                    # Create metric table
                    headers = ["Metric", "Value"]
                    metrics_data = [
                        ["Accuracy", f"{acc:.4f}"],
                        ["Precision", f"{prec:.4f}"],
                        ["Recall", f"{rec:.4f}"],
                        ["F1 Score", f"{f1:.4f}"],
                        ["Matthews Correlation Coefficient", f"{mcc:.4f}"]
                    ]
                    
                    if auc_roc is not None:
                        metrics_data.append(["ROC AUC", f"{auc_roc:.4f}"])
                    
                    # Add class distribution
                    for i, (count, pct) in enumerate(zip(class_dist, class_dist_pct)):
                        metrics_data.append([f"Class {i} Count", f"{count} ({pct:.1f}%)"])
                    
                    # Print metrics table
                    print("\nClassification Metrics:")
                    print(tabulate(metrics_data, headers=headers, tablefmt="pretty"))
                    
                    # Print confusion matrix
                    print("\nConfusion Matrix:")
                    cm_table = [[f"Actual {i}", *[f"{x}" for x in row]] for i, row in enumerate(cm)]
                    cm_headers = ["", *[f"Pred {i}" for i in range(len(cm))]]
                    print(tabulate(cm_table, headers=cm_headers, tablefmt="pretty"))
                    
                    # Print classification report
                    print("\nClassification Report:")
                    print(classification_report(targets, preds))
                    
                    # Update results with metrics
                    results.update({
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1': f1,
                        'mcc': mcc,
                        'confusion_matrix': cm,
                    })
                    if auc_roc is not None:
                        results['roc_auc'] = auc_roc
                    
                else:
                    # Regression metrics
                    mae = mean_absolute_error(targets, preds)
                    mse = mean_squared_error(targets, preds)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(targets, preds)
                    
                    # Create metric table
                    headers = ["Metric", "Value"]
                    metrics_data = [
                        ["Mean Absolute Error", f"{mae:.4f}"],
                        ["Mean Squared Error", f"{mse:.4f}"],
                        ["Root Mean Squared Error", f"{rmse:.4f}"],
                        ["R-squared", f"{r2:.4f}"]
                    ]
                    
                    # Print metrics table
                    print("\nRegression Metrics:")
                    print(tabulate(metrics_data, headers=headers, tablefmt="pretty"))
                    
                    # Update results with metrics
                    results.update({
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2
                    })
            
            print(f"{'='*80}\n")
            return results
        
        return wrapper