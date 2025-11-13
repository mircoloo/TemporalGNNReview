
import numpy as np
from tabulate import tabulate
from model_runners.base_runner import evaluate_decorator, BaseModelRunner
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.utils.data import DataLoader
from model_runners.runner_utils import BaseGraphDataset
import torch
from model_runners.models_dataset import HyperStockGATDataset
import matplotlib.pyplot as plt
from pathlib import Path
import json


class HyperStockGATRunner(BaseModelRunner):
    def __init__(self, model, device, market_name):
        super().__init__(model, device, market_name)
        self.model_name = "HyperStockGAT"

    def train(self, 
              train_dataset, 
              validation_dataset, 
              optimizer, 
              criterion, 
              epochs: int, 
              seq_length: int, 
              num_features: int,
              early_stopping_patience=50,
              early_stopping_metric='loss',
              early_stopping_min_delta=1e-4,
              plot_dir='training_plots'):
        """
        Train the HyperStockGAT model with optional early stopping.
        
        Args:
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            early_stopping_metric: Metric to monitor ('loss', 'acc', 'f1', 'mcc')
            early_stopping_min_delta: Minimum change to qualify as an improvement
            plot_dir: Directory to save training plots
        """
        # Convert to HyperStockGATDataset
        train_set = HyperStockGATDataset(train_dataset)
        validation_set = HyperStockGATDataset(validation_dataset)
        
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=1)

        # Early stopping setup
        best_val_metric = float('inf') if early_stopping_metric == 'loss' else float('-inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        
        # Tracking metrics for plotting
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_prec': [],
            'val_rec': [],
            'val_f1': [],
            'val_mcc': []
        }
        
        for epoch in range(1,epochs+1):
            self.model.train()
            train_loss = 0.0
            total_samples = 0
            for batch in train_loader:
                x, y, adj = batch
                #y = y.squeeze(0) # remove the batch dimension
                #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)
                #print(f"hyperstockgat input x.shape: {x.shape}, adj.shape: {adj.shape}, y.shape: {y.shape}")
                optimizer.zero_grad()
                emb = self.model.encode(x, adj)
                outputs = self.model.decode(emb, adj)
                loss = criterion(outputs, y.squeeze(0))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                total_samples += 1
            avg_train_loss = train_loss / float(max(total_samples, 1))
            history['train_loss'].append(avg_train_loss)
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}")
            

            if epoch % 1 == 0:
                
                self.model.eval()
                val_metrics = {
                    'loss': 0.0, 'acc': 0.0, 'prec': 0.0, 
                    'f1': 0.0, 'mcc': 0.0, 'rec': 0.0
                }
                n_val = 0
                
                with torch.no_grad():
                    for batch in val_loader:                            
                        x, y, adj = batch
                        #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                        x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)
                        #print(f"hyperstockgat input x.shape: {x.shape}, adj.shape: {adj.shape}, y.shape: {y.shape}")
                        targets = y.squeeze(0)
                        optimizer.zero_grad()
                        emb = self.model.encode(x, adj)
                        outputs = self.model.decode(emb, adj)
                        loss = criterion(outputs, targets)

                        # Compute metrics for batch
                        val_metrics['loss'] += criterion(outputs, targets).item()
                        
                        # Convert predictions to binary
                        preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
                        targets = targets.int().cpu()
                        
                        # Compute metrics
                        val_metrics['acc'] += accuracy_score(
                            targets.flatten(), preds.flatten())
                        val_metrics['f1'] += f1_score(
                            targets.flatten(), preds.flatten(), zero_division=0)
                        val_metrics['rec'] += recall_score(
                            targets.flatten(), preds.flatten())
                        val_metrics['mcc'] += matthews_corrcoef(
                            targets.flatten(), preds.flatten())
                        val_metrics['prec'] += precision_score(
                            targets.flatten(), preds.flatten(), zero_division=0)
                        
                        n_val += 1

                # Average metrics
                if n_val > 0:
                    for k in val_metrics:
                        val_metrics[k] /= n_val
                    
                    # Store metrics for plotting
                    history['val_loss'].append(val_metrics['loss'])
                    history['val_acc'].append(val_metrics['acc'])
                    history['val_prec'].append(val_metrics['prec'])
                    history['val_rec'].append(val_metrics['rec'])
                    history['val_f1'].append(val_metrics['f1'])
                    history['val_mcc'].append(val_metrics['mcc'])
                    
                    # Early stopping logic
                    current_metric = val_metrics[early_stopping_metric]
                    
                    # Check if improvement occurred
                    if early_stopping_metric == 'loss':
                        improved = (best_val_metric - current_metric) > early_stopping_min_delta
                    else:
                        improved = (current_metric - best_val_metric) > early_stopping_min_delta
                    
                    if improved:
                        best_val_metric = current_metric
                        patience_counter = 0
                        best_epoch = epoch
                        # Save best model state
                        best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                        improvement_marker = " ✓ NEW BEST"
                    else:
                        patience_counter += 1
                        improvement_marker = ""
                    
                    # Print results in a table format
                    if epoch % 5 == 0:
                        headers = ["Metric", "Value"]
                        table_data = [
                            ["Train Loss", f"{avg_train_loss:.4f}"],
                            ["Val Loss", f"{val_metrics['loss']:.4f}"],
                            ["Accuracy", f"{val_metrics['acc']:.4f}"],
                            ["Precision", f"{val_metrics['prec']:.4f}"],
                            ["Recall", f"{val_metrics['rec']:.4f}"],
                            ["F1 Score", f"{val_metrics['f1']:.4f}"],
                            ["MCC", f"{val_metrics['mcc']:.4f}"],
                            ["---", "---"],
                            ["Best Epoch", f"{best_epoch}"],
                            ["Patience", f"{patience_counter}/{early_stopping_patience}"]
                        ]
                        
                        print(f"\nEpoch {epoch}/{epochs} Results:{improvement_marker}")
                        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
                        print("\n")
                    
                    # Check early stopping condition
                    if patience_counter >= early_stopping_patience:
                        print(f"\n{'='*60}")
                        print(f"Early stopping triggered at epoch {epoch}")
                        print(f"Best {early_stopping_metric}: {best_val_metric:.4f} at epoch {best_epoch}")
                        print(f"{'='*60}\n")
                        
                        # Restore best model
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                            print("✓ Best model weights restored")
                        break

                self.model.train()
        
        # Plot training curves
        self._plot_training_curves(history, epochs, seq_length, num_features, optimizer, plot_dir)
    
    def _plot_training_curves(self, history, epochs, seq_length, num_features, 
                             optimizer, plot_dir='training_plots'):
        """Plot and save training and validation curves"""
        # Create directory structure
        hyperparams_str = f"seq{seq_length}_feat{num_features}_lr{optimizer.param_groups[0]['lr']:.0e}"
        save_dir = Path(plot_dir) / self.model_name / self.market_name / hyperparams_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training history as JSON
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot 1: Loss curves
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='o')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{self.model_name} - Loss Curves ({self.market_name})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Validation metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} - Validation Metrics ({self.market_name})', 
                     fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('val_acc', 'Accuracy', axes[0, 0]),
            ('val_prec', 'Precision', axes[0, 1]),
            ('val_rec', 'Recall', axes[0, 2]),
            ('val_f1', 'F1 Score', axes[1, 0]),
            ('val_mcc', 'MCC', axes[1, 1])
        ]
        
        for metric_key, metric_name, ax in metrics_to_plot:
            if history[metric_key]:
                ax.plot(epochs_range, history[metric_key], 'g-', linewidth=2, marker='o')
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel(metric_name, fontsize=10)
                ax.set_title(metric_name, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])
        
        # Hide the last subplot
        axes[1, 2].axis('off')
        
        # Add hyperparameters text
        hyperparams_text = (
            f"Hyperparameters:\n"
            f"Seq Length: {seq_length}\n"
            f"Features: {num_features}\n"
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.0e}\n"
            f"Total Epochs: {epochs}"
        )
        axes[1, 2].text(0.1, 0.5, hyperparams_text, fontsize=11, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'validation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Training plots saved to: {save_dir}")
    
    
    @evaluate_decorator
    def test(self, test_dataset):
        test_dataset = HyperStockGATDataset(test_dataset)  
        test_loader = DataLoader(test_dataset, batch_size=1)
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y, adj = batch
                #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)
                #print(f"hyperstockgat input x.shape: {x.shape}, adj.shape: {adj.shape}, y.shape: {y.shape}")
                emb = self.model.encode(x, adj)
                outputs = self.model.decode(emb, adj)
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
                all_preds.extend(preds.flatten().tolist())
                all_labels.extend(y.squeeze(0).cpu().flatten().tolist())

        return {'preds': np.array(all_preds), 'targets': np.array(all_labels)}
