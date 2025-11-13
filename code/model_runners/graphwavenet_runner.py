#from model_runners.runner_utils import BaseGraphDataset
import numpy as np
from tabulate import tabulate
from model_runners.models_dataset import GraphWaveNetDataset
from model_runners.base_runner import BaseModelRunner, evaluate_decorator
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import matplotlib.pyplot as plt
from pathlib import Path
import json 

        


class GraphWaveNetRunner(BaseModelRunner):
    def __init__(self, model, device, market_name):
        super().__init__(model, device, market_name)
        self.model_name = "GWNet"
        
        
    def train(self, train_dataset, 
              val_dataset, 
              optimizer, 
              criterion, 
              num_epochs, 
              seq_length, 
              num_features, 
              batch_size=32,
              threshold=0.5,
              early_stopping_patience=50,
              early_stopping_metric='loss',
              early_stopping_min_delta=1e-4,
              plot_dir='training_plots'):
        """
        Train the GraphWaveNet model with optional early stopping.
        
        Args:
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            early_stopping_metric: Metric to monitor ('loss', 'acc', 'f1', 'mcc')
            early_stopping_min_delta: Minimum change to qualify as an improvement
            plot_dir: Directory to save training plots
        """
        # Create organized TensorBoard writer
        train_set = GraphWaveNetDataset(train_dataset)
        val_set = GraphWaveNetDataset(val_dataset)
        # Use actual batching
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1)

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
        
        for epoch in range(1, num_epochs+1):
            self.model.train()
            train_loss = 0.0
            n_train = 0

            for batch in train_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                                
                optimizer.zero_grad()
                
                # Forward pass with batched inputs
                output = self.model(x)  # [B, num_features, num_nodes, seq_length]
                output_for_loss = output[:, :, :, -1]  # Take last timestep
                predict = output_for_loss.squeeze(1)  # Remove feature dimension
                real = y.float()  # [B, num_nodes]
                
                loss = criterion(predict, real)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_train += 1

            # Training metrics
            avg_train_loss = train_loss / n_train
            history['train_loss'].append(avg_train_loss)
            
            # Validation every 5 epochs
            if val_dataset and (epoch % 5 == 0):
                self.model.eval()
                val_metrics = {
                    'loss': 0.0, 'acc': 0.0, 'prec': 0.0, 
                    'f1': 0.0, 'mcc': 0.0, 'rec': 0.0
                }
                n_val = 0
                
                with torch.no_grad():
                    for batch in val_loader:          
                        x, y = batch
                        x = x.to(self.device)
                        y = y.to(self.device)
                        
                        output = self.model(x)
                        output_for_loss = output[:, :, :, -1]  # Take last timestep
                        predict = output_for_loss.squeeze(1)  # Remove feature dimension
                        targets = y.float().squeeze()  # [B, num_nodes]
                        output = output.squeeze()
                        y = y.squeeze()
                        #print(f"Validation batch x.shape: {x.shape}, y.shape: {y.shape}, outputs.shape: {outputs.shape}, predict.shape: {predict.shape}, targets.shape: {targets.shape}")
                        # Compute metrics for batch
                        #print(f"{outputs.shape=}, {targets.shape=}")
                        val_metrics['loss'] += criterion(output, targets).item()

                        # Convert predictions to binary
                        preds = (torch.sigmoid(output) > 0.5).int().cpu()
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
                    if epoch % 5 == 0 or epoch == num_epochs:
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
                        
                        print(f"\nEpoch {epoch}/{num_epochs} Results:{improvement_marker}")
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
        self._plot_training_curves(history, num_epochs, seq_length, num_features, 
                                   batch_size, optimizer, plot_dir)

    def _plot_training_curves(self, history, num_epochs, seq_length, num_features, 
                             batch_size, optimizer, plot_dir='training_plots'):
        """Plot and save training and validation curves"""
        # Create directory structure
        hyperparams_str = f"seq{seq_length}_feat{num_features}_bs{batch_size}_lr{optimizer.param_groups[0]['lr']:.0e}"
        save_dir = Path(plot_dir) / self.model_name / self.market_name / hyperparams_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training history as JSON
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot 1: Loss curves
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, len(history['train_loss']) + 1)
        val_epochs = range(5, len(history['train_loss']) + 1, 5)  # Validation every 5 epochs
        
        ax.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(val_epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='o')
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
                ax.plot(val_epochs, history[metric_key], 'g-', linewidth=2, marker='o')
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
            f"Batch Size: {batch_size}\n"
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.0e}\n"
            f"Total Epochs: {num_epochs}"
        )
        axes[1, 2].text(0.1, 0.5, hyperparams_text, fontsize=11, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'validation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Training plots saved to: {save_dir}")

    @evaluate_decorator
    def test(self, test_dataset, seq_length, num_features, batch_size=32, config=None, threshold=0.5):
        test_set = GraphWaveNetDataset(test_dataset)
        test_loader = DataLoader(test_set, batch_size=1)
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                output_for_loss = output[:, :, :, -1]  # Take last timestep
                predict = output_for_loss.squeeze(1)  # Remove feature dimension
                targets = y.float().squeeze()  # [B, num_nodes]
                output = output.squeeze()
                y = y.squeeze()
                #print(f"Validation batch x.shape: {x.shape}, y.shape: {y.shape}, outputs.shape: {outputs.shape}, predict.shape: {predict.shape}, targets.shape: {targets.shape}")
                # Compute metrics for batch
                #print(f"{outputs.shape=}, {targets.shape=}")
                # Convert predictions to binary
                preds = (torch.sigmoid(output) > 0.5).int().cpu()
                targets = targets.int().cpu()
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        # Concatenate all predictions and targets
        y_pred = torch.cat(all_preds, dim=0).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()
        
        # Calculate metrics
        test_metrics = {
            'accuracy': accuracy_score(y_true.flatten(), y_pred.flatten()),
            'precision': precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
            'recall': recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
            'f1': f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
            'mcc': matthews_corrcoef(y_true.flatten(), y_pred.flatten())
        }
        
        # Log experiment if config is provided
        if config is not None:
            dataset_info = {
                'seq_length': seq_length,
                'num_features': num_features,
                'num_samples': len(test_dataset),
                'num_nodes': test_dataset[0].x.shape[0]
            }
            
            train_params = {
                'batch_size': batch_size,
                'num_epochs': config.get('num_epochs', None),
                'learning_rate': config.get('learning_rate', None),
                'optimizer': config.get('optimizer', 'Adam')
            }


        return {'preds': np.array(y_pred), 'targets': np.array(y_true)}