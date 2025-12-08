from tabulate import tabulate
from .base_runner import BaseModelRunner, evaluate_decorator
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from model_runners.runner_utils import BaseGraphDataset
from torch_geometric.loader import DataLoader
import numpy as np
from model_runners.models_dataset import DARNNDataset
import matplotlib.pyplot as plt
from pathlib import Path
import json


class DARNNRunner(BaseModelRunner):
    def __init__(self, model, device, market_name):
        super().__init__(model, device, market_name)
        self.model_name = "DARNN"


    def train(self, train_dataset, val_dataset, optimizer, criterion, num_epochs, seq_length, batch_size=32,
              early_stopping_patience=20,
              early_stopping_metric='loss',
              early_stopping_min_delta=1e-5,
              plot_dir='training_plots'):
        """
        Train the DARNN model with optional early stopping.
        
        Args:
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            early_stopping_metric: Metric to monitor ('loss', 'acc', 'f1', 'mcc')
            early_stopping_min_delta: Minimum change to qualify as an improvement
            plot_dir: Directory to save training plots
        """
        train_set = DARNNDataset(train_dataset)
        val_set = DARNNDataset(val_dataset)
        
        # Use actual batching
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Early stopping setup
        best_val_metric = float('inf') if early_stopping_metric == 'loss' else float('-inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        
        # Tracking metrics for plotting
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_prec': [],
            'train_rec': [],
            'train_f1': [],
            'train_mcc': [],
            'val_loss': [],
            'val_acc': [],
            'val_prec': [],
            'val_rec': [],
            'val_f1': [],
            'val_mcc': []
        }
        
        # Track loss improvement across epochs
        prev_epoch_loss = float('inf')
        loss_improvement_history = []
        
        for epoch in range(1,num_epochs+1):
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            # Store training predictions for manual metric calculation
            train_all_preds = []
            train_all_targets = []
            train_all_logits = []
            train_batch_losses = []
            train_batch_improvements = []
            epoch_grad_norms = []
            
            print(f"\n{'#'*80}")
            print(f"# EPOCH {epoch}/{num_epochs}")
            print(f"{'#'*80}")
            
            # Training loop
            batch_idx = 0
            batch_acc = 0.0
            # Training loop
            batch_idx = 0
            batch_acc = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                X, y_target, target = batch  # Assuming your dataset returns (x, y)
                # Move batch to device
                X = X.to(self.device)
                y_target = y_target.to(self.device)
                target = target.to(self.device)
                outputs = self.model(X, y_target)
                
                # Calculate batch accuracy
                batch_acc += ((torch.round(torch.sigmoid(outputs))).detach().int().cpu() == target.detach().int().cpu()).float().mean().item()
                
                loss = criterion(outputs, target.float())
                
                loss.backward()
                
                # Debug gradients
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                epoch_grad_norms.append(total_norm)
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx} Grad Norm: {total_norm:.4f}")

                optimizer.step()

                current_batch_loss = loss.item()
                train_loss += current_batch_loss
                train_batch_losses.append(current_batch_loss)
                
                # Track batch improvement
                if n_batches > 0:
                    prev_batch_loss = train_batch_losses[-2]
                    improvement = prev_batch_loss - current_batch_loss
                    train_batch_improvements.append(improvement)
                    
                n_batches += 1
                batch_idx += 1
                
                # Collect predictions for manual metrics
                preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                train_all_preds.extend(preds.flatten().tolist())
                train_all_targets.extend(target.int().cpu().flatten().tolist())
                train_all_logits.extend(outputs.detach().cpu().flatten().tolist())
            
            # Save attention visualizations every 5 epochs
            if epoch % 5 == 0:
                self._save_attention_visualizations(epoch)

            avg_train_loss = train_loss / n_batches
            history['train_loss'].append(avg_train_loss)
            
            # Track epoch-level improvement
            epoch_improvement = prev_epoch_loss - avg_train_loss
            loss_improvement_history.append(epoch_improvement)
            
            prev_epoch_loss = avg_train_loss
            
            # Calculate training metrics manually
            train_all_preds = np.array(train_all_preds)
            train_all_targets = np.array(train_all_targets)
            train_all_logits = np.array(train_all_logits)
            
            # Gradient Analysis Summary
            if len(epoch_grad_norms) > 0:
                grad_mean = np.mean(epoch_grad_norms)
                grad_std = np.std(epoch_grad_norms)
                grad_max = np.max(epoch_grad_norms)
                grad_min = np.min(epoch_grad_norms)
                print(f"  Gradient Norms: Mean={grad_mean:.4f}, Std={grad_std:.4f}, Min={grad_min:.4f}, Max={grad_max:.4f}")
                
                if grad_max > 100:
                    print("  ⚠️ WARNING: Exploding gradients detected!")
                if grad_mean < 1e-4:
                    print("  ⚠️ WARNING: Vanishing gradients detected!")
            
            if epoch % 5 == 0:
                self._save_attention_visualizations(epoch)

            train_metrics = {
                'acc': accuracy_score(train_all_targets, train_all_preds),
                'prec': precision_score(train_all_targets, train_all_preds, zero_division=0),
                'rec': recall_score(train_all_targets, train_all_preds, zero_division=0),
                'f1': f1_score(train_all_targets, train_all_preds, zero_division=0),
                'mcc': matthews_corrcoef(train_all_targets, train_all_preds)
            }
            
            # Store training metrics in history
            history['train_acc'].append(train_metrics['acc'])
            history['train_prec'].append(train_metrics['prec'])
            history['train_rec'].append(train_metrics['rec'])
            history['train_f1'].append(train_metrics['f1'])
            history['train_mcc'].append(train_metrics['mcc'])
            
            # Get num_nodes from data shape
            num_nodes = target.shape[-1] if len(target.shape) > 1 else len(target)
            
            # Validation every 5 epochs
            if (epoch % 5 == 0):
                self.model.eval()
                val_metrics = {
                    'loss': 0.0, 'acc': 0.0, 'prec': 0.0, 
                    'f1': 0.0, 'mcc': 0.0, 'rec': 0.0
                }
                n_val = 0
                
                # Store validation predictions for manual metric calculation
                val_all_preds = []
                val_all_targets = []
                val_all_logits = []
                val_batch_losses = []
                
                with torch.no_grad():
                    for batch in val_loader:                            
                        X, y_target, target = batch  # Assuming your dataset returns (x, y)
                        # Move batch to device
                        X = X.to(self.device)
                        y_target = y_target.to(self.device)
                        targets = target.to(self.device)
                        outputs = self.model(X, y_target)
                        
                        # Compute loss for batch
                        batch_loss = criterion(outputs, targets).item()
                        val_metrics['loss'] += batch_loss
                        val_batch_losses.append(batch_loss)
                        
                        # Convert predictions to binary
                        preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                        targets_cpu = targets.int().cpu()
                        
                        # Collect for manual calculation
                        val_all_preds.extend(preds.flatten().tolist())
                        val_all_targets.extend(targets_cpu.flatten().tolist())
                        val_all_logits.extend(outputs.cpu().flatten().tolist())
                        
                        # Compute per-batch metrics (will be averaged later)
                        val_metrics['acc'] += accuracy_score(
                            targets_cpu.flatten(), preds.flatten())
                        val_metrics['f1'] += f1_score(
                            targets_cpu.flatten(), preds.flatten(), zero_division=0)
                        val_metrics['rec'] += recall_score(
                            targets_cpu.flatten(), preds.flatten())
                        val_metrics['mcc'] += matthews_corrcoef(
                            targets_cpu.flatten(), preds.flatten())
                        val_metrics['prec'] += precision_score(
                            targets_cpu.flatten(), preds.flatten(), zero_division=0)
                        
                        n_val += 1

                # Average metrics
                if n_val > 0:
                    # Average per-batch metrics
                    for k in val_metrics:
                        val_metrics[k] /= n_val
                    
                    # Calculate manual metrics on all predictions
                    val_all_preds = np.array(val_all_preds)
                    val_all_targets = np.array(val_all_targets)
                    val_all_logits = np.array(val_all_logits)
                    
                    manual_val_metrics = {
                        'acc': accuracy_score(val_all_targets, val_all_preds),
                        'prec': precision_score(val_all_targets, val_all_preds, zero_division=0),
                        'rec': recall_score(val_all_targets, val_all_preds, zero_division=0),
                        'f1': f1_score(val_all_targets, val_all_preds, zero_division=0),
                        'mcc': matthews_corrcoef(val_all_targets, val_all_preds)
                    }
                    
                    # Print validation debug info
                    print(f"\n{'='*80}")
                    print(f"VALIDATION DEBUG - Epoch {epoch}/{num_epochs}")
                    print(f"{'='*80}")
                    print(f"Val Loss (avg): {val_metrics['loss']:.6f}")
                    print(f"Val Loss (min batch): {min(val_batch_losses):.6f}")
                    print(f"Val Loss (max batch): {max(val_batch_losses):.6f}")
                    
                    print(f"\n🔍 VALIDATION LOGITS ANALYSIS:")
                    print(f"  Logits - Min: {val_all_logits.min():.4f}, Max: {val_all_logits.max():.4f}, "
                          f"Mean: {val_all_logits.mean():.4f}, Std: {val_all_logits.std():.4f}")
                    
                    # Check if logits are all the same
                    unique_val_logits = np.unique(np.round(val_all_logits, 4))
                    print(f"  Unique logit values: {len(unique_val_logits)}")
                    if len(unique_val_logits) < 10:
                        print(f"  ⚠️ WARNING: Very few unique logit values! {unique_val_logits[:10]}")
                    
                    # Check variance across different nodes
                    val_all_logits_reshaped = np.array(val_all_logits).reshape(-1, num_nodes)
                    per_node_variance_val = val_all_logits_reshaped.var(axis=0)
                    print(f"  Per-node variance - Mean: {per_node_variance_val.mean():.6f}, "
                          f"Min: {per_node_variance_val.min():.6f}, Max: {per_node_variance_val.max():.6f}")
                    print(f"  Nodes with zero variance: {np.sum(per_node_variance_val < 1e-6)}/{num_nodes}")
                    
                    print(f"\n📊 VALIDATION PREDICTIONS:")
                    print(f"  Sample logits (first 20): {val_all_logits[:20]}")
                    print(f"  Sample targets (first 20): {val_all_targets[:20]}")
                    print(f"  Sample preds (first 20): {val_all_preds[:20]}")
                    
                    # Check if all predictions are the same
                    unique_val_preds = np.unique(val_all_preds)
                    print(f"  Unique predictions: {unique_val_preds}")
                    if len(unique_val_preds) == 1:
                        print(f"  ⚠️ CRITICAL: Model always predicts class {unique_val_preds[0]}!")
                    
                    print(f"\n📈 VALIDATION METRICS:")
                    print(f"  Manual Validation Metrics:")
                    print(f"    Acc: {manual_val_metrics['acc']:.4f}, "
                          f"Prec: {manual_val_metrics['prec']:.4f}, "
                          f"Rec: {manual_val_metrics['rec']:.4f}, "
                          f"F1: {manual_val_metrics['f1']:.4f}, "
                          f"MCC: {manual_val_metrics['mcc']:.4f}")
                    print(f"  Per-Batch Averaged Metrics:")
                    print(f"    Acc: {val_metrics['acc']:.4f}, "
                          f"Prec: {val_metrics['prec']:.4f}, "
                          f"Rec: {val_metrics['rec']:.4f}, "
                          f"F1: {val_metrics['f1']:.4f}, "
                          f"MCC: {val_metrics['mcc']:.4f}")
                    print(f"  Class distribution - Targets: {np.bincount(val_all_targets.astype(int))}")
                    print(f"  Class distribution - Preds: {np.bincount(val_all_preds.astype(int))}")
                    
                    # Store metrics for plotting (using manual metrics for validation)
                    history['val_loss'].append(val_metrics['loss'])
                    history['val_acc'].append(manual_val_metrics['acc'])
                    history['val_prec'].append(manual_val_metrics['prec'])
                    history['val_rec'].append(manual_val_metrics['rec'])
                    history['val_f1'].append(manual_val_metrics['f1'])
                    history['val_mcc'].append(manual_val_metrics['mcc'])
                    
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
                    if epoch % 1 == 0 or epoch == num_epochs:
                        headers = ["Metric", "Train", "Val (Manual)"]
                        table_data = [
                            ["Loss", f"{avg_train_loss:.4f}", f"{val_metrics['loss']:.4f}"],
                            ["Accuracy", f"{train_metrics['acc']:.4f}", f"{manual_val_metrics['acc']:.4f}"],
                            ["Precision", f"{train_metrics['prec']:.4f}", f"{manual_val_metrics['prec']:.4f}"],
                            ["Recall", f"{train_metrics['rec']:.4f}", f"{manual_val_metrics['rec']:.4f}"],
                            ["F1 Score", f"{train_metrics['f1']:.4f}", f"{manual_val_metrics['f1']:.4f}"],
                            ["MCC", f"{train_metrics['mcc']:.4f}", f"{manual_val_metrics['mcc']:.4f}"],
                            ["---", "---", "---"],
                            ["Best Epoch", "", f"{best_epoch}"],
                            ["Patience", "", f"{patience_counter}/{early_stopping_patience}"]
                        ]
                        
                        print(f"\n{'='*70}")
                        print(f"Epoch {epoch}/{num_epochs} Summary:{improvement_marker}")
                        print(f"{'='*70}")
                        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
                        print(f"{'='*70}\n")
                        
                        # Plot and save training curves after every 5 epochs
                        self._plot_training_curves_interim(history, epoch, seq_length, batch_size, 
                                                           optimizer, plot_dir)
                    
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
        
        # Plot final training curves
        self._plot_training_curves(history, num_epochs, seq_length, batch_size, optimizer, plot_dir)
    
    def _plot_training_curves_interim(self, history, current_epoch, seq_length, batch_size, 
                                     optimizer, plot_dir='training_plots'):
        """Plot and save interim training curves during training"""
        # Create directory structure
        hyperparams_str = f"seq{seq_length}_bs{batch_size}_lr{optimizer.param_groups[0]['lr']:.0e}"
        save_dir = Path(plot_dir) / self.model_name / self.market_name / hyperparams_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with loss and accuracy subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        epochs_range = range(1, len(history['train_loss']) + 1)
        val_epochs = range(5, current_epoch + 1, 5)  # Validation every 5 epochs
        
        # Plot 1: Loss curves
        ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o')
        ax1.plot(val_epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title(f'Loss Curves (Epoch {current_epoch})', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2.plot(epochs_range, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o')
        ax2.plot(val_epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2, marker='s')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title(f'Accuracy Curves (Epoch {current_epoch})', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_dir / f'training_progress_epoch_{current_epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_training_curves(self, history, num_epochs, seq_length, batch_size, 
                             optimizer, plot_dir='training_plots'):
        """Plot and save training and validation curves"""
        # Create directory structure
        hyperparams_str = f"seq{seq_length}_bs{batch_size}_lr{optimizer.param_groups[0]['lr']:.0e}"
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
    def test(self, test_dataset, save_attention=False):
        test_dataset = DARNNDataset(test_dataset)  
        test_loader = DataLoader(test_dataset, batch_size=1)
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                X, y_target, target = batch
                X = X.to(self.device)
                y_target = y_target.to(self.device)
                target = target.to(self.device)
                
                outputs = self.model(X, y_target)
                preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                all_preds.extend(preds.flatten().tolist())
                all_labels.extend(target.int().cpu().flatten().tolist())
                
                if save_attention and i == 0:
                    self._save_attention_visualizations('test_final')

        return {'preds': np.array(all_preds), 'targets': np.array(all_labels)}
    
    def _save_attention_visualizations(self, epoch_or_tag):
        """Save attention visualizations for DARNN"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        
        # Handle both int epoch and string tag
        if isinstance(epoch_or_tag, int):
            folder_name = f'epoch_{epoch_or_tag}'
            title_suffix = f'Epoch {epoch_or_tag}'
        else:
            folder_name = str(epoch_or_tag)
            title_suffix = str(epoch_or_tag)
            
        save_dir = Path('training_plots') / 'DARNN' / 'attention' / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Input Attention (Encoder)
        # self.model.encoder.input_attention_weights is a list of tensors [batch_size, N] for each timestep T
        if hasattr(self.model.encoder, 'input_attention_weights') and self.model.encoder.input_attention_weights:
            # Stack to get [T, batch_size, N]
            input_attn = torch.stack(self.model.encoder.input_attention_weights)
            # Permute to [batch_size, T, N]
            input_attn = input_attn.permute(1, 0, 2)
            # Take first sample in batch: [T, N]
            attn_map = input_attn[0].numpy()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(attn_map.T, cmap='viridis', annot=False) # Transpose to have N on y-axis, T on x-axis
            plt.title(f'Input Attention (Encoder) - {title_suffix}')
            plt.xlabel('Time Step')
            plt.ylabel('Stock / Series')
            plt.tight_layout()
            plt.savefig(save_dir / 'input_attention.png')
            plt.close()
            
        # 2. Temporal Attention (Decoder)
        # self.model.decoders is a ModuleList. We can visualize a few of them.
        num_stocks_to_viz = min(5, len(self.model.decoders))
        
        for i in range(num_stocks_to_viz):
            decoder = self.model.decoders[i]
            if hasattr(decoder, 'temporal_attention_weights') and decoder.temporal_attention_weights:
                try:
                    # Stack to get [T, batch_size, M] (M is encoder hidden size)
                    temp_attn = torch.stack(decoder.temporal_attention_weights)
                    
                    # Fix: Handle 4D tensor case (e.g. [T, Batch, 1, M])
                    if temp_attn.dim() == 4:
                        temp_attn = temp_attn.squeeze(2)
                        
                    # Permute to [batch_size, T, M]
                    temp_attn = temp_attn.permute(1, 0, 2)
                    # Take first sample: [T, M]
                    attn_map = temp_attn[0].detach().cpu().numpy()
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(attn_map.T, cmap='viridis', annot=False)
                    plt.title(f'Temporal Attention (Decoder Stock {i}) - {title_suffix}')
                    plt.xlabel('Time Step')
                    plt.ylabel('Encoder Hidden State')
                    plt.tight_layout()
                    plt.savefig(save_dir / f'temporal_attention_stock_{i}.png')
                    plt.close()
                except Exception as e:
                    print(f"Failed to plot temporal attention for stock {i}: {e}")
                    plt.close()
