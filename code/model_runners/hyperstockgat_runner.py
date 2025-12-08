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
              batch_size=32,
              early_stopping_patience=20,
              early_stopping_metric='loss',
              early_stopping_min_delta=1e-4,
              plot_dir='training_plots'):
        """
        Train the HyperStockGAT model with optional early stopping.
        
        Args:
            batch_size: Batch size for training
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            early_stopping_metric: Metric to monitor ('loss', 'acc', 'f1', 'mcc')
            early_stopping_min_delta: Minimum change to qualify as an improvement
            plot_dir: Directory to save training plots
        """
        # Convert to HyperStockGATDataset
        train_set = HyperStockGATDataset(train_dataset)
        validation_set = HyperStockGATDataset(validation_dataset)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=1)

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
        
        for epoch in range(1,epochs+1):
            self.model.train()
            train_loss = 0.0
            total_samples = 0
            
            # Store training predictions for manual metric calculation
            train_all_preds = []
            train_all_targets = []
            train_all_logits = []
            train_batch_losses = []
            train_batch_improvements = []
            epoch_grad_norms = []
            
            print(f"\n{'#'*80}")
            print(f"# EPOCH {epoch}/{epochs}")
            print(f"{'#'*80}")
            
            # Training loop
            batch_idx = 0
            batch_acc = 0.0
            for batch in train_loader:
                x, y, adj = batch
                #y = y.squeeze(0) # remove the batch dimension
                #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)
                
                # Fix for HyperStockGAT: Model expects [Nodes, Features], not [Batch, Nodes, Features]
                if x.dim() == 3 and x.size(0) == 1:
                    x = x.squeeze(0)
                if adj.dim() == 3 and adj.size(0) == 1:
                    adj = adj.squeeze(0)

                #print(f"hyperstockgat input x.shape: {x.shape}, adj.shape: {adj.shape}, y.shape: {y.shape}")
                optimizer.zero_grad()
                emb = self.model.encode(x, adj)
                outputs = self.model.decode(emb, adj)
                targets = y.squeeze(0)
                
                # Calculate batch accuracy
                batch_acc += ((torch.round(torch.sigmoid(outputs))).detach().int().cpu() == targets.detach().int().cpu()).float().mean().item()
                
                loss = criterion(outputs, targets)
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
                if total_samples > 0:
                    prev_batch_loss = train_batch_losses[-2]
                    improvement = prev_batch_loss - current_batch_loss
                    train_batch_improvements.append(improvement)
                    
                total_samples += 1
                batch_idx += 1
                
                # Collect predictions for manual metrics
                preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                train_all_preds.extend(preds.flatten().tolist())
                train_all_targets.extend(targets.int().cpu().flatten().tolist())
                train_all_logits.extend(outputs.detach().cpu().flatten().tolist())
            
            avg_train_loss = train_loss / float(max(total_samples, 1))
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
            
            # Get num_nodes from data
            num_nodes = targets.shape[-1] if len(targets.shape) > 1 else len(targets)
            

            if epoch % 5 == 0:
                # Save attention visualizations
                self._save_attention_visualizations(epoch)
                
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
                        x, y, adj = batch
                        #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                        x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)
                        
                        # Fix for HyperStockGAT: Model expects [Nodes, Features], not [Batch, Nodes, Features]
                        if x.dim() == 3 and x.size(0) == 1:
                            x = x.squeeze(0)
                        if adj.dim() == 3 and adj.size(0) == 1:
                            adj = adj.squeeze(0)

                        #print(f"hyperstockgat input x.shape: {x.shape}, adj.shape: {adj.shape}, y.shape: {y.shape}")
                        targets = y.squeeze(0)
                        optimizer.zero_grad()
                        emb = self.model.encode(x, adj)
                        outputs = self.model.decode(emb, adj)
                        
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
                            targets_cpu.flatten(), preds.flatten(), zero_division=0)
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
                    if epoch % 5 == 0:
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
                        print(f"Epoch {epoch}/{epochs} Summary:{improvement_marker}")
                        print(f"{'='*70}")
                        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
                        print(f"{'='*70}\n")
                        
                        # Plot and save training curves after every 5 epochs
                        #self._plot_training_curves_interim(history, epoch, seq_length, num_features, optimizer, plot_dir)
                    
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
        self._plot_training_curves(history, epochs, seq_length, num_features, optimizer, plot_dir)
    
    def _plot_training_curves_interim(self, history, current_epoch, seq_length, num_features, 
                                     optimizer, plot_dir='training_plots'):
        """Plot and save interim training curves during training"""
        # Create directory structure
        hyperparams_str = f"seq{seq_length}_feat{num_features}_lr{optimizer.param_groups[0]['lr']:.0e}"
        save_dir = Path(plot_dir) / self.model_name / self.market_name / hyperparams_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with loss and accuracy subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        # Plot 1: Loss curves
        ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o')
        ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title(f'Loss Curves (Epoch {current_epoch})', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2.plot(epochs_range, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o')
        ax2.plot(epochs_range, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2, marker='s')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title(f'Accuracy Curves (Epoch {current_epoch})', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_dir / f'training_progress_epoch_{current_epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        
        # Plot 1: Loss curves (Train vs Val)
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o')
        ax.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.model_name} - Loss Curves ({self.market_name})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Accuracy curves (Train vs Val)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs_range, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o')
        ax.plot(epochs_range, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2, marker='s')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.model_name} - Accuracy Curves ({self.market_name})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(save_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: All metrics comparison (2x3 grid)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{self.model_name} - All Metrics ({self.market_name})', 
                     fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            (('train_acc', 'val_acc'), 'Accuracy', axes[0, 0]),
            (('train_prec', 'val_prec'), 'Precision', axes[0, 1]),
            (('train_rec', 'val_rec'), 'Recall', axes[0, 2]),
            (('train_f1', 'val_f1'), 'F1 Score', axes[1, 0]),
            (('train_mcc', 'val_mcc'), 'MCC', axes[1, 1])
        ]
        
        for (train_key, val_key), metric_name, ax in metrics_to_plot:
            if history[train_key] and history[val_key]:
                ax.plot(epochs_range, history[train_key], 'b-', linewidth=2, marker='o', label='Train')
                ax.plot(epochs_range, history[val_key], 'r-', linewidth=2, marker='s', label='Val')
                ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
                ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
                ax.set_title(metric_name, fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])
        
        # Add hyperparameters text in the last subplot
        axes[1, 2].axis('off')
        hyperparams_text = (
            f"Hyperparameters:\n"
            f"Seq Length: {seq_length}\n"
            f"Features: {num_features}\n"
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.0e}\n"
            f"Total Epochs: {len(epochs_range)}"
        )
        axes[1, 2].text(0.1, 0.5, hyperparams_text, fontsize=11, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'all_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Training plots saved to: {save_dir}")
        print(f"  - loss_curves.png")
        print(f"  - accuracy_curves.png")
        print(f"  - all_metrics.png")
        print(f"  - training_history.json")
    
    
    @evaluate_decorator
    def test(self, test_dataset, save_attention=False):
        test_dataset = HyperStockGATDataset(test_dataset)  
        test_loader = DataLoader(test_dataset, batch_size=1)
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                x, y, adj = batch
                #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)
                
                # Fix for HyperStockGAT: Model expects [Nodes, Features], not [Batch, Nodes, Features]
                if x.dim() == 3 and x.size(0) == 1:
                    x = x.squeeze(0)
                if adj.dim() == 3 and adj.size(0) == 1:
                    adj = adj.squeeze(0)

                #print(f"hyperstockgat input x.shape: {x.shape}, adj.shape: {adj.shape}, y.shape: {y.shape}")
                emb = self.model.encode(x, adj)
                outputs = self.model.decode(emb, adj)
                preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                all_preds.extend(preds.flatten().tolist())
                all_labels.extend(y.squeeze(0).cpu().flatten().tolist())
                
                if save_attention and i == 0:
                    self._save_attention_visualizations('test_final')

        return {'preds': np.array(all_preds), 'targets': np.array(all_labels)}
    
    def _save_attention_visualizations(self, epoch_or_tag):
        """Save attention visualizations for HyperStockGAT"""
        # Only for HGCN model which has temporal attention
        if not hasattr(self.model, 'temporal_attention_1'):
            return
            
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
        
        save_dir = Path('training_plots') / 'HyperStockGAT' / 'attention' / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Temporal Attention 1
        if self.model.temporal_attention_1 is not None:
            # Shape: [B, T, T] -> Take first sample [T, T]
            attn = self.model.temporal_attention_1[0].numpy()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn, cmap='viridis', annot=False)
            plt.title(f'Temporal Attention 1 (Layer 1) - {title_suffix}')
            plt.xlabel('Time Step')
            plt.ylabel('Time Step')
            plt.tight_layout()
            plt.savefig(save_dir / 'temporal_attention_1.png')
            plt.close()
            
        # 2. Temporal Attention 2
        if hasattr(self.model, 'temporal_attention_2') and self.model.temporal_attention_2 is not None:
            # Shape: [B, T, T]
            attn = self.model.temporal_attention_2[0].numpy()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn, cmap='viridis', annot=False)
            plt.title(f'Temporal Attention 2 (Layer 2) - {title_suffix}')
            plt.xlabel('Time Step')
            plt.ylabel('Time Step')
            plt.tight_layout()
            plt.savefig(save_dir / 'temporal_attention_2.png')
            plt.close()
