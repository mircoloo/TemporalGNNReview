from model_runners.base_runner import evaluate_decorator, BaseModelRunner
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import numpy as np
from tqdm.auto import tqdm
from tabulate import tabulate
from pathlib import Path
from model_runners.models_dataset import DTMLDataset
import pandas as pd
import matplotlib.pyplot as plt
import json

class DTMLRunner(BaseModelRunner):

    def __init__(self, model, device, market_name, feature_normalization = 'none', dates: list[list[str]] = None, window_size: int = None, market: str = ''):
        super().__init__(model, device, market_name)
        self.model_name = 'DTML'
        self.feature_normalization = feature_normalization
        self.dates = dates
        self.window_size = window_size
        self.market = market.upper()
        self.norm_values = {'minmax': None, 'zscore': None} #minmax [min, max], zscore [mean, std]
        
        if self.dates:
            self.train_indexes = self.load_index(self.dates[0], is_train=True)
            self.val_indexes = self.load_index(self.dates[1])
            self.test_indexes = self.load_index(self.dates[2])
        

    def train(self, train_dataset, 
              val_dataset, 
              optimizer, 
              criterion, 
              num_epochs,
              n_features,
              batch_size=1, 
              use_validation=True,
              early_stopping_patience=20,
              early_stopping_metric='loss',
              early_stopping_min_delta=1e-6,
              plot_dir='training_plots'):
        """
        Train the DTML model with optional early stopping.
        
        Args:
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            early_stopping_metric: Metric to monitor ('loss', 'acc', 'f1', 'mcc')
            early_stopping_min_delta: Minimum change to qualify as an improvement
        """
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.batch_size = batch_size

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

        train_set = DTMLDataset(train_dataset, self.train_indexes, self.window_size)
        val_set = DTMLDataset(val_dataset, self.val_indexes, self.window_size)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            n_train = 0
            
            # Store training predictions for manual metric calculation
            train_all_preds = []
            train_all_targets = []
            train_all_logits = []
            train_batch_losses = []
            train_batch_improvements = []
            epoch_grad_norms = []
            
            print(f"\n{'#'*80}")
            print(f"# EPOCH {epoch+1}/{num_epochs}")
            print(f"{'#'*80}")
            
            # Training loop
            batch_idx = 0
            batch_acc = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [T]"):
                if batch is None:
                    continue
                
                x, index_x, y = batch
                x = x.to(self.device).float()
                index_x = index_x.to(self.device).float()
                y = y.to(self.device)

                optimizer.zero_grad()

                x = x.squeeze(0)  # Remove batch dimension if batch_size=1
                index_x = index_x.squeeze(0)  # Remove batch dimension if batch_size=1

                # Capture attention periodically
                if epoch % 5 == 0 and batch_idx == 0:
                    out_dict = self.model(x, index_x, rt_attn=True)
                    outputs = out_dict['output'].permute(1,0)
                    self._save_attention_visualizations(epoch, out_dict)
                else:
                    outputs = self.model(x, index_x)['output'].permute(1,0)
                
                # Calculate batch accuracy
                batch_acc += ((torch.round(torch.sigmoid(outputs))).detach().int().cpu() == y.detach().int().cpu()).float().mean().item()
                
                loss = self.criterion(outputs.float(), y.float())
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
                if n_train > 0:
                    prev_batch_loss = train_batch_losses[-2]
                    improvement = prev_batch_loss - current_batch_loss
                    train_batch_improvements.append(improvement)
                    
                n_train += 1
                batch_idx += 1
                
                # Collect predictions for manual metrics
                preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                train_all_preds.extend(preds.flatten().tolist())
                train_all_targets.extend(y.int().cpu().flatten().tolist())
                train_all_logits.extend(outputs.detach().cpu().flatten().tolist())
                
                # Save attention for the first batch of every 5th epoch
                if epoch % 5 == 0 and batch_idx == 1:
                    # We need to run forward pass with rt_attn=True to get attention weights
                    # But the training loop already ran forward. 
                    # The model definition of DTML.forward has rt_attn=False by default.
                    # We need to modify the training call or do a separate pass.
                    # Let's do a separate pass for visualization to avoid changing the training loop logic too much
                    with torch.no_grad():
                        viz_outputs = self.model(x, index_x, rt_attn=True)
                        self._save_attention_visualizations(epoch, viz_outputs)
            
            avg_train_loss = train_loss / n_train if n_train > 0 else 0.0
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
            num_nodes = y.shape[-1] if len(y.shape) > 1 else len(y.flatten())
                
            if use_validation and ( epoch % 5 == 0 or epoch == num_epochs - 1):
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
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [V]"):
                        if batch is None:
                            continue
                        
                        
                        x, index_x, targets = batch
                        x = x.to(self.device).float()
                        index_x = index_x.to(self.device).float()
                        targets = targets.to(self.device)


                        x = x.squeeze(0)  # Remove batch dimension if batch_size=1
                        index_x = index_x.squeeze(0)  # Remove batch dimension if batch_size=1

                        outputs = self.model(x, index_x)['output'].permute(1,0)
                        
                        # Compute loss for batch
                        batch_loss = self.criterion(outputs.float(), targets.float()).item()
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
                        val_metrics['acc'] += accuracy_score(targets_cpu.flatten(), preds.flatten())
                        val_metrics['f1'] += f1_score(targets_cpu.flatten(), preds.flatten(), zero_division=0)
                        val_metrics['rec'] += recall_score(targets_cpu.flatten(), preds.flatten(), zero_division=0)
                        val_metrics['mcc'] += matthews_corrcoef(targets_cpu.flatten(), preds.flatten())
                        val_metrics['prec'] += precision_score(targets_cpu.flatten(), preds.flatten(), zero_division=0)
                        
                        n_val += 1

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
                    print(f"Epoch {epoch+1}/{num_epochs} Summary:{improvement_marker}")
                    print(f"{'='*70}")
                    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
                    print(f"{'='*70}\n")
                    
                    # Plot and save training curves after every 5 epochs
                    self._plot_training_curves_interim(history, epoch, n_features, self.window_size, 
                                                       optimizer, plot_dir)
                    # Check early stopping condition
                    if patience_counter >= early_stopping_patience:
                        print(f"\n{'='*60}")
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        print(f"Best {early_stopping_metric}: {best_val_metric:.4f} at epoch {best_epoch+1}")
                        print(f"{'='*60}\n")
                        
                        # Restore best model
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                            print("✓ Best model weights restored")
                        break
        
        # Plot final training curves
        self._plot_training_curves(history, num_epochs, n_features, self.window_size, 
                                   optimizer, plot_dir)
    
    def _plot_training_curves_interim(self, history, current_epoch, n_features, window_size, 
                                     optimizer, plot_dir='training_plots'):
        """Plot and save interim training curves during training"""
        # Create directory structure
        hyperparams_str = f"feat{n_features}_win{window_size}_lr{optimizer.param_groups[0]['lr']:.0e}"
        save_dir = Path(plot_dir) / self.model_name / self.market_name / hyperparams_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with loss and accuracy subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate epochs for train (all) and val (every 5th)
        train_epochs = range(1, len(history['train_loss']) + 1)
        val_epochs = [i for i in train_epochs if i % 5 == 0 or i == len(history['train_loss'])][:len(history['val_loss'])]
        
        # Plot 1: Loss curves
        ax1.plot(train_epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o')
        if len(history['val_loss']) > 0:
            val_len = min(len(val_epochs), len(history['val_loss']))
            ax1.plot(val_epochs[:val_len],
                    history['val_loss'][:val_len],
                    'r-', label='Val Loss', linewidth=2, marker='s')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title(f'Loss Curves (Epoch {current_epoch+1})', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2.plot(train_epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o')
        if len(history['val_acc']) > 0:
            val_len = min(len(val_epochs), len(history['val_acc']))
            ax2.plot(val_epochs[:val_len],
                    history['val_acc'][:val_len],
                    'r-', label='Val Accuracy', linewidth=2, marker='s')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title(f'Accuracy Curves (Epoch {current_epoch+1})', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_dir / f'training_progress_epoch_{current_epoch+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_curves(self, history, num_epochs, n_features, window_size, 
                             optimizer, plot_dir='training_plots'):
        """Plot and save training and validation curves"""
        # Create directory structure
        hyperparams_str = f"feat{n_features}_win{window_size}_lr{optimizer.param_groups[0]['lr']:.0e}"
        save_dir = Path(plot_dir) / self.model_name / self.market_name / hyperparams_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training history as JSON
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot 1: Loss curves (Train vs Val)
        fig, ax = plt.subplots(figsize=(10, 6))
        train_epochs = range(1, len(history['train_loss']) + 1)
        # Build val_epochs to match actual validation points (epoch % 5 == 0 or epoch == num_epochs - 1)
        # Convert from 0-indexed loop epochs to 1-indexed plot epochs
        val_epochs = [i for i in train_epochs if ((i-1) % 5 == 0 or (i-1) == num_epochs - 1)][:len(history['val_loss'])]
        
        ax.plot(train_epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o')
        if len(history['val_loss']) > 0:
            val_len = min(len(val_epochs), len(history['val_loss']))
            ax.plot(val_epochs[:val_len],
                    history['val_loss'][:val_len],
                    'r-', label='Val Loss', linewidth=2, marker='s')

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
        
        ax.plot(train_epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o')
        if len(history['val_acc']) > 0:
            val_len = min(len(val_epochs), len(history['val_acc']))
            ax.plot(val_epochs[:val_len],
                    history['val_acc'][:val_len],
                    'r-', label='Val Accuracy', linewidth=2, marker='s')

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
                ax.plot(train_epochs, history[train_key], 'b-', linewidth=2, marker='o', label='Train')
                ax.plot(val_epochs, history[val_key], 'r-', linewidth=2, marker='s', label='Val')
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
            f"Features: {n_features}\n"
            f"Window Size: {window_size}\n"
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.0e}\n"
            f"Total Epochs: {len(train_epochs)}"
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
    def test(self, test_dataset, batch_size=1):
        test_set = DTMLDataset(test_dataset, self.test_indexes, self.window_size)
        test_loader = DataLoader(test_set, batch_size=1)
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                if batch is None:
                    continue
                    
                x, index_x, targets = batch
                x = x.to(self.device).float()
                index_x = index_x.to(self.device).float()
                targets = targets.to(self.device)

                x = x.squeeze(0)  # Remove batch dimension if batch_size=1
                index_x = index_x.squeeze(0)  # Remove batch dimension if batch_size=1



                outputs = self.model(x, index_x)['output'].permute(1,0)
                loss = self.criterion(outputs.float(), targets.float())
                preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                
                all_preds.extend(preds.flatten().tolist())
                all_labels.extend(targets.flatten().tolist())

        return {'preds': np.array(all_preds), 'targets': np.array(all_labels)}
    
    
    def load_index(self, dates: list[str], is_train: bool = False) -> torch.tensor:
        market_to_path = {
            'NASDAQ': '~/tests/TemporalGNNReview/code/data/datasets/hist_prices/NASDAQ/NASDAQ_NDX_index.csv',
            'SSE': '~/tests/TemporalGNNReview/code/data/datasets/hist_prices/SSE/SSE_000001SS_index.csv',
            'NYSE': '~/tests/TemporalGNNReview/code/data/datasets/hist_prices/NYSE/NYSE_NYA_index.csv'
        }
        # Load from csv
        df: pd.DataFrame = pd.read_csv(Path(market_to_path[self.market]), index_col=0, parse_dates=True)
        # Get the right range
        df = df[dates[0]:dates[1]] 
        # Handle missing values
        df = df.bfill().ffill()
        
        #If is train set, load normalization values for other datasets
        if is_train:
            self.load_norm_values(df)

        # Transform to numpy 
        df_np = df.to_numpy()
        # Normalize
        df_np_normalized = self.normalize_features(df_np)
        # Transform to tensor
        return torch.from_numpy(df_np_normalized)
    
    
    def normalize_features(self, x: torch.tensor) -> torch.tensor:
        if self.feature_normalization == 'minmax':
            min_vals, max_vals = self.norm_values['minmax']
            x = (x - min_vals) / (max_vals - min_vals + 1e-8)
        elif self.feature_normalization == 'zscore':
            mean_vals, std_vals = self.norm_values['zscore']
            x = (x - mean_vals) / (std_vals + 1e-8)
        return x
    

    def load_norm_values(self, df: pd.DataFrame):
        if self.feature_normalization == 'minmax':
            self.norm_values['minmax'] = [df.min().to_numpy(), df.max().to_numpy()]
        elif self.feature_normalization == 'zscore':
            self.norm_values['zscore'] = [df.mean().to_numpy(), df.std().to_numpy()]
    
    def _save_attention_visualizations(self, epoch, outputs):
        """Save attention visualizations for DTML"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        
        save_dir = Path('training_plots') / 'DTML' / 'attention' / f'epoch_{epoch}'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # DTML returns attention weights in the output dictionary
        # 'tx_attn_stocks': [D, W] (Stocks x Window)
        # 'tx_attn_index': [1, W]
        # 'dx_attn_stocks': [D, D] (Stocks x Stocks)
        
        if 'tx_attn_stocks' in outputs and outputs['tx_attn_stocks'] is not None:
            attn = outputs['tx_attn_stocks'].detach().cpu().numpy()
            plt.figure(figsize=(12, 8))
            sns.heatmap(attn, cmap='viridis', annot=False)
            plt.title(f'Time-Axis Attention (Stocks) - Epoch {epoch}')
            plt.xlabel('Time Step')
            plt.ylabel('Stock Index')
            plt.tight_layout()
            plt.savefig(save_dir / 'time_attention_stocks.png')
            plt.close()
            
        if 'dx_attn_stocks' in outputs and outputs['dx_attn_stocks'] is not None:
            attn = outputs['dx_attn_stocks'].detach().cpu().numpy()
            plt.figure(figsize=(10, 10))
            sns.heatmap(attn, cmap='viridis', annot=False)
            plt.title(f'Data-Axis Attention (Stock-to-Stock) - Epoch {epoch}')
            plt.xlabel('Stock Index')
            plt.ylabel('Stock Index')
            plt.tight_layout()
            plt.savefig(save_dir / 'data_attention_stocks.png')
            plt.close()

