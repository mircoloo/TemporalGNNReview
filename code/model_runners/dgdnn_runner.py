from pathlib import Path
from model_runners.base_runner import *
import torch
from torch_geometric.utils import to_dense_adj
import pandas as pd
from tqdm.auto import tqdm
from tabulate import tabulate
from model_runners.base_runner import evaluate_decorator, BaseModelRunner
from model_runners.models_dataset import DGDNNDataset
import matplotlib.pyplot as plt
import json
from torch_geometric.loader import DataLoader

class DGDNNRunner(BaseModelRunner):

    def __init__(self, model, device, market_name):
        super().__init__(model, device, market_name)
        self.model_name = 'DGDNN'

    def train(self, train_dataset, 
              val_dataset, 
              optimizer, 
              criterion, 
              num_epochs, 
              alpha, 
              window_size, num_nodes, 
              batch_size=32, 
              use_validation=True,
              early_stopping_patience=20,
              early_stopping_metric='mcc',
              early_stopping_min_delta=1e-4,
              plot_dir='training_plots'):
        """
        Train the DGDNN model with optional early stopping.
        
        Args:
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            early_stopping_metric: Metric to monitor ('loss', 'acc', 'f1', 'mcc')
            early_stopping_min_delta: Minimum change to qualify as an improvement
            plot_dir: Directory to save training plots
        """
        self.optimizer, \
        self.criterion, \
        self.num_epochs, \
        self.alpha, \
        self.window_size, \
        self.num_nodes, \
        self.batch_size = optimizer, criterion, num_epochs, alpha, window_size, num_nodes, batch_size
        print(f"DGDNN parameters: {self.num_nodes=} {self.window_size=} {self.alpha=}")
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
        
        # Create organized TensorBoard writer
        train_set = DGDNNDataset(train_dataset)
        val_set = DGDNNDataset(val_dataset)
        # Use actual batching
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        
        # Track loss improvement across epochs
        prev_epoch_loss = float('inf')
        loss_improvement_history = []
        
        # Update training loop to handle None batches
        for epoch in range(num_epochs + 1):
            self.model.train()
            
            # Enable visualization saving periodically (e.g., every 5 epochs)
            if epoch % 5 == 0:
                save_dir_attn = Path(plot_dir) / 'DGDNN' / 'attention' / f'epoch_{epoch}'
                save_dir_diff = Path(plot_dir) / 'DGDNN' / 'diffusion' / f'epoch_{epoch}'
                self.model.enable_attention_saving(save_dir=str(save_dir_attn), max_saves=1)
                self.model.enable_diffusion_saving(save_dir=str(save_dir_diff), max_saves=1)
            else:
                self.model.disable_attention_saving()
                self.model.disable_diffusion_saving()
                
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
            for batch in train_loader:
                if batch is None:
                    continue
                batch = batch.to(self.device)
                number_of_features = batch.x.size(-1)

                X = batch.x.view(-1, num_nodes, number_of_features)  # [B, N, F]
                A = to_dense_adj(
                    batch.edge_index, 
                    batch=batch.batch,
                    edge_attr=batch.edge_attr,
                    max_num_nodes=num_nodes
                )
                optimizer.zero_grad()
                # Forward pass with batched inputs
                outputs = self.model(X, A)  # [B, N, 1]
                targets = batch.y.view(-1, num_nodes, 1).float()  # [B, N, 1]
                
                # Compute loss
                loss = criterion(outputs, targets) + theta_regularizer(self.model.theta) - alpha * neighbor_distance_regularizer(self.model.theta)
                
                loss.backward()
                
                # Gradient monitoring
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                epoch_grad_norms.append(total_norm)

                if n_train % 10 == 0:
                    print(f"Batch {n_train} Grad Norm: {total_norm:.4f}")

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
                
                # Collect predictions for manual metrics
                preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                train_all_preds.extend(preds.flatten().tolist())
                train_all_targets.extend(targets.int().cpu().flatten().tolist())
                train_all_logits.extend(outputs.detach().cpu().flatten().tolist())


            # Gradient Analysis
            if epoch_grad_norms:
                grad_mean = np.mean(epoch_grad_norms)
                grad_std = np.std(epoch_grad_norms)
                grad_max = np.max(epoch_grad_norms)
                grad_min = np.min(epoch_grad_norms)
                print(f"Gradient Stats - Mean: {grad_mean:.4f}, Std: {grad_std:.4f}, Min: {grad_min:.4f}, Max: {grad_max:.4f}")
                
                if grad_max > 100:
                    print("WARNING: Exploding gradients detected!")
                if grad_mean < 1e-4:
                    print("WARNING: Vanishing gradients detected!")

            # Training metrics
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
            
            train_metrics = {
                'acc': accuracy_score(train_all_targets, train_all_preds),
                'prec': precision_score(train_all_targets, train_all_preds, zero_division=0),
                'rec': recall_score(train_all_targets, train_all_preds, zero_division=0),
                'f1': f1_score(train_all_targets, train_all_preds, zero_division=0),
                'mcc': matthews_corrcoef(train_all_targets, train_all_preds)
            }
            
            # Calculate directional accuracy for training
            up_mask_train = train_all_targets == 1
            down_mask_train = train_all_targets == 0
            train_metrics['dir_acc_up'] = accuracy_score(
                train_all_targets[up_mask_train], 
                train_all_preds[up_mask_train]
            ) if up_mask_train.sum() > 0 else 0.0
            train_metrics['dir_acc_down'] = accuracy_score(
                train_all_targets[down_mask_train], 
                train_all_preds[down_mask_train]
            ) if down_mask_train.sum() > 0 else 0.0
            
            # Store training metrics in history
            history['train_acc'].append(train_metrics['acc'])
            history['train_prec'].append(train_metrics['prec'])
            history['train_rec'].append(train_metrics['rec'])
            history['train_f1'].append(train_metrics['f1'])
            history['train_mcc'].append(train_metrics['mcc'])
                
            # Validation loop
            if use_validation and (epoch % 1 == 0):
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
                        batch = batch.to(self.device)
                        X = batch.x.view(-1, num_nodes, 5 * window_size)
                        A = to_dense_adj(batch.edge_index, 
                                       batch=batch.batch,
                                       edge_attr=batch.edge_attr,
                                       max_num_nodes=num_nodes)
                        
                        outputs = self.model(X, A)
                        targets = batch.y.view(-1, num_nodes, 1).float()
                        
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
                    
                    # Calculate directional accuracy for validation
                    up_mask_val = val_all_targets == 1
                    down_mask_val = val_all_targets == 0
                    manual_val_metrics['dir_acc_up'] = accuracy_score(
                        val_all_targets[up_mask_val], 
                        val_all_preds[up_mask_val]
                    ) if up_mask_val.sum() > 0 else 0.0
                    manual_val_metrics['dir_acc_down'] = accuracy_score(
                        val_all_targets[down_mask_val], 
                        val_all_preds[down_mask_val]
                    ) if down_mask_val.sum() > 0 else 0.0
                    
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
                    if epoch % 5 == 0 or epoch == num_epochs:
                        headers = ["Metric", "Train", "Val"]
                        table_data = [
                            ["Loss", f"{avg_train_loss:.4f}", f"{val_metrics['loss']:.4f}"],
                            ["Accuracy", f"{train_metrics['acc']:.4f}", f"{manual_val_metrics['acc']:.4f}"],
                            ["Dir. Acc (Up)", f"{train_metrics['dir_acc_up']:.4f}", f"{manual_val_metrics['dir_acc_up']:.4f}"],
                            ["Dir. Acc (Down)", f"{train_metrics['dir_acc_down']:.4f}", f"{manual_val_metrics['dir_acc_down']:.4f}"],
                            ["Precision", f"{train_metrics['prec']:.4f}", f"{manual_val_metrics['prec']:.4f}"],
                            ["Recall", f"{train_metrics['rec']:.4f}", f"{manual_val_metrics['rec']:.4f}"],
                            ["F1 Score", f"{train_metrics['f1']:.4f}", f"{manual_val_metrics['f1']:.4f}"],
                            ["MCC", f"{train_metrics['mcc']:.4f}", f"{manual_val_metrics['mcc']:.4f}"],
                            ["---", "---", "---"],
                            ["Best Epoch", "", f"{best_epoch}"],
                            ["Patience", "", f"{patience_counter}/{early_stopping_patience}"]
                        ]
                        
                        print(f"\n{'='*70}")
                        print(f"Epoch {epoch+1}/{num_epochs} Summary{improvement_marker}")
                        print(f"{'='*70}")
                        print(tabulate(table_data, headers=headers, tablefmt="simple"))
                        print(f"{'='*70}\n")
                        
                        # Plot and save training curves after every 5 epochs
                        self._plot_training_curves_interim(history, epoch, alpha, window_size, 
                                                           num_nodes, batch_size, optimizer, plot_dir)
                    
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

                self.model.train()
        
        # Plot final training curves
        self._plot_training_curves(history, num_epochs, alpha, window_size, 
                                   num_nodes, batch_size, optimizer, plot_dir)
    
    def _plot_training_curves_interim(self, history, current_epoch, alpha, window_size, 
                                     num_nodes, batch_size, optimizer, plot_dir='training_plots'):
        """Plot and save interim training curves during training"""
        # Create directory structure
        hyperparams_str = f"alpha{alpha}_win{window_size}_nodes{num_nodes}_bs{batch_size}_lr{optimizer.param_groups[0]['lr']:.0e}"
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
        ax1.set_title(f'Loss Curves (Epoch {current_epoch+1})', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2.plot(epochs_range, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o')
        ax2.plot(epochs_range, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2, marker='s')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title(f'Accuracy Curves (Epoch {current_epoch+1})', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_dir / f'training_progress_epoch_{current_epoch+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_curves(self, history, num_epochs, alpha, window_size, 
                             num_nodes, batch_size, optimizer, plot_dir='training_plots'):
        """Plot and save training and validation curves"""
        # Create directory structure
        hyperparams_str = f"alpha{alpha}_win{window_size}_nodes{num_nodes}_bs{batch_size}_lr{optimizer.param_groups[0]['lr']:.0e}"
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
            f"Alpha: {alpha}\n"
            f"Window Size: {window_size}\n"
            f"Num Nodes: {num_nodes}\n"
            f"Batch Size: {batch_size}\n"
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
    def test(self, test_dataset, window_size, num_nodes, batch_size=1, 
             save_attention=False, attention_save_dir='dgdnn_attention_outputs', max_saves=50):
        """
        Test the DGDNN model
        
        Args:
            save_attention: Whether to save attention visualizations during forward passes
            attention_save_dir: Directory to save attention outputs
            max_saves: Maximum number of forward passes to save attention for
        """
        # Enable attention saving if requested
        if save_attention:
            self.model.enable_attention_saving(
                save_dir=attention_save_dir,
                max_saves=max_saves
            )
            
        self.model.enable_diffusion_saving('dgdnn_diffusion_outputs', max_saves=30)
        
        test_dataset = DGDNNDataset(test_dataset)  
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                if batch.x.shape[-1] != 5 * window_size:
                    continue
                    
                batch = batch.to(self.device)
                X = batch.x.view(-1, num_nodes, 5 * window_size)
                A = to_dense_adj(batch.edge_index, 
                               batch=batch.batch,
                               edge_attr=batch.edge_attr,
                               max_num_nodes=num_nodes)
                
                outputs = self.model(X, A)
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
                all_preds.extend(preds.flatten().tolist())
                all_labels.extend(batch.y.cpu().flatten().tolist())
        
        # Disable attention saving after test
        if save_attention:
            self.model.disable_attention_saving()
            print(f"\n✓ Saved attention for {self.model.forward_pass_counter} forward passes to {attention_save_dir}")

        return {'preds': np.array(all_preds), 'targets': np.array(all_labels)}
    
    def visualize_attention(self, test_dataset, window_size, num_nodes, 
                           stock_names=None, num_samples=5,
                           output_dir='dgdnn_attention_vis'):
        """
        Visualize attention mechanisms for DGDNN model
        
        Args:
            test_dataset: Test dataset
            window_size: Window size used in training
            num_nodes: Number of nodes/stocks
            stock_names: List of stock ticker names (optional)
            num_samples: Number of samples to visualize
            output_dir: Directory to save visualizations
        """
        from models.DGDNN.attention_visualizer import DGDNNAttentionVisualizer
        from pathlib import Path
        
        print(f"\n{'='*80}")
        print(f"DGDNN Attention Visualization")
        print(f"{'='*80}\n")
        
        # Create visualizer
        visualizer = DGDNNAttentionVisualizer(self.model, device=self.device)
        
        # Create output directory
        output_path = Path(output_dir) / self.market_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        test_set = DGDNNDataset(test_dataset)
        
        # Visualize multiple samples
        for sample_idx in range(min(num_samples, len(test_set))):
            print(f"\nProcessing sample {sample_idx + 1}/{num_samples}...")
            
            batch = test_set[sample_idx]
            
            if batch.x.shape[-1] != 5 * window_size:
                continue
            
            # Prepare inputs
            X = batch.x.view(1, num_nodes, 5 * window_size)  # Add batch dim
            A = to_dense_adj(
                batch.edge_index,
                edge_attr=batch.edge_attr,
                max_num_nodes=num_nodes
            ).unsqueeze(0)  # Add batch dim
            
            # Move to device
            X = X.to(self.device)
            A = A.to(self.device)
            
            # Create sample-specific output directory
            sample_dir = output_path / f'sample_{sample_idx}'
            sample_dir.mkdir(exist_ok=True)
            
            # Extract attention weights
            visualizer.extract_attention_weights(X, A, stock_names=stock_names)
            
            # Generate visualizations
            print(f"  - Generating theta weights plot...")
            visualizer.plot_theta_weights(
                save_path=sample_dir / 'theta_weights.png'
            )
            
            num_layers = visualizer.attention_weights['num_layers']
            num_heads = visualizer.attention_weights['num_heads']
            
            for layer_idx in range(num_layers):
                print(f"  - Layer {layer_idx}...")
                
                # All heads
                visualizer.plot_all_heads(
                    layer_idx=layer_idx,
                    save_path=sample_dir / f'layer_{layer_idx}_all_heads.png'
                )
                
                # Statistics
                visualizer.plot_attention_statistics(
                    layer_idx=layer_idx,
                    save_path=sample_dir / f'layer_{layer_idx}_statistics.png'
                )
                
                # First head heatmap
                visualizer.plot_attention_heatmap(
                    layer_idx=layer_idx, 
                    head_idx=0,
                    save_path=sample_dir / f'layer_{layer_idx}_head_0_heatmap.png'
                )
        
        print(f"\n✓ Attention visualizations saved to {output_path}")
        print(f"{'='*80}\n")

# Define optimizer and objective function
def theta_regularizer(theta):
    row_sums = torch.sum(theta, dim=-1)
    ones = torch.ones_like(row_sums)
    return torch.sum(torch.abs(row_sums - ones))

def neighbor_distance_regularizer(theta):
    box = torch.sum(theta, dim=-1)
    result = torch.zeros_like(theta)

    for idx, row in enumerate(theta):
        for i, j in enumerate(row):
            result[idx, i] = i * j

    result_sum = torch.sum(result, dim=1)
    return torch.sum(result / result_sum[:, None])

