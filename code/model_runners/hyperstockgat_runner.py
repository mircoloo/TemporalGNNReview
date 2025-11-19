
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
              early_stopping_patience=20,
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
                #print(f"hyperstockgat input x.shape: {x.shape}, adj.shape: {adj.shape}, y.shape: {y.shape}")
                optimizer.zero_grad()
                emb = self.model.encode(x, adj)
                outputs = self.model.decode(emb, adj)
                targets = y.squeeze(0)
                
                # Calculate batch accuracy
                batch_acc += ((torch.round(torch.sigmoid(outputs))).detach().int().cpu() == targets.detach().int().cpu()).float().mean().item()
                
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Check gradient norms before optimizer step
                total_grad_norm = 0.0
                num_params_with_grad = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                        num_params_with_grad += 1
                total_grad_norm = total_grad_norm ** 0.5
                
                optimizer.step()
                
                current_batch_loss = loss.item()
                train_loss += current_batch_loss
                train_batch_losses.append(current_batch_loss)
                
                # Track batch improvement
                if total_samples > 0:
                    prev_batch_loss = train_batch_losses[-2]
                    improvement = prev_batch_loss - current_batch_loss
                    train_batch_improvements.append(improvement)
                    
                    # Report non-improving batches
                    if improvement <= 0:
                        if batch_idx % 10 == 0 or epoch < 3:  # More verbose in early epochs
                            print(f"  ⚠️ Batch {batch_idx}: Loss NOT improving "
                                  f"(prev: {prev_batch_loss:.6f}, curr: {current_batch_loss:.6f}, "
                                  f"diff: {improvement:.6f}, grad_norm: {total_grad_norm:.6f})")
                total_samples += 1
                batch_idx += 1
                
                # Verbose logging for first few batches of early epochs
                if epoch < 2 and batch_idx <= 5:
                    print(f"  📊 Batch {batch_idx}: loss={current_batch_loss:.6f}, "
                          f"grad_norm={total_grad_norm:.6f}, "
                          f"output_range=[{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                
                # Debug first batch of first epoch
                if epoch == 1 and total_samples == 1:
                    print(f"\n🔬 FIRST BATCH DEBUG:")
                    print(f"  Input X shape: {x.shape}")
                    print(f"  Adj shape: {adj.shape}")
                    print(f"  Targets shape: {targets.shape}")
                    print(f"  Outputs shape: {outputs.shape}")
                    print(f"  Sample outputs[:5]: {outputs.flatten()[:5].detach().cpu().numpy()}")
                    print(f"  Sample targets[:5]: {targets.flatten()[:5].detach().cpu().numpy()}")
                    
                    # Check if all outputs are the same
                    outputs_flat = outputs.flatten().detach().cpu().numpy()
                    print(f"  Output variance: {outputs_flat.var():.8f}")
                    if outputs_flat.var() < 1e-6:
                        print(f"  ⚠️ CRITICAL: All outputs are nearly identical!")
                    
                    # Check gradient flow
                    print(f"  Gradient norm: {total_grad_norm:.6f}")
                    print(f"  Params with gradients: {num_params_with_grad}")
                
                # Collect predictions for manual metrics
                preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                train_all_preds.extend(preds.flatten().tolist())
                train_all_targets.extend(targets.int().cpu().flatten().tolist())
                train_all_logits.extend(outputs.detach().cpu().flatten().tolist())
            
            print(f"MANUAL BATCH ACC = {(batch_acc/total_samples):.4f}")
            
            # Batch improvement summary
            if len(train_batch_improvements) > 0:
                improving_batches = sum(1 for imp in train_batch_improvements if imp > 0)
                worsening_batches = sum(1 for imp in train_batch_improvements if imp < 0)
                stable_batches = len(train_batch_improvements) - improving_batches - worsening_batches
                print(f"\n  📈 Batch Improvement Summary:")
                print(f"    Improving batches: {improving_batches}/{len(train_batch_improvements)} "
                      f"({100*improving_batches/len(train_batch_improvements):.1f}%)")
                print(f"    Worsening batches: {worsening_batches}/{len(train_batch_improvements)} "
                      f"({100*worsening_batches/len(train_batch_improvements):.1f}%)")
                print(f"    Stable batches: {stable_batches}/{len(train_batch_improvements)} "
                      f"({100*stable_batches/len(train_batch_improvements):.1f}%)")
            
            avg_train_loss = train_loss / float(max(total_samples, 1))
            history['train_loss'].append(avg_train_loss)
            
            # Track epoch-level improvement
            epoch_improvement = prev_epoch_loss - avg_train_loss
            loss_improvement_history.append(epoch_improvement)
            
            if epoch_improvement <= 0:
                print(f"\n  ⚠️⚠️⚠️ EPOCH LOSS NOT IMPROVING ⚠️⚠️⚠️")
                print(f"  Previous epoch loss: {prev_epoch_loss:.6f}")
                print(f"  Current epoch loss:  {avg_train_loss:.6f}")
                print(f"  Difference:          {epoch_improvement:.6f}")
            else:
                print(f"\n  ✅ Epoch improvement: {epoch_improvement:.6f} "
                      f"(prev: {prev_epoch_loss:.6f} → curr: {avg_train_loss:.6f})")
            
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
            
            # Store training metrics in history
            history['train_acc'].append(train_metrics['acc'])
            history['train_prec'].append(train_metrics['prec'])
            history['train_rec'].append(train_metrics['rec'])
            history['train_f1'].append(train_metrics['f1'])
            history['train_mcc'].append(train_metrics['mcc'])
            
            # Get num_nodes from data
            num_nodes = targets.shape[-1] if len(targets.shape) > 1 else len(targets)
            
            # Print training debug info
            print(f"\n{'='*80}")
            print(f"TRAINING DEBUG - Epoch {epoch}/{epochs}")
            print(f"{'='*80}")
            print(f"📉 LOSS STATISTICS:")
            print(f"  Train Loss (avg): {avg_train_loss:.6f}")
            print(f"  Train Loss (min batch): {min(train_batch_losses):.6f}")
            print(f"  Train Loss (max batch): {max(train_batch_losses):.6f}")
            print(f"  Train Loss (std): {np.std(train_batch_losses):.6f}")
            
            # Loss distribution analysis
            loss_quartiles = np.percentile(train_batch_losses, [25, 50, 75])
            print(f"  Loss quartiles [Q1, Q2, Q3]: [{loss_quartiles[0]:.6f}, {loss_quartiles[1]:.6f}, {loss_quartiles[2]:.6f}]")
            
            # Identify problematic batches
            high_loss_threshold = avg_train_loss + 2 * np.std(train_batch_losses)
            high_loss_batches = [i for i, loss in enumerate(train_batch_losses) if loss > high_loss_threshold]
            if len(high_loss_batches) > 0:
                print(f"  ⚠️ High loss batches (>{high_loss_threshold:.6f}): {len(high_loss_batches)} batches")
                print(f"    Batch indices: {high_loss_batches[:10]}{'...' if len(high_loss_batches) > 10 else ''}")
            
            print(f"\n🔍 LOGITS ANALYSIS:")
            print(f"  Logits - Min: {train_all_logits.min():.4f}, Max: {train_all_logits.max():.4f}, "
                  f"Mean: {train_all_logits.mean():.4f}, Std: {train_all_logits.std():.4f}")
            
            # Check if logits are all the same (CRITICAL ISSUE)
            unique_logits = np.unique(np.round(train_all_logits, 4))
            print(f"  Unique logit values (rounded to 4 decimals): {len(unique_logits)}")
            if len(unique_logits) < 10:
                print(f"  ⚠️ WARNING: Very few unique logit values! {unique_logits[:10]}")
            
            # Check variance across different nodes
            train_all_logits_reshaped = np.array(train_all_logits).reshape(-1, num_nodes)
            per_node_variance = train_all_logits_reshaped.var(axis=0)
            per_node_mean = train_all_logits_reshaped.mean(axis=0)
            print(f"  Per-node variance - Mean: {per_node_variance.mean():.6f}, "
                  f"Min: {per_node_variance.min():.6f}, Max: {per_node_variance.max():.6f}")
            print(f"  Per-node mean - Mean: {per_node_mean.mean():.6f}, "
                  f"Min: {per_node_mean.min():.6f}, Max: {per_node_mean.max():.6f}")
            print(f"  Nodes with zero variance: {np.sum(per_node_variance < 1e-6)}/{num_nodes}")
            
            # Check if nodes have different means
            nodes_with_same_mean = np.sum(np.abs(per_node_mean - per_node_mean.mean()) < 1e-4)
            if nodes_with_same_mean > num_nodes * 0.9:
                print(f"  ⚠️ CRITICAL: {nodes_with_same_mean}/{num_nodes} nodes have nearly identical means!")
            
            print(f"\n📊 PREDICTIONS ANALYSIS:")
            print(f"  Sample logits (first 20): {train_all_logits[:20]}")
            print(f"  Sample targets (first 20): {train_all_targets[:20]}")
            print(f"  Sample preds (first 20): {train_all_preds[:20]}")
            
            # Check if all predictions are the same
            unique_preds = np.unique(train_all_preds)
            print(f"  Unique predictions: {unique_preds}")
            if len(unique_preds) == 1:
                print(f"  ⚠️ CRITICAL: Model always predicts class {unique_preds[0]}!")
            
            print(f"\n📈 METRICS:")
            print(f"  Manual Train Accuracy: {train_metrics['acc']:.4f}")
            print(f"  Class distribution - Targets: {np.bincount(train_all_targets.astype(int))}")
            print(f"  Class distribution - Preds: {np.bincount(train_all_preds.astype(int))}")
            
            # Check gradient flow
            print(f"\n🔧 MODEL PARAMETERS:")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"  Total parameters: {total_params}, Trainable: {trainable_params}")
            
            # Check if parameters are updating
            has_grad = sum(1 for p in self.model.parameters() if p.grad is not None)
            total_model_params = sum(1 for _ in self.model.parameters())
            print(f"  Parameters with gradients: {has_grad}/{total_model_params}")
            

            if epoch % 1 == 0:
                
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
                    
                    # Print validation debug info
                    print(f"\n{'='*80}")
                    print(f"VALIDATION DEBUG - Epoch {epoch}/{epochs}")
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
                        self._plot_training_curves_interim(history, epoch, seq_length, num_features, 
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
                preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                all_preds.extend(preds.flatten().tolist())
                all_labels.extend(y.squeeze(0).cpu().flatten().tolist())

        return {'preds': np.array(all_preds), 'targets': np.array(all_labels)}
