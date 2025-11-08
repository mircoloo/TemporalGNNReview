from pathlib import Path
from model_runners.base_runner import *
import torch
from torch_geometric.utils import to_dense_adj
import pandas as pd
from tqdm.auto import tqdm
from tabulate import tabulate
from model_runners.base_runner import evaluate_decorator, BaseModelRunner
from model_runners.models_dataset import DGDNNDataset

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
              early_stopping_patience=50,
              early_stopping_metric='loss',
              early_stopping_min_delta=1e-4):
        """
        Train the DGDNN model with optional early stopping.
        
        Args:
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            early_stopping_metric: Metric to monitor ('loss', 'acc', 'f1', 'mcc')
            early_stopping_min_delta: Minimum change to qualify as an improvement
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
        
        
        # Create organized TensorBoard writer
        train_set = DGDNNDataset(train_dataset)
        val_set = DGDNNDataset(val_dataset)
        # Use actual batching
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        
        # Update training loop to handle None batches
        for epoch in range(num_epochs + 1):
            train_loss = 0.0
            n_train = 0
            
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
                train_loss = criterion(outputs, targets) + theta_regularizer(self.model.theta) - alpha * neighbor_distance_regularizer(self.model.theta)
                
                train_loss.backward()
                optimizer.step()
                
                train_loss += train_loss.item()
                n_train += 1

            # Training metrics
            avg_train_loss = train_loss / n_train
                
            # Validation loop
            if use_validation and (epoch % 1 == 0):
                self.model.eval()
                val_metrics = {
                    'loss': 0.0, 'acc': 0.0, 'prec': 0.0, 
                    'f1': 0.0, 'mcc': 0.0, 'rec': 0.0
                }
                n_val = 0
                
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
                        
                        print(f"\nEpoch {epoch+1}/{num_epochs} Results:{improvement_marker}")
                        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
                        print("\n")
                    
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
                
    
    
    @evaluate_decorator
    def test(self, test_dataset, window_size, num_nodes, batch_size=1):
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

        return {'preds': np.array(all_preds), 'targets': np.array(all_labels)}

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

