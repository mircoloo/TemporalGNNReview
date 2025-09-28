from tabulate import tabulate
from .base_runner import BaseModelRunner, evaluate_decorator
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from model_runners.runner_utils import BaseGraphDataset
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model_runners.models_dataset import DARNNDataset


class DARNNRunner(BaseModelRunner):
    def __init__(self, model, device, market_name):
        super().__init__(model, device, market_name)
        self.model_name = "DARNN"


    def train(self, train_dataset, val_dataset, optimizer, criterion, num_epochs, seq_length, batch_size=32):
        writer = SummaryWriter('runs/')
        train_set = DARNNDataset(train_dataset)
        val_set = DARNNDataset(val_dataset)
        
        # Use actual batching
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        for epoch in range(1,num_epochs+1):
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()
                X, y_target, target = batch  # Assuming your dataset returns (x, y)
                # Move batch to device
                X = X.to(self.device)
                y_target = y_target.to(self.device)
                target = target.to(self.device)
                outputs = self.model(X, y_target)
                loss = criterion(outputs, target.float())
                
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / n_batches
            # Validation every 5 epochs
            if (epoch % 5 == 0):
                self.model.eval()
                val_metrics = {
                    'loss': 0.0, 'acc': 0.0, 'prec': 0.0, 
                    'f1': 0.0, 'mcc': 0.0, 'rec': 0.0
                }
                n_val = 0
                
                with torch.no_grad():
                    for batch in val_loader:                            
                        X, y_target, target = batch  # Assuming your dataset returns (x, y)
                        # Move batch to device
                        X = X.to(self.device)
                        y_target = y_target.to(self.device)
                        targets = target.to(self.device)
                        outputs = self.model(X, y_target)
                        
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
                    
                    # Print results in a table format
                    if epoch % 1 == 0 or epoch == num_epochs:
                        headers = ["Metric", "Value"]
                        table_data = [
                            ["Train Loss", f"{avg_train_loss:.4f}"],
                            ["Val Loss", f"{val_metrics['loss']:.4f}"],
                            ["Accuracy", f"{val_metrics['acc']:.4f}"],
                            ["Precision", f"{val_metrics['prec']:.4f}"],
                            ["Recall", f"{val_metrics['rec']:.4f}"],
                            ["F1 Score", f"{val_metrics['f1']:.4f}"],
                            ["MCC", f"{val_metrics['mcc']:.4f}"]
                        ]
                        
                        print(f"\nEpoch {epoch+1}/{num_epochs} Results:")
                        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
                        print("\n")

                self.model.train()

    
    @evaluate_decorator
    def test(self, test_dataset):
        test_dataset = DARNNDataset(test_dataset)  
        test_loader = DataLoader(test_dataset, batch_size=1)
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                X, y_target, target = batch  # Assuming your dataset returns (x, y)
                
                # Move batch to device
                X = X.to(self.device)
                y_target = y_target.to(self.device)
                target = target.to(self.device)
                outputs = self.model(X, y_target)
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
                all_preds.extend(preds.flatten().tolist())
                all_labels.extend(target.cpu().flatten().tolist())

        return {'preds': np.array(all_preds), 'targets': np.array(all_labels)}
