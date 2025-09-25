#from model_runners.runner_utils import BaseGraphDataset
import numpy as np
from tabulate import tabulate
from model_runners.models_dataset import GraphWaveNetDataset
from model_runners.base_runner import BaseModelRunner, evaluate_decorator
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score 

        


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
              threshold=0.5):
        # Create organized TensorBoard writer
        train_set = GraphWaveNetDataset(train_dataset)
        val_set = GraphWaveNetDataset(val_dataset)
        # Use actual batching
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1)

        best_val_loss = float('inf')
        for epoch in range(num_epochs):
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
                        
                        outputs = self.model(x)
                        output_for_loss = output[:, :, :, -1]  # Take last timestep
                        predict = output_for_loss.squeeze(1)  # Remove feature dimension
                        targets = y.float().squeeze()  # [B, num_nodes]
                        outputs = output.squeeze()
                        y = y.squeeze()
                        #print(f"Validation batch x.shape: {x.shape}, y.shape: {y.shape}, outputs.shape: {outputs.shape}, predict.shape: {predict.shape}, targets.shape: {targets.shape}")
                        # Compute metrics for batch
                        print(f"{outputs.shape=}, {targets.shape=}")
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
                    if epoch % 5 == 0 or epoch == num_epochs:
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
    def test(self, test_dataset, seq_length, num_features, batch_size=32, config=None):
        test_set = GraphWaveNetDataset(test_dataset)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                output_for_loss = output[:, :, :, -1]
                predict = output_for_loss.squeeze(1)
                
                preds = (torch.sigmoid(predict) > self.threshold).int()
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
            
            self.log_experiment(
                config=config,
                dataset_info=dataset_info,
                train_params=train_params,
                test_metrics=test_metrics
            )

        return {'preds': np.array(y_pred), 'targets': np.array(y_true)}