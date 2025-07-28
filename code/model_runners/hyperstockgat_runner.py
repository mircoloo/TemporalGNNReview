import sys
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_runners.runner_utils import BaseGraphDataset
from model_runners.base_runner import BaseModelRunner
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch_geometric.utils import to_dense_adj
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class HyperStockGATDataset(BaseGraphDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, idx):
        data_sample = self.dataset[idx]
        x_real = data_sample.x
        res = super().is_input_correct_shaped(x_real)
        if not res:
            x_real = super().adjust_input_shape(x_real)
        
        # Transform data for HyperStockGAT format
        # Keep the shape as [n_nodes, n_features * seq_length]
        # The encoder's Temporal_Attention_layer will handle reshaping internally
        x = x_real  # Don't reshape here, let the model handle it
        adj = to_dense_adj(data_sample.edge_index, edge_attr=data_sample.edge_attr)[0]
        y = data_sample.y.float()  # Keep as float for BCE loss
        
        print(f"Input shape: {x.shape}")
        
        return x, adj, y

class HyperStockGATRunner(BaseModelRunner):
    def __init__(self, model, device, market_name):
        super().__init__(model, device, market_name)
        self.model_name = "HyperStockGAT"
        
    def train(self, train_dataset, val_dataset, optimizer, criterion, num_epochs, window_size, num_features, batch_size=32):
        writer = SummaryWriter(f'runs/{self.market_name}/{self.model_name}')
        train_set = HyperStockGATDataset(train_dataset)
        val_set = HyperStockGATDataset(val_dataset)
        
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1)

        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            n_train = 0

            for batch in train_loader:
                x, adj, y = [b.to(self.device) for b in batch]
                optimizer.zero_grad()

                print(f"Epoch {epoch+1}/{num_epochs}, Batch size: {x.shape[0]} adj shape: {adj.shape} y shape: {y.shape}")

                # Forward pass
                embeddings = self.model.encode(x, adj)
                output = self.model.decode(embeddings, adj)
                loss = criterion(output.squeeze(), y)
                
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_train += 1

            avg_train_loss = train_loss / n_train
            writer.add_scalar('Training/Loss', avg_train_loss, epoch)

            # Validation every 5 epochs
            if epoch % 5 == 0:
                self.model.eval()
                val_metrics = {
                    'loss': 0.0,
                    'acc': 0.0,
                    'prec': 0.0,
                    'rec': 0.0,
                    'f1': 0.0,
                    'mcc': 0.0
                }
                n_val = 0
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for batch in val_loader:
                        x_val, adj_val, y_val = [b.to(self.device) for b in batch]
                        
                        embeddings = self.model.encode(x_val, adj_val)
                        output = self.model.decode(embeddings, adj_val)
                        val_loss = criterion(output.squeeze(), y_val)
                        val_metrics['loss'] += val_loss.item()

                        preds = (torch.sigmoid(output) > 0.5).int()
                        val_preds.append(preds.cpu())
                        val_targets.append(y_val.cpu())
                        n_val += 1

                if n_val > 0:
                    y_pred = torch.cat(val_preds, dim=0)
                    y_true = torch.cat(val_targets, dim=0)

                    val_metrics['acc'] = accuracy_score(y_true.flatten(), y_pred.flatten())
                    val_metrics['prec'] = precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
                    val_metrics['rec'] = recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
                    val_metrics['f1'] = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
                    val_metrics['mcc'] = matthews_corrcoef(y_true.flatten(), y_pred.flatten())
                    val_metrics['loss'] /= n_val

                    # Log all metrics
                    for metric, value in val_metrics.items():
                        writer.add_scalar(f'Validation/{metric}', value, epoch)

                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} - "
                          f"Val Loss: {val_metrics['loss']:.4f} - "
                          f"Acc: {val_metrics['acc']:.4f} - "
                          f"Prec: {val_metrics['prec']:.4f} - "
                          f"Rec: {val_metrics['rec']:.4f} - "
                          f"F1: {val_metrics['f1']:.4f} - "
                          f"MCC: {val_metrics['mcc']:.4f}")

        writer.close()

    def test(self, test_dataset, window_size, num_features, batch_size=32, config=None):
        test_set = HyperStockGATDataset(test_dataset)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, adj, y = [b.to(self.device) for b in batch]
                
                embeddings = self.model.encode(x, adj)
                output = self.model.decode(embeddings, adj)
                preds = (torch.sigmoid(output) > 0.5).int()
                
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

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
                'seq_length': window_size,
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

        return y_pred, y_true
