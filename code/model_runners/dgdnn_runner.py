from pathlib import Path
from model_runners.base_runner import *
import torch
from torch_geometric.utils import to_dense_adj
import pandas as pd



def custom_collate_fn(data_list):
            # Check if all tensors have the same dimensions
            if len(data_list) == 0:
                return Batch()
            # Handle variable feature dimensions by padding
            max_node_features = max([data.x.size(1) if data.x is not None else 0 for data in data_list])
            
            for data in data_list:
                if data.x is not None and data.x.size(1) < max_node_features:
                    # Pad with zeros
                    print("Found one tensor with less features, padding it from ", data.x.size(1), "to", max_node_features  )
                    padding = data.x[:, - (max_node_features - data.x.size(1)):]
                    data.x = torch.cat([data.x, padding], dim=1)

            return Batch.from_data_list(data_list)


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
              use_validation=True):

        self.optimizer, \
        self.criterion, \
        self.num_epochs, \
        self.alpha, \
        self.window_size, \
        self.num_nodes, \
        self.batch_size = optimizer, criterion, num_epochs, alpha, window_size, num_nodes, batch_size
        
        
        train_loader = TorchDataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            collate_fn=custom_collate_fn
        )

        val_loader = TorchDataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        
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
                train_loss = criterion(outputs, targets)
                if alpha > 0:
                    train_loss = train_loss + alpha * neighbor_distance_regularizer(self.model.theta) \
                          + theta_regularizer(self.model.theta)
                
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
                    
                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} - "
                          f"Val Loss: {val_metrics['loss']:.4f} - "
                          f"Acc: {val_metrics['acc']:.4f} - "
                          f"Prec: {val_metrics['prec']:.4f} - "
                          f"Rec: {val_metrics['rec']:.4f} - "
                          f"F1: {val_metrics['f1']:.4f} - "
                          f"MCC: {val_metrics['mcc']:.4f}")

                self.model.train()
                
    
    

    def test(self, test_dataset, window_size, num_nodes, batch_size=1):
        test_loader = TorchDataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            collate_fn=custom_collate_fn
        )

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

        return all_preds, all_labels





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

