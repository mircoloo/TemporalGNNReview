from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from .base_runner import BaseModelRunner
import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from runner_utils import BaseGraphDataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch



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
    

    def train(self, train_dataset, val_dataset, optimizer, criterion, num_epochs, alpha, 
          neighbor_distance_regularizer, theta_regularizer, window_size, num_nodes, 
          batch_size=32, use_validation=True):
    
        # Create organized TensorBoard writer
        writer = SummaryWriter(f'runs/{self.market_name}/{self.model_name}')
        self.model.train()
        
        # Custom collate function to handle different sizes
        
        
        # Use actual batching with custom collate
        # Use actual batching with custom collate
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
                optimizer.zero_grad()
                
                # Handle batched data
                X = batch.x.view(-1, num_nodes, batch.x.size(-1))  # [B, N, F]
                A = to_dense_adj(
                    batch.edge_index, 
                    batch=batch.batch,
                    edge_attr=batch.edge_attr,
                    max_num_nodes=num_nodes
                )
                
                # Forward pass with batched inputs
                outputs = self.model(X, A)  # [B, N, 1]
                targets = batch.y.view(-1, num_nodes, 1).float()  # [B, N, 1]
                
                # Compute loss
                loss = criterion(outputs, targets)
                if alpha > 0:
                    loss = loss + alpha * neighbor_distance_regularizer(self.model.theta) \
                          + theta_regularizer(self.model.theta)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_train += 1

            # Training metrics
            avg_train_loss = train_loss / n_train
            writer.add_scalar('Training/Loss', avg_train_loss, epoch)
            
            # Add regularization terms separately
            if alpha > 0:
                writer.add_scalar('Training/Neighbor_Distance_Reg', 
                                neighbor_distance_regularizer(self.model.theta), epoch)
                writer.add_scalar('Training/Theta_Reg', 
                                theta_regularizer(self.model.theta), epoch)

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
                        if batch.x.shape[-1] != 5 * window_size:
                            continue
                            
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
                    
                    # Log metrics in organized sections
                    writer.add_scalar('Validation/Loss', val_metrics['loss'], epoch)
                    writer.add_scalar('Validation/Accuracy', val_metrics['acc'], epoch)
                    writer.add_scalar('Validation/Precision', val_metrics['prec'], epoch)
                    writer.add_scalar('Validation/Recall', val_metrics['rec'], epoch)
                    writer.add_scalar('Validation/F1', val_metrics['f1'], epoch)
                    writer.add_scalar('Validation/MCC', val_metrics['mcc'], epoch)

                    # Add model parameter distributions
                    for name, param in self.model.named_parameters():
                        writer.add_histogram(f'Parameters/{name}', param, epoch)

                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} - "
                          f"Val Loss: {val_metrics['loss']:.4f} - "
                          f"Acc: {val_metrics['acc']:.4f} - "
                          f"Prec: {val_metrics['prec']:.4f} - "
                          f"Rec: {val_metrics['rec']:.4f} - "
                          f"F1: {val_metrics['f1']:.4f} - "
                          f"MCC: {val_metrics['mcc']:.4f}")

                self.model.train()

        writer.close()

    def test(self, test_dataset, window_size, num_nodes, batch_size=32):
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

