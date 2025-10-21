
import numpy as np
from tabulate import tabulate
from model_runners.base_runner import evaluate_decorator, BaseModelRunner
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.utils.data import DataLoader
from model_runners.runner_utils import BaseGraphDataset
import torch
from model_runners.models_dataset import HyperStockGATDataset


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
              num_features: int):
        # Convert to HyperStockGATDataset
        train_set = HyperStockGATDataset(train_dataset)
        validation_set = HyperStockGATDataset(validation_dataset)
        
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=1)

        for epoch in range(1,epochs+1):
            self.model.train()
            train_loss = 0.0
            total_samples = 0
            for batch in train_loader:
                x, y, adj = batch
                #y = y.squeeze(0) # remove the batch dimension
                #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)
                #print(f"hyperstockgat input x.shape: {x.shape}, adj.shape: {adj.shape}, y.shape: {y.shape}")
                optimizer.zero_grad()
                emb = self.model.encode(x, adj)
                outputs = self.model.decode(emb, adj)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                total_samples += 1
            avg_train_loss = train_loss / float(max(total_samples, 1))
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}")
            
            # Run validation every epoch

            if epoch % 1 == 0:
                self.model.eval()
                val_metrics = {
                    'loss': 0.0, 'acc': 0.0, 'prec': 0.0, 
                    'f1': 0.0, 'mcc': 0.0, 'rec': 0.0
                }
                n_val = 0
                
                with torch.no_grad():
                    for batch in val_loader:                            
                        x, y, adj = batch
                        #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                        x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)
                        #print(f"hyperstockgat input x.shape: {x.shape}, adj.shape: {adj.shape}, y.shape: {y.shape}")
                        targets = y
                        optimizer.zero_grad()
                        emb = self.model.encode(x, adj)
                        outputs = self.model.decode(emb, adj)
                        loss = criterion(outputs, targets)

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
                    if epoch % 5 == 0 :
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
                        
                        print(f"\nEpoch {epoch+1}/{epochs} Results:")
                        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
                        print("\n")

                self.model.train()
    
    
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
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
                all_preds.extend(preds.flatten().tolist())
                all_labels.extend(y.cpu().flatten().tolist())

        return {'preds': np.array(all_preds), 'targets': np.array(all_labels)}
