from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from .base_runner import BaseModelRunner
import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader
from runner_utils import BaseGraphDataset




class DGDNNRunner(BaseModelRunner):
    def train(self, train_dataset, val_dataset, optimizer, criterion, num_epochs, alpha, neighbor_distance_regularizer, theta_regularizer, window_size, num_nodes, use_validation=True):
        self.model.train()
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        for epoch in range(num_epochs + 1):
            train_loss = 0.0
            for train_sample in train_loader:
                if train_sample.x.shape[-1] != 5 * window_size:
                    print(f"Warning: Skipping sample with incorrect shape: {train_sample.x.shape}")
                    continue
                train_sample = train_sample.to(self.device)
                optimizer.zero_grad()
                A = to_dense_adj(train_sample.edge_index, batch=train_sample.batch, edge_attr=train_sample.edge_attr, max_num_nodes=num_nodes).squeeze(0)
                C = train_sample.y.unsqueeze(dim=1).float()
                outputs = self.model(train_sample.x, A)
                loss = criterion(outputs, C) \
                       + alpha * neighbor_distance_regularizer(self.model.theta) \
                       + theta_regularizer(self.model.theta)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            if use_validation and (epoch % 1 == 0):
                val_loss, val_acc, val_f1, val_mcc = 0.0, 0.0, 0.0, 0.0
                self.model.eval()
                n_val = 0
                with torch.no_grad():
                    for val_sample in val_loader:
                        if val_sample.x.shape[-1] != 5 * window_size: continue
                        val_sample = val_sample.to(self.device)
                        A = to_dense_adj(val_sample.edge_index, batch=val_sample.batch, edge_attr=val_sample.edge_attr, max_num_nodes=num_nodes).squeeze(0)
                        logits = self.model(val_sample.x, A)
                        y_true = val_sample.y.detach().cpu()
                        y_pred = (torch.sigmoid(logits) > 0.5).int().detach().cpu().squeeze()
                        val_loss += criterion(logits, val_sample.y.unsqueeze(1).float()).item()
                        val_acc += accuracy_score(y_true, y_pred)
                        val_f1 += f1_score(y_true, y_pred, zero_division=0)
                        val_mcc += matthews_corrcoef(y_true, y_pred)
                        n_val += 1
                        #print(f"{y_pred=}")
                if n_val > 0:
                    avg_val_loss = val_loss / n_val
                    avg_val_acc = val_acc / n_val
                    avg_val_f1 = val_f1 / n_val
                    avg_val_mcc = val_mcc / n_val
                    print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Val F1: {avg_val_f1:.4f},  Val MCC: {avg_val_mcc:.4f}")
                self.model.train()

    def test(self, test_dataset, window_size, num_nodes):
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        self.model.eval()
        all_logits = torch.tensor([]).to(self.device)
        all_labels = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for test_sample in test_loader:
                if test_sample.x.shape[-1] != 5 * window_size: continue
                test_sample = test_sample.to(self.device)
                A = to_dense_adj(test_sample.edge_index, batch=test_sample.batch, edge_attr=test_sample.edge_attr, max_num_nodes=num_nodes).squeeze(0)
                logits = self.model(test_sample.x, A)
                all_logits = torch.cat((all_logits, logits.squeeze()), dim=0)
                all_labels = torch.cat((all_labels, test_sample.y), dim=0) 

        y_pred = [1 if torch.sigmoid(logit).item() > 0.5 else 0 for logit in all_logits] # Assuming all_logits now holds the accumulated logits
        # Also, ensure all_labels is used for y_true for the final return
        y_true = all_labels.detach().cpu().tolist() # Convert accumulated labels to list for sklearn metrics

        return y_pred, y_true
 
        