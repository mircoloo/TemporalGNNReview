from .base_runner import BaseModelRunner
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch_geometric.utils import to_dense_adj


class GraphWaveNetRunner(BaseModelRunner):

    def convert_data(self, data, seq_length, num_features):
        # data.x: [num_nodes, seq_length * num_features]
        x = data.x
        num_nodes = x.shape[0]
        x = x.view(num_nodes, 5, -1).permute(0, 2, 1) # [num_nodes, seq_length, num_features]
        x = x.unsqueeze(0)  # batch size 1
        x = x.permute(0, 3, 1, 2) #(batch_size, num_features, num_nodes, sequence_length)             # rechanged the size 15/07/2025   
        y = data.y.long()  # Ensure y is long for classification

        return x, y

    def train(self, train_loader, val_loader, optimizer, criterion, num_epochs, seq_length, num_features):
        self.model.train()
        for epoch in range(num_epochs):
            for data in train_loader:
                if data.x.shape[-1] != seq_length * num_features:
                    (f"Warning: Skipping sample with incorrect shape: {data.x.shape}")
                    continue
                x, y = self.convert_data(data, seq_length, num_features)
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)  # [1, 1, num_nodes]
                outputs = outputs.squeeze(0).squeeze(0)  # [num_nodes]
                outputs = outputs[:, -1] if outputs.dim() == 2 else outputs  # [num_nodes] (last time step for all nodes)
                loss = criterion(outputs, y.float())
                preds = (torch.sigmoid(outputs) > 0.5).long()
                acc = accuracy_score(y.cpu(), preds.cpu())
                loss.backward()
                optimizer.step()
            if epoch % 5 == 0 and val_loader is not None:
                self.model.eval()
                total_val_loss = 0.0
                all_preds = []
                all_targets = []

                with torch.no_grad():
                    for val_data in val_loader:
                        if val_data.x.shape[-1] != seq_length * num_features:
                            print(f"Warning: Skipping sample with incorrect shape: {val_data.x.shape}")
                            continue

                        x_val, y_val = self.convert_data(val_data, seq_length, num_features)
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)

                        val_outputs = self.model(x_val)
                        val_outputs = val_outputs.squeeze(0).squeeze(0)  # [num_nodes]
                        val_outputs = val_outputs[:, -1] if val_outputs.dim() == 2 else val_outputs  # [num_nodes]

                        val_loss = criterion(val_outputs, y_val.float())
                        total_val_loss += val_loss.item()

                        preds = (torch.sigmoid(val_outputs) > 0.5).long()

                        all_preds.append(preds.cpu())
                        all_targets.append(y_val.cpu())

                # After all batches
                if all_preds:
                    y_true = torch.cat(all_targets)
                    y_pred = torch.cat(all_preds)

                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='weighted')
                    mcc = matthews_corrcoef(y_true, y_pred)
                    avg_val_loss = total_val_loss / len(all_preds)

                    print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.4f} | "
                        f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | MCC: {mcc:.4f}")  


    def test(self, test_loader, seq_length, num_features):
        self.model.eval()
        all_preds = []
        all_targets = []    
        with torch.no_grad():
            for data in test_loader:
                if data.x.shape[-1] != seq_length * num_features:
                    print(f"Warning: Skipping sample with incorrect shape: {data.x.shape}")
                    continue
                x, y = self.convert_data(data, seq_length, num_features)
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)
                outputs = outputs.squeeze(0).squeeze(0)
                outputs = outputs[:, -1] if outputs.dim() == 2 else outputs  # [num_nodes]
                preds = (torch.sigmoid(outputs) > 0.5).long()
                all_preds.append(preds.cpu())           
                all_targets.append(y.cpu())
        if all_preds:
            y_true = torch.cat(all_targets)
            y_pred = torch.cat(all_preds)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_true, y_pred)
            print(f"Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | MCC: {mcc:.4f}")
            
        