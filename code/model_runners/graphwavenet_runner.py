from model_runners.runner_utils import BaseGraphDataset
from model_runners.base_runner import BaseModelRunner
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score
from torch_geometric.utils import to_dense_adj


class GraphWaveNetDataset(BaseGraphDataset):
    def __init__(self, dataset):
        # Data(x=[1171, 110], edge_index=[2, 1369852], edge_attr=[1369852], y=[1171])
        super().__init__(dataset)
    def __getitem__(self, idx):
        data_sample = self.dataset[idx]
        data_sample = self.dataset[idx] 
        x = data_sample.x 
        res = super().is_input_correct_shaped(x) # check if the input is correct shape, since some samples are wrong
        if not res:
            print(data_sample)
            x = super().adjust_input_shape(x) # in case reshape the tensor appending the last timestamp features
        x = data_sample.x
        x = x.view(self.n_nodes, self.n_features, self.seq_length).permute(0, 2, 1) # [num_nodes, seq_length, num_features]
        x = x.unsqueeze(0)  # batch size 1
        x = x.permute(0, 3, 1, 2) #(batch_size, num_features, num_nodes, sequence_length)             # rechanged the size 15/07/2025   
        y = data_sample.y.long()  # Ensure y is long for classification
        return x, y
        


class GraphWaveNetRunner(BaseModelRunner):

    def _convert_data(self, data, seq_length, num_features):
        # data.x: [num_nodes, seq_length * num_features]
        x = data.x
        num_nodes = x.shape[0]
        x = x.view(num_nodes, num_features, seq_length).permute(0, 2, 1) # [num_nodes, seq_length, num_features]
        x = x.unsqueeze(0)  # batch size 1
        x = x.permute(0, 3, 1, 2) #(batch_size, num_features, num_nodes, sequence_length)             # rechanged the size 15/07/2025   
        y = data.y.long()  # Ensure y is long for classification

        return x, y

    def train(self, train_dataset, val_dataset, optimizer, criterion, num_epochs, seq_length, num_features):
        train_set = GraphWaveNetDataset(train_dataset)
        val_set = GraphWaveNetDataset(val_dataset)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1)

        self.model.train()
        for epoch in range(num_epochs):
            for data in train_loader:
                print(f"{data=}")
                if data.x.shape[-1] != seq_length * num_features:
                    (f"Warning: Skipping sample with incorrect shape: {data.x.shape}")
                    continue
                x, y = self._convert_data(data, seq_length, num_features)
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

                        x_val, y_val = self._convert_data(val_data, seq_length, num_features)
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


    def test(self, test_dataset, seq_length, num_features):
        test_set = GraphWaveNetDataset(test_dataset)
        test_loader = DataLoader(test_set, batch_size=1)
        self.model.eval()
        all_preds_logits = [] # Renamed for clarity: these are raw logits
        all_targets = []    
        with torch.no_grad():
            for data in test_loader:
                if data.x.shape[-1] != seq_length * num_features:
                    print(f"Warning: Skipping sample with incorrect shape: {data.x.shape}")
                    continue
                x, y = self._convert_data(data, seq_length, num_features)
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                logits = logits.squeeze(0).squeeze(0)
                logits = logits[:, -1] if logits.dim() == 2 else logits  # [num_nodes]
                all_preds_logits.append(logits.cpu())           
                all_targets.append(y.cpu())

        # Concatenate all raw logit tensors into one
        concatenated_logits = torch.cat(all_preds_logits, dim=0) 

        # Apply sigmoid and thresholding to the single tensor
        y_pred = (torch.sigmoid(concatenated_logits) > .5).int()

        # Concatenate true values (this line was already correct)
        y_true = torch.cat(all_targets)

        # Convert to numpy arrays for compatibility with sklearn metrics if used outside
        return y_pred.numpy(), y_true.numpy()