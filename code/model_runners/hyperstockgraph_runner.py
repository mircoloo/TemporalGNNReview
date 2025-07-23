
from models.HyperStockGAT.training.models.base_models import NCModel
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
from model_runners.runner_utils import BaseGraphDataset
import torch


class HyperStockGraphDataset(BaseGraphDataset):
    def __init__(self, dataset):
        # Data(x=[n_nodes, features (5) * timestamps ], edge_index=[2, 1369852], edge_attr=[1369852], y=[1171])
        super().__init__(dataset)
        
    def __getitem__(self, idx):
        data_sample = self.dataset[idx] 
        x = data_sample.x 
        res = super().is_input_correct_shaped(x) # check if the input is correct shape, since some samples are wrong
        if not res:
            print(data_sample)
            x = super().adjust_input_shape(x) # in case reshape the tensor appending the last timestamp features
        
        # Write specifit reshape code
        x = x.reshape((self.n_nodes, self.n_features, self.seq_length)).permute(0,2,1)
        adj = to_dense_adj(edge_index=data_sample.edge_index, edge_attr=data_sample.edge_attr).squeeze() #create adjacency list
        y =  torch.tensor(data_sample.y.clone().detach(), dtype=torch.float32).unsqueeze(1)
        return x, y, adj



class HyperStockGraphRunner:
    def __init__(self, model, device):
        self.model: NCModel = model
        self.device = device

    def train(self, 
              train_dataset, 
              validation_dataset, 
              optimizer, 
              criterion, 
              epochs: int, 
              seq_length: int, 
              num_features: int):
        # Convert to HyperStockGraphDataset
        train_set = HyperStockGraphDataset(train_dataset)
        validation_set = HyperStockGraphDataset(validation_dataset)
        
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=1)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            total_samples = 0
            for batch in train_loader:
                x, y, adj = batch
                
                y = y.squeeze(0) # remove the batch dimension
                #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)
                
                optimizer.zero_grad()
                emb = self.model.encode(x, adj)
                output = self.model.decode(emb, adj)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                total_samples += 1

            avg_train_loss = train_loss / max(total_samples, 1)
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}")

            # Run validation every epoch

            val_loss = self.evaluate(val_loader, criterion, seq_length, num_features)
            print(f"[Epoch {epoch}] Validation Loss: {val_loss:.4f}")

    def evaluate(self, val_loader, criterion, seq_length, num_features):
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y, adj = batch
                
                y = y.squeeze(0) # remove the batch dimension
                #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)

                emb = self.model.encode(x, adj)
                output = self.model.decode(emb, adj)
                loss = criterion(output, y)

                total_loss += loss.item()
                total_samples += 1

        return total_loss / max(total_samples, 1)

    def test(self, test_dataset, seq_length, num_features):
        validation_set = HyperStockGraphDataset(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=1)
        self.model.eval()
        predictions = []
        true_values = []

        with torch.no_grad():
            for batch in test_loader:
                x, y, adj = batch
                if x.shape[-1] * x.shape[-2] != seq_length * num_features:
                    print(f"Wrong input dimensions")
                    continue
                
                y = y.squeeze(0) # remove the batch dimension
                #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)

                emb = self.model.encode(x, adj)
                output = self.model.decode(emb, adj)

                predictions.append(output.cpu())
                true_values.append(y.cpu())

        # Concatenate all prediction tensors into one
        all_predictions_tensor = torch.cat(predictions, dim=0) 

        # Apply sigmoid and thresholding to the single tensor
        preds = (torch.sigmoid(all_predictions_tensor) > .5).int()

        # Concatenate true values as well if they are still a list, and convert to appropriate format
        all_true_values_tensor = torch.cat(true_values, dim=0)

        # Ensure return types match what you expect for sklearn metrics (e.g., NumPy arrays or Python lists)
        return preds.numpy(), all_true_values_tensor.numpy() # Or .tolist() if preferred
