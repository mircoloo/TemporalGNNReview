
from models.HyperStockGAT.training.models.base_models import NCModel
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Dataset, DataLoader
from runner_utils import BaseGraphDataset 
import torch


class HyperStockGraphDataset(BaseGraphDataset):
    def __init__(self, dataset):
        # Data(x=[1171, 110], edge_index=[2, 1369852], edge_attr=[1369852], y=[1171])
        super().__init__(dataset)
        assert len(dataset) > 0, "0 data in dataset, impossibile to create the HyperStockDataset" 
        self.n_nodes = self.dataset[0].x.shape[0]
        self.n_features = int(self.dataset[0].x.shape[1] / 5)
        self.seq_length = int(self.dataset[0].x.shape[1] / self.n_features)
        self.n_edges = self.dataset[0].edge_index.shape[1]
    def __getitem__(self, idx):
        data_sample = self.dataset[idx] 
        x = data_sample.x 
        x = x.reshape((self.n_nodes, self.n_features, self.seq_length)).permute(0,2,1)
        adj = to_dense_adj(edge_index=data_sample.edge_index, edge_attr=data_sample.edge_attr).squeeze() #create adjacency list
        y =  torch.tensor(data_sample.y.clone().detach(), dtype=torch.float32).unsqueeze(1)
        return x, y, adj



class HyperStockGraphRunner:
    def __init__(self, model, device):
        self.model: NCModel = model
        self.device = device

    def _convert_data(self, data, seq_length, num_features, num_nodes):
        x = data['x']  # Node features
        x = x.reshape((num_nodes, num_features, seq_length)).permute(0,2,1)
        num_edges = data['edge_index'].shape[1]  # Number of edges
        adj = to_dense_adj(edge_index=data.edge_index, edge_attr=data.edge_attr).squeeze() #create adjacency list
        y = torch.tensor(data.y, dtype=torch.float32)  # Assuming binary classification targets
        # - x.shape [n_nodes, seq_length, features] 
        # - y.shape [n_nodes, 1] 
        # - adj.shape [n_nodes, n_nodes] 
        return x, y, adj

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
                if x.shape[-1] * x.shape[-2] != seq_length * num_features:
                    print(f"Wrong input dimensions")
                    continue
                
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

    def evaluate(self, loader, criterion, seq_length, num_features):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                x, y, adj = batch
                if x.shape[-1] * x.shape[-2] != seq_length * num_features:
                    print(f"Wrong input dimensions")
                    continue
                
                y = y.squeeze(0) # remove the batch dimension
                #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)

                emb = self.model.encode(x, adj)
                output = self.model.decode(emb, adj)
                loss = criterion(output, y)

                total_loss += loss.item()
                total_samples += 1

        return total_loss / max(total_samples, 1)

    def test(self, test_loader, seq_length, num_features):
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
