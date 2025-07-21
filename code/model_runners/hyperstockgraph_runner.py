
from models.HyperStockGAT.training.models.base_models import NCModel
from torch_geometric.utils import to_dense_adj

import torch

class HyperStockGraphRunner:
    def __init__(self, model, device):
        self.model: NCModel = model
        self.device = device
        

    def _convert_data(self, data, seq_length, num_features, num_nodes):
        x = data['x']  # Node features
        x = x.reshape((num_nodes, num_features, seq_length)).permute(0,2,1)  # Reshape to (n_nodes, l, feat_dim)
        num_edges = data['edge_index'].shape[1]  # Number of edges
        adj = to_dense_adj(edge_index=data.edge_index, edge_attr=data.edge_attr).squeeze() 
        y =  torch.tensor(data.y, dtype=torch.float32).unsqueeze(1)  # Assuming binary classification targets

        return x, y, adj

    def train(self, train_loader, validation_loader, optimizer, criterion, epochs, seq_length, num_features):
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            total_samples = 0

            for batch in train_loader:
                if batch.x.shape[-1] != num_features * seq_length:
                    print(f"[Epoch {epoch}] Skipping sample with shape {batch.x.shape}")
                    continue

                x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
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
            val_loss = self.evaluate(validation_loader, criterion, seq_length, num_features)
            print(f"[Epoch {epoch}] Validation Loss: {val_loss:.4f}")

    def evaluate(self, loader, criterion, seq_length, num_features):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                if batch.x.shape[-1] != num_features * seq_length:
                    continue

                x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
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
                if batch.x.shape[-1] != num_features * seq_length:
                    continue

                x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)

                emb = self.model.encode(x, adj)
                output = self.model.decode(emb, adj)

                predictions.append(output.cpu())
                true_values.append(y.cpu())

        return torch.cat(predictions), torch.cat(true_values)