from .base_runner import BaseModelRunner
import torch

class GraphWaveNetRunner(BaseModelRunner):
    def convert_data(self, data, seq_length, num_features):
        # data.x: [num_nodes, seq_length * num_features]
        x = data.x
        num_nodes = x.shape[0]
        x = x.view(num_nodes, seq_length, num_features)  # [num_nodes, seq_length, num_features]
        # Graph-WaveNet expects [batch, num_nodes, seq_length, num_features]
        x = x.unsqueeze(0)  # batch size 1
        return x, data.edge_index, data.y

    def train(self, train_loader, val_loader, optimizer, criterion, num_epochs, seq_length, num_features):
        self.model.train()
        for epoch in range(num_epochs):
            for data in train_loader:
                x, edge_index, y = self.convert_data(data, seq_length, num_features)
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x, edge_index)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
        # Add validation logic as needed

    def test(self, test_loader, seq_length, num_features):
        self.model.eval()
        # ...testing logic...