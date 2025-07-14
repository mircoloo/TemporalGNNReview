from .base_runner import BaseModelRunner
import torch


class GraphWaveNetRunner(BaseModelRunner):

    def convert_data(self, data, seq_length, num_features):
        # data.x: [num_nodes, seq_length * num_features]
        x = data.x
        num_nodes = x.shape[0]
        x = x.view(num_nodes, 5, -1).permute(0, 2, 1) # [num_nodes, seq_length, num_features]
        # Graph-WaveNet expects [batch, num_nodes, seq_length, num_features]
        x = x.unsqueeze(0)  # batch size 1
        y = data.y.long()  # Ensure y is long for classification
        return x, y

    def train(self, train_loader, val_loader, optimizer, criterion, num_epochs, seq_length, num_features):
        self.model.train()
        for epoch in range(num_epochs):
            for data in train_loader:
                print(data)
                
                x, y = self.convert_data(data, seq_length, num_features)
                print(x.shape, y.shape)
                print(f"{x=}, {y=}")  # Debugging line to check shapes

                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                # outputs: [batch, num_nodes, num_classes] or [batch, num_nodes] if binary
                # Flatten for criterion: [batch * num_nodes, num_classes], y: [batch * num_nodes]
                if outputs.dim() == 4:
                    # [batch, num_nodes, 1, num_classes] -> [batch, num_nodes, num_classes]
                    outputs = outputs.squeeze(2)
                loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
        # Add validation logic as needed

    def test(self, test_loader, seq_length, num_features):
        self.model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for data in test_loader:
                x, y = self.convert_data(data, seq_length, num_features)
                x = x.to(self.device)
                logits = self.model(x)
                if logits.dim() == 4:
                    logits = logits.squeeze(2)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return all_logits, all_labels