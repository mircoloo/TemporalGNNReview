from .base_runner import BaseModelRunner
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

class DARNNRunner(BaseModelRunner):

    def _convert_data(self, data, seq_length): 
        """
        Convert data to the format expected by the DA-RNN model.
        """
        X = data.x 
        y = data.y
        # Assuming data.x is of shape [num_nodes, seq_length * num_features]
        num_nodes = X.shape[0]
        close_price_index = 0  
        X = X.reshape(num_nodes, -1, seq_length).permute(0,2,1)[:,:,close_price_index]  # Reshape to [num_nodes, seq_length, num_features] and matain only the closing price (index = 0)
        # X is shape [n_nodes, seq_length]
        # select a random index from nodes 
        node_idx = torch.randint(0, num_nodes).item()  # Randomly select a node index    
        # NOW WE NEED TO RETURN THE DATA IN A FORMAT THAT THE MODEL EXPECTS
        # 16/07/2025

        print(X.shape, X[0])
        
    

    def train(self, train_loader, val_loader, optimizer, criterion, num_epochs, seq_length):
        min_val_loss = float('inf')
        counter = 0
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for train_sample in train_loader:
                if train_sample.x.shape[-1] != 5 * seq_length:
                    print(f"Warning: Skipping sample with incorrect shape: {train_sample.x.shape}")
                    continue
                train_sample = train_sample.to(self.device)
                drivers, targets, y   = self._convert_data(train_sample, seq_length)
                return
                optimizer.zero_grad()
                outputs = self.model(train_sample.x)
                loss = criterion(outputs, train_sample.y.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

           

    def test(self, test_loader, criterion):
      pass