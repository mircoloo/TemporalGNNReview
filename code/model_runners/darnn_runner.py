from .base_runner import BaseModelRunner
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from model_runners.runner_utils import BaseGraphDataset
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class DARNNDataset(BaseGraphDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
    
    def __getitem__(self, idx):
        data_sample = self.dataset[idx] 
        x = data_sample.x 
        res = super().is_input_correct_shaped(x) # check if the input is correct shape, since some samples are wrong
        if not res:
            print(data_sample)
            x = super().adjust_input_shape(x) # in case reshape the tensor appending the last timestamp features
        
        # Write specifit reshape code
        X = x
        y = data_sample.y
        # Assuming data.x is of shape [num_nodes, seq_length * num_features]
        num_nodes = X.shape[0]
        close_price_index = 0 

        X = X.reshape(num_nodes, -1, self.seq_length).permute(0,2,1)[:,:,close_price_index]  # Reshape to [num_nodes, seq_length, num_features] and matain only the closing price (index = 0)
        # X is shape [n_nodes, seq_length]
        # select a random index from nodes 

        retrieve_index: int = torch.randint(low=0,high=num_nodes, size=(1,1)).item()  # Randomly select a node index    

        target_series = X[retrieve_index, :]
        target = y[retrieve_index]
        mask = torch.arange(num_nodes) != retrieve_index
        drivers = X[mask, :]  

        return drivers, target_series, target




class DARNNRunner(BaseModelRunner):
    def __init__(self, model, device, market_name):
        super().__init__(model, device, market_name)
        self.model_name = "DARNN"

    def _convert_data(self, data, seq_length, retrieve_index=None): 
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
        if retrieve_index is None:
            retrieve_index: int = torch.randint(low=0,high=num_nodes, size=(1,1)).item()  # Randomly select a node index    

        target_series = X[retrieve_index, :]
        target = y[retrieve_index]
        mask = torch.arange(num_nodes) != retrieve_index
        drivers = X[mask, :]  


        return drivers, target_series, target

        
    

    def train(self, train_dataset, val_dataset, optimizer, criterion, num_epochs, seq_length, batch_size=32):
        writer = SummaryWriter('runs/')
        train_set = DARNNDataset(train_dataset)
        val_set = DARNNDataset(val_dataset)
        
        # Use actual batching
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x_batch, y_batch = batch  # Assuming your dataset returns (x, y)
                batch_size = x_batch.size(0)
                
                # Move batch to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch.float())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / n_batches
            writer.add_scalar(f'{self.model_name}/{self.market_name}/loss_train', avg_train_loss, epoch)

            # Validation every 5 epochs
            if epoch % 5 == 0:
                self.model.eval()
                val_loss = 0.0
                val_preds = []
                val_targets = []
                n_val = 0

                with torch.no_grad():
                    for batch in val_loader:
                        x_val, y_val = batch
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)

                        outputs = self.model(x_val)
                        loss = criterion(outputs, y_val.float())
                        val_loss += loss.item()

                        # Store predictions and targets
                        preds = (torch.sigmoid(outputs) > 0.5).int()
                        val_preds.extend(preds.cpu().numpy())
                        val_targets.extend(y_val.cpu().numpy())
                        n_val += 1

                avg_val_loss = val_loss / n_val
                writer.add_scalar(f'{self.model_name}/{self.market_name}/loss_val', avg_val_loss, epoch)

                # Calculate metrics
                val_preds = np.array(val_preds)
                val_targets = np.array(val_targets)
                
                acc = accuracy_score(val_targets, val_preds)
                f1 = f1_score(val_targets, val_preds, zero_division=0)
                mcc = matthews_corrcoef(val_targets, val_preds)

                print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, '
                      f'Val Loss = {avg_val_loss:.4f}, Acc = {acc:.4f}, '
                      f'F1 = {f1:.4f}, MCC = {mcc:.4f}')

                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), f'best_model_{self.model_name}_{self.market_name}.pt')

        writer.close()
    def test(self, test_dataset, criterion):
        test_set = DARNNDataset(test_dataset)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
        self.model.eval()  
        predictions = []
        true_values = []
        total_loss = 0.0

        with torch.no_grad():
            for test_sample in test_loader:
                for idx in range(len(test_sample.x)):
                    test_sample = test_sample.to(self.device)
                    drivers, targets, y = self._convert_data(test_sample, self.seq_length, idx)
                    drivers = drivers.unsqueeze(0).permute(0, 2, 1).to(self.device)  # [1, seq_len, num_drivers]
                    targets = targets.reshape(1, -1, 1).to(self.device)  # [1, seq_len, 1]
                    logits = self.model(drivers, targets)
                    loss = criterion(logits, y.reshape(1,1))
                    total_loss += loss.item()
                    predictions.append(logits.item())
                    true_values.append(y.item())

        y_pred = [1 if torch.sigmoid(logit) > .5 else 0 for logit in predictions]


        return y_pred, true_values
