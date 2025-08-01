from .base_runner import BaseModelRunner
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from model_runners.runner_utils import BaseGraphDataset
from torch_geometric.loader import DataLoader

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

        
    

    def train(self, train_dataset, val_dataset, optimizer, criterion, num_epochs, seq_length):
        train_set = DARNNDataset(train_dataset)
        val_set = DARNNDataset(val_dataset)

        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

        min_val_loss = float('inf')
        #print(self.model)
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for train_sample in train_loader:
                # if train_sample.x.shape[-1] != 5 * seq_length:
                #     print(f"Warning: Skipping sample with incorrect shape: {train_sample.x.shape}")
                #     continue
                
                drivers, targets, y  = train_sample
                optimizer.zero_grad()                        

                print(f"{drivers.shape=} {targets.shape=} {y.shape=}")
                
                drivers = drivers.permute(0,2,1).to(self.device) # 
                targets = targets.reshape(1,-1,1).to(self.device) 
                print(f"{drivers.shape=} {targets.shape=} {y.shape=}")

                outputs = self.model(drivers, targets)


                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            # ===== VALIDATION =====
            if epoch % 5 == 0:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    logits: list = []
                    y_trues: list = []
                    for val_sample in val_loader:
                        if val_sample.x.shape[-1] != 5 * seq_length:
                            print(f"Warning: Skipping val sample with incorrect shape: {val_sample.x.shape}")
                            continue
                        val_sample = val_sample.to(self.device)
                        for idx in range(len(val_sample.x)):
                            drivers, targets, y = self._convert_data(val_sample, seq_length, idx)
                            # drivers: [num_drivers, seq_length], targets: [seq_length], y
                            drivers = drivers.unsqueeze(0).permute(0, 2, 1).to(self.device)
                            targets = targets.reshape(1, -1, 1).to(self.device)
                            y = y.reshape(1, 1).to(self.device)
                            outputs = self.model(drivers, targets)
                            loss = criterion(outputs, y)
                            logits.append(outputs.item())
                            y_trues.append(y.item())
                            val_loss += loss.item()
                            total_nodes_validated_this_epoch += 1

                avg_val_loss = val_loss / max(total_nodes_validated_this_epoch, 1)
                # Convert raw outputs to binary predictions
                preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()
                acc = accuracy_score(y_trues, preds)
                rec = recall_score(y_trues, preds)
                f1 = f1_score(y_trues, preds, zero_division=0)
                mcc = matthews_corrcoef(y_trues, preds)
                print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_loss:.4f} - Acc: {acc:.4f} - Rec: {rec:.4f} - F1: {f1:.4f} - MCC: {mcc:.4f}")
                # Save best model
                # if avg_val_loss < min_val_loss:
                #     min_val_loss = avg_val_loss
                #     best_model_state = self.model.state_dict()

        # Optional: reload best model after training
        #self.model.load_state_dict(best_model_state)

           

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
    