from .base_runner import BaseModelRunner
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

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
        # 16/07/2025
        target_series = X[retrieve_index, :]
        target = y[retrieve_index]
        mask = torch.arange(num_nodes) != retrieve_index
        drivers = X[mask, :]  
        return drivers, target_series, target

        
    

    def train(self, train_loader, val_loader, optimizer, criterion, num_epochs, seq_length):
        min_val_loss = float('inf')
        print(self.model)
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
                optimizer.zero_grad()                        

                drivers = drivers.unsqueeze(0).permute(0,2,1).to(self.device)
                targets = targets.reshape(1,-1,1).to(self.device)
                outputs = self.model(drivers, targets)
                y = y.reshape(1,1)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            # ===== VALIDATION =====
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                y_hats: list = []
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
                        y_hats.append(outputs.item())
                        y_trues.append(y.item())
                        val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            # Convert raw outputs to binary predictions
            y_preds = [1 if o > 0 else 0 for o in y_hats]
            acc = accuracy_score(y_trues, y_preds)
            f1 = f1_score(y_trues, y_preds, zero_division=0)
            mcc = matthews_corrcoef(y_trues, y_preds)
            print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_loss:.4f} - Acc: {acc:.4f} - F1: {f1:.4f} - MCC: {mcc:.4f}")

            # Save best model
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()

        # Optional: reload best model after training
        #self.model.load_state_dict(best_model_state)

           

    def test(self, test_loader, criterion):
        self.model.eval()
        predictions = []
        true_values = []
        total_loss = 0.0

        with torch.no_grad():
            for test_sample in test_loader:
                if test_sample.x.shape[-1] != 5 * self.seq_length:
                    print(f"Warning: Skipping sample with incorrect shape: {test_sample.x.shape}")
                    continue
                test_sample = test_sample.to(self.device)
                drivers, targets, y = self._convert_data(test_sample, self.seq_length)
                drivers = drivers.unsqueeze(0).permute(0, 2, 1).to(self.device)  # [1, seq_len, num_drivers]
                targets = targets.reshape(1, -1, 1).to(self.device)  # [1, seq_len, 1]
                output = self.model(drivers, targets)
                loss = criterion(output, y.reshape(1,1))
                total_loss += loss.item()
                predictions.append(output.item())
                true_values.append(y.item())

        y_pred = torch.tensor([1 if o > 0 else 0 for o in predictions])
        y_true = torch.tensor([1 if t > 0 else 0 for t in true_values])

        return y_pred, y_true
    