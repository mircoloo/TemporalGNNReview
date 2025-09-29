
from model_runners.base_runner import BaseModelRunner
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score
from torch.utils.data import DataLoader
from model_runners.runner_utils import BaseGraphDataset
from torch.utils.tensorboard import SummaryWriter
import torch
from model_runners.models_dataset import HyperStockGATDataset


class HyperStockGATRunner(BaseModelRunner):
    def __init__(self, model, device, market_name):
        super().__init__(model, device, market_name)
        self.model_name = "HyperStockGAT"

    def train(self, 
              train_dataset, 
              validation_dataset, 
              optimizer, 
              criterion, 
              epochs: int, 
              seq_length: int, 
              num_features: int):
        # Convert to HyperStockGATDataset
        train_set = HyperStockGATDataset(train_dataset)
        validation_set = HyperStockGATDataset(validation_dataset)
        
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=1)

        for epoch in range(1,epochs+1):
            self.model.train()
            train_loss = 0.0
            total_samples = 0
            for batch in train_loader:
                x, y, adj = batch
                #y = y.squeeze(0) # remove the batch dimension
                #x, y, adj = self._convert_data(batch, seq_length, num_features, batch.x.shape[0])
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)
                #print(f"hyperstockgat input x.shape: {x.shape}, adj.shape: {adj.shape}, y.shape: {y.shape}")
                optimizer.zero_grad()
                emb = self.model.encode(x, adj)
                output = self.model.decode(emb, adj)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                if torch.isnan(loss):
                    print(f"For x:{x},\n\n adj:{adj},\n\n y:{y}, \n\n output:{output}")
                    return
                train_loss += loss.item()
                total_samples += 1
            avg_train_loss = train_loss / float(max(total_samples, 1))
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}")
            
            # Run validation every epoch

            self.evaluate(val_loader, criterion, seq_length, num_features)




    def evaluate(self, val_loader, criterion, seq_length, num_features, epoch=None, num_epochs=None):
        """
        Evaluates the model on the validation set, calculates loss and various classification metrics.

        Args:
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            criterion (torch.nn.Module): The loss function (e.g., BCEWithLogitsLoss).
            seq_length (int): The expected sequence length of input features.
            num_features (int): The expected number of features per node.
            epoch (int, optional): The current epoch number (for printing). Defaults to None.
            num_epochs (int, optional): The total number of epochs (for printing). Defaults to None.
        """
        self.model.eval() # Set the model to evaluation mode
        total_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad(): # Disable gradient calculations for inference
            for val_data in val_loader:
                # Check if val_data.x exists and has the expected shape for features.
                # This handles cases where samples might be malformed or filtered.


                # Convert raw data from val_data object into x_val and y_val tensors.
                # This method (_convert_data) should handle any necessary restructuring,
                # including potentially extracting/processing adjacency matrices if required by the model.
                x, y, adj = val_data
                x, y, adj = x.to(self.device), y.to(self.device), adj.to(self.device)
                
                # Move input and target tensors to the specified device (CPU/GPU)
                emb = self.model.encode(x, adj)
                val_outputs = self.model.decode(emb, adj)
                val_loss = criterion(val_outputs, y)

                # Perform the forward pass through the model

                # Adjust output shape:
                # - .squeeze(0).squeeze(0) removes any batch or singleton dimensions if they exist,
                #   resulting in a shape like [num_nodes] or [num_nodes, 1].
                # - val_outputs[:, -1] is used if the model's output is a sequence (e.g., [num_nodes, sequence_len])
                #   and you only need the prediction from the last time step.
                val_outputs = val_outputs.squeeze(0).squeeze(0)
                val_outputs = val_outputs[:, -1] if val_outputs.dim() == 2 else val_outputs

                # Calculate the loss for the current batch
                # y_val.float() ensures the target labels are float, which is required by common loss functions
                # like BCEWithLogitsLoss for binary classification.
                total_val_loss += val_loss.item() # Accumulate the loss

                # Generate binary predictions from model outputs (logits)
                # - torch.sigmoid converts logits to probabilities.
                # - > 0.5 thresholds probabilities to binary classes (0 or 1).
                # - .long() converts boolean results to integer (0 or 1).
                preds = (torch.sigmoid(val_outputs) > 0.5).long()
                # Collect predictions and true labels for overall metric calculation later
                # Move to CPU before appending as scikit-learn metrics typically operate on CPU numpy arrays/tensors.
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        # --- After iterating through all batches ---
        # Concatenate all collected predictions and targets into single tensors
        if all_preds: # Ensure that there were valid samples processed
            y_true = torch.cat(all_targets)
            y_pred = torch.cat(all_preds)

            # Calculate average validation loss over all processed samples
            avg_val_loss = total_val_loss / len(all_preds)

            # Calculate various classification metrics using scikit-learn
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted') # 'weighted' considers label imbalance
            mcc = matthews_corrcoef(y_true, y_pred)
            rec = recall_score(y_true, y_pred, average='weighted') # 'weighted' for potentially imbalanced binary classes

            # Print the evaluation results
            if epoch is not None and num_epochs is not None:
                print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_loss:.4f} - Acc: {acc:.4f} - Rec: {rec:.4f} - F1: {f1:.4f} - MCC: {mcc:.4f}")
            else:
                print(f"Validation Results - Val Loss: {avg_val_loss:.4f} - Acc: {acc:.4f} - Rec: {rec:.4f} - F1: {f1:.4f} - MCC: {mcc:.4f}")
        else:
            # Message if no samples were processed (e.g., all were skipped due to shape mismatch)
            print("No valid samples processed during evaluation. All samples were skipped or val_loader was empty.")
            # Return default values if no evaluation was performed
            avg_val_loss, acc, f1, mcc, rec = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

        # Return the calculated metrics
        return avg_val_loss, acc, f1, mcc, rec

    def test(self, test_dataset, seq_length, num_features):
        validation_set = HyperStockGATDataset(test_dataset)
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
        
        #df['preds'] = all_preds
        #df['targets'] = all_labels

        return df
        # Ensure return types match what you expect for sklearn metrics (e.g., NumPy arrays or Python lists)
        return preds.numpy(), all_true_values_tensor.numpy() # Or .tolist() if preferred
