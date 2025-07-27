from model_runners.runner_utils import BaseGraphDataset
from model_runners.base_runner import BaseModelRunner
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch_geometric.utils import to_dense_adj
from torch.utils.tensorboard import SummaryWriter


class GraphWaveNetDataset(BaseGraphDataset):
    def __init__(self, dataset):
        # Data(x=[1171, 110], edge_index=[2, 1369852], edge_attr=[1369852], y=[1171])
        super().__init__(dataset)
    def __getitem__(self, idx):
        data_sample = self.dataset[idx]
        data_sample = self.dataset[idx] 
        x_real = data_sample.x
        res = super().is_input_correct_shaped(x_real) # check if the input is correct shape, since some samples are wrong
        if not res:
            #print(data_sample)
            x_real = super().adjust_input_shape(x_real) # in case reshape the tensor appending the last timestamp features
        
        # x = x.view(self.n_nodes, self.n_features, self.seq_length).permute(0, 2, 1) # [num_nodes, seq_length, num_features]
        # x = x.unsqueeze(0)  # batch size 1
        # x = x.permute(0, 3, 1, 2) #(batch_size, num_features, num_nodes, sequence_length)             # rechanged the size 15/07/2025   
        y = data_sample.y.long()  # Ensure y is long for classification
        x = x_real.view(self.n_nodes, self.n_features, self.seq_length).permute(1, 0, 2)        # rechanged the size 25/07/2025
        
    
    
        return x, y
        


class GraphWaveNetRunner(BaseModelRunner):
    def __init__(self, model, device, market_name):
        super().__init__(model, device, market_name)
        self.model_name = "GWNet"

    def train(self, train_dataset, val_dataset, optimizer, criterion, num_epochs, seq_length, num_features):
        writer = SummaryWriter('runs/')
        train_set = GraphWaveNetDataset(train_dataset)
        val_set = GraphWaveNetDataset(val_dataset)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1)

        self.model.train()
        for epoch in range(num_epochs):
            for data in train_loader:
                
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()

                output = self.model(x) 
                output_for_loss = output[:, :, :, -1] 
                predict = output_for_loss.squeeze()
                real = y.float().squeeze() # This is (batch_size, num_nodes)
                loss = criterion(predict, real)
                writer.add_scalar(f'{self.model_name}/{self.market_name}/loss_train', loss.item(), epoch)
                loss.backward()
                optimizer.step()


            if epoch % 5 == 0 and val_loader is not None:
                self.model.eval()
                total_val_loss = 0.0
                all_preds = []
                all_targets = []

                with torch.no_grad():
                    for val_data in val_loader:
                        x, y = data
                        x = x.to(self.device)
                        y = y.to(self.device)
                        optimizer.zero_grad()

                        output = self.model(x) 
                        output_for_loss = output[:, :, :, -1] 
                        predict = output_for_loss.squeeze()
                        real = y.float().squeeze() # This is (batch_size, num_nodes)
                        loss = criterion(predict, real)
                        
                        total_val_loss += loss.item()

                        preds = (torch.sigmoid(predict) > 0.5).long()

                        all_preds.append(preds.cpu())
                        all_targets.append(real.cpu())

                writer.add_scalar(f'{self.model_name}/{self.market_name}/loss_val', total_val_loss, epoch)
                
                # After all batches
                if all_preds:
                    y_true = torch.cat(all_targets)
                    y_pred = torch.cat(all_preds)

                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='weighted')
                    mcc = matthews_corrcoef(y_true, y_pred)
                    avg_val_loss = total_val_loss / len(all_preds)
                    rec = recall_score(y_true, y_pred)
                    
                    writer.add_scalar(f'{self.model_name}/{self.market_name}/acc_val', acc, epoch)
                    writer.add_scalar(f'{self.model_name}/{self.market_name}/prec_val', prec, epoch)
                    writer.add_scalar(f'{self.model_name}/{self.market_name}/rec_val', rec, epoch)
                    writer.add_scalar(f'{self.model_name}/{self.market_name}/mcc_val', mcc, epoch)

                    print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_loss:.4f} - Acc: {acc:.4f} - Rec: {rec:.4f} - F1: {f1:.4f} - MCC: {mcc:.4f}")
  
        writer.close()

    def test(self, test_dataset, seq_length, num_features):
        test_set = GraphWaveNetDataset(test_dataset)
        test_loader = DataLoader(test_set, batch_size=1)
        self.model.eval()
        all_preds_logits = [] # Renamed for clarity: these are raw logits
        all_targets = []    
        with torch.no_grad():
            for data in test_loader:
                if data.x.shape[-1] != seq_length * num_features:
                    print(f"Warning: Skipping sample with incorrect shape: {data.x.shape}")
                    continue
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x) 
                output_for_loss = output[:, :, :, -1] 
                predict = output_for_loss.squeeze()
                real = y.float().squeeze() # This is (batch_size, num_nodes)
                preds = (torch.sigmoid(predict) > 0.5).long()
                all_preds_logits.append(preds.cpu())           
                all_targets.append(real.cpu())

        # Concatenate all raw logit tensors into one
        concatenated_logits = torch.cat(all_preds_logits, dim=0) 

        # Apply sigmoid and thresholding to the single tensor
        y_pred = (torch.sigmoid(concatenated_logits) > .5).int()

        # Concatenate true values (this line was already correct)
        y_true = torch.cat(all_targets)

        # Convert to numpy arrays for compatibility with sklearn metrics if used outside
        return y_pred.numpy(), y_true.numpy()