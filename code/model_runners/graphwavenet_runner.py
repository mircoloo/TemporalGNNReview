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
        
    def log_experiment(self, config, dataset_info, train_params, test_metrics, log_dir='logs'):
        """
        Log experiment configuration and results
        
        Args:
            config (dict): Model configuration parameters
            dataset_info (dict): Dataset information
            train_params (dict): Training hyperparameters
            test_metrics (dict): Test results metrics
            log_dir (str): Directory to save logs
        """
        import json
        from pathlib import Path
        from datetime import datetime
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_dir) / self.market_name / self.model_name
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare experiment info
        experiment_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'model': {
                'name': self.model_name,
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'config': config
            },
            'dataset': dataset_info,
            'training': train_params,
            'test_results': test_metrics,
            'device': str(self.device)
        }
        
        # Save to JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.market_name}_{self.model_name}_{timestamp}.json"
        with open(log_path / filename, 'w') as f:
            json.dump(experiment_info, indent=4, default=str)
            
        print(f"📄 Experiment logged to: {log_path / filename}")
        
        # Also save a summary to a CSV for easy comparison
        import pandas as pd
        summary_file = log_path / 'experiments_summary.csv'
        summary = {
            'timestamp': [experiment_info['timestamp']],
            'market': [self.market_name],
            'model': [self.model_name],
            'num_parameters': [experiment_info['model']['num_parameters']],
            'epochs': [train_params['num_epochs']],
            'batch_size': [train_params['batch_size']],
            'learning_rate': [train_params['learning_rate']],
            'test_accuracy': [test_metrics['accuracy']],
            'test_f1': [test_metrics['f1']],
            'test_precision': [test_metrics['precision']],
            'test_recall': [test_metrics['recall']],
            'test_mcc': [test_metrics['mcc']]
        }
        
        df = pd.DataFrame(summary)
        if summary_file.exists():
            df_existing = pd.read_csv(summary_file)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(summary_file, index=False)
        
    def train(self, train_dataset, val_dataset, optimizer, criterion, num_epochs, seq_length, num_features, batch_size=32):
        # Create organized TensorBoard writer
        writer = SummaryWriter(f'runs/{self.market_name}/{self.model_name}')
        train_set = GraphWaveNetDataset(train_dataset)
        val_set = GraphWaveNetDataset(val_dataset)
        
        # Use actual batching
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            n_train = 0

            for batch in train_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with batched inputs
                output = self.model(x)  # [B, num_features, num_nodes, seq_length]
                output_for_loss = output[:, :, :, -1]  # Take last timestep
                predict = output_for_loss.squeeze(1)  # Remove feature dimension
                real = y.float()  # [B, num_nodes]
                
                loss = criterion(predict, real)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_train += 1

            # Training metrics
            avg_train_loss = train_loss / n_train
            writer.add_scalar('Training/Loss', avg_train_loss, epoch)

            # Validation every 5 epochs
            if epoch % 5 == 0:
                self.model.eval()
                val_metrics = {
                    'loss': 0.0,
                    'acc': 0.0,
                    'prec': 0.0,
                    'rec': 0.0,
                    'f1': 0.0,
                    'mcc': 0.0
                }
                n_val = 0
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for batch in val_loader:
                        x_val, y_val = batch
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)

                        output = self.model(x_val)
                        output_for_loss = output[:, :, :, -1]
                        predict = output_for_loss.squeeze(1)
                        real = y_val.float()

                        loss = criterion(predict, real)
                        val_metrics['loss'] += loss.item()

                        # Store predictions and targets
                        preds = (torch.sigmoid(predict) > 0.5).int()
                        val_preds.append(preds.cpu())
                        val_targets.append(real.cpu())
                        n_val += 1

                # Process all predictions
                if n_val > 0:
                    y_pred = torch.cat(val_preds, dim=0)
                    y_true = torch.cat(val_targets, dim=0)

                    # Calculate metrics
                    val_metrics['acc'] = accuracy_score(y_true.flatten(), y_pred.flatten())
                    val_metrics['prec'] = precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
                    val_metrics['rec'] = recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
                    val_metrics['f1'] = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
                    val_metrics['mcc'] = matthews_corrcoef(y_true.flatten(), y_pred.flatten())
                    val_metrics['loss'] /= n_val

                    # Log metrics in organized sections
                    writer.add_scalar('Validation/Loss', val_metrics['loss'], epoch)
                    writer.add_scalar('Validation/Accuracy', val_metrics['acc'], epoch)
                    writer.add_scalar('Validation/Precision', val_metrics['prec'], epoch)
                    writer.add_scalar('Validation/Recall', val_metrics['rec'], epoch)
                    writer.add_scalar('Validation/F1', val_metrics['f1'], epoch)
                    writer.add_scalar('Validation/MCC', val_metrics['mcc'], epoch)

                    # Add histogram of predictions
                    writer.add_histogram('Predictions/Validation', y_pred, epoch)

                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} - "
                          f"Val Loss: {val_metrics['loss']:.4f} - "
                          f"Acc: {val_metrics['acc']:.4f} - "
                          f"Prec: {val_metrics['prec']:.4f} - "
                          f"Rec: {val_metrics['rec']:.4f} - "
                          f"F1: {val_metrics['f1']:.4f} - "
                          f"MCC: {val_metrics['mcc']:.4f}")

        writer.close()

    def test(self, test_dataset, seq_length, num_features, batch_size=32, config=None):
        test_set = GraphWaveNetDataset(test_dataset)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                output_for_loss = output[:, :, :, -1]
                predict = output_for_loss.squeeze(1)
                
                preds = (torch.sigmoid(predict) > 0.5).int()
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        # Concatenate all predictions and targets
        y_pred = torch.cat(all_preds, dim=0).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()
        
        # Calculate metrics
        test_metrics = {
            'accuracy': accuracy_score(y_true.flatten(), y_pred.flatten()),
            'precision': precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
            'recall': recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
            'f1': f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
            'mcc': matthews_corrcoef(y_true.flatten(), y_pred.flatten())
        }
        
        # Log experiment if config is provided
        if config is not None:
            dataset_info = {
                'seq_length': seq_length,
                'num_features': num_features,
                'num_samples': len(test_dataset),
                'num_nodes': test_dataset[0].x.shape[0]
            }
            
            train_params = {
                'batch_size': batch_size,
                'num_epochs': config.get('num_epochs', None),
                'learning_rate': config.get('learning_rate', None),
                'optimizer': config.get('optimizer', 'Adam')
            }
            
            self.log_experiment(
                config=config,
                dataset_info=dataset_info,
                train_params=train_params,
                test_metrics=test_metrics
            )

        return y_pred, y_true