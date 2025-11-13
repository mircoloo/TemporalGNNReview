from model_runners.base_runner import evaluate_decorator, BaseModelRunner
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import numpy as np
from tqdm.auto import tqdm
from tabulate import tabulate
from pathlib import Path
from model_runners.models_dataset import DTMLDataset
import pandas as pd

class DTMLRunner(BaseModelRunner):

    def __init__(self, model, device, market_name, feature_normalization = 'none', dates: list[list[str]] = None, window_size: int = None, market: str = ''):
        super().__init__(model, device, market_name)
        self.model_name = 'DTML'
        self.feature_normalization = feature_normalization
        self.dates = dates
        self.window_size = window_size
        self.market = market.upper()
        self.norm_values = {'minmax': None, 'zscore': None} #minmax [min, max], zscore [mean, std]
        

    def train(self, train_dataset, 
              val_dataset, 
              optimizer, 
              criterion, 
              num_epochs,
              n_features,
              batch_size=1, 
              use_validation=True,
              early_stopping_patience=10,
              early_stopping_metric='loss',
              early_stopping_min_delta=1e-4):
        """
        Train the DTML model with optional early stopping.
        
        Args:
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            early_stopping_metric: Metric to monitor ('loss', 'acc', 'f1', 'mcc')
            early_stopping_min_delta: Minimum change to qualify as an improvement
        """
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_indexes = self.load_index(self.dates[0], is_train=True)

        self.val_indexes = self.load_index(self.dates[1])
        self.test_indexes = self.load_index(self.dates[2])

        # Early stopping setup
        best_val_metric = float('inf') if early_stopping_metric == 'loss' else float('-inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = 0

        train_set = DTMLDataset(train_dataset, self.train_indexes, self.window_size)
        val_set = DTMLDataset(val_dataset, self.val_indexes, self.window_size)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            n_train = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [T]"):
                if batch is None:
                    continue
                
                x, index_x, y = batch
                x = x.to(self.device).float()
                index_x = index_x.to(self.device).float()
                y = y.to(self.device)

                optimizer.zero_grad()

                x = x.squeeze(0)  # Remove batch dimension if batch_size=1
                index_x = index_x.squeeze(0)  # Remove batch dimension if batch_size=1

                outputs = self.model(x, index_x)['output'].permute(1,0)
                loss = self.criterion(outputs.float(), y.float())

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_train += 1

            avg_train_loss = train_loss / n_train if n_train > 0 else 0.0
                
            if use_validation and ( epoch % 5 == 0 or epoch == num_epochs - 1):
                self.model.eval()
                val_metrics = {
                    'loss': 0.0, 'acc': 0.0, 'prec': 0.0, 
                    'f1': 0.0, 'mcc': 0.0, 'rec': 0.0
                }
                n_val = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [V]"):
                        if batch is None:
                            continue
                        
                        
                        x, index_x, targets = batch
                        x = x.to(self.device).float()
                        index_x = index_x.to(self.device).float()
                        targets = y.to(self.device)


                        x = x.squeeze(0)  # Remove batch dimension if batch_size=1
                        index_x = index_x.squeeze(0)  # Remove batch dimension if batch_size=1

                        outputs = self.model(x, index_x)['output'].permute(1,0)
                        loss = self.criterion(outputs.float(), targets.float())

                        val_metrics['loss'] += self.criterion(outputs.float(), targets.float()).item()
                        
                        preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                        targets_cpu = targets.int().cpu()
                        
                        
                        val_metrics['acc'] += accuracy_score(targets_cpu.flatten(), preds.flatten())
                        val_metrics['f1'] += f1_score(targets_cpu.flatten(), preds.flatten(), zero_division=0)
                        val_metrics['rec'] += recall_score(targets_cpu.flatten(), preds.flatten())
                        val_metrics['mcc'] += matthews_corrcoef(targets_cpu.flatten(), preds.flatten())
                        val_metrics['prec'] += precision_score(targets_cpu.flatten(), preds.flatten(), zero_division=0)
                        
                        n_val += 1

                if n_val > 0:
                    for k in val_metrics:
                        val_metrics[k] /= n_val
                    
                    # Early stopping logic
                    current_metric = val_metrics[early_stopping_metric]
                    
                    # Check if improvement occurred
                    if early_stopping_metric == 'loss':
                        improved = (best_val_metric - current_metric) > early_stopping_min_delta
                    else:
                        improved = (current_metric - best_val_metric) > early_stopping_min_delta
                    
                    if improved:
                        best_val_metric = current_metric
                        patience_counter = 0
                        best_epoch = epoch
                        # Save best model state
                        best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                        improvement_marker = " ✓ NEW BEST"
                    else:
                        patience_counter += 1
                        improvement_marker = ""
                    
                    headers = ["Metric", "Value"]
                    table_data = [
                        ["Train Loss", f"{avg_train_loss:.4f}"],
                        ["Val Loss", f"{val_metrics['loss']:.4f}"],
                        ["Accuracy", f"{val_metrics['acc']:.4f}"],
                        ["Precision", f"{val_metrics['prec']:.4f}"],
                        ["Recall", f"{val_metrics['rec']:.4f}"],
                        ["F1 Score", f"{val_metrics['f1']:.4f}"],
                        ["MCC", f"{val_metrics['mcc']:.4f}"],
                        ["---", "---"],
                        ["Best Epoch", f"{best_epoch}"],
                        ["Patience", f"{patience_counter}/{early_stopping_patience}"]
                    ]
                    
                    print(f"\nEpoch {epoch+1}/{num_epochs} Results:{improvement_marker}")
                    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
                    print("\n")
                    
                    # Check early stopping condition
                    if patience_counter >= early_stopping_patience:
                        print(f"\n{'='*60}")
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        print(f"Best {early_stopping_metric}: {best_val_metric:.4f} at epoch {best_epoch+1}")
                        print(f"{'='*60}\n")
                        
                        # Restore best model
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                            print("✓ Best model weights restored")
                        break

    @evaluate_decorator
    def test(self, test_dataset, batch_size=1):
        test_set = DTMLDataset(test_dataset, self.test_indexes, self.window_size)
        test_loader = DataLoader(test_set, batch_size=1)
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                if batch is None:
                    continue
                    
                x, index_x, targets = batch
                x = x.to(self.device).float()
                index_x = index_x.to(self.device).float()
                targets = targets.to(self.device)

                x = x.squeeze(0)  # Remove batch dimension if batch_size=1
                index_x = index_x.squeeze(0)  # Remove batch dimension if batch_size=1



                outputs = self.model(x, index_x)['output'].permute(1,0)
                loss = self.criterion(outputs.float(), targets.float())
                preds = (torch.round(torch.sigmoid(outputs))).int().cpu()
                
                all_preds.extend(preds.flatten().tolist())
                all_labels.extend(targets.flatten().tolist())

        return {'preds': np.array(all_preds), 'targets': np.array(all_labels)}
    
    
    def load_index(self, dates: list[str], is_train: bool = False) -> torch.tensor:
        market_to_path = {
            'NASDAQ': '~/tests/TemporalGNNReview/code/data/datasets/hist_prices/NASDAQ/NASDAQ_NDX_index.csv',
            'SSE': '~/tests/TemporalGNNReview/code/data/datasets/hist_prices/SSE/SSE_000001SS_index.csv',
            'NYSE': '~/tests/TemporalGNNReview/code/data/datasets/hist_prices/NYSE/NYSE_NYA_index.csv'
        }
        # Load from csv
        df: pd.DataFrame = pd.read_csv(Path(market_to_path[self.market]), index_col=0, parse_dates=True)
        # Get the right range
        df = df[dates[0]:dates[1]] 
        # Handle missing values
        df = df.bfill().ffill()
        
        #If is train set, load normalization values for other datasets
        if is_train:
            self.load_norm_values(df)

        # Transform to numpy 
        df_np = df.to_numpy()
        # Normalize
        df_np_normalized = self.normalize_features(df_np)
        # Transform to tensor
        return torch.from_numpy(df_np_normalized)
    
    
    def normalize_features(self, x: torch.tensor) -> torch.tensor:
        if self.feature_normalization == 'minmax':
            min_vals, max_vals = self.norm_values['minmax']
            x = (x - min_vals) / (max_vals - min_vals + 1e-8)
        elif self.feature_normalization == 'zscore':
            mean_vals, std_vals = self.norm_values['zscore']
            x = (x - mean_vals) / (std_vals + 1e-8)
        return x
    

    def load_norm_values(self, df: pd.DataFrame):
        if self.feature_normalization == 'minmax':
            self.norm_values['minmax'] = [df.min().to_numpy(), df.max().to_numpy()]
        elif self.feature_normalization == 'zscore':
            self.norm_values['zscore'] = [df.mean().to_numpy(), df.std().to_numpy()]
            
            