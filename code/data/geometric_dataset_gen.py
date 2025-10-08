import torch
import math
import csv
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import List, Tuple, Dict
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from scipy.linalg import expm
from pathlib import Path


class MyDataset(Dataset):
    def __init__(self, root: str, 
                 desti: str, 
                 market: str, 
                 comlist: List[str], 
                 start: str, 
                 end: str, 
                 window: int, 
                 dataset_type: str, 
                 fast_approx, 
                 normalize_method: str = '',
                train_dates: List[str] = '',
                minmax_normalize_adj: bool = True):
        
        super().__init__()
        
        # Add a clear print statement at initialization
        print(f"\n{'='*80}")
        print(f"Creating {market} {dataset_type.upper()} dataset ({start} to {end})")
        print(f"Normalization method: {normalize_method if normalize_method else 'None'}")
        print(f"Window size: {window} days")
        print(f"{'='*80}\n")
        
        self.market = market
        self.root = Path(root)
        self.desti = Path(desti)
        self.start = start
        self.end = end
        self.window = window
        self.normalize_method = normalize_method
        self.global_data_search_cutoff = '2025-07-05'


        #params for minmax normalizing adj matrix
        self.minmax_normalize_adj = minmax_normalize_adj
        self.adj_min = torch.inf
        self.adj_max = -torch.inf

        


        self.dataset_type = dataset_type #(train, test, val)
        self.fast_approx = fast_approx
        

        # 1 step find the common dates and valid companies
        self.company_list, self.dates, self.next_day = self.find_dates(start, end, comlist, self.global_data_search_cutoff)


        if not self.dates or len(self.dates) < self.window + 1:
            print(f"Insufficient common dates ({len(self.dates)}) found for a window of size {self.window}. Dataset will be empty.")
            self.company_list = []

        if not self.company_list:
            print(f"No valid companies with data found. {self.dataset_type} Dataset will be empty.")
            self.dates = []

        
        # 2 Create directory path to store the snapshot graphs
        self.directory_path = self.desti / f'{market}_{dataset_type}_{start}_{end}_{window}{f"_{self.normalize_method}" if self.normalize_method else ""}'
        
        # 3 Define path for index to ticker mapping
        self.index_mapping_path = self.directory_path / 'ticker_index_mapping.json'
        
        # 4 Define path for normalization parameters
        self.norm_params = {}
        # 5 Find train dataset's normalization parameters path
        self.train_dir = self.desti / f'{market}_Train_{train_dates[0]}_{train_dates[1]}_{window}{f"_{self.normalize_method}" if self.normalize_method else ""}'
        self.norm_params_path = self.train_dir / 'norm_params.pt'
        self.adj_minmax_path = self.train_dir / 'adj_minmax.pt'


        # This will be False only one snapshot graph file or mapping file is missing
        graph_files_exist = all((self.directory_path / f'graph_{i}.pt').exists() for i in range(len(self.dates) - window + 1))
        mapping_file_exists = self.index_mapping_path.exists()

        
        if self.minmax_normalize_adj:
            if self.train_dir.exists() and (self.adj_minmax_path).exists():
                print("Loading Min-Max normalizing adjacency matrix")
                adj_data = torch.load(self.adj_minmax_path)
                self.adj_min = adj_data['adj_min']
                self.adj_max = adj_data['adj_max']

        # 6 Create graphs only if they don't already exist
        if not graph_files_exist or not mapping_file_exists:
            if self.dates and len(self.dates) >= self.window + 1 and self.company_list:
                # For normalization that requires statistics
                if self.normalize_method in ['zscore', 'minmax', 'robust', 'maxabs']:
                    # For training set, compute and save parameters
                    if self.dataset_type.lower() == 'train':
                        # Create norm_params directory if it doesn't exist
                        self.train_dir.mkdir(parents=True, exist_ok=True)
                        # Compute stock-level normalization parameters
                        self._compute_stock_norm_params()

                        
                    # For validation and test sets, check if parameters exist
                    elif self.dataset_type.lower() in ['val', 'test']:
                        if not self.norm_params_path.exists():
                            print(f"WARNING: Normalization parameters file not found at {self.norm_params_path}")
                            print(f"This may lead to data leakage. Train dataset should be created first!")
                        else:
                            # Load normalization parameters
                            print(f"Loading normalization parameters from {self.norm_params_path}")
                            self._load_norm_params()

                # 6 Create snapshot graphs
                self._create_graphs()
            else:
                print("Skipping graph creation due to insufficient common dates or no valid companies.")

    # Helper method for filepath construction
    def _get_ticker_filepath(self, ticker: str) -> Path:
        """Constructs the full path for a company's ticker CSV file."""
        filename = f'{self.market}_{ticker}.csv'
        return self.root / filename

    def __len__(self):
        if not self.dates or len(self.dates) < self.window + 1 or not self.company_list:
            return 0
        return len(self.dates) - self.window + 1

    def __getitem__(self, idx: int):
        data_path = self.directory_path / f'graph_{idx}.pt'
        if data_path.exists():
            sample = torch.load(data_path, weights_only=False)
            if self.minmax_normalize_adj and self.adj_min != torch.inf and self.adj_max != -torch.inf:
                # Normalize edge_attr using stored min and max
                threshold = .004 #to change
                sample.edge_attr = (sample.edge_attr - self.adj_min) / (self.adj_max - self.adj_min + 1e-9)
                sample.edge_attr = torch.clamp(sample.edge_attr, 0, 1)  # Ensure values are within [0, 1]
                sample.edge_attr[sample.edge_attr < threshold] = 0  # Thresholding to ensure no very small entries
            return sample
        else:
            raise FileNotFoundError(f"No graph data found for index {idx}")
    
    # Get ticker from node index
    def get_ticker_from_index(self, node_idx: int) -> str:
        """Returns the ticker symbol corresponding to a node index."""
        if self.index_mapping_path.exists():
            with open(self.index_mapping_path, 'r') as f:
                mapping = json.load(f)
                # Convert keys from strings back to integers
                mapping = {int(k): v for k, v in mapping.items()}
                return mapping.get(node_idx, "Unknown")
        else:
            raise FileNotFoundError(f"Ticker index mapping file not found at {self.index_mapping_path}")

    # Get node index from ticker
    def get_index_from_ticker(self, ticker: str) -> int:
        """Returns the node index corresponding to a ticker symbol."""
        if self.index_mapping_path.exists():
            with open(self.index_mapping_path, 'r') as f:
                mapping = json.load(f)
                # Convert keys from strings back to integers
                mapping = {int(k): v for k, v in mapping.items()}
                # Create reverse mapping
                reverse_mapping = {v: k for k, v in mapping.items()}
                return reverse_mapping.get(ticker, -1)
        else:
            raise FileNotFoundError(f"Ticker index mapping file not found at {self.index_mapping_path}")

    def check_years(self, date_str: str, start_str: str, end_str: str) -> bool:
        date_format = "%Y-%m-%d"
        try:
            date = datetime.strptime(date_str, date_format)
            start = datetime.strptime(start_str, date_format)
            end = datetime.strptime(end_str, date_format)
            return start <= date <= end
        except ValueError:
            return False

    def find_dates(self, start: str, end: str, initial_comlist: List[str], search_until_date: str) -> Tuple[List[str], List[str], str]:
        print(f"Searching for common dates between {start} and {end} for {len(initial_comlist)} companies...")
        
        filtered_comlist: list[str] = []
        not_inserted_companies: list[str] = []
        for h in initial_comlist:
            d_path = self._get_ticker_filepath(h)
            if d_path.is_file():
                filtered_comlist.append(h)
            else:
                print(f"Stock {h} data file not found at {d_path}. **Excluding this stock from analysis.**")
                not_inserted_companies.append(h)
        if not filtered_comlist:
            print("No valid stock data files found after initial filtering. Returning empty lists.")
            return [], [], None

        date_sets_valid = []
        after_end_date_sets_valid = []
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        search_until_dt = datetime.strptime(search_until_date, "%Y-%m-%d")

        for company_ticker in filtered_comlist:
            dates: set = set()
            after_end_dates: set = set()
            d_path = self._get_ticker_filepath(company_ticker)
            with open(d_path, 'r') as f:
                file = csv.reader(f)
                next(file, None)
                for line in file:
                    if not line: continue
                    try:
                        current_date_dt = datetime.strptime(line[0][:10], "%Y-%m-%d")
                        if start_dt <= current_date_dt <= end_dt:
                            dates.add(line[0][:10])
                        elif end_dt < current_date_dt <= search_until_dt:
                            after_end_dates.add(line[0][:10])
                    except (ValueError, IndexError):
                        continue
            if not dates:
                print(f"Warning: No valid dates found for {company_ticker} in the specified range.")
                not_inserted_companies.append(company_ticker)
                continue
            
            if dates: date_sets_valid.append(dates)
            if after_end_dates: after_end_date_sets_valid.append(after_end_dates)

        if not date_sets_valid:
            return filtered_comlist, [], None

        all_dates = sorted(list(set.intersection(*date_sets_valid))) if date_sets_valid else []
        all_after_end_dates = sorted(list(set.intersection(*after_end_date_sets_valid))) if after_end_date_sets_valid else []
        
        if not all_dates:
            return filtered_comlist, [], None
        
        next_common_day = all_after_end_dates[0] if all_after_end_dates else None

        print(f"Found {len(filtered_comlist)} valid companies with {len(all_dates)} common dates")
        if next_common_day:
            print(f"Next common day after end date: {next_common_day}")
        else:
            print(f"Warning: No common next_day found after {end}. This will prevent graph generation.")
            
        return filtered_comlist, all_dates, next_common_day

    # Method to load stock data for all dates
    def _load_stock_data(self, all_dates=None):
        """Load stock data for all companies across all dates."""
        if all_dates is None:
            # Use the full date range from start to end
            start_dt = datetime.strptime(self.start, "%Y-%m-%d")
            end_dt = datetime.strptime(self.end, "%Y-%m-%d")
            all_dates_dt = pd.date_range(start=start_dt, end=end_dt)
            all_dates = [dt.strftime("%Y-%m-%d") for dt in all_dates_dt]
        else:
            all_dates_dt = pd.to_datetime(all_dates)
        
        # Dictionary to store dataframes for each stock
        stock_data = {}
        
        
        for ticker in tqdm(self.company_list):
            d_path = self._get_ticker_filepath(ticker)
            df = pd.read_csv(d_path, parse_dates=[0], index_col=0)
            
            # Ensure the DataFrame index is just the date part for clean matching
            df.index = pd.to_datetime(df.index.date)
            
            # Reindex to match all dates in the range
            df_reindexed = df.reindex(all_dates_dt, fill_value=0)
            
            # Select only the first 5 features
            df_features = df_reindexed.iloc[:, :5]
            
            stock_data[ticker] = df_features
        
        return stock_data, all_dates_dt

    # Method to compute stock-level normalization parameters
    def _compute_stock_norm_params(self):
        """Compute normalization parameters for each stock's features across all dates."""
        if self.normalize_method not in ['zscore', 'minmax', 'robust', 'maxabs']:
            print(f"Skipping normalization parameter computation (method: {self.normalize_method})")
            return
            
        print(f"\n>>> Computing stock-level {self.normalize_method} normalization parameters for {self.dataset_type} dataset...")
        
        # Load all stock data
        stock_data, _ = self._load_stock_data()
        
        # Dictionary to store normalization parameters for each feature of each stock
        print(f"Computing normalization parameters for {len(stock_data)} stocks...")
        norm_params = {
            'stock_params': {},
            'feature_names': ['Open', 'High', 'Low', 'Close', 'Volume']  # Assuming these are the 5 features
        }
        

        for ticker, df in tqdm(stock_data.items()):
            # Initialize parameters for this stock
            stock_params = {}
            
            # Compute parameters for each feature column
            for col_idx, col_name in enumerate(df.columns[:5]):  # First 5 columns only
                series = df[col_name]
                
                # Skip if all values are NaN
                if series.isna().all():
                    print(f"Warning: All values are NaN for {ticker} feature {col_name}. Skipping this feature.")
                    continue
                    
                # Compute parameters based on normalization method
                if self.normalize_method == 'zscore':
                    mean = series.mean()
                    std = series.std()
                    if std < 1e-5:  # Handle zero std case
                        std = 1.0
                    stock_params[str(col_idx)] = {'mean': mean, 'std': std}
                    
                elif self.normalize_method == 'minmax':
                    min_val = series.min()
                    max_val = series.max()
                    if abs(max_val - min_val) < 1e-5:  # Handle equal min and max
                        min_val = min_val - 1.0
                    stock_params[str(col_idx)] = {'min': min_val, 'max': max_val}
                    
                elif self.normalize_method == 'robust':
                    q25 = series.quantile(0.25)
                    q75 = series.quantile(0.75)
                    iqr = q75 - q25
                    if iqr < 1e-5:  # Handle zero IQR
                        iqr = 1.0
                    stock_params[str(col_idx)] = {'q25': q25, 'q75': q75, 'iqr': iqr}
                    
                elif self.normalize_method == 'maxabs':
                    maxabs = series.abs().max()
                    if maxabs < 1e-5:  # Handle zero maxabs
                        maxabs = 1.0
                    stock_params[str(col_idx)] = {'maxabs': maxabs}
            
            # Store parameters for this stock
            norm_params['stock_params'][ticker] = stock_params
        
        # Save normalization parameters
        self.norm_params = norm_params
        torch.save(norm_params, self.norm_params_path)
        
        num_stocks = len(norm_params['stock_params'])
        print(f">>> Computed {self.normalize_method} parameters for {num_stocks} stocks")
        print(f">>> Saved to: {self.norm_params_path}")

    def _load_adj_minmax(self, path: Path):
        if path.exists():
            adj_params = torch.load(path, weights_only=False)
            self.adj_min = adj_params.get('adj_min', torch.inf)
            self.adj_max = adj_params.get('adj_max', -torch.inf)
            print(f"Loaded adjacency matrix min: {self.adj_min}, max: {self.adj_max} from {path}")
        else:
            print(f"Adjacency min-max file not found at {path}. Using default values.")

    # Method to load normalization parameters
    def _load_norm_params(self):
        """Load normalization parameters from file."""
        if self.norm_params_path.exists():
            self.norm_params = torch.load(self.norm_params_path, weights_only=False)
            num_stocks = len(self.norm_params['stock_params']) if 'stock_params' in self.norm_params else 0
            return True
        else:
            if self.dataset_type != 'train':
                print(f"\n!!! WARNING: No normalization parameters found at {self.norm_params_path}")
                print(f"!!! {self.dataset_type.upper()} dataset will compute its own statistics, which is incorrect for proper evaluation!")
            return False
    
    # Create index to ticker mapping JSON
    def _create_ticker_mapping(self):
        """Creates a JSON file mapping node indices to ticker symbols."""
        if not self.company_list:
            print("No valid companies to create ticker mapping.")
            return
        
        # Create mapping dictionary {index: ticker}
        mapping = {idx: ticker for idx, ticker in enumerate(self.company_list)}
        
        # Ensure directory exists
        self.directory_path.mkdir(parents=True, exist_ok=True)
        
        # Save mapping to JSON file
        with open(self.index_mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"Ticker to index mapping created and saved to {self.index_mapping_path}")

    

    def _create_graphs(self):
        # 6 Create snapshot graphs
        if len(self.dates) < self.window + 1 or not self.company_list:
            print(f"Skipping graph generation due to insufficient data.")
            return

        dates_with_next_day = self.dates + ([self.next_day] if self.next_day else [])
        if len(dates_with_next_day) < self.window + 1:
            print("Not enough dates to create even one graph.")
            return

        self.directory_path.mkdir(parents=True, exist_ok=True)

        # 7 Create ticker mapping JSON before generating graphs
        self._create_ticker_mapping()
        
        # For validation and test sets, load normalization parameters if they exist
        
        if self.dataset_type.lower() in ['validation', 'test'] and self.normalize_method in ['zscore', 'minmax', 'robust', 'maxabs']:
            loaded = self._load_norm_params()
            if loaded:
                print(f">>> Using training set normalization parameters for {self.dataset_type} dataset")
                
        
        print(f"\n>>> Creating {len(self.dates) - self.window + 1} graphs for {self.dataset_type} dataset")
        print(f">>> Using {self.normalize_method if self.normalize_method else 'no'} normalization")
        
        # 8 For each possible time window, create a graph snapshot
        for i in tqdm(range(len(self.dates) - self.window + 1)):  # w=3 -> 1,2,3,4,5 -> 1,2,3, ... , len(dates)-w+1=3
            filename = self.directory_path / f'graph_{i}.pt'
            if filename.exists():
                continue
                
            # 9 Get the date window (including next day for target)
            box = self.dates[i : i + self.window + 1] 
            
            # 10 Create and normalize feature matrix
            X = self.create_feature_node_timestamp_matrix(box) # [feature, node, timestep]



            if X.shape[1] == 0: # X.shape[1] is the number of nodes
                print(f"Skipping graph {i}: No nodes.")
                continue

            # Target C is based on the 'Close' price (row index 0)
            C = torch.zeros(X.shape[1]) # A vector of size [num_nodes]

            # X is [feature, node, timestep] - for each node, if Close[today] - Close[yesterday] > 0, C[node] = 1 else 0
            for j in range(C.shape[0]):
                if X[0, j, -1] - X[0, j, -2] > 0:
                    C[j] = 1

            # Remove the last timestep (prediction one)
            X_features = X[:, :, :-1]
            
            if X_features.nelement() == 0:
                print(f"Skipping graph {i}: Feature matrix is empty.")
                continue
                
            # from [feature, node, timestep] to [node, feature, timestep] -> [node, feature * timestamps]
            X_final = X_features.permute(1, 0, 2).reshape(X_features.shape[1], -1)
            
            # Handle any remaining NaN values
            X_final = torch.nan_to_num(X_final, 0)

            try:
                # 11 Create adjacency matrix and convert to edge_index and edge_attr
                A = self.adjacency_matrix(X_final)
                edge_index, edge_attr = dense_to_sparse(A) # [N;F*T] -> [N;N] -> edge_index [2;E], edge_attr [E]
            except Exception as e:
                print(f"Skipping graph {i} due to adjacency matrix error: {e}")
                continue

            data = Data(x=X_final, edge_index=edge_index, edge_attr=edge_attr, y=C)
            torch.save(data, filename)
        
        print(f"\n>>> Finished creating {self.dataset_type.upper()} dataset with {len(self.dates) - self.window + 1} graphs")
        print(f">>> Saved to: {self.directory_path}")

        print("Calculating Min-Max normalizing adjacency matrix")
        for i in tqdm(range(len(self.dates) - self.window + 1)):
            edges = torch.load(self.directory_path / f'graph_{i}.pt', weights_only=False).edge_attr
            self.adj_min = min(self.adj_min, edges.min())
            self.adj_max = max(self.adj_max, edges.max())
        torch.save({'adj_min': self.adj_min, 'adj_max': self.adj_max}, self.train_dir / 'adj_minmax.pt')
        print(f"Adjacency matrix min: {self.adj_min}, max: {self.adj_max}")
    

    def create_feature_node_timestamp_matrix(self, dates: List[str]) -> torch.Tensor:
        """Create and normalize the node feature matrix for given dates (used with box)."""
        # Convert date strings to datetime objects for indexing
        dates_dt = pd.to_datetime(dates)
        
        # Initialize the feature tensor X with the correct dimensions
        # 5 features, number of companies, and number of time steps in the window [F;N;T]
        X = torch.zeros((5, len(self.company_list), len(dates_dt)))

        # 11 For each company in the company list
        for idx, ticker in enumerate(self.company_list):
            # Get the stock filpath (csv with historical data)
            d_path = self._get_ticker_filepath(ticker)
            df = pd.read_csv(d_path, parse_dates=[0], index_col=0)
            # Ensure the DataFrame index is just the date part for clean matching
            df.index = pd.to_datetime(df.index.date)

            # Reindex the DataFrame to match the exact dates of the window
            df_reindexed = df.reindex(dates_dt, fill_value=0) # [Total_days, Features]
            
            # Get the first 5 features
            features_df = df_reindexed.iloc[:, :5].astype(float)
            # Apply stock-level normalization if parameters exist
            if (self.normalize_method.lower() in ['zscore', 'minmax', 'robust', 'maxabs'] and 
                hasattr(self, 'norm_params') and 
                'stock_params' in self.norm_params and 
                ticker in self.norm_params['stock_params']):
                
                stock_params = self.norm_params['stock_params'][ticker]
                
                # Normalize each feature column independently
                for col_idx in range(5):  # Assuming 5 features
                    str_idx = str(col_idx)
                    if str_idx not in stock_params:
                        continue
                        
                    series = features_df.iloc[:, col_idx]
                    
                    # Apply normalization based on method
                    if self.normalize_method == 'zscore':
                        mean = stock_params[str_idx]['mean']
                        std = stock_params[str_idx]['std']
                        features_df.iloc[:, col_idx] = (series - mean) / std
                        
                    elif self.normalize_method == 'minmax':
                        min_val = stock_params[str_idx]['min']
                        max_val = stock_params[str_idx]['max']
                        features_df.iloc[:, col_idx] = (series - min_val) / (max_val - min_val)
                        
                    elif self.normalize_method == 'robust':
                        q25 = stock_params[str_idx]['q25']
                        iqr = stock_params[str_idx]['iqr']
                        features_df.iloc[:, col_idx] = (series - q25) / iqr
                        
                    elif self.normalize_method == 'maxabs':
                        maxabs = stock_params[str_idx]['maxabs']
                        features_df.iloc[:, col_idx] = series / maxabs
            
            # Apply log1p normalization (doesn't require pre-computed parameters)
            elif self.normalize_method == 'log1p':
                features_df = np.log1p(features_df)
            
            # Apply row normalization (doesn't require pre-computed parameters)
            elif self.normalize_method == 'row':
                # Normalize each row (date) independently
                row_sums = features_df.sum(axis=1)
                # Avoid division by zero
                row_sums[row_sums < 1e-5] = 1.0
                for col_idx in range(5):
                    features_df.iloc[:, col_idx] = features_df.iloc[:, col_idx].div(row_sums)
            
            # Handle NaN values
            features_df = features_df.fillna(0)
            
            # Transpose to get [features, timestamps]
            df_features = features_df.transpose()
            
            # Assign to tensor
            X[:, idx, :] = torch.from_numpy(df_features.to_numpy()) #[F;N;T] maybe save this to recover all the dataset
            
        return X

    
    def adjacency_matrix(self, X: torch.Tensor) -> torch.Tensor:
        num_nodes = X.shape[0]
        A = torch.zeros((num_nodes, num_nodes)) # [N;N]
        X_np = X.numpy() # convert into numpy array [N;F*T]
        energy = np.array([self.signal_energy(tuple(x)) for x in X_np]) # energy for each node x.shape is [F*T]
        entropy = np.array([self.information_entropy(tuple(x)) for x in X_np])

        for i in range(num_nodes): #
            for j in range(num_nodes):
                concat_x = np.concatenate((X_np[i], X_np[j])) # concatenate two nodes [F*T + F*T]
                A[i, j] = \
                torch.tensor(
                    (energy[i] / (energy[j] + 1e-9)) * (math.exp(entropy[i] + entropy[j] - self.information_entropy(tuple(concat_x)
                                                                                                                    ))),dtype=torch.float32)

        # if self.fast_approx:
        #     t = 5
        #     A_np = A.numpy()
        #     num_nodes = A_np.shape[0]
        #     A_tilde = A_np + np.eye(num_nodes)
        #     sum_A_tilde_rows = A_tilde.sum(axis=1)
        #     D_tilde_diag = np.where(sum_A_tilde_rows > 0, 1 / np.sqrt(sum_A_tilde_rows), 0)
        #     D_tilde = np.diag(D_tilde_diag)
        #     H = D_tilde @ A_np @ D_tilde
        #     return torch.from_numpy(expm(-t * (np.eye(num_nodes) - H))).float()
        A[A < 1] = 1 # Thresholding to ensure no zero entries
        return torch.log(A)

    def signal_energy(self, x_tuple: Tuple[float]) -> float:
        """Calculates the signal energy of a given tuple of floats."""
        x = np.array(x_tuple)
        return np.sum(np.square(x))
        
    def information_entropy(self, x_tuple: Tuple[float]) -> float:
        """Calculates the information entropy of a given tuple of floats."""
        x = np.array(x_tuple)
        unique, counts = np.unique(x, return_counts=True)
        total_counts = np.sum(counts)
        probabilities = counts / total_counts
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
        return entropy
    