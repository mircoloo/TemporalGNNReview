import torch
import math
import csv
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from scipy.linalg import expm
from pathlib import Path


class MyDataset(Dataset):
    def __init__(self, root: str, desti: str, market: str, comlist: List[str], start: str, end: str, window: int, dataset_type: str, fast_approx):
        super().__init__()
        self.market = market
        self.root = root
        self.desti = desti
        self.start = start
        self.end = end
        self.window = window

        self.global_data_search_cutoff = '2025-07-05'

        self.comlist, self.dates, self.next_day = self.find_dates(start, end, comlist, self.global_data_search_cutoff)

        self.dataset_type = dataset_type
        self.fast_approx = fast_approx

        if not self.dates or len(self.dates) < self.window + 1:
            print(f"Insufficient common dates ({len(self.dates)}) found for a window of size {self.window}. Dataset will be empty.")
            self.comlist = []

        if not self.comlist:
            print(f"No valid companies with data found. {self.dataset_type} Dataset will be empty.")
            self.dates = []

        graph_files_exist = all(os.path.exists(os.path.join(desti, f'{market}_{dataset_type}_{start}_{end}_{window}/graph_{i}.pt')) for i in range(len(self.dates) - window + 1))

        if not graph_files_exist:
            if self.dates and len(self.dates) >= self.window + 1 and self.comlist:
                self._create_graphs() # Removed arguments as they are instance attributes
            else:
                print("Skipping graph creation due to insufficient common dates or no valid companies.")

    # NEW HELPER METHOD: Centralizes filename creation for easier future changes.
    def _get_ticker_filepath(self, ticker: str) -> str:
        """Constructs the full path for a company's ticker CSV file."""
        filename = f'{self.market}_{ticker}.csv'
        return os.path.join(self.root, filename)

    def __len__(self):
        if not self.dates or len(self.dates) < self.window + 1 or not self.comlist:
            return 0
        return len(self.dates) - self.window + 1

    def __getitem__(self, idx: int):
        directory_path = os.path.join(self.desti, f'{self.market}_{self.dataset_type}_{self.start}_{self.end}_{self.window}')
        data_path = os.path.join(directory_path, f'graph_{idx}.pt')
        if os.path.exists(data_path):
            return torch.load(data_path, weights_only=False)
        else:
            raise FileNotFoundError(f"No graph data found for index {idx}")

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
        filtered_comlist: list[str] = []
        not_inserted_companies: list[str] = []
        for h in initial_comlist:
            # CHANGED: Using the new helper method to get the file path.
            d_path = self._get_ticker_filepath(h)
            if os.path.isfile(d_path):
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
            # CHANGED: Using the new helper method again.
            d_path: str = self._get_ticker_filepath(company_ticker)
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
                        else:
                            not_inserted_companies.append(company_ticker)
                    except (ValueError, IndexError):
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

        if not next_common_day:
            print(f"Warning: No common next_day found after {end}. This will prevent graph generation.")
            return filtered_comlist, all_dates, None


        return filtered_comlist, all_dates, next_common_day

    def signal_energy(self, x_tuple: Tuple[float]) -> float:
        x = np.array(x_tuple)
        return np.sum(np.square(x))

    def information_entropy(self, x_tuple: Tuple[float]) -> float:
        x = np.array(x_tuple)
        unique, counts = np.unique(x, return_counts=True)
        probabilities = counts / np.sum(counts)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
        return entropy

    def adjacency_matrix(self, X: torch.Tensor) -> torch.Tensor:
        A = torch.zeros((X.shape[0], X.shape[0]))
        X_np = X.numpy()
        energy = np.array([self.signal_energy(tuple(x)) for x in X_np])
        entropy = np.array([self.information_entropy(tuple(x)) for x in X_np])

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                concat_x = np.concatenate((X_np[i], X_np[j]))
                A[i, j] = torch.tensor((energy[i] / (energy[j] + 1e-9)) * (math.exp(entropy[i] + entropy[j] - self.information_entropy(tuple(concat_x)))), dtype=torch.float32)

        if self.fast_approx:
            t = 5
            A_np = A.numpy()
            num_nodes = A_np.shape[0]
            A_tilde = A_np + np.eye(num_nodes)
            sum_A_tilde_rows = A_tilde.sum(axis=1)
            D_tilde_diag = np.where(sum_A_tilde_rows > 0, 1 / np.sqrt(sum_A_tilde_rows), 0)
            D_tilde = np.diag(D_tilde_diag)
            H = D_tilde @ A_np @ D_tilde
            return torch.from_numpy(expm(-t * (np.eye(num_nodes) - H))).float()

        A[A < 1] = 1
        return torch.log(A)

    def node_feature_matrix(self, dates: List[str]) -> torch.Tensor:
        
        # Convert date strings to datetime objects for indexing
        dates_dt = pd.to_datetime(dates)
        
        # Initialize the feature tensor X with the correct dimensions
        # 5 features, number of companies, and number of time steps in the window
        X = torch.zeros((5, len(self.comlist), len(dates_dt)))

        for idx, h in enumerate(self.comlist):
            d_path = self._get_ticker_filepath(h)
            df = pd.read_csv(d_path, parse_dates=[0], index_col=0)
            
            # Ensure the DataFrame index is just the date part for clean matching
            df.index = pd.to_datetime(df.index.date)

            # Reindex the DataFrame to match the exact dates of the window.
            # This adds rows with `fill_value=0` for any missing dates.
            df_reindexed = df.reindex(dates_dt, fill_value=0)

            # --- THIS IS THE CORRECTED LOGIC ---
            # Select all rows (the full time window) and the first 5 columns (features).
            # .iloc[:, :5] gets all rows and columns from index 0 up to 4.
            # Then, transpose the result to get the desired shape [5_features, num_timesteps].
            df_features = df_reindexed.iloc[:, :5].transpose()
            
            # Assign the resulting tensor to the correct slice in X
            # The shape of df_features.to_numpy() is now [5, 20], which matches the target slice.
            X[:, idx, :] = torch.from_numpy(df_features.to_numpy())

        return X



    def _create_graphs(self):
        if len(self.dates) < self.window + 1 or not self.comlist:
            print(f"Skipping graph generation due to insufficient data.")
            return

        dates_with_next_day = self.dates + ([self.next_day] if self.next_day else [])
        if len(dates_with_next_day) < self.window + 1:
            print("Not enough dates to create even one graph.")
            return

        directory_path = os.path.join(self.desti, f'{self.market}_{self.dataset_type}_{self.start}_{self.end}_{self.window}')
        os.makedirs(directory_path, exist_ok=True)
        
        for i in tqdm(range(len(self.dates) - self.window + 1)):
            filename = os.path.join(directory_path, f'graph_{i}.pt')
            if os.path.exists(filename):
                continue
            # Time window dates + the prediction date
            box = self.dates[i : i + self.window + 1]
            
            #print(f"{box=}")

            X = self.node_feature_matrix(box)
            
            if X.shape[1] == 0:
                print(f"Skipping graph {i}: No nodes.")
                continue

            # Target C is based on the 'Close' price (row index 0)

            C = torch.zeros(X.shape[1])

            # X is [feature, node, timestep]
            
            for j in range(C.shape[0]):
                if X[0, j, -1] - X[0, j, -2] > 0:
                    C[j] = 1

            # Remove the last timestep (preditction one)
            X_features = X[:, :, :-1]
            
            if X_features.nelement() == 0:
                print(f"Skipping graph {i}: Feature matrix is empty.")
                continue
                
            # from [feature, node, timestep] to [node, feature, timestep] -> [node, feature * timestamps]
            X_final = X_features.permute(1, 0, 2).reshape(X_features.shape[1], -1)
            
            try:
                X_final = torch.nan_to_num(torch.log1p(X_final), 0)
            except Exception as e:
                print(f"Skipping graph {i} due to log1p error: {e}")
                continue

            try:
                edge_index, edge_attr = dense_to_sparse(self.adjacency_matrix(X_final))
            except Exception as e:
                print(f"Skipping graph {i} due to adjacency matrix error: {e}")
                continue

            data = Data(x=X_final, edge_index=edge_index, edge_attr=edge_attr, y=C)
            torch.save(data, filename)