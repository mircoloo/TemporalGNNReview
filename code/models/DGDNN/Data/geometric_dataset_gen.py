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


class MyDataset(Dataset):
    def __init__(self, root: str, desti: str, market: str, comlist: List[str], start: str, end: str, window: int, dataset_type: str, fast_approx):
        super().__init__()
        self.market = market
        self.root = root
        self.desti = desti
        self.start = start
        self.end = end
        self.window = window

        # Determine the maximum end date among all your dataset splits to set the global search limit
        self.global_data_search_cutoff = '2025-07-05' # Ensure this is a date beyond your latest 'end' date in test_sedate

        # Modify find_dates to also return the *filtered* comlist
        self.comlist, self.dates, self.next_day = self.find_dates(start, end, root, comlist, market, self.global_data_search_cutoff)

        self.dataset_type = dataset_type
        self.fast_approx = fast_approx

        # Check if we have valid dates to proceed
        if not self.dates or len(self.dates) < self.window + 1:
            print(f"Insufficient common dates ({len(self.dates)}) found across all companies for a window of size {self.window}. Dataset will be empty.")
            self.comlist = [] # Ensure comlist is empty if dataset is empty

        if not self.comlist: # Added check for empty comlist after filtering
            print("No valid companies with data found. Dataset will be empty.")
            self.dates = [] # Ensure dates is empty if no companies


        # Check if graph files already exist
        graph_files_exist = all(os.path.exists(os.path.join(desti, f'{market}_{dataset_type}_{start}_{end}_{window}/graph_{i}.pt')) for i in range(len(self.dates) - window + 1))

        if not graph_files_exist:
            if self.dates and len(self.dates) >= self.window + 1 and self.comlist: # Add self.comlist check here too
                self._create_graphs(self.dates, desti, self.comlist, market, root, window) # Pass self.comlist
            else:
                print("Skipping graph creation due to insufficient common dates or no valid companies.")


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
        """
        Check if the provided dates are chronologically correct within a given range [start_str, end_str]
        """
        date_format = "%Y-%m-%d"
        try:
            date = datetime.strptime(date_str, date_format)
            start = datetime.strptime(start_str, date_format)
            end = datetime.strptime(end_str, date_format)
            return start <= date <= end
        except ValueError:
            return False

    def find_dates(self, start: str, end: str, path: str, initial_comlist: List[str], market: str, search_until_date: str) -> Tuple[List[str], List[str], str]:
        """
        Identifies and returns common trading dates across a list of companies, filtering out those without data files
        and ensuring date compliance for dataset construction.

        Args:
            start (str): The start date (YYYY-MM-DD) for the primary data window.
            end (str): The end date (YYYY-MM-DD) for the primary data window.
            path (str): The root directory where company CSV data files are located.
            initial_comlist (List[str]): The initial, unfiltered list of company tickers.
            market (str): The market identifier (e.g., 'NASDAQ') used in the CSV filename.
            search_until_date (str): The absolute latest date (YYYY-MM-DD) to consider for
                                     'after-end' dates (should be beyond the latest `end` in your splits).

        Returns:
            Tuple[List[str], List[str], str]: A tuple containing:
                -   `List[str]`: The `filtered_comlist` of companies that have existing data files.
                -   `List[str]`: A sorted list of common trading dates found across all `filtered_comlist`
                                 companies within the `[start, end]` range. Returns an empty list if
                                 no common dates are found.
                -   `str` (or `None`): The earliest common date immediately following the `end` date,
                                       intended for prediction targets. Returns `None` if no such common day exists.
        """
        
        # Filter comlist upfront based on file existence

        filtered_comlist: list[str] = []
        for h in initial_comlist:
            d_path = os.path.join(path, f'{market}_{h}_30Y.csv')
            if os.path.isfile(d_path):
                filtered_comlist.append(h)
            else:
                print(f"Stock {h} data file not found at {d_path}. **Excluding this stock from analysis.**")

        if not filtered_comlist:
            print("No valid stock data files found after initial filtering. Returning empty lists.")
            return [], [], None # Return empty list for comlist, dates, and None for next_day

        date_sets_valid = []
        after_end_date_sets_valid = []

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        search_until_dt = datetime.strptime(search_until_date, "%Y-%m-%d")

        for company_ticker in filtered_comlist: # Iterate over the filtered list
            dates: set = set()
            after_end_dates: set = set()
            d_path: str = os.path.join(path, f'{market}_{company_ticker}_30Y.csv')
            with open(d_path, 'r') as f:
                file = csv.reader(f)
                next(file, None) #Skip the first row
                for line in file:
                    if not line:
                        continue
                    date_str = line[0][:10]
                    try:
                        current_date_dt = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        continue

                    if start_dt <= current_date_dt <= end_dt:
                        dates.add(date_str)
                    elif end_dt < current_date_dt <= search_until_dt:
                        after_end_dates.add(date_str)

            if len(dates) == 0:
                print(f"For stock {h}, there are no dates in the primary range {start} to {end}. This stock's dates won't contribute to the common dates.")
            else:
                date_sets_valid.append(dates)

            if len(after_end_dates) == 0:
                print(f"For stock {h}, there are no 'after-end' dates in range {end} to '{search_until_date}'. This stock's dates won't contribute to the common 'next day'.")
            else:
                after_end_date_sets_valid.append(after_end_dates)

        if not date_sets_valid:
            print("No companies had valid dates in the primary range. Returning empty common dates.")
            return filtered_comlist, [], None # Still return filtered_comlist, but no dates/next_day

        if not after_end_date_sets_valid:
            print("No companies had valid dates in the 'after-end' range. Returning no common 'next day'.")
            return filtered_comlist, [], None


        all_dates = list(set.intersection(*date_sets_valid))
        all_after_end_dates = list(set.intersection(*after_end_date_sets_valid))

        if not all_dates:
            print("No common dates found across all *valid* companies within the primary range.")
            return filtered_comlist, [], None

        next_common_day = min(all_after_end_dates) if all_after_end_dates else None

        if next_common_day is None:
            print(f"Warning: No common next_day found across all *valid* companies after {end} and up to {search_until_date}. This will prevent graph generation.")
            return filtered_comlist, [], None

        print(f"Max date in range: {max(all_dates)}, Min date after range: {next_common_day}")

        return filtered_comlist, sorted(all_dates), next_common_day # Return the filtered comlist as well

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

        A[A<1] = 1
        return torch.log(A)

    def node_feature_matrix(self, dates: List[str], comlist: List[str], market: str, path: str) -> torch.Tensor:
        dates_dt = [pd.to_datetime(date).date() for date in dates]
        # X will have dimensions [num_features, num_nodes, num_timestamps]
        # num_nodes is now len(comlist) which is already filtered
        X = torch.zeros((5, len(comlist), len(dates_dt)))

        for idx, h in enumerate(comlist): # Iterate over the *filtered* comlist
            d_path = os.path.join(path, f'{market}_{h}_30Y.csv')
            # No need for os.path.isfile(d_path) check here, as it's guaranteed to exist for h in comlist
            df = pd.read_csv(d_path, parse_dates=[0], index_col=0)
            df.index = df.index.astype(str).str.split(" ").str[0]
            df.index = pd.to_datetime(df.index)

            df_selected_and_reindexed = df.reindex(pd.to_datetime(dates_dt))
            df_selected_and_reindexed = df_selected_and_reindexed.fillna(0) # Keep fillna for actual data gaps, not missing files

            df_T = df_selected_and_reindexed.transpose()
            df_selected = df_T.iloc[0:5]

            if df_selected.shape[1] != len(dates_dt):
                print(f"Warning: Stock {h} has missing data for some dates in the current window. Data will be partially filled with zeros.")

            X[:, idx, :] = torch.from_numpy(df_selected.to_numpy())

        return X

    def _create_graphs(self, dates: List[str], desti: str, comlist: List[str], market: str, root: str, window: int):
            if len(dates) < window + 1 or not comlist: # Add comlist check
                print(f"Not enough common dates ({len(dates)}) or no valid companies ({len(comlist)}) to form a window of size {window} plus a prediction day. Skipping graph generation.")
                return

            dates_with_next_day = list(dates)
            if self.next_day:
                dates_with_next_day.append(self.next_day)
            else:
                print("Warning: No 'next_day' found for prediction target. Graph generation might be incomplete or problematic.")
                return

            for i in tqdm(range(len(dates_with_next_day) - window)):
                directory_path = os.path.join(desti, f'{market}_{self.dataset_type}_{self.start}_{self.end}_{window}')
                filename = os.path.join(directory_path, f'graph_{i}.pt')

                if os.path.exists(filename):
                    continue

                box = dates_with_next_day[i : i + window + 1]

                if len(box) != (window + 1):
                    print(f"Skipping graph {i}: Incomplete date 'box' ({len(box)} dates instead of {window + 1}).")
                    continue

                # Pass the already filtered self.comlist
                X = self.node_feature_matrix(box, self.comlist, market, root)

                if X.shape[1] == 0 or X.shape[2] == 0:
                    print(f"Skipping graph {i}: node_feature_matrix returned empty data for this window.")
                    continue

                C = torch.zeros(X.shape[1])

                for j in range(C.shape[0]):
                    if X.shape[2] >= 2:
                        if X[3, j, -1] - X[3, j, -2] > 0:
                            C[j] = 1
                    else:
                        print(f"Warning: Insufficient time steps for prediction for stock {self.comlist[j]} in graph {i}. Setting C[{j}]=0.")
                        C[j] = 0


                X_features = X[:, :, :-1]

                if X_features.shape[0] == 0 or X_features.shape[1] == 0 or X_features.shape[2] == 0:
                    print(f"Skipping graph {i}: Feature matrix is empty after removing prediction day.")
                    continue

                X_dim = [X_features.shape[0], X_features.shape[-1]]
                X_reshaped = X_features.view(-1, X_dim[-1])

                if X_reshaped.shape[0] == 0 or X_reshaped.shape[1] == 0:
                     print(f"Skipping graph {i}: Reshaped feature matrix is empty.")
                     continue

                X_final = torch.chunk(X_reshaped, X_dim[0], dim=0)
                X_final = torch.cat(X_final, dim=1)

                try:
                    X_final = torch.Tensor(np.log1p(X_final.numpy()))
                except ValueError as e:
                    print(f"Skipping graph {i} due to ValueError during log1p transform (e.g., negative input): {e}")
                    continue


                try:
                    edge_index, edge_attr = dense_to_sparse(self.adjacency_matrix(X_final))
                except Exception as e:
                    print(f"Skipping graph {i} due to error in adjacency matrix calculation: {e}")
                    continue

                data = Data(x=X_final, edge_index=edge_index, edge_attr=edge_attr, y=C.long())
                os.makedirs(directory_path, exist_ok=True)
                torch.save(data, filename)