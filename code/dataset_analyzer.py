from pathlib import Path
import torch
import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def get_dataset_path() -> Path:
    """
    Returns the path to the dataset graph directory.
    """
    return Path(__file__).parent / "data" / "datasets" / "graph"




class MarketAnalyzer():
    def __init__(self, dataset_path: Path, num_features: int = 5, convert_x_to_standard_form: bool =True):
        self.dataset_path = dataset_path
        self.num_features = num_features
        self.convert_x_to_standard_form = convert_x_to_standard_form
        self.graph_snapshots = self.load_graph_snapshots() # List of graph snapshots loaded from the dataset directory
        self.load_dataset_info()  # Load dataset information from the directory
        self.load_ticker_maps() # Initialize ticker maps (index to stock and stock to index)
        self.features = ["Close", "High", "Low", "Open", "Volume"] if num_features == 5 else ["Close"] # hard coded, to change with more features
        self.analysis_results_path = self.create_folder("analysis_results")  # Create a folder for analysis results
        


    def load_ticker_maps(self):
        json_stocks_mapping = json.load(open(self.dataset_path / "ticker_index_mapping.json", "r"))
        self.index_to_stock = {int(k): v for k, v in json_stocks_mapping.items()} # map index to stock ticker 
        self.stock_to_index = {v: int(k) for k, v in json_stocks_mapping.items()} # map stock ticker to index

    def create_folder(self, folder_name: str):
        """
        Create a folder in the dataset directory if it does not exist.
        """
        folder_path = self.dataset_path / folder_name
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Folder {folder_name} created at {self.dataset_path}")
        else:
            print(f"Folder {folder_name} already exists at {self.dataset_path}")
        return folder_path

    def load_graph_snapshots(self, convert_x_to_standard_form: bool = True):
        """
        Load graph snapshots from the dataset directory.
        """
        graph_snapshots = []
        for file in self.dataset_path.glob("graph_*.pt"):
            graph_snapshots.append(torch.load(file, weights_only=False))

        if convert_x_to_standard_form:
            for data in graph_snapshots:
                data.x = data.x.reshape(data.x.shape[0], self.num_features, -1)
                data.x = data.x.permute(0, 2, 1)  # [B, N, F]
        return graph_snapshots

    def load_dataset_info(self):
        """
        Load dataset information from the dataset directory.
        """
        folder_name = self.dataset_path.name
        folder_name_chunk = folder_name.split("_")
        self.market_name = folder_name_chunk[0]  # e.g., NASDAQ
        self.dataset_type = folder_name_chunk[1]  # e.g., Validation
        self.start_date = folder_name_chunk[2]  # e.g., 2017
        self.end_date = folder_name_chunk[3]  # e.g., 2017  
        self.window_size = int(folder_name_chunk[4])
        self.normalizer = folder_name_chunk[5] if len(folder_name_chunk) > 5 else 'None'
        self.num_nodes = self.graph_snapshots[0].x.shape[0] if self.graph_snapshots else 0
        self.num_snapshots = len(self.graph_snapshots)

        self.print_dataset_info()
        
    def print_dataset_info(self):
        """
        Print dataset information with aligned columns.
        """
        # Define the width for the entire content area
        content_width = 45
        
        print(f"{'=' * (content_width + 4)}")
        print(f"|{' Dataset Information '.center(content_width)}|")
        print(f"{'=' * (content_width + 4)}")
        print(f"| {'Market Name:':<20} {str(self.market_name):<{content_width-22}} |")
        print(f"| {'Time Period:':<20} {str(self.dataset_type):<{content_width-22}} |") 
        print(f"| {'Start Date:':<20} {str(self.start_date):<{content_width-22}} |")
        print(f"| {'End Date:':<20} {str(self.end_date):<{content_width-22}} |")
        print(f"| {'Window Size:':<20} {str(self.window_size):<{content_width-22}} |")
        print(f"| {'Normalizer:':<20} {str(self.normalizer):<{content_width-22}} |")
        print(f"| {'Number of Features:':<20} {str(self.num_features):<{content_width-22}} |")
        print(f"| {'Number of Nodes:':<20} {str(self.num_nodes):<{content_width-22}} |")
        print(f"| {'Number of Snapshots:':<20} {str(self.num_snapshots):<{content_width-22}} |")
        print(f"{'=' * (content_width + 4)}")

        

    def analyze_node_features(self):
        """
        Analyze node features across all snapshots and return a DataFrame.

        """
        all_features = []
        for data in self.graph_snapshots:
            x = data.x  # shape: [num_nodes, time_steps, features] oppure [num_nodes, num_features]

            if x.size(1) < self.window_size:
                # If the second dimension is smaller than the window size, pad it
                padding_size = self.window_size - x.size(1)
                last_timestep = x[:, padding_size, :].unsqueeze(1)  # Get the last timestep
                x = torch.cat([x, last_timestep], dim=1)
            if x.dim() == 3:
                x = x.mean(dim=1)  # media nel tempo se 3D
            all_features.append(x)

        all_features = torch.cat(all_features, dim=0).numpy()  # [num_snapshots * num_nodes, num_features]
        df = pd.DataFrame(all_features, columns=self.features)
        df['ticker'] = [self.index_to_stock[i % self.num_nodes ]  for i in range(len(df))]  # Add ticker column
        df['snapshot_idx'] = [i // self.num_nodes for i in range(len(df))]  # Add snapshot index column
        return df

    def get_snapshots_adjacency_matrix(self, snapshot_index: list[int] = None):
        
        if snapshot_index is None:
            snapshot_index = list(range(self.num_snapshots))

        A = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)  # Initialize adjacency matrix

        adj_matrixes = []

        for i in snapshot_index:
            snapshot = self.graph_snapshots[i]
            edge_index = snapshot.edge_index.numpy()
            edge_attr = snapshot.edge_attr.numpy() 
            for edge in range(edge_index.shape[1]):
                A[edge_index[0][edge], edge_index[1][edge]] = edge_attr[edge]
            adj_matrixes.append(A.copy())  # Append a copy of the current adjacency matrix
        print(f"Adjacency matrix shape: {A.shape}")
        return adj_matrixes

    def get_snapshot_info(self, snapshot_index: int):
        snapshot = self.graph_snapshots[snapshot_index] # get the snapshot at the specified index
        num_edges = snapshot.edge_index.shape[1]  # get the edge index of the snapshot
        num_nodes = snapshot.x.shape[0]
        up_target = sum(snapshot.y > 0).item()  # count the number of up targets
        down_target = num_nodes - up_target  # count the number of down targets
        up_ratio = up_target / num_nodes if num_nodes > 0 else 0
        down_ratio = 1 - up_ratio
        connectivness = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

        # print(f"Snapshot {snapshot_index} Analysis:")
        # print(f"Number of Nodes: {num_nodes}")
        # print(f"Number of Edges: {num_edges}")
        # print(f"Number of Up Targets: {up_target}")
        # print(f"Number of Down Targets: {down_target}")
        # print(f"Up Ratio: {up_ratio:.2f}")
        # print(f"Down Ratio: {down_ratio:.2f}")
        # print(f"Connectivity: {connectivness:.4f}")

        snapshot_info = {
            "snapshot_index": snapshot_index,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "up_target": up_target,
            "down_target": down_target,
            "up_ratio": up_ratio,
            "down_ratio": down_ratio,
            "connectivity": connectivness
        }

        return snapshot_info
    

    def get_snapshots_info(self):
        """
        Get information for all snapshots in the dataset.
        """
        snapshots_info = []
        for i in range(self.num_snapshots):
            snapshots_info.append(self.get_snapshot_info(i))
        return snapshots_info
    

    def get_snapshots_info_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.get_snapshots_info())


    

def main():
    pass

    

if __name__ == "__main__":
    main()