from pathlib import Path
import torch
import json
from matplotlib import pyplot as plt
import seaborn as sns



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
    def load_ticker_maps(self):
        json_stocks_mapping = json.load(open(self.dataset_path / "ticker_index_mapping.json", "r"))
        self.index_to_stock = {int(k): v for k, v in json_stocks_mapping.items()} # map index to stock ticker 
        self.stock_to_index = {v: int(k) for k, v in json_stocks_mapping.items()} # map stock ticker to index

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
        self.time_period = folder_name_chunk[1]  # e.g., Validation
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
        print(f"| {'Time Period:':<20} {str(self.time_period):<{content_width-22}} |") 
        print(f"| {'Start Date:':<20} {str(self.start_date):<{content_width-22}} |")
        print(f"| {'End Date:':<20} {str(self.end_date):<{content_width-22}} |")
        print(f"| {'Window Size:':<20} {str(self.window_size):<{content_width-22}} |")
        print(f"| {'Normalizer:':<20} {str(self.normalizer):<{content_width-22}} |")
        print(f"| {'Number of Features:':<20} {str(self.num_features):<{content_width-22}} |")
        print(f"| {'Number of Nodes:':<20} {str(self.num_nodes):<{content_width-22}} |")
        print(f"| {'Number of Snapshots:':<20} {str(self.num_snapshots):<{content_width-22}} |")
        print(f"{'=' * (content_width + 4)}")

        


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



def main():
    dataset_path =  get_dataset_path() / "NASDAQ_Train_2016-05-01_2017-06-30_19"
    analyzer = MarketAnalyzer(dataset_path)
    snapshots_info = analyzer.get_snapshots_info()  # Analyze the first snapshot
    print([ snap['up_ratio'] for snap in snapshots_info])  # Print the up ratio for each snapshot

if __name__ == "__main__":
    main()