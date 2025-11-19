from pathlib import Path
import itertools
import math
import random
import torch
import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import numpy as np


def get_dataset_path() -> Path:
    """
    Returns the path to the dataset graph directory.
    """
    return Path(__file__).parent / "data" / "datasets" / "graph"


class MarketAnalyzer():
    def __init__(self, dataset_path: Path, train_path: Path = None, num_features: int = 5, convert_x_to_standard_form: bool =True, adj_minmax: bool = False, threshold=0):
        self.dataset_path = dataset_path
        self.train_path = train_path
        self.num_features = num_features
        self.convert_x_to_standard_form = convert_x_to_standard_form
        self.threshold = threshold
        self.adj_minmax = adj_minmax
        self.graph_snapshots = self.load_graph_snapshots(adj_minmax=adj_minmax, threshold=self.threshold)
        self.load_dataset_info()
        self.load_ticker_maps()
        self.features = ["Close", "High", "Low", "Open", "Volume"] if num_features == 5 else ["Close"]
        self.analysis_results_path = self.create_folder("analysis_results")

    def load_ticker_maps(self):
        json_stocks_mapping = json.load(open(self.dataset_path / "ticker_index_mapping.json", "r"))
        self.index_to_stock = {int(k): v for k, v in json_stocks_mapping.items()}
        self.stock_to_index = {v: int(k) for k, v in json_stocks_mapping.items()}

    def create_folder(self, folder_name: str):
        folder_path = self.dataset_path / folder_name
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Folder {folder_name} created at {self.dataset_path}")
        else:
            print(f"Folder {folder_name} already exists at {self.dataset_path}")
        return folder_path

    def load_graph_snapshots(self, convert_x_to_standard_form: bool = True, adj_minmax: bool = False, threshold: float = 0):
        graph_snapshots = []
        for file in self.dataset_path.glob("graph_*.pt"):
            graph_snapshots.append(torch.load(file, weights_only=False))

        if convert_x_to_standard_form:
            for data in graph_snapshots:
                if adj_minmax:
                    if self.train_path:
                        adj_params = torch.load(self.train_path / 'adj_minmax.pt', weights_only=False)
                    else:
                        adj_params = torch.load(self.dataset_path / 'adj_minmax.pt', weights_only=False)
                    adj_max = adj_params.get('adj_max', -torch.inf)
                    adj_min = adj_params.get('adj_min', torch.inf)
                    data.edge_attr = (data.edge_attr - adj_min) / (adj_max - adj_min + 1e-9)
                    data.edge_attr[data.edge_attr < threshold] = 0                    
                    mask = data.edge_attr.squeeze() > 0
                    data.edge_index = data.edge_index[:, mask]
                    data.edge_attr  = data.edge_attr[mask]
                data.x = data.x.reshape(data.x.shape[0], self.num_features, -1)
                data.x = data.x.permute(0, 2, 1)
        return graph_snapshots

    def load_dataset_info(self):
        folder_name = self.dataset_path.name
        folder_name_chunk = folder_name.split("_")
        self.market_name = folder_name_chunk[0]
        self.dataset_type = folder_name_chunk[1]
        self.start_date = folder_name_chunk[2]
        self.end_date = folder_name_chunk[3]
        self.window_size = int(folder_name_chunk[4])
        self.normalizer = folder_name_chunk[5] if len(folder_name_chunk) > 5 else 'None'
        self.num_nodes = self.graph_snapshots[0].x.shape[0] if self.graph_snapshots else 0
        self.num_snapshots = len(self.graph_snapshots)
        self.print_dataset_info()
        
    def print_dataset_info(self):
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
        all_features = []
        for data in self.graph_snapshots:
            x = data.x
            if x.size(1) < self.window_size:
                padding_size = self.window_size - x.size(1)
                last_timestep = x[:, padding_size, :].unsqueeze(1)
                x = torch.cat([x, last_timestep], dim=1)
            if x.dim() == 3:
                x = x.mean(dim=1)
            all_features.append(x)

        all_features = torch.cat(all_features, dim=0).numpy()
        df = pd.DataFrame(all_features, columns=self.features)
        df['node_idx'] = [i % self.num_nodes for i in range(len(df))]
        df['ticker'] = [self.index_to_stock[i % self.num_nodes ]  for i in range(len(df))]
        df['snapshot_idx'] = [i // self.num_nodes for i in range(len(df))]
        return df
    
    # --- REFACTORED METHODS START HERE ---

    def graph_snapshot_to_networkx(self, snapshot_index: int) -> nx.DiGraph:
        """
        Converts a single graph snapshot from PyG data to a NetworkX graph.
        This is more efficient than adding edges one by one in a loop.
        """
        snapshot = self.graph_snapshots[snapshot_index]
        edge_index = snapshot.edge_index.numpy().T  # Shape: [num_edges, 2]
        edge_attr = snapshot.edge_attr.numpy().flatten()
        
        # Create a list of weighted edges in the format (u, v, weight)
        weighted_edges = [(u, v, w) for (u, v), w in zip(edge_index, edge_attr)]
        
        G = nx.DiGraph()
        # Explicitly add all nodes to include any isolated nodes
        G.add_nodes_from(range(self.num_nodes))
        # Add all edges at once from the list
        G.add_weighted_edges_from(weighted_edges)
        return G
    
    def graph_snapshots_to_networkx(self) -> list[nx.DiGraph]:
        """
        Converts all graph snapshots to a list of NetworkX graphs.
        """
        return [self.graph_snapshot_to_networkx(i) for i in range(self.num_snapshots)]

    def get_snapshots_adjacency_matrix(self, snapshot_index: list[int] = None) -> list[np.ndarray]:
        """
        REFACTORED: Uses NetworkX's built-in function to create an adjacency matrix.
        This is more robust and ensures correct, consistent node ordering.
        """
        if snapshot_index is None:
            snapshot_index = list(range(len(self.graph_snapshots)))

        adj_matrices = []
        # Define a consistent node order for all matrices
        nodelist = list(range(self.num_nodes))

        for i in snapshot_index:
            G = self.graph_snapshot_to_networkx(i)
            # Convert graph to NumPy matrix using a fixed nodelist for consistency
            adj_matrix = nx.to_numpy_array(G, nodelist=nodelist, weight='weight', dtype=np.float32)
            adj_matrices.append(adj_matrix)
        
        if adj_matrices:
            print(f"Adjacency matrix shape: {adj_matrices[0].shape}")
        if self.adj_minmax:
            for adj_matrix in adj_matrices:
                adj_matrix[adj_matrix < self.threshold] = 0
        return adj_matrices

    def get_snapshot_info(self, snapshot_index: int) -> dict:
        """
        OPTIMIZED: Calculates snapshot info directly from tensors without NetworkX conversion.
        """
        snapshot = self.graph_snapshots[snapshot_index]
        
        num_nodes = snapshot.num_nodes
        num_edges = snapshot.num_edges

        # For a directed graph, density = M / (N * (N - 1))
        # where M is number of edges, N is number of nodes.
        if num_nodes > 1:
            connectivity = num_edges / (num_nodes * (num_nodes - 1))
        else:
            connectivity = 0.0

        up_target = sum(snapshot.y > 0).item()
        down_target = num_nodes - up_target if num_nodes > 0 else 0
        up_ratio = up_target / num_nodes if num_nodes > 0 else 0

        return {
            "snapshot_index": snapshot_index,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "up_target": up_target,
            "down_target": down_target,
            "up_ratio": up_ratio,
            "down_ratio": 1 - up_ratio,
            "connectivity": connectivity
        }
    
    def get_snapshots_info(self) -> list[dict]:
        """Returns a list of info dictionaries for all snapshots."""
        return [self.get_snapshot_info(i) for i in range(self.num_snapshots)]
    
    def get_snapshots_info_df(self) -> pd.DataFrame:
        """Returns snapshot info as a pandas DataFrame."""
        return pd.DataFrame(self.get_snapshots_info())

    def get_degree_dist(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        OPTIMIZED: Calculates degree distributions directly from PyG tensors
        without converting to NetworkX graphs, which is much faster and more memory-efficient.
        """
        num_all_nodes = self.num_nodes * self.num_snapshots
        
        # Initialize arrays to hold degree values for all nodes across all snapshots
        all_in_degrees = np.zeros(num_all_nodes, dtype=np.int32)
        all_out_degrees = np.zeros(num_all_nodes, dtype=np.int32)
        all_weighted_in_degrees = np.zeros(num_all_nodes, dtype=np.float32)
        all_weighted_out_degrees = np.zeros(num_all_nodes, dtype=np.float32)

        for i, snapshot in enumerate(self.graph_snapshots):
            start_idx = i * self.num_nodes
            end_idx = start_idx + self.num_nodes
            
            edge_index = snapshot.edge_index
            edge_attr = snapshot.edge_attr.squeeze()

            # Unweighted degrees
            # In-degree: Count occurrences of each node index in the destination (target) part of edge_index
            in_degree = np.bincount(edge_index[1].numpy(), minlength=self.num_nodes)
            # Out-degree: Count occurrences in the source part of edge_index
            out_degree = np.bincount(edge_index[0].numpy(), minlength=self.num_nodes)
            
            all_in_degrees[start_idx:end_idx] = in_degree
            all_out_degrees[start_idx:end_idx] = out_degree

            # Weighted degrees
            # Weighted In-degree: Sum of weights for incoming edges
            weighted_in_degree = np.zeros(self.num_nodes, dtype=np.float32)
            np.add.at(weighted_in_degree, edge_index[1].numpy(), edge_attr.numpy())
            all_weighted_in_degrees[start_idx:end_idx] = weighted_in_degree
            
            # Weighted Out-degree: Sum of weights for outgoing edges
            weighted_out_degree = np.zeros(self.num_nodes, dtype=np.float32)
            np.add.at(weighted_out_degree, edge_index[0].numpy(), edge_attr.numpy())
            all_weighted_out_degrees[start_idx:end_idx] = weighted_out_degree

        return all_in_degrees, all_out_degrees, all_weighted_in_degrees, all_weighted_out_degrees

    def get_homophily_score(self, snapshot_index: int) -> float:
        """
        OPTIMIZED: Calculates homophily directly from tensors.
        """
        snapshot = self.graph_snapshots[snapshot_index]
        targets = snapshot.y.numpy()
        edge_index = snapshot.edge_index.numpy()

        total_edges = snapshot.num_edges
        if total_edges == 0:
            return 0.0

        # Count edges where connected nodes have the same class/label
        source_labels = targets[edge_index[0]]
        target_labels = targets[edge_index[1]]
        homophilous_edges = np.sum(source_labels == target_labels)
            
        return homophilous_edges / total_edges

    def get_average_homophily_score(self) -> float:
        """
        Calculates the average homophily score across all snapshots in the dataset.
        """
        if self.num_snapshots == 0:
            return 0.0
        
        scores = [self.get_homophily_score(i) for i in range(self.num_snapshots)]
        return np.mean(scores) if scores else 0.0

    def get_feature_timeseries(self, stock_ticker=None, feature='Close', num_snapshots=None):
        if feature not in self.features:
            raise ValueError(f"Feature {feature} not found. Available features: {self.features}")
        
        timeseries_data = []
        stock_idx = self.stock_to_index.get(stock_ticker) if stock_ticker else None
        snapshots_to_use = self.graph_snapshots[:num_snapshots] if num_snapshots else self.graph_snapshots
        
        for snapshot_idx, snapshot in enumerate(snapshots_to_use):
            x = snapshot.x
            if x.dim() == 3:
                feature_values = x[:, -1, self.features.index(feature)].numpy()
            else:
                feature_values = x[:, self.features.index(feature)].numpy()
                
            if stock_idx is not None:
                timeseries_data.append({
                    'snapshot': snapshot_idx,
                    'value': feature_values[stock_idx],
                    'ticker': stock_ticker
                })
            else:
                for node_idx in range(len(feature_values)):
                    ticker = self.index_to_stock.get(node_idx, f"Node_{node_idx}")
                    timeseries_data.append({
                        'snapshot': snapshot_idx,
                        'value': feature_values[node_idx],
                        'ticker': ticker
                    })
        
        return pd.DataFrame(timeseries_data)

    def calculate_derived_features(self, df=None):
        if df is None:
            df = self.analyze_node_features()
        
        result = df.copy()
        
        if 'High' in df.columns and 'Low' in df.columns:
            result['volatility'] = df['High'] - df['Low']
            
        if 'Close' in df.columns and 'Open' in df.columns:
            result['daily_return'] = (df['Close'] - df['Open']) / df['Open']
            
        if 'Volume' in df.columns and 'Close' in df.columns:
            result['volume_price_ratio'] = df['Volume'] / df['Close']
        
        return result

    def get_class_dist(self) -> pd.Series:
        """
        Calculates the distribution of classes (Up/Down) across all snapshots.
        """
        all_labels = []
        for snapshot in self.graph_snapshots:
            all_labels.extend(snapshot.y.numpy())
        
        labels_series = pd.Series(all_labels)
        class_dist = labels_series.value_counts(normalize=True)
        return class_dist
    
    def get_connectivity(self) -> float:
        """
        OPTIMIZED: Calculates average graph connectivity (density) across all snapshots
        without creating a list of NetworkX graphs.
        """
        if self.num_snapshots == 0:
            return 0.0
        
        total_connectivity = 0.0
        for snapshot in self.graph_snapshots:
            num_nodes = snapshot.num_nodes
            num_edges = snapshot.num_edges
            if num_nodes > 1:
                total_connectivity += num_edges / (num_nodes * (num_nodes - 1))
        
        return total_connectivity / self.num_snapshots
    
    def get_average_node_degrees(self) -> pd.DataFrame:
        """
        OPTIMIZED: Calculates the average in-degree and out-degree for each node
        across all snapshots using vectorized operations.
        """
        # Get the total degrees for all nodes across all snapshots
        in_degrees, out_degrees, weighted_in_degrees, weighted_out_degrees = self.get_degree_dist()
        
        # Reshape the flat arrays into (num_snapshots, num_nodes)
        in_degrees_reshaped = in_degrees.reshape(self.num_snapshots, self.num_nodes)
        out_degrees_reshaped = out_degrees.reshape(self.num_snapshots, self.num_nodes)
        weighted_in_reshaped = weighted_in_degrees.reshape(self.num_snapshots, self.num_nodes)
        weighted_out_reshaped = weighted_out_degrees.reshape(self.num_snapshots, self.num_nodes)

        # Calculate the mean across the snapshots (axis 0)
        avg_in_degree = in_degrees_reshaped.mean(axis=0)
        avg_out_degree = out_degrees_reshaped.mean(axis=0)
        avg_weighted_in_degree = weighted_in_reshaped.mean(axis=0)
        avg_weighted_out_degree = weighted_out_reshaped.mean(axis=0)

        # Create a DataFrame for the results
        degree_df = pd.DataFrame({
            'node_idx': range(self.num_nodes),
            'ticker': [self.index_to_stock.get(i, f'Node_{i}') for i in range(self.num_nodes)],
            'avg_in_degree': avg_in_degree,
            'avg_out_degree': avg_out_degree,
            'avg_weighted_in_degree': avg_weighted_in_degree,
            'avg_weighted_out_degree': avg_weighted_out_degree
        })
        
        return degree_df

    def compute_snapshot_hyperbolicity(self, snapshot_index: int, sample_size: int | None = None, random_state: int | None = None) -> float:
        """Compute the graph hyperbolicity for a single snapshot.

        The exact computation inspects every 4-tuple of distinct nodes when ``sample_size``
        is ``None`` or larger than the total number of combinations. For larger graphs you
        can provide ``sample_size`` to obtain an approximate estimate via random sampling.
        """

        if snapshot_index < 0 or snapshot_index >= self.num_snapshots:
            raise ValueError(f"Snapshot index {snapshot_index} is out of range (0-{self.num_snapshots - 1}).")

        G_directed = self.graph_snapshot_to_networkx(snapshot_index)
        if G_directed.number_of_nodes() < 4:
            return 0.0

        # Convert to an undirected view for distance computation; ignore weights for geodesic length.
        #G = G_directed.to_undirected()
        G = G_directed
        length = dict(nx.all_pairs_shortest_path_length(G))

        nodes = list(G.nodes())
        total_quads = math.comb(len(nodes), 4)
        rng = random.Random(random_state)

        if sample_size is None or sample_size >= total_quads:
            quads_iter = itertools.combinations(nodes, 4)
        else:
            sampled = set()
            quads_iter = []
            while len(quads_iter) < sample_size:
                quad = tuple(sorted(rng.sample(nodes, 4)))
                if quad in sampled:
                    continue
                sampled.add(quad)
                quads_iter.append(quad)

        max_hyp = 0.0

        def dist(u: int, v: int) -> float:
            try:
                return length[u][v]
            except KeyError:
                return math.inf

        for quad in quads_iter:
            a, b, c, d = quad
            distances = {
                'S1': dist(a, b) + dist(d, c),
                'S2': dist(a, c) + dist(b, d),
                'S3': dist(a, d) + dist(b, c)
            }

            if math.inf in distances.values():
                continue  # Skip disconnected selections.

            largest_two = sorted(distances.values(), reverse=True)[:2]
            hyp_value = largest_two[0] - largest_two[1]
            if hyp_value > max_hyp:
                max_hyp = hyp_value

        return 0.5 * max_hyp

    def compute_average_hyperbolicity(self, sample_size: int | None = None, random_state: int | None = None) -> float:
        """Compute the average hyperbolicity across all snapshots."""
        if self.num_snapshots == 0:
            return 0.0

        scores = []
        for idx in range(self.num_snapshots):
            score = self.compute_snapshot_hyperbolicity(idx, sample_size=sample_size, random_state=random_state)
            if not math.isnan(score):
                scores.append(score)

        return float(np.mean(scores)) if scores else 0.0
        

# --- UNCHANGED PLOTTING AND COMPARISON FUNCTIONS ---

def compare_dataset_stats(analyzers, names):
    stats = []
    for i, analyzer in enumerate(analyzers):
        stats.append({
            'Dataset': names[i],
            'Market': analyzer.market_name,
            'Time Period': f"{analyzer.start_date}-{analyzer.end_date}",
            'Num Nodes': analyzer.num_nodes,
            'Num Snapshots': analyzer.num_snapshots,
            'Window Size': analyzer.window_size,
            'Normalizer': analyzer.normalizer
        })
    return pd.DataFrame(stats)

def plot_degree_distributions(analyzers, names):
    plt.figure(figsize=(18, 16))

    # In-Degree Distribution
    plt.subplot(2, 2, 1)
    for i, analyzer in enumerate(analyzers):
        in_degrees, _, _, _ = analyzer.get_degree_dist()
        sns.kdeplot(in_degrees, label=names[i], fill=True, alpha=0.3)
    plt.title('In-Degree Distribution Comparison', fontsize=18)
    plt.xlabel('In-Degree', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(fontsize=12)

    # Out-Degree Distribution
    plt.subplot(2, 2, 2)
    for i, analyzer in enumerate(analyzers):
        _, out_degrees, _, _ = analyzer.get_degree_dist()
        sns.kdeplot(out_degrees, label=names[i], fill=True, alpha=0.3)
    plt.title('Out-Degree Distribution Comparison', fontsize=18)
    plt.xlabel('Out-Degree', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(fontsize=12)

    # Weighted In-Degree Distribution
    plt.subplot(2, 2, 3)
    for i, analyzer in enumerate(analyzers):
        _, _, weighted_in_degrees, _ = analyzer.get_degree_dist()
        sns.kdeplot(weighted_in_degrees, label=names[i], fill=True, alpha=0.3)
    plt.title('Weighted In-Degree Distribution', fontsize=18)
    plt.xlabel('Weighted In-Degree', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(fontsize=12)

    # Weighted Out-Degree Distribution
    plt.subplot(2, 2, 4)
    for i, analyzer in enumerate(analyzers):
        _, _, _, weighted_out_degrees = analyzer.get_degree_dist()
        sns.kdeplot(weighted_out_degrees, label=names[i], fill=True, alpha=0.3)
    plt.title('Weighted Out-Degree Distribution', fontsize=18)
    plt.xlabel('Weighted Out-Degree', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

def compare_snapshot_stats(analyzers, names):
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    
    for i, analyzer in enumerate(analyzers):
        snapshot_info = analyzer.get_snapshots_info_df()
        
        axs[0, 0].plot(snapshot_info['snapshot_index'], snapshot_info['up_ratio'], 
                        label=f"{names[i]} (Up Ratio)", marker='o', markersize=8)
        
        axs[0, 1].plot(snapshot_info['snapshot_index'], snapshot_info['connectivity'], 
                       label=f"{names[i]}", marker='s', markersize=8)
        
        axs[1, 0].plot(snapshot_info['snapshot_index'], snapshot_info['num_edges'], 
                       label=f"{names[i]}", marker='^', markersize=8)
        
        homophily_scores = [analyzer.get_homophily_score(idx) for idx in range(analyzer.num_snapshots)]
        axs[1, 1].plot(range(len(homophily_scores)), homophily_scores, 
                       label=f"{names[i]}", marker='*', markersize=8)
    
    axs[0, 0].set_title('Up Ratio Over Time', fontsize=16)
    axs[0, 0].set_xlabel('Snapshot Index', fontsize=14)
    axs[0, 0].set_ylabel('Up Ratio', fontsize=14)
    
    axs[0, 1].set_title('Network Connectivity', fontsize=16)
    axs[0, 1].set_xlabel('Snapshot Index', fontsize=14)
    axs[0, 1].set_ylabel('Connectivity', fontsize=14)
    
    axs[1, 0].set_title('Number of Edges', fontsize=16)
    axs[1, 0].set_xlabel('Snapshot Index', fontsize=14)
    axs[1, 0].set_ylabel('Edge Count', fontsize=14)
    
    axs[1, 1].set_title('Homophily Score', fontsize=16)
    axs[1, 1].set_xlabel('Snapshot Index', fontsize=14)
    axs[1, 1].set_ylabel('Homophily', fontsize=14)
    
    for ax in axs.flatten():
        ax.legend(fontsize=12)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# Function to compare feature distributions
def compare_feature_distributions(analyzers, names):
    feature_dfs = []
    for i, analyzer in enumerate(analyzers):
        df = analyzer.analyze_node_features()
        df['dataset'] = names[i]
        feature_dfs.append(df)
    
    all_features = pd.concat(feature_dfs)
    
    for feature in analyzers[0].features:
        plt.figure(figsize=(18, 8))
        sns.boxplot(x='dataset', y=feature, data=all_features)
        plt.title(f'{feature} Distribution Comparison', fontsize=18)
        plt.xlabel('Dataset', fontsize=16)
        plt.ylabel(feature, fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.show()

# Function to analyze time series data
def compare_time_series(analyzers, names, feature='Close'):
    plt.figure(figsize=(20, 10))
    
    # Select common stocks across all datasets
    common_stocks = set(analyzers[0].index_to_stock.values())
    for analyzer in analyzers[1:]:
        common_stocks = common_stocks.intersection(set(analyzer.index_to_stock.values()))
    
    # Take a sample of stocks if there are too many
    sample_size = min(5, len(common_stocks))
    sample_stocks = list(common_stocks)[:sample_size]
    
    # Create subplots for each stock
    fig, axes = plt.subplots(sample_size, 1, figsize=(18, 5*sample_size))
    if sample_size == 1:
        axes = [axes]  # Make sure axes is iterable
    
    for i, stock in enumerate(sample_stocks):
        for j, analyzer in enumerate(analyzers):
            ts_data = analyzer.get_feature_timeseries(stock_ticker=stock, feature=feature)
            axes[i].plot(ts_data['snapshot'], ts_data['value'], label=names[j], marker='o', linewidth=2)
        
        axes[i].set_title(f'{stock} - {feature} Price Across Datasets', fontsize=16)
        axes[i].set_xlabel('Snapshot Index', fontsize=14)
        axes[i].set_ylabel(feature, fontsize=14)
        axes[i].legend(fontsize=12)
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# Function to analyze network adjacency heatmaps
def compare_adjacency_heatmaps(analyzers, names):
    fig, axes = plt.subplots(1, len(analyzers), figsize=(20, 6))
    if len(analyzers) == 1:
        axes = [axes]  # Make axes iterable if there's only one analyzer
    
    # First determine the global min and max values across all adjacency matrices
    vmin, vmax = float('inf'), float('-inf')
    all_matrices = []
    for analyzer in analyzers:
        adj_matrix = analyzer.get_snapshots_adjacency_matrix([0])[0]  # Get first snapshot
        max_nodes = min(2000, adj_matrix.shape[0])
        matrix_subset = adj_matrix[:max_nodes, :max_nodes]
        all_matrices.append(matrix_subset)
        vmin = min(vmin, np.min(matrix_subset))
        vmax = max(vmax, np.max(matrix_subset))
    
    # Now plot all matrices with the same color scale
    for i, matrix_subset in enumerate(all_matrices):
        im = axes[i].imshow(matrix_subset, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'{names[i]} Adjacency Matrix\n(First {len(matrix_subset)} nodes)', fontsize=16)
    
    # Add a single colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
    plt.show()


# Function to analyze derived features
def compare_derived_features(analyzers, names):
    derived_dfs = []
    for i, analyzer in enumerate(analyzers):
        df = analyzer.calculate_derived_features()
        df['dataset'] = names[i]
        derived_dfs.append(df)
    
    all_derived = pd.concat(derived_dfs)
    
    # Check which derived features are available
    derived_features = [col for col in all_derived.columns 
                       if col not in analyzers[0].features + ['ticker', 'snapshot_idx', 'dataset']]
    
    if derived_features:
        for feature in derived_features:
            plt.figure(figsize=(18, 8))
            sns.boxplot(x='dataset', y=feature, data=all_derived)
            plt.title(f'{feature} Comparison', fontsize=18)
            plt.xlabel('Dataset', fontsize=16)
            plt.ylabel(feature, fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.show()
# ... (the rest of the plotting functions remain unchanged)


def main():
    # Example usage (you would need to adjust paths)
    # try:
    #     dataset_path = Path("./data/datasets/graph/NASDAQ_Validation_2017_2017_10_MinMax")
    #     analyzer = MarketAnalyzer(dataset_path)
    #     
    #     # Test a refactored function
    #     df_info = analyzer.get_snapshots_info_df()
    #     print("Snapshot Info DataFrame:")
    #     print(df_info.head())
    #
    #     abs_deg, weighted_deg = analyzer.get_degree_dist()
    #     print(f"\nCalculated {len(abs_deg)} absolute degree values.")
    #     
    # except FileNotFoundError:
    #     print("Example dataset not found. Please adjust the path in main().")
    pass
if __name__ == "__main__":
    main()