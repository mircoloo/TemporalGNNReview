"""
Example script demonstrating how to visualize DGDNN attention mechanisms
"""

import torch
import sys
sys.path.append('/home/mbisoffi/tests/TemporalGNNReview/code')

from models.DGDNN.dgdnn import DGDNN
from models.DGDNN.attention_visualizer import DGDNNAttentionVisualizer

# Example: Visualize attention for a trained DGDNN model

def visualize_dgdnn_attention(model, test_data, stock_names=None, device='cpu'):
    """
    Visualize attention mechanisms for DGDNN model
    
    Args:
        model: Trained DGDNN model
        test_data: Test dataset (PyG Data objects)
        stock_names: List of stock ticker names
        device: Device to run on
    """
    
    # Initialize visualizer
    visualizer = DGDNNAttentionVisualizer(model, device=device)
    
    # Get a sample from test data
    sample = test_data[0]  # Use first test sample
    
    # Prepare inputs
    X = sample.x.unsqueeze(0)  # Add batch dimension [1, N, F]
    
    # Create adjacency matrix from edge_index
    num_nodes = X.size(1)
    A = torch.zeros(1, num_nodes, num_nodes)
    edge_index = sample.edge_index
    A[0, edge_index[0], edge_index[1]] = sample.edge_attr if hasattr(sample, 'edge_attr') else 1.0
    
    # Extract attention weights
    print("Extracting attention weights...")
    visualizer.extract_attention_weights(X, A, stock_names=stock_names)
    
    # 1. Plot theta weights (diffusion step importance)
    print("\n1. Plotting theta weights...")
    visualizer.plot_theta_weights()
    
    # 2. Plot attention heatmap for layer 0, head 0
    print("\n2. Plotting attention heatmap...")
    visualizer.plot_attention_heatmap(layer_idx=0, head_idx=0)
    
    # 3. Plot all attention heads for layer 0
    print("\n3. Plotting all attention heads...")
    visualizer.plot_all_heads(layer_idx=0)
    
    # 4. Plot attention statistics
    print("\n4. Plotting attention statistics...")
    visualizer.plot_attention_statistics(layer_idx=0)
    
    # 5. Plot attention as a graph (requires networkx)
    print("\n5. Plotting attention graph...")
    try:
        visualizer.plot_stock_attention_graph(layer_idx=0, head_idx=0, 
                                             threshold=0.05, top_k_edges=30)
    except ImportError:
        print("   Skipping graph plot (networkx not installed)")
    
    # 6. Generate full report
    print("\n6. Generating full attention report...")
    visualizer.generate_full_report(X, A, stock_names=stock_names,
                                   output_dir='attention_visualizations')
    
    print("\n✓ Visualization complete!")


# Example usage with dummy data
if __name__ == "__main__":
    # Model parameters (adjust to match your trained model)
    num_nodes = 20
    num_features = 70  # 5 features * 14 window size
    
    config = {
        'diffusion_size': [num_features, 64, 64],
        'embedding_size': [128, 128],
        'embedding_hidden_size': 64,
        'embedding_output_size': 64,
        'raw_feature_size': 64,
        'classes': 1,
        'layers': 2,
        'num_nodes': num_nodes,
        'expansion_step': 5,
        'num_heads': 4,
        'active': [True, True]
    }
    
    # Create model
    model = DGDNN(**config)
    
    # Create dummy data for demonstration
    X = torch.randn(1, num_nodes, num_features)
    A = torch.rand(1, num_nodes, num_nodes)
    A = (A + A.transpose(1, 2)) / 2  # Make symmetric
    A = A / A.sum(dim=-1, keepdim=True)  # Normalize
    
    # Stock names (optional)
    stock_names = [f"STOCK_{i:02d}" for i in range(num_nodes)]
    
    # Visualize
    visualizer = DGDNNAttentionVisualizer(model, device='cpu')
    visualizer.extract_attention_weights(X, A, stock_names=stock_names)
    
    # Individual plots
    visualizer.plot_theta_weights()
    visualizer.plot_attention_heatmap(layer_idx=0, head_idx=0)
    visualizer.plot_all_heads(layer_idx=0)
    visualizer.plot_attention_statistics(layer_idx=0)
    
    # Full report
    # visualizer.generate_full_report(X, A, stock_names=stock_names)
