import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DGDNNAttentionVisualizer:
    """Visualizer for DGDNN attention mechanisms"""
    
    def __init__(self, model, device='cpu'):
        """
        Args:
            model: DGDNN model instance
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.attention_weights = {}
        self.theta_weights = {}
        
    def extract_attention_weights(self, X, A, stock_names=None):
        """
        Extract attention weights from all layers
        
        Args:
            X: Input features [B, N, F]
            A: Adjacency matrix [B, N, N]
            stock_names: List of stock names/tickers (optional)
            
        Returns:
            Dictionary containing attention weights and theta values
        """
        self.model.eval()
        X = X.to(self.device)
        A = A.to(self.device)
        
        batch_size = X.size(0)
        num_nodes = X.size(1)
        
        # Store original forward hooks
        attention_outputs = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                # For MultiheadAttention, we need to get attn_weights
                # We'll modify the forward pass temporarily
                pass
            return hook
        
        # Modified forward pass to capture attention
        h = X
        h_prime = self.model.raw_h_prime(X)
        
        # Get theta weights (diffusion step importance)
        theta_soft = torch.softmax(self.model.theta, dim=-1)
        self.theta_weights = theta_soft.detach().cpu().numpy()
        
        layer_attentions = []
        
        # Process each layer
        for l in range(len(self.model.diffusion_layers) - 1):
            theta_l = theta_soft[l].unsqueeze(0).expand(batch_size, -1)
            
            # Apply diffusion
            h_list = []
            for b in range(batch_size):
                h_b = self.model.diffusion_layers[l](
                    theta_l[b],
                    self.model.T[l],
                    h[b],
                    A[b]
                )
                h_list.append(h_b)
            h = torch.stack(h_list)
            
            # Apply attention and capture weights
            x = torch.cat([h, h_prime], dim=-1)  # [B, N, input_time]
            
            # Get attention weights manually
            attn_layer = self.model.cat_attn_layers[l].multi_head_attn
            attn_output, attn_weights = attn_layer(x, x, x, need_weights=True, average_attn_weights=False)
            
            # Store attention weights [B, num_heads, N, N]
            layer_attentions.append(attn_weights.detach().cpu().numpy())
            
            # Continue with h_prime update
            out = self.model.cat_attn_layers[l].out_proj(attn_output)
            if self.model.cat_attn_layers[l].use_activation:
                out = torch.relu(out)
            
            if l == 0:
                h_prime = out
            else:
                h_prime = h_prime + out
        
        self.attention_weights = {
            'layer_attentions': layer_attentions,  # List of [B, num_heads, N, N]
            'num_layers': len(layer_attentions),
            'num_heads': layer_attentions[0].shape[1] if layer_attentions else 0,
            'num_nodes': num_nodes,
            'stock_names': stock_names if stock_names else [f"Stock_{i}" for i in range(num_nodes)]
        }
        
        return self.attention_weights
    
    def plot_attention_heatmap(self, layer_idx=0, head_idx=0, batch_idx=0, 
                               save_path=None, figsize=(12, 10)):
        """
        Plot attention heatmap for a specific layer and head
        
        Args:
            layer_idx: Which layer to visualize
            head_idx: Which attention head to visualize
            batch_idx: Which sample in batch to visualize
            save_path: Path to save the figure
            figsize: Figure size
        """
        if not self.attention_weights:
            raise ValueError("No attention weights extracted. Run extract_attention_weights first.")
        
        attn = self.attention_weights['layer_attentions'][layer_idx]
        stock_names = self.attention_weights['stock_names']
        
        # Get attention weights [N, N]
        attn_matrix = attn[batch_idx, head_idx, :, :]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        num_nodes = attn_matrix.shape[0]
        if num_nodes <= 20:
            ax.set_xticks(range(num_nodes))
            ax.set_yticks(range(num_nodes))
            ax.set_xticklabels(stock_names, rotation=45, ha='right')
            ax.set_yticklabels(stock_names)
        else:
            # Show fewer labels for large graphs
            step = max(1, num_nodes // 20)
            ticks = range(0, num_nodes, step)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([stock_names[i] for i in ticks], rotation=45, ha='right')
            ax.set_yticklabels([stock_names[i] for i in ticks])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=12)
        
        # Labels and title
        ax.set_xlabel('Key Stocks', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query Stocks', fontsize=12, fontweight='bold')
        ax.set_title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention heatmap to {save_path}")
        
        plt.show()
        
    def plot_all_heads(self, layer_idx=0, batch_idx=0, save_path=None, figsize=(20, 12)):
        """
        Plot attention weights for all heads in a specific layer
        
        Args:
            layer_idx: Which layer to visualize
            batch_idx: Which sample in batch
            save_path: Path to save the figure
            figsize: Figure size
        """
        if not self.attention_weights:
            raise ValueError("No attention weights extracted. Run extract_attention_weights first.")
        
        num_heads = self.attention_weights['num_heads']
        attn = self.attention_weights['layer_attentions'][layer_idx]
        
        # Create subplots
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if num_heads == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each head
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            attn_matrix = attn[batch_idx, head_idx, :, :]
            
            im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
            ax.set_title(f'Head {head_idx}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        # Remove extra subplots
        for idx in range(num_heads, len(axes)):
            fig.delaxes(axes[idx])
        
        fig.suptitle(f'All Attention Heads - Layer {layer_idx}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-head visualization to {save_path}")
        
        plt.show()
    
    def plot_theta_weights(self, save_path=None, figsize=(12, 6)):
        """
        Plot theta weights (diffusion step importance) for all layers
        
        Args:
            save_path: Path to save the figure
            figsize: Figure size
        """
        if not hasattr(self, 'theta_weights') or self.theta_weights is None:
            raise ValueError("No theta weights extracted. Run extract_attention_weights first.")
        
        num_layers, num_steps = self.theta_weights.shape
        
        fig, axes = plt.subplots(1, num_layers, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for l in range(num_layers):
            ax = axes[l]
            theta_l = self.theta_weights[l]
            
            # Bar plot
            ax.bar(range(num_steps), theta_l, color='steelblue', alpha=0.7)
            ax.set_xlabel('Diffusion Step (k)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Weight (θ)', fontsize=10, fontweight='bold')
            ax.set_title(f'Layer {l}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(num_steps))
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Diffusion Step Importance (Theta Weights)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved theta weights visualization to {save_path}")
        
        plt.show()
    
    def plot_attention_statistics(self, layer_idx=0, batch_idx=0, 
                                  save_path=None, figsize=(15, 8)):
        """
        Plot statistical analysis of attention weights
        
        Args:
            layer_idx: Which layer to analyze
            batch_idx: Which sample in batch
            save_path: Path to save the figure
            figsize: Figure size
        """
        if not self.attention_weights:
            raise ValueError("No attention weights extracted. Run extract_attention_weights first.")
        
        attn = self.attention_weights['layer_attentions'][layer_idx]
        num_heads = attn.shape[1]
        stock_names = self.attention_weights['stock_names']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Attention distribution across heads
        ax = axes[0, 0]
        for head_idx in range(num_heads):
            attn_matrix = attn[batch_idx, head_idx, :, :].flatten()
            ax.hist(attn_matrix, bins=50, alpha=0.5, label=f'Head {head_idx}')
        ax.set_xlabel('Attention Weight', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.set_title('Attention Weight Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Average attention received by each stock
        ax = axes[0, 1]
        avg_attn_received = attn[batch_idx].mean(axis=(0, 1))  # Average across heads and queries
        top_k = 10
        top_indices = np.argsort(avg_attn_received)[-top_k:]
        
        ax.barh(range(top_k), avg_attn_received[top_indices], color='coral')
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([stock_names[i] for i in top_indices])
        ax.set_xlabel('Average Attention Received', fontsize=10, fontweight='bold')
        ax.set_title(f'Top {top_k} Most Attended Stocks', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 3. Attention entropy per head
        ax = axes[1, 0]
        entropies = []
        for head_idx in range(num_heads):
            attn_matrix = attn[batch_idx, head_idx, :, :]
            # Calculate entropy for each query
            eps = 1e-10
            entropy = -(attn_matrix * np.log(attn_matrix + eps)).sum(axis=1).mean()
            entropies.append(entropy)
        
        ax.bar(range(num_heads), entropies, color='skyblue')
        ax.set_xlabel('Head Index', fontsize=10, fontweight='bold')
        ax.set_ylabel('Average Entropy', fontsize=10, fontweight='bold')
        ax.set_title('Attention Entropy per Head', fontsize=12, fontweight='bold')
        ax.set_xticks(range(num_heads))
        ax.grid(True, alpha=0.3)
        
        # 4. Attention sparsity (% of weights above threshold)
        ax = axes[1, 1]
        thresholds = np.linspace(0, 0.5, 20)
        sparsity_per_head = []
        
        for head_idx in range(num_heads):
            attn_matrix = attn[batch_idx, head_idx, :, :]
            sparsity = [(attn_matrix > t).mean() * 100 for t in thresholds]
            sparsity_per_head.append(sparsity)
            ax.plot(thresholds, sparsity, marker='o', label=f'Head {head_idx}')
        
        ax.set_xlabel('Attention Threshold', fontsize=10, fontweight='bold')
        ax.set_ylabel('% Weights Above Threshold', fontsize=10, fontweight='bold')
        ax.set_title('Attention Sparsity Analysis', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention statistics to {save_path}")
        
        plt.show()
    
    def plot_stock_attention_graph(self, layer_idx=0, head_idx=0, batch_idx=0,
                                   threshold=0.1, top_k_edges=50,
                                   save_path=None, figsize=(14, 14)):
        """
        Plot attention as a directed graph showing stock relationships
        
        Args:
            layer_idx: Which layer to visualize
            head_idx: Which attention head
            batch_idx: Which sample in batch
            threshold: Minimum attention weight to show edge
            top_k_edges: Show only top-k strongest edges
            save_path: Path to save the figure
            figsize: Figure size
        """
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX not installed. Install with: pip install networkx")
            return
        
        if not self.attention_weights:
            raise ValueError("No attention weights extracted. Run extract_attention_weights first.")
        
        attn = self.attention_weights['layer_attentions'][layer_idx]
        stock_names = self.attention_weights['stock_names']
        attn_matrix = attn[batch_idx, head_idx, :, :]
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, name in enumerate(stock_names):
            G.add_node(i, label=name)
        
        # Add edges (filter by threshold and top-k)
        edges = []
        for i in range(len(stock_names)):
            for j in range(len(stock_names)):
                if i != j and attn_matrix[i, j] > threshold:
                    edges.append((i, j, attn_matrix[i, j]))
        
        # Sort by weight and keep top-k
        edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_k_edges]
        
        for i, j, weight in edges:
            G.add_edge(i, j, weight=weight)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use spring layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes
        node_sizes = [300 + 1000 * attn_matrix[:, i].mean() for i in range(len(stock_names))]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7, ax=ax)
        
        # Draw edges with varying width
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_widths = [w * 5 for w in edge_weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                              edge_color='gray', arrows=True, 
                              arrowsize=20, ax=ax)
        
        # Draw labels
        labels = {i: name[:8] for i, name in enumerate(stock_names)}  # Truncate long names
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Stock Attention Graph - Layer {layer_idx}, Head {head_idx}\n'
                    f'(Top {top_k_edges} edges, threshold={threshold})',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention graph to {save_path}")
        
        plt.show()
    
    def generate_full_report(self, X, A, stock_names=None, output_dir='attention_visualizations'):
        """
        Generate a full visualization report with all plots
        
        Args:
            X: Input features
            A: Adjacency matrix
            stock_names: List of stock names
            output_dir: Directory to save all visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Extracting attention weights...")
        self.extract_attention_weights(X, A, stock_names)
        
        num_layers = self.attention_weights['num_layers']
        
        print(f"Generating visualizations for {num_layers} layers...")
        
        # 1. Theta weights
        print("- Theta weights...")
        self.plot_theta_weights(save_path=output_path / 'theta_weights.png')
        
        # 2. For each layer
        for layer_idx in range(num_layers):
            layer_dir = output_path / f'layer_{layer_idx}'
            layer_dir.mkdir(exist_ok=True)
            
            print(f"- Layer {layer_idx}...")
            
            # All heads
            self.plot_all_heads(layer_idx=layer_idx, 
                               save_path=layer_dir / 'all_heads.png')
            
            # Statistics
            self.plot_attention_statistics(layer_idx=layer_idx,
                                          save_path=layer_dir / 'statistics.png')
            
            # Individual head heatmaps
            num_heads = self.attention_weights['num_heads']
            for head_idx in range(num_heads):
                self.plot_attention_heatmap(layer_idx=layer_idx, head_idx=head_idx,
                                           save_path=layer_dir / f'head_{head_idx}_heatmap.png')
                
                # Attention graph for first head of first layer
                if layer_idx == 0 and head_idx == 0:
                    self.plot_stock_attention_graph(layer_idx=layer_idx, head_idx=head_idx,
                                                   save_path=layer_dir / f'head_{head_idx}_graph.png')
        
        print(f"\n✓ Full attention report saved to {output_path}")
