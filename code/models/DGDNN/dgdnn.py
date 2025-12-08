import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DGDNN.ggd import GeneralizedGraphDiffusion
from models.DGDNN.catattn import CatMultiAttn


class DGDNN(nn.Module):
    def __init__(
        self,
        diffusion_size: list,     # e.g., [F0, F1, F2]. This defines the input/output feature dimensions for each GeneralizedGraphDiffusion layer.
        embedding_size: list,     # e.g., [F1+F1, E1, E1+F2, E2, ...]. This defines the input dimensions for CatMultiAttn layers, which concatenate features.
        embedding_hidden_size: int, # Hidden dimension for the projection layers within CatMultiAttn.
        embedding_output_size: int, # Output dimension of each CatMultiAttn layer. This will be the dimension of the 'h_prime' representation.
        raw_feature_size: int,    # The size to which raw features (X) are projected before the first attention layer.
        classes: int,             # Number of output classes for stock movement prediction (e.g., up, down, flat). [cite: 43]
        layers: int,              # Number of DGDNN layers, each consisting of a GGD and CatMultiAttn.
        num_nodes: int,           # Total number of nodes (stocks) in the graph. [cite: 73]
        expansion_step: int,      # K in the paper, representing the maximum diffusion step in Generalized Graph Diffusion. [cite: 103]
        num_heads: int,           # Number of attention heads for CatMultiAttn.
        active: list              # Boolean list, one per layer, indicating whether to use activation functions.
    ):
        super().__init__()

        # Transition matrices and weights
        # --- Part related to Generalized Graph Diffusion and Learning Task-Optimal Topology ---
        # As per Section 4.2 "Generalized Graph Diffusion", the model learns task-optimal graph topology.
        # This involves learning diffusion matrices (T_slices) and their corresponding weights (theta). [cite: 109]
        self.T = nn.Parameter(torch.empty(layers, expansion_step, num_nodes, num_nodes)) # T_l,k in Eq. 6[cite: 103], trainable matrices for diffusion. 
        self.theta = nn.Parameter(torch.empty(layers, expansion_step)) # theta_l,k in Eq. 6[cite: 103], trainable weights for combining T_slices. 

        # Scaling factor adjustment based on num_heads.
        # This is a common practice to maintain model capacity or parameter count when changing num_heads,
        # ensuring that total capacity scales appropriately with the number of heads.
        if num_heads != 2:
            # compute a scaling factor (float!) relative to 2-head base
            scale = num_heads / 2.0    # e.g. 3 / 2 = 1.5

            # scale the scalar sizes
            embedding_output_size  = int(round(embedding_output_size   * scale))
            raw_feature_size = int(round(raw_feature_size  * scale))

            # leave the first element of diffusion_size unchanged,
            # scale the rest via a list comprehension
            diffusion_size = [
                diffusion_size[0]
            ] + [
                int(round(x * scale))
                for x in diffusion_size[1:]
            ]

            # similarly scale each per-layer emb_size
            embedding_size = [
                int(round(x * scale))
                for x in embedding_size
            ]



        # --- Part related to Generalized Graph Diffusion Layers ---
        # Implements the Generalized Graph Diffusion (GGD) component mentioned in Section 4.2. 
        # Each layer performs a diffusion process to learn task-optimal topology. [cite: 101]
        self.diffusion_layers = nn.ModuleList([
            GeneralizedGraphDiffusion(diffusion_size[i], diffusion_size[i + 1], active[i])
            for i in range(len(diffusion_size) - 1)
        ])
        
        



        # --- Part related to Hierarchical Decoupled Representation Learning (Cat Attention) ---
        # Implements the "Hierarchical Decoupled Representation Learning" described in Section 4.3[cite: 112].
        # It uses multi-head attention (CatMultiAttn) to combine features at different hierarchical levels. [cite: 117, 118]
        self.cat_attn_layers = nn.ModuleList([
            CatMultiAttn(
                input_time=embedding_size[i],        # Expected concatenated dimension (h and h_prime) = diffusion_size[i] + embedding_output_size.
                num_heads=num_heads,                 # Number of attention heads.
                hidden_dim=embedding_hidden_size,    # Hidden dimension for internal projection in attention.
                output_dim=embedding_output_size,    # Output dimension of the attention layer.
                use_activation=active[i]             # Whether to use activation within the attention block.
            )
            for i in range(len(embedding_size))
        ])

        # Initial linear projection for raw features
        # This prepares the initial 'h_prime' (original features for attention fusion) for the attention mechanism,
        # ensuring its dimension is compatible with the attention layer's input expectations.
        self.raw_h_prime = nn.Linear(diffusion_size[0], raw_feature_size)

        # Final classifier layer
        # Maps the combined, learned representations to the final class probabilities for stock movement prediction. [cite: 43]
        self.linear = nn.Linear(embedding_output_size, classes)
        #self.linear2 = nn.Linear(embedding_output_size, embedding_output_size)
        #self.linear3 = nn.Linear(embedding_output_size, classes)

        # Initialize the learned transition parameters (T and theta)
        self._init_transition_params()
        
        # Attention visualization settings
        self.save_attention = False
        self.attention_save_dir = None
        self.forward_pass_counter = 0
        self.attention_weights_history = []
        
        # Diffusion visualization settings
        self.save_diffusion = False
        self.diffusion_save_dir = None
        self.diffusion_pass_counter = 0

    def _init_transition_params(self):
        nn.init.xavier_uniform_(self.T)
        # Inizializza theta con valori leggermente random invece che uniformi
        nn.init.normal_(self.theta, mean=0.0, std=0.1)  # Piccola varianza iniziale
    
    def enable_attention_saving(self, save_dir='attention_outputs', max_saves=100):
        """
        Enable automatic saving of attention visualizations during forward passes
        
        Args:
            save_dir: Directory to save attention visualizations
            max_saves: Maximum number of forward passes to save (to avoid disk overflow)
        """
        self.save_attention = True
        self.attention_save_dir = save_dir
        self.max_saves = max_saves
        self.forward_pass_counter = 0
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"✓ Attention saving enabled. Outputs will be saved to: {save_dir}")
    
    def disable_attention_saving(self):
        """Disable automatic saving of attention visualizations"""
        self.save_attention = False
        print("✓ Attention saving disabled")
    
    def enable_diffusion_saving(self, save_dir='diffusion_outputs', max_saves=100):
        """
        Enable automatic saving of diffusion matrix visualizations during forward passes
        
        Args:
            save_dir: Directory to save diffusion visualizations
            max_saves: Maximum number of forward passes to save (to avoid disk overflow)
        """
        self.save_diffusion = True
        self.diffusion_save_dir = save_dir
        self.max_diffusion_saves = max_saves
        self.diffusion_pass_counter = 0
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"✓ Diffusion saving enabled. Outputs will be saved to: {save_dir}")
    
    def disable_diffusion_saving(self):
        """Disable automatic saving of diffusion visualizations"""
        self.save_diffusion = False
        print("✓ Diffusion saving disabled")

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [B, N, F_in] - batched node features
            A: [B, N, N] - batched adjacency matrices
        """
        batch_size = X.size(0)
        h = X  # [B, N, F_in]
        h_prime = self.raw_h_prime(X)  # [B, N, raw_feature_size]
        
        # Normalize theta per layer
        theta_soft = F.softmax(self.theta, dim=-1)
        
        # Store attention weights if saving is enabled
        layer_attention_weights = [] if self.save_attention else None
        
        # Store diffusion matrices if saving is enabled
        diffusion_data = {
            'T_matrices': [],
            'theta_weights': theta_soft.detach().cpu() if self.save_diffusion else None,
            'diffused_features': []
        } if self.save_diffusion else None
        
        # Process each layer
        for l in range(len(self.diffusion_layers) - 1):
            # Expand theta for batch processing
            theta_l = theta_soft[l].unsqueeze(0).expand(batch_size, -1)  # [B, expansion_step]
            
            # Apply diffusion to each sample in batch
            h_list = []
            for b in range(batch_size):
                h_b = self.diffusion_layers[l](
                    theta_l[b],
                    self.T[l],
                    h[b],
                    A[b]
                )
                h_list.append(h_b)
            h = torch.stack(h_list)  # [B, N, diffusion_size[l+1]]
            
            # Save diffusion data if enabled
            if self.save_diffusion:
                diffusion_data['T_matrices'].append(self.T[l].detach().cpu())
                diffusion_data['diffused_features'].append(h.detach().cpu())
            
            # Apply attention and capture weights if needed
            x = torch.cat([h, h_prime], dim=-1)  # [B, N, input_time]
            
            if self.save_attention:
                # Get attention weights
                attn_layer = self.cat_attn_layers[l].multi_head_attn
                attn_output, attn_weights = attn_layer(x, x, x, need_weights=True, average_attn_weights=False)
                layer_attention_weights.append(attn_weights.detach().cpu())
                
                # Apply output projection
                out = self.cat_attn_layers[l].out_proj(attn_output)
                if self.cat_attn_layers[l].use_activation:
                    out = F.relu(out)
            else:
                # Normal forward without capturing attention
                if l == 0:
                    out = self.cat_attn_layers[l](h, h_prime)
                else:
                    out = self.cat_attn_layers[l](h, h_prime)
            
            # Update h_prime
            if l == 0:
                h_prime = out
            else:
                h_prime = h_prime + out
        
        # Save attention visualizations if enabled
        if self.save_attention and self.forward_pass_counter < self.max_saves:
            self._save_attention_visualization(
                layer_attention_weights, 
                theta_soft.detach().cpu(),
                X.size(1)  # num_nodes
            )
            self.forward_pass_counter += 1
        
        # Save diffusion visualizations if enabled
        if self.save_diffusion and self.diffusion_pass_counter < self.max_diffusion_saves:
            self._save_diffusion_visualization(
                diffusion_data,
                A.detach().cpu(),
                X.size(1)  # num_nodes
            )
            self.diffusion_pass_counter += 1
    
        # Final prediction
        return self.linear(h_prime)
    
    def _save_attention_visualization(self, attention_weights, theta_weights, num_nodes):
        """
        Save attention visualization for current forward pass
        
        Args:
            attention_weights: List of attention weight tensors per layer
            theta_weights: Theta weights tensor
            num_nodes: Number of nodes
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        
        save_path = Path(self.attention_save_dir) / f'forward_pass_{self.forward_pass_counter:05d}'
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save theta weights
        num_layers, num_steps = theta_weights.shape
        fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 3), squeeze=False)
        axes = axes.flatten()
        
        for l in range(num_layers):
            theta_l = theta_weights[l].numpy()
            axes[l].bar(range(num_steps), theta_l, color='steelblue', alpha=0.7)
            axes[l].set_xlabel('Diffusion Step (k)', fontsize=8)
            axes[l].set_ylabel('Weight (θ)', fontsize=8)
            axes[l].set_title(f'Layer {l}', fontsize=10, fontweight='bold')
            axes[l].set_xticks(range(num_steps))
            axes[l].grid(True, alpha=0.3)
        
        fig.suptitle(f'Theta Weights - Pass {self.forward_pass_counter}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path / 'theta_weights.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Save attention heatmaps for each layer and head
        for layer_idx, attn in enumerate(attention_weights):
            batch_idx = 0  # Use first sample in batch
            num_heads = attn.size(1)
            attn_np = attn[batch_idx].numpy()  # [num_heads, N, N]
            
            # Create subplots for all heads
            cols = min(4, num_heads)
            rows = (num_heads + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
            axes = axes.flatten()
            
            for head_idx in range(num_heads):
                ax = axes[head_idx]
                attn_matrix = attn_np[head_idx]
                
                im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
                ax.set_title(f'Head {head_idx}', fontsize=9, fontweight='bold')
                ax.set_xlabel('Key', fontsize=7)
                ax.set_ylabel('Query', fontsize=7)
                
                # Only show ticks for small graphs
                if num_nodes <= 20:
                    ax.set_xticks(range(num_nodes))
                    ax.set_yticks(range(num_nodes))
                    ax.tick_params(labelsize=6)
                
                plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Remove extra subplots
            for idx in range(num_heads, len(axes)):
                fig.delaxes(axes[idx])
            
            fig.suptitle(f'Layer {layer_idx} Attention - Pass {self.forward_pass_counter}', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path / f'layer_{layer_idx}_attention.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Save a summary text file
        with open(save_path / 'summary.txt', 'w') as f:
            f.write(f"Forward Pass: {self.forward_pass_counter}\n")
            f.write(f"Number of Nodes: {num_nodes}\n")
            f.write(f"Number of Layers: {len(attention_weights)}\n")
            f.write(f"Number of Heads: {attention_weights[0].size(1)}\n\n")
            
            f.write("Theta Weights:\n")
            for l in range(num_layers):
                theta_l = theta_weights[l].numpy()
                f.write(f"  Layer {l}: {theta_l}\n")
            
            f.write("\nAttention Statistics:\n")
            for layer_idx, attn in enumerate(attention_weights):
                attn_np = attn[0].numpy()  # First batch
                f.write(f"  Layer {layer_idx}:\n")
                for head_idx in range(attn_np.shape[0]):
                    attn_matrix = attn_np[head_idx]
                    f.write(f"    Head {head_idx}: mean={attn_matrix.mean():.4f}, "
                           f"std={attn_matrix.std():.4f}, "
                           f"max={attn_matrix.max():.4f}, "
                           f"min={attn_matrix.min():.4f}\n")


    def _save_diffusion_visualization(self, diffusion_data, adjacency_matrices, num_nodes):
        """
        Save diffusion matrix visualization for current forward pass
        
        Args:
            diffusion_data: Dictionary containing T_matrices, theta_weights, and diffused_features
            adjacency_matrices: Adjacency matrices [B, N, N]
            num_nodes: Number of nodes
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        
        save_path = Path(self.diffusion_save_dir) / f'forward_pass_{self.diffusion_pass_counter:05d}'
        save_path.mkdir(parents=True, exist_ok=True)
        
        T_matrices = diffusion_data['T_matrices']
        theta_weights = diffusion_data['theta_weights']
        diffused_features = diffusion_data['diffused_features']
        num_layers = len(T_matrices)
        
        # 1. Save theta weights (same as attention visualization but for diffusion context)
        expansion_step = theta_weights.size(1)
        fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 3), squeeze=False)
        axes = axes.flatten()
        
        for l in range(num_layers):
            theta_l = theta_weights[l].numpy()
            axes[l].bar(range(expansion_step), theta_l, color='darkorange', alpha=0.7)
            axes[l].set_xlabel('Diffusion Step (k)', fontsize=8)
            axes[l].set_ylabel('Weight (θ)', fontsize=8)
            axes[l].set_title(f'Layer {l}', fontsize=10, fontweight='bold')
            axes[l].set_xticks(range(expansion_step))
            axes[l].grid(True, alpha=0.3)
            axes[l].set_ylim([0, max(1.0, theta_l.max() * 1.1)])
        
        fig.suptitle(f'Diffusion Theta Weights - Pass {self.diffusion_pass_counter}', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path / 'diffusion_theta_weights.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Save T matrices for each layer and diffusion step
        for layer_idx, T_layer in enumerate(T_matrices):
            # T_layer shape: [expansion_step, N, N]
            T_np = T_layer.numpy()
            expansion_step_layer = T_np.shape[0]
            
            # Create subplots for all diffusion steps
            cols = min(4, expansion_step_layer)
            rows = (expansion_step_layer + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
            axes = axes.flatten()
            
            # Calculate global min/max for consistent colorbar
            vmin = T_np.min()
            vmax = T_np.max()
            
            for step_idx in range(expansion_step_layer):
                ax = axes[step_idx]
                T_matrix = T_np[step_idx]
                
                im = ax.imshow(T_matrix, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
                ax.set_title(f'Step k={step_idx} (θ={theta_weights[layer_idx, step_idx]:.3f})', 
                           fontsize=9, fontweight='bold')
                ax.set_xlabel('Node j', fontsize=7)
                ax.set_ylabel('Node i', fontsize=7)
                
                # Only show ticks for small graphs
                if num_nodes <= 20:
                    ax.set_xticks(range(num_nodes))
                    ax.set_yticks(range(num_nodes))
                    ax.tick_params(labelsize=6)
                
                plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Remove extra subplots
            for idx in range(expansion_step_layer, len(axes)):
                fig.delaxes(axes[idx])
            
            fig.suptitle(f'Layer {layer_idx} Diffusion Matrices T - Pass {self.diffusion_pass_counter}', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path / f'layer_{layer_idx}_T_matrices.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Save combined weighted T matrix (sum of theta_k * T_k) for each layer
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4), squeeze=False)
        axes = axes.flatten()
        
        for layer_idx, T_layer in enumerate(T_matrices):
            T_np = T_layer.numpy()
            theta_l = theta_weights[layer_idx].numpy()
            
            # Compute weighted sum: T_combined = sum_k(theta_k * T_k)
            T_combined = np.zeros((num_nodes, num_nodes))
            for k in range(len(theta_l)):
                T_combined += theta_l[k] * T_np[k]
            
            ax = axes[layer_idx]
            im = ax.imshow(T_combined, cmap='RdBu_r', aspect='auto')
            ax.set_title(f'Layer {layer_idx} Combined T', fontsize=10, fontweight='bold')
            ax.set_xlabel('Node j', fontsize=8)
            ax.set_ylabel('Node i', fontsize=8)
            
            if num_nodes <= 20:
                ax.set_xticks(range(num_nodes))
                ax.set_yticks(range(num_nodes))
                ax.tick_params(labelsize=7)
            
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        fig.suptitle(f'Combined Weighted Diffusion Matrices', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path / 'combined_T_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Save adjacency matrix for reference
        batch_idx = 0
        A_np = adjacency_matrices[batch_idx].numpy()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(A_np, cmap='Greys', aspect='auto')
        ax.set_title(f'Input Adjacency Matrix', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Node j', fontsize=9)
        ax.set_ylabel('Node i', fontsize=9)
        
        if num_nodes <= 20:
            ax.set_xticks(range(num_nodes))
            ax.set_yticks(range(num_nodes))
            ax.tick_params(labelsize=7)
        
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(save_path / 'adjacency_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Save diffused features statistics
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3*num_layers), squeeze=False)
        axes = axes.flatten()
        
        for layer_idx, h_features in enumerate(diffused_features):
            # h_features shape: [B, N, F]
            h_np = h_features[batch_idx].numpy()  # [N, F]
            
            # Plot feature statistics per node
            ax = axes[layer_idx]
            feature_means = h_np.mean(axis=1)  # Mean across features for each node
            feature_stds = h_np.std(axis=1)    # Std across features for each node
            
            x = np.arange(num_nodes)
            ax.errorbar(x, feature_means, yerr=feature_stds, fmt='o-', 
                       capsize=3, color='darkgreen', alpha=0.7)
            ax.set_xlabel('Node', fontsize=9)
            ax.set_ylabel('Feature Value (mean ± std)', fontsize=9)
            ax.set_title(f'Layer {layer_idx} Diffused Features', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if num_nodes <= 20:
                ax.set_xticks(range(num_nodes))
        
        fig.suptitle(f'Diffused Feature Statistics', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path / 'diffused_features.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 6. Save a detailed summary text file
        with open(save_path / 'diffusion_summary.txt', 'w') as f:
            f.write(f"Diffusion Forward Pass: {self.diffusion_pass_counter}\n")
            f.write(f"Number of Nodes: {num_nodes}\n")
            f.write(f"Number of Layers: {num_layers}\n")
            f.write(f"Expansion Steps: {expansion_step}\n\n")
            
            f.write("="*60 + "\n")
            f.write("THETA WEIGHTS (Diffusion Step Importance)\n")
            f.write("="*60 + "\n")
            for l in range(num_layers):
                theta_l = theta_weights[l].numpy()
                f.write(f"\nLayer {l}:\n")
                for k, weight in enumerate(theta_l):
                    f.write(f"  Step k={k}: θ={weight:.6f}\n")
                f.write(f"  Sum: {theta_l.sum():.6f} (should be ~1.0)\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("DIFFUSION MATRIX STATISTICS\n")
            f.write("="*60 + "\n")
            for layer_idx, T_layer in enumerate(T_matrices):
                T_np = T_layer.numpy()
                f.write(f"\nLayer {layer_idx}:\n")
                for k in range(T_np.shape[0]):
                    T_k = T_np[k]
                    f.write(f"  Step k={k}:\n")
                    f.write(f"    Mean: {T_k.mean():.6f}\n")
                    f.write(f"    Std:  {T_k.std():.6f}\n")
                    f.write(f"    Min:  {T_k.min():.6f}\n")
                    f.write(f"    Max:  {T_k.max():.6f}\n")
                    f.write(f"    Sparsity: {(np.abs(T_k) < 1e-6).sum() / T_k.size * 100:.2f}%\n")
                
                # Combined matrix statistics
                theta_l = theta_weights[layer_idx].numpy()
                T_combined = sum(theta_l[k] * T_np[k] for k in range(len(theta_l)))
                f.write(f"  Combined (weighted sum):\n")
                f.write(f"    Mean: {T_combined.mean():.6f}\n")
                f.write(f"    Std:  {T_combined.std():.6f}\n")
                f.write(f"    Min:  {T_combined.min():.6f}\n")
                f.write(f"    Max:  {T_combined.max():.6f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("DIFFUSED FEATURES STATISTICS\n")
            f.write("="*60 + "\n")
            for layer_idx, h_features in enumerate(diffused_features):
                h_np = h_features[batch_idx].numpy()
                f.write(f"\nLayer {layer_idx}:\n")
                f.write(f"  Shape: {h_np.shape}\n")
                f.write(f"  Mean: {h_np.mean():.6f}\n")
                f.write(f"  Std:  {h_np.std():.6f}\n")
                f.write(f"  Min:  {h_np.min():.6f}\n")
                f.write(f"  Max:  {h_np.max():.6f}\n")
                
                # Per-node statistics
                f.write(f"  Per-node feature means: min={h_np.mean(axis=1).min():.6f}, "
                       f"max={h_np.mean(axis=1).max():.6f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("ADJACENCY MATRIX STATISTICS\n")
            f.write("="*60 + "\n")
            A_np = adjacency_matrices[batch_idx].numpy()
            f.write(f"Shape: {A_np.shape}\n")
            f.write(f"Number of edges: {(A_np > 0).sum()}\n")
            f.write(f"Density: {(A_np > 0).sum() / A_np.size * 100:.2f}%\n")
            f.write(f"Mean edge weight: {A_np[A_np > 0].mean():.6f}\n")
            f.write(f"Is symmetric: {np.allclose(A_np, A_np.T)}\n")
        
        print(f"✓ Diffusion visualization saved to: {save_path}")


## For those who use fast implementation version.

# class DGDNN(nn.Module):
#     def __init__(
#         self,
#         diffusion_size: list,
#         embedding_size: list,
#         embedding_hidden_size: int,
#         embedding_output_size: int,
#         raw_feature_size: int,
#         classes: int,
#         layers: int,
#         num_heads: int,
#         active: list
#     ):
#         super().__init__()
#         assert len(diffusion_size) - 1 == layers, "Mismatch in diffusion layers"
#         assert len(embedding_size) == layers, "Mismatch in attention layers"

#         self.layers = layers

#         self.diffusion_layers = nn.ModuleList([
#             GeneralizedGraphDiffusion(diffusion_size[i], diffusion_size[i + 1], active[i])
#             for i in range(layers)
#         ])

#         self.cat_attn_layers = nn.ModuleList([
#             CatMultiAttn(
#                 input_time=embedding_size[i],        # e.g., input = concat[h, h_prime] dim
#                 num_heads=num_heads,
#                 hidden_dim=embedding_hidden_size,      
#                 output_dim=embedding_output_size,
#                 use_activation=active[i]             
#             )
#             for i in range(len(embedding_size))
#         ])
#         # Transform raw features to be divisible by num_heads
#         self.raw_h = nn.Linear(diffusion_size[0], raw_feature_size)
        
#         self.linear = nn.Linear(embedding_output_size, classes)

#     def forward(self, X: torch.Tensor, A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             X: [N, F_in]         - node features
#             A: [2, E]            - adjacency (sparse index)
#             W: [E]               - edge weights (if using sparse edge_index)

#         Returns:
#             logits: [N, classes]
#         """
#         z = X
#         h = X

#         for l in range(self.layers):
#             z = self.diffusion_layers[l](z, A, W)  # GeneralizedGraphDiffusion (e.g. GCNConv)
#             if l == 0:
#                 h = self.cat_attn_layers[l](z, self.raw_h(h))
#             else:
#                 h = h + self.cat_attn_layers[l](z, h)

#         return self.linear(h)  # [N, classes]