import torch
import torch.nn as nn
import torch.nn.functional as F
from .ggd import GeneralizedGraphDiffusion
from .catattn import CatMultiAttn


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
                input_time=embedding_size[i],        # Expected concatenated dimension (h and h_prime).
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

        # Initialize the learned transition parameters (T and theta)
        self._init_transition_params()

    def _init_transition_params(self):
        nn.init.xavier_uniform_(self.T)
        nn.init.constant_(self.theta, 1.0 / self.theta.size(-1))  # normalize

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
            
            # Apply attention
            if l == 0:
                h_prime = self.cat_attn_layers[l](h, h_prime)  # [B, N, embedding_size[l]]
            else:
                h_prime = h_prime + self.cat_attn_layers[l](h, h_prime)
    
        # Final prediction
        out = self.linear(h_prime)  # [B, N, classes]
        return out


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