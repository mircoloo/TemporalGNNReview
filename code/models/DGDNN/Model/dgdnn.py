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
            X: [N, F_in]  - node features
            A: [N, N]     - adjacency matrix
        Returns:
            logits: [N, classes]
        """
        h = X              # diffused features
        h_prime = X              # original features for attention fusion
        theta_soft = F.softmax(self.theta, dim=-1)  # normalize theta to summation of 1 per layer as the regularization

        # --- Layer-wise processing (Generalized Graph Diffusion and Decoupled Representation Learning) ---
        # This loop implements the core of the DGDNN architecture as shown in Fig. 2 [cite: 111] and Eq. 8.
        # It combines generalized graph diffusion for topology learning [cite: 101] with decoupled representation learning[cite: 117].
        for l in range(len(self.diffusion_layers) - 1):
            # 1. Generalized Graph Diffusion Step:
            # Propagates information on the graph using learned diffusion matrices and weights.
            # 'theta_soft[l]' are the learned weights (theta_l,k) for the current layer.
            # 'self.T[l]' are the learned transition matrices (T_l,k) for the current layer.
            # 'h' is the current diffused feature representation.
            # 'A' is the static adjacency matrix used to filter the diffused output, as described in GeneralizedGraphDiffusion.
            h = self.diffusion_layers[l](theta_soft[l], self.T[l], h, A)  # h sized [N, diffusion_size[l+1]]
            # 2. Hierarchical Decoupled Representation Learning (CatMultiAttn Step):
            # Combines the *newly diffused* features (h) with the *hierarchical/original* features (h_prime) using attention.
            # This aims to preserve distinctive local information by decoupling representation transformation and message passing. [cite: 115]
            if l == 0:
                # For the first layer, raw features (h_prime) are projected before attention.
                h_prime = self.cat_attn_layers[l](h, self.raw_h_prime(h_prime))  # [N, embedding_output_size]
            else:
                # For subsequent layers, a residual connection is used (h_prime + attention_output).
                # This helps in preserving hierarchical information across layers. [cite: 37]
                h_prime = h_prime + self.cat_attn_layers[l](h, h_prime)

            


        # --- Final Classification ---
        # The final 'h_prime' (which holds the refined hierarchical and diffused representations) is passed
        # through a linear layer to produce the final classification logits for each node (stock). [cite: 43]
        out = self.linear(h_prime)  # [N, classes]
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