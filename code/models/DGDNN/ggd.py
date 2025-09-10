import torch.nn as nn
import torch
from torch import Tensor

class GeneralizedGraphDiffusion(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, active: bool):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.PReLU(num_parameters=input_dim) if active else nn.Identity()

    def forward(
        self,
        theta: Tensor,        # [K] - Learned weights (theta_l,k from DGDNN), where K is expansion_step. [cite: 103]
        T_slices: Tensor,     # [K, N, N] - Learned transition matrices (T_l,k from DGDNN). [cite: 103]
        x: Tensor,            # [N, F_in] - Input node features for this diffusion layer.
        a: Tensor             # [N, N] - The original static adjacency matrix. This is used to filter the learned diffusion. [cite: 74, 95]
    ) -> Tensor:              # [N, F_out] - Output features after diffusion and transformation.
        

        # --- Core of Generalized Graph Diffusion (Eq. 6) ---
        # Q_l = sum_{k=0}^{K-1} (theta_l,k * T_l,k) [cite: 103]
        # This performs a weighted linear combination of the learned transition matrices (T_slices)
        # using the learned weights (theta). The result 'q' is the combined diffusion matrix for this layer.
        q = torch.einsum('k,kij->ij', theta, T_slices)  # [N, N] q is the flexible diffusion matrix
        
        # Apply the original adjacency matrix 'a' as a mask or filter.
        # This implies that the learned diffusion 'q' is constrained or guided by the initially constructed graph.
        # The paper mentions that the original graph is 'constructed' and then the model 'learns task-optimal topology'
        # by a generalized diffusion process *on* the constructed graph. [cite: 8, 101]
        # Multiplying by 'a' ensures that connections not present in the original graph (or with zero weight)
        # are also zeroed out in the learned diffusion matrix, or that the learned diffusion is applied only
        # where there's an original connection (if 'a' is binary).
        q = q * a      
        
        # Convert the diffusion matrix 'q' to sparse format for efficient matrix multiplication,
        # especially important for large graphs where 'q' might be sparse.                                 # [N, N]
        q = q.to_sparse()

        # Perform the graph diffusion operation: matrix multiplication of the
        # (sparse) diffusion matrix 'q' with the node features 'x'.
        # This propagates information across nodes according to the learned diffusion.
        out = torch.sparse.mm(q, x)                    # [N, F_in]

        # Apply activation function and final linear projection
        # In the paper formula (8)
        out = self.activation(out)
        out = self.fc(out)                             # [N, F_out]
        return out


# Fast version using GCNConv for efficiency
# from torch_geometric.nn import GCNConv
# import torch.nn as nn
# import torch

# class GeneralizedGraphDiffusion(nn.Module):
#     def __init__(self, input_dim, output_dim, active: bool):
#         super().__init__()
#         self.gconv = GCNConv(input_dim, output_dim)
#         self.activation = nn.PReLU(num_parameters=output_dim) if active else nn.Identity()

#     def forward(
#         self,
#         x: torch.Tensor,                  # [N, F_in]
#         edge_index: torch.Tensor,        # [2, E] COO format
#         edge_weight: torch.Tensor        # [E] edge weights (like q_ij values)
#     ) -> torch.Tensor:
#         x = self.gconv(x, edge_index, edge_weight)  # [N, F_out]
#         x = self.activation(x)
#         return x
