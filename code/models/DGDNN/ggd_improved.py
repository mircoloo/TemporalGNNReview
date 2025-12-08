"""
Improved version of Generalized Graph Diffusion (GGD) module
with better gradient flow for theta parameters

This version includes multiple options for handling adjacency masking
to ensure gradients flow properly to theta weights.
"""

import torch.nn as nn
import torch
from torch import Tensor

class GeneralizedGraphDiffusion(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, active: bool, 
                 masking_mode: str = 'soft', masking_alpha: float = 1.0):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            active: Whether to use activation function
            masking_mode: How to apply adjacency masking
                - 'none': No masking (learn full diffusion matrix)
                - 'hard': Hard masking q = q * a (original, blocks gradients)
                - 'soft': Soft masking q = q * (1 + alpha * a) (recommended)
                - 'blend': Blend masking q = (1-beta)*q + beta*(q*a)
                - 'additive': Additive masking q = q + alpha * a
            masking_alpha: Hyperparameter for soft/additive masking (default: 1.0)
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.PReLU(num_parameters=input_dim) if active else nn.Identity()
        
        # Masking configuration
        self.masking_mode = masking_mode
        self.masking_alpha = masking_alpha
        
        # For blend mode
        if masking_mode == 'blend':
            self.blend_beta = nn.Parameter(torch.tensor(0.5))  # Learnable blend factor

    def forward(
        self,
        theta: Tensor,        # [K] - Learned weights (theta_l,k from DGDNN)
        T_slices: Tensor,     # [K, N, N] - Learned transition matrices (T_l,k from DGDNN)
        x: Tensor,            # [N, F_in] - Input node features
        a: Tensor             # [N, N] - Original adjacency matrix
    ) -> Tensor:              # [N, F_out] - Output features
        
        # Core diffusion: weighted combination of T matrices
        # Q_l = sum_{k=0}^{K-1} (theta_l,k * T_l,k)
        q = torch.einsum('k,kij->ij', theta, T_slices)  # [N, N]
        
        # Apply adjacency masking based on selected mode
        if self.masking_mode == 'none':
            # No masking - learn completely free diffusion matrix
            # Best gradient flow but may ignore graph structure
            q_final = q
            
        elif self.masking_mode == 'hard':
            # Original implementation - hard masking
            # Blocks gradients where a=0
            q_final = q * a
            
        elif self.masking_mode == 'soft':
            # Soft masking - adjacency acts as multiplier
            # Good gradient flow while respecting graph structure
            # When alpha=1: edges get 2x weight, non-edges get 1x weight
            q_final = q * (1.0 + self.masking_alpha * a)
            
        elif self.masking_mode == 'blend':
            # Blend between free and masked diffusion
            # Learnable blend factor beta
            beta = torch.sigmoid(self.blend_beta)  # Keep in [0,1]
            q_masked = q * a
            q_final = (1.0 - beta) * q + beta * q_masked
            
        elif self.masking_mode == 'additive':
            # Additive influence - adjacency as bias
            # Preserves all gradients while incorporating structure
            q_final = q + self.masking_alpha * a
            
        else:
            raise ValueError(f"Unknown masking_mode: {self.masking_mode}")
        
        # Convert to sparse for efficient multiplication
        q_sparse = q_final.to_sparse()
        
        # Perform graph diffusion
        out = torch.sparse.mm(q_sparse, x)  # [N, F_in]
        
        # Apply activation and projection
        out = self.activation(out)
        out = self.fc(out)  # [N, F_out]
        
        return out


# Original version for reference
class GeneralizedGraphDiffusion_Original(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, active: bool):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.PReLU(num_parameters=input_dim) if active else nn.Identity()

    def forward(
        self,
        theta: Tensor,
        T_slices: Tensor,
        x: Tensor,
        a: Tensor
    ) -> Tensor:
        q = torch.einsum('k,kij->ij', theta, T_slices)
        q = q * a  # ⚠️ This blocks gradients!
        q = q.to_sparse()
        out = torch.sparse.mm(q, x)
        out = self.activation(out)
        out = self.fc(out)
        return out
