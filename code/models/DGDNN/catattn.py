import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CatMultiAttn(nn.Module):
    def __init__(self, input_time, num_heads, hidden_dim, output_dim, use_activation=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_activation = use_activation

        # Multi-head attention layer
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=input_time,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.out_proj = nn.Linear(input_time, output_dim)

    def forward(self, h: torch.Tensor, h_prime: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with batch support
        Args:
            h: [B, N, diffusion_size]
            h_prime: [B, N, embedding_size]
        Returns:
            [B, N, output_dim]
        """
        batch_size, num_nodes = h.size(0), h.size(1)
        
        # Concatenate along feature dimension
        x = torch.cat([h, h_prime], dim=-1)  # [B, N, diffusion_size + embedding_size]
        
        # Apply multi-head attention
        # Note: MultiheadAttention expects [B, N, C] format
        attn_out, _ = self.multi_head_attn(x, x, x)  # [B, N, input_time]
        
        # Project to output dimension
        out = self.out_proj(attn_out)  # [B, N, output_dim]
        
        if self.use_activation:
            out = F.relu(out)
            
        return out