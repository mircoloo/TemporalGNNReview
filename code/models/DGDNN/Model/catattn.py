import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CatMultiAttn(nn.Module):
    def __init__(
        self,
        input_time: int,       # T1 + T2: This is the combined feature dimension after concatenating 'h' (diffused) and 'h_prime' (hierarchical/original). 
        num_heads: int,        # Number of attention heads for MultiheadAttention.
        hidden_dim: int,       # Hidden dimension for the projection layers within the attention block.
        output_dim: int,       # Final output dimension of this attention module.
        use_activation: bool   # Whether to apply GELU activation in the projection layers.
    ):
        """
        Args:
            input_time (int): Combined time dimension after concatenation (T1 + T2)
            num_heads (int): Number of attention heads
            hidden_dim (int): Hidden output dimension (E_h)
            use_activation (bool): Whether to apply GELU activation
        """
        super().__init__()
        self.use_activation = use_activation

        self.attn = nn.MultiheadAttention(embed_dim=input_time, num_heads=num_heads)
        self.norm = nn.LayerNorm(input_time)  # Apply norm on attention output

        self.proj = nn.Sequential(
            nn.Linear(input_time, hidden_dim),
            nn.GELU() if use_activation else nn.Identity(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, h: Tensor, h_prime: Tensor) -> Tensor:
        """
        Args:
            h (Tensor): [N, T1]
            h_prime (Tensor): [N, T2]

        Returns:
            Tensor: [N, output_dim] — per-series representation
        """
        assert h.shape[0] == h_prime.shape[0], "Number of time series (N) must match."
        x = torch.cat([h, h_prime], dim=1)              # [N,  diffusion_size[l+1] (h) + embedding_output_size[l] (h_prime)]
        x = x.unsqueeze(1).transpose(0, 1)              # [1, N, T]

        attn_out, _ = self.attn(x, x, x)                # [1, N, T]
        attn_out = self.norm(attn_out)                  # [1, N, T]

        x = attn_out.squeeze(0)                         # [N, T]
        x = self.proj(x)                                # [N, embedding_output_size[l]]
        return x