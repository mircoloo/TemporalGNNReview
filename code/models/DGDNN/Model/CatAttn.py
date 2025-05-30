import torch
import torch.nn as nn


class CatMultiAttn(torch.nn.Module):
    def __init__(self, embedding, num_heads, output, active, timestamp):
        super(CatMultiAttn, self).__init__()
        self.attn_layer = nn.MultiheadAttention(embedding, num_heads)
        self.linear_layer = nn.Linear(embedding*timestamp, output*timestamp)
        self.activation = nn.PReLU()
        self.active = active
        self.timestamp = timestamp
        self.layernorm = nn.functional.layer_norm

    def forward(self, h, h_prime):
        h = torch.cat((h, h_prime), dim=1).view(self.timestamp, h.shape[0], -1)
        h, _ = self.attn_layer(h, h, h)
        h = h.reshape(h.shape[1], -1)
        h = self.linear_layer(h)
        #print(f"H shape in CatAttn: {h.shape}") ### Added check because in 30/05/25 gived an error on inserting 'normalized_shape' patameter below
        h = self.layernorm(h, normalized_shape=[h.shape[1]])
        if self.active:
            h = self.activation(h)
            return h
        else:
            return h
