"""Graph decoders."""
import models.hyperstockgat.training.manifolds as manifolds
import torch.nn as nn
import torch.nn.functional as F

from models.hyperstockgat.training.layers.att_layers import GraphAttentionLayer
from models.hyperstockgat.training.layers.layers import GraphConvolution, Linear


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class LinearDecoder2(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False
        self.time_conv = nn.Conv2d(int(args.l), 1, kernel_size=(1, 3), stride=(1,  1), padding=(0, 1))
    def decode2(self, x, adj):
        print(f"LinearDecoder decode input {x.shape=} {adj.shape=}")
        print(f"Manifold: {self.manifold.name}")
        #x = x.permute(0,2,1,3) # to remove maybe
        #x = x[..., -2:]
        h_tangent = self.manifold.logmap0(x, c=self.c)
        print(f"{h_tangent.shape=}")
        h = self.manifold.proj_tan0(h_tangent, c=self.c)
        h = self.time_conv(h)
        h = h.squeeze(0).squeeze(0)
        return F.leaky_relu(super(LinearDecoder, self).decode(h, adj))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )

class LinearDecoder(Decoder):
    """
    Temporal Conv + Linear Decoder for binary node classification (up/down).
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()

        # Input: F_in = feature dim after encoder
        self.input_dim = args.dim
        self.output_dim = args.n_classes  # usually 1 for binary classification
        self.bias = args.bias

        # Temporal conv over time dimension
        # Input will be permuted to (B, F_in, N, T)
        self.time_conv = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.input_dim,  # keep same feature dim, can set larger
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1)
        )

        # Final linear classifier per node
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)

        self.decode_adj = False  # we don’t use adj here

    def decode(self, x, adj):
        """
        x: Z with shape (B, T, N, F_in)
        adj: unused here (decoder does not need adjacency)
        returns: logits shape (B, N, n_classes)
        """
        B, T, N, F_in = x.shape

        # 1) Prepare for Conv2d: (B, F_in, N, T)
        x_perm = x.permute(0, 3, 2, 1).contiguous()

        # 2) Temporal conv
        conv_out = self.time_conv(x_perm)   # (B, F_in, N, T)

        # 3) Aggregate over time (mean pooling)
        conv_mean = conv_out.mean(dim=-1)   # (B, F_in, N)

        # 4) Permute back: (B, N, F_in)
        node_feats = conv_mean.permute(0, 2, 1).contiguous()

        # 5) Flatten per-node
        node_feats_flat = node_feats.view(B * N, -1)

        # 6) Linear classifier → logits
        logits_flat = self.cls(node_feats_flat)   # (B*N, n_classes)
        logits = logits_flat.view(B, N, -1)

        return logits

    def extra_repr(self):
        return f'in_features={self.input_dim}, out_features={self.output_dim}, bias={self.bias}, c={self.c}'


model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
}
