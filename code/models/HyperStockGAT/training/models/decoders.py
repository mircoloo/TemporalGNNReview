"""Graph decoders."""
import manifolds
import torch.nn as nn
import torch.nn.functional as F
import torch
from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear


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


class LinearDecoder(Decoder):
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

    def decode(self, x, adj):
        # --- Inspection before hyperbolic ops ---
        if torch.isnan(x).any():
            print("ALERT: NaNs found in input 'x' to decode!")
        if torch.isinf(x).any():
            print("ALERT: Infs found in input 'x' to decode!")

        logmap = self.manifold.logmap0(x, c=self.c)
       
        if torch.isnan(logmap).any():
            print("!!! ALERT: NaNs found in 'h' AFTER hyperbolic ops (BEFORE time_conv) !!!")
        if torch.isinf(logmap).any():
            print("!!! ALERT: Infs found in 'h' AFTER hyperbolic ops (BEFORE time_conv) !!!")
        h = self.manifold.proj_tan0(logmap, c=self.c)

        # --- Inspection after hyperbolic ops (this 'h' is input to time_conv) ---
        if torch.isnan(h).any():
            print("!!! ALERT: NaNs found in 'h' AFTER hyperbolic ops (BEFORE time_conv) !!!")
        if torch.isinf(h).any():
            print("!!! ALERT: Infs found in 'h' AFTER hyperbolic ops (BEFORE time_conv) !!!")

        # ... rest of your code from here
        h = self.time_conv(h) # This is the problematic line

        if torch.isnan(h).any():
            print("ALERT: NaNs found in output of time_conv!")
        if torch.isinf(h).any():
            print("ALERT: Infs found in output of time_conv!")
        h = h.squeeze(0).squeeze(0)
        return F.leaky_relu(super(LinearDecoder, self).decode(h, adj))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )


model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
}
