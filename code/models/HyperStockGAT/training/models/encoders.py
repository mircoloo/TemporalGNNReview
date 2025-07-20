"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True


class Temporal_Attention_layer(nn.Module):
    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        # --- Corrected Parameter Initializations ---
        # U1: Often for a vector, Glorot/Xavier uniform is a good starting point.
        # Shape: (num_of_vertices,)
        self.U1 = nn.Parameter(torch.empty(num_of_vertices))
        nn.init.xavier_uniform_(self.U1.unsqueeze(0)) # Unsqueeze to make it 2D for xavier_uniform

        # U2: This looks like a weight matrix for a linear transformation.
        # Shape: (in_channels, num_of_vertices)
        self.U2 = nn.Parameter(torch.empty(in_channels, num_of_vertices))
        nn.init.xavier_uniform_(self.U2) # Xavier/Glorot is suitable for linear layers

        # U3: Often for a vector, Glorot/Xavier uniform.
        # Shape: (in_channels,)
        self.U3 = nn.Parameter(torch.empty(in_channels))
        nn.init.xavier_uniform_(self.U3.unsqueeze(0)) # Unsqueeze to make it 2D

        # be: Often for a bias or a specific learnable tensor.
        # Shape: (1, num_of_timesteps, num_of_timesteps)
        self.be = nn.Parameter(torch.empty(1, num_of_timesteps, num_of_timesteps))
        nn.init.xavier_uniform_(self.be) # Initialize as a matrix

        # Ve: Another weight matrix for a linear transformation or attention.
        # Shape: (num_of_timesteps, num_of_timesteps)
        self.Ve = nn.Parameter(torch.empty(num_of_timesteps, num_of_timesteps))
        nn.init.xavier_uniform_(self.Ve) # Xavier/Glorot is suitable

        # Optional: You can also use zeros for biases if that fits your design
        # nn.init.zeros_(self.be) # if be is meant to be an additive bias starting at zero

        

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # print(self.U1)
        print(" -- num of vertices", num_of_vertices, "num_of_features", num_of_features, "num_of_timesteps", num_of_timesteps)
        print(" -- U1 shape", self.U1.shape, "U2 shape", self.U2.shape, "U3 shape", self.U3.shape, "be shape", self.be.shape, "Ve shape", self.Ve.shape)
        

        print(f"Temporal attention network x nan? {x.isnan().any()}")
        lhs = x.permute(0, 3, 2, 1) @ self.U1 @  self.U2

        print(f"Temmporal attention layer nan? {lhs.isnan().any()}")
        if lhs.isnan().any():
            print(x.permute(0, 3, 2, 1).isnan().any(), self.U1.isnan().any(), self.U2.isnan().any())
            x_permuted = x.permute(0, 3, 2, 1)
            print(f"x_permuted - min: {x_permuted.min().item():.6e}, max: {x_permuted.max().item():.6e}")
            print(f"U1 - min: {self.U1.min().item():.6e}, max: {self.U1.max().item():.6e}")
            print(f"U2 - min: {self.U2.min().item():.6e}, max: {self.U2.max().item():.6e}")
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)
        # print('lhs',lhs)
        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
        # print('rhs', rhs)
        print(f"{rhs.shape=}")
        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        # print('product', product)
        print(f"{product.shape=}")

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        print(f"{E.shape=}")
        # print('E', E)
        E_normalized = F.softmax(E, dim=1)
        # print('E_norm', E_normalized)
        return E_normalized

def normalize(input):    
    input += 1e-5  #For Numerical Stability
    stdv = torch.std(input)
    input = (input - torch.mean(input)) #/ np.std(input)        #0 mean 1 std
    input = input / stdv #np.max(abs(input))
    if torch.isnan(torch.sum(input)):
        print("[Nan Values in Normalize is ::]", torch.isnan(torch.sum(input)))
    return input

class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        # self.grup = gru(5,32)
        # self.attention_temp = Attention(32)
        self.n_nodes = args.n_nodes
        self.tat = Temporal_Attention_layer(args.feat_dim, self.n_nodes, int(args.l))
        self.tat2 = Temporal_Attention_layer(args.feat_dim, self.n_nodes, int(args.l))
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.time_conv = nn.Conv2d(int(args.feat_dim), int(args.feat_dim), kernel_size=(1, 3), stride=(1,  1), padding=(0, 1), bias=True)
        self.time_conv2 = nn.Conv2d(int(args.feat_dim), int(args.feat_dim), kernel_size=(1, 3), stride=(1,  1), padding=(0, 1), bias=True)


    def encode(self, x, adj):
        
        x = x.unsqueeze(0)
        x = x.permute(0,1,3,2)
        print(" - in encode of HGCN x shape", x.shape, "adj shape", adj.shape)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        print(" - batch_size", batch_size, "num_of_vertices", num_of_vertices, "num_of_features", num_of_features, "num_of_timesteps", num_of_timesteps)
        temporal_At = self.tat(x)
        print(" - temporal_At shape", temporal_At.shape)
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        print(" - x_TAt shape before time_conv", x_TAt.permute(0, 2, 1, 3).shape)
        x_TAt = self.time_conv(x_TAt.permute(0, 2, 1, 3)).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        #x_TAt = self.time_conv(x_TAt.permute(0, 3, 1, 2)).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        
        print(" - x_TAt shape after time_conv", x_TAt.shape)
        outputs = []
        for time_step in range(num_of_timesteps):
            y = x_TAt[:,:,:,time_step]
            y = y.reshape((self.n_nodes,5))
            x_tan = self.manifold.proj_tan0(y, self.curvatures[0])
            x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
            x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

            temp = super(HGCN, self).encode(x_hyp, adj)
            outputs.append(temp.reshape(1,self.n_nodes, 5))
        spatial_At = torch.stack(outputs).permute(1, 0, 2, 3)
        h = spatial_At.permute(0, 2, 3, 1)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = h.shape
        num_of_features = 5
        print("2 - batch_size", batch_size, "num_of_vertices", num_of_vertices, "num_of_features", num_of_features, "num_of_timesteps", num_of_timesteps)
        print("2 - x.shape", x.shape, "adj shape", adj.shape)
        temporal_At = self.tat2(x)
        print("2 - temporal_At shape", temporal_At.shape, "x.reshape(batch_size, -1, num_of_timesteps)", x.reshape(batch_size, -1, num_of_timesteps).shape)
        x_TAt_tmp = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At)
        print(f"2 -  {x_TAt_tmp.shape=}")
        print("2 - num_of_vertices", num_of_vertices, "num_of_features", num_of_features, "num_of_timesteps", num_of_timesteps)
        x_TAt = x_TAt_tmp.reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        x_TAt = self.time_conv2(x_TAt.permute(0, 2, 1, 3)).reshape(batch_size, num_of_timesteps,num_of_vertices, num_of_features)
        print(f"2 - {x_TAt.shape=}")
        return x_TAt



class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)
