"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import torch.nn.init as init
#import utilities.math_utils as pmath
#from models.HyperStockGAT.training.utilities.math_utils import pmath



class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            if len(adj.shape) != 2:
                adj = adj.squeeze()
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
        self.U1 = nn.Parameter(torch.empty(num_of_vertices))
        init.uniform_(self.U1, a=-0.1, b=0.1)  # initialize U1 in [-0.1, 0.1]

        self.U2 = nn.Parameter(torch.empty(in_channels, num_of_vertices))
        init.xavier_uniform_(self.U2)  # Xavier uniform initialization for 2D weight matrix

        self.U3 = nn.Parameter(torch.empty(in_channels))
        init.uniform_(self.U3, a=-0.1, b=0.1)

        self.be = nn.Parameter(torch.empty(1, num_of_timesteps, num_of_timesteps))
        init.uniform_(self.be, a=-0.1, b=0.1)

        self.Ve = nn.Parameter(torch.empty(num_of_timesteps, num_of_timesteps))
        init.xavier_uniform_(self.Ve)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        batch_size , num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # print(self.U1)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)
        # print('lhs',lhs)
        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
        # print('rhs', rhs)
        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        # print('product', product)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
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
        self.tat = Temporal_Attention_layer(args.feat_dim, args.num_nodes, int(args.l))
        self.tat2 = Temporal_Attention_layer(args.dim, args.num_nodes, int(args.l))
        self.manifold = getattr(manifolds, args.manifold)()
        self.feat_dim = args.feat_dim
        self.num_nodes = args.num_nodes
        self.dim = args.dim
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
        self.time_conv = nn.Conv2d(int(args.l), int(args.l), kernel_size=(1, 3), stride=(1,  1), padding=(0, 1))
        self.time_conv2 = nn.Conv2d(int(args.l), int(args.l), kernel_size=(1, 3), stride=(1,  1), padding=(0, 1))
        
    def encode(self, x, adj):
        
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # Temporal attention network
        temporal_At = self.tat(x) # First temporal attention  
        x_TAt = (x.reshape(batch_size, -1, num_of_timesteps) @  temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        x_TAt = x_TAt.permute(0, 3, 1, 2)
        ########  First Temporal Convolution 
        x_TAt = self.time_conv(x_TAt)
        x_TAt = x_TAt.permute(0,2,3,1) # Added to reshape and put the timestamp as last

        outputs = []
        ########  Spatial Hyperbolic Graph Encoding Loop
        for time_step in range(num_of_timesteps):
            y = x_TAt[:,:,:,time_step]
            y = y.reshape((self.num_nodes, self.feat_dim))
            x_tan = self.manifold.proj_tan0(y, self.curvatures[0])
            x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
            x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
            temp = super(HGCN, self).encode(x_hyp, adj)
            outputs.append(temp.reshape(1,self.num_nodes,self.dim))
        
        spatial_At = torch.stack(outputs, dim=0).permute(1, 0, 2, 3)
        h = spatial_At.permute(0, 2, 3, 1)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = h.shape

        temporal_At = self.tat2(h) # second attention matrix 
        x_TAt = h.reshape(batch_size, -1, num_of_timesteps) @ temporal_At

        x_TAt = x_TAt.reshape(batch_size, num_of_vertices, self.dim, num_of_timesteps)
        ##################  Second Temporal Attention Layer
        x_TAt = self.time_conv2(x_TAt.permute(0, 3, 1, 2)).reshape(batch_size, num_of_timesteps,num_of_vertices, num_of_features)
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
                dims[0] = args.dim_feat + weights.shape[1]
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