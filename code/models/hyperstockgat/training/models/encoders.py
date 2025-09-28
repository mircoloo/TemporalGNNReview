"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.hyperstockgat.training.manifolds as manifolds
from models.hyperstockgat.training.layers.att_layers import GraphAttentionLayer
import models.hyperstockgat.training.layers.hyp_layers as hyp_layers
from models.hyperstockgat.training.layers.layers import GraphConvolution, Linear, get_dim_act
import models.hyperstockgat.training.utils.math_utils as pmath


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj.squeeze())  
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
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices)) # N
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices)) # F x N
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels)) # F
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps)) # 1 x T x T
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps)) # T x T

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
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
        self.tat = Temporal_Attention_layer(args.feat_dim, args.n_nodes, int(args.l)) # features x num_nodes x num_timesteps [FxNxT]
        self.tat2 = Temporal_Attention_layer(args.feat_dim, args.n_nodes, int(args.l)) # the second attention, at the end
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
        self.time_conv = nn.Conv2d(int(args.feat_dim), int(args.feat_dim), kernel_size=(1, 3), stride=(1,  1), padding=(0, 1)) #changed to feat_dim from args.l
        self.time_conv2 = nn.Conv2d(int(args.feat_dim), int(args.feat_dim), kernel_size=(1, 3), stride=(1,  1), padding=(0, 1))
    def encode(self, x, adj):
        """
        args: x: (N, F, T) where N is the number of nodes, F is the number of features, T is the number of time steps
                adj: (N, N) adjacency matrix 
        
        """
        #x = x.unsqueeze(0) #add the batch dimension
        

        x = x.permute(0,1,3,2) # (B, N, F, T)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        #print(f"batch_size: {batch_size}, num_of_vertices: {num_of_vertices}, num_of_features: {num_of_features}, num_of_timesteps: {num_of_timesteps}  ")
        
        temporal_At = self.tat(x)
        #print(f"temporal_At shape: {temporal_At.shape}" )
        
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        #print(f"x_TAt before conv {x_TAt.shape=}")
        #print(f"self.time_conv weight shape: {self.time_conv.weight.shape}, bias shape: {self.time_conv.bias.shape}")
        x_TAt_conved = self.time_conv(x_TAt.permute(0, 2, 1, 3))
        #print(f"x_TAt_conved shape: {x_TAt_conved.shape}")
        x_TAt = x_TAt_conved.reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        # (B, N, F, T) -> (B, F, N, T)
        outputs = []
        for time_step in range(num_of_timesteps): # for each timestamp
            y = x_TAt[:,:,:,time_step]
            y = y.reshape((num_of_vertices, num_of_features)) # (N, F)
            x_tan = self.manifold.proj_tan0(y, self.curvatures[0])
            x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
            x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
            #print(f"y shape: {y.shape}, x_tan shape: {x_tan.shape}, x_hyp shape: {x_hyp.shape}")
            temp = super(HGCN, self).encode(x_hyp, adj)
            outputs.append(temp.reshape(1,num_of_vertices,6))
        #print(f"outputs length: {len(outputs)}")
        spatial_At = torch.stack(outputs).permute(1, 0, 2, 3)
        h = spatial_At.permute(0, 2, 3, 1)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = h.shape
        temporal_At = self.tat2(x)
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        x_TAt = self.time_conv2(x_TAt.permute(0, 2, 1, 3)).reshape(batch_size, num_of_timesteps,num_of_vertices, num_of_features)
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
