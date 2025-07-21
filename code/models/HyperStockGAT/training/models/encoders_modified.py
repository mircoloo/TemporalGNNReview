"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utilities.math_utils as pmath


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
        

        lhs = x.permute(0, 3, 2, 1) @ self.U1 @  self.U2

        if lhs.isnan().any():
            x_permuted = x.permute(0, 3, 2, 1)
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
        self.n_nodes = args.n_nodes
        self.output_dim = args.dim # This is the final output dimension of the GNN part
        
        # args.feat_dim is the feature dimension PER NODE, args.l is the number of time steps
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
        
        # Conv2d needs (B, C, H, W). If C is Features, and W is Time, then kernel_size=(1,3) makes sense.
        # This implies Features are the channels, and the convolution happens across Timesteps.
        self.time_conv = nn.Conv2d(int(args.feat_dim), int(args.feat_dim), kernel_size=(1, 3), stride=(1,  1), padding=(0, 1), bias=True)
        self.time_conv2 = nn.Conv2d(int(args.feat_dim), int(args.feat_dim), kernel_size=(1, 3), stride=(1,  1), padding=(0, 1), bias=True)


    def encode(self, x, adj):
        # Initial `x` from BaseModel is (N_nodes, feat_dim_actual, timesteps)
        # feat_dim_actual is (args.feat_dim + 1) if Hyperboloid.
        # timesteps is args.l.
        
        x = x.unsqueeze(0) # (1, N_nodes, feat_dim_actual, timesteps)
        # No further permute needed if we assume (Batch, Nodes, Features, Timesteps)
        # This aligns with the expected input format for Temporal_Attention_layer (B, N, F_in, T)

        print(" - in encode of HGCN x shape", x.shape, "adj shape", adj.shape)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # Here: num_of_features = feat_dim_actual (e.g., 6), num_of_timesteps = args.l (e.g., 10)
        
        print(" - batch_size", batch_size, "num_of_vertices", num_of_vertices, "num_of_features", num_of_features, "num_of_timesteps", num_of_timesteps)
        
        temporal_At = self.tat(x) # x is (B,N,F,T), tat expects this and returns (B,T,T)
        print(" - temporal_At shape", temporal_At.shape) # (B, T, T)

        # x.reshape(batch_size, -1, num_of_timesteps) makes it (B, N*F, T)
        # Matmul (B, N*F, T) @ (B, T, T) -> (B, N*F, T)
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At)
        x_TAt = x_TAt.reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps) # Back to (B, N, F, T)
        
        print(" - x_TAt shape before time_conv", x_TAt.shape)
        
        # Permute to (B, C, H, W) for Conv2d: (B, Features, Nodes, Timesteps)
        # This aligns with self.time_conv(in_channels=args.feat_dim)
        x_TAt = self.time_conv(x_TAt.permute(0, 2, 1, 3))
        # Now x_TAt is (B, Features, Nodes, Timesteps) after conv.
        
        # Reshape back to (B, N, F, T) for the loop
        x_TAt_for_loop = x_TAt.permute(0, 2, 1, 3) 
        print(" - x_TAt shape after time_conv and permute", x_TAt_for_loop.shape)

        outputs = []
        for time_step in range(num_of_timesteps): # Iterating args.l times
            y = x_TAt_for_loop[:,:,:,time_step] # (Batch, N, Features)
            
            # FIX: Use dynamic num_of_features from `x.shape`
            # `num_of_features` here is `feat_dim_actual` (args.feat_dim + 1)
            y = y.reshape((self.n_nodes, num_of_features)) 
            
            x_tan = self.manifold.proj_tan0(y, self.curvatures[0])
            x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
            x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

            temp = super(HGCN, self).encode(x_hyp, adj) # Output is (N_nodes, self.output_dim)
            
            # FIX: Use self.output_dim (which is args.dim, e.g., 32)
            outputs.append(temp.reshape(1, self.n_nodes, self.output_dim))
        
        spatial_At = torch.stack(outputs).permute(1, 0, 2, 3) # (Batch, Timesteps, Nodes, Final_Dim)
        h = spatial_At.permute(0, 2, 3, 1) # (Batch, Nodes, Final_Dim, Timesteps)

        print("2 - h shape after first block", h.shape) # (B, N, args.dim, args.l)

        # Second temporal attention block. `x` here is the original input `(1, N, F, T)`
        temporal_At = self.tat2(x) # x is (B,N,F,T), tat returns (B,T,T)
        print("2 - temporal_At shape (second block)", temporal_At.shape)

        # x.reshape(batch_size, -1, num_of_timesteps) makes it (B, N*F, T)
        x_TAt_tmp = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At)
        x_TAt_2 = x_TAt_tmp.reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps) # Back to (B, N, F, T)
        
        print("2 - x_TAt_2 shape before time_conv2", x_TAt_2.shape)

        # Permute to (B, C, H, W) for Conv2d: (B, Features, Nodes, Timesteps)
        x_TAt_2_convolved = self.time_conv2(x_TAt_2.permute(0, 2, 1, 3)) 
        # Output is (B, Features, Nodes, Timesteps)
        
        # Your final return is (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
        # This implies (B, T, N, F)
        # So you need to permute `x_TAt_2_convolved` (B, F, N, T) to (B, T, N, F)
        final_output = x_TAt_2_convolved.permute(0, 3, 2, 1) 
        print(f"2 - final_output shape {final_output.shape=}")
        return final_output




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
