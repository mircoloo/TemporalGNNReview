import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True) # why use kernel size 1x1?

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        # c_in: input channels, c_out: output channels, dropout: dropout rate,
        # support_len: number of adjacency matrices (supports), order: Chebyshev polynomial order (k)
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        # x: input features [batch_size, channels, num_nodes, sequence_length]
        # support: list of adjacency matrices (e.g., A, A_normalized, A_adaptive)
        out = [x] # Start with the original features
        for a in support: # Iterate through each adjacency matrix (support)
            x1 = self.nconv(x,a) # First order diffusion (X * A)
            out.append(x1)
            # Higher order diffusion (Chebyshev approximation of graph convolution)
            # This implements the powers of A: A^2*X, A^3*X, etc., up to order 'self.order'.
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a) # X * A^k (recursively apply A)
                out.append(x2)
                x1 = x2 # Update x1 for the next power of A

        # Concatenate features from all orders of diffusion and all support matrices
        # E.g., for 1 support and order=2: [X, XA, XA^2]
        h = torch.cat(out,dim=1) # Concatenate along the channel dimension
        h = self.mlp(h) # Apply linear transformation (1x1 conv)
        h = F.dropout(h, self.dropout, training=self.training) # Apply dropout for regularization
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks # Number of WaveNet blocks
        self.layers = layers # Number of dilated convolution layers within each block
        self.gcn_bool = gcn_bool # Flag to enable/disable GCN layers
        self.addaptadj = addaptadj # Flag to enable/disable adaptive adjacency matrix learning

        # ModuleLists to hold the sequential layers for WaveNet and GCN components
        self.filter_convs = nn.ModuleList()  # For the 'filter' part of gated activation
        self.gate_convs = nn.ModuleList()    # For the 'gate' part of gated activation
        self.residual_convs = nn.ModuleList() # 1x1 convs for residual connections
        self.skip_convs = nn.ModuleList()    # 1x1 convs for skip connections
        self.bn = nn.ModuleList()            # Batch normalization layers
        self.gconv = nn.ModuleList()         # Graph Convolution (gcn) layers

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1  # Tracks the total receptive field size for temporal convolutions

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        # --- Adaptive Adjacency Matrix Learning (Section 3.1.1 "Adaptive Graph Convolution") ---
        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        # input: [batch_size, in_dim, num_nodes, sequence_length]

        in_len = input.size(3) # Current input sequence length
        # Pad input if its length is less than the model's total receptive field.
        # This ensures all layers have enough context. Padding is on the left (past).
        if in_len < self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        
        x = self.start_conv(x) # Initial projection of input features
        skip = 0 # Accumulator for skip connections

        # --- Adaptive Adjacency Matrix Calculation (Fig. 1 and Section 3.1.1) ---
        # The adaptive matrix is calculated once per forward pass.
        new_supports = None
        if self.gcn_bool and self.addaptadj:
            # A_adp = softmax(ReLU(e1 * e2))
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            # Combine static supports with the newly learned adaptive support
            if self.supports is None: # Handle case where supports was initialized empty
                 new_supports = [adp]
            else:
                 new_supports = self.supports + [adp]

        # --- Stacked Dilated Convolution Blocks (WaveNet Component) ---
        # Iterates through each (block, layer) combination
        for i in range(self.blocks * self.layers):
            # Diagram from WaveNet:
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + --> *input* (next layer's input)
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + -------------> *skip* (accumulated for final output)

            residual = x # Current input for the block becomes the residual connection's input

            # Gated Activation Unit (Section 3.1.3 "Dilated Causal Convolution Layer")
            # This is a common WaveNet component to control information flow.
            filter = self.filter_convs[i](residual) # Convolution for filter (tanh branch)
            filter = torch.tanh(filter) # Apply tanh
            gate = self.gate_convs[i](residual) # Convolution for gate (sigmoid branch)
            gate = torch.sigmoid(gate) # Apply sigmoid
            x = filter * gate # Element-wise multiplication of filter and gate outputs

            # --- Parametrized Skip Connection (Section 3.1.3) ---
            # s = x (output of gated activation)
            s = self.skip_convs[i](x) # 1x1 conv for skip connection path
            
            # This 'try-except' block is a bit unusual. It seems to handle the first iteration where 'skip' is 0.
            # For subsequent iterations, it ensures that 'skip' (the accumulated output) is truncated to match
            # the current 's' in terms of sequence length before addition.
            try:
                # Truncate 'skip' to match the current sequence length of 's'
                skip = skip[:, :, :,  -s.size(3):]
            except:
                # If skip is 0 (first iteration) or causes error, initialize it with 0
                skip = 0
            skip = s + skip # Accumulate skip connections

            # --- Spatial Graph Convolution (Section 3.1.2) ---
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    # If adaptive adjacency is used, pass the combined supports (static + adaptive)
                    x = self.gconv[i](x, new_supports)
                else:
                    # Otherwise, use only the static supports
                    x = self.gconv[i](x,self.supports)
            else:
                # If GCN is not used, just apply a 1x1 conv to maintain residual_channels.
                # This makes it a pure WaveNet-like temporal model.
                x = self.residual_convs[i](x) # Re-uses the residual_convs from WaveNet part for this path

            # --- Residual Connection (Section 3.1.3) ---
            # Adds the input 'residual' (from before gated convs) to the output 'x' (after GCN/1x1 conv).
            # The '[:, :, :, -x.size(3):]' part ensures sequence length matching for the addition.
            x = x + residual[:, :, :, -x.size(3):]

            # Batch Normalization
            x = self.bn[i](x) # Batch normalization after each block's output

        # --- Final Output Layers ---
        # The accumulated skip connections are passed through final 1x1 convolutions.
        # This aggregates information from all layers.
        x = F.relu(skip) # Apply ReLU to the accumulated skip connections
        x = F.relu(self.end_conv_1(x)) # First end conv
        x = self.end_conv_2(x) # Final end conv to produce output_dim predictions

        return x





