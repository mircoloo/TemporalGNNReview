import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# --- Sezione 1: Utilità e Placeholder ---
# Sostituzioni minimali per le dipendenze esterne (scaler e util.loss)

def unmasked_mae(preds, labels, null_val=0.0):
    """
    Mean Absolute Error non mascherato.
    Sostituisce util.masked_mae per questo esempio minimale.
    """
    # Se null_val non è 0, lo ignoriamo per semplicità in questo esempio
    loss = torch.abs(preds - labels)
    return torch.mean(loss)

class DummyScaler:
    """
    Classe fittizia per sostituire lo scaler dei dati reali.
    Non fa nulla, restituisce solo i dati così come sono.
    """
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

# --- Sezione 2: Definizione del Modello GraphWaveNet ---

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h 

class gwnet(nn.Module):
    """
    The main GraphWaveNet model architecture.
    Combines dilated temporal convolutions, graph convolutions, and skip connections
    to capture spatio-temporal dependencies in graph-structured time series data.
    """
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout # Dropout rate for regularization
        self.blocks = blocks # Number of spatio-temporal blocks
        self.layers = layers # Number of dilated convolution layers within each block
        self.gcn_bool = gcn_bool # Flag to enable/disable Graph Convolutional Network (GCN) layers
        self.addaptadj = addaptadj # Flag to enable/disable adaptive adjacency matrix learning

        # ModuleLists to store layers for each block/layer iteration
        self.filter_convs = nn.ModuleList() # Dilated convolutions for the 'filter' path
        self.gate_convs = nn.ModuleList() # Dilated convolutions for the 'gate' path
        self.residual_convs = nn.ModuleList() # 1x1 convolutions for residual connections (if no GCN)
        self.skip_convs = nn.ModuleList() # 1x1 convolutions for skip connections
        self.bn = nn.ModuleList() # Batch normalization layers
        self.gconv = nn.ModuleList() # GCN layers

        # Initial 1x1 convolution to project input features to residual_channels
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        #print(self, device, num_nodes, dropout, supports, gcn_bool, addaptadj, aptinit, in_dim, out_dim, residual_channels, dilation_channels, skip_channels, end_channels, kernel_size, blocks, layers)
        self.supports = supports # Pre-defined adjacency matrices (if any)
        receptive_field = 1 # Tracks the total receptive field of temporal convolutions
        self.supports_len = 0 # Number of support matrices
        if supports is not None:
            self.supports_len += len(supports)

        # Adaptive Adjacency Matrix Learning (if enabled)
        # This mechanism learns a graph structure directly from the data
        if gcn_bool and addaptadj:
            if aptinit is None: # If no initial adaptive adjacency is provided
                if supports is None: # Initialize self.supports if it's empty
                    self.supports = []
                # Learnable node embeddings (E1 and E2 in the paper, formula 5)
                # These are multiplied to form the adaptive adjacency matrix
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device) 
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device) 
                self.supports_len += 1 # Increment support length for the adaptive matrix
            # The 'aptinit' part from the original paper is omitted for brevity as per user's input

        # Building the stacked spatio-temporal blocks
        for b in range(blocks): # Loop through each block
            additional_scope = kernel_size - 1 # Additional receptive field added by current dilated conv
            new_dilation = 1 # Initial dilation rate for the first layer in a block
            for i in range(layers): # Loop through layers within each block
                # Dilated Causal Convolution layers for capturing temporal patterns
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                
                # 1x1 convolution for the residual connection path
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))

                # 1x1 convolution for the skip connection path
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                
                self.bn.append(nn.BatchNorm2d(residual_channels)) # Batch Normalization
                
                new_dilation *= 2 # Double dilation rate for the next layer (exponential growth)
                receptive_field += additional_scope # Update total receptive field
                additional_scope *= 2 # Update additional scope for next iteration
                
                if self.gcn_bool: # If GCN layers are enabled
                    # Append a GCN layer to process spatial dependencies
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        # Output layers to project features to the final output dimension
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field # Total effective receptive field of the network

    def forward(self, input):
        #print(f"{input.shape=}" )
        in_len = input.size(3) # Get input sequence length (time dimension)
        # Pad input if its length is less than the network's receptive field
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        
        # Initial transformation of input features
        x = self.start_conv(x) 
        #print(f"{x.shape=} after start conv (should remain the same)")
        skip = 0 # Initialize skip connection accumulation
        new_supports = None # Placeholder for combined support matrices
        
        # Create or update adaptive adjacency matrix
        if self.gcn_bool and self.addaptadj:
            # Generate adaptive adjacency matrix (A_adp) from learned node embeddings
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            if self.supports is None:
                 new_supports = [adp] # If no fixed supports, adaptive is the only one
            else:
                 new_supports = self.supports + [adp] # Combine fixed supports with adaptive

        # Main loop for spatio-temporal processing
        for i in range(self.blocks * self.layers): # Iterate through each combined block-layer
            residual = x # Store input for residual connection
            
            # Dilated Causal Convolution (Temporal part)
            filter = self.filter_convs[i](residual) # Filter path convolution
            filter = torch.tanh(filter) # Apply tanh activation
            
            gate = self.gate_convs[i](residual) # Gate path convolution
            gate = torch.sigmoid(gate) # Apply sigmoid activation

            #print(f"{filter.shape=} {gate.shape=}")

            x = filter * gate # Gating mechanism: element-wise product of filter and gate outputs
                               # This acts as a temporal attention mechanism

            # Skip Connection
            s = self.skip_convs[i](x) # Transform x for skip connection
            try:
                # Align skip connection length with current output length
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0 # If it's the first skip connection
            skip = s + skip # Accumulate skip connections

            # Spatial GCN Layer or Direct Residual Path
            if self.gcn_bool and (self.supports is not None or self.addaptadj):
                # Apply Graph Convolution using the (newly formed) support matrices
                x = self.gconv[i](x, new_supports) if self.addaptadj else self.gconv[i](x, self.supports) 
            else:
                # If GCN is not used or no supports, apply a simple 1x1 convolution
                x = self.residual_convs[i](x) 
            
            # Residual Connection
            # Add the original input (residual) to the processed output, aligning lengths
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x) # Apply Batch Normalization

        # Final output layers
        x = F.relu(skip) # Apply ReLU to the accumulated skip connections
        x = F.relu(self.end_conv_1(x)) # First end layer with ReLU
        x = self.end_conv_2(x) # Second end layer to produce the final output
        return x