import torch
import torch.nn as nn

import torch.nn.functional as F
from .GGD import GeneralizedGraphDiffusion
from .CatAttn import CatMultiAttn

class DGDNN(nn.Module):
    def __init__(self, diffusion_size: list # e.g., [F0, F1, F2]
                 ,embedding_size: list # e.g., [F1+F1, E1, E1+F2, E2, ...]
                 , classes
                 ,layers
                 , num_nodes
                 , expansion_step
                 , num_heads
                 , active
                 , timestamp
                 ):
        #assert len(embedding_size) == 2*layers, f"len of emb_size must have 2* the num_layers, current len_emb_size:{len(embedding_size)} instead of {layers*2}"
        assert len(diffusion_size) == layers + 1, f"The length of diffusion size must be one more than the length of layers. Obtained {len(diffusion_size)} and expected {len(layers)+1}"
        assert diffusion_size[0] == 5*timestamp, f"First diffusion size dimension must match timestamp*num_features, expected {timestamp*5} got {diffusion_size[0]}"
        assert len(embedding_size) == 2*layers, f"The length of embedding sizes must be 2*num_layers, one for input and one for output dimension... Expected {2*layers} got {len(embedding_size)}"

        super().__init__()

        # T =  transition matrix params [1 x layer, expansion_step, num_nodes, num_nodes]
        # theta = weight coefficents [1 x layer, expansion step]
        self.T     = nn.Parameter(torch.empty(layers,
                                              expansion_step,
                                              num_nodes,
                                              num_nodes))
        self.theta = nn.Parameter(torch.empty(layers, expansion_step))

        # Initialize the diffusion layers other modules
        #input dim = diffusionsize[i], ouput dim = diffusionsize[i+1]
        self.diffusion_layers = nn.ModuleList(
            [GeneralizedGraphDiffusion(diffusion_size[i],
                                       diffusion_size[i + 1],
                                       active[i])
             for i in range(len(diffusion_size) - 1)]  ## Various layers of graph diffusions
        )
        self.cat_attn_layers = nn.ModuleList(

            #input = embedding_size[even] output = embedding_size[odd]
            [CatMultiAttn(embedding_size[2 * i],
                          num_heads,
                          embedding_size[2 * i + 1],
                          active[i],
                          timestamp)
             for i in range(len(embedding_size) // 2)] ## List of CatMultiAttn, each for each layer
        )
        self.linear = nn.Linear(embedding_size[-1] * timestamp, classes)

        # Perform smart initialization
        self._init_transition_params()

    def _init_transition_params(self):
        # Xavier init for transition matrices
        nn.init.xavier_uniform_(self.T)
        # Uniform or constant for theta
        nn.init.constant_(self.theta, 1.0 / self.theta.size(-1))

    def forward(self, X, A):
        z, h = X, X # z output of the diffusion mechanism and h output of CatAtnn #z shape (num_nodes, timestamp*features) #h shape (num_nodes, timestamp*features)
        # Optionally constrain theta to be non-negative or sum-to-one:
        theta_pos  = F.softplus(self.theta)             # >= 0
        theta_prob = F.softmax(self.theta, dim=-1)      # sum-to-1

    
        # for each diffusion layers
        for l in range(self.T.shape[0]):
            # pass the theta at layer l (shape (expansion_step))
            # pass the difufsion layers l (shape (expansion_step, n_nodes, n_nodes))
            # pass last z
            # pass adjacency matrix
            z = self.diffusion_layers[l](theta_pos[l], self.T[l], z, A) # shape (num_nodes, diffusion_size[l+1])
            # h is new h from diffusion and h_prime is previous h prime
            h = self.cat_attn_layers[l](h=z, h_prime=h)
        
        return self.linear(h)

# For those who use fast implementation version.
#class DGDNN(nn.Module):
    #def __init__(self, diffusion_size, embedding_size, classes, num_heads, active, timestamp, layers):
        #super(DGDNN, self).__init__()

        # Initialize different module layers at all levels
        #self.diffusion_layers = nn.ModuleList(
            #[GeneralizedGraphDiffusion(diffusion_size[i], diffusion_size[i + 1], active[i]) for i in range(len(diffusion_size) - 1)])
        #self.cat_attn_layers = nn.ModuleList(
            #[CatMultiAttn(embedding_size[2 * i], num_heads, embedding_size[2 * i + 1], active[i], timestamp) for i in range(len(embedding_size) // 2)])
        #self.linear = nn.Linear(embedding_size[-1]*timestamp, classes)

    #def forward(self, X, A, W):
        #z, h = X, X

        #for l in range(layers):
            #z = self.diffusion_layers[l](z, A, W)
            #h = self.cat_attn_layers[l](z, h)

        #h = self.linear(h)

        #return h
