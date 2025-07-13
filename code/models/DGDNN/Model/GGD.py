import torch
import torch.nn as nn

class GeneralizedGraphDiffusion(torch.nn.Module):
    def __init__(self, input_dim, output_dim, active):
        super(GeneralizedGraphDiffusion, self).__init__()
        self.output = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation0 = torch.nn.PReLU()
        self.active = active

    def forward(self, theta, T, x, A):
        Q = 0
        for i in range(theta.shape[0]): # for each diffusion step            
            Q += theta[i] * T[i] # scalar * (num_nodes x num_nodes)
        # Q is ((num_nodes * num_nodes) * (num_nodes * num_nodes)) @ (num_nodes, diffusion_size[i])
        x = self.fc((Q * A) @ x) #diffusion step input=diffusion_size[i], output=diffusion_size[i+1]
        # x.shape = (num_nodes, diffusion_size[i+1])
        if self.active:
            x = self.activation0(x)

        return x

# you can use the below for fast implementation if computational resources are limited.
#class GeneralizedGraphDiffusion(torch.nn.Module):
    #def __init__(self, input_dim, output_dim, active):
        #super(GeneralizedGraphDiffusion, self).__init__()
        #self.output = output_dim
        #self.gconv = GCNConv(input_dim, output_dim)
        #self.activation0 = torch.nn.PReLU()
        #self.active = active

    #def forward(self, x, a, w):
        #x = self.gconv(x, a, w)
        #if self.active:
            #x = self.activation0(x)

        #return x
