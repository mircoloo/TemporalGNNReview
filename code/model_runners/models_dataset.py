from model_runners.runner_utils import BaseGraphDataset
from torch_geometric.utils import to_dense_adj
import torch

class DGDNNDataset(BaseGraphDataset):
    def __init__(self, dataset):
        # Data(x=[1171, 110], edge_index=[2, 1369852], edge_attr=[1369852], y=[1171])
        super().__init__(dataset)
    def __getitem__(self, idx):
        data_sample = self.dataset[idx]
        x_real = data_sample.x
        res = super().is_input_correct_shaped(x_real) # check if the input is correct shape, since some samples are wrong
        if not res:
            #print(data_sample)
            x_real = super().adjust_input_shape(x_real) # in case reshape the tensor appending the last timestamp features   
        data_sample.x = x_real
        return data_sample
class GraphWaveNetDataset(BaseGraphDataset):
    def __init__(self, dataset):
        # Data(x=[1171, 110], edge_index=[2, 1369852], edge_attr=[1369852], y=[1171])
        super().__init__(dataset)
    def __getitem__(self, idx):
        data_sample = self.dataset[idx]
        x_real = data_sample.x
        res = super().is_input_correct_shaped(x_real) # check if the input is correct shape, since some samples are wrong
        if not res:
            #print(data_sample)
            x_real = super().adjust_input_shape(x_real) # in case reshape the tensor appending the last timestamp features        
        # x = x.view(self.n_nodes, self.n_features, self.seq_length).permute(0, 2, 1) # [num_nodes, seq_length, num_features]
        # x = x.unsqueeze(0)  # batch size 1
        # x = x.permute(0, 3, 1, 2) #(batch_size, num_features, num_nodes, sequence_length)             # rechanged the size 15/07/2025   
        y = data_sample.y.long()  # Ensure y is long for classification
        x = x_real.view(self.n_nodes, self.n_features, self.seq_length).permute(1, 0, 2)        # rechanged the size 25/07/2025

        return torch.tensor(x), torch.tensor(y)
    
class HyperStockGATDataset(BaseGraphDataset):
    def __init__(self, dataset):
        # Data(x=[n_nodes, features (5) * timestamps ], edge_index=[2, 1369852], edge_attr=[1369852], y=[1171])
        super().__init__(dataset)
        
    def __getitem__(self, idx):
        data_sample = self.dataset[idx] 
        x = data_sample.x 
        res = super().is_input_correct_shaped(x) # check if the input is correct shape, since some samples are wrong
        if not res:
            x = super().adjust_input_shape(x) # in case reshape the tensor appending the last timestamp features
        # Write specifit reshape code
        x = x.reshape((self.n_nodes, self.n_features, self.seq_length)).permute(0,2,1)
        adj = to_dense_adj(edge_index=data_sample.edge_index, edge_attr=data_sample.edge_attr).squeeze() #create adjacency list
        y =  torch.tensor(data_sample.y.clone().detach(), dtype=torch.float32).unsqueeze(1)
        return x, y, adj


class DARNNDataset(BaseGraphDataset):

    def __init__(self, dataset):
        super().__init__(dataset)
    
    def __getitem__(self, idx):
        data_sample = self.dataset[idx] 
        x = data_sample.x

        res = super().is_input_correct_shaped(x) # check if the input is correct shape, since some samples are wrong
        if not res:
            x = super().adjust_input_shape(x) # in case reshape the tensor appending the last timestamp features
        target = data_sample.y
        num_nodes = x.shape[0]
        N = num_nodes  # number of nodes
        T = int(x.shape[1] / 5) - 1
        # number of time steps
        X = x.view(num_nodes, 5, T+1).permute(0, 2, 1)  # [num_nodes, T]
        X = X[:, :, :1].squeeze()
        X = X.permute(1, 0)  # [T, N]
        y_target = torch.zeros((T, num_nodes), dtype=torch.long)
        for t in range(T):
            y_target[t, :] = (X[t, :] > X[t+1, :]).long() 

        X = X[:-1, :]  # use all but last timestep as input
        return X, y_target, target