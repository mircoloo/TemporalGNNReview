from model_runners.runner_utils import BaseGraphDataset

class DGDNNDataset(BaseGraphDataset):
    def __init__(self, dataset):
        # Data(x=[1171, 110], edge_index=[2, 1369852], edge_attr=[1369852], y=[1171])
        super().__init__(dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]

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

        return x, y