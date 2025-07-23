from torch_geometric.data import Dataset, Batch
import torch

class BaseGraphDataset(Dataset):
    def __init__(self, dataset):
        """
        Args:
            data (Dataset): The Dataset of paper DGDNN Data(x,edge_index,edge_attr,y)
        """
        self.dataset = dataset
        assert len(dataset) > 0, "0 data in dataset, impossibile to create Dataset" 
        self.n_nodes = self.dataset[0].x.shape[0]
        self.n_features = int(self.dataset[0].x.shape[1] / 5)
        self.seq_length = int(self.dataset[0].x.shape[1] / self.n_features)
        self.n_edges = self.dataset[0].edge_index.shape[1]
        # self.x = dataset.x
        # self.edge_index = dataset.edge_index
        # self.edge_attr = dataset.edge_attr
        # self.y = dataset.y

    def __len__(self):
        """
            Returns the total number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
            This method must be implemented by subclasses.
            It should return a single processed sample (e.g., x_tensor, y_tensor)
            for the given index.
        """
        raise NotImplementedError("Subclasses must implement __getitem__ method")
    
    def is_input_correct_shaped(self, x) -> bool:
        #print(f"{x.shape[1]} {self.n_features * self.seq_length}")
        return x.shape[1] == self.n_features * self.seq_length

    def adjust_input_shape(self,x) -> torch.tensor:
        to_cat_dim = abs(x.shape[1] - (self.n_features * self.seq_length))
        last_values = x[:, -to_cat_dim:]
        x = torch.cat( (x, last_values), dim = 1)
        return x 
    