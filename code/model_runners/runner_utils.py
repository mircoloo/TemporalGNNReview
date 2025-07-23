from torch_geometric.data import Dataset

class BaseGraphDataset(Dataset):
    def __init__(self, dataset):
        """
        Args:
            data (Dataset): The Dataset of paper DGDNN
        """
        self.dataset = dataset
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