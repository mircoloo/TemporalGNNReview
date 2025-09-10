from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch


class BaseModelRunner:
    def __init__(self, model, device, market_name=''):
        self.model = model
        self.device = device
        self.market_name = market_name
    def train(self, train_loader, val_loader, **kwargs):
        raise NotImplementedError
    def test(self, test_loader, **kwargs):
        raise NotImplementedError
    
    