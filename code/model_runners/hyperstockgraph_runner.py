
from models.HyperStockGAT.training.models.base_models import NCModel
import torch

class HyperStockGraphRunner:
    def __init__(self, model, device):
        self.model: NCModel = model
        self.device = device

    def train(self, train_loader, validation_loader, optimizer, criterion, epochs, seq_length):
        self.model.train()
        for epoch in range(epochs):
            for data in train_loader:
                # Training logic here
                pass

    def test(self, test_loader, seq_length, num_features):
        self.model.eval()
        predictions = []
        true_values = []
        with torch.no_grad():
            for data in test_loader:
                # Testing logic here
                pass
        return predictions, true_values