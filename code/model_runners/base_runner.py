class BaseModelRunner:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train(self, train_loader, val_loader, **kwargs):
        raise NotImplementedError

    def test(self, test_loader, **kwargs):
        raise NotImplementedError
    