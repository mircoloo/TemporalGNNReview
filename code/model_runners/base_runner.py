class BaseModelRunner:
    def __init__(self, model, device, market_name=''):
        self.model = model
        self.device = device
        self.market_name = market_name

    def train(self, train_loader, val_loader, **kwargs):
        raise NotImplementedError

    def test(self, test_loader, **kwargs):
        raise NotImplementedError
    