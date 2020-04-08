from specusticc.predictive_models.neural_networks.neural_network import NeuralNetwork


class NeuralNetworkBuilder:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.model = None

    def build(self):
        self._choose_network()
        return self.model

    def _choose_network(self):
        target = self.config['model']['target']
        if target == 'classification':
            self._choose_classification_network()
        elif target == 'regression':
            self._choose_regression_network()
        else:
            raise NotImplementedError

    def _choose_classification_network(self):
        name = self.config['model']['name']
        if name == 'cnn':
            from specusticc.predictive_models.neural_networks.models.classification.cnn import CNN
            self.model = CNN(self.config)
        elif name == 'lstm':
            from specusticc.predictive_models.neural_networks.models.classification.lstm import LSTM
            self.model = LSTM(self.config)
        elif name == 'mlp':
            from specusticc.predictive_models.neural_networks.models.classification.mlp import MLP
            self.model = MLP(self.config)
        else:
            raise NotImplementedError

    def _choose_regression_network(self):
        name = self.config['model']['name']
        if name == 'basic':
            from specusticc.predictive_models.neural_networks.models.regression.basic_net import BasicNet
            self.model = BasicNet(self.config)
        elif name == 'cnn':
            from specusticc.predictive_models.neural_networks.models.regression.cnn import CNN
            self.model = CNN(self.config)
        elif name == 'lstm':
            from specusticc.predictive_models.neural_networks.models.regression.lstm import LSTM
            self.model = LSTM(self.config)
        elif name == 'lstm-attention':
            from specusticc.predictive_models.neural_networks.models.regression.lstm_attention import LSTMAttention
            self.model = LSTMAttention(self.config)
        elif name == 'mlp':
            from specusticc.predictive_models.neural_networks.models.regression.mlp import MLP
            self.model = MLP(self.config)
        else:
            raise NotImplementedError


