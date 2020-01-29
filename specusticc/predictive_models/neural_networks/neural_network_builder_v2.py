from specusticc.predictive_models.neural_networks.model_v2 import ModelV2


class NeuralNetworkBuilderV2:
    def __init__(self, target: str, name: str) -> None:
        self.target = target
        self.name = name
        self.model = None

        self._choose_model()

    def _choose_model(self):
        if self.target == 'classification':
            self._choose_classification_model()
        elif self.target == 'regression':
            self._choose_regression_model()
        else:
            raise NotImplementedError

    def _choose_classification_model(self):
        if self.name == 'cnn':
            from specusticc.predictive_models.neural_networks.models.classification.cnn import CNN
            self.model = CNN()
        elif self.name == 'lstm':
            from specusticc.predictive_models.neural_networks.models.classification.lstm import LSTM
            self.model = LSTM()
        elif self.name == 'mlp':
            from specusticc.predictive_models.neural_networks.models.classification.mlp import MLP
            self.model = MLP()
        else:
            raise NotImplementedError

    def _choose_regression_model(self):
        if self.name == 'cnn':
            from specusticc.predictive_models.neural_networks.models.regression.cnn import CNN
            self.model = CNN()
        elif self.name == 'lstm':
            from specusticc.predictive_models.neural_networks.models.regression.lstm import LSTM
            self.model = LSTM()
        elif self.name == 'lstm-attention':
            from specusticc.predictive_models.neural_networks.models.regression.lstm_attention import LSTMAttention
            self.model = LSTMAttention()
        elif self.name == 'mlp':
            from specusticc.predictive_models.neural_networks.models.regression.mlp import MLP
            self.model = MLP()
        else:
            raise NotImplementedError

    def build(self) -> ModelV2:
        return self.model
