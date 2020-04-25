from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.neural_networks.neural_network import NeuralNetwork


class NeuralNetworkBuilder:
    def __init__(self, config: ModelCreatorConfig):
        self.config = config
        self.model = None

    def build(self) -> NeuralNetwork:
        self._choose_network()
        return self.model

    def _choose_network(self):
        target = self.config.machine_learning_target
        if target == 'classification':
            self._choose_classification_network()
        elif target == 'regression':
            self._choose_regression_network()
        else:
            raise NotImplementedError

    def _choose_classification_network(self):
        from specusticc.model_creating.neural_networks.models.classification.cnn import CNN
        from specusticc.model_creating.neural_networks.models.classification.lstm import LSTM
        from specusticc.model_creating.neural_networks.models.classification.mlp import MLP

        name = self.config.nn_name
        if name == 'cnn':
            self.model = CNN(self.config)
        elif name == 'lstm':
            self.model = LSTM(self.config)
        elif name == 'mlp':
            self.model = MLP(self.config)
        else:
            raise NotImplementedError

    def _choose_regression_network(self):
        from specusticc.model_creating.neural_networks.models.regression.basic_net import BasicNet
        from specusticc.model_creating.neural_networks.models.regression.cnn import CNN
        from specusticc.model_creating.neural_networks.models.regression.lstm import LSTM
        from specusticc.model_creating.neural_networks.models.regression.lstm_attention import LSTMAttention
        from specusticc.model_creating.neural_networks.models.regression.mlp import MLP
        from specusticc.model_creating.neural_networks.models.regression.lstm_enc_dec import LSTMEncoderDecoder

        name = self.config.nn_name
        if name == 'basic':
            self.model = BasicNet(self.config)
        elif name == 'cnn':
            self.model = CNN(self.config)
        elif name == 'lstm':
            self.model = LSTM(self.config)
        elif name == 'lstm-attention':
            self.model = LSTMAttention(self.config)
        elif name == 'encoder-decoder':
            self.model = LSTMEncoderDecoder(self.config)
        elif name == 'mlp':
            self.model = MLP(self.config)
        else:
            raise NotImplementedError