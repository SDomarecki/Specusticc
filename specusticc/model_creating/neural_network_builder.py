from specusticc.configs_init.model.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.neural_network import NeuralNetwork


class NeuralNetworkBuilder:
    def __init__(self, config: ModelCreatorConfig):
        self.config = config
        self.model = None

    def build(self) -> NeuralNetwork:
        self._choose_network()
        return self.model

    def _choose_network(self):
        from specusticc.model_creating.models.basic_net import BasicNet
        from specusticc.model_creating.models.cnn import CNN
        from specusticc.model_creating.models.lstm import LSTM
        from specusticc.model_creating.models.lstm_attention import LSTMAttention
        from specusticc.model_creating.models.mlp import MLP
        from specusticc.model_creating.models.lstm_enc_dec import LSTMEncoderDecoder
        from specusticc.model_creating.models.transformer import ModelTransformer

        name = self.config.name
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
        elif name == 'transformer':
            self.model = ModelTransformer(self.config)
        else:
            raise NotImplementedError
