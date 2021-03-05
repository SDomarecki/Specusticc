from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.model_creating.models.basic_net import BasicNet
from specusticc.model_creating.models.cnn import CNN
from specusticc.model_creating.models.lstm import LSTM
from specusticc.model_creating.models.cnnlstm import CNNLSTM
from specusticc.model_creating.models.lstm_attention import LSTMAttention
from specusticc.model_creating.models.mlp import MLP
from specusticc.model_creating.models.lstm_enc_dec import LSTMEncoderDecoder
from specusticc.model_creating.models.transformer import ModelTransformer
from specusticc.model_creating.models.gan_wrapper import GANWrapper


def choose_network(model_name: str, config: AgentConfig):
    if model_name == "basic":
        return BasicNet(config)
    elif model_name == "cnn":
        return CNN(config)
    elif model_name == "lstm":
        return LSTM(config)
    elif model_name == "cnn-lstm":
        return CNNLSTM(config)
    elif model_name == "lstm-attention":
        return LSTMAttention(config)
    elif model_name == "encoder-decoder":
        return LSTMEncoderDecoder(config)
    elif model_name == "mlp":
        return MLP(config)
    elif model_name == "transformer":
        return ModelTransformer(config)
    elif model_name == "gan":
        return GANWrapper(config)
    else:
        raise NotImplementedError
