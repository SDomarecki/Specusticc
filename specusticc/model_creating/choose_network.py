from configs_init.model.agent_config import AgentConfig


def choose_network(model_name: str, config: AgentConfig):
    from model_creating.models.basic_net import BasicNet
    from model_creating.models.cnn import CNN
    from model_creating.models.lstm import LSTM
    from model_creating.models.cnnlstm import CNNLSTM
    from model_creating.models.lstm_attention import LSTMAttention
    from model_creating.models.mlp import MLP
    from model_creating.models.lstm_enc_dec import LSTMEncoderDecoder
    from model_creating.models.transformer import ModelTransformer
    from model_creating.models.gan_wrapper import GANWrapper

    if model_name == 'basic':
        return BasicNet(config)
    elif model_name == 'cnn':
        return CNN(config)
    elif model_name == 'lstm':
        return LSTM(config)
    elif model_name == 'cnn-lstm':
        return CNNLSTM(config)
    elif model_name == 'lstm-attention':
        return LSTMAttention(config)
    elif model_name == 'encoder-decoder':
        return LSTMEncoderDecoder(config)
    elif model_name == 'mlp':
        return MLP(config)
    elif model_name == 'transformer':
        return ModelTransformer(config)
    elif model_name == 'gan':
        return GANWrapper(config)
    else:
        raise NotImplementedError